from __future__ import annotations

from pivot.frontend.base import FunctionInfo, ParsedUnit, Use, UseContext
from pivot.frontend.scope_tracker import ScopeTracker
from pivot.frontend.types import CallStatement, ConstantArg, VariableDecl
from pivot.ir.source_span import SourceSpan
from pivot.isa.immediate_constants import fold_constant, fold_intrinsic_call
from pivot.isa.intrinsic_registry import IntrinsicRegistry, get_intrinsic_registry
from pivot.lang.grammars import FILE_START, callee_name, span, text


class TreeSitterFrontend:
    """What the tree-sitter frontends share: registry lookups, immediate folding,
    function and block lowering, and the walk that records a statement's uses.
    A frontend supplies ``_function_info`` and ``_lower_stmt``; ``_base`` is where
    the tree being lowered starts in the file (a re-parsed macro body)."""

    # Node types that open a lexical scope.
    _SCOPES: frozenset[str] = frozenset()
    # Literal node types, which are arguments as they are spelled.
    _LITERALS: frozenset[str] = frozenset()

    def __init__(self, intr_reg: IntrinsicRegistry | None = None) -> None:
        self._intr_reg = intr_reg or get_intrinsic_registry()

    def _begin(self, path: str) -> bytes:
        """Reads ``path`` and resets the per-file state; returns the source."""
        with open(path, "rb") as f:
            self._src = f.read()
        self._path = path
        self._base = FILE_START
        self._scope = ScopeTracker()
        self._current_func_info: FunctionInfo | None = None
        return self._src

    def _output_type(self, name: str) -> str:
        outs = self._intr_reg.intrinsic_output_types_of(name)
        return outs[0] if outs else "void"

    def _lower_function(self, func_node, unit: ParsedUnit) -> None:
        """A function in a scope of its own, which its parameters are declared in."""
        self._scope.push()
        self._current_func_info = self._function_info(func_node, unit)
        if (body := func_node.child_by_field_name("body")) is not None:
            self._lower_block(body, unit)
        unit.functions.append(self._current_func_info)
        self._current_func_info = None
        self._scope.pop()

    def _lower_block(self, block_node, unit: ParsedUnit) -> None:
        scoped = block_node.type in self._SCOPES
        if scoped:
            self._scope.push()
        for stmt in block_node.named_children:
            self._lower_stmt(stmt, unit)
        if scoped:
            self._scope.pop()

    @staticmethod
    def _int_literal_value(text: str) -> int | None:
        """Value of `text` as an integer literal of the frontend's language (C's
        `0261` is octal, Rust's decimal), else None."""
        return None

    def _int_literal_key(self, text: str) -> str | None:
        """Decimal matching key of an integer literal, else None."""
        value = self._int_literal_value(text)
        return None if value is None else str(value)

    def _fold_call(self, node) -> str | None:
        """The folded integer of a fold pseudo-intrinsic call (`_MM_SHUFFLE`) with
        constant arguments, else None."""
        if node.type != "call_expression":
            return None
        args_node = node.child_by_field_name("arguments")
        texts = [self._txt(c) for c in args_node.named_children] if args_node else []
        arg_texts = [self._int_literal_key(text) or text for text in texts]
        return fold_intrinsic_call(self._intr_reg, self._callee_name(node), arg_texts)

    def _constant_arg(self, node) -> ConstantArg | None:
        """``node`` as a constant argument: a folded immediate call, a literal, or a
        predicate enumerator (`_CMP_LE_OQ`), which selects the operation and so
        binds by its value."""
        if (folded := self._fold_call(node)) is not None:
            return ConstantArg(value=folded, extent=self._span(node))
        if node.type in self._LITERALS:
            literal = self._txt(node)
            return ConstantArg(value=literal, extent=self._span(node), key=self._int_literal_key(literal))
        if node.type == "identifier" and (pred := fold_constant(self._intr_reg, self._txt(node))) is not None:
            return ConstantArg(value=pred, extent=self._span(node))
        return None

    # Nested function node types (lambdas, closures): their names do not resolve
    # in the enclosing scope.
    _NESTED_FUNCTIONS: tuple[str, ...] = ()

    def _collect_uses(self, node, unit: ParsedUnit, lowered_from: int, roles=None) -> None:
        """Record a Use per variable read under ``node``, in source order.  The
        calls lowered from this statement start at ``lowered_from``; a variable
        argument of one is a use with ``call`` set.  ``roles`` presets roles."""
        lowered = unit.call_statements[lowered_from:]
        operand_of: dict[tuple[str, int], CallStatement] = {}
        pending = list(lowered)
        while pending:
            call = pending.pop()
            for arg, extent in zip(call.args, call.arg_extents):
                if isinstance(arg, CallStatement):
                    pending.append(arg)
                elif isinstance(arg, VariableDecl) and extent is not None:
                    operand_of[(arg.id, extent.start.byte)] = call
        roles = dict(roles or {})
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type in self._NESTED_FUNCTIONS:
                continue
            if n.type != "identifier":
                self._assign_roles(n, roles, lowered)
                stack.extend(reversed(n.named_children))
                continue
            decl = self._scope.lookup(self._txt(n))
            if decl is None or not self._is_read(n):
                continue
            extent = self._span(n)
            context, other, boundary = roles.get(n.start_byte, (UseContext.EXPRESSION, None, None))
            unit.uses.append(Use(
                decl, extent, context, other=other, boundary=boundary,
                call=operand_of.get((decl.id, extent.start.byte)),
                function=self._current_func_info if context is UseContext.RETURN else None,
            ))

    def _assign_roles(self, node, roles: dict[int, tuple], lowered: list[CallStatement]) -> None:
        """Record ``(context, other, boundary)`` for each variable ``node`` reads
        directly, keyed by the identifier's start byte; unrecorded reads are
        EXPRESSION uses."""

    def _role(self, roles: dict[int, tuple], node, context: UseContext, other: VariableDecl | None = None) -> None:
        """Presets the role of ``node`` if it is a variable inside casts and parentheses."""
        ident = self._unwrap(node)
        if ident is not None and ident.type == "identifier":
            roles[ident.start_byte] = (context, other, None)

    def _argument_roles(self, args_node, roles: dict[int, tuple]) -> None:
        intrinsic = self._intr_reg.is_typed_intrinsic(self._callee_name(args_node.parent))
        for arg in args_node.named_children:
            self._role(roles, arg, UseContext.INTRINSIC_ARGUMENT if intrinsic else UseContext.ARGUMENT)

    @staticmethod
    def _is_read(ident) -> bool:
        """Whether this mention of a variable reads it (not a callee, a binding
        or the target of a plain assignment)."""
        return True

    def _txt(self, node) -> str:
        return text(node, self._src)

    def _span(self, node) -> SourceSpan:
        return span(node, self._path, self._base)

    def _span_between(self, first, last) -> SourceSpan:
        """The source text after ``first`` up to the end of ``last``."""
        return SourceSpan(self._span(first).end, self._span(last).end, self._path)

    def _callee_name(self, call_node) -> str:
        return callee_name(call_node, self._src)

    def _unwrap(self, node):
        return self._grammar.unwrap(node)
