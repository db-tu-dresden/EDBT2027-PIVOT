from __future__ import annotations

import re

from pivot.frontend.base import FunctionInfo, ParsedUnit, UseContext
from pivot.frontend.tree_sitter_frontend import TreeSitterFrontend
from pivot.ir.source_span import SourceSpan
from pivot.frontend.types import CallStatement, VariableDecl
from pivot.lang.grammars import RUST, find

# A Rust integer literal: decimal (a leading 0 stays decimal), 0x / 0o / 0b,
# `_` separators, and an optional integer type suffix.
_RUST_INT_LITERAL_RE = re.compile(
    r"(?:0x(?P<hex>[0-9a-fA-F_]+)|0o(?P<oct>[0-7_]+)|0b(?P<bin>[01_]+)|(?P<dec>[0-9][0-9_]*))"
    r"(?:[iu](?:8|16|32|64|128|size))?"
)


def rust_int_literal_value(text: str) -> int | None:
    """Value of a Rust integer literal (`0xB1`, `0o261`, `0b1011_0001`, `177u32`),
    else None."""
    m = _RUST_INT_LITERAL_RE.fullmatch(text.strip())
    if m is None:
        return None
    for group, base in (("hex", 16), ("oct", 8), ("bin", 2), ("dec", 10)):
        if (digits := m.group(group)) is not None:
            digits = digits.replace("_", "")
            return int(digits, base) if digits else None
    return None


class RustFrontend(TreeSitterFrontend):
    """Tree-sitter based front-end. Lowers Rust `core::arch` intrinsic code into
    the shared CallStatement/VariableDecl IR (see ``ParsedUnit``)."""

    _int_literal_value = staticmethod(rust_int_literal_value)
    _grammar = RUST
    _SCOPES = frozenset({"block", "unsafe_block"})
    _LITERALS = frozenset({"integer_literal", "float_literal"})

    def parse(self, path: str) -> ParsedUnit:
        root = self._grammar.parse(self._begin(path))
        unit = ParsedUnit(path=path)
        for func in find(root, "function_item"):
            self._lower_function(func, unit)
        for use in find(root, "use_declaration"):
            unit.simd_headers[self._txt(use)] = self._use_span_with_attributes(use)
        return unit

    def _function_info(self, func_node, unit: ParsedUnit) -> FunctionInfo:
        name_node = func_node.child_by_field_name("name")
        ret_type_node = func_node.child_by_field_name("return_type")
        return FunctionInfo(
            name=self._txt(name_node) if name_node is not None else "",
            return_type=self._txt(ret_type_node) if ret_type_node is not None else "void",
            return_type_extent=self._span(ret_type_node) if ret_type_node is not None else None,
            params=self._lower_params(func_node.child_by_field_name("parameters"), unit),
        )

    def _lower_params(self, plist, unit: ParsedUnit) -> list[VariableDecl]:
        """Each parameter with its type text and the span that spells it."""
        params = []
        for p in plist.named_children if plist is not None else ():
            if p.type != "parameter":
                continue
            pname_node = p.child_by_field_name("pattern")
            ptype_node = p.child_by_field_name("type")
            params.append(self._scope.declare(
                self._txt(pname_node), self._span(pname_node),
                self._txt(ptype_node) if ptype_node else "auto", "param", unit,
                type_extent=self._span(ptype_node) if ptype_node else None,
            ))
        return params

    # Control-flow/blocks we recurse through to reach nested statements.
    _RECURSE = {
        "block", "unsafe_block", "while_expression", "for_expression",
        "loop_expression", "if_expression", "else_clause",
        "match_expression", "match_block", "match_arm",
    }

    def _lower_stmt(self, node, unit: ParsedUnit, stmt_span=None) -> None:
        t = node.type
        lowered_from = len(unit.call_statements)
        roles = None
        if t == "expression_statement":
            # The inner call, assignment or control flow, with the statement's span.
            inner = node.named_children[0] if node.named_children else None
            if inner is not None:
                self._lower_stmt(inner, unit, stmt_span=self._span(node))
            return
        if t in self._SCOPES:
            self._lower_block(node, unit)
            return
        if t in self._RECURSE:
            for child in node.named_children:
                self._lower_stmt(child, unit)
            return
        if t == "let_declaration":
            self._lower_let(node, unit)
        elif t in ("call_expression", "assignment_expression"):
            self._lower_expr(node, unit, stmt_span or self._span(node))
        elif t == "identifier":
            # A bare identifier in statement position is a block's value: at the
            # end of the function body, what the function returns.
            roles = {node.start_byte: (UseContext.RETURN, None, None)}
        elif not t.endswith("_expression"):
            return
        self._collect_uses(node, unit, lowered_from, roles)

    def _lower_expr(self, node, unit: ParsedUnit, stmt_span) -> None:
        if node.type == "call_expression":
            call = self._build_call(node, unit, None)
        elif node.type == "assignment_expression":
            rhs = node.child_by_field_name("right")
            if rhs is None or rhs.type != "call_expression":
                return
            lhs = node.child_by_field_name("left")
            target = self._scope.resolve(self._txt(lhs), self._span(lhs), unit)
            call = self._build_call(rhs, unit, target)
        else:
            return
        call.extent_with_var = stmt_span
        unit.call_statements.append(call)

    def _lower_let(self, let_node, unit: ParsedUnit) -> None:
        value = let_node.child_by_field_name("value")
        if value is None:
            return
        if value.type == "call_expression":
            name_node = let_node.child_by_field_name("pattern")
            type_node = let_node.child_by_field_name("type")
            type_extent = self._span(type_node) if type_node is not None else None
            out_type = self._txt(type_node) if type_node is not None else self._output_type(self._callee_name(value))
            decl = self._scope.declare(self._txt(name_node), self._span(name_node), out_type, "var", unit, type_extent=type_extent)
            call = self._build_call(value, unit, decl)
            call.extent_with_var = self._span(let_node)
            unit.call_statements.append(call)
        elif value.type == "identifier":
            name_node = let_node.child_by_field_name("pattern")
            if name_node is not None:
                type_node = let_node.child_by_field_name("type")
                type_extent = self._span(type_node) if type_node is not None else None
                out_type = self._txt(type_node) if type_node is not None else "auto"
                self._scope.declare(self._txt(name_node), self._span(name_node), out_type, "var", unit, type_extent=type_extent)

    _NESTED_FUNCTIONS = ("closure_expression",)

    def _assign_roles(self, node, roles: dict[int, tuple], lowered: list[CallStatement]) -> None:
        def variable(child):
            return self._scope.lookup(self._txt(child)) if child is not None else None

        t = node.type
        if t == "arguments":
            self._argument_roles(node, roles)
        elif t == "return_expression" and node.named_children:
            self._role(roles, node.named_children[0], UseContext.RETURN)
        elif t == "let_declaration" and (value := node.child_by_field_name("value")) is not None:
            self._role(roles, value, UseContext.COPY, variable(node.child_by_field_name("pattern")))
        elif t in ("assignment_expression", "compound_assignment_expr"):
            left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
            target = variable(left) if left.type == "identifier" else None
            if t == "assignment_expression":
                self._role(roles, right, UseContext.COPY, target)
            else:
                self._role(roles, left, UseContext.ASSIGNED)
                self._role(roles, right, UseContext.COMPOUND_VALUE, target)

    @staticmethod
    def _is_read(ident) -> bool:
        parent = ident.parent
        if parent is None or parent.type == "scoped_identifier":
            return False
        for field, owner in (("function", "call_expression"), ("pattern", None), ("left", "assignment_expression")):
            child = parent.child_by_field_name(field)
            if child is not None and child.start_byte == ident.start_byte and child.end_byte == ident.end_byte:
                return owner is not None and parent.type != owner
        return True

    def _build_call(self, call_node, unit: ParsedUnit, defines: VariableDecl | None) -> CallStatement:
        name = self._callee_name(call_node)
        args, arg_extents = self._lower_args(call_node, unit)
        call = CallStatement(
            name=name,
            is_intrinsic=self._intr_reg.is_typed_intrinsic(name),
            extent=self._span(call_node),
            args=args,
            arg_extents=arg_extents,
            returns_decl=defines,
            is_strict_assignment=True,   # rhs is exactly the call in all our forms
        )
        # The producer link lets a hoisted temp (`pivot_tmp = call;`, dtype
        # "auto") be typed from its producer.
        if defines is not None:
            defines.producer_call_id = call.id
        return call

    def _lower_args(self, call_node, unit: ParsedUnit) -> tuple[list, list]:
        """Each argument, stripped of an `as` cast, with its call-site span."""
        args_node = call_node.child_by_field_name("arguments")
        out: list = []
        extents: list = []
        for a in args_node.named_children:
            if a.type == "type_cast_expression":
                a = a.child_by_field_name("value")
            extents.append(self._span(a))
            if (constant := self._constant_arg(a)) is not None:
                out.append(constant)
            elif a.type == "call_expression" and self._intr_reg.is_typed_intrinsic(self._callee_name(a)):
                out.append(self._nested_call_arg(a, unit))
            else:
                # A variable, or an opaque expression (`a.add(i)`) resolved as a symbol.
                out.append(self._scope.resolve(self._txt(a), self._span(a), unit))
        return out, extents

    def _nested_call_arg(self, call_node, unit: ParsedUnit):
        """A nested intrinsic call argument, embedded as a sub-``CallStatement``."""
        return self._build_call(call_node, unit, defines=None)

    def _use_span_with_attributes(self, use_node) -> SourceSpan:
        """Span of a ``use`` declaration extended over the outer attributes before
        it (``#[cfg(target_arch = "x86_64")]``): tree-sitter makes them siblings,
        and left behind they would bind to the companion binding inserted there."""
        start = use_node
        sibling = use_node.prev_named_sibling
        while sibling is not None and sibling.type == "attribute_item":
            start = sibling
            sibling = sibling.prev_named_sibling
        return SourceSpan(self._span(start).start, self._span(use_node).end, self._path)
