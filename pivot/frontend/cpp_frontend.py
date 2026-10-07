from __future__ import annotations

import re
from typing import NamedTuple, Optional

from pivot.frontend.base import FunctionInfo, ParsedUnit, RedundantCast, TypeAlias, TypeSite, UseContext
from pivot.frontend.tree_sitter_frontend import TreeSitterFrontend
from pivot.frontend.types import CallStatement, ConstantArg, ElementArg, VariableDecl
from pivot.ir.source_span import SourcePos, SourceSpan
from pivot.lang.grammars import c_family_grammar, find, macro_bodies, walk
from pivot.lang.temps import is_pivot_temp_name

# Element bytes of the integer types a constant array is materialized from.
_INT_TYPE_BYTES = {
    "int8_t": 1, "uint8_t": 1, "char": 1, "signed char": 1, "unsigned char": 1,
    "int16_t": 2, "uint16_t": 2, "short": 2, "unsigned short": 2,
    "int32_t": 4, "uint32_t": 4, "int": 4, "unsigned": 4, "unsigned int": 4,
    "int64_t": 8, "uint64_t": 8, "long": 8, "unsigned long": 8,
    "long long": 8, "size_t": 8, "ssize_t": 8, "ptrdiff_t": 8,
}

# An element of a materialized constant array: decimal or hex, int suffixes.
_INT_LITERAL_RE = re.compile(r"^[+-]?(?:0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*$")

# A C/C++ integer literal; tree-sitter folds an adjacent sign into the token (`-1`).
_C_INT_LITERAL_RE = re.compile(
    r"(?P<sign>[+-]?)"
    r"(?:0[xX](?P<hex>[0-9a-fA-F](?:'?[0-9a-fA-F])*)"
    r"|0[bB](?P<bin>[01](?:'?[01])*)"
    r"|(?P<oct>0(?:'?[0-7])*)"
    r"|(?P<dec>[1-9](?:'?[0-9])*))"
    r"(?:[uU](?:ll|LL|l|L)?|(?:ll|LL|l|L)[uU]?)?"
)

# Header spellings that pull in SIMD intrinsics.
_SIMD_HEADER_HINTS = (
    "immintrin", "xmmintrin", "emmintrin", "smmintrin", "tmmintrin",
    "nmmintrin", "wmmintrin", "x86intrin", "arm_neon", "arm_sve",
)

# The type of a C-style cast to a raw x86 SIMD pointer.
_SIMD_POINTER_TYPE_RE = re.compile(r"^(?:const\s+)?__m(?:128|256|512)[id]?\s*\*$")

_NAMED_CASTS = ("reinterpret_cast", "static_cast", "const_cast")


def c_int_literal_value(text: str) -> int | None:
    """Value of a C/C++ integer literal (`0xB1`, `0261`, `0b10110001`, `177u`),
    else None."""
    m = _C_INT_LITERAL_RE.fullmatch(text.strip())
    if m is None:
        return None
    for group, base in (("hex", 16), ("bin", 2), ("oct", 8), ("dec", 10)):
        if (digits := m.group(group)) is not None:
            value = int(digits.replace("'", ""), base)
            return -value if m.group("sign") == "-" else value
    return None


class _PointerCast(NamedTuple):
    """A cast to a pointer to SIMD vectors, spelled raw or through ``alias``."""
    node: object
    type_name: object
    base_type: str
    alias: Optional[TypeAlias]
    inner: object


class CppFrontend(TreeSitterFrontend):
    """Lowers C/C++ intrinsic code into a ``ParsedUnit`` without expanding macros."""

    _int_literal_value = staticmethod(c_int_literal_value)
    _SCOPES = frozenset({"compound_statement"})
    _LITERALS = frozenset({"number_literal", "char_literal"})
    _NESTED_FUNCTIONS = ("lambda_expression",)

    # Each part of these may run a different number of times than the code around
    # it, so it is a straight-line region of its own, which no pattern spans.
    _CONTROL_FLOW = {
        "if_statement", "else_clause", "for_statement", "while_statement",
        "do_statement", "switch_statement",
    }
    # One region each.
    _SINGLE_REGION = {"case_statement", "labeled_statement"}
    # The preprocessor is not run, so every branch of an in-body `#if` is lowered.
    _PREPROC_BRANCHES = {"preproc_if", "preproc_ifdef", "preproc_elif", "preproc_else"}

    def parse(self, path: str) -> ParsedUnit:
        src = self._begin(path)
        self._grammar = c_family_grammar(path)
        root = self._grammar.parse(src)
        self._region = self._last_region = 0
        # name -> [(scope start byte, scope end byte, alias)]
        self._aliases: dict[str, list[tuple[int, int, TypeAlias]]] = {}
        # Start bytes of the pointer casts a pointer holder keeps, and of the raw
        # type names inside them -> the holder they type.
        self._kept_casts: set[int] = set()
        self._site_owner: dict[int, VariableDecl] = {}

        unit = ParsedUnit(path=path)
        self._collect_type_aliases(root, unit)
        for func in find(root, "function_definition"):
            self._lower_function(func, unit)
        for body, base in macro_bodies(root, src):
            self._lower_macro_body(body, base, unit)
        self._collect_includes(root, unit)
        self._collect_redundant_casts(root, unit)
        self._collect_type_sites(root, unit)
        return unit

    def _collect_includes(self, root, unit: ParsedUnit) -> None:
        for inc in find(root, "preproc_include"):
            if (path_node := inc.child_by_field_name("path")) is None:
                continue
            simd = any(h in self._txt(path_node) for h in _SIMD_HEADER_HINTS)
            (unit.simd_headers if simd else unit.std_headers)[self._txt(inc).strip()] = self._span(inc)

    def _function_info(self, func_node, unit: ParsedUnit) -> FunctionInfo:
        declarator = func_node.child_by_field_name("declarator")
        name_node = declarator
        if declarator is not None and declarator.type == "function_declarator":
            name_node = declarator.child_by_field_name("declarator")
        rtype_node = func_node.child_by_field_name("type")
        ralias = self._alias_of_type(rtype_node)
        if ralias is not None:
            rtype = ralias.base_type
        else:
            rtype = self._txt(rtype_node) if rtype_node is not None else ""
        return FunctionInfo(
            name=self._txt(name_node) if name_node is not None else "",
            return_type=rtype,
            return_type_extent=self._span(rtype_node) if rtype_node is not None else None,
            params=self._lower_params(declarator, unit),
            return_type_alias=ralias.id if ralias is not None else None,
        )

    def _lower_params(self, declarator, unit: ParsedUnit) -> list[VariableDecl]:
        plist = declarator.child_by_field_name("parameters") if declarator is not None else None
        params: list[VariableDecl] = []
        for p in plist.named_children if plist is not None else ():
            if p.type != "parameter_declaration":
                continue
            ptype_node = p.child_by_field_name("type")
            if (p.child_by_field_name("declarator") is None and ptype_node is not None
                    and self._txt(ptype_node) == "void"):
                continue  # `f(void)` declares no parameters
            params.append(self._lower_param(p, unit))
        return params

    def _lower_param(self, param_node, unit: ParsedUnit) -> VariableDecl:
        declarator = param_node.child_by_field_name("declarator")
        name_node, ptr, dims = self._declarator_info(declarator) if declarator is not None else (None, 0, [])
        dtype, type_extent, alias = self._declared_type(param_node, declarator, ptr, dims)
        if name_node is None:
            # Not declared, but its type still tells overloads apart.
            return VariableDecl("", dtype, self._span(param_node), "param", type_extent=type_extent)
        decl = self._scope.declare(
            self._txt(name_node), self._span(param_node), dtype, "param", unit, type_extent=type_extent,
        )
        if alias is not None:
            decl.type_alias, decl.alias_spelled = alias.id, True
        decl.holds_elements = self._is_element_pointer(dtype, ptr, dims)
        return decl

    def _lower_declaration(self, decl_node, unit: ParsedUnit) -> None:
        """Declares each variable of a declaration and lowers the call that initializes it."""
        for declarator in decl_node.children_by_field_name("declarator"):
            inner, value = declarator, None
            if declarator.type == "init_declarator":
                inner = declarator.child_by_field_name("declarator")
                value = declarator.child_by_field_name("value")
            decl = self._declare_variable(decl_node, inner, value, unit)
            call_node = self._unwrap(value)
            if decl is None or call_node is None or call_node.type != "call_expression":
                continue
            call = self._build_call(call_node, unit, defines=decl, is_declaration=True)
            call.extent_with_var = self._span(decl_node)
            call.initializer_extent = self._span_between(inner, declarator)
            call.is_strict_assignment = value is call_node  # no cast or parentheses around it
            unit.call_statements.append(call)

    def _declare_variable(self, decl_node, declarator, value, unit: ParsedUnit) -> VariableDecl | None:
        name_node, ptr, dims = self._declarator_info(declarator)
        if name_node is None:
            return None
        dtype, type_extent, alias = self._declared_type(decl_node, declarator, ptr, dims)
        holds_elements = self._is_element_pointer(dtype, ptr, dims)
        cast = self._simd_pointer_cast(value)
        if (cast is not None and not holds_elements and not dims and ptr <= 1
                and self._is_placeholder(decl_node.child_by_field_name("type"))):
            # `auto* p = (T*)x`: the cast spells the element type, `auto` stays.
            dtype, type_extent, holds_elements = f"{cast.base_type} *", None, True
        decl = self._scope.declare(
            self._txt(name_node), self._span(decl_node), dtype, "var", unit,
            type_extent=type_extent, const_init=self._const_int_array(decl_node, dims, value),
        )
        if alias is not None:
            decl.type_alias, decl.alias_spelled = alias.id, True
        decl.holds_elements = holds_elements
        if holds_elements and cast is not None:
            self._keep_holder_cast(decl, cast)
        return decl

    def _declared_type(self, decl_node, declarator, ptr: int, dims: list[str]):
        """``(dtype, type span, alias)`` of a declaration; an alias's dtype is its SIMD type."""
        dtype = self._assemble_dtype(decl_node, ptr, dims)
        alias = self._alias_of_type(decl_node.child_by_field_name("type"))
        if alias is not None:
            dtype = self._dealias(dtype, alias)
        return dtype, self._type_span(decl_node, declarator, ptr), alias

    def _lower_stmt(self, node, unit: ParsedUnit) -> None:
        t = node.type
        lowered_from = len(unit.call_statements)
        if t == "declaration":
            self._lower_declaration(node, unit)
            self._collect_uses(node, unit, lowered_from)
        elif t == "expression_statement":
            if node.named_children:
                self._lower_expr(node.named_children[0], unit, stmt_span=self._span(node))
            self._collect_uses(node, unit, lowered_from)
        elif t in self._SCOPES:
            self._lower_block(node, unit)
        elif t in self._CONTROL_FLOW:
            for child in node.named_children:
                self._lower_in_new_region([child], unit)
        elif t in self._SINGLE_REGION:
            self._lower_in_new_region(node.named_children, unit)
        elif t in self._PREPROC_BRANCHES:
            for child in node.named_children:
                self._lower_stmt(child, unit)
        elif t in ("return_statement", "condition_clause") or t.endswith("_expression"):
            # A return, a condition, a `for` initializer or update.
            self._collect_uses(node, unit, lowered_from)

    def _lower_in_new_region(self, nodes, unit: ParsedUnit) -> None:
        saved = self._region
        self._last_region += 1
        self._region = self._last_region
        for node in nodes:
            self._lower_stmt(node, unit)
        self._region = saved

    def _lower_expr(self, node, unit: ParsedUnit, stmt_span) -> None:
        if node.type == "call_expression":
            call = self._build_call(node, unit, None, False)
            call.extent_with_var = stmt_span
            unit.call_statements.append(call)
        elif node.type == "assignment_expression":
            # Fetched once: tree-sitter wraps a node anew per fetch, so `rhs is right`
            # tells a bare call from one inside a cast or parentheses.
            right = node.child_by_field_name("right")
            rhs = self._unwrap(right)
            if rhs is not None and rhs.type == "call_expression":
                lhs = node.child_by_field_name("left")
                target = self._scope.resolve(self._txt(lhs), self._span(lhs), unit)
                call = self._build_call(rhs, unit, target, False)
                call.extent_with_var = stmt_span
                call.is_strict_assignment = self._assignment_operator(node) == "=" and rhs is right
                unit.call_statements.append(call)
            self._lower_element_assignment(node, unit)

    def _build_call(self, call_node, unit: ParsedUnit, defines: VariableDecl | None,
                    is_declaration: bool) -> CallStatement:
        name = self._callee_name(call_node)
        is_intrinsic = self._intr_reg.is_typed_intrinsic(name)
        args, arg_extents = self._lower_args(call_node, unit, is_intrinsic)
        call = CallStatement(
            name=name,
            is_intrinsic=is_intrinsic,
            extent=self._span(call_node),
            args=args,
            arg_extents=arg_extents,
            returns_decl=defines,
            is_strict_assignment=is_declaration,
            region=self._region,
        )
        if defines is not None:
            defines.producer_call_id = call.id
        return call

    def _lower_args(self, call_node, unit: ParsedUnit, is_intrinsic: bool) -> tuple[list, list]:
        """Each argument inside its casts and parentheses, with its call-site span."""
        args_node = call_node.child_by_field_name("arguments")
        args: list = []
        extents: list = []
        for arg in args_node.named_children if args_node is not None else ():
            value = self._unwrap(arg)
            extents.append(self._span(value))
            args.append(self._lower_arg(value, unit, is_intrinsic))
        return args, extents

    def _lower_arg(self, a, unit: ParsedUnit, is_intrinsic: bool):
        if (constant := self._constant_arg(a)) is not None:
            return constant
        if a.type == "identifier":
            # An unresolved name (an extern, a macro parameter) stays an opaque constant.
            decl = self._scope.lookup(self._txt(a))
            return decl if decl is not None else ConstantArg(value=self._txt(a), extent=self._span(a))
        if a.type == "call_expression" and self._intr_reg.is_typed_intrinsic(self._callee_name(a)):
            return self._build_call(a, unit, defines=None, is_declaration=False)
        if not is_intrinsic and (holder := self._element_holder(a)) is not None:
            return ElementArg(value=self._txt(a), extent=self._span(a), pointer=holder)
        # `&rids[i]`, `a + b`, `f(x)`: opaque text.
        return ConstantArg(value=self._txt(a), extent=self._span(a),
                           callee=self._callee_name(a) if a.type == "call_expression" else None)

    def _lower_macro_body(self, body: bytes, base, unit: ParsedUnit) -> None:
        """Lowers a `#define` body in place, in a scope of its own.  The macro's
        parameters and the call site's locals stay unresolved, as expansion
        substitutes them per site."""
        frag_root = self._grammar.parse(body)
        saved = (self._src, self._base, self._current_func_info)
        self._src, self._base = body, base
        self._current_func_info = None  # a `return` in a body belongs to the caller
        self._scope.push()
        try:
            for stmt in frag_root.named_children:
                self._lower_stmt(stmt, unit)
            self._collect_redundant_casts(frag_root, unit)
            self._collect_type_sites(frag_root, unit)
        finally:
            self._scope.pop()
            self._src, self._base, self._current_func_info = saved

    def _assign_roles(self, node, roles: dict[int, tuple], lowered: list[CallStatement]) -> None:
        if node.type == "argument_list":
            self._argument_roles(node, roles)
        elif node.type == "return_statement" and node.named_children:
            self._role(roles, node.named_children[0], UseContext.RETURN)
        elif node.type == "init_declarator" and (value := node.child_by_field_name("value")) is not None:
            self._initializer_roles(node, value, roles)
        elif node.type == "assignment_expression":
            self._assignment_roles(node, roles, lowered)

    def _initializer_roles(self, declarator, value, roles: dict[int, tuple]) -> None:
        name_node, _, _ = self._declarator_info(declarator.child_by_field_name("declarator"))
        target = self._scope.lookup(self._txt(name_node)) if name_node is not None else None
        self._role(roles, value, UseContext.COPY, target)
        if target is not None and (base := self._element_base(value)) is not None:
            roles[base.start_byte] = (UseContext.ELEMENT_LOAD, target, self._span(self._strip_parens(value)))

    def _assignment_roles(self, node, roles: dict[int, tuple], lowered: list[CallStatement]) -> None:
        left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
        plain = self._assignment_operator(node) == "="
        target = self._scope.lookup(self._txt(left)) if left.type == "identifier" else None
        self._role(roles, left, UseContext.ASSIGNED)
        self._role(roles, right, UseContext.COPY if plain else UseContext.COMPOUND_VALUE, target)
        if not plain:
            return
        if (base := self._element_base(left)) is not None:
            stored = self._stored_value(self._unwrap(right), lowered)
            if stored is not None:
                roles[base.start_byte] = (UseContext.ELEMENT_STORE, stored, self._span(self._unwrap(right)))
        elif (target is not None and self._holder_ident(left) is None
              and (base := self._element_base(right)) is not None):
            roles[base.start_byte] = (UseContext.ELEMENT_LOAD, target, self._span(self._strip_parens(right)))

    def _stored_value(self, value, lowered: list[CallStatement]) -> VariableDecl | None:
        """The variable holding what `p[i] = value` stores: the value's own, or
        the result of the call lowered from it."""
        if value.type == "identifier":
            return self._scope.lookup(self._txt(value))
        if value.type == "call_expression":
            start = self._span(value).start.byte
            return next((c.returns_decl for c in lowered if c.extent.start.byte == start), None)
        return None

    @staticmethod
    def _assignment_operator(node) -> str:
        return node.child(1).type if node.child_count >= 2 else "="

    @staticmethod
    def _is_read(ident) -> bool:
        parent = ident.parent
        if parent is None:
            return False
        field = next(
            (name for name in ("function", "declarator", "left")
             if any(child.start_byte == ident.start_byte and child.end_byte == ident.end_byte
                    for child in parent.children_by_field_name(name))),
            None,
        )
        if field == "function":
            return parent.type != "call_expression"
        if field == "declarator":
            return not (parent.type == "declaration" or parent.type.endswith("declarator"))
        if field == "left" and parent.type == "assignment_expression":
            return parent.child(1).type != "="
        return True

    def _is_vector_type(self, name: str) -> bool:
        return self._intr_reg.is_dtype(name)

    def _collect_type_aliases(self, root_node, unit: ParsedUnit) -> None:
        """Every `using A = T;` / `typedef T A;` of a SIMD vector type or of such
        an alias, visible within the scope that declares it."""
        for node in walk(root_node):
            if node.type == "alias_declaration":
                name_node, desc = node.child_by_field_name("name"), node.child_by_field_name("type")
                if name_node is not None and desc is not None and desc.child_by_field_name("declarator") is None:
                    self._add_alias(node, name_node, desc.child_by_field_name("type"), unit)
            elif node.type == "type_definition":
                for name_node in node.children_by_field_name("declarator"):
                    if name_node.type == "type_identifier":
                        self._add_alias(node, name_node, node.child_by_field_name("type"), unit)

    def _add_alias(self, decl_node, name_node, type_node, unit: ParsedUnit) -> None:
        if type_node is None or type_node.type not in ("type_identifier", "primitive_type"):
            return
        target = self._txt(type_node)
        name, span = self._txt(name_node), self._span(name_node)
        alias_id = f"{name}@{span.start.line}:{span.start.column}"
        if self._is_vector_type(target):
            alias = TypeAlias(id=alias_id, name=name, base_type=target, target_extent=self._span(type_node))
        elif (chained := self._alias_named(target, self._span(type_node).start.byte)) is not None:
            alias = TypeAlias(id=alias_id, name=name, base_type=chained.base_type,
                              target_extent=None, target_alias_id=chained.id)
        else:
            return
        scope = self._span(decl_node.parent or decl_node)
        self._aliases.setdefault(name, []).append((scope.start.byte, scope.end.byte, alias))
        unit.type_aliases.append(alias)

    def _alias_named(self, name: str, at_byte: int) -> TypeAlias | None:
        """The innermost alias called `name` whose scope contains `at_byte`."""
        visible = [(end - start, alias) for start, end, alias in self._aliases.get(name, ()) if start <= at_byte < end]
        return min(visible, key=lambda v: v[0])[1] if visible else None

    def _alias_of_type(self, type_node) -> TypeAlias | None:
        if type_node is None or type_node.type != "type_identifier":
            return None
        return self._alias_named(self._txt(type_node), self._span(type_node).start.byte)

    @staticmethod
    def _dealias(dtype: str, alias: TypeAlias) -> str:
        return re.sub(rf"\b{re.escape(alias.name)}\b", alias.base_type, dtype, count=1)

    def _is_element_pointer(self, dtype: str, ptr: int, dims: list[str]) -> bool:
        """A single-level pointer to SIMD vectors (`const __m128i *p`)."""
        return ptr == 1 and not dims and self._is_vector_type(dtype.replace("*", " ").strip())

    @staticmethod
    def _is_placeholder(type_node) -> bool:
        return type_node is not None and type_node.type == "placeholder_type_specifier"

    @staticmethod
    def _strip_parens(node):
        while node is not None and node.type == "parenthesized_expression" and node.named_children:
            node = node.named_children[0]
        return node

    def _simd_pointer_cast(self, node) -> _PointerCast | None:
        """A C-style or named cast to a single-level pointer to a SIMD vector type,
        spelled raw or through an alias."""
        node = self._strip_parens(node)
        desc, inner = self._cast_parts(node) if node is not None else (None, None)
        if desc is None or inner is None:
            return None
        star = desc.child_by_field_name("declarator")
        if star is None or star.type != "abstract_pointer_declarator" or star.child_by_field_name("declarator"):
            return None
        type_name = desc.child_by_field_name("type")
        if type_name is None:
            return None
        name = self._txt(type_name)
        if self._is_vector_type(name):
            return _PointerCast(node, type_name, name, None, inner)
        alias = self._alias_of_type(type_name)
        return _PointerCast(node, type_name, alias.base_type, alias, inner) if alias is not None else None

    def _cast_parts(self, node):
        """``(type descriptor, operand)`` of a C-style or C++ named cast, else Nones."""
        if node.type == "cast_expression":
            return node.child_by_field_name("type"), node.child_by_field_name("value")
        fn = node.child_by_field_name("function") if node.type == "call_expression" else None
        if fn is None or fn.type != "template_function":
            return None, None
        fname = fn.child_by_field_name("name")
        if fname is None or self._txt(fname) not in _NAMED_CASTS:
            return None, None
        targs = fn.child_by_field_name("arguments")
        descs = [c for c in targs.named_children if c.type == "type_descriptor"] if targs is not None else []
        call_args = node.child_by_field_name("arguments")
        operands = call_args.named_children if call_args is not None else []
        return (descs[0] if len(descs) == 1 else None), (operands[0] if len(operands) == 1 else None)

    def _keep_holder_cast(self, holder: VariableDecl, cast: _PointerCast) -> None:
        """The cast that initializes or assigns a pointer holder spells its element
        type: it stays, and its type name is rewritten with the holder's type."""
        self._kept_casts.add(self._span(cast.node).start.byte)
        if cast.alias is not None:
            if holder.type_alias is None:
                holder.type_alias = cast.alias.id
        else:
            self._site_owner[self._span(cast.type_name).start.byte] = holder

    def _is_indirection(self, node) -> bool:
        """`*x`, where a `pointer_expression` may also be `&x`."""
        return node.type == "pointer_expression" and bool(node.children) and self._txt(node.children[0]) == "*"

    def _element_base(self, node):
        """The identifier of the pointer holder an element access `p[i]`, `*p`,
        `*(p + k)` reads or writes, else None."""
        node = self._strip_parens(node)
        if node is None:
            return None
        if node.type == "subscript_expression":
            return self._holder_ident(node.child_by_field_name("argument"))
        if not self._is_indirection(node):
            return None
        base = self._strip_parens(node.child_by_field_name("argument"))
        if base is not None and base.type == "binary_expression":
            operator = base.child_by_field_name("operator")
            if operator is None or self._txt(operator) not in ("+", "-"):
                return None
            sides = [self._holder_ident(base.child_by_field_name(f)) for f in ("left", "right")]
            holders = [side for side in sides if side is not None]
            return holders[0] if len(holders) == 1 else None
        return self._holder_ident(base)

    def _holder_ident(self, node):
        node = self._strip_parens(node)
        if node is None or node.type != "identifier":
            return None
        decl = self._scope.lookup(self._txt(node))
        return node if decl is not None and decl.holds_elements else None

    def _element_holder(self, node) -> VariableDecl | None:
        base = self._element_base(node)
        return None if base is None else self._scope.lookup(self._txt(base))

    def _lower_element_assignment(self, node, unit: ParsedUnit) -> None:
        """A plain `x = rhs`: a pointer holder keeps the SIMD pointer cast it is
        assigned from; a PIVOT temp an element is loaded into is declared here,
        as a hoisted temp has no declaration of its own."""
        lhs = node.child_by_field_name("left")
        if self._assignment_operator(node) != "=" or lhs is None or lhs.type != "identifier":
            return
        right = node.child_by_field_name("right")
        if self._holder_ident(lhs) is not None:
            if (cast := self._simd_pointer_cast(right)) is not None:
                self._keep_holder_cast(self._scope.lookup(self._txt(lhs)), cast)
            return
        name = self._txt(lhs)
        if (self._element_base(right) is not None and self._scope.lookup(name) is None
                and is_pivot_temp_name(name)):
            self._scope.resolve(name, self._span(lhs), unit)

    def _collect_type_sites(self, root_node, unit: ParsedUnit) -> None:
        """Every spelling of a SIMD vector type name, so none is left unrewritten."""
        for node in walk(root_node):
            if node.type not in ("type_identifier", "primitive_type", "identifier"):
                continue
            name = self._txt(node)
            if not self._is_vector_type(name):
                continue
            span = self._span(node)
            unit.type_sites.append(TypeSite(
                extent=span, type_name=name, owner=self._site_owner.get(span.start.byte),
            ))

    def _collect_redundant_casts(self, root_node, unit: ParsedUnit) -> None:
        """The SIMD pointer casts the translation drops: neither dereferenced nor
        typing a pointer holder."""
        for node in walk(root_node):
            operand = self._dropped_cast_operand(node)
            if operand is None or self._is_dereferenced(node) or self._span(node).start.byte in self._kept_casts:
                continue
            unit.redundant_casts.append(RedundantCast(cast_extent=self._span(node), inner_text=self._txt(operand)))

    def _dropped_cast_operand(self, node):
        """The operand of a C-style cast to a raw `__m*` pointer or an alias pointer,
        or of a named SIMD pointer cast; else None."""
        if node.type == "cast_expression":
            type_node, value = node.child_by_field_name("type"), node.child_by_field_name("value")
            if type_node is None or value is None:
                return None
            if _SIMD_POINTER_TYPE_RE.match(" ".join(self._txt(type_node).split())):
                return value
            cast = self._simd_pointer_cast(node)
            return value if cast is not None and cast.alias is not None else None
        if node.type == "call_expression":
            cast = self._simd_pointer_cast(node)
            return cast.inner if cast is not None else None
        return None

    def _is_dereferenced(self, cast_node) -> bool:
        child, parent = cast_node, cast_node.parent
        while parent is not None and parent.type == "parenthesized_expression":
            child, parent = parent, parent.parent
        if parent is None:
            return False
        if parent.type == "subscript_expression":
            return parent.child_by_field_name("argument") == child
        return self._is_indirection(parent)

    def _declarator_info(self, node):
        """``(name node, pointer levels, array dimension texts)`` of a declarator."""
        t = node.type
        if t in ("identifier", "field_identifier", "type_identifier"):
            return node, 0, []
        if t == "init_declarator":
            return self._declarator_info(node.child_by_field_name("declarator"))
        if t == "pointer_declarator":
            name, p, dims = self._declarator_info(node.child_by_field_name("declarator"))
            return name, p + 1, dims
        if t == "reference_declarator":
            # `&`/`&&` is no pointer level; the name is a named child, not a field.
            inner = next(iter(node.named_children), None)
            return self._declarator_info(inner) if inner is not None else (None, 0, [])
        if t == "array_declarator":
            name, p, dims = self._declarator_info(node.child_by_field_name("declarator"))
            size = node.child_by_field_name("size")
            return name, p, dims + [self._txt(size) if size is not None else ""]
        return None, 0, []

    def _assemble_dtype(self, decl_node, ptr: int, dims: list[str]) -> str:
        """The declared type as `const int32_t *`, `__m512i`, `const int32_t[8]`:
        qualifiers kept, storage class dropped."""
        quals = [self._txt(c) for c in decl_node.children if c.type == "type_qualifier"]
        base_node = decl_node.child_by_field_name("type")
        base = self._txt(base_node) if base_node is not None else ""
        dtype = " ".join([*quals, base]).strip()
        if ptr:
            dtype += " " + "*" * ptr
        for d in dims:
            dtype += f"[{d}]"
        return dtype

    def _type_span(self, decl_node, declarator, ptr: int) -> SourceSpan | None:
        """The base type's span, stretched over the declarator's `*` when there is
        exactly one."""
        base_node = decl_node.child_by_field_name("type")
        if base_node is None:
            return None
        base = self._span(base_node)
        star = self._first_star(declarator) if ptr == 1 else None
        if star is None:
            return base
        ep = star.end_point
        return SourceSpan(base.start, SourcePos(ep[0] + 1, ep[1] + 1, star.end_byte), self._path)

    def _first_star(self, node):
        """The `*` of a one-level pointer declarator."""
        if node.type == "init_declarator":
            return self._first_star(node.child_by_field_name("declarator"))
        if node.type == "pointer_declarator":
            return next((c for c in node.children if c.type == "*"), None)
        return None

    def _const_int_array(self, decl_node, dims: list[str], value):
        """``(brace text, count, element bytes)`` of a `const` integer array
        initialized by plain literals, else None: conservative, so the literal it
        materializes is bit-identical to the array."""
        if value is None or value.type != "initializer_list" or len(dims) != 1:
            return None
        if not any(c.type == "type_qualifier" and self._txt(c) == "const" for c in decl_node.children):
            return None
        base_node = decl_node.child_by_field_name("type")
        elem_bytes = _INT_TYPE_BYTES.get(self._txt(base_node) if base_node is not None else "")
        if elem_bytes is None:
            return None
        elements = value.named_children
        if not elements or any(e.type != "number_literal" for e in elements):
            return None
        texts = [self._txt(e) for e in elements]
        if not all(_INT_LITERAL_RE.match(t) for t in texts):
            return None
        if dims[0].isdigit() and int(dims[0]) != len(texts):
            return None
        return "{" + ", ".join(texts) + "}", len(texts), elem_bytes
