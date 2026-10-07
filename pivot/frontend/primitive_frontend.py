from __future__ import annotations

from pivot.frontend.base import ParsedUnit
from pivot.frontend.cpp_frontend import c_int_literal_value
from pivot.frontend.rust_frontend import RustFrontend, rust_int_literal_value
from pivot.lang.grammars import PATTERN


class PrimitiveFrontend(RustFrontend):
    """Front-end for PIVOT's pseudo-code pattern DSL, used offline to build the
    pattern graphs.

    The DSL is bare calls and assignments (``name(args);``, ``x = name(args);``,
    integer literals, identifiers), one grammar for every translated language.
    The Rust grammar accepts it, so RustFrontend's lowering applies unchanged,
    and its ScopeTracker resolves an unknown identifier to a param
    ``VariableDecl``, which is what makes a pattern's free inputs ``variable``
    graph nodes.  Only :meth:`parse` differs: patterns are top-level statements,
    and any ERROR node is rejected.
    """

    @staticmethod
    def _int_literal_value(text: str) -> int | None:
        # Patterns spell C intrinsic code, so a leading 0 is octal as in C; the
        # Rust-only spellings (`0o17`, `1_000`, `17u32`) are read as well.
        value = c_int_literal_value(text)
        return value if value is not None else rust_int_literal_value(text)

    def parse(self, path: str) -> ParsedUnit:
        src = self._begin(path)
        root = PATTERN.parse(src)
        if root.has_error:
            raise ValueError(
                f"Pattern definition {path!r} contains disallowed syntax "
                f"(only assignments and call statements are permitted): {src.decode()!r}"
            )
        unit = ParsedUnit(path=path)
        # One scope hosts every pattern local, and ScopeTracker.resolve
        # fabricates a param decl for each free input.
        self._scope.push()
        for stmt in root.named_children:
            self._lower_stmt(stmt, unit)
        return unit

    def _nested_call_arg(self, call_node, unit: ParsedUnit):
        """Flatten a nested call now, as no unnesting pass runs on patterns: its own
        statement feeds a fresh temp, a graph node the outer call depends on."""
        name = self._callee_name(call_node)
        tmp = self._scope.declare(
            f"_tmp_{len(unit.variable_decls)}", self._span(call_node),
            self._output_type(name), "var", unit,
        )
        nested = self._build_call(call_node, unit, defines=tmp)
        unit.call_statements.append(nested)
        return tmp
