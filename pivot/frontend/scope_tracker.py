"""Lexical scopes for the tree-sitter frontends: a stack of symbol tables, so a
name resolves to the innermost visible ``VariableDecl``."""
from __future__ import annotations

from pivot.frontend.base import ParsedUnit
from pivot.ir.source_span import SourceSpan
from pivot.frontend.types import VariableDecl


class ScopeTracker:
    def __init__(self) -> None:
        self._frames: list[dict[str, VariableDecl]] = []

    def push(self) -> None:
        """Enter a scope: a function body (its parameters' scope) or a block."""
        self._frames.append({})

    def pop(self) -> None:
        self._frames.pop()

    def declare(self, name: str, span: SourceSpan, dtype: str, kind: str,
                unit: ParsedUnit, *, type_extent=None,
                const_init=None) -> VariableDecl:
        """Create a VariableDecl, register it in the current scope and in the
        ParsedUnit, and return it."""
        decl = VariableDecl(
            name=name, dtype=dtype, extent=span, kind=kind, type_extent=type_extent,
            const_init_text=const_init[0] if const_init else None,
            const_elem_count=const_init[1] if const_init else None,
            const_elem_bytes=const_init[2] if const_init else None,
        )
        if self._frames:
            self._frames[-1][name] = decl
        unit.variable_decls.append(decl)
        return decl

    def lookup(self, name: str) -> VariableDecl | None:
        """The innermost declaration of ``name``, or None."""
        for symbols in reversed(self._frames):
            if name in symbols:
                return symbols[name]
        return None

    def resolve(self, name: str, span: SourceSpan, unit: ParsedUnit) -> VariableDecl:
        """The innermost declaration of ``name``; an unknown name gets an
        ``auto``/``param`` declaration in the current scope (an extern or forward reference, or
        a free input of the pattern DSL)."""
        decl = self.lookup(name)
        return decl if decl is not None else self.declare(name, span, "auto", "param", unit)
