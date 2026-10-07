"""The tree-sitter grammars of the source languages, and the reads of their trees
that the frontends, the preprocessing locators and the signature inference share."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_rust
from tree_sitter import Language, Node, Parser

from pivot.ir.source_span import SourcePos, SourceSpan

# Where a tree starts in its file as (byte, row, column), 0-based: a `#define` body
# is re-parsed on its own and placed back by its base.
FragmentBase = tuple[int, int, int]
FILE_START: FragmentBase = (0, 0, 0)


@dataclass(frozen=True)
class Grammar:
    language: Language
    # Node types that wrap a value without changing which value it is.
    wrappers: tuple[str, ...]
    # The slots an expression fills whole, needing no parentheses there, as
    # (parent type, field; None for any child).
    whole_slots: frozenset[tuple[str, str | None]] = frozenset()
    # GNU `__attribute__` derails the C/C++ grammars' error recovery.
    blanks_attributes: bool = False

    def parse(self, src: bytes) -> Node:
        """The root of ``src``'s tree; blanking keeps every offset of ``src``."""
        return Parser(self.language).parse(_blank_attributes(src) if self.blanks_attributes else src).root_node

    def unwrap(self, node: Node | None) -> Node | None:
        """The value inside ``node``'s casts and parentheses."""
        while node is not None and node.type in self.wrappers:
            inner = node.child_by_field_name("value")
            if inner is None:
                inner = next(iter(node.named_children), None)
            if inner is None:
                break
            node = inner
        return node

    def fills_slot(self, node: Node) -> bool:
        """Whether ``node`` is the whole of an argument, a return value, an
        initializer, a right-hand side or a condition."""
        parent = node.parent
        return parent is not None and any(
            parent.type == kind and (field is None or parent.child_by_field_name(field) == node)
            for kind, field in self.whole_slots
        )


_C_SLOTS = frozenset({
    ("argument_list", None), ("return_statement", None), ("init_declarator", "value"),
    ("assignment_expression", "right"), ("for_statement", "condition"),
    # C's conditions are parenthesized, as is anything the user parenthesized.
    ("parenthesized_expression", None),
})
C = Grammar(
    Language(tree_sitter_c.language()), ("cast_expression", "parenthesized_expression"),
    whole_slots=_C_SLOTS, blanks_attributes=True,
)
CPP = Grammar(
    Language(tree_sitter_cpp.language()), ("cast_expression", "parenthesized_expression"),
    whole_slots=_C_SLOTS | {("condition_clause", "value")}, blanks_attributes=True,
)
RUST = Grammar(
    Language(tree_sitter_rust.language()), ("type_cast_expression", "parenthesized_expression"),
    whole_slots=frozenset({
        ("arguments", None), ("return_expression", None), ("let_declaration", "value"),
        ("assignment_expression", "right"), ("compound_assignment_expr", "right"),
        ("parenthesized_expression", None),
        # An expression right inside a block is its tail.
        ("block", None),
    }),
)
# Primitive `direct` bodies are bare calls and assignments in every language,
# which the Rust grammar parses.
PATTERN = RUST


def c_family_grammar(path: str, cxx: bool = False) -> Grammar:
    """C for a `.c` file, else C++: a superset of C that also takes the ambiguous `.h`.
    ``cxx`` code is C++ in any file."""
    return C if path.endswith(".c") and not cxx else CPP


def walk(node: Node) -> Iterator[Node]:
    """Every named node under ``node`` in pre-order, without recursion (deep trees)."""
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.named_children))


def find(node: Node, node_type: str) -> Iterator[Node]:
    return (n for n in walk(node) if n.type == node_type)


def text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode()


def span(node: Node, path: str = "", base: FragmentBase = FILE_START) -> SourceSpan:
    return SourceSpan(_position(node.start_point, node.start_byte, base),
                      _position(node.end_point, node.end_byte, base), path)


def _position(point: tuple[int, int], byte: int, base: FragmentBase) -> SourcePos:
    # A fragment's first row continues the base's column; later rows own theirs.
    base_byte, base_row, base_col = base
    row, col = point
    return SourcePos(base_row + row + 1, (base_col + col if row == 0 else col) + 1, base_byte + byte)


def callee_name(call: Node, src: bytes) -> str:
    """The name a call calls; a qualified `ns::f` or scoped `core::arch::…::f` gives `f`."""
    fn = call.child_by_field_name("function")
    if fn is None:
        return ""
    if fn.type in ("qualified_identifier", "scoped_identifier"):
        fn = fn.child_by_field_name("name") or fn
    return text(fn, src)


def macro_bodies(root: Node, src: bytes, base: FragmentBase = FILE_START) -> list[tuple[bytes, FragmentBase]]:
    """Each `#define` body under ``root`` with its base.  tree-sitter keeps a body as
    one `preproc_arg` token, so its code is only seen by re-parsing it."""
    bodies = []
    for node in walk(root):
        if node.type not in ("preproc_function_def", "preproc_def"):
            continue
        body = node.child_by_field_name("value")
        if body is not None and body.type == "preproc_arg":
            start = span(body, base=base).start
            bodies.append((src[body.start_byte:body.end_byte], (start.byte, start.line - 1, start.column - 1)))
    return bodies


_ATTRIBUTE = b"__attribute__"


def _blank_attributes(src: bytes) -> bytes:
    """``src`` with every ``__attribute__((...))`` replaced by spaces, newlines kept,
    so the spans of its tree still index ``src``."""
    if _ATTRIBUTE not in src:
        return src
    out = bytearray(src)
    i = 0
    while (start := src.find(_ATTRIBUTE, i)) != -1:
        j = start + len(_ATTRIBUTE)
        while j < len(src) and src[j:j + 1].isspace():
            j += 1
        if src[j:j + 1] != b"(":
            i = j
            continue
        depth = 0
        while j < len(src):
            depth += {0x28: 1, 0x29: -1}.get(src[j], 0)
            j += 1
            if depth == 0:
                break
        for k in range(start, j):
            if out[k] != 0x0A:
                out[k] = 0x20
        i = j
    return bytes(out)
