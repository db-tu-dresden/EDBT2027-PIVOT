"""Where the preprocessing passes hoist: the expressions a frontend does not lower
in place, found by one algorithm over each grammar's node types; and the temps'
assignments and reads, which nesting inlines again."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Iterator, NamedTuple

from tree_sitter import Node

from pivot.ir.source_span import SourceSpan
from pivot.isa.intrinsic_registry import get_intrinsic_registry
from pivot.lang.grammars import (
    FILE_START, RUST, FragmentBase, Grammar, c_family_grammar, callee_name, macro_bodies, span, text, walk,
)
from pivot.lang.temps import is_pivot_temp_name

_CALL = "call_expression"


@dataclass
class HoistSite:
    """An expression to lift into a temporary assigned before its statement."""

    # Replaced by the temp's name; its text becomes the temp's value.
    expr_span: SourceSpan
    expr_text: str
    # The statement the assignment is inserted before.
    anchor_span: SourceSpan
    indent: str = ""
    # The anchor is an unbraced body (`if (c) f(x);`): the insertion needs braces.
    requires_block: bool = False
    # Names the temp `pivot_tmp_<kind>_<n>`.
    kind: str = "tmp"
    # A void function's `return expr;`, spanned whole, becomes `expr; return;`.
    is_void_return: bool = False
    # Ends every inserted line: ` \` inside a `#define` body.
    line_continuation: str = ""


class TempAssignment(NamedTuple):
    """A temp's statement ``name = value;``, by character offsets of the file."""

    name: str
    start: int
    end: int
    value: str
    value_start: int


class _Tree(NamedTuple):
    """The file's tree, or a macro body's placed at ``base``."""

    root: Node
    src: bytes
    base: FragmentBase
    continuation: str


class Locator:
    """Finds a file's hoist sites.  A grammar's subclass names its node types and
    its few genuine differences."""

    _grammar: Grammar
    # Statements a temp is inserted before.
    STATEMENTS: frozenset[str] = frozenset()
    # Statements whose body may be one unbraced statement: inserting before it needs braces.
    UNBRACED_BODY_PARENTS: frozenset[str] = frozenset()
    # Statements whose condition the frontend does not lower.
    CONDITION_STATEMENTS: frozenset[str] = frozenset()
    # Expressions whose operands the frontend does not lower (arithmetic, `?:`, an index).
    OPERAND_PARENTS: frozenset[str] = frozenset()
    RETURN = ""
    # What stays in place when returned or passed to an intrinsic.
    TRIVIAL_RETURNS: frozenset[str] = frozenset()
    TRIVIAL_ARGUMENTS: frozenset[str] = frozenset()

    def __init__(self, path: str, cxx: bool = False) -> None:
        """``cxx``: the code is C++ whatever ``path``'s extension (a C++ target's output)."""
        self._path = path
        self._intr_reg = get_intrinsic_registry()

    def declared_names(self, source: str) -> set[str]:
        """The temp names in ``source``, which a new temp must not reuse."""
        return {text(node, tree.src) for node, tree in self._temps(source.encode())}

    def bare_temp_reads(self, source: str) -> set[int]:
        """The character offsets of the temp reads that fill a whole slot (an
        argument, a return value, a condition …): an inlined value needs no
        parentheses there."""
        src = source.encode()
        char = _char_offsets(src, source)
        reads = {tree.base[0] + node.start_byte for node, tree in self._temps(src) if self._grammar.fills_slot(node)}
        return {char(byte) for byte in reads}

    def temp_assignments(self, source: str) -> list[TempAssignment]:
        """Every statement ``t = value;`` that assigns a temp, in source order."""
        src = source.encode()
        char = _char_offsets(src, source)
        found = []
        for node, tree in self._temps(src):
            assignment = node.parent
            if assignment.type != "assignment_expression" or assignment.child_by_field_name("left") != node:
                continue
            # C names the operator, `+=` too; Rust's `=` has no field (`+=` is another node).
            operator = assignment.child_by_field_name("operator")
            statement, value = assignment.parent, assignment.child_by_field_name("right")
            if statement.type != "expression_statement" or value is None:
                continue
            if operator is not None and operator.type != "=":
                continue
            base = tree.base[0]
            found.append(TempAssignment(
                text(node, tree.src), char(base + statement.start_byte), char(base + statement.end_byte),
                text(value, tree.src), char(base + value.start_byte),
            ))
        return sorted(found, key=lambda a: a.start)

    def find_complex_returns(self, source: str) -> list[HoistSite]:
        return self._sites(source, self._return_sites)

    def find_nested_calls(self, source: str) -> list[HoistSite]:
        return self._sites(source, self._nested_call_sites)

    def find_complex_args(self, source: str) -> list[HoistSite]:
        return self._sites(source, self._complex_arg_sites)

    def _fragments(self, root: Node, src: bytes, base: FragmentBase) -> list[tuple[bytes, FragmentBase]]:
        """Code the tree holds as one token, re-parsed on its own."""
        return []

    def _returns_void(self, statement: Node) -> bool:
        return False

    def _tail_value(self, node: Node) -> Node | None:
        """The value ``node`` returns without a return statement."""
        return None

    def _trees(self, src: bytes, base: FragmentBase = FILE_START, continuation: str = "") -> Iterator[_Tree]:
        root = self._grammar.parse(src)
        yield _Tree(root, src, base, continuation)
        for body, body_base in self._fragments(root, src, base):
            yield from self._trees(body, body_base, " \\")

    def _temps(self, src: bytes) -> Iterator[tuple[Node, _Tree]]:
        for tree in self._trees(src):
            for node in walk(tree.root):
                if node.type == "identifier" and is_pivot_temp_name(text(node, tree.src)):
                    yield node, tree

    def _sites(self, source: str, sites_in: Callable[[_Tree], list[HoistSite]]) -> list[HoistSite]:
        return [site for tree in self._trees(source.encode()) for site in sites_in(tree)]

    def _return_sites(self, tree: _Tree) -> list[HoistSite]:
        sites = []
        for node in walk(tree.root):
            if node.type == self.RETURN and node.named_children:
                value = node.named_children[0]
                if self._grammar.unwrap(value).type not in self.TRIVIAL_RETURNS:
                    site = self._site(tree, value, node, "ret")
                    if self._returns_void(node):
                        # `return f(x);` becomes `f(x); return;`.
                        site = replace(site, expr_span=site.anchor_span, is_void_return=True)
                    sites.append(site)
            elif (tail := self._tail_value(node)) is not None:
                if self._grammar.unwrap(tail).type not in self.TRIVIAL_RETURNS:
                    # No statement to insert before: the tail becomes a block that ends in the temp.
                    sites.append(self._site(tree, tail, tail, "ret", needs_braces=True))
        return sites

    def _nested_call_sites(self, tree: _Tree) -> list[HoistSite]:
        """Intrinsic calls passed to a call, controlling a condition statement or
        buried in an operand.  Only the innermost are kept, so the edits of one
        round never overlap and each round peels one level."""
        sites = []
        # Arguments, call by call.
        for call in walk(tree.root):
            args = call.child_by_field_name("arguments") if call.type == _CALL else None
            for arg in args.named_children if args is not None else ():
                inner = self._grammar.unwrap(arg)
                if inner.type == _CALL and self._hoistable(callee_name(inner, tree.src)):
                    if (statement := self._statement_of(inner)) is not None:
                        sites.append(self._site(tree, inner, statement, "unnest"))
        # Conditions; nesting inlines the call again, so a loop still evaluates it per iteration.
        for call in walk(tree.root):
            if call.type != _CALL or not self._hoistable(callee_name(call, tree.src)):
                continue
            statement = self._statement_of(call)
            if statement is not None and statement.type in self.CONDITION_STATEMENTS and not self._inside_call(call):
                sites.append(self._site(tree, call, statement, "cond"))
        # Operands.
        for call in walk(tree.root):
            if call.type != _CALL or not self._hoistable(callee_name(call, tree.src)):
                continue
            parent = call.parent
            while parent is not None and parent.type in self._grammar.wrappers:
                parent = parent.parent
            if parent is None or parent.type not in self.OPERAND_PARENTS:
                continue
            statement = self._statement_of(call)
            if statement is not None and statement.type not in self.CONDITION_STATEMENTS:
                sites.append(self._site(tree, call, statement, "unnest"))
        return _by_containment(sites, keep_inner=True)

    def _complex_arg_sites(self, tree: _Tree) -> list[HoistSite]:
        """Intrinsic arguments that are neither a variable nor a literal (literals
        stay: many parameters are immediates).  Only the outermost are kept."""
        sites = []
        for call in walk(tree.root):
            if call.type != _CALL or not self._intr_reg.is_typed_intrinsic(callee_name(call, tree.src)):
                continue
            args = call.child_by_field_name("arguments")
            statement = self._statement_of(call)
            if args is None or statement is None:
                continue
            for arg in args.named_children:
                value = self._grammar.unwrap(arg)
                if value.type in self.TRIVIAL_ARGUMENTS:
                    continue
                # A fold pseudo-intrinsic (`_MM_SHUFFLE`) is an immediate the frontend folds.
                if value.type == _CALL and self._intr_reg.is_fold_intrinsic(callee_name(value, tree.src)):
                    continue
                sites.append(self._site(tree, arg, statement, "arg"))
        return _by_containment(sites, keep_inner=False)

    def _hoistable(self, name: str) -> bool:
        return self._intr_reg.is_typed_intrinsic(name) and not self._intr_reg.is_fold_intrinsic(name)

    def _statement_of(self, node: Node) -> Node | None:
        while node is not None and node.type not in self.STATEMENTS:
            node = node.parent
        return node

    @staticmethod
    def _inside_call(node: Node) -> bool:
        while (node := node.parent) is not None:
            if node.type == _CALL:
                return True
        return False

    def _site(self, tree: _Tree, expr: Node, anchor: Node, kind: str, needs_braces: bool | None = None) -> HoistSite:
        if needs_braces is None:
            needs_braces = anchor.parent is not None and anchor.parent.type in self.UNBRACED_BODY_PARENTS
        return HoistSite(
            expr_span=span(expr, self._path, tree.base),
            expr_text=text(expr, tree.src),
            anchor_span=span(anchor, self._path, tree.base),
            indent=" " * anchor.start_point[1],
            requires_block=needs_braces,
            kind=kind,
            line_continuation=tree.continuation,
        )


def _char_offsets(src: bytes, source: str) -> Callable[[int], int]:
    """Maps a byte offset of ``src``, ``source`` encoded, to its character offset."""
    if len(src) == len(source):
        return lambda byte: byte
    return lambda byte: len(src[:byte].decode())


def _by_containment(sites: list[HoistSite], keep_inner: bool) -> list[HoistSite]:
    """The sites that contain no other site (``keep_inner``) or lie in none.  Spans
    are tree nodes, so any two nest or are disjoint: one sweep by start, widest
    first, with a stack of open spans finds every pair.  Identical spans contain
    each other, so every copy goes."""
    spans = [(s.expr_span.start.byte, s.expr_span.end.byte) for s in sites]
    duplicates = Counter(spans)
    keep = [duplicates[s] == 1 for s in spans]
    open_spans: list[int] = []
    for i in sorted(range(len(sites)), key=lambda i: (spans[i][0], -spans[i][1])):
        while open_spans and spans[open_spans[-1]][1] < spans[i][0]:
            open_spans.pop()
        if open_spans:
            keep[open_spans[-1] if keep_inner else i] = False
        open_spans.append(i)
    return [site for site, kept in zip(sites, keep) if kept]


class CppLocator(Locator):
    STATEMENTS = frozenset({
        "declaration", "expression_statement", "return_statement", "if_statement", "for_statement",
        "while_statement", "do_statement", "switch_statement", "compound_statement",
        "labeled_statement", "case_statement",
    })
    UNBRACED_BODY_PARENTS = frozenset({"if_statement", "for_statement", "while_statement", "do_statement", "else_clause"})
    CONDITION_STATEMENTS = frozenset({"if_statement", "while_statement", "for_statement", "do_statement", "switch_statement"})
    OPERAND_PARENTS = frozenset({"binary_expression", "conditional_expression", "unary_expression", "subscript_expression"})
    RETURN = "return_statement"
    TRIVIAL_RETURNS = frozenset({
        "identifier", "number_literal", "char_literal", "string_literal", "concatenated_string",
        "raw_string_literal", "true", "false", "null",
    })
    TRIVIAL_ARGUMENTS = frozenset({"identifier", "number_literal", "char_literal", "true", "false", "null"})

    def __init__(self, path: str, cxx: bool = False) -> None:
        super().__init__(path)
        self._grammar = c_family_grammar(path, cxx)

    _fragments = staticmethod(macro_bodies)

    def _returns_void(self, statement: Node) -> bool:
        """Whether the function around ``statement`` returns `void`: no temp can hold its value."""
        func = statement.parent
        while func is not None and func.type != "function_definition":
            func = func.parent
        if func is None:
            return False
        type_node = func.child_by_field_name("type")
        if type_node is None or type_node.text.decode().strip() != "void":
            return False
        declarator = func.child_by_field_name("declarator")
        return declarator is None or declarator.type != "pointer_declarator"


class RustLocator(Locator):
    _grammar = RUST
    STATEMENTS = frozenset({"let_declaration", "expression_statement"})
    # Every body is braced, and a condition is an expression: its call is an operand.
    OPERAND_PARENTS = frozenset({"binary_expression", "unary_expression", "index_expression", "range_expression"})
    RETURN = "return_expression"
    TRIVIAL_RETURNS = TRIVIAL_ARGUMENTS = frozenset({"identifier", "integer_literal", "float_literal", "boolean_literal"})

    def _tail_value(self, node: Node) -> Node | None:
        """The last expression of a function body, which the function returns."""
        if node.type != "block" or node.parent is None or node.parent.type != "function_item":
            return None
        last = None
        for child in node.named_children:
            if child.type not in ("line_comment", "block_comment"):
                last = child
        return None if last is None or last.type in self.STATEMENTS else last
