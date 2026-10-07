"""The preprocessing passes: each lifts the sites one locator finder reports into
temporaries, so the frontend sees single-assignment, variable-only calls."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from pivot.ir.edits import EditBuffer
from pivot.lang.languages import get_language_for_file
from pivot.lang.syntax import LanguageSyntax
from pivot.lang.temps import make_pivot_temp_name
from pivot.passes.preprocessing.locators import HoistSite, Locator

# Real nesting is a handful of levels deep.
_MAX_ROUNDS = 64


@dataclass(frozen=True)
class HoistPass:
    # Calls the locator's own finder, so a grammar's override applies.
    find: Callable[[Locator, str], list[HoistSite]]
    # The name of the snapshot taken after the pass.
    snapshot: str
    # Unnesting peels one level per round, until no nested call is left.
    repeats: bool = False

    def run(self, path: str) -> None:
        """Rewrites ``path`` in place."""
        language = get_language_for_file(path)
        locator, syntax = language.load_locator(path), language.load_syntax()
        with open(path, encoding="utf-8") as f:
            source = f.read()
        changed = False
        for _ in range(_MAX_ROUNDS if self.repeats else 1):
            if not (sites := self.find(locator, source)):
                break
            source = hoist_to_temp(source, sites, syntax, existing_names=locator.declared_names(source))
            changed = True
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                f.write(source)


# By the pass names of the language manifests.
HOIST_PASSES = {
    "return_normalization": HoistPass(
        lambda locator, source: locator.find_complex_returns(source), "01_return_normalized"),
    "unnesting": HoistPass(
        lambda locator, source: locator.find_nested_calls(source), "02_unnested", repeats=True),
    "argument_normalization": HoistPass(
        lambda locator, source: locator.find_complex_args(source), "03_argument_normalized"),
}


def hoist_to_temp(
    source: str, sites: list[HoistSite], syntax: LanguageSyntax, *, existing_names: set[str] | None = None,
) -> str:
    """``source`` with every site lifted into a temporary."""
    if not sites:
        return source

    buf = EditBuffer(source)
    used: set[str] = set(existing_names or ())
    counter: dict[str, int] = {}

    def fresh(kind: str) -> str:
        n = counter.get(kind, 0)
        while True:
            name = make_pivot_temp_name(kind, n)
            n += 1
            if name not in used:
                used.add(name)
                counter[kind] = n
                return name

    def assign_text(site: HoistSite, name: str) -> str:
        # Never a declaration: typing derives the temp's type from its producer,
        # and nesting inlines it again.
        return syntax.expr_statement(f"{name} = {site.expr_text}")

    value_sites = [s for s in sites if not s.is_void_return]
    void_sites = [s for s in sites if s.is_void_return]

    # The temps of one statement are inserted together, in site order.
    groups: OrderedDict[int, list[tuple[HoistSite, str]]] = OrderedDict()
    for site in value_sites:
        groups.setdefault(site.anchor_span.start.byte, []).append((site, fresh(site.kind)))

    for entries in groups.values():
        first = entries[0][0]
        anchor, indent, cont = first.anchor_span, first.indent, first.line_continuation
        assigns = [assign_text(site, name) for site, name in entries]
        if not first.requires_block:
            # Each line keeps the anchor's indent, as formatting touches only
            # changed lines; inside a `#define` it ends in the continuation.
            buf.insert_before(anchor, "".join(f"{a}{cont}\n{indent}" for a in assigns))
            for site, name in entries:
                buf.replace(site.expr_span, name)
            continue
        # One edit for the braced statement, so it overlaps none of its own replacements.
        base = anchor.start.byte
        stmt_text = _splice(buf.slice(anchor), [
            (site.expr_span.start.byte - base, site.expr_span.end.byte - base, name)
            for site, name in entries
        ])
        block = syntax.wrap_block([*assigns, stmt_text], indent)
        buf.replace(anchor, _continue_interior(block, cont))

    for site in void_sites:
        if site.requires_block:
            replacement = syntax.wrap_block(
                [syntax.expr_statement(site.expr_text), syntax.return_void()], site.indent
            )
        else:
            replacement = f"{syntax.expr_statement(site.expr_text)}\n{site.indent}{syntax.return_void()}"
        buf.replace(site.expr_span, _continue_interior(replacement, site.line_continuation))

    return buf.apply()


def _splice(text: str, edits: list[tuple[int, int, str]]) -> str:
    """``text`` with disjoint ``(start, end, replacement)`` edits applied."""
    for start, end, repl in sorted(edits, key=lambda e: e[0], reverse=True):
        text = text[:start] + repl + text[end:]
    return text


def _continue_interior(text: str, cont: str) -> str:
    """``text`` with ``cont`` ending every line but the last, which the file's
    own continuation after the replaced span ends."""
    if not cont:
        return text
    lines = text.split("\n")
    return "\n".join(f"{ln}{cont}" if i < len(lines) - 1 else ln for i, ln in enumerate(lines))
