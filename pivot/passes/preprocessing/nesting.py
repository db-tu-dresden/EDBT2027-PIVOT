from __future__ import annotations

import re

from pivot.lang.languages import get_language_for_file
from pivot.lang.temps import TEMP_NAME_PATTERN
from pivot.passes.preprocessing.locators import Locator
from pivot.passes.translation.emit import UNKNOWN_TYPE_MARKER

_TEMP_PATTERN = re.compile(rf"\b{TEMP_NAME_PATTERN}\b")
# A string or char literal, whose parentheses and commas are text; else one of those.
_COMMA_OR_PAREN = re.compile(r""""(?:\\.|[^"\\])*"|'(?:\\.[^']*|[^'\\])'|[(),]""")
# What may follow a statement that ends its line: a `#define` body's continuation.
_LINE_END = re.compile(r"[ \t\r]*\\?[ \t\r]*")
_INDENT = re.compile(r"[ \t]*")


def nest_temps(source: str, locator: Locator) -> str:
    """``source`` with every temp inlined into its reads and its assignment removed.
    Assignments come in dependency order, so each value is expanded once.  A value
    is parenthesized unless its read fills a whole slot."""
    bare_reads = locator.bare_temp_reads(source)
    values: dict[str, str] = {}
    removed: list[tuple[int, int]] = []

    def inline(text: str, offset: int) -> str:
        def value(m: re.Match[str]) -> str:
            if (v := values.get(m.group(0))) is None:
                return m.group(0)
            return v if offset + m.start() in bare_reads and not _has_top_level_comma(v) else f"({v})"
        return _TEMP_PATTERN.sub(value, text)

    for assignment in locator.temp_assignments(source):
        line_start = source.rfind("\n", 0, assignment.start) + 1
        # An unresolved type left a Fixme marker: the temp stays, so the build
        # fails at the marked line instead of hiding a missed conversion.
        if source[line_start:assignment.start].rstrip().endswith(UNKNOWN_TYPE_MARKER):
            continue
        values[assignment.name] = inline(assignment.value, assignment.value_start)
        removed.append(_statement_span(source, line_start, assignment.start, assignment.end))

    if not values:
        return source

    pieces: list[str] = []
    cursor = 0
    for start, end in _joined(source, removed):
        pieces.append(inline(source[cursor:start], cursor))
        cursor = end
    pieces.append(inline(source[cursor:], cursor))
    return "".join(pieces)


def _statement_span(source: str, line_start: int, start: int, end: int) -> tuple[int, int]:
    """What removing the statement at ``start``..``end`` removes: its whole line
    when nothing else is on it, else the statement and, when it ends the line,
    the line break the hoist put after it."""
    newline = source.find("\n", end)
    line_end = len(source) if newline == -1 else newline
    if not _LINE_END.fullmatch(source, end, line_end):
        return start, end
    return (start if source[line_start:start].strip() else line_start), min(line_end + 1, len(source))


def _joined(source: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The disjoint ``spans``, merged where they touch.  A span from within a line
    to the start of another also takes that line's indent: the code before it
    (a `case` label, a `{`) continues with the next kept line."""
    touching: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if touching and touching[-1][1] == start:
            start = touching.pop()[0]
        touching.append((start, end))
    joined = []
    for start, end in touching:
        if start and source[start - 1] != "\n" and source[end - 1] == "\n":
            end = _INDENT.match(source, end).end()
        joined.append((start, end))
    return joined


def _has_top_level_comma(value: str) -> bool:
    """Whether a comma of ``value`` lies outside every parenthesis: bare, it would
    split an argument list, a macro's too, where braces and angles group nothing."""
    depth = 0
    for token in _COMMA_OR_PAREN.findall(value):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif token == "," and depth == 0:
            return True
    return False


def run(path: str, original: str, cxx: bool = False) -> None:
    """Inline the temporaries of ``path``, C++ code if ``cxx``, again and format
    what differs from ``original``, the user's file, in its style."""
    language = get_language_for_file(path)
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    nested = nest_temps(source, language.load_locator(path, cxx))
    with open(path, "w", encoding="utf-8") as f:
        f.write(nested)
    language.load_syntax().format_file(path, original)
