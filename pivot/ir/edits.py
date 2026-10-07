"""Source edits: planned changes, the coverage rule among them, and the
byte-range buffer that applies them."""
from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pivot.ir.source_span import SourcePos, SourceSpan


class ChangeType(Enum):
    REMOVE = "remove"
    REPLACE = "replace"
    INSERT_BEFORE = "insert_before"


@dataclass
class Change:
    """One planned source modification."""

    change_type: ChangeType
    location: Optional[SourcePos] = None
    extent: Optional[SourceSpan] = None
    text: str = ""

    @property
    def sort_key(self) -> tuple:
        """Position of the change: its insertion point, else its extent's start."""
        if self.location:
            return self.location.line, self.location.column
        return self.extent.start.line, self.extent.start.column

    def __str__(self):
        if self.location:
            return f"{self.change_type.value}: {self.text} @ {self.location.line}:{self.location.column}"
        return f"{self.change_type.value}: {self.text}"


class ChangePlan:
    """The changes of one file in planning order.  A candidate whose span lies
    within a change planned before it is dropped: that change already rewrites
    the span."""

    def __init__(self) -> None:
        self.changes: list[Change] = []
        # The planned extents as (start, end) positions, in sorted blocks of
        # halving size (merged like a binary counter), each block with the
        # running max of its ends.
        self._blocks: list[tuple[list, list, list]] = []

    def add(self, span: SourceSpan | None, *changes: Change) -> bool:
        """Plan ``changes`` for ``span`` unless a planned change covers it."""
        if span is not None and self.covers(span):
            return False
        self.changes.extend(changes)
        for change in changes:
            if change.extent:
                self._index((_position(change.extent.start), _position(change.extent.end)))
        return True

    def covers(self, span: SourceSpan) -> bool:
        """Whether ``span`` lies within a planned change (inclusive)."""
        start, end = _position(span.start), _position(span.end)
        for _, starts, max_ends in self._blocks:
            i = bisect.bisect_right(starts, start) - 1
            if i >= 0 and max_ends[i] >= end:
                return True
        return False

    def _index(self, extent: tuple) -> None:
        extents = [extent]
        while self._blocks and len(self._blocks[-1][0]) <= len(extents):
            extents = sorted(self._blocks.pop()[0] + extents)
        starts = [start for start, _ in extents]
        self._blocks.append((extents, starts, list(itertools.accumulate((end for _, end in extents), max))))


def _position(pos: SourcePos) -> tuple[int, int]:
    return pos.line, pos.column


class EditBuffer:
    """A source buffer plus (byte-range -> replacement) edits, applied all at
    once.  Byte offsets come from the frontends' SourceSpans."""

    def __init__(self, source: str | bytes) -> None:
        self._src = source.encode() if isinstance(source, str) else source
        self._edits: list[tuple[int, int, bytes]] = []  # (start, end, text)

    def _byte(self, loc) -> int:
        return loc.byte

    def replace(self, span: SourceSpan, text: str) -> None:
        self._edits.append((self._byte(span.start), self._byte(span.end), text.encode()))

    def insert_before(self, span: SourceSpan, text: str) -> None:
        self._edits.append((self._byte(span.start), self._byte(span.start), text.encode()))

    def slice(self, span: SourceSpan) -> str:
        """Original text under a span."""
        return self._src[self._byte(span.start):self._byte(span.end)].decode()

    def apply(self) -> str:
        # Two planning stages may emit the same edit; collapse duplicates.
        seen: set[tuple[int, int, bytes]] = set()
        unique_edits: list[tuple[int, int, bytes]] = []
        for edit in self._edits:
            if edit not in seen:
                seen.add(edit)
                unique_edits.append(edit)

        ordered = sorted(unique_edits, key=lambda e: (e[0], e[1]))
        last_end = -1
        for start, end, _ in ordered:
            if start < last_end:
                raise ValueError(f"overlapping edits near byte {start}")
            if end > start:
                last_end = end

        # Back-to-front, so earlier offsets stay valid.  At one offset a removal
        # goes before a zero-width insert (whose text it would otherwise eat);
        # inserts sharing an offset keep their order (stable sort).
        out = self._src
        for start, end, text in sorted(unique_edits, key=lambda e: (e[0], e[1]), reverse=True):
            out = out[:start] + text + out[end:]
        return out.decode()


def apply_changes(edit_buffer: EditBuffer, changes: list[Change]) -> None:
    """Record ``changes`` in ``edit_buffer``, bottom to top."""
    for change in sorted(changes, key=lambda c: c.sort_key, reverse=True):
        span = change.extent
        if not span and change.location:
            span = SourceSpan(start=change.location, end=change.location)

        if change.change_type == ChangeType.REMOVE:
            edit_buffer.replace(span, "")
            continue
        if change.change_type == ChangeType.REPLACE:
            edit_buffer.replace(span, change.text)
            continue
        if change.change_type == ChangeType.INSERT_BEFORE:
            edit_buffer.insert_before(span, change.text)
