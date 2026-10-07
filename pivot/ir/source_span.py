from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourcePos:
    line: int        # 1-indexed
    column: int      # 1-indexed
    byte: int = 0    # offset into the source buffer, which edits use


@dataclass(frozen=True)
class SourceSpan:
    """Language-neutral source range, which every front-end lowers its parser's
    node locations into."""
    start: SourcePos
    end: SourcePos
    file: str = ""

