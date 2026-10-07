"""Expands a primitive definition into its concrete variants: its dtype
expression first, then per dtype the expander of its isa."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pivot.lang.backends import build_emitter_registry, source_expanders
from pivot.isa.templates import DTYPE_WIDTHS, parse_width_list, unknown_tokens


@dataclass(frozen=True)
class ExpansionRequest:
    """The primitive whose definitions are expanded."""

    primitive: str
    inputs: list[str]
    output: str | None
    allowed_dtypes: frozenset[str]
    # Parameter -> the shape classes its target signatures declare; a source
    # family checks its swept intrinsics against it.  None disables the check.
    io_shape: dict[str, set[str]] | None


def note_dropped(primitive: str, isa: str, detail: str) -> None:
    print(f"[DefinitionExpander] Dropped definition: primitive={primitive}, isa={isa}, {detail}")


# `{int, uint}`, `{int, uint}32`, `{int, uint}{8, 16}`; and `int{8, 16}`.
_FAMILY_SET_RE = re.compile(r"\{\s*([^{}]+?)\s*\}(\d+|\{\s*[^{}]+?\s*\})?")
_FAMILY_WIDTHS_RE = re.compile(r"(int|uint|float|bool)(\{\s*[^{}]+?\s*\})")


def expand_dtypes(expr: str | None, allowed_dtypes: frozenset[str]) -> list[str | None]:
    """The allowed dtypes a dtype expression names: one dtype, a family (`int`),
    or a family set and width list as above.  No expression gives the one dtype None."""
    expr = (expr or "").strip()
    if not expr:
        return [None]
    widths: list[int] | None = None
    if expr in DTYPE_WIDTHS:
        families = [expr]
    elif m := _FAMILY_SET_RE.fullmatch(expr):
        families = [p.strip() for p in m.group(1).split(",") if p.strip()]
        widths = parse_width_list(m.group(2)) if m.group(2) else None
    elif m := _FAMILY_WIDTHS_RE.fullmatch(expr):
        families = [m.group(1)]
        widths = parse_width_list(m.group(2))
    else:
        return [expr] if expr in allowed_dtypes else []

    expanded: list[str] = []
    for family in families:
        if family not in DTYPE_WIDTHS:
            continue
        valid = DTYPE_WIDTHS[family]
        selected = valid if widths is None else [w for w in widths if w in valid]
        expanded.extend(f"{family}{w}" for w in selected if f"{family}{w}" in allowed_dtypes)
    return list(dict.fromkeys(expanded))


def _template_texts(definition: dict[str, Any]) -> list[str]:
    include = definition.get("include") or {}
    return [
        *(str(v) for v in (definition.get("signature") or {}).values()),
        *definition["direct"],
        *(str(line) for line in include.get("direct", [])),
    ]


class DefinitionExpander:
    """Dispatches by isa label to a source family's expander or a target
    emitter's, as the active language's manifest names them."""

    def __init__(self) -> None:
        sources = source_expanders()
        self._expanders: dict[str, Any] = {
            **sources,
            **{label: emitter.definition_expander for label, emitter in build_emitter_registry().items()},
        }
        self._slot_tokens = frozenset(token for source in sources.values() for token in source.slot_tokens)

    def expand(self, definition: dict[str, Any], request: ExpansionRequest) -> list[dict[str, Any]]:
        isa = definition["isa"]
        if unknown := unknown_tokens(_template_texts(definition), self._slot_tokens):
            note_dropped(request.primitive, isa, f"unknown_tokens={sorted(unknown)}")
            return []
        dtypes = expand_dtypes(definition["dtype"], request.allowed_dtypes)
        if not dtypes:
            note_dropped(request.primitive, isa, f"dtype_expr={definition['dtype']!r} expanded to no concrete dtypes")
            return []
        concrete = [{**definition, "dtype": dtype} for dtype in dtypes]
        if (expander := self._expanders.get(isa)) is None:
            return concrete
        return [variant for c in concrete for variant in expander.expand(c, request)]
