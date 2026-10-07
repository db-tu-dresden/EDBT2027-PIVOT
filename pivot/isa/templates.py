"""The template vocabulary of primitive definitions: `{token}` placeholders and
the dtypes that bind them."""

from __future__ import annotations

import re
from typing import Iterable

# A template placeholder, `{v}` / `{dtype_suffix}`.
TOKEN_RE = re.compile(r"\{([a-z_]+)\}")

# IR tokens, bound from a target definition's dtype and register width.
VECTOR_TOKENS = frozenset({"v", "vint", "vuint", "vfloat", "vidx"})
MASK_TOKENS = frozenset({"vbool", "vbool_all"})
SCALAR_TOKENS = frozenset({"s", "sint", "suint", "sfloat", "sbool"})
IR_TOKENS = VECTOR_TOKENS | MASK_TOKENS | SCALAR_TOKENS | {"elem_bits"}
# The IR tokens swept over register widths; `{v}{128,256}` names the widths.
WIDTH_TOKENS = frozenset({"v", "vint", "vuint", "vfloat", "vbool", "vbool_all"})
WIDTH_QUALIFIED_RE = re.compile(
    rf"\{{(?P<token>{'|'.join(sorted(WIDTH_TOKENS))})\}}\{{(?P<widths>[0-9,\s]+)\}}"
)

# `{simd_<param>}`: a target's spelling of the simd type of parameter <param>.
SIMD_PARAM_PREFIX = "simd_"
SIMD_PARAM_RE = re.compile(r"\{simd_(?P<param>[a-z_][a-z0-9_]*)\}")

# A derived signature value `{elem@width}`: the element type of parameter `elem`
# at the register width of parameter `width`.
DERIVED_RE = re.compile(r"\{(?P<elem>[a-z_][a-z0-9_]*)@(?P<width>[a-z_][a-z0-9_]*)\}")

# The element bit widths of each dtype family.
DTYPE_WIDTHS: dict[str, tuple[int, ...]] = {
    "int": (8, 16, 32, 64),
    "uint": (8, 16, 32, 64),
    "float": (32, 64),
    "bool": (8, 16, 32, 64),
}
_DTYPE_RE = re.compile(r"(int|uint|float|bool)(8|16|32|64)")


def split_dtype(dtype: str | None) -> tuple[str | None, int | None]:
    """``("int", 32)`` for ``int32``; ``(None, None)`` for anything else."""
    m = _DTYPE_RE.fullmatch(dtype.strip()) if dtype is not None else None
    return (m.group(1), int(m.group(2))) if m else (None, None)


def parse_width_list(text: str | None) -> list[int]:
    """The widths of ``256``, ``{128, 256}`` or ``128, 256``."""
    if not text:
        return []
    value = text.strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    return [int(p.strip()) for p in value.split(",") if p.strip()]


def substitute(text: str, values: dict[str, str]) -> str:
    """``text`` with each ``{token}`` of ``values`` replaced by its value."""
    for token, value in values.items():
        text = text.replace("{" + token + "}", value)
    return text


def tokens_in(texts: Iterable[str]) -> set[str]:
    """The ``{token}`` names in ``texts``."""
    return {token for text in texts for token in TOKEN_RE.findall(text)}


def is_simd_param(token: str) -> bool:
    return token.startswith(SIMD_PARAM_PREFIX)


def unknown_tokens(texts: Iterable[str], slot_tokens: frozenset[str]) -> set[str]:
    """The tokens that are neither IR tokens nor a source family's ``slot_tokens``.
    Any `simd_<param>` passes; its target checks that the parameter exists."""
    return {t for t in tokens_in(texts) if t not in IR_TOKENS and t not in slot_tokens and not is_simd_param(t)}
