from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from pivot.lang.languages import get_language
from pivot.ir.types import Domain, ElemBits

if TYPE_CHECKING:
    pass

# Macro tokens used in intrinsic_type_hints_x86.yaml patterns.
_X86_WIDTHS_REGEX = r"(?:256|512)?"
_HINT_REGEX = (
    r"(?:ep[iu](?:8|16|32|64)"
    r"|ps|pd|ss|sd"
    r"|f(?:32|64)x\d+"
    r"|i(?:8|16|32|64)"
    r"|si(?:128|256|512|64|32|16|8))"
)
# Trailing (N) in a pattern string constrains the match to intrinsics with exactly N arguments.
_ARG_COUNT_RE = re.compile(r"\((\d+)\)$")

# The dtype hints in an intrinsic name.
_DTYPE_HINTS = re.compile(
    r"ep[iu](?:8|16|32|64)(?=$|_)"
    r"|_(?:ps|pd|ss|sd)(?=$|_)"             # _ps/_pd and scalar-float _ss/_sd
    r"|(?:ps|pd|ss|sd)(?=_)"                # source-type embedded in verb: cvtps_, cvtss_
    r"|_f(?:32|64)x\d+"                      # _f32x4, _f64x4 (tile broadcast dtypes)
    r"|_i(?:8|16|32|64)(?!\d)"
    r"|(?<![a-zA-Z0-9])si(?:128|256|512|64|32|16|8)(?!\d)"
)

_ELEM_BITS_MAP: dict[int, ElemBits] = {
    8: ElemBits.BITS_8,
    16: ElemBits.BITS_16,
    32: ElemBits.BITS_32,
    64: ElemBits.BITS_64,
}


def _hint_to_domain_bits(hint: str) -> tuple[Domain | None, ElemBits] | None:
    """Convert one raw hint token to (Domain, ElemBits).

    Width-only hints (_i32, _i64: gather/scatter index widths) yield
    (None, bits): the element width is known but the domain (int/uint) is not,
    so the node is typed from the variable declaration or the int32 default.
    si* hints carry neither domain nor element width and yield None entirely.
    """
    m = re.match(r"ep([iu])(\d+)$", hint)
    if m:
        bits = _ELEM_BITS_MAP.get(int(m.group(2)))
        domain = Domain.INT if m.group(1) == "i" else Domain.UINT
        return (domain, bits) if bits else None
    if hint in ("_ps", "_ss", "ps", "ss"):
        return Domain.FLOAT, ElemBits.BITS_32
    if hint in ("_pd", "_sd", "pd", "sd"):
        return Domain.FLOAT, ElemBits.BITS_64
    m = re.match(r"_f(32|64)x\d+$", hint)
    if m:
        bits = _ELEM_BITS_MAP.get(int(m.group(1)))
        return (Domain.FLOAT, bits) if bits else None
    m = re.match(r"_i(\d+)$", hint)
    if m:
        bits = _ELEM_BITS_MAP.get(int(m.group(1)))
        return (None, bits) if bits else None
    # si* are weak width-only hints with no element type.
    return None


class IntrinsicTypeHintRegistry:
    """Regex-based registry mapping multi-dtype intrinsic names to positional type annotations.

    Loaded from a YAML file.  Each entry has a regex pattern and a mapping from
    positional keys ("ret", "arg0", "arg1", ...) to THN references (TH1, TH2, ...).
    The N-th hint is extracted left-to-right from the intrinsic name.
    """

    def __init__(self, yaml_path: Path) -> None:
        self._compiled: list[tuple[re.Pattern, dict[str, str], int | None]] = []
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        for entry in raw.get("hints", []):
            raw_pat = entry["pattern"]
            arg_count: int | None = None
            if m := _ARG_COUNT_RE.search(raw_pat):
                arg_count = int(m.group(1))
                raw_pat = raw_pat[:m.start()]
            raw_pat = raw_pat.replace("{x86_widths}", _X86_WIDTHS_REGEX)
            raw_pat = raw_pat.replace("{hint}", _HINT_REGEX)
            mapping = {k: v for k, v in entry.items() if k != "pattern"}
            self._compiled.append((re.compile(raw_pat), mapping, arg_count))

    def lookup(self, intrinsic_name: str, arg_count: int | None = None) -> dict[str, tuple[Domain | None, ElemBits]] | None:
        """Return positional annotation for intrinsic_name, or None if no entry matches.

        Keys: "ret", "arg0", "arg1", ...
        Values: (Domain | None, ElemBits) pairs derived from the name's dtype
        hints.  Domain is None for width-only hints (_i32 etc.).
        arg_count, when provided, skips entries whose (N) constraint differs.
        """
        for pattern, mapping, expected_args in self._compiled:
            if expected_args is not None and arg_count != expected_args:
                continue
            if not pattern.match(intrinsic_name):
                continue
            raw_hints = _DTYPE_HINTS.findall(intrinsic_name)
            th: list[tuple[Domain | None, ElemBits] | None] = [
                _hint_to_domain_bits(h) for h in raw_hints
            ]
            result: dict[str, tuple[Domain | None, ElemBits]] = {}
            for pos_key, th_ref in mapping.items():
                if not (isinstance(th_ref, str) and th_ref.startswith("TH")):
                    continue
                try:
                    idx = int(th_ref[2:]) - 1  # TH1 → index 0
                except ValueError:
                    continue
                if 0 <= idx < len(th) and th[idx] is not None:
                    result[pos_key] = th[idx]
            return result if result else None
        return None


_registry: IntrinsicTypeHintRegistry | None = None


def get_intrinsic_type_hint_registry() -> IntrinsicTypeHintRegistry:
    """Return the shared singleton registry, loading from the configured intrinsics dir."""
    global _registry
    if _registry is None:
        intrinsics_dir = get_language().intrinsics_dir
        yaml_path = Path(intrinsics_dir) / "intrinsic_type_hints_x86.yaml"
        _registry = IntrinsicTypeHintRegistry(yaml_path)
    return _registry
