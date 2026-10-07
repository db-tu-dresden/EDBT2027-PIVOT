"""Expands a source-family (x86, arm) definition over the slots its intrinsic
names carry, keeping each variant whose intrinsics exist; its signature is read
off them."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from pivot.isa.definition_expander import note_dropped
from pivot.isa.intrinsic_registry import get_intrinsic_registry
from pivot.isa.signature_inference import infer_signature
from pivot.isa.templates import DTYPE_WIDTHS, substitute

if TYPE_CHECKING:
    from pivot.isa.definition_expander import ExpansionRequest


@dataclass(frozen=True)
class Slot:
    """A `{token}` of intrinsic names and the values it sweeps."""

    token: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class SourceFamily:
    """How a family's intrinsic names spell a variant."""

    width: Slot
    # The tokens a dtype spells, and each dtype's values for them in sweep order.
    dtype_tokens: tuple[str, ...]
    dtypes: dict[str, dict[str, str]]
    # Per width, a spelling with no element type that a dtype sweep also tries.
    registers: dict[str, dict[str, str]] = field(default_factory=dict)
    # Slots whose values spell one operation: only the first value's miss is noted.
    aliases: tuple[Slot, ...] = ()


class SourceFamilyExpander:
    """Sweeps the width, dtype and alias slots a definition's body uses.  With no
    named dtype every dtype the family spells is tried, quietly."""

    def __init__(self, family: SourceFamily) -> None:
        self._family = family
        # The `{token}`s this family's intrinsic names are spelled with.
        self.slot_tokens = frozenset({
            family.width.token, *family.dtype_tokens, *(slot.token for slot in family.aliases),
        })

    def expand(self, definition: dict[str, Any], request: ExpansionRequest) -> list[dict[str, Any]]:
        family = self._family
        isa, dtype, lines = definition["isa"], definition["dtype"], definition["direct"]
        body = "\n".join(lines)
        intrinsics = get_intrinsic_registry()

        def values(slot: Slot) -> tuple[str, ...]:
            return slot.values if "{" + slot.token + "}" in body else ("",)

        sweeping = dtype is None and any("{" + t + "}" in body for t in family.dtype_tokens)
        alias_tokens = [slot.token for slot in family.aliases]

        expanded: list[dict[str, Any]] = []
        for width in values(family.width):
            for cand_dtype, spelling in self._dtype_choices(dtype, sweeping, width, request.allowed_dtypes):
                for alias_values in itertools.product(*(values(slot) for slot in family.aliases)):
                    slots = {family.width.token: width, **spelling, **dict(zip(alias_tokens, alias_values))}
                    direct = [substitute(line, slots) for line in lines]
                    signature, reason = infer_signature(
                        direct, request.inputs, request.output, intrinsics, request.io_shape
                    )
                    if signature is None:
                        # A sweep expects misses; a named dtype's is worth a note.
                        if not sweeping and all(
                            v in ("", slot.values[0]) for slot, v in zip(family.aliases, alias_values)
                        ):
                            note_dropped(request.primitive, isa, f"dtype={dtype}, reason={reason}")
                        continue
                    expanded.append({**definition, "dtype": cand_dtype, "direct": direct, "signature": signature})

        if sweeping and not expanded:
            note_dropped(request.primitive, isa, f"reason=no existing intrinsic for any dtype of {lines}")
        return expanded

    def _dtype_choices(
        self, named: str | None, sweeping: bool, width: str, allowed_dtypes: frozenset[str]
    ) -> list[tuple[str | None, dict[str, str]]]:
        """``(dtype, slot values)`` pairs to try at one width."""
        family = self._family
        if named is not None:
            return [(named, family.dtypes.get(named, {}))]
        if not sweeping:
            return [(None, {})]
        choices = [(dt, spelling) for dt, spelling in family.dtypes.items() if dt in allowed_dtypes]
        if (register := family.registers.get(width)) is not None:
            choices.append((None, register))
        return choices


_X86_DTYPE_SUFFIX: dict[str, str] = {
    "int8": "epi8", "uint8": "epu8",
    "int16": "epi16", "uint16": "epu16",
    "int32": "epi32", "uint32": "epu32",
    "int64": "epi64", "uint64": "epu64",
    "float32": "ps", "float64": "pd",
    "bool8": "mask8", "bool16": "mask16",
    "bool32": "mask32", "bool64": "mask64",
}

X86_EXPANDER = SourceFamilyExpander(SourceFamily(
    width=Slot("x86_widths", ("", "256", "512")),
    dtype_tokens=("dtype_suffix",),
    dtypes={dt: {"dtype_suffix": suffix} for dt, suffix in _X86_DTYPE_SUFFIX.items()},
    # The whole-register integer `si{bits}` of the `_mm{width}` register.
    registers={width: {"dtype_suffix": f"si{bits}"} for width, bits in (("", 128), ("256", 256), ("512", 512))},
))

_ARM_TYPE_PREFIX: dict[str, str] = {"int": "s", "uint": "u", "float": "f", "bool": "b"}

ARM_EXPANDER = SourceFamilyExpander(SourceFamily(
    width=Slot("neon_widths", ("", "q")),
    dtype_tokens=("dtype_suffix", "elem_bits"),
    # Swept in name order.
    dtypes=dict(sorted(
        (f"{family}{bits}", {"dtype_suffix": f"{_ARM_TYPE_PREFIX[family]}{bits}", "elem_bits": str(bits)})
        for family, widths in DTYPE_WIDTHS.items() for bits in widths
    )),
    # Under an all-true predicate the SVE governing modes are one unmasked operation.
    aliases=(Slot("sve_gov", ("x", "z", "m")),),
))
