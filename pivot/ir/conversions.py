"""Which conversion a value needs to be read through a slot of another type: the
one decision behind a variant's cast cost and every conversion emit writes (the
backend only spells it).  Every conversion preserves the bits; x86 performs
them implicitly on its untyped `__m*i` registers and integer `__mmaskN` masks,
strongly typed targets need them explicit."""
from __future__ import annotations

from enum import Enum

from pivot.ir.types import ElemBits, IRKind, IRType, parse_ir_token

# The slot of a read C evaluates as an integer (`if (m)`, `popcount(m)`).
INTEGER_SLOT = "integer"


class Conversion(Enum):
    # Same type, or nothing the translation converts (scalars, pointers, immediates,
    # tokens outside the IR vocabulary).
    NONE = "none"
    # A data vector read at another element view of the same register width.
    REINTERPRET = "reinterpret"
    # A mask resized (widen zero-extends, truncate keeps the low lanes) or read at
    # another element.
    MASK_CAST = "mask_cast"
    # A mask read as its integer bitmap.
    MASK_BITS = "mask_bits"
    # No faithful conversion (another width, data vs. mask, vector vs. scalar).
    IMPOSSIBLE = "impossible"


def conversion(value_token: str | None, slot_token: str | None) -> Conversion:
    """The conversion a value of ``value_token`` needs in a ``slot_token`` slot."""
    if not value_token or not slot_token or value_token == slot_token:
        return Conversion.NONE
    value = parse_ir_token(value_token)
    if slot_token == INTEGER_SLOT:
        return Conversion.MASK_BITS if value is not None and value.kind is IRKind.MASK else Conversion.NONE
    slot = parse_ir_token(slot_token)
    if value is None or slot is None:
        return Conversion.NONE
    kinds = {value.kind, slot.kind}
    if kinds & {IRKind.POINTER, IRKind.IMMEDIATE} or kinds == {IRKind.SCALAR}:
        return Conversion.NONE
    if kinds == {IRKind.MASK}:
        return Conversion.MASK_CAST if _mask_cast_needed(value, slot) else Conversion.NONE
    if kinds == {IRKind.VECTOR}:
        return _vector_conversion(value, slot)
    return Conversion.IMPOSSIBLE


def _mask_cast_needed(value: IRType, slot: IRType) -> bool:
    """Two different mask tokens: a cast unless an element-agnostic (`_bALL`)
    side meets the same lane count."""
    if value.lanes is not None and slot.lanes is not None and value.lanes != slot.lanes:
        return True
    return value.elem_bits is not ElemBits.ALL and slot.elem_bits is not ElemBits.ALL


def _vector_conversion(value: IRType, slot: IRType) -> Conversion:
    """Two different data vector tokens.  A length-open side is one whole register,
    so only the element view can differ; fixed lanes must keep the total width."""
    if value.lanes is None or slot.lanes is None:
        same_view = (value.domain, value.elem_bits) == (slot.domain, slot.elem_bits)
        return Conversion.NONE if same_view else Conversion.REINTERPRET
    if value.concrete_total_bits() == slot.concrete_total_bits():
        return Conversion.REINTERPRET
    return Conversion.IMPOSSIBLE
