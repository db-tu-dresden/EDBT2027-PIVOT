"""Which target variants a match's source signature admits, from the matched
intrinsics alone.  It decides whether a match is a selection candidate and
bounds the typing pass's choice."""
from __future__ import annotations

from pivot.ir.types import IRKind, IRType, ir_unifies, parse_ir_token, Domain
from pivot.passes.translation.model import Operand, Variant

_INT_UINT = frozenset({Domain.INT, Domain.UINT})


def slot_ir(variant: Variant, param: str) -> IRType | None:
    token = variant.typing_signature.get(param, "").strip()
    return parse_ir_token(token) if token else None


def _operand_admits(operand: Operand, variant: Variant) -> bool:
    """Whether this operand's own source type can fill the variant's slot; an
    opaque operand or slot never rejects.  Two int/uint scalars admit each other at
    any width, as the host language converts a scalar implicitly (the typing pass
    ranks the width gap).  Everything else is `ir_unifies`: a vector keeps its sign
    and width, which the intrinsic name pins."""
    source = operand.ir
    if source is None:
        return True
    target = slot_ir(variant, operand.param)
    if target is None:
        return True
    if (
        source.kind is IRKind.SCALAR and target.kind is IRKind.SCALAR
        and source.domain in _INT_UINT and target.domain in _INT_UINT
    ):
        return True
    return ir_unifies(source, target)


def admissible_variants(operands: tuple[Operand, ...], variants: list[Variant]) -> list[Variant]:
    """The variants a match's source operands admit.  A variant missing a slot for
    an operand is inadmissible."""
    return [
        variant for variant in variants
        if all(
            variant.typing_signature.get(operand.param, "").strip() and _operand_admits(operand, variant)
            for operand in operands
        )
    ]
