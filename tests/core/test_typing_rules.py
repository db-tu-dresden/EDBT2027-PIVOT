"""Unit tests for the mask typing rules.

1. A length-open whole-register predicate (`vb_bALL`) stays length-open and
   admits every element; the typing pass types the op from its operands.
2. `admissible_variants`: which target variants a match's own signature admits.
3. `_canonical_rank`: the cast-free tie-break, which names a mask slot's element
   from the op's own source register (the byte register for an open whole
   predicate, the representative for a fixed x86 k-register, its own width for a
   `_bN`).
"""
from __future__ import annotations

import pytest

from pivot.ir.types import concretize_scalable_token, resolve_scalable_ir, parse_ir_token
from pivot.passes.translation.admissibility import admissible_variants
from pivot.passes.translation.model import Match, Operand, Variant
from pivot.passes.translation.type_resolution import _canonical_rank


def _match(slot_tokens: dict[str, str], variant_sigs: list[dict[str, str]], sve_bits=128) -> Match:
    """A match whose operands carry the source types as matching binds them, with
    every variant (admissible or not) in `variants`."""
    operands = {
        name: Operand(name, ir=resolve_scalable_ir(parse_ir_token(token), sve_bits))
        for name, token in slot_tokens.items()
    }
    return Match(
        index=0, primitive=None, pattern=None, mapping={}, calls=frozenset(), result="", kept=frozenset(),
        inputs=tuple(operand for name, operand in operands.items() if name != "res"),
        output=operands.get("res"),
        variants=tuple(
            Variant(
                definition=None,
                typing_signature={k: concretize_scalable_token(v, sve_bits) for k, v in sig.items()},
                cost=0,
            )
            for sig in variant_sigs
        ),
    )


def _admissible(match: Match) -> list[Variant]:
    return admissible_variants(match.operands, match.variants)


def _tsl_vla_sweep(params: list[str]) -> list[dict[str, str]]:
    """One width-symbolic `{vbool}` variant per element, as the tsl vla expander emits."""
    return [{p: f"vb_{s}" for p in params} for s in ("c", "s", "i", "l", "uc", "us", "ui", "ul", "f", "d")]


def _tsl_samewidth_sweep(params: list[str]) -> list[dict[str, str]]:
    """Per-register-width fixed-lane sweep (x86 samewidth): three widths per element."""
    out = []
    for suffix, bits in (("c", 8), ("s", 16), ("i", 32), ("l", 64), ("uc", 8), ("us", 16), ("ui", 32), ("ul", 64)):
        for reg in (128, 256, 512):
            out.append({p: f"v{reg // bits}b_{suffix}" for p in params})
    return out


def _canonical_pick(match: Match) -> Variant:
    """The variant the cast-free canonical rank would choose among the admissible."""
    return min(_admissible(match), key=lambda v: _canonical_rank(match, v))


# Rule 1: whole-register predicate stays length-open

@pytest.mark.parametrize("token,bits,expected", [
    ("vb_bALL", 128, "vb_bALL"),     # whole register: no divisor, stays length-open
    ("vb_bALL", 512, "vb_bALL"),
    ("vb_b32", 128, "v4b_b32"),      # granularity-carrying: lanes = bits / width
    ("vb_i", 256, "v8b_i"),
    ("vi", 128, "v4i"),
    ("v16b_bALL", 128, "v16b_bALL"), # already fixed (x86 __mmask16): untouched
    ("v8b_i", 512, "v8b_i"),
])
def test_scalable_resolution(token, bits, expected):
    assert concretize_scalable_token(token, bits) == expected


def test_whole_register_stays_length_open():
    ir = resolve_scalable_ir(parse_ir_token("vb_bALL"), 128)
    assert ir.lanes is None
    assert parse_ir_token("vb_bALL") == ir


# Rule 2: admissible_variants (source-only fit)

def test_open_ball_admits_every_element():
    """A length-open whole predicate unifies with every variant (any lanes, any
    element): the `_bALL` element wildcard and the open lane count reject nothing."""
    match = _match({"res": "vb_bALL"}, _tsl_vla_sweep(["res"]))
    assert len(_admissible(match)) == 10


def test_fixed_ball_admits_only_its_lane_count():
    """A fixed x86 `v16b_bALL` admits only 16-lane variants (the element is still a
    wildcard, but the lane count must match)."""
    match = _match({"res": "v16b_bALL"}, _tsl_samewidth_sweep(["res"]))
    admissible = _admissible(match)
    assert admissible and all(parse_ir_token(v.typing_signature["res"]).lanes == 16 for v in admissible)


def test_scalar_source_admits_any_width():
    """Two int/uint scalars admit each other at any width (the host converts a
    scalar operand implicitly)."""
    match = _match({"a": "int64_t"}, [{"a": "int32_t"}, {"a": "uint16_t"}])
    assert len(_admissible(match)) == 2


def test_concrete_vector_sign_not_admitted():
    """A concrete data vector keeps strict sign: v4i does not admit a v4ui slot."""
    match = _match({"a": "v4i"}, [{"a": "v4i"}, {"a": "v4ui"}])
    admissible = _admissible(match)
    assert len(admissible) == 1 and admissible[0].typing_signature["a"] == "v4i"


# Rule 3: canonical rank names the mask from its own source register

def test_canonical_open_whole_register_is_byte_granular():
    """svpfalse_b: an open whole predicate with no operand -> the byte register
    (uint8), the finest granularity."""
    match = _match({"res": "vb_bALL"}, _tsl_vla_sweep(["res"]))
    assert _canonical_pick(match).typing_signature["res"] == "v16b_uc"


def test_canonical_fixed_kmask_prefers_representative():
    """kand on __mmask16: uint at the width-preserving representative (uint32 @512)."""
    match = _match({"res": "v16b_bALL"}, _tsl_samewidth_sweep(["res"]))
    assert _canonical_pick(match).typing_signature["res"] == "v16b_ui"


def test_canonical_granular_source_keeps_its_own_width():
    """svzip1_b32 slots `vb_b32`: uint at the slot's own 32-bit granularity (v4b_ui
    at 128), never the register-filling representative."""
    match = _match({"a": "vb_b32", "res": "vb_b32"}, _tsl_vla_sweep(["a", "res"]))
    assert _canonical_pick(match).typing_signature["res"] == "v4b_ui"


def test_canonical_admits_scalar_output():
    """svcntp: a mask input and a uint64 scalar count output: the scalar slot does
    not disturb the mask slot's canonical (v2b_ul at its own 64-bit granularity)."""
    match = _match({"a": "vb_b64", "res": "uint64_t"},
               [{"a": f"vb_{s}", "res": "uint64_t"} for s in ("i", "l", "ui", "ul", "f", "d")])
    assert _canonical_pick(match).typing_signature["a"] == "v2b_ul"
