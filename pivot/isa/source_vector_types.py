"""The data-vector token a source-ISA vector type name stands for when nothing
more specific is known: a spelling that carries no value (`sizeof(__m128i)`), or a
pointer / alias no typed value reaches.  Only the register width must survive, so
the element is the typing pass's canonical default (signed int32 for an
element-agnostic x86 integer register); NEON and SVE names carry their element."""
from __future__ import annotations

import re

from pivot.ir.types import IRKind, IRType, spell_ir_token, Domain, ElemBits

_X86 = re.compile(r"__m(?P<bits>128|256|512)(?P<kind>[id]?)")
_NEON = re.compile(r"(?P<dom>u?int|float)(?P<elem>8|16|32|64)x(?P<lanes>\d+)_t")
_SVE = re.compile(r"sv(?P<dom>u?int|float)(?P<elem>8|16|32|64)_t")

_DOMAIN = {"int": Domain.INT, "uint": Domain.UINT, "float": Domain.FLOAT}


def _vector(domain: Domain, elem: int, lanes: int) -> str | None:
    return spell_ir_token(IRType(IRKind.VECTOR, domain, ElemBits(elem), lanes))


def canonical_vector_token(type_name: str, sve_bits: int | None) -> str | None:
    """Canonical data-vector token of a source vector type name, or None for a
    mask/predicate type or a name this module does not know."""
    name = type_name.strip()
    if m := _X86.fullmatch(name):
        bits = int(m.group("bits"))
        if m.group("kind") == "d":
            return _vector(Domain.FLOAT, 64, bits // 64)
        if m.group("kind") == "i":
            return _vector(Domain.INT, 32, bits // 32)
        return _vector(Domain.FLOAT, 32, bits // 32)
    if m := _NEON.fullmatch(name):
        return _vector(_DOMAIN[m.group("dom")], int(m.group("elem")), int(m.group("lanes")))
    if (m := _SVE.fullmatch(name)) and sve_bits:
        elem = int(m.group("elem"))
        return _vector(_DOMAIN[m.group("dom")], elem, sve_bits // elem)
    return None
