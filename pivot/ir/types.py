"""The language-neutral IR type vocabulary: the token grammar, IRType, its
domain / element-width enums, unification and scalable-width resolution."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Domain(str, Enum):
    INT = "int"
    UINT = "uint"
    FLOAT = "float"
    # Mask-only: a predicate over lanes of this width whose element domain is
    # irrelevant (SVE `svcntp_b32`, x86 `kand`).  A mask with a known element
    # carries INT/UINT/FLOAT instead.
    BITS = "b"


class ElemBits(int, Enum):
    BITS_8 = 8
    BITS_16 = 16
    BITS_32 = 32
    BITS_64 = 64
    # Mask-only `_bALL`: the whole register, element width irrelevant.  Falsy
    # (0): excluded from `n // width` lane resolution, loud if misused.
    ALL = 0


# One vocabulary types both source-intrinsic operands and target-definition
# signatures; the backend emitters spell it.
#
# Token grammar:
#
#   scalar          int{N}_t | uint{N}_t | float | double
#   vector (fixed)  v{lanes}{suffix}          e.g. v4i, v8f, v2d, v16uc
#   vector (open)   v{suffix}                 length-open / scalable (SVE),
#                                             e.g. vi (scalable int32)
#   mask (fixed)    v{lanes}b_{elem}          predicate over `lanes` elements of
#                                             `elem`, e.g. v16b_i (16×int32),
#                                             v16b_bALL (16 bits, granularity-
#                                             agnostic)
#   mask (open)     vb_{elem}                 length-open / scalable predicate,
#                                             e.g. vb_i (SVE svbool_t over int32)
#   integer blob    blob{bits}                width-fixed, element/lane/domain
#                                             open (x86 si128/si256/si512)
#   pointer         ptr                       opaque memory address; the pointee
#                                             carries no selection signal (target
#                                             defs spell every pointer as void*)
#   immediate       imm{bits}                 compile-time integer constant, e.g.
#                                             imm8, imm32; sign-less, width is
#                                             informational.  The marker a backend
#                                             uses to route an operand to a
#                                             template / const-generic argument.
#
# data suffix ∈ {c,s,i,l, uc,us,ui,ul, f,d}; a fixed IR vector spells identically
# to a backend signature token.  A mask always carries a predicated-element suffix
# `_{elem}`, where elem is a data suffix (concrete: v16b_i), a domain-agnostic
# granularity (b8,b16,b32,b64: v16b_b32), or the whole-mask sentinel (bALL:
# v16b_bALL).  There is no element-less mask token.

# suffix -> (domain, elem_bits).  Data vectors only; masks use `_MASK_ELEM_TO_TYPE`.
_SUFFIX_TO_TYPE: dict[str, tuple[Domain, ElemBits]] = {
    "c": (Domain.INT, ElemBits.BITS_8),
    "s": (Domain.INT, ElemBits.BITS_16),
    "i": (Domain.INT, ElemBits.BITS_32),
    "l": (Domain.INT, ElemBits.BITS_64),
    "uc": (Domain.UINT, ElemBits.BITS_8),
    "us": (Domain.UINT, ElemBits.BITS_16),
    "ui": (Domain.UINT, ElemBits.BITS_32),
    "ul": (Domain.UINT, ElemBits.BITS_64),
    "f": (Domain.FLOAT, ElemBits.BITS_32),
    "d": (Domain.FLOAT, ElemBits.BITS_64),
}
_TYPE_TO_SUFFIX: dict[tuple[Domain, ElemBits], str] = {
    v: k for k, v in _SUFFIX_TO_TYPE.items()
}

# Mask predicated-element suffix -> (domain, elem_bits).  A mask is always
# (domain, width) with both present: a concrete data element (reusing the data
# suffixes), a domain-agnostic fixed granularity (BITS + 8/16/32/64), or the
# granularity-agnostic whole-mask sentinel (BITS + ALL).
_MASK_ELEM_TO_TYPE: dict[str, tuple[Domain, ElemBits]] = {
    **_SUFFIX_TO_TYPE,
    "b8": (Domain.BITS, ElemBits.BITS_8),
    "b16": (Domain.BITS, ElemBits.BITS_16),
    "b32": (Domain.BITS, ElemBits.BITS_32),
    "b64": (Domain.BITS, ElemBits.BITS_64),
    "bALL": (Domain.BITS, ElemBits.ALL),
}
_MASK_TYPE_TO_ELEM: dict[tuple[Domain, ElemBits], str] = {
    v: k for k, v in _MASK_ELEM_TO_TYPE.items()
}

_BLOB_TOKEN = re.compile(r"^blob(?P<bits>\d+)$")
_VECTOR_TOKEN = re.compile(r"^v(?P<lanes>\d*)(?P<suffix>[a-z]+)$")
_MASK_TOKEN = re.compile(r"^v(?P<lanes>\d*)b_(?P<elem>[A-Za-z0-9]+)$")
_SCALAR_INT_TOKEN = re.compile(r"^(?P<sign>u?)int(?P<bits>8|16|32|64)_t$")
_IMMEDIATE_TOKEN = re.compile(r"^imm(?P<bits>8|16|32|64)$")


class IRKind(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    MASK = "mask"
    BLOB = "blob"
    POINTER = "pointer"
    IMMEDIATE = "immediate"


@dataclass(frozen=True)
class IRType:
    """One language-neutral IR type.

    `open` types leave part of the shape unresolved:
      * BLOB           — only total_bits is known (structureless integer register)
      * VECTOR w/ lanes=None — element type known, lane count open (scalable)
      * MASK  w/ lanes=None — predicated element known, lane count open (scalable
                              SVE predicate); resolved via `n // width`

    For a MASK, `domain`/`elem_bits` describe the *predicated* element (INT/UINT/
    FLOAT + width, a domain-agnostic BITS granularity, or the BITS+ALL whole-mask
    sentinel); the mask-ness itself is the `kind`, not the domain.
    """

    kind: IRKind
    domain: Domain | None = None       # None only for BLOB
    elem_bits: ElemBits | None = None  # None only for BLOB
    lanes: int | None = None           # 1 for SCALAR; None = open for VECTOR/MASK/BLOB
    total_bits: int | None = None      # set for BLOB; None otherwise

    def concrete_total_bits(self) -> int | None:
        """Total register width for a fully-concrete, element-sized vector/scalar.

        `None` for a MASK: a predicate carries no data register width in this model
        (its bit-packed width is `lanes`, read where that is what's wanted)."""
        if self.kind is IRKind.BLOB:
            return self.total_bits
        if self.kind is IRKind.MASK:
            return None
        if self.elem_bits is None or self.lanes is None:
            return None
        return self.elem_bits.value * self.lanes


def parse_ir_token(token: str) -> IRType | None:
    """Parse one IR token string into an IRType, or None if unrecognized."""
    token = token.strip()

    if token == "float":
        return IRType(IRKind.SCALAR, Domain.FLOAT, ElemBits.BITS_32, 1)
    if token == "double":
        return IRType(IRKind.SCALAR, Domain.FLOAT, ElemBits.BITS_64, 1)
    if m := _SCALAR_INT_TOKEN.match(token):
        domain = Domain.UINT if m.group("sign") == "u" else Domain.INT
        return IRType(IRKind.SCALAR, domain, ElemBits(int(m.group("bits"))), 1)

    if token == "ptr":
        return IRType(IRKind.POINTER)

    if m := _IMMEDIATE_TOKEN.match(token):
        # A compile-time integer constant.  Sign-less; the width is carried for
        # spelling only and is not used to reject a unification.
        return IRType(IRKind.IMMEDIATE, elem_bits=ElemBits(int(m.group("bits"))))

    if m := _BLOB_TOKEN.match(token):
        return IRType(IRKind.BLOB, total_bits=int(m.group("bits")))

    if m := _MASK_TOKEN.match(token):
        elem = m.group("elem")
        if elem not in _MASK_ELEM_TO_TYPE:
            return None
        domain, elem_bits = _MASK_ELEM_TO_TYPE[elem]
        lanes_digits = m.group("lanes")
        lanes = int(lanes_digits) if lanes_digits else None  # absent => scalable
        return IRType(IRKind.MASK, domain, elem_bits, lanes)

    if m := _VECTOR_TOKEN.match(token):
        suffix = m.group("suffix")
        if suffix not in _SUFFIX_TO_TYPE:
            return None
        domain, elem_bits = _SUFFIX_TO_TYPE[suffix]
        lanes_digits = m.group("lanes")
        lanes = int(lanes_digits) if lanes_digits else None  # absent => scalable/open
        return IRType(IRKind.VECTOR, domain, elem_bits, lanes)

    return None


def spell_ir_token(ir: IRType) -> str | None:
    """Render an IRType back to its canonical token string, or None if unspellable."""
    if ir.kind is IRKind.POINTER:
        return "ptr"

    if ir.kind is IRKind.BLOB:
        return f"blob{ir.total_bits}" if ir.total_bits is not None else None

    if ir.kind is IRKind.IMMEDIATE:
        return f"imm{ir.elem_bits.value}" if ir.elem_bits is not None else None

    if ir.kind is IRKind.SCALAR:
        if ir.domain is Domain.INT and ir.elem_bits is not None:
            return f"int{ir.elem_bits.value}_t"
        if ir.domain is Domain.UINT and ir.elem_bits is not None:
            return f"uint{ir.elem_bits.value}_t"
        if ir.domain is Domain.FLOAT and ir.elem_bits is ElemBits.BITS_32:
            return "float"
        if ir.domain is Domain.FLOAT and ir.elem_bits is ElemBits.BITS_64:
            return "double"
        return None

    if ir.kind is IRKind.MASK:
        if ir.domain is None or ir.elem_bits is None:
            return None
        elem = _MASK_TYPE_TO_ELEM.get((ir.domain, ir.elem_bits))
        if elem is None:
            return None
        lanes = "" if ir.lanes is None else str(ir.lanes)
        return f"v{lanes}b_{elem}"

    # VECTOR
    suffix = _TYPE_TO_SUFFIX.get((ir.domain, ir.elem_bits)) if ir.domain is not None else None
    if suffix is None:
        return None
    lanes = "" if ir.lanes is None else str(ir.lanes)
    return f"v{lanes}{suffix}"


def immediate_bits(token: str) -> int | None:
    """Bit width of an immediate token (``imm{bits}``), else None."""
    ir = parse_ir_token(token.strip()) if token else None
    if ir is not None and ir.kind is IRKind.IMMEDIATE and ir.elem_bits is not None:
        return ir.elem_bits.value
    return None


def is_immediate_token(token: str) -> bool:
    """Whether ``token`` names an immediate operand (``imm{bits}``)."""
    return immediate_bits(token) is not None


def data_vector_shape(token: str) -> tuple[int, Domain, ElemBits] | None:
    """``(lanes, domain, elem_bits)`` for a fixed-lane data vector token, the shape
    a backend spells an operand type from; ``None`` for masks, scalars, pointers,
    blobs, immediates and length-open vectors.  Corpus tokens are canonical, so
    no cv-qualifiers are stripped."""
    ir = parse_ir_token(token)
    if (
        ir is None
        or ir.kind is not IRKind.VECTOR
        or ir.lanes is None
        or ir.elem_bits is None
    ):
        return None
    return ir.lanes, ir.domain, ir.elem_bits


def is_mask_token(token: str) -> bool:
    """Whether ``token`` names a mask, ``v{N}b_{elem}`` or width-open ``vb_{elem}``."""
    ir = parse_ir_token(token)
    return ir is not None and ir.kind is IRKind.MASK


# A `_bALL` mask is a whole mask register (kand, svpfalse_b): no element, but a lane
# count.  An x86 `__mmaskN` has N lanes, assumed to predicate an AVX-512 register
# (`mask_representative_bits`).  An SVE `svbool_t` has a bit per byte, so `vb_bALL`
# stays length-open and takes its operands' element; with no operand, the byte
# granularity, which loses no lanes.

# Widest register a concretized fixed-lane mask is assumed to predicate (AVX-512).
MAX_MASK_REGISTER_BITS = 512

# Native lane granularity of a length-scalable predicate register: one bit per byte.
SCALABLE_MASK_NATIVE_BITS = ElemBits.BITS_8


def mask_representative_bits(lanes: int) -> int | None:
    """The element width for a fixed-lane `_bALL` mask of `lanes` lanes: the largest
    of {8,16,32,64} with ``lanes * width <= MAX_MASK_REGISTER_BITS``, so an x86
    mask stays a native k-register (16 lanes -> uint32 @512, 8 -> uint64).
    ``None`` for more than 64 lanes."""
    want = min(64, MAX_MASK_REGISTER_BITS // lanes)
    return want if want >= 8 else None


def mask_lanes(token: str) -> int | None:
    """Lane count of a fixed-lane mask token (``v{N}b_{elem}``), else ``None`` (a
    scalable mask ``vb_{elem}`` or a non-mask)."""
    ir = parse_ir_token(token)
    if ir is None or ir.kind is not IRKind.MASK:
        return None
    return ir.lanes


def mask_witness_data_token(token: str) -> str | None:
    """The data vector token a backend names a mask type from (TSL's
    ``Vec::mask_type``): ``v16b_i`` -> ``v16i``, ``vb_i`` -> ``vi``, and a
    granularity mask an unsigned vector of its width (``v16b_b32`` -> ``v16ui``).

    A ``_bALL`` mask reaching here belongs to a width-changing op (``kunpackb``)
    and takes :func:`mask_representative_bits` (``v16b_bALL`` -> ``v16ui``).
    ``None`` for a non-mask or unspellable token."""
    ir = parse_ir_token(token)
    if ir is None or ir.kind is not IRKind.MASK or ir.elem_bits is None:
        return None
    if ir.elem_bits is ElemBits.ALL:
        if ir.lanes is None:
            return None
        rep = mask_representative_bits(ir.lanes)
        if rep is None:
            return None
        return spell_ir_token(IRType(IRKind.VECTOR, Domain.UINT, ElemBits(rep), ir.lanes))
    domain = ir.domain if ir.domain in (Domain.INT, Domain.UINT, Domain.FLOAT) else Domain.UINT
    return spell_ir_token(IRType(IRKind.VECTOR, domain, ir.elem_bits, ir.lanes))


def ir_unifies(a: IRType, b: IRType) -> bool:
    """True when two IR types share a common concrete refinement (order-insensitive).

      * BLOB{W} unifies with any data vector of total width W or a BLOB of equal
        width; never with a mask.
      * MASK unifies only with MASK, on equal lanes and a compatible predicated
        element: `_bALL` matches any element, `_bN` any element of width N, two
        concrete elements only each other.
      * everything else is strict equality; a sign or width mismatch is the
        conversion layer's job (a reinterpret), not variant selection's.
    """
    if a.kind is IRKind.POINTER or b.kind is IRKind.POINTER:
        return a.kind is IRKind.POINTER and b.kind is IRKind.POINTER

    if a.kind is IRKind.IMMEDIATE or b.kind is IRKind.IMMEDIATE:
        # An immediate slot takes a compile-time integer constant: another
        # immediate (width is informational, so any width) or an integer scalar:
        # the source operand's literal is typed int{N}_t by the intrinsic ir, and
        # a constant folds to any narrower immediate the instruction wants.
        imm, other = (a, b) if a.kind is IRKind.IMMEDIATE else (b, a)
        if other.kind is IRKind.IMMEDIATE:
            return True
        return other.kind is IRKind.SCALAR and other.domain in (Domain.INT, Domain.UINT)

    if a.kind is IRKind.BLOB or b.kind is IRKind.BLOB:
        blob, other = (a, b) if a.kind is IRKind.BLOB else (b, a)
        if other.kind is IRKind.BLOB:
            return blob.total_bits == other.total_bits
        if other.kind is IRKind.VECTOR:
            return blob.total_bits is not None and blob.total_bits == other.concrete_total_bits()
        return False

    if a.kind is IRKind.MASK or b.kind is IRKind.MASK:
        # A mask unifies only with a mask.
        if a.kind is not IRKind.MASK or b.kind is not IRKind.MASK:
            return False
        # Lanes as for a data vector: `_bALL` is a wildcard on the element only, so
        # v8b_bALL does not unify with v16b_bALL.
        if a.lanes is not None and b.lanes is not None and a.lanes != b.lanes:
            return False
        # Predicated element:
        #   * `_bALL` (BITS, ALL) is a whole-mask wildcard -> any element.
        #   * a domain-agnostic granularity (BITS, N) unifies with any element of
        #     the same width N (a concrete `_i`/`_ui`/`_f` or another `_bN`).
        #   * two concrete elements must match exactly (domain + width); a sign /
        #     domain mismatch is the reinterpret layer's job, not selection's.
        if a.elem_bits is ElemBits.ALL or b.elem_bits is ElemBits.ALL:
            return True
        if a.domain is Domain.BITS or b.domain is Domain.BITS:
            return a.elem_bits == b.elem_bits
        return a.domain == b.domain and a.elem_bits == b.elem_bits

    if a.kind is not b.kind:
        return False

    if a.kind is IRKind.SCALAR:
        return a.domain == b.domain and a.elem_bits == b.elem_bits

    # VECTOR vs VECTOR: both sides are concrete by now (source operands and
    # typing signatures are resolved at the assumed width), so a `None` lane
    # count is an unresolved token and fails.
    if a.domain != b.domain:
        return False
    if a.elem_bits != b.elem_bits:
        return False
    return a.lanes == b.lanes


def resolve_scalable_ir(ir: IRType, sve_bits: int | None) -> IRType:
    """Resolve a length-open vector/mask (`vi`, `vb_ui`) to a concrete lane count
    at the assumed SVE width (``lanes = sve_bits / elem_bits``).

    A length-open ``_bALL`` mask (`vb_bALL`) has no element to divide by and
    stays open: it admits every granularity, and the typing pass types it from
    its operands.  A shapeless or already-concrete type passes through."""
    if ir.kind not in (IRKind.VECTOR, IRKind.MASK) or ir.lanes is not None or not sve_bits:
        return ir
    if ir.elem_bits is None or ir.elem_bits is ElemBits.ALL:
        return ir
    return IRType(ir.kind, ir.domain, ir.elem_bits, sve_bits // ir.elem_bits.value)


def concretize_scalable_token(token: str, sve_bits: int | None) -> str:
    """Token form of :func:`resolve_scalable_ir`; a token that does not parse or
    re-spell is left as-is."""
    ir = parse_ir_token(token)
    if ir is None:
        return token
    spelled = spell_ir_token(resolve_scalable_ir(ir, sve_bits))
    return spelled if spelled is not None else token
