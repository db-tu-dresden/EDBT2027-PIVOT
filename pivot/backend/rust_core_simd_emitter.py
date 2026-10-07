from __future__ import annotations

from typing import TYPE_CHECKING

from pivot.backend.backend_emitter import BackendEmitter, Target
from pivot.backend.helpers import Param, bitmap_bits, expression, returning
from pivot.ir.types import (
    Domain,
    ElemBits,
    data_vector_shape,
    immediate_bits,
    mask_lanes,
    mask_witness_data_token,
)
from pivot.isa.ir_signature_expander import IRSignatureExpander
from pivot.lang.syntax import RUST_SYNTAX

if TYPE_CHECKING:
    from pivot.isa.primitive_registry import Definition


# An untyped pointer's raw-pointer prefix; the helper is generic over the pointee,
# so any caller pointer binds without a cast.
_POINTER_TOKENS: dict[str, str] = {"const_ptr": "*const", "mut_ptr": "*mut"}

# (domain, element bits) -> Rust scalar type.
_RUST_SCALAR: dict[tuple[Domain, ElemBits], str] = {
    (Domain.INT, ElemBits.BITS_8): "i8",    (Domain.UINT, ElemBits.BITS_8): "u8",
    (Domain.INT, ElemBits.BITS_16): "i16",  (Domain.UINT, ElemBits.BITS_16): "u16",
    (Domain.INT, ElemBits.BITS_32): "i32",  (Domain.UINT, ElemBits.BITS_32): "u32",
    (Domain.INT, ElemBits.BITS_64): "i64",  (Domain.UINT, ElemBits.BITS_64): "u64",
    (Domain.FLOAT, ElemBits.BITS_32): "f32", (Domain.FLOAT, ElemBits.BITS_64): "f64",
}

# A mask compares a vector's elements as a signed integer of their width:
# `Simd<f32, 8>` and `Simd<u32, 8>` both compare to `Mask<i32, 8>`.
_MASK_ELEMENT_BY_SCALAR: dict[str, str] = {
    "i8": "i8",  "u8": "i8",
    "i16": "i16", "u16": "i16",
    "i32": "i32", "u32": "i32", "f32": "i32",
    "i64": "i64", "u64": "i64", "f64": "i64",
}

# Re-exported, since the translated file's own signatures name `Simd`/`Mask` and
# the helper bodies call the `cmp`/`num` traits.  Which ones a companion uses
# depends on its primitives, and mask parameters are named `k__raw`.
_CORE_SIMD_USES: list[str] = [
    "#![allow(unused_imports, non_snake_case)]",
    "",
    "pub use core::simd::*;",
    "pub use core::simd::cmp::*;",
    "pub use core::simd::num::*;",
]


def _lanes_and_scalar(token: str) -> tuple[int, str] | None:
    """``(lanes, scalar)`` of a fixed-lane data-vector token, else None."""
    shape = data_vector_shape(token)
    if shape is None:
        return None
    lanes, domain, elem_bits = shape
    scalar = _RUST_SCALAR.get((domain, elem_bits))
    return (lanes, scalar) if scalar else None


def _mask_uint(token: str) -> str | None:
    """The ``uN`` bitmap a fixed-lane mask crosses helpers as."""
    lanes = mask_lanes(token)
    bits = bitmap_bits(lanes) if lanes is not None else None
    return f"u{bits}" if bits is not None else None


def core_simd_type_for_token(token: str) -> str | None:
    """The ``Simd<T, N>`` of a vector token (``v8i`` -> ``Simd<i32, 8>``)."""
    parsed = _lanes_and_scalar(token)
    if parsed is None:
        return None
    lanes, scalar = parsed
    return f"Simd<{scalar}, {lanes}>"


def _mask_type(token: str) -> str | None:
    """The ``Mask::<T, N>`` (expression position) of a mask token, from its own
    element; None for an element-agnostic mask."""
    data_token = mask_witness_data_token(token)
    parsed = _lanes_and_scalar(data_token) if data_token is not None else None
    if parsed is None:
        return None
    lanes, scalar = parsed
    element = _MASK_ELEMENT_BY_SCALAR.get(scalar)
    return f"Mask::<{element}, {lanes}>" if element is not None else None


# `Simd<T, N>` / `Mask<T, N>` exist for power-of-two lane counts up to this bound
# (`LaneCount<N>: SupportedLaneCount`); a wider variant cannot be spelled.
_MAX_SUPPORTED_LANES = 64


class RustCoreSimdEmitter(BackendEmitter):
    """Rust portable SIMD (``core::simd``): each primitive an ``#[inline] pub fn``
    over ``Simd<T, N>``, which builds for any target.  A mask crosses helpers as
    its ``uN`` bitmap, so user code reading it as an integer keeps compiling;
    immediates stay runtime parameters (``Simd::splat(imm)`` folds once inlined)."""

    target = Target(
        name="core_simd",
        companion_filename="pivot_core_simd.rs",
        companion_guard="pivot_core_simd companion (generated)",
        # The corpus expands to ~1000 monomorphic helpers; unused ones are noise.
        emits_only_used_definitions=True,
        # `core::simd` is nightly-only; the feature gate must sit at the crate root.
        file_prologue=("#![feature(portable_simd)]",),
    )
    language = RUST_SYNTAX
    # Bodies name `{simd_<param>}` in type position (`ptr as *const {simd_res}`,
    # `transmute` arguments), so the spelling has no turbofish.
    definition_expander = IRSignatureExpander(
        label="core_simd", simd_type=core_simd_type_for_token, max_lanes=_MAX_SUPPORTED_LANES
    )

    def spell_type(self, token: str) -> str:
        # An immediate is signed, as `core::arch` types it.
        if (bits := immediate_bits(token)) is not None:
            return f"i{bits}"
        if mask_lanes(token) is not None:
            return _mask_uint(token) or token
        return core_simd_type_for_token(token) or token

    def reinterpret_syntax(self, source: str, target: str) -> tuple[str, str]:
        # Portable SIMD has no general reinterpret (`to_bits` covers float only).
        src_simd, tgt_simd = core_simd_type_for_token(source), core_simd_type_for_token(target)
        if src_simd is None or tgt_simd is None:
            raise NotImplementedError(
                f"core_simd reinterpret needs two data-vector tokens (source={source!r}, target={target!r})"
            )
        return (f"unsafe {{ core::mem::transmute::<{src_simd}, {tgt_simd}>(", ") }")

    def mask_cast_syntax(self, source: str, target: str) -> tuple[str, str]:
        # Masks cross as integer bitmaps, so this is an integer `as`.
        if (uint := _mask_uint(target)) is None:
            raise NotImplementedError(f"core_simd mask cast needs a concrete mask target (target={target!r})")
        return ("(", f") as {uint}")

    def generic_pointer(self, token: str) -> str | None:
        return _POINTER_TOKENS.get(token.strip())

    def mask_param(self, name: str, token: str) -> Param:
        if mask_lanes(token) is None:
            return Param(self.spell_type(token), name)
        uint, mask = self._mask_boundary(token)
        # `from_bitmask` takes a u64 whatever the lane count.
        return Param(uint, f"{name}__raw", (f"let {name} = {mask}::from_bitmask({name}__raw as u64);",))

    def mask_return(self, token: str, body: list[str], output: str) -> tuple[str, list[str]]:
        if mask_lanes(token) is None:
            return self.spell_type(token), returning(body, output)
        uint, _mask = self._mask_boundary(token)
        if len(body) != 1:
            raise ValueError("a mask result needs a single direct expression")
        return uint, [f"return ({expression(body[0])}).to_bitmask() as {uint};"]

    @staticmethod
    def _mask_boundary(token: str) -> tuple[str, str]:
        """The bitmap a mask crosses helpers as, and its ``Mask`` inside one."""
        uint, mask = _mask_uint(token), _mask_type(token)
        if uint is None or mask is None:
            raise ValueError(f"mask {token} has no concrete element")
        return uint, mask

    def companion_prelude(self, definitions: list["Definition"]) -> list[str]:
        return [*_CORE_SIMD_USES, ""]
