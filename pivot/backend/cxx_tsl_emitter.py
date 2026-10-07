from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pivot.backend.backend_emitter import BackendEmitter, Target
from pivot.backend.helpers import Param, returning
from pivot.lang.syntax import CXX_SYNTAX
from pivot.driver.run_options import POLICY_CLANG_FIXED, WIDTH_MODE_VLA, active_options
from pivot.isa.ir_signature_expander import IRSignatureExpander
from pivot.ir.types import (
    IRKind,
    data_vector_shape,
    immediate_bits,
    is_mask_token,
    mask_witness_data_token,
    parse_ir_token,
    Domain,
    ElemBits,
)

if TYPE_CHECKING:
    from pivot.isa.primitive_registry import Definition


# Replaces the header selection by one `#include` of the named header.
_TSL_PROFILE_ENV = "PIVOT_TSL_PROFILE"

# samewidth bakes concrete `fixed<N>` lane counts, so its one profile header must be
# as wide as the widest vector: avx2 up to 256 bits, else the AVX-512 profile.
_PROFILE_BASE_DEFAULT = "avx2"
_PROFILE_BASE_AVX512 = "icelake_rockerlake"

# (domain, element bits) -> the element type TSL is instantiated with.
_CXX_SCALAR: dict[tuple[Domain, ElemBits], str] = {
    (Domain.INT, ElemBits.BITS_8): "int8_t",    (Domain.UINT, ElemBits.BITS_8): "uint8_t",
    (Domain.INT, ElemBits.BITS_16): "int16_t",  (Domain.UINT, ElemBits.BITS_16): "uint16_t",
    (Domain.INT, ElemBits.BITS_32): "int32_t",  (Domain.UINT, ElemBits.BITS_32): "uint32_t",
    (Domain.INT, ElemBits.BITS_64): "int64_t",  (Domain.UINT, ElemBits.BITS_64): "uint64_t",
    (Domain.FLOAT, ElemBits.BITS_32): "float",  (Domain.FLOAT, ElemBits.BITS_64): "double",
}

# vla's one knob, the vector register width in bits: every vector has
# `PIVOT_TSL_VLEN_BITS / elem_bits` lanes (SVE's model), so an SVE kernel stays
# length-open and an x86 kernel becomes width-tunable.
_VLEN_MACRO = "PIVOT_TSL_VLEN_BITS"


def _policy_template_head(policy: str, lane_expr: str) -> str:
    """The ``POLICY<lanes>`` head of a simd type.  `clang_fixed` also names its
    mask tag, since a vector's `mask_type` must match the mask passed to it."""
    if policy == POLICY_CLANG_FIXED:
        return f"clang_fixed<{lane_expr}, tsl::dataparallel::clang_mask::comparison_vector>"
    return f"{policy}<{lane_expr}>"


def _lanes_and_scalar(token: str, *, symbolic: bool) -> tuple[str, str] | None:
    """``(lane count, element type)`` of a data-vector token: symbolically
    ``PIVOT_TSL_VLEN_BITS / elem`` for any lane count, else its own fixed lane
    count; None for a token with no simd type."""
    ir = parse_ir_token(token)
    if ir is None or ir.kind is not IRKind.VECTOR or ir.elem_bits is None:
        return None
    scalar = _CXX_SCALAR.get((ir.domain, ir.elem_bits))
    if scalar is None:
        return None
    if symbolic:
        return f"{_VLEN_MACRO} / {ir.elem_bits.value}", scalar
    if ir.lanes is None:
        return None
    return str(ir.lanes), scalar


def _tsl_type(template: str, token: str, policy: str, symbolic: bool) -> str | None:
    parsed = _lanes_and_scalar(token, symbolic=symbolic)
    if parsed is None:
        return None
    lanes, scalar = parsed
    return f"tsl::dataparallel::{template}<tsl::dataparallel::{_policy_template_head(policy, lanes)},{scalar}>"


def tsl_simd_type_for_token(token: str, policy: str, *, symbolic: bool) -> str | None:
    """The TSL simd type (``simd_for_t<fixed<8>,int32_t>``) of a vector token."""
    return _tsl_type("simd_for_t", token, policy, symbolic)


def tsl_register_type_for_token(token: str, policy: str, *, symbolic: bool) -> str | None:
    """The TSL register type of a vector token, what a value of it is declared as."""
    return _tsl_type("register_t", token, policy, symbolic)


def _mask_witness(token: str, policy: str, *, symbolic: bool) -> str | None:
    """The simd type whose ``mask_type`` a mask token is, from its own element
    (``v16b_i`` -> that of ``v16i``); None for an element-agnostic mask."""
    data_token = mask_witness_data_token(token)
    if data_token is None:
        return None
    return tsl_simd_type_for_token(data_token, policy, symbolic=symbolic)


def _max_vector_bits(definitions: list["Definition"]) -> int:
    """The widest fixed-lane data vector the definitions name, in bits; 0 if none."""
    widest = 0
    for definition in definitions:
        for token in definition.signature.values():
            if (shape := data_vector_shape(token)) is not None:
                lanes, _domain, elem_bits = shape
                widest = max(widest, lanes * elem_bits.value)
    return widest


# vla's header switch: the arch macro picks the family, the VLEN knob the tier.  A
# tier names one header, or several: the first included, the rest offered commented
# out (NEON beside SVE@128).  `clang_fixed` uses each one's `*_clang` sibling.
_VLA_HEADER_LADDER: tuple[tuple[str, tuple[tuple[int, str | tuple[str, ...]], ...]], ...] = (
    ("defined(__x86_64__) || defined(_M_X64)", (
        (512, "icelake_rockerlake"), (256, "avx2"), (128, "sse2"),
    )),
    ("defined(__aarch64__)", (
        (512, "sve512"), (256, "sve256"), (128, ("sve128", "neon")),
    )),
    # Fixed-length RVV profiles: one VLEN per build (-mrvv-vector-bits=zvl), wider
    # vectors group registers (rvv1024m4 = four VLEN-256 registers), half a
    # register (mf2) has no 64-bit lanes.
    ("defined(__riscv) && defined(__riscv_v_fixed_vlen) && __riscv_v_fixed_vlen == 1024", (
        (1024, "rvv1024m1"), (512, "rvv512mf2"),
    )),
    ("defined(__riscv)", (
        (1024, "rvv1024m4"), (512, "rvv512m2"), (256, "rvv256m1"), (128, "rvv128mf2"),
    )),
)
_VLA_HEADER_FALLBACK = "scalar"  # an unknown arch gets the portable scalar backend


# A mask changes element type through its integral mask: an integer cast where
# the integral masks share a type (SVE) or are both integers (x86), else the lane
# window extract_imask / insert_imask (RVV, one predicate type per element width).
_MASK_CAST_HELPER = "pivot_mask_cast"
_MASK_CAST_LINES = [
    "#include <type_traits>",
    "",
    "template <class ToVec, class Vec>",
    f"static inline typename ToVec::mask_type {_MASK_CAST_HELPER}(typename Vec::mask_type mask) {{",
    "    using From = typename Vec::imask_type;",
    "    using To = typename ToVec::imask_type;",
    "    if constexpr (std::is_same_v<From, To> || (std::is_integral_v<From> && std::is_integral_v<To>)) {",
    "        return tsl::to_mask<ToVec>((To) tsl::to_integral<Vec>(mask));",
    "    } else if constexpr (Vec::lane_count_v > ToVec::lane_count_v) {",
    "        return tsl::to_mask<ToVec>(tsl::extract_imask<Vec, ToVec>(tsl::to_integral<Vec>(mask), 0));",
    "    } else {",
    "        return tsl::to_mask<ToVec>(tsl::insert_imask<Vec, ToVec>(",
    "            tsl::to_integral<ToVec>(tsl::mask_false<ToVec>()), tsl::to_integral<Vec>(mask), 0));",
    "    }",
    "}",
    "",
]


class CxxTSLEmitter(BackendEmitter):
    """C++ TSL v2: one ``static inline`` wrapper per primitive, under the run's
    policy (generic | fixed | clang_fixed) and width mode.  ``vla`` spells every
    vector ``simd_for_t<POLICY<PIVOT_TSL_VLEN_BITS / elem>,T>`` and picks the
    header by an arch×width ``#if`` switch; ``samewidth`` keeps each vector's own
    lane count and includes one header wide enough for the widest.  A mask
    crosses helpers as its native ``mask_type``, a variable of it is ``auto``."""

    target = Target(
        name="tsl",
        companion_filename="pivot_cxx_tsl.h",
        companion_guard="PIVOT_CXX_TSL_H",
        emits_only_used_definitions=True,
        routes_immediates_to_template_args=True,
        masks_are_integers=False,
        writes_cxx=True,
    )
    language = CXX_SYNTAX

    def __init__(self) -> None:
        options = active_options()
        self._policy = options.tsl_policy
        self._width_symbolic = options.tsl_width_mode == WIDTH_MODE_VLA
        # Bodies spell their simd types as the call boundaries do.
        self.definition_expander = IRSignatureExpander(
            label=f"tsl[{options.tsl_policy},{options.tsl_width_mode}]",
            width_symbolic=self._width_symbolic,
            simd_type=self._simd,
        )

    def _simd(self, token: str) -> str | None:
        return tsl_simd_type_for_token(token, self._policy, symbolic=self._width_symbolic)

    def _witness(self, token: str) -> str | None:
        return _mask_witness(token, self._policy, symbolic=self._width_symbolic)

    def accepts_source_register_widths(self, widths: set[int]) -> bool:
        # vla re-widths every vector through one knob, so the source must use one
        # register width; samewidth keeps each op's own.
        return not self._width_symbolic or len(widths) <= 1

    def spell_type(self, token: str) -> str:
        # An immediate is the `int{bits}_t` of its template parameter; scalars and
        # pointers are C++ types already.
        if (bits := immediate_bits(token)) is not None:
            return f"int{bits}_t"
        if is_mask_token(token):
            return "auto"
        return tsl_register_type_for_token(token, self._policy, symbolic=self._width_symbolic) or token

    def reinterpret_syntax(self, source: str, target: str) -> tuple[str, str]:
        src_simd, tgt_simd = self._simd(source), self._simd(target)
        if src_simd is None or tgt_simd is None:
            raise NotImplementedError(
                f"tsl reinterpret needs two data-vector tokens (source={source!r}, target={target!r})"
            )
        return (f"tsl::reinterpret<{src_simd}, {tgt_simd}>(", ")")

    def mask_cast_syntax(self, source: str, target: str) -> tuple[str, str]:
        src_simd, tgt_simd = self._witness(source), self._witness(target)
        if src_simd is None or tgt_simd is None:
            raise NotImplementedError(
                f"tsl mask cast needs two concrete mask tokens (source={source!r}, target={target!r})"
            )
        return (f"{_MASK_CAST_HELPER}<{tgt_simd}, {src_simd}>(", ")")

    def mask_param(self, name: str, token: str) -> Param:
        return Param(self._mask_type(token), name)

    def mask_return(self, token: str, body: list[str], output: str) -> tuple[str, list[str]]:
        return self._mask_type(token), returning(body, output)

    def _mask_type(self, token: str) -> str:
        if (witness := self._witness(token)) is None:
            raise ValueError(f"mask {token} has no concrete element")
        return f"typename {witness}::mask_type"

    def companion_includes(self, definitions: list["Definition"]) -> list[str]:
        return [] if self._width_symbolic else [self._select_profile_header(definitions)]

    def companion_prelude(self, definitions: list["Definition"]) -> list[str]:
        # The VLEN knob, then the header switch that reads it.
        prelude = self._vlen_macro_prelude() + self._vla_include_switch() if self._width_symbolic else []
        return prelude + _MASK_CAST_LINES

    def _select_profile_header(self, definitions: list["Definition"]) -> str:
        """samewidth's one profile header, wide enough for the widest vector."""
        if override := os.environ.get(_TSL_PROFILE_ENV):
            return override
        base = _PROFILE_BASE_AVX512 if _max_vector_bits(definitions) > 256 else _PROFILE_BASE_DEFAULT
        return self._header_name(base)

    @staticmethod
    def _vlen_macro_prelude() -> list[str]:
        return [
            f"#ifndef {_VLEN_MACRO}",
            f"#define {_VLEN_MACRO} 512  // logical vector register width in bits; tune with -D{_VLEN_MACRO}=N",
            "#endif",
            f"static_assert({_VLEN_MACRO} % 128 == 0, "
            f'"{_VLEN_MACRO} must be a whole number of 128-bit registers");',
            "",
        ]

    def _header_name(self, base: str) -> str:
        suffix = "_clang" if self._policy == POLICY_CLANG_FIXED else ""
        return f"tsl_{base}{suffix}.hpp"

    def _tier_includes(self, base: "str | tuple[str, ...]", indent: str) -> list[str]:
        """One tier's `#include`: the first header active, the others commented out."""
        bases = (base,) if isinstance(base, str) else base
        out = [f"{indent}include <{self._header_name(bases[0])}>"]
        for alt in bases[1:]:
            out.append(f"// {indent}include <{self._header_name(alt)}>   // alternative ISA for this tier: uncomment (and comment the line above) to use it")
        return out

    def _vla_include_switch(self) -> list[str]:
        """vla's arch×width `#if` block selecting the TSL profile header."""
        if override := os.environ.get(_TSL_PROFILE_ENV):
            return [f"#include <{override}>", ""]
        lines: list[str] = []
        for idx, (arch_cond, ladder) in enumerate(_VLA_HEADER_LADDER):
            lines.append(f"{'#if' if idx == 0 else '#elif'} {arch_cond}")
            for w_idx, (bits, base) in enumerate(ladder):
                if w_idx == 0:
                    lines.append(f"#  if {_VLEN_MACRO} >= {bits}")
                elif w_idx < len(ladder) - 1:
                    lines.append(f"#  elif {_VLEN_MACRO} >= {bits}")
                else:
                    lines.append("#  else")
                lines.extend(self._tier_includes(base, "#    "))
            lines.append("#  endif")
        lines.append("#else")
        lines.extend(self._tier_includes(_VLA_HEADER_FALLBACK, "#  "))
        lines.append("#endif")
        lines.append("")
        return lines
