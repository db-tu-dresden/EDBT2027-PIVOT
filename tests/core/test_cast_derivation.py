"""End-to-end cast-derivation tests.

Drives the real single-file pipeline (cpp, x86 -> clang_builtins) on tiny
snippets and asserts on the reinterpret casts in the translated output. Every
SIMD value's type is decided by its first producer; a same-size lane-view change
at any other edge must surface as an explicit reinterpret, and matching views
must not be over-cast.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from main import run_single_file_pipeline

CLANG = shutil.which("clang")
AVX = ["-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl", "-mavx512cd", "-mavx512vbmi"]


def _translate(tmp_path: Path, name: str, src: str) -> tuple[str, Path]:
    """Translate one snippet; return (output_text, companion_include_dir)."""
    source = tmp_path / name
    source.write_text(textwrap.dedent(src).strip() + "\n")
    out_root = tmp_path / "out"
    out_file = run_single_file_pipeline(
        source_path=str(source),  # language is inferred from the .c extension
        source_isa="x86",
        target_isa="clang_builtins",
        args=[],
        output_root=str(out_root),
    )
    companion_dir = out_root / ".pivot" / "debug" / name
    return Path(out_file).read_text(), companion_dir


def _assert_compiles(tmp_path: Path, name: str, text: str, companion_dir: Path) -> None:
    """clang -fsyntax-only must be clean (skipped when clang is unavailable)."""
    if CLANG is None:
        pytest.skip("clang not available")
    gen = tmp_path / f"{name}.gen.c"
    gen.write_text(text)
    proc = subprocess.run(
        [CLANG, "-x", "c", "-std=gnu11", "-D_GNU_SOURCE", *AVX,
         "-fsyntax-only", "-w", "-I", str(companion_dir), str(gen)],
        capture_output=True, text=True,
    )
    errors = [ln for ln in proc.stderr.splitlines() if "error:" in ln.lower()]
    assert not errors, "translated snippet failed to compile:\n" + "\n".join(errors)


def test_differing_lane_view_inserts_reinterpret(tmp_path):
    """make_vec produces a 64-bit view; the epi32 add consumes a 32-bit view."""
    text, comp = _translate(tmp_path, "diff.c", """
        #include <immintrin.h>
        __m512i make_vec(long long v) {
            return _mm512_set1_epi64(v);
        }
        __m512i use_it(long long v, __m512i other) {
            return _mm512_add_epi32(make_vec(v), other);
        }
    """)
    assert "v8l make_vec" in text                                # producer typed v8l
    assert re.search(r"\(v16i\)\(+\s*make_vec", text)            # reinterpret at the add
    _assert_compiles(tmp_path, "diff", text, comp)


def test_matching_lane_view_has_no_cast(tmp_path):
    """Producer and consumer agree (epi32 both) -> no reinterpret."""
    text, comp = _translate(tmp_path, "same.c", """
        #include <immintrin.h>
        __m512i make_vec32(int v) {
            return _mm512_set1_epi32(v);
        }
        __m512i use_it(int v, __m512i other) {
            return _mm512_add_epi32(make_vec32(v), other);
        }
    """)
    assert "v16i make_vec32(int v)" in text
    assert "(v8l)(" not in text and "(v16i)(" not in text        # no over-casting
    _assert_compiles(tmp_path, "same", text, comp)


def test_userfunc_temp_typed_as_producer(tmp_path):
    """A temp fed by a user function is typed by that function, cast at the use."""
    text, comp = _translate(tmp_path, "temp.c", """
        #include <immintrin.h>
        __m512i make_vec(long long v) {
            return _mm512_set1_epi64(v);
        }
        __m512i use_it(long long v, __m512i other) {
            __m512i t = make_vec(v);
            return _mm512_add_epi32(t, other);
        }
    """)
    assert "v8l t = make_vec(v)" in text                         # temp = producer type
    assert re.search(r"\(v16i\)\(+\s*t\s*\)", text)              # one cast at the add
    _assert_compiles(tmp_path, "temp", text, comp)


def test_parameter_fed_into_differently_viewed_slot(tmp_path):
    """A parameter typed by its caller (v16i) feeds an epi64 slot -> cast on it."""
    text, comp = _translate(tmp_path, "param.c", """
        #include <immintrin.h>
        static __m512i shift_it(__m512i v) {
            return _mm512_srli_epi64(v, 5);
        }
        __m512i run(int a) {
            __m512i x = _mm512_set1_epi32(a);
            return shift_it(x);
        }
    """)
    assert "shift_it(v16i v)" in text                            # param typed by caller
    assert re.search(r"\(v8l\)\(+\s*v\s*\)", text)               # cast inside the body
    _assert_compiles(tmp_path, "param", text, comp)


def test_reassignment_casts_at_non_first_producer(tmp_path):
    """First producer fixes the type; a later differently-viewed producer casts."""
    text, comp = _translate(tmp_path, "reassign.c", """
        #include <immintrin.h>
        __m512i reassign(int a, long long b) {
            __m512i x = _mm512_set1_epi32(a);
            x = _mm512_set1_epi64(b);
            return _mm512_add_epi32(x, x);
        }
    """)
    assert "v16i x = " in text                                   # first producer wins
    assert re.search(r"x = \(v16i\)\(+\s*pivot_set1_int64", text)  # cast at reassignment
    _assert_compiles(tmp_path, "reassign", text, comp)


def test_mask_widen_inserts_cast(tmp_path):
    """A mask's type comes from its producer, an op's mask slot from the op; a
    disagreement is bridged by an explicit mask cast, as for data vectors.

    `cmpgt_epi64_mask` produces an 8-lane mask (clang `uint8_t`); `kortestz` is a
    16-lane mask op, so each operand is widened to the op's 16-lane `uint16_t`, the
    zero-extend x86 did implicitly on `__mmask8 -> __mmask16`.
    """
    text, comp = _translate(tmp_path, "mask_widen.c", """
        #include <immintrin.h>
        int run(__m512i a, __m512i b) {
            __mmask16 k = _mm512_cmpgt_epi64_mask(a, b);
            return _mm512_kortestz(k, k);
        }
    """)
    assert re.search(r"\(uint16_t\)\(+\s*k\s*\)", text)          # widen cast on the mask arg
    _assert_compiles(tmp_path, "mask_widen", text, comp)


def test_matching_mask_view_has_no_cast(tmp_path):
    """Producer and consuming op agree on the mask type -> no cast.

    `cmpgt_epu32_mask` produces a 16-lane unsigned mask; `kortestz` names its slot
    16-lane unsigned too (a pure-mask op is unsigned), so the tokens match and no
    mask cast is inserted (masks are not over-cast, mirroring the vector rule).
    """
    text, comp = _translate(tmp_path, "mask_same.c", """
        #include <immintrin.h>
        int run(__m512i a, __m512i b) {
            __mmask16 k = _mm512_cmpgt_epu32_mask(a, b);
            return _mm512_kortestz(k, k);
        }
    """)
    assert "(uint16_t)(" not in text and "(uint8_t)(" not in text  # no mask over-cast
    _assert_compiles(tmp_path, "mask_same", text, comp)


def test_zero_arg_function_return_types_value(tmp_path):
    """A `f(void)` user function has no parameters: its zero-argument call still
    resolves, so the call's return types the assigned value (producer-first) and
    the differently-viewed consumer casts."""
    text, comp = _translate(tmp_path, "voidfn.c", """
        #include <immintrin.h>
        static inline __m256i setall_si256(void) {
            __m256i x = _mm256_undefined_si256();
            return _mm256_cmpeq_epi32(x, x);
        }
        __m512i kernel(void) {
            const __m256i m = setall_si256();
            return _mm512_cvtepu16_epi32(m);
        }
    """)
    assert "v8i m = setall_si256()" in text                        # producer types the value
    assert re.search(r"\(v16us\)\(+\s*m\s*\)", text)              # cast at the widening consumer
    _assert_compiles(tmp_path, "voidfn", text, comp)
