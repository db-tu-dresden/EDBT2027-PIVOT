import pytest

from pivot.driver.run_options import RunOptions, activate
from pivot.frontend.base import UseContext as C
from pivot.frontend.cpp_frontend import CppFrontend
from pivot.isa.intrinsic_registry import get_intrinsic_registry
from pivot.ir.value_flow import _reads_integer

SOURCE = """#include <immintrin.h>
int user(__mmask16 k);
__mmask16 f(__m512i *p, __m512i a, __mmask16 m, uint64_t u) {
  __m512i x = a;
  __mmask16 k = _mm512_cmpeq_epi32_mask(x, a);
  u |= m;
  if (k) user(m);
  p[0] = x;
  __m512i y = p[1];
  return k;
}
"""


@pytest.fixture
def uses(tmp_path):
    activate(RunOptions.build("cpp", "x86", "clang_builtins"))
    path = tmp_path / "uses.c"
    path.write_text(SOURCE)
    return CppFrontend().parse(str(path)).uses


def test_each_read_has_its_context(uses):
    assert [
        (u.decl.name, u.extent.start.line, u.context, u.other and u.other.name, u.call and u.call.name)
        for u in uses
    ] == [
        ("a", 4, C.COPY, "x", None),
        ("x", 5, C.INTRINSIC_ARGUMENT, None, "_mm512_cmpeq_epi32_mask"),
        ("a", 5, C.INTRINSIC_ARGUMENT, None, "_mm512_cmpeq_epi32_mask"),
        ("u", 6, C.ASSIGNED, None, None),
        ("m", 6, C.COMPOUND_VALUE, "u", None),
        ("k", 7, C.EXPRESSION, None, None),
        ("m", 7, C.ARGUMENT, None, "user"),
        ("p", 8, C.ELEMENT_STORE, "x", None),
        ("x", 8, C.COPY, None, None),
        ("p", 9, C.ELEMENT_LOAD, "y", None),
        ("k", 10, C.RETURN, None, None),
    ]


def test_integer_reads_of_masks(uses):
    registry = get_intrinsic_registry()
    reads = [
        (u.decl.name, u.extent.start.line) for u in uses
        if registry.is_integer_mask_type(u.decl.dtype) and _reads_integer(u, registry)
    ]
    # Not the intrinsic operand, nor the mask the mask-typed function returns.
    assert reads == [("m", 6), ("k", 7), ("m", 7)]
