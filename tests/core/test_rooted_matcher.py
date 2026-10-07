import pytest

from pivot.frontend.cpp_frontend import CppFrontend
from pivot.driver.run_options import RunOptions, activate
from pivot.ir.graph import Graph
from pivot.ir.graph_matcher import GraphMatcher


@pytest.fixture(autouse=True)
def _cpp():
    activate(RunOptions.build("cpp", "x86", "clang_builtins"))


def _graph(tmp_path, name: str, body: str) -> Graph:
    path = tmp_path / f"{name}.c"
    path.write_text("#include <immintrin.h>\n" + body)
    return Graph(matches=CppFrontend().parse(str(path)).call_statements)


def _pattern(tmp_path, body: str) -> Graph:
    graph = _graph(tmp_path, "pattern", body)
    graph.result_node_id = graph._single_result_node("pattern.c")
    return graph


def _matches(tmp_path, pattern_body: str, world_body: str):
    pattern = _pattern(tmp_path, pattern_body)
    world = _graph(tmp_path, "world", world_body)
    return world, GraphMatcher(world).find_pattern_matches(pattern)


HADD = """
int p(__m128i vec) {
  __m128i t0 = _mm_shuffle_epi32(vec, 177);
  __m128i s0 = _mm_add_epi32(vec, t0);
  __m128i t1 = _mm_shuffle_epi32(s0, 78);
  __m128i s1 = _mm_add_epi32(s0, t1);
  int res = _mm_cvtsi128_si32(s1);
  return res;
}
"""


def test_ladder_matches_once_rooted_at_its_result(tmp_path):
    world, matches = _matches(tmp_path, HADD, """
int f(__m128i v) {
  __m128i a = _mm_shuffle_epi32(v, 0xB1);
  __m128i b = _mm_add_epi32(v, a);
  __m128i c = _mm_shuffle_epi32(b, 78);
  __m128i d = _mm_add_epi32(b, c);
  int r = _mm_cvtsi128_si32(d);
  return r;
}
""")
    assert len(matches) == 1
    (match,) = matches
    assert world.nodes[match.result_node_id].name == "_mm_cvtsi128_si32"
    calls = [w for w in match.mapping.values() if not world.nodes[w].is_constant]
    assert len(calls) == 5 and len(set(calls)) == 5


def test_constant_operand_must_match_by_value(tmp_path):
    _, matches = _matches(tmp_path, HADD, """
int f(__m128i v) {
  __m128i a = _mm_shuffle_epi32(v, 78);
  __m128i b = _mm_add_epi32(v, a);
  __m128i c = _mm_shuffle_epi32(b, 177);
  __m128i d = _mm_add_epi32(b, c);
  return _mm_cvtsi128_si32(d);
}
""")
    assert matches == []


def test_reused_pattern_node_needs_the_same_program_value(tmp_path):
    # The second add reads a different value than the one the second shuffle read.
    _, matches = _matches(tmp_path, HADD, """
int f(__m128i v, __m128i w) {
  __m128i a = _mm_shuffle_epi32(v, 177);
  __m128i b = _mm_add_epi32(v, a);
  __m128i c = _mm_shuffle_epi32(b, 78);
  __m128i b2 = _mm_add_epi32(w, a);
  __m128i d = _mm_add_epi32(b2, c);
  return _mm_cvtsi128_si32(d);
}
""")
    assert matches == []


SAME_INPUT = """
__m128i p(__m128i data) {
  __m128i res = _mm_and_si128(data, data);
  return res;
}
"""


def test_one_pattern_input_binds_one_value(tmp_path):
    _, same = _matches(tmp_path, SAME_INPUT, """
__m128i f(__m128i x) { __m128i r = _mm_and_si128(x, x); return r; }
""")
    assert len(same) == 1

    # Distinct values that share a producer name are still distinct values.
    _, distinct = _matches(tmp_path, SAME_INPUT, """
__m128i f(__m128i *p) {
  __m128i a = _mm_loadu_si128(p);
  __m128i b = _mm_loadu_si128(p + 1);
  __m128i r = _mm_and_si128(a, b);
  return r;
}
""")
    assert distinct == []


def test_distinct_pattern_inputs_may_bind_one_value(tmp_path):
    _, matches = _matches(tmp_path, """
__m128i p(__m128i a, __m128i b) { __m128i res = _mm_and_si128(a, b); return res; }
""", """
__m128i f(__m128i x) { __m128i r = _mm_and_si128(x, x); return r; }
""")
    assert len(matches) == 1
