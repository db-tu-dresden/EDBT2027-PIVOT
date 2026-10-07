import pytest

from pivot.frontend.cpp_frontend import CppFrontend
from pivot.driver.run_options import RunOptions, activate
from pivot.ir.graph import Graph
from pivot.ir.graph_matcher import GraphMatcher
from pivot.passes.translation.matching import _kept_calls

LADDER_HEAD = """
  __m128i t0 = _mm_shuffle_epi32(v, 177);
  __m128i s0 = _mm_add_epi32(v, t0);
"""
LADDER_TAIL = """
  __m128i t1 = _mm_shuffle_epi32(s0, 78);
  __m128i s1 = _mm_add_epi32(s0, t1);
"""


@pytest.fixture(autouse=True)
def _cpp():
    activate(RunOptions.build("cpp", "x86", "clang_builtins"))


def _graph(tmp_path, name: str, body: str) -> Graph:
    path = tmp_path / f"{name}.c"
    path.write_text("#include <immintrin.h>\n__m128i g;\nvoid sink(__m128i x);\n" + body)
    unit = CppFrontend().parse(str(path))
    return Graph(matches=unit.call_statements, uses=unit.uses)


def _pattern(tmp_path, body: str) -> Graph:
    pattern = _graph(tmp_path, "pattern", body)
    pattern.result_node_id = pattern._single_result_node("pattern.c")
    return pattern


def _kept(world: Graph, pattern: Graph) -> tuple[list[str], bool]:
    """The names of the calls the match keeps, and whether one of them takes a pointer."""
    calls = {n for n, node in world.nodes.items() if node.kind == "intrinsic" and not node.is_constant}
    (match,) = GraphMatcher(world).find_pattern_matches(pattern)
    intrinsic_ids = frozenset(n for n in match.mapping.values() if n in calls)
    kept = _kept_calls(world, intrinsic_ids, match.result_node_id)
    pointer_calls = {match.mapping[p] for p in pattern.calls_with_pointer_operand()}
    return sorted(world.nodes[n].name for n in kept), bool(kept & pointer_calls)


def _ladder_kept(tmp_path, signature: str, use: str, tail: str = "") -> list[str]:
    pattern = _pattern(tmp_path, "int p(__m128i v) {" + LADDER_HEAD + LADDER_TAIL
                       + "  int res = _mm_cvtsi128_si32(s1);\n  return res;\n}\n")
    world = _graph(tmp_path, "world", signature + " {" + LADDER_HEAD + use + LADDER_TAIL
                   + "  int r = _mm_cvtsi128_si32(s1);\n" + tail + "  return r;\n}\n")
    kept, touches_memory = _kept(world, pattern)
    assert not touches_memory
    return kept


def test_straight_line_ladder_keeps_nothing(tmp_path):
    assert _ladder_kept(tmp_path, "int f(__m128i v)", "") == []


@pytest.mark.parametrize("signature, use", [
    ("int f(__m128i v, __m128i *out)", "  _mm_storeu_si128(out, s0);\n"),
    ("int f(__m128i v)", "  sink(s0);\n"),
    ("int f(__m128i v)", "  g = s0;\n"),
    ("int f(__m128i v, __m128i *out)", "  *out = s0;\n"),
    ("int f(__m128i v, __m128i *out)", "  out[1] = s0 & v;\n"),
    ("int f(__m128i v)", "  if (sizeof(s0)) g = v;\n"),
])
def test_intermediate_used_outside_the_match_is_kept_with_its_feeders(tmp_path, signature, use):
    assert _ladder_kept(tmp_path, signature, use) == ["_mm_add_epi32", "_mm_shuffle_epi32"]


def test_read_after_the_result_sees_the_intermediate(tmp_path):
    assert _ladder_kept(tmp_path, "int f(__m128i v)", "", tail="  g = t0;\n") == ["_mm_shuffle_epi32"]


def test_reads_of_a_reassigned_variable_see_the_new_value(tmp_path):
    # `s0` is overwritten before the read, so the ladder's s0 is not read there.
    assert _ladder_kept(tmp_path, "int f(__m128i v)", "", tail="  s0 = _mm_setzero_si128();\n  g = s0;\n") == []


LOAD_ADD = """
  __m128i x = _mm_loadu_si128(p);
  __m128i y = _mm_add_epi32(x, x);
"""


def test_a_kept_load_would_run_twice(tmp_path):
    pattern = _pattern(tmp_path, "int f(const __m128i *p) {" + LOAD_ADD + "  int r = _mm_cvtsi128_si32(y);\n  return r;\n}\n")
    world = _graph(tmp_path, "world", "int f(const __m128i *p) {" + LOAD_ADD + "  g = x;\n"
                   + "  int r = _mm_cvtsi128_si32(y);\n  return r;\n}\n")
    assert _kept(world, pattern) == (["_mm_loadu_si128"], True)


def test_a_load_used_only_inside_the_match_is_not_kept(tmp_path):
    body = "int f(const __m128i *p) {" + LOAD_ADD + "  int r = _mm_cvtsi128_si32(y);\n  return r;\n}\n"
    assert _kept(_graph(tmp_path, "world", body), _pattern(tmp_path, body)) == ([], False)
