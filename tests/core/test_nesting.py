"""Hoisting and then nesting gives the user's code back: a temp's assignment goes,
whatever shares its line with it stays."""

import re

import pytest

from pivot.driver.run_options import RunOptions, activate
from pivot.lang.languages import get_language_for_file
from pivot.passes.preprocessing.hoist import HOIST_PASSES
from pivot.passes.preprocessing.nesting import nest_temps
from pivot.passes.translation.emit import UNKNOWN_TYPE_MARKER

SOURCES = {
    "return_after_case": """\
__m128i f(int op, __m128i a, __m128i b) {
  switch (op) {
    case 'f': return _mm_add_epi32(a, b);
    case 'g': return _mm_add_epi32(_mm_mullo_epi32(a, b), _mm_mullo_epi32(b, a));
  }
  return a;
}
""",
    "call_after_goto_label": """\
__m128i f(__m128i a, __m128i b, __m128i c) {
  __m128i x = a;
  goto done;
done: x = _mm_add_epi32(a, _mm_mullo_epi32(b, c));
  return x;
}
""",
    "call_after_default": """\
__m128i f(int op, __m128i a, __m128i b) {
  __m128i x = a;
  switch (op) {
    default: x = _mm_add_epi32(a, _mm_mullo_epi32(a, b));
  }
  return x;
}
""",
    "one_line_block": """\
__m128i f(__m128i b, __m128i c, __m128i d) {
  __m128i a = b;
  { a = _mm_add_epi32(b, _mm_mullo_epi32(c, d)); }
  return a;
}
""",
    "one_line_if_block": """\
__m128i f(int c, __m128i b, __m128i d, __m128i e) {
  __m128i a = b;
  if (c) { a = _mm_add_epi32(b, _mm_mullo_epi32(d, e)); }
  return a;
}
""",
    "define_body_on_its_line": """\
#define MADD(r, a, b) r = _mm_add_epi32(a, _mm_mullo_epi32(a, b));
""",
    "indented_case": """\
__m128i f(int op, __m128i a, __m128i b) {
  switch (op) {
    case 1:
      return _mm_add_epi32(a, b);
  }
  return a;
}
""",
}


def _hoisted(tmp_path, source: str) -> tuple[str, str]:
    """``source`` after the hoist passes, and the path of its copy."""
    activate(RunOptions.build("cpp", "x86", "clang_builtins"))
    path = tmp_path / "input.c"
    path.write_text(f"#include <immintrin.h>\n{source}")
    language = get_language_for_file(str(path))
    for step in language.passes:
        if (hoist := HOIST_PASSES.get(step)) is not None:
            hoist.run(str(path))
    hoisted = path.read_text()
    assert "pivot_tmp_" in hoisted
    return hoisted, str(path)


def _nested(hoisted: str, path: str) -> str:
    return nest_temps(hoisted, get_language_for_file(path).load_locator(path))


def _code(text: str) -> str:
    """``text`` without whitespace and parentheses: nesting may parenthesize a value."""
    return re.sub(r"[\s()]", "", text)


@pytest.mark.parametrize("name", SOURCES)
def test_nesting_restores_the_hoisted_source(tmp_path, name):
    hoisted, path = _hoisted(tmp_path, SOURCES[name])
    nested = _nested(hoisted, path)
    assert "pivot_tmp_" not in nested
    assert _code(nested) == _code(f"#include <immintrin.h>\n{SOURCES[name]}")


def test_indented_case_is_unchanged(tmp_path):
    source = SOURCES["indented_case"]
    assert _nested(*_hoisted(tmp_path, source)) == f"#include <immintrin.h>\n{source}"


def test_unresolved_temp_stays_with_its_marker(tmp_path):
    hoisted, path = _hoisted(tmp_path, SOURCES["return_after_case"])
    marked = hoisted.replace("pivot_tmp_ret_0 =", f"{UNKNOWN_TYPE_MARKER} pivot_tmp_ret_0 =")
    nested = _nested(marked, path)
    assert f"case 'f': {UNKNOWN_TYPE_MARKER} pivot_tmp_ret_0 = _mm_add_epi32(a, b);" in nested
    assert "return pivot_tmp_ret_0;" in nested
