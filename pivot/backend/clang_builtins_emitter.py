from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pivot.backend.backend_emitter import BackendEmitter, Companion, Target
from pivot.backend.helpers import Param, bitmap_bits, build_helpers, expression, returning
from pivot.backend.naming import helper_name
from pivot.driver.run_options import active_options
from pivot.ir.types import immediate_bits, mask_lanes
from pivot.isa.intrinsic_registry import get_intrinsic_registry
from pivot.isa.ir_signature_expander import IRSignatureExpander
from pivot.isa.primitive_registry import get_primitive_registry
from pivot.lang.syntax import CXX_SYNTAX, HelperSignature

if TYPE_CHECKING:
    from pivot.isa.primitive_registry import Definition


_VECTOR_TYPE_PATTERN = re.compile(r"\bv(?P<lanes>\d+)(?P<suffix>[a-z]+)\b")
# A mask body type (`v16b_i`); `_` blocks the word boundary of the pattern above.
_MASK_TYPE_PATTERN = re.compile(r"\bv(?P<lanes>\d+)b(?:_[A-Za-z0-9]+)?\b")

# The primitives a mask crosses a helper boundary through, bool vector -> bitmap and
# back, where a definition says how; else it is a bit_cast.
_TO_BITS, _FROM_BITS = "MASK_BITS", "MASK_FROM_BITS"

_BASE_TYPE_BY_SUFFIX: dict[str, str] = {
    "c": "signed char",
    "uc": "unsigned char",
    "s": "short",
    "us": "unsigned short",
    "i": "int",
    "ui": "unsigned int",
    "l": "long",
    "ul": "unsigned long",
    "f": "float",
    "d": "double",
    "b": "bool",
}


def token_to_c_type(token: str) -> str:
    """The C type of a token: a mask its ``__mmaskN``-like ``uint{N}_t`` bitmap (an
    integer in user code), an immediate ``int{bits}_t``, else the token itself (a
    typedef)."""
    if (bits := immediate_bits(token)) is not None:
        return f"int{bits}_t"
    lanes = mask_lanes(token)
    if lanes is None or (bits := bitmap_bits(lanes)) is None:
        return token
    return f"uint{bits}_t"


class ClangBuiltinsEmitter(BackendEmitter):
    """C with clang vector extensions: a vector is an ``ext_vector_type`` typedef
    named by its token, a mask the x86 ``uint{N}_t`` bitmap."""

    target = Target(
        name="clang_builtins",
        companion_filename="pivot_clang_builtins.h",
        companion_guard="PIVOT_CLANG_BUILTINS_H",
        supports_const_vector_literal=True,
    )
    language = CXX_SYNTAX
    # Its C types are the IR tokens (`v16i` is a typedef): nothing to spell.
    definition_expander = IRSignatureExpander(label="clang_builtins")

    def spell_type(self, token: str) -> str:
        return token_to_c_type(token)

    def reinterpret_syntax(self, source: str, target: str) -> tuple[str, str]:
        return f"({target})(", ")"

    def mask_cast_syntax(self, source: str, target: str) -> tuple[str, str]:
        # A mask is its `uint{N}_t`, so this is C's own integer widen / truncate.
        return f"({token_to_c_type(target)})(", ")"

    def mask_param(self, name: str, token: str) -> Param:
        # A mask crosses as its `uint{N}_t` and is rebound to the bit-packed bool
        # vector the body is written against (incl. `&m`, `sizeof(m)`).
        c_type = token_to_c_type(token)
        if c_type == token:
            return Param(c_type, name)
        return Param(c_type, f"{name}__raw", (f"{token} {name} = {_from_bits(token, f'{name}__raw')};",))

    def mask_return(self, token: str, body: list[str], output: str) -> tuple[str, list[str]]:
        c_type = token_to_c_type(token)
        if c_type == token:
            return c_type, returning(body, output)
        return c_type, _mask_return_lines(body, token, c_type, output)

    def companion(self, definitions: list["Definition"]) -> Companion:
        # Every helper with a mask operand may call a crossing, so the crossings
        # come first; their own mask is the bool vector, not its bitmap.
        crossings = [d for d in definitions if d.primitive.name in (_TO_BITS, _FROM_BITS)]
        helpers = build_helpers(self, [d for d in definitions if d.primitive.name not in (_TO_BITS, _FROM_BITS)])
        lines = self.companion_prelude(definitions)
        for definition in crossings:
            lines += [*self._crossing_helper(definition).splitlines(), ""]
        text = self.language.frame_companion(
            guard=self.target.companion_guard,
            system_includes=self.companion_includes(definitions),
            body_lines=lines + helpers.lines,
        )
        return Companion(text, helpers.collisions)

    def _crossing_helper(self, definition: "Definition") -> str:
        primitive, signature = definition.primitive, definition.signature
        params = [(signature[name], name) for name in primitive.input]
        body = [line.strip() for line in definition.direct if line.strip()]
        helper = HelperSignature(helper_name(definition), signature[primitive.output], params)
        return self.language.render_helper(helper, returning(body, primitive.output))

    def companion_includes(self, definitions: list["Definition"]) -> list[str]:
        return ["stdint.h", "stddef.h", "stdbool.h"]

    def companion_prelude(self, definitions: list["Definition"]) -> list[str]:
        # Permute/table-lookup definitions reach for the ACLE headers; both are guarded
        # so the same companion still compiles on x86, where the ARM arms are dead.
        lines = [
            "#if defined(__ARM_FEATURE_SVE)",
            "#include <arm_sve.h>",
            "#endif",
            "#if defined(__ARM_NEON)",
            "#include <arm_neon.h>",
            "#endif",
            "",
            *self._emit_assumed_vlen_macro(),
            "",
            *self._emit_compare_predicate_constants(),
            "",
        ]
        if aliases := self._emit_vector_aliases(definitions):
            lines += [*aliases, ""]
        return lines

    @staticmethod
    def _emit_assumed_vlen_macro() -> list[str]:
        """The assumed SVE width, for the register width queries (``svcntw``,
        ``LANE_COUNT_*``) that have no operand to read it from."""
        bits = active_options().sve_assumed_bits
        return [
            "// Assumed vector-register width (bits) this translation was generated for.",
            "#ifndef PIVOT_CB_VLEN_BITS",
            f"#define PIVOT_CB_VLEN_BITS {bits}",
            "#endif",
        ]

    @staticmethod
    def _emit_compare_predicate_constants() -> list[str]:
        """The x86 compare predicates (`_MM_CMPINT_*`, `_CMP_*`) the removed SIMD
        header defined, from their `fold:` entries; a header defining them wins."""
        lines = ["// x86 compare-predicate constants (re-provided after SIMD-header removal)."]
        for name, value in get_intrinsic_registry().nullary_fold_constants():
            lines += [f"#ifndef {name}", f"#define {name} {value}", "#endif"]
        return lines

    @staticmethod
    def _emit_vector_aliases(definitions: list["Definition"]) -> list[str]:
        """A typedef for each vector type a signature or body names (a body may
        name one no signature has, as MUL_WIDEN's `v8ul`); a mask is a bool vector."""
        aliases: dict[str, str] = {}
        for definition in definitions:
            for spelling in [*definition.signature.values(), *definition.direct]:
                for lanes_text, suffix in _VECTOR_TYPE_PATTERN.findall(spelling):
                    if suffix in _BASE_TYPE_BY_SUFFIX:
                        lanes = int(lanes_text)
                        aliases.setdefault(
                            f"v{lanes}{suffix}",
                            f"typedef {_BASE_TYPE_BY_SUFFIX[suffix]} v{lanes}{suffix} "
                            f"__attribute__((ext_vector_type({lanes})));",
                        )
                for match in _MASK_TYPE_PATTERN.finditer(spelling):
                    alias = match.group(0)
                    aliases.setdefault(
                        alias, f"typedef bool {alias} __attribute__((ext_vector_type({int(match.group('lanes'))})));"
                    )
        return [aliases[name] for name in sorted(aliases, key=lambda name: (len(name), name))]

def _mask_return_lines(body: list[str], token: str, c_type: str, output: str) -> list[str]:
    """Each returned value bound to a ``token`` temp (clang converts a comparison's
    vector result implicitly), then converted to ``c_type``.  Bits past the lanes
    are unspecified (garbage on AArch64) but zero in an x86 mask, so they are
    cleared."""
    bits = _to_bits(token, c_type, "__pivot_ret")
    lanes = mask_lanes(token)
    width = re.fullmatch(r"uint(\d+)_t", c_type)
    if lanes is not None and width is not None and lanes < int(width.group(1)):
        bits = f"({c_type})({bits} & {hex((1 << lanes) - 1)})"

    if any(line.startswith("return ") for line in body):
        # One block per `return`, so several never collide on the temp.
        return [
            f"{{ {token} __pivot_ret = {expression(line[len('return '):].rstrip())}; return {bits}; }}"
            if line.startswith("return ") else line
            for line in body
        ]
    if len(body) == 1:
        body, value = [], expression(body[0])
    else:
        value = output
    return [*body, f"{token} __pivot_ret = {value};", f"return {bits};"]


def _crossing(primitive: str, mask_slot: str, token: str) -> "Definition | None":
    """The ``primitive`` definition converting a mask of ``token``, if there is one."""
    definitions = get_primitive_registry().definitions_for_primitive(ClangBuiltinsEmitter.target.name, primitive)
    return next((d for d in definitions if d.signature.get(mask_slot) == token), None)


def _to_bits(token: str, c_type: str, value: str) -> str:
    """``value``, a mask of ``token``, as its ``c_type`` bitmap."""
    if (definition := _crossing(_TO_BITS, "k", token)) is not None:
        return f"{helper_name(definition)}({value})"
    return f"__builtin_bit_cast({c_type}, {value})"


def _from_bits(token: str, value: str) -> str:
    """``value``, a bitmap, as a mask of ``token``."""
    if (definition := _crossing(_FROM_BITS, "res", token)) is not None:
        return f"{helper_name(definition)}({value})"
    return f"__builtin_bit_cast({token}, {value})"
