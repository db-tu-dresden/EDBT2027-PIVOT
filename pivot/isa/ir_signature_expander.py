"""Expands a target definition: binds its IR tokens (`{v}`, `{s}`, `{vbool}`, ...)
at its dtype over register widths, then spells its `{simd_<param>}` types."""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from pivot.driver.run_options import active_options
from pivot.ir.types import (
    IRKind,
    IRType,
    concretize_scalable_token,
    data_vector_shape,
    is_mask_token,
    mask_lanes,
    mask_witness_data_token,
    spell_ir_token,
)
from pivot.isa.definition_expander import note_dropped
from pivot.isa.templates import (
    DERIVED_RE,
    SIMD_PARAM_RE,
    TOKEN_RE,
    WIDTH_QUALIFIED_RE,
    WIDTH_TOKENS,
    is_simd_param,
    parse_width_list,
    split_dtype,
    substitute,
    tokens_in,
)

if TYPE_CHECKING:
    from pivot.isa.definition_expander import ExpansionRequest

_VECTOR_WIDTHS_BITS: tuple[int, ...] = (128, 256, 512, 1024, 2048)
# Data-vector element suffixes; a mask `v{lanes}b_{elem}` names its element by one.
_VECTOR_SUFFIX: dict[str, dict[int, str]] = {
    "int":   {8: "c",  16: "s",  32: "i",  64: "l"},
    "uint":  {8: "uc", 16: "us", 32: "ui", 64: "ul"},
    "float": {32: "f", 64: "d"},
}
_SCALAR: dict[str, dict[int, str]] = {
    "int":   {8: "int8_t",   16: "int16_t",   32: "int32_t",   64: "int64_t"},
    "uint":  {8: "uint8_t",  16: "uint16_t",  32: "uint32_t",  64: "uint64_t"},
    "float": {32: "float",   64: "double"},
    "bool":  {8: "bool",     16: "bool",      32: "bool",      64: "bool"},
}


def _ir_replacements(family: str, elem_bits: int, lanes: int | None, tokens: set[str]) -> dict[str, str]:
    """The IR spelling of each token in ``tokens`` at ``lanes`` lanes of a
    ``family``/``elem_bits`` element; ``lanes=None`` spells them length-open."""
    n = "" if lanes is None else str(lanes)

    def vector(fam: str) -> str:
        return f"v{n}{_VECTOR_SUFFIX[fam][elem_bits]}"

    def scalar(fam: str) -> str:
        return _SCALAR[fam][elem_bits]

    spell: dict[str, Any] = {
        # A whole mask, whatever its element.
        "vbool_all": lambda: f"v{n}b_bALL",
        "elem_bits": lambda: str(elem_bits),
    }
    # A mask-only (bool) primitive has no data element: only `{vbool_all}` binds.
    if family != "bool":
        spell.update({
            "v": lambda: vector(family), "vint": lambda: vector("int"),
            "vuint": lambda: vector("uint"), "vfloat": lambda: vector("float"),
            "vbool": lambda: f"v{n}b_{_VECTOR_SUFFIX[family][elem_bits]}",
            "s": lambda: scalar(family), "sint": lambda: scalar("int"),
            "suint": lambda: scalar("uint"), "sfloat": lambda: scalar("float"),
            "sbool": lambda: scalar("bool"),
            # An int32 index vector of the same lane count (a gather's indices).
            "vidx": lambda: f"v{n}i",
        })
    # Lazily: a cross-family token may have no spelling at this element width.
    return {t: spell[t]() for t in tokens if t in spell}


def _width_fields(
    signature: dict[str, str], widths: list[int]
) -> tuple[dict[str, tuple[str, list[int]]], list[str]]:
    """The width-qualified vector parameters with their token and widths, and the
    plain vector parameters, which sweep every width."""
    qualified: dict[str, tuple[str, list[int]]] = {}
    plain: list[str] = []
    for name, raw_value in signature.items():
        value = str(raw_value).strip()
        if m := WIDTH_QUALIFIED_RE.fullmatch(value):
            allowed = [w for w in parse_width_list(m.group("widths")) if w in widths]
            if not allowed:
                raise ValueError(f"Width-qualified token for '{name}' has no valid widths: {value}")
            qualified[name] = (m.group("token"), allowed)
        elif (m := TOKEN_RE.fullmatch(value)) and m.group(1) in WIDTH_TOKENS:
            plain.append(name)
    return qualified, plain


def _resolve_derived_signature(signature: dict[str, str], primitive: str) -> dict[str, str]:
    """``signature`` with each ``{elem@width}`` value replaced by the vector of
    parameter ``elem``'s element at parameter ``width``'s register width."""
    resolved = dict(signature)
    for name, value in signature.items():
        m = DERIVED_RE.fullmatch(str(value).strip())
        if not m:
            continue
        elem_ref, width_ref = m.group("elem"), m.group("width")
        elem_tok = str(signature.get(elem_ref, "")).strip()
        width_tok = str(signature.get(width_ref, "")).strip()
        elem_shape = data_vector_shape(elem_tok)
        width_shape = data_vector_shape(width_tok)
        if elem_shape is None or width_shape is None:
            raise ValueError(
                f"{primitive}: derived signature '{name}: {value}' references a "
                f"non-data-vector param (elem {elem_ref!r}={elem_tok!r}, "
                f"width {width_ref!r}={width_tok!r})"
            )
        _, elem_domain, elem_bits = elem_shape
        width_lanes, _, width_elem_bits = width_shape
        total_bits = width_lanes * width_elem_bits.value
        if total_bits % elem_bits.value != 0:
            raise ValueError(
                f"{primitive}: derived signature '{name}: {value}' — width "
                f"{total_bits} bits is not a whole multiple of the "
                f"{elem_bits.value}-bit element"
            )
        resolved[name] = spell_ir_token(
            IRType(IRKind.VECTOR, elem_domain, elem_bits, total_bits // elem_bits.value)
        )
    return resolved


def _lanes_of(token: str) -> int | None:
    shape = data_vector_shape(token)
    return shape[0] if shape is not None else mask_lanes(token)


@dataclass(frozen=True)
class IRSignatureExpander:
    """How one target expands its definitions."""

    # Names the target in drop notes.
    label: str
    # One length-open variant per dtype instead of a register-width sweep, for a
    # target whose width is a compile-time knob.  Width-qualified and concrete
    # signatures still sweep.
    width_symbolic: bool = False
    # The target's simd type of a vector token, which `{simd_<param>}` spells.
    simd_type: Callable[[str], str | None] | None = None
    # The most lanes a vector of the target may have.
    max_lanes: int | None = None

    def expand(self, definition: dict[str, Any], request: ExpansionRequest) -> list[dict[str, Any]]:
        variants = self._bind_ir_tokens(definition, request.primitive)
        if self.simd_type is not None:
            spelled = (self._spell_simd_params(v, request.primitive) for v in variants)
            variants = [v for v in spelled if v is not None]
        if self.max_lanes is not None:
            variants = [v for v in variants if self._fits(v, request.primitive)]
        return variants

    def _unresolved(self, tokens: set[str]) -> set[str]:
        """The tokens left once IR tokens are bound; `{simd_<param>}` is spelled later."""
        if self.simd_type is None:
            return tokens
        return {t for t in tokens if not is_simd_param(t)}

    def _bind_ir_tokens(self, definition: dict[str, Any], primitive: str) -> list[dict[str, Any]]:
        template = definition["signature"]
        tokens = tokens_in([*(str(v) for v in template.values()), *definition["direct"]])
        dtype = definition["dtype"]
        family, elem_bits = split_dtype(dtype)
        if family is None or elem_bits is None:
            if not self._unresolved(tokens):
                return [dict(definition)]
            note_dropped(primitive, self.label, f"dtype={dtype!r} is not concrete")
            return []

        @functools.cache
        def replacements(lanes: int | None) -> dict[str, str]:
            return _ir_replacements(family, elem_bits, lanes, tokens)

        def bind(signature: dict[str, str], lanes: int | None, param_lanes: dict[str, int]) -> dict[str, Any] | None:
            """The definition at ``lanes`` lanes (a parameter of ``param_lanes`` at
            its own), or None after a note if a token stays unresolved."""
            bound = {
                name: substitute(str(value), replacements(param_lanes.get(name, lanes)))
                for name, value in signature.items()
            }
            direct = [substitute(line, replacements(lanes)) for line in definition["direct"]]
            if unresolved := self._unresolved(tokens_in([*bound.values(), *direct])):
                note_dropped(primitive, self.label, f"dtype={dtype}, unresolved_tokens={sorted(unresolved)}")
                return None
            return {**definition, "signature": bound, "direct": direct}

        variants = (bind(*binding) for binding in self._lane_bindings(template, family, elem_bits))
        return [v for v in variants if v is not None]

    def _lane_bindings(
        self, template: dict[str, str], family: str, elem_bits: int
    ) -> list[tuple[dict[str, str], int | None, dict[str, int]]]:
        """Each ``(signature, lanes, param_lanes)`` to bind: the lanes of the
        result, and of each parameter that has a width of its own."""
        # boolN is N lanes of bool, not an element width to sweep.
        if family == "bool":
            return [(template, elem_bits, {})]
        widths = [w for w in _VECTOR_WIDTHS_BITS if w % elem_bits == 0]
        qualified, plain = _width_fields(template, widths)
        if self.width_symbolic and plain and not qualified:
            return [(template, None, {})]
        # A width-qualified parameter binds its bare token at its own width.
        swept = {**template, **{name: "{" + token + "}" for name, (token, _) in qualified.items()}}
        bindings = []
        for default in widths if plain else [None]:
            for combo in itertools.product(*(w for _, w in qualified.values())):
                param_widths = dict(zip(qualified, combo))
                if default is not None:
                    param_widths.update({name: default for name in plain})
                # The direct body and the other parameters take the result's width.
                anchor = param_widths.get("res") or default or next(iter(param_widths.values()), widths[0])
                bindings.append((swept, anchor // elem_bits, {name: w // elem_bits for name, w in param_widths.items()}))
        return bindings

    def _spell_simd_params(self, variant: dict[str, Any], primitive: str) -> dict[str, Any] | None:
        """``variant`` with each ``{simd_<param>}`` spelled from that parameter's
        token, or None after a note if one has no simd type."""
        signature = variant["signature"]
        # A fixed-width target pins length-open tokens (an SVE WHILELT's `vd`) to
        # the assumed SVE width, as typing pins source operands.
        if not self.width_symbolic:
            bits = active_options().sve_assumed_bits
            signature = {name: concretize_scalable_token(str(token), bits) for name, token in signature.items()}
        signature = _resolve_derived_signature(signature, primitive)

        replacements: dict[str, str] = {}
        unresolved: set[str] = set()
        for line in variant["direct"]:
            for match in SIMD_PARAM_RE.finditer(line):
                param = match.group("param")
                token = str(signature.get(param, "")).strip()
                # A mask is spelled by its witness data vector, as the emitter does.
                data_token = mask_witness_data_token(token) if is_mask_token(token) else token
                simd_type = self.simd_type(data_token) if data_token else None
                if simd_type is None:
                    unresolved.add(match.group(0))
                else:
                    replacements[f"simd_{param}"] = simd_type
        if unresolved:
            note_dropped(primitive, self.label, f"unresolved_simd_params={sorted(unresolved)}")
            return None
        return {
            **variant,
            "signature": signature,
            "direct": [substitute(line, replacements) for line in variant["direct"]],
        }

    def _fits(self, variant: dict[str, Any], primitive: str) -> bool:
        lanes = (_lanes_of(token) for token in variant["signature"].values())
        if over := sorted({n for n in lanes if n is not None and n > self.max_lanes}):
            note_dropped(
                primitive, self.label,
                f"reason=lane count exceeds the target's supported maximum ({self.max_lanes}): {over}",
            )
            return False
        return True
