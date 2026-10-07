"""The companion's helper functions, built alike for every target: one helper per
name, in name order.  Each operand crosses the helper boundary by its kind; the
target spells only how a mask crosses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pivot.backend.naming import helper_name
from pivot.ir.types import is_immediate_token, is_mask_token
from pivot.lang.syntax import HelperSignature

if TYPE_CHECKING:
    from pivot.backend.backend_emitter import BackendEmitter
    from pivot.isa.primitive_registry import Definition


@dataclass(frozen=True)
class Param:
    """A helper parameter, and the body lines rebinding it to the form the body
    is written against."""

    type: str
    name: str
    rebind: tuple[str, ...] = ()


@dataclass(frozen=True)
class Helpers:
    lines: list[str]
    # Each helper name several distinct definitions share, with them (the emitted one first).
    collisions: dict[str, list["Definition"]]


def definition_key(definition: "Definition") -> tuple:
    """Definitions with one key are one helper."""
    return (
        definition.primitive.name,
        tuple(sorted(definition.signature.items())),
        tuple(definition.direct),
    )


def build_helpers(backend: "BackendEmitter", definitions: list["Definition"]) -> Helpers:
    """The helpers realizing ``definitions``, as companion lines."""
    by_name: dict[str, dict[tuple, "Definition"]] = {}
    for definition in definitions:
        group = by_name.setdefault(helper_name(definition), {})
        group.setdefault(definition_key(definition), definition)
    lines: list[str] = []
    for symbol in sorted(by_name):
        first = next(iter(by_name[symbol].values()))
        lines.extend(_helper(backend, first, symbol).splitlines())
        lines.append("")
    collisions = {name: list(group.values()) for name, group in by_name.items() if len(group) > 1}
    return Helpers(lines, collisions)


def returning(body: list[str], output: str) -> list[str]:
    """``body`` returning its value: a lone expression is returned, a longer body
    returns its ``output`` variable."""
    if any(line.startswith("return ") for line in body):
        return body
    if len(body) == 1:
        return [f"return {expression(body[0])};"]
    return [*body, f"return {output};"]


def expression(statement: str) -> str:
    """The expression of an expression statement."""
    return statement[:-1].rstrip() if statement.endswith(";") else statement


def bitmap_bits(lanes: int) -> int | None:
    """Width of the narrowest integer (up to 64 bits) holding a mask's bitmap."""
    return next((bits for bits in (8, 16, 32, 64) if lanes <= bits), None)


def _helper(backend: "BackendEmitter", definition: "Definition", symbol: str) -> str:
    try:
        return _render(backend, definition, symbol)
    except ValueError as error:
        raise ValueError(
            f"{backend.target.name}: {symbol}: {error} (signature={definition.signature})"
        ) from error


def _render(backend: "BackendEmitter", definition: "Definition", symbol: str) -> str:
    primitive, signature = definition.primitive, definition.signature
    params: list[tuple[str, str]] = []
    template_params: list[tuple[str, str]] = []
    type_params: list[str] = []
    rebind: list[str] = []
    for name in primitive.input:
        token = signature[name]
        if is_immediate_token(token):
            slot = template_params if backend.target.routes_immediates_to_template_args else params
            slot.append((backend.spell_type(token), name))
        elif (pointer := backend.generic_pointer(token)) is not None:
            # Any pointee binds: the pointer takes a fresh type parameter.
            generic = f"P{len(type_params)}" if type_params else "P"
            type_params.append(generic)
            params.append((f"{pointer} {generic}", name))
        elif is_mask_token(token):
            param = backend.mask_param(name, token)
            params.append((param.type, param.name))
            rebind.extend(param.rebind)
        else:
            params.append((backend.spell_type(token), name))

    body = [line.strip() for line in definition.direct if line.strip()]
    output = primitive.output
    out_token = signature[output] if output else None
    if not out_token:
        return_type, lines = "void", body
    elif is_mask_token(out_token):
        return_type, lines = backend.mask_return(out_token, body, output)
    else:
        return_type, lines = backend.spell_type(out_token), returning(body, output)
    helper = HelperSignature(symbol, return_type, params, template_params, type_params)
    return backend.language.render_helper(helper, rebind + lines)
