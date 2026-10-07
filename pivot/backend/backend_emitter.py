"""A target backend: the record of what the target is, the spelling hooks each
emitter supplies, and the companion every target builds alike."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, NamedTuple, TYPE_CHECKING

from pivot.backend.helpers import Param, build_helpers
from pivot.backend.naming import helper_name
from pivot.ir.conversions import Conversion
from pivot.ir.types import is_immediate_token
from pivot.isa.primitive_registry import get_primitive_registry

if TYPE_CHECKING:
    from pivot.isa.ir_signature_expander import IRSignatureExpander
    from pivot.isa.primitive_registry import Definition
    from pivot.lang.syntax import LanguageSyntax


@dataclass(frozen=True)
class Target:
    """What a target is, independent of the run settings."""

    name: str
    companion_filename: str
    # C include guard; the Rust companion's first comment line.
    companion_guard: str
    # The companion holds only the definitions a translation calls (the library
    # ships a subset), else every definition of the target.
    emits_only_used_definitions: bool = False
    # Immediates (`imm{bits}`) are template arguments, as TSL's `shift_left_imm<Vec, shift>` needs.
    routes_immediates_to_template_args: bool = False
    # A load of a constant array may become an inline vector literal.
    supports_const_vector_literal: bool = False
    # A mask variable is an integer bitmap (x86 `__mmaskN`); else reading it as one calls MASK_BITS.
    masks_are_integers: bool = True
    # Lines that must open a translated file, ahead of the companion binding (a
    # Rust crate-root attribute is honoured only there).
    file_prologue: tuple[str, ...] = ()
    # It writes C++, into a `.c` file too, so nesting parses its output as C++.
    writes_cxx: bool = False


@dataclass(frozen=True)
class Companion:
    """A companion unit, and each helper name several distinct definitions share
    (with them, the emitted one first)."""

    text: str
    collisions: dict[str, list["Definition"]]


class ConversionSpelling(NamedTuple):
    """A conversion as the text around the converted expression."""

    prefix: str
    suffix: str
    # The companion definition it calls, if any.
    definition: "Definition | None" = None


class BackendEmitter(ABC):
    """A target's vocabulary: its types, conversions and how a mask crosses a
    helper boundary.  The host language (:attr:`language`) spells calls, helpers
    and the companion frame.  Shared passes reach a target through these methods
    only."""

    target: ClassVar[Target]
    language: ClassVar["LanguageSyntax"]
    # Turns this backend's shorthand `isa:` definitions into concrete ones.
    definition_expander: "IRSignatureExpander"

    def companion(self, definitions: list["Definition"]) -> Companion:
        """The companion unit of the helpers realizing ``definitions``."""
        helpers = build_helpers(self, definitions)
        text = self.language.frame_companion(
            guard=self.target.companion_guard,
            system_includes=self.companion_includes(definitions),
            body_lines=self.companion_prelude(definitions) + helpers.lines,
        )
        return Companion(text, helpers.collisions)

    def companion_includes(self, definitions: list["Definition"]) -> list[str]:
        """System headers the companion includes."""
        return []

    def companion_prelude(self, definitions: list["Definition"]) -> list[str]:
        """Companion lines ahead of the helpers."""
        return []

    def companion_binding(self) -> list[str]:
        """The directive line(s) binding the companion into a translated source."""
        return self.language.companion_binding(self.target.companion_filename)

    def emit_call(self, definition: "Definition", args: list[str]) -> str:
        """A call of the helper realizing ``definition``; ``args`` align with its
        inputs, and immediates move to template position if the target says so."""
        symbol = helper_name(definition)
        if not self.target.routes_immediates_to_template_args:
            return self.language.emit_call(symbol, args)
        runtime_args: list[str] = []
        template_args: list[str] = []
        for name, arg in zip(definition.primitive.input, args):
            token = definition.signature.get(name, "")
            (template_args if is_immediate_token(token) else runtime_args).append(arg)
        return self.language.emit_call(symbol, runtime_args, template_args=template_args or None)

    @abstractmethod
    def spell_type(self, token: str) -> str:
        """The target type a value of IR ``token`` has (``v16i``, a clang mask's
        ``uint16_t``)."""

    def declared_type(self, token: str, pointer: bool = False) -> str:
        """How a variable of ``token`` is declared, ``T *`` for a pointer to it."""
        return self.language.normalize_declared_type(self.spell_type(token), pointer)

    def vector_literal(self, token: str, elements_text: str) -> str:
        """An inline vector literal of ``token`` from ``elements_text``."""
        return self.language.vector_literal(token, elements_text)

    def accepts_source_register_widths(self, widths: set[int]) -> bool:
        """Whether a kernel whose vector registers span these bit widths can be
        lowered; a width-parametric target takes one."""
        return True

    def generic_pointer(self, token: str) -> str | None:
        """The pointer prefix (``*const``) when an untyped pointer ``token`` is
        passed as a generic type parameter, else None."""
        return None

    @abstractmethod
    def mask_param(self, name: str, token: str) -> Param:
        """How mask operand ``name`` of ``token`` crosses the helper boundary."""

    @abstractmethod
    def mask_return(self, token: str, body: list[str], output: str) -> tuple[str, list[str]]:
        """The return type and body lines of a helper returning a mask ``token``."""

    def spell_conversion(self, kind: Conversion, value_token: str, slot_token: str) -> ConversionSpelling | None:
        """The conversion :func:`~pivot.ir.conversions.conversion` chose for a
        value of ``value_token`` read as ``slot_token``; None if none is spelled."""
        if kind is Conversion.REINTERPRET:
            return ConversionSpelling(*self.reinterpret_syntax(value_token, slot_token))
        if kind is Conversion.MASK_CAST:
            return ConversionSpelling(*self.mask_cast_syntax(value_token, slot_token))
        if kind is Conversion.MASK_BITS and not self.target.masks_are_integers:
            to_bits = {
                d.signature["k"]: d
                for d in get_primitive_registry().definitions_for_primitive(self.target.name, "MASK_BITS")
            }
            if (definition := to_bits.get(value_token)) is not None:
                return ConversionSpelling(f"{helper_name(definition)}(", ")", definition)
        return None

    @abstractmethod
    def reinterpret_syntax(self, source: str, target: str) -> tuple[str, str]:
        """A same-size reinterpret from token ``source`` to ``target``, as the
        ``(prefix, suffix)`` around the expression."""

    @abstractmethod
    def mask_cast_syntax(self, source: str, target: str) -> tuple[str, str]:
        """A mask cast from ``source`` to ``target`` as ``(prefix, suffix)``: C's
        implicit ``__mmaskN`` conversion (zero-extend, or keep the low bits)."""
