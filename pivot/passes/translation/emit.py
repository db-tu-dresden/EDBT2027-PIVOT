"""Emit: the source changes a typed translation implies, planned in this order:
each selected match's rewrite, the retyped declarations and the conversions
between them; once any call is translated, also the imports, the SIMD-pointer
casts, the remaining vector type names and the integer reads of masks.  A
change inside one planned before it is dropped."""
from __future__ import annotations

import os
from dataclasses import dataclass

from pivot.backend.backend_emitter import BackendEmitter, ConversionSpelling
from pivot.driver.translation_logger import get_translation_logger
from pivot.frontend.types import CallStatement, VariableDecl
from pivot.ir.conversions import conversion
from pivot.ir.edits import Change, ChangePlan, ChangeType
from pivot.ir.source_span import SourcePos
from pivot.ir.types import IRKind, parse_ir_token, Domain
from pivot.ir.value_flow import BridgeSite, NodeType
from pivot.isa.primitive_registry import Definition, Primitive
from pivot.isa.source_vector_types import canonical_vector_token
from pivot.passes.translation.model import Match, TranslationContext, Variant
from pivot.passes.translation.type_resolution import input_token, value_token


# Marks a SIMD value whose type did not resolve.  Deliberately not valid code:
# the build must fail there, and the nesting pass keeps a temp marked this way.
UNKNOWN_TYPE_MARKER = "/* Fixme: unknown type */"


@dataclass
class Translation:
    changes: list[Change]
    # Name of every call the rewrites translate.
    translated_names: list[str]
    # The target definitions the rewrites call; the companion is generated from them.
    used_definitions: list[Definition]


def emit(ctx: TranslationContext, backend: BackendEmitter, typed: list[tuple[Match, Variant]]) -> Translation:
    """All changes of the file, sorted by position, stably, so changes at one
    position apply in planning order."""
    plan = ChangePlan()
    translated_names = _rewrite_matches(ctx, backend, typed, plan)
    used_definitions = [variant.definition for _, variant in typed]
    _declaration_types(ctx, backend, plan)
    used_definitions += _conversions(ctx, backend, ctx.value_flow.bridge_sites, plan)
    if typed:
        _imports(ctx, backend, plan)
        _pointer_casts(ctx, plan)
        _type_sites(ctx, backend, plan)
        # A mask read as its integer converts only in translated code.
        used_definitions += _conversions(ctx, backend, ctx.value_flow.integer_reads, plan)
    changes = sorted(plan.changes, key=lambda change: change.sort_key)
    return Translation(changes, translated_names, used_definitions)


def _rewrite_matches(
    ctx: TranslationContext, backend: BackendEmitter, typed: list[tuple[Match, Variant]], plan: ChangePlan
) -> list[str]:
    """Per match: the removal of every call it removes, then its replacement at
    the result call.  Returns the names of the calls the matches translate."""
    removed = frozenset().union(*(match.removed for match, _ in typed))
    rewritten = frozenset(match.result for match, _ in typed)
    names: list[str] = []
    for match, variant in typed:
        for node_id in sorted(match.removed):
            call = ctx.calls[node_id]
            extent = call.extent_with_var or call.extent
            # A variable the source reassigns later keeps its
            # declaration once its initializing call is gone.
            if call.initializer_extent is not None and _declared_var_still_used(ctx, call, removed, rewritten):
                extent = call.initializer_extent
            plan.add(extent, Change(change_type=ChangeType.REMOVE, location=extent.start, extent=extent))
        replacement = _replacement(ctx, backend, match, variant)
        plan.add(replacement.extent, replacement)
        names += [ctx.calls[node_id].name for node_id in sorted(match.translated)]
    # Selected matches translate disjoint calls, so no name is counted twice.
    get_translation_logger().log(
        f"[Emit] Selected matches: matches={len(typed)}, translated_intrinsics={len(names)}"
    )
    return names


def _declared_var_still_used(
    ctx: TranslationContext, call: CallStatement, removed: frozenset[str], rewritten: frozenset[str]
) -> bool:
    """Whether a call that survives the rewrite reads or assigns the variable a
    removed call declares.  A rewritten result keeps only its assignment target;
    its arguments are re-emitted from the primitive's inputs."""
    decl = call.returns_decl
    if decl is None:
        return False
    for node_id, other in ctx.calls.items():
        if node_id in removed:
            continue
        if other.returns_decl is not None and other.returns_decl.id == decl.id:
            return True
        if node_id in rewritten:
            continue
        if any(isinstance(arg, VariableDecl) and arg.id == decl.id for arg in other.args):
            return True
    return False


def _replacement(ctx: TranslationContext, backend: BackendEmitter, match: Match, variant: Variant) -> Change:
    """The target call replacing the match's result call.  Each argument is
    converted to its slot's type (the reinterpret x86 gets free on untyped
    `__m*i` registers, the widen/truncate C gets free on `__mmask{N}`), and the
    result to its variable's type when this op is not the value's first producer."""
    definition = variant.definition
    anchor = ctx.calls[match.result]
    text = None
    if match.primitive.name == "LOAD" and backend.target.supports_const_vector_literal:
        text = _const_load_literal(backend, match.primitive, definition, anchor)
    if text is None:
        args = [
            _converted(
                backend,
                operand.text if operand.text is not None else operand.param,
                input_token(ctx, operand),
                (definition.signature.get(operand.param) or "").strip(),
            )
            for operand in match.inputs
        ]
        text = backend.emit_call(definition, args)
    if match.output is not None:
        text = _converted(
            backend,
            text,
            (definition.signature.get(match.output.param) or "").strip(),
            value_token(ctx, match.output.decl),
        )
    return Change(change_type=ChangeType.REPLACE, extent=anchor.extent, text=text)


def _spelled_conversion(backend: BackendEmitter, value: str | None, slot: str | None) -> ConversionSpelling | None:
    """The conversion a value of token ``value`` needs in a ``slot`` slot, spelled."""
    return backend.spell_conversion(conversion(value, slot), value, slot)


def _converted(backend: BackendEmitter, expr: str, value: str | None, slot: str | None) -> str:
    if (spelled := _spelled_conversion(backend, value, slot)) is None:
        return expr
    return f"{spelled.prefix}{expr}{spelled.suffix}"


def _const_load_literal(
    backend: BackendEmitter, primitive: Primitive, definition: Definition, call: CallStatement
) -> str | None:
    """An inline vector literal ``(v16i){0, 2, ...}`` for a load of a ``const``
    integer array whose element count and width match the target vector, or
    None.  The memcpy-based load would hide the constant, so a constant
    permute mask would scalarize instead of folding to ``vpermt2d``."""
    if list(primitive.input) != ["ptr"] or not primitive.output:
        return None
    out_token = (definition.signature.get(primitive.output) or "").strip()
    ir = parse_ir_token(out_token)
    if ir is None or ir.kind is not IRKind.VECTOR:
        return None
    if ir.lanes is None or ir.elem_bits is None:
        return None
    if ir.domain not in (Domain.INT, Domain.UINT):
        return None
    if not call.args:
        return None
    arg = call.args[0]
    if not isinstance(arg, VariableDecl) or arg.const_init_text is None:
        return None
    if arg.const_elem_count != ir.lanes or arg.const_elem_bytes != ir.elem_bits.value // 8:
        return None
    return backend.vector_literal(out_token, arg.const_init_text)


def _declaration_types(ctx: TranslationContext, backend: BackendEmitter, plan: ChangePlan) -> None:
    """Every typed declaration retyped; a slot-less value whose type did not
    resolve gets the unknown-type marker before its name, so the nesting pass
    keeps it and the build fails at a labelled line."""
    for node in ctx.value_flow.nodes:
        if node.node_type is NodeType.INTRINSIC_RETURN:
            continue
        if node.extent is not None:
            # Spelled with an alias of its own type: the alias is retyped instead.
            if node.spelled_by is not None and node.token is not None and node.token == node.spelled_by.token:
                continue
            text = backend.declared_type(node.token, node.pointer) if node.token is not None else UNKNOWN_TYPE_MARKER
            plan.add(node.extent, Change(change_type=ChangeType.REPLACE, extent=node.extent, text=text))
        elif node.token is None and node.anchor_extent is not None:
            plan.add(node.anchor_extent, Change(
                change_type=ChangeType.INSERT_BEFORE,
                location=node.anchor_extent.start,
                text=f"{UNKNOWN_TYPE_MARKER} ",
            ))


def _conversions(
    ctx: TranslationContext, backend: BackendEmitter, sites: list[BridgeSite], plan: ChangePlan
) -> list[Definition]:
    """A conversion wherever a value flows into a slot of another type, at most
    one per expression.  Returns the definitions they call."""
    used: list[Definition] = []
    seen: set[tuple[int, int]] = set()
    # Offsets are relative to the file being translated.
    path = os.path.realpath(ctx.unit.path)
    for site in sites:
        ext = site.use_extent
        if not ext.file or os.path.realpath(ext.file) != path or (ext.start.byte, ext.end.byte) in seen:
            continue
        if (spelled := _spelled_conversion(backend, site.value.token, site.slot.token)) is None:
            continue
        # Two point insertions, so nested conversions (an argument inside a return) compose.
        if plan.add(
            ext,
            Change(change_type=ChangeType.INSERT_BEFORE, location=ext.start, text=spelled.prefix),
            Change(change_type=ChangeType.INSERT_BEFORE, location=ext.end, text=spelled.suffix),
        ):
            seen.add((ext.start.byte, ext.end.byte))
            if spelled.definition is not None and spelled.definition not in used:
                used.append(spelled.definition)
    return used


def _imports(ctx: TranslationContext, backend: BackendEmitter, plan: ChangePlan) -> None:
    """Drop the source ISA's import directives (C ``#include``, Rust ``use``)
    and bind the companion in their place."""
    simd = list(ctx.unit.simd_headers.values())
    for span in sorted(simd, key=lambda r: (r.start.line, r.start.column), reverse=True):
        plan.add(span, Change(change_type=ChangeType.REMOVE, location=span.start, extent=span))

    if simd:
        insertion = min(simd, key=lambda r: (r.start.line, r.start.column)).start
    elif std := list(ctx.unit.std_headers.values()):
        insertion = min(std, key=lambda r: (r.start.line, r.start.column)).start
    else:
        insertion = SourcePos(line=1, column=1, byte=0)
    if binding := backend.companion_binding():
        plan.add(None, Change(
            change_type=ChangeType.INSERT_BEFORE,
            location=insertion,
            text="".join(line + "\n" for line in binding),
        ))


def _pointer_casts(ctx: TranslationContext, plan: ChangePlan) -> None:
    """A standalone SIMD-pointer cast ``(__m128i*)p`` would reference the
    stripped register type; each becomes its inner expression."""
    for cast in ctx.unit.redundant_casts:
        extent = cast.cast_extent
        plan.add(extent, Change(change_type=ChangeType.REPLACE, extent=extent, text=cast.inner_text))


def _type_sites(ctx: TranslationContext, backend: BackendEmitter, plan: ChangePlan) -> None:
    """Rewrite every remaining source vector type name (a kept pointer cast,
    `sizeof`, a template argument, a field): to the type of the pointer the
    spelling types, else to a register of the same width.  A name that has
    neither stays loud: the unknown-type marker plus a warning."""
    for site in ctx.unit.type_sites:
        token = ctx.value_flow.token_for(site.owner) or canonical_vector_token(site.type_name, ctx.sve_bits)
        # The name alone: a pointer's `*` lies outside the site.
        text = backend.declared_type(token) if token is not None else UNKNOWN_TYPE_MARKER
        change = Change(change_type=ChangeType.REPLACE, extent=site.extent, text=text)
        if plan.add(site.extent, change) and token is None:
            print(f"[type-sites] no target type for '{site.type_name}' at "
                  f"{site.extent.file}:{site.extent.start.line}:{site.extent.start.column}")
