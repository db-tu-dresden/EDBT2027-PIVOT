"""Matching: every rooted match of a pattern whose primitive the target defines,
with its operands bound to the source and the target variants its source
signature admits.

A match is kept if all its calls lie in one control-flow region and every call
it keeps is pure.  Nothing is typed here."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pivot.driver.translation_logger import get_translation_logger
from pivot.frontend.types import CallStatement, ConstantArg, VariableDecl
from pivot.ir.graph import Graph
from pivot.ir.graph_matcher import GraphMatcher
from pivot.ir.types import IRKind, IRType, concretize_scalable_token, parse_ir_token, resolve_scalable_ir
from pivot.isa.primitive_registry import get_primitive_registry
from pivot.passes.translation.admissibility import admissible_variants
from pivot.passes.translation.model import Match, Operand, Variant

if TYPE_CHECKING:
    from pivot.isa.primitive_registry import Definition, Primitive


def find_matches(program: Graph, patterns: list[Graph], target: str, sve_bits: int) -> list[Match]:
    """The matches of ``patterns`` in ``program`` that stay in one region and
    keep only pure calls, numbered in pattern and anchor order."""
    matcher = GraphMatcher(program)
    registry = get_primitive_registry()
    matches: list[Match] = []
    for pattern in patterns:
        if not (primitive := registry.primitive_of(pattern.primitive_name)):
            get_translation_logger().log(f"Definition not found for {pattern.primitive_name}, {pattern.isa}")
            continue
        if not (definitions := registry.definitions_for_primitive(target, pattern.primitive_name)):
            continue
        variants = _variants(definitions, sve_bits)
        pointer_calls = pattern.calls_with_pointer_operand()
        for found in matcher.find_pattern_matches(pattern):
            calls = frozenset(n for n in found.mapping.values() if n in program.calls)
            # Calls in different control-flow regions may run a different number
            # of times, so they cannot fuse into one operation.
            if len({program.calls[n].region for n in calls}) > 1:
                continue
            kept = _kept_calls(program, calls, found.result_node_id)
            # A kept call runs twice, itself and inside the replacement, so it
            # must not touch memory or state.
            if kept & {found.mapping[p] for p in pointer_calls}:
                continue
            inputs, output = _bind_operands(primitive, pattern, found.mapping, program.calls, sve_bits)
            operands = inputs if output is None else (*inputs, output)
            matches.append(Match(
                index=len(matches),
                primitive=primitive,
                pattern=pattern,
                mapping=found.mapping,
                calls=calls,
                result=found.result_node_id,
                kept=kept,
                inputs=inputs,
                output=output,
                variants=tuple(admissible_variants(operands, variants)),
            ))
    return matches


def _kept_calls(program: Graph, calls: frozenset[str], result: str) -> frozenset[str]:
    """The calls of a match that stay in the program: every intermediate whose
    value is used outside the match (by another call, a read outside call
    operands, or another region), and the calls of the match feeding one."""
    pending = [
        node_id for node_id in calls
        if node_id != result
        and any(consumer not in calls for consumer in program.nodes[node_id].outgoing)
    ]
    kept = set(pending)
    while pending:
        for producer in program.nodes[pending.pop()].incoming:
            if producer in calls and producer not in kept:
                kept.add(producer)
                pending.append(producer)
    return frozenset(kept)


def _variants(definitions: list[Definition], sve_bits: int) -> list[Variant]:
    """One variant per distinct rewrite, in corpus order.  Definitions equal after
    expansion (same signature and body) would make the typing pass see a choice
    where there is none."""
    seen: set[tuple] = set()
    variants: list[Variant] = []
    for definition in definitions:
        key = (tuple(sorted(definition.signature.items())), tuple(definition.direct))
        if key in seen:
            continue
        seen.add(key)
        variants.append(Variant(
            definition=definition,
            typing_signature={
                name: concretize_scalable_token(token, sve_bits)
                for name, token in definition.signature.items()
            },
            cost=len(definition.direct),
        ))
    return variants


def _bind_operands(
    primitive: Primitive,
    pattern: Graph,
    mapping: dict[str, str],
    program_calls: dict[str, CallStatement],
    sve_bits: int,
) -> tuple[tuple[Operand, ...], Operand | None]:
    """Each primitive input bound to the argument of its first consumer in the
    pattern, and the output to the call the result node matched."""
    calls = {
        pattern_id: program_calls[mapping[pattern_id]]
        for pattern_id, node in pattern.nodes.items()
        if node.kind == "intrinsic" and mapping.get(pattern_id) in program_calls
    }
    inputs = tuple(_bind_input(name, pattern, calls, sve_bits) for name in primitive.input)
    if not primitive.output:
        return inputs, None
    result = pattern.result_node_id
    if (call := calls.get(result)) is None:
        return inputs, Operand(primitive.output)
    return inputs, Operand(
        primitive.output, _source_ir(pattern.nodes[result].type, sve_bits), decl=call.returns_decl)


def _bind_input(name: str, pattern: Graph, calls: dict[str, CallStatement], sve_bits: int) -> Operand:
    for pattern_id, call in calls.items():
        for position, operand_id in enumerate(pattern.nodes[pattern_id].incoming):
            node = pattern.nodes.get(operand_id)
            if node is None or node.kind != "variable" or node.name != name:
                continue
            arg = call.args[position] if position < len(call.args) else None
            return Operand(
                name,
                _source_ir(node.type, sve_bits),
                decl=arg if isinstance(arg, VariableDecl) else None,
                text=arg.name if arg is not None else None,
                callee=arg.callee if isinstance(arg, ConstantArg) else None,
            )
    return Operand(name)


def _source_ir(token: str | None, sve_bits: int) -> IRType | None:
    """The pattern token's type at the assumed width; a pointer stays opaque."""
    ir = parse_ir_token(token) if token is not None else None
    if ir is None or ir.kind is IRKind.POINTER:
        return None
    return resolve_scalable_ir(ir, sve_bits)
