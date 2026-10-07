"""The translation pass on one parsed file: match -> select -> type -> emit."""
from __future__ import annotations

import json
import time
from pathlib import Path

from pivot.backend.backend_emitter import BackendEmitter
from pivot.driver.run_options import RunOptions
from pivot.driver.translation_logger import get_translation_logger
from pivot.frontend.base import ParsedUnit
from pivot.ir.graph import Graph
from pivot.ir.graph_loader import GraphLoader
from pivot.ir.types import IRKind
from pivot.ir.value_flow import build_value_flow
from pivot.passes.translation.emit import Translation, emit
from pivot.passes.translation.match_selection import MatchCandidate, select_matches
from pivot.passes.translation.matching import find_matches
from pivot.passes.translation.model import Match, TranslationContext
from pivot.passes.translation.type_resolution import resolve_types


class NoTranslationSolutionError(ValueError):
    """Raised when no match can translate the file."""


def translate(
    unit: ParsedUnit,
    options: RunOptions,
    *,
    backend: BackendEmitter,
    graph_loader: GraphLoader,
    artifact_dir: str,
) -> Translation:
    """The source changes that translate ``unit`` to the options' target."""
    ctx = TranslationContext(
        unit=unit,
        target=options.target_isa,
        sve_bits=options.sve_assumed_bits,
        program=Graph(matches=unit.call_statements, uses=unit.uses),
        value_flow=build_value_flow(unit),
    )
    graph_loader.load_all_graphs()
    patterns = graph_loader.get_by_source(options.source_isa, arch=options.arch, nto1=options.nto1)
    start = time.perf_counter()
    matches = find_matches(ctx.program, patterns, ctx.target, ctx.sve_bits)
    matching_seconds = time.perf_counter() - start
    if not matches:
        raise NoTranslationSolutionError(
            "No valid translation solution could be constructed for the current source graph."
        )
    # The backend's width gate, over every match, so a mixed-width kernel
    # selected at one width is still gated.
    widths = _source_register_widths(matches)
    if not backend.accepts_source_register_widths(widths):
        raise NoTranslationSolutionError(
            f"Target '{options.target_isa}' requires all source vectors to share one "
            f"register width, but the kernel mixes register widths (bits): "
            f"{sorted(widths)}."
        )
    selected = _select(ctx, matches, matching_seconds, artifact_dir)
    typed = resolve_types(ctx, selected)
    return emit(ctx, backend, typed)


def _source_register_widths(matches: list[Match]) -> set[int]:
    """Register bit-widths (lanes * elem_bits) of the vector/blob source operands."""
    widths: set[int] = set()
    for match in matches:
        for operand in match.operands:
            if operand.ir is None or operand.ir.kind not in (IRKind.VECTOR, IRKind.BLOB):
                continue
            if (bits := operand.ir.concrete_total_bits()) is not None:
                widths.add(bits)
    return widths


def _select(ctx: TranslationContext, matches: list[Match], matching_seconds: float, artifact_dir: str) -> list[Match]:
    """The selected candidates, a candidate being a match with an admissible variant."""
    candidates = [_candidate(ctx, match) for match in matches if match.variants]
    start = time.perf_counter()
    selected_ids, stats = select_matches(candidates)
    selection_seconds = time.perf_counter() - start
    _write_selection_stats(artifact_dir, {
        **stats.to_json(),
        "matching_seconds": round(matching_seconds, 4),
        "selection_seconds": round(selection_seconds, 4),
    })
    by_id = {match.id: match for match in matches}
    return [by_id[match_id] for match_id in selected_ids]


def _candidate(ctx: TranslationContext, match: Match) -> MatchCandidate:
    positions = sorted(
        (ctx.calls[node_id].extent.start.line, ctx.calls[node_id].extent.start.column)
        for node_id in match.calls
    )
    return MatchCandidate(
        match_id=match.id,
        intrinsics=match.translated,
        cost=min(variant.cost for variant in match.variants),
        constants=sum(1 for node in match.pattern.nodes.values() if node.is_constant),
        order_key=(match.primitive.name, tuple(positions), match.index),
        recomputed=len(match.kept),
        removed=match.removed,
        inputs=frozenset(
            producer
            for node_id in match.calls
            for producer in ctx.program.nodes[node_id].incoming
            if producer in ctx.calls and producer not in match.calls
        ),
    )


def _write_selection_stats(artifact_dir: str, stats: dict) -> None:
    """Record the matching/selection problem size of this file in its debug
    artifacts, for the evaluation."""
    get_translation_logger().log(f"[MatchSelection] {json.dumps(stats, sort_keys=True)}")
    path = Path(artifact_dir) / "match_selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
