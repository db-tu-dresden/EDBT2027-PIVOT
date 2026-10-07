"""Match selection: the set of compatible candidate matches that, in this order,
translates the most intrinsics, emits the fewest target statements, recomputes
the fewest kept calls, fixes the most operands as constants, and prefers earlier
matches in a stable order.

Two matches conflict if they translate a common intrinsic, or if one removes a
call whose value the other's replacement reads.  Matches may share kept calls:
those stay in the program, and the replacement recomputes them.

This is weighted set packing.  It is solved exactly as a 0/1 ILP (HiGHS), one
connected component of the conflict graph at a time: x_m selects match m, and
each conflict allows at most one of the matches involved."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import highspy
import numpy as np


@dataclass(frozen=True)
class MatchCandidate:
    match_id: str
    intrinsics: frozenset[str]
    # Statements of the cheapest admissible target variant.
    cost: int
    # Constant operands the pattern fixes.
    constants: int
    # Stable order; the last tie-break prefers earlier matches.
    order_key: tuple
    # Calls of the match kept for a use outside it, which the replacement recomputes.
    recomputed: int = 0
    # Translated calls other than the result; the rewrite deletes them.
    removed: frozenset[str] = frozenset()
    # Calls outside the match whose values the replacement reads.
    inputs: frozenset[str] = frozenset()


@dataclass
class SelectionStats:
    candidates: int = 0
    component_matches: list[int] = field(default_factory=list)
    component_intrinsics: list[int] = field(default_factory=list)
    max_matches_per_intrinsic: int = 0
    recomputing_candidates: int = 0
    selected: int = 0
    selected_recomputing: int = 0
    recomputed_intrinsics: int = 0

    def to_json(self) -> dict:
        return {
            "candidate_matches": self.candidates,
            "components": len(self.component_matches),
            "solved_components": sum(1 for size in self.component_matches if size > 1),
            "max_component_matches": max(self.component_matches, default=0),
            "max_component_intrinsics": max(self.component_intrinsics, default=0),
            "max_matches_per_intrinsic": self.max_matches_per_intrinsic,
            "recomputing_candidates": self.recomputing_candidates,
            "selected_matches": self.selected,
            "selected_recomputing_matches": self.selected_recomputing,
            "recomputed_intrinsics": self.recomputed_intrinsics,
            "component_matches_histogram": _histogram(self.component_matches),
            "component_intrinsics_histogram": _histogram(self.component_intrinsics),
        }


def _histogram(sizes: list[int]) -> dict[str, int]:
    return {str(size): count for size, count in sorted(Counter(sizes).items())}


def select_matches(candidates: list[MatchCandidate]) -> tuple[list[str], SelectionStats]:
    """The selected match ids and the shape of the problem."""
    ordered = sorted(candidates, key=lambda c: c.order_key)
    by_intrinsic: dict[str, list[int]] = {}
    for index, candidate in enumerate(ordered):
        for node_id in candidate.intrinsics:
            by_intrinsic.setdefault(node_id, []).append(index)
    conflicts = [group for group in by_intrinsic.values() if len(group) > 1]
    conflicts += _input_conflicts(ordered)

    stats = SelectionStats(
        candidates=len(ordered),
        max_matches_per_intrinsic=max((len(m) for m in by_intrinsic.values()), default=0),
        recomputing_candidates=sum(1 for c in ordered if c.recomputed),
    )
    components = _components(len(ordered), conflicts)
    component_of = {index: c for c, component in enumerate(components) for index in component}
    rows: list[list[list[int]]] = [[] for _ in components]
    for group in conflicts:
        rows[component_of[group[0]]].append(group)

    selected: list[MatchCandidate] = []
    for component, component_rows in zip(components, rows):
        members = [ordered[index] for index in component]
        stats.component_matches.append(len(members))
        stats.component_intrinsics.append(len(frozenset().union(*(m.intrinsics for m in members))))
        if len(members) == 1:
            # A lone match always adds coverage.
            selected.extend(members)
            continue
        position = {index: pos for pos, index in enumerate(component)}
        selected.extend(_solve(members, [[position[i] for i in group] for group in component_rows]))
    stats.selected = len(selected)
    stats.selected_recomputing = sum(1 for m in selected if m.recomputed)
    stats.recomputed_intrinsics = sum(m.recomputed for m in selected)
    return [m.match_id for m in selected], stats


def _input_conflicts(ordered: list[MatchCandidate]) -> list[list[int]]:
    """Pairs where one match removes a call the other's replacement reads.  This
    only happens between a recomputing match and a match that extends past its
    inputs: otherwise the two would translate a common intrinsic."""
    removed_by: dict[str, list[int]] = {}
    for index, candidate in enumerate(ordered):
        for node_id in candidate.removed:
            removed_by.setdefault(node_id, []).append(index)
    pairs = {
        tuple(sorted((reader, remover)))
        for reader, candidate in enumerate(ordered)
        for node_id in candidate.inputs
        for remover in removed_by.get(node_id, ())
    }
    return [list(pair) for pair in sorted(pairs)]


def _components(count: int, groups) -> list[list[int]]:
    """Connected components of the conflict graph, members in index order."""
    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for group in groups:
        root = find(group[0])
        for other in group[1:]:
            parent[find(other)] = root
    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(find(index), []).append(index)
    return sorted(components.values())


def _solve(members: list[MatchCandidate], rows: list[list[int]]) -> list[MatchCandidate]:
    """Solve one component exactly.

    Levels 1-4 form one integer objective whose weights make it strictly
    lexicographic: each level's weight exceeds the range of all levels below it.
    A second solve keeps that optimum and maximizes the stable order weight."""
    n = len(members)
    w_recomputed = sum(m.constants for m in members) + 1
    w_cost = w_recomputed * (sum(m.recomputed for m in members) + 1)
    w_intrinsics = w_cost * (sum(m.cost for m in members) + 1)
    primary = [
        w_intrinsics * len(m.intrinsics) - w_cost * m.cost - w_recomputed * m.recomputed + m.constants
        for m in members
    ]
    if w_intrinsics * (sum(len(m.intrinsics) for m in members) + 1) >= _EXACT_FLOAT_INTS:
        raise OverflowError(f"match selection weights exceed exact float range ({n} matches)")

    highs = _model(n, rows)
    x = _maximize(highs, primary)
    best = _value(primary, x)
    # The objective is integral, so a dual bound below best + 1 proves best optimal.
    if highs.getInfo().mip_dual_bound >= best + 1:
        raise RuntimeError(f"match selection ILP stopped short of optimality ({n} matches)")
    highs.addRow(best, highspy.kHighsInf, n, np.arange(n, dtype=np.int32), np.array(primary, dtype=float))
    x = _maximize(highs, [n - index for index in range(n)])
    if _value(primary, x) != best or any(sum(x[i] for i in row) > 1 for row in rows):
        raise RuntimeError(f"match selection ILP returned an inexact solution ({n} matches)")
    return [member for member, chosen in zip(members, x) if chosen]


_EXACT_FLOAT_INTS = 2 ** 53


def _value(weights: list[int], x: list[bool]) -> int:
    return sum(w for w, chosen in zip(weights, x) if chosen)


def _model(n: int, packing_rows: list[list[int]]) -> highspy.Highs:
    highs = highspy.Highs()
    for option, value in (
        ("output_flag", False),
        ("threads", 1),
        ("random_seed", 0),
        ("mip_rel_gap", 0.0),
        ("mip_abs_gap", 0.0),
    ):
        highs.setOptionValue(option, value)
    cols = np.arange(n, dtype=np.int32)
    highs.addVars(n, np.zeros(n), np.ones(n))
    highs.changeColsIntegrality(n, cols, np.array([highspy.HighsVarType.kInteger] * n))
    highs.changeObjectiveSense(highspy.ObjSense.kMaximize)
    for row in packing_rows:
        highs.addRow(-highspy.kHighsInf, 1.0, len(row), np.array(row, dtype=np.int32), np.ones(len(row)))
    return highs


def _maximize(highs: highspy.Highs, weights: list[int]) -> list[bool]:
    n = len(weights)
    highs.changeColsCost(n, np.arange(n, dtype=np.int32), np.array(weights, dtype=float))
    highs.run()
    status = highs.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"match selection ILP not solved to optimality: {highs.modelStatusToString(status)}")
    return [value > 0.5 for value in highs.getSolution().col_value]
