"""Rooted pattern matching: a pattern is anchored at its single result node and
matched top-down along operand edges only (Hoffmann & O'Donnell's top-down tree
matching, applied to DAGs as instruction selectors do)."""
from dataclasses import dataclass

from pivot.ir.graph import Graph


@dataclass(frozen=True)
class PatternMatch:
    # Pattern intrinsic/constant node id -> world node id.
    mapping: dict[str, str]
    # World node the pattern's result node maps to.
    result_node_id: str


class GraphMatcher:
    """Every world value has exactly one producer per operand slot, so walking
    from the result node to its operands leaves no choice: each root candidate
    yields at most one mapping, in O(|pattern|)."""

    def __init__(self, world_graph: Graph):
        self._world = world_graph
        self._calls_by_name: dict[str, list[str]] = {}
        for node_id, node in world_graph.nodes.items():
            if node.kind == "intrinsic" and not node.is_constant:
                self._calls_by_name.setdefault(node.name, []).append(node_id)

    def find_pattern_matches(self, pattern: Graph) -> list[PatternMatch]:
        root = pattern.result_node_id
        matches = []
        for world_root in self._calls_by_name.get(pattern.nodes[root].name, ()):
            mapping = self._match_at(pattern, root, world_root)
            if mapping is not None:
                matches.append(PatternMatch(mapping=mapping, result_node_id=world_root))
        return matches

    def _match_at(self, pattern: Graph, root: str, world_root: str) -> dict[str, str] | None:
        world = self._world.nodes
        mapping = {root: world_root}
        mapped_calls = {world_root}
        inputs: dict[str, str] = {}
        stack = [root]
        while stack:
            pattern_id = stack.pop()
            operands = pattern.nodes[pattern_id].incoming
            world_operands = world[mapping[pattern_id]].incoming
            if len(operands) != len(world_operands):
                return None
            for p, q in zip(operands, world_operands):
                p_node, q_node = pattern.nodes[p], world[q]
                if p_node.kind == "variable":
                    # One pattern input binds one program value.
                    if inputs.setdefault(p, q) != q:
                        return None
                elif p_node.is_constant:
                    if not (q_node.is_constant and q_node.name == p_node.name):
                        return None
                    mapping[p] = q
                elif p in mapping:
                    if mapping[p] != q:
                        return None
                else:
                    if (q_node.kind != "intrinsic" or q_node.is_constant
                            or q_node.name != p_node.name or q in mapped_calls):
                        return None
                    mapping[p] = q
                    mapped_calls.add(q)
                    stack.append(p)
        return mapping
