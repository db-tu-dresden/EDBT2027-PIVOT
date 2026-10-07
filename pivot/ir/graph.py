from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import yaml

from pivot.frontend.types import CallStatement, VariableDecl, ConstantArg
from pivot.isa.intrinsic_registry import get_intrinsic_registry

if TYPE_CHECKING:
    from pivot.frontend.base import Use

# Constant operands are "intrinsic"-kind leaves named by their match key; their
# ids carry this suffix.
CONSTANT_ID_SUFFIX = "@constant"


@dataclass
class Node:
    id: str
    name: str
    # IR token, read from pattern graphs only (matching is structural).
    type: str
    kind: str
    incoming: list[str] = field(default_factory=list)
    outgoing: list[str] = field(default_factory=list)

    @property
    def is_constant(self) -> bool:
        return self.id.endswith(CONSTANT_ID_SUFFIX)

    def to_yaml_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "kind": self.kind,
            "incoming": list(self.incoming),
            "outgoing": list(self.outgoing),
        }

def _position(pos) -> tuple[int, int]:
    return pos.line, pos.column


class Graph:
    def __init__(
        self,
        matches: list[CallStatement] = None,
        primitive_name: str = '',
        isa: str = '',
        blocked_arches: list[str] = None,
        nto1: bool = False,
        uses: list["Use"] = None,
    ) -> None:
        # The pattern fields below are empty on the program graph.
        self.primitive_name: str = primitive_name
        self.isa: str = isa
        # Target architectures on which the pattern is not matched.
        self.blocked_arches: list[str] = list(blocked_arches or [])
        # The pattern fuses several source intrinsics into one target op.
        self.nto1: bool = nto1
        self.nodes: dict[str, Node] = {}
        # The one intrinsic node no other intrinsic of the pattern consumes.
        self.result_node_id: Optional[str] = None

        # Program graph: call node id (`CallStatement.id`) -> the call.
        self.calls: dict[str, CallStatement] = {}
        self._producers: dict[str, str] = {}  # declaration id -> current producer node id

        if matches:
            # A use the program graph does not link as a call operand is a read.
            reads = [use for use in uses or () if use.call is None]
            self._build(self._sort_matches(matches), sorted(reads, key=lambda r: _position(r.extent.start)))

    def __repr__(self) -> str:
        node_ids = list(self.nodes.keys())
        return (
            f'Graph(primitive_name={self.primitive_name}, isa={self.isa}, '
            f'nodes={node_ids})'
        )

    @staticmethod
    def _sort_matches(matches: list[CallStatement]) -> list[CallStatement]:
        # By line only, stably: the frontends list calls in evaluation order, and
        # a column tiebreak would put a nested call's outer consumer (lower
        # column) before its inner producer.
        return sorted(matches, key=lambda m: m.extent.start.line)

    def _build(self, matches: list[CallStatement], reads: list["Use"]) -> None:
        """Build the data-flow graph in statement order.

        A read outside call operands becomes a "read" node consuming the value
        current at that point; a read inside a call's statement sees the value
        from before the statement."""
        next_read = 0

        for match in matches:
            statement_end = _position((match.extent_with_var or match.extent).end)
            while next_read < len(reads) and _position(reads[next_read].extent.start) < statement_end:
                self._add_read(reads[next_read])
                next_read += 1

            intrinsic_id = match.id
            self.calls[intrinsic_id] = match
            self._add_node(intrinsic_id, match.name, "intrinsic")

            for var in match.args:
                if isinstance(var, ConstantArg):
                    constant_id = f"{var.id}{CONSTANT_ID_SUFFIX}"
                    if constant_id not in self.nodes:
                        self._add_node(constant_id, var.key, "intrinsic")
                    self._add_edge(constant_id, intrinsic_id)
                    continue

                if not isinstance(var, VariableDecl):
                    raise ValueError(
                        f"Unsupported variable argument type {type(var).__name__} "
                        f"for intrinsic {match.id}"
                    )

                # A non-strict producer (e.g. one branch of a ternary) does not
                # define the variable, so the operand reads the stable input node.
                producer_id = self._producers.get(var.id)
                producer = self.calls.get(producer_id) if producer_id else None
                if producer is not None and not producer.is_strict_assignment:
                    producer_id = None
                if producer_id is None:
                    input_var_id = f"{var.id}@input"
                    if input_var_id not in self.nodes:
                        self._add_node(input_var_id, var.name, "variable")
                    self._producers[var.id] = input_var_id
                    producer_id = input_var_id
                self._add_edge(producer_id, intrinsic_id)

            if match.returns_decl:
                if not isinstance(match.returns_decl, VariableDecl):
                    raise ValueError(
                        f"Unsupported return type for intrinsic {match.id}: {type(match.returns_decl).__name__}"
                    )
                self._producers[match.returns_decl.id] = intrinsic_id

        for read in reads[next_read:]:
            self._add_read(read)
        self._attach_ir_types()

    def _add_read(self, read: "Use") -> None:
        producer_id = self._producers.get(read.decl.id)
        producer = self.calls.get(producer_id) if producer_id else None
        if producer is None or not producer.is_strict_assignment:
            return
        start = read.extent.start
        read_id = f"{read.decl.name}@{start.line}:{start.column}@read"
        self._add_node(read_id, read.decl.name, "read")
        self._add_edge(producer_id, read_id)

    def _attach_ir_types(self) -> None:
        """Type every node from the intrinsic registry's `ir:` signatures: an
        intrinsic node by its `ret` token, a variable leaf by the argument slot of
        its first consumer.  Only pattern graphs read the types."""
        reg = get_intrinsic_registry()
        for node in self.nodes.values():
            if node.kind == "intrinsic":
                # Constant leaves have kind "intrinsic" but are not real intrinsics.
                if not reg.is_known_intrinsic(node.name):
                    continue
                token = reg.ir_token_at(node.name, "ret")
                node.type = token if token is not None else "void"
            elif node.kind == "variable" and node.outgoing:
                consumer = self.nodes.get(node.outgoing[0])
                if consumer is None or consumer.kind != "intrinsic":
                    continue
                try:
                    arg_pos = consumer.incoming.index(node.id)
                except ValueError:
                    continue
                token = reg.ir_token_at(consumer.name, arg_pos)
                if token is not None:
                    node.type = token

    def _add_node(self, node_id: str, name: str, kind: str) -> None:
        # Typed afterwards, in one pass (_attach_ir_types).
        self.nodes[node_id] = Node(node_id, name, "", kind)

    def _add_edge(self, source_id: str, target_id: str) -> None:
        self.nodes[target_id].incoming.append(source_id)
        self.nodes[source_id].outgoing.append(target_id)

    @classmethod
    def load_all_from_yaml(cls, filename: str) -> list['Graph']:
        try:
            loader = yaml.CSafeLoader
        except AttributeError:  # pragma: no cover - libyaml not built
            loader = yaml.SafeLoader
        with open(filename, "r") as f:
            graph_docs = [doc for doc in yaml.load_all(f, Loader=loader) if doc]

        graphs: list[Graph] = []
        for graph_data in graph_docs:
            graph = cls()
            graph.primitive_name = graph_data.get("primitive_name", "unknown")
            graph.isa = graph_data.get("isa", "unknown")
            graph.blocked_arches = list(graph_data.get("blocked_arches", []))
            graph.nto1 = bool(graph_data.get("nto1", False))

            for node_id, node_data in graph_data["nodes"].items():
                graph.nodes[node_id] = Node(
                    id=node_data["id"],
                    name=node_data["name"],
                    type=node_data["type"],
                    kind=node_data["kind"],
                    incoming=list(node_data.get("incoming", [])),
                    outgoing=list(node_data.get("outgoing", [])),
                )
            graph.result_node_id = graph._single_result_node(filename)
            graphs.append(graph)

        return graphs

    def _single_result_node(self, filename: str) -> str:
        """Rooted matching anchors a pattern at its result node, so a pattern must
        have exactly one intrinsic node whose value no other intrinsic consumes."""
        sinks = [
            node_id for node_id, node in self.nodes.items()
            if node.kind == "intrinsic" and not node.is_constant
            and not any(
                self.nodes[out].kind == "intrinsic" for out in node.outgoing if out in self.nodes
            )
        ]
        if len(sinks) != 1:
            raise ValueError(
                f"Pattern graph {self.primitive_name} ({self.isa}) in {filename} must have "
                f"exactly one result node, found {len(sinks)}: {sorted(sinks)}"
            )
        return sinks[0]

    def calls_with_pointer_operand(self) -> frozenset[str]:
        """Pattern graphs only: the calls that take a pointer, i.e. the ones that
        touch memory or state (load, store, gather, rdrand) and must not run twice."""
        return frozenset(
            node_id for node_id, node in self.nodes.items()
            if node.kind == "intrinsic" and not node.is_constant
            and any(self.nodes[src].type == "ptr" for src in node.incoming if src in self.nodes)
        )
