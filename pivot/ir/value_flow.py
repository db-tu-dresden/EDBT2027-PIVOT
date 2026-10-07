"""The value-flow graph of one parsed unit: a node per place a SIMD type is
written (a declaration, a user function's return type, an alias, an intrinsic
result) and an edge per value flowing between them (a copy, a return, a call
argument, an element load or store).  It stores the types; the typing pass
(`passes/translation/type_resolution`) decides them."""
from __future__ import annotations

import math
import re
from enum import Enum
from typing import Optional

from pivot.frontend.base import ParsedUnit, Use, UseContext
from pivot.frontend.types import ElementArg, VariableDecl
from pivot.ir.conversions import INTEGER_SLOT
from pivot.ir.source_span import SourceSpan
from pivot.isa.intrinsic_registry import IntrinsicRegistry, get_intrinsic_registry


class NodeType(Enum):
    VARIABLE = "variable"
    FUNCTION = "function"
    # The result of one intrinsic call; it flows to the variable the call writes.
    INTRINSIC_RETURN = "intrinsic_return"
    # A type alias of a SIMD vector type, one type for every declaration spelled
    # with it; rewritten at its own definition.
    ALIAS = "alias"
    # The host integer a C mask is read as.
    INTEGER = "integer"


class ValueNode:
    """A place a SIMD type is written, typed by one IR token."""

    def __init__(self, node_type: NodeType, name: str, extent: SourceSpan | None,
                 anchor_extent: SourceSpan | None = None):
        self.node_type: NodeType = node_type
        self.name: str = name
        # The declared type, rewritten with the resolved one; None when there is
        # none (a hoisted temp is a bare assignment).
        self.extent: SourceSpan | None = extent
        # The value's name, where the diagnostic for an unresolved slot-less value goes.
        self.anchor_extent: SourceSpan | None = anchor_extent
        self.token: str | None = None
        # Declared as a pointer to the value (`T *p`): its type is spelled with the `*`.
        self.pointer: bool = False
        # Forward edges: the nodes this node's type flows to.
        self.dependents: list[ValueNode] = []
        # Deferred edges: the nodes this node types once the forward walk is done.
        self.deferred: list[ValueNode] = []
        # A pointer to SIMD vectors, typed by its element.
        self.holds_elements: bool = False
        # The source vector type a pointer or alias is declared with.
        self.source_type: str | None = None
        # The alias whose spelling this declaration keeps.
        self.spelled_by: ValueNode | None = None


class BridgeSite:
    """A value (an argument, an RHS, a returned value) flowing into a slot (a
    parameter, an assigned variable); the expression at ``use_extent`` converts
    when their types differ."""

    def __init__(self, use_extent: SourceSpan, value: ValueNode, slot: ValueNode):
        self.use_extent: SourceSpan = use_extent
        self.value: ValueNode = value
        self.slot: ValueNode = slot


class ValueFlowGraph:
    """The nodes in source order, their forward and deferred edges, and the
    sites where a value meets a slot that may differ in type."""

    def __init__(self, nodes: list[ValueNode], declarations: dict[str, ValueNode],
                 intrinsic_returns: dict[str, ValueNode]):
        self.nodes: list[ValueNode] = sorted(nodes, key=_source_order)
        # The node of each typed declaration, by `VariableDecl.id`.
        self.declarations = declarations
        # The result node of each intrinsic call, by call id.
        self.intrinsic_returns = intrinsic_returns
        # The first user function of each name returning a SIMD type.
        self._functions: dict[str, ValueNode] = {}
        for node in self.nodes:
            if node.node_type is NodeType.FUNCTION:
                self._functions.setdefault(node.name, node)
        self.bridge_sites: list[BridgeSite] = []
        # The reads of a mask C evaluates as its integer, into the INTEGER slot.
        self.integer_reads: list[BridgeSite] = []

    def add_edge(self, producer: ValueNode | None, dependent: ValueNode | None) -> None:
        """The producer's type flows to the dependent."""
        if producer is not None and dependent is not None and producer is not dependent:
            if dependent not in producer.dependents:
                producer.dependents.append(dependent)

    def link(self, a: ValueNode | None, b: ValueNode | None) -> None:
        """Two nodes of one type: whichever is typed first types the other."""
        self.add_edge(a, b)
        self.add_edge(b, a)

    def add_deferred(self, source: ValueNode | None, target: ValueNode | None) -> None:
        """Once the forward walk is done, a typed ``source`` types a still untyped
        ``target`` (a parameter its call-site argument, an element its producer),
        so forward evidence wins."""
        if source is not None and target is not None and source is not target:
            if target not in source.deferred:
                source.deferred.append(target)

    def add_bridge_site(self, use_extent: SourceSpan | None, value: ValueNode | None, slot: ValueNode | None) -> None:
        if use_extent is not None and value is not None and slot is not None:
            self.bridge_sites.append(BridgeSite(use_extent, value, slot))

    def node_for(self, decl) -> Optional[ValueNode]:
        """The node of a declaration, None when the graph does not type it."""
        return self.declarations.get(decl.id) if decl is not None else None

    def token_for(self, decl) -> str | None:
        node = self.node_for(decl)
        return node.token if node is not None else None

    def function_token(self, name: str) -> str | None:
        """The token a user function's return type resolved to, None while untyped."""
        node = self._functions.get(name)
        return node.token if node is not None else None


def _source_order(node: ValueNode) -> tuple:
    return (node.extent.start.line, node.extent.start.column) if node.extent else (math.inf, math.inf)


def _strip_qualifiers(dtype: str) -> str:
    """A source type spelling without its cv-qualifiers."""
    parts = [p for p in (dtype or "").replace("\t", " ").split()
             if p not in {"const", "volatile", "restrict"}]
    return " ".join(parts).strip()


def _strip_pointer_suffix(dtype: str) -> str:
    token = (dtype or "").strip()
    while token.endswith("*"):
        token = token[:-1].rstrip()
    return token


def _norm_dtype(dtype: str) -> str:
    """The bare element type of a spelling, for overload matching: ``const T &``,
    ``T *``, ``T[4]`` and ``T`` compare equal."""
    token = re.sub(r"\[[^\]]*\]", " ", _strip_qualifiers(dtype))
    for deco in ("&&", "&", "*"):
        token = token.replace(deco, " ")
    return " ".join(token.split())


def _select_overload(call, candidates):
    """The overload among same-named ``candidates`` a call resolves to: same arity,
    no argument type contradicting its parameter, most argument types matching.
    None when that is ambiguous, so no edges are guessed."""
    arity = len(call.args)
    viable = [c for c in candidates if len(c.params) == arity]
    if len(viable) <= 1:
        return viable[0] if viable else None

    arg_types = [_norm_dtype(getattr(a, "dtype", "")) for a in call.args]

    def score(cand) -> int | None:
        matches = 0
        for at, param in zip(arg_types, cand.params):
            if not at or at == "auto":
                continue  # unknown arg type: neither confirms nor rules out
            if at == _norm_dtype(param.dtype):
                matches += 1
            else:
                return None
        return matches

    scored = [(s, c) for c in viable if (s := score(c)) is not None]
    if not scored:
        return None
    scored.sort(key=lambda sc: sc[0], reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def build_value_flow(unit: ParsedUnit) -> ValueFlowGraph:
    intr_reg = get_intrinsic_registry()

    def is_dtype(dtype: str) -> bool:
        return intr_reg.is_dtype(dtype=_strip_pointer_suffix(dtype))

    call_by_id = {c.id: c for c in unit.call_statements}
    func_simd = {f.name: is_dtype(f.return_type) for f in unit.functions}

    def _has_simd_producer(decl) -> bool:
        """Written by a SIMD-producing call."""
        producer = call_by_id.get(getattr(decl, "producer_call_id", None))
        if producer is None:
            return False
        return producer.is_intrinsic or func_simd.get(producer.name, False)

    element_uses = [use for use in unit.uses
                    if use.context in (UseContext.ELEMENT_LOAD, UseContext.ELEMENT_STORE)]
    element_values = {id(use.other) for use in element_uses}

    def _keeps_declared_type(decl) -> bool:
        """Declared with a non-SIMD type of its own (`uint64_t m`), not `auto`
        or a hoisted temp's bare assignment."""
        return (decl.type_extent is not None and _strip_qualifiers(decl.dtype) != "auto"
                and not is_dtype(decl.dtype) and not decl.holds_elements)

    def is_graph_value(decl) -> bool:
        """A declared SIMD variable, a pointer to SIMD vectors, or a temp fed by a
        SIMD-producing call or an element load."""
        return (is_dtype(decl.dtype) or decl.holds_elements
                or (_has_simd_producer(decl) and not _keeps_declared_type(decl))
                or id(decl) in element_values)

    # Nodes: type aliases of SIMD vector types
    root_nodes: list[ValueNode] = []
    alias_nodes: dict[str, ValueNode] = {}
    for alias in unit.type_aliases:
        anode = ValueNode(node_type=NodeType.ALIAS, name=alias.name, extent=alias.target_extent)
        anode.source_type = alias.base_type
        alias_nodes[alias.id] = anode
        root_nodes.append(anode)

    # Nodes: SIMD-typed variable declarations
    decl_nodes: list[tuple] = []
    node_of: dict[str, ValueNode] = {}   # declaration id -> its node
    for decl in unit.variable_decls:
        if not is_graph_value(decl):
            continue
        node = ValueNode(
            node_type=NodeType.VARIABLE, name=decl.name, extent=decl.type_extent,
            anchor_extent=decl.extent,
        )
        node.pointer = "*" in (decl.dtype or "")
        node.holds_elements = decl.holds_elements
        if decl.holds_elements:
            node.source_type = _norm_dtype(decl.dtype)
        if decl.alias_spelled:
            node.spelled_by = alias_nodes.get(decl.type_alias)
        decl_nodes.append((decl, node))
        node_of[decl.id] = node
        root_nodes.append(node)

    # A variable declared with another type (`uint64_t m`) keeps it; a SIMD
    # result written to it converts as a mask read as its integer.
    integer = ValueNode(node_type=NodeType.INTEGER, name=INTEGER_SLOT, extent=None)
    integer.token = INTEGER_SLOT
    for decl in unit.variable_decls:
        if _has_simd_producer(decl) and not is_graph_value(decl):
            node_of[decl.id] = integer

    # Nodes: user functions returning a SIMD type
    # Paired with the function, not keyed by name: overloads share one.
    func_return_nodes: list[tuple] = []           # (FunctionInfo, FUNCTION node)
    for func in unit.functions:
        if is_dtype(func.return_type):
            fnode = ValueNode(node_type=NodeType.FUNCTION, name=func.name, extent=func.return_type_extent)
            fnode.spelled_by = alias_nodes.get(func.return_type_alias)
            func_return_nodes.append((func, fnode))
            root_nodes.append(fnode)

    # Nodes: results of intrinsic calls that write a tracked value
    intrinsic_return_node: dict[str, ValueNode] = {}   # call.id -> node
    for call in unit.call_statements:
        if not call.is_intrinsic:
            continue
        rd = call.returns_decl
        if rd is None or not is_graph_value(rd):
            continue
        irn = ValueNode(node_type=NodeType.INTRINSIC_RETURN, name=call.name, extent=call.extent)
        intrinsic_return_node[call.id] = irn
        root_nodes.append(irn)

    graph = ValueFlowGraph(root_nodes, node_of, intrinsic_return_node)

    # Edges: an alias and everything typed through it are one type
    for alias in unit.type_aliases:
        if alias.target_alias_id is not None:
            graph.link(alias_nodes[alias.id], alias_nodes.get(alias.target_alias_id))
    for decl, node in decl_nodes:
        if decl.type_alias is not None:
            graph.link(node, alias_nodes.get(decl.type_alias))
    for func, fnode in func_return_nodes:
        graph.link(fnode, fnode.spelled_by)

    def node(value) -> ValueNode | None:
        """The node of a variable, None for another argument or an untyped variable."""
        return node_of.get(value.id) if isinstance(value, VariableDecl) else None

    # Edges: returned variable -> its function's return type
    returned: dict[int, list] = {}
    for use in unit.uses:
        if use.context is UseContext.RETURN and use.function is not None:
            returned.setdefault(id(use.function), []).append(use)
    for func, fnode in func_return_nodes:
        for use in returned.get(id(func), ()):
            if (producer := node(use.decl)) is not None:
                graph.add_edge(producer, fnode)
                # A return type shared with an alias can be typed before the
                # returned value; the value then converts at the return.
                if fnode.spelled_by is not None:
                    graph.add_bridge_site(use_extent=use.extent, value=producer, slot=fnode)

    # Edges: intrinsic result -> the variable it writes
    for call in unit.call_statements:
        irn = intrinsic_return_node.get(call.id)
        if irn is None:
            continue
        graph.add_edge(irn, node(call.returns_decl))

    # Edges: user-function calls (return value, arguments)
    func_overloads: dict[str, list] = {}
    for f in unit.functions:
        func_overloads.setdefault(f.name, []).append(f)

    return_node_of = {id(func): fnode for func, fnode in func_return_nodes}
    for call in unit.call_statements:
        if call.is_intrinsic or call.name not in func_overloads:
            continue
        callee_func = _select_overload(call, func_overloads[call.name])
        if callee_func is None:
            continue

        if call.returns_decl is not None and (returned_by := return_node_of.get(id(callee_func))) is not None:
            graph.add_edge(returned_by, node(call.returns_decl))
            graph.add_bridge_site(use_extent=call.extent, value=returned_by, slot=node(call.returns_decl))

        if len(call.args) == len(callee_func.params):
            for arg, arg_extent, param in zip(call.args, call.arg_extents, callee_func.params):
                if not is_dtype(param.dtype) or arg_extent is None:
                    continue
                if isinstance(arg, ElementArg) and arg.pointer is not None:
                    # `f(p[i])` reads a value of the pointer's element type.
                    arg = arg.pointer
                graph.add_edge(node(arg), node(param))
                graph.add_deferred(node(param), node(arg))
                graph.add_bridge_site(use_extent=arg_extent, value=node(arg), slot=node(param))

    # Edges: copies into SIMD variables (`T b = a;`, `b = a;`, `b op= a`)
    for use in unit.uses:
        target = use.other
        if use.context not in (UseContext.COPY, UseContext.COMPOUND_VALUE) or target is None:
            continue
        # A variable of another type (`uint64_t m`) converts what is copied into it.
        if not is_dtype(target.dtype):
            continue
        graph.add_edge(node(use.decl), node(target))
        graph.add_bridge_site(use_extent=use.extent, value=node(use.decl), slot=node(target))

    # Edges: element loads / stores through a pointer to SIMD vectors
    # A load produces the loaded value, a store the element; either end nothing
    # else typed takes the other's type.
    for use in element_uses:
        pointer, value = node(use.decl), node(use.other)
        producer, slot = (value, pointer) if use.context is UseContext.ELEMENT_STORE else (pointer, value)
        graph.add_edge(producer, slot)
        graph.add_deferred(slot, producer)
        graph.add_bridge_site(use_extent=use.boundary, value=producer, slot=slot)

    # Integer reads of masks (`if (m)`, `popcount(m)`)
    for use in unit.uses:
        if intr_reg.is_integer_mask_type(use.decl.dtype) and _reads_integer(use, intr_reg):
            if (value := node(use.decl)) is not None:
                graph.integer_reads.append(BridgeSite(use.extent, value, integer))

    return graph


def _reads_integer(use: Use, registry: IntrinsicRegistry) -> bool:
    """Whether C evaluates this read of an integer mask as the integer: in an
    operator, a condition or a user-call argument, or copied or returned as a
    non-mask.  An intrinsic's own slot types its argument, and the target of a
    compound assignment is written in place."""
    if use.context in (UseContext.EXPRESSION, UseContext.ARGUMENT, UseContext.COMPOUND_VALUE):
        return True
    if use.context is UseContext.COPY:
        # A copy into a mask, or into a variable whose type it deduces, stays a mask.
        target = use.other
        return target is None or not (
            registry.is_integer_mask_type(target.dtype) or _strip_qualifiers(target.dtype) == "auto")
    if use.context is UseContext.RETURN:
        return use.function is None or not registry.is_integer_mask_type(use.function.return_type)
    return False
