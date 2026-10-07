"""The typing pass, the sole typer.  Walking the selected matches in
source order, each op takes the admissible variant whose incoming values need
the fewest conversions, ties going to one canonical default per kind, and its
output slot types the value it produces (first writer wins).  So a consumer
adapts to its producers: `and_si512` adopts its operand's element, and `kand`
of a signed and an unsigned compare falls to the default.  The fallbacks then
type what no producer reached.
"""
from __future__ import annotations

from collections import deque

from pivot.frontend.types import VariableDecl
from pivot.ir.conversions import Conversion, conversion
from pivot.ir.types import (
    MAX_MASK_REGISTER_BITS,
    SCALABLE_MASK_NATIVE_BITS,
    IRKind,
    IRType,
    mask_representative_bits,
    parse_ir_token,
    Domain,
    ElemBits,
)
from pivot.ir.value_flow import NodeType, ValueNode
from pivot.isa.source_vector_types import canonical_vector_token
from pivot.passes.translation.admissibility import slot_ir
from pivot.passes.translation.model import Match, Operand, TranslationContext, Variant


def _mask_slot_rank(slot: IRType, source: IRType | None) -> tuple:
    """Canonical rank of one mask slot, named from the op's own source slot,
    unsigned first:

      * a known granularity (`_i`, `_ui`, `_bN`): that width;
      * a fixed-lane `_bALL` (x86 `__mmaskN`): the representative width of its
        lane count, then the widest that fits one register, then the narrowest;
      * an open-lane `_bALL` (an SVE predicate) or none: the byte register, the
        finest granularity and so lossless under the bitmap cast.
    """
    width = slot.elem_bits.value
    not_uint = slot.domain is not Domain.UINT
    if (
        source is not None and source.kind is IRKind.MASK
        and source.elem_bits is not None and source.elem_bits is not ElemBits.ALL
    ):
        return (not_uint, width != source.elem_bits.value)
    if (
        source is not None and source.kind is IRKind.MASK
        and source.elem_bits is ElemBits.ALL and source.lanes is not None
    ):
        rep = mask_representative_bits(source.lanes)
        fits = (slot.lanes or 0) * width <= MAX_MASK_REGISTER_BITS
        return (not_uint, width != rep, not fits, -width if fits else width)
    return (not_uint, width != SCALABLE_MASK_NATIVE_BITS.value, width)


def _canonical_rank(match: Match, variant: Variant) -> tuple:
    """Tie-break among equally cast-free variants, slot by slot: a data vector or
    scalar signed int first, then closest to 32 bits; a mask by `_mask_slot_rank`."""
    ranks: list[tuple] = []
    for operand in match.operands:
        token = (variant.typing_signature.get(operand.param) or "").strip()
        ir = parse_ir_token(token)
        if ir is None or ir.elem_bits is None:
            continue
        if ir.kind is IRKind.MASK:
            ranks.append(_mask_slot_rank(ir, operand.ir))
        elif ir.kind in (IRKind.VECTOR, IRKind.SCALAR):
            ranks.append((ir.domain is not Domain.INT, abs(ir.elem_bits.value - 32)))
    return tuple(ranks)


def _scalar_gap(operand: Operand, variant: Variant) -> int:
    """|target - source| element-width gap for a SCALAR operand (scalars convert
    implicitly, so this is exactness, not a cast)."""
    source = operand.ir
    target = slot_ir(variant, operand.param)
    if (
        source is None or target is None
        or source.kind is not IRKind.SCALAR or target.kind is not IRKind.SCALAR
        or source.elem_bits is None or target.elem_bits is None
    ):
        return 0
    return abs(source.elem_bits.value - target.elem_bits.value)


def _variant_key(ctx: TranslationContext, match: Match, variant: Variant, index: int) -> tuple:
    """(unbridgeable edges, emitted conversions, scalar width gap, canonical rank,
    corpus order); lower is better."""
    edges: list[Conversion] = []
    gap = 0
    signature = variant.definition.signature
    for operand in match.inputs:
        if slot := (signature.get(operand.param) or "").strip():
            edges.append(conversion(input_token(ctx, operand), slot))
            gap += _scalar_gap(operand, variant)
    if match.output is not None and (slot := (signature.get(match.output.param) or "").strip()):
        # A later producer of an already-typed value is converted at its output.
        edges.append(conversion(slot, value_token(ctx, match.output.decl)))
    unbridgeable = edges.count(Conversion.IMPOSSIBLE)
    casts = len(edges) - unbridgeable - edges.count(Conversion.NONE)
    return (unbridgeable, casts, gap, _canonical_rank(match, variant), index)


def resolve_types(ctx: TranslationContext, selected: list[Match]) -> list[tuple[Match, Variant]]:
    """Each selected match with the variant cast avoidance chooses for it, in
    source order; each produced value is typed as the walk reaches it."""
    typed: list[tuple[Match, Variant]] = []
    for match in sorted(selected, key=lambda m: _anchor(ctx, m)):
        index, variant = min(
            enumerate(match.variants),
            key=lambda item: _variant_key(ctx, match, item[1], item[0]),
        )
        typed.append((match, variant))
        output = match.output
        if output is not None and output.decl is not None \
                and (token := (variant.definition.signature.get(output.param) or "").strip()):
            # Through the call's result node when it has one; it flows to the variable.
            graph = ctx.value_flow
            node = graph.intrinsic_returns.get(output.decl.producer_call_id) or graph.node_for(output.decl)
            if node is not None:
                _type(node, token)
    _type_unreached(ctx, typed)
    return typed


def _anchor(ctx: TranslationContext, match: Match) -> tuple:
    """Source position of the match's result call: the walk order."""
    start = ctx.calls[match.result].extent.start
    return (start.line, start.column, match.id)


def _type_unreached(ctx: TranslationContext, typed: list[tuple[Match, Variant]]) -> None:
    """An input value no producer typed takes its first consumer slot in
    source order; then the deferred edges run until nothing changes; last, a
    pointer or alias takes the register of its declared type."""
    graph = ctx.value_flow
    for match, variant in typed:
        for operand in match.inputs:
            node = graph.node_for(operand.decl)
            token = variant.definition.signature.get(operand.param, "").strip()
            if node is not None and token:
                _type(node, token)

    # Each round types what the last one reached (l1 -> l2 -> l3); first writer
    # wins, so this terminates.
    while True:
        newly_typed = []
        for node in graph.nodes:
            if node.token is None:
                continue
            for target in node.deferred:
                if _assign(target, node.token):
                    newly_typed.append(target)
        if not newly_typed:
            break
        for node in newly_typed:
            _propagate(node)

    for node in graph.nodes:
        if node.token is None and node.source_type and _holds_register(node):
            if (token := canonical_vector_token(node.source_type, ctx.sve_bits)) is not None:
                _type(node, token)


def _type(node: ValueNode, token: str) -> None:
    if _assign(node, token):
        _propagate(node)


def _assign(node: ValueNode, token: str) -> bool:
    """Type an untyped node; a pointer to vectors or an alias takes only a
    vector or mask."""
    if node.token is not None:
        return False
    if _holds_register(node) and not _is_register(token):
        return False
    node.token = token
    return True


def _propagate(node: ValueNode) -> None:
    """Forward a node's type along its edges, transitively; first writer wins."""
    queue = deque([node])
    while queue:
        current = queue.popleft()
        for dependent in current.dependents:
            if _assign(dependent, current.token):
                queue.append(dependent)


def _holds_register(node: ValueNode) -> bool:
    return node.holds_elements or node.node_type is NodeType.ALIAS


def _is_register(token: str) -> bool:
    ir = parse_ir_token(token)
    return ir is not None and ir.kind in (IRKind.VECTOR, IRKind.MASK)


def value_token(ctx: TranslationContext, decl: VariableDecl | None) -> str | None:
    """The token the value-flow graph resolved for a program variable, or None."""
    return ctx.value_flow.token_for(decl)


def input_token(ctx: TranslationContext, operand: Operand) -> str | None:
    """The type of an input's value; an inline call to a user function binds no
    variable and takes its callee's resolved return type."""
    token = value_token(ctx, operand.decl)
    if token is None and operand.callee:
        token = ctx.value_flow.function_token(operand.callee)
    return token
