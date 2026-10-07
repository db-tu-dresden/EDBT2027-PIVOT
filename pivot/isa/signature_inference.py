"""A source definition's signature, read off the intrinsics its direct body calls
and checked against the primitive's I/O shape classes."""

from __future__ import annotations

import re

from tree_sitter import Node

from pivot.isa.intrinsic_registry import IntrinsicRegistry
from pivot.isa.templates import MASK_TOKENS, SCALAR_TOKENS, TOKEN_RE, VECTOR_TOKENS
from pivot.lang.grammars import PATTERN, callee_name, text, walk

# Coarse I/O shape classes.  An existing intrinsic of the wrong kind (the
# predicate `svdup_n_b8` for a data-vector `set1`) is rejected by its class.
SHAPE_VECTOR = "vector"
SHAPE_MASK = "mask"
SHAPE_SCALAR = "scalar"
SHAPE_POINTER = "pointer"

# NEON `x4_t`-style vector types (int32x4_t, float32x4x2_t, ...).
_NEON_VECTOR_RE = re.compile(r"x\d+(x\d+)?_t$")


def classify_intrinsic_type(type_spelling: str) -> str:
    """The shape class of a concrete intrinsic C type."""
    s = type_spelling.strip()
    if "*" in s:
        return SHAPE_POINTER
    # Before the vector prefixes: `svbool_t` starts with `sv`, `__mmask16` with `__m`.
    if s == "svbool_t" or s.startswith("__mmask"):
        return SHAPE_MASK
    if s.startswith("__m") or s.startswith("sv") or _NEON_VECTOR_RE.search(s):
        return SHAPE_VECTOR
    return SHAPE_SCALAR


def classify_canonical_token(token: str) -> str | None:
    """The shape class of a primitive signature token (``{v}``, ``{s}``, a
    pointer, ``imm*``); None for a concrete spelling (``v8f``), which disables
    the check rather than risk misclassifying it."""
    s = token.strip()
    if "*" in s:
        return SHAPE_POINTER
    if re.fullmatch(r"imm\d+", s):
        return SHAPE_SCALAR
    if not (m := TOKEN_RE.fullmatch(s)):
        return None
    key = m.group(1)
    if key in MASK_TOKENS:
        return SHAPE_MASK
    if key in VECTOR_TOKENS:
        return SHAPE_VECTOR
    if key in SCALAR_TOKENS:
        return SHAPE_SCALAR
    return None


def _shapes_compatible(canonical: str, intrinsic: str) -> bool:
    # NEON has no predicate registers: a mask is an all-ones data vector there.
    return canonical == intrinsic or (canonical == SHAPE_MASK and intrinsic == SHAPE_VECTOR)


def infer_signature(
    direct_lines: list[str],
    inputs: list[str],
    output: str | None,
    intrinsics: IntrinsicRegistry,
    io_shape: dict[str, set[str]] | None = None,
) -> tuple[dict[str, str] | None, str]:
    """Each parameter's type from the intrinsic argument it is first passed to
    (the output's from the returning call), or None and the reason."""
    src = "\n".join(direct_lines).encode()
    root = PATTERN.parse(src)
    signature: dict[str, str] = {}

    for var_name in inputs:
        if (use := _first_argument_use(root, src, var_name)) is None:
            return None, f"input variable '{var_name}' not used in direct body"
        call_name, arg_index = use
        arg_types = intrinsics.intrinsic_input_types_of(call_name)
        if not arg_types:
            return None, f"intrinsic metadata has no input types: {call_name}"
        if arg_index >= len(arg_types):
            return None, (
                f"intrinsic argument index out of bounds: "
                f"{call_name} arg_index={arg_index} input_count={len(arg_types)}"
            )

        inferred = arg_types[arg_index]
        if var_name in signature and signature[var_name] != inferred:
            return None, (
                f"conflicting inferred input types for '{var_name}': "
                f"{signature[var_name]} vs {inferred}"
            )
        if bad := _shape_mismatch(io_shape, var_name, inferred):
            return None, bad
        signature[var_name] = inferred

    if output:
        if not (ret_call := _returning_call(root, src)):
            return None, "could not locate returning intrinsic call for primitive output"
        out_types = intrinsics.intrinsic_output_types_of(ret_call)
        if not out_types:
            return None, f"intrinsic metadata has no output types: {ret_call}"
        if bad := _shape_mismatch(io_shape, output, out_types[0]):
            return None, bad
        signature[output] = out_types[0]

    return signature, ""


def _first_argument_use(root: Node, src: bytes, name: str) -> tuple[str, int] | None:
    """The call the first mention of ``name`` is an argument of, and its slot."""
    node = next((n for n in walk(root) if n.type == "identifier" and text(n, src) == name), None)
    while node is not None and node.parent is not None:
        if node.parent.type == "parenthesized_expression":
            return None
        if node.parent.type == "arguments":
            return callee_name(node.parent.parent, src), node.parent.named_children.index(node)
        node = node.parent
    return None


def _returning_call(root: Node, src: bytes) -> str | None:
    """The call the value of a body comes from: in its last statement with a
    call, the last call not inside parentheses."""
    for statement in reversed(root.named_children):
        calls = []
        stack = [statement]
        while stack:
            node = stack.pop()
            if node.type == "call_expression":
                calls.append(node)
            if node.type not in ("arguments", "parenthesized_expression"):
                stack.extend(reversed(node.named_children))
        if calls:
            return callee_name(calls[-1], src)
    return None


def _shape_mismatch(
    io_shape: dict[str, set[str]] | None, param: str, inferred_type: str
) -> str | None:
    """Why ``inferred_type`` fits none of the classes ``io_shape`` allows ``param``,
    or None.  A parameter may allow several classes (a vector in one target
    definition, a scalar in another)."""
    if not io_shape or param not in io_shape:
        return None
    allowed = io_shape[param]
    actual = classify_intrinsic_type(inferred_type)
    if not any(_shapes_compatible(canonical, actual) for canonical in allowed):
        return (
            f"IO shape mismatch for '{param}': primitive expects "
            f"{'/'.join(sorted(allowed))}, intrinsic type '{inferred_type}' is {actual}"
        )
    return None
