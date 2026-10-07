"""The name of the helper realizing a definition: one scheme for every target."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pivot.isa.primitive_registry import Definition

_UNSAFE = re.compile(r"[^a-zA-Z0-9_]")


def helper_name(definition: "Definition") -> str:
    """``pivot_<primitive>_<suffix>``: the inputs' one token, else all of them
    ``x``-joined, then the output token unless an input has it."""
    primitive, signature = definition.primitive, definition.signature
    inputs = [signature.get(name, "unknown") for name in primitive.input]
    output = signature.get(primitive.output, "") if primitive.output else None
    if not inputs:
        suffix = output or "void"
    else:
        suffix = inputs[0] if len(set(inputs)) == 1 else "x".join(inputs)
        if output and output not in inputs:
            suffix = f"{suffix}x{output}"
    return f"pivot_{_safe(primitive.name)}_{_safe(suffix)}"


def _safe(text: str) -> str:
    return _UNSAFE.sub("_", text.strip().lower())
