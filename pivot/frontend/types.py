from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pivot.ir.source_span import SourceSpan


def _loc_key(extent: SourceSpan) -> str:
    return f"{extent.start.line}:{extent.start.column}"


@dataclass
class VariableDecl:
    id: str = field(init=False)
    name: str
    dtype: str
    extent: SourceSpan
    kind: str  # "var" | "param"
    # Span of the declared type; None for a bare-assignment temp.
    type_extent: Optional[SourceSpan] = None
    producer_call_id: Optional[str] = None
    # Set only for a `const` integer array initialized by plain literals (a
    # permute mask), so a load of it can become an inline vector literal and
    # the constant survives to its use.
    const_init_text: Optional[str] = None   # normalized brace text, e.g. "{0, 2, 4}"
    const_elem_count: Optional[int] = None
    const_elem_bytes: Optional[int] = None
    # Id of the TypeAlias whose type the declaration shares: it is spelled with
    # the alias (`alias_spelled`, `dtype` then holds the underlying SIMD type and
    # the spelling is kept) or initialized from a pointer cast through it.
    type_alias: Optional[str] = None
    alias_spelled: bool = False
    # A pointer to SIMD vectors: the type graph types it by its pointee, and
    # `p[i]` / `*p` are uses of that value.
    holds_elements: bool = False

    def __post_init__(self) -> None:
        self.id = f"{self.name}@{_loc_key(self.extent)}"

    def __repr__(self) -> str:
        return (
            f"VariableDecl(id={self.id!r}, name={self.name!r}, dtype={self.dtype!r}, "
            f"kind={self.kind!r})"
        )


@dataclass
class ConstantArg:
    id: str = field(init=False)
    # Text written back into translated code: the source spelling, or the folded
    # integer of an immediate macro (`_MM_SHUFFLE(...)`, `_CMP_LE_OQ`).
    value: str
    extent: SourceSpan
    # What pattern matching compares.  An integer literal keys by its decimal
    # value, so `0xB1`, `0261` and `177u` bind the same pattern constant; any
    # other constant keys by `value`.
    key: Optional[str] = None
    # An argument that is a call to a user function: the callee, whose return
    # type is the argument's.
    callee: Optional[str] = None

    def __post_init__(self) -> None:
        self.id = f"{self.value}@{_loc_key(self.extent)}"
        if self.key is None:
            self.key = self.value

    def __repr__(self) -> str:
        return f"ConstantArg(id={self.id!r}, value={self.value!r})"

    @property
    def name(self) -> str:
        return self.value

    @property
    def dtype(self) -> str:
        return "auto"


@dataclass
class ElementArg(ConstantArg):
    """A user-function argument that reads a SIMD element through a pointer
    (`f(p[i])`, `f(*p)`): opaque text like any complex argument, but the type
    graph types it as the pointer's pointee."""
    pointer: Optional[VariableDecl] = None


@dataclass
class CallStatement:
    id: str = field(init=False)
    # Callee name, e.g. "_mm_add_epi32".
    name: str
    is_intrinsic: bool
    # The call expression itself.
    extent: SourceSpan
    # Each argument is a variable reference or an inline constant.
    args: list[VariableDecl | ConstantArg] = field(default_factory=list)
    # Call-site span of each argument, parallel to `args`: a variable argument
    # is the shared declaration, whose `.extent` is the declaration's span.
    arg_extents: list[Optional[SourceSpan]] = field(default_factory=list)
    # The declaration receiving the result, None for an unused/void call.
    returns_decl: Optional[VariableDecl] = None
    # The call with the assignment/declaration around it.
    extent_with_var: Optional[SourceSpan] = None
    # For a declaration, the " = call(...)" initializer alone; cutting it leaves "T x;".
    initializer_extent: Optional[SourceSpan] = None
    # The assignment's right-hand side is exactly the call (no cast or operator).
    is_strict_assignment: bool = False
    # Straight-line region (unique id) of the call: each part of a control-flow
    # statement (branch, loop body, case) opens a new one, 0 is the top level.
    region: int = 0

    def __post_init__(self) -> None:
        self.id = f"{self.name}@{_loc_key(self.extent)}"

    def __repr__(self) -> str:
        arg_preview = [a.id for a in self.args]
        return (
            f"CallStatement(id={self.id!r}, name={self.name!r}, "
            f"intrinsic={self.is_intrinsic}, args={arg_preview}, "
            f"strict={self.is_strict_assignment}, returns_decl={self.returns_decl!r})"
        )
