"""The middle-end's records: one Match per rooted pattern match, carried from
matching through selection and typing to emit, and the context the stages of
one file's translation share."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pivot.frontend.base import ParsedUnit
    from pivot.frontend.types import CallStatement, VariableDecl
    from pivot.ir.graph import Graph
    from pivot.ir.types import IRType
    from pivot.ir.value_flow import ValueFlowGraph
    from pivot.isa.primitive_registry import Definition, Primitive


@dataclass(frozen=True)
class Operand:
    """A primitive parameter bound to the matched source."""

    param: str
    # The type the matched intrinsic declares for it, at the assumed SVE width;
    # None for a pointer or an unbound parameter.
    ir: IRType | None = None
    # The program variable the argument is, if it is one.
    decl: VariableDecl | None = None
    # The argument's source text; None when no argument binds the parameter.
    text: str | None = None
    # An argument that is a call to a user function: its callee.
    callee: str | None = None


@dataclass(frozen=True)
class Variant:
    """A target definition a match admits."""

    # Spelled by the backend (width-symbolic on tsl vla: `vb_ui`, `vi`); emit reads it.
    definition: Definition
    # Its signature at concrete lane counts for the assumed SVE width
    # (`vb_ui` -> `v4b_ui`); admissibility and typing read it.
    typing_signature: dict[str, str]
    # Target statements emitted.
    cost: int


@dataclass(frozen=True)
class Match:
    """One rooted match of a pattern in the program graph."""

    # Position in matching order.
    index: int
    primitive: Primitive
    pattern: Graph
    # Pattern node -> program node.
    mapping: dict[str, str]
    # Every call of the match (program node ids, i.e. `CallStatement.id`).
    calls: frozenset[str]
    # The call the pattern's result maps to; the replacement goes here.
    result: str
    # Calls kept because a value they compute is used outside the match; the
    # replacement computes them a second time.
    kept: frozenset[str]
    # One per primitive input, in order.
    inputs: tuple[Operand, ...]
    output: Operand | None
    # The target variants its source signature admits, in corpus order.
    variants: tuple[Variant, ...]

    @property
    def id(self) -> str:
        return f"occ_{self.index}"

    @property
    def operands(self) -> tuple[Operand, ...]:
        return self.inputs if self.output is None else (*self.inputs, self.output)

    @property
    def translated(self) -> frozenset[str]:
        """The calls the replacement stands for."""
        return self.calls - self.kept

    @property
    def removed(self) -> frozenset[str]:
        """Translated calls other than the result; the rewrite deletes them."""
        return self.translated - {self.result}


@dataclass(frozen=True)
class TranslationContext:
    """What the stages of one file's translation share."""

    unit: ParsedUnit
    # Target label (`clang_builtins`, `tsl`, `core_simd`).
    target: str
    # The SVE vector width matching and typing assume.
    sve_bits: int
    # The program graph the patterns match in.
    program: Graph
    # The one value-flow graph; the typing pass types it, emit reads it.
    value_flow: ValueFlowGraph

    @property
    def calls(self) -> dict[str, CallStatement]:
        return self.program.calls
