from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Optional

from pivot.ir.source_span import SourceSpan
from pivot.frontend.types import CallStatement, VariableDecl


@dataclass
class FunctionInfo:
    """A user-defined function: the type graph links its return type to the
    values it returns and its parameters to the arguments of its calls."""
    name: str
    return_type: str
    return_type_extent: Optional[SourceSpan] = None
    # In order; an unnamed parameter is not declared in the unit.
    params: list[VariableDecl] = field(default_factory=list)
    # Id of the TypeAlias the return type is spelled with (see VariableDecl.type_alias).
    return_type_alias: Optional[str] = None


@dataclass
class RedundantCast:
    """A SIMD pointer cast that becomes redundant after translation.

    E.g. C ``(__m512i*)p`` or Rust ``p as *const __m512i``: the translated
    helpers take ``void*``, so the cast is stripped.
    """
    cast_extent: SourceSpan   # span of the entire cast expression
    inner_text: str           # text of the unwrapped expression (replacement)


@dataclass
class TypeAlias:
    """A type alias of a SIMD vector type: C++ ``using A = __m128i;`` / C
    ``typedef __m128i A;``, chains (``using B = A;``) included.

    In the source language every spelling of the alias is one type, so the type
    graph types it as one value: every declaration spelled with it and every
    pointer cast through it shares its node.  Only ``target_extent`` (the vector
    type name the alias is defined as) is rewritten; a chained alias names
    another alias there and keeps its spelling."""
    id: str
    name: str
    base_type: str                          # underlying SIMD type, e.g. "__m128i"
    target_extent: Optional[SourceSpan]     # the raw type name to rewrite, None for a chain
    target_alias_id: Optional[str] = None   # the alias a chain names


@dataclass
class TypeSite:
    """A spelling of a SIMD vector type outside the declarations the type graph
    owns: a cast type, a ``sizeof``/``alignof`` operand, a template argument, a
    field, a prototype.  Rewritten to the type of ``owner`` (the declaration the
    spelling types, e.g. a pointer initialized by the cast) or, for a site that
    carries no value, to a register of the same width."""
    extent: SourceSpan
    type_name: str                          # the raw SIMD type name spelled here
    owner: Optional[VariableDecl] = None


class UseContext(Enum):
    """How a statement reads a variable: the role of the mention, seen through
    the parentheses and casts around it."""
    COPY = "copy"                        # the whole value of `T x = v;`, `x = v;`
    COMPOUND_VALUE = "compound_value"    # v in `x op= v`
    ASSIGNED = "assigned"                # x in `x op= v`: read, then written in place
    RETURN = "return"                    # `return v;`
    ARGUMENT = "argument"                # an argument of a user-function call
    INTRINSIC_ARGUMENT = "intrinsic_argument"
    ELEMENT_LOAD = "element_load"        # p in `x = p[i];`, `T x = *p;`
    ELEMENT_STORE = "element_store"      # p in `p[i] = v;`
    EXPRESSION = "expression"            # anything else: an operator operand, a condition, an index


@dataclass
class Use:
    """One read of a variable.  The program graph, the value-flow graph and the
    integer reads of masks each take the uses they need."""
    decl: VariableDecl
    extent: SourceSpan                   # the identifier
    context: UseContext
    # The variable at the other end: the one a copy or a load writes (None when
    # that is no variable), the value a store stores.
    other: Optional[VariableDecl] = None
    # An argument's call, when the frontend lowers it: the program graph links
    # the value as its operand.
    call: Optional[CallStatement] = None
    # RETURN: the enclosing function (None in a macro body).
    function: Optional[FunctionInfo] = None
    # ELEMENT_*: the expression crossing the pointer boundary (the access of a
    # load, the stored value of a store), where a conversion goes.
    boundary: Optional[SourceSpan] = None


@dataclass
class ParsedUnit:
    """Everything the language-agnostic core needs from one source file."""
    path: str
    call_statements: list[CallStatement] = field(default_factory=list)
    variable_decls: list[VariableDecl] = field(default_factory=list)
    # Header/import directives that pull in the SIMD intrinsics (Rust `use`
    # decls, C `#include`s), each mapped to its source span for later rewriting.
    simd_headers: dict[str, SourceSpan] = field(default_factory=dict)
    std_headers: dict[str, SourceSpan] = field(default_factory=dict)
    functions: list[FunctionInfo] = field(default_factory=list)
    redundant_casts: list[RedundantCast] = field(default_factory=list)
    type_aliases: list[TypeAlias] = field(default_factory=list)
    type_sites: list[TypeSite] = field(default_factory=list)
    # Every read of a variable, statement by statement (function bodies, then
    # macro bodies).
    uses: list[Use] = field(default_factory=list)


class Frontend(Protocol):
    def parse(self, path: str) -> ParsedUnit: ...
