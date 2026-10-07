"""How code is spelled in one host language: the calls, helpers and companion
unit (C's header, Rust's module) a backend generates, and the statements the
preprocessing passes insert.  Shared by every backend of that language; the
:class:`~pivot.backend.backend_emitter.BackendEmitter` owns what is called.
"""

from __future__ import annotations

import bisect
import difflib
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from xml.etree import ElementTree


@dataclass
class HelperSignature:
    """The language-independent shape of one generated helper function.

    ``return_type`` is the already-lowered target type text ("void" / an empty
    string denote no return).  ``params`` are ``(type_text, name)`` pairs, their
    types already lowered by the backend.  The body is passed separately to
    :meth:`LanguageSyntax.render_helper` as indent-free logical lines.

    ``template_params`` are compile-time ``(type_text, name)`` parameters: the
    immediates a library takes as template arguments (TSL's ``shift_left_imm<Vec,
    shift>``).

    ``type_params`` are generic type parameter names (C++ ``typename P``, Rust
    ``<P>``), for the untyped-pointer helpers that take ``*const P`` for any ``P``
    (Rust's analog of ``const void*``).  They precede ``template_params``."""

    name: str
    return_type: str
    params: list[tuple[str, str]]
    template_params: list[tuple[str, str]] = field(default_factory=list)
    type_params: list[str] = field(default_factory=list)


class LanguageSyntax(ABC):
    """How generated target code is spelled in one host language."""

    @abstractmethod
    def emit_call(self, symbol: str, args: list[str], template_args: list[str] | None = None) -> str:
        """Spell a call to helper ``symbol`` with runtime ``args``.

        ``template_args`` are compile-time arguments placed in the language's
        template / const-generic position (C++ ``symbol<…>(…)``, Rust
        ``symbol::<…>(…)``); empty/None for the common no-immediate case."""

    @abstractmethod
    def render_helper(self, signature: HelperSignature, body_lines: list[str]) -> str:
        """Render one helper function: its signature plus an indent-free body."""

    @abstractmethod
    def frame_companion(
        self, guard: str, system_includes: list[str], body_lines: list[str]
    ) -> str:
        """Wrap generated ``body_lines`` (helpers, type aliases, constants…) into
        a compilable companion unit: include guard / module framing plus the
        language's own required ``system_includes``."""

    @abstractmethod
    def companion_binding(self, filename: str) -> list[str]:
        """Directive line(s) that make the companion file ``filename``'s symbols
        visible at the translated call sites (C ``#include``; Rust ``mod``/``use``)."""

    def vector_literal(self, type_text: str, elements_text: str) -> str:
        """Spell an inline vector literal of type ``type_text`` from an element
        list ``elements_text``; only reached for a backend that supports literals."""
        raise NotImplementedError(f"{type(self).__name__} has no vector literal")

    # A hoisted temporary is a bare assignment (`t = expr;`, via expr_statement),
    # never a declaration: only tree-sitter parses the intermediate file, the
    # type graph types the temp from its producer, and the nesting pass inlines
    # it again.

    @abstractmethod
    def expr_statement(self, expr: str) -> str:
        """Spell ``expr`` as a statement executed for its side effects."""

    @abstractmethod
    def return_void(self) -> str:
        """Spell a bare value-less return statement."""

    @abstractmethod
    def normalize_declared_type(self, type_text: str, pointer_decl: bool) -> str:
        """Normalize an emitted declared-type spelling into the shape the
        type-extent rewrite expects, adding the pointer form when the original
        declaration was a pointer."""

    @abstractmethod
    def split_direct_statements(self, value: str | list[str]) -> list[str]:
        """Split a primitive ``direct`` body into one statement per element.

        Accepts the list-of-lines form and multi-line YAML scalars.  Physical
        lines are joined at this language's statement boundaries so every
        consumer (return heuristics, line-count costs, pattern synthesis) sees
        exactly one statement per element."""

    @staticmethod
    def _raw_direct_lines(value: str | list[str]) -> list[str]:
        """Flatten a ``direct`` body to stripped, non-empty physical lines."""
        raw_lines = value.splitlines() if isinstance(value, str) else [
            segment for item in value for segment in str(item).splitlines()
        ]
        return [stripped for raw in raw_lines if (stripped := raw.strip())]

    def wrap_block(self, lines: list[str], indent: str) -> str:
        """Wrap ``lines`` in a brace-delimited block, used when a temp must be
        inserted before a single-statement child of a control-flow statement.
        Brace syntax is shared by C/C++/Rust, so this has a default."""
        inner = f"{indent}    "
        body = "\n".join(f"{inner}{line}" if line else "" for line in lines)
        return f"{{\n{body}\n{indent}}}"

    def format_file(self, path: str, original: str) -> None:
        """Pretty-format ``path`` in place, in the style configured for
        ``original`` (the user's file it copies); by default a no-op."""


class CxxSyntax(LanguageSyntax):
    """C / C++ syntax: textual ``#include``, ``static inline`` helpers, guarded
    headers.  Shared by the clang_builtins and TSL backends."""

    def emit_call(self, symbol: str, args: list[str], template_args: list[str] | None = None) -> str:
        template = f"<{', '.join(template_args)}>" if template_args else ""
        return f"{symbol}{template}({', '.join(args)})"

    def render_helper(self, signature: HelperSignature, body_lines: list[str]) -> str:
        params = ", ".join(f"{type_text} {name}" for type_text, name in signature.params)
        return_type = signature.return_type or "void"
        lines = []
        generic = [f"typename {t}" for t in signature.type_params] + [
            f"{type_text} {name}" for type_text, name in signature.template_params
        ]
        if generic:
            lines.append(f"template <{', '.join(generic)}>")
        lines.append(f"static inline {return_type} {signature.name}({params}) {{")
        lines.extend(f"    {line}" if line else "" for line in body_lines)
        lines.append("}")
        return "\n".join(lines)

    def frame_companion(
        self, guard: str, system_includes: list[str], body_lines: list[str]
    ) -> str:
        parts = [f"#ifndef {guard}", f"#define {guard}", ""]
        parts.extend(f"#include <{header}>" for header in system_includes)
        parts.append("")
        parts.extend(body_lines)
        parts.append(f"#endif  // {guard}")
        parts.append("")
        return "\n".join(parts)

    def companion_binding(self, filename: str) -> list[str]:
        return [f'#include "{filename}"']

    def vector_literal(self, type_text: str, elements_text: str) -> str:
        # C-style cast of an ext_vector brace initializer: `(v16i){0, 2, ...}`.
        return f"({type_text}){elements_text}"

    def expr_statement(self, expr: str) -> str:
        return f"{expr};"

    def return_void(self) -> str:
        return "return;"

    def normalize_declared_type(self, type_text: str, pointer_decl: bool) -> str:
        parts = [p for p in (type_text or "").replace("\t", " ").split()
                 if p not in {"const", "volatile", "restrict"}]
        base = " ".join(parts).replace("*", " ").strip()
        return f"{base} *" if pointer_decl else base

    def split_direct_statements(self, value: str | list[str]) -> list[str]:
        # A line ending in ';', '{' or '}' ends a statement; any other ending is
        # a wrapped continuation joined with the next line.  Preprocessor
        # directives ('#...') are line-oriented: they always stand alone and
        # never join with, or absorb, a neighbouring statement.
        statements: list[str] = []
        for line in self._raw_direct_lines(value):
            if (
                line.startswith("#")
                or not statements
                or statements[-1].endswith((";", "{", "}"))
                or statements[-1].startswith("#")
            ):
                statements.append(line)
            else:
                statements[-1] = f"{statements[-1]} {line}"
        return statements

    def format_file(self, path: str, original: str) -> None:
        # Only the lines that differ from ``original``: the user's own code keeps
        # its layout.  Via stdin, so clang-format looks up the .clang-format beside it.
        try:
            with open(path, "rb") as f:
                source = f.read()
            with open(original, "rb") as f:
                ranges = _changed_lines(f.read(), source)
            if not ranges:
                return
            result = subprocess.run(
                ["clang-format", f"--assume-filename={original}", "--output-replacements-xml",
                 *(f"--lines={a}:{b}" for a, b in ranges)],
                input=source, stdout=subprocess.PIPE, check=True,
            )
            changed = {line for a, b in ranges for line in range(a - 1, b)}
            with open(path, "wb") as f:
                f.write(_apply_within(source, result.stdout, changed))
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            print(f"[nesting] clang-format skipped: {e}")


class RustSyntax(LanguageSyntax):
    """Rust syntax: a companion file is bound as a module (``mod x; use x::*;``)."""

    def emit_call(self, symbol: str, args: list[str], template_args: list[str] | None = None) -> str:
        turbofish = f"::<{', '.join(template_args)}>" if template_args else ""
        return f"{symbol}{turbofish}({', '.join(args)})"

    def render_helper(self, signature: HelperSignature, body_lines: list[str]) -> str:
        params = ", ".join(f"{name}: {type_text}" for type_text, name in signature.params)
        ret = signature.return_type
        arrow = f" -> {ret}" if ret and ret != "void" else ""
        generic = list(signature.type_params) + [
            f"const {name}: {type_text}" for type_text, name in signature.template_params
        ]
        generics = ("<" + ", ".join(generic) + ">") if generic else ""
        lines = ["#[inline]", f"pub fn {signature.name}{generics}({params}){arrow} {{"]
        lines.extend(f"    {line}" if line else "" for line in body_lines)
        lines.append("}")
        return "\n".join(lines)

    def frame_companion(
        self, guard: str, system_includes: list[str], body_lines: list[str]
    ) -> str:
        parts = [f"// {guard}"]
        parts.extend(f"use {path};" for path in system_includes)
        if system_includes:
            parts.append("")
        parts.extend(body_lines)
        parts.append("")
        return "\n".join(parts)

    def companion_binding(self, filename: str) -> list[str]:
        module = filename.rsplit(".", 1)[0]
        return [f"mod {module};", f"use {module}::*;"]

    def expr_statement(self, expr: str) -> str:
        return f"{expr};"

    def return_void(self) -> str:
        return "return;"

    def normalize_declared_type(self, type_text: str, pointer_decl: bool) -> str:
        # Rust has no C-style `const`/`volatile`/`restrict` type keywords to strip.
        base = (type_text or "").strip()
        return f"*const {base}" if pointer_decl else base

    def split_direct_statements(self, value: str | list[str]) -> list[str]:
        # Rust statements/blocks end in ';', '{' or '}'; there are no
        # preprocessor directives (a leading '#' is an attribute on the item).
        statements: list[str] = []
        for line in self._raw_direct_lines(value):
            if not statements or statements[-1].endswith((";", "{", "}")):
                statements.append(line)
            else:
                statements[-1] = f"{statements[-1]} {line}"
        return statements

    def format_file(self, path: str, original: str) -> None:
        # The whole file: rustfmt has no stable line ranges.
        # Via stdin: by path, rustfmt resolves `mod pivot_core_simd;` and fails,
        # since the companion is not beside the file yet.  Formatting is
        # cosmetic, so a missing or failing rustfmt is only reported.
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            result = subprocess.run(
                ["rustfmt", "--edition", "2021"],
                input=source, capture_output=True, text=True, check=True,
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            print(f"[nesting] rustfmt skipped: {e}")


def _changed_lines(original: bytes, text: bytes) -> list[tuple[int, int]]:
    """The 1-based inclusive line ranges of ``text`` that differ from ``original``."""
    matcher = difflib.SequenceMatcher(None, original.splitlines(), text.splitlines(), autojunk=False)
    return [(j1 + 1, j2) for tag, _, _, j1, j2 in matcher.get_opcodes() if tag in ("replace", "insert")]


def _apply_within(source: bytes, replacements_xml: bytes, lines: set[int]) -> bytes:
    """``source`` with clang-format's replacements that edit only ``lines`` (0-based).
    clang-format lays out a statement as a whole, even past the requested lines, so
    edits that share a line stand or fall together."""
    line_starts = [0, *(i + 1 for i, byte in enumerate(source) if byte == 0x0A)]
    # Each group: its edits and the lines they touch.
    groups: list[tuple[list[tuple[int, int, bytes]], set[int]]] = []
    reach = -1
    for r in ElementTree.fromstring(replacements_xml).iter("replacement"):
        start = int(r.get("offset"))
        end = start + int(r.get("length"))
        old, new = source[start:end], (r.text or "").encode()
        # Only the part that changes counts: `\n    ` -> `\n  ` edits the second line alone.
        head = len(os.path.commonprefix([old, new]))
        tail = len(os.path.commonprefix([old[head:][::-1], new[head:][::-1]]))
        start, end, new = start + head, end - tail, new[head:len(new) - tail]
        # Replacing a line's newline joins the next line: the end's line counts too.
        first = bisect.bisect_right(line_starts, start) - 1
        last = bisect.bisect_right(line_starts, end) - 1
        if first > reach:
            groups.append(([], set()))
        reach = max(reach, last)
        groups[-1][0].append((start, end, new))
        groups[-1][1].update(range(first, last + 1))
    out = bytearray(source)
    for edits, touched in reversed(groups):
        if touched <= lines:
            for start, end, new in reversed(edits):
                out[start:end] = new
    return bytes(out)


# Process-wide singletons: one syntax object per host language, shared by every
# backend targeting it.
CXX_SYNTAX = CxxSyntax()
RUST_SYNTAX = RustSyntax()
