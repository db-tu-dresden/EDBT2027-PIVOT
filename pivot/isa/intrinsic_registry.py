from __future__ import annotations

import os
import re
from dataclasses import dataclass

import yaml

from pivot.driver.run_options import register_reset_hook
from pivot.lang.languages import get_language
from pivot.isa.immediate_constants import eval_fold
from pivot.utils.corpus_cache import load_corpus_per_file

# The intrinsic corpus is several MB of YAML (sve.yaml, AVX512F.yaml, …); the
# libyaml C loader parses it in a fraction of the pure-Python loader's time.
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

# x86 SIMD extensions, low to high, as the intrinsic flags spell them; an
# intrinsic's audit bucket is its highest.
_X86_ISA_ORDER = ("SSE", "SSE2", "SSE3", "SSSE3", "SSE4.1", "SSE4.2", "AVX", "AVX2", "AVX512")

# Bump when the cached payload shape changes so an older layout is rejected.
_CACHE_VERSION = "3"
_CACHE_FILENAME = ".intrinsic_cache.pkl"


@dataclass(frozen=True)
class Intrinsic:
    """What the corpus says about one intrinsic."""

    # ISA flags as `AVX512F`, `SSE4.1`, `NEON`.
    flags: frozenset[str] = frozenset()
    # The source-language types of its parameters and results, in order.
    input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    # An immediate pseudo-intrinsic's parameter names and `fold:` expression.
    fold: tuple[tuple[str, ...], str] | None = None
    # Its `ir:` signature, ``{input: [...], output: [...]}``.
    ir: dict[str, list[str]] | None = None

    def merged(self, later: Intrinsic) -> Intrinsic:
        """Flags accumulate; every other field keeps the first entry that gives it."""
        return Intrinsic(
            flags=self.flags | later.flags,
            input_types=self.input_types or later.input_types,
            output_types=self.output_types or later.output_types,
            fold=self.fold if self.fold is not None else later.fold,
            ir=self.ir if self.ir is not None else later.ir,
        )

    @property
    def isa(self) -> str:
        """The ISA an audit groups it under: SVE2, SVE or NEON, else its highest
        x86 extension (SSE for any other flags); UNKNOWN without flags."""
        if not self.flags:
            return "UNKNOWN"
        if arm := next((isa for isa in ("SVE2", "SVE", "NEON") if isa in self.flags), None):
            return arm
        highest = "SSE"
        for isa in _X86_ISA_ORDER:
            if isa in self.flags or (isa == "AVX512" and any(flag.startswith("AVX512") for flag in self.flags)):
                highest = isa
        return highest


_NO_ENTRY = Intrinsic()


def _types(section) -> tuple[str, ...]:
    """The type spellings of an `input:` or `output:` mapping."""
    return tuple(t.strip() for t in section.values() if isinstance(t, str)) if isinstance(section, dict) else ()


def _parse_entry(item: dict) -> Intrinsic:
    flags = item.get("flags") or []
    inputs = item.get("input") or {}
    fold, ir = item.get("fold"), item.get("ir")
    return Intrinsic(
        flags=frozenset(
            flag.strip().replace("_", ".").upper()
            for flag in (flags if isinstance(flags, list) else ())
            if isinstance(flag, str) and flag.strip()
        ),
        input_types=_types(inputs),
        output_types=_types(item.get("output") or {}),
        fold=((tuple(p for p in inputs if isinstance(p, str)), fold.strip())
              if isinstance(inputs, dict) and isinstance(fold, str) and fold.strip() else None),
        ir=({
            "input": [str(t) for t in (ir.get("input") or {}).values()],
            "output": [str(t) for t in (ir.get("output") or {}).values()],
        } if isinstance(ir, dict) else None),
    )


def _merge_into(entries: dict[str, Intrinsic], name: str, entry: Intrinsic) -> None:
    entries[name] = entries[name].merged(entry) if name in entries else entry


class IntrinsicRegistry:
    def __init__(self, intrinsics_dir: str) -> None:
        self._assemble(self._load_file_partials(intrinsics_dir))

    def _load_file_partials(self, intrinsics_dir: str) -> list[tuple[str, tuple]]:
        """Each corpus file's parsed contribution as ``(kind, data)``, files sorted
        so the first-writer-wins merge is deterministic.  Only changed files are
        reparsed; ``PIVOT_NO_INTRINSIC_CACHE`` bypasses the cache."""
        return load_corpus_per_file(
            intrinsics_dir,
            cache_filename=_CACHE_FILENAME,
            version=_CACHE_VERSION,
            parse_file=self._parse_file,
            accept=lambda name: name.endswith((".yaml", ".yml")),
            sort_files=True,
            enabled=not os.environ.get("PIVOT_NO_INTRINSIC_CACHE"),
        )

    def _parse_file(self, path: str, fname: str) -> tuple:
        """One file's contribution: a dtype map, or its intrinsics by name."""
        if fname == "dtype_mapping.yaml":
            return ("dtype", self._load_dtype_map(path))
        return ("meta", self._file_entries(path))

    def _assemble(self, ordered: list[tuple[str, tuple]]) -> None:
        """Merge the files in walk order: dtype maps update, intrinsics merge."""
        dtype_map: dict[str, str] = {}
        entries: dict[str, Intrinsic] = {}
        for _rel, (kind, data) in ordered:
            if kind == "dtype":
                dtype_map.update(data)
                continue
            for name, entry in data.items():
                _merge_into(entries, name, entry)
        self._intrinsic_type_to_dtype = dtype_map
        self._intrinsics = entries

    @staticmethod
    def _file_entries(file_path: str) -> dict[str, Intrinsic]:
        """One file's intrinsics by name; the items of one name merge as files do."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = yaml.load(f, Loader=_YAML_LOADER) or []
        except OSError:
            payload = []
        entries: dict[str, Intrinsic] = {}
        for item in payload if isinstance(payload, list) else ():
            name = item.get("name") if isinstance(item, dict) else None
            if isinstance(name, str) and name:
                _merge_into(entries, name, _parse_entry(item))
        return entries

    def _load_dtype_map(self, file_path: str) -> dict[str, str]:
        with open(file_path, "r") as f:
            data = yaml.load(f, Loader=_YAML_LOADER) or {}
        dtype_map = data.get("dtype_mapping", {})

        # Source type spelling -> abstract dtype.
        intrinsic_type_to_dtype: dict[str, str] = {}
        for _, dtype_mapping in dtype_map.items():
            for dtype, intrinsic_type in dtype_mapping.items():
                if intrinsic_type:
                    intrinsic_type_to_dtype[intrinsic_type] = dtype
        return intrinsic_type_to_dtype

    def _entry(self, name: str) -> Intrinsic:
        return self._intrinsics.get(name, _NO_ENTRY)

    def is_known_intrinsic(self, name: str) -> bool:
        """Whether the corpus lists ``name``, a nullary void one such as
        `_mm_sfence` included."""
        return name in self._intrinsics

    def is_typed_intrinsic(self, name: str) -> bool:
        """Whether the corpus gives ``name`` an input or output type: an intrinsic
        that takes or returns a value, which a frontend lowers as a call."""
        entry = self._entry(name)
        return bool(entry.input_types or entry.output_types)

    def is_fold_intrinsic(self, intrinsic_name: str) -> bool:
        """Whether ``intrinsic_name`` builds an immediate (`_MM_SHUFFLE`) rather than
        computing: its call folds to a compile-time integer."""
        return self._entry(intrinsic_name).fold is not None

    def fold_of(self, intrinsic_name: str) -> tuple[tuple[str, ...], str] | None:
        """``(ordered param names, fold expr)`` of a fold pseudo-intrinsic, else
        None.  The expression is C integer arithmetic over the param names."""
        return self._entry(intrinsic_name).fold

    def nullary_fold_constants(self) -> list[tuple[str, str]]:
        """Each parameterless fold pseudo-intrinsic (`_CMP_LE_OQ`) with its value,
        which a backend defines again once the source SIMD header is gone."""
        out: list[tuple[str, str]] = []
        for name, entry in self._intrinsics.items():
            if entry.fold is None or entry.fold[0]:
                continue
            value = eval_fold(entry.fold[1], [], [])
            if value is not None:
                out.append((name, value))
        return out

    def is_dtype(self, dtype: str) -> bool:
        """Whether ``dtype`` spells a source vector or mask type (`__m128i`, `float32x4_t`)."""
        return self._normalize_dtype_spelling(dtype) in self._intrinsic_type_to_dtype

    def is_integer_mask_type(self, dtype: str) -> bool:
        """Whether a type spelling is an N-lane mask the source language defines
        as a plain N-bit integer (x86 `__mmaskN`, abstract dtype `boolN`)."""
        abstract = self._intrinsic_type_to_dtype.get(self._normalize_dtype_spelling(dtype), "")
        return re.fullmatch(r"bool\d+", abstract) is not None

    def intrinsic_isa_of(self, intrinsic_name: str) -> str:
        """The ISA an audit groups an intrinsic under (see ``Intrinsic.isa``)."""
        return self._entry((intrinsic_name or "").strip()).isa

    def ir_signature_of(self, intrinsic_name: str) -> dict[str, list[str]] | None:
        """``{input: [...], output: [...]}``, or None.  Pattern graphs are typed
        from it, so a change makes theirs stale."""
        return self._entry(intrinsic_name).ir

    def ir_token_at(self, intrinsic_name: str, arg_position) -> str | None:
        """Return the IR token for one call position, an int arg index or 'ret'."""
        sig = self.ir_signature_of(intrinsic_name)
        if sig is None:
            return None
        if arg_position == "ret":
            outputs = sig["output"]
            return outputs[0] if outputs else None
        if isinstance(arg_position, int) and 0 <= arg_position < len(sig["input"]):
            return sig["input"][arg_position]
        return None

    def intrinsic_input_types_of(self, intrinsic_name: str) -> list[str]:
        """The C types of an intrinsic's parameters, in order."""
        return list(self._entry(intrinsic_name).input_types)

    def intrinsic_output_types_of(self, intrinsic_name: str) -> list[str]:
        """The C types of an intrinsic's results."""
        return list(self._entry(intrinsic_name).output_types)

    @staticmethod
    def _normalize_dtype_spelling(dtype: str) -> str:
        """``dtype`` without qualifiers; pointers stay."""
        if not dtype:
            return dtype
        tokens = dtype.replace("\t", " ").split()
        tokens = [t for t in tokens if t not in ("const", "volatile", "restrict")]
        return " ".join(tokens)


_cached_intrinsic_registry: IntrinsicRegistry | None = None


def get_intrinsic_registry() -> IntrinsicRegistry:
    global _cached_intrinsic_registry
    if _cached_intrinsic_registry is None:
        _cached_intrinsic_registry = IntrinsicRegistry(intrinsics_dir=get_language().intrinsics_dir)
    return _cached_intrinsic_registry


def _reset_cache() -> None:
    global _cached_intrinsic_registry
    _cached_intrinsic_registry = None


register_reset_hook(_reset_cache)
