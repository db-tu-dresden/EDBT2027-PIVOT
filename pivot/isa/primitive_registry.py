from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from pivot.driver.run_options import register_reset_hook
from pivot.isa.definition_expander import DefinitionExpander, ExpansionRequest
from pivot.isa.signature_inference import classify_canonical_token
from pivot.lang.languages import get_language

# libyaml C loader: much faster than the pure-Python parser on the primitive corpus.
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

# A call whose intrinsic name the dtype sweep changes.
_DTYPE_SWEPT_CALL_RE = re.compile(r"\{dtype_suffix\}[A-Za-z0-9_]*\s*\(")

# The target whose signatures are the neutral reference for a primitive's I/O shape.
_REFERENCE_TARGET = "clang_builtins"

CANONICAL_DTYPES: frozenset[str] = frozenset({
    "int8", "uint8",
    "int16", "uint16",
    "int32", "uint32",
    "int64", "uint64",
    "float32", "float64",
    "bool8", "bool16", "bool32", "bool64",
})


@dataclass
class Primitive:
    name: str
    input: list[str]
    output: Optional[str]
    definitions: list["Definition"] = field(default_factory=list)


@dataclass
class Definition:
    isa: str
    signature: dict[str, str]
    direct: list[str]
    include: Optional[dict[str, list[str]]]
    primitive: Primitive
    # Architectures on which this source pattern is excluded from matching (see
    # GraphLoader.get_by_source). Lowercased. Only meaningful on source-family
    # (x86/arm) definitions, which are the ones that generate pattern graphs.
    blocked_arches: list[str] = field(default_factory=list)
    # Source pattern that collapses several source intrinsics into one target op;
    # a run with `translation.nto1: false` excludes it (see GraphLoader.get_by_source).
    nto1: bool = False


def _isa(raw_def: dict[str, Any]) -> str:
    return str(raw_def.get("isa", "")).strip().lower()


class PrimitiveRegistry:
    def __init__(self, prim_dir: str) -> None:
        self.mapping: dict[str, Primitive] = {}
        self._definition_expander = DefinitionExpander()
        language = get_language()
        self._source_families = frozenset(language.source_backends)
        self._allowed_isas = self._source_families | frozenset(language.target_backends)
        self._load_primitives(prim_dir)

    def is_source_family(self, isa: str) -> bool:
        """Whether ``isa`` is a source family (x86, arm), whose definitions are
        swept and read their signature off their intrinsics."""
        return isa in self._source_families

    @staticmethod
    def _yaml_files(root_dir: str) -> list[str]:
        files: list[str] = []
        for root, dirnames, filenames in os.walk(root_dir):
            dirnames.sort()
            for filename in sorted(filenames):
                if filename.endswith(".yaml"):
                    files.append(os.path.join(root, filename))
        return files

    @staticmethod
    def _validate_single_swept_intrinsic(primitive_name: str, file_path: str, isa: str, direct: list[str]) -> None:
        """A source body calls at most one swept intrinsic: the sweep keeps a
        variant only if its intrinsic exists, which it cannot check for two jointly."""
        swept = _DTYPE_SWEPT_CALL_RE.findall("\n".join(direct))
        if len(swept) > 1:
            raise ValueError(
                f"{file_path}: primitive '{primitive_name}' has an {isa} definition with "
                f"{len(swept)} type-swept intrinsic calls (each containing '{{dtype_suffix}}') "
                f"in one body; only one is allowed, because template expansion cannot guarantee "
                f"every swept variant compiles when several swept intrinsics are composed. Split "
                f"the lowering into separate primitives, or spell the other operations as concrete "
                f"(non-templated) intrinsics."
            )

    @staticmethod
    def _validate_signature_keys(primitive_name: str, expected_keys: list[str], signature: dict[str, str]) -> None:
        """Every parameter has a signature entry.  Extra keys name the helper types
        a body spells as `{simd_<key>}`."""
        expected = set(expected_keys)
        got = set(signature.keys())
        if not expected.issubset(got):
            missing = sorted(expected - got)
            raise ValueError(
                f"{primitive_name}: signature is missing required IO keys {missing} "
                f"(expected at least {sorted(expected)}, got {sorted(got)})"
            )

    def _load_primitives(self, prim_dir: str) -> None:
        for file_path in self._yaml_files(prim_dir):
            with open(file_path, "r", encoding="utf-8") as f:
                # One YAML document per primitive.
                for payload in yaml.load_all(f, Loader=_YAML_LOADER):
                    if not payload:
                        continue
                    if not isinstance(payload, dict):
                        raise ValueError(f"{file_path}: primitive YAML document root must be a mapping")
                    self._load_primitive_document(payload, file_path)

    def _load_primitive_document(self, payload: dict[str, Any], file_path: str) -> None:
        primitive_name = str(payload["primitive"]).strip()
        if primitive_name in self.mapping:
            raise ValueError(f"{file_path}: primitive '{primitive_name}' already defined")

        input_params = list(payload.get("input", []))
        output_param = payload.get("output")
        expected_signature_keys = [*input_params, *([output_param] if output_param else [])]
        request = ExpansionRequest(
            primitive=primitive_name,
            inputs=input_params,
            output=output_param,
            allowed_dtypes=CANONICAL_DTYPES,
            io_shape=self._io_shape(payload.get("definitions", [])),
        )
        primitive = Primitive(name=primitive_name, input=input_params, output=output_param)
        syntax = get_language().load_syntax()

        for raw_def in payload.get("definitions", []):
            isa = _isa(raw_def)
            if isa not in self._allowed_isas:
                raise ValueError(f"{file_path}: invalid isa '{isa}' in primitive '{primitive_name}'")
            seed = {
                "isa": isa,
                "dtype": str(raw_def["dtype"]).strip() if "dtype" in raw_def else None,
                "direct": syntax.split_direct_statements(raw_def["direct"]),
                "include": raw_def.get("include"),
                "blocked_arches": [str(a).strip().lower() for a in raw_def.get("blocked_arches", [])],
                "nto1": bool(raw_def.get("nto1", False)),
            }
            if self.is_source_family(isa):
                self._validate_single_swept_intrinsic(primitive_name, file_path, isa, seed["direct"])
            else:
                self._validate_signature_keys(primitive_name, expected_signature_keys, raw_def["signature"])
                seed["signature"] = dict(raw_def["signature"])

            for item in self._definition_expander.expand(seed, request):
                # The dtype is only an expansion axis; the signature carries the types.
                dtype = item["dtype"]
                if dtype is not None and dtype not in CANONICAL_DTYPES:
                    raise ValueError(f"{file_path}: invalid dtype '{dtype}' in primitive '{primitive_name}'")
                primitive.definitions.append(Definition(
                    isa=item["isa"],
                    signature=item["signature"],
                    direct=item["direct"],
                    include=item.get("include"),
                    primitive=primitive,
                    blocked_arches=list(item.get("blocked_arches", [])),
                    nto1=bool(item.get("nto1", False)),
                ))

        self.mapping[primitive.name] = primitive

    def _io_shape(self, raw_defs: list[dict[str, Any]]) -> dict[str, set[str]] | None:
        """Each parameter's shape classes across the target signatures, the
        reference target's if it has any.  A parameter may allow several: `add`
        takes a vector where ARM's `svadd_n` takes a broadcast scalar."""
        targets = [d for d in raw_defs if d.get("signature") and not self.is_source_family(_isa(d))]
        candidates = [d for d in targets if _isa(d) == _REFERENCE_TARGET] or targets
        if not candidates:
            return None
        shape: dict[str, set[str]] = {}
        for raw_def in candidates:
            for param, token in raw_def["signature"].items():
                cls = classify_canonical_token(str(token))
                if cls is None:
                    # A concrete spelling: no check rather than a misclassified one.
                    return None
                shape.setdefault(param, set()).add(cls)
        return shape

    def definitions_for_primitive(self, isa: str, prim_name: str) -> list[Definition]:
        """Return all target definitions for one primitive/isa."""
        if not (primitive := self.mapping.get(prim_name)):
            return []
        return [definition for definition in primitive.definitions if definition.isa == isa]

    def primitive_of(self, prim_name: str) -> Optional[Primitive]:
        return self.mapping.get(prim_name)


_cached_primitive_registry: PrimitiveRegistry | None = None


def get_primitive_registry() -> PrimitiveRegistry:
    global _cached_primitive_registry
    if _cached_primitive_registry is None:
        _cached_primitive_registry = PrimitiveRegistry(prim_dir=get_language().primitives_dir)
    return _cached_primitive_registry


def _reset_cache() -> None:
    global _cached_primitive_registry
    _cached_primitive_registry = None


register_reset_hook(_reset_cache)
