import hashlib
import json
import os
import shutil
from typing import Iterator

import yaml

from pivot.driver.config import get_config
from pivot.frontend.primitive_frontend import PrimitiveFrontend
from pivot.lang.languages import get_language
from pivot.ir.graph import Graph
from pivot.isa.intrinsic_registry import get_intrinsic_registry
from pivot.isa.primitive_registry import get_primitive_registry, Definition


class PatternGraphGenerator:
    """Builds the pattern graph YAMLs from the primitive definitions, regenerating
    only the combos whose definition or intrinsic IR changed."""

    def __init__(self) -> None:
        self._prim_reg = get_primitive_registry()
        self._intr_reg = get_intrinsic_registry()

    @staticmethod
    def _signature_shape_key(signature: dict[str, str]) -> str:
        encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:10]

    def _iter_graphifiable_combos(self) -> Iterator[tuple[str, str, str, list[Definition]]]:
        for primitive_name, primitive in self._prim_reg.mapping.items():
            combos: dict[tuple[str, str, str], list[Definition]] = {}
            for definition in primitive.definitions:
                if not self._prim_reg.is_source_family(definition.isa):
                    continue
                signature_key = self._signature_shape_key(definition.signature)
                combo_key = (primitive_name, definition.isa, signature_key)
                combos.setdefault(combo_key, []).append(definition)

            for (name, isa, signature_key), definitions in combos.items():
                yield name, isa, signature_key, definitions

    @staticmethod
    def _combo_paths(
        base_dir: str,
        prim_name: str,
        isa: str,
        signature_key: str,
    ) -> tuple[str, str, str]:
        yaml_dir = os.path.join(base_dir, "yaml", isa, prim_name)
        yaml_file = os.path.join(yaml_dir, f"{prim_name}_{signature_key}.yaml")
        cache_key = f"{prim_name}:{isa}:{signature_key}"
        return yaml_dir, yaml_file, cache_key

    @staticmethod
    def _write_graph_payload_docs(yaml_file: str, payloads: list[dict]) -> None:
        """Write one multi-document YAML file, one document per variant."""
        with open(yaml_file, "w") as f:
            for idx, payload in enumerate(payloads):
                if idx == 0:
                    yaml.dump(payload, f)
                else:
                    yaml.dump(payload, f, explicit_start=True)

    def _create_graph_from_variant(self, prim_name: str, isa: str, source_file: str) -> Graph:
        """Parse one variant source file and construct its pattern graph."""
        unit = PrimitiveFrontend().parse(source_file)

        # Fold pseudo-intrinsics (immediate constructors) are inlined, so they
        # need no `ir:` signature; every other intrinsic's operands are typed by it.
        untyped = sorted({
            call.name
            for call in unit.call_statements
            if self._intr_reg.is_known_intrinsic(call.name)
            and not self._intr_reg.is_fold_intrinsic(call.name)
            and self._intr_reg.ir_signature_of(call.name) is None
        })
        if untyped:
            raise ValueError(
                "pattern intrinsics lack an `ir:` signature in their intrinsic "
                f"YAML (re-run scripts/annotate_ir/annotate.py): {untyped}"
            )
        return Graph(matches=unit.call_statements, primitive_name=prim_name, isa=isa)

    def _variant_payload(self, prim_name: str, variant_idx: int, definition: Definition, variant_dir: str) -> dict:
        """Generate one variant's graph and return its YAML document."""
        if os.path.exists(variant_dir):
            shutil.rmtree(variant_dir)
        os.makedirs(variant_dir, exist_ok=True)
        # Pattern bodies are PIVOT's own pseudo-C, parsed by one fixed grammar
        # whatever language is being translated, so the extension is not the
        # active language's.
        source_file = os.path.join(variant_dir, "source.pseudo")
        with open(source_file, "w") as file:
            file.write("\n".join(definition.direct))

        graph = self._create_graph_from_variant(prim_name, definition.isa, source_file)
        if not graph.nodes:
            raise ValueError(
                f"Generated empty graph for {prim_name} ({definition.isa}) variant #{variant_idx}"
            )
        payload = {
            "primitive_name": graph.primitive_name,
            "isa": graph.isa,
            "nodes": {node_id: node.to_yaml_dict() for node_id, node in graph.nodes.items()},
        }
        if definition.blocked_arches:
            payload["blocked_arches"] = list(definition.blocked_arches)
        if definition.nto1:
            payload["nto1"] = True
        return payload

    def generate_pattern_graph_artifacts(self) -> None:
        """Regenerate every stale combo's graph YAML (one file per primitive, ISA
        and signature), and prune the files and hash entries of removed combos."""
        cfg = get_config()
        base_dir: str = get_language().graphs_base_dir
        graph_debug_root = os.path.join(cfg["paths"]["logs_dir"], ".graphs")
        hash_path: str = os.path.join(base_dir, "primitives.hash")
        hash_cache: dict[str, str] = self._load_hash_cache(hash_path)
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(graph_debug_root, exist_ok=True)

        combo_specs = list(self._iter_graphifiable_combos())
        active_combo_keys: set[str] = set()
        expected_yaml_files: set[str] = set()
        for primitive_name, isa, signature_key, _ in combo_specs:
            _, yaml_file, cache_key = self._combo_paths(base_dir, primitive_name, isa, signature_key)
            active_combo_keys.add(cache_key)
            expected_yaml_files.add(yaml_file)

        pruned_hash_entries = self._prune_stale_hash_entries(hash_cache, active_combo_keys)
        pruned_graph_files = self._prune_stale_graph_yaml_files(os.path.join(base_dir, "yaml"), expected_yaml_files)
        if pruned_hash_entries or pruned_graph_files:
            print(
                f"Pruned stale graph artifacts: hash_entries={pruned_hash_entries}, "
                f"graph_yaml_files={pruned_graph_files}"
            )

        failed_combos: list[tuple[str, list[str]]] = []
        for primitive_name, isa, signature_key, definitions in combo_specs:
            yaml_dir, yaml_file, cache_key = self._combo_paths(base_dir, primitive_name, isa, signature_key)
            current_hash = self._compute_combo_hash(primitive_name, definitions)
            if os.path.exists(yaml_file) and self._combo_cache_is_current(hash_cache.get(cache_key), current_hash):
                continue
            if os.path.exists(yaml_file):
                os.remove(yaml_file)
            os.makedirs(yaml_dir, exist_ok=True)

            payloads: list[dict] = []
            failures: list[str] = []
            for variant_idx, definition in enumerate(definitions):
                print(f"Generating {primitive_name} ({definition.isa}) variant #{variant_idx}")
                variant_label = f"{primitive_name}_{self._signature_shape_key(definition.signature)}_{variant_idx}"
                variant_dir = os.path.join(graph_debug_root, definition.isa, primitive_name, variant_label)
                try:
                    payloads.append(self._variant_payload(primitive_name, variant_idx, definition, variant_dir))
                except Exception as exc:
                    failures.append(f"variant #{variant_idx}: {type(exc).__name__}: {exc}")

            if failures:
                hash_cache.pop(cache_key, None)
                failed_combos.append((cache_key, failures))
                continue
            self._write_graph_payload_docs(yaml_file, payloads)
            names = self._intrinsic_names_in_payloads(payloads)
            registry_fp = self._registry_ir_fingerprint(names)
            hash_cache[cache_key] = f"{current_hash}:{registry_fp}:{','.join(names)}"
            # Only after the YAML is on disk, so an interrupted run regenerates it.
            self._write_hash_cache(hash_path, hash_cache)

        self._write_hash_cache(hash_path, hash_cache)
        if failed_combos:
            print(f"Skipped {len(failed_combos)} combos due to graph generation failures:")
            for cache_key, reasons in failed_combos:
                print(f"  - {cache_key}")
                for reason in reasons:
                    print(f"      {reason}")

    @staticmethod
    def _compute_combo_hash(prim_name: str, definitions: list[Definition]) -> str:
        payload = {
            "primitive": prim_name,
            "variants": [
                {
                    "isa": definition.isa,
                    "signature": definition.signature,
                    "direct": definition.direct,
                    "include": definition.include,
                    "blocked_arches": definition.blocked_arches,
                    "nto1": definition.nto1,
                }
                for definition in definitions
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    # A combo's staleness key is "{def_hash}:{registry_ir_fp}:{intrinsic_names}":
    # the definition hash, plus the IR fingerprint of the intrinsics the graph
    # references (their registry YAML is not part of the definition).

    def _registry_ir_fingerprint(self, intrinsic_names) -> str:
        """Short hash of the current registry IR signatures of these intrinsics."""
        payload = [
            [name, self._intr_reg.ir_signature_of(name)]
            for name in sorted(set(intrinsic_names))
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @staticmethod
    def _intrinsic_names_in_payloads(payloads: list[dict]) -> list[str]:
        """Every intrinsic-node name across a combo's variant graph payloads."""
        names: set[str] = set()
        for payload in payloads:
            for node in (payload.get("nodes") or {}).values():
                if node.get("kind") == "intrinsic" and node.get("name"):
                    names.add(node["name"])
        return sorted(names)

    def _combo_cache_is_current(self, cached: str | None, def_hash: str) -> bool:
        """True when the cached staleness key matches the current definition and the
        current registry IR of the intrinsics the graph references."""
        if not cached:
            return False
        parts = cached.split(":", 2)
        if len(parts) != 3:
            return False
        cached_def, cached_registry_fp, cached_names = parts
        if cached_def != def_hash:
            return False
        names = cached_names.split(",") if cached_names else []
        return self._registry_ir_fingerprint(names) == cached_registry_fp

    @staticmethod
    def _load_hash_cache(hash_path: str) -> dict[str, str]:
        if not os.path.exists(hash_path):
            return {}
        with open(hash_path, "r") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError(f"Hash cache at {hash_path} must be a JSON object")
        return {str(key): str(value) for key, value in data.items()}

    @staticmethod
    def _write_hash_cache(hash_path: str, hash_cache: dict[str, str]) -> None:
        # Atomic, so a reader never sees a truncated file.
        os.makedirs(os.path.dirname(hash_path), exist_ok=True)
        tmp_path = f"{hash_path}.{os.getpid()}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(hash_cache, f, sort_keys=True, indent=2)
        os.replace(tmp_path, hash_path)

    @staticmethod
    def _prune_stale_hash_entries(hash_cache: dict[str, str], active_combo_keys: set[str]) -> int:
        stale_keys = [key for key in hash_cache if key not in active_combo_keys]
        for key in stale_keys:
            hash_cache.pop(key, None)
        return len(stale_keys)

    @staticmethod
    def _prune_stale_graph_yaml_files(pattern_graphs_yaml_dir: str, expected_yaml_files: set[str]) -> int:
        if not os.path.isdir(pattern_graphs_yaml_dir):
            return 0

        expected = {os.path.normpath(path) for path in expected_yaml_files}
        existing: set[str] = set()
        for root, _, files in os.walk(pattern_graphs_yaml_dir):
            for file in files:
                if file.endswith(".yaml"):
                    existing.add(os.path.normpath(os.path.join(root, file)))

        stale_files = sorted(existing - expected)
        for stale_file in stale_files:
            os.remove(stale_file)

        # Remove empty directories left after orphan graph removal.
        for root, dirnames, filenames in os.walk(pattern_graphs_yaml_dir, topdown=False):
            if dirnames or filenames:
                continue
            os.rmdir(root)

        return len(stale_files)
