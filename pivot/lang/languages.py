"""A language's manifest resolved into a :class:`Language`: the implementations it
names (frontend, syntax, locator, backends) and the resource dirs of the run's
primitive set.  Only the engine code is shared between languages."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from pivot.driver.run_options import active_options
from pivot.lang.manifests import file_language_name, language_manifests


@dataclass(frozen=True)
class Language:
    """Resource bindings for one supported source language."""

    name: str
    # Importable dotted path to the Frontend class (parses source -> ParsedUnit).
    frontend: str
    # Importable dotted path to the source LanguageSyntax class (spells temps).
    syntax: str
    # Importable dotted path to the structural preprocessing locator class.
    locator: str
    # Source-file extensions owned by this language (used to detect a file's
    # language and to drive frontend/locator/output selection off the filename).
    extensions: tuple[str, ...]
    # Subset of `extensions` that are headers rather than compilable sources
    # (a C/C++ concept; empty for languages without headers, e.g. Rust).
    header_extensions: tuple[str, ...]
    # Directory scanned for the intrinsic vocabulary (dtype_mapping + catalogs).
    intrinsics_dir: str
    # Directory scanned for primitive definitions (patterns + replacements).
    primitives_dir: str
    # Directory GraphLoader reads generated pattern graph YAMLs from.
    graphs_yaml_dir: str
    # Directory the pattern-graph generator writes artifacts under.
    graphs_base_dir: str
    # The list of pipeline passes to run for this language.
    passes: list[str]
    # Dotted paths keyed by bare isa label: a source family (x86/arm) to its
    # expander, a target (clang_builtins/tsl/core_simd) to its BackendEmitter class.
    source_backends: dict[str, str]
    target_backends: dict[str, str]

    def load_source_expanders(self) -> dict[str, Any]:
        """This language's source-family expanders, keyed by isa label."""
        return {label: _load_symbol(dotted) for label, dotted in self.source_backends.items()}

    def load_target_emitters(self) -> dict[str, Any]:
        """Instantiate this language's target emitters, keyed by isa label."""
        return {label: _load_symbol(dotted)() for label, dotted in self.target_backends.items()}

    def load_frontend(self, **kwargs: Any) -> Any:
        """Instantiate this language's frontend from its declared class path."""
        return _load_symbol(self.frontend)(**kwargs)

    def load_syntax(self) -> Any:
        """Instantiate this language's source-syntax object."""
        return _load_symbol(self.syntax)()

    def load_locator(self, path: str, cxx: bool = False) -> Any:
        """Instantiate this language's locator for ``path``, whose code is C++ if ``cxx``."""
        return _load_symbol(self.locator)(path, cxx)


def _load_symbol(dotted: str) -> Any:
    """Import and return the object named by a dotted path (``pkg.mod.Attr``)."""
    module_path, _, attr = dotted.rpartition(".")
    if not module_path or not attr:
        raise ValueError(f"Invalid dotted class path: '{dotted}'.")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _require(value: str | None, key: str, name: str) -> str:
    if not value:
        raise ValueError(f"Language '{name}' manifest is missing '{key}'.")
    return value


def _resolve_primitive_set(
    manifest: dict[str, Any], language: str
) -> tuple[str, str, str]:
    """Resolve ``(primitives_dir, graphs_yaml_dir, graphs_base_dir)`` for the
    run's primitive set.  The manifest's flat fields are the implicit ``default``
    set; a named set is an entry of the manifest's ``primitive_sets`` map."""
    flat = (
        _require(manifest.get("primitives_dir"), "primitives_dir", language),
        _require(manifest.get("graphs_yaml_dir"), "graphs_yaml_dir", language),
        _require(manifest.get("graphs_base_dir"), "graphs_base_dir", language),
    )
    selected = active_options().primitive_set
    if selected is None:
        return flat

    sets = manifest.get("primitive_sets") or {}
    entry = sets.get(selected)
    if entry is None:
        raise ValueError(
            f"Unknown primitive_set '{selected}' for language '{language}'. "
            f"Configured sets: {sorted(sets.keys())} (omit or use 'default' for "
            "the language's built-in primitives)."
        )
    scope = f"primitive_sets.{selected}"
    return (
        _require(entry.get("primitives_dir"), f"{scope}.primitives_dir", language),
        _require(entry.get("graphs_yaml_dir"), f"{scope}.graphs_yaml_dir", language),
        _require(entry.get("graphs_base_dir"), f"{scope}.graphs_base_dir", language),
    )


def get_language(name: str | None = None) -> Language:
    """The resource manifest of ``name``, by default the active language."""
    resolved = name or active_options().language
    languages = language_manifests()
    manifest = languages.get(resolved)
    if manifest is None:
        raise ValueError(
            f"Unknown language '{resolved}'. Configured languages: "
            f"{sorted(languages.keys())}."
        )

    primitives_dir, graphs_yaml_dir, graphs_base_dir = _resolve_primitive_set(
        manifest, resolved
    )

    return Language(
        name=resolved,
        frontend=_require(manifest.get("frontend"), "frontend", resolved),
        syntax=_require(manifest.get("syntax"), "syntax", resolved),
        locator=_require(manifest.get("locator"), "locator", resolved),
        extensions=tuple(e.lower() for e in (manifest.get("extensions") or [])),
        header_extensions=tuple(e.lower() for e in (manifest.get("header_extensions") or [])),
        intrinsics_dir=_require(manifest.get("intrinsics_dir"), "intrinsics_dir", resolved),
        primitives_dir=primitives_dir,
        graphs_yaml_dir=graphs_yaml_dir,
        graphs_base_dir=graphs_base_dir,
        passes=manifest.get("passes", []),
        source_backends=dict(manifest.get("source_backends") or {}),
        target_backends=dict(manifest.get("target_backends") or {}),
    )


def get_language_for_file(path: str) -> Language:
    """Resolve the full manifest for a file from its extension."""
    return get_language(file_language_name(path))
