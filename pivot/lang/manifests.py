"""The language manifests (``resources/languages/*/config.yaml``) and the
extension map they declare: a source file's language follows from its
extension alone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_manifests_cache: dict[str, dict[str, Any]] | None = None


def language_manifests() -> dict[str, dict[str, Any]]:
    """Every language manifest by name, discovered once."""
    global _manifests_cache
    if _manifests_cache is not None:
        return _manifests_cache

    configs = {}
    lang_dir = Path("resources/languages")
    if not lang_dir.exists():
        return configs

    for config_path in lang_dir.glob("*/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
            if manifest and "name" in manifest:
                configs[manifest["name"]] = manifest

    _manifests_cache = configs
    return configs


def language_names() -> list[str]:
    return list(language_manifests().keys())


def _extension_index() -> dict[str, str]:
    """Map each declared file extension to its owning language name.

    Raises if two languages claim the same extension."""
    index: dict[str, str] = {}
    for name, manifest in language_manifests().items():
        for ext in manifest.get("extensions") or []:
            ext = ext.lower()
            owner = index.get(ext)
            if owner is not None and owner != name:
                raise ValueError(
                    f"Extension '{ext}' is claimed by both '{owner}' and '{name}'. "
                    "Each extension must map to exactly one language."
                )
            index[ext] = name
    return index


def known_extensions() -> tuple[str, ...]:
    """Every source-file extension across all configured languages (for scans)."""
    return tuple(sorted(_extension_index().keys()))


def language_name_for_file(path: str) -> str | None:
    """Owning language name for a file by extension, or None if unrecognized."""
    return _extension_index().get(Path(path).suffix.lower())


def file_language_name(path: str) -> str:
    """The language owning ``path``'s extension; raises if none does."""
    if (name := language_name_for_file(path)) is None:
        raise ValueError(
            f"No language registered for the extension of: {path} "
            f"(known extensions: {list(known_extensions())})."
        )
    return name


def detect_language_name_for_paths(paths: list[str]) -> str:
    """Single language shared by all recognized files in ``paths``.

    Ignores files with unrecognized extensions.  Raises if none are recognized
    or if more than one language is present."""
    found: dict[str, int] = {}
    for path in paths:
        name = language_name_for_file(path)
        if name is not None:
            found[name] = found.get(name, 0) + 1
    if not found:
        raise ValueError(
            "No files with a recognized source-language extension were found "
            f"(known extensions: {list(known_extensions())})."
        )
    if len(found) > 1:
        detail = ", ".join(f"{k} ({v} file(s))" for k, v in sorted(found.items()))
        raise ValueError(
            f"Mixed source languages detected: {detail}. A single run must be "
            "one language."
        )
    return next(iter(found))
