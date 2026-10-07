"""A project's worklist: the files of its language whose text looks like it uses
SIMD intrinsics.  Each is translated on its own, since tree-sitter parses files
standalone."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pivot.driver.run_options import RunOptions
from pivot.lang.languages import get_language
from pivot.lang.manifests import detect_language_name_for_paths, known_extensions
from pivot.utils.util import norm

_X86_INTRINSIC_PATTERNS = [
    re.compile(r"\b_mm(256|512)?_[A-Za-z0-9_]+\b"),
]
_ARM_INTRINSIC_PATTERNS = [
    re.compile(r"\bv[a-z0-9_]*q?_[A-Za-z0-9_]+\b"),
    re.compile(r"\bsv[a-z0-9_]+\b"),
]
_PATTERNS_BY_SOURCE_ISA = {"x86": _X86_INTRINSIC_PATTERNS, "arm": _ARM_INTRINSIC_PATTERNS}


@dataclass(frozen=True)
class Worklist:
    """The candidate files of one project run, split into sources and headers."""

    project_root: str
    options: RunOptions
    sources: list[str]
    headers: list[str]

    @property
    def files(self) -> list[str]:
        return sorted(self.sources + self.headers)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


def project_language(project_root: str) -> str:
    """The one language of the project's source files (mixed projects are rejected)."""
    return detect_language_name_for_paths(_project_files(norm(project_root), known_extensions()))


def build_worklist(project_root: str, options: RunOptions) -> Worklist:
    """The files under ``project_root`` of the options' language that look like
    they call ``options.source_isa`` intrinsics.  A false positive only costs a parse."""
    project_root = norm(project_root)
    language = get_language(options.language)
    patterns = _PATTERNS_BY_SOURCE_ISA.get(options.source_isa.lower(), _X86_INTRINSIC_PATTERNS + _ARM_INTRINSIC_PATTERNS)
    candidates = [path for path in _project_files(project_root, language.extensions) if _mentions(path, patterns)]
    headers = [path for path in candidates if Path(path).suffix.lower() in language.header_extensions]
    sources = [path for path in candidates if path not in headers]
    return Worklist(project_root=project_root, options=options, sources=sources, headers=headers)


def _project_files(project_root: str, extensions: tuple[str, ...]) -> list[str]:
    """The files under ``project_root`` with one of ``extensions``, sorted."""
    return sorted({
        norm(os.path.join(root, filename))
        for root, _, filenames in os.walk(project_root)
        for filename in filenames
        if Path(filename).suffix.lower() in extensions
    })


def _mentions(path: str, patterns: list[re.Pattern[str]]) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return False
    return any(pattern.search(text) for pattern in patterns)
