"""Per-file load cache for a directory of corpus files.

A translation corpus (pattern graphs, intrinsic definitions) is many files
that change rarely and independently.  This caches each file's parsed
contribution keyed on the file's ``(mtime, size)``, so a load reparses only the
files that changed and reuses the rest from a single pickle.  The caller owns
what parsing one file yields and how to assemble the contributions; this owns the
directory walk, change detection, and the versioned, atomically written cache.
"""
from __future__ import annotations

import os
import pickle
import tempfile
from typing import Callable


def _read_cache(cache_path: str, version: str) -> dict:
    """The stored ``{relpath: {mtime_ns, size, parsed}}`` map, or empty on a
    missing / version-mismatched / unreadable cache."""
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("version") != version:
            return {}
        return payload.get("files", {})
    except Exception:
        return {}


def _write_cache(cache_path: str, version: str, files: dict) -> None:
    """Atomically write the cache (temp file + replace); best-effort: a failure
    just means the next load reparses."""
    try:
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(cache_path), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                pickle.dump({"version": version, "files": files}, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception:
        pass


def load_corpus_per_file(
    corpus_dir: str,
    *,
    cache_filename: str,
    version: str,
    parse_file: Callable[[str, str], object],
    accept: Callable[[str], bool] = lambda name: name.endswith(".yaml"),
    sort_files: bool = False,
    enabled: bool = True,
) -> list[tuple[str, object]]:
    """Return ``(relpath, parsed)`` for every accepted file under ``corpus_dir``,
    in walk order, reparsing only files whose ``(mtime, size)`` changed.

    ``parse_file(path, name)`` yields one file's parsed contribution and must be
    picklable.  ``accept`` selects files by name; ``sort_files`` sorts filenames
    within each directory (callers whose assembly is order-sensitive choose the
    order that reproduces their non-cached result).  ``enabled=False`` parses
    everything and neither reads nor writes the cache.  The cache is rewritten
    whenever the file set or any file's stamp changed.
    """
    cache_path = os.path.join(corpus_dir, cache_filename)
    cached = _read_cache(cache_path, version) if enabled else {}

    current: dict[str, dict] = {}
    ordered: list[tuple[str, object]] = []
    dirty = False
    for root, _, files in os.walk(corpus_dir):
        for name in (sorted(files) if sort_files else files):
            if not accept(name):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, corpus_dir)
            try:
                st = os.stat(path)
            except OSError:
                continue
            entry = cached.get(rel)
            if entry is not None and entry["mtime_ns"] == st.st_mtime_ns and entry["size"] == st.st_size:
                current[rel] = entry
            else:
                current[rel] = {"mtime_ns": st.st_mtime_ns, "size": st.st_size,
                                "parsed": parse_file(path, name)}
                dirty = True
            ordered.append((rel, current[rel]["parsed"]))

    # A file removed since the last write leaves the set smaller than the cache.
    if not dirty and len(cached) != len(current):
        dirty = True
    if enabled and dirty:
        _write_cache(cache_path, version, current)
    return ordered
