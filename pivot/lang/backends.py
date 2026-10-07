"""The active language's backend handlers: a target emitter per target label and
a definition expander per source family, loaded from the manifest's
`target_backends` / `source_backends` by bare label (`tsl`, `x86`), so a label
may exist in one language only or bind to a different handler in each."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from pivot.driver.run_options import register_reset_hook
from pivot.lang.languages import get_language

if TYPE_CHECKING:
    from pivot.backend.backend_emitter import BackendEmitter

# Instances are cached per language and dropped when other run options activate.
_emitter_cache: dict[str, dict[str, "BackendEmitter"]] = {}
_source_expander_cache: dict[str, dict[str, Any]] = {}


def build_emitter_registry() -> dict[str, "BackendEmitter"]:
    """Map each target isa label to its emitter, for the active language."""
    lang = get_language()
    if lang.name not in _emitter_cache:
        _emitter_cache[lang.name] = lang.load_target_emitters()
    return dict(_emitter_cache[lang.name])


def source_expanders() -> dict[str, Any]:
    """Map each source-family isa label (x86/arm) to its expander, for the active
    language."""
    lang = get_language()
    if lang.name not in _source_expander_cache:
        _source_expander_cache[lang.name] = lang.load_source_expanders()
    return dict(_source_expander_cache[lang.name])


register_reset_hook(_emitter_cache.clear)
register_reset_hook(_source_expander_cache.clear)
