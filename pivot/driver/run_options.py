"""The options of a run and the ones active in this process.  A registry cached
against the active options registers a reset hook, run when others activate."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

from pivot.driver.config import get_config
from pivot.lang.manifests import language_names

# TSL data-parallel policies; `generic` compiles on any profile.
POLICY_FIXED = "fixed"
POLICY_GENERIC = "generic"
POLICY_CLANG_FIXED = "clang_fixed"
TSL_POLICIES = frozenset({POLICY_GENERIC, POLICY_FIXED, POLICY_CLANG_FIXED})

# TSL width models: `vla` re-widths every vector by one `PIVOT_TSL_VLEN_BITS`
# knob (needs one source register width); `samewidth` keeps each op's lane count.
WIDTH_MODE_VLA = "vla"
WIDTH_MODE_SAMEWIDTH = "samewidth"
TSL_WIDTH_MODES = frozenset({WIDTH_MODE_VLA, WIDTH_MODE_SAMEWIDTH})

# Names of each language's built-in primitive set (its manifest's flat dirs).
_BUILTIN_PRIMITIVE_SETS = frozenset({"default", "handwritten", "pivot"})


@dataclass(frozen=True)
class RunOptions:
    """What a run translates with, besides the source files."""

    language: str
    source_isa: str
    target_isa: str
    primitive_set: str | None  # None: the language's built-in set
    tsl_policy: str
    tsl_width_mode: str
    sve_assumed_bits: int
    arch: str | None  # excludes the patterns that block it
    nto1: bool  # False excludes every n:1 pattern

    @classmethod
    def build(cls, language: str, source_isa: str, target_isa: str, **overrides: Any) -> RunOptions:
        """The options a run config's keys set over the pivot_config defaults."""
        if unknown := sorted(set(overrides) - _OVERRIDE_KEYS):
            raise ValueError(f"Unknown run-config key(s): {unknown}.")
        if language not in language_names():
            raise ValueError(f"Unknown language '{language}'. Configured languages: {sorted(language_names())}.")
        config = get_config()
        translation = config.get("translation") or {}

        def setting(key: str, section: dict[str, Any], default: Any = None) -> Any:
            value = overrides.get(key)
            return section.get(key, default) if value in (None, "") else value

        primitive_set = _text(setting("primitive_set", config))
        options = cls(
            language=language,
            source_isa=source_isa,
            target_isa=target_isa,
            primitive_set=None if primitive_set in _BUILTIN_PRIMITIVE_SETS else primitive_set or None,
            tsl_policy=_text(setting("tsl_policy", config)) or POLICY_GENERIC,
            tsl_width_mode=_text(setting("tsl_width_mode", config)) or WIDTH_MODE_VLA,
            sve_assumed_bits=int(setting("sve_assumed_bits", translation, 128)),
            arch=_text(setting("arch", translation)).lower() or None,
            nto1=bool(setting("nto1", translation, True)),
        )
        if options.tsl_policy not in TSL_POLICIES:
            raise ValueError(f"Unknown tsl_policy '{options.tsl_policy}'. Valid: {sorted(TSL_POLICIES)}.")
        if options.tsl_width_mode not in TSL_WIDTH_MODES:
            raise ValueError(f"Unknown tsl_width_mode '{options.tsl_width_mode}'. Valid: {sorted(TSL_WIDTH_MODES)}.")
        return options


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


# The run-config keys that override a pivot_config default.
_OVERRIDE_KEYS = frozenset(f.name for f in fields(RunOptions)) - {"language", "source_isa", "target_isa"}

_active: RunOptions | None = None
_reset_hooks: list[Callable[[], None]] = []


def register_reset_hook(hook: Callable[[], None]) -> None:
    """Call ``hook`` whenever other options activate (a registry drops its cache)."""
    _reset_hooks.append(hook)


def activate(options: RunOptions) -> None:
    """Make ``options`` the active ones; a no-op if they already are."""
    global _active
    if options == _active:
        return
    _active = options
    for hook in _reset_hooks:
        hook()


def active_options() -> RunOptions:
    """The active options, or raise if none were activated."""
    if _active is None:
        raise RuntimeError("No run options are active: call activate(RunOptions.build(...)) first.")
    return _active
