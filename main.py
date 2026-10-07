"""Run the translation job a YAML run config describes: ``main.py <run_config.yaml>``.

Modes (each also takes ``output_root`` and ``name``, the label of the run's
``logs/<name>_<stamp>/`` folder):
- 'single_file': source_path; optional output_dir, output_name.
- 'project_translate': project_root.
- 'project_debugger': project_root, debug_target.

Every mode takes source_isa ("x86" or "arm") and target_isa, a label of the file
language's `target_backends` ("clang_builtins", "tsl" or "core_simd"), and may set
the run options primitive_set, tsl_policy, tsl_width_mode, sve_assumed_bits, arch
and nto1 (see pivot/driver/run_options.py).
"""

from __future__ import annotations

import sys
from dataclasses import MISSING, fields
from typing import Any

import yaml

from pivot.driver.run_options import RunOptions
from pivot.driver.runner import ProjectDebugger, ProjectTranslation, SingleFile, run

_MODES = {"single_file": SingleFile, "project_translate": ProjectTranslation, "project_debugger": ProjectDebugger}
_IGNORED_KEYS = {"args"}  # compiler flags of older run configs


def run_config(config: dict[str, Any]) -> str:
    """Run the job ``config`` describes; the path of its output."""
    config = {key: value for key, value in config.items() if key not in _IGNORED_KEYS}
    mode_name = config.pop("mode", None)
    if mode_name not in _MODES:
        raise ValueError(f"Run config 'mode' must be one of {sorted(_MODES)}, not {mode_name!r}.")
    mode_type = _MODES[mode_name]
    mode_fields = fields(mode_type)
    required = {"source_isa", "target_isa"} | {spec.name for spec in mode_fields if spec.default is MISSING}
    if missing := sorted(required - config.keys()):
        raise ValueError(f"Run config of mode '{mode_name}' lacks {missing}.")
    mode = mode_type(**{spec.name: config.pop(spec.name) for spec in mode_fields if spec.name in config})
    output_root, name = config.pop("output_root", None), config.pop("name", None)
    options = RunOptions.build(mode.language(), config.pop("source_isa"), config.pop("target_isa"), **config)
    return run(mode, options, output_root=output_root, name=name)


def run_single_file_pipeline(source_path: str, source_isa: str, target_isa: str, **config: Any) -> str:
    """Translate one file; ``config`` takes the other single_file run-config keys.
    The path of the translated file."""
    return run_config(
        {"mode": "single_file", "source_path": source_path, "source_isa": source_isa, "target_isa": target_isa, **config}
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: main.py <run_config.yaml>")
    print(f"Loading run configuration from: {sys.argv[1]}")
    with open(sys.argv[1], encoding="utf-8") as f:
        run_config(yaml.safe_load(f) or {})


if __name__ == "__main__":
    main()
