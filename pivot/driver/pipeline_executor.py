"""The language's pass schedule on one working copy: preprocessing rewrites it,
translation re-parses and rewrites it, nesting inlines the temporaries again."""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pivot.backend.backend_companion_generator import BackendCompanionGenerator
from pivot.driver.run_options import RunOptions, activate
from pivot.ir.edits import EditBuffer, apply_changes
from pivot.ir.graph_loader import GraphLoader
from pivot.isa.primitive_registry import get_primitive_registry
from pivot.lang.backends import build_emitter_registry
from pivot.lang.languages import get_language
from pivot.passes.preprocessing import nesting
from pivot.passes.preprocessing.hoist import HOIST_PASSES
from pivot.passes.translation.translate import translate
from pivot.utils.util import norm

TRANSLATION = "translation"
NESTING = "nesting"


@dataclass(frozen=True)
class PipelineJob:
    """A working copy of the user's file ``origin``, translated in place and
    copied into ``artifact_dir`` after every stage."""

    source: str
    origin: str
    artifact_dir: str
    options: RunOptions


@dataclass(frozen=True)
class PipelineStats:
    """The intrinsic calls of one file, found and translated, by name."""

    found: Counter[str]
    translated: Counter[str]


def run_pipeline(job: PipelineJob) -> PipelineStats:
    """Run the active language's passes on ``job.source``, a snapshot after each."""
    activate(job.options)
    language = get_language()
    source = norm(job.source)
    artifact_dir = Path(norm(job.artifact_dir))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(stage: str) -> None:
        shutil.copy2(source, artifact_dir / f"{stage}{Path(source).suffix.lower()}")

    stage = 1
    for step in language.passes:
        if (hoist := HOIST_PASSES.get(step)) is not None:
            print(f"[{step}(neutral)] Stage {stage}: file={source}")
            hoist.run(source)
            snapshot(hoist.snapshot)
            stage += 1

    unit = language.load_frontend().parse(source)
    found = Counter(call.name for call in unit.call_statements if call.is_intrinsic)
    translated: Counter[str] = Counter()
    writes_cxx = False
    if TRANSLATION in language.passes:
        print(f"[TranslationPass(neutral)] Stage {stage}: file={source}")
        target = job.options.target_isa
        if (backend := build_emitter_registry().get(target)) is None:
            raise ValueError(f"No backend emitter registered for target '{target}'")
        translation = translate(
            unit, job.options, backend=backend, graph_loader=GraphLoader(), artifact_dir=str(artifact_dir)
        )
        with open(source, encoding="utf-8") as f:
            edit_buffer = EditBuffer(f.read())
        apply_changes(edit_buffer, translation.changes)
        with open(source, "w", encoding="utf-8") as f:
            f.write(_with_file_prologue(edit_buffer.apply(), backend.target.file_prologue))
        if translation.used_definitions:
            BackendCompanionGenerator().generate(
                target=target,
                output_dir=str(artifact_dir),
                prim_reg=get_primitive_registry(),
                used_definitions=translation.used_definitions,
            )
        translated = Counter(translation.translated_names)
        writes_cxx = backend.target.writes_cxx
        snapshot("04_translated")

    if NESTING in language.passes:
        print(f"[nesting(neutral)] Stage {stage + 1}: file={source}")
        nesting.run(source, original=job.origin, cxx=writes_cxx)
        snapshot("05_nested")
    return PipelineStats(found=found, translated=translated)


def _with_file_prologue(text: str, prologue: tuple[str, ...]) -> str:
    """``text`` with the backend's prologue lines first, each once.  A text
    prepend, not an edit at offset 0: the source's import block there is often
    removed, and a prologue line (a Rust crate-root attribute) must come first."""
    missing = [line for line in prologue if line not in text]
    return "".join(f"{line}\n" for line in missing) + text
