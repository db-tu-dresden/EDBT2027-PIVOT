"""One runner for every mode: activate the options, regenerate the pattern graphs,
translate working copies inside the run root, audit, publish, mirror to logs.  A
mode names only its jobs and what it publishes."""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pivot.driver.config import get_config
from pivot.driver.pipeline_executor import PipelineJob, PipelineStats, run_pipeline
from pivot.driver.run_options import RunOptions, activate
from pivot.driver.translation_logger import get_translation_logger
from pivot.driver.worklist import build_worklist, project_language
from pivot.ir.pattern_graph_generator import PatternGraphGenerator
from pivot.lang.backends import build_emitter_registry
from pivot.lang.languages import get_language
from pivot.lang.manifests import file_language_name
from pivot.utils.util import norm, print_audit_report


@dataclass(frozen=True)
class JobResult:
    job: PipelineJob
    stats: PipelineStats | None  # None: the job failed


class Mode(ABC):
    """What a run translates and what it publishes."""

    log_level = "summary"
    parallel = False  # jobs run in worker processes; a failing one is skipped

    @abstractmethod
    def language(self) -> str:
        """The language of the files to translate."""

    @abstractmethod
    def default_name(self) -> str:
        """The run's label when the run config names none."""

    @abstractmethod
    def jobs(self, run_root: Path, options: RunOptions) -> list[PipelineJob]:
        """Set up the working copies in ``run_root``, one job each."""

    @abstractmethod
    def publish(self, run_root: Path, results: list[JobResult]) -> str:
        """Place and report the translated files; the path of the output."""


@dataclass(frozen=True)
class SingleFile(Mode):
    """One file.  ``output_dir`` receives only the deliverable: the translated file
    (named ``output_name``, default the source's basename) and its companion."""

    source_path: str
    output_dir: str | None = None
    output_name: str | None = None

    def language(self) -> str:
        return file_language_name(self.source_path)

    def default_name(self) -> str:
        return "single_file_pivot"

    def jobs(self, run_root: Path, options: RunOptions) -> list[PipelineJob]:
        source = norm(self.source_path)
        return [_copy_job(source, run_root, Path(source).name, options)]

    def publish(self, run_root: Path, results: list[JobResult]) -> str:
        (result,) = results
        job = result.job
        output = run_root / f"translated{Path(job.origin).suffix}"
        shutil.copyfile(job.source, output)
        # The translated file binds its companion as a sibling.
        companion_name = build_emitter_registry()[job.options.target_isa].target.companion_filename
        companion: Path | None = None
        if (generated := Path(job.artifact_dir) / companion_name).exists():
            companion = Path(shutil.copy2(generated, run_root / companion_name))
        if self.output_dir:
            deliverable_dir = Path(self.output_dir)
            deliverable_dir.mkdir(parents=True, exist_ok=True)
            output = Path(shutil.copy2(output, deliverable_dir / (self.output_name or Path(job.origin).name)))
            if companion:
                companion = Path(shutil.copy2(companion, deliverable_dir / companion_name))
        print("Single-file translation complete.")
        print(f"Output: {output}")
        if companion:
            print(f"Companion: {companion}")
        print(f"Artifacts: {job.artifact_dir}")
        return str(output)


@dataclass(frozen=True)
class ProjectTranslation(Mode):
    """Every candidate file of a project, translated in a full copy of it, the
    published ``translated_project/``."""

    project_root: str
    parallel = True

    def language(self) -> str:
        return project_language(self.project_root)

    def default_name(self) -> str:
        return _project_run_name(self.project_root)

    def jobs(self, run_root: Path, options: RunOptions) -> list[PipelineJob]:
        worklist = build_worklist(self.project_root, options)
        manifest = run_root / ".pivot" / "worklist.json"
        worklist.write(manifest)
        print(f"Project worklist: {len(worklist.sources)} sources, {len(worklist.headers)} headers ({manifest})")
        debug_root = run_root / ".pivot" / "debug"
        translated_root = run_root / "translated_project"
        for stale in (debug_root, translated_root):
            shutil.rmtree(stale, ignore_errors=True)
        debug_root.mkdir(parents=True)
        shutil.copytree(worklist.project_root, translated_root)
        return [
            PipelineJob(
                source=str(copy),
                origin=path,
                artifact_dir=str(debug_root / _debug_dir_name(worklist.project_root, path)),
                options=options,
            )
            for path in worklist.files
            if (copy := translated_root / Path(path).relative_to(worklist.project_root)).exists()
        ]

    def publish(self, run_root: Path, results: list[JobResult]) -> str:
        header_extensions = get_language().header_extensions
        done = [result.job.origin for result in results if result.stats is not None]
        failed = [result.job.origin for result in results if result.stats is None]
        headers = sum(1 for path in done if Path(path).suffix.lower() in header_extensions)
        translated_root = run_root / "translated_project"
        if failed:
            print(f"[project_translate] {len(failed)} file(s) failed and were skipped:")
            for path in failed:
                print(f"  - {path}")
        lines = [
            f"worklist={run_root / '.pivot' / 'worklist.json'}",
            f"translated_sources={len(done) - headers}",
            f"processed_headers={headers}",
            f"failed_sources={len(failed)}",
            *(f"failed={path}" for path in failed),
            f"translated_project_root={translated_root}",
        ]
        (run_root / ".pivot" / "project_translation_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("Project translation complete.")
        print(f"Translated source files: {len(done) - headers}")
        print(f"Processed headers: {headers}")
        print(f"Failed source files: {len(failed)}")
        print(f"Translated project: {translated_root}")
        return str(translated_root)


@dataclass(frozen=True)
class ProjectDebugger(Mode):
    """One project file, logged at debug level."""

    project_root: str
    debug_target: str
    log_level = "debug"

    def language(self) -> str:
        return file_language_name(self.debug_target)

    def default_name(self) -> str:
        return _project_run_name(self.project_root)

    def jobs(self, run_root: Path, options: RunOptions) -> list[PipelineJob]:
        source = norm(self.debug_target)
        return [_copy_job(source, run_root, _debug_dir_name(norm(self.project_root), source), options)]

    def publish(self, run_root: Path, results: list[JobResult]) -> str:
        (result,) = results
        job, stats = result.job, result.stats
        translated, found = sum(stats.translated.values()), sum(stats.found.values())
        (Path(job.artifact_dir) / "debug_summary.txt").write_text(
            f"source={job.origin}\ntranslated={translated}\nfound={found}\nartifacts={job.artifact_dir}\n",
            encoding="utf-8",
        )
        print("Project debugger run complete.")
        print(f"Debug target: {job.origin}")
        print(f"Intrinsics translated: {translated}")
        print(f"Intrinsics found: {found}")
        print(f"Artifacts: {job.artifact_dir}")
        return job.artifact_dir


def run(mode: Mode, options: RunOptions, *, output_root: str | None = None, name: str | None = None) -> str:
    """Translate what ``mode`` names under ``options``; the path of its output."""
    print(f"Run options: {options}")
    activate(options)
    PatternGraphGenerator().generate_pattern_graph_artifacts()
    label = name or mode.default_name()
    run_root = Path(output_root).resolve() if output_root else _fresh_logs_dir(label)
    run_root.mkdir(parents=True, exist_ok=True)
    get_translation_logger().set_log_path(
        run_root / ".pivot" / "translation_log.txt", reset=True, auto_flush=True, min_level=mode.log_level
    )
    jobs = mode.jobs(run_root, options)
    results = _run_in_workers(jobs) if mode.parallel else [JobResult(job, run_pipeline(job)) for job in jobs]
    _audit(results)
    output = mode.publish(run_root, results)
    if (logs_copy := _mirror_to_logs(run_root, label)) is not None:
        print(f"Logs copy: {logs_copy}")
    return output


def _copy_job(source: str, run_root: Path, debug_name: str, options: RunOptions) -> PipelineJob:
    """A job on a copy of ``source`` in the run root's work dir, keeping its basename."""
    work_copy = run_root / ".pivot" / "work" / Path(source).name
    work_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, work_copy)
    artifact_dir = run_root / ".pivot" / "debug" / debug_name
    return PipelineJob(source=str(work_copy), origin=source, artifact_dir=str(artifact_dir), options=options)


def _run_in_workers(jobs: list[PipelineJob]) -> list[JobResult]:
    """Run ``jobs`` in worker processes; a file that fails is reported and skipped."""
    workers = min(_worker_count(), max(1, len(jobs)))
    print(f"[project_translate] workers={workers} jobs={len(jobs)}")
    results: list[JobResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_pipeline, job) for job in jobs]
        for job, future in zip(jobs, futures):
            try:
                results.append(JobResult(job, future.result()))
            except Exception as exc:  # noqa: BLE001 (one file must not abort the project)
                message = f"[project_translate] FAILED {job.origin}: {type(exc).__name__}: {exc}"
                print(message)
                get_translation_logger().log(message, level="summary")
                results.append(JobResult(job, None))
    return results


def _worker_count() -> int:
    """``PIVOT_PROJECT_TRANSLATE_WORKERS`` if a positive integer, else all cores but one."""
    value = os.getenv("PIVOT_PROJECT_TRANSLATE_WORKERS", "").strip()
    if value.isdigit() and int(value) > 0:
        return int(value)
    return max(1, (os.cpu_count() or 1) - 1)


def _audit(results: list[JobResult]) -> None:
    """Log each translated file's audit report, then the global one."""
    log = get_translation_logger().log
    found: Counter[str] = Counter()
    translated: Counter[str] = Counter()
    for result in results:
        if result.stats is not None:
            print_audit_report(result.job.origin, result.stats.found, result.stats.translated, log)
            found.update(result.stats.found)
            translated.update(result.stats.translated)
    print_audit_report("GLOBAL", found, translated, log, is_global=True)


def _debug_dir_name(project_root: str, path: str) -> str:
    """A project file's artifact dir name: its project-relative path joined by ``__``."""
    return "__".join(Path(path).relative_to(project_root).parts)


def _project_run_name(project_root: str) -> str:
    return f"{Path(norm(project_root)).name or 'project'}_pivot"


def _logs_dir() -> Path:
    return Path(get_config()["paths"]["logs_dir"]).resolve()


def _fresh_logs_dir(label: str) -> Path:
    """A new ``<logs_dir>/<label>_<stamp>`` directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path, suffix = _logs_dir() / f"{label}_{stamp}", 1
    while path.exists():
        path, suffix = _logs_dir() / f"{label}_{stamp}_{suffix}", suffix + 1
    path.mkdir(parents=True)
    return path


def _mirror_to_logs(run_root: Path, label: str) -> Path | None:
    """Copy a run root outside the logs dir into a fresh one there, so every run
    leaves a trail; None for a run root inside it."""
    logs_dir = _logs_dir()
    if run_root == logs_dir or logs_dir in run_root.parents:
        return None
    archive = _fresh_logs_dir(label)
    shutil.copytree(run_root, archive, dirs_exist_ok=True)
    return archive
