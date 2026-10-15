"""The benchmark runner — execute tasks through every source, then grade.

For each task the runner re-exposes the *same* dataset through each requested
source connector (local file, SQLite database, in-memory object storage, …),
resolves it back to a local file via :func:`avaloka.io.materialize`, runs the
real mission, and applies the scorers. Batch tasks additionally prove exact
map-reduce convergence; tasks with ``coordinate_workers`` additionally run the
multi-Avaloka fleet and check that the converged answer matches a single pass.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

from avaloka import ray_batch, workload
from avaloka.benchmark import comprehension, scoring
from avaloka.benchmark.datasets import ensure_kaggle_dataset
from avaloka.benchmark.spec import (BenchmarkTask, CheckResult, Scorecard,
                                    TaskResult)
from avaloka.coordinator import MainAvaloka
from avaloka.io import load_dataset, materialize
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze, run_train


#: What a plain install is told when a benchmark task needs a library that only
#: the `benchmark` extra in pyproject.toml brings in.
BENCHMARK_EXTRA_HINT = 'pip install "avaloka[benchmark]"'


class BenchmarkRunner:
    """Runs a list of :class:`BenchmarkTask` and returns a :class:`Scorecard`."""

    def __init__(self, base_dir: str | Path, *, verbose: bool = False,
                 edgar: bool = False) -> None:
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        # EDGAR tasks reach the network, so they are opt-in exactly as Kaggle
        # tasks are. Default off keeps `python -m avaloka.benchmark run`
        # hermetic with every key unset.
        self.edgar = edgar

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    @staticmethod
    def _read_any(path: Path) -> pd.DataFrame:
        """Read a prepared dataset back, whatever format it is in.

        Goes through :func:`avaloka.io.load_dataset` -- the same reader the
        missions use -- so the benchmark never sees a table the product could
        not have seen. A private CSV read here would have quietly given the
        suite a better view of the file than Avaloka gets, which is the one
        thing a benchmark must not do.
        """
        return load_dataset(str(path)).frame

    # -- source exposure ---------------------------------------------------
    def _expose(self, base_csv: Path, frame: pd.DataFrame, scheme: str,
                workdir: Path) -> Optional[str]:
        """Re-publish the dataset through a source connector; return its URI (or None to skip)."""
        if scheme in ("file", "local"):
            return str(base_csv)
        workdir.mkdir(parents=True, exist_ok=True)
        if scheme in ("sqlite", "database"):
            if frame is None:
                raise ValueError(f"{base_csv.name}: only a CSV dataset can be re-published "
                                 "through sqlite")
            import sqlalchemy
            db = workdir / "bench.db"
            engine = sqlalchemy.create_engine(f"sqlite:///{db}")
            frame.to_sql("bench_data", engine, if_exists="replace", index=False)
            engine.dispose()
            return f"sqlite:///{db}#bench_data"
        if scheme in ("object_storage", "memory"):
            import fsspec
            key = f"memory://avaloka-bench/{workdir.name}/{base_csv.name}"
            with fsspec.open(key, "wb") as fh:
                fh.write(base_csv.read_bytes())
            return key
        if scheme in ("s3", "gs", "az"):
            self._log(f"    · skip {scheme}:// (fsspec backend not installed)")
            return None
        return None

    # -- dataset preparation ----------------------------------------------
    def _prepare_dataset(self, task: BenchmarkTask, workdir: Path) -> Optional[tuple[Path, str]]:
        """Return ``(local_path, target)`` or None when unavailable (Kaggle/EDGAR skip).

        The suffix comes from the spec rather than being hard-coded ``.csv``.
        That one assumption is why eight of the nine declared upload formats had
        no benchmark coverage: a builder could only ever write a CSV, so there
        was nowhere to put a workbook or an Avro container.
        """
        spec = task.dataset
        if spec.kind == "synthetic":
            dest = workdir / f"{spec.name}{spec.suffix}"
            spec.builder(dest)
            return dest, (task.target or spec.target or "")
        if spec.kind == "kaggle":
            path = ensure_kaggle_dataset(spec.kaggle_slug, spec.preferred_file)
            if path is None:
                return None
            return path, (task.target or spec.target or "")
        if spec.kind == "edgar":
            if not self.edgar:
                return None
            from avaloka.benchmark.edgar import ensure_edgar_filing
            path = ensure_edgar_filing(spec.edgar_cik, spec.edgar_accession, workdir)
            if path is None:
                return None
            return path, (task.target or spec.target or "")
        return None

    # -- one (task, scheme) run -------------------------------------------
    def _run_mission(self, task: BenchmarkTask, uri: str, scheme: str,
                     out: Path, target: str) -> TaskResult:
        started = time.perf_counter()
        checks: list[CheckResult] = []
        try:
            resolved = materialize(uri)
            local = str(resolved.local_path)
            wl = workload.route(local)

            kind = MissionKind.TRAIN if task.kind == "train" else MissionKind.ANALYZE
            ctx = MissionContext(
                kind=kind, goal=task.goal, data_source=local, output_dir=out,
                budget=Budget(limit_usd=task.budget), target=(target or None),
                metric=task.metric, deployable=task.deployable)
            ctx.blackboard["workload"] = wl.as_dict()
            ctx.blackboard["resolved_source"] = resolved.as_dict()

            (run_train if kind is MissionKind.TRAIN else run_analyze)(ctx)

            # source-connector correctness: profile must reflect the real data
            handle = load_dataset(local)
            frame = handle.frame

            checks += scoring.score_workload(out, task.expectations)
            checks += scoring.score_roles(out, task.expectations)
            checks += scoring.score_quality(out, task.expectations)
            checks += scoring.score_schema(out, task.expectations)
            checks += scoring.score_transformation(out)
            checks += scoring.score_leakage(out, task.expectations)
            checks += scoring.score_deployment_level(out, task.expectations)
            checks += scoring.score_fireflies(out, task.expectations)
            checks += scoring.score_economics(out, task.expectations)
            checks += scoring.score_plan(out, task.expectations)
            checks += scoring.score_goal_analysis(out, task.expectations)
            checks += scoring.score_split_was_performed(out, task.expectations, frame)
            # Did it understand the file at all? Every scorer above grades what
            # Avaloka did with `frame`; none asks whether `frame` is the table
            # that was in the file.
            checks += comprehension.score_comprehension(out, task.expectations, frame)
            if task.kind == "train" and target:
                model_frame = frame if target in frame.columns else \
                    pd.read_parquet(out / "transformed_dataset.parquet")
                checks += scoring.score_model(out, task.expectations, model_frame, target)
                checks += scoring.score_baseline(out, task.expectations)
                checks += scoring.score_features_used(out, task.expectations)
            checks = scoring.mark_guards(checks, task.expectations)

            error = None
        except Exception:
            error = traceback.format_exc()
        return TaskResult(task=task.name, family=task.family, kind=task.kind,
                          source_scheme=scheme, checks=checks, error=error,
                          artifacts_dir=str(out), duration_s=time.perf_counter() - started)

    def _run_batch(self, task: BenchmarkTask, base_csv: Path, out: Path) -> TaskResult:
        started = time.perf_counter()
        checks: list[CheckResult] = []
        try:
            frame = self._read_any(base_csv)
            wl = workload.route(str(base_csv))
            if task.expectations.expected_lane:
                checks.append(CheckResult(
                    "workload_routing", wl.lane.value == task.expectations.expected_lane,
                    f"router chose '{wl.lane.value}'", weight=2.0))
            res = ray_batch.run_ray_local(str(base_csv), n_partitions=8)
            cols = task.expectations.exact_columns or \
                [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])][:3]
            checks += scoring.score_exact_convergence(res.converged, frame, cols)
            checks.append(CheckResult("batch_engine", res.engine in {"ray-local", "single-process"},
                                      f"engine={res.engine}, partitions={res.n_partitions}"))
            error = None
        except Exception:
            error = traceback.format_exc()
        return TaskResult(task=task.name, family=task.family, kind="batch",
                          source_scheme="file", checks=checks, error=error,
                          artifacts_dir=str(out), duration_s=time.perf_counter() - started)

    def _run_coordination(self, task: BenchmarkTask, base_csv: Path, out: Path,
                          target: str) -> TaskResult:
        started = time.perf_counter()
        checks: list[CheckResult] = []
        try:
            frame = self._read_any(base_csv)
            kind = MissionKind.TRAIN if task.kind == "train" else MissionKind.ANALYZE
            main = MainAvaloka()
            cr = main.coordinate(
                str(base_csv), goal=task.goal, output_dir=out,
                workers=task.coordinate_workers or 4, kind=kind,
                target=(target or None), metric=task.metric, deployable=task.deployable,
                budget=task.budget)
            cols = task.expectations.exact_columns or \
                [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])][:3]
            checks += scoring.score_exact_convergence(cr.converged_profile, frame, cols)
            checks.append(CheckResult(
                "coordination_workers", cr.workers == (task.coordinate_workers or 4),
                f"dispatched {cr.workers} sub-Avalokas"))
            valid_verdicts = {"pass", "warn", "fail"}
            checks.append(CheckResult("coordination_verdict", cr.verdict in valid_verdicts,
                                      f"converged verdict '{cr.verdict}'"))
            if task.expectations.expected_verdict:
                checks.append(CheckResult(
                    "coordination_verdict_expected",
                    cr.verdict == task.expectations.expected_verdict,
                    f"'{cr.verdict}' vs expected '{task.expectations.expected_verdict}'"))
            error = None
        except Exception:
            error = traceback.format_exc()
        return TaskResult(task=task.name, family=task.family, kind=f"{task.kind}+coordinate",
                          source_scheme="fleet", checks=checks, error=error,
                          artifacts_dir=str(out), duration_s=time.perf_counter() - started)

    def _grade_refusal(self, task: BenchmarkTask, path: Path,
                       workdir: Path) -> TaskResult:
        """Grade a dataset the reader must refuse rather than guess at."""
        started = time.perf_counter()
        wanted = list(task.expectations.expected_refusal)
        checks: list[CheckResult] = []
        try:
            frame = self._read_any(path)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            missing = [token for token in wanted
                       if token.lower() not in message.lower()]
            checks.append(CheckResult(
                "reader_refuses_ambiguous_document", True,
                f"refused, as it must: {message[:200]}", weight=2.0))
            checks.append(CheckResult(
                "refusal_names_the_problem", not missing,
                (f"the refusal does not name {missing}; a refusal that does not "
                 f"say what is wrong leaves the user no next step. Message: "
                 f"{message[:160]}") if missing else
                f"names {wanted}, so the user knows what to extract",
                weight=2.0))
        else:
            checks.append(CheckResult(
                "reader_refuses_ambiguous_document", False,
                (f"the reader returned a frame of {frame.shape[0]} rows and "
                 f"columns {list(frame.columns)[:8]} for a document with no "
                 f"single correct reading. It should have refused naming "
                 f"{wanted}: silently choosing one interpretation is the defect "
                 "this task exists for."), weight=2.0))
            checks.append(CheckResult(
                "refusal_names_the_problem", False,
                "not reached: the reader did not refuse", weight=2.0))
        return TaskResult(
            task=task.name, family=task.family, kind=task.kind,
            source_scheme="file", checks=checks, artifacts_dir=str(workdir),
            duration_s=time.perf_counter() - started)

    # -- public API --------------------------------------------------------
    def run_task(self, task: BenchmarkTask) -> list[TaskResult]:
        workdir = self.base_dir / task.name
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            prepared = self._prepare_dataset(task, workdir)
        except Exception as exc:
            # A dataset that cannot be built is a FAILED run, with its reason,
            # and the other tasks still run. It must never read as a skip: the
            # exit rule skips an error containing one particular word, so the
            # text is composed here and checked not to contain it.
            reason = f"dataset could not be built: {type(exc).__name__}: {exc}"
            if isinstance(exc, ImportError):
                # The spreadsheet tasks write their own file with a library the
                # CLI does not depend on. Say what to install.
                reason += f' -- install the benchmark extra: {BENCHMARK_EXTRA_HINT}'
            reason = reason.replace("unavailable", "not available")
            self._log(f"  · {task.name}: {reason}")
            return [TaskResult(task=task.name, family=task.family, kind=task.kind,
                               source_scheme="—", checks=[], error=reason)]
        if prepared is None:
            self._log(f"  · {task.name}: dataset unavailable — skipped")
            return [TaskResult(task=task.name, family=task.family, kind=task.kind,
                               source_scheme="—", checks=[],
                               error="dataset unavailable (Kaggle not downloaded)")]
        base_csv, target = prepared

        # A dataset the reader is *supposed* to refuse is graded here, not by a
        # mission: the decision happens at load time, and no mission should run
        # on a document whose shape has no single correct reading.
        if task.expectations.expected_refusal:
            return [self._grade_refusal(task, base_csv, workdir)]

        try:
            frame = self._read_any(base_csv)
        except Exception:
            # A dataset the loader cannot read is a red task, not a dead run.
            # This read was `pd.read_csv` with no guard, so the first
            # unreadable format would have aborted the whole suite before any
            # scorecard was written.
            return [TaskResult(
                task=task.name, family=task.family, kind=task.kind,
                source_scheme="file", checks=[CheckResult(
                    "dataset_is_readable", False,
                    f"the benchmark could not read its own {task.dataset.suffix} "
                    f"dataset back: {traceback.format_exc().strip().splitlines()[-1]}",
                    weight=2.0)],
                error=traceback.format_exc(), artifacts_dir=str(workdir))]
        results: list[TaskResult] = []

        if task.kind == "batch":
            self._log(f"  · {task.name} [batch convergence]")
            results.append(self._run_batch(task, base_csv, workdir / "batch"))
        else:
            for scheme in task.source_schemes:
                uri = self._expose(base_csv, frame, scheme, workdir / scheme)
                if uri is None:
                    continue
                self._log(f"  · {task.name} via {scheme}")
                results.append(self._run_mission(
                    task, uri, scheme, workdir / scheme / "mission", target))

        if task.coordinate_workers:
            self._log(f"  · {task.name} [multi-Avaloka x{task.coordinate_workers}]")
            results.append(self._run_coordination(task, base_csv, workdir / "coordinate", target))
        return results

    def run(self, tasks: list[BenchmarkTask]) -> Scorecard:
        card = Scorecard()
        for task in tasks:
            for result in self.run_task(task):
                card.add(result)
        return card
