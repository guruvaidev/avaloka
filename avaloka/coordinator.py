"""Multi-Avaloka coordination — the open-claw *main* agent.

The per-mission firefly swarm (``avaloka.swarm``) is one Avaloka cloning herself
across the *steps* of a single job. This module lifts that one level up: a
**main Avaloka** decomposes work over the *same dataset* into shards and
dispatches a fleet of **sub-Avalokas** — each a full clone that runs its own
firefly swarm on its shard — then **converges** their partial answers into one
result she will stand behind.

This is the swarm-of-swarms the spec describes: multiple data scientists working
the same dataset in parallel under one coordinator, sampling, analysing, building
features, training and reconciling into a single deliverable.

Two convergence guarantees make the coordination honest rather than theatrical:

* **Profiling converges *exactly*.** Each sub-Avaloka emits sufficient statistics
  for its shard (counts, missing, min, max, and mean/variance via sums); the main
  Avaloka combines them losslessly (the same map-reduce as ``avaloka.ray_batch``),
  so the converged full-dataset profile equals a single-pass profile — no sampling
  error.
* **Decisions converge *conservatively*.** Validation verdicts merge worst-case
  (any shard that fails fails the fleet) and the selected model is the shard model
  with the best honest score; the coordinator never averages away a red flag.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from avaloka import workload
from avaloka.fireflies import sampling_specialist
from avaloka.drivers_answer import drivers_table
from avaloka.goal_analysis import driver_target
from avaloka.goal_merge import answer_markdown, goal_partials, merge_goal_answers
from avaloka.io import ResolvedSource, load_dataset, materialize
from avaloka.io.sharding import ShardFile, ShardPlan, peek_columns, write_shards
from avaloka.mission.budget import Budget
from avaloka.mission.context import ExecutionMode, MissionContext, MissionKind
from avaloka.mission.economics import EconomicReport, compute_economics
from avaloka.mission.ledger import Ledger, WorkUnit
from avaloka.missions import run_analyze, run_train
from avaloka.ray_batch import _combine, _partition_stats
from avaloka.swarm_limits import SwarmGrant, SwarmLimits, grant, limits_from_env
from avaloka.swarm_sizing import FALLBACK_CLONES, FleetSize, size_fleet
from avaloka.swarm_trace import SwarmTraceRecorder, clone_id_for, clone_name_for
from avaloka.understanding_trace import (AGENT_RECORD, EGRESS_FILE, LOG_FILE, SWARM_RECORD,
                                         AgentUnderstanding, SwarmUnderstanding, UnderstandingLog,
                                         egress_events)
from avaloka.util import write_json

#: Where the per-clone trace lands, beside ``coordination.json``.
logger = logging.getLogger(__name__)

SWARM_TRACE_FILE = "swarm_trace.json"
#: The fleet's merged answer and the report that leads with it.
GOAL_ANALYSIS_FILE = "goal_analysis.json"
EXECUTIVE_REPORT_FILE = "executive_report.md"

# A shard is dispatched to a sub-Avaloka; this is the callback the main agent
# uses to narrate the fleet (mirrors the per-mission ``on_progress`` API).
FleetProgress = Callable[[str, str], None]

#: A shard with no verdict (or one this table does not know) was not
#: validated, so it ranks with "fail": a missing verdict is never a pass.
_VERDICT_RANK = {"pass": 0, "warn": 1, "fail": 2}
_UNVERIFIED_RANK = 2
#: The floor of the deployment ladder (avaloka.deploy reads a missing level as 1).
_UNVALIDATED_LEVEL = 1


def _noop(_event: str, _msg: str) -> None:  # pragma: no cover
    pass


def derive_workers(source: str) -> tuple[FleetSize, ResolvedSource | None]:
    """Size the fleet from the source, before anything is sharded.

    Resolves the source (a download or a database transfer happens once; the
    resolved copy is handed on to the coordinator), reads its workload plan
    cheaply (:func:`avaloka.workload.route`: rows, bytes, lane) and its
    native statistics (row groups, partitions, indexes) and applies
    :func:`avaloka.swarm_sizing.size_fleet`. A source that cannot be sized
    gets :data:`FALLBACK_CLONES` and a ``derived_from`` that says so; the
    coordinator then fails on the same source with its trace finished, as it
    did before.
    """
    try:
        resolved = materialize(source)
        plan = workload.route(str(resolved.local_path))
    except Exception as exc:  # noqa: BLE001 - the coordinator raises the real error
        return FleetSize(FALLBACK_CLONES, f"source could not be sized ({type(exc).__name__}); "
                         f"default {FALLBACK_CLONES}"), None
    statistics = dict(resolved.detail.get("statistics") or {})
    return size_fleet(plan.n_rows, plan.n_bytes, lane=plan.lane.value, statistics=statistics,
                      estimated=plan.estimated), resolved


#: Identifier columns say nothing as a finding ("id averages 36,518").
_ID_NAME = re.compile(r"((?i:^id$|_id$|^id_|uuid)|[a-z]ID$)")


def _goal_columns(goal: str, profile: dict[str, Any]) -> list[str]:
    """The profile's columns the goal names: the outcome of a "what drives X"
    goal as avaloka.goal_analysis resolves it, then whole names, then a goal
    word that a single column's name starts with ("churn" -> ``churned``)."""
    names = [str(c.get("name")) for c in profile.get("columns", []) if c.get("name")]
    target = driver_target(goal or "", profile) or {}
    hits = [target["column"]] if target.get("column") in names else []
    low = f" {re.sub(r'[^a-z0-9]+', ' ', (goal or '').lower())} "
    hits += sorted((n for n in names if f" {re.sub(r'[^a-z0-9]+', ' ', n.lower()).strip()} " in low),
                   key=lambda n: (-len(n), n))
    for word in low.split():
        starts = [n for n in names if len(word) >= 4 and n.lower().startswith(word)]
        if len(starts) == 1:
            hits.append(starts[0])
    return list(dict.fromkeys(hits))


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.4g}"


def _column_finding(col: dict[str, Any], n_rows: int) -> str | None:
    """One sentence of fact about one merged column, or ``None``."""
    name, count = col.get("name"), int(col.get("count") or 0)
    if not name or not count:
        return None
    if col.get("role") == "numeric":
        lo, hi, mean = col.get("min"), col.get("max"), col.get("mean")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lo, hi, mean)):
            return None
        if lo == 0 and hi == 1:
            ones = int(round(mean * count))
            return f"{name} is 1 in {mean:.1%} of the {count:,} rows with a value ({ones:,} rows)"
        return f"{name} averages {_fmt(mean)} over {count:,} rows (range {_fmt(lo)} to {_fmt(hi)})"
    tops = [t for t in col.get("top_values") or [] if isinstance(t, dict)]
    if not tops or not n_rows:
        return None
    top = tops[0]
    count = int(top["count"])
    tied = [t for t in tops if int(t["count"]) == count]
    if len(tied) == 1:
        return (f"{name} is most often “{top['value']}” ({count:,} of {n_rows:,} rows, "
                f"{count / n_rows:.1%})")
    # "Most often “Recovered”" over a 20/20/20 split named one of three. The
    # profile keeps the ten most frequent values, so when all ten tie there
    # may be more.
    names = [f"“{t['value']}”" for t in tied]
    each = f"{count:,} of {n_rows:,} rows each ({count / n_rows:.1%})"
    if len(names) <= 3:
        return (f"{name} has no single most common value: {', '.join(names[:-1])} and {names[-1]} "
                f"tie at {each}")
    many = "at least " if len(tied) == len(tops) >= 10 else ""
    return (f"{name} has no single most common value: {many}{len(tied)} values tie at {each}, "
            f"among them {', '.join(names[:2])} and {names[2]}")


def merged_findings(profile: dict[str, Any], *, goal: str = "", target: str | None = None,
                    limit: int = 2) -> list[str]:
    """What the merged profile says, in sentences: the facts the swarm's story
    reports as its finding.

    Every figure is the exact merged statistic (sufficient statistics summed
    over the shards), never a clone's estimate. The column the mission is
    about comes first -- the target, then the columns the goal names; the
    worst gap follows when the data has one, then the first column with
    something to say. Local stdout and the chat only: the trace's telemetry
    projection (:mod:`avaloka.understanding_trace`) does not carry it.
    """
    columns = [c for c in profile.get("columns", []) if isinstance(c, dict)]
    n_rows = int(profile.get("n_rows") or 0)
    by_name = {c.get("name"): c for c in columns}
    focus = ([target] if target in by_name else []) + [
        n for n in _goal_columns(goal, profile) if n != target]
    out: list[str] = []
    for name in focus:
        line = _column_finding(by_name[name], n_rows)
        if line and line not in out:
            out.append(line)
        if len(out) >= limit:
            return out
    gaps = sorted((c for c in columns if int(c.get("missing") or 0) > 0),
                  key=lambda c: (-int(c["missing"]), str(c.get("name"))))
    if gaps and n_rows:
        worst = gaps[0]
        out.append(f"{worst['name']} is empty in {int(worst['missing']):,} of {n_rows:,} rows "
                   f"({int(worst['missing']) / n_rows:.1%})")
    for col in columns:
        if len(out) >= limit:
            break
        line = _column_finding(col, n_rows)
        if line and line not in out and col.get("name") not in focus \
                and not _ID_NAME.search(str(col.get("name"))):
            out.append(line)
    return out[:limit]


def _safe_source_uri(uri: str) -> str:
    """Keep source provenance without persisting passwords or signed queries."""
    split = urlsplit(uri)
    if not split.scheme or len(split.scheme) == 1:
        return uri
    netloc = split.netloc
    if split.username is not None:
        host = split.hostname or ""
        if split.port is not None:
            host = f"{host}:{split.port}"
        user = split.username
        netloc = f"{user}:***@{host}" if split.password is not None else f"{user}@{host}"
    query = "redacted=1" if split.query else ""
    return urlunsplit((split.scheme, netloc, split.path, query, split.fragment))


@dataclass
class ShardOutcome:
    """What one sub-Avaloka returned for its shard of the dataset."""

    shard_id: int
    n_rows: int
    output_dir: str
    verdict: str | None
    max_safe_deployment_level: int | None
    manual_minutes: float
    cost_usd: float
    model: dict[str, Any] | None = field(default=None)
    sampling: dict[str, Any] = field(default_factory=dict)
    partition_stats: dict[str, Any] = field(default_factory=dict)
    #: Her answer to the goal over her shard (her goal_analysis.json) and the
    #: sufficient statistics behind it, which the merge combines. The partials
    #: stay in memory: they can hold every group, and are not the report.
    goal_analysis: dict[str, Any] | None = field(default=None)
    goal_partials: dict[str, Any] | None = field(default=None, repr=False)


@dataclass
class CoordinationResult:
    """The main Avaloka's converged answer over all sub-Avalokas."""

    kind: str
    goal: str
    source: str
    workers: int
    n_rows: int
    converged_profile: dict[str, Any]
    verdict: str
    max_safe_deployment_level: int
    shards: list[ShardOutcome]
    economics: EconomicReport
    selected_model: dict[str, Any] | None
    output_dir: str
    source_resolution: dict[str, Any] = field(default_factory=dict)
    sampling_summary: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    # Where the per-clone trace was written and how many records it holds.
    trace: dict[str, Any] = field(default_factory=dict)
    # The one answer to the goal, merged from the clones' (avaloka.goal_merge).
    answer: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "goal": self.goal,
            "answer": self.answer,
            "source": self.source,
            "workers": self.workers,
            "n_rows": self.n_rows,
            "verdict": self.verdict,
            "max_safe_deployment_level": self.max_safe_deployment_level,
            "selected_model": self.selected_model,
            "converged_profile": self.converged_profile,
            "economics": self.economics.as_dict(),
            "shards": [{k: v for k, v in s.__dict__.items() if k != "goal_partials"}
                       for s in self.shards],
            "output_dir": self.output_dir,
            "source_resolution": self.source_resolution,
            "sampling_summary": self.sampling_summary,
            "execution": self.execution,
            "trace": self.trace,
        }


class MainAvaloka:
    """The coordinating (open-claw) Avaloka.

    She streams the dataset once into one shard file per clone
    (:mod:`avaloka.io.sharding`) without holding it whole, dispatches a
    sub-Avaloka mission per shard, then converges their outputs. Sub-missions are
    genuinely independent — each loads only its own shard file — so this models
    a real fleet, not a single process pretending to be many, and her memory is
    the clones' shards, not the dataset again on top of them.
    """

    def __init__(self, *, tier: str = "community", loaded_hourly_rate: float = 80.0,
                 execution_mode: ExecutionMode = ExecutionMode.LOCAL,
                 swarm_limits: SwarmLimits | None = None) -> None:
        self.tier = tier
        self.loaded_hourly_rate = loaded_hourly_rate
        self.execution_mode = execution_mode
        # The edition's fan-out limit. Resolved from the installed build when
        # not given, so every caller is capped, not just the CLI.
        self.swarm_limits = swarm_limits if swarm_limits is not None else limits_from_env()

    # -- sharding ----------------------------------------------------------
    @staticmethod
    def _write_shards(path: Path, out: Path, clones: int,
                      question: str | None = None) -> ShardPlan:
        """One ``sub-avaloka-NN`` directory per clone, each with its shard file;
        a clone that would receive no rows gets no directory. ``question`` picks
        a workbook's sheet, as it does for a single mission."""
        return write_shards(path, [out / f"sub-avaloka-{i:02d}" for i in range(max(1, clones))],
                            question=question)

    def _run_shard(
        self,
        *,
        shard_id: int,
        shard: ShardFile,
        shard_dir: Path,
        kind: MissionKind,
        goal: str,
        target: str | None,
        metric: str | None,
        deployable: bool,
        budget: float | None,
        on_progress: FleetProgress,
        recorder: SwarmTraceRecorder,
        understanding: UnderstandingLog | None = None,
    ) -> tuple[ShardOutcome, list[WorkUnit]]:
        """Run one independent sub-Avaloka and return collectable state.

        The partition statistics are read from the frame the clone's own Data
        Scout loaded, so a shard is loaded once, by its clone.
        """
        shard_path, n_rows = shard.path, shard.n_rows
        on_progress(
            "dispatch",
            f"Ava-{shard_id:02d} takes shard {shard_id} ({n_rows:,} rows)",
        )
        clone_id = clone_id_for(shard_id)
        recorder.record(
            "clone.dispatched",
            clone_id=clone_id, name=clone_name_for(shard_id), shard_id=shard_id,
            received={
                "rows": n_rows, "columns": list(shard.columns),
                "goal": goal, "kind": kind.value, "target": target, "metric": metric,
                "shard_path": str(shard_path),
            },
        )
        ctx = MissionContext(
            kind=kind,
            goal=goal,
            data_source=str(shard_path),
            output_dir=shard_dir / "mission",
            budget=Budget(limit_usd=budget),
            target=target,
            metric=metric,
            deployable=deployable,
            execution_mode=self.execution_mode,
            tier=self.tier,
            loaded_hourly_rate=self.loaded_hourly_rate,
        )
        # A shard is a full-data lane from the sub-Avaloka's perspective until
        # it crosses the Sampling Specialist's own threshold. Recording this
        # makes the sampling contract visible in coordination.json.
        ctx.blackboard["workload"] = {
            "lane": "online" if n_rows <= sampling_specialist.SAMPLE_THRESHOLD_ROWS else "sampled_online",
            "n_rows": n_rows,
            "scope": "partition",
        }
        run_fn = run_train if kind is MissionKind.TRAIN else run_analyze
        # Every firefly step the clone runs lands in the trace as it happens:
        # "start" opens the step, "done" closes it from the unit the firefly
        # just put on the ledger (Firefly.activate records before _run narrates).
        steps = recorder.clone_progress(clone_id, ctx.ledger)
        # Behind each completed step, what the firefly understood and decided
        # (avaloka.understanding_trace): the open step is captured on "start"
        # because completing it closes it, and read back on "done".
        current: dict[str, Any] = {}

        def understood(outcome: str, error_class: str | None = None) -> None:
            if not current:
                return
            unit = ctx.ledger.units[-1] if outcome == "ok" and ctx.ledger.units else None
            self._understood(
                recorder, understanding, AGENT_RECORD,
                lambda: AgentUnderstanding.from_firefly(
                    step=dict(current), unit=unit, blackboard=ctx.blackboard, clone_id=clone_id,
                    clone_name=clone_name_for(shard_id), clone_index=shard_id,
                    data_source=str(shard_path), outcome=outcome,
                    error_class=error_class).as_record())
            current.clear()

        def progress(event: str, message: str) -> None:
            steps(event, message)
            if event == "start" and steps.open_step is not None:
                current.clear()
                current.update(steps.open_step)
            elif event == "done":
                understood("ok")

        clone_started = time.perf_counter()
        try:
            result = run_fn(ctx, on_progress=progress)
        except BaseException as exc:
            # A firefly that raised recorded no unit; close its step as failed,
            # then let the failure propagate exactly as it did before.
            steps.complete("failed", error_class=type(exc).__name__, error=str(exc))
            understood("failed", type(exc).__name__)
            recorder.record(
                "clone.returned", clone_id=clone_id, status="failed", verdict=None,
                max_safe_deployment_level=None, model=None, sampling={},
                duration_ms=int(round((time.perf_counter() - clone_started) * 1000)),
                cost_usd=round(ctx.ledger.compute_cost_usd + ctx.ledger.model_cost_usd, 6),
                compute_cost_usd=round(ctx.ledger.compute_cost_usd, 6),
                model_cost_usd=round(ctx.ledger.model_cost_usd, 6),
                steps=len(ctx.ledger.units),
                error_class=type(exc).__name__, error=str(exc),
            )
            raise
        handle = ctx.blackboard.get("handle")
        frame = handle.frame if handle is not None else load_dataset(str(shard_path)).frame
        n_rows, stats = len(frame), _partition_stats(frame)
        del frame, handle
        analysis, partials = self._goal_part(ctx)
        summary = result.summary
        outcome = ShardOutcome(
            shard_id=shard_id,
            n_rows=n_rows,
            output_dir=str(shard_dir / "mission"),
            verdict=summary.get("verdict"),
            max_safe_deployment_level=summary.get("max_safe_deployment_level"),
            manual_minutes=ctx.ledger.manual_minutes,
            cost_usd=round(ctx.ledger.compute_cost_usd + ctx.ledger.model_cost_usd, 6),
            model=summary.get("model"),
            sampling=dict(ctx.blackboard.get("sampling") or {}),
            partition_stats=stats,
            goal_analysis=analysis,
            goal_partials=partials,
        )
        # "done" is a clone that came back validated. One whose mission ended
        # without a verdict did not do its work, and its record says so.
        unverified = outcome.verdict not in _VERDICT_RANK
        recorder.record(
            "clone.returned", clone_id=clone_id, status="failed" if unverified else "done",
            verdict=outcome.verdict,
            max_safe_deployment_level=outcome.max_safe_deployment_level,
            model=outcome.model, sampling=outcome.sampling,
            duration_ms=int(round((time.perf_counter() - clone_started) * 1000)),
            cost_usd=outcome.cost_usd, steps=len(ctx.ledger.units),
            # Which part is a nominal CPU charge and which is model calls: the
            # narration must not call the first a "cost" next to "No model was called".
            compute_cost_usd=round(ctx.ledger.compute_cost_usd, 6),
            model_cost_usd=round(ctx.ledger.model_cost_usd, 6),
            **({"error_class": "NoVerdict", "error": "the clone's mission ended without a verdict"}
               if unverified else {}),
        )
        on_progress("report", f"Ava-{shard_id:02d} back without a verdict" if unverified
                    else f"Ava-{shard_id:02d} back — verdict={outcome.verdict}")
        return outcome, list(ctx.ledger.units)

    @staticmethod
    def _goal_part(ctx: MissionContext) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Her answer and its mergeable statistics, from the very frame and
        filters her Analysis Planner answered over. A failure here costs the
        merge this clone's figures, never the mission."""
        analysis = ctx.blackboard.get("goal_analysis")
        if not analysis:
            return None, None
        # A shard over the sampling threshold is still answered over every
        # row she holds (analyze_goal's full pass); her statistics must come
        # from those rows too, or the merge reweights a sample's means.
        basis = analysis.get("basis") or {}
        rows = ctx.blackboard.get("frame") if basis.get("full_pass") else ctx.blackboard.get("working_frame")
        try:
            return analysis, goal_partials(rows, analysis)
        except Exception:  # noqa: BLE001
            logger.warning("[swarm] no mergeable goal statistics for this clone: %s",
                           type(sys.exc_info()[1]).__name__, exc_info=True)
            return analysis, None

    @staticmethod
    def _understood(recorder: SwarmTraceRecorder, log: UnderstandingLog | None, kind: str,
                    build: Callable[[], dict[str, Any]]) -> None:
        """Record an understanding record and append it to the local log.
        Understanding is derived from what the fireflies already wrote, so a
        bug in deriving it must never fail the mission: it is logged and the
        step simply has no understanding record."""
        try:
            fields = build()
            # Through the log when there is one, so understanding.jsonl stays
            # in seq order while clones record at the same time.
            if log is not None:
                log.record(recorder, kind, **fields)
            else:
                recorder.record(kind, **fields)
        except Exception:  # noqa: BLE001
            # Not silent: a swallowed derivation bug once cost every clone its
            # feature-engineer record while the mission read as complete.
            logger.warning("[swarm] no %s record for this step: %s", kind, type(sys.exc_info()[1]).__name__,
                           exc_info=True)

    # -- coordination ------------------------------------------------------
    def coordinate(
        self,
        source: str,
        *,
        goal: str,
        output_dir: str | Path,
        workers: int | None = None,
        kind: MissionKind = MissionKind.ANALYZE,
        target: str | None = None,
        metric: str | None = None,
        deployable: bool = False,
        budget: float | None = None,
        max_parallel: int | None = None,
        on_progress: FleetProgress = _noop,
        recorder: SwarmTraceRecorder | None = None,
        derived_from: str | None = None,
    ) -> CoordinationResult:
        """Fan sub-Avalokas out in parallel and converge one auditable answer.

        ``workers`` is the caller's explicit fleet (``--swarm N``). When it is
        ``None`` the count is derived from the data by :func:`derive_workers`
        (rows, bytes and lane from the workload plan; row groups or
        partitions from the source statistics; the formula is in
        :mod:`avaloka.swarm_sizing`), announced through ``on_progress("sizing",
        reason)`` and recorded as ``derived_from`` on ``swarm.requested``. A
        caller that derived the count itself passes both ``workers`` and
        ``derived_from``. The edition cap then applies as it always did, so
        the open-source and Free editions run their preview of two and are
        told why.

        Every clone's work is recorded on ``recorder`` (one is created when none
        is given, so the trace always exists) and written to
        ``<output_dir>/swarm_trace.json`` beside ``coordination.json``. What
        each firefly understood and decided (``agent.understanding``) and the
        fleet's summary (``swarm.understanding``) go on the same recorder and
        into ``<output_dir>/understanding.jsonl``; the metadata-only
        projection telemetry would send is written to
        ``<output_dir>/understanding_egress.json`` so a reviewer can read it.
        """
        resolved: ResolvedSource | None = None
        if workers is None:
            fleet, resolved = derive_workers(source)
            workers, derived_from = fleet.clones, fleet.derived_from
            on_progress("sizing", derived_from)
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if max_parallel is not None and max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")
        # The edition limit applies before anything is sharded, and a capped
        # request is announced, never silently shrunk.
        allowed = grant(workers, max_parallel, self.swarm_limits, derived_from=derived_from)
        if allowed.limited:
            on_progress("limit", allowed.message)
        rec = recorder if recorder is not None else SwarmTraceRecorder()
        rec.record(
            "swarm.requested", kind=kind.value, goal=goal, source=_safe_source_uri(source),
            requested_clones=allowed.requested_clones, requested_parallel=allowed.requested_parallel,
            granted_clones=allowed.clones, granted_parallel=allowed.parallel,
            limited=allowed.limited, message=allowed.message, derived_from=allowed.derived_from,
            limits=self.swarm_limits.as_dict(),
        )
        started = time.perf_counter()
        out = Path(output_dir).expanduser()
        try:
            return self._coordinate(
                source, out, allowed, rec, started, goal=goal, kind=kind, target=target,
                metric=metric, deployable=deployable, budget=budget, on_progress=on_progress,
                resolved=resolved,
            )
        except BaseException as exc:
            # Whatever failed -- an unreachable source, a missing target column,
            # a clone that raised -- the trace says so, and the error propagates
            # unchanged. The file is best effort: the directory may not exist.
            rec.record("swarm.finished", status="failed",
                       duration_ms=int(round((time.perf_counter() - started) * 1000)),
                       economics=None, error_class=type(exc).__name__)
            try:
                if out.is_dir():
                    write_json(out / SWARM_TRACE_FILE, rec.mission())
            except Exception:  # noqa: BLE001
                pass
            raise

    def _coordinate(
        self,
        source: str,
        out: Path,
        allowed: SwarmGrant,
        rec: SwarmTraceRecorder,
        started: float,
        *,
        goal: str,
        kind: MissionKind,
        target: str | None,
        metric: str | None,
        deployable: bool,
        budget: float | None,
        on_progress: FleetProgress,
        resolved: ResolvedSource | None = None,
    ) -> CoordinationResult:
        # Sizing may have resolved the source already; a download or a
        # database transfer is not repeated for the shard read.
        resolved = resolved if resolved is not None else materialize(source)
        local = Path(resolved.local_path)
        if kind is MissionKind.TRAIN:
            if not target:
                raise ValueError("a training swarm requires a target column")
            # The header decides when the format has a cheap one; otherwise
            # the shards are written first and their columns decide.
            _require_target(target, peek_columns(local))
        # The dataset is never loaded whole here: it is streamed into the
        # shard files (or, for a format that cannot stream, loaded, split
        # and released) before any clone runs.
        out.mkdir(parents=True, exist_ok=True)
        plan = self._write_shards(local, out, allowed.clones, question=goal or None)
        if kind is MissionKind.TRAIN:
            _require_target(target, plan.columns)
        rec.record(
            "swarm.sharded", clones=len(plan.shards), total_rows=plan.total_rows,
            rows_per_clone=[s.n_rows for s in plan.shards], columns=plan.columns,
            method=plan.method,
        )

        understanding = UnderstandingLog(out / LOG_FILE)
        try:
            return self._run_fleet(
                plan, resolved, source, out, allowed, rec, started, understanding,
                goal=goal, kind=kind, target=target, metric=metric, deployable=deployable,
                budget=budget, on_progress=on_progress)
        finally:
            understanding.close()

    def _run_fleet(
        self,
        plan: ShardPlan,
        resolved: Any,
        source: str,
        out: Path,
        allowed: SwarmGrant,
        rec: SwarmTraceRecorder,
        started: float,
        understanding: UnderstandingLog,
        *,
        goal: str,
        kind: MissionKind,
        target: str | None,
        metric: str | None,
        deployable: bool,
        budget: float | None,
        on_progress: FleetProgress,
    ) -> CoordinationResult:
        shards = plan.shards
        on_progress("announce", f"{len(shards)}")
        fleet_ledger = Ledger()
        outcomes: list[ShardOutcome] = []
        prepared = [(shard.shard_id, shard, shard.path.parent) for shard in shards]

        parallelism = max(1, min(allowed.parallel, len(prepared)))
        with cf.ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = [
                executor.submit(
                    self._run_shard,
                    shard_id=i,
                    shard=shard,
                    shard_dir=shard_dir,
                    kind=kind,
                    goal=goal,
                    target=target,
                    metric=metric,
                    deployable=deployable,
                    budget=budget,
                    on_progress=on_progress,
                    recorder=rec,
                    understanding=understanding,
                )
                for i, shard, shard_dir in prepared
            ]
            for future in cf.as_completed(futures):
                outcome, units = future.result()
                outcomes.append(outcome)
                for unit in units:
                    fleet_ledger.record(unit)

        # Completion order is intentionally nondeterministic; persisted output
        # and model tie-breaking remain stable by sorting on shard identity.
        outcomes.sort(key=lambda outcome: outcome.shard_id)
        part_stats = _align_partition_stats(
            [outcome.partition_stats for outcome in outcomes],
            {shard.shard_id: shard.path for shard in shards},
            [outcome.shard_id for outcome in outcomes])

        # -- converge ------------------------------------------------------
        converged_profile = _combine(part_stats)
        verdict = self._merge_verdicts(outcomes)
        # A shard that reported no level was not cleared for any: it is the floor,
        # not the top of the ladder.
        max_level = (min((_UNVALIDATED_LEVEL if o.max_safe_deployment_level is None
                          else o.max_safe_deployment_level) for o in outcomes)
                     if outcomes else _UNVALIDATED_LEVEL)
        answer = merge_goal_answers(goal, _goal_contributions(outcomes, shards))
        if answer.get("kind") == "drivers" and answer.get("status") == "answered":
            # "What predicts X": the table the web chat and the planner graph
            # show (avaloka.drivers_answer), from the merged ranking.
            answer["table"] = drivers_table(answer)
        verdict = self._cap_sampled_answer(verdict, answer)
        selected = self._select_model(outcomes, metric) if kind is MissionKind.TRAIN else None
        merged = dict(
            rules={"profile": "exact sufficient statistics",
                   "verdict": "worst case, at most WARN for an answer estimated from samples"
                   if answer.get("verdict_cap") else "worst case",
                   "model": "best honest score",
                   "answer": "sufficient statistics (sums and counts add, means weighted by n)"},
            # The answer itself: local to the trace file. Telemetry projects
            # swarm.merged through a closed key list that carries none of it.
            answer={"headline": answer.get("headline"), "status": answer.get("status"),
                    "exact": answer.get("exact"), "sampled": bool(answer.get("sampled")),
                    "contributions": [
                        c["finding"] for c in answer.get("contributions", [])],
                    "approximate": [a["statistic"] for a in answer.get("approximate", [])],
                    **({"table": answer["table"]} if answer.get("table") else {})},
            verdict=verdict, max_safe_deployment_level=max_level, selected_model=selected,
            contributions=[{
                "clone_id": clone_id_for(o.shard_id), "rows": o.n_rows, "verdict": o.verdict,
                "selected": bool(selected) and selected.get("from_shard") == o.shard_id,
                "cost_usd": o.cost_usd,
            } for o in outcomes],
            n_rows=converged_profile["n_rows"],
            columns=[c["name"] for c in converged_profile.get("columns", [])],
            findings=merged_findings(converged_profile, goal=goal, target=target),
        )
        # The fleet's understanding is folded from the records so far plus the
        # merge about to be announced; it precedes swarm.merged so the trace
        # still ends merged, finished.
        self._understood(
            rec, understanding, SWARM_RECORD,
            lambda: SwarmUnderstanding.from_records(
                rec.records, merge={"type": "swarm.merged", **merged},
                duration_ms=int(round((time.perf_counter() - started) * 1000)),
                outcome="done").as_record())
        rec.record("swarm.merged", **merged)

        economics = compute_economics(
            fleet_ledger, tier=self.tier, loaded_hourly_rate=self.loaded_hourly_rate)

        sampled = [outcome for outcome in outcomes if outcome.sampling.get("sampled")]
        sampling_summary = {
            "input_rows": sum(outcome.n_rows for outcome in outcomes),
            "working_rows": sum(
                int(outcome.sampling.get("n_working_rows", outcome.n_rows))
                for outcome in outcomes
            ),
            "sampled_shards": len(sampled),
            "strategies": sorted(
                {str(outcome.sampling.get("strategy")) for outcome in outcomes}
                - {"", "None"}
            ),
        }
        source_resolution = resolved.as_dict()
        # A generated cache path is an implementation detail; provenance keeps
        # the original URI, scheme and transfer kind without leaking temp paths.
        source_resolution.pop("local_path", None)
        source_resolution["uri"] = _safe_source_uri(str(source_resolution["uri"]))
        if plan.structure:
            source_resolution["structure"] = plan.structure
        if plan.read.get("kind") == "excel":
            # Which sheet was read, and what was not: a workbook of two equal
            # sheets used to report half its rows with nothing said.
            source_resolution["sheet"] = {
                "analysed": plan.read.get("sheet_name"),
                "others": plan.read.get("other_sheets", []),
            }
        execution = {
            "engine": "thread-pool",
            "parallelism": parallelism,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "mission_engine": "avaloka-firefly",
            # How the shards were cut: streamed or loaded, and rows per clone.
            "sharding": plan.as_dict(),
            # What was asked, what the edition allowed, and why.
            "swarm": allowed.as_dict(),
        }
        rec.record("swarm.finished", status="done",
                   duration_ms=int(round((time.perf_counter() - started) * 1000)),
                   economics=economics.as_dict())
        write_json(out / SWARM_TRACE_FILE, rec.mission())
        # Exactly what telemetry would send for this mission, for the reviewer.
        # Nothing here sends anything.
        write_json(out / EGRESS_FILE, {"schema": "understanding", "events": egress_events(rec.records)})

        coordinated = CoordinationResult(
            kind=kind.value, goal=goal, source=_safe_source_uri(source), workers=len(shards),
            n_rows=converged_profile["n_rows"], converged_profile=converged_profile,
            verdict=verdict, max_safe_deployment_level=max_level, shards=outcomes,
            economics=economics, selected_model=selected, output_dir=str(out),
            source_resolution=source_resolution,
            sampling_summary=sampling_summary,
            execution=execution,
            trace={"path": SWARM_TRACE_FILE, "records": rec.last_seq, "mission_id": rec.mission_id},
            answer=answer,
        )
        write_json(out / "coordination.json", coordinated.as_dict())
        # The fleet's answer where a single mission's is: goal_analysis.json,
        # and the executive report a reader opens first.
        write_json(out / GOAL_ANALYSIS_FILE, answer)
        (out / EXECUTIVE_REPORT_FILE).write_text(_executive_report(coordinated), encoding="utf-8")
        on_progress("converge",
                    f"{len(shards)} sub-Avalokas merged — {converged_profile['n_rows']:,} rows, "
                    f"verdict={verdict}")
        return coordinated

    # -- convergence rules -------------------------------------------------
    @staticmethod
    def _merge_verdicts(outcomes: list[ShardOutcome]) -> str:
        """Worst-case wins: any shard that fails fails the fleet."""
        # No outcomes at all is no evidence, and that is not a pass either.
        worst = max((_VERDICT_RANK.get(o.verdict, _UNVERIFIED_RANK) for o in outcomes),
                    default=_UNVERIFIED_RANK)
        return {0: "pass", 1: "warn", 2: "fail"}[worst]

    @staticmethod
    def _cap_sampled_answer(verdict: str, answer: dict[str, Any]) -> str:
        """A merged answer estimated from samples is at most WARN.

        Each clone's Validator reconciles her figures against the rows she
        answered from, so a sample reconciles with itself and passes. Nothing
        reconciled the merged figures against every row, so they have not
        earned PASS; the answer records why the verdict was lowered."""
        if answer.get("sampled") and _VERDICT_RANK.get(verdict, _UNVERIFIED_RANK) < _VERDICT_RANK["warn"]:
            answer["verdict_cap"] = ("WARN, not PASS: the merged answer is estimated from the clones' "
                                     "samples and was not reconciled against a computation over every row.")
            return "warn"
        return verdict

    @staticmethod
    def _select_model(outcomes: list[ShardOutcome], metric: str | None) -> dict[str, Any] | None:
        """Pick the shard model with the best honest score (lower is better for error metrics)."""
        scored = [o for o in outcomes if o.model and o.model.get("score") is not None]
        if not scored:
            return None
        lower_better = (metric or "").lower() in {"rmse", "mae"}
        best = min(scored, key=lambda o: o.model["score"]) if lower_better \
            else max(scored, key=lambda o: o.model["score"])
        chosen = dict(best.model)
        chosen["from_shard"] = best.shard_id
        chosen["candidates_considered"] = len(scored)
        return chosen


# --------------------------------------------------------------------------
def _require_target(target: str, columns: list[str] | None) -> None:
    """Refuse a training swarm whose target is not a column; ``None`` means
    the columns are not known yet, and the check runs again once they are."""
    if columns is not None and target not in columns:
        raise ValueError(
            f"target column {target!r} was not found; available columns: "
            + ", ".join(map(str, columns))
        )


def _align_partition_stats(parts: list[dict[str, Any]], paths: dict[int, Path],
                           shard_ids: list[int]) -> list[dict[str, Any]]:
    """Make the clones' partition statistics combine as one frame's would.

    Each clone types its own shard, so a column with text far down the file
    is numeric in the shards that hold none of it, and an all-empty column is
    dropped from the shards where it is empty. Combined as they are, the
    first shard's reading would win (or a missing key would raise). Here a
    column missing from a shard counts as all missing there, and a column
    that any shard read as text is text in every shard: the shards that read
    it as numbers have that column's value counts taken again from their
    shard files, as strings, which is what the whole frame would have held.
    """
    if not parts:
        return parts
    names: list[str] = []
    for part in parts:
        names.extend(name for name in part["columns"] if name not in names)
    text = {name for name in names
            if any(name in p["columns"] and not p["columns"][name].get("is_numeric") for p in parts)}
    aligned = []
    for part, shard_id in zip(parts, shard_ids):
        cols = dict(part["columns"])
        recount = [name for name in text if name in cols and cols[name].get("is_numeric")]
        if recount:
            frame = load_dataset(str(paths[shard_id])).frame
            for name in recount:
                values = frame[name].dropna().astype(str)
                cols[name] = {"missing": cols[name]["missing"], "count": cols[name]["count"],
                              "is_numeric": False,
                              "top": {str(k): int(n) for k, n in values.value_counts().head(20).items()}}
            del frame
        for name in names:
            if name not in cols:
                empty = {"missing": part["n_rows"], "count": 0, "is_numeric": name not in text}
                empty.update({"sum": 0.0, "sumsq": 0.0, "min": math.inf, "max": -math.inf}
                             if name not in text else {"top": {}})
                cols[name] = empty
        aligned.append({**part, "columns": {name: cols[name] for name in names}})
    return aligned


def _goal_contributions(outcomes: list[ShardOutcome], shards: list[ShardFile]) -> list[dict[str, Any]]:
    """Each clone's part of the answer, with the source rows she was given
    (shards are contiguous, in order, so the ranges follow from the counts)."""
    starts, at = {}, 0
    for shard in sorted(shards, key=lambda s: s.shard_id):
        starts[shard.shard_id] = (at, at + shard.n_rows - 1)
        at += shard.n_rows
    return [{"name": clone_name_for(o.shard_id),
             "rows": list(starts.get(o.shard_id, (0, o.n_rows - 1))),
             "analysis": o.goal_analysis, "partials": o.goal_partials} for o in outcomes]


def _executive_report(cr: CoordinationResult) -> str:
    """The fleet's executive report: the answer first, then how it was made."""
    lines = [f"# Executive report — {cr.goal}", "",
             answer_markdown(cr.answer) if cr.answer else "", "## The fleet", "",
             f"{cr.workers} sub-Avalokas each analysed one contiguous shard of {cr.n_rows:,} rows; "
             f"the converged verdict is **{cr.verdict.upper()}** (worst case of the clones; safe to "
             f"deployment level {cr.max_safe_deployment_level}/4).", ""]
    for s in cr.shards:
        lines.append(f"- Ava-{s.shard_id:02d}: {s.n_rows:,} rows, verdict {s.verdict or '—'} — "
                     f"her own report: `sub-avaloka-{s.shard_id:02d}/mission/`")
    lines += ["", "Measured values and the merge: `goal_analysis.json`; the full record: "
              "`coordination.json`.", ""]
    return "\n".join(lines)


def converged_mean(profile: dict[str, Any], column: str) -> float | None:
    """Convenience: pull a numeric column's converged mean from a coordination profile."""
    for col in profile.get("columns", []):
        if col["name"] == column and col.get("role") == "numeric":
            return col["mean"]
    return None


def load_coordination(output_dir: str | Path) -> dict[str, Any]:
    """Read back a written ``coordination.json`` report."""
    return json.loads((Path(output_dir) / "coordination.json").read_text())
