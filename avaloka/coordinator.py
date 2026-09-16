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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from avaloka.io import load_dataset, materialize
from avaloka.mission.budget import Budget
from avaloka.mission.context import ExecutionMode, MissionContext, MissionKind
from avaloka.mission.economics import EconomicReport, compute_economics
from avaloka.mission.ledger import Ledger, WorkUnit
from avaloka.missions import run_analyze, run_train
from avaloka.ray_batch import _combine, _partition_stats
from avaloka.util import write_json

# A shard is dispatched to a sub-Avaloka; this is the callback the main agent
# uses to narrate the fleet (mirrors the per-mission ``on_progress`` API).
FleetProgress = Callable[[str, str], None]

_VERDICT_RANK = {"pass": 0, "warn": 1, "fail": 2, None: 0}


def _noop(_event: str, _msg: str) -> None:  # pragma: no cover
    pass


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
    partition_stats: dict[str, Any] = field(default_factory=dict)


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "goal": self.goal,
            "source": self.source,
            "workers": self.workers,
            "n_rows": self.n_rows,
            "verdict": self.verdict,
            "max_safe_deployment_level": self.max_safe_deployment_level,
            "selected_model": self.selected_model,
            "converged_profile": self.converged_profile,
            "economics": self.economics.as_dict(),
            "shards": [s.__dict__ for s in self.shards],
            "output_dir": self.output_dir,
        }


class MainAvaloka:
    """The coordinating (open-claw) Avaloka.

    She reads the whole dataset once to size and shard it, dispatches a
    sub-Avaloka mission per shard, then converges their outputs. Sub-missions are
    genuinely independent — each reloads only its own shard file — so this models
    a real fleet, not a single process pretending to be many.
    """

    def __init__(self, *, tier: str = "community", loaded_hourly_rate: float = 80.0,
                 execution_mode: ExecutionMode = ExecutionMode.LOCAL) -> None:
        self.tier = tier
        self.loaded_hourly_rate = loaded_hourly_rate
        self.execution_mode = execution_mode

    # -- sharding ----------------------------------------------------------
    def _shard(self, frame: pd.DataFrame, workers: int) -> list[pd.DataFrame]:
        workers = max(1, min(workers, len(frame)))
        idx_parts = np.array_split(np.arange(len(frame)), workers)
        return [frame.iloc[idx].reset_index(drop=True) for idx in idx_parts if len(idx)]

    def _write_shard(self, shard: pd.DataFrame, shard_dir: Path) -> Path:
        shard_dir.mkdir(parents=True, exist_ok=True)
        path = shard_dir / "shard.csv"
        shard.to_csv(path, index=False)
        return path

    # -- coordination ------------------------------------------------------
    def coordinate(
        self,
        source: str,
        *,
        goal: str,
        output_dir: str | Path,
        workers: int = 4,
        kind: MissionKind = MissionKind.ANALYZE,
        target: str | None = None,
        metric: str | None = None,
        deployable: bool = False,
        budget: float | None = None,
        on_progress: FleetProgress = _noop,
    ) -> CoordinationResult:
        """Fan a fleet of sub-Avalokas across ``source`` and converge one answer."""
        resolved = materialize(source)
        handle = load_dataset(str(resolved.local_path))
        frame = handle.frame
        shards = self._shard(frame, workers)

        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        run_fn = run_train if kind is MissionKind.TRAIN else run_analyze

        on_progress("announce", f"{len(shards)}")
        fleet_ledger = Ledger()
        outcomes: list[ShardOutcome] = []
        part_stats: list[dict[str, Any]] = []

        for i, shard in enumerate(shards):
            shard_dir = out / f"sub-avaloka-{i:02d}"
            shard_path = self._write_shard(shard, shard_dir)
            on_progress("dispatch", f"Ava-{i:02d} takes shard {i} ({len(shard):,} rows)")

            ctx = MissionContext(
                kind=kind, goal=goal, data_source=str(shard_path),
                output_dir=shard_dir / "mission",
                budget=Budget(limit_usd=budget), target=target, metric=metric,
                deployable=deployable, execution_mode=self.execution_mode,
                tier=self.tier, loaded_hourly_rate=self.loaded_hourly_rate,
            )
            result = run_fn(ctx)

            # Roll the sub-Avaloka's real ledger into the fleet ledger so the
            # aggregate economics reflect the whole workforce.
            for unit in ctx.ledger.units:
                fleet_ledger.record(unit)

            stats = _partition_stats(shard)
            part_stats.append(stats)
            summary = result.summary
            outcomes.append(ShardOutcome(
                shard_id=i, n_rows=len(shard), output_dir=str(shard_dir / "mission"),
                verdict=summary.get("verdict"),
                max_safe_deployment_level=summary.get("max_safe_deployment_level"),
                manual_minutes=ctx.ledger.manual_minutes,
                cost_usd=round(ctx.ledger.compute_cost_usd + ctx.ledger.model_cost_usd, 6),
                model=summary.get("model"),
                partition_stats=stats,
            ))
            on_progress("report", f"Ava-{i:02d} back — verdict={outcomes[-1].verdict}")

        # -- converge ------------------------------------------------------
        converged_profile = _combine(part_stats)
        verdict = self._merge_verdicts(outcomes)
        max_level = min((o.max_safe_deployment_level or 4) for o in outcomes) if outcomes else 1
        selected = self._select_model(outcomes, metric) if kind is MissionKind.TRAIN else None

        economics = compute_economics(
            fleet_ledger, tier=self.tier, loaded_hourly_rate=self.loaded_hourly_rate)

        coordinated = CoordinationResult(
            kind=kind.value, goal=goal, source=source, workers=len(shards),
            n_rows=converged_profile["n_rows"], converged_profile=converged_profile,
            verdict=verdict, max_safe_deployment_level=max_level, shards=outcomes,
            economics=economics, selected_model=selected, output_dir=str(out),
        )
        write_json(out / "coordination.json", coordinated.as_dict())
        on_progress("converge",
                    f"{len(shards)} sub-Avalokas merged — {converged_profile['n_rows']:,} rows, "
                    f"verdict={verdict}")
        return coordinated

    # -- convergence rules -------------------------------------------------
    @staticmethod
    def _merge_verdicts(outcomes: list[ShardOutcome]) -> str:
        """Worst-case wins: any shard that fails fails the fleet."""
        worst = max((_VERDICT_RANK.get(o.verdict, 0) for o in outcomes), default=0)
        return {0: "pass", 1: "warn", 2: "fail"}[worst]

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
def converged_mean(profile: dict[str, Any], column: str) -> float | None:
    """Convenience: pull a numeric column's converged mean from a coordination profile."""
    for col in profile.get("columns", []):
        if col["name"] == column and col.get("role") == "numeric":
            return col["mean"]
    return None


def load_coordination(output_dir: str | Path) -> dict[str, Any]:
    """Read back a written ``coordination.json`` report."""
    return json.loads((Path(output_dir) / "coordination.json").read_text())
