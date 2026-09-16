"""Mission orchestration.

A mission activates exactly the fireflies its kind requires, in dependency
order, then closes the books: it computes the economic report, writes
``economics.json`` and ``mission.json`` and returns a structured result. The
ordering mirrors the planner's execution graph; fireflies activate only when
their expertise is needed (analysis missions never wake the Model Scientist or
ML Engineer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from avaloka.fireflies import (AnalysisPlanner, DataEngineer, DataScout, FinOps,
                               MLEngineer, ModelScientist, Reporter,
                               SamplingSpecialist, Validator)
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.mission.economics import EconomicReport, compute_economics
from avaloka.util import write_json

ProgressFn = Callable[[str, str], None]


@dataclass
class MissionResult:
    context: MissionContext
    economics: EconomicReport
    summary: dict[str, Any]


def _noop(_a: str, _b: str) -> None:  # pragma: no cover
    pass


def _run(ctx: MissionContext, fireflies: list[type], on_progress: ProgressFn) -> MissionResult:
    ctx.ensure_output()
    if ctx.blackboard.get("workload"):
        write_json(ctx.path("workload.json"), ctx.blackboard["workload"])
    for cls in fireflies:
        firefly = cls(ctx)
        on_progress("start", cls.name)
        unit = firefly.activate()
        on_progress("done", f"{cls.name}: {unit.output}")

    econ = compute_economics(ctx.ledger, tier=ctx.tier,
                             loaded_hourly_rate=ctx.loaded_hourly_rate,
                             wall_clock_seconds=ctx.wall_clock_seconds())

    validation = ctx.blackboard.get("validation", {})
    summary = {
        "mission_id": ctx.mission_id,
        "kind": ctx.kind.value,
        "goal": ctx.goal,
        "data_source": ctx.data_source,
        "output_dir": str(ctx.output_dir),
        "task": ctx.blackboard.get("plan", {}).get("task"),
        "workload": ctx.blackboard.get("workload"),
        "verdict": validation.get("verdict"),
        "max_safe_deployment_level": validation.get("max_safe_deployment_level"),
        "economics": econ.as_dict(),
        "fireflies": ctx.ledger.as_list(),
        "budget": {"limit_usd": ctx.budget.limit_usd, "spent_usd": round(ctx.budget.spent_usd, 4)},
    }
    if ctx.blackboard.get("training_summary"):
        ts = ctx.blackboard["training_summary"]
        # The baseline travels with the score. A metric reported on its own
        # invites the reader to decide whether 0.71 is good, which is exactly
        # the judgement the baseline exists to make for them.
        summary["model"] = {"selected": ts["selected"], "metric": ts["metric"],
                            "score": ts["selected_metrics"].get(ts["metric"]),
                            "baseline_score": ts.get("baseline_score"),
                            "baseline_strategy": ts.get("baseline_strategy"),
                            "beats_baseline": ts.get("beats_baseline")}

    write_json(ctx.path("economics.json"), econ.as_dict())
    write_json(ctx.path("mission.json"), summary)
    return MissionResult(context=ctx, economics=econ, summary=summary)


def analyze_fireflies() -> list[type]:
    return [DataScout, SamplingSpecialist, DataEngineer, AnalysisPlanner, Validator, Reporter]


def train_fireflies(deployable: bool) -> list[type]:
    fireflies = [DataScout, SamplingSpecialist, DataEngineer, AnalysisPlanner,
                 ModelScientist, Validator, FinOps]
    if deployable:
        fireflies.append(MLEngineer)
    fireflies.append(Reporter)
    return fireflies


def run_analyze(ctx: MissionContext, on_progress: ProgressFn = _noop) -> MissionResult:
    """Release 1 — Avaloka Analyze. Produces the analysis deliverable bundle."""
    ctx.kind = MissionKind.ANALYZE
    return _run(ctx, analyze_fireflies(), on_progress)


def run_train(ctx: MissionContext, on_progress: ProgressFn = _noop) -> MissionResult:
    """Release 2 — Avaloka Predict. Adds modelling, cost-quality and packaging."""
    ctx.kind = MissionKind.TRAIN
    return _run(ctx, train_fireflies(ctx.deployable), on_progress)
