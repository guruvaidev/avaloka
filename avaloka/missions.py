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

from avaloka.fireflies import (AnalysisPlanner, ConformalCalibrator, DataEngineer,
                               DataScout, Experimenter, FeatureEngineer, FinOps,
                               MLEngineer, ModelScientist, Reporter,
                               SamplingSpecialist, Statistician, Validator,
                               WorkloadOptimizer)
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


def _specialists_due(ctx: MissionContext, queue: list[type]) -> list[type]:
    """The analytical specialists the blackboard now calls for and the queue lacks.

    The planner writes ``statistical_plan``, ``experiment_plan`` and
    ``conformal_config`` while the mission is already running, so the
    firefly list drawn up at the start cannot know about them; they are
    slotted in ahead of the Validator at the moment it is reached.
    """
    wanted = [(Statistician, "statistical_plan"), (Experimenter, "experiment_plan"),
              (ConformalCalibrator, "conformal_config")]
    return [cls for cls, key in wanted
            if ctx.blackboard.get(key) is not None and cls not in queue]


def _run(ctx: MissionContext, fireflies: list[type], on_progress: ProgressFn) -> MissionResult:
    ctx.ensure_output()
    if ctx.blackboard.get("workload"):
        write_json(ctx.path("workload.json"), ctx.blackboard["workload"])
    queue = list(fireflies)
    index = 0
    while index < len(queue):
        if queue[index] is Validator:
            queue[index:index] = _specialists_due(ctx, queue)
        cls = queue[index]
        firefly = cls(ctx)
        on_progress("start", cls.name)
        unit = firefly.activate()
        on_progress("done", f"{cls.name}: {unit.output}")
        index += 1

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
        "workload_optimization": ctx.blackboard.get("workload_optimization"),
        "verdict": validation.get("verdict"),
        "max_safe_deployment_level": validation.get("max_safe_deployment_level"),
        "economics": econ.as_dict(),
        "fireflies": ctx.ledger.as_list(),
        "budget": {"limit_usd": ctx.budget.limit_usd, "spent_usd": round(ctx.budget.spent_usd, 4)},
    }
    if ctx.blackboard.get("goal_analysis"):
        summary["goal_analysis"] = {
            "kind": ctx.blackboard["goal_analysis"]["kind"],
            "status": ctx.blackboard["goal_analysis"]["status"],
            "scope": ctx.blackboard["goal_analysis"]["scope"],
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
                            "beats_baseline": ts.get("beats_baseline"),
                            "ensemble_vs_single": ts.get("ensemble_vs_single")}
    # The specialists' verdicts travel with the summary for the same reason
    # the baseline does: a lift without its interval and its integrity check
    # is a number, not a finding.
    experiment = ctx.blackboard.get("experiment_result")
    if experiment is not None:
        summary["experiment"] = {
            "status": experiment.status, "effect": experiment.effect,
            "relative_effect": experiment.relative_effect,
            "confidence_interval": experiment.confidence_interval,
            "sample_ratio_mismatch": experiment.sample_ratio_mismatch,
            "power": experiment.power,
            "control_n": experiment.control_n, "treatment_n": experiment.treatment_n,
        }
    statistic = ctx.blackboard.get("statistical_result")
    if statistic is not None:
        summary["statistic"] = {"method": statistic.method, "estimate": statistic.estimate,
                                "p_value": statistic.p_value,
                                "confidence_interval": statistic.confidence_interval,
                                "sample_size": statistic.sample_size}
    conformal = ctx.blackboard.get("conformal_result")
    if conformal is not None:
        summary["conformal"] = {"task": conformal.task, "target_coverage": conformal.target_coverage,
                                "empirical_coverage": conformal.empirical_coverage,
                                "efficiency": conformal.efficiency}

    write_json(ctx.path("economics.json"), econ.as_dict())
    write_json(ctx.path("mission.json"), summary)
    return MissionResult(context=ctx, economics=econ, summary=summary)


def analyze_fireflies() -> list[type]:
    return [DataScout, SamplingSpecialist, WorkloadOptimizer, DataEngineer, FeatureEngineer,
            AnalysisPlanner, Validator, Reporter]


def train_fireflies(deployable: bool) -> list[type]:
    fireflies = [DataScout, SamplingSpecialist, WorkloadOptimizer, DataEngineer,
                 FeatureEngineer, AnalysisPlanner,
                 ModelScientist, Validator, FinOps]
    if deployable:
        fireflies.append(MLEngineer)
    fireflies.append(Reporter)
    return fireflies


def run_analyze(ctx: MissionContext, on_progress: ProgressFn = _noop) -> MissionResult:
    """Release 1 — Avaloka Analyze. Produces the analysis deliverable bundle."""
    ctx.kind = MissionKind.ANALYZE
    fireflies = analyze_fireflies()
    validator_index = fireflies.index(Validator)
    if ctx.blackboard.get("statistical_plan") is not None:
        fireflies.insert(validator_index, Statistician)
        validator_index += 1
    if ctx.blackboard.get("experiment_plan") is not None:
        fireflies.insert(validator_index, Experimenter)
    return _run(ctx, fireflies, on_progress)


def run_train(ctx: MissionContext, on_progress: ProgressFn = _noop) -> MissionResult:
    """Release 2 — Avaloka Predict. Adds modelling, cost-quality and packaging."""
    ctx.kind = MissionKind.TRAIN
    fireflies = train_fireflies(ctx.deployable)
    if ctx.blackboard.get("conformal_config") is not None:
        fireflies.insert(fireflies.index(Validator), ConformalCalibrator)
    return _run(ctx, fireflies, on_progress)
