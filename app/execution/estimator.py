"""Workload + cost estimator — Avaloka's economical-data-scientist core.

Implements a first, transparent version of the vision's

    W = f(B, R, C, T, K, A, P, L, Q, $)

where B=bytes, R=rows, C=columns, T=type/transform complexity,
K=cardinality/skew, A=analysis type, P=privacy, L=latency objective,
Q=confidence target, $=budget. The numbers here are deliberately simple,
documented heuristics calibrated against the vision's worked examples
(≈84M rows ⇒ ≈$14 / ≈17 min full pass). They are meant to be replaced by
measured coefficients later — the *contract* (DatasetStats → WorkloadEstimate)
is the stable part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from app.missions.schema import DataMission, ExecutionMode, GoalType


# --- tunable coefficients (v1 heuristic) -----------------------------------
_USD_PER_MILLION_ROWS = 0.17       # full distributed scan, baseline complexity
_ROWS_PER_MINUTE_FULL = 5_000_000  # full distributed throughput
_INTERACTIVE_ROW_BUDGET = 2_000_000  # rows a single interactive pass handles
_SAMPLED_FLOOR_USD = 0.10          # fixed overhead of building/scoring a sample
_RANGE = 0.20                      # ± band reported on point estimates

# analysis-type multipliers (the "A" term): model training dominates cost
_ANALYSIS_MULT = {
    GoalType.ETL: 1.0,
    GoalType.DATA_QUALITY: 1.2,
    GoalType.ANALYSIS: 1.0,
    GoalType.PREDICTIVE_MODEL: 4.0,
    GoalType.INFERENCE: 0.5,
}


@dataclass
class DatasetStats:
    """What inspection knows about the data before committing real compute.

    All optional: a cold ``avaloka plan`` with only a row estimate still
    produces a usable (wider) estimate.
    """

    rows: Optional[int] = None
    columns: Optional[int] = None
    bytes: Optional[int] = None
    type_complexity: float = 0.0      # T, 0=simple numerics … 1=text/images
    cardinality_skew: float = 0.0     # K, 0=balanced … 1=rare-class/long-tail
    rare_class_fraction: Optional[float] = None  # feeds full-pass value logic


@dataclass
class WorkloadEstimate:
    recommended_mode: ExecutionMode
    workload_score: float
    full_cost_usd: Tuple[float, float]
    full_runtime_minutes: Tuple[float, float]
    sampled_cost_usd: Tuple[float, float]
    sampled_rows: Optional[int]
    full_pass_worthwhile: bool
    rationale: str
    notes: list = field(default_factory=list)


def _band(point: float) -> Tuple[float, float]:
    return (round(point * (1 - _RANGE), 2), round(point * (1 + _RANGE), 2))


def _complexity_mult(stats: DatasetStats) -> float:
    # type complexity adds up to 3x; skew adds up to 0.5x (heavier sampling).
    return 1.0 + 3.0 * max(0.0, min(stats.type_complexity, 1.0)) \
        + 0.5 * max(0.0, min(stats.cardinality_skew, 1.0))


def _sample_rows(stats: DatasetStats, mission: DataMission) -> Optional[int]:
    """Pick a defensible sample size (the 'Q' term).

    Honours an explicit ``sampling.sample_size``; otherwise targets the
    requested margin of error, falling back to a 1.5% stratified preview with a
    sane floor/ceiling. Conservative, documented — not a substitute for the
    sampling specialist agent that will own this later.
    """
    s = mission.spec.sampling
    if s.sample_size:
        return s.sample_size
    if stats.rows is None:
        return None
    if s.margin_error:
        # n ≈ (z/2E)^2 for a proportion at z=1.96 (95%); clamp to dataset size.
        n = int((0.98 / s.margin_error) ** 2)
    else:
        n = max(50_000, int(stats.rows * 0.015))
    return min(stats.rows, max(50_000, n))


def estimate_workload(mission: DataMission, stats: Optional[DatasetStats] = None) -> WorkloadEstimate:
    stats = stats or DatasetStats()
    goal = mission.spec.goal.type
    analysis_mult = _ANALYSIS_MULT.get(goal, 1.0)
    cmult = _complexity_mult(stats)
    notes: list = []

    rows = stats.rows
    if rows is None:
        notes.append("No row count available; estimate is a wide upper bound.")

    eff_rows = (rows or _INTERACTIVE_ROW_BUDGET) * cmult * analysis_mult
    workload_score = eff_rows / _INTERACTIVE_ROW_BUDGET

    full_cost_point = (eff_rows / 1_000_000) * _USD_PER_MILLION_ROWS
    full_runtime_point = eff_rows / _ROWS_PER_MINUTE_FULL

    sampled_rows = _sample_rows(stats, mission)
    if sampled_rows and rows:
        frac = sampled_rows / rows
        sampled_cost_point = max(_SAMPLED_FLOOR_USD, full_cost_point * frac)
    else:
        sampled_cost_point = _SAMPLED_FLOOR_USD

    full_cost = _band(full_cost_point)
    sampled_cost = _band(sampled_cost_point)
    full_runtime = _band(full_runtime_point)

    recommended, full_worthwhile, rationale = _recommend(
        mission, stats, workload_score, full_cost, sampled_cost
    )

    return WorkloadEstimate(
        recommended_mode=recommended,
        workload_score=round(workload_score, 3),
        full_cost_usd=full_cost,
        full_runtime_minutes=(round(full_runtime[0], 1), round(full_runtime[1], 1)),
        sampled_cost_usd=sampled_cost,
        sampled_rows=sampled_rows,
        full_pass_worthwhile=full_worthwhile,
        rationale=rationale,
        notes=notes,
    )


def _recommend(mission, stats, score, full_cost, sampled_cost):
    """Translate workload + constraints + budget into a recommended mode.

    Encodes the vision's rule: prefer the cheapest plan that satisfies the
    quality objective, and only run a full pass when its marginal value is real
    (rare classes underrepresented, exhaustive validation requested, or it is
    cheap relative to the budget).
    """
    ex = mission.spec.execution
    val = mission.spec.validation
    budget = ex.max_cost_usd

    # honour an explicit, non-auto mode choice.
    if ex.mode is not ExecutionMode.AUTO:
        worthwhile = ex.mode in (ExecutionMode.HYBRID, ExecutionMode.BATCH)
        return ex.mode, worthwhile, f"Mode '{ex.mode.value}' was explicitly requested."

    rare = (stats.rare_class_fraction is not None and stats.rare_class_fraction < 0.05)
    exhaustive = (
        val.level.value == "deployment"
        or val.detect_leakage
        or mission.spec.goal.type is GoalType.DATA_QUALITY
    )
    full_is_cheap = budget is not None and full_cost[1] <= budget * 0.5

    # small enough to just do fully and interactively.
    if score <= 1.0:
        return ExecutionMode.INTERACTIVE, False, (
            "Workload fits an interactive full pass; no sampling needed."
        )

    # large work that genuinely benefits from (or requires) the full data.
    if rare or exhaustive or full_is_cheap:
        reasons = []
        if rare:
            reasons.append("rare class is underrepresented")
        if exhaustive:
            reasons.append("exhaustive validation requested")
        if full_is_cheap:
            reasons.append("full pass is cheap relative to budget")
        # if it also fits the budget, go straight to batch; else preview first.
        mode = (
            ExecutionMode.BATCH
            if budget is not None and full_cost[1] <= budget
            else ExecutionMode.HYBRID
        )
        return mode, True, "Full data is worthwhile (" + "; ".join(reasons) + ")."

    # large work where a sample is statistically sufficient and far cheaper.
    return ExecutionMode.SAMPLED, False, (
        "A representative sample answers this question; the full pass is not "
        "worth the additional cost yet."
    )
