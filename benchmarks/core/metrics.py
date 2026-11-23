"""Metric aggregation and the reusable statistical primitives.

Two layers live here:

1. *Cross-trial aggregation* — mean/median/std, Pass@1, Best-of-N, and the
   ``(Quality, Cost, Latency, Data Scanned)`` 4-tuple that every result must report
   (design rules #2 and #3).
2. *Reusable quality primitives* — answer error, CI coverage, rare-event recall,
   distribution divergence — imported by track evaluators so the definitions live in
   exactly one place.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Sequence

from .task import TaskResult, TrialResult


# --------------------------------------------------------------------------- #
# Cross-trial aggregation
# --------------------------------------------------------------------------- #

def _summary(values: Sequence[float]) -> Dict[str, float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def aggregate(trials: List[TrialResult]) -> TaskResult:
    """Aggregate trials of one task/config into a :class:`TaskResult`.

    We never reduce to a single "best" number: Pass@1 and Best-of-N are reported
    side by side, and every quality metric plus the cost/latency/data 4-tuple gets a
    full mean/median/std/min/max summary.
    """
    if not trials:
        raise ValueError("aggregate() requires at least one trial")

    task_id, track, config_id = trials[0].task_id, trials[0].track, trials[0].config_id
    n = len(trials)

    # Pass@1 = fraction of trials that passed. Best-of-N = did any trial pass.
    passes = [t.passed for t in trials]
    pass_at_1 = sum(passes) / n
    best_of_n = 1.0 if any(passes) else 0.0

    # Per-quality-key summaries across trials.
    quality_keys = sorted({k for t in trials for k, value in t.quality.items()
                           if isinstance(value, (int, float)) and not isinstance(value, bool)})
    quality_stats = {k: _summary([t.quality.get(k, float("nan")) for t in trials])
                     for k in quality_keys}

    outcomes = [t.outcome for t in trials]
    tuple_stats = {
        "cost_usd": _summary([o.cost_usd for o in outcomes if o.cost_usd is not None]),
        "latency_s": _summary([o.latency_s for o in outcomes]),
        "data_scanned_ratio": _summary(
            [o.data_scanned_ratio for o in outcomes if o.data_scanned_ratio is not None]
        ),
        "tool_calls": _summary([o.tool_calls for o in outcomes]),
        "tokens": _summary([o.tokens for o in outcomes]),
        "prompt_tokens": _summary([o.prompt_tokens for o in outcomes]),
        "completion_tokens": _summary([o.completion_tokens for o in outcomes]),
        "retries": _summary([o.retries for o in outcomes]),
    }
    errors = [o.error for o in outcomes if o.error]
    failure_kinds = [o.error_kind.value for o in outcomes if o.error_kind]

    # Cost provenance travels with the cost. A mean of $0.0000 means something
    # very different when every trial was "unpriced" than when it was "catalog",
    # and a scorecard that shows only the number cannot distinguish them.
    cost_sources = sorted({o.cost_source for o in outcomes if o.cost_source})
    # cost_usd is None when the cost is unknown. That is not $0: leave it out of
    # the total and count it, rather than summing it as zero or crashing.
    total_cost_usd = sum(o.cost_usd for o in outcomes if o.cost_usd is not None)
    n_cost_unknown = sum(1 for o in outcomes if o.cost_usd is None)

    stats = {
        "n_trials": n,
        "pass_at_1": pass_at_1,
        "best_of_n": best_of_n,
        "quality": quality_stats,
        "tuple": tuple_stats,          # the (Quality, Cost, Latency, Data Scanned) 4-tuple
        "total_cost_usd": total_cost_usd,
        "n_cost_unknown": n_cost_unknown,
        "cost_sources": cost_sources,
        "n_errors": len(errors),
        "errors": errors[:5],
        "failure_kinds": {kind: failure_kinds.count(kind)
                          for kind in sorted(set(failure_kinds))},
    }
    return TaskResult(task_id=task_id, track=track, config_id=config_id,
                      trials=trials, stats=stats)


def cost_quality_score(quality: float, cost_usd: float, *, cost_ref: float = 1.0) -> float:
    """A blunt cost-quality scalar used for router-regret style comparisons.

    Rewards quality, penalises spend relative to a reference cost. Deliberately
    simple and monotone; tracks that need a different trade-off define their own.
    """
    return quality - (cost_usd / cost_ref if cost_ref else 0.0)


# --------------------------------------------------------------------------- #
# Reusable quality primitives (imported by track evaluators)
# --------------------------------------------------------------------------- #

def relative_error(estimate: float, truth: float) -> float:
    """|estimate - truth| / |truth| (falls back to absolute error when truth == 0)."""
    denom = abs(truth)
    return abs(estimate - truth) / denom if denom > 1e-12 else abs(estimate - truth)


def quality_from_error(rel_err: float) -> float:
    """Map a relative error to a [0, 1] quality score. 0 error -> 1.0."""
    return max(0.0, 1.0 - rel_err)


def ci_covers(true_value: float, ci_low: float, ci_high: float) -> float:
    """1.0 if the true value lies inside the reported confidence interval, else 0.0."""
    return 1.0 if ci_low <= true_value <= ci_high else 0.0


def rare_event_recall(retained: Sequence[Any], population: Sequence[Any]) -> float:
    """Fraction of the rare/important population retained by a sample.

    ``retained`` and ``population`` are identifier collections for the rare segment.
    """
    pop = set(population)
    if not pop:
        return 1.0
    return len(pop & set(retained)) / len(pop)


def population_stability_index(sample_dist: Dict[Any, float],
                               pop_dist: Dict[Any, float],
                               eps: float = 1e-6) -> float:
    """Population Stability Index between a sample and population distribution.

    Standard divergence proxy for "how different is the sample from the population".
    Lower is better; 0 means identical. Both inputs are category -> proportion maps.
    """
    keys = set(sample_dist) | set(pop_dist)
    psi = 0.0
    for k in keys:
        s = max(sample_dist.get(k, 0.0), eps)
        p = max(pop_dist.get(k, 0.0), eps)
        psi += (s - p) * math.log(s / p)
    return psi
