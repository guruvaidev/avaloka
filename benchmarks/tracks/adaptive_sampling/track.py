"""Track 1 tasks, evaluators, configs, and suite registration.

Two representative tasks (the full Track-1 task matrix from benchmarks.md §4 is much
larger — means/quantiles/group-by/correlation/segment-discovery/anomaly/...):

* ``mean_amount``  — estimate the mean of a long-tailed numeric column.
* ``fraud_recall`` — estimate a rare-event rate and retain the rare segment.

The comparison configs are exactly the six Track-1 strategies, so a single run
produces the headline table: *quality retained vs data scanned vs cost* for
random / stratified / reservoir / domain / adaptive against the full-data reference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ...core.config import RunConfig
from ...core.metrics import (
    ci_covers,
    quality_from_error,
    rare_event_recall,
    relative_error,
)
from ...core.registry import register_suite
from ...core.task import BenchmarkTask, DatasetSpec, Evaluator, Split
from ...prompts import render as render_prompt
from .datasets import imbalanced_transactions, reference_for


# --------------------------------------------------------------------------- #
# Evaluators
# --------------------------------------------------------------------------- #

def _mean_evaluator(dataset: DatasetSpec) -> Evaluator:
    ref = reference_for(dataset)
    truth = ref["mean_amount"]

    def _eval(task: BenchmarkTask, answer: Any) -> Dict[str, float]:
        if not isinstance(answer, dict) or "estimate" not in answer:
            return {"pass": 0.0, "score": 0.0}
        rel = relative_error(answer["estimate"], truth)
        score = quality_from_error(rel)
        ci = answer.get("ci") or [float("nan"), float("nan")]
        coverage = ci_covers(truth, ci[0], ci[1])
        # "pass" = the estimate is within a 3% tolerance of full-data truth.
        return {
            "score": score,
            "answer_rel_error": rel,
            "ci_coverage": coverage,
            "pass": 1.0 if rel <= task.params.get("tolerance", 0.03) else 0.0,
        }

    return _eval


def _fraud_recall_evaluator(dataset: DatasetSpec) -> Evaluator:
    ref = reference_for(dataset)
    truth_rate = ref["fraud_rate"]
    truth_ids = ref["fraud_ids"]

    def _eval(task: BenchmarkTask, answer: Any) -> Dict[str, float]:
        if not isinstance(answer, dict) or "rate_estimate" not in answer:
            return {"pass": 0.0, "score": 0.0}
        rate_rel = relative_error(answer["rate_estimate"], truth_rate)
        recall = rare_event_recall(answer.get("retained_ids", []), truth_ids)
        # Quality blends rate accuracy and rare-segment retention.
        score = 0.5 * quality_from_error(rate_rel) + 0.5 * recall
        return {
            "score": score,
            "rate_rel_error": rate_rel,
            "rare_event_recall": recall,
            "pass": 1.0 if recall >= task.params.get("min_recall", 0.8) else 0.0,
        }

    return _eval


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

def _tasks() -> List[BenchmarkTask]:
    ds = imbalanced_transactions()
    return [
        BenchmarkTask(
            id="t1_mean_amount",
            track="adaptive_sampling",
            mission=render_prompt("mean_estimate", target_column="amount", tolerance=0.03),
            dataset=ds,
            evaluator=_mean_evaluator(ds),
            reference=reference_for(ds)["mean_amount"],
            split=Split.DEV,
            params={"operation": "mean", "target_column": "amount",
                    "stratify_column": "region", "tolerance": 0.03},
            tags=("means", "long_tailed"),
        ),
        BenchmarkTask(
            id="t1_fraud_recall",
            track="adaptive_sampling",
            mission=render_prompt("rare_event_recall", rare_label_column="is_fraud"),
            dataset=ds,
            evaluator=_fraud_recall_evaluator(ds),
            reference=reference_for(ds)["fraud_rate"],
            split=Split.DEV,
            params={"operation": "rare_recall", "rare_label_column": "is_fraud",
                    "id_column": "txn_id", "min_recall": 0.8},
            tags=("rare_event", "anomaly_detection"),
        ),
    ]


# --------------------------------------------------------------------------- #
# Configs — the six Track-1 strategies (benchmarks.md §4 Track 1)
# --------------------------------------------------------------------------- #

_STRATEGY_SUTS = [
    ("full", "full_data_reference"),
    ("random", "random_sample"),
    ("stratified", "stratified_sample"),
    ("reservoir", "reservoir_sample"),
    ("domain", "domain_sample"),
    ("adaptive", "adaptive_sample"),
]


def _configs(trials: int = 5) -> List[RunConfig]:
    configs = []
    for name, sut in _STRATEGY_SUTS:
        configs.append(RunConfig(
            id=f"t1_{name}",
            sut=sut,
            sampling_mode=name,
            environment="local",
            llm="n/a",  # Track 1 baselines are non-LLM; Avaloka config sets its own.
            trials=trials,
            extra={"sample_fraction": "0.05", "stratify_column": "region",
                   "target_precision": "0.02", "max_fraction": "0.5"},
        ))
    return configs


@register_suite(
    "track1_adaptive_sampling",
    kind="track",
    description="Adaptive Sampling: quality retained vs data scanned vs cost across "
                "random/stratified/reservoir/domain/adaptive against full-data truth.",
)
def build_track1() -> Tuple[List[BenchmarkTask], List[RunConfig]]:
    return _tasks(), _configs()
