"""Track 5 tasks, fault scenarios, and suite registration.

One analytical mission (mean estimation) run under a matrix of injected failures and
recovery regimes. Reliability is orthogonal to the analytical task, so a single task
keeps the scenario matrix readable; the interesting axis is the fault + recovery config.

Scenarios encode the core claim from benchmarks.md §4 Track 5:
* transient faults recover whenever retry/checkpointing is on;
* structural faults need a specific mitigation — a blind retry does not save them;
* every run, recovered or not, must still inform the user.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ...core.config import Budget, RunConfig
from ...core.registry import register_suite
from ...core.task import BenchmarkTask, Evaluator, Split
from ...prompts import render as render_prompt
from ..adaptive_sampling.datasets import imbalanced_transactions, reference_for
from ..adaptive_sampling.track import _mean_evaluator


def _reliability_evaluator(dataset) -> Evaluator:
    base = _mean_evaluator(dataset)

    def _eval(task: BenchmarkTask, answer: Any) -> Dict[str, float]:
        if not isinstance(answer, dict):
            return {"pass": 0.0, "score": 0.0}
        fault = answer.get("_fault", "none")
        recovered = answer.get("_recovered")  # True / False / None(no fault)
        has_answer = answer.get("estimate") is not None

        q: Dict[str, float] = base(task, answer) if has_answer else {
            "score": 0.0, "answer_rel_error": float("nan"), "ci_coverage": 0.0}

        q["recovered"] = 1.0 if recovered or recovered is None else 0.0
        q["work_repeated_ratio"] = float(answer.get("_work_repeated_ratio", 0.0))
        q["cost_wasted_usd"] = float(answer.get("_cost_wasted_usd", 0.0))
        q["user_informed"] = 1.0 if answer.get("_user_informed") else 0.0

        if fault == "none":
            q["pass"] = 1.0 if q.get("score", 0.0) >= 0.5 else 0.0
        else:
            # A fault scenario passes only if the run recovered AND produced a good
            # analytical answer.
            q["pass"] = 1.0 if (recovered and q.get("score", 0.0) >= 0.5) else 0.0
        return q

    return _eval


def _tasks() -> List[BenchmarkTask]:
    ds = imbalanced_transactions()
    return [
        BenchmarkTask(
            id="t5_mean_amount",
            track="reliability",
            mission=render_prompt("mean_estimate", target_column="amount", tolerance=0.03),
            dataset=ds,
            evaluator=_reliability_evaluator(ds),
            reference=reference_for(ds)["mean_amount"],
            split=Split.DEV,
            params={"operation": "mean", "target_column": "amount", "tolerance": 0.03},
            tags=("means", "reliability"),
        ),
    ]


# (config-id-suffix, fault, recovery on/off, mitigations)
_SCENARIOS: List[Tuple[str, str, str, str]] = [
    ("no_fault", "none", "off", ""),
    ("transient_no_recovery", "rate_limit", "off", ""),
    ("transient_with_recovery", "rate_limit", "on", ""),
    ("worker_crash_with_recovery", "worker_crash", "on", ""),
    ("structural_blind_retry", "schema_change", "on", ""),
    ("structural_with_mitigation", "schema_change", "on", "schema_change"),
    ("corrupt_parquet_no_recovery", "corrupt_parquet", "off", ""),
    ("corrupt_parquet_with_mitigation", "corrupt_parquet", "on", "corrupt_parquet"),
]


def _configs(trials: int = 5) -> List[RunConfig]:
    configs = []
    for suffix, fault, recovery, mitigations in _SCENARIOS:
        configs.append(RunConfig(
            id=f"t5_{suffix}",
            sut="fault_injecting",
            sampling_mode="fixed",
            environment="local",
            llm="local-fault-injection",
            trials=trials,
            budget=Budget(),
            extra={"sample_fraction": "0.2", "fault": fault,
                   "recovery": recovery, "mitigations": mitigations},
        ))
    return configs


@register_suite(
    "track5_reliability",
    kind="track",
    description="Reliability & recovery (local fault injection, no cloud): schema "
                "drift / crashes / corrupt data / rate limits under retry+checkpoint "
                "vs mitigation-aware recovery. Measures recovered, work repeated, cost "
                "wasted, user informed.",
)
def build_track5() -> Tuple[List[BenchmarkTask], List[RunConfig]]:
    return _tasks(), _configs()
