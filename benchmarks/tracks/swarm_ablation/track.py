"""Track 4 tasks, ablation configs, and suite registration.

Reuses the Track-1 analytical tasks (same synthetic dataset + full-data reference) but
runs them through the local reference swarm (``local_swarm`` adapter) under eight agent
configurations. A governance-aware evaluator layers ``leakage`` and
``unsupported_claims`` on top of the base Track-1 quality so the diagnostics are
visible in the scorecard:

    single agent        -> no sampling: scans everything (cost/data up)
    - validator         -> leakage / higher answer error
    - reviewer          -> unsupported_claims > 0
    - planner           -> tool_calls & retries up
    - memory            -> reproducibility (cross-trial std) down
    - adaptive sampling -> fixed random sample instead of adaptive
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ...core.config import RunConfig
from ...core.registry import register_suite
from ...core.task import BenchmarkTask, Evaluator, Split
from ...prompts import FULL_SWARM, render as render_prompt
from ..adaptive_sampling.datasets import imbalanced_transactions, reference_for
from ..adaptive_sampling.track import _fraud_recall_evaluator, _mean_evaluator


def _governance_aware(base: Evaluator) -> Evaluator:
    """Wrap a Track-1 evaluator to also surface swarm governance diagnostics."""

    def _eval(task: BenchmarkTask, answer: Any) -> Dict[str, float]:
        q = base(task, answer)
        if isinstance(answer, dict):
            q["leakage"] = float(answer.get("_leakage", 0))
            q["unsupported_claims"] = float(answer.get("_unsupported_claims", 0))
        return q

    return _eval


def _tasks() -> List[BenchmarkTask]:
    ds = imbalanced_transactions()
    return [
        BenchmarkTask(
            id="t4_mean_amount",
            track="swarm_ablation",
            mission=render_prompt("mean_estimate", target_column="amount", tolerance=0.03),
            dataset=ds,
            evaluator=_governance_aware(_mean_evaluator(ds)),
            reference=reference_for(ds)["mean_amount"],
            split=Split.DEV,
            params={"operation": "mean", "target_column": "amount",
                    "stratify_column": "region", "tolerance": 0.03},
            tags=("means", "ablation"),
        ),
        BenchmarkTask(
            id="t4_fraud_recall",
            track="swarm_ablation",
            mission=render_prompt("rare_event_recall", rare_label_column="is_fraud"),
            dataset=ds,
            evaluator=_governance_aware(_fraud_recall_evaluator(ds)),
            reference=reference_for(ds)["fraud_rate"],
            split=Split.DEV,
            params={"operation": "rare_recall", "rare_label_column": "is_fraud",
                    "id_column": "txn_id", "min_recall": 0.8},
            tags=("rare_event", "ablation"),
        ),
    ]


# The eight configurations from benchmarks.md §4 Track 4. ``agents_enabled=None`` means
# the full swarm (exercises the default path); the "no_X" variants subtract one role.
def _ablations() -> List[Tuple[str, frozenset | None, str]]:
    return [
        ("single_general", frozenset({"coder"}), "adaptive"),
        ("general_tools", frozenset({"coder", "sampling"}), "adaptive"),
        ("specialized_only", frozenset({"coder", "sampling", "validator"}), "adaptive"),
        ("full_swarm", None, "adaptive"),
        ("no_reviewer", FULL_SWARM - {"reviewer"}, "adaptive"),
        ("no_planner", FULL_SWARM - {"planner"}, "adaptive"),
        ("no_memory", FULL_SWARM - {"memory"}, "adaptive"),
        ("no_adaptive_sampling", FULL_SWARM, "random"),
    ]


def _configs(trials: int = 5) -> List[RunConfig]:
    configs = []
    for name, agents, sampling_mode in _ablations():
        configs.append(RunConfig(
            id=f"t4_{name}",
            sut="local_swarm",
            sampling_mode=sampling_mode,
            environment="local",
            llm="local-reference-swarm",
            agents_enabled=agents,
            trials=trials,
            extra={"sample_fraction": "0.05", "target_precision": "0.02",
                   "max_fraction": "0.5", "min_rare_positives": "40"},
        ))
    return configs


@register_suite(
    "track4_swarm_ablation",
    kind="track",
    description="Agent & swarm ablation (local, no cloud): toggles planner/validator/"
                "reviewer/memory/sampling to show each agent's contribution to "
                "quality, cost, leakage, and tool calls.",
)
def build_track4() -> Tuple[List[BenchmarkTask], List[RunConfig]]:
    return _tasks(), _configs()
