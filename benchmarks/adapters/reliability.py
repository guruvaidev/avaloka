"""Local fault-injection adapter for Track-5 (Reliability & Recovery).

Track 5 (benchmarks.md §4) argues a production data-science agent must be evaluated on
*degraded* environments, not ideal ones. Doing that against the real graph would need
cloud/Groq; per the local-first policy this adapter models the mission + a fault
deterministically so the recovery story is measurable offline.

It runs a Track-1 analytical operation over a sample and injects one failure mid-scan.
Whether the run recovers depends on two things — matching how the real system behaves:

* **Transient faults** (rate limit, worker crash, DB timeout, expired signed URL) are
  recoverable whenever retry/checkpointing is enabled.
* **Structural faults** (schema change, missing partition, corrupt Parquet) recover
  only if a *specific mitigation* is configured — a blind retry loops on them.

A handled-but-failed run is **data, not a crash**: we never set ``outcome.error`` for an
injected fault (that is reserved for real adapter bugs). Instead the answer carries
recovery diagnostics so the evaluator can score recovery even when the analytical
answer is missing. ``requires_cloud = False`` so it always runs by default.
"""

from __future__ import annotations

import random
import statistics
from typing import Any, Dict, List

from . import register_adapter
from .baselines import _COST_PER_ROW_USD, _SEC_PER_ROW, _records, _run_operation
from ..core.config import RunConfig
from ..core.task import BenchmarkTask, MissionOutcome

TRANSIENT_FAULTS = {"rate_limit", "worker_crash", "db_timeout", "expired_url"}
STRUCTURAL_FAULTS = {"schema_change", "missing_partition", "corrupt_parquet"}
ALL_FAULTS = TRANSIENT_FAULTS | STRUCTURAL_FAULTS

# Where in the scan the fault fires (mid-execution) and how often we checkpoint.
_FAULT_FRACTION = 0.55
_CHECKPOINT_INTERVAL = 0.10
_RETRY_OVERHEAD_S = 0.05


@register_adapter("fault_injecting")
class FaultInjectingAdapter:
    key = "fault_injecting"
    requires_cloud = False

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        recs = _records(task)
        total = len(recs)
        trial = int(config.extra.get("_trial", 0))
        rng = random.Random(config.seed * 100003 + trial)

        # A modest fixed sample is enough to compute a good answer when we complete.
        frac = float(config.extra.get("sample_fraction", 0.2))
        k = max(1, int(total * frac))
        order = recs[:]
        rng.shuffle(order)
        sample = order[:k]

        fault = config.extra.get("fault", "none")
        recovery_on = config.extra.get("recovery", "off") == "on"
        mitigations = {m for m in config.extra.get("mitigations", "").split(",") if m}

        base_cost = k * _COST_PER_ROW_USD
        base_latency = k * _SEC_PER_ROW

        if fault == "none":
            answer = _answer(task, sample)
            answer.update(_recovery_fields(recovered=None, work_repeated=0.0,
                                           cost_wasted=0.0, user_informed=True,
                                           fault="none"))
            return MissionOutcome(answer=answer, cost_usd=base_cost,
                                  latency_s=base_latency,
                                  data_scanned_ratio=k / total if total else None,
                                  tool_calls=3, recovered=None,
                                  trajectory=[{"fault": "none"}])

        if fault not in ALL_FAULTS:
            return MissionOutcome(answer=None,
                                  error=f"config_error: unknown fault {fault!r}; "
                                        f"known {sorted(ALL_FAULTS)}")

        can_recover = recovery_on and (fault in TRANSIENT_FAULTS or fault in mitigations)
        # Work already done when the fault fires.
        work_before = _FAULT_FRACTION * k

        if can_recover:
            # Resume from the last checkpoint: redo only the work since that checkpoint.
            repeated = (_FAULT_FRACTION % _CHECKPOINT_INTERVAL) * k
            answer = _answer(task, sample)  # completes successfully after recovery
            cost_wasted = repeated * _COST_PER_ROW_USD
            latency = base_latency + repeated * _SEC_PER_ROW + _RETRY_OVERHEAD_S
            answer.update(_recovery_fields(
                recovered=True, work_repeated=repeated / k,
                cost_wasted=cost_wasted, user_informed=True, fault=fault))
            return MissionOutcome(
                answer=answer, cost_usd=base_cost + cost_wasted, latency_s=latency,
                data_scanned_ratio=(k + repeated) / total if total else None,
                tool_calls=4, retries=1, recovered=True,
                trajectory=[{"fault": fault, "recovered": True,
                             "checkpoint_interval": _CHECKPOINT_INTERVAL}])

        # Unrecovered: the run reports a failure (still informs the user). Work up to
        # the fault is wasted; there is no analytical answer.
        cost_wasted = work_before * _COST_PER_ROW_USD
        answer = {"estimate": None, "rate_estimate": None, "retained_ids": [],
                  "n_scanned": int(work_before)}
        answer.update(_recovery_fields(
            recovered=False, work_repeated=0.0, cost_wasted=cost_wasted,
            user_informed=True, fault=fault))
        return MissionOutcome(
            answer=answer, cost_usd=work_before * _COST_PER_ROW_USD,
            latency_s=work_before * _SEC_PER_ROW,
            data_scanned_ratio=work_before / total if total else None,
            tool_calls=4, retries=1 if recovery_on else 0, recovered=False,
            trajectory=[{"fault": fault, "recovered": False,
                         "reason": "no_mitigation" if recovery_on else "no_recovery"}])


def _answer(task: BenchmarkTask, sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    ans = _run_operation(task.params["operation"], sample, task)
    ans["n_scanned"] = len(sample)
    return ans


def _recovery_fields(*, recovered, work_repeated, cost_wasted, user_informed,
                     fault) -> Dict[str, Any]:
    return {
        "_recovered": recovered,
        "_work_repeated_ratio": work_repeated,
        "_cost_wasted_usd": cost_wasted,
        "_user_informed": user_informed,
        "_fault": fault,
    }
