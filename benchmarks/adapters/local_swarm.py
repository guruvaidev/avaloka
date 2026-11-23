"""Local, cloud-free reference swarm for Track-4 ablation.

Track 4 (benchmarks.md §4) must prove that Avaloka's *multiple agents provide real
value rather than organizational theater*. Running the real graph for that needs Groq
+ cloud, which we deliberately avoid for the default, local-first suite (the user's
"test everything locally so we don't use cloud first").

So this adapter is a **deterministic local model of the swarm** over the same Track-1
analytical tasks. It toggles exactly the agent roles in ``benchmarks.prompts.AGENT_ROLES``
and reproduces the expected diagnostics from the strategy table:

    Sampling removed   -> full scan (cost/data up) or rare groups disappear
    Validator removed  -> leakage / statistical error up (answer quality down)
    Reviewer removed   -> unsupported conclusions appear
    Planner removed    -> tool calls & retries increase
    Memory removed     -> reproducibility drops (per-trial variance)

It is NOT the real Avaloka. When cloud is available, the ``avaloka`` adapter runs the
genuine graph under the same configs; this local swarm is the offline yardstick and CI
guardrail. ``requires_cloud = False`` so it always runs by default.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, List, Tuple

from . import register_adapter
from .baselines import _COST_PER_ROW_USD, _SEC_PER_ROW, _records, _run_operation
from ..core.config import RunConfig
from ..core.task import BenchmarkTask, MissionOutcome
from ..prompts import FULL_SWARM

# Each tool call costs a little money — so removing the Planner (more tool calls)
# shows up as higher cost, not just a higher count.
_COST_PER_TOOL_USD = 2e-3
# Without the Validator, leakage/statistical error inflates a numeric estimate.
_LEAKAGE_INFLATION = 1.12


def _select_sample(task: BenchmarkTask, config: RunConfig, agents: frozenset,
                   rng: random.Random) -> Tuple[List[Dict[str, Any]], bool]:
    """Sampling agent's job. Returns (sample, escalated)."""
    recs = _records(task)
    total = len(recs)
    if "sampling" not in agents:
        # No sampling agent -> scan everything (the cost/data blow-up we want to show).
        return recs, False

    frac = float(config.extra.get("sample_fraction", 0.05))
    adaptive = config.sampling_mode == "adaptive"
    order = recs[:]
    rng.shuffle(order)

    if not adaptive:
        # Sampling present but not adaptive -> fixed random fraction.
        k = max(1, int(total * frac))
        return order[:k], False

    # Adaptive: grow until CI half-width / rare-positive floor is met (reference policy).
    op = task.params["operation"]
    target_precision = float(config.extra.get("target_precision", 0.02))
    max_fraction = float(config.extra.get("max_fraction", 0.5))
    batch = max(50, int(total * 0.01))
    seen: List[Dict[str, Any]] = []
    escalated = False
    i = 0
    while i < total and len(seen) < total * max_fraction:
        seen.extend(order[i:i + batch])
        i += batch
        if op == "mean":
            vals = [float(r[task.params["target_column"]]) for r in seen]
            if len(vals) > 30:
                mean = statistics.fmean(vals)
                se = statistics.pstdev(vals) / math.sqrt(len(vals))
                rel = (1.96 * se) / abs(mean) if abs(mean) > 1e-9 else float("inf")
                if rel <= target_precision:
                    break
                escalated = True
        elif op == "rare_recall":
            rare_seen = sum(1 for r in seen if r.get(task.params["rare_label_column"]))
            if rare_seen >= int(config.extra.get("min_rare_positives", 40)):
                break
            escalated = True
    return seen, escalated


@register_adapter("local_swarm")
class LocalSwarmAdapter:
    """Deterministic offline model of the Avaloka swarm (see module docstring)."""

    key = "local_swarm"
    requires_cloud = False

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        agents = config.agents_enabled if config.agents_enabled is not None else FULL_SWARM
        trial = int(config.extra.get("_trial", 0))

        # Memory agent -> reproducible across trials (seed ignores the trial index).
        # Without it, each trial draws differently, so aggregate std (reproducibility)
        # is worse.
        seed_salt = 0 if "memory" in agents else trial
        rng = random.Random(config.seed * 100003 + seed_salt)

        total = len(_records(task))
        sample, escalated = _select_sample(task, config, agents, rng)
        answer = _run_operation(task.params["operation"], sample, task)
        answer["n_scanned"] = len(sample)

        # Validator absent -> inject a leakage/statistical error into numeric answers.
        leakage = 0
        if "validator" not in agents:
            leakage = 1
            if "estimate" in answer:
                answer["estimate"] *= _LEAKAGE_INFLATION
            if "rate_estimate" in answer:
                answer["rate_estimate"] *= _LEAKAGE_INFLATION
        answer["_leakage"] = leakage

        # Reviewer absent -> unsupported conclusions slip through.
        answer["_unsupported_claims"] = 0 if "reviewer" in agents else 2

        # Planner present -> tight plan; absent -> more tool calls and retries.
        if "planner" in agents:
            tool_calls, retries = 3, 0
        else:
            tool_calls, retries = 9, 3

        n = len(sample)
        cost = n * _COST_PER_ROW_USD + tool_calls * _COST_PER_TOOL_USD
        latency = n * _SEC_PER_ROW + tool_calls * 0.01

        return MissionOutcome(
            answer=answer,
            cost_usd=cost,
            latency_s=latency,
            data_scanned_ratio=(n / total) if total else None,
            tool_calls=tool_calls,
            retries=retries,
            recovered=None,
            trajectory=[{"agents_enabled": sorted(agents),
                         "sampling_mode": config.sampling_mode,
                         "escalated": escalated}],
            artifacts={"leakage": leakage,
                       "unsupported_claims": answer["_unsupported_claims"]},
        )
