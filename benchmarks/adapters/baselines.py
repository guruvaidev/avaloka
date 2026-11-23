"""Baseline systems-under-test for Track 1 (Adaptive Sampling).

These implement the six comparison strategies from ``benchmarks.md`` §4 Track 1 as
pure-Python analytics executors over a tabular dataset. They are *not* Avaloka — they
are the yardsticks Avaloka is measured against, plus one reference implementation of a
confidence-driven adaptive policy.

Why they live here and are dependency-free (stdlib only): the harness and Track 1 must
be runnable and unit-testable in CI without pandas/numpy, cloud creds, or a LangGraph
server. When the real Avaloka Sampling agent (``sample_with_profiling`` in
``app/agents/sampling_agent_daft.py``) is wired through ``avaloka_graph.py``, the
``adaptive_sample`` baseline here becomes the *reference policy* the real agent is
compared against — not a stand-in for it.

Each adapter reads ``task.params['operation']`` and runs it over the sample its
strategy selects, reporting an honest ``data_scanned_ratio`` and a simple linear cost
model so the (Quality, Cost, Latency, Data Scanned) 4-tuple is meaningful.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, List, Sequence, Tuple

from . import register_adapter
from ..core.config import RunConfig
from ..core.task import BenchmarkTask, MissionOutcome

# Simple, transparent cost model: reading a row costs money and time. The absolute
# scale is arbitrary; only *ratios between strategies* are interpreted.
_COST_PER_ROW_USD = 1e-5
_SEC_PER_ROW = 5e-6


def _records(task: BenchmarkTask) -> List[Dict[str, Any]]:
    data = task.dataset.load()
    return data["records"] if isinstance(data, dict) and "records" in data else data


def _rng(config: RunConfig, trial_salt: Any = 0) -> random.Random:
    # Seed varies per trial so repeated trials show real variance (design rule #2)
    # while staying reproducible given (config.seed, trial_salt).
    #
    # trial_salt is coerced rather than trusted: it arrives from config.extra,
    # which any caller can populate. A str once reached the arithmetic below and
    # raised TypeError inside every baseline adapter, which the runner recorded
    # as a failed trial rather than a harness fault -- a silent, plausible-looking
    # zero. Coercing here means a bad caller costs nothing.
    try:
        salt = int(trial_salt)
    except (TypeError, ValueError):
        salt = 0
    return random.Random(config.seed * 100003 + salt)


def _finish(sample: Sequence[Dict[str, Any]], total: int, task: BenchmarkTask,
            strategy: str, extra: Dict[str, Any] | None = None) -> MissionOutcome:
    n = len(sample)
    op = task.params["operation"]
    answer = _run_operation(op, sample, task)
    if extra:
        answer.update(extra)
    answer["n_scanned"] = n
    return MissionOutcome(
        answer=answer,
        cost_usd=n * _COST_PER_ROW_USD,
        latency_s=n * _SEC_PER_ROW,
        data_scanned_ratio=(n / total) if total else None,
        tool_calls=1,
        trajectory=[{"strategy": strategy, "n": n, "total": total}],
    )


def _run_operation(op: str, sample: Sequence[Dict[str, Any]],
                   task: BenchmarkTask) -> Dict[str, Any]:
    if op == "mean":
        col = task.params["target_column"]
        vals = [float(r[col]) for r in sample]
        est = statistics.fmean(vals) if vals else float("nan")
        # Normal-approx 95% CI on the mean — the harness scores whether it covers truth.
        if len(vals) > 1:
            se = statistics.pstdev(vals) / math.sqrt(len(vals))
        else:
            se = float("inf")
        return {"estimate": est, "ci": [est - 1.96 * se, est + 1.96 * se], "se": se}
    if op == "rare_recall":
        label = task.params["rare_label_column"]
        id_col = task.params["id_column"]
        retained = [r[id_col] for r in sample if r.get(label)]
        rate = (len(retained) / len(sample)) if sample else 0.0
        return {"rate_estimate": rate, "retained_ids": retained}
    raise ValueError(f"unknown operation {op!r}")


# --------------------------------------------------------------------------- #
# Strategy adapters
# --------------------------------------------------------------------------- #

class _Base:
    key = "base"

    def _fraction(self, config: RunConfig) -> float:
        return float(config.extra.get("sample_fraction", 0.05))


@register_adapter("full_data_reference")
class FullDataReference(_Base):
    """Scans everything — the ground-truth executor (ratio = 1.0)."""

    key = "full_data_reference"

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        recs = _records(task)
        return _finish(recs, len(recs), task, "full")


@register_adapter("random_sample")
class RandomSample(_Base):
    key = "random_sample"

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        recs = _records(task)
        rng = _rng(config, config.extra.get("_trial", 0))
        k = max(1, int(len(recs) * self._fraction(config)))
        return _finish(rng.sample(recs, min(k, len(recs))), len(recs), task, "random")


@register_adapter("stratified_sample")
class StratifiedSample(_Base):
    """Proportional allocation across a stratification column."""

    key = "stratified_sample"

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        recs = _records(task)
        rng = _rng(config, config.extra.get("_trial", 0))
        strat = task.params.get("stratify_column") or config.extra.get("stratify_column")
        frac = self._fraction(config)
        if not strat:
            k = max(1, int(len(recs) * frac))
            return _finish(rng.sample(recs, min(k, len(recs))), len(recs), task,
                           "stratified(no-col→random)")
        buckets: Dict[Any, List[Dict[str, Any]]] = {}
        for r in recs:
            buckets.setdefault(r.get(strat), []).append(r)
        sample: List[Dict[str, Any]] = []
        for _, rows in buckets.items():
            k = max(1, int(len(rows) * frac))
            sample.extend(rng.sample(rows, min(k, len(rows))))
        return _finish(sample, len(recs), task, "stratified")


@register_adapter("reservoir_sample")
class ReservoirSample(_Base):
    """Fixed-size reservoir (Algorithm R) — one streaming pass, bounded memory."""

    key = "reservoir_sample"

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        recs = _records(task)
        rng = _rng(config, config.extra.get("_trial", 0))
        k = max(1, int(len(recs) * self._fraction(config)))
        reservoir: List[Dict[str, Any]] = []
        for i, r in enumerate(recs):
            if i < k:
                reservoir.append(r)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = r
        return _finish(reservoir, len(recs), task, "reservoir")


@register_adapter("domain_sample")
class DomainSample(_Base):
    """Domain-specific sample: oversample the rare/important segment.

    Represents a hand-tuned sampler that a domain expert would build. For rare-event
    tasks it deliberately keeps the whole rare segment plus a random slice of the rest.
    """

    key = "domain_sample"

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        recs = _records(task)
        rng = _rng(config, config.extra.get("_trial", 0))
        label = task.params.get("rare_label_column")
        frac = self._fraction(config)
        if not label:
            k = max(1, int(len(recs) * frac))
            return _finish(rng.sample(recs, min(k, len(recs))), len(recs), task,
                           "domain(no-label→random)")
        rare = [r for r in recs if r.get(label)]
        common = [r for r in recs if not r.get(label)]
        k_common = max(1, int(len(common) * frac))
        sample = rare + rng.sample(common, min(k_common, len(common)))
        return _finish(sample, len(recs), task, "domain",
                       extra={"_note": "oversampled rare segment"})


@register_adapter("adaptive_sample")
class AdaptiveSample(_Base):
    """Reference confidence-driven adaptive policy.

    THIS IS THE POLICY AVALOKA SHOULD IMPLEMENT (and today does NOT — the shipped
    sampler uses size-tiers + a multi-strategy portfolio with no statistical stopping;
    see benchmarks.md §4 Track 1 "Implementation status"). It is included here as the
    *reference target* the real Sampling agent will be measured against once wired
    through ``avaloka_graph.py``.

    Behaviour: draw in batches; after each batch estimate the standard error of the
    target statistic; stop as soon as the relative CI half-width is within
    ``target_precision`` (default 2%), or a hard ``max_fraction`` cap is hit. For
    rare-event tasks it also guarantees a floor of rare-segment coverage, modelling
    "escalate when the rare signal is under-sampled".
    """

    key = "adaptive_sample"

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        recs = _records(task)
        total = len(recs)
        rng = _rng(config, config.extra.get("_trial", 0))
        order = recs[:]
        rng.shuffle(order)

        target_precision = float(config.extra.get("target_precision", 0.02))
        max_fraction = float(config.extra.get("max_fraction", 0.5))
        batch = max(50, int(total * 0.01))
        op = task.params["operation"]

        seen: List[Dict[str, Any]] = []
        escalated = False
        i = 0
        while i < total and len(seen) < total * max_fraction:
            seen.extend(order[i:i + batch])
            i += batch
            if op == "mean":
                col = task.params["target_column"]
                vals = [float(r[col]) for r in seen]
                if len(vals) > 30:
                    mean = statistics.fmean(vals)
                    se = statistics.pstdev(vals) / math.sqrt(len(vals))
                    half = 1.96 * se
                    rel = half / abs(mean) if abs(mean) > 1e-9 else float("inf")
                    if rel <= target_precision:
                        break
                    escalated = True
            elif op == "rare_recall":
                label = task.params["rare_label_column"]
                rare_seen = sum(1 for r in seen if r.get(label))
                # Escalate until we have enough positives to estimate the rate stably.
                if rare_seen >= int(config.extra.get("min_rare_positives", 40)):
                    break
                escalated = True

        return _finish(seen, total, task, "adaptive", extra={
            "escalated": escalated,
            "stopped_early": len(seen) < total,
        })
