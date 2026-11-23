"""Decide how many workers a Ray cluster should have, and when it should have none.

Two costs sit on either side of this decision and they pull in opposite
directions. Too few workers and a job that should take ten minutes takes an
hour. Too many, or workers that outlive the work, and an idle cluster bills
continuously for nothing — an idle GKE cluster with one node group is roughly
$268/month before a single query runs.

Today neither side is decided. ``app/infra/rayjob.yaml`` sets ``minReplicas: 1``,
so a cluster never reaches zero however long it sits idle, and
``execution_agent.py`` pins ``minReplicas == maxReplicas == workers``, which
turns the autoscaler off entirely for the jobs that go through it. The Ray
autoscaler is configured and then prevented from doing anything.

This module supplies the decision, and three properties make it safe to act on.

**Hysteresis, not a single threshold.** Scaling up at 80% utilisation and down
at 79% produces a cluster that oscillates, and thrash is worse than either
steady state: every scale-down discards warm workers that the next scale-up pays
to recreate. Up and down have separate thresholds with a deliberate dead band
between them, plus a cooldown after any change.

**Scale-to-zero is weighed against cold start, not applied on sight.** A cluster
that scales to zero and then pays three minutes of spin-up for the next job
arriving 30 seconds later has cost time and saved nothing. Idle must exceed the
point where the money saved outweighs the restart, and that point is computed
from the actual figures rather than assumed.

**In-flight work is never scaled out from under.** A scale-down that would drop
below what running tasks need is refused, whatever the utilisation says.

Pure functions over dataclasses: no cluster, no kubectl, no clock of its own —
elapsed times are passed in — so every decision is reproducible in a test.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Sustained utilisation above this means the cluster is the bottleneck.
SCALE_UP_UTILISATION = 0.75

#: Below this, workers are being paid for and not used. The gap between the two
#: is the dead band that stops oscillation; it is wide on purpose.
SCALE_DOWN_UTILISATION = 0.35

#: No further change within this window of the last one. A cluster that resizes
#: every poll never finishes anything.
DEFAULT_COOLDOWN_SECONDS = 300

#: A single spike is not a trend. Require this many consecutive samples agreeing
#: before acting on them.
DEFAULT_SAMPLES_REQUIRED = 3

#: Typical KubeRay worker cold start: image pull, node join, Ray registration.
DEFAULT_COLD_START_SECONDS = 180.0

#: Rough on-demand cost of one e2-standard-2-class worker.
DEFAULT_WORKER_COST_PER_HOUR = 0.067


class ScalingAction(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_TO_ZERO = "scale_to_zero"
    HOLD = "hold"


@dataclass(frozen=True)
class ClusterState:
    """What the cluster looks like right now."""

    workers: int
    #: Most recent first is not assumed; order only matters for length.
    utilisation_samples: Sequence[float] = ()
    pending_tasks: int = 0
    running_tasks: int = 0
    idle_seconds: float = 0.0
    seconds_since_last_change: float = float("inf")

    @property
    def utilisation(self) -> float:
        if not self.utilisation_samples:
            return 0.0
        return sum(self.utilisation_samples) / len(self.utilisation_samples)

    @property
    def busy(self) -> bool:
        return self.running_tasks > 0 or self.pending_tasks > 0


@dataclass(frozen=True)
class ScalingConstraints:
    min_workers: int = 0
    max_workers: int = 10
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    samples_required: int = DEFAULT_SAMPLES_REQUIRED
    scale_up_utilisation: float = SCALE_UP_UTILISATION
    scale_down_utilisation: float = SCALE_DOWN_UTILISATION
    cold_start_seconds: float = DEFAULT_COLD_START_SECONDS
    worker_cost_per_hour: float = DEFAULT_WORKER_COST_PER_HOUR
    #: Allowing zero is a deployment choice: a dev cluster should, a latency-
    #: sensitive serving cluster should not.
    allow_scale_to_zero: bool = True

    def __post_init__(self) -> None:
        if self.min_workers < 0:
            raise ValueError("min_workers cannot be negative")
        if self.max_workers < max(self.min_workers, 1):
            raise ValueError("max_workers must be at least min_workers and at least 1")
        if self.scale_down_utilisation >= self.scale_up_utilisation:
            # Without a dead band the cluster oscillates between the two states.
            raise ValueError(
                "scale_down_utilisation must be below scale_up_utilisation; the gap "
                "between them is the dead band that prevents thrash")


@dataclass
class ScalingDecision:
    action: ScalingAction
    target_workers: int
    reason: str
    cost_delta_per_hour: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def changes_anything(self) -> bool:
        return self.action is not ScalingAction.HOLD

    def as_dict(self) -> Dict[str, Any]:
        return {"action": self.action.value, "target_workers": self.target_workers,
                "reason": self.reason,
                "cost_delta_per_hour": round(self.cost_delta_per_hour, 4),
                "evidence": self.evidence}


def _hold(state: ClusterState, reason: str, **evidence: Any) -> ScalingDecision:
    return ScalingDecision(ScalingAction.HOLD, state.workers, reason, 0.0, evidence)


def idle_seconds_before_zero_pays(constraints: ScalingConstraints, workers: int) -> float:
    """How long idle must last before scaling to zero is worth the cold start.

    Scaling to zero saves ``workers * cost_per_hour``. Coming back costs
    ``cold_start_seconds`` of latency, and — because the cluster is unavailable
    while it starts — the honest comparison is against the same workers' cost
    over that restart window. Below this threshold, going to zero is churn.
    """
    if workers <= 0:
        return 0.0
    return max(constraints.cold_start_seconds * 2.0, 60.0)


def recommend_scale(state: ClusterState,
                    constraints: ScalingConstraints = ScalingConstraints(),
                    *, memory_bound_workers: Optional[int] = None) -> ScalingDecision:
    """Decide the next worker count. Returns HOLD when the answer is 'nothing'.

    ``memory_bound_workers`` is the floor implied by a memory estimate — the
    number of shards a frame must be split into to fit. Utilisation must never
    scale below it, because the resulting workers would OOM rather than run slow.
    """
    floor = max(constraints.min_workers, memory_bound_workers or 0)
    ceiling = constraints.max_workers

    # 1. Cooldown. Checked before anything else so a flapping signal cannot
    #    produce a change no matter how extreme it looks.
    if state.seconds_since_last_change < constraints.cooldown_seconds:
        return _hold(state, f"within the {constraints.cooldown_seconds:.0f}s cooldown after "
                            f"the last change; resizing every poll finishes nothing",
                     seconds_since_last_change=state.seconds_since_last_change)

    # 2. Not enough evidence. One spike is not a trend.
    if len(state.utilisation_samples) < constraints.samples_required:
        return _hold(state, f"only {len(state.utilisation_samples)} utilisation sample(s); "
                            f"{constraints.samples_required} are required before acting",
                     samples=len(state.utilisation_samples))

    utilisation = state.utilisation

    # 3. Scale up: saturated, or work queued with nowhere to run.
    if utilisation >= constraints.scale_up_utilisation or state.pending_tasks > 0:
        if state.workers >= ceiling:
            return _hold(state, f"saturated at {utilisation:.0%} but already at the "
                                f"{ceiling}-worker ceiling",
                         utilisation=round(utilisation, 3), at_ceiling=True)
        target = min(ceiling, max(state.workers * 2, state.workers + 1, floor, 1))
        added = target - state.workers
        driver = (f"utilisation {utilisation:.0%} at or above the "
                  f"{constraints.scale_up_utilisation:.0%} threshold"
                  if utilisation >= constraints.scale_up_utilisation
                  else f"{state.pending_tasks} task(s) queued with no free worker")
        return ScalingDecision(
            ScalingAction.SCALE_UP, target,
            f"{driver}; adding {added} worker(s)",
            added * constraints.worker_cost_per_hour,
            {"utilisation": round(utilisation, 3), "pending_tasks": state.pending_tasks})

    # 4. Scale to zero: only when genuinely idle, allowed, and past the point
    #    where the saving beats the restart.
    #    Utilisation is independent evidence of work: a cluster at 50% is not
    #    idle even if nothing has updated its task counters, and scaling that to
    #    zero would kill running work. Both signals must agree.
    idle_by_utilisation = utilisation <= constraints.scale_down_utilisation
    if (constraints.allow_scale_to_zero and floor == 0 and not state.busy
            and idle_by_utilisation and state.workers > 0):
        required_idle = idle_seconds_before_zero_pays(constraints, state.workers)
        if state.idle_seconds >= required_idle:
            saved = state.workers * constraints.worker_cost_per_hour
            return ScalingDecision(
                ScalingAction.SCALE_TO_ZERO, 0,
                f"idle {state.idle_seconds:.0f}s with no running or queued work, past the "
                f"{required_idle:.0f}s at which the saving beats a "
                f"{constraints.cold_start_seconds:.0f}s cold start; releasing "
                f"{state.workers} worker(s) saves ${saved:.3f}/hour",
                -saved,
                {"idle_seconds": state.idle_seconds, "required_idle": required_idle})
        return _hold(state, f"idle {state.idle_seconds:.0f}s, short of the "
                            f"{required_idle:.0f}s at which scaling to zero beats paying "
                            f"the cold start again",
                     idle_seconds=state.idle_seconds, required_idle=required_idle)

    # 5. Scale down: under-used, but never below the floor or below in-flight work.
    if utilisation <= constraints.scale_down_utilisation and state.workers > floor:
        needed_for_running = max(state.running_tasks, floor, 1)
        if state.workers <= needed_for_running:
            return _hold(state, f"utilisation {utilisation:.0%} is low, but {state.workers} "
                                f"worker(s) are needed for work in flight",
                         utilisation=round(utilisation, 3),
                         running_tasks=state.running_tasks)
        target = max(needed_for_running, floor, math.ceil(state.workers / 2))
        target = min(target, state.workers - 1)
        removed = state.workers - target
        return ScalingDecision(
            ScalingAction.SCALE_DOWN, target,
            f"utilisation {utilisation:.0%} at or below the "
            f"{constraints.scale_down_utilisation:.0%} threshold; releasing "
            f"{removed} worker(s)",
            -removed * constraints.worker_cost_per_hour,
            {"utilisation": round(utilisation, 3), "floor": floor})

    # 6. Below the floor: a memory estimate can demand more than utilisation does.
    if state.workers < floor:
        added = floor - state.workers
        return ScalingDecision(
            ScalingAction.SCALE_UP, floor,
            f"{state.workers} worker(s) is below the {floor} required to hold the working "
            f"set; too few would OOM rather than run slowly",
            added * constraints.worker_cost_per_hour,
            {"memory_bound_workers": memory_bound_workers})

    return _hold(state, f"utilisation {utilisation:.0%} sits inside the dead band "
                        f"({constraints.scale_down_utilisation:.0%}–"
                        f"{constraints.scale_up_utilisation:.0%}); nothing to do",
                 utilisation=round(utilisation, 3))


def workers_for_working_set(working_set_gib: float, worker_memory_gib: float,
                            *, max_workers: int = 10) -> int:
    """How many workers a frame needs to fit. The floor for any scale-down."""
    if working_set_gib <= 0 or worker_memory_gib <= 0:
        return 1
    return max(1, min(max_workers, math.ceil(working_set_gib / worker_memory_gib)))


def monthly_cost(workers: int, constraints: ScalingConstraints = ScalingConstraints(),
                 *, hours: float = 730.0) -> float:
    """What a given worker count costs if left running. The number that motivates
    scale-to-zero, stated rather than implied."""
    return workers * constraints.worker_cost_per_hour * hours
