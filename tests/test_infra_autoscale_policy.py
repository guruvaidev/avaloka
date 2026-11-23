"""Autoscaling: the decision, the dead band, and the refusal to thrash."""

import pytest

from app.infra.autoscale_policy import (ClusterState, ScalingAction, ScalingConstraints,
                                        idle_seconds_before_zero_pays, monthly_cost,
                                        recommend_scale, workers_for_working_set)

READY = {"seconds_since_last_change": 10_000.0}


def busy_state(**kw):
    base = dict(workers=2, utilisation_samples=(0.9, 0.92, 0.88), running_tasks=4, **READY)
    base.update(kw)
    return ClusterState(**base)


def idle_state(**kw):
    base = dict(workers=4, utilisation_samples=(0.0, 0.0, 0.0), idle_seconds=3600.0, **READY)
    base.update(kw)
    return ClusterState(**base)


# --------------------------------------------------------------------------- #
# Guards that run before any scaling logic
# --------------------------------------------------------------------------- #

def test_cooldown_blocks_even_an_extreme_signal():
    state = ClusterState(workers=2, utilisation_samples=(1.0, 1.0, 1.0),
                         pending_tasks=99, seconds_since_last_change=10.0)
    d = recommend_scale(state)
    assert d.action is ScalingAction.HOLD
    assert "cooldown" in d.reason


def test_one_sample_is_not_a_trend():
    state = ClusterState(workers=2, utilisation_samples=(0.99,), **READY)
    d = recommend_scale(state)
    assert d.action is ScalingAction.HOLD
    assert "sample" in d.reason


def test_a_dead_band_is_mandatory():
    """Thresholds that touch produce a cluster that oscillates."""
    with pytest.raises(ValueError, match="dead band"):
        ScalingConstraints(scale_up_utilisation=0.5, scale_down_utilisation=0.5)


def test_utilisation_inside_the_dead_band_does_nothing():
    state = ClusterState(workers=3, utilisation_samples=(0.5, 0.55, 0.5), **READY)
    d = recommend_scale(state)
    assert d.action is ScalingAction.HOLD
    assert "dead band" in d.reason
    assert d.cost_delta_per_hour == 0.0


# --------------------------------------------------------------------------- #
# Scaling up
# --------------------------------------------------------------------------- #

def test_saturation_scales_up_and_prices_it():
    d = recommend_scale(busy_state())
    assert d.action is ScalingAction.SCALE_UP
    assert d.target_workers > 2
    assert d.cost_delta_per_hour > 0, "adding workers costs money and must say so"


def test_queued_work_scales_up_even_at_low_utilisation():
    """Idle workers with a queue means the queue cannot reach them."""
    state = ClusterState(workers=1, utilisation_samples=(0.1, 0.1, 0.1),
                         pending_tasks=20, **READY)
    d = recommend_scale(state)
    assert d.action is ScalingAction.SCALE_UP
    assert "queued" in d.reason


def test_scale_up_respects_the_ceiling():
    d = recommend_scale(busy_state(workers=10), ScalingConstraints(max_workers=10))
    assert d.action is ScalingAction.HOLD
    assert "ceiling" in d.reason


# --------------------------------------------------------------------------- #
# Scaling down and to zero
# --------------------------------------------------------------------------- #

def test_idle_cluster_scales_to_zero_and_reports_the_saving():
    d = recommend_scale(idle_state())
    assert d.action is ScalingAction.SCALE_TO_ZERO
    assert d.target_workers == 0
    assert d.cost_delta_per_hour < 0


def test_briefly_idle_does_not_scale_to_zero():
    """Zero then a cold start 30 seconds later costs time and saves nothing."""
    d = recommend_scale(idle_state(idle_seconds=30.0))
    assert d.action is ScalingAction.HOLD
    assert "cold start" in d.reason


def test_scale_to_zero_can_be_forbidden():
    c = ScalingConstraints(allow_scale_to_zero=False, min_workers=1)
    d = recommend_scale(idle_state(), c)
    assert d.action is not ScalingAction.SCALE_TO_ZERO


def test_min_workers_prevents_zero_even_when_idle():
    d = recommend_scale(idle_state(), ScalingConstraints(min_workers=2))
    assert d.action is not ScalingAction.SCALE_TO_ZERO
    assert d.target_workers >= 2


def test_low_utilisation_scales_down():
    state = ClusterState(workers=8, utilisation_samples=(0.2, 0.15, 0.2),
                         running_tasks=1, **READY)
    d = recommend_scale(state)
    assert d.action is ScalingAction.SCALE_DOWN
    assert d.target_workers < 8
    assert d.cost_delta_per_hour < 0


def test_never_scales_out_from_under_running_work():
    state = ClusterState(workers=4, utilisation_samples=(0.1, 0.1, 0.1),
                         running_tasks=4, **READY)
    d = recommend_scale(state)
    assert d.action is ScalingAction.HOLD
    assert "in flight" in d.reason


def test_running_work_blocks_scale_to_zero_regardless_of_idle_time():
    state = ClusterState(workers=2, utilisation_samples=(0.0, 0.0, 0.0),
                         running_tasks=1, idle_seconds=99_999.0, **READY)
    assert recommend_scale(state).action is not ScalingAction.SCALE_TO_ZERO


# --------------------------------------------------------------------------- #
# Memory floor beats utilisation
# --------------------------------------------------------------------------- #

def test_memory_floor_prevents_a_scale_down_that_would_oom():
    state = ClusterState(workers=4, utilisation_samples=(0.1, 0.1, 0.1), **READY)
    d = recommend_scale(state, memory_bound_workers=4)
    assert d.action is not ScalingAction.SCALE_DOWN


def test_below_the_memory_floor_it_scales_up_even_when_idle():
    state = ClusterState(workers=1, utilisation_samples=(0.05, 0.05, 0.05), **READY)
    d = recommend_scale(state, memory_bound_workers=4)
    assert d.action is ScalingAction.SCALE_UP
    assert d.target_workers == 4
    assert "OOM" in d.reason


def test_memory_floor_blocks_scale_to_zero():
    d = recommend_scale(idle_state(), memory_bound_workers=2)
    assert d.action is not ScalingAction.SCALE_TO_ZERO


@pytest.mark.parametrize("working_set,per_worker,expected", [
    (30.0, 8.0, 4), (8.0, 8.0, 1), (0.5, 8.0, 1), (0.0, 8.0, 1), (1000.0, 8.0, 10),
])
def test_workers_for_working_set(working_set, per_worker, expected):
    assert workers_for_working_set(working_set, per_worker, max_workers=10) == expected


# --------------------------------------------------------------------------- #
# No oscillation
# --------------------------------------------------------------------------- #

def test_a_decision_is_stable_when_immediately_re_evaluated():
    """Applying a recommendation must not immediately recommend the reverse."""
    c = ScalingConstraints(cooldown_seconds=0.0)
    state = ClusterState(workers=8, utilisation_samples=(0.2, 0.2, 0.2),
                         running_tasks=1, seconds_since_last_change=10_000.0)
    down = recommend_scale(state, c)
    assert down.action is ScalingAction.SCALE_DOWN

    # Same total work, now spread over fewer workers: utilisation rises but must
    # not immediately cross the up threshold, or the cluster flaps.
    scaled = ClusterState(workers=down.target_workers,
                          utilisation_samples=(0.2 * 8 / down.target_workers,) * 3,
                          running_tasks=1, seconds_since_last_change=10_000.0)
    back = recommend_scale(scaled, c)
    assert back.action is not ScalingAction.SCALE_UP, "the dead band must absorb this"


def test_cost_of_leaving_a_cluster_running_is_stated_not_implied():
    assert monthly_cost(3) == pytest.approx(3 * 0.067 * 730, rel=1e-6)
    assert monthly_cost(0) == 0.0


def test_idle_threshold_scales_with_cold_start():
    slow = ScalingConstraints(cold_start_seconds=600.0)
    fast = ScalingConstraints(cold_start_seconds=30.0)
    assert idle_seconds_before_zero_pays(slow, 4) > idle_seconds_before_zero_pays(fast, 4)


def test_every_decision_serialises_with_its_reason():
    d = recommend_scale(busy_state())
    out = d.as_dict()
    assert out["action"] and out["reason"] and "cost_delta_per_hour" in out


def test_utilisation_alone_blocks_scale_to_zero_when_counters_are_stale():
    """A cluster at 50% is working, whatever its task counters say."""
    state = ClusterState(workers=3, utilisation_samples=(0.5, 0.55, 0.5),
                         running_tasks=0, pending_tasks=0,
                         idle_seconds=99_999.0, seconds_since_last_change=10_000.0)
    d = recommend_scale(state)
    assert d.action is not ScalingAction.SCALE_TO_ZERO, (
        "task counters can be stale; utilisation is independent evidence of work")
