"""Idle-scaling policy, Spot placement, and the legacy-gateway guard.

The policy functions take an explicit clock, so every scaling decision is
exercised here without a cluster.
"""

from __future__ import annotations

import json

import pytest

from app.agents.mta_v2.inference_autoscale import (AutoscalePolicy,
                                                   RayServiceScaler,
                                                   apply_spot_to_worker_groups,
                                                   idle_seconds, mark_activity,
                                                   read_last_activity,
                                                   should_ramp_up,
                                                   should_scale_to_zero,
                                                   spot_pod_overrides,
                                                   target_replicas)

HOUR = 3600.0
NOW = 1_800_000_000.0


# --------------------------------------------------------------------------- #
# Idle policy
# --------------------------------------------------------------------------- #

def test_scales_to_zero_after_two_idle_hours():
    policy = AutoscalePolicy(idle_timeout_s=2 * HOUR)
    assert should_scale_to_zero(NOW - 2 * HOUR, 2, policy, now=NOW) is True


def test_stays_up_just_under_the_threshold():
    policy = AutoscalePolicy(idle_timeout_s=2 * HOUR)
    assert should_scale_to_zero(NOW - (2 * HOUR - 1), 2, policy, now=NOW) is False


def test_already_at_the_floor_is_not_scaled_again():
    """No repeated patches once the service is down."""
    policy = AutoscalePolicy(idle_timeout_s=2 * HOUR, min_replicas=0)
    assert should_scale_to_zero(NOW - 99 * HOUR, 0, policy, now=NOW) is False


def test_unknown_activity_counts_as_idle():
    """A missing marker must scale down, not pin capacity up forever."""
    assert idle_seconds(None, now=NOW) == float("inf")
    assert should_scale_to_zero(None, 3, AutoscalePolicy(), now=NOW) is True


def test_clock_skew_does_not_produce_negative_idle():
    assert idle_seconds(NOW + 500, now=NOW) == 0.0


def test_ramps_up_from_zero_on_demand():
    policy = AutoscalePolicy(warm_replicas=1)
    assert should_ramp_up(0, policy) is True
    assert should_ramp_up(1, policy) is False
    assert should_ramp_up(4, policy) is False


def test_in_flight_request_beats_an_idle_scale_down():
    """A request arriving at the boundary is served, not cold-started."""
    policy = AutoscalePolicy(idle_timeout_s=2 * HOUR, warm_replicas=1)
    assert target_replicas(NOW - 5 * HOUR, 2, policy, now=NOW, serving_request=True) == 2
    assert target_replicas(NOW - 5 * HOUR, 0, policy, now=NOW, serving_request=True) == 1
    # ...and without a request in flight, the same state scales down.
    assert target_replicas(NOW - 5 * HOUR, 2, policy, now=NOW) == 0


def test_busy_service_is_left_alone():
    policy = AutoscalePolicy(idle_timeout_s=2 * HOUR)
    assert target_replicas(NOW - 60, 3, policy, now=NOW) == 3


# --------------------------------------------------------------------------- #
# Policy validation — misconfiguration fails loudly at construction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kwargs", [
    {"idle_timeout_s": 0},
    {"idle_timeout_s": -1},
    {"min_replicas": -1},
    {"max_replicas": 0},
    {"warm_replicas": 9, "max_replicas": 4},
])
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        AutoscalePolicy(**kwargs)


def test_scale_to_zero_is_the_default_floor():
    """Autopilot bills per running pod, so the idle floor must be zero."""
    assert AutoscalePolicy().min_replicas == 0


def test_default_idle_window_is_two_hours():
    assert AutoscalePolicy().idle_timeout_s == 2 * 60 * 60


# --------------------------------------------------------------------------- #
# Activity marker
# --------------------------------------------------------------------------- #

def test_activity_marker_round_trips(tmp_path):
    p = tmp_path / "activity.json"
    mark_activity(now=NOW, path=p)
    assert read_last_activity(p) == NOW


def test_missing_marker_reads_as_unknown(tmp_path):
    assert read_last_activity(tmp_path / "absent.json") is None


def test_corrupt_marker_reads_as_unknown_rather_than_raising(tmp_path):
    p = tmp_path / "activity.json"
    p.write_text("{ not json", encoding="utf-8")
    assert read_last_activity(p) is None


def test_marking_activity_never_raises_into_the_request_path(tmp_path):
    # A directory where the file should be makes the write fail.
    bad = tmp_path / "activity.json"
    bad.mkdir()
    mark_activity(now=NOW, path=bad)  # must not raise


# --------------------------------------------------------------------------- #
# Spot placement (Autopilot: per-pod, not per-node-pool)
# --------------------------------------------------------------------------- #

def test_spot_overrides_carry_selector_and_matching_toleration():
    o = spot_pod_overrides()
    assert o["nodeSelector"]["cloud.google.com/gke-spot"] == "true"
    tol = o["tolerations"][0]
    assert tol["key"] == "cloud.google.com/gke-spot" and tol["effect"] == "NoSchedule"


def test_workers_move_to_spot_and_the_head_does_not():
    """Preempting the head restarts the whole Ray cluster — leave it on-demand."""
    spec = {
        "headGroupSpec": {"template": {"spec": {"containers": []}}},
        "workerGroupSpecs": [
            {"groupName": "small", "template": {"spec": {"containers": []}}},
            {"groupName": "large", "template": {"spec": {"containers": []}}},
        ],
    }
    out = apply_spot_to_worker_groups(spec)
    for group in out["workerGroupSpecs"]:
        assert group["template"]["spec"]["nodeSelector"]["cloud.google.com/gke-spot"] == "true"
        assert any(t["key"] == "cloud.google.com/gke-spot"
                   for t in group["template"]["spec"]["tolerations"])
    assert "nodeSelector" not in out["headGroupSpec"]["template"]["spec"]


def test_applying_spot_twice_does_not_duplicate_the_toleration():
    spec = {"workerGroupSpecs": [{"template": {"spec": {}}}]}
    once = apply_spot_to_worker_groups(spec)
    twice = apply_spot_to_worker_groups(once)
    assert len(twice["workerGroupSpecs"][0]["template"]["spec"]["tolerations"]) == 1


def test_applying_spot_does_not_mutate_the_input():
    spec = {"workerGroupSpecs": [{"template": {"spec": {}}}]}
    apply_spot_to_worker_groups(spec)
    assert spec["workerGroupSpecs"][0]["template"]["spec"] == {}


def test_existing_node_selectors_are_preserved():
    spec = {"workerGroupSpecs": [
        {"template": {"spec": {"nodeSelector": {"disktype": "ssd"}}}}]}
    out = apply_spot_to_worker_groups(spec)
    sel = out["workerGroupSpecs"][0]["template"]["spec"]["nodeSelector"]
    assert sel["disktype"] == "ssd" and sel["cloud.google.com/gke-spot"] == "true"


# --------------------------------------------------------------------------- #
# Scaler effector — kubectl stubbed
# --------------------------------------------------------------------------- #

class _Result:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_unreadable_replica_count_blocks_any_scaling_action(monkeypatch):
    """Unknown != zero. If we cannot read state, we must not act on it."""
    s = RayServiceScaler()
    monkeypatch.setattr(s, "_kubectl", lambda *a, **k: _Result(rc=1, err="boom"))
    assert s.current_replicas() is None
    assert s.reap_if_idle(now=NOW) is False


def test_non_numeric_replica_output_is_treated_as_unknown(monkeypatch):
    s = RayServiceScaler()
    monkeypatch.setattr(s, "_kubectl", lambda *a, **k: _Result(out="<none>"))
    assert s.current_replicas() is None


def test_reap_scales_down_when_idle(monkeypatch, tmp_path):
    import app.agents.mta_v2.inference_autoscale as mod
    monkeypatch.setattr(mod, "ACTIVITY_FILE", tmp_path / "a.json")
    mark_activity(now=NOW - 5 * HOUR, path=tmp_path / "a.json")

    s = RayServiceScaler(policy=AutoscalePolicy(idle_timeout_s=2 * HOUR))
    patched = {}
    def fake(*args, **kwargs):
        if args and args[0] == "get":
            return _Result(out="2")
        patched["payload"] = json.loads(args[args.index("-p") + 1])
        return _Result()
    monkeypatch.setattr(s, "_kubectl", fake)

    assert s.reap_if_idle(now=NOW) is True
    groups = patched["payload"]["spec"]["rayClusterConfig"]["workerGroupSpecs"]
    assert groups[0]["replicas"] == 0


def test_reap_leaves_a_recently_used_service_running(monkeypatch, tmp_path):
    import app.agents.mta_v2.inference_autoscale as mod
    monkeypatch.setattr(mod, "ACTIVITY_FILE", tmp_path / "a.json")
    mark_activity(now=NOW - 60, path=tmp_path / "a.json")

    s = RayServiceScaler(policy=AutoscalePolicy(idle_timeout_s=2 * HOUR))
    monkeypatch.setattr(s, "_kubectl", lambda *a, **k: _Result(out="2"))
    assert s.reap_if_idle(now=NOW) is False


def test_ensure_capacity_ramps_from_zero(monkeypatch):
    s = RayServiceScaler(policy=AutoscalePolicy(warm_replicas=1))
    calls = []
    def fake(*args, **kwargs):
        calls.append(args[0])
        return _Result(out="0") if args[0] == "get" else _Result()
    monkeypatch.setattr(s, "_kubectl", fake)
    assert s.ensure_capacity() is True
    assert "patch" in calls


def test_ensure_capacity_is_a_noop_when_already_warm(monkeypatch):
    s = RayServiceScaler(policy=AutoscalePolicy(warm_replicas=1))
    calls = []
    def fake(*args, **kwargs):
        calls.append(args[0])
        return _Result(out="2")
    monkeypatch.setattr(s, "_kubectl", fake)
    assert s.ensure_capacity() is True
    assert "patch" not in calls
