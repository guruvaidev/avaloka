"""Memory failure diagnosis and bounded recovery."""

import pytest

from app.infra.memory_recovery import (DEFAULT_MEMORY_CEILING_GIB, Diagnosis, FailureKind,
                                       FailureSignal, RecoveryExhausted, RecoveryPolicy,
                                       Strategy, WorkloadResources, classify_failure,
                                       estimate_from_profile, estimate_working_set_gib,
                                       propose_remediation)

BASE = WorkloadResources(memory_gib=8.0, cpus=2.0, workers=2, object_store_gib=0.5)


# --------------------------------------------------------------------------- #
# Classification prefers facts over log text
# --------------------------------------------------------------------------- #

def test_pod_reason_is_conclusive():
    d = classify_failure(FailureSignal(pod_reason="OOMKilled"))
    assert d.kind is FailureKind.OOM_KILLED and d.confidence == 1.0


def test_exit_137_is_the_kernel_oom_killer():
    d = classify_failure(FailureSignal(exit_code=137))
    assert d.kind is FailureKind.OOM_KILLED
    assert "128" in d.evidence[0]


def test_structured_signal_beats_misleading_logs():
    """A traceback can mention memory without memory being the cause."""
    d = classify_failure(FailureSignal(
        pod_reason="OOMKilled",
        logs="ModuleNotFoundError: No module named 'memory_profiler'"))
    assert d.kind is FailureKind.OOM_KILLED


def test_a_variable_named_memory_is_not_an_oom():
    """The failure mode QA flagged: classifying on incidental log content."""
    d = classify_failure(FailureSignal(
        exit_code=1,
        logs="NameError: name 'memory_limit_gib' is not defined"))
    assert d.kind is FailureKind.NOT_MEMORY
    assert not d.remediable


@pytest.mark.parametrize("logs,expected", [
    ("MemoryError", FailureKind.HEAP_EXHAUSTED),
    ("numpy.core._exceptions._ArrayMemoryError: Unable to allocate 4.5 GiB",
     FailureKind.HEAP_EXHAUSTED),
    ("ObjectStoreFullError: Failed to put object", FailureKind.OBJECT_STORE_FULL),
    ("Spilled 12000 MiB to local disk", FailureKind.OBJECT_STORE_FULL),
    ("RayActorError: The actor died unexpectedly", FailureKind.WORKER_LOST),
    ("FileNotFoundError: s3://bucket/missing.parquet", FailureKind.NOT_MEMORY),
    ("botocore 403 Forbidden", FailureKind.NOT_MEMORY),
])
def test_log_classification(logs, expected):
    assert classify_failure(FailureSignal(logs=logs)).kind is expected


def test_no_signal_is_unknown_not_assumed_memory():
    d = classify_failure(FailureSignal())
    assert d.kind is FailureKind.UNKNOWN
    assert not d.remediable, "an unknown failure must never be resized blindly"


def test_repeated_restarts_are_recorded_as_evidence():
    d = classify_failure(FailureSignal(logs="MemoryError", restart_count=3))
    assert any("restart" in e for e in d.evidence)


# --------------------------------------------------------------------------- #
# Estimating what the workload needs
# --------------------------------------------------------------------------- #

def test_object_columns_dominate_the_estimate():
    numeric = estimate_working_set_gib(rows=10_000_000, dtypes=["float64"] * 10)
    text = estimate_working_set_gib(rows=10_000_000, dtypes=["object"] * 10)
    assert text > numeric * 8, "a string column is far more than 8 bytes per row"


def test_estimate_includes_transform_overhead():
    raw_gib = (1_000_000 * 8 * 10) / (1024 ** 3)
    est = estimate_working_set_gib(rows=1_000_000, dtypes=["float64"] * 10)
    assert est == pytest.approx(raw_gib * 3.0, rel=0.01)


def test_empty_inputs_estimate_zero():
    assert estimate_working_set_gib(rows=0, dtypes=["int64"]) == 0.0
    assert estimate_working_set_gib(rows=100, dtypes=[]) == 0.0


def test_estimate_from_the_profile_the_profiler_already_produces():
    profile = {"dataset": {"n_rows": 2_000_000},
               "columns": [{"name": "a", "dtype": "int64"},
                           {"name": "b", "dtype": "object"}]}
    assert estimate_from_profile(profile) > 0


def test_unknown_dtype_falls_back_rather_than_crashing():
    assert estimate_working_set_gib(rows=1000, dtypes=["some_extension_type"]) > 0


# --------------------------------------------------------------------------- #
# Remediation
# --------------------------------------------------------------------------- #

def test_non_memory_failure_is_escalated_never_resized():
    d = Diagnosis(FailureKind.NOT_MEMORY, 0.9, [])
    r = propose_remediation(d, BASE)
    assert r.strategy is Strategy.ESCALATE
    assert r.resources is None
    assert r.needs_human_approval


def test_oom_at_least_doubles_memory():
    d = Diagnosis(FailureKind.OOM_KILLED, 1.0, [])
    r = propose_remediation(d, BASE)
    assert r.strategy is Strategy.RAISE_MEMORY
    assert r.resources.memory_gib >= 16.0


def test_estimate_overrides_a_doubling_that_would_still_be_too_small():
    d = Diagnosis(FailureKind.OOM_KILLED, 1.0, [])
    r = propose_remediation(d, BASE, estimated_need_gib=25.0)
    assert r.resources.memory_gib >= 25.0, "doubling to 16 GiB would fail again"


def test_object_store_pressure_raises_the_store_not_every_worker():
    d = Diagnosis(FailureKind.OBJECT_STORE_FULL, 0.9, [])
    r = propose_remediation(d, BASE)
    assert r.strategy is Strategy.RAISE_OBJECT_STORE
    assert r.resources.object_store_gib > BASE.object_store_gib
    assert r.resources.memory_gib == BASE.memory_gib


def test_beyond_the_ceiling_it_shards_rather_than_buying_a_bigger_machine():
    big = WorkloadResources(memory_gib=48.0, workers=2)
    r = propose_remediation(Diagnosis(FailureKind.OOM_KILLED, 1.0, []), big)
    assert r.strategy is Strategy.ADD_WORKERS
    assert r.resources.workers > big.workers
    assert r.resources.memory_gib == big.memory_gib


def test_at_the_worker_ceiling_it_reduces_the_batch():
    maxed = WorkloadResources(memory_gib=48.0, workers=10, batch_rows=100_000)
    r = propose_remediation(Diagnosis(FailureKind.OOM_KILLED, 1.0, []), maxed, max_workers=10)
    assert r.strategy is Strategy.REDUCE_BATCH
    assert r.resources.batch_rows == 50_000


def test_out_of_levers_escalates_rather_than_looping():
    stuck = WorkloadResources(memory_gib=64.0, workers=10, batch_rows=None)
    r = propose_remediation(Diagnosis(FailureKind.OOM_KILLED, 1.0, []), stuck, max_workers=10)
    assert r.strategy is Strategy.ESCALATE and r.needs_human_approval


def test_a_large_jump_is_a_spending_decision():
    big = WorkloadResources(memory_gib=20.0)
    r = propose_remediation(Diagnosis(FailureKind.OOM_KILLED, 1.0, []), big)
    assert r.resources.memory_gib >= 32.0
    assert r.needs_human_approval, "40 GiB is a cost decision, not a routine bump"


# --------------------------------------------------------------------------- #
# The policy: bounded, auditable, never a no-op retry
# --------------------------------------------------------------------------- #

def test_attempts_are_capped():
    policy = RecoveryPolicy(max_attempts=2)
    res = BASE
    for _ in range(2):
        rem = policy.next_step(FailureSignal(pod_reason="OOMKilled"), res)
        res = rem.resources
    with pytest.raises(RecoveryExhausted, match="attempts already made"):
        policy.next_step(FailureSignal(pod_reason="OOMKilled"), res)


def test_identical_resources_are_refused():
    """Retrying unchanged reproduces the failure and bills for it."""
    policy = RecoveryPolicy()
    stuck = WorkloadResources(memory_gib=DEFAULT_MEMORY_CEILING_GIB, workers=10)
    with pytest.raises(RecoveryExhausted):
        for _ in range(5):
            rem = policy.next_step(FailureSignal(pod_reason="OOMKilled"), stuck)
            if rem.resources:
                stuck = rem.resources


def test_each_attempt_strictly_increases_the_resource():
    policy = RecoveryPolicy(max_attempts=3)
    res = WorkloadResources(memory_gib=4.0, workers=1)
    seen = [res.memory_gib]
    for _ in range(3):
        rem = policy.next_step(FailureSignal(pod_reason="OOMKilled"), res)
        if rem.resources is None:
            break
        res = rem.resources
        seen.append(res.memory_gib)
    assert seen == sorted(seen) and len(set(seen)) == len(seen)


def test_routine_bumps_apply_automatically_large_ones_do_not():
    policy = RecoveryPolicy()
    small = policy.next_step(FailureSignal(pod_reason="OOMKilled"),
                             WorkloadResources(memory_gib=4.0))
    assert policy.may_apply_automatically(small)

    policy2 = RecoveryPolicy()
    large = policy2.next_step(FailureSignal(pod_reason="OOMKilled"),
                              WorkloadResources(memory_gib=24.0))
    assert not policy2.may_apply_automatically(large)


def test_non_memory_failure_is_never_applied_automatically():
    policy = RecoveryPolicy()
    rem = policy.next_step(FailureSignal(logs="ImportError: no module named pandas"), BASE)
    assert rem.strategy is Strategy.ESCALATE
    assert not policy.may_apply_automatically(rem)


def test_audit_trail_records_every_decision_and_its_reason():
    policy = RecoveryPolicy()
    policy.next_step(FailureSignal(pod_reason="OOMKilled"), BASE, estimated_need_gib=30.0)
    trail = policy.audit_trail()
    assert len(trail) == 1
    entry = trail[0]
    assert entry["diagnosis"]["kind"] == "oom_killed"
    assert entry["remediation"]["reason"]
    assert entry["remediation"]["estimated_need_gib"] == 30.0
