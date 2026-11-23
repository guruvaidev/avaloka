"""Track 5 smoke tests — recovery diagnostics, all local (no cloud)."""

from __future__ import annotations

from dataclasses import replace

from benchmarks.adapters import requires_cloud
from benchmarks.core.registry import get_suite
from benchmarks.core.runner import BenchmarkRunner


def _run():
    tasks, configs = get_suite("track5_reliability")
    configs = [replace(c, trials=3) for c in configs]
    runner = BenchmarkRunner()
    results = runner.run_suite(tasks, configs)
    return {r.config_id: r for r in results}


def test_track5_is_local_only():
    _tasks, configs = get_suite("track5_reliability")
    assert all(not requires_cloud(c.sut) for c in configs)


def test_no_fault_passes_and_wastes_nothing():
    res = _run()
    r = res["t5_no_fault"]
    assert r.stats["pass_at_1"] == 1.0
    assert r.stats["quality"]["cost_wasted_usd"]["mean"] == 0.0
    assert r.stats["quality"]["recovered"]["mean"] == 1.0


def test_transient_fault_needs_recovery():
    res = _run()
    off = res["t5_transient_no_recovery"]
    on = res["t5_transient_with_recovery"]
    assert off.stats["quality"]["recovered"]["mean"] == 0.0   # no retry -> stuck
    assert on.stats["quality"]["recovered"]["mean"] == 1.0    # retry -> recovers
    assert on.stats["pass_at_1"] == 1.0
    assert off.stats["pass_at_1"] == 0.0


def test_structural_fault_needs_mitigation_not_blind_retry():
    res = _run()
    blind = res["t5_structural_blind_retry"]       # recovery on, no mitigation
    mitigated = res["t5_structural_with_mitigation"]
    assert blind.stats["quality"]["recovered"]["mean"] == 0.0
    assert mitigated.stats["quality"]["recovered"]["mean"] == 1.0
    assert mitigated.stats["pass_at_1"] == 1.0


def test_recovery_wastes_some_work_but_less_than_total():
    res = _run()
    recovered = res["t5_transient_with_recovery"]
    # Recovery redoes only the work since the last checkpoint: > 0 but a small fraction.
    wr = recovered.stats["quality"]["work_repeated_ratio"]["mean"]
    assert 0.0 < wr < 0.2


def test_user_always_informed_even_on_unrecovered_failure():
    res = _run()
    for cid in ("t5_transient_no_recovery", "t5_corrupt_parquet_no_recovery"):
        assert res[cid].stats["quality"]["user_informed"]["mean"] == 1.0
