"""Track 4 smoke tests — the ablation diagnostics must hold, all local (no cloud)."""

from __future__ import annotations

from dataclasses import replace

from benchmarks.core.registry import get_suite
from benchmarks.core.runner import BenchmarkRunner


def _run():
    tasks, configs = get_suite("track4_swarm_ablation")
    configs = [replace(c, trials=3) for c in configs]
    runner = BenchmarkRunner()
    results = runner.run_suite(tasks, configs)
    # index by (config_id, task_id)
    return {(r.config_id, r.task_id): r for r in results}


def test_track4_is_local_only():
    from benchmarks.core.registry import get_suite as gs
    from benchmarks.adapters import requires_cloud
    _tasks, configs = gs("track4_swarm_ablation")
    assert all(not requires_cloud(c.sut) for c in configs)


def test_removing_sampling_agent_scans_everything():
    res = _run()
    full = res[("t4_full_swarm", "t4_mean_amount")]
    single = res[("t4_single_general", "t4_mean_amount")]  # no sampling agent
    single_scan = single.stats["tuple"]["data_scanned_ratio"]["mean"]
    full_scan = full.stats["tuple"]["data_scanned_ratio"]["mean"]
    assert single_scan == 1.0            # no sampling -> full scan
    assert full_scan < 1.0               # swarm samples
    # ... and the full swarm therefore costs less on this task.
    assert (full.stats["tuple"]["cost_usd"]["mean"]
            < single.stats["tuple"]["cost_usd"]["mean"])


def test_removing_validator_raises_leakage_and_error():
    res = _run()
    full = res[("t4_full_swarm", "t4_mean_amount")]
    no_val = res[("t4_specialized_only", "t4_mean_amount")]  # has validator
    single = res[("t4_single_general", "t4_mean_amount")]     # no validator
    assert full.stats["quality"]["leakage"]["mean"] == 0.0
    assert no_val.stats["quality"]["leakage"]["mean"] == 0.0
    assert single.stats["quality"]["leakage"]["mean"] == 1.0
    # Leakage inflates the estimate -> worse answer error than the full swarm.
    assert (single.stats["quality"]["answer_rel_error"]["mean"]
            > full.stats["quality"]["answer_rel_error"]["mean"])


def test_removing_reviewer_produces_unsupported_claims():
    res = _run()
    full = res[("t4_full_swarm", "t4_mean_amount")]
    no_rev = res[("t4_no_reviewer", "t4_mean_amount")]
    assert full.stats["quality"]["unsupported_claims"]["mean"] == 0.0
    assert no_rev.stats["quality"]["unsupported_claims"]["mean"] > 0.0


def test_removing_planner_raises_tool_calls_and_retries():
    res = _run()
    full = res[("t4_full_swarm", "t4_mean_amount")]
    no_plan = res[("t4_no_planner", "t4_mean_amount")]
    assert (no_plan.stats["tuple"]["tool_calls"]["mean"]
            > full.stats["tuple"]["tool_calls"]["mean"])
    assert (no_plan.stats["tuple"]["retries"]["mean"]
            > full.stats["tuple"]["retries"]["mean"])
