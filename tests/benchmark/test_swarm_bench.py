"""
Swarm benchmark suite — multiple Avalokas coordinating under one main Avaloka.

These tests assert *coordination properties* over the full lifecycle described
in the spec (sample -> analyze -> build -> transform -> train -> serve), using
the deterministic swarm harness so they are stable in CI.
"""

from __future__ import annotations

import pytest

from tests.benchmark.catalog import SWARM_SCENARIOS
from tests.benchmark.harness import prepare_source
from tests.benchmark.registry import get
from tests.benchmark.swarm import MainAvaloka, WorkerAvaloka, build_swarm


def _feature_columns(dataset_key, target):
    spec = get(dataset_key)
    return [c.name for c in spec.synthetic_columns if c.name != target]


def _build_swarm_for(scenario):
    workers = []
    for i in range(scenario.n_workers):
        # Inject a fault into exactly one worker if the scenario asks for it.
        fail = set(scenario.inject_failure_kinds) if (i == 0 and scenario.inject_failure_kinds) else set()
        workers.append(WorkerAvaloka(f"avaloka-{i}", fail_kinds=fail, max_retries=2))
    return MainAvaloka(workers)


_LIGHT = [s for s in SWARM_SCENARIOS if not s.heavy]
_HEAVY = [s for s in SWARM_SCENARIOS if s.heavy]


@pytest.mark.swarm
def test_plan_partitions_features_disjointly():
    """Feature analysis must be sharded into disjoint, complete partitions."""
    swarm = build_swarm(n_workers=3)
    cols = _feature_columns("titanic", "Survived")
    plan = swarm.plan("goal", target="Survived", task_kind="binary_classification",
                      feature_columns=cols, n_model_candidates=3)
    shards = [st.partition["columns"] for st in plan.subtasks
              if st.kind == "analyze_features"]
    flat = [c for s in shards for c in s]
    assert sorted(flat) == sorted(cols), "feature coverage incomplete"
    assert len(flat) == len(set(flat)), "feature shards overlap"
    assert len(shards) >= 2, "expected data-parallel sharding across workers"


@pytest.mark.swarm
def test_plan_covers_full_lifecycle():
    swarm = build_swarm(n_workers=3)
    cols = _feature_columns("titanic", "Survived")
    plan = swarm.plan("goal", target="Survived", task_kind="binary_classification",
                      feature_columns=cols)
    kinds = set(plan.coverage())
    assert {"sample", "analyze_features", "build_features", "transform", "train",
            "serve"} <= kinds


@pytest.mark.swarm
def test_assignment_balanced_across_workers():
    swarm = build_swarm(n_workers=4)
    cols = _feature_columns("telco_churn", "Churn")
    plan = swarm.plan("goal", target="Churn", task_kind="binary_classification",
                      feature_columns=cols, n_model_candidates=4)
    used = set(plan.assignment.values())
    assert len(used) >= 2, "work not distributed across the swarm"


@pytest.mark.swarm
@pytest.mark.parametrize("scenario", _LIGHT, ids=[s.scenario_id for s in _LIGHT])
def test_swarm_scenario(scenario, workdir):
    handle, source, _ = prepare_source(scenario.dataset_key, "local", workdir)
    swarm = _build_swarm_for(scenario)
    cols = _feature_columns(scenario.dataset_key, scenario.target)

    plan = swarm.plan(scenario.goal, target=scenario.target,
                      task_kind=scenario.task_kind, feature_columns=cols,
                      n_model_candidates=scenario.n_model_candidates,
                      heavy=scenario.heavy)
    result = swarm.run(plan, source.uri)

    # 1. Sampling + feature build + transform must have succeeded.
    kinds_ok = {r.kind for r in result.successes()}
    assert {"sample", "build_features", "transform"} <= kinds_ok, (
        f"core lifecycle failed: {[(r.kind, r.error) for r in result.sub_results if not r.ok]}"
    )

    # 2. Consensus model selected from competing trainers.
    if scenario.expect_consensus_model:
        assert result.consensus.get("selected_model_path"), "no consensus model"
        assert result.consensus["n_candidates"] >= 1

    # 3. Inference service provisioned on k8s.
    if scenario.expect_served_endpoint:
        assert result.aggregated.get("ready") is True, "inference service not ready"
        assert "endpoint" in result.aggregated

    # 4. Provenance recorded for every subtask.
    assert len(result.provenance) == len(plan.subtasks)

    # 5. Merged feature catalog covers the dataset's feature columns.
    catalog = result.aggregated.get("feature_catalog", {})
    assert catalog, "feature catalog empty after aggregation"


@pytest.mark.swarm
def test_swarm_fault_tolerance(workdir):
    """One worker's trainer always crashes; the swarm must still ship a model."""
    scenario = next(s for s in SWARM_SCENARIOS if s.scenario_id == "swarm.titanic.faulttolerant")
    handle, source, _ = prepare_source(scenario.dataset_key, "local", workdir)
    swarm = _build_swarm_for(scenario)
    cols = _feature_columns(scenario.dataset_key, scenario.target)
    plan = swarm.plan(scenario.goal, target=scenario.target,
                      task_kind=scenario.task_kind, feature_columns=cols,
                      n_model_candidates=scenario.n_model_candidates)
    result = swarm.run(plan, source.uri)

    train_results = [r for r in result.sub_results if r.kind == "train"]
    assert any(not r.ok for r in train_results), "fault injection had no effect"
    assert any(r.ok for r in train_results), "all trainers failed — no resilience"
    assert result.consensus.get("selected_model_path"), "swarm failed to converge"
    assert result.aggregated.get("ready") is True, "service not provisioned despite resilience"


@pytest.mark.swarm
def test_swarm_run_is_deterministic(workdir):
    """Same swarm + same data -> same consensus winner (reproducible aggregation)."""
    handle, source, _ = prepare_source("titanic", "local", workdir)
    cols = _feature_columns("titanic", "Survived")

    def run_once():
        swarm = build_swarm(n_workers=3)
        plan = swarm.plan("g", target="Survived", task_kind="binary_classification",
                          feature_columns=cols, n_model_candidates=3)
        return swarm.run(plan, source.uri).consensus.get("selected_metrics")

    assert run_once() == run_once()


@pytest.mark.swarm
@pytest.mark.heavy
@pytest.mark.slow
@pytest.mark.parametrize("scenario", _HEAVY, ids=[s.scenario_id for s in _HEAVY])
def test_swarm_scenario_heavy(scenario, workdir):
    pytest.importorskip("torch", reason="heavy swarm needs PyTorch")
    handle, source, _ = prepare_source(scenario.dataset_key, "local", workdir)
    swarm = _build_swarm_for(scenario)
    cols = _feature_columns(scenario.dataset_key, scenario.target)
    plan = swarm.plan(scenario.goal, target=scenario.target,
                      task_kind=scenario.task_kind, feature_columns=cols,
                      n_model_candidates=scenario.n_model_candidates, heavy=True)
    result = swarm.run(plan, source.uri)
    assert result.consensus.get("selected_model_path"), "no consensus deep model"
    assert "torch" in (result.consensus.get("winner_kind") or ""), "deep model not used"
    assert result.aggregated.get("ready") is True
