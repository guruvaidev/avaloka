"""
Swarm scenarios — end-to-end "many Avalokas under one main Avaloka" jobs.

Each scenario describes a full-lifecycle goal over one dataset and the
coordination properties the swarm must satisfy. The harness builds a swarm,
plans, runs, and asserts these properties (see test_swarm_bench.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SwarmScenario:
    scenario_id: str
    dataset_key: str
    goal: str
    target: str
    task_kind: str  # binary_classification | multiclass_classification | regression
    n_workers: int = 3
    n_model_candidates: int = 3
    heavy: bool = False
    # Coordination properties asserted by the harness.
    expect_disjoint_feature_shards: bool = True
    expect_consensus_model: bool = True
    expect_served_endpoint: bool = True
    # Fault-injection: kinds that a single worker will fail, to test resilience.
    inject_failure_kinds: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


SWARM_SCENARIOS = [
    SwarmScenario(
        scenario_id="swarm.titanic.full",
        dataset_key="titanic",
        goal="Build & serve a survival-prediction model from a shared dataset.",
        target="Survived",
        task_kind="binary_classification",
        n_workers=3,
        n_model_candidates=3,
        tags=["full-lifecycle", "classification"],
    ),
    SwarmScenario(
        scenario_id="swarm.churn.competitive",
        dataset_key="telco_churn",
        goal="Five data scientists race to the best churn model; coordinator picks the winner.",
        target="Churn",
        task_kind="binary_classification",
        n_workers=5,
        n_model_candidates=5,
        tags=["competitive-training", "consensus"],
    ),
    SwarmScenario(
        scenario_id="swarm.house.regression",
        dataset_key="house_prices",
        goal="Swarm builds a price regressor and provisions inference on k8s.",
        target="price",
        task_kind="regression",
        n_workers=4,
        n_model_candidates=3,
        tags=["regression", "serving"],
    ),
    SwarmScenario(
        scenario_id="swarm.titanic.faulttolerant",
        dataset_key="titanic",
        goal="One worker's trainer keeps crashing; the swarm must still ship a model.",
        target="Survived",
        task_kind="binary_classification",
        n_workers=4,
        n_model_candidates=4,
        inject_failure_kinds=["train"],  # one worker fails its train subtasks
        expect_consensus_model=True,
        tags=["fault-tolerance", "resilience"],
    ),
    SwarmScenario(
        scenario_id="swarm.titanic.deep",
        dataset_key="titanic",
        goal="Swarm trains competing deep tabular models and serves the best.",
        target="Survived",
        task_kind="binary_classification",
        n_workers=3,
        n_model_candidates=3,
        heavy=True,
        tags=["deep-learning", "heavy", "full-lifecycle"],
    ),
]
