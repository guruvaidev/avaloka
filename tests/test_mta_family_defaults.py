"""Gradient boosting and random forests train with their own defaults, never the MLP's.

rc3: the planner extracted hyperparameters from the MLP template before it
picked a family, so a gradient-boosted plan carried ``learning_rate=0.001``
and ``epochs=30``; ``build_estimator`` then trained 30 boosting rounds
(epochs fed max_iter) at a hundredth of the usual rate. On the same split
stroke went 0.9497 -> 0.9630 (baseline 0.9573) and diabetes 0.5391 ->
0.6091 (baseline 0.5392) once each family got its own defaults.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from app.agents.mta_v2 import estimators

MLP_TEMPLATE = dict(estimators.MLP_DEFAULTS)


def _gb(config):
    return estimators.build_estimator("gradient_boosting", "classification", config).get_params()


def test_a_gb_plan_from_the_mlp_template_gets_gbs_own_defaults():
    hp = estimators.family_hyperparameters("gradient_boosting", MLP_TEMPLATE)
    assert hp["learning_rate"] == 0.1
    assert hp["max_iter"] == 200
    assert hp["early_stopping_patience"] == 10
    assert not {"epochs", "optimizer_name", "batch_size", "hidden_layer_sizes"} & set(hp)


def test_gb_never_gets_the_mlp_learning_rate_unless_it_is_set():
    # From the plan builder, and from a plan that names no learning rate at all.
    for hp in (estimators.family_hyperparameters("gradient_boosting", MLP_TEMPLATE), {}):
        config = {"epochs": 30, "batch_size": 32, **hp}      # what LocalTrainer adds
        config.setdefault("learning_rate", estimators.default_for("gradient_boosting", "learning_rate"))
        params = _gb(config)
        assert params["learning_rate"] != MLP_TEMPLATE["learning_rate"]
        assert params["max_iter"] == 200, "epochs=30 must not cap the boosting rounds"


def test_an_explicit_value_is_kept():
    hp = estimators.family_hyperparameters(
        "gradient_boosting", {**MLP_TEMPLATE, "learning_rate": 0.05, "max_depth": 6, "epochs": 120})
    assert hp["learning_rate"] == 0.05 and hp["max_depth"] == 6
    assert hp["max_iter"] == 120, "an epochs request for GB means boosting rounds"
    rf = estimators.family_hyperparameters("random_forest", {**MLP_TEMPLATE, "n_estimators": 150})
    assert rf["n_estimators"] == 150 and rf["max_features"] == "sqrt"
    # Set explicitly on the plan the trainer reads, 0.001 is honoured.
    assert _gb({"learning_rate": 0.001})["learning_rate"] == 0.001


def test_an_mlp_plan_is_untouched():
    assert estimators.family_hyperparameters("mlp", MLP_TEMPLATE) == MLP_TEMPLATE
    assert estimators.default_for("mlp", "learning_rate") == 0.001


@pytest.mark.parametrize("algorithm,expected_lr,expected_patience", [
    ("gradient_boosting", 0.1, 10),
    ("mlp", 0.001, 5),
])
def test_local_trainer_fills_a_missing_key_with_the_familys_default(algorithm, expected_lr, expected_patience):
    from app.agents.mta_v2 import local_trainer
    captured = {}

    def _fake_main(**kwargs):
        captured.update(kwargs)
        return {"status": "success"}

    plan = {"model_type": "classification", "algorithm": algorithm, "hyperparameter_config": {},
            "data_config": {"dataset_uri": "/tmp/x.csv", "feature_columns": ["a"], "target_column": "y"}}
    with patch.object(local_trainer, "training_main", side_effect=_fake_main), \
         patch.object(local_trainer, "setup_mlflow"):
        local_trainer.LocalTrainer("u", "s").train(plan)
    assert captured["learning_rate"] == expected_lr
    assert captured["early_stopping_pat"] == expected_patience


def test_the_plan_builder_gives_a_gb_plan_its_own_hyperparameters():
    from app.agents.mta_v2 import agent as mta
    agent = mta.ModelTrainingAgent()
    data_config = {"dataset_uri": "/tmp/x.csv", "feature_columns": ["a", "b"], "target_column": "y"}
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps({
        "model_type": "classification", "algorithm": "gradient_boosting",
        "model_name": "m", "model_description": "d", "model_version": "v1.0"}))
    state = {"messages": [], "training_plan": {}}
    with patch.object(mta, "llm", llm), \
         patch.object(agent, "_extract_hyperparameter_config", return_value=dict(MLP_TEMPLATE)), \
         patch.object(agent, "_extract_ray_config", return_value={}), \
         patch.object(agent, "_extract_initial_data_config", return_value=data_config), \
         patch.object(agent, "_reconcile_model_type_with_target", side_effect=lambda t, *a, **k: t), \
         patch.object(agent, "_training_backend_decision", return_value=("local", "local file", 1)), \
         patch.object(agent, "_state_dataset_size_bytes", return_value=1):
        plan = agent._extract_training_plan(state)
    assert plan["algorithm"] == "gradient_boosting"
    hp = plan["hyperparameter_config"]
    assert hp["learning_rate"] == 0.1 and hp["max_iter"] == 200
    assert "epochs" not in hp and "optimizer_name" not in hp


def test_ray_never_hands_an_mlp_a_tree_familys_learning_rate(monkeypatch):
    from app.agents.mta_v2.ray_trainer import RayTrainer
    trainer = RayTrainer(user_id="u", session_id="s")
    plan = {"algorithm": "gradient_boosting", "model_type": "classification",
            "hyperparameter_config": estimators.family_hyperparameters("gradient_boosting", MLP_TEMPLATE),
            "data_config": {"dataset_uri": "gs://b/x.csv", "feature_columns": ["a"], "target_column": "y"}}
    env = trainer._build_job_env(plan)
    assert env["LEARNING_RATE"] == "0.001"
