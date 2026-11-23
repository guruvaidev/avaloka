"""The training plan text describes the family that will train, with its real values.

rc3 web re-check: for diabetes the plan read "30 epochs, Adam optimizer" while
the model was gradient boosting, and for nullxlsx "None epochs, None
optimizer". The plan text is now built from the chosen family's real
hyperparameters -- read back off the estimator ``build_estimator`` makes --
and epochs and an optimizer appear only for the neural network.
"""
import re

import pytest

from app.agents.mta_v2 import estimators

# The diabetes plan as the planner emitted it (MLP keys under a GB algorithm).
_MLP_KEYS = {"learning_rate": 0.001, "epochs": 30, "batch_size": 32, "optimizer_name": "adam",
             "random_seed": 42, "early_stopping_patience": 5, "hidden_layer_sizes": [128, 64],
             "activation": "relu", "dropout_rate": 0.2, "batch_norm": True}


def _plan(algorithm, hp, model_type="classification"):
    return {
        "model_type": model_type, "algorithm": algorithm, "model_name": "m", "model_version": "v1.0",
        "model_description": "d", "hyperparameter_config": hp, "ray_config": None,
        "data_config": {"dataset_uri": "/tmp/x.csv", "feature_columns": ["a", "b"], "target_column": "y"},
    }


@pytest.fixture(scope="module")
def agent():
    from app.agents.mta_v2.agent import ModelTrainingAgent
    return ModelTrainingAgent()


def _training_line(text):
    return next(line for line in text.splitlines() if line.startswith("**Training:**"))


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("algorithm,hp,model_type", [
    ("gradient_boosting", dict(_MLP_KEYS), "classification"),          # diabetes
    ("gradient_boosting", {"max_depth": 6, "learning_rate": 0.05}, "classification"),
    ("gradient_boosting", {}, "regression"),                           # nullxlsx: no epochs at all
    ("random_forest", {"n_estimators": 150, "max_depth": None}, "classification"),
    ("random_forest", dict(_MLP_KEYS), "regression"),
    ("linear", {}, "classification"),
    ("linear", {}, "regression"),
])
def test_tree_and_linear_plans_never_mention_epochs_optimizer_or_none(agent, algorithm, hp, model_type, full):
    text = agent._format_training_plan_response(_plan(algorithm, hp, model_type), full=full)
    assert "None" not in text
    assert not re.search(r"\bepochs?\b", text, re.IGNORECASE)
    assert "optimizer" not in text.lower()
    assert "DynamicMLP" not in text and ".pth" not in text


def test_gradient_boosting_plan_shows_the_values_it_will_train_with(agent):
    hp = {"max_depth": 6, "learning_rate": 0.05, "max_iter": 400}
    text = agent._format_training_plan_response(_plan("gradient_boosting", hp))
    line = _training_line(text)
    assert "gradient-boosted trees" in line
    assert "up to 400 estimators" in line and "max depth 6" in line and "`0.05`" in line

    # A plan that still carries the MLP's values (stored before family
    # defaults existed) trains with them, so the text must say so; epochs no
    # longer cap the boosting rounds.
    real = estimators.build_estimator("gradient_boosting", "classification", dict(_MLP_KEYS)).get_params()
    line = _training_line(agent._format_training_plan_response(_plan("gradient_boosting", dict(_MLP_KEYS))))
    assert f"up to {real['max_iter']} estimators" in line
    assert f"`{real['learning_rate']}`" in line
    assert "max depth no limit" in line


def test_random_forest_plan_shows_estimators_and_depth(agent):
    line = _training_line(agent._format_training_plan_response(
        _plan("random_forest", {"n_estimators": 150, "max_depth": 8})))
    assert "random forest, 150 estimators, max depth 8" in line
    line = _training_line(agent._format_training_plan_response(_plan("random_forest", {})))
    assert "300 estimators, max depth no limit" in line


def test_full_plan_lists_the_family_hyperparameters(agent):
    text = agent._format_training_plan_response(
        _plan("gradient_boosting", {"max_depth": 4, "learning_rate": 0.1, "max_iter": 250}), full=True)
    assert "- **Trainer / Architecture:** Gradient-boosted trees" in text
    assert "- **Estimators:** 250" in text
    assert "- **Max depth:** 4" in text
    assert "- **Learning rate:** 0.1" in text
    assert "scikit-learn model" in text


@pytest.mark.parametrize("hp,expected", [
    (dict(_MLP_KEYS), "30 epochs, Adam optimizer, learning rate `0.001`, batch size `32`"),
    ({"epochs": 12, "optimizer_name": "sgd", "learning_rate": 0.01, "batch_size": 64},
     "12 epochs, SGD optimizer, learning rate `0.01`, batch size `64`"),
    # Only family-foreign keys, or explicit nulls: the trainer's defaults, never "None".
    ({"max_depth": 3}, "30 epochs, Adam optimizer, learning rate `0.001`, batch size `32`"),
    ({"epochs": None, "optimizer_name": None}, "30 epochs, Adam optimizer"),
])
def test_neural_plan_keeps_epochs_and_optimizer(agent, hp, expected):
    text = agent._format_training_plan_response(_plan("mlp", hp))
    assert expected in _training_line(text)
    assert "None" not in text


def test_a_plan_without_an_algorithm_is_the_mlp(agent):
    plan = _plan("mlp", dict(_MLP_KEYS))
    plan.pop("algorithm")
    assert "30 epochs, Adam optimizer" in _training_line(agent._format_training_plan_response(plan))
