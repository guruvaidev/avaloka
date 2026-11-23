"""The training agent can train something other than a neural network.

Before this, `mta_v2` had exactly one model. `TrainingPlan` carried no field
naming an algorithm -- `model_type` is the *task* -- and `_execute_training`
chose only a backend, never a family. So a user asking Avaloka to predict a
column got an MLP whether or not one suited the data, and on tabular data it
usually does not.
"""
import numpy as np
import pandas as pd
import pytest

from app.agents.mta_v2 import estimators

FEATURES = ["tenure", "monthly", "support_calls", "noise"]


def _frame(n=600, seed=0, regression=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure": rng.integers(1, 72, n).astype(float),
        "monthly": rng.normal(70, 20, n),
        "support_calls": rng.poisson(2, n).astype(float),
        "noise": rng.normal(0, 1, n),
    })
    signal = 0.05 * df.tenure - 0.04 * df.monthly + 0.6 * df.support_calls
    df["__label__"] = (signal + rng.normal(0, 0.5, n) if regression
                       else (signal + rng.normal(0, 1, n) > signal.mean()).astype(int))
    return df.iloc[:int(n * 0.75)], df.iloc[int(n * 0.75):]


class TestAlgorithmNormalisation:
    def test_the_default_is_the_mlp(self):
        """A plan written before this field existed must train what it trained
        yesterday, so anything unrecognised resolves to the MLP."""
        assert estimators.normalise_algorithm(None) == "mlp"
        assert estimators.normalise_algorithm("") == "mlp"
        assert estimators.DEFAULT_ALGORITHM == "mlp"

    @pytest.mark.parametrize("alias,expected", [
        ("xgboost", "gradient_boosting"), ("GBM", "gradient_boosting"),
        ("lightgbm", "gradient_boosting"), ("hist-gradient-boosting", "gradient_boosting"),
        ("random-forest", "random_forest"), ("rf", "random_forest"),
        ("Ridge", "linear"), ("logistic", "linear"),
        ("neural_network", "mlp"), ("NN", "mlp"),
    ])
    def test_accepts_the_names_people_actually_write(self, alias, expected):
        assert estimators.normalise_algorithm(alias) == expected

    def test_an_unknown_family_falls_back_rather_than_raising(self):
        """Failing at the last step of a long training job because the planner
        invented a model name is worse than training the default and saying so."""
        assert estimators.normalise_algorithm("quantum_forest") == "mlp"

    def test_every_advertised_family_is_implemented(self):
        train, val = _frame()
        for algorithm in estimators.ALGORITHMS:
            if algorithm == "mlp":
                continue
            estimators.build_estimator(algorithm, "classification", {"random_seed": 42})

    def test_every_family_has_a_rationale_a_person_can_read(self):
        for algorithm in estimators.ALGORITHMS:
            assert estimators.FAMILY_RATIONALE.get(algorithm)
            assert estimators.FAMILY_HYPERPARAMETERS.get(algorithm)


class TestContract:
    """Every family returns train_local's (model, history, final_metrics)."""

    @pytest.mark.parametrize("algorithm", ["gradient_boosting", "random_forest", "linear"])
    def test_classification_returns_the_same_triple(self, algorithm):
        train, val = _frame()
        model, history, metrics = estimators.fit_sklearn(train, val, {
            "algorithm": algorithm, "feature_cols": FEATURES,
            "model_type": "classification", "random_seed": 42,
        })
        assert hasattr(model, "predict")
        assert isinstance(history, list) and history
        for key in ("epoch", "train_loss", "val_loss", "accuracy",
                    "val_accuracy", "f1_score", "val_f1_score"):
            assert key in metrics, f"{algorithm} dropped {key} from the contract"
        assert metrics["val_accuracy"] > 0.6, "should beat a coin flip on separable data"

    @pytest.mark.parametrize("algorithm", ["gradient_boosting", "random_forest", "linear"])
    def test_regression_returns_the_same_triple(self, algorithm):
        train, val = _frame(regression=True)
        model, history, metrics = estimators.fit_sklearn(train, val, {
            "algorithm": algorithm, "feature_cols": FEATURES,
            "model_type": "regression", "random_seed": 42,
        })
        assert metrics["val_loss"] >= 0
        assert len(history) == 1, (
            "these families do not train in epochs; inventing a curve for them "
            "would put a shape in the UI that means nothing"
        )

    def test_a_classifier_is_built_for_classification_and_a_regressor_for_regression(self):
        clf = estimators.build_estimator("random_forest", "classification", {})
        reg = estimators.build_estimator("random_forest", "regression", {})
        assert "Classifier" in type(clf).__name__
        assert "Regressor" in type(reg).__name__

    def test_hyperparameters_reach_the_estimator(self):
        est = estimators.build_estimator("random_forest", "classification", {"n_estimators": 7})
        assert est.get_params()["n_estimators"] == 7


class TestTorchDetection:
    def test_a_sklearn_estimator_is_not_a_torch_model(self):
        """Drives the branch that decides .pth/.onnx and mlflow.pytorch versus
        joblib and mlflow.sklearn. Getting it wrong writes a pickle nothing can
        load back, which fails at load time rather than here."""
        est = estimators.build_estimator("linear", "regression", {})
        assert estimators.is_torch_model(est) is False


class TestRegressionOnTheScaledLabels:
    """develop-1.6 standardises regression labels before train_local.

    The MLP path inverts that before reporting MAE/RMSE/R-squared. A non-MLP
    family must do the same, or training_main finds no train_/val_ mae keys and
    logs final_mae = final_rmse = 0.0: a perfect-looking score for any model.
    """

    @pytest.mark.parametrize("algorithm", ["gradient_boosting", "linear"])
    def test_reports_mae_in_the_targets_own_units(self, algorithm):
        from app.agents.mta_v2.local_trainer import (_standard_scaler_metadata,
                                                     train_local)
        train, val = _frame(regression=True)
        train, val = train.copy(), val.copy()
        # A target in the thousands, as real prices or revenues are.
        train["__label__"] = 1000 + 400 * train["__label__"]
        val["__label__"] = 1000 + 400 * val["__label__"]
        scaler = _standard_scaler_metadata(train["__label__"])
        for frame in (train, val):
            frame["__label__"] = (frame["__label__"] - scaler["mean"]) / scaler["scale"]

        model, history, metrics = train_local(train, val, {
            "algorithm": algorithm, "feature_cols": FEATURES,
            "model_type": "regression", "random_seed": 42,
            "target_scaling": scaler,
        })

        for key in ("train_mae", "val_mae", "train_rmse", "val_rmse",
                    "train_r2_score", "val_r2_score"):
            assert key in metrics, f"{algorithm} omitted {key}; final_{key} would log 0.0"
        assert "val_accuracy" not in metrics
        # Noise on the label is N(0, 0.5) before the x400 scale: an honest MAE
        # in original units is in the hundreds, never the ~0.3 of scaled units.
        assert 50 < metrics["val_mae"] < 400, metrics["val_mae"]
        assert metrics["val_rmse"] >= metrics["val_mae"]
        assert 0 < metrics["val_r2_score"] <= 1
        # The history row is what MLflow logs metric by metric, so it stays
        # flat; the fold report and the matrix ride only on the final metrics.
        evaluation_keys = {"evaluation_strategy", "cross_validation", "classification"}
        assert history == [{k: v for k, v in metrics.items() if k not in evaluation_keys}]
        assert metrics["cross_validation"]["primary_metric"] == "rmse"

    def test_the_baseline_verdict_can_be_reached_for_a_non_mlp_family(self):
        """The below-baseline guard (develop-1.6 #502) reads the evaluation
        report's beats_baseline, which is only decided when val_rmse is present
        and in the target's units. A family that omitted it would leave the
        verdict None, so a worse-than-trivial model could never be flagged."""
        from app.agents.mta_v2.local_trainer import (_standard_scaler_metadata,
                                                     train_local)
        from app.agents.mta_v2.training_docker_image.src.evaluation import (
            build_holdout_evaluation_report)
        rng = np.random.default_rng(3)
        train, val = _frame(regression=True)
        train, val = train.copy(), val.copy()
        # Pure noise: no family can beat predicting the mean by much, if at all.
        train["__label__"] = 500 + 100 * rng.normal(size=len(train))
        val["__label__"] = 500 + 100 * rng.normal(size=len(val))
        raw_train, raw_val = train["__label__"].tolist(), val["__label__"].tolist()
        scaler = _standard_scaler_metadata(train["__label__"])
        for frame in (train, val):
            frame["__label__"] = (frame["__label__"] - scaler["mean"]) / scaler["scale"]

        _, _, metrics = train_local(train, val, {
            "algorithm": "random_forest", "feature_cols": FEATURES,
            "model_type": "regression", "random_seed": 42, "target_scaling": scaler,
        })
        report = build_holdout_evaluation_report(raw_train, raw_val, "regression", metrics)
        assert report["beats_baseline"] in (True, False), report
