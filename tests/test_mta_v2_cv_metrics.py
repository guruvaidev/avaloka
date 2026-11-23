"""k-fold and the confusion matrix in the app MTA.

Until this, the app's local trainer validated on one hashed 80/20 holdout and
reported accuracy and weighted F1 -- numbers that cannot tell a classifier
that found every class from one that found only the largest, and that nothing
contested with a second split. ``evaluate_model`` had the k-fold machinery
and no caller. Here the sklearn families are cross-validated after the fit,
the MLP keeps its holdout and says so, and every classifier reports a
labelled confusion matrix whose rows sum to the class counts.
"""
import numpy as np
import pandas as pd
import pytest

from app.agents.mta_v2 import estimators, local_trainer, model_card
from app.agents.mta_v2.training_docker_image.src.evaluation import (
    build_holdout_evaluation_report,
)

FEATURES = ["x1", "x2", "x3"]


def _three_class(n=450, seed=3):
    """Three separable-ish blobs, so every class is present in both splits."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 3, n)
    centres = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 1.0], [0.0, 3.0, -1.0]])
    X = centres[label] + rng.normal(0, 0.8, (n, 3))
    df = pd.DataFrame(X, columns=FEATURES)
    df["__label__"] = label
    return df.iloc[:int(n * 0.75)].reset_index(drop=True), df.iloc[int(n * 0.75):].reset_index(drop=True)


def _config(algorithm, model_type="classification", **extra):
    base = {
        "algorithm": algorithm, "feature_cols": FEATURES, "model_type": model_type,
        "class_names": ["red", "green", "blue"], "random_seed": 42,
        "num_classes": 3, "output_dim": 3, "hidden_sizes": [8], "activation": "relu",
        "dropout": 0.0, "batch_norm": False, "learning_rate": 0.01, "epochs": 3,
        "batch_size": 64, "optimizer": "adam", "early_stopping_patience": 2,
        "use_gpu": False, "target_scaling": {}, "class_weights": None,
    }
    base.update(extra)
    return base


class TestConfusionMatrix:
    def test_three_class_matrix_rows_sum_to_the_class_counts(self):
        """The matrix is 3x3, labelled with the class names, and each row
        adds up to how many validation rows actually carry that class. A
        mutation that drops a class from the matrix fails on the shape and
        on the row sums."""
        train, val = _three_class()
        model, _, final = local_trainer.train_local(train, val, _config("gradient_boosting"))
        detail = final["classification"]
        table = detail["confusion_matrix"]
        assert table[0] == ["actual \\ predicted", "red", "green", "blue"]
        assert [row[0] for row in table[1:]] == ["red", "green", "blue"]
        counts = val["__label__"].value_counts()
        for code, row in enumerate(table[1:]):
            assert len(row[1:]) == 3
            assert sum(row[1:]) == counts[code], f"row {row[0]} does not sum to its class count"
        assert set(detail["per_class"]) == {"red", "green", "blue"}
        for pc in detail["per_class"].values():
            assert 0.0 <= pc["precision"] <= 1.0 and 0.0 <= pc["recall"] <= 1.0
        assert 0.0 <= detail["precision_macro"] <= 1.0
        assert 0.0 <= detail["recall_macro"] <= 1.0
        assert detail["roc_auc"] is None, "ROC AUC is binary-only"
        # the flat record carries the macro scores for MLflow
        assert 0.0 <= final["val_precision_macro"] <= 1.0
        assert 0.0 <= final["val_recall_macro"] <= 1.0

    def test_a_never_predicted_class_still_has_a_row(self):
        """Dropping an absent prediction column would make the model look
        better than it is: the class the model never says is the row of
        misses, not an omission."""
        detail = estimators.classification_detail([0, 1, 2, 2], [0, 1, 1, 0], ["a", "b", "c"])
        assert detail["labels"] == ["a", "b", "c"]
        assert detail["confusion_matrix"][3] == ["c", 1, 1, 0]
        assert detail["per_class"]["c"] == {"precision": 0.0, "recall": 0.0, "support": 2}

    def test_binary_reports_roc_auc_when_probabilities_exist(self):
        detail = estimators.classification_detail([0, 1, 0, 1], [0, 1, 0, 0], ["no", "yes"],
                                                  y_score=[0.1, 0.9, 0.2, 0.4])
        assert detail["roc_auc"] == 1.0
        assert detail["confusion_matrix"][0] == ["actual \\ predicted", "no", "yes"]

    def test_model_card_quotes_the_matrix_and_per_class_scores(self):
        train, val = _three_class()
        model, _, final = local_trainer.train_local(train, val, _config("random_forest"))
        card = model_card.build_model_card(
            model, algorithm="random_forest", model_type="classification",
            train_df=train, val_df=val, feature_cols=FEATURES, metrics=final)
        assert card["evaluation"]["classification"]["confusion_matrix"][0][1:] == ["red", "green", "blue"]
        text = card["summary_text"]
        assert "Confusion matrix" in text and "red" in text and "blue" in text
        assert "Per-class precision/recall" in text
        assert "macro precision" in text and "macro recall" in text


class TestKFold:
    def test_sklearn_families_are_cross_validated_after_the_fit(self):
        """A mutation that skips the CV call leaves cross_validation None and
        the strategy at plain holdout, and this fails."""
        train, val = _three_class()
        _, history, final = local_trainer.train_local(train, val, _config("gradient_boosting"))
        cv = final["cross_validation"]
        assert final["evaluation_strategy"] == "holdout+kfold"
        assert cv["splitter"] == "StratifiedKFold"
        assert cv["n_folds"] == 5
        assert cv["primary_metric"] == "accuracy"
        assert len(cv["metrics"]["accuracy"]["folds"]) == 5
        assert cv["cv_mean"] == pytest.approx(np.mean(cv["metrics"]["accuracy"]["folds"]))
        assert cv["cv_std"] >= 0.0
        assert cv["beats_baseline"] is True
        # the history row is what MLflow logs, metric by metric: it stays flat
        assert all(not isinstance(v, (dict, list)) for v in history[0].values())

    def test_regression_folds_are_in_the_targets_units(self):
        train, val = _three_class()
        scaling = {"mean": 10.0, "std": 2.0}
        for frame in (train, val):
            frame["__label__"] = ((frame.x1 + frame.x2 - 10.0) / 2.0).astype(float)
        _, _, final = local_trainer.train_local(
            train, val, _config("linear", model_type="regression", target_scaling=scaling,
                                 class_names=None, num_classes=1, output_dim=1))
        cv = final["cross_validation"]
        assert cv["splitter"] == "KFold" and cv["primary_metric"] == "rmse"
        # unscaled rmse would be about half the holdout's; in target units they agree
        assert cv["cv_mean"] == pytest.approx(final["val_rmse"], rel=0.5)
        assert final["classification"] is None

    def test_the_mlp_keeps_its_holdout_and_says_so(self):
        train, val = _three_class()
        _, _, final = local_trainer.train_local(train, val, _config("mlp"))
        assert final["evaluation_strategy"] == "holdout"
        assert final["cross_validation"] is None
        assert final["classification"]["confusion_matrix"][0][1:] == ["red", "green", "blue"]

    def test_time_ordered_training_folds_with_time_series_split(self):
        train, val = _three_class()
        _, _, final = local_trainer.train_local(train, val, _config("linear", time_ordered=True))
        assert final["cross_validation"]["splitter"] == "TimeSeriesSplit"

    def test_evaluation_report_carries_the_fold_report_and_the_matrix(self):
        train, val = _three_class()
        _, _, final = local_trainer.train_local(train, val, _config("linear"))
        report = build_holdout_evaluation_report(
            train["__label__"].tolist(), val["__label__"].tolist(), "classification", final)
        assert report["evaluation_strategy"] == "holdout+kfold"
        assert report["cross_validation"]["cv_mean"] is not None
        assert report["classification"]["labels"] == ["red", "green", "blue"]

    def test_the_ray_path_is_reported_as_a_holdout(self):
        """Ray hands back its own metrics dict with no fold report; the
        shared report must then say "holdout", not leave the field out for
        a reader to assume the better thing."""
        report = build_holdout_evaluation_report([0, 1, 0, 1], [0, 1], "classification",
                                                 {"accuracy": 1.0})
        assert report["evaluation_strategy"] == "holdout"
        assert report["cross_validation"] is None

    def test_model_card_states_the_folds_or_the_holdout(self):
        train, val = _three_class()
        model, _, final = local_trainer.train_local(train, val, _config("random_forest"))
        card = model_card.build_model_card(
            model, algorithm="random_forest", model_type="classification",
            train_df=train, val_df=val, feature_cols=FEATURES, metrics=final)
        assert card["evaluation"]["strategy"] == "holdout+kfold"
        assert card["evaluation"]["cross_validation"]["cv_mean"] is not None
        assert "5-fold StratifiedKFold" in card["summary_text"]
        holdout_only = model_card.render_summary(
            algorithm="mlp", model_type="classification",
            metrics={"val_accuracy": 0.9, "val_f1_score": 0.9, "evaluation_strategy": "holdout",
                     "cross_validation": None},
            importances=[], baseline={}, feature_count=3, row_count=100)
        assert "not cross-validated" in holdout_only
