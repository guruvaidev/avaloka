"""Focused regression-metric and chat-report tests for MTA v2."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pandas as pd


def _mlflow_run(*, model_type, metrics):
    return SimpleNamespace(
        info=SimpleNamespace(run_id="run-summary", status="FINISHED"),
        data=SimpleNamespace(
            tags={
                "user_id": "test-user",
                "session_id": "test-session",
                "model_name": "summary-test",
                "model_type": model_type,
                "model_description": "Summary metric test",
                "model_version": "v1",
                "created_at": "2026-08-14T00:00:00+00:00",
            },
            metrics=metrics,
        ),
    )


def test_model_summary_uses_metrics_for_the_model_type():
    from app.agents.mta_v2.mlflow_manager import MLflowManager

    manager = object.__new__(MLflowManager)
    regression = manager._build_summary(
        _mlflow_run(
            model_type="regression",
            metrics={
                "final_val_loss": 0.247355,
                "final_mae": 0.31,
                "final_rmse": 0.497,
                "final_r2_score": 0.73,
            },
        ),
        "avaloka-training",
    )
    classification = manager._build_summary(
        _mlflow_run(
            model_type="classification",
            metrics={
                "final_val_loss": 0.509707,
                "final_accuracy": 0.722302,
                "final_f1_score": 0.711986,
            },
        ),
        "avaloka-training",
    )

    assert regression["final_mae"] == 0.31
    assert regression["final_rmse"] == 0.497
    assert regression["final_r2_score"] == 0.73
    assert "final_accuracy" not in regression
    assert "final_f1_score" not in regression
    assert classification["final_accuracy"] == 0.722302
    assert classification["final_f1_score"] == 0.711986
    assert "final_mae" not in classification


def test_regression_metric_calculation_uses_mae_rmse_and_r2():
    from app.agents.mta_v2.local_trainer import _compute_regression_metrics

    metrics = _compute_regression_metrics(
        labels=[1.0, 2.0, 3.0],
        predictions=[1.0, 2.0, 4.0],
    )

    assert metrics["mae"] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert metrics["rmse"] == pytest.approx((1.0 / 3.0) ** 0.5, abs=1e-6)
    assert metrics["r2_score"] == pytest.approx(0.5)


def test_local_regression_training_emits_no_classification_metrics():
    from app.agents.mta_v2.local_trainer import train_local

    train_df = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0], "__label__": [1.0, 3.0, 5.0, 7.0]})
    val_df = pd.DataFrame({"feature": [4.0, 5.0], "__label__": [9.0, 11.0]})
    config = {
        "random_seed": 42,
        "use_gpu": False,
        "feature_cols": ["feature"],
        "model_type": "regression",
        "output_dim": 1,
        "hidden_sizes": [4],
        "activation": "relu",
        "dropout": 0.0,
        "batch_norm": False,
        "optimizer": "adam",
        "learning_rate": 0.01,
        "batch_size": 2,
        "epochs": 1,
        "early_stopping_patience": 1,
    }

    _, history, final_metrics = train_local(train_df, val_df, config)

    assert len(history) == 1
    assert {"train_mae", "val_mae", "train_rmse", "val_rmse", "train_r2_score", "val_r2_score"} <= final_metrics.keys()
    assert "accuracy" not in final_metrics
    assert "f1_score" not in final_metrics


def test_local_trainer_returns_regression_metrics_in_final_payload(tmp_path):
    from app.agents.mta_v2.local_trainer import LocalTrainer

    csv_path = tmp_path / "regression.csv"
    rows = list(range(24))
    pd.DataFrame({"feature": rows, "target": [2.5 * value + 3.0 for value in rows]}).to_csv(csv_path, index=False)
    plan = {
        "model_type": "regression",
        "model_name": "regression-test",
        "model_description": "Regression metric test",
        "model_version": "v1",
        "hyperparameter_config": {
            "learning_rate": 0.01,
            "epochs": 1,
            "batch_size": 8,
            "optimizer_name": "adam",
            "random_seed": 42,
            "early_stopping_patience": 1,
            "hidden_layer_sizes": [4],
            "activation": "relu",
            "dropout_rate": 0.0,
            "batch_norm": False,
        },
        "ray_config": None,
        "data_config": {
            "dataset_uri": str(csv_path),
            "feature_columns": ["feature"],
            "target_column": "target",
        },
    }

    with patch.dict(os.environ, {
        "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "MLFLOW_BACKEND_STORE_URI": "",
        "MLFLOW_DEFAULT_ARTIFACT_ROOT": f"file://{tmp_path / 'artifacts'}",
        "MLFLOW_EXPERIMENT_NAME": "regression-metric-test",
    }):
        result = LocalTrainer(user_id="test-user", session_id="test-session").train(plan)

    assert result["model_type"] == "regression"
    assert {"final_mae", "final_rmse", "final_r2_score"} <= result.keys()
    assert "final_accuracy" not in result
    assert "final_f1_score" not in result


def test_regression_report_displays_regression_metrics_only():
    from app.agents.mta_v2.agent import ModelTrainingAgent

    report = ModelTrainingAgent()._format_training_result_report(
        {
            "model_name": "FareRegressionModel",
            "model_type": "regression",
            "model_version": "v1.0",
            "mlflow_run_id": "run-regression",
            "rows_processed": 891,
            "num_epochs_trained": 30,
            "num_features": 6,
            "num_classes": 1,
            "final_train_loss": 1883.9,
            "final_val_loss": 849.2,
            "final_accuracy": 0.0,
            "final_f1_score": 0.0,
            "final_mae": 18.25,
            "final_rmse": 29.14,
            "final_r2_score": 0.61,
        },
        use_local=True,
    )

    assert "Training Loss" in report
    assert "Validation Loss" in report
    assert "Average Error" in report
    assert "Large-Error Score" in report
    assert "Explained Variation" in report
    assert "61.00%" in report
    assert "MAE" not in report
    assert "RMSE" not in report
    assert "R²" not in report
    assert "accuracy" not in report.lower()
    assert "f1" not in report.lower()
    assert "Number of Classes" not in report
    assert "regression training was successful" in report


def test_classification_report_keeps_accuracy_and_f1():
    from app.agents.mta_v2.agent import ModelTrainingAgent

    report = ModelTrainingAgent()._format_training_result_report(
        {
            "model_type": "classification",
            "mlflow_run_id": "run-classification",
            "final_accuracy": 0.91,
            "final_f1_score": 0.89,
            "num_classes": 3,
        },
        use_local=True,
    )

    assert "Final Accuracy" in report
    assert "Final F1 Score" in report
    assert "Number of Classes" in report
    assert "Validation MAE" not in report
