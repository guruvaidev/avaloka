from unittest.mock import MagicMock, patch

from app.agents.mta_v2.inference_service_manager import InferenceServiceManager
from app.agents.mta_v2.model import ModelConfig


def _config(model_type: str = "regression") -> ModelConfig:
    return ModelConfig(
        input_size=1,
        output_size=1 if model_type == "regression" else 2,
        hidden_sizes=[4],
        feature_names=["value"],
        class_names=[] if model_type == "regression" else ["no", "yes"],
        model_type=model_type,
        model_name="test-model",
        model_version="v1",
        mlflow_run_id="run-123",
        preprocessing={
            "numeric_features": ["value"],
            "feature_scaling": {
                "value": {"method": "standard", "mean": 10.0, "scale": 2.0},
            },
            "target_scaling": {"method": "standard", "mean": 100.0, "scale": 20.0},
        },
    )


def test_rayserve_gateway_preserves_regression_scaling(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    manager = InferenceServiceManager()
    manager._resolve_model_uri = MagicMock(return_value="runs:/run-123/model_files/model.onnx")
    manager._request_rayserve = MagicMock(return_value={
        "scores": [[0.5]],
        "model_uri": "runs:/run-123/model_files/model.onnx",
        "model_loaded": True,
        "backend": "onnxruntime",
    })

    mlflow = MagicMock()
    mlflow.get_model_config.return_value = _config()
    with patch("app.agents.mta_v2.inference_service_manager.MLflowManager", return_value=mlflow):
        result = manager._inference_rayserve("run-123", {"value": 12.0})

    payload = manager._request_rayserve.call_args.args[0]
    assert payload["features"] == [1.0]
    assert result["prediction"] == 110.0
    assert result["model_type"] == "regression"


def test_rayserve_gateway_decodes_classification_probabilities(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    manager = InferenceServiceManager()
    manager._resolve_model_uri = MagicMock(return_value="runs:/run-123/model_files/model.onnx")
    manager._request_rayserve = MagicMock(return_value={
        "scores": [[-0.5, 1.0]],
        "model_uri": "runs:/run-123/model_files/model.onnx",
        "model_loaded": True,
        "backend": "onnxruntime",
    })

    mlflow = MagicMock()
    mlflow.get_model_config.return_value = _config("classification")
    with patch("app.agents.mta_v2.inference_service_manager.MLflowManager", return_value=mlflow):
        result = manager._inference_rayserve("run-123", {"value": 12.0})

    assert result["prediction"] == "yes"
    assert result["predicted_class_index"] == 1
    assert result["probabilities"]["yes"] > result["probabilities"]["no"]


def test_rayserve_gateway_returns_every_batch_row(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    manager = InferenceServiceManager()
    manager._resolve_model_uri = MagicMock(return_value="runs:/run-123/model_files/model.onnx")
    manager._request_rayserve = MagicMock(side_effect=[
        {"scores": [[0.0]], "model_loaded": True, "backend": "onnxruntime"},
        {"scores": [[1.0]], "model_loaded": True, "backend": "onnxruntime"},
    ])

    mlflow = MagicMock()
    mlflow.get_model_config.return_value = _config()
    with patch("app.agents.mta_v2.inference_service_manager.MLflowManager", return_value=mlflow):
        result = manager._inference_rayserve(
            "run-123",
            [{"value": 10.0}, {"value": 14.0}],
        )

    assert [row["prediction"] for row in result] == [100.0, 120.0]
    assert manager._request_rayserve.call_count == 2


def test_rayserve_gateway_uses_saved_pii_and_one_hot_contract(monkeypatch):
    from app.agents.pii_agent import pseudonym

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("AVALOKA_PII_KEY", "test-only-key")
    token = pseudonym("alice@example.com", salt="customer_email")
    config = ModelConfig(
        input_size=2,
        output_size=2,
        hidden_sizes=[4],
        feature_names=["customer_email"],
        class_names=["no", "yes"],
        model_type="classification",
        model_name="pii-model",
        model_version="v1",
        mlflow_run_id="run-pii",
        preprocessing={
            "version": 3,
            "pii_transformations": {
                "customer_email": {
                    "kind": "email",
                    "strategy": "pseudonymise",
                }
            },
            "categorical_features": {
                "customer_email": {
                    "encoding": "one_hot",
                    "categories": [token],
                    "output_columns": ["email_known"],
                    "unknown_column": "email_unknown",
                }
            },
            "imputation": {"customer_email": {"fill_value": "__MISSING__"}},
            "feature_scaling": {},
            "model_feature_names": ["email_known", "email_unknown"],
        },
    )
    manager = InferenceServiceManager()
    manager._resolve_model_uri = MagicMock(return_value="runs:/run-pii/model.onnx")
    manager._request_rayserve = MagicMock(return_value={
        "scores": [[0.0, 1.0]],
        "model_loaded": True,
        "backend": "onnxruntime",
    })
    mlflow = MagicMock()
    mlflow.get_model_config.return_value = config

    with patch("app.agents.mta_v2.inference_service_manager.MLflowManager", return_value=mlflow):
        manager._inference_rayserve(
            "run-pii", {"customer_email": "alice@example.com"}
        )

    payload = manager._request_rayserve.call_args.args[0]
    assert payload["features"] == [1.0, 0.0]
    assert "alice@example.com" not in str(payload)
