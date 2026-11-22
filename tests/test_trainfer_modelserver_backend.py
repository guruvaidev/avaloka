"""InferenceServiceManager's default `modelserver` backend.

Base classes (develop-1.6 @ 9cd9e122): the default-backend test fails by
assertion (the base defaults to "rayserve"); the rest error because the
modelserver methods and app.infra.model_server_scaler do not exist.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.agents.mta_v2 import inference_service_manager as ism
from app.agents.mta_v2.model import ModelConfig


def _config():
    return ModelConfig(input_size=1, output_size=1, hidden_sizes=[4], feature_names=["value"], class_names=[],
                       model_type="regression", model_name="m", model_version="1", mlflow_run_id="run-1",
                       preprocessing={"numeric_features": ["value"],
                                      "feature_scaling": {"value": {"method": "standard", "mean": 10.0, "scale": 2.0}},
                                      "target_scaling": {"method": "standard", "mean": 100.0, "scale": 20.0}})


def test_the_default_backend_is_the_model_server(monkeypatch):
    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
    import importlib
    mod = importlib.reload(ism)
    try:
        assert mod.INFERENCE_BACKEND == "modelserver"
        assert mod.InferenceServiceManager._backend() == "modelserver"
    finally:
        importlib.reload(ism)


def test_connection_refused_wakes_the_deployment_and_raises_serving_asleep(monkeypatch):
    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "modelserver")
    mgr = ism.InferenceServiceManager()
    scaler = MagicMock(); scaler.replicas.return_value = 0
    mgr._scaler_for = MagicMock(return_value=scaler)
    with patch.object(ism.requests, "post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(ism.ServingAsleep):
            mgr._post_modelserver("run-1", "predict", {"features": [1.0]})
    scaler.wake.assert_called_once()


def test_connection_refused_while_running_is_a_real_error(monkeypatch):
    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "modelserver")
    mgr = ism.InferenceServiceManager()
    scaler = MagicMock(); scaler.replicas.return_value = 2
    mgr._scaler_for = MagicMock(return_value=scaler)
    with patch.object(ism.requests, "post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(requests.ConnectionError):
            mgr._post_modelserver("run-1", "predict", {"features": [1.0]})
    scaler.wake.assert_not_called()


def test_prediction_posts_the_registry_hint_and_decodes_like_rayserve(monkeypatch):
    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "modelserver")
    mgr = ism.InferenceServiceManager()
    mgr._registry_hint = MagicMock(return_value={"registered_name": "avaloka.u.m", "version": 2})
    resp = MagicMock(status_code=200, headers={"content-type": "application/json"})
    resp.json.return_value = {"scores": [[0.5]], "model_loaded": True, "backend": "onnxruntime", "model_uri": "models:/avaloka.u.m/2"}
    mlflow = MagicMock(); mlflow.get_model_config.return_value = _config()
    with patch.object(ism.requests, "post", return_value=resp) as post, \
         patch.object(ism, "MLflowManager", return_value=mlflow):
        out = mgr.inference("run-1", {"value": 12.0})
    url, = post.call_args.args
    assert url.endswith("/models/run-1/predict")
    payload = post.call_args.kwargs["json"]
    assert payload == {"features": [1.0], "registered_name": "avaloka.u.m", "version": 2}
    assert out["prediction"] == 110.0 and out["model_type"] == "regression"
    assert "mlflow_tracking_uri" not in payload, "credentials and tracking URIs never travel in a request body"


def test_prepare_is_cold_and_configure_is_warm(monkeypatch):
    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "modelserver")
    mgr = ism.InferenceServiceManager()
    mgr._ensure_self_contained_onnx = MagicMock()
    mgr._persist_details = MagicMock()
    mgr._registry_hint = MagicMock(return_value={"registered_name": "avaloka.u.m", "version": 1})
    scaler = MagicMock(); scaler.replicas.return_value = 0
    mgr._scaler_for = MagicMock(return_value=scaler)

    with patch.object(ism.requests, "post") as post:
        cold = mgr.prepare_inference_service("run-1")
    post.assert_not_called()
    assert cold["warm"] is False and cold["model_uri"] == "models:/avaloka.u.m/1"

    resp = MagicMock(status_code=200, headers={"content-type": "application/json"})
    resp.json.return_value = {"loaded": True, "bytes": 1234, "flavor": "onnx"}
    with patch.object(ism.requests, "post", return_value=resp) as post:
        warm = mgr.configure_inference_service("run-1")
    scaler.wake.assert_called_once()
    scaler.wait_ready.assert_called_once()
    assert post.call_args.args[0].endswith("/models/run-1/load")
    assert post.call_args.kwargs["json"] == {"registered_name": "avaloka.u.m", "version": 1}
    assert warm["warm"] is True and warm["bytes"] == 1234


def test_a_422_from_the_server_is_a_plain_runtime_error(monkeypatch):
    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "modelserver")
    mgr = ism.InferenceServiceManager()
    resp = MagicMock(status_code=422, headers={"content-type": "application/json"}, text="")
    resp.json.return_value = {"detail": {"error": "ONNX artifact missing"}}
    with patch.object(ism.requests, "post", return_value=resp):
        with pytest.raises(RuntimeError, match="ONNX artifact missing"):
            mgr._post_modelserver("run-1", "load", {})


# Compatibility guard, not a feature test: it passes on the base too (rayserve
# has always been selectable). Counted as vacuous in the report.
def test_rayserve_stays_available_as_an_opt_in(monkeypatch):
    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "rayserve")
    assert ism.InferenceServiceManager._backend() == "rayserve"
