# app/serve/inference.py
"""Ray Serve inference-as-a-service.

Ray Serve application bound by the ``RayService`` CR in
``deploy/helm/ray/rayservice.yaml`` (import path ``app.serve.inference:app``).

The service loads an ONNX model from ``MODEL_URI`` (the artifact MTA writes —
``model_artifacts.onnx_path``) and runs real inference. When no model can be
loaded it falls back to a **deterministic** transform (clearly labelled — not a
"scaffold" placeholder) so the platform is still exercisable, and ``model_loaded``
always reports the truth.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

try:
    from ray import serve
    from starlette.requests import Request
except ImportError:  # allows importing/inspecting this module without ray installed
    serve = None  # type: ignore
    Request = Any  # type: ignore


def _materialize(uri: str, mlflow_tracking_uri: str | None = None) -> str:
    """Return a local path for a filesystem, MLflow, or cloud artifact URI."""
    if uri.startswith(("runs:/", "models:/", "mlflow-artifacts:/", "http://", "https://")):
        import mlflow

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        return mlflow.artifacts.download_artifacts(artifact_uri=uri)
    if uri.startswith("gs://"):
        import gcsfs  # optional; only needed for cloud artifacts

        fs = gcsfs.GCSFileSystem()
        fd, local = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        fs.get(uri, local)
        return local
    if uri.startswith("file://"):
        return unquote(urlparse(uri).path)
    return os.path.expanduser(uri)


def _to_vector(features: Any) -> List[float]:
    """Coerce a feature dict/list into a float vector without reordering it."""
    if isinstance(features, dict):
        return [float(value) for value in features.values()]
    if isinstance(features, (list, tuple)):
        return [float(x) for x in features]
    return [float(features)]


class _Model:
    """Model wrapper independent of Ray, so it is unit-testable without a cluster."""

    def __init__(self, model_uri: str | None = None, mlflow_tracking_uri: str | None = None):
        self.model_uri = model_uri
        self.session = None
        self.input_name = None
        self.load_error = None
        self.backend = "deterministic-stub"
        if model_uri:
            self._load(model_uri, mlflow_tracking_uri)

    def _load(self, uri: str, mlflow_tracking_uri: str | None = None) -> None:
        try:
            local = _materialize(uri, mlflow_tracking_uri)
            import onnxruntime as ort

            self.session = ort.InferenceSession(local, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.backend = "onnxruntime"
            logger.info("[inference] Loaded ONNX model from %s (input=%s)", uri, self.input_name)
        except Exception as e:  # noqa: BLE001 - degrade to deterministic stand-in
            logger.warning("[inference] Could not load model from %s: %s — using deterministic stand-in", uri, e)
            self.session = None
            self.load_error = str(e)
            self.backend = "deterministic-stub"

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def predict(self, features: Any) -> Dict[str, Any]:
        vec = _to_vector(features)
        if self.session is not None:
            import numpy as np

            arr = np.array([vec], dtype=np.float32)
            out = self.session.run(None, {self.input_name: arr})
            pred = out[0]
            scores = pred.tolist() if hasattr(pred, "tolist") else pred
            if pred.size > 1:
                prediction = int(np.argmax(pred, axis=-1).reshape(-1)[0])
            else:
                prediction = float(pred.reshape(-1)[0])
            return {
                "prediction": prediction,
                "scores": scores,
                "model_uri": self.model_uri,
                "model_loaded": True,
                "backend": "onnxruntime",
            }
        # Deterministic stand-in — NOT a scaffold; the value is a stable function of
        # the inputs so tests can assert reproducibility.
        return {
            "prediction": round(sum(vec), 6),
            "model_uri": self.model_uri,
            "model_loaded": False,
            "backend": "deterministic-stub",
            "note": "deterministic stand-in: no ONNX model loaded from MODEL_URI",
        }


if serve is not None:

    @serve.deployment(
        num_replicas=int(os.environ.get("SERVE_REPLICAS", "1")),
        ray_actor_options={"num_cpus": 1},
    )
    class InferenceService:
        """A Ray Serve deployment that lazily loads selected MLflow ONNX models."""

        def __init__(self, model_uri: str | None = None):
            self._model = _Model(model_uri)
            self._models: Dict[str, _Model] = {}
            if model_uri and self._model.loaded:
                self._models[f"|{model_uri}"] = self._model

        def _get_model(
            self,
            model_uri: str | None,
            mlflow_tracking_uri: str | None = None,
        ) -> _Model:
            if not model_uri:
                return self._model
            cache_key = f"{mlflow_tracking_uri or ''}|{model_uri}"
            if cache_key not in self._models:
                model = _Model(model_uri, mlflow_tracking_uri)
                if model.loaded:
                    self._models[cache_key] = model
                return model
            return self._models[cache_key]

        def predict(
            self,
            features: Dict[str, Any],
            model_uri: str | None = None,
            mlflow_tracking_uri: str | None = None,
        ) -> Dict[str, Any]:
            return self._get_model(model_uri, mlflow_tracking_uri).predict(features)

        async def __call__(self, request: "Request") -> Dict[str, Any]:
            if request.method == "GET":
                return {
                    "status": "ready",
                    "service": "avaloka-inference",
                    "model_loaded": self._model.loaded,
                    "model_uri": self._model.model_uri,
                    "backend": self._model.backend,
                }
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            model_uri = payload.get("model_uri")
            mlflow_tracking_uri = payload.get("mlflow_tracking_uri")
            model = self._get_model(model_uri, mlflow_tracking_uri)
            if payload.get("load_only"):
                result = {
                    "status": "ready" if model.loaded else "error",
                    "model_uri": model_uri,
                    "model_loaded": model.loaded,
                    "backend": model.backend,
                }
                if model.load_error:
                    result["error"] = model.load_error
                return result
            return model.predict(payload.get("features", payload))

    # Bound application object referenced by the RayService import_path.
    app = InferenceService.bind(model_uri=os.environ.get("MODEL_URI"))
else:  # pragma: no cover - only when ray is absent
    app = None
