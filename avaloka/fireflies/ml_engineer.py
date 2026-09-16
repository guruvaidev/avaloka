"""ML Engineer — replaces the ML platform engineer.

Economic output: a real deployment package — container, API server, schemas,
dependency lock, tests, monitoring config and manifests for local / Docker /
Kubernetes / Ray. Avaloka integrates well-trodden serving primitives rather
than rebuilding them; the value is that the package is generated from the
*validated* model with its feature contract, not hand-assembled.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from avaloka.fireflies.base import Firefly
from avaloka.util import write_json, write_text, write_yaml

_JSON_TYPE = {
    "numeric": "number", "boolean": "boolean", "datetime": "string",
    "categorical": "string", "text": "string", "id": "string", "constant": "string",
}


def _pkg(name: str, fallback: str) -> str:
    try:
        return f"{name}=={version(name)}"
    except PackageNotFoundError:  # pragma: no cover
        return f"{name}>={fallback}"


class MLEngineer(Firefly):
    name = "ml_engineer"
    human_role = "ML platform engineer"

    def perform(self) -> tuple[str, float, dict[str, Any]]:
        training = self.ctx.blackboard["training"]
        profile = self.ctx.blackboard["profile"]
        feature_cols = training.X_columns
        col_by_name = {c["name"]: c for c in profile["columns"]}

        self._inference_schema(feature_cols, col_by_name, training)
        self._feature_contract(feature_cols, col_by_name)
        self._requirements_lock()
        self._service_py(training)
        self._dockerfile()
        self._tests(feature_cols, col_by_name)
        self._monitoring_config(feature_cols)
        self._deployment_manifests()
        self._mlflow(training)

        return (
            "Generated Level-2 deployment package: service.py, Dockerfile, inference "
            "schema, feature contract, tests, monitoring + k8s/ray/compose manifests.",
            6 * 60.0,
            {"artifacts": ["service.py", "Dockerfile", "deployment/"]},
        )

    # --- schemas ----------------------------------------------------------
    def _example_value(self, col: dict[str, Any]) -> Any:
        if col.get("sample_values"):
            return col["sample_values"][0]
        return 0 if col["role"] == "numeric" else "example"

    def _inference_schema(self, feature_cols, col_by_name, training) -> None:
        props = {}
        for name in feature_cols:
            col = col_by_name.get(name, {"role": "numeric"})
            props[name] = {"type": _JSON_TYPE.get(col["role"], "string")}
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Avaloka inference request",
            "type": "object",
            "properties": {"instances": {"type": "array", "items": {
                "type": "object", "properties": props,
                "required": list(feature_cols)}}},
            "required": ["instances"],
            "x-avaloka": {"task": training.task, "metric": training.metric,
                          "target": self.ctx.target},
            "x-response": {"predictions": "array", "model_version": "string"},
        }
        write_json(self.ctx.path("inference_schema.json"), schema)
        example = {"instances": [{n: self._example_value(col_by_name.get(n, {})) for n in feature_cols}]}
        write_json(self.ctx.path("tests", "example_request.json"), example)

    def _feature_contract(self, feature_cols, col_by_name) -> None:
        features = []
        for name in feature_cols:
            col = col_by_name.get(name, {"role": "numeric"})
            entry: dict[str, Any] = {"name": name, "role": col["role"],
                                     "dtype": col.get("dtype", "object"),
                                     "required": col.get("missing_pct", 0) < 0.5}
            if col.get("role") == "numeric" and col.get("stats"):
                entry["expected_range"] = [col["stats"]["min"], col["stats"]["max"]]
            elif col.get("top_values"):
                entry["known_categories"] = [t["value"] for t in col["top_values"]]
            features.append(entry)
        write_yaml(self.ctx.path("feature_contract.yaml"), {
            "target": self.ctx.target,
            "features": features,
            "enforcement": "Reject or flag requests whose features fall outside the contract.",
        })

    def _requirements_lock(self) -> None:
        reqs = [
            _pkg("scikit-learn", "1.3"), _pkg("pandas", "2.0"), _pkg("numpy", "1.24"),
            "fastapi>=0.110", "uvicorn[standard]>=0.29", "pydantic>=2.5",
        ]
        write_text(self.ctx.path("requirements.lock"), "\n".join(reqs) + "\n")

    # --- service ----------------------------------------------------------
    def _service_py(self, training) -> None:
        src = f'''"""Avaloka inference service for mission {self.ctx.mission_id}.

Loads the validated model pipeline and serves predictions over a typed REST API
with a health endpoint. Generated by Avaloka; safe to edit.
"""

import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"
MODEL_VERSION = "{self.ctx.mission_id}"
TASK = "{training.task}"
FEATURES = {training.X_columns!r}

with MODEL_PATH.open("rb") as fh:
    _model = pickle.load(fh)

app = FastAPI(title="Avaloka inference", version=MODEL_VERSION)


class PredictRequest(BaseModel):
    instances: list[dict]


@app.get("/health")
def health() -> dict:
    return {{"status": "ok", "model_version": MODEL_VERSION, "task": TASK}}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    df = pd.DataFrame(req.instances)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = None
    df = df[FEATURES]
    preds = _model.predict(df).tolist()
    out = {{"predictions": preds, "model_version": MODEL_VERSION}}
    if TASK == "binary_classification" and hasattr(_model, "predict_proba"):
        out["probabilities"] = _model.predict_proba(df)[:, 1].tolist()
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
'''
        write_text(self.ctx.path("service.py"), src)

    def _dockerfile(self) -> None:
        write_text(self.ctx.path("Dockerfile"), '''# Avaloka-generated inference container
FROM python:3.11-slim

WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY model/ ./model/
COPY service.py ./

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; \\
  sys.exit(0) if urllib.request.urlopen('http://localhost:8080/health').status==200 else sys.exit(1)"
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]
''')

    def _tests(self, feature_cols, col_by_name) -> None:
        src = f'''"""Smoke + contract tests for the Avaloka inference service."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

import service

client = TestClient(service.app)
EXAMPLE = json.loads((Path(__file__).parent / "example_request.json").read_text())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_shape():
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == len(EXAMPLE["instances"])


def test_feature_contract():
    assert service.FEATURES == {list(feature_cols)!r}
'''
        write_text(self.ctx.path("tests", "test_service.py"), src)

    def _monitoring_config(self, feature_cols) -> None:
        write_yaml(self.ctx.path("monitoring_config.yaml"), {
            "model_version": self.ctx.mission_id,
            "data_drift": {"method": "population_stability_index", "warn_threshold": 0.1,
                           "alert_threshold": 0.2, "features": list(feature_cols)},
            "prediction_drift": {"method": "psi", "alert_threshold": 0.2},
            "performance": {"recompute_on": "labelled_feedback",
                            "alert_if_metric_drops_by": 0.05},
            "operational": {"latency_p95_ms_alert": self.ctx.deployment_latency_ms or 250,
                            "error_rate_alert": 0.02},
            "retrain_triggers": ["drift_score>0.20", "performance_drop>0.05", "schema_change"],
        })

    # --- manifests --------------------------------------------------------
    def _deployment_manifests(self) -> None:
        name = f"avaloka-{self.ctx.mission_id[:8]}"
        write_yaml(self.ctx.path("deployment", "kubernetes.yaml"), [
            {"apiVersion": "apps/v1", "kind": "Deployment",
             "metadata": {"name": name, "labels": {"app": name}},
             "spec": {"replicas": 1, "selector": {"matchLabels": {"app": name}},
                      "template": {"metadata": {"labels": {"app": name}},
                                   "spec": {"containers": [{
                                       "name": "inference", "image": f"{name}:latest",
                                       "ports": [{"containerPort": 8080}],
                                       "readinessProbe": {"httpGet": {"path": "/health", "port": 8080}},
                                       "resources": {"requests": {"cpu": "250m", "memory": "512Mi"},
                                                     "limits": {"cpu": "1", "memory": "1Gi"}}}]}}}},
            {"apiVersion": "v1", "kind": "Service",
             "metadata": {"name": name},
             "spec": {"selector": {"app": name}, "ports": [{"port": 80, "targetPort": 8080}]}},
        ])
        write_yaml(self.ctx.path("deployment", "rayservice.yaml"), {
            "apiVersion": "ray.io/v1", "kind": "RayService",
            "metadata": {"name": name},
            "spec": {"serveConfigV2": (
                "applications:\n  - name: avaloka\n    import_path: service:app\n"
                "    deployments:\n      - name: inference\n"
                "        autoscaling_config:\n          min_replicas: 0\n          max_replicas: 4\n")},
        })
        write_yaml(self.ctx.path("deployment", "local-compose.yaml"), {
            "services": {"inference": {
                "build": ".", "ports": ["8080:8080"],
                "healthcheck": {"test": ["CMD", "python", "-c",
                                         "import urllib.request;urllib.request.urlopen('http://localhost:8080/health')"],
                                "interval": "30s"}}}})

    def _mlflow(self, training) -> None:
        # Minimal MLmodel descriptor so the package is registry-ready (MLflow
        # provides lineage/versioning/aliases — Avaloka integrates, not rebuilds).
        write_yaml(self.ctx.path("mlflow", "MLmodel"), {
            "artifact_path": "model",
            "flavors": {"python_function": {"loader_module": "mlflow.sklearn",
                                            "model_path": "../model/model.pkl",
                                            "python_version": self.ctx.environment()["python"]}},
            "run_id": self.ctx.mission_id,
            "avaloka": {"task": training.task, "metric": training.metric,
                        "selected": training.best.name},
        })
        write_text(self.ctx.path("mlflow", "register.py"), f'''"""Register this model with an MLflow tracking server."""

import mlflow, pickle

with open("model/model.pkl", "rb") as fh:
    model = pickle.load(fh)

with mlflow.start_run(run_name="avaloka-{self.ctx.mission_id}"):
    mlflow.sklearn.log_model(model, "model", registered_model_name="{self.ctx.target or 'avaloka_model'}")
    mlflow.log_metric("{training.metric}", {training.best.primary_score!r})
''')
