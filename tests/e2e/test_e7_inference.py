"""
E7 -- Ray Serve inference (the 1.5.2 headline feature), unit/contract level.

Everything in this module runs with NO Kubernetes cluster and NO network:
kubectl/subprocess and ``requests`` are stubbed, ONNX artifacts are exported to a
tmp dir with torch, and the two API-level legs ride the hermetic in-process app
from tests/e2e/conftest.py. Nothing here is deployed-only, so no leg skips for
mode; individual tests skip only when ray/onnxruntime are absent from the venv.

Coverage map:
  E7.02  configure/stop lifecycle   -- selected MLflow run is loaded and persisted
  E7.03  serve honesty              -- stub must never claim model_loaded=True
  E7.04  backend dispatch           -- default, explicit, and invalid values
  E7.05  _to_vector feature order   -- caller order is preserved
  E7.06  model lifecycle            -- soft-delete risk pinned at source level
  E7.07  autoscaling (with E12.07)  -- chart numbers pinned so the plan cannot drift
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
import subprocess
import types
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest
import requests

from tests.e2e.conftest import REPO_ROOT

yaml = pytest.importorskip("yaml", reason="E7.07 parses the Ray charts with PyYAML")
pytest.importorskip("onnxruntime", reason="E7.03/E7.05 need onnxruntime to load a real ONNX model")

import app.agents.mta_v2.inference_service_manager as ism  # noqa: E402
import app.serve.inference as serve_inference  # noqa: E402
from app.agents.mta_v2.mlflow_manager import MLflowManager  # noqa: E402
from app.agents.mta_v2.model import DynamicMLP, ModelConfig  # noqa: E402

_HAS_RAY = serve_inference.serve is not None

RAYSERVICE_CHART = REPO_ROOT / "deploy" / "helm" / "ray" / "rayservice.yaml"
RAYCLUSTER_CHART = REPO_ROOT / "deploy" / "helm" / "ray" / "raycluster.yaml"

# The order local_trainer.py:374 builds tensors in (config["feature_cols"]).
# Deliberately not alphabetical -- that is the whole point of E7.05.
TRAINING_FEATURE_ORDER: List[str] = ["zip_code", "age", "income"]
TRAINING_FEATURES: Dict[str, float] = {"zip_code": 90210.0, "age": 41.0, "income": 82000.0}

DISPATCH_METHODS = (
    "_configure_rayserve",
    "_inference_rayserve",
    "_stop_rayserve",
    "_configure_gateway",
    "_inference_gateway",
    "_stop_gateway",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _trained_model_config(onnx_uri: str) -> ModelConfig:
    """A ModelConfig exactly as MTA writes it for a completed training run."""
    return ModelConfig(
        input_size=len(TRAINING_FEATURE_ORDER),
        output_size=2,
        hidden_sizes=[8],
        feature_names=list(TRAINING_FEATURE_ORDER),
        class_names=["no", "yes"],
        target_column="churn",
        model_name="churn-mlp",
        model_version="1",
        task_id="task-1",
        mlflow_run_id="run-e7",
    )


def _fake_mlflow_manager(cfg: Optional[ModelConfig]) -> type:
    class _FakeMLflowManager:
        ONNX_ARTIFACT_PATH = MLflowManager.ONNX_ARTIFACT_PATH

        def __init__(self, *a: Any, **k: Any) -> None:
            self.updated: List[Any] = []

        def get_model_config(self, mlflow_run_id: str) -> Optional[ModelConfig]:
            return cfg

        def update_model_config(self, mlflow_run_id: str, config: Any) -> None:
            self.updated.append(config)

    return _FakeMLflowManager


def _record_subprocess(monkeypatch: pytest.MonkeyPatch, returncode: int = 0) -> List[List[str]]:
    """Replace subprocess.run inside the manager module; never spawn a process."""
    calls: List[List[str]] = []

    def _run(argv: List[str], **kw: Any) -> subprocess.CompletedProcess:
        calls.append(list(argv))
        return subprocess.CompletedProcess(
            argv, returncode, stdout="", stderr="" if returncode == 0 else "error: kubectl failed"
        )

    monkeypatch.setattr(ism, "subprocess", types.SimpleNamespace(run=_run))
    return calls


def _stub_dispatch(monkeypatch: pytest.MonkeyPatch, module: types.ModuleType) -> List[str]:
    """Neutralise every backend implementation and record which one is reached."""
    calls: List[str] = []

    def _make(name: str) -> Callable[..., None]:
        def _impl(self: Any, *a: Any, **k: Any) -> None:
            calls.append(name)

        return _impl

    for name in DISPATCH_METHODS:
        monkeypatch.setattr(module.InferenceServiceManager, name, _make(name))
    return calls


def _load_chart(path: Path) -> List[Dict[str, Any]]:
    """Plain yaml.safe_load: a guard asserts these charts carry no Helm templating."""
    text = path.read_text(encoding="utf-8")
    assert "{{" not in text, f"{path.name} gained Helm templating; switch this test to regex"
    return [doc for doc in yaml.safe_load_all(text) if doc]


def _serve_app_config(rayservice: Dict[str, Any]) -> Dict[str, Any]:
    return yaml.safe_load(rayservice["spec"]["serveConfigV2"])["applications"][0]


@pytest.fixture(scope="module")
def onnx_model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A real 3-feature ONNX artifact, exported the way DynamicMLP.save_onnx does."""
    torch = pytest.importorskip("torch", reason="exporting the ONNX fixture needs torch")
    assert torch is not None
    model = DynamicMLP(
        input_size=len(TRAINING_FEATURE_ORDER),
        output_size=2,
        hidden_sizes=[4],
        dropout=0.0,
        batch_norm=False,
    )
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    model.save_onnx(str(path))
    return str(path)


@pytest.fixture()
def ism_fresh() -> Iterator[Callable[[Optional[str]], types.ModuleType]]:
    """Re-import the manager under a chosen INFERENCE_BACKEND env value."""
    original = os.environ.get("INFERENCE_BACKEND")

    def _load(backend: Optional[str]) -> types.ModuleType:
        if backend is None:
            os.environ.pop("INFERENCE_BACKEND", None)
        else:
            os.environ["INFERENCE_BACKEND"] = backend
        return importlib.reload(ism)

    yield _load

    if original is None:
        os.environ.pop("INFERENCE_BACKEND", None)
    else:
        os.environ["INFERENCE_BACKEND"] = original
    importlib.reload(ism)


def _wire_hermetic_inference(
    monkeypatch: pytest.MonkeyPatch, user_id: str, model_uri: Optional[str] = None
) -> None:
    """Point POST /api/models/{run_id}/inference at an in-process _Model, no HTTP."""
    import app.agents.mta_v2.mlflow_manager as mlflow_manager_module

    class _FakeMLflowManager:
        ONNX_ARTIFACT_PATH = MLflowManager.ONNX_ARTIFACT_PATH

        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def get_model_details(self, run_id: str) -> Dict[str, Any]:
            return {
                "user_id": user_id,
                "mlflow_run_id": run_id,
                "inference_service_details": {"backend": "rayserve"},
            }

        def get_model_config(self, run_id: str) -> ModelConfig:
            return _trained_model_config(model_uri or "")

    monkeypatch.setattr(mlflow_manager_module, "MLflowManager", _FakeMLflowManager)
    monkeypatch.setattr(ism, "MLflowManager", _FakeMLflowManager)

    model = serve_inference._Model(model_uri)

    def _post(url: str, json: Optional[Dict[str, Any]] = None, **kw: Any) -> Any:
        payload = model.predict((json or {}).get("features"))
        return types.SimpleNamespace(
            status_code=200, raise_for_status=lambda: None, json=lambda: payload
        )

    monkeypatch.setattr(
        ism,
        "requests",
        types.SimpleNamespace(post=_post, RequestException=requests.RequestException),
    )
    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "rayserve")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


# ---------------------------------------------------------------------------
# E7.03 -- serve honesty: a stub must never masquerade as a loaded model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri_kind",
    ["none", "empty", "missing-file", "directory", "not-an-onnx-file"],
)
def test_e7_03_stub_never_claims_model_loaded(uri_kind: str, tmp_path: Path) -> None:
    """Every un-loadable MODEL_URI degrades to a labelled stub reporting model_loaded=False."""
    garbage = tmp_path / "garbage.onnx"
    garbage.write_bytes(b"this is not a protobuf")
    uris: Dict[str, Optional[str]] = {
        "none": None,
        "empty": "",
        "missing-file": str(tmp_path / "absent.onnx"),
        "directory": str(tmp_path),
        "not-an-onnx-file": str(garbage),
    }

    model = serve_inference._Model(uris[uri_kind])

    assert model.session is None
    assert model.loaded is False
    assert model.backend == "deterministic-stub"

    out = model.predict(TRAINING_FEATURES)
    assert out["model_loaded"] is False
    assert out["backend"] == "deterministic-stub"
    assert "no ONNX model" in out["note"]
    assert isinstance(out["prediction"], float)


def test_e7_03_stub_prediction_is_deterministic_and_labelled() -> None:
    """The stand-in is a stable function of its inputs and always carries the stub label."""
    model = serve_inference._Model(None)

    first = model.predict(TRAINING_FEATURES)
    second = model.predict(dict(TRAINING_FEATURES))
    other = model.predict({"zip_code": 1.0, "age": 2.0, "income": 3.0})

    assert first == second
    assert first["prediction"] != other["prediction"]
    assert {"note", "backend", "model_loaded"} <= set(first)


def test_e7_03_loaded_model_reports_model_loaded_true(onnx_model_path: str) -> None:
    """A genuinely loaded ONNX session runs onnxruntime and reports model_loaded=True."""
    model = serve_inference._Model(onnx_model_path)

    assert model.loaded is True
    assert model.backend == "onnxruntime"

    out = model.predict(TRAINING_FEATURES)
    assert out["model_loaded"] is True
    assert out["backend"] == "onnxruntime"
    assert "note" not in out
    assert out["model_uri"] == onnx_model_path


@pytest.mark.skipif(not _HAS_RAY, reason="ray is not installed; the Serve deployment cannot be built")
def test_e7_03_status_endpoint_reports_backend_truthfully(onnx_model_path: str) -> None:
    """The Serve deployment's readiness payload mirrors the wrapped model's real state."""
    cls = serve_inference.InferenceService.func_or_class

    stub = cls(model_uri=None)
    real = cls(model_uri=onnx_model_path)

    assert (stub._model.loaded, stub._model.backend) == (False, "deterministic-stub")
    assert (real._model.loaded, real._model.backend) == (True, "onnxruntime")


def test_e7_03_api_rejects_an_unloaded_gateway_model(api, monkeypatch: pytest.MonkeyPatch) -> None:
    """The API never returns a deterministic stub as a successful model prediction."""
    if api.mode != "hermetic":
        pytest.skip("this leg stubs in-process collaborators; it is meaningless against a deployment")
    _wire_hermetic_inference(monkeypatch, "user-1", model_uri=None)

    resp = api.post("/api/models/run-e7/inference", user="user-1", json={"features": TRAINING_FEATURES})

    assert resp.status_code == 502
    assert "did not load" in resp.text


# ---------------------------------------------------------------------------
# E7.04 -- backend dispatch
# ---------------------------------------------------------------------------


def test_e7_04_default_backend_is_rayserve(ism_fresh) -> None:
    """With INFERENCE_BACKEND unset the manager selects the in-cluster RayServe backend."""
    module = ism_fresh(None)
    assert module.INFERENCE_BACKEND == "rayserve"


@pytest.mark.parametrize("raw", ["  RayServe  ", "RAYSERVE", "RayServe", "rayserve\n"])
def test_e7_04_pins_backend_value_is_case_and_space_normalised(ism_fresh, raw: str) -> None:
    """Case/whitespace variants of a known backend normalise; only unknown words are at risk."""
    module = ism_fresh(raw)
    assert module.INFERENCE_BACKEND == "rayserve"


def test_e7_04_rayserve_backend_dispatches_to_rayserve(monkeypatch: pytest.MonkeyPatch) -> None:
    """INFERENCE_BACKEND=rayserve routes configure/inference/stop to the RayServe methods."""
    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "rayserve")
    calls = _stub_dispatch(monkeypatch, ism)
    manager = ism.InferenceServiceManager()

    manager.configure_inference_service("run-e7")
    manager.inference("run-e7", {"features": TRAINING_FEATURES})
    manager.stop_inference_service("run-e7")

    assert calls == ["_configure_rayserve", "_inference_rayserve", "_stop_rayserve"]


def test_e7_04_explicit_gateway_backend_dispatches_to_legacy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """INFERENCE_BACKEND=gateway still routes to the legacy GCP API Gateway methods."""
    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "gateway")
    calls = _stub_dispatch(monkeypatch, ism)
    manager = ism.InferenceServiceManager()

    manager.configure_inference_service("run-e7")
    manager.inference("run-e7", {"features": TRAINING_FEATURES})
    manager.stop_inference_service("run-e7")

    assert calls == ["_configure_gateway", "_inference_gateway", "_stop_gateway"]


@pytest.mark.parametrize("backend", ["ray-serve", "ray_serve", "rayserv", "serve", "kserve", ""])
def test_e7_04_unknown_backend_is_rejected(ism_fresh, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """An unrecognised INFERENCE_BACKEND raises a configuration error, never falls back to gateway."""
    module = ism_fresh(backend)
    calls = _stub_dispatch(monkeypatch, module)
    manager = module.InferenceServiceManager()

    with pytest.raises((ValueError, RuntimeError)):
        manager.configure_inference_service("run-e7")

    assert calls == []


# ---------------------------------------------------------------------------
# E7.05 -- feature ordering into the ONNX session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "features, order",
    [
        (TRAINING_FEATURES, TRAINING_FEATURE_ORDER),
        ({"b_second": 2.0, "a_first": 1.0}, ["b_second", "a_first"]),
    ],
)
def test_e7_05_to_vector_follows_training_feature_order(
    features: Dict[str, float], order: List[str]
) -> None:
    """A feature dict is vectorised in the model's training order, not alphabetically."""
    assert serve_inference._to_vector(features) == [features[k] for k in order]


def test_e7_05_pins_list_input_passes_through_unreordered() -> None:
    """List/tuple features are passed through in caller order (only dicts get re-sorted)."""
    assert serve_inference._to_vector([3, 1, 2]) == [3.0, 1.0, 2.0]
    assert serve_inference._to_vector((3, 1, 2)) == [3.0, 1.0, 2.0]
    assert serve_inference._to_vector(7) == [7.0]


def test_e7_05_mapping_input_matches_explicit_training_order(onnx_model_path: str) -> None:
    """Mapping input preserves caller/training order rather than sorting feature names."""
    model = serve_inference._Model(onnx_model_path)

    as_sent = model.predict(TRAINING_FEATURES)["prediction"]
    in_training_order = model.predict([TRAINING_FEATURES[k] for k in TRAINING_FEATURE_ORDER])["prediction"]

    assert serve_inference._to_vector(TRAINING_FEATURES) == [
        TRAINING_FEATURES[name] for name in TRAINING_FEATURE_ORDER
    ]
    assert as_sent == in_training_order


def test_e7_05_pins_missing_feature_raises_onnx_shape_error(onnx_model_path: str) -> None:
    """A missing feature shortens the vector and surfaces as an ONNX shape error, not a 4xx."""
    model = serve_inference._Model(onnx_model_path)
    short = {"age": 41.0, "income": 82000.0}

    assert len(serve_inference._to_vector(short)) == 2

    with pytest.raises(Exception) as exc:
        model.predict(short)
    assert type(exc.value).__module__.startswith("onnxruntime")


def test_e7_05_pins_non_numeric_feature_raises_valueerror_uncaught() -> None:
    """Non-numeric features raise ValueError out of _to_vector; nothing converts it to a 4xx."""
    with pytest.raises(ValueError):
        serve_inference._to_vector({"age": "forty-one"})
    with pytest.raises(ValueError):
        serve_inference._Model(None).predict({"age": "forty-one"})


def test_e7_05_bad_features_do_not_return_a_stub_prediction(api, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed input cannot turn an unloaded gateway into a successful prediction."""
    if api.mode != "hermetic":
        pytest.skip("this leg stubs in-process collaborators; it is meaningless against a deployment")
    _wire_hermetic_inference(monkeypatch, "user-1", model_uri=None)

    resp = api.post(
        "/api/models/run-e7/inference",
        user="user-1",
        json={"features": {"age": "forty-one"}},
    )

    assert resp.status_code == 502
    assert "Inference gateway failed" in resp.text


# ---------------------------------------------------------------------------
# E7.02 -- configure / stop lifecycle
# ---------------------------------------------------------------------------


def test_e7_02_pins_model_config_has_no_model_artifacts_field() -> None:
    """Root cause: ModelConfig defines no model_artifacts field and from_dict drops the key."""
    assert "model_artifacts" not in ModelConfig.__dataclass_fields__

    cfg = ModelConfig.from_dict(
        {
            "input_size": 3,
            "output_size": 2,
            "hidden_sizes": [8],
            "model_artifacts": {"onnx_path": "gs://bucket/run-e7/model.onnx"},
        }
    )
    assert getattr(cfg, "model_artifacts", None) is None


def test_e7_02_resolves_the_selected_runs_onnx_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an override, Ray Serve receives the selected run's canonical ONNX URI."""
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setattr(
        ism, "MLflowManager", _fake_mlflow_manager(_trained_model_config("gs://bucket/model.onnx"))
    )

    assert ism.InferenceServiceManager()._resolve_model_uri("run-e7") == (
        f"runs:/run-e7/{MLflowManager.ONNX_ARTIFACT_PATH}"
    )


def test_e7_02_pins_model_uri_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """MODEL_URI remains an explicit deployment override."""
    monkeypatch.setenv("MODEL_URI", "gs://bucket/run-e7/model.onnx")
    monkeypatch.setattr(ism, "MLflowManager", _fake_mlflow_manager(None))

    assert ism.InferenceServiceManager()._resolve_model_uri("run-e7") == "gs://bucket/run-e7/model.onnx"


def test_e7_02_configure_binds_the_runs_onnx_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configuring a completed training run resolves that run's ONNX artifact URI."""
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "rayserve")
    monkeypatch.setattr(
        ism, "MLflowManager", _fake_mlflow_manager(_trained_model_config("gs://bucket/run-e7/model.onnx"))
    )
    monkeypatch.setattr(ism.InferenceServiceManager, "_ensure_self_contained_onnx", lambda self, run_id: None)
    monkeypatch.setattr(
        ism.InferenceServiceManager,
        "_load_rayserve_model",
        lambda self, uri: {"model_loaded": True, "model_uri": uri},
    )

    details = ism.InferenceServiceManager().configure_inference_service("run-e7")

    assert details["model_uri"].endswith(".onnx")


def test_e7_02_configure_loads_before_persisting(monkeypatch: pytest.MonkeyPatch) -> None:
    """A service is marked configured only after Ray Serve loads the selected run."""
    from unittest.mock import MagicMock

    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "rayserve")
    manager = ism.InferenceServiceManager()
    manager._ensure_self_contained_onnx = MagicMock()
    manager._load_rayserve_model = MagicMock(return_value={"model_loaded": True})
    manager._persist_details = MagicMock()

    details = manager.configure_inference_service("run-e7")

    model_uri = f"runs:/run-e7/{MLflowManager.ONNX_ARTIFACT_PATH}"
    manager._ensure_self_contained_onnx.assert_called_once_with("run-e7")
    manager._load_rayserve_model.assert_called_once_with(model_uri)
    manager._persist_details.assert_called_once_with("run-e7", details)


def test_e7_02_stop_clears_only_the_selected_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stopping a model clears its registry state without shutting down shared Ray Serve."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "rayserve")
    manager = ism.InferenceServiceManager()
    manager._persist_details = MagicMock()

    assert manager.stop_inference_service("run-e7") is None
    manager._persist_details.assert_called_once_with("run-e7", None)


# ---------------------------------------------------------------------------
# E7.06 -- model lifecycle after delete
# ---------------------------------------------------------------------------


def test_e7_06_pins_deleted_model_details_have_no_lifecycle_guard() -> None:
    """delete_model is MLflow's soft delete_run and get_model_details never checks
    lifecycle_stage, so 'inference on a deleted model -> 404' is not guaranteed."""
    delete_src = inspect.getsource(MLflowManager.delete_model)
    details_src = inspect.getsource(MLflowManager.get_model_details)

    assert "delete_run" in delete_src
    assert "lifecycle_stage" not in details_src
    assert "lifecycle_stage" not in inspect.getsource(MLflowManager.get_model_config)


# ---------------------------------------------------------------------------
# E7.07 / E12.07 -- what the shipped charts actually scale
# ---------------------------------------------------------------------------


def test_e7_07_pins_inference_serve_app_is_fixed_at_one_replica() -> None:
    """The inference Serve app pins num_replicas: 1 with no autoscaling_config."""
    rayservice = _load_chart(RAYSERVICE_CHART)[0]
    assert rayservice["kind"] == "RayService"

    application = _serve_app_config(rayservice)
    deployment = application["deployments"][0]

    assert deployment["name"] == "InferenceService"
    assert deployment["num_replicas"] == 1
    assert "autoscaling_config" not in deployment
    assert application["import_path"] == "app.serve.inference:app"
    assert application["runtime_env"]["env_vars"]["MODEL_URI"] == ""


def test_e7_07_pins_inference_rayservice_has_no_cluster_autoscaler() -> None:
    """The inference RayService sets no enableInTreeAutoscaling; its workers cap at 3."""
    rayservice = _load_chart(RAYSERVICE_CHART)[0]
    cluster = rayservice["spec"]["rayClusterConfig"]
    worker = cluster["workerGroupSpecs"][0]

    assert "enableInTreeAutoscaling" not in cluster
    assert "autoscalerOptions" not in cluster
    assert worker["groupName"] == "serve-workers"
    assert (worker["replicas"], worker["minReplicas"], worker["maxReplicas"]) == (1, 1, 3)


def test_e7_07_rayservice_receives_mlflow_and_object_store_environment() -> None:
    """The API and Ray pods must resolve the same runs:/ URI and S3 artifacts."""
    rayservice = _load_chart(RAYSERVICE_CHART)[0]
    cluster = rayservice["spec"]["rayClusterConfig"]
    head = cluster["headGroupSpec"]["template"]["spec"]["containers"][0]
    worker = cluster["workerGroupSpecs"][0]["template"]["spec"]["containers"][0]

    for container in (head, worker):
        refs = container["envFrom"]
        assert any(ref.get("configMapRef", {}).get("name") == "avaloka-config" for ref in refs)
        assert any(ref.get("secretRef", {}).get("name") == "avaloka-secrets" for ref in refs)

    timeout = {item["name"]: item["value"] for item in head["env"]}
    assert timeout["RAY_DASHBOARD_SUBPROCESS_MODULE_WAIT_READY_TIMEOUT"] == "180"
    assert head["resources"]["limits"]["memory"] == "6Gi"


def test_e7_07_pins_autoscaling_belongs_to_the_separate_compute_cluster() -> None:
    """maxReplicas 4 + enableInTreeAutoscaling live on the compute RayCluster, which serves no inference."""
    cluster = next(d for d in _load_chart(RAYCLUSTER_CHART) if d.get("kind") == "RayCluster")
    spec = cluster["spec"]
    worker = spec["workerGroupSpecs"][0]

    assert cluster["metadata"]["name"] == "avaloka-raycluster"
    assert spec["enableInTreeAutoscaling"] is True
    assert worker["groupName"] == "cpu-workers"
    assert worker["maxReplicas"] == 4
    assert "serveConfigV2" not in spec


def test_e7_07_pins_serve_deployment_decorator_has_no_autoscaling() -> None:
    """@serve.deployment takes num_replicas from SERVE_REPLICAS (default 1) and declares no autoscaler."""
    src = inspect.getsource(serve_inference)
    decorator = re.search(r"@serve\.deployment\((.*?)\)\s*\n\s*class InferenceService", src, re.S)

    assert decorator is not None
    body = decorator.group(1)
    assert 'os.environ.get("SERVE_REPLICAS", "1")' in body
    assert "autoscaling_config" not in body
