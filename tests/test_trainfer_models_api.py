"""The /api/models surface the inference panel calls, on the real router.

Base classes (develop-1.6 @ 9cd9e122): tests marked A fail by assertion on
the old server.py handlers (200 "running", Redis 503, no `endpoint` key,
/predict 404); the rest error because app.api.models_routes does not exist.
The base handlers are reached through the same small app by mounting the
server module's routes, so the comparison is like for like.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

try:
    from app.core import ml_store
except ImportError:  # pragma: no cover - base tree only
    ml_store = None


RUN = {"mlflow_run_id": "run-1", "user_id": "u1", "model_name": "m", "model_type": "classification",
       "metrics": {"final_accuracy": 0.9}, "inference_service_details": None}


class FakeMLflow:
    def __init__(self, runs=None):
        self.runs = runs if runs is not None else {"run-1": dict(RUN)}
        self.deleted = []

    def list_models(self):
        return list(self.runs.values())

    def get_model_details(self, run_id):
        return self.runs.get(run_id)

    def delete_model(self, run_id):
        self.deleted.append(run_id)


@pytest.fixture
def store():
    if ml_store is None:
        yield None
        return
    s = ml_store.InMemoryMLStore()
    ml_store.set_ml_store(s)
    yield s
    ml_store.set_ml_store(None)


async def _premium(user_id, feature):
    return None


def _user(req):
    return req.headers.get("X-User") or None


@pytest.fixture
def client(store):
    """The models router on a bare app; on the base, the server's routes instead."""
    app = FastAPI()
    try:
        from app.api.models_routes import build_router
        app.include_router(build_router(_user, _premium))
    except ImportError:
        import app.api.server as server
        for route in server.app.routes:
            if getattr(route, "path", "").startswith("/api/models"):
                app.router.routes.append(route)
        server._resolve_user_id = _user
    return TestClient(app)


def _patch_mlflow(fake):
    return patch("app.agents.mta_v2.mlflow_manager.MLflowManager", return_value=fake)


H = {"X-User": "u1"}


# T-07 (A) -----------------------------------------------------------------------

def test_list_models_works_without_redis(client):
    with _patch_mlflow(FakeMLflow()), patch("app.services.session_service.cache", None, create=True):
        r = client.get("/api/models", headers=H)
    assert r.status_code == 200, r.text
    assert [m["mlflow_run_id"] for m in r.json()["models"]] == ["run-1"]


def test_list_models_is_owner_scoped(client):
    fake = FakeMLflow({"run-1": dict(RUN), "run-2": {**RUN, "mlflow_run_id": "run-2", "user_id": "someone-else"}})
    with _patch_mlflow(fake):
        r = client.get("/api/models", headers=H)
    assert [m["mlflow_run_id"] for m in r.json()["models"]] == ["run-1"]


# T-08 (A) -----------------------------------------------------------------------

def test_detail_carries_endpoint_state_from_postgres(client, store):
    assert store is not None, "app.core.ml_store is missing"
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="u1", status="deploying"))
    store.upsert_model(ml_store.MLModel(mlflow_run_id="run-1", user_id="u1", registered_name="avaloka.u1.m",
                                        version=3, task_type="classification"))
    with _patch_mlflow(FakeMLflow()):
        r = client.get("/api/models/run-1", headers=H)
    body = r.json()
    assert body["endpoint"]["status"] == "deploying"
    assert body["endpoint"]["servable"] is False
    assert body["inference_service_details"] is None, "only a ready endpoint counts as running"
    assert (body["registry"]["registered_name"], body["registry"]["version"]) == ("avaloka.u1.m", 3)


def test_ready_endpoint_sets_the_legacy_running_flag(client, store):
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="u1", status="ready"))
    with _patch_mlflow(FakeMLflow()):
        r = client.get("/api/models/run-1", headers=H)
    body = r.json()
    assert body["endpoint"]["url"] == "/api/models/run-1/predict"
    assert body["inference_service_details"] is not None


def test_detail_of_another_users_run_is_404_not_403(client):
    """TI-24: a 403 would confirm the model exists; another user's run is indistinguishable from none."""
    with _patch_mlflow(FakeMLflow({"run-1": {**RUN, "user_id": "other"}})):
        r = client.get("/api/models/run-1", headers=H)
    assert r.status_code == 404, r.text
    assert r.json()["detail"] == "Model not found."


# T-06 (A for the status code; E for /endpoint) -----------------------------------

def test_configure_returns_202_and_a_job_without_loading_in_the_request(client, store):
    manager = MagicMock()
    manager.configure_inference_service.side_effect = AssertionError("configure ran inside the request")
    deploy = MagicMock(); deploy.apply_async.return_value = MagicMock(id="c-1")
    with _patch_mlflow(FakeMLflow()), \
         patch("app.agents.mta_v2.inference_service_manager.InferenceServiceManager", return_value=manager), \
         _maybe_patch("app.infra.training_tasks._worker_available", return_value=True), \
         _maybe_patch("app.infra.training_tasks.tasks", return_value=(deploy, MagicMock())):
        r = client.post("/api/models/run-1/configure-inference-service", headers=H)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "deploying" and body["job_id"]
    assert body["endpoint"]["status"] == "pending"

    r2 = client.get("/api/models/run-1/endpoint", headers=H)
    assert r2.status_code == 200
    assert r2.json()["status"] == "pending"
    r3 = client.get(f"/api/ml-jobs/{body['job_id']}", headers=H)
    assert r3.json()["kind"] == "deploy_endpoint" and r3.json()["status"] == "queued"


def test_configure_when_ready_is_200_and_starts_no_job(client, store):
    """TI-09: a configure on an endpoint that is already serving is a no-op."""
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="u1", status="ready"))
    with _patch_mlflow(FakeMLflow()), _maybe_patch("app.infra.training_tasks._worker_available", return_value=True):
        r = client.post("/api/models/run-1/configure-inference-service", headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ready" and r.json()["job_id"] is None
    assert store.list_jobs("u1", kind="deploy_endpoint") == []


def test_configure_while_deploying_is_409(client, store):
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="u1", status="deploying"))
    with _patch_mlflow(FakeMLflow()):
        r = client.post("/api/models/run-1/configure-inference-service", headers=H)
    assert r.status_code == 409


def test_configure_without_a_worker_is_503_and_the_job_is_failed(client, store):
    with _patch_mlflow(FakeMLflow()), _maybe_patch("app.infra.training_tasks._worker_available", return_value=False):
        r = client.post("/api/models/run-1/configure-inference-service", headers=H)
    assert r.status_code == 503
    jobs = store.list_jobs("u1", kind="deploy_endpoint")
    assert jobs and jobs[0].status == "failed"


def test_stop_returns_202_and_flips_the_badge_at_once(client, store):
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="u1", status="ready"))
    stop = MagicMock(); stop.apply_async.return_value = MagicMock(id="c-2")
    with _patch_mlflow(FakeMLflow()), \
         _maybe_patch("app.infra.training_tasks._worker_available", return_value=True), \
         _maybe_patch("app.infra.training_tasks.tasks", return_value=(MagicMock(), stop)):
        r = client.post("/api/models/run-1/stop-inference-service", headers=H)
    assert r.status_code == 202, r.text
    assert r.json()["endpoint"]["status"] == "stopped"


# predictions: T-20 (409 / 503) and T-40 (/predict + alias) ------------------------

def test_predict_requires_a_ready_endpoint(client, store):
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="u1", status="deploying"))
    with _patch_mlflow(FakeMLflow()):
        r = client.post("/api/models/run-1/predict", headers=H, json={"f": 1})
    assert r.status_code == 409


def test_predict_and_inference_alias_share_one_handler_and_touch_the_endpoint(client, store):
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="u1", status="ready"))
    manager = MagicMock(); manager.inference.return_value = {"prediction": 1}
    with _patch_mlflow(FakeMLflow()), \
         patch("app.agents.mta_v2.inference_service_manager.InferenceServiceManager", return_value=manager):
        a = client.post("/api/models/run-1/predict", headers=H, json={"f": 1})
        b = client.post("/api/models/run-1/inference", headers=H, json={"f": 1})
    assert a.status_code == 200 and b.status_code == 200, (a.text, b.text)
    assert a.json() == b.json() == {"result": {"prediction": 1}}
    assert manager.inference.call_count == 2
    assert store.get_endpoint("run-1").last_request_at is not None


def test_predict_on_another_users_endpoint_is_404_for_both_routes(client, store):
    """T-40 / TI-24: both routes, one handler, and 404 rather than 403."""
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="other", status="ready"))
    with _patch_mlflow(FakeMLflow({"run-1": {**RUN, "user_id": "other"}})):
        a = client.post("/api/models/run-1/predict", headers=H, json={"f": 1})
        b = client.post("/api/models/run-1/inference", headers=H, json={"f": 1})
    assert (a.status_code, b.status_code) == (404, 404), (a.text, b.text)


def test_every_run_route_answers_404_to_another_user(client, store):
    """TI-24 across the surface: configure, stop, delete, endpoint, detail."""
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="other", status="ready"))
    with _patch_mlflow(FakeMLflow({"run-1": {**RUN, "user_id": "other"}})):
        codes = {
            "detail": client.get("/api/models/run-1", headers=H).status_code,
            "endpoint": client.get("/api/models/run-1/endpoint", headers=H).status_code,
            "configure": client.post("/api/models/run-1/configure-inference-service", headers=H).status_code,
            "stop": client.post("/api/models/run-1/stop-inference-service", headers=H).status_code,
            "delete": client.delete("/api/models/run-1", headers=H).status_code,
        }
    assert codes == {k: 404 for k in codes}, codes


def test_predict_answers_503_with_retry_after_while_the_server_wakes(client, store):
    from app.agents.mta_v2.inference_service_manager import ServingAsleep
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="u1", status="ready"))
    manager = MagicMock(); manager.inference.side_effect = ServingAsleep("scaled to zero")
    with _patch_mlflow(FakeMLflow()), \
         patch("app.agents.mta_v2.inference_service_manager.InferenceServiceManager", return_value=manager):
        r = client.post("/api/models/run-1/predict", headers=H, json={"f": 1})
    assert r.status_code == 503
    assert r.headers.get("Retry-After") == "15"


# T-24 delete -> registry ------------------------------------------------------------

def test_delete_removes_registry_version_and_rows(client, store):
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="u1", status="stopped"))
    store.upsert_model(ml_store.MLModel(mlflow_run_id="run-1", user_id="u1", registered_name="avaloka.u1.m",
                                        version=1, task_type="classification"))
    fake = FakeMLflow()
    with _patch_mlflow(fake), patch("app.core.ml_registry.delete_run_version") as del_version:
        r = client.delete("/api/models/run-1", headers=H)
    assert r.status_code == 200
    del_version.assert_called_once_with("run-1")
    assert fake.deleted == ["run-1"]
    assert store.get_model("run-1") is None and store.get_endpoint("run-1") is None


# helpers ---------------------------------------------------------------------------

from contextlib import nullcontext


def _maybe_patch(target, *args, **kwargs):
    module = target.rsplit(".", 1)[0]
    try:
        __import__(module)
    except ImportError:
        return nullcontext(MagicMock())
    return patch(target, *args, **kwargs)
