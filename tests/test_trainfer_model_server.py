"""The default inference backend: one process, many models, bounded LRU, 0<->1 scaler.

Base classes (develop-1.6 @ 9cd9e122): everything here errors on the base --
app.serve.model_server and app.infra.model_server_scaler do not exist. The
idle-boundary test's *default-value* half would pass on the base
(inference_autoscale already defaults to 7200) and is therefore not asserted
on its own (see T-38 in DESIGN.md).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _server():
    from app.serve import model_server
    return model_server


# ModelCache (T-18) -------------------------------------------------------------------

def test_cache_evicts_least_recently_used_to_stay_within_the_byte_budget():
    ms = _server()
    cache = ms.ModelCache(budget_bytes=100, max_models=10)
    cache.put("a", ms.CacheEntry(model=object(), bytes=60))
    cache.put("b", ms.CacheEntry(model=object(), bytes=30))
    assert cache.keys() == ["a", "b"]
    cache.get("a")                                   # a is now the most recent
    evicted = cache.put("c", ms.CacheEntry(model=object(), bytes=40))
    assert evicted == ["b"], "the least recently used entry goes first"
    assert cache.keys() == ["a", "c"]
    assert cache.bytes() <= cache.budget_bytes


def test_cache_honours_the_count_cap_and_keeps_at_least_the_newest():
    ms = _server()
    cache = ms.ModelCache(budget_bytes=10_000, max_models=2)
    for k in ("a", "b", "c"):
        cache.put(k, ms.CacheEntry(model=object(), bytes=1))
    assert cache.keys() == ["b", "c"]
    big = ms.ModelCache(budget_bytes=10, max_models=5)
    big.put("huge", ms.CacheEntry(model=object(), bytes=500))
    assert big.keys() == ["huge"], "an oversized single model is kept, not thrashed"


def test_cache_get_updates_recency_and_hits():
    ms = _server()
    cache = ms.ModelCache(budget_bytes=100)
    cache.put("a", ms.CacheEntry(model=object(), bytes=1))
    cache.put("b", ms.CacheEntry(model=object(), bytes=1))
    e = cache.get("a", now=123.0)
    assert e is not None and e.hits == 1 and e.last_used == 123.0
    assert cache.keys() == ["b", "a"]
    assert cache.get("zzz") is None
    assert cache.evict("a") is True and cache.evict("a") is False


def test_cache_rejects_a_nonpositive_budget():
    ms = _server()
    with pytest.raises(ValueError):
        ms.ModelCache(budget_bytes=0)


# resolve_model_uri -------------------------------------------------------------------

def test_registry_version_is_preferred_over_the_run_artifact():
    ms = _server()
    assert ms.resolve_model_uri("r1", registered_name="avaloka.u.m", version=3) == "models:/avaloka.u.m/3"
    assert ms.resolve_model_uri("r1") == "runs:/r1/model_files/model.onnx"
    assert ms.resolve_model_uri("r1", model_uri="s3://x/model.onnx", registered_name="n", version=1) == "s3://x/model.onnx"


# the HTTP surface (T-19) ---------------------------------------------------------------

class _FakeModel:
    loaded = True
    backend = "fake"

    def __init__(self, uri):
        self.model_uri = uri

    def predict(self, features):
        return {"prediction": 1.0, "scores": [[1.0]], "model_uri": self.model_uri, "model_loaded": True, "backend": "fake"}


@pytest.fixture
def served():
    from fastapi.testclient import TestClient
    ms = _server()
    loads = []

    def loader(run_id, uri):
        loads.append((run_id, uri))
        if run_id == "broken":
            raise ms.ModelLoadError("ONNX artifact missing")
        return ms.CacheEntry(model=_FakeModel(uri), bytes=10, model_uri=uri)

    cache = ms.ModelCache(budget_bytes=25, max_models=8)
    client = TestClient(ms.create_app(cache=cache, loader=loader))
    return SimpleNamespace(client=client, loads=loads, cache=cache)


def test_predict_loads_once_then_serves_from_the_cache(served):
    c = served.client
    a = c.post("/models/r1/predict", json={"features": [1, 2, 3]})
    b = c.post("/models/r1/predict", json={"features": [4, 5, 6]})
    assert a.status_code == 200 and b.status_code == 200
    assert a.json()["prediction"] == 1.0 and a.json()["run_id"] == "r1"
    assert served.loads == [("r1", "runs:/r1/model_files/model.onnx")], "loaded exactly once"


def test_load_uses_the_registry_hint_and_reports_the_entry(served):
    r = served.client.post("/models/r2/load", json={"registered_name": "avaloka.u.m", "version": 2})
    assert r.status_code == 200
    assert r.json() == {"loaded": True, "run_id": "r2", "bytes": 10, "flavor": "onnx", "model_uri": "models:/avaloka.u.m/2"}
    assert [m["run_id"] for m in served.client.get("/models").json()["models"]] == ["r2"]


def test_a_model_that_cannot_load_is_422_with_the_reason_not_a_stack(served):
    r = served.client.post("/models/broken/load", json={})
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "ONNX artifact missing"
    assert "broken" not in served.cache


def test_unload_and_eviction_under_budget(served):
    c = served.client
    for k in ("a", "b", "c"):          # 10 bytes each, budget 25 -> the third evicts the first
        c.post(f"/models/{k}/load", json={})
    assert served.cache.keys() == ["b", "c"]
    assert c.post("/models/b/unload").json() == {"unloaded": True, "run_id": "b"}
    assert c.post("/models/b/unload").json()["unloaded"] is False


def test_health_before_any_model(served):
    assert served.client.get("/healthz").json() == {"status": "ok"}
    r = served.client.get("/readyz").json()
    assert r["status"] == "ready" and r["models"] == 0


def test_predict_without_features_is_400(served):
    r = served.client.post("/models/r1/predict", json={})
    assert r.status_code == 400


# the scaler (T-21, T-38) ----------------------------------------------------------------

class _FakeApps:
    def __init__(self, replicas=0, ready=0):
        self.replicas, self.ready, self.patches = replicas, ready, []

    def read_namespaced_deployment_scale(self, name, ns):
        return SimpleNamespace(spec=SimpleNamespace(replicas=self.replicas))

    def read_namespaced_deployment(self, name, ns):
        return SimpleNamespace(status=SimpleNamespace(ready_replicas=self.ready))

    def patch_namespaced_deployment_scale(self, name, ns, body):
        self.patches.append((name, ns, body["spec"]["replicas"]))
        self.replicas = body["spec"]["replicas"]


def test_wake_scales_zero_to_one_once_and_leaves_a_running_deployment_alone():
    from app.infra.model_server_scaler import ModelServerScaler
    api = _FakeApps(replicas=0)
    s = ModelServerScaler("avaloka-model-server", "ns", api=api)
    assert s.wake() is True and s.wake() is True
    assert api.patches == [("avaloka-model-server", "ns", 1)], "one patch, then a no-op at >= 1"
    api2 = _FakeApps(replicas=3)
    assert ModelServerScaler("d", "ns", api=api2).wake() is True and api2.patches == []


def test_idle_boundary_7199_stays_7200_sleeps():
    from app.infra.model_server_scaler import ModelServerScaler
    api = _FakeApps(replicas=1)
    s = ModelServerScaler("d", "ns", api=api, idle_timeout_s=7200)
    now = 1_000_000.0
    assert s.sleep_if_idle(now - 7199, now=now) is False and api.patches == []
    assert s.sleep_if_idle(now - 7200, now=now) is True and api.patches == [("d", "ns", 0)]


def test_sleep_never_patches_when_already_zero_or_unreadable():
    from app.infra.model_server_scaler import ModelServerScaler
    api = _FakeApps(replicas=0)
    s = ModelServerScaler("d", "ns", api=api)
    assert s.sleep_if_idle(0.0, now=1e9) is False and api.patches == []

    class Broken:
        def read_namespaced_deployment_scale(self, *a):
            raise RuntimeError("403 forbidden")

    s2 = ModelServerScaler("d", "ns", api=Broken())
    assert s2.replicas() is None and s2.sleep_if_idle(0.0, now=1e9) is False and s2.wake() is False


def test_unknown_last_request_counts_as_idle():
    from app.infra.model_server_scaler import ModelServerScaler
    api = _FakeApps(replicas=1)
    assert ModelServerScaler("d", "ns", api=api).sleep_if_idle(None, now=1.0) is True
