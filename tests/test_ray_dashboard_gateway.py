"""Gateway rules for Ray's own dashboard at /ray/ (app/api/ray_jobs.py).

Runs against the real router with FastAPI's TestClient; the plan check and the
Ray service are faked, so no Supabase, Redis or Ray is needed.
"""
from __future__ import annotations

import time
import types

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api import ray_jobs

SECRET = "test-secret"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("RAY_SESSION_SECRET", SECRET)
    monkeypatch.setenv("COOKIE_SECURE", "")


def _client(plan_features: set[str], user="user-a", hooks: dict | None = None, monkeypatch=None):
    async def require_premium(user_id, feature):
        if feature not in plan_features:
            raise HTTPException(403, {"reason": "plan_upgrade_required", "feature": feature})

    if monkeypatch is not None:
        fake = types.SimpleNamespace(RayJobsError=Exception, **(hooks or {}))
        monkeypatch.setattr(ray_jobs, "svc", fake)

    app = FastAPI()
    app.include_router(ray_jobs.build_router(lambda req: req.headers.get("X-User") or None, require_premium))
    c = TestClient(app)
    c.headers.update({"X-User": user} if user else {})
    return c


# ------------------------------------------------------------ plan gate --

def test_free_user_cannot_open_the_dashboard(monkeypatch):
    c = _client(set(), monkeypatch=monkeypatch)
    r = c.post("/api/ray/session")
    assert r.status_code == 403


def test_not_signed_in_gets_401(monkeypatch):
    c = _client({"data_transfer"}, user=None, monkeypatch=monkeypatch)
    assert c.post("/api/ray/session").status_code == 401


@pytest.mark.parametrize("feature", ["data_transfer", "model_training", "scheduled_large_jobs", "ray_dashboard"])
def test_any_ray_feature_on_the_plan_is_enough(feature, monkeypatch):
    """Not only data transfer: large scheduled jobs and training also run on Ray."""
    c = _client({feature}, monkeypatch=monkeypatch)
    r = c.post("/api/ray/session")
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "running"


# ---------------------------------------------------- session and authz --

def _session_cookie(c) -> str:
    r = c.post("/api/ray/session")
    assert r.status_code == 200
    return r.cookies.get(ray_jobs.COOKIE_NAME)


def test_session_cookie_is_scoped_and_http_only(monkeypatch):
    c = _client({"data_transfer"}, monkeypatch=monkeypatch)
    r = c.post("/api/ray/session")
    header = r.headers["set-cookie"].lower()
    assert "path=/ray/" in header and "httponly" in header and "samesite=lax" in header


def test_authz_returns_the_users_cluster(monkeypatch):
    hooks = {"user_ray_cluster": lambda uid: {"state": "running", "upstream": f"ray-{uid}-head-svc.ray:8265"}}
    c = _client({"data_transfer"}, user="user-a", hooks=hooks, monkeypatch=monkeypatch)
    cookie = _session_cookie(c)
    r = c.get("/api/ray/authz", cookies={ray_jobs.COOKIE_NAME: cookie},
              headers={"X-Original-Method": "GET", "X-Original-URI": "/ray/api/jobs/"})
    assert r.status_code == 200
    assert r.headers["X-Ray-Upstream"] == "ray-user-a-head-svc.ray:8265"


def test_users_never_share_a_cluster(monkeypatch):
    hooks = {"user_ray_cluster": lambda uid: {"state": "running", "upstream": f"ray-{uid}-head-svc.ray:8265"}}
    a = _client({"data_transfer"}, user="user-a", hooks=hooks, monkeypatch=monkeypatch)
    b = _client({"data_transfer"}, user="user-b", hooks=hooks, monkeypatch=monkeypatch)
    ua = a.get("/api/ray/authz", cookies={ray_jobs.COOKIE_NAME: _session_cookie(a)}).headers["X-Ray-Upstream"]
    ub = b.get("/api/ray/authz", cookies={ray_jobs.COOKIE_NAME: _session_cookie(b)}).headers["X-Ray-Upstream"]
    assert ua != ub


def test_authz_without_cookie_is_401(monkeypatch):
    c = _client({"data_transfer"}, monkeypatch=monkeypatch)
    assert c.get("/api/ray/authz").status_code == 401


def test_tampered_or_expired_cookie_is_401(monkeypatch):
    c = _client({"data_transfer"}, monkeypatch=monkeypatch)
    cookie = _session_cookie(c)
    body, sig = cookie.split(".", 1)
    forged = ray_jobs.sign_session({"sub": "x", "upstream": "evil.example:8265", "exp": time.time() + 60}, b"wrong")
    expired = ray_jobs.sign_session({"sub": "x", "upstream": "a.b:8265", "exp": time.time() - 1}, SECRET.encode())
    for bad in (body + ".AAAA", forged, expired, "garbage"):
        assert c.get("/api/ray/authz", cookies={ray_jobs.COOKIE_NAME: bad}).status_code == 401


# --------------------------------------------------- what /ray/ allows --

@pytest.mark.parametrize("method,uri,allowed", [
    ("GET",    "/ray/",                                   True),
    ("GET",    "/ray/api/jobs/",                          True),
    ("HEAD",   "/ray/api/version",                        True),
    ("POST",   "/ray/api/jobs/raysubmit_abc123/stop",     True),   # stop own job
    ("POST",   "/ray/api/jobs/",                          False),  # submit a job
    ("DELETE", "/ray/api/jobs/raysubmit_abc123",          False),  # delete a job
    ("PUT",    "/ray/api/serve/applications/",            False),  # Serve change
    ("POST",   "/ray/api/jobs/../../x/stop",              False),
    ("POST",   "/ray/api/jobs/../stop",                   False),
    ("PATCH",  "/ray/",                                   False),
])
def test_dashboard_actions(method, uri, allowed, monkeypatch):
    c = _client({"data_transfer"}, monkeypatch=monkeypatch)
    cookie = _session_cookie(c)
    r = c.get("/api/ray/authz", cookies={ray_jobs.COOKIE_NAME: cookie},
              headers={"X-Original-Method": method, "X-Original-URI": uri})
    assert (r.status_code == 200) is allowed


# ------------------------------------------------- stopped cluster / idle --

def test_stopped_cluster_is_started_not_replaced(monkeypatch):
    started = []
    hooks = {
        "user_ray_cluster": lambda uid: {"state": "stopped", "upstream": ""},
        "start_user_ray_cluster": lambda uid: started.append(uid),
    }
    c = _client({"data_transfer"}, user="user-a", hooks=hooks, monkeypatch=monkeypatch)
    r = c.post("/api/ray/session")
    body = r.json()
    assert (body["state"], body["url"]) == ("starting", None)
    assert body["message"].startswith("Starting your Ray cluster")
    assert body["idle_minutes"] == 15
    assert started == ["user-a"]
    assert ray_jobs.COOKIE_NAME not in r.cookies  # never sent to another cluster meanwhile


def test_heartbeat_keeps_the_users_cluster_alive(monkeypatch):
    touched = []
    c = _client({"data_transfer"}, user="user-a",
                hooks={"touch_user_ray_cluster": lambda uid: touched.append(uid)}, monkeypatch=monkeypatch)
    assert c.post("/api/ray/heartbeat").status_code == 200
    assert touched == ["user-a"]