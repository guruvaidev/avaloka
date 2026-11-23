"""The premium gate (server._require_premium_plan) on model inference, by
deployment and plan-lookup outcome.

A plan lookup has three outcomes and the gate must keep them apart: a paid plan
passes, a lookup that succeeds and finds Free is refused with the upgrade
payload, and a lookup that *failed* (error or timeout) is not a Free plan. On
1.7 the decision is app/core/plan_gate.check_feature over
plan_limits.try_resolve_plan, which reports a failed lookup as "unknown"
instead of Free; the deployment comes from editions.deployment_state() and
nothing else:

                free           premium   failed
  hosted        403 upgrade    pass      503 plan_check_unavailable, Retry-After
  self_hosted   pass           pass      pass        VACUOUS: no lookup runs
  undeclared    pass           pass      pass        VACUOUS: no lookup runs

Self-hosted and undeclared deployments have no cost gates (plan_gate), so the
lookup -- and so its failure -- never happens there; the lookup-count assertion
shows that vacuity rather than hiding it. Only the subscription lookup itself
is stubbed (plan_limits._lookup_plan); the real resolver, cache and billing
switch run.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

import app.api.server as server
import app.core.plan_limits as pl

_DEPLOYMENT_ENV = {"hosted": "avaloka_hosted", "self_hosted": "self_hosted", "undeclared": None}

MATRIX = [
    ("hosted", "free", 403), ("hosted", "premium", None), ("hosted", "failed", 503),
    ("self_hosted", "free", None), ("self_hosted", "premium", None), ("self_hosted", "failed", None),
    ("undeclared", "free", None), ("undeclared", "premium", None), ("undeclared", "failed", None),
]


def _ids(cell):
    deployment, lookup, _ = cell
    return f"{deployment}-{lookup}" + ("-vacuous" if deployment != "hosted" else "")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("AVALOKA_EDITION", "AVALOKA_PLAN_LIMITS", "AVALOKA_DEPLOYMENT",
                "AVALOKA_DISABLE_PLAN_GATES", "AVALOKA_DEFAULT_PLAN"):
        monkeypatch.delenv(var, raising=False)
    pl.clear_plan_cache()
    yield
    pl.clear_plan_cache()


def _gate(user_id: str = "auth-1"):
    try:
        asyncio.run(server._require_premium_plan(user_id, "model_inference"))
    except HTTPException as exc:
        return exc
    return None


def _stub_lookup(monkeypatch, lookup: str):
    calls = []

    def lookup_plan(user_id):
        calls.append(user_id)
        if lookup == "failed":
            raise TimeoutError("supabase timed out")
        return "free" if lookup == "free" else "professional"
    monkeypatch.setattr(pl, "_lookup_plan", lookup_plan)
    return calls


@pytest.mark.parametrize("cell", MATRIX, ids=[_ids(c) for c in MATRIX])
def test_the_premium_gate_by_deployment_and_lookup_outcome(cell, monkeypatch):
    deployment, lookup, expected = cell
    if _DEPLOYMENT_ENV[deployment]:
        monkeypatch.setenv("AVALOKA_DEPLOYMENT", _DEPLOYMENT_ENV[deployment])
    calls = _stub_lookup(monkeypatch, lookup)

    exc = _gate()

    if deployment == "hosted":
        assert calls == ["auth-1"]
    else:
        assert calls == [], "a deployment without cost gates has no plan to look up"
    if expected is None:
        assert exc is None, (exc.status_code, exc.detail)
        return
    assert exc is not None, f"{deployment}/{lookup}: the request was allowed"
    assert exc.status_code == expected, (exc.status_code, exc.detail)
    if expected == 403:
        assert exc.detail["reason"] == "plan_upgrade_required"
    if expected == 503:
        # The plan is unknown: neither "upgrade" (they may be paying) nor a free
        # pass is true. The failure is cached for _FAILURE_TTL_S, hence Retry-After.
        assert "plan_upgrade_required" not in str(exc.detail)
        assert exc.detail["reason"] == "plan_check_unavailable"
        assert "unavailable" in exc.detail["message"].lower()
        assert (exc.headers or {}).get("Retry-After") == str(int(pl._FAILURE_TTL_S))


def test_a_hosted_failed_lookup_is_a_503_response_on_the_predict_route(monkeypatch):
    """The hosted-failed cell through the real models router and the server's
    gate: the HTTP response, not the exception."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.models_routes import build_router

    monkeypatch.setenv("AVALOKA_DEPLOYMENT", "avaloka_hosted")
    _stub_lookup(monkeypatch, "failed")

    app = FastAPI()
    app.include_router(build_router(lambda req: "auth-1", server._require_premium_plan))
    r = TestClient(app).post("/api/models/run-1/predict", json={"f": 1})
    assert r.status_code == 503, r.text
    assert "plan_upgrade_required" not in r.text
    assert r.json()["detail"]["reason"] == "plan_check_unavailable"
    assert r.headers.get("retry-after") == str(int(pl._FAILURE_TTL_S))
