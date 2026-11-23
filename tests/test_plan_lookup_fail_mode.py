"""What a failed plan lookup does, per declared deployment.

The owner's ruling, through the one deployment signal
(app/core/editions.declared_deployment, from AVALOKA_DEPLOYMENT):

* hosted (avaloka_hosted): a successful "free" lookup is refused as an upgrade
  (403 plan_upgrade_required); a premium plan is allowed; a FAILED lookup is
  refused as unavailable: 503 "plan check unavailable", never
  plan_upgrade_required, and chat says the check could not run.
* self_hosted and undeclared (unset): every case is allowed, as before.
* undeclared logs one warning per process.

Each case runs through check_feature, the real /api/models/{run_id}/inference
route (a signed token, the real resolver with only the Supabase query
replaced), GET /api/limits (plan_check_unavailable and feature_states tell
"could not be checked" apart from "not on your plan") and the planner's
training and data-transfer gates.
"""
import logging

import jwt
import pytest
from fastapi.testclient import TestClient

from app.agents import planner
from app.core import editions, plan_gate, plan_limits

DEPLOYMENTS = ["hosted", "self_hosted", "undeclared"]
LOOKUPS = ["free", "premium", "failed"]
_ENV = {"hosted": "avaloka_hosted", "self_hosted": "self_hosted", "undeclared": None}
_SECRET = "plan-fail-mode-test-secret-0123456789abcdef"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("AVALOKA_DEPLOYMENT", "AVALOKA_DISABLE_PLAN_GATES", "AVALOKA_EDITION",
                 "AVALOKA_PLAN_LIMITS", "AVALOKA_DEFAULT_PLAN"):
        monkeypatch.delenv(name, raising=False)
    plan_limits.clear_plan_cache()
    yield
    plan_limits.clear_plan_cache()


def _arrange(monkeypatch, deployment, lookup):
    if _ENV[deployment]:
        monkeypatch.setenv("AVALOKA_DEPLOYMENT", _ENV[deployment])

    def fake_lookup(user_id):
        if lookup == "failed":
            raise RuntimeError("supabase down")
        return "free" if lookup == "free" else "enterprise"

    monkeypatch.setattr(plan_limits, "_lookup_plan", fake_lookup)


def _expected(deployment, lookup):
    """'allowed', 'upgrade' or 'unavailable'."""
    if deployment != "hosted" or lookup == "premium":
        return "allowed"
    return "upgrade" if lookup == "free" else "unavailable"


MATRIX = [(d, l) for d in DEPLOYMENTS for l in LOOKUPS]


# ------------------------------------------------------------ check_feature

@pytest.mark.parametrize("deployment, lookup", MATRIX)
def test_check_feature(monkeypatch, deployment, lookup):
    _arrange(monkeypatch, deployment, lookup)
    d = plan_gate.check_feature("model_inference", user_id="user-1")
    expected = _expected(deployment, lookup)
    assert d.allowed is (expected == "allowed"), d.reason
    assert getattr(d, "unavailable", False) is (expected == "unavailable"), d.reason


# --------------------------------------------------------------- HTTP route

@pytest.fixture
def client(monkeypatch):
    from app.api import server
    monkeypatch.setattr(server, "JWT_SECRET", _SECRET)
    # Past the gate the route reaches Redis, storage and MLflow; whatever they
    # do there (a 500 in a full run, where earlier tests leave them configured)
    # is answered, not raised, so only the gate's answer is asserted.
    return TestClient(server.app, raise_server_exceptions=False)


def _post_inference(client):
    token = jwt.encode({"sub": "user-1"}, _SECRET, algorithm="HS256")
    return client.post("/api/models/run-1/inference", json={"x": 1},
                       headers={"Authorization": f"Bearer {token}"})


def _plan_reason(resp):
    try:
        body = resp.json()
    except ValueError:              # e.g. a plain-text 500 past the gate
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail.get("reason") if isinstance(detail, dict) else None


@pytest.mark.parametrize("deployment, lookup", MATRIX)
def test_inference_route(monkeypatch, client, deployment, lookup):
    _arrange(monkeypatch, deployment, lookup)
    resp = _post_inference(client)
    expected = _expected(deployment, lookup)
    if expected == "unavailable":
        assert resp.status_code == 503, resp.text
        assert _plan_reason(resp) == "plan_check_unavailable", resp.text
        assert "plan_upgrade_required" not in resp.text
        assert "plan check unavailable" in resp.json()["detail"]["message"].lower()
    elif expected == "upgrade":
        assert resp.status_code == 403, resp.text
        assert _plan_reason(resp) == "plan_upgrade_required", resp.text
    else:
        # Past the plan gate; what the route answers next (no Redis, no model)
        # is not the gate's business.
        assert _plan_reason(resp) not in ("plan_upgrade_required", "plan_check_unavailable"), resp.text


# ------------------------------------------------------------ GET /api/limits

def _get_limits(client):
    token = jwt.encode({"sub": "user-1"}, _SECRET, algorithm="HS256")
    return client.get("/api/limits", headers={"Authorization": f"Bearer {token}"})


@pytest.mark.parametrize("deployment, lookup", MATRIX)
def test_limits_report_unavailable_apart_from_not_on_plan(monkeypatch, client, deployment, lookup):
    _arrange(monkeypatch, deployment, lookup)
    resp = _get_limits(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    states = body.get("feature_states") or {}
    assert set(states) == set(plan_gate.FEATURE_CAPABILITY), body
    expected = _expected(deployment, lookup)
    assert body.get("plan_check_unavailable") is (expected == "unavailable"), body
    if expected == "unavailable":
        # Not an upgrade case: nothing says "not on your plan" or asks to upgrade.
        assert set(states.values()) == {"unavailable"}, states
        assert "plan_upgrade_required" not in resp.text and "upgrade" not in resp.text.lower()
    elif expected == "upgrade":
        assert "not_on_plan" in states.values() and "unavailable" not in states.values(), states
    else:
        assert set(states.values()) == {"allowed"}, states
    assert body["features"] == {f: s == "allowed" for f, s in states.items()}


# ------------------------------------------------------------------ planner

def _training_out():
    return {"enable_training": True, "skip_to_training": True, "messages": []}


@pytest.mark.parametrize("deployment, lookup", MATRIX)
def test_planner_training_gate(monkeypatch, deployment, lookup):
    _arrange(monkeypatch, deployment, lookup)
    out = planner._apply_training_plan_gate({"user_id": "user-1", "messages": []}, _training_out())
    expected = _expected(deployment, lookup)
    if expected == "allowed":
        assert out["enable_training"] is True
        assert not out.get("plan_upgrade_required")
        return
    assert out["enable_training"] is False
    message = out["messages"][-1].content
    if expected == "upgrade":
        assert out["plan_upgrade_required"] == "model_training"
        assert "Upgrade your plan" in message
    else:
        assert not out.get("plan_upgrade_required")
        assert "Plan check unavailable" in message
        assert "upgrade" not in message.lower()


@pytest.mark.parametrize("deployment, lookup", MATRIX)
def test_planner_data_transfer_gate(monkeypatch, deployment, lookup):
    _arrange(monkeypatch, deployment, lookup)
    monkeypatch.setattr(planner, "_handle_initiate_transfer", lambda state, params: "PROCEEDED")
    state = {"user_id": "user-1", "messages": []}
    reply = planner._resolve_pending_dta(state, {"collected": {}, "awaiting": "filename"}, "out.csv")
    expected = _expected(deployment, lookup)
    if expected == "allowed":
        assert reply == "PROCEEDED"
        assert not state.get("plan_upgrade_required")
    elif expected == "upgrade":
        assert state.get("plan_upgrade_required") == "data_transfer"
        assert "Upgrade your plan" in reply
    else:
        assert reply != "PROCEEDED"
        assert not state.get("plan_upgrade_required")
        assert "Plan check unavailable" in reply
        assert "upgrade" not in reply.lower()


# -------------------------------------------------------- one-time warning

def _undeclared_warnings(caplog):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and "undeclared" in r.getMessage()
            and "AVALOKA_DEPLOYMENT" in r.getMessage()]


def test_undeclared_deployment_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(editions, "_undeclared_warned", False, raising=False)
    caplog.set_level(logging.WARNING)
    for _ in range(3):
        plan_gate.check_feature("model_inference", user_id="user-1", plan="free")
        editions.deployment_from_env()
    assert len(_undeclared_warnings(caplog)) == 1


@pytest.mark.parametrize("value", ["self_hosted", "avaloka_hosted"])
def test_declared_deployment_does_not_warn(monkeypatch, caplog, value):
    monkeypatch.setattr(editions, "_undeclared_warned", False, raising=False)
    monkeypatch.setenv("AVALOKA_DEPLOYMENT", value)
    caplog.set_level(logging.WARNING)
    plan_gate.check_feature("model_inference", user_id="user-1", plan="enterprise")
    assert _undeclared_warnings(caplog) == []


@pytest.mark.parametrize("value, declared", [
    ("avaloka_hosted", "hosted"), ("Avaloka-Hosted", "hosted"), ("self_hosted", "self_hosted"),
    ("", "undeclared"), ("nonsense", "undeclared"),
])
def test_declared_deployment_values(monkeypatch, value, declared):
    monkeypatch.setattr(editions, "_undeclared_warned", True, raising=False)
    monkeypatch.setenv("AVALOKA_DEPLOYMENT", value)
    assert editions.declared_deployment() == declared


# ------------------------------------------------------------------ billing

@pytest.mark.parametrize("deployment, edition, disabled", [
    ("hosted", "", False), ("hosted", "oss", False),
    ("self_hosted", "", False), ("self_hosted", "oss", True),
    ("undeclared", "", False), ("undeclared", "community", True),
])
def test_billing_reads_the_signal(monkeypatch, deployment, edition, disabled):
    if _ENV[deployment]:
        monkeypatch.setenv("AVALOKA_DEPLOYMENT", _ENV[deployment])
    if edition:
        monkeypatch.setenv("AVALOKA_EDITION", edition)
    assert plan_limits._billing_disabled() is disabled


@pytest.mark.parametrize("deployment", DEPLOYMENTS)
def test_plan_limits_off_disables_billing_everywhere(monkeypatch, deployment):
    if _ENV[deployment]:
        monkeypatch.setenv("AVALOKA_DEPLOYMENT", _ENV[deployment])
    monkeypatch.setenv("AVALOKA_PLAN_LIMITS", "off")
    assert plan_limits._billing_disabled() is True
