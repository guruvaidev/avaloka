"""The swarm console is Professional and Enterprise; the swarm is everyone's.

The console is the live /swarm page and the swarm/trace routes behind it.
These tests enumerate the console routes from the app itself — anything whose
template matches ``app.api.swarm_gate.CONSOLE_PATH_RE`` — so a new trace route
that forgets ``Depends(require_swarm_console)`` fails here rather than shipping
open. Driven over HTTP through the real server harness.
"""

from __future__ import annotations

import pytest

from app.api import swarm_gate
from app.core.editions import EDITIONS, GateReason, swarm_console_gate

try:  # tests/ as a package, or tests/ on sys.path
    from tests.server_harness import client, make_auth_headers  # noqa: F401
except ImportError:  # pragma: no cover
    from server_harness import client, make_auth_headers  # noqa: F401

FREE = make_auth_headers("free-user")
PRO = make_auth_headers("pro-user")
ENT = make_auth_headers("ent-user")
USERS = ("free-user", "pro-user", "ent-user")


@pytest.fixture(autouse=True)
def _no_mission_outlives_its_test(client, tmp_path, monkeypatch):
    """A paid caller's POST starts a real mission on a daemon thread. Join it
    and let its persist run inside a request before the harness tears down,
    or the thread races the fake cache's dicts in the next test's setup."""
    from app.trace import swarm_store

    monkeypatch.setenv("AVALOKA_SWARM_OUTPUT_DIR", str(tmp_path / "swarm-out"))
    swarm_store.registry.clear()
    yield
    for user in USERS:
        for entry in swarm_store.registry.for_user(user):
            if entry.thread is not None:
                entry.thread.join(timeout=180)
    client.get("/version")
    swarm_store.registry.clear()

#: Every console route must produce this when a Free caller hits it.
GATED = 403


def _as_plan(monkeypatch, plan: str) -> None:
    import app.api.server as server

    monkeypatch.delenv("AVALOKA_EDITION", raising=False)
    monkeypatch.setattr(server, "resolve_plan", lambda uid, _p=plan: _p)


def console_routes():
    """(method, concrete path) for every route the regex names, from the app."""
    from app.api.server import app

    found = []
    for route in app.routes:
        path = getattr(route, "path", None)
        if not path or not swarm_gate.is_console_path(path):
            continue
        concrete = path.replace("{thread_id}", "t-1").replace("{mission_id}", "m-1").replace("{clone_id}", "ava-00")
        for method in sorted(getattr(route, "methods", None) or ()):
            if method in ("HEAD", "OPTIONS"):
                continue
            found.append((method, concrete, path))
    return found


def _hit(client, method, path, headers):
    body = {"source": {"sample": "churn"}, "goal": "g"} if method == "POST" else None
    return client.request(method, path, json=body, headers=headers)


# ------------------------------------------------------------------ the gate
def test_the_route_family_is_the_console_and_nothing_else():
    paths = {p for _, _, p in console_routes()}
    for expected in ("/api/swarm/missions", "/api/swarm/missions/{mission_id}/records",
                     "/api/swarm/missions/{mission_id}/telemetry", "/api/agent-metrics"):
        assert expected in paths, expected
    # The user's own trace and planner graph are not the console.
    for open_path in swarm_gate.OPEN_EVERYWHERE:
        assert open_path not in paths, open_path
        assert not swarm_gate.is_console_path(open_path)


@pytest.mark.parametrize("edition", ["oss", "free"])
def test_every_console_route_is_closed_to_oss_and_free(client, monkeypatch, edition):
    if edition == "oss":
        monkeypatch.setenv("AVALOKA_EDITION", "oss")
    else:
        _as_plan(monkeypatch, "free")
    routes = console_routes()
    assert routes
    for method, path, template in routes:
        resp = _hit(client, method, path, FREE)
        assert resp.status_code == GATED, (method, template, resp.status_code, resp.text)
        detail = resp.json()["detail"]
        assert detail["code"] == swarm_gate.GATE_CODE and detail["reason"] == "commercial"
        assert "Professional and Enterprise" in detail["message"]


def test_anonymous_callers_still_get_401_not_a_hint_about_the_edition(client, monkeypatch):
    _as_plan(monkeypatch, "free")
    for method, path, template in console_routes():
        assert _hit(client, method, path, None).status_code == 401, (method, template)


@pytest.mark.parametrize("plan", ["professional", "enterprise"])
def test_paid_editions_pass_the_gate_on_every_console_route(client, monkeypatch, plan):
    _as_plan(monkeypatch, plan)
    for method, path, template in console_routes():
        resp = _hit(client, method, path, ENT)
        # Past the gate the route answers for itself: 200, or 404 for the
        # made-up mission and thread ids. Never the gate's 403.
        assert resp.status_code in (200, 404), (method, template, resp.status_code, resp.text)


def test_the_thread_trace_and_planner_graph_stay_open_on_oss(client, monkeypatch):
    """The owner's first 1.7 ask: a user sees how their prompt was acted upon,
    in every edition. The gate must never grow back over these two."""
    try:
        from tests.test_agent_trace import _chat
    except ImportError:  # pragma: no cover
        from test_agent_trace import _chat

    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    thread_id, body = _chat(client, monkeypatch, user="oss-user")
    headers = make_auth_headers("oss-user")
    trace = client.get(body["trace_url"], headers=headers)
    assert trace.status_code == 200, trace.text
    assert trace.json()["thread_id"] == thread_id
    graph = client.get(f"/threads/{thread_id}/planner-graph?format=json", headers=headers)
    assert graph.status_code == 200, graph.text
    # ...while the operator's console stays closed to the same caller.
    assert client.get("/api/agent-metrics", headers=headers).status_code == GATED


def test_the_dependency_is_declared_on_the_route_not_hidden_in_its_body():
    """Structural: each matched route carries require_swarm_console as a
    dependency, so reading the signature tells the truth about the gate."""
    from app.api.server import app, require_swarm_console

    for route in app.routes:
        path = getattr(route, "path", None)
        if not path or not swarm_gate.is_console_path(path):
            continue
        calls = [d.call for d in route.dependant.dependencies]
        assert require_swarm_console in calls, path


# ----------------------------------------------------------- the capability
def test_capabilities_tell_the_ui_whether_the_console_exists(client, monkeypatch):
    _as_plan(monkeypatch, "free")
    free = client.get("/api/capabilities", headers=FREE).json()["features"]
    assert free["swarm_console"] == {
        "enabled": False, "reason": "commercial", "page": "/swarm",
        "message": swarm_gate.explain_swarm_console("free"),
    }
    # The swarm itself is still there, at the preview limit.
    assert free["swarm"]["limits"]["max_clones_per_mission"] == 2

    _as_plan(monkeypatch, "professional")
    pro = client.get("/api/capabilities", headers=PRO).json()["features"]
    assert pro["swarm_console"]["enabled"] is True and pro["swarm_console"]["reason"] == "open"

    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    oss = client.get("/api/capabilities").json()["features"]
    assert oss["swarm_console"]["enabled"] is False and oss["swarm"]["enabled"] is not None


# ------------------------------------------------------------- the decision
def test_the_gate_is_commercial_outside_professional_and_enterprise():
    for edition in EDITIONS:
        reason = swarm_console_gate(edition)
        if edition in ("professional", "enterprise"):
            assert reason is GateReason.OPEN, edition
        else:
            assert reason is GateReason.COMMERCIAL, edition
    # Unknown and missing editions fall to the closed side.
    assert swarm_console_gate(None) is GateReason.COMMERCIAL
    assert swarm_console_gate("platinum") is GateReason.COMMERCIAL


def test_the_gate_never_nags_an_edition_that_has_the_console():
    assert "Professional" not in swarm_gate.explain_swarm_console("enterprise")
    assert "Professional and Enterprise" in swarm_gate.explain_swarm_console("oss")


def test_the_swarm_is_not_gated_by_the_console():
    """The owner's line: the swarm runs everywhere, the console is paid."""
    from app.core.editions import GATE_REASON, resolve_capabilities

    assert GATE_REASON["swarm_intelligence"] is GateReason.OPEN
    for edition in EDITIONS:
        assert resolve_capabilities("avaloka_hosted", edition, has_commercial_build=False).swarm_intelligence
