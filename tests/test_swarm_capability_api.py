"""/api/capabilities states the swarm limit that applies to the caller.

Driven over HTTP through the real server harness. The limit itself is enforced
in avaloka/coordinator.py and tested in tests/avaloka/test_swarm_limits.py;
this file holds the report to the same numbers.
"""

from __future__ import annotations

try:  # tests/ as a package, or tests/ on sys.path
    from tests.server_harness import client, make_auth_headers  # noqa: F401
except ImportError:  # pragma: no cover
    from server_harness import client, make_auth_headers  # noqa: F401


def test_capabilities_report_the_swarm_limit_plainly(client, monkeypatch):
    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    swarm = client.get("/api/capabilities").json()["features"]["swarm"]
    assert swarm["limits"]["max_clones_per_mission"] == 2
    assert swarm["limits"]["execution"] == "local" and swarm["limits"]["cluster_fan_out"] is False
    if swarm["enabled"]:
        assert swarm["limited"] is True
        assert swarm["message"] == ("Swarm limited to 2 clones per mission on the free open-source "
                                    "edition, run locally.")


def test_capabilities_follow_the_callers_plan_in_a_commercial_build(client, monkeypatch):
    import app.api.server as server
    import avaloka.swarm_limits as sl

    monkeypatch.delenv("AVALOKA_EDITION", raising=False)
    monkeypatch.setattr(sl, "commercial_build", lambda: True)
    import app.core.editions as editions
    monkeypatch.setattr(editions, "commercial_build", lambda: True)
    monkeypatch.setattr(server, "resolve_plan", lambda uid: "professional")
    doc = client.get("/api/capabilities", headers=make_auth_headers("pro-user")).json()
    assert doc["features"]["swarm"]["limits"]["max_clones_per_mission"] == 16
    monkeypatch.setattr(server, "resolve_plan", lambda uid: "free")
    doc = client.get("/api/capabilities", headers=make_auth_headers("free-user")).json()
    assert doc["features"]["swarm"]["limits"]["max_clones_per_mission"] == 2


def test_a_build_without_swarm_reports_no_limit_either(client, monkeypatch):
    """Limits sit beside availability and never imply it.

    Availability comes only from the import probe. A build that does not ship
    the swarm module must not read as "limited" — there is nothing to limit.
    """
    import sys

    import app.agents.avaloka_agent as avaloka_agent_pkg
    from app.agents.avaloka_agent import capabilities as swarm_capabilities

    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "app.agents.avaloka_agent.swarm", None)
        m.delattr(avaloka_agent_pkg, "swarm", raising=False)
        swarm_capabilities.reset_cache()
        try:
            swarm = client.get("/api/capabilities").json()["features"]["swarm"]
        finally:
            swarm_capabilities.reset_cache()
    assert swarm["enabled"] is False
    assert swarm["reason"] == "not present in this build"
    assert swarm["limited"] is False and swarm["message"] is None
