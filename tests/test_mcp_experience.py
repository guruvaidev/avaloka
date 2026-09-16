"""MCP experience tests — the avaloka MCP server's pure tools, dispatch, cross-surface
parity, remote delegation, the visualize tool, and graceful SDK wiring."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.interfaces.mcp import server as mcp
from app.interfaces.service import plan_mission, planned_to_dict

INTENT = {"source_uri": "customers.parquet", "goal_description": "Explain churn",
          "rows": 84_000_000, "max_cost_usd": 20}
ROWS = [{"country": "USA", "price": 10}, {"country": "UK", "price": 20}, {"country": "USA", "price": 30}]


# ------------------------------------------------------------------ tool shapes
def test_plan_mission_tool_shape():
    out = mcp.tool_plan_mission(INTENT)
    assert set(out) == {"mission", "plan", "estimate"}
    assert out["plan"]["mode"]


def test_estimate_cost_tool_shape():
    out = mcp.tool_estimate_cost(INTENT)
    assert {"recommended_mode", "workload_score", "sampled_cost_usd",
            "full_cost_usd", "rationale"}.issubset(out)


def test_describe_environment_tool_shape():
    out = mcp.tool_describe_environment(allow_network=False)
    assert "provider" in out and "service_catalog" in out and "agent_context" in out


def test_inspect_intent_returns_canonical_mission():
    out = mcp.tool_inspect_intent(INTENT)
    assert "mission" in out and out["mission"]["spec"]["source"]["uri"] == "customers.parquet"


# ------------------------------------------------------------------ dispatch + specs
def test_dispatch_and_specs_list_every_tool():
    for name in ("plan_mission", "estimate_cost", "inspect_intent", "visualize"):
        assert name in mcp._DISPATCH
    spec_names = {s["name"] for s in mcp.TOOL_SPECS}
    assert {"plan_mission", "estimate_cost", "describe_environment",
            "inspect_intent", "visualize"}.issubset(spec_names)


# ------------------------------------------------------------------ cross-surface parity
def test_mcp_plan_equals_the_shared_service():
    assert mcp.tool_plan_mission(INTENT) == planned_to_dict(plan_mission(INTENT))


# ------------------------------------------------------------------ remote delegation
def test_plan_mission_delegates_to_endpoint_when_set():
    remote_payload = {"mission": {}, "plan": {}, "estimate": {}}

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return remote_payload

    with patch("requests.post", return_value=FakeResp()) as post:
        out = mcp.tool_plan_mission(INTENT, endpoint="http://api:9000", token="jwt")
    assert out is remote_payload
    url = post.call_args.args[0]
    assert url == "http://api:9000/api/missions/plan"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer jwt"


def test_plan_mission_uses_env_endpoint(monkeypatch):
    monkeypatch.setenv("AVALOKA_API_URL", "http://env-api:9000")

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"ok": True}

    with patch("requests.post", return_value=FakeResp()) as post:
        mcp.tool_plan_mission(INTENT)
    assert post.call_args.args[0] == "http://env-api:9000/api/missions/plan"


# ------------------------------------------------------------------ visualize tool
def test_visualize_returns_spec_and_rendered_html():
    out = mcp.tool_visualize(ROWS, dataset_id="sales")
    assert out["chart_count"] >= 1
    assert out["visualization_status"] == "ready"
    assert Path(out["html_path"]).exists()
    assert out["html_url"].startswith("file://")
    assert "charts" in out["visualization_config"]


def test_visualize_can_skip_rendering():
    out = mcp.tool_visualize(ROWS, render=False)
    assert "html_path" not in out
    assert "visualization_config" in out


# ------------------------------------------------------------------ SDK wiring
def test_build_server_either_builds_or_raises_actionable_error():
    try:
        srv = mcp.build_server()
        # SDK present -> a FastMCP server object was constructed
        assert srv is not None
    except RuntimeError as e:
        # SDK absent -> the message must be actionable
        assert "mcp" in str(e).lower()
