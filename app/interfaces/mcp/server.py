"""Avaloka MCP server — mission-level tools over a stdio/SSE transport.

Two layers, deliberately separated so the logic is testable without the SDK:

  * **pure functions** (``tool_plan_mission`` …) take a plain intent dict and
    return JSON-able dicts by delegating to ``app.interfaces.service``. They have
    no MCP dependency and are what the tests exercise.
  * **``build_server()`` / ``main()``** lazily import the ``mcp`` SDK and register
    those functions as MCP tools. Missing SDK raises a clear, actionable error
    only when you actually try to *serve*, never on import.

Run (once the ``mcp`` SDK is installed):

    python -m app.interfaces.mcp.server        # stdio transport
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from app.execution.environment import detect_environment
from app.interfaces.service import plan_mission, planned_to_dict


def _plan_remote(intent: Dict[str, Any], endpoint: str, token: Optional[str]) -> Dict[str, Any]:
    """Delegate planning to a deployed API (/api/missions/plan). Returns the same
    planned_to_dict shape as the in-process path, so remote and local agree."""
    import requests

    url = endpoint.rstrip("/") + "/api/missions/plan"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(url, json=intent, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

SERVER_NAME = "avaloka"
SERVER_INSTRUCTIONS = (
    "Avaloka is an AI data scientist. Describe a data mission (a source URI plus "
    "an optional goal, budget and constraints) and Avaloka compiles it into a "
    "canonical DataMission, estimates cost/runtime, and returns an execution "
    "plan. It is cloud-environment-aware: on GCP it reasons in GCP primitives, "
    "on AWS in AWS, on Azure in Azure, and locally on-prem."
)

# Declarative specs — reused for the SDK registration and returned by an
# introspection tool so a caller can discover intent fields without the SDK.
TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "plan_mission",
        "description": (
            "Compile intent into a DataMission and return the full execution plan "
            "(steps, recommended mode, cost/runtime estimate, cloud target)."
        ),
    },
    {
        "name": "estimate_cost",
        "description": (
            "Return just the workload/cost/runtime estimate and the recommended "
            "execution mode for the given intent (no full plan)."
        ),
    },
    {
        "name": "describe_environment",
        "description": (
            "Report the detected cloud environment (gcp|aws|azure|local), how it "
            "was detected, and the service catalog Avaloka will reason with."
        ),
    },
    {
        "name": "inspect_intent",
        "description": (
            "Echo the canonical DataMission that the given intent compiles to, "
            "without estimating or routing — useful for validating input."
        ),
    },
    {
        "name": "visualize",
        "description": (
            "Build recommended charts for a small sample of rows and return the "
            "visualization spec (Vega-Lite-flavored) plus a rendered self-contained "
            "HTML artifact. MCP hosts can render the spec inline (no browser needed — "
            "the terminal CLI opens the HTML instead)."
        ),
    },
]


# --- pure tool functions (no MCP dependency) --------------------------------

def tool_plan_mission(intent: Dict[str, Any], endpoint: Optional[str] = None,
                      token: Optional[str] = None) -> Dict[str, Any]:
    """Full plan: mission + estimate + execution plan (the primary tool).

    When ``endpoint`` (or ``AVALOKA_API_URL``) is set, delegates to the deployed
    API so the MCP tool can drive a cloud backend; otherwise plans in-process.
    Either way the returned shape is identical for the same intent.
    """
    endpoint = endpoint or os.getenv("AVALOKA_API_URL")
    if endpoint:
        return _plan_remote(intent, endpoint, token or os.getenv("AVALOKA_API_TOKEN"))
    return planned_to_dict(plan_mission(intent))


def tool_estimate_cost(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Just the estimate + recommended mode — the cheap 'what will this cost?'."""
    result = planned_to_dict(plan_mission(intent))
    est = result["estimate"]
    return {
        "mission_name": result["plan"]["mission_name"],
        "recommended_mode": est["recommended_mode"],
        "workload_score": est["workload_score"],
        "sampled_cost_usd": est["sampled_cost_usd"],
        "full_cost_usd": est["full_cost_usd"],
        "full_runtime_minutes": est["full_runtime_minutes"],
        "full_pass_worthwhile": est["full_pass_worthwhile"],
        "rationale": est["rationale"],
    }


def tool_describe_environment(allow_network: bool = False) -> Dict[str, Any]:
    """Expose Avaloka's cloud-environment awareness to the calling agent."""
    env = detect_environment(allow_network=allow_network)
    return {
        "provider": env.provider.value,
        "detected_from": env.source,
        "region": env.region,
        "is_cloud": env.is_cloud,
        "service_catalog": env.service_catalog,
        "agent_context": env.system_prompt_fragment(),
    }


def tool_inspect_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Compile intent → DataMission and return it, without routing/estimating."""
    from app.missions.compiler import compile_mission

    return {"mission": compile_mission(intent).canonical()}


def tool_visualize(
    sample_rows: List[Dict[str, Any]],
    dataset_id: str = "dataset",
    target_column: Optional[str] = None,
    render: bool = True,
) -> Dict[str, Any]:
    """Recommend charts for ``sample_rows`` and return the spec (+ a rendered HTML
    artifact). The spec's charts are Vega-Lite-flavored, so an MCP host can render
    them inline; the ``html_path``/``html_url`` is for hosts/CLIs that prefer a file.
    """
    from app.agents.visualization_agent import build_visualization_config_from_sample
    from app.interfaces import artifacts

    cfg = build_visualization_config_from_sample(
        dataset_id=dataset_id, sample_rows=sample_rows, target_column=target_column)
    out: Dict[str, Any] = {
        "visualization_config": cfg,
        "chart_count": len(cfg.get("charts") or []),
        "visualization_status": cfg.get("visualization_status"),
    }
    if render:
        html = artifacts.visualization_to_html(cfg, sample_rows, title=f"Avaloka — {dataset_id}")
        path = artifacts.write_artifact(html, suffix=".html")
        out["html_path"] = str(path)
        out["html_url"] = artifacts.to_file_url(path)
    return out


_DISPATCH = {
    "plan_mission": tool_plan_mission,
    "estimate_cost": tool_estimate_cost,
    "inspect_intent": tool_inspect_intent,
    "visualize": tool_visualize,
}


# --- MCP SDK wiring (lazy) ---------------------------------------------------

def build_server():  # pragma: no cover - exercised only when the SDK is present
    """Construct a FastMCP server exposing the mission tools.

    Imported lazily so the module works without the ``mcp`` package installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # actionable, only when actually serving
        raise RuntimeError(
            "The Avaloka MCP server requires the 'mcp' package. Install it with "
            "`pip install mcp` (or add it to requirements.txt) to serve; the "
            "pure tool functions in this module work without it."
        ) from exc

    server = FastMCP(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)

    @server.tool(description=TOOL_SPECS[0]["description"])
    def plan_mission_tool(intent: Dict[str, Any]) -> Dict[str, Any]:
        return tool_plan_mission(intent)

    @server.tool(description=TOOL_SPECS[1]["description"])
    def estimate_cost(intent: Dict[str, Any]) -> Dict[str, Any]:
        return tool_estimate_cost(intent)

    @server.tool(description=TOOL_SPECS[2]["description"])
    def describe_environment(allow_network: bool = False) -> Dict[str, Any]:
        return tool_describe_environment(allow_network)

    @server.tool(description=TOOL_SPECS[3]["description"])
    def inspect_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
        return tool_inspect_intent(intent)

    @server.tool(description=TOOL_SPECS[4]["description"])
    def visualize(sample_rows: List[Dict[str, Any]], dataset_id: str = "dataset",
                  target_column: Optional[str] = None) -> Dict[str, Any]:
        return tool_visualize(sample_rows, dataset_id=dataset_id, target_column=target_column)

    return server


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    """Serve over stdio (the default MCP transport for local/gateway use)."""
    build_server().run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
