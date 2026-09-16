"""Avaloka MCP server interface.

This is the fourth thin interface (after CLI, web, REST). It does **not** wrap
Avaloka's internal agents — exposing ``generate_code`` would leak internal
structure. Instead it exposes mission-level tools (``avaloka.plan_mission``,
``avaloka.estimate_cost``, ``avaloka.describe_environment``) that compile the
same flat *intent* into the same canonical ``DataMission`` as every other
interface. The invariant ``cli_mission == mcp_mission`` holds because both call
``app.interfaces.service``.

The ``mcp`` SDK is imported lazily inside :func:`build_server`, so this module
(and its pure-Python tool functions) import and unit-test cleanly whether or not
the SDK is installed.
"""

from app.interfaces.mcp.server import (
    TOOL_SPECS,
    tool_describe_environment,
    tool_estimate_cost,
    tool_inspect_intent,
    tool_plan_mission,
)

__all__ = [
    "TOOL_SPECS",
    "tool_plan_mission",
    "tool_estimate_cost",
    "tool_describe_environment",
    "tool_inspect_intent",
]
