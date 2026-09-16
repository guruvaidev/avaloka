"""Avaloka mission layer.

The DataMission is the single canonical contract that every interface
(CLI, web conversation, REST API, MCP server) compiles user intent into.
The shared engine — estimator, router, swarm — decides how to execute it.

Invariant: cli_mission == web_mission == api_mission == mcp_mission.
"""

from app.missions.schema import (
    DataMission,
    MissionSpec,
    Source,
    Goal,
    Execution,
    Sampling,
    Validation,
    Output,
    GoalType,
    ExecutionMode,
    FullDataEngine,
)
from app.missions.compiler import compile_mission

__all__ = [
    "DataMission",
    "MissionSpec",
    "Source",
    "Goal",
    "Execution",
    "Sampling",
    "Validation",
    "Output",
    "GoalType",
    "ExecutionMode",
    "FullDataEngine",
    "compile_mission",
]
