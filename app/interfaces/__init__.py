"""Avaloka interfaces.

Each interface (CLI, conversation, REST, MCP) is a thin translator from a user
surface into the canonical DataMission. None of them contains analytical
decision logic — they all funnel through ``app.missions.compile_mission`` and
``app.execution.route`` so the invariant holds:

    cli_mission == web_mission == api_mission == mcp_mission
"""
