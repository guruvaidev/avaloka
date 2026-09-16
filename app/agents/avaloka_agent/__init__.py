"""Top-level conversational Avaloka agent.

Sits above ``plan_etl`` in the LangGraph and acts as the user-facing
dispatcher: classifies intent, reasons about dataset size / fidelity /
execution mode, resolves infra preferences, loads Redis session context,
and either replies directly or delegates to the existing planner.
"""

from app.agents.avaloka_agent.agent import avaloka_agent_node, route_avaloka

__all__ = ["avaloka_agent_node", "route_avaloka"]
