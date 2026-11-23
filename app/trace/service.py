"""The server's view of the trace: persist a finished turn, serve a thread's.

Kept out of ``app/api/server.py`` so the routes there stay a few lines each and
this logic is testable without the HTTP stack.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from app.trace import store
from app.trace.build import agent_metrics, build_turn, planner_graph, trace_response

logger = logging.getLogger(__name__)


def turn_from_state(final: Mapping[str, Any], *, execution_context: Optional[Mapping[str, Any]] = None,
                    question: Optional[str] = None,
                    dataset_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """One turn record from a finished graph state, or ``None`` if nothing was recorded."""
    entries = final.get("agent_trace") or []
    if not entries:
        return None
    scope = {}
    if execution_context:
        scope = {k: execution_context.get(k) for k in
                 ("mode", "mode_label", "sample", "rows_analyzed", "total_rows", "is_sampled")
                 if execution_context.get(k) is not None}
        if "mode" in scope:
            scope.setdefault("fidelity", scope["mode"])
    if dataset_name:
        # The name the user uploaded, not the working copy the graph reads.
        scope["dataset"] = dataset_name
    return build_turn(
        entries,
        turn_id=final.get("trace_turn_id"),
        question=question,
        data_scope=scope,
        swarm_plan=final.get("avaloka_swarm"),
        swarm_message=final.get("avaloka_swarm_message"),
    )


async def persist_from_state(thread_id: str, user_id: Optional[str], final: Mapping[str, Any], *,
                             execution_context: Optional[Mapping[str, Any]] = None,
                             question: Optional[str] = None,
                             dataset_name: Optional[str] = None) -> Optional[str]:
    """Store this turn's trace; returns its turn id when stored. Never raises."""
    try:
        turn = turn_from_state(final, execution_context=execution_context, question=question,
                               dataset_name=dataset_name)
        if turn and await store.append_turn(thread_id, user_id, turn):
            return turn.get("turn_id")
    except Exception:  # noqa: BLE001 - the trace must never fail a chat turn
        logger.warning("[trace] could not persist the trace for %s", thread_id, exc_info=True)
    return None


async def thread_trace(thread_id: str, *, checkpoint_state: Optional[Mapping[str, Any]] = None,
                       swarm_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    turns = await store.load_turns(thread_id)
    if not turns and checkpoint_state:
        # Redis lost it (or was down when the turn finished): the in-process
        # checkpoint still holds the latest turn.
        turn = turn_from_state(checkpoint_state)
        turns = [turn] if turn else []
    return trace_response(thread_id, turns, swarm_limits=swarm_limits)


def pick_turn(turns: List[Dict[str, Any]], turn_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not turns:
        return None
    if not turn_id or turn_id == "latest":
        return turns[-1]
    return next((t for t in turns if t.get("turn_id") == turn_id), None)


def graph_for(turn: Dict[str, Any], *, plan_steps: Optional[list] = None) -> Dict[str, Any]:
    return planner_graph(turn, plan_steps=plan_steps)


async def user_agent_metrics(user_id: str, *, max_threads: int = 50) -> Dict[str, Any]:
    """Per-agent aggregates across the caller's traced threads."""
    threads = (await store.user_trace_threads(user_id))[-max_threads:]
    turns: List[Dict[str, Any]] = []
    for tid in threads:
        turns.extend(await store.load_turns(tid))
    return {"threads": len(threads), "turns": len(turns), "agents": agent_metrics(turns)}


def swarm_limits_for(plan: Optional[str]) -> Dict[str, Any]:
    """The swarm limits for a plan in this build, as the API reports them."""
    from app.core.editions import swarm_limits

    return swarm_limits(plan).as_dict()
