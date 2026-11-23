""""What predicts X", answered the same way from every entry point.

The rule and the template are :mod:`avaloka.drivers_answer` (re-exported
here). This module is where an entry point hands over the rows it holds and
gets back the answer or None, so there is ONE routing decision:
:func:`answer_drivers_turn`.

Every entry point passes through it:

- the planner graph's first node, :func:`drivers_route_node`
  (app/api/workflow.py ``build_graph``). That graph is the ``langgraph dev``
  assistant / runs API (app/api/langgraph_app.py), the web chat's GRAPH and
  the Streamlit app, so a drivers question that reaches any of them is
  answered before memory, the conversational agent, the planner or the coder
  run, and the turn ends there: no narrator or verifier rewrites a figure.
- the web chat's fast path (app/api/server.py ``send_message``), which calls
  the same function with the same inputs (:func:`inputs_from_session`) before
  it builds the graph state, and leaves those inputs on the state so the node
  decides identically.
- the swarm and the CLI never reach the coder: each clone and ``avaloka
  analyze`` answer through :func:`avaloka.goal_analysis.analyze_goal`, and the
  swarm's merged answer carries :func:`drivers_table` (avaloka/coordinator.py).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from avaloka.drivers_answer import (  # noqa: F401  (the shared rule, re-exported)
    DRIVERS_QUESTION,
    MODEL_REQUEST,
    SHOWN,
    drivers_route,
    drivers_table,
    is_drivers_question,
    narrate_drivers,
    shown_drivers,
    table_markdown,
    wants_a_model,
)

logger = logging.getLogger(__name__)

#: What the server leaves on the graph state (ETLState.drivers_inputs).
STATE_KEY = "drivers_inputs"
#: The output table's action id; clients and tests find the table by it.
ACTION_ID = "drivers_route"


def _loaded(value: Any) -> Any:
    """A session value as stored: JSON text is parsed, anything else kept."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def inputs_from_session(sess: Mapping[str, Any], *, total_rows: Optional[int], datasets: int) -> Dict[str, Any]:
    """The rows a web-chat thread holds, and what they are a sample of.

    Upload threads read the working file; DB-table threads read the rows the
    session drew from the table (``db_sample`` says N of M)."""
    db_sample = _loaded(sess.get("db_sample"))
    pending = _loaded(sess.get("pending_clarification"))
    local = sess.get("active_data_source_location_local") or sess.get("active_data_source_location")
    return {
        "sources": [[local, False], [sess.get("work_local_input"), False], [sess.get("sample_local_input"), True]],
        "sheet_name": sess.get("sheet_name") or None,
        "db_sample": db_sample if isinstance(db_sample, dict) else None,
        "table": sess.get("source_table") or sess.get("filename") or None,
        "total_rows": total_rows,
        "datasets": int(datasets),
        "pending_clarification": bool(pending),
    }


def inputs_from_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """The same inputs from a planner-graph state: the server's own when it
    left them, otherwise what a ``langgraph dev`` / runs-API caller passed."""
    left = state.get(STATE_KEY)
    if isinstance(left, Mapping) and left.get("sources"):
        return dict(left)
    sources = [[state.get(k), False] for k in (
        "active_data_source_location_local", "data_source_location_local", "active_data_source_location",
        "data_source_location", "full_data_location")]
    sources.append([state.get("sample_data_location"), True])
    return {
        "sources": sources,
        "sheet_name": state.get("sheet_name") or None,
        "db_sample": None,
        "table": state.get("active_db_source_table") or None,
        "total_rows": None,
        "datasets": len(state.get("active_dataset_ids") or []) or 1,
        "pending_clarification": bool(state.get("pending_clarification")),
    }


def _basis(inputs: Mapping[str, Any], rows: int, from_sample_file: bool) -> str:
    """What the drivers answer was computed over: all N rows, or N of M."""
    from app.core.db_sampling import DbSample, sample_scope

    sample = DbSample.from_session(inputs.get("db_sample"))
    if sample is not None:
        table = inputs.get("table") or "database"
        if not sample.is_partial:
            return f"Computed over all {rows:,} rows of the {table} table."
        return (f"Computed from {sample_scope(sample)} of the {table} table, so each figure is a "
                "sample estimate.")
    total = inputs.get("total_rows")
    if total and rows < total:
        return f"Estimated from a {rows:,}-row sample of {total:,} rows, so each figure is a sample estimate."
    if from_sample_file and not total:
        return f"Computed over the {rows:,} rows of the working sample."
    return f"Computed over all {rows:,} rows."


def answer_drivers_turn(inputs: Mapping[str, Any], question: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """THE ROUTING DECISION, for every entry point: ``(answer, output_table)``
    for "what predicts X" from the shared driver ranking over the rows in
    ``inputs``, or None, and the turn runs as before.

    None when the question is not a drivers question or names a model (the
    MODEL_REQUEST escape hatch: it goes to the coder), when the thread has
    more than one dataset or a clarification pending, when the outcome does
    not resolve to a column, or when no local file under the recompute cap
    can be read.
    """
    from app.agents.claim_verifier import RECOMPUTE_MAX_BYTES
    from app.agents.input_reader import read_input_frame
    from file_handler.null_policy import read_csv_best_effort

    if not is_drivers_question(question):
        return None
    if int(inputs.get("datasets") or 1) != 1 or inputs.get("pending_clarification"):
        return None
    path, from_sample = next(((str(p), bool(s)) for p, s in inputs.get("sources") or []
                              if p and "://" not in str(p) and Path(str(p)).is_file()), (None, False))
    if path is None or Path(path).stat().st_size > RECOMPUTE_MAX_BYTES:
        return None
    try:
        frame = read_input_frame(path, read_csv_best_effort, sheet_name=inputs.get("sheet_name") or None)
        result = drivers_route(frame, question) if isinstance(frame, pd.DataFrame) else None
    except Exception as exc:  # noqa: BLE001 - unreadable rows: the normal pipeline answers
        logger.warning("[drivers-route] not answered: %s", type(exc).__name__)
        return None
    if result is None:
        return None
    table = drivers_table(result)
    answer = narrate_drivers(result, _basis(inputs, int(len(frame)), from_sample))
    target = (result.get("metrics") or {}).get("target")
    return answer, {"title": f"Association with {target}", "rows": table, "action_ids": [ACTION_ID],
                    "group_by": [], "metric_columns": ["association"]}


def turn_question(state: Mapping[str, Any]) -> str:
    """The user's words this turn: the latest human message, without the
    run context the web chat appends to it."""
    from app.agents.user_text import last_user_text

    return last_user_text(state)


#: The verifier does not run on this answer: it holds no figure the ranking
#: did not produce, and the turn ends at the node.
_NOT_VERIFIED = {"safe_to_present": True, "claims_checked": 0, "n_problems": 0, "findings": [],
                 "skipped": "deterministic drivers answer: every figure is the shared ranking's own"}


def drivers_route_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    """The planner graph's first node. A drivers question it can answer ends
    the turn here with the answer as the final AI message and the table on
    ``output_tables`` / ``output_json``; anything else leaves the state as it
    was (``drivers_answered`` False) and the graph runs as before."""
    question = turn_question(state)
    answered = answer_drivers_turn(inputs_from_state(state), question) if question else None
    if answered is None:
        return {"drivers_answered": False}
    from langchain_core.messages import AIMessage

    from app.api import reply_segments as rs

    text, table = answered
    return {
        "drivers_answered": True,
        # The template wrote every word of it (#581): plain, whole.
        "messages": [rs.tag(AIMessage(content=text), rs.PLAIN)],
        "analysis_narrative": text,
        "output_tables": [table],
        "output_json": table["rows"],
        "verification_report": dict(_NOT_VERIFIED),
        "verification_safe_to_present": True,
    }


def route_after_drivers(state: Mapping[str, Any]) -> str:
    """``answered`` ends the turn; ``continue`` runs the graph as before."""
    return "answered" if state.get("drivers_answered") else "continue"


__all__: List[str] = [
    "ACTION_ID", "DRIVERS_QUESTION", "MODEL_REQUEST", "SHOWN", "STATE_KEY", "answer_drivers_turn",
    "drivers_route", "drivers_route_node", "drivers_table", "inputs_from_session", "inputs_from_state",
    "is_drivers_question", "narrate_drivers", "route_after_drivers", "shown_drivers", "table_markdown",
    "turn_question", "wants_a_model",
]
