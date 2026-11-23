"""Small adapter between HTTP sessions and PR 157's graph entry point."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional


def build_swarm_state(
    *, thread_id: str, session_id: str, session: Mapping[str, Any]
) -> dict[str, Any]:
    """Return graph identity/transient state without conflating ids."""
    return {
        "session_id": session_id,
        "thread_id": thread_id,
        "connection_event": session.get("connection_event"),
    }


def consume_connection_event(
    session: MutableMapping[str, Any], final_state: Mapping[str, Any]
) -> bool:
    """Persist the agent's one-shot event consumption decision.

    Returns whether the session changed. If the graph did not return the key,
    the stored event is retained rather than being lost accidentally.
    """
    if "connection_event" not in final_state:
        return False
    value = final_state.get("connection_event")
    if value is None:
        if "connection_event" not in session:
            return False
        session.pop("connection_event", None)
        return True
    if session.get("connection_event") == value:
        return False
    session["connection_event"] = value
    return True


def database_connection_event(
    *,
    customer_id: str,
    test_result: Mapping[str, Any],
    created_at: str,
    connection_id: Optional[str] = None,
    database_type_hint: Optional[str] = None,
) -> dict[str, Any]:
    """The ``connection_event`` /api/database/connect stores for onboarding.

    One builder so the event cannot drift from the connect response:

    * ``connection_id`` is the id the response hands back -- the saved
      connection id when the client connected through one (the legacy-UI
      path #248 adds), otherwise the customer id. Same rule as that
      response's ``response_connection_id``.
    * ``database_type`` is what the table listing reports, falling back to
      the type recorded on the saved connection when the listing has none.

    Carries no credential: the api_key never enters it, and this is the
    object the agent places in its prompt.
    """
    meta = test_result.get("meta") or {}
    tables = list(test_result.get("tables") or [])
    return {
        "kind": "database",
        "customer_id": customer_id,
        "connection_id": (connection_id or "").strip() or customer_id,
        "database_type": meta.get("database_type") or database_type_hint,
        "tables_available": len(tables),
        "tables_preview": tables[:20],
        "created_at": created_at,
    }
