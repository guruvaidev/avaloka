"""rc18c live regrade: two chat-registration follow-ups (synthetic values only).

1. chatB: the details were re-sent after a successful registration, and the
   model registered them again as a second connection (postgres_sample_*).
   The same host, database and user now reuse the existing connection.
2. chatC1: "list connections", asked for the second time in the thread, was
   answered with the registration refusal. The graph reducer dropped the new
   message as a duplicate of the first "list connections" while appending the
   history's display copies of earlier replies, so the planner answered the
   previous "yes" again.
"""
from __future__ import annotations

import hashlib
import secrets

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.planner as planner
from app.graph.etl_state import filter_messages

PLACEHOLDER = "⟦credential⟧"
DETAILS = ("Connect my PostgreSQL database: host postgres, port 5432, database sample, "
           "username rw18b, password Dummy123")


@pytest.fixture
def sealing_key(monkeypatch):
    import app.api.cloud_connections as cc
    key = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)


@pytest.fixture
def store(monkeypatch, sealing_key):
    """The onboarding register-and-save, recorded; the saved-connections store, empty."""
    import app.api.db_connections as dbc
    from app.api.cloud_connections import encrypt_secret
    seen, rows = [], []

    async def register_and_save(user_id, body):
        seen.append(dict(body))
        return {"id": f"row-{len(seen)}", "customer_id": body["customer_id"]}, encrypt_secret("ak_chat")

    class _Rows:
        def list_for(self, user_id):
            return [dict(r) for r in rows]

    monkeypatch.setattr(dbc, "register_and_save", register_and_save, raising=False)
    monkeypatch.setattr(dbc, "connection_store", lambda: _Rows())
    for name in ("_seed_ray_fields", "_seed_infra_request_if_needed"):
        monkeypatch.setattr(planner, name, lambda *a, **k: None)
    monkeypatch.setattr(planner, "llm", object(), raising=False)
    monkeypatch.setattr(planner, "_invoke_planner_with_recovery", lambda msgs, tool_choice, kw: AIMessage(
        content="", tool_calls=[{"name": "register_database", "id": "r", "args": {
            "alias": "postgres_sample", "db_type": "postgresql", "host": "postgres", "port": 5432,
            "database": "sample", "user": "rw18b", "password": PLACEHOLDER}}]))
    return seen, rows


def _turn(state, text):
    from app.core.typed_secrets import seal_typed_secrets
    state["messages"] = list(state.get("messages") or []) + [HumanMessage(content=seal_typed_secrets(text)[0])]
    state.update(planner.plan_etl_job(state))
    return [m for m in state["messages"] if isinstance(m, AIMessage)][-1].content


def test_resending_the_same_details_reuses_the_connection(store):
    seen, _rows = store
    state = {"user_id": "user-1", "dta_database_registry": {}, "messages": []}
    first = _turn(state, DETAILS)
    assert len(seen) == 1 and "Registered" in first, first
    again = _turn(state, DETAILS)
    assert len(seen) == 1, f"registered a second copy: {[b['customer_id'] for b in seen]}"
    assert "already registered" in again and "postgres_sample" in again
    assert list(state["dta_database_registry"]) == ["postgres_sample"]


def test_the_same_details_in_another_thread_reuse_the_saved_connection(store):
    seen, rows = store
    from app.api.cloud_connections import encrypt_secret
    user_enc = encrypt_secret("rw18b")
    rows.append({"user_id": "user-1", "connection_name": "postgres_sample", "customer_id": "postgres_sample_4a02a4a0",
                 "database_type": "postgresql", "host": "postgres", "port": 5432, "database_name": "sample",
                 "status": "active", "username": "", "username_ciphertext": user_enc["ciphertext"],
                 "username_iv": user_enc["iv"]})
    state = {"user_id": "user-1", "dta_database_registry": {}, "messages": []}
    reply = _turn(state, DETAILS)
    assert not seen, "registered a duplicate of a saved connection"
    assert "already registered" in reply
    assert state["dta_database_registry"]["postgres_sample"]["conn_id"] == "postgres_sample_4a02a4a0"


def test_another_user_or_database_is_still_a_new_connection(store):
    seen, _rows = store
    state = {"user_id": "user-1", "dta_database_registry": {}, "messages": []}
    _turn(state, DETAILS)
    planner._invoke_planner_with_recovery = lambda msgs, tool_choice, kw: AIMessage(  # noqa: E731 (restored by monkeypatch)
        content="", tool_calls=[{"name": "register_database", "id": "r", "args": {
            "alias": "postgres_wh", "db_type": "postgresql", "host": "postgres", "port": 5432,
            "database": "warehouse", "user": "rw18b", "password": PLACEHOLDER}}])
    _turn(state, DETAILS.replace("database sample", "database warehouse"))
    assert len(seen) == 2


# ---- the repeated "list connections" ----

CTX = "\n\n[Analysis context] Current fidelity: portfolio_samples"


def test_a_repeated_message_is_this_turns_last_message():
    first = HumanMessage(content="list connections" + CTX)
    reply = AIMessage(content="To register rw18cb, I need the password.")
    x = [first, AIMessage(content="Registered Data Sources ..."), HumanMessage(content="yes"), reply]
    # What the server sends: its history (replies as displayed, marked as
    # commentary, so not equal to the state's) and this turn's message.
    y = [HumanMessage(content="list connections" + CTX),
         AIMessage(content="Registered Data Sources ... (commentary — not checked)"),
         HumanMessage(content="yes"),
         AIMessage(content="To register rw18cb, I need the password. (commentary — not checked)"),
         HumanMessage(content="list connections" + CTX)]
    merged = filter_messages(x, y)
    assert isinstance(merged[-1], HumanMessage) and merged[-1].content.startswith("list connections"), \
        [type(m).__name__ + ":" + m.content[:20] for m in merged]


def test_list_connections_after_a_refused_registration_lists(store):
    """chatC1's last turn, through the reducer and the planner."""
    seen, _rows = store
    state = {"user_id": "user-1", "messages": [
        HumanMessage(content="list connections" + CTX),
        AIMessage(content="Registered Data Sources ..."),
        HumanMessage(content="Register a new PostgreSQL database connection named rw18cb: host postgres, "
                             "port 5432, database sample, username rw18cb."),
        AIMessage(content="I couldn't find the password for rw18cb in what you typed, so nothing was stored."),
        HumanMessage(content="yes"),
        AIMessage(content="To register rw18cb, I need the password.")],
        "dta_database_registry": {"rw18ca": {"db_type": "postgresql", "conn_id": "rw18ca_1", "host": "postgres",
                                             "port": 5432, "database": "sample", "user": "rw18ca"}}}
    y = [HumanMessage(content=m.content + (" (commentary — not checked)" if isinstance(m, AIMessage) else ""))
         if isinstance(m, HumanMessage) else AIMessage(content=m.content + " (commentary — not checked)")
         for m in state["messages"]] + [HumanMessage(content="list connections" + CTX)]
    state["messages"] = filter_messages(state["messages"], y)
    monkey_lists = []
    real = planner._handle_list_databases
    planner._handle_list_databases = lambda st: (monkey_lists.append(1), real(st))[1]
    try:
        out = planner.plan_etl_job(state)
    finally:
        planner._handle_list_databases = real
    reply = [m for m in out["messages"] if isinstance(m, AIMessage)][-1].content
    assert monkey_lists, reply
    assert "Registered Data Sources" in reply and "couldn't find the password" not in reply
    assert not seen
