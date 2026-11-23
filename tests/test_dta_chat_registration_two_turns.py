"""Registering a database by chat, when the credentials and the go-ahead are two turns.

Shape of the rc17 live run (~/avaloka-work/states/rweb17/cred599/chat2, minimum
fields, synthetic password): the user types the connection and its password;
the first reply is a question (there, the fabrication guard); the user says
"yes"; the model, which was shown ``⟦credential⟧`` where the password was,
calls register_database with that placeholder. At 8270ae3e the password was
looked up only in the newest human message -- "yes" -- so the placeholder
itself went to Postgres as the password and authentication failed.
"""
from __future__ import annotations

import hashlib
import secrets

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.planner as planner

SYN = "ZZ-SYNTH-PW-1"
PLACEHOLDER = "⟦credential⟧"
TYPED = ("Register a new PostgreSQL database connection named rw17chat: host postgres, port 5432, "
         "database warehouse, username rw17chat, password " + SYN)


@pytest.fixture
def sealing_key(monkeypatch):
    import app.api.cloud_connections as cc
    key = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)


@pytest.fixture
def store(monkeypatch, sealing_key):
    """The onboarding API's register-and-save, recorded."""
    import app.api.db_connections as dbc
    from app.api.cloud_connections import encrypt_secret
    seen = []

    async def register_and_save(user_id, body):
        seen.append(dict(body))
        return {"id": "row-1", "customer_id": body["customer_id"]}, encrypt_secret("ak_chat")

    monkeypatch.setattr(dbc, "register_and_save", register_and_save, raising=False)
    for name in ("_seed_ray_fields", "_seed_infra_request_if_needed"):
        monkeypatch.setattr(planner, name, lambda *a, **k: None)
    return seen


def _sealed(text: str) -> str:
    from app.core.typed_secrets import seal_typed_secrets
    return seal_typed_secrets(text)[0]


def _model_registers(monkeypatch, alias="rw17chat", password=PLACEHOLDER):
    """The planner model: calls register_database with what it was shown."""
    monkeypatch.setattr(planner, "llm", object(), raising=False)
    monkeypatch.setattr(planner, "_invoke_planner_with_recovery", lambda msgs, tool_choice, kw: AIMessage(
        content="", tool_calls=[{"name": "register_database", "id": "r", "args": {
            "alias": alias, "db_type": "postgresql", "host": "postgres", "port": 5432,
            "database": "warehouse", "user": "rw17chat", "password": password}}]))


def _turn(state, text):
    state["messages"] = list(state.get("messages") or []) + [HumanMessage(content=text)]
    state.update(planner.plan_etl_job(state))
    return [m for m in state["messages"] if isinstance(m, AIMessage)][-1].content


def test_yes_after_typed_credentials_registers_with_the_typed_password(store, monkeypatch):
    state = {"user_id": "user-1", "dta_database_registry": {},
             "messages": [HumanMessage(content=_sealed(TYPED)),
                          AIMessage(content="That request refers to fields this dataset doesn't contain.")]}
    _model_registers(monkeypatch)
    reply = _turn(state, "yes")
    assert store, reply
    assert store[0]["password"] == SYN, "the placeholder (or nothing) was sent as the password"
    assert PLACEHOLDER not in store[0]["password"]
    assert "Registered" in reply and SYN not in reply


def test_the_placeholder_is_never_sent_as_a_password(store, monkeypatch):
    """No typed credential in this flow: refuse, never register with the placeholder."""
    state = {"user_id": "user-1", "dta_database_registry": {}, "messages": []}
    _model_registers(monkeypatch)
    reply = _turn(state, "register rw17chat on postgres:5432 database warehouse user rw17chat")
    assert not store, f"registered with {store[0]['password']!r}" if store else ""
    assert "nothing was stored" in reply.lower()


def test_a_token_typed_for_another_connection_is_not_used(store, monkeypatch):
    state = {"user_id": "user-1", "dta_database_registry": {},
             "messages": [HumanMessage(content=_sealed(TYPED.replace("rw17chat:", "otherdb:")
                                                       .replace("named rw17chat", "named otherdb"))),
                          AIMessage(content="Registered **otherdb** as `postgresql`.")]}
    _model_registers(monkeypatch)
    reply = _turn(state, "now register rw17chat the same way, yes")
    assert not store, "a token typed for otherdb was opened for rw17chat"
    assert "nothing was stored" in reply.lower()


# ---- rc18 live shapes (~/avaloka-work/states/rweb18/cred18), synthetic values ----

UNNAMED = ("Connect my PostgreSQL database: host postgres, port 5432, database sample, "
           "username rw18b, password Dummy123.")


def _model_registers_as(monkeypatch, alias, user, database="sample", password=PLACEHOLDER):
    monkeypatch.setattr(planner, "llm", object(), raising=False)
    args = {"alias": alias, "db_type": "postgresql", "host": "postgres", "port": 5432,
            "database": database, "user": user}
    if password is not None:
        args["password"] = password
    monkeypatch.setattr(planner, "_invoke_planner_with_recovery", lambda msgs, tool_choice, kw: AIMessage(
        content="", tool_calls=[{"name": "register_database", "id": "r", "args": args}]))


def test_details_without_a_name_then_yes_registers(store, monkeypatch):
    """chatB: no connection name typed; the model picks "sample_postgres"."""
    state = {"user_id": "user-1", "dta_database_registry": {},
             "messages": [HumanMessage(content=_sealed(UNNAMED)),
                          AIMessage(content="Shall I register this PostgreSQL connection?")]}
    _model_registers_as(monkeypatch, "sample_postgres", "rw18b")
    reply = _turn(state, "yes")
    assert store, reply
    assert store[0]["password"] == "Dummy123"


def test_resending_the_details_as_the_refusal_asks_registers(store, monkeypatch):
    """chatB's loop: details, refusal, the same details again, yes -- the retry must work."""
    state = {"user_id": "user-1", "dta_database_registry": {}, "messages": [
        HumanMessage(content=_sealed(UNNAMED)),
        AIMessage(content="I couldn't find the password for `sample_postgres` in what you typed, "
                          "so nothing was stored. Send the connection details again in one message, "
                          "including the password."),
        HumanMessage(content=_sealed(UNNAMED)),
        AIMessage(content="Shall I register it?")]}
    _model_registers_as(monkeypatch, "sample_postgres", "rw18b")
    reply = _turn(state, "yes")
    assert store and store[0]["password"] == "Dummy123", reply


@pytest.mark.parametrize("model_password", [PLACEHOLDER, None], ids=["placeholder", "omitted"])
def test_a_password_typed_for_rw18ca_is_never_used_for_rw18cb(store, monkeypatch, model_password):
    """chatC1: rw18ca's details (with password) failed; rw18cb's details carry none."""
    state = {"user_id": "user-1", "dta_database_registry": {}, "messages": [
        HumanMessage(content=_sealed("Register a new PostgreSQL database connection named rw18ca: host postgres, "
                                     "port 5432, database sample, username rw18ca, password Dummy123.")),
        AIMessage(content="Couldn't register rw18ca: password authentication failed. Nothing was stored."),
        HumanMessage(content="Register a new PostgreSQL database connection named rw18cb: host postgres, "
                             "port 5432, database sample, username rw18cb."),
        AIMessage(content="Shall I register rw18cb?")]}
    _model_registers_as(monkeypatch, "rw18cb", "rw18cb", password=model_password)
    reply = _turn(state, "yes")
    assert not store, f"rw18cb was tried with {store[0].get('password')!r}" if store else ""
    assert "nothing was stored" in reply.lower()
