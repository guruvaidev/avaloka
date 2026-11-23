"""A typed password that ends a sentence is the password, not the password plus its full stop.

rc18 live: "... password X." sealed "X." and Postgres refused the login; without
the full stop it worked. Sentence punctuation followed by whitespace or the end
of the message is not part of an unquoted value; a quoted or backticked value
is taken exactly, so a password that really ends in punctuation can be typed.
Synthetic values only.
"""
from __future__ import annotations

import hashlib
import re
import secrets

import pytest
from langchain_core.messages import AIMessage, HumanMessage

PW = "Zsynth9"
HEAD = "Register a new PostgreSQL database connection named rw18chat: host postgres, port 5432, database warehouse, "


@pytest.fixture
def sealing_key(monkeypatch):
    import app.api.cloud_connections as cc
    key = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)


def _sealed_values(text):
    """{label: [value, ...]} -- what each sealed token in the sealed text opens to."""
    from app.core.typed_secrets import TOKEN_RE, open_sealed, seal_typed_secrets
    sealed, _n = seal_typed_secrets(text)
    out = {}
    for m in TOKEN_RE.finditer(sealed):
        out.setdefault(m.group(1), []).append(open_sealed(m.group(0)))
    return sealed, out


@pytest.mark.parametrize("typed, password", [
    (f"username rw18chat, password {PW}.", PW),                    # ends the message
    (f"password {PW}. Thanks", PW),                                # ends a sentence
    (f"password {PW}, user rw18chat", PW),                         # a list goes on
    (f"password {PW}! please", PW),
    (f"password '{PW}.'", f"{PW}."),                               # quoted: exact
    (f"password `{PW}!`", f"{PW}!"),                               # backticked: exact
    (f'password "{PW}?"', f"{PW}?"),
    ("password a.b", "a.b"),                                       # internal punctuation kept
    ("password a.b.", "a.b"),
    ("password a,b;c", "a,b;c"),
])
def test_the_sealed_password_is_exactly_what_was_meant(sealing_key, typed, password):
    sealed, values = _sealed_values(HEAD + typed)
    assert values.get("password") == [password], values
    assert password not in sealed.replace("⟦", " ")  # sealed, not left in the text


@pytest.mark.parametrize("typed, label, value", [
    ("secret_key Zsec9. region us-east-1", "secret_key", "Zsec9"),
    ("access_key AKIASYNTH1, secret_key Zsec9", "access_key", "AKIASYNTH1"),
    ("sas token sv=2024&sig=Zsig9.", "az_sas_token", "sv=2024&sig=Zsig9"),
    ("account key 'Zacct9=='", "az_access_key", "Zacct9=="),
])
def test_other_sealed_labels_stop_at_sentence_punctuation_too(sealing_key, typed, label, value):
    _sealed, values = _sealed_values("register s3 bucket b, " + typed)
    assert values.get(label) == [value], values


def test_a_service_account_private_key_is_taken_exactly(sealing_key):
    # Not PEM-shaped on purpose (repo hygiene forbids key headers in tracked files);
    # what matters is that the escaped newlines and the trailing full stop survive.
    key = "synthetic-key-line-1\\nZkey9.\\nsynthetic-key-line-3."
    _sealed, values = _sealed_values('register gcs bucket b service account {"private_key": "' + key + '"}.')
    assert values.get("private_key") == [key]


def test_a_chat_registration_ending_in_a_full_stop_stores_the_password(sealing_key, monkeypatch):
    """End to end through the planner: what reaches the sealed store is the password."""
    import app.agents.planner as planner
    import app.api.db_connections as dbc
    from app.api.cloud_connections import encrypt_secret
    from app.core.typed_secrets import seal_typed_secrets
    stored = []

    async def register_and_save(user_id, body):
        stored.append(dict(body))
        return {"id": "row-1", "customer_id": body["customer_id"]}, encrypt_secret("ak_chat")

    monkeypatch.setattr(dbc, "register_and_save", register_and_save, raising=False)
    for name in ("_seed_ray_fields", "_seed_infra_request_if_needed"):
        monkeypatch.setattr(planner, name, lambda *a, **k: None)
    monkeypatch.setattr(planner, "llm", object(), raising=False)
    monkeypatch.setattr(planner, "_invoke_planner_with_recovery", lambda msgs, tool_choice, kw: AIMessage(
        content="", tool_calls=[{"name": "register_database", "id": "r", "args": {
            "alias": "rw18chat", "db_type": "postgresql", "host": "postgres", "port": 5432,
            "database": "warehouse", "user": "rw18chat", "password": "⟦credential⟧"}}]))
    text = seal_typed_secrets(HEAD + f"username rw18chat, password {PW}.")[0]
    state = {"user_id": "user-1", "dta_database_registry": {}, "messages": [HumanMessage(content=text)]}
    state.update(planner.plan_etl_job(state))
    assert stored and stored[0]["password"] == PW, [s.get("password") for s in stored]
