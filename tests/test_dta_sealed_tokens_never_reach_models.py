"""A sealed credential token never reaches a model, and never reaches a person.

#599 sealed a password typed into the chat: the transcript, the logs and the
model saw ``⟦sealed:password:<ciphertext>.<nonce>⟧`` instead of the value. The
token is ciphertext, but a model can copy it or quote pieces of it, and a live
run showed a reply quoting two base64 fragments of one as "fields this dataset
doesn't contain". Now:

* every message any LangChain chat model is sent carries ``⟦credential⟧`` in
  place of each token (captured here at the HTTP layer, as the provider sees it);
* every thread response a person reads has whole tokens and any run of 8 or
  more of a token's characters removed.

The stored transcript keeps the token: register_database still opens it.
"""
from __future__ import annotations

import hashlib
import json
import secrets

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.planner as planner
import app.core.inference  # noqa: F401  (installs the model-side scrub, as in every process)

try:  # tests/ as a package, or tests/ on sys.path
    from tests.server_harness import client, make_auth_headers, _do_upload  # noqa: F401
except ImportError:  # pragma: no cover
    from server_harness import client, make_auth_headers, _do_upload  # noqa: F401

SYN = "ZZ-SYNTH-PW-1"
SEALED = "⟦sealed:"


@pytest.fixture
def sealing_key(monkeypatch):
    import app.api.cloud_connections as cc
    key = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)


def _seal(text: str) -> str:
    """The message as the API stores it: the typed password replaced by its token."""
    import app.core.typed_secrets as ts
    sealed, n = ts.seal_typed_secrets(text)
    assert n == 1 and SEALED in sealed
    return sealed


def _fragments(token_text: str):
    """Every 8-character run of each token's ciphertext and nonce."""
    import re
    out = set()
    for m in re.finditer(r"⟦sealed:[a-z_]+:([A-Za-z0-9+/=]+)\.([A-Za-z0-9+/=]+)⟧", token_text):
        for part in m.groups():
            out |= {part[i:i + 8] for i in range(len(part) - 7)}
    assert out
    return out


def _leaks(text: str, frags) -> list:
    return sorted(f for f in frags if f in text) + (["<whole token>"] if SEALED in text else []) \
        + (["<plaintext>"] if SYN in text else [])


class _Provider:
    """An OpenAI-compatible endpoint that records each request body it is sent."""

    def __init__(self, reply):
        self.bodies, self.reply = [], reply

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(json.dumps(body, ensure_ascii=False))
        return httpx.Response(200, json=self.reply(body))


def _completion(message: dict) -> dict:
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _chat_model(provider: _Provider):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="m", api_key="k", base_url="http://provider.test/v1", max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(provider)))


def test_a_chat_model_is_never_sent_a_sealed_token(sealing_key):
    text = _seal(f"register a postgres connection wh: host db port 5432 user u password {SYN}")
    frags = _fragments(text)
    provider = _Provider(lambda body: _completion({"role": "assistant", "content": "ok"}))
    model = _chat_model(provider)
    model.invoke([HumanMessage(content=text)])
    model.invoke(text)                                   # a bare string prompt
    model.generate([[AIMessage(content=f"earlier: {text}"), HumanMessage(content="and now?")]])
    assert len(provider.bodies) == 3
    for body in provider.bodies:
        assert not _leaks(body, frags), _leaks(body, frags)
        assert "⟦credential⟧" in body


def test_the_planner_registers_a_typed_password_without_any_model_seeing_its_token(sealing_key, monkeypatch):
    """End to end through plan_etl_job and a real chat model client."""
    import app.api.db_connections as dbc
    from app.api.cloud_connections import encrypt_secret
    stored = []

    async def register_and_save(user_id, body):
        stored.append(dict(body))
        return {"id": "row-1", "customer_id": body["customer_id"]}, encrypt_secret("ak_chat")

    monkeypatch.setattr(dbc, "register_and_save", register_and_save, raising=False)

    def reply(body):
        if body.get("tools"):   # the planner: it copies what it was shown into the password
            shown = body["messages"][-1]["content"]
            password = shown.split("password ", 1)[1].split()[0] if "password " in shown else "?"
            args = {"alias": "wh_chat", "db_type": "postgresql", "host": "db", "port": 5432,
                    "database": "w", "user": "u", "password": password}
            return _completion({"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "register_database", "arguments": json.dumps(args)}}]})
        return _completion({"role": "assistant", "content": "data_analysis"})

    provider = _Provider(reply)
    model = _chat_model(provider)
    monkeypatch.setattr(planner, "llm", model, raising=False)
    monkeypatch.setattr(planner, "_routing_llm", lambda: model, raising=False)
    for name in ("_seed_ray_fields", "_seed_infra_request_if_needed"):
        monkeypatch.setattr(planner, name, lambda *a, **k: None)
    sealed = _seal(f"register a postgres connection wh_chat: host db port 5432 database w user u password {SYN}")
    state = {"user_id": "user-1", "dta_database_registry": {}, "messages": [HumanMessage(content=sealed)]}
    state.update(planner.plan_etl_job(state))
    frags = _fragments(sealed)
    assert provider.bodies, "no model was called"
    for body in provider.bodies:
        assert not _leaks(body, frags), _leaks(body, frags)
    assert stored and stored[0]["password"] == SYN, "the sealed store must still get the password"
    replies = [m.content for m in state["messages"] if isinstance(m, AIMessage)]
    assert replies and not any(_leaks(r, frags) for r in replies)


def _post(client, text):
    upload = _do_upload(client, user_id="u-reply")
    resp = client.post(f"/threads/{upload['thread_id']}/messages",
                       json={"content": text, "metadata": {"dataset_id": upload["dataset_id"]}},
                       headers=make_auth_headers("u-reply"))
    assert resp.status_code == 200, resp.text
    history = client.get(f"/threads/{upload['thread_id']}/messages", headers=make_auth_headers("u-reply"))
    return resp.text, history.text


def test_a_reply_never_shows_a_sealed_token(client, sealing_key):
    """The harness graph echoes the message it was given -- the token, verbatim."""
    text = f"register a postgres connection wh: host db port 5432 user u password {SYN}"
    frags = _fragments(_seal(text))
    turn, history = _post(client, text)
    assert not _leaks(turn, frags) and not _leaks(history, frags)


def test_a_reply_never_shows_fragments_of_a_sealed_token(client, sealing_key, monkeypatch):
    """A guard that quotes 'unknown fields' from the user's text, as the live run showed."""
    import re
    import app.api.server as server
    real = server.GRAPH.invoke

    def quoting_graph(state_in, config=None):
        out = real(state_in, config)
        last = state_in["messages"][-1].content
        m = re.search(r":([A-Za-z0-9+/=]{12,})\.([A-Za-z0-9+/=]{12,})⟧", last)
        out["messages"][-1] = AIMessage(content=(
            "That request refers to fields this dataset doesn't contain. I couldn't map these "
            f"to the data: {m.group(1)[3:15]}, {m.group(2)[:12]}."))
        return out

    monkeypatch.setattr(server.GRAPH, "invoke", quoting_graph)
    text = f"register a postgres connection wh: host db port 5432 user u password {SYN}"
    turn, history = _post(client, text)
    assert "couldn't map these" in turn
    # Whatever the graph quoted, no 8-character run of base64 from the token survives.
    assert not re.search(r"to the data: [A-Za-z0-9+/=]{8,}", turn), turn
    assert not re.search(r"to the data: [A-Za-z0-9+/=]{8,}", history), history


def test_a_sealed_password_does_not_trip_the_fabrication_guard(sealing_key, monkeypatch):
    """#599 regression: the token's base64 read as camelCase "fields" this dataset lacks,
    so the deterministic guard refused a registration before any model saw it
    (129 of 300 seals at a8e7c1fa; the live run's reply quoted two fragments)."""
    monkeypatch.setattr(planner, "llm", None, raising=False)
    state = {"schema": {"id": "int", "age": "int", "stroke": "int"},
             "uploaded_csv_columns": ["id", "age", "stroke"]}
    text = "register a postgres connection chatwh: host postgres port 5432 database synthdst user synth password " + SYN
    refused = [r for r in (planner._detect_fabricated_reference(_seal(text), state) for _ in range(60)) if r]
    assert not refused, refused[0]["message"][:200]
