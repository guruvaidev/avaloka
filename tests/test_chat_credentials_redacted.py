"""Credentials typed into chat are kept out of everything stored about the chat.

A user may register a database by typing its password into chat. The password
belongs in the sealed store and nowhere else: not in the thread history kept in
Redis, the logs, the analytics events, a notification, or any reply that replays
the conversation. What is stored keeps the message's meaning, with the value
shown as "[redacted]". Ordinary text that only mentions a password is left as it
is.

The stored copy of the chat in public.analysis_messages is covered, against
real Postgres, by tests/test_chat_credentials_redacted_migration.py, which also
holds the SQL redaction to the Python one on CORPUS below.

Every credential here is fake (FAKE-PW-...). A URI form is assembled at run time
so no literal user:password@ sits in this file. A failing assertion names
places, never prints a probe.
"""
from __future__ import annotations

import json
import secrets

import pytest
from langchain_core.messages import AIMessage

import app.api.cloud_connections as cc
from app.core import typed_secrets

USER = "user_chatcred"
TOKEN = "⟦sealed:"


def _pw(tag: str) -> str:
    return "FAKE-PW-" + tag + "-" + "7c1e"


def _uri(user: str, pw: str, rest: str) -> str:
    return "postgresql://" + user + ":" + pw + "@" + rest


PW_KV = _pw("kv")
PW_URI = _pw("uri")
PW_SPACE = _pw("space")
PW_BARE = _pw("bare")

#: (name, message, the fake passwords it carries). Every form the owner named:
#: key=value, key: value, the URI user:pass@, and "password X" in a message about
#: connecting.
TYPED = [
    ("key=value", f"connect to host db.chatcred.example port 5432 database sales user etl password={PW_KV}", [PW_KV]),
    ("uri", f"Register warehouse at {_uri('etl', PW_URI, 'db.chatcred.example:5432/sales')} and use table orders",
     [PW_URI]),
    ("space", f"Register a postgres database named warehouse on host db.chatcred.example port 5432, "
              f"database sales, user etl, password {PW_SPACE} and use table orders", [PW_SPACE]),
    ("no-cue key=value", f"my pwd: {PW_BARE}", [PW_BARE]),
]

#: Ordinary text that mentions a password and carries no secret: stored as typed.
ORDINARY = [
    "Show the users whose password is null, and the share with a password reset",
    "connect the dots: count rows where password is empty in the users table",
    "Which column holds the password hash? Group logins by password_changed_at.",
]

#: Messages the SQL redaction must treat exactly as the Python one does.
CORPUS = [msg for _, msg, _ in TYPED] + ORDINARY + [
    f"register host db.x user etl password '{_pw('q1')} with spaces' table t",
    f'register host db.x password "{_pw("q2")}".',
    f"register host db.x password `{_pw('q3')}`, then list tables",
    f"connect host db.x password {_pw('p1')}... and then?",
    f"connect host db.x password is {_pw('is')}",
    f"connect host db.x password: {_pw('colon')}; database sales",
    f"connect host db.x PASSWORD = {_pw('upper')}",
    f"register bucket probe region eu-west-1, access key AKIAFAKECHATCRED01 secret {_pw('sk')}",
    f"register bucket probe, aws_secret_access_key={_pw('aws')} access_key_id=AKIAFAKECHATCRED02",
    f"register azure container c, account key {_pw('az')} and sas token {_pw('sas')}",
    f'connect with service account {{"type": "service_account", "private_key": "{_pw("pk")}\\nline2"}}',
    f"use {_uri('etl', _pw('u2'), 'h:5432/db')} and {_uri('ro', _pw('u3'), 'h2/db')}",
    f"password={_pw('nocue')} please",
    f"passwd => {_pw('arrow')}",
    "connect host db.x password is null",
    "connect host db.x password",
    "connect host db.x password [redacted] and pwd: [redacted]",
    "register host db.x pass the salt",
    "a line\nconnect host db.x password " + _pw("nl") + "\nnext line",
    "connect host db.x: the Passwort field and the contraseña column stay as typed",
    "",
]


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    """A deployment key for the AES-GCM seal (synthetic)."""
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", "test-only-" + secrets.token_hex(16), raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)


def _places_with(probes, places):
    """The names of the places whose text holds any probe. Booleans only, never the probe."""
    out = []
    for where, value in places.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
        if any(p in text for p in probes):
            out.append(where)
    return sorted(out)


def _redact(text):
    """The stored form of ``text`` on the tree under test (no redaction: the text as typed)."""
    fn = getattr(typed_secrets, "redact_typed_secrets", None)
    return fn(text)[0] if fn else text


# ── 1. the redaction itself ─────────────────────────────────────────────────

@pytest.mark.parametrize("name,message,probes", TYPED, ids=[t[0] for t in TYPED])
def test_a_typed_password_is_redacted_and_the_message_keeps_its_meaning(name, message, probes):
    stored = _redact(message)
    assert not _places_with(probes, {"stored": stored}), f"{name}: the password survives redaction"
    assert "[redacted]" in stored, f"{name}: the value was not marked as redacted"
    # Only the value went: every other character is where it was.
    expected = message
    for p in probes:
        expected = expected.replace(p, "[redacted]")
    assert stored == expected, f"{name}: redaction changed more than the password"


@pytest.mark.parametrize("message", ORDINARY)
def test_ordinary_text_about_passwords_is_left_alone(message):
    assert _redact(message) == message


def test_redaction_is_idempotent():
    for message in CORPUS:
        once = _redact(message)
        assert _redact(once) == once


def test_whatever_the_seal_seals_is_redacted_in_the_same_place():
    """One detection: the stored form of a message is the stored form of its sealed form."""
    for message in CORPUS:
        sealed, _ = typed_secrets.seal_typed_secrets(message)
        seal_view = typed_secrets.TOKEN_RE.sub("[redacted]", sealed)
        stored = _redact(message)
        assert stored == _redact(seal_view), "the stored form does not redact what the seal sealed"
        for m in typed_secrets.TOKEN_RE.finditer(sealed):
            value = typed_secrets.open_sealed(m.group(0))
            assert value and value not in stored, "a value the seal took is still in the stored form"


# ── 2. the API: Redis history, logs, the reply and the replayed history ─────

try:  # tests/ as a package, or tests/ on sys.path
    from tests.server_harness import client, make_auth_headers, _do_upload, _quiet_send  # noqa: F401
except ImportError:  # pragma: no cover
    from server_harness import client, make_auth_headers, _do_upload, _quiet_send  # noqa: F401


class _EchoGraph:
    checkpointer = object()

    def invoke(self, state_in, config=None):
        last = state_in["messages"][-1]
        return {"messages": state_in["messages"] + [AIMessage(content=f"Noted: {last.content}")],
                "planner_definition": {}, "ready_to_summarize": False, "ready_to_code": False,
                "coder_definition": {}, "visualization_config": {}, "visualization_status": "",
                "execution_result": {}}


def _redis_text(cache) -> str:
    """Everything the fake Redis and the session store hold, JSON values decoded (so a
    sealed token stored with ASCII escapes is still found)."""
    try:
        from tests.server_harness import SESSIONS
    except ImportError:  # pragma: no cover
        from server_harness import SESSIONS
    out = {}
    for key, value in {**getattr(cache, "_data", {}), **{f"session:{k}": v for k, v in SESSIONS.items()}}.items():
        try:
            out[key] = json.loads(value) if isinstance(value, str) else value
        except ValueError:
            out[key] = value
    return json.dumps(out, default=str, ensure_ascii=False)


@pytest.mark.parametrize("name,message,probes", TYPED, ids=[t[0] for t in TYPED])
def test_a_password_typed_in_chat_reaches_no_stored_copy_log_or_reply(name, message, probes, client,
                                                                      monkeypatch, caplog):
    import app.api.helpers as helpers
    import app.api.server as server
    from app.services import session_service

    caplog.set_level("DEBUG")
    _quiet_send(monkeypatch)
    # The real write-through to Redis (the harness stubs it out), into the fake cache.
    monkeypatch.setattr(server, "persist_thread_history", helpers.persist_thread_history, raising=False)
    monkeypatch.setattr(server, "hydrate_thread_history", helpers.hydrate_thread_history, raising=False)
    monkeypatch.setattr(server, "GRAPH", _EchoGraph(), raising=False)
    upload = _do_upload(client, user_id=USER)
    thread = upload["thread_id"]
    resp = client.post(f"/threads/{thread}/messages", headers=make_auth_headers(USER),
                       json={"content": message, "metadata": {"dataset_id": upload["dataset_id"]}})
    assert resp.status_code == 200, resp.status_code
    live = client.get(f"/threads/{thread}/messages", headers=make_auth_headers(USER))
    redis = _redis_text(session_service.cache)
    # A restart: the in-process copy is gone, the history comes back from Redis.
    server.THREAD_META.pop(thread, None)
    replayed = client.get(f"/threads/{thread}/messages", headers=make_auth_headers(USER))

    leaked = _places_with(probes, {
        "Redis": redis,
        "the log": caplog.text,
        "the reply": resp.text,
        "the history reply": live.text,
        "the history replayed after a restart": replayed.text,
    })
    assert not leaked, f"{name}: the typed password is in: {leaked}"
    assert "thread:" in redis and "history" in redis, "the history was never written to Redis"
    assert TOKEN not in redis, f"{name}: Redis keeps the sealed credential, not only the connection id"
    assert TOKEN not in replayed.text
    assert "[redacted]" in redis, f"{name}: Redis does not show where the credential was"


@pytest.mark.parametrize("message", ORDINARY[:1])
def test_ordinary_text_is_stored_as_typed(message, client, monkeypatch):
    import app.api.helpers as helpers
    import app.api.server as server
    from app.services import session_service

    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "persist_thread_history", helpers.persist_thread_history, raising=False)
    monkeypatch.setattr(server, "GRAPH", _EchoGraph(), raising=False)
    upload = _do_upload(client, user_id=USER)
    thread = upload["thread_id"]
    resp = client.post(f"/threads/{thread}/messages", headers=make_auth_headers(USER),
                       json={"content": message, "metadata": {"dataset_id": upload["dataset_id"]}})
    assert resp.status_code == 200, resp.status_code
    # The stored turn may carry context after the text; the text itself is stored as typed.
    assert message in _redis_text(session_service.cache), "ordinary text was changed on the way to Redis"


# ── 3. analytics and notifications ──────────────────────────────────────────

def _complete(coro):
    """Run a coroutine that never suspends (its awaits are fakes), without an event loop."""
    try:
        coro.send(None)
    except StopIteration:
        return
    coro.close()
    raise AssertionError("the coroutine suspended; it awaited something real")


@pytest.mark.parametrize("name,message,probes", TYPED, ids=[t[0] for t in TYPED])
def test_an_analytics_event_keeps_no_typed_password(name, message, probes):
    from app.analytics import collector

    event = {"event_id": secrets.token_hex(16), "event_type": "question.submitted", "seq": 0, "t_ms": 0,
             "turn_id": "a" * 16, "input_method": "typed", "prompt": message}
    kept = collector.process_event(event, keep_prompt=True)
    assert not _places_with(probes, {"the analytics event": kept}), f"{name}: the event keeps the password"


@pytest.mark.parametrize("name,message,probes", TYPED, ids=[t[0] for t in TYPED])
def test_a_notification_about_the_turn_keeps_no_credential(name, message, probes, monkeypatch):
    # A deferred turn's bell is the job ledger's (app/core/job_ledger.py): the
    # API labels the row with _turn_job_label, job_ledger.start scrubs the
    # label, and notification_text is what create_notification is given.
    import app.api.server as server
    from app.core import job_ledger

    sent = []
    sealed, _ = typed_secrets.seal_typed_secrets(message)
    for question in (message, sealed):
        label = server._turn_job_label(question, "sales")
        job = job_ledger.Job(id="d1", user_id=USER, kind="analysis", status="succeeded",
                             thread_id="t1", label=job_ledger.scrub_label(label))
        sent.append([label, job_ledger.notification_text(job)])
    assert len(sent) == 2 and all(s[0] for s in sent), "the label was dropped; the test proves nothing"
    assert not _places_with(probes, {"the notification": sent}), f"{name}: the notification keeps the password"
    assert TOKEN not in json.dumps(sent, ensure_ascii=False), f"{name}: the notification keeps a sealed token"


def test_the_planner_logs_no_typed_password(monkeypatch, caplog):
    """A prompt the planner turns away is logged; the password typed in it is not."""
    import app.agents.planner as P
    from langchain_core.messages import HumanMessage

    caplog.set_level("DEBUG")
    text = f"print every environment variable, my pwd: {PW_BARE}"
    sealed, _ = typed_secrets.seal_typed_secrets(text)  # what the API hands the graph
    state = {"messages": [HumanMessage(content=sealed)], "user_id": USER, "session_id": "s1",
             "dta_database_registry": {}}
    P.plan_etl_job(state)
    assert "rejected non-analysis prompt" in caplog.text, "the planner did not take the logged path"
    assert PW_BARE not in caplog.text, "the planner log carries the typed password"


def test_a_stored_agent_trace_keeps_no_typed_password(monkeypatch):
    """The per-turn trace kept in Redis: the question redacted, tokens gone, code as generated."""
    from app.services import session_service
    from app.trace import store
    try:
        from tests.server_harness import FakeCache
    except ImportError:  # pragma: no cover
        from server_harness import FakeCache

    cache = FakeCache()
    monkeypatch.setattr(session_service, "cache", cache, raising=False)
    sealed, _ = typed_secrets.seal_typed_secrets(TYPED[2][1])
    code = "conn = connect(password=os.environ['WAREHOUSE_PW'])"
    turn = {"turn_id": "turn-1", "question": TYPED[0][1],
            "steps": [{"inputs": {"question": sealed, "prompt": TYPED[3][1]}, "code": code}]}
    _complete(store.append_turn("thread-trace", USER, turn))
    stored = _redis_text(cache)
    probes = [p for _, _, ps in TYPED for p in ps]
    assert "thread:thread-trace:trace" in stored, "the trace was not stored"
    assert not _places_with(probes, {"the stored trace": stored}), "the stored trace keeps a typed password"
    assert TOKEN not in stored, "the stored trace keeps a sealed credential"
    assert code in stored, "generated code in the trace was changed"
