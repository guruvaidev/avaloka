"""A database typed into chat: its password lives in the sealed store and nowhere else.

1.6 port of 2.0's #599 chat sealing (app/core/typed_secrets.py, the planner's
register_database, app/api/db_connections.register_and_save). Every probe here
is synthetic; a failing assertion names places, never prints the probe.

The tests reach the code only through what 1.6 already had (the planner node,
the API server, the Redis history reader, httpx), so on a tree without the fix
they fail on their assertions, not on an import.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sys
import types
from types import SimpleNamespace
from urllib.parse import quote_plus, urlsplit

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.planner as P
import app.api.cloud_connections as cc

USER = "user_chat"
# Synthetic, and not a real secret: built so no literal DSN with a password sits in the file.
PROBE_PW = "PROBE-chat-pw-" + "4be1d09a"
AWS_AK = "AKIAPROBECHAT0001"
AWS_SK = "probe/chat+secret=0001k"
TOKEN = "⟦sealed:"


def _places_with(probes, places):
    """The names of the places whose text holds any probe (raw or URL-encoded). Booleans only."""
    out = []
    for where, text in places.items():
        text = text if isinstance(text, str) else json.dumps(text, default=str, ensure_ascii=False)
        if any(p in text or quote_plus(p) in text for p in probes):
            out.append(where)
    return sorted(out)


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    """A deployment key for the AES-GCM seal (synthetic), and the service token."""
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", "test-only-" + secrets.token_hex(16), raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    monkeypatch.setenv("MCP_SERVICE_TOKEN", "svc-" + secrets.token_hex(8))
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setenv("DTA_GKE_REACHES_PRIVATE", "1")


# ── fakes: the planner's model, and the onboarding API over httpx ───────────

def _typed_value(text, word):
    m = re.search(r"\b" + word + r"\s+(\S+?)(?=[.,;]?(?:\s|$))", text or "")
    return m.group(1) if m else None


class _Model:
    """The planner model: picks the tool the script says, copying what it was shown.

    A real model copies a password it can read into the tool call; shown the
    sealed placeholder (or a token) it can only copy that. So does this one.
    """

    def __init__(self, tool, args_from_text):
        self.tool, self.args_from_text, self.seen = tool, args_from_text, []

    def invoke(self, msgs, *a, **k):
        self.seen.append("\n".join(str(getattr(m, "content", m)) for m in (msgs or [])))
        last = next((m for m in reversed(msgs or []) if isinstance(m, HumanMessage)), None)
        text = last.content if last is not None and isinstance(last.content, str) else ""
        return SimpleNamespace(tool_calls=[{"name": self.tool, "args": self.args_from_text(text)}],
                               content="")

    def bind(self, **_k):
        return self


def _register_args(text):
    pw = _typed_value(text, "password")
    if pw and pw.startswith("⟦"):
        pw = "⟦credential⟧"
    return {"alias": "warehouse", "db_type": "postgresql", "host": "db.chat.example", "port": 5432,
            "database": "sales", "user": "etl", "password": pw, "table": "orders"}


class _Onboarding:
    """The onboarding API's /customers routes, reached over httpx (register) and requests (read)."""

    def __init__(self, register_status=200, register_detail=None):
        self.register_status, self.register_detail = register_status, register_detail
        self.registered = {}      # customer_id -> the request body (the sealed store's copy)
        self.bodies = []

    def install(self, monkeypatch):
        store = self

        class _Resp:
            def __init__(self, code, body):
                self.status_code, self._body = code, body

            def json(self):
                return self._body

        class _AsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def request(self, method, url, content=None, headers=None, **_k):
                body = json.loads(content or b"{}")
                store.bodies.append(body)
                if not (headers or {}).get("X-Avaloka-User"):
                    return _Resp(401, {"detail": "Authentication required"})
                if store.register_status != 200:
                    return _Resp(store.register_status, {"detail": store.register_detail})
                store.registered[body["customer_id"]] = {**body, "owner": headers["X-Avaloka-User"]}
                return _Resp(200, {"customer_id": body["customer_id"], "api_key": "ak_" + secrets.token_hex(8),
                                   "status": "active", "database_type": body["database_type"]})

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, headers=None, **_k):
                parts = url[len(P.ONBOARDING_API_URL):].strip("/").split("/")
                if len(parts) == 3 and parts[0] == "customers" and parts[2] == "credentials":
                    rec = store.registered.get(parts[1])
                    if rec and (headers or {}).get("X-Avaloka-User") == rec["owner"]:
                        return _Resp(200, {"host": rec["host"], "port": rec["port"],
                                           "database": rec["database_name"], "username": rec["username"],
                                           "password": rec["password"], "table": "first_table",
                                           "database_type": rec["database_type"]})
                    return _Resp(404, {})
                if parts == ["customers"]:
                    return _Resp(200, [])
                return _Resp(404, {})

        monkeypatch.setattr(httpx, "AsyncClient", _AsyncClient)
        monkeypatch.setattr(P.requests, "Session", _Session)
        return self


def _api_seal(text):
    """What the API server does to a chat message before the graph sees it (send_message).

    Read from the tree under test; a tree with no seal passes the text on as typed.
    """
    try:
        from app.core.typed_secrets import seal_typed_secrets
    except ImportError:
        return text
    return seal_typed_secrets(text)[0]


def _register_turn(monkeypatch, text, seal=True):
    model = _Model("register_database", _register_args)
    monkeypatch.setattr(P, "llm", model)
    state = {"messages": [HumanMessage(content=_api_seal(text) if seal else text)],
             "user_id": USER, "session_id": "s1", "dta_database_registry": {}}
    out = P.plan_etl_job(state)
    return out if isinstance(out, dict) else state, model


def _reply(state):
    ai = [m for m in state.get("messages") or [] if isinstance(m, AIMessage)]
    return ai[-1].content if ai else ""


REGISTER_TEXT = (f"Register a postgres database named warehouse on host db.chat.example port 5432, "
                 f"database sales, user etl, password {PROBE_PW} and use table orders")


# ── 1. the registry and the transcript ──────────────────────────────────────

def test_a_chat_typed_password_goes_to_the_sealed_store_only(monkeypatch):
    onboarding = _Onboarding().install(monkeypatch)
    state, model = _register_turn(monkeypatch, REGISTER_TEXT)
    leaked = _places_with([PROBE_PW], {
        "the stored transcript (messages)": [getattr(m, "content", m) for m in state["messages"]],
        "the registry": state.get("dta_database_registry"),
        "the reply": _reply(state),
        "what the planner model was sent": model.seen,
    })
    assert not leaked, f"the chat-typed password is in: {leaked}"
    entry = (state.get("dta_database_registry") or {}).get("warehouse") or {}
    assert entry.get("_source") == "chat" and entry.get("conn_id"), \
        f"the registry entry is not a handle: fields {sorted(entry)}"
    stored = onboarding.registered.get(entry["conn_id"]) or {}
    assert stored.get("password") == PROBE_PW, "the sealed store did not receive the typed password"
    assert stored.get("owner") == USER, "the registration is not bound to the signed-in user"
    assert entry.get("host") == "db.chat.example" and entry.get("table") == "orders"


def test_the_users_own_message_is_sealed_in_state_even_without_the_api(monkeypatch):
    """A turn that reached the graph without the API server is sealed by the planner, in place."""
    _Onboarding().install(monkeypatch)
    state, _model = _register_turn(monkeypatch, REGISTER_TEXT, seal=False)
    human = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    assert human and not _places_with([PROBE_PW], {"m": human}), \
        "the user's message still holds the typed password in graph state"
    assert TOKEN in human[0], "the user's message was not sealed"


def test_a_refused_registration_stores_nothing_and_never_echoes_the_password(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    _Onboarding(register_status=400,
                register_detail=f"Database connection failed: login for etl with {PROBE_PW} refused").install(monkeypatch)
    state, _model = _register_turn(monkeypatch, REGISTER_TEXT)
    assert not (state.get("dta_database_registry") or {}), "a refused registration was kept in the registry"
    leaked = _places_with([PROBE_PW], {"the reply": _reply(state), "the log": caplog.text,
                                       "the transcript": [m.content for m in state["messages"]]})
    assert not leaked, f"the refused password is in: {leaked}"


def test_a_transfer_opens_the_chat_handle_at_run_time(monkeypatch):
    onboarding = _Onboarding().install(monkeypatch)
    state, _model = _register_turn(monkeypatch, REGISTER_TEXT)
    assert onboarding.registered, "nothing reached the sealed store"
    handed = {}

    def _pipeline(**kw):
        handed.update(kw)
        return {"error_message": None, "warnings": None, "runtime_secret_env": {}}, "# copy"

    fake_dta = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")
    fake_dta.data_transfer_pipeline = _pipeline
    fake_gke = types.ModuleType("app.data_transfer_docker.gke_run")
    fake_gke.launch_gke_pipeline = lambda script, **kw: SimpleNamespace(rows_inserted=3, table="orders_copy")
    monkeypatch.setitem(sys.modules, fake_dta.__name__, fake_dta)
    monkeypatch.setitem(sys.modules, fake_gke.__name__, fake_gke)
    monkeypatch.setattr(P, "_list_dest_tables", lambda *a, **k: ["orders_copy"])
    state["dta_database_registry"]["sink"] = {"db_type": "postgresql", "conn_id": "warehouse_sink",
                                              "host": "db.chat.example", "database": "sink", "_source": "chat"}
    onboarding.registered["warehouse_sink"] = {"host": "db.chat.example", "port": 5432, "database_name": "sink",
                                               "username": "etl", "password": "sink-pw-" + secrets.token_hex(4),
                                               "database_type": "postgresql", "owner": USER}
    msg = P._handle_initiate_transfer(state, P.InitiateTransferParams(
        source_alias="warehouse", destination_alias="sink", user_prompt="copy orders from warehouse to sink",
        write_mode="append", dest_table="orders_copy"))
    src = handed.get("source_credentials") or {}
    assert src.get("password") == PROBE_PW, f"the handle was not opened for the transfer: {msg[:120]}"
    assert src.get("table") == "orders", "the chat registration's table was lost"
    assert not _places_with([PROBE_PW], {"graph state": state.get("dta_database_registry"), "reply": msg}), \
        "the opened password went back into state or the reply"


def test_a_chat_typed_bucket_keeps_its_keys_sealed_in_cloud_datasets(monkeypatch):
    inserted = []

    class _Table:
        def insert(self, row):
            inserted.append(row)
            return self

        def execute(self):
            return SimpleNamespace(data=[{"id": "7d1c2a3b-4d5e-4f60-8a7b-9c0d1e2f3a4b"}])

    import app.agents.sampling_persistence as sp
    monkeypatch.setattr(sp, "get_supabase_client", lambda: SimpleNamespace(table=lambda name: _Table()))
    monkeypatch.setenv("SUPABASE_URL", "http://supabase.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-only-role")

    def _args(text):
        ak, sk = _typed_value(text, "key"), _typed_value(text, "secret")
        return {"alias": "exports", "db_type": "aws_s3", "bucket_name": "probe-bucket", "region": "eu-west-1",
                "access_key": "⟦credential⟧" if (ak or "").startswith("⟦") else ak,
                "secret_key": "⟦credential⟧" if (sk or "").startswith("⟦") else sk}

    monkeypatch.setattr(P, "llm", _Model("register_database", _args))
    text = f"Register bucket probe-bucket region eu-west-1, access key {AWS_AK} secret {AWS_SK}"
    state = {"messages": [HumanMessage(content=_api_seal(text))], "user_id": USER, "session_id": "s1",
             "dta_database_registry": {}}
    out = P.plan_etl_job(state)
    state = out if isinstance(out, dict) else state
    leaked = _places_with([AWS_AK, AWS_SK], {"the registry": state.get("dta_database_registry"),
                                             "the transcript": [m.content for m in state["messages"]],
                                             "the cloud_datasets row": inserted})
    assert not leaked, f"the bucket keys are in: {leaked}"
    row = inserted[0] if inserted else {}
    assert cc.decrypt_secret({"ciphertext": row.get("secret_key_ciphertext"), "iv": row.get("secret_key_iv")}) == AWS_SK
    assert (state["dta_database_registry"].get("exports") or {}).get("conn_id")


# ── 2. checkpoints and history written before sealing ───────────────────────

def test_an_older_checkpoint_holding_a_chat_password_is_scrubbed_on_read(monkeypatch):
    """Registry entry and transcript as a pre-sealing checkpoint left them; one planner read later, gone."""
    _Onboarding().install(monkeypatch)
    monkeypatch.setattr(P, "llm", _Model("list_databases", lambda text: {}))
    state = {
        "user_id": USER, "session_id": "s1",
        "dta_database_registry": {"warehouse": {
            "db_type": "postgresql", "host": "db.chat.example", "port": 5432, "database": "sales",
            "user": "etl", "password": PROBE_PW, "table": "orders"}},
        "messages": [
            HumanMessage(content=REGISTER_TEXT),
            AIMessage(content="Registered warehouse as postgresql."),
            HumanMessage(content=f"connect to host db.chat.example user etl password {PROBE_PW}"),
            HumanMessage(content="list databases"),
        ],
    }
    out = P.plan_etl_job(state)
    state = out if isinstance(out, dict) else state
    leaked = _places_with([PROBE_PW], {
        "the registry": state.get("dta_database_registry"),
        "the transcript": [getattr(m, "content", m) for m in state["messages"]],
        "the reply": _reply(state),
    })
    assert not leaked, f"an older checkpoint's chat password survives a read, in: {leaked}"
    assert "warehouse" in state["dta_database_registry"], "the scrub dropped the whole entry"


def test_redis_history_written_before_sealing_is_sealed_as_it_is_read():
    from app.api import helpers
    msgs = helpers._jsonable_to_lc_msgs([
        {"role": "human", "content": REGISTER_TEXT},
        {"role": "ai", "content": "Registered warehouse."},
    ])
    assert not _places_with([PROBE_PW], {"history": [m.content for m in msgs]}), \
        "thread history read back from Redis still holds the typed password"
    assert msgs[1].content == "Registered warehouse."


# ── 3. the API server: the seal on the way in, the redaction on the way out ─

try:  # tests/ as a package, or tests/ on sys.path
    from tests.server_harness import client, make_auth_headers, _do_upload, _quiet_send  # noqa: F401
except ImportError:  # pragma: no cover
    from server_harness import client, make_auth_headers, _do_upload, _quiet_send  # noqa: F401


def test_the_api_seals_the_message_before_the_graph_and_redacts_every_reply(client, monkeypatch):
    import app.api.server as server
    _quiet_send(monkeypatch)
    seen = []

    class _EchoGraph:
        checkpointer = object()

        def invoke(self, state_in, config=None):
            seen.append(json.dumps([getattr(m, "content", m) for m in state_in["messages"]], default=str,
                                   ensure_ascii=False))
            last = state_in["messages"][-1]
            return {"messages": state_in["messages"] + [AIMessage(content=f"You said: {last.content}")],
                    "planner_definition": {}, "ready_to_summarize": False, "ready_to_code": False,
                    "coder_definition": {}, "visualization_config": {}, "visualization_status": "",
                    "execution_result": {}}

    monkeypatch.setattr(server, "GRAPH", _EchoGraph(), raising=False)
    upload = _do_upload(client, user_id=USER)
    thread = upload["thread_id"]
    resp = client.post(f"/threads/{thread}/messages", headers=make_auth_headers(USER),
                       json={"content": REGISTER_TEXT, "metadata": {"dataset_id": upload["dataset_id"]}})
    assert resp.status_code == 200, resp.status_code
    history = client.get(f"/threads/{thread}/messages", headers=make_auth_headers(USER))
    leaked = _places_with([PROBE_PW], {
        "what the graph was handed": seen,
        "the reply": resp.text,
        "the history reply": history.text,
        "the in-process transcript": [m.content for m in server.THREAD_META[thread]["lc_msgs"]],
    })
    assert not leaked, f"the typed password is in: {leaked}"
    assert seen and TOKEN in seen[0], "the graph was not handed the sealed token"
    assert TOKEN not in resp.text and TOKEN not in history.text, "a reply carries a sealed token"


# ── 4. end to end: real Postgres, the real onboarding router, a full transfer ─
#
# Through the real API server (send_message, the reply middleware, the history
# route), a graph whose planner node is the real plan_etl_job, the real
# onboarding router (/customers/register tests the login against Postgres and
# keeps the connection; /credentials releases it to the service), and the
# #635 transfer path in KubeRay mode with a pretend cluster that really runs
# the copy. Only code generation and the model are replaced.
#
# 1.6 ships no Postgres checkpointer (the API's is a MemorySaver), so the
# graph's checkpointer here writes every checkpoint and every pending write it
# is given, serialized by its own serde exactly as a durable saver stores them,
# into Postgres tables -- and the scan reads those tables back with SQL.

try:
    from tests.test_dta_credentials_out_of_rayjob import COPY_SCRIPT, ROWS, _FakeCluster, _engine, pg  # noqa: F401
except ImportError:  # pragma: no cover
    from test_dta_credentials_out_of_rayjob import COPY_SCRIPT, ROWS, _FakeCluster, _engine, pg  # noqa: F401


def _pg_checkpointer(admin_url, table):
    from langgraph.checkpoint.memory import MemorySaver
    from sqlalchemy import text as sql

    eng = _engine(admin_url)
    with eng.connect() as conn:
        conn.execute(sql(f'CREATE TABLE IF NOT EXISTS "{table}" (id bigserial PRIMARY KEY, thread text, '
                         f'kind text, type text, blob bytea)'))

    class _PgMirroredSaver(MemorySaver):
        def _store(self, config, kind, obj):
            typ, blob = self.serde.dumps_typed(obj)
            with eng.connect() as conn:
                conn.execute(sql(f'INSERT INTO "{table}" (thread, kind, type, blob) '
                                 f'VALUES (:thread, :kind, :type, :blob)'),
                             {"thread": str(config["configurable"].get("thread_id")), "kind": kind,
                              "type": typ, "blob": blob})

        def put(self, config, checkpoint, metadata, new_versions):
            self._store(config, "checkpoint", {"checkpoint": checkpoint, "metadata": metadata})
            return super().put(config, checkpoint, metadata, new_versions)

        def put_writes(self, config, writes, task_id, task_path=""):
            self._store(config, "writes", list(writes))
            return super().put_writes(config, writes, task_id, task_path)

    return _PgMirroredSaver(), eng


def _checkpoint_rows(eng, table):
    from sqlalchemy import text as sql
    try:
        with eng.connect() as conn:
            rows = conn.execute(sql(f'SELECT kind, blob FROM "{table}" ORDER BY id')).all()
            conn.execute(sql(f'DROP TABLE IF EXISTS "{table}"'))
    finally:
        eng.dispose()
    return [(kind, bytes(blob)) for kind, blob in rows]


class _ChatModel:
    """The planner model for the e2e: one scripted tool per turn, copying what it was shown."""

    def __init__(self):
        self.seen = []

    def invoke(self, msgs, *a, **k):
        text_in = "\n".join(str(getattr(m, "content", m)) for m in (msgs or []))
        self.seen.append(text_in)
        last = next((m for m in reversed(msgs or []) if isinstance(m, HumanMessage)), None)
        text = last.content if last is not None and isinstance(last.content, str) else ""
        if text.lower().startswith("transfer"):
            return SimpleNamespace(content="", tool_calls=[{"name": "initiate_transfer", "args": {
                "source_alias": "chatsrc", "destination_alias": "chatdst", "user_prompt": text,
                "write_mode": "append", "dest_table": "orders_copy"}}])
        alias = re.search(r"named (\w+)", text).group(1)
        pw = _typed_value(text, "password")
        if pw and pw.startswith("⟦"):
            pw = "⟦credential⟧"
        return SimpleNamespace(content="", tool_calls=[{"name": "register_database", "args": {
            "alias": alias, "db_type": "postgresql", "host": _typed_value(text, "host"),
            "port": int(_typed_value(text, "port")), "database": _typed_value(text, "database"),
            "user": _typed_value(text, "user"), "password": pw,
            "table": _typed_value(text, "table")}}])

    def bind(self, **_k):
        return self


@pytest.mark.parametrize("sealing", ["on", "off (positive control)"])
def test_real_postgres_chat_connection_transfer_and_probe_scan(pg, client, monkeypatch, caplog, capsys,
                                                              tmp_path, sealing):
    import app.api.server as server
    import app.data_transfer_docker.gke_run as gk
    import app.infra.ray_job_runner as rjr
    import app.mcp_server.customer_dbs as cdb
    import app.mcp_server.multi_tenant_mcp_server as mts
    import subprocess
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from langgraph.graph import END, START, StateGraph
    from sqlalchemy import text as sql
    from app.graph.etl_state import ETLState

    caplog.set_level("DEBUG")
    _quiet_send(monkeypatch)
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://ray.invalid:8265")
    monkeypatch.delenv("RAY_DASHBOARD_URL_INCLUSTER", raising=False)
    monkeypatch.delenv("RAYJOB_DRYRUN", raising=False)
    monkeypatch.setenv("KUBE_NAMESPACE", "dta-jobs")
    monkeypatch.setenv("ONBOARDING_API_URL", P.ONBOARDING_API_URL)
    if sealing != "on":
        try:  # a tree without the seal has nothing to switch off
            import app.core.typed_secrets as ts
            monkeypatch.setattr(ts, "seal_typed_secrets", lambda t: (t, 0))
            monkeypatch.setattr(server, "seal_typed_secrets", lambda t: (t, 0), raising=False)
        except ImportError:
            pass
        try:  # ...and the seal at rest and in transit
            import app.mcp_server.credentials as cr
            monkeypatch.setattr(cr, "seal_record", lambda rec: dict(rec))
            monkeypatch.setattr(cdb, "credential_handle", lambda v: {"scheme": cr.HANDLE_SCHEME, "plain": v},
                                raising=False)
            real_open = cr.open_credential_handle
            monkeypatch.setattr(mts, "open_credential_handle",
                                lambda h: h.get("plain") if isinstance(h, dict) and "plain" in h else real_open(h),
                                raising=False)
        except ImportError:
            pass

    e = _engine(pg.dbs["dst"].url)
    with e.connect() as conn:
        conn.execute(sql("TRUNCATE orders_copy"))
    e.dispose()

    # The onboarding store, behind its real router; MCP's admin route recorded.
    store = tmp_path / "customers.json"
    store.write_text("{}")
    mgr = cdb.customer_manager
    monkeypatch.setattr(mgr, "customers_file", str(store))
    monkeypatch.setattr(mgr, "customers", {})
    # The real MCP server shares the registry file, as in the chart's MCP pod.
    monkeypatch.setattr(mts, "CUSTOMERS_FILE", str(store), raising=False)
    monkeypatch.setattr(mts, "_registry_cache", {"stat": None, "data": {}}, raising=False)
    onboarding = FastAPI()
    onboarding.include_router(cdb.router)
    reads = TestClient(onboarding)
    mcp_payloads = []
    onboarding_host = urlsplit(P.ONBOARDING_API_URL).netloc
    mcp_host = urlsplit(mgr.mcp_server_url).netloc

    class _Routes(httpx.AsyncBaseTransport):
        def __init__(self):
            self.asgi = httpx.ASGITransport(app=onboarding)
            self.mcp = httpx.ASGITransport(app=mts.app)

        async def handle_async_request(self, request):
            host = request.url.netloc.decode()
            if host == onboarding_host:
                return await self.asgi.handle_async_request(request)
            if host == mcp_host:
                body = await request.aread()
                mcp_payloads.append(body.decode("utf-8", "replace"))
                return await self.mcp.handle_async_request(request)
            return httpx.Response(404, json={})

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_async_client(*a, transport=_Routes(), **k))

    class _RoutedSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None, **_kw):
            assert url.startswith(P.ONBOARDING_API_URL)
            return reads.get(url[len(P.ONBOARDING_API_URL):], headers=headers or {})

    monkeypatch.setattr(P.requests, "Session", _RoutedSession)

    # Code generation is replaced; the planner's env merge, launch_gke_pipeline,
    # the runner and the job Secret are real, against a cluster that runs the copy.
    def _pipeline(**kw):
        def url(c):
            return f"postgresql://{c['user']}:{quote_plus(c['password'])}@{c['host']}:{c['port']}/{c['database']}"
        src, dst = kw["source_credentials"], kw["destination_credentials"]
        return ({"error_message": None, "warnings": None,
                 "runtime_secret_env": {"DTA_SOURCE_CONN_STR": url(src), "DTA_DEST_CONN_STR": url(dst)}},
                COPY_SCRIPT.format(src_table=src["table"], dst_table=dst["table"]))

    fake_dta = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")
    fake_dta.data_transfer_pipeline = _pipeline
    monkeypatch.setitem(sys.modules, fake_dta.__name__, fake_dta)
    monkeypatch.setitem(sys.modules, "app.data_transfer_docker.gke_run", gk)
    cluster = _FakeCluster(subprocess.run)
    monkeypatch.setattr(subprocess, "run", cluster)
    monkeypatch.setattr(rjr, "_which_kubectl", lambda: "kubectl", raising=False)
    real_runner, seen_job = rjr.run_rayjob_from_yaml, {}

    def _runner(**kw):
        seen_job.update(kw)
        monkeypatch.delenv("RAY_DASHBOARD_URL", raising=False)
        return real_runner(**{**kw, "dashboard_url": None})

    monkeypatch.setattr(gk, "run_rayjob_from_yaml", _runner)

    # The graph: the real planner node, checkpointed into Postgres.
    table = f"e2e_ckpt_{secrets.token_hex(4)}"
    saver, ckpt_engine = _pg_checkpointer(os.environ["AVALOKA_TEST_PG_ADMIN_URL"], table)
    builder = StateGraph(ETLState)
    builder.add_node("planner", P.plan_etl_job)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", END)
    graph = builder.compile(checkpointer=saver)
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)
    model = _ChatModel()
    monkeypatch.setattr(P, "llm", model)

    upload = _do_upload(client, user_id=USER)
    thread, dsid = upload["thread_id"], upload["dataset_id"]
    src, dst = pg.dbs["src"], pg.dbs["dst"]
    turns = [
        f"Register a postgres connection named chatsrc on host {pg.host} port {pg.port} database {src.db} "
        f"user {src.role} password {src.pw} table orders",
        f"Register a postgres connection named chatdst on host {pg.host} port {pg.port} database {dst.db} "
        f"user {dst.role} password {dst.pw} table orders_copy",
        "transfer from chatsrc to chatdst into table orders_copy",
    ]
    replies = []
    try:
        for t in turns:
            r = client.post(f"/threads/{thread}/messages", headers=make_auth_headers(USER),
                            json={"content": t, "metadata": {"dataset_id": dsid}})
            assert r.status_code == 200, r.status_code
            replies.append(r.text)
        history = client.get(f"/threads/{thread}/messages", headers=make_auth_headers(USER)).text
        state = graph.get_state({"configurable": {"thread_id": f"ui:{thread}"}}).values
        store_text = store.read_text()
        try:  # what the store holds, opened with the deployment key
            from app.mcp_server.credentials import load_registry
            store_opened = json.dumps(load_registry(str(store), migrate=False))
        except ImportError:
            store_opened = store_text
    finally:
        rows = _checkpoint_rows(ckpt_engine, table)
        store.unlink(missing_ok=True)
        for cid in list(getattr(mts, "CUSTOMER_API_KEYS", {})):
            mts._evict_customer(cid)

    e = _engine(dst.url)
    with e.connect() as conn:
        copied = conn.execute(sql('SELECT count(*) FROM "orders_copy"')).scalar_one()
    e.dispose()
    def _safe(text):
        for p in (src.pw, dst.pw):
            text = text.replace(p, "***").replace(quote_plus(p), "***")
        return text[:1500]

    assert copied == ROWS, f"copied {copied} of {ROWS} rows; replies {[_safe(r) for r in replies]}"

    probes = [src.pw, dst.pw]
    out = capsys.readouterr()
    registry = state.get("dta_database_registry") or {}
    places = {
        "the stored transcript (graph messages)": [getattr(m, "content", m) for m in state.get("messages") or []],
        "the checkpoint tables": [blob.decode("utf-8", "replace") for _k, blob in rows],
        "graph state": json.dumps(state, default=str),
        "the registry": registry,
        "the API's transcript (THREAD_META)": [m.content for m in server.THREAD_META[thread]["lc_msgs"]],
        "the log": caplog.text,
        "stdout/stderr": out.out + out.err,
        "the replies": replies,
        "the history reply": history,
        "what the planner model was sent": model.seen,
        "the rendered RayJob/ConfigMap": seen_job.get("yaml_text", ""),
        "what the cluster was sent (RayJob, ConfigMap)": [t for t in cluster.texts if '"kind": "Secret"' not in t],
        "customers.json on disk": store_text,
        "the MCP /admin/customers request bodies": mcp_payloads,
    }
    leaked = _places_with(probes, places)
    assert rows, "the checkpointer wrote nothing"
    if sealing == "on":
        assert not leaked, f"the probe password is in: {leaked}"
    # The sealed store does hold them (that is where they belong), sealed.
    assert all(p in store_opened or quote_plus(p) in store_opened for p in probes), \
        "the onboarding store does not hold the registered connections"
    assert mcp_payloads, "the onboarding service never registered with the MCP server"
    if sealing == "on":
        assert all((registry.get(a) or {}).get("_source") == "chat" for a in ("chatsrc", "chatdst"))
        rayjob = seen_job["rayjob_name"]
        assert ("secret", f"{rayjob}-env") in cluster.deleted, "the job Secret outlived the job"
    else:
        # The scan is live: with the seal switched off it finds the typed password.
        for where in ("the checkpoint tables", "the stored transcript (graph messages)",
                      "customers.json on disk", "the MCP /admin/customers request bodies"):
            assert where in leaked, f"the scan missed the unsealed {where}; it found: {leaked}"
