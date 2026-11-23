"""Sealed credential handles for every database the caller owns, and no other.

The onboarding API's credential routes answer only a customer's own api_key.
The server gives the planner that key, sealed under DB_ENCRYPTION_KEY, for
every database the AUTHENTICATED caller connected
(``state["db_credential_handles"] = {customer_id: handle}``), so "transfer from
my DB to my other DB" resolves both ends. These tests hold the ownership line:
user A naming user B's database gets no handle and causes no fetch, even when
A's session index has been made to list one of B's sessions. They also check
the handles never reach a log, the trace or the telemetry projection.

Every key, handle and password here is generated in the test.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import types
from typing import Any, Dict

import pytest

import app.agents.planner as planner_mod
from app.agents.planner import InitiateTransferParams, _auto_fetch_from_ui, _handle_initiate_transfer
from app.services import session_service

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import FakeCache, FakeGraph, _do_upload, client, make_auth_headers  # noqa: F401
    from tests.onboarding_identity import assertion_user, credentials_status
except ImportError:
    from server_harness import FakeCache, FakeGraph, _do_upload, client, make_auth_headers  # noqa: F401
    from onboarding_identity import assertion_user, credentials_status

USER_A, USER_B = "user-a", "user-b"


@pytest.fixture
def sealing_key(monkeypatch):
    import app.api.cloud_connections as cc
    key = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)


def _db_session(user_id: str, customer_id: str, api_key: str, created_at: str) -> Dict[str, Any]:
    from app.api.cloud_connections import encrypt_secret
    return {"user_id": user_id, "type": "database", "customer_id": customer_id,
            "connection_id": customer_id, "created_at": created_at,
            "api_key_enc": encrypt_secret(api_key)}


def _plant(cache: FakeCache, sid: str, data: Dict[str, Any], *indexes: str) -> None:
    cache._data[session_service._k_session(sid)] = json.dumps(data)
    for uid in indexes:
        cache._sets.setdefault(session_service._k_user_sessions(uid), set()).add(sid)


@pytest.fixture
def two_users(monkeypatch, sealing_key):
    """A owns a_sales and a_warehouse; B owns b_finance and b_hr.

    B's ``b_hr`` session is ALSO listed in A's session index -- a stale or
    polluted index entry -- so the ownership check on the session itself is
    what keeps it out of A's map, not the index alone.
    """
    cache = FakeCache()
    monkeypatch.setattr(session_service, "cache", cache)
    keys = {cid: "ak_" + secrets.token_hex(8) for cid in ("a_sales", "a_warehouse", "b_finance", "b_hr")}
    _plant(cache, "sa1", _db_session(USER_A, "a_sales", "ak_old_" + secrets.token_hex(4), "2026-10-01T00:00:00"), USER_A)
    _plant(cache, "sa2", _db_session(USER_A, "a_sales", keys["a_sales"], "2026-10-02T00:00:00"), USER_A)
    _plant(cache, "sa3", _db_session(USER_A, "a_warehouse", keys["a_warehouse"], "2026-10-03T00:00:00"), USER_A)
    _plant(cache, "sb1", _db_session(USER_B, "b_finance", keys["b_finance"], "2026-10-04T00:00:00"), USER_B)
    _plant(cache, "sb2", _db_session(USER_B, "b_hr", keys["b_hr"], "2026-10-05T00:00:00"), USER_B, USER_A)
    # A non-database session of A's never yields a handle either.
    _plant(cache, "sa4", {"user_id": USER_A, "type": "upload", "customer_id": "a_csv"}, USER_A)
    return types.SimpleNamespace(cache=cache, keys=keys)


def _opened(handles: Dict[str, Any]) -> Dict[str, str]:
    from app.mcp_server.client import api_key_from_handle
    return {cid: api_key_from_handle(h) for cid, h in handles.items()}


# ---------------------------------------------------------------------------
# the map is the caller's own databases
# ---------------------------------------------------------------------------

def test_handle_map_holds_only_the_callers_own_databases(two_users):
    a = asyncio.run(session_service.db_credential_handles_for_user(USER_A))
    b = asyncio.run(session_service.db_credential_handles_for_user(USER_B))
    assert set(a) == {"a_sales", "a_warehouse"}
    assert set(b) == {"b_finance", "b_hr"}
    # The newest connect session wins, and every handle is ciphertext.
    assert _opened(a) == {"a_sales": two_users.keys["a_sales"], "a_warehouse": two_users.keys["a_warehouse"]}
    assert all(set(h) == {"ciphertext", "iv"} for h in a.values())
    assert asyncio.run(session_service.db_credential_handle_for_user(USER_A, "b_hr")) is None
    assert asyncio.run(session_service.db_credential_handles_for_user(None)) == {}


def _creds_payload(customer_id: str) -> Dict[str, Any]:
    from app.mcp_server.credentials import credential_handle
    password = secrets.token_hex(12)
    return {"host": "db.internal", "port": 5432, "database": customer_id, "username": "svc", "table": "t",
            "credential_handle": credential_handle(f"postgresql://svc:{password}@db.internal:5432/{customer_id}")}


class _Network:
    """The onboarding API: /credentials answers only the owner's assertion, like the real route.

    (Since the merge of develop-1.7, decision 1, the planner asks as the
    signed-in user, never with a database's api_key; ownership decides.)
    """

    def __init__(self, customers):
        self.calls = []
        self.routes = {f"/customers/{c}/credentials": _creds_payload(c) for c in customers}
        self.routes["/customers"] = {"customers": [{"customer_id": c, "database_type": "postgresql"}
                                                   for c in customers]}

    def Session(self):  # noqa: N802 - stands in for requests.Session
        net = self

        class _S:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, **kwargs):
                headers = dict(kwargs.get("headers") or {})
                net.calls.append((url, headers))
                if url.endswith("/credentials"):
                    cid = url.split("/customers/")[-1].split("/")[0]
                    owner = USER_A if cid.startswith("a_") else USER_B
                    status = credentials_status(headers, owner)
                    if status != 200:
                        return types.SimpleNamespace(status_code=status, json=lambda: {})
                for suffix, payload in net.routes.items():
                    if url.endswith(suffix):
                        return types.SimpleNamespace(status_code=200, json=lambda p=payload: p)
                return types.SimpleNamespace(status_code=404, json=lambda: {})

        return _S()


@pytest.fixture
def stub_pipeline(monkeypatch):
    import sys
    fake = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")
    fake.data_transfer_pipeline = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("pipeline stubbed out"))
    monkeypatch.setitem(sys.modules, "app.agents.data_transfer_agent.data_transfer_agent", fake)
    monkeypatch.setattr(planner_mod, "_list_dest_tables", lambda creds, db_type: None)
    # A name that is not a database falls through to the user's cloud connections
    # (Supabase); none here, and no network.
    monkeypatch.setattr(planner_mod, "_resolve_cloud_conn_by_name", lambda *a, **k: None)


def _state_for(user_id: str) -> Dict[str, Any]:
    """What the server puts in state for ``user_id``: the active pair plus the map."""
    handles = asyncio.run(session_service.db_credential_handles_for_user(user_id))
    return {"user_id": user_id, "dta_database_registry": {}, "messages": [],
            "active_db_customer_id": "a_sales", "active_db_credential_handle": handles.get("a_sales"),
            "db_credential_handles": handles}


@pytest.mark.parametrize("victim", ["b_finance", "b_hr"])
def test_user_a_naming_user_bs_database_gets_no_handle_and_no_fetch(two_users, stub_pipeline, monkeypatch, victim):
    net = _Network(["a_sales", "a_warehouse", "b_finance", "b_hr"])
    monkeypatch.setattr(planner_mod, "requests", net)
    state = _state_for(USER_A)
    assert victim not in state["db_credential_handles"]
    assert planner_mod._db_api_key_for(state, victim) is None
    msg = _handle_initiate_transfer(state, InitiateTransferParams(
        source_alias="a_sales", destination_alias=victim, dest_table="t",
        user_prompt=f"transfer orders from a_sales to {victim}"))
    assert f"Destination **`{victim}`** not registered" in msg
    assert victim not in state["dta_database_registry"]
    # Asked as A (ownership decides, decision 1 of the 1.7 merge), and refused;
    # nothing A sent carried B's key or anything but A's own identity.
    victim_calls = [h for u, h in net.calls if f"/customers/{victim}/" in u]
    assert all(assertion_user(h) == USER_A for h in victim_calls)
    assert not [h for _, h in net.calls if two_users.keys[victim] in json.dumps(h)]


def test_user_a_reaches_their_own_other_database(two_users, monkeypatch):
    """The same planner, the same network: A's second database resolves, as itself."""
    net = _Network(["a_sales", "a_warehouse", "b_finance", "b_hr"])
    monkeypatch.setattr(planner_mod, "requests", net)
    entry = _auto_fetch_from_ui("a_warehouse", _state_for(USER_A))
    assert entry and entry["db_type"] == "postgresql" and entry["database"] == "a_warehouse"
    cred = [h for u, h in net.calls if u.endswith("/customers/a_warehouse/credentials")]
    assert len(cred) == 1 and assertion_user(cred[0]) == USER_A
    assert two_users.keys["a_warehouse"] not in json.dumps(net.calls)


def test_active_pair_alone_still_resolves_the_active_database(sealing_key, monkeypatch):
    """Back-compat: a state with only active_db_* (no map) resolves the active database and no other."""
    from app.api.cloud_connections import encrypt_secret
    api_key = "ak_" + secrets.token_hex(8)
    state = {"active_db_customer_id": "acme", "active_db_credential_handle": encrypt_secret(api_key)}
    assert planner_mod._db_api_key_for(state, "acme") == api_key
    assert planner_mod._db_api_key_for(state, "other") is None
    assert planner_mod._db_api_key_for({**state, "db_credential_handles": {}}, "acme") == api_key


# ---------------------------------------------------------------------------
# the chat route builds the map from the authenticated caller
# ---------------------------------------------------------------------------

def test_chat_route_puts_only_the_callers_handles_in_state(client, monkeypatch, sealing_key):
    import app.api.server as server

    cache = session_service.cache
    keys = {cid: "ak_" + secrets.token_hex(8) for cid in ("a_sales", "a_warehouse", "b_finance", "b_hr")}
    _plant(cache, "sa2", _db_session(USER_A, "a_sales", keys["a_sales"], "2026-10-02T00:00:00"), USER_A)
    _plant(cache, "sa3", _db_session(USER_A, "a_warehouse", keys["a_warehouse"], "2026-10-03T00:00:00"), USER_A)
    _plant(cache, "sb1", _db_session(USER_B, "b_finance", keys["b_finance"], "2026-10-04T00:00:00"), USER_B)
    _plant(cache, "sb2", _db_session(USER_B, "b_hr", keys["b_hr"], "2026-10-05T00:00:00"), USER_B, USER_A)

    captured: Dict[str, Any] = {}

    class CapturingGraph(FakeGraph):
        def invoke(self, state_in, config=None):
            captured.update(state_in)
            return super().invoke(state_in, config)

    monkeypatch.setattr(server, "GRAPH", CapturingGraph(), raising=False)
    out = _do_upload(client, user_id=USER_A)
    resp = client.post(
        f"/threads/{out['thread_id']}/messages",
        json={"content": "transfer orders from a_sales to b_hr", "metadata": {"dataset_id": out["dataset_id"]}},
        headers=make_auth_headers(USER_A),
    )
    assert resp.status_code == 200, resp.text
    handles = captured["db_credential_handles"]
    assert set(handles) == {"a_sales", "a_warehouse"}
    assert _opened(handles) == {"a_sales": keys["a_sales"], "a_warehouse": keys["a_warehouse"]}
    # The response carries no handle and no key.
    for h in handles.values():
        assert h["ciphertext"] not in resp.text
    assert "db_credential_handles" not in resp.text
    assert not any(k in resp.text for k in keys.values())


# ---------------------------------------------------------------------------
# handles never reach a log, the trace or telemetry
# ---------------------------------------------------------------------------

def test_handles_never_reach_the_trace_or_the_telemetry_projection(sealing_key):
    from langchain_core.messages import HumanMessage
    from app.api.cloud_connections import encrypt_secret
    from app.trace.build import trace_response
    from app.trace.service import turn_from_state
    from app.trace.telemetry import agent_events
    from tests.test_agent_trace import build_test_graph

    api_keys = {c: "ak_" + secrets.token_hex(8) for c in ("a_sales", "a_warehouse")}
    handles = {c: encrypt_secret(k) for c, k in api_keys.items()}
    final = build_test_graph().invoke(
        {"messages": [HumanMessage(content="transfer orders from a_sales to a_warehouse")],
         "user_id": "u", "session_id": "s",
         "active_db_customer_id": "a_sales", "active_db_credential_handle": handles["a_sales"],
         "db_credential_handles": handles},
        config={"configurable": {"thread_id": "handles"}},
    )
    # The handles were in the graph state for the whole run (so a leak would show) ...
    assert final["db_credential_handles"] == handles
    assert handles["a_warehouse"]["ciphertext"] in json.dumps(final, default=str)
    # ... and neither the trace a user reads nor the telemetry projection carries them.
    turn = turn_from_state(final, question="transfer")
    assert turn is not None
    payloads = {
        "trace": json.dumps(trace_response("t", [turn]), default=str),
        "telemetry": json.dumps(agent_events(final["agent_trace"]), default=str),
    }
    planted = ["db_credential_handles", "active_db_credential_handle", "api_key_enc",
               *api_keys.values(), *(h["ciphertext"] for h in handles.values()),
               *(h["iv"] for h in handles.values())]
    for where, text in payloads.items():
        for p in planted:
            assert p not in text, f"{p[:12]}... reached the {where} payload"


def test_auto_fetch_logs_no_handle_and_no_key(two_users, monkeypatch, caplog):
    net = _Network(["a_sales", "a_warehouse"])
    monkeypatch.setattr(planner_mod, "requests", net)
    state = _state_for(USER_A)
    with caplog.at_level(logging.DEBUG):
        assert _auto_fetch_from_ui("a_warehouse", state)
        assert _auto_fetch_from_ui("b_hr", state) is None
    text = "\n".join(r.getMessage() for r in caplog.records)
    for cid, handle in state["db_credential_handles"].items():
        assert handle["ciphertext"] not in text and handle["iv"] not in text
    assert not any(k in text for k in two_users.keys.values())
    assert "db_credential_handles" not in text
