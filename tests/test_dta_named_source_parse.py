"""A transfer from a database that is not the session's own can be asked for in chat.

The live run on the #513 stack (2026-10-31) found three phrasings that failed,
and the same parser is on integration/2.0 and develop-1.7:

1. "Transfer the legacy_items table from m513legacy to m513dst as legacy_copy."
   The alias regex allows only filler words between the verb and "from", so the
   named source was dropped and the session's own database used in its place:
   "There is no table legacy_items in m513src". The bot's own help text
   suggests exactly this shape.
2. "Transfer from m513legacy to m513dst, table legacy_items as legacy_copy"
   reached the source, but read legacy_items as the DESTINATION table and asked
   "Which table in m513legacy should I transfer (legacy_items)?". That question
   armed no slot, so the answer ("legacy_items", or "yes") went to the LLM door,
   which could not see the source and asked again, forever.
3. Owned -> owned only worked when the source happened to be the session's
   database; "from m513src" was never parsed either.

Everything here runs the planner's real DTA path. The fakes are the onboarding
API (with #623's identities and answers), the source catalogs and the row count.

Credentials resolve as #623 has them: the user's own databases through the
signed-in user's assertion; any other name only through the service token,
which first asks /status (no handle) and only on a 200 -- an active record
bound to nobody -- asks /credentials. Another user's database is refused at
/status, and its credentials are never requested.

Every key, handle and password here is generated in the test.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import types
from typing import Any, Dict, List
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.planner as P
from app.agents.planner import plan_etl_job

USER = "user-a"
OWNED = ("m513src", "m513other", "m513dst")
# Registered before owners existed: bound to nobody. #623 reaches it with the
# service token (#618), never listed and never in the user's handles.
LEGACY = "m513legacy"
FOREIGN = "b_hr"          # another user's database
NONEXISTENT = "nosuchdb"  # no database at all
OWNERS = {**{c: USER for c in OWNED}, LEGACY: None, FOREIGN: "user-b"}
CATALOGS = {
    "m513src": ["items"],
    "m513other": ["sales"],
    "m513legacy": ["legacy_items"],
    "m513dst": ["existing"],
    "b_hr": ["hr_people"],
}


@pytest.fixture
def sealing_key(monkeypatch):
    import app.api.cloud_connections as cc
    key = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)


class _Onboarding:
    """The onboarding API as #623 has it, for the routes a transfer uses.

    Identities: a user assertion (Bearer avu1…) or the service token naming a
    user. ``/customers`` lists the caller's OWN records. ``/status`` and
    ``/credentials`` answer a record the caller owns, or -- to the service token
    only -- an unbound one; anything else is 404, the same as no record. A
    customer api_key would be 403 on /credentials; the planner never sends one.
    Every call is recorded as (url, identity, status).
    """

    def __init__(self, token: str):
        from app.mcp_server.credentials import credential_handle
        self.token = token
        self.calls: List[tuple] = []
        self.passwords = {cid: "pw_" + secrets.token_hex(12) for cid in OWNERS}
        self.payloads = {
            cid: {"host": "db.internal", "port": 5432, "database": cid, "username": "svc",
                  "database_type": "postgresql", "table": CATALOGS[cid][0],
                  "credential_handle": credential_handle(
                      f"postgresql://svc:{self.passwords[cid]}@db.internal:5432/{cid}")}
            for cid in OWNERS
        }

    def _caller(self, headers):
        from app.mcp_server.credentials import verify_user_assertion
        headers = headers or {}
        if self.token and headers.get("X-Avaloka-Service-Token") == self.token:
            return "service", headers.get("X-Avaloka-User")
        auth = headers.get("Authorization", "")
        user = verify_user_assertion(auth[len("Bearer "):]) if auth.startswith("Bearer ") else None
        return ("user", user) if user else ("none", None)

    def Session(self):  # noqa: N802 - stands in for requests.Session
        net = self

        def answer(url, headers):
            kind, user = net._caller(headers)
            if kind == "none":
                return 401, {}
            if url.endswith("/customers"):
                return 200, {"customers": [{"customer_id": c, "database_type": "postgresql"}
                                           for c, o in OWNERS.items() if o and o == user]}
            m = re.search(r"/customers/([^/]+)/(status|credentials)$", url)
            if not m:
                return 404, {}
            cid, route = m.groups()
            if cid not in OWNERS:
                return 404, {}
            owner = OWNERS[cid]
            if not ((owner and owner == user) or (kind == "service" and not owner)):
                return 404, {}
            if route == "status":
                return 200, {"customer_id": cid, "status": "active", "connection_healthy": True}
            return 200, net.payloads[cid]

        class _S:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, headers=None, **_kw):
                status, body = answer(url, headers)
                net.calls.append((url, net._caller(headers)[0], status))
                return types.SimpleNamespace(status_code=status, json=lambda b=body: b)

        return _S()

    def urls(self, cid):
        return [c for c in self.calls if f"/customers/{cid}/" in c[0]]


@pytest.fixture
def world(sealing_key, monkeypatch):
    from app.api.cloud_connections import encrypt_secret
    token = secrets.token_urlsafe(24)
    monkeypatch.setenv("MCP_SERVICE_TOKEN", token)
    net = _Onboarding(token)
    monkeypatch.setattr(P, "requests", net)
    monkeypatch.setattr(P, "_list_source_tables",
                        lambda creds, db_type, schema=None: list(CATALOGS[creds["database"]]))
    monkeypatch.setattr(P, "_list_dest_tables", lambda creds, db_type: list(CATALOGS[creds["database"]]))
    monkeypatch.setattr(P, "_count_source_rows", lambda res: 37)
    monkeypatch.setattr(P, "_resolve_cloud_conn_by_name", lambda *a, **k: None)
    monkeypatch.setattr(P, "_auto_fetch_cloud_from_ui", lambda alias: None)
    monkeypatch.setattr(P, "_seed_ray_fields", lambda state: None)
    monkeypatch.setattr(P, "_seed_infra_request_if_needed", lambda state: None)
    # What the server hands this session: handles for the user's own saved
    # databases only. The legacy one is reached by name, through the service.
    handles = {cid: encrypt_secret("ak_" + secrets.token_hex(8)) for cid in OWNED}
    return types.SimpleNamespace(net=net, handles=handles)


def _state(world, session_db="m513src") -> Dict[str, Any]:
    return {"messages": [], "user_prompt": "", "user_id": USER, "dta_database_registry": {},
            "active_db_customer_id": session_db,
            "active_db_credential_handle": world.handles[session_db],
            "db_credential_handles": dict(world.handles)}


def _say(state: Dict[str, Any], text: str) -> Dict[str, Any]:
    state = dict(state)
    state["messages"] = list(state.get("messages") or []) + [HumanMessage(content=text)]
    state["user_prompt"] = ""
    return plan_etl_job(state)


def _reply(state) -> str:
    return str(state["messages"][-1].content)


def _awaiting(state):
    pending = state.get("pending_clarification")
    return pending.get("awaiting") if isinstance(pending, dict) else None


# ---------------------------------------------------------------------------
# the three phrasings, through the planner's DTA path
# ---------------------------------------------------------------------------

def test_phrasing_1_legacy_source_named_before_from_wins_over_the_session(world):
    out = _say(_state(world), "Transfer the legacy_items table from m513legacy to m513dst as legacy_copy.")
    assert "Copy m513legacy.legacy_items → m513dst.legacy_copy" in _reply(out), _reply(out)
    assert "new table" in _reply(out)
    assert _awaiting(out) == "confirm_transfer"
    assert out["pending_clarification"]["collected"]["source_alias"] == LEGACY


def test_phrasing_2_table_after_both_connections_is_the_source(world):
    out = _say(_state(world), "Transfer from m513legacy to m513dst, table legacy_items as legacy_copy")
    assert "Copy m513legacy.legacy_items → m513dst.legacy_copy" in _reply(out), _reply(out)
    assert _awaiting(out) == "confirm_transfer"


def test_phrasing_3_owned_session_source_is_parsed_not_implied(world, monkeypatch):
    seen = []
    real = P._handle_initiate_transfer
    monkeypatch.setattr(P, "_handle_initiate_transfer",
                        lambda state, params: (seen.append(params.model_copy()), real(state, params))[1])
    out = _say(_state(world), "Transfer the items table from m513src to m513dst as items_copy.")
    assert seen and seen[0].source_alias == "m513src", "the named source was not parsed"
    assert seen[0].source_table == "items"
    assert "Copy m513src.items → m513dst.items_copy" in _reply(out), _reply(out)


def test_a_non_session_owned_source_is_used_not_the_session_database(world):
    out = _say(_state(world, session_db="m513src"), "Transfer the sales table from m513other to m513dst as sales_copy.")
    assert "Copy m513other.sales → m513dst.sales_copy" in _reply(out), _reply(out)
    assert "m513src" not in _reply(out)


@pytest.mark.parametrize("prompt", [
    "copy the legacy_items table from m513legacy to m513dst as legacy_copy",
    "move table legacy_items from m513legacy to m513dst as legacy_copy",
    "transfer legacy_items from m513legacy to m513dst as legacy_copy",
    "Transfer `legacy_items` from `m513legacy` to `m513dst` as legacy_copy",
    'transfer "legacy_items" from m513legacy to m513dst named legacy_copy',
])
def test_the_reasonable_variants_parse_the_same(world, prompt):
    out = _say(_state(world), prompt)
    assert "Copy m513legacy.legacy_items → m513dst.legacy_copy" in _reply(out), _reply(out)


@pytest.mark.parametrize("prompt", [
    "move revenue from q1 to q2",          # a column request, not a transfer
    "transfer data from m513legacy to m513dst",   # "data" names no table
    "transfer from m513legacy to m513dst into table orders",
])
def test_the_new_parser_leaves_other_shapes_alone(prompt):
    """A no-regression guard (passes before and after): these keep their old routes."""
    parse = getattr(P, "_extract_named_table_transfer", lambda text: None)
    assert parse(prompt) is None


# ---------------------------------------------------------------------------
# the clarify loop
# ---------------------------------------------------------------------------

def _llm_reaching_for(source_alias, source_table, destination_alias="m513dst"):
    """The planner LLM as the live run saw it: it rebuilds the transfer from history."""
    fake = mock.MagicMock()
    fake.invoke.return_value = AIMessage(content="", tool_calls=[{
        "name": "initiate_transfer", "id": "t1",
        "args": {"destination_alias": destination_alias, "user_prompt": "transfer",
                 "source_alias": source_alias, "source_table": source_table}}])
    return fake


@pytest.mark.parametrize("answer", ["legacy_items", "yes"])
def test_answering_which_table_in_the_source_continues_the_transfer(world, monkeypatch, answer):
    monkeypatch.setattr(P, "llm", _llm_reaching_for(LEGACY, "legacy_items"))
    out = _say(_state(world), "Transfer from m513legacy to m513dst as legacy_copy")
    assert _reply(out).startswith("❓ Which table in m513legacy should I transfer (legacy_items)?"), _reply(out)
    out = _say(out, answer)
    assert "couldn't work out" not in _reply(out)
    assert "Copy m513legacy.legacy_items → m513dst.legacy_copy" in _reply(out), _reply(out)
    # Consent is still its own step: only a plain yes to THIS question runs it (#569).
    assert _awaiting(out) == "confirm_transfer"


def test_a_table_the_source_lacks_is_refused_not_looped(world):
    out = _say(_state(world), "Transfer from m513legacy to m513dst as legacy_copy")
    out = _say(out, "items")
    assert "There is no table items in m513legacy" in _reply(out), _reply(out)
    assert _awaiting(out) is None


def test_yes_with_several_tables_offered_asks_for_the_name(world, monkeypatch):
    monkeypatch.setitem(CATALOGS, "m513legacy", ["legacy_items", "legacy_orders"])
    out = _say(_state(world), "Transfer from m513legacy to m513dst as legacy_copy")
    out = _say(out, "yes")
    assert "name of the table" in _reply(out), _reply(out)
    assert _awaiting(out) == "source_table"
    out = _say(out, "legacy_orders")
    assert "Copy m513legacy.legacy_orders → m513dst.legacy_copy" in _reply(out), _reply(out)


@pytest.mark.parametrize("answer", ["legacy_items", "yes"])
def test_an_answer_to_the_bots_own_question_counts_as_named_without_a_slot(world, monkeypatch, answer):
    """A thread whose question was asked before the slot existed (a checkpoint from
    before this fix): the LLM door carries the answer, and the "user never named"
    guard must count it as named by the user."""
    monkeypatch.setattr(P, "llm", _llm_reaching_for(LEGACY, "legacy_items"))
    state = _state(world)
    state["messages"] = [
        HumanMessage(content="Transfer from m513legacy to m513dst"),
        AIMessage(content="❓ Which table in m513legacy should I transfer (legacy_items)?"),
    ]
    out = _say(state, answer)
    assert "couldn't work out" not in _reply(out), _reply(out)
    # The source and its table are settled; the next question is the destination table.
    assert "Which table in m513dst should I transfer into" in _reply(out), _reply(out)
    collected = out["pending_clarification"]["collected"]
    assert (collected["source_alias"], collected["source_table"]) == (LEGACY, "legacy_items")
    out = _say(out, "new table legacy_copy")
    assert "Copy m513legacy.legacy_items → m513dst.legacy_copy" in _reply(out), _reply(out)


def test_the_never_named_guard_keeps_a_table_the_user_accepted_from_the_bots_question(world, monkeypatch):
    """The live log: "DTA: dropping LLM source_table 'legacy_items' the user never named"
    on a "yes" to the bot's own one-table question. Here the LLM names no source at
    all, so only the guard decides whether legacy_items survives."""
    monkeypatch.setattr(P, "llm", _llm_reaching_for(None, "legacy_items"))
    state = _state(world)
    state["messages"] = [
        HumanMessage(content="Transfer from m513legacy to m513dst"),
        AIMessage(content="❓ Which table in `m513legacy` should I transfer (legacy_items)?"),
    ]
    out = _say(state, "yes")
    collected = (out.get("pending_clarification") or {}).get("collected") or {}
    assert collected.get("source_table") == "legacy_items", _reply(out)
    assert collected.get("source_alias") == LEGACY, _reply(out)


def test_the_guard_still_drops_a_table_nobody_offered_or_typed():
    """A no-regression guard (passes before and after)."""
    params = P.InitiateTransferParams(destination_alias="m513dst", user_prompt="go on then",
                                      source_table="patients")
    assert P._normalize_llm_transfer_params(params, "go on then").source_table is None


# ---------------------------------------------------------------------------
# security: no new path to another user's database
# ---------------------------------------------------------------------------

def test_the_legacy_source_is_reached_through_the_service_after_a_status_probe(world):
    out = _say(_state(world), "Transfer the legacy_items table from m513legacy to m513dst as legacy_copy.")
    assert "Copy m513legacy.legacy_items → m513dst.legacy_copy" in _reply(out), _reply(out)
    calls = world.net.urls(LEGACY)
    # Not the user's: never asked for as the user, probed before its credentials.
    assert all(ident == "service" for _, ident, _ in calls), calls
    assert calls[0][0].endswith(f"/customers/{LEGACY}/status") and calls[0][2] == 200, calls


def test_an_owned_source_is_asked_for_as_the_user_only(world):
    _say(_state(world), "Transfer the sales table from m513other to m513dst as sales_copy.")
    calls = world.net.urls("m513other")
    assert calls and all(ident == "user" for _, ident, _ in calls), calls
    assert not [c for c in calls if c[0].endswith("/status")]


@pytest.mark.parametrize("prompt", [
    "Transfer the hr_people table from b_hr to m513dst as hr_copy.",
    "Transfer from b_hr to m513dst, table hr_people as hr_copy",
    "transfer from b_hr to m513dst",
])
def test_another_users_database_is_refused_without_a_credentials_request(world, caplog, prompt):
    import json
    import logging
    with caplog.at_level(logging.DEBUG):
        out = _say(_state(world), prompt)
    calls = world.net.urls(FOREIGN)
    # Only the status probe names it, as the service, and is refused.
    assert calls == [(f"{P.ONBOARDING_API_URL}/customers/{FOREIGN}/status", "service", 404)], calls
    assert not [u for u, _, _ in world.net.calls if u.endswith(f"/customers/{FOREIGN}/credentials")]
    assert f"Source {FOREIGN} not registered" in _reply(out), _reply(out)
    # Nothing copied or waiting to be: no confirmation, nothing registered.
    assert _awaiting(out) != "confirm_transfer"
    assert FOREIGN not in out["dta_database_registry"]
    assert "dta_last_transfer" not in out
    # The planted password is nowhere: replies, logs, state.
    planted = world.net.passwords[FOREIGN]
    assert planted not in json.dumps([str(m.content) for m in out["messages"]])
    assert planted not in caplog.text
    assert planted not in json.dumps(out, default=str)


def _turn_fingerprint(world, cid):
    out = _say(_state(world), f"Transfer the hr_people table from {cid} to m513dst as hr_copy.")
    last = out["messages"][-1]
    view = {
        "content": str(last.content),
        "kwargs": getattr(last, "additional_kwargs", {}),
        "metadata": getattr(last, "response_metadata", {}),
        "awaiting": _awaiting(out),
        "is_dta_request": out.get("is_dta_request"),
        "execution_result": out.get("execution_result"),
        "prompts": out.get("prompts"),
    }
    text = json.dumps(view, default=str, sort_keys=True).replace(cid, "<ID>")
    calls = [(u.replace(cid, "<ID>"), ident, st) for u, ident, st in world.net.calls]
    return text, calls


def test_no_existence_oracle_between_a_missing_and_a_foreign_database(world):
    foreign_text, foreign_calls = _turn_fingerprint(world, FOREIGN)
    world.net.calls.clear()
    missing_text, missing_calls = _turn_fingerprint(world, NONEXISTENT)
    assert foreign_text == missing_text
    assert foreign_calls == missing_calls


# ---------------------------------------------------------------------------
# every phrasing the bot suggests parses
# ---------------------------------------------------------------------------

_FILL = {
    "<table>": "orders", "<source connection>": "clinic", "<destination connection>": "warehouse",
    "<connection>": "warehouse", "<new table>": "orders_copy", "<destination>": "warehouse",
    "<source file>": "in/orders.csv", "<destination file>": "out/orders.parquet",
}
_EXPECTED = {
    "transfer the orders table from clinic to warehouse":
        {"source_alias": "clinic", "source_table": "orders", "destination_alias": "warehouse"},
    "transfer from clinic to warehouse, from in/orders.csv to out/orders.parquet":
        {"source_alias": "clinic", "source_object": "in/orders.csv", "destination_alias": "warehouse",
         "dest_object": "out/orders.parquet"},
    "transfer to warehouse": {"source_alias": None, "destination_alias": "warehouse"},
    "transfer to My Warehouse, to file exports/result.parquet":
        {"source_alias": None, "destination_alias": "My Warehouse", "dest_object": "exports/result.parquet"},
    "Copy the orders table to warehouse as orders_copy":
        {"source_alias": None, "source_table": "orders", "destination_alias": "warehouse",
         "dest_table": "orders_copy", "create_if_missing": True},
}


def _fill(phrase: str) -> str:
    phrase = phrase.replace("`", "").replace("*", "").strip()
    for k, v in _FILL.items():
        phrase = phrase.replace(k, v)
    return phrase


def _suggested_phrases(monkeypatch) -> List[str]:
    """Collect the phrasings from the messages the bot actually renders."""
    out: List[str] = []
    # No active source: 'e.g. "…"'.
    msg = P._resolve_implicit_source({})
    out += re.findall(r'e\.g\. "([^"]+)"', msg)
    # The LLM door's "couldn't work out which one": its quoted lines and the bold form.
    monkeypatch.setattr(P, "llm", _llm_reaching_for("somewhere", None, "warehouse"))
    door = _reply(plan_etl_job({"messages": [HumanMessage(content="shift the stuff from somewhere to warehouse")],
                                "user_prompt": "", "dta_database_registry": {}}))
    assert "couldn't work out" in door, door
    out += [ln[1:].strip() for ln in door.splitlines() if ln.startswith(">")]
    out += re.findall(r"just say (transfer to [^\n]+?) and I'll", door)
    # The cloud-datasets list tip.
    import app.api.cloud_connections as cc

    async def _conns(user_id):
        return [{"id": "c1", "provider": "gcs", "name": "My Warehouse", "bucket_name": "b"}]
    monkeypatch.setattr(cc, "list_cloud_connections", _conns)
    tip = P._handle_list_cloud_datasets({"user_id": USER})
    out += re.findall(r"e\.g\. `([^`]+)`", tip)
    # The database-write refusal.
    from app.agents.db_write_guard import DB_WRITE_REFUSAL
    out += re.findall(r'"([^"]+)"', DB_WRITE_REFUSAL)
    return [_fill(p) for p in out]


def test_every_phrasing_the_bot_suggests_parses(monkeypatch):
    monkeypatch.setattr(P, "_seed_ray_fields", lambda state: None)
    monkeypatch.setattr(P, "_seed_infra_request_if_needed", lambda state: None)
    phrases = _suggested_phrases(monkeypatch)
    assert set(phrases) == set(_EXPECTED), f"a suggestion has no expectation here: {phrases}"
    for phrase in phrases:
        with mock.patch.object(P, "_handle_initiate_transfer", return_value="HANDLED") as m:
            plan_etl_job({"messages": [HumanMessage(content=phrase)], "user_prompt": "",
                          "dta_database_registry": {}})
        assert m.call_count == 1, f"{phrase!r} did not parse as a transfer"
        params = m.call_args[0][1]
        for field, want in _EXPECTED[phrase].items():
            assert getattr(params, field) == want, f"{phrase!r}: {field}={getattr(params, field)!r}, want {want!r}"


def test_the_wizard_replies_the_bot_suggests_are_understood(world):
    out = _say(_state(world), "Transfer the legacy_items table from m513legacy to m513dst")
    assert "Which table in m513dst should I transfer into" in _reply(out), _reply(out)
    assert "new table <name>" in _reply(out), _reply(out)
    out = _say(out, "new table legacy_copy")
    assert "Copy m513legacy.legacy_items → m513dst.legacy_copy" in _reply(out), _reply(out)
