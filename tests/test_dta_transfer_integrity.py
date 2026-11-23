"""J1: a transfer moves exactly the table the user named, or refuses.

The rc5 web re-check asked "Transfer the no_such_table table to e2ewhgemma as
ghost_copy", then "new table ghost_copy", and got "SUCCESS ... 5,110 copied":
``patients`` (the table being analyzed) was copied into ghost_copy, on both
models. No parser read the named source table, so the handler fell back to the
analyzed one; and the wizard's follow-up turn kept no source slot at all, so
even a correctly named table would have been replaced by the active one there.

These drive the planner node (``plan_etl_job``) with the user's own sentences
against real SQLite databases standing in for the two PostgreSQL connections,
with no Ray configured, so the local copy (#547) really runs.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.planner as planner

PATIENTS = 5110   # the evidence table's size
ORDERS = 7


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    for var in ("RAY_DASHBOARD_URL", "RAY_DASHBOARD_URL_INCLUSTER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setenv("DTA_GKE_REACHES_PRIVATE", "1")


@pytest.fixture
def sealing_key(monkeypatch):
    import app.api.cloud_connections as cc
    key = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)


def _tables(path: Path):
    con = sqlite3.connect(path)
    try:
        return sorted(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    finally:
        con.close()


def _count(path: Path, table: str) -> int:
    con = sqlite3.connect(path)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


@pytest.fixture
def world(tmp_path, monkeypatch, sealing_key):
    """The user is analyzing clinic.patients; clinic also holds orders."""
    src, dst = tmp_path / "clinic.db", tmp_path / "warehouse.db"
    wh2 = tmp_path / "wh2.db"   # a second destination, for "copy this table to wh2"
    con = sqlite3.connect(src)
    con.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, age REAL, stroke INTEGER)")
    con.executemany("INSERT INTO patients VALUES (?, ?, ?)",
                    [(i, 30 + i % 50, 1 if i < 10 else 0) for i in range(PATIENTS)])
    con.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, units INTEGER)")
    con.executemany("INSERT INTO orders VALUES (?, ?)", [(i, i * 2) for i in range(ORDERS)])
    con.commit()
    con.close()
    sqlite3.connect(dst).close()
    sqlite3.connect(wh2).close()
    files = {"clinic_db": src, "warehouse_db": dst, "wh2_db": wh2}

    import app.agents.data_transfer_agent.data_transfer_agent as dta
    monkeypatch.setattr(dta, "build_connection_string",
                        lambda creds, db_type: (f"sqlite:///{files[creds['database']]}", {}))
    by_db = files
    monkeypatch.setattr(planner, "_list_dest_tables",
                        lambda creds, db_type: _tables(by_db[creds["database"]]))
    monkeypatch.setattr(planner, "_list_source_tables",
                        lambda creds, db_type, schema=None: _tables(by_db[creds["database"]]),
                        raising=False)

    # Credentials endpoint unreachable: the registry entries are used as they are.
    resp = MagicMock()
    resp.status_code = 404
    session = patch("app.agents.planner.requests.Session")
    m = session.start()
    m.return_value.__enter__.return_value.get.return_value = resp
    for name in ("_seed_ray_fields", "_seed_infra_request_if_needed"):
        p = patch(f"app.agents.planner.{name}")
        p.start()

    from app.api.cloud_connections import encrypt_secret
    state = {
        "dta_database_registry": {
            # As the credentials endpoint returns it: "table" is the first table
            # it auto-detected, not anything the user chose.
            "clinic": {"db_type": "postgresql", "host": "db", "port": 5432, "database": "clinic_db",
                       "user": "u", "password": "p", "table": "orders", "_source": "ui"},
            "warehouse": {"db_type": "postgresql", "host": "db", "port": 5432,
                          "database": "warehouse_db", "user": "u", "password": "p",
                          "table": None, "_source": "ui"},
            "wh2": {"db_type": "postgresql", "host": "db", "port": 5432,
                    "database": "wh2_db", "user": "u", "password": "p",
                    "table": None, "_source": "ui"},
        },
        "db_credential_handles": {"clinic": encrypt_secret("ak_c"), "warehouse": encrypt_secret("ak_w"),
                                  "wh2": encrypt_secret("ak_w2")},
        "active_db_customer_id": "clinic",
        "active_db_source_table": "patients",
        "user_prompt": "",
        "messages": [],
    }
    state["_wh2_path"] = wh2  # not a planner key; the consent tests read it
    yield state, src, dst
    patch.stopall()


def _turn(state, text: str) -> str:
    state["messages"] = list(state.get("messages") or []) + [HumanMessage(content=text)]
    out = planner.plan_etl_job(state)
    state.update(out)
    reply = [m for m in state["messages"] if isinstance(m, AIMessage)][-1].content
    return reply


# ---- (a) a missing table is refused, nothing is created, the tables are listed ----

def test_missing_source_table_is_refused_before_anything_is_created(world):
    state, src, dst = world
    before = _tables(dst)
    reply = _turn(state, "Transfer the no_such_table table to warehouse as ghost_copy.")
    assert "no table no_such_table" in reply.lower(), reply
    assert "SUCCESS" not in reply and "copied" not in reply, reply
    # The available tables are listed, so the user can ask again.
    assert "• orders" in reply and "• patients" in reply, reply
    assert _tables(dst) == before == []
    # No transfer slot is left open, so "new table ghost_copy" cannot complete it.
    pending = state.get("pending_clarification")
    assert not (isinstance(pending, dict) and pending.get("type") == "dta_transfer"), pending
    assert state.get("dta_last_transfer") is None


def test_close_name_is_suggested_not_substituted(world):
    state, _src, dst = world
    reply = _turn(state, "Transfer the patient table to warehouse as p2.")
    assert "no table patient " in reply.lower() or "no table patient," in reply.lower(), reply
    assert "Did you mean patients?" in reply, reply
    assert _tables(dst) == []


def test_case_insensitive_name_uses_the_catalog_spelling(world):
    state, src, dst = world
    ask = _turn(state, "Transfer the ORDERS table to warehouse as orders_copy.")
    assert "Copy clinic.orders → warehouse.orders_copy" in ask, ask
    reply = _turn(state, "yes")
    assert "SUCCESS" in reply, reply
    assert _count(dst, "orders_copy") == _count(src, "orders") == ORDERS


# ---- (b) the success message names both tables, and the counts match ----

def test_success_names_source_and_destination_and_counts_match(world):
    state, src, dst = world
    _turn(state, "Transfer the patients table to warehouse as patients_copy.")
    reply = _turn(state, "yes")
    assert "SUCCESS" in reply, reply
    assert f"from source table patients into destination table patients_copy" in reply, reply
    assert (f"Row counts match: {PATIENTS:,} in source table patients, "
            f"{PATIENTS:,} copied into patients_copy") in reply, reply
    assert _count(dst, "patients_copy") == _count(src, "patients") == PATIENTS


# ---- (c) no path substitutes the active table for the named one ----

def test_wizard_follow_up_keeps_the_named_table_not_the_active_one(world):
    """The user names orders while analyzing patients; the follow-up turn must copy orders."""
    state, src, dst = world
    ask = _turn(state, "Transfer the orders table to warehouse.")
    assert "Which table in warehouse" in ask, ask
    ask = _turn(state, "new table orders_copy")
    assert "Copy clinic.orders → warehouse.orders_copy" in ask, ask
    reply = _turn(state, "yes")
    assert "SUCCESS" in reply, reply
    assert "from source table orders" in reply, reply
    assert _count(dst, "orders_copy") == ORDERS          # not PATIENTS


def test_no_named_and_no_analyzed_table_never_uses_the_first_table(world):
    """The credentials endpoint's auto-detected first table is never a fallback."""
    state, _src, dst = world
    state["active_db_source_table"] = None
    reply = _turn(state, "Transfer it to warehouse as copy1.")
    assert reply == "❓ Which table in clinic should I transfer (orders, patients)?", reply
    assert _tables(dst) == []


def test_llm_door_takes_the_users_table_and_drops_an_invented_one():
    p = planner.InitiateTransferParams(destination_alias="warehouse", user_prompt="x",
                                       source_table="patients")
    planner._normalize_llm_transfer_params(p, "Transfer the no_such_table table to warehouse")
    assert p.source_table == "no_such_table"

    q = planner.InitiateTransferParams(destination_alias="warehouse", user_prompt="x",
                                       source_table="patients")
    planner._normalize_llm_transfer_params(
        q, "move it over to warehouse please\n\n[Analysis context] rows of `patients`")
    assert q.source_table is None


@pytest.mark.parametrize("prompt, table", [
    ("Transfer the no_such_table table to e2ewhgemma as ghost_copy.", "no_such_table"),
    ("copy table orders to warehouse", "orders"),
    ("move my `orders` table into warehouse", "orders"),
    ("transfer the data to warehouse", None),
    ("transfer to warehouse into table orders", None),
])
def test_source_table_extraction(prompt, table):
    got = planner._extract_source_table_from_prompt(prompt)
    assert (got[0] if got else None) == table


def test_source_clause_is_not_read_as_the_destination_table():
    with patch("app.agents.planner._handle_initiate_transfer", return_value="HANDLED") as m, \
         patch("app.agents.planner._seed_ray_fields"), \
         patch("app.agents.planner._seed_infra_request_if_needed"):
        planner.plan_etl_job({"messages": [HumanMessage(content="copy table orders to warehouse")],
                              "user_prompt": "", "dta_database_registry": {}})
    params = m.call_args[0][1]
    assert params.source_table == "orders"
    assert params.dest_table is None


# ===========================================================================
# Follow-up: all four names explicit or from this thread, one confirmation
# naming them before any write, "as <name>" is the destination table.
# ===========================================================================

def _no_source_in_thread(state):
    state["active_db_customer_id"] = None
    state["active_db_source_table"] = None


def test_a_no_source_connection_in_context_asks_one_line_and_writes_nothing(world):
    state, _src, dst = world
    _no_source_in_thread(state)
    reply = _turn(state, "Transfer the patients table to warehouse as ghost_copy.")
    assert reply.startswith("❓") and "\n" not in reply, reply
    assert "which database and table should I transfer from" in reply, reply
    assert _tables(dst) == []
    assert state.get("dta_last_transfer") is None


def test_c_confirmation_names_both_connections_and_tables_and_writes_nothing(world):
    state, _src, dst = world
    ask = _turn(state, "Transfer the patients table to warehouse as ghost_copy.")
    assert ask.startswith(f"❓ Copy clinic.patients → warehouse.ghost_copy ({PATIENTS:,} rows, new table)?"), ask
    assert "\n" not in ask
    assert state["pending_clarification"]["awaiting"] == "confirm_transfer"
    assert _tables(dst) == [], "nothing is created before the user confirms"


def test_c_no_cancels_and_writes_nothing(world):
    state, _src, dst = world
    _turn(state, "Transfer the patients table to warehouse as ghost_copy.")
    reply = _turn(state, "no")
    assert "cancelled" in reply and "Nothing was written" in reply, reply
    assert _tables(dst) == []
    assert not state.get("pending_clarification")


def test_d_as_name_sets_the_destination_table(world):
    with patch("app.agents.planner._handle_initiate_transfer", return_value="HANDLED") as m, \
         patch("app.agents.planner._seed_ray_fields"), \
         patch("app.agents.planner._seed_infra_request_if_needed"):
        for text in ("Transfer the patients table to warehouse as ghost_copy.",
                     "Transfer the patients table to warehouse named ghost_copy",
                     "copy the patients table to warehouse into ghost_copy"):
            planner.plan_etl_job({"messages": [HumanMessage(content=text)],
                                  "user_prompt": "", "dta_database_registry": {}})
            params = m.call_args[0][1]
            assert (params.destination_alias, params.dest_table) == ("warehouse", "ghost_copy"), text
            assert params.source_table == "patients"
            # "as"/"named" ask for a new table; "into" names one that may exist.
            assert params.create_if_missing is (not text.startswith("copy")), text


def test_d_invalid_name_is_asked_for_again(world):
    state, _src, dst = world
    reply = _turn(state, "Transfer the patients table to warehouse as ghost-copy.")
    assert "isn't a valid table name" in reply and "\n" not in reply, reply
    assert state["pending_clarification"]["awaiting"] == "new_table_name"
    assert _tables(dst) == []


def test_e_existing_destination_table_is_asked_about_not_overwritten(world):
    state, _src, dst = world
    con = sqlite3.connect(dst)
    con.execute("CREATE TABLE ghost_copy (id INTEGER, age REAL, stroke INTEGER)")
    con.execute("INSERT INTO ghost_copy VALUES (-1, 1, 0)")
    con.commit()
    con.close()
    reply = _turn(state, "Transfer the patients table to warehouse as ghost_copy.")
    assert "already exists in warehouse" in reply and "append" in reply, reply
    assert state["pending_clarification"]["awaiting"] == "dest_exists"
    assert _count(dst, "ghost_copy") == 1, "the existing table was written to"
    # A new name moves on to the confirmation, still nothing written.
    ask = _turn(state, "ghost_copy2")
    assert "Copy clinic.patients → warehouse.ghost_copy2" in ask, ask
    assert _count(dst, "ghost_copy") == 1 and "ghost_copy2" not in _tables(dst)


def test_f_happy_path_copies_every_row(world):
    state, src, dst = world
    ask = _turn(state, "Transfer the patients table to warehouse as ghost_copy.")
    assert "Copy clinic.patients → warehouse.ghost_copy (5,110 rows" in ask, ask
    reply = _turn(state, "yes")
    assert "SUCCESS" in reply, reply
    assert "Row counts match: 5,110 in source table patients, 5,110 copied into ghost_copy" in reply
    assert _count(dst, "ghost_copy") == _count(src, "patients") == 5110


# ===========================================================================
# rc7 web re-check: on a database-table thread "Transfer the patients table to
# e2ewh as patients_copy" answered "Table read doesn't exist", then wrote 5,110
# rows into a NEW table named `read`. The chat endpoint appends an
# "[Analysis context]" line whose "Database table read in full" gave the bare
# "table <word>" pattern a destination table, and that pre-empted "as <name>".
# ===========================================================================

from app.core.db_sampling import DbSample, sample_context_line  # noqa: E402

# The line rc7 shipped, verbatim: the parsers must ignore it whatever it says.
_RC7_LINE = "Database table read in full (5,110 rows); figures are exact."


def _with_context(text: str, line: str) -> str:
    """The message as app/api/server.py builds it for a database-table thread."""
    return "\n".join([text, (
        "\n[Analysis context] Current fidelity: full; Effective fidelity for this "
        "request: full; Selected sample: none; Resolved analysis source: patients; "
        "Dataset size bytes: 0; " + line)])


_CONTEXT_LINES = [
    pytest.param(_RC7_LINE, id="rc7-line"),
    pytest.param(sample_context_line(DbSample(method="full", n=5110, total=5110)), id="current-line"),
]


def _fast_path_params(text: str):
    with patch("app.agents.planner._handle_initiate_transfer", return_value="HANDLED") as m, \
         patch("app.agents.planner._seed_ray_fields"), \
         patch("app.agents.planner._seed_infra_request_if_needed"):
        planner.plan_etl_job({"messages": [HumanMessage(content=text)],
                              "user_prompt": "", "dta_database_registry": {}})
    assert m.called, "the transfer fast path did not fire"
    return m.call_args[0][1]


@pytest.mark.parametrize("line", _CONTEXT_LINES)
@pytest.mark.parametrize("name", ["patients_copy", "ghost_copy"])
def test_context_line_never_names_the_destination_table(line, name):
    params = _fast_path_params(_with_context(f"Transfer the patients table to wh as {name}", line))
    assert (params.destination_alias, params.source_table) == ("wh", "patients")
    assert params.dest_table == name
    assert params.create_if_missing is True


@pytest.mark.parametrize("line", _CONTEXT_LINES)
def test_context_line_alone_names_no_destination_table(line):
    params = _fast_path_params(_with_context("Transfer the patients table to wh", line))
    assert params.dest_table is None
    assert params.create_if_missing is False


def test_as_name_outranks_a_bare_table_word_in_the_users_own_words():
    # "as <name>" is anchored to the destination; a loose "table <word>" elsewhere
    # in the sentence must not replace it.
    params = _fast_path_params("Transfer the patients table to wh as patients_copy, table stays")
    assert (params.dest_table, params.create_if_missing) == ("patients_copy", True)


@pytest.mark.parametrize("line", _CONTEXT_LINES)
def test_llm_door_ignores_the_context_line(line):
    p = planner.InitiateTransferParams(destination_alias="wh", user_prompt="x")
    planner._normalize_llm_transfer_params(
        p, _with_context("Transfer the patients table to wh as patients_copy", line))
    assert (p.source_table, p.dest_table, p.create_if_missing) == ("patients", "patients_copy", True)

    q = planner.InitiateTransferParams(destination_alias="wh", user_prompt="x")
    planner._normalize_llm_transfer_params(q, _with_context("move the data over to wh please", line))
    assert q.dest_table is None


def test_context_line_does_not_read_as_table_word():
    line = sample_context_line(DbSample(method="full", n=5110, total=5110))
    assert planner._extract_dest_table_from_prompt(line) is None, line


@pytest.mark.parametrize("line", _CONTEXT_LINES)
def test_end_to_end_confirmation_and_write_name_the_as_table(world, line):
    state, src, dst = world
    ask = _turn(state, _with_context("Transfer the patients table to warehouse as patients_copy", line))
    assert ask.startswith(
        f"❓ Copy clinic.patients → warehouse.patients_copy ({PATIENTS:,} rows, new table)?"), ask
    reply = _turn(state, "yes")
    assert "SUCCESS" in reply, reply
    assert _tables(dst) == ["patients_copy"], _tables(dst)
    assert _count(dst, "patients_copy") == PATIENTS


# ===========================================================================
# rc8 web check. (C1) While "❓ Copy A → B (…)?" was pending, "Copy this table
# to e2ewh." ran the pending transfer and wrote 5,110 rows: the consent regex
# matched any reply that merely STARTED with copy/start/go/ok. (C2) In the
# "already exists" slot every non-append reply was a new table name: "no"
# became wh.no, and "create" then "yes" wrote a table named `create`.
# ===========================================================================

_ASK = "Transfer the patients table to warehouse as ghost_copy."


def _nothing_written(state, dst):
    assert _tables(dst) == [], _tables(dst)
    assert _tables(state["_wh2_path"]) == [], _tables(state["_wh2_path"])
    assert state.get("dta_last_transfer") is None


@pytest.mark.parametrize("reply", [
    "Copy this table to wh2.",
    "Copy this table to warehouse.",
    "Copy this table to e2ewh.",   # a connection that is not registered
    "yes but to wh2",
    "ok copy it to wh2 instead",
    "start the transfer to wh2",
    "go to wh2",
    "copy everything",
])
@pytest.mark.parametrize("web", [False, True], ids=["bare", "with-analysis-context"])
def test_c1_a_reply_with_its_own_intent_never_runs_the_pending_transfer(world, reply, web):
    # The web chat appends an [Analysis context] line to every message; that is
    # how rc8's "Copy this table to e2ewh." slipped past the new-request check.
    say = (lambda t: _with_context(t, _RC7_LINE)) if web else (lambda t: t)
    state, _src, dst = world
    ask = _turn(state, say(_ASK))
    assert state["pending_clarification"]["awaiting"] == "confirm_transfer", ask
    out = _turn(state, say(reply))
    assert "SUCCESS" not in out, out
    _nothing_written(state, dst)
    # Whatever is pending now is not the old plan waiting to be run on a later "yes".
    pending = state.get("pending_clarification") or {}
    if pending.get("awaiting") == "confirm_transfer":
        assert "ghost_copy" not in str(pending.get("collected", {}).get("confirm_key")), pending
    # And a later bare "yes" does not resurrect the old plan.
    _turn(state, "yes")
    assert "ghost_copy" not in _tables(dst)


@pytest.mark.parametrize("web", [False, True], ids=["bare", "with-analysis-context"])
def test_c1_a_new_copy_request_is_handled_as_a_fresh_request(world, web):
    say = (lambda t: _with_context(t, _RC7_LINE)) if web else (lambda t: t)
    state, _src, dst = world
    _turn(state, say(_ASK))
    out = _turn(state, say("Copy this table to wh2."))
    # The new request is asked about on its own terms, naming wh2; nothing runs.
    assert "wh2" in out and "ghost_copy" not in out, out
    _nothing_written(state, dst)


@pytest.mark.parametrize("reply", [
    "yes", "Yes!", "yes please", "ok, go ahead", "do it", "confirm", "copy", "please proceed", "👍 yes",
])
def test_c1_a_wholly_affirmative_reply_writes(world, reply):
    state, src, dst = world
    _turn(state, _ASK)
    out = _turn(state, reply)
    assert "SUCCESS" in out, (reply, out)
    assert _count(dst, "ghost_copy") == _count(src, "patients") == PATIENTS


@pytest.mark.parametrize("reply", ["no", "cancel", "stop", "never mind"])
def test_c1_a_negative_cancels(world, reply):
    state, _src, dst = world
    _turn(state, _ASK)
    out = _turn(state, reply)
    assert "cancelled" in out and "Nothing was written" in out, out
    _nothing_written(state, dst)
    assert not state.get("pending_clarification")


def _dest_exists(state, dst):
    con = sqlite3.connect(dst)
    con.execute("CREATE TABLE ghost_copy (id INTEGER, age REAL, stroke INTEGER)")
    con.commit()
    con.close()
    reply = _turn(state, _ASK)
    assert state["pending_clarification"]["awaiting"] == "dest_exists", reply


@pytest.mark.parametrize("reply", ["no", "cancel", "stop", "never mind", "nope"])
def test_c2_no_in_dest_exists_cancels(world, reply):
    state, _src, dst = world
    _dest_exists(state, dst)
    out = _turn(state, reply)
    assert "cancelled" in out and "Nothing was written" in out, out
    assert "warehouse.no" not in out and "(new table)" not in out, out
    assert _tables(dst) == ["ghost_copy"]
    assert not state.get("pending_clarification")


@pytest.mark.parametrize("reply", ["create", "new", "create new table", "new table", "create table"])
def test_c2_create_without_a_name_asks_for_one(world, reply):
    state, _src, dst = world
    _dest_exists(state, dst)
    out = _turn(state, reply)
    assert out.strip().endswith("What should the new table be called?"), out
    assert state["pending_clarification"]["awaiting"] == "new_table_name"
    assert _tables(dst) == ["ghost_copy"]


@pytest.mark.parametrize("reply", ["yes", "ok", "table", "data", "the", "Create!"])
def test_c2_reserved_words_never_become_table_names(world, reply):
    state, _src, dst = world
    _dest_exists(state, dst)
    out = _turn(state, reply)
    for word in ("yes", "ok", "table", "data", "the", "create"):
        assert f"warehouse.{word} " not in out and f"warehouse.{word}(" not in out, out
    _turn(state, "yes")
    assert set(_tables(dst)) <= {"ghost_copy"}, _tables(dst)


def test_c2_create_then_yes_never_creates_a_table_named_create(world):
    state, _src, dst = world
    _dest_exists(state, dst)
    assert _turn(state, "create").endswith("What should the new table be called?")
    _turn(state, "yes")   # not a name either: asked again
    _turn(state, "yes")
    assert "create" not in _tables(dst) and "yes" not in _tables(dst), _tables(dst)
    assert _tables(dst) == ["ghost_copy"]


def test_c2_a_valid_new_name_still_moves_on_to_the_confirmation(world):
    state, _src, dst = world
    _dest_exists(state, dst)
    ask = _turn(state, "create ghost_copy2")
    assert "Copy clinic.patients → warehouse.ghost_copy2" in ask, ask
    assert _tables(dst) == ["ghost_copy"]


def test_c2_an_invalid_name_is_not_accepted(world):
    state, _src, dst = world
    _dest_exists(state, dst)
    out = _turn(state, "ghost-copy")
    assert "Copy clinic.patients" not in out, out
    _turn(state, "yes")
    assert _tables(dst) == ["ghost_copy"]


@pytest.mark.parametrize("reply,expect", [
    ("no", "cancelled"),
    ("cancel", "cancelled"),
    ("create", "Copy clinic.patients → warehouse.ghost_missing"),
    ("yes", "Copy clinic.patients → warehouse.ghost_missing"),
    ("table", "Reply create to build ghost_missing"),
    ("ok then", "Copy clinic.patients → warehouse.ghost_missing"),
])
def test_c2_use_existing_reply_never_takes_a_reserved_word_as_the_table(world, reply, expect):
    # "into ghost_missing" names a table that is not there: the create-offer slot,
    # whose "reply with the name of an existing table" branch took split()[0].
    state, _src, dst = world
    ask = _turn(state, "copy the patients table to warehouse into ghost_missing")
    assert state["pending_clarification"]["awaiting"] == "confirm_create_table", ask
    out = _turn(state, reply)
    assert expect in out, out
    assert f"{reply.split()[0]} isn't a table" not in out, out
    _turn(state, "yes")
    assert set(_tables(dst)) <= {"ghost_missing"}, _tables(dst)


# ===========================================================================
# rc9 web check, D1: a cancelled transfer came back. After the cancel nothing
# was pending, so the next bare "yes" went to the LLM planner, which rebuilt the
# cancelled plan from the thread history and offered it again ("❓ Copy … →
# wh.c1_old_8"); a second "yes" wrote 5,110 rows. After "no" (e_nono) it did the
# same. The planner LLM is stood in for by one that does exactly that.
# ===========================================================================

_NOTHING_PENDING = ("There's no transfer waiting — the last one was cancelled, so nothing "
                    "was written. Tell me what to copy and where.")
_NOTHING_PENDING_NO = "Okay — there's no transfer waiting; the last one was cancelled and nothing was written."


def _planner_rebuilds_the_old_plan(monkeypatch):
    """A planner LLM that, like rc9's, answers "yes" by re-offering the cancelled plan."""
    calls = []

    def _invoke(msgs, tool_choice, invoke_kwargs):
        calls.append(msgs)
        return AIMessage(content="", tool_calls=[{
            "name": "initiate_transfer", "id": "rebuilt",
            "args": {"destination_alias": "warehouse", "dest_table": "ghost_copy",
                     "create_if_missing": True, "user_prompt": _ASK},
        }])

    monkeypatch.setattr(planner, "llm", MagicMock(), raising=False)
    monkeypatch.setattr(planner, "_invoke_planner_with_recovery", _invoke)
    return calls


@pytest.mark.parametrize("cancel", ["yes but to wh2", "no"])
@pytest.mark.parametrize("web", [False, True], ids=["bare", "with-analysis-context"])
def test_d1_a_bare_yes_after_a_cancel_never_revives_the_old_plan(world, monkeypatch, cancel, web):
    say = (lambda t: _with_context(t, _RC7_LINE)) if web else (lambda t: t)
    calls = _planner_rebuilds_the_old_plan(monkeypatch)
    state, _src, dst = world
    ask = _turn(state, say(_ASK))
    assert state["pending_clarification"]["awaiting"] == "confirm_transfer", ask
    out = _turn(state, say(cancel))
    assert "cancelled" in out and "Nothing was written" in out, out
    assert not state.get("pending_clarification")
    for _ in range(2):   # rc9: the first "yes" re-offered the plan, the second wrote it
        out = _turn(state, say("yes"))
        assert "❓ Copy" not in out and "SUCCESS" not in out, out
        _nothing_written(state, dst)
        assert not state.get("pending_clarification"), state.get("pending_clarification")
        assert out == _NOTHING_PENDING, out
    assert calls == [], "a bare yes after a cancel must not reach the planner LLM"


@pytest.mark.parametrize("web", [False, True], ids=["bare", "with-analysis-context"])
def test_d1_a_bare_no_after_a_cancel_is_answered_not_planned(world, monkeypatch, web):
    # e_nono: "no" cancelled, and a second "no" made the planner re-offer the transfer.
    say = (lambda t: _with_context(t, _RC7_LINE)) if web else (lambda t: t)
    calls = _planner_rebuilds_the_old_plan(monkeypatch)
    state, _src, dst = world
    _turn(state, say(_ASK))
    _turn(state, say("no"))
    out = _turn(state, say("no"))
    assert "❓ Copy" not in out and "⚠️ Table" not in out, out
    assert out == _NOTHING_PENDING_NO, out
    out = _turn(state, say("yes"))
    assert "❓ Copy" not in out and "SUCCESS" not in out, out
    _nothing_written(state, dst)
    assert out == _NOTHING_PENDING, out
    assert calls == []


@pytest.mark.parametrize("cancel", ["yes but to wh2", "no"])
def test_d1_a_real_request_after_a_cancel_gets_its_own_confirmation(world, cancel):
    state, src, dst = world
    _turn(state, _ASK)
    _turn(state, cancel)
    out = _turn(state, "Transfer the patients table to wh2 as fresh_copy.")
    assert "❓ Copy" in out and "wh2" in out and "fresh_copy" in out, out
    assert "ghost_copy" not in out, out
    _nothing_written(state, dst)
    out = _turn(state, "yes")
    assert "SUCCESS" in out, out
    assert _count(state["_wh2_path"], "fresh_copy") == _count(src, "patients") == PATIENTS
    assert _tables(dst) == []


def test_d1_a_yes_answering_a_question_asked_since_the_cancel_is_the_planners(world, monkeypatch):
    # #570's marker expired once anything else was said, and rc10 (e_expiry) showed
    # why that was too soon (see the rc10 tests below). What a bare yes answers now
    # is the last question the assistant asked: here, one asked after the cancel.
    calls = _planner_rebuilds_the_old_plan(monkeypatch)
    state, _src, dst = world
    _turn(state, _ASK)
    _turn(state, "no")
    state["messages"] = list(state["messages"]) + [
        HumanMessage(content="what is the mean age?"),
        AIMessage(content="The mean age is 54.5.\n\nWant it broken down by gender?")]
    _turn(state, "yes")
    assert len(calls) == 1


@pytest.mark.parametrize("reply, consent, negative", [
    ("yes", True, False), ("Yes please!", True, False), ("👍 ok", True, False),
    ("no", False, True), ("no thanks", False, True), ("cancel", False, True),
    ("yes but to wh2", False, False), ("no, copy it to wh2", False, False),
])
def test_d1_bare_reply_classification(reply, consent, negative):
    assert planner._dta_reply_is_consent(reply) is consent
    assert planner._dta_reply_is_bare_negative(reply) is negative


def test_d1_the_canned_replies_are_the_ones_the_planner_sends():
    assert planner._DTA_NOTHING_PENDING == _NOTHING_PENDING
    assert planner._DTA_NOTHING_PENDING_NO == _NOTHING_PENDING_NO



# ===========================================================================
# rc10 web check, D1 e_expiry: a write planned outside the transfer path.
#
# cancel -> "What is the stroke rate by gender?" -> "yes" -> "yes". #570's
# marker had expired (the stroke answer came in between), so the planner took
# the second yes as a job: generate_code, a plan about the stroke question,
# and a coder blueprint -- built from the history and the memory plane's
# "Transfer patients table to e2ewhgroq as d1_e" hint -- that appended the
# table to the warehouse with to_sql. No confirmation was shown; only the
# model's choice to raise kept the rows out.
# ===========================================================================

from app.agents.db_write_guard import DB_WRITE_REFUSAL  # noqa: E402

_STROKE_Q = "What is the stroke rate by gender?"
_STROKE_A = ("**Stroke rate by gender** (overall 0.0487):\n\n| gender | stroke rate | rows with a value |\n"
             "|---|---|---|\n| Female | 0.0471 | 2,994 |\n| Male | 0.0511 | 2,115 |\n| Other | 0.0000 | 1 |\n\n"
             "Computed by the database over all 5,110 rows of `patients`, so these figures are exact, "
             "not a sample estimate.")
# The plan rc10's planner wrote for the second yes (planner_definition in q4.json).
_RC10_PLAN = "1. Group by gender and count rows.\n2. Sum strokes per gender."
_WRITE_PLAN = ("1. Open a connection to the target database warehouse.\n"
               "2. Write the patients rows to the target table ghost_copy with "
               "df.to_sql(name='ghost_copy', con=tgt_conn, if_exists='append').")


def _planner_like_rc10(monkeypatch, plan_text):
    """A planner LLM that answers the stroke question and turns any other turn into generate_code."""
    calls = []

    def _invoke(msgs, tool_choice, invoke_kwargs):
        calls.append(msgs)
        last = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
        if last is not None and "stroke rate" in str(last.content):
            return AIMessage(content="", tool_calls=[{
                "name": "respond_to_user", "id": "a", "args": {"response_text": _STROKE_A}}])
        return AIMessage(content="", tool_calls=[{"name": "generate_code", "id": "g", "args": {}}])

    llm = MagicMock(return_value=AIMessage(content=plan_text))
    llm.invoke.return_value = AIMessage(content=plan_text)
    monkeypatch.setattr(planner, "llm", llm, raising=False)
    monkeypatch.setattr(planner, "_invoke_planner_with_recovery", _invoke)
    monkeypatch.setattr(planner, "_classify_output_replaces_dataset", lambda **kw: False)
    return calls


def _turn_reply(state, text: str):
    """Like _turn, but a turn that adds no assistant message (a coder-bound one) returns None."""
    before = len(state.get("messages") or []) + 1
    state["messages"] = list(state.get("messages") or []) + [HumanMessage(content=text)]
    state.update(planner.plan_etl_job(state))
    added = [m for m in state["messages"][before:] if isinstance(m, AIMessage)]
    return added[-1].content if added else None


def _no_coder_step(state):
    assert not state.get("ready_to_summarize"), "the turn was sent on to the coder"
    assert not state.get("ready_to_code"), "the turn was sent on to the coder"
    assert not state.get("plan"), state.get("plan")


@pytest.mark.parametrize("plan_text", [_RC10_PLAN, _WRITE_PLAN], ids=["rc10-plan", "write-plan"])
@pytest.mark.parametrize("web", [False, True], ids=["bare", "with-analysis-context"])
def test_rc10_cancel_question_yes_yes_writes_nothing_and_never_codes(world, monkeypatch, plan_text, web):
    say = (lambda t: _with_context(t, _RC7_LINE)) if web else (lambda t: t)
    state, _src, dst = world
    ask = _turn(state, say(_ASK))
    assert state["pending_clarification"]["awaiting"] == "confirm_transfer", ask
    out = _turn(state, say("no"))
    assert "cancelled" in out and "Nothing was written" in out, out
    calls = _planner_like_rc10(monkeypatch, plan_text)
    out = _turn(state, say(_STROKE_Q))
    assert out == _STROKE_A, out
    assert len(calls) == 1
    for _ in range(2):
        out = _turn_reply(state, say("yes"))
        _no_coder_step(state)
        assert out and "❓ Copy" not in out and "SUCCESS" not in out, out
        _nothing_written(state, dst)
        assert not state.get("pending_clarification"), state.get("pending_clarification")
        assert out == _NOTHING_PENDING, out
    assert len(calls) == 1, "a bare yes aimed at the cancelled transfer must not reach the planner LLM"


def test_rc10_a_bare_no_later_on_is_answered_too(world, monkeypatch):
    state, _src, dst = world
    _turn(state, _ASK)
    _turn(state, "no")
    calls = _planner_like_rc10(monkeypatch, _RC10_PLAN)
    _turn(state, _STROKE_Q)
    out = _turn_reply(state, "no")
    _no_coder_step(state)
    assert out == _NOTHING_PENDING_NO, out
    _nothing_written(state, dst)
    assert len(calls) == 1


def test_rc10_an_offer_that_is_no_longer_pending_is_not_consented_to(world, monkeypatch):
    # The offer is in the history but nothing is pending (a lost wizard slot, a
    # restarted server): a later bare yes is not consent to it.
    state, _src, dst = world
    _turn(state, _ASK)
    state["pending_clarification"] = None
    calls = _planner_like_rc10(monkeypatch, _WRITE_PLAN)
    _turn(state, _STROKE_Q)
    out = _turn_reply(state, "yes")
    _no_coder_step(state)
    assert out == planner._DTA_NOTHING_WAITING, out
    _nothing_written(state, dst)
    assert len(calls) == 1


def test_rc10_a_real_request_after_all_that_still_gets_its_confirmation(world, monkeypatch):
    state, src, dst = world
    _turn(state, _ASK)
    _turn(state, "no")
    _planner_like_rc10(monkeypatch, _RC10_PLAN)
    _turn(state, _STROKE_Q)
    _turn_reply(state, "yes")
    out = _turn_reply(state, "Transfer the patients table to wh2 as fresh_copy.")
    assert out, "the transfer request got no reply"
    assert "❓ Copy" in out and "fresh_copy" in out, out
    out = _turn(state, "yes")
    assert "SUCCESS" in out, out
    assert _count(state["_wh2_path"], "fresh_copy") == _count(src, "patients") == PATIENTS
    assert _tables(dst) == []


# ---- layer 1: the planner never sends a write plan to the coder ----

@pytest.mark.parametrize("plan_text", [
    _WRITE_PLAN,
    "Compute the stroke rate per gender and save the summary into the warehouse database.",
    "1. Create a SQLAlchemy engine for the destination.\n2. Append the rows.",
    "Run INSERT INTO ghost_copy SELECT * FROM patients on the target.",
])
def test_planner_turns_a_write_plan_into_a_question_never_a_coder_step(world, monkeypatch, plan_text):
    state, _src, dst = world
    calls = _planner_like_rc10(monkeypatch, plan_text)
    # No transfer in the history and no destination the fast path can parse:
    # only the plan says "write".
    out = _turn_reply(state, "Put the stroke summary somewhere permanent in our database")
    assert len(calls) == 1
    _no_coder_step(state)
    assert out == DB_WRITE_REFUSAL, out
    _no_coder_step(state)
    _nothing_written(state, dst)


def test_planner_still_sends_an_analysis_plan_to_the_coder(world, monkeypatch):
    state, _src, _dst = world
    _planner_like_rc10(monkeypatch, _RC10_PLAN)
    state["messages"] = [HumanMessage(content="Count rows and strokes per gender")]
    state.update(planner.plan_etl_job(state))   # a coder-bound turn adds no reply here
    assert state.get("ready_to_summarize") is True
    assert state.get("plan") == _RC10_PLAN


# ---- rc11: a read plan reaches the coder -------------------------------------
# rc11's planner refused read plans that named a driver or staged a scratch
# table ('psycopg2', 'sqlalchemy', 'CREATE TABLE'). The plans are the
# regenerated rweb11 ones in tests/fixtures/rc11_refused_read_plans.json (see
# tests/test_no_db_writes_outside_transfer.py for how they were produced).

_RC11_PLANS = json.loads((Path(__file__).parent / "fixtures" / "rc11_refused_read_plans.json")
                          .read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _RC11_PLANS,
                         ids=[f"{c['rweb11_turn']}-{c['logged_match']}" for c in _RC11_PLANS])
def test_rc11_planner_sends_a_read_plan_to_the_coder(world, monkeypatch, case):
    state, _src, dst = world
    _planner_like_rc10(monkeypatch, case["plan"])
    state["messages"] = [HumanMessage(content="average age of patients with stroke")]
    state.update(planner.plan_etl_job(state))
    assert state.get("ready_to_summarize") is True, state.get("messages")[-1]
    assert state.get("plan") == case["plan"]
    assert not any(getattr(m, "content", None) == DB_WRITE_REFUSAL for m in state["messages"])
    _nothing_written(state, dst)


# Registry credentials these tests seed live in a stand-in sealed store, not in
# state: the planner opens them by handle after the confirm gate (tests/dta_sealed.py).
try:  # tests/ as a package, or tests/ on sys.path
    from tests import dta_sealed as _dta_sealed
except ImportError:  # pragma: no cover
    import dta_sealed as _dta_sealed


@pytest.fixture(autouse=True)
def _sealed_store(monkeypatch):
    return _dta_sealed.install(monkeypatch)
