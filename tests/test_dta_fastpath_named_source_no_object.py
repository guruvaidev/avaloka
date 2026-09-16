"""'transfer from <X> to <dest>' with no object/table clause must not drop X.

The tabled parser needs a literal "table" keyword and the named-source parser needs
a source OBJECT, so this shape fell to _extract_transfer_destination, whose legacy
grammar consumed the "from <X>" and discarded it — the transfer then ran on the
implicit ACTIVE dataset. User names customers_db, gets whatever they were looking
at, reported as success. Same silent-swap class as the LLM-door bug (cb77e2b), one
door earlier.

Now the fast-path threads X through as source_alias, and the handler adds one
fallback rung: when X is not a connection but IS the analyzed table (the legacy
"transfer from <my_table> to <dest>" phrasing that used to work only by accident),
the implicit source is used with that table — otherwise the ladder's own
"not registered" message fires. Never a silent guess.
"""
from unittest import mock

from langchain_core.messages import HumanMessage

import app.agents.planner as P
from app.agents.planner import _implicit_source_matching_alias, plan_etl_job


def _route(prompt):
    state = {
        "messages": [HumanMessage(content=prompt)],
        "user_prompt": "",
        "dta_database_registry": {},
    }
    with mock.patch("app.agents.planner._handle_initiate_transfer", return_value="HANDLED") as m, \
         mock.patch("app.agents.planner._seed_ray_fields"), \
         mock.patch("app.agents.planner._seed_infra_request_if_needed"):
        plan_etl_job(state)
    m.assert_called_once()
    return m.call_args[0][1]


def test_named_source_without_object_is_threaded_not_dropped():
    p = _route("transfer from customers_db to warehouse")
    assert p.source_alias == "customers_db"
    assert p.destination_alias == "warehouse"


def test_named_source_with_dest_table_keeps_both():
    p = _route("transfer from customers_db to warehouse into table orders")
    assert p.source_alias == "customers_db"
    assert p.destination_alias == "warehouse"
    assert p.dest_table == "orders"


def test_plain_implicit_transfer_still_has_no_source():
    p = _route("transfer to warehouse into table orders")
    assert p.source_alias is None
    assert p.destination_alias == "warehouse"


def test_file_source_token_is_not_treated_as_connection():
    # "from in.csv to out.json" is the object-only form — no named connection.
    p = _route("transfer from in.csv to gcs_backup")
    assert p.source_alias is None
    assert p.destination_alias == "gcs_backup"


# ── the handler fallback rung ─────────────────────────────────────────────────

def _implicit(entry_table=None, obj=None):
    return {"alias": "conn1", "entry": {"db_type": "postgres", "source_table": entry_table}, "object": obj}


def test_alias_matching_analyzed_table_uses_implicit_source(monkeypatch):
    monkeypatch.setattr(P, "_resolve_implicit_source", lambda s: _implicit(entry_table="adult_income"))
    out = _implicit_source_matching_alias({}, "adult_income")
    assert out is not None and out["alias"] == "conn1"
    # case-insensitive: users type table names loosely
    assert _implicit_source_matching_alias({}, "Adult_Income") is not None


def test_alias_matching_cloud_object_stem_uses_implicit_source(monkeypatch):
    monkeypatch.setattr(P, "_resolve_implicit_source",
                        lambda s: _implicit(obj="data/patient_data.csv"))
    assert _implicit_source_matching_alias({}, "patient_data.csv") is not None
    assert _implicit_source_matching_alias({}, "patient_data") is not None


def test_unrelated_alias_never_gets_the_implicit_source(monkeypatch):
    """The whole point: an unresolvable token must fail loudly, not silently swap."""
    monkeypatch.setattr(P, "_resolve_implicit_source", lambda s: _implicit(entry_table="adult_income"))
    assert _implicit_source_matching_alias({}, "customers_db") is None


def test_no_active_source_returns_none(monkeypatch):
    monkeypatch.setattr(P, "_resolve_implicit_source", lambda s: "no active source")
    assert _implicit_source_matching_alias({}, "adult_income") is None
