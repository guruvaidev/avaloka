"""A transfer's destination clause names a connection AND a table in it.

The demo turn "Transfer to churn_dst table customers_copy" answered "Table
churn_dst doesn't exist in churn_dst": the table extractor read "to <X> table"
as a table named X. On 2.x, "to targetdb new table X" named a connection
"targetdb new", because the destination extractor stopped only at " table".
Both are the same clause read in two halves; it is now read whole.

Every row goes through the planner's own fast-path (``plan_etl_job`` with the
transfer handler mocked), so the assertion is on the parameters a real turn
would hand to the transfer, not on a helper.
"""
from unittest import mock

import pytest
from langchain_core.messages import HumanMessage

from app.agents.planner import (
    _extract_dest_table_from_prompt,
    _extract_transfer_destination,
    _normalize_llm_transfer_params,
    InitiateTransferParams,
    plan_etl_job,
)

# The demo appends this to the user's words before the planner sees them.
_CONTEXT = "\n\n[Analysis context] Current fidelity: portfolio_samples; Selected sample: none"


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
    if not m.called:
        return None
    return m.call_args[0][1]


# (prompt, destination connection, destination table)
POSITIVE = [
    # the demo turn, verbatim, with and without the analysis-context suffix
    ("Transfer to churn_dst table customers_copy", "churn_dst", "customers_copy"),
    ("Transfer to churn_dst table customers_copy" + _CONTEXT, "churn_dst", "customers_copy"),
    ("Transfer to churn_dst table customers_copy.", "churn_dst", "customers_copy"),
    # "to X table Y" and close variants
    ("copy the data to warehouse table orders_copy", "warehouse", "orders_copy"),
    ("transfer to churn_dst, table customers_copy", "churn_dst", "customers_copy"),
    ("transfer to churn_dst as table customers_copy", "churn_dst", "customers_copy"),
    ("transfer to the churn_dst database table customers_copy", "churn_dst", "customers_copy"),
    ("transfer it to churn_dst db, table customers_copy", "churn_dst", "customers_copy"),
    # "to X new table Y" (the 2.x report)
    ("transfer to targetdb new table X", "targetdb", "X"),
    ("transfer to targetdb a new table orders_2024", "targetdb", "orders_2024"),
    # "into X.Y"
    ("transfer into churn_dst.customers_copy", "churn_dst", "customers_copy"),
    ("copy it into churn_dst.customers_copy" + _CONTEXT, "churn_dst", "customers_copy"),
    # schema-qualified
    ("transfer to churn_dst table sales.customers_copy", "churn_dst", "sales.customers_copy"),
    ("transfer into churn_dst.sales.customers_copy", "churn_dst", "sales.customers_copy"),
    # quoted names
    ('transfer to "churn dst" table "Customers Copy"', "churn dst", "Customers Copy"),
    ("transfer to `churn_dst` table `customers_copy`", "churn_dst", "customers_copy"),
    ('transfer to churn_dst table "sales"."Customers Copy"', "churn_dst", "sales.Customers Copy"),
    # table first
    ("transfer to a new table customers_copy in churn_dst", "churn_dst", "customers_copy"),
    # with a named source in front
    ("transfer from churn_src to churn_dst table customers_copy", "churn_dst", "customers_copy"),
]


@pytest.mark.parametrize("prompt,conn,table", POSITIVE, ids=[p[0][:50] for p in POSITIVE])
def test_fast_path_reads_connection_and_table(prompt, conn, table):
    p = _route(prompt)
    assert p is not None, "the transfer fast-path did not fire"
    assert p.destination_alias == conn
    assert p.dest_table == table


@pytest.mark.parametrize("prompt,conn,table", POSITIVE, ids=[p[0][:50] for p in POSITIVE])
def test_extractors_agree_with_the_fast_path(prompt, conn, table):
    parsed = _extract_transfer_destination(prompt)
    assert parsed is not None and parsed[0] == conn
    assert _extract_dest_table_from_prompt(prompt) == table


def test_the_demo_turn_is_never_the_connection_as_a_table():
    p = _route("Transfer to churn_dst table customers_copy" + _CONTEXT)
    assert p.dest_table != p.destination_alias


def test_new_table_is_a_create_request():
    p = _route("transfer to targetdb new table X")
    assert p.destination_alias == "targetdb"
    # the handler's own intent check reads "new table X" from the prompt
    from app.agents.planner import _dta_table_intent
    assert _dta_table_intent(p.user_prompt) == ("new", "X")


def test_llm_door_does_not_keep_the_connection_as_the_table():
    """The LLM path re-derives the table from the user's words, not the model's copy of the alias."""
    params = InitiateTransferParams(destination_alias="churn_dst", dest_table="churn_dst",
                                    user_prompt="summary by the model")
    out = _normalize_llm_transfer_params(params, "Transfer to churn_dst table customers_copy")
    assert out.dest_table == "customers_copy"


# Phrasings that must keep their old meaning: (prompt, destination, table or None)
UNCHANGED = [
    ("transfer to warehouse into table orders", "warehouse", "orders"),
    ("transfer to warehouse", "warehouse", None),
    ("transfer to gcs_backup", "gcs_backup", None),
    ("transfer to my_bucket, to file out/result.json", "my_bucket", None),
    ("transfer to churn_dst to table customers_copy", "churn_dst", "customers_copy"),
]


@pytest.mark.parametrize("prompt,conn,table", UNCHANGED, ids=[p[0][:50] for p in UNCHANGED])
def test_older_phrasings_keep_their_meaning(prompt, conn, table):
    p = _route(prompt)
    assert p is not None
    assert p.destination_alias == conn
    assert p.dest_table == table


# Not a database destination at all: the fast-path must not read one.
NEGATIVE = [
    "move revenue to profit",
    "let's move to the next table please",
    "add the revenue column to the sales table",
]


@pytest.mark.parametrize("prompt", NEGATIVE)
def test_non_transfers_do_not_become_transfers(prompt):
    assert _route(prompt) is None


def test_a_file_name_is_not_connection_dot_table():
    p = _route("transfer into churn_dst.customers_copy.csv")
    assert p is None or (p.destination_alias, p.dest_table) != ("churn_dst", "customers_copy.csv")
    assert p is None or p.dest_table != "csv"


def test_a_grammar_word_is_never_a_table_name():
    p = _route("transfer to churn_dst table with only active rows")
    assert p is not None and p.destination_alias == "churn_dst"
    assert p.dest_table != "with"
