"""parse_db_destination: the shared reader of "<connection> + <table>" clauses.

The same module ships on 1.6 and 2.x; these rows are the contract both planners
rely on. Planner-level behaviour is in test_dta_destination_phrasings.py.
"""
import pytest

from app.agents.data_transfer_agent.destination_parser import parse_db_destination

# (text, connection, schema, table, create_new, dotted)
ROWS = [
    ("Transfer to churn_dst table customers_copy", "churn_dst", None, "customers_copy", False, False),
    ("to X table Y", "X", None, "Y", False, False),
    ("to targetdb new table X", "targetdb", None, "X", True, False),
    ("to targetdb a fresh table X", "targetdb", None, "X", True, False),
    ("into X.Y", "X", None, "Y", False, True),
    ("into X.S.T", "X", "S", "T", False, True),
    ("to X new table Y", "X", None, "Y", True, False),
    ("to X table sales.orders", "X", "sales", "orders", False, False),
    ('to "churn dst" table "Customers Copy"', "churn dst", None, "Customers Copy", False, False),
    ('to X table "sales"."Customers Copy"', "X", "sales", "Customers Copy", False, False),
    ("to `X` table `Y`", "X", None, "Y", False, False),
    ("to My Warehouse table orders", "My Warehouse", None, "orders", False, False),
    ("to db.example.com table x", "db.example.com", None, "x", False, False),
    ("to a new table Y in X", "X", None, "Y", True, False),
    ("into table Y on the X database", "X", None, "Y", False, False),
    ("from src table a to dst table b", "dst", None, "b", False, False),
]


@pytest.mark.parametrize("text,conn,schema,table,new,dotted", ROWS, ids=[r[0] for r in ROWS])
def test_reads_the_clause_whole(text, conn, schema, table, new, dotted):
    d = parse_db_destination(text)
    assert d is not None
    assert (d.connection, d.schema, d.table, d.create_new, d.dotted) == (conn, schema, table, new, dotted)
    assert d.qualified_table == (f"{schema}.{table}" if schema else table)


def test_dotted_form_keeps_the_raw_token_for_a_registry_check():
    d = parse_db_destination("copy into prod.orders")
    assert d.raw == "prod.orders"


@pytest.mark.parametrize("text", [
    "move revenue to profit",
    "let's move to the next table please",
    "to gs://bucket/out.csv",
    "to out.csv",
    "into out.csv",
    "into churn_dst.customers_copy.csv",
    "to my_bucket/folder/x.parquet",
    "to db1.internal",                        # "to" + dots: a host-style name, not <conn>.<table>
    "to churn_dst",
    "to churn_dst table",
    "to churn_dst table with only active rows",
    "to the table this",
    "send to me the table results",
    "to my db table x",
    "",
])
def test_negatives(text):
    assert parse_db_destination(text) is None
