"""
Supabase persistence migration has the UNIQUE index the upserts need.

The upsert helpers call .upsert(row, on_conflict='dataset_id,phase'), which
PostgREST rejects unless a UNIQUE index/constraint over exactly those columns
exists. The original MIGRATION_SQL created only a PK on `id` and non-unique
indexes, so every profile/sample persist was a silent no-op.

The structural test needs no database. The live-Postgres test is opt-in via
TEST_PG_DSN (e.g. "host=localhost port=55432 dbname=test user=postgres
password=pw") and is skipped otherwise.
"""

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents import sampling_persistence as sp

TABLES = {"dataset_profiles", "dataset_samples"}


def _unique_indexes_by_table():
    """Parse MIGRATION_SQL and return {table: set(frozenset(cols))} for UNIQUE indexes."""
    import sqlglot
    from sqlglot import exp

    out = {}
    for stmt in sqlglot.parse(sp.MIGRATION_SQL, read="postgres"):
        if not isinstance(stmt, exp.Create):
            continue
        if (stmt.args.get("kind") or "").upper() != "INDEX" or not stmt.args.get("unique"):
            continue
        idx = stmt.this
        table = idx.args["table"].name if idx.args.get("table") else None
        cols = frozenset(c.name for c in idx.find_all(exp.Column))
        out.setdefault(table, set()).add(cols)
    return out


def test_migration_is_valid_postgres():
    import sqlglot

    stmts = [s for s in sqlglot.parse(sp.MIGRATION_SQL, read="postgres") if s]
    assert stmts, "MIGRATION_SQL parsed to nothing"


def test_on_conflict_columns_have_matching_unique_index():
    """Every on_conflict=... used in the module must be backed by a UNIQUE index."""
    src = Path(sp.__file__).read_text()
    conflicts = re.findall(r"on_conflict=[\"']([^\"']+)[\"']", src)
    assert conflicts, "expected at least one on_conflict= in the persistence module"

    unique_idx = _unique_indexes_by_table()
    for cols_csv in conflicts:
        cols = frozenset(c.strip() for c in cols_csv.split(","))
        matched = any(cols in idxs for idxs in unique_idx.values())
        assert matched, f"no UNIQUE index matches on_conflict='{cols_csv}'"


def test_both_tables_have_unique_dataset_id_phase():
    unique_idx = _unique_indexes_by_table()
    for table in TABLES:
        assert frozenset({"dataset_id", "phase"}) in unique_idx.get(table, set()), (
            f"{table} is missing a UNIQUE index on (dataset_id, phase)"
        )


@pytest.mark.skipif(not os.getenv("TEST_PG_DSN"), reason="set TEST_PG_DSN to run the live-Postgres test")
def test_migration_applies_and_upsert_works_on_postgres():
    import psycopg2

    conn = psycopg2.connect(os.environ["TEST_PG_DSN"])
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute("DROP TABLE IF EXISTS dataset_profiles, dataset_samples CASCADE")
        cur.execute(sp.MIGRATION_SQL)
        cur.execute(sp.MIGRATION_SQL)  # idempotent

        for table in TABLES:
            for _ in range(2):  # insert, then update-in-place
                cur.execute(
                    f"INSERT INTO {table} (dataset_id, phase) VALUES ('ds1','full') "
                    "ON CONFLICT (dataset_id, phase) DO UPDATE SET dataset_id = EXCLUDED.dataset_id"
                )
            cur.execute(f"SELECT count(*) FROM {table} WHERE dataset_id='ds1' AND phase='full'")
            assert cur.fetchone()[0] == 1
    finally:
        cur.close()
        conn.close()
