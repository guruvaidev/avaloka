"""Database answers are honest about sampling (web e2e, 2.0).

The e2e run answered "stroke rate is 0.50 with hypertension, 0.21 without" for
a table whose true rates are 0.1325 / 0.0397: the API read
``SELECT * FROM patients LIMIT 1000`` and the table is stored with every
stroke=1 row first. These tests build that shape -- a real SQLite table whose
head is all positives -- and run the sampler and the aggregate pushdown against
it through an MCP-shaped runner that wraps every statement exactly as the MCP
server does (``SELECT * FROM (<sql>) AS _mcp_sub LIMIT <max_rows>``).
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from pathlib import Path

import pytest

from app.core.db_pushdown import parse_aggregate_intent, render_pushdown_answer, run_pushdown
from app.core.db_sampling import (
    DbSample,
    DbSampleError,
    draw_table_sample,
    random_sample_sql,
    sample_scope,
)

N_ROWS = 5000
N_POSITIVE = 250       # 5% stroke, all stored first
SAMPLE_N = 1000
QUESTION = "What is the stroke rate for patients with hypertension versus without?"


def _hypertension(i: int) -> int:
    return 1 if (i < 150 or i % 9 == 0) else 0


def make_sorted_patients(path: Path) -> dict:
    """The e2e shape: positives first. Returns the full-table truth."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, age REAL, hypertension INTEGER, stroke INTEGER)")
    con.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?)",
        [(i, 20 + (i * 7) % 60, _hypertension(i), 1 if i < N_POSITIVE else 0) for i in range(N_ROWS)],
    )
    con.commit()
    con.close()
    by = {0: [0, 0], 1: [0, 0]}
    for i in range(N_ROWS):
        g = _hypertension(i)
        by[g][0] += 1 if i < N_POSITIVE else 0
        by[g][1] += 1
    return {
        "rate": {g: pos / n for g, (pos, n) in by.items()},
        "n": {g: n for g, (_pos, n) in by.items()},
        "overall": N_POSITIVE / N_ROWS,
        "head_overall": min(N_POSITIVE, SAMPLE_N) / SAMPLE_N,
    }


def mcp_runner(path: Path, seen: list | None = None, max_rows: int = 10_000, dialect: str = "sqlite"):
    async def run(sql: str) -> dict:
        if seen is not None:
            seen.append(sql)
        con = sqlite3.connect(path)
        try:
            cur = con.execute(f"SELECT * FROM ({sql}) AS _mcp_sub LIMIT {max_rows}")
            return {
                "columns": [d[0] for d in cur.description],
                "rows": [list(r) for r in cur.fetchall()],
                "meta": {"database_type": dialect, "row_limit": max_rows},
            }
        except Exception as exc:  # the MCP tool reports errors, it does not raise
            return {"error": f"{type(exc).__name__}: {exc}", "meta": {}}
        finally:
            con.close()
    return run


@pytest.fixture
def patients(tmp_path):
    path = tmp_path / "clinic.db"
    return path, make_sorted_patients(path)


def test_fixture_head_is_not_the_table(patients):
    """The fixture bites: a head sample of it is far from the truth."""
    path, truth = patients
    con = sqlite3.connect(path)
    head = [r[0] for r in con.execute(f"SELECT stroke FROM patients LIMIT {SAMPLE_N}")]
    con.close()
    assert sum(head) / len(head) == truth["head_overall"] == 0.25
    assert abs(truth["overall"] - 0.05) < 1e-12


# ---- (a) pushdown: exact on the full table ------------------------------------

def test_pushdown_rate_by_group_equals_full_table_truth(patients):
    path, truth = patients
    intent = parse_aggregate_intent(QUESTION, ["id", "age", "hypertension", "stroke"])
    assert (intent.operation, intent.measure, intent.group_by) == ("rate", "stroke", "hypertension")

    result = asyncio.run(run_pushdown(mcp_runner(path), "patients", intent, "sqlite"))
    assert result is not None
    got = {g: (fig, n) for g, fig, n in result.groups}
    for g in (0, 1):
        assert got[g][0] == pytest.approx(truth["rate"][g], abs=1e-12)
        assert got[g][1] == truth["n"][g]
    assert result.overall == pytest.approx(truth["overall"], abs=1e-12)
    assert result.total_rows == N_ROWS

    text = render_pushdown_answer(result, "patients")
    assert f"{truth['rate'][1]:.4f}" in text and f"{truth['rate'][0]:.4f}" in text
    assert f"all {N_ROWS:,} rows" in text and "exact" in text


@pytest.mark.parametrize("question", [
    "stroke rate for patients over 60 by hypertension",   # a filter
    "stroke rate by hypertension and age",                 # two groupings
    "Transfer the patients table to the warehouse",        # not an aggregate
    "median age by hypertension",                          # not in the family
])
def test_pushdown_declines_what_it_cannot_answer_exactly(question):
    assert parse_aggregate_intent(question, ["id", "age", "hypertension", "stroke"]) is None


def test_pushdown_rate_refuses_a_non_binary_measure(patients):
    path, _truth = patients
    intent = parse_aggregate_intent("age rate by hypertension", ["id", "age", "hypertension", "stroke"])
    assert intent is not None and intent.measure == "age"
    assert asyncio.run(run_pushdown(mcp_runner(path), "patients", intent, "sqlite")) is None


# ---- (a)+(c) random sample: labelled and near the truth ------------------------

def test_random_sample_is_labelled_n_of_m_and_within_tolerance(patients):
    path, truth = patients
    cols, rows, sample = asyncio.run(draw_table_sample(mcp_runner(path), "patients", SAMPLE_N, table="patients"))
    assert sample.method == "random"
    assert (sample.n, sample.total, sample.total_is_estimate) == (SAMPLE_N, N_ROWS, False)
    assert sample.is_partial
    assert sample_scope(sample) == f"a random sample of {SAMPLE_N:,} of {N_ROWS:,} rows"

    stroke = cols.index("stroke")
    rate = sum(r[stroke] for r in rows) / len(rows)
    # sd of a 1,000-row sample at p=0.05 is ~0.007; the head would read 0.25.
    assert abs(rate - truth["overall"]) < 0.035, rate
    ids = [r[cols.index("id")] for r in rows]
    assert ids != list(range(SAMPLE_N)), "the sample must not be the first rows"


def test_small_table_is_read_whole_and_called_exact(tmp_path):
    path = tmp_path / "small.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (id INTEGER, v INTEGER)")
    con.executemany("INSERT INTO t VALUES (?, ?)", [(i, i % 2) for i in range(40)])
    con.commit()
    con.close()
    _cols, rows, sample = asyncio.run(draw_table_sample(mcp_runner(path), "t", SAMPLE_N, table="t"))
    assert (sample.method, sample.n, sample.total, sample.is_partial) == ("full", 40, 40, False)
    assert len(rows) == 40


def test_row_cap_on_the_server_does_not_turn_a_full_read_into_a_head(tmp_path):
    """COUNT says it fits, but the MCP server's own max_rows cuts the read: sample instead."""
    path = tmp_path / "capped.db"
    make_sorted_patients(path)
    seen: list = []
    _cols, rows, sample = asyncio.run(
        draw_table_sample(mcp_runner(path, seen, max_rows=300), "patients", 10_000, table="patients"))
    assert sample.method == "random" and sample.n == 300 and sample.total == N_ROWS
    assert "ORDER BY random() LIMIT 300" in seen[-1]


# ---- (b) never a LIMIT-N head ---------------------------------------------------

_HEAD = re.compile(r"^\s*SELECT \* FROM \S+ LIMIT \d+\s*$", re.IGNORECASE)


def test_sampler_never_sends_a_head_query(patients):
    path, _truth = patients
    seen: list = []
    asyncio.run(draw_table_sample(mcp_runner(path, seen), "patients", SAMPLE_N, table="patients"))
    assert seen, "the sampler ran no query"
    for sql in seen:
        assert not _HEAD.match(sql), sql
        if "LIMIT" in sql.upper():
            assert "ORDER BY random()" in sql, sql


@pytest.mark.parametrize("dialect,marker", [
    ("postgresql", "ORDER BY random()"), ("mysql", "ORDER BY RAND()"), ("mariadb", "ORDER BY RAND()"),
    ("sqlite", "ORDER BY random()"), ("mssql", "ORDER BY NEWID()"), ("oracle", "ORDER BY DBMS_RANDOM.VALUE"),
])
@pytest.mark.parametrize("total", [5_000, 50_000_000])
def test_every_dialect_orders_at_random_before_its_bound(dialect, marker, total):
    sql = random_sample_sql("patients", SAMPLE_N, dialect, total_rows=total)
    assert marker in sql
    assert not _HEAD.match(sql)
    if total > 1_000_000:
        # Large tables are pre-filtered at random, not cut at the head.
        assert any(k in sql for k in ("TABLESAMPLE BERNOULLI", "RAND() <", "random()", "CHECKSUM(NEWID())", "SAMPLE ("))


def test_unknown_dialect_uses_a_keyed_sample_not_the_head(patients):
    path, truth = patients
    seen: list = []
    cols, rows, sample = asyncio.run(
        draw_table_sample(mcp_runner(path, seen, dialect="exotic"), "patients", SAMPLE_N, table="patients"))
    assert sample.method == "keyed" and sample.total == N_ROWS
    assert any("MOD(id," in s for s in seen)
    assert all(not _HEAD.match(s) for s in seen)
    rate = sum(r[cols.index("stroke")] for r in rows) / len(rows)
    assert abs(rate - truth["overall"]) < 0.035
    assert sample_scope(sample).startswith("a key-spaced sample of")


def test_no_honest_sample_raises_rather_than_reading_the_head(tmp_path):
    path = tmp_path / "nokey.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (label TEXT)")
    con.executemany("INSERT INTO t VALUES (?)", [(str(i),) for i in range(3000)])
    con.commit()
    con.close()
    seen: list = []
    with pytest.raises(DbSampleError):
        asyncio.run(draw_table_sample(mcp_runner(path, seen, dialect="exotic"), "t", SAMPLE_N, table="t"))
    assert all(not _HEAD.match(s) for s in seen)


# ---- (c) N of M wording ---------------------------------------------------------

def test_scope_wording_exact_estimate_and_unknown():
    assert sample_scope(DbSample("random", 1000, 5110)) == "a random sample of 1,000 of 5,110 rows"
    est = sample_scope(DbSample("random", 1000, 2_000_000, total_is_estimate=True))
    assert "1,000 of about 2,000,000 rows" in est and "estimate" in est
    assert "could not be read" in sample_scope(DbSample("random", 1000, None))
    assert DbSample("random", 1000, None).is_partial
    assert not DbSample("full", 40, 40).is_partial


def test_catalog_estimate_is_used_and_labelled_when_count_fails(patients):
    path, _truth = patients
    base = mcp_runner(path, dialect="postgresql")

    async def run(sql):
        if sql.startswith("SELECT COUNT(*)"):
            return {"error": "QueryCanceled: statement timeout"}
        if "pg_class" in sql:
            return {"columns": ["row_count"], "rows": [[4900]], "meta": {"database_type": "postgresql"}}
        return await base(sql)

    _cols, _rows, sample = asyncio.run(draw_table_sample(run, "patients", SAMPLE_N, table="patients"))
    assert sample.total == 4900 and sample.total_is_estimate
    assert "of about 4,900 rows" in sample_scope(sample)
