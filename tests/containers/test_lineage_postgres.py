"""Lineage against a real Postgres, from more than one process.

``tests/test_lineage.py`` already runs every store test against Postgres when
``AVALOKA_TEST_POSTGRES_URL`` is set, and ``scripts/integration.sh`` sets it, so
the query-level tenant scoping is exercised there. This file covers what one
process and one engine cannot: the reason lineage moved to Postgres at all was
that the API pod and the LangGraph pod each kept their own SQLite file, so the
properties that matter are the ones between *processes* --

* what one process records, another reads, column names included;
* processes that start together on an empty database all come up, and agree on
  one salt (a second salt would make one pod's column ids unreadable to the
  other, silently);
* two tenants written by different processes stay apart;
* a rejected credential turns lineage off rather than failing a request, and
  does not put the password in the log;
* a connection the server dropped is replaced, not reported as "no lineage".

Every test asserts something positive that an unreachable database cannot
satisfy. That is deliberate: ``LineageStore`` never raises, so "tenant B sees
nothing of tenant A" is also true of a store that is switched off.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.core import lineage
from app.core.lineage import LineageStore

TABLES = ("lineage_node", "lineage_edge", "lineage_name_vault", "lineage_meta")
A, B = "tenant-a", "tenant-b"

# What a pod does: import the store, record with the tenant it was given, and
# say what it saw. argv: tenant, start-at (epoch seconds), then a JSON list of
# record_dataset calls as [dataset_id, kwargs].
POD = r"""
import json, sys, time
from app.core.lineage import LineageStore

tenant, start_at, calls = sys.argv[1], float(sys.argv[2]), json.loads(sys.argv[3])
time.sleep(max(0.0, start_at - time.time()))
store = LineageStore(tenant)
for dataset_id, kwargs in calls:
    kwargs["columns"] = [tuple(c) for c in kwargs.get("columns", [])]
    store.record_dataset(dataset_id, **kwargs)
out = {"available": store.available,
       "column_ids": {f"{d}/{name}": store.column_id(d, name)
                      for d, kwargs in calls for name, _ in kwargs.get("columns", [])}}
store.close()
print(json.dumps(out))
"""


def _pod_env(url: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("POSTGRES_URL", lineage.ENV_ENABLED)}
    env[lineage.ENV_DB_URL] = url
    return env


def _start_pod(url: str, tenant: str, calls, start_at: float = 0.0) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", POD, tenant, repr(start_at), json.dumps(calls)],
        env=_pod_env(url), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _finish(pod: subprocess.Popen) -> dict:
    out, err = pod.communicate(timeout=120)
    assert pod.returncode == 0, f"the pod exited {pod.returncode}:\n{err[-2000:]}"
    return json.loads(out.strip().splitlines()[-1])


def _rows(url: str, sql: str, **params):
    """Read with a connection of our own, not through the code under test."""
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return conn.execute(text(sql), params).fetchall()
    finally:
        engine.dispose()


@pytest.fixture
def db(postgres_url, monkeypatch):
    """An empty database: the lineage tables do not exist yet, as on a first rollout."""
    lineage.reset_for_tests()
    engine = create_engine(postgres_url)
    with engine.begin() as conn:
        for table in TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
    engine.dispose()
    monkeypatch.setenv(lineage.ENV_DB_URL, postgres_url)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv(lineage.ENV_ENABLED, raising=False)
    yield postgres_url
    lineage.reset_for_tests()


def test_what_one_process_records_another_process_reads(db):
    wrote = _finish(_start_pod(db, A, [
        ["ds:raw", {"label": "sales.csv", "columns": [["customer_email", "object"]]}],
        ["ds:clean", {"label": "clean.csv", "derived_from": "ds:raw"}],
    ]))
    assert wrote["available"] is True

    reader = LineageStore(A)
    assert reader.ancestors("ds:clean") == ["ds:raw"]
    assert reader.get_node("ds:raw").label == "sales.csv"

    # The column id is a salted digest. Another process reproduces it only if
    # it read the same salt from the database rather than minting its own.
    cid = wrote["column_ids"]["ds:raw/customer_email"]
    assert reader.column_id("ds:raw", "customer_email") == cid
    assert reader.resolve_column(cid) == "customer_email"

    stored = _rows(db, "SELECT id FROM lineage_node WHERE tenant_id = :t ORDER BY id", t=A)
    assert [r[0] for r in stored] == sorted([cid, "ds:clean", "ds:raw"])


def test_processes_starting_together_on_an_empty_database_agree_on_one_salt(db):
    pods = 6
    start_at = time.time() + 4.0   # past the slowest import, so they hit the database together
    running = [_start_pod(db, A, [
        ["ds:shared", {"label": "shared.csv", "columns": [["amount", "float64"]]}],
        [f"ds:pod-{n}", {"derived_from": "ds:shared"}],
    ], start_at) for n in range(pods)]
    results = [_finish(pod) for pod in running]

    assert [r["available"] for r in results] == [True] * pods
    column_ids = {r["column_ids"]["ds:shared/amount"] for r in results}
    assert len(column_ids) == 1, f"pods disagree on the id of one column: {sorted(column_ids)}"

    salts = _rows(db, "SELECT v FROM lineage_meta WHERE tenant_id = :t AND k = 'column_salt'", t=A)
    assert len(salts) == 1

    reader = LineageStore(A)
    assert sorted(reader.descendants("ds:shared")) == [f"ds:pod-{n}" for n in range(pods)]
    assert reader.resolve_column(column_ids.pop()) == "amount"
    # One dataset and one column, however many pods recorded them.
    assert reader.stats()["node_column"] == 1
    assert _rows(db, "SELECT count(*) FROM lineage_node WHERE tenant_id = :t AND id = 'ds:shared'",
                 t=A)[0][0] == 1


def test_tenants_written_by_different_processes_stay_apart(db):
    start_at = time.time() + 3.0
    # Same ids, same column name, on purpose. B's graph also continues past the
    # point where A's stops: a walk that followed another tenant's edges would
    # carry A from ds:mid into ds:b-source.
    pod_a = _start_pod(db, A, [
        ["ds:mid", {"label": "a-mid.csv", "columns": [["salary", "float64"]]}],
        ["ds:out", {"label": "a-out.csv", "derived_from": "ds:mid"}],
    ], start_at)
    pod_b = _start_pod(db, B, [
        ["ds:b-source", {"label": "b-payroll.csv"}],
        ["ds:mid", {"label": "b-mid.csv", "derived_from": "ds:b-source",
                    "columns": [["salary", "float64"]]}],
        ["ds:out", {"label": "b-out.csv", "derived_from": "ds:mid"}],
    ], start_at)
    wrote_a, wrote_b = _finish(pod_a), _finish(pod_b)
    assert wrote_a["available"] is True and wrote_b["available"] is True

    # Both tenants' rows are really there, under the same ids.
    owners = _rows(db, "SELECT tenant_id, label FROM lineage_node WHERE id = 'ds:mid' ORDER BY 1")
    assert [tuple(r) for r in owners] == [(A, "a-mid.csv"), (B, "b-mid.csv")]

    a, b = LineageStore(A), LineageStore(B)
    assert a.get_node("ds:mid").label == "a-mid.csv"
    assert b.get_node("ds:mid").label == "b-mid.csv"
    assert a.ancestors("ds:out") == ["ds:mid"]
    assert sorted(b.ancestors("ds:out")) == ["ds:b-source", "ds:mid"]
    assert a.get_node("ds:b-source") is None
    assert a.descendants("ds:b-source") == []
    assert a.stats()["node_dataset"] == 2 and b.stats()["node_dataset"] == 3

    cid_a = wrote_a["column_ids"]["ds:mid/salary"]
    cid_b = wrote_b["column_ids"]["ds:mid/salary"]
    assert cid_a != cid_b, "one salt serves both tenants: a column id is comparable across them"
    assert a.resolve_column(cid_a) == "salary" and b.resolve_column(cid_b) == "salary"
    assert a.resolve_column(cid_b) is None and b.resolve_column(cid_a) is None


def test_a_rejected_password_turns_lineage_off_without_raising_or_logging_it(db, caplog):
    good = make_url(db)
    wrong_password = (good.password or "") + "-wrong"
    bad_url = good.set(password=wrong_password).render_as_string(hide_password=False)

    with caplog.at_level(logging.DEBUG):
        refused = LineageStore(A, bad_url)
        assert refused.available is False
        refused.record_dataset("ds:raw", label="sales.csv", columns=[("amount", "float64")])
        assert refused.ancestors("ds:raw") == []
        assert refused.get_node("ds:raw") is None
    assert "store unavailable" in caplog.text
    assert wrong_password not in caplog.text
    assert good.password not in caplog.text

    # Nothing reached the database through the refused login ...
    present = _rows(db, "SELECT to_regclass('lineage_node')")[0][0]
    assert present is None
    # ... and the same calls with the right password do, so it was the
    # credential that switched it off and not something else about this setup.
    accepted = LineageStore(A, db)
    assert accepted.available is True
    accepted.record_dataset("ds:raw", label="sales.csv", columns=[("amount", "float64")])
    assert accepted.get_node("ds:raw").label == "sales.csv"


def test_a_connection_the_server_dropped_is_replaced_not_reported_as_no_lineage(db):
    store = LineageStore(A)
    store.record_dataset("ds:raw", label="sales.csv")
    store.record_dataset("ds:clean", derived_from="ds:raw")
    assert store.ancestors("ds:clean") == ["ds:raw"]

    # What a Postgres restart or failover does to a pooled connection.
    killed = _rows(db, "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                       "WHERE datname = current_database() AND pid <> pg_backend_pid()")
    assert len(killed) >= 1, "the store held no pooled connection, so nothing was dropped"

    assert store.ancestors("ds:clean") == ["ds:raw"]
    store.record_dataset("ds:features", derived_from="ds:clean")
    assert sorted(store.ancestors("ds:features")) == ["ds:clean", "ds:raw"]
