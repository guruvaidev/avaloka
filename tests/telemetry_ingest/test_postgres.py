"""Tests for the Postgres backend.

Written because this module shipped with none: 177 lines of production database
code that had never been executed once. "It's the same interface" is a claim,
not a test.

Most of this file needs no Postgres server: a fake driver stands in. That
cannot prove the SQL is accepted by a real server — only a test against one can
do that, and those are the last class in the file. They are marked
``integration`` and need ``AVALOKA_TEST_POSTGRES_URL`` to name a scratch
database, so the hermetic run deselects them. What the fake does prove is
everything else: that the class constructs, that the interface really does
match, that connections are returned to the pool on both success and failure,
that the dedup uses ON CONFLICT rather than a check-then-insert that would race,
and that no column exists which could hold a sender's identity.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from telemetry_service.store import Store, open_store


# ------------------------------------------------------------------ fake driver

class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rowcount = 0
        self._result = None
        self.closed = False

    def execute(self, sql, params=None):
        self.connection.statements.append((" ".join(sql.split()), params))
        lowered = sql.lower()

        if lowered.strip().startswith("create"):
            self.rowcount = -1
        elif "insert into events" in lowered:
            event_id = params[0]
            if event_id in self.connection.rows:
                self.rowcount = 0          # ON CONFLICT DO NOTHING
            else:
                self.connection.rows[event_id] = params
                self.rowcount = 1
        elif "select count(*)" in lowered:
            if "received_at <" in lowered:
                cutoff = params[0]
                self._result = (sum(1 for r in self.connection.rows.values() if r[3] < cutoff),)
            else:
                self._result = (len(self.connection.rows),)
        elif "select event_id" in lowered:
            self._result = [(k,) for k in self.connection.rows]
        elif "select payload, install_id" in lowered:
            self._result = [(r[4], r[2]) for r in self.connection.rows.values()]
        elif "column_name" in lowered:
            self._result = [("event_id",), ("event_type",), ("install_id",),
                            ("received_at",), ("payload",)]
        elif "delete from events" in lowered:
            cutoff = params[0]
            doomed = [k for k, r in self.connection.rows.items() if r[3] < cutoff]
            for k in doomed:
                del self.connection.rows[k]
            self.rowcount = len(doomed)
        else:
            raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self):
        return self._result

    def fetchall(self):
        return self._result

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, shared):
        self.rows = shared
        self.statements = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakePool:
    def __init__(self, minconn, maxconn, dsn):
        self.dsn = dsn
        self.rows = {}
        self.connection = FakeConnection(self.rows)
        self.checked_out = 0
        self.closed = False

    def getconn(self):
        self.checked_out += 1
        return self.connection

    def putconn(self, connection):
        self.checked_out -= 1

    def closeall(self):
        self.closed = True


@pytest.fixture()
def pg(monkeypatch):
    """A PostgresStore backed by the fake driver."""
    fake_pool_module = types.SimpleNamespace(ThreadedConnectionPool=FakePool)
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_psycopg2.pool = fake_pool_module
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.pool", fake_pool_module)

    from telemetry_service.postgres import PostgresStore

    return PostgresStore(dsn="postgresql://user:pass@localhost/telemetry")


def event(index, **extra):
    return {"event_id": f"{index:032x}", "event_type": "agent.completed",
            "agent": "data_scout", "outcome": "success", **extra}


# ------------------------------------------------------------------- interface

class TestInterfaceParity:
    """The whole "swapping the database is a connection string" claim rests on
    these two classes being interchangeable. That was asserted, never checked."""

    def test_every_method_the_callers_use_exists_on_both(self):
        from telemetry_service.postgres import PostgresStore

        for name in ["store", "count", "records", "columns", "count_older_than",
                     "purge_older_than", "close", "batch_install_id"]:
            assert hasattr(Store, name), f"Store is missing {name}"
            assert hasattr(PostgresStore, name), f"PostgresStore is missing {name}"

    def test_the_signatures_match(self):
        import inspect

        from telemetry_service.postgres import PostgresStore

        for name in ["store", "count", "records", "columns",
                     "count_older_than", "purge_older_than"]:
            assert (str(inspect.signature(getattr(Store, name)))
                    == str(inspect.signature(getattr(PostgresStore, name)))), name

    def test_a_postgres_url_selects_postgres(self, monkeypatch):
        fake_pool_module = types.SimpleNamespace(ThreadedConnectionPool=FakePool)
        fake_psycopg2 = types.ModuleType("psycopg2")
        fake_psycopg2.pool = fake_pool_module
        monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
        monkeypatch.setitem(sys.modules, "psycopg2.pool", fake_pool_module)

        from telemetry_service.postgres import PostgresStore

        assert isinstance(open_store("postgresql://u@h/db"), PostgresStore)
        assert isinstance(open_store("postgres://u@h/db"), PostgresStore)

    def test_anything_else_selects_sqlite(self, tmp_path):
        assert isinstance(open_store(str(tmp_path / "t.db")), Store)


# ----------------------------------------------------------------------- write

class TestWriting:
    def test_the_schema_is_created_on_construction(self, pg):
        assert any("create table" in sql.lower()
                   for sql, _ in pg._pool.connection.statements)

    def test_records_are_stored(self, pg):
        result = pg.store([event(0), event(1)], received_at=1000.0)
        assert result.stored == 2 and pg.count() == 2

    def test_redelivery_is_counted_once(self, pg):
        pg.store([event(0), event(1)], received_at=1000.0)
        again = pg.store([event(0), event(1)], received_at=1000.0)
        assert again.stored == 0 and again.duplicates == 2

    def test_dedup_uses_on_conflict_not_a_racy_check(self, pg):
        """Two instances handling the same redelivered batch would both pass a
        check-then-insert and both write. The database must decide."""
        pg.store([event(0)], received_at=1000.0)
        inserts = [sql for sql, _ in pg._pool.connection.statements
                   if "insert into events" in sql.lower()]
        assert inserts and all("ON CONFLICT (event_id) DO NOTHING" in s for s in inserts)

    def test_batch_events_inherit_the_install_id(self, pg):
        pg.store([
            {"event_id": "a" * 32, "event_type": "install", "install_id": "1" * 32},
            event(1),
        ], received_at=1000.0)
        assert {r.get("install_id") for r in pg.records()} == {"1" * 32}


# ------------------------------------------------------------------ connections

class TestConnectionHandling:
    def test_connections_are_returned_to_the_pool(self, pg):
        before = pg._pool.checked_out
        pg.store([event(0)], received_at=1000.0)
        pg.count()
        pg.records()
        assert pg._pool.checked_out == before, "a connection was leaked"

    def test_a_connection_is_returned_even_when_the_query_raises(self, pg):
        before = pg._pool.checked_out
        with pytest.raises(AssertionError):
            with pg._cursor() as cursor:
                cursor.execute("SELECT something_unexpected")
        assert pg._pool.checked_out == before, "a failed query leaked a connection"

    def test_a_failure_rolls_back_rather_than_committing(self, pg):
        before = pg._pool.connection.rollbacks
        with pytest.raises(AssertionError):
            with pg._cursor() as cursor:
                cursor.execute("SELECT something_unexpected")
        assert pg._pool.connection.rollbacks == before + 1

    def test_success_commits(self, pg):
        before = pg._pool.connection.commits
        pg.store([event(0)], received_at=1000.0)
        assert pg._pool.connection.commits > before

    def test_close_closes_the_pool(self, pg):
        pg.close()
        assert pg._pool.closed


# --------------------------------------------------------------------- privacy

class TestPrivacy:
    def test_there_is_nowhere_to_put_a_sender_identity(self, pg):
        columns = set(pg.columns())
        for forbidden in ["ip", "ip_address", "remote_addr", "hostname", "user_agent"]:
            assert forbidden not in columns
        assert columns == {"event_id", "event_type", "install_id", "received_at", "payload"}

    def test_the_schema_declares_no_identity_column(self):
        from telemetry_service.postgres import SCHEMA

        lowered = SCHEMA.lower()
        for forbidden in ["ip", "hostname", "remote", "user_agent", "sender"]:
            assert f"{forbidden} " not in lowered


# ------------------------------------------------------------------- retention

class TestRetention:
    def test_old_events_are_deleted(self, pg):
        pg.store([event(0)], received_at=1000.0)
        pg.store([event(1)], received_at=9000.0)
        assert pg.purge_older_than(5000.0) == 1
        assert pg.count() == 1

    def test_counting_deletes_nothing(self, pg):
        pg.store([event(0)], received_at=1000.0)
        assert pg.count_older_than(5000.0) == 1
        assert pg.count() == 1


# ------------------------------------------------------------- a real server

PG_URL = os.environ.get("AVALOKA_TEST_POSTGRES_URL", "").strip()
PG_SCHEMA = "telemetry_ingest_test"


def _postgres_url_or_skip():
    """A skip reads as a pass in a summary line. On a laptop that is fine. On
    the run that gates a merge it would mean this SQL had still never met a
    server, so there it is a failure. Same rule as tests/test_lineage.py."""
    if PG_URL:
        return PG_URL
    message = "AVALOKA_TEST_POSTGRES_URL is not set; the Postgres store was not exercised"
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
        pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
    pytest.skip(message)


@pytest.fixture()
def real_pg():
    """A PostgresStore on a real server, with the real driver.

    The one thing the fake above cannot prove: that the SQL is valid. A fake
    accepts whatever it is given, so a typo in the DDL, a column type Postgres
    rejects, or a JSONB value that comes back as text would all pass every
    test above and fail on first contact with production.

    The scratch database is shared with other suites (tests/test_lineage.py
    uses the same variable), and the store's table is called simply `events`.
    So these tests work in a schema of their own, created here and dropped with
    everything in it afterwards: nothing of theirs is visible to another suite
    and nothing is left behind.
    """
    import psycopg2

    from telemetry_service.postgres import PostgresStore

    url = _postgres_url_or_skip()

    def admin(sql):
        connection = psycopg2.connect(url)
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
            connection.commit()
        finally:
            connection.close()

    admin(f"DROP SCHEMA IF EXISTS {PG_SCHEMA} CASCADE")
    admin(f"CREATE SCHEMA {PG_SCHEMA}")
    # search_path set on the connection, so the store's unqualified `events`
    # resolves inside the scratch schema and nowhere else.
    scoped = url + ("&" if "?" in url else "?") + f"options=-csearch_path%3D{PG_SCHEMA}"
    store = PostgresStore(dsn=scoped)
    try:
        yield store
    finally:
        store.close()
        admin(f"DROP SCHEMA IF EXISTS {PG_SCHEMA} CASCADE")


@pytest.mark.integration
class TestAgainstARealServer:
    def test_the_schema_is_accepted_and_holds_no_identity_column(self, real_pg):
        assert set(real_pg.columns()) == {
            "event_id", "event_type", "install_id", "received_at", "payload"}

    def test_the_table_lives_in_the_scratch_schema_only(self, real_pg):
        """Otherwise this suite would be writing into whatever else shares the
        database."""
        with real_pg._cursor() as cursor:
            cursor.execute("SELECT table_schema FROM information_schema.tables "
                           "WHERE table_name = 'events'")
            assert [row[0] for row in cursor.fetchall()] == [PG_SCHEMA]

    def test_creating_the_schema_twice_is_harmless(self, real_pg):
        from telemetry_service.postgres import PostgresStore

        again = PostgresStore(dsn=real_pg.dsn)
        assert again.count() == 0
        again.close()

    def test_a_batch_round_trips_and_is_attributed(self, real_pg):
        batch = [{"event_id": "a" * 32, "event_type": "install", "install_id": "1" * 32,
                  "os": "linux"}, event(1, duration_ms=42), event(2)]
        result = real_pg.store(batch, received_at=1000.0)
        assert (result.stored, result.duplicates) == (3, 0)
        assert real_pg.count() == 3 and real_pg.event_ids() == {r["event_id"] for r in batch}

        records = {r["event_id"]: r for r in real_pg.records()}
        # JSONB comes back as a dict, with the values it went in with.
        assert records[event(1)["event_id"]]["duration_ms"] == 42
        assert {r["install_id"] for r in records.values()} == {"1" * 32}

    def test_redelivery_is_counted_once(self, real_pg):
        real_pg.store([event(0), event(1)], received_at=1000.0)
        again = real_pg.store([event(1), event(2)], received_at=2000.0)
        assert (again.stored, again.duplicates) == (1, 1)
        assert real_pg.count() == 3

    def test_session_records_carry_their_arrival_time(self, real_pg):
        real_pg.store([{"event_id": "b" * 32, "event_type": "agent.started",
                        "agent": "validator", "session_id": "5" * 32, "seq": 1}],
                      received_at=1000.0)
        real_pg.store([{"event_id": "c" * 32, "event_type": "dataset",
                        "file_format": "csv"}], received_at=1000.0)
        rows = real_pg.session_records()
        assert [r["event_type"] for r in rows] == ["agent.started"]
        assert rows[0]["received_at"] == 1000.0
        assert real_pg.session_records(older_than=500.0) == []

    def test_retention_deletes_what_it_counted(self, real_pg):
        real_pg.store([event(0)], received_at=1000.0)
        real_pg.store([event(1)], received_at=9000.0)
        assert real_pg.count_older_than(5000.0) == 1 and real_pg.count() == 2
        assert real_pg.purge_older_than(5000.0) == 1
        assert real_pg.count() == 1

    def test_an_analytics_record_round_trips(self, real_pg):
        record = {"event_id": "e" * 32, "event_type": "analytics.page.viewed",
                  "install_id": "d" * 32, "session_id": "5" * 32, "seq": 1, "t_ms": 10,
                  "route": "/reports/$reportId", "schema_version": "1.1"}
        real_pg.store([record], received_at=1000.0)
        assert real_pg.records() == [record]

    def test_the_reports_run_against_it(self, real_pg):
        from telemetry_service import reports

        real_pg.store([{"event_id": "a" * 32, "event_type": "install",
                        "install_id": "1" * 32, "os": "linux"}, event(1), event(2)],
                      received_at=1000.0)
        result = reports.run_all(real_pg)
        assert result["adoption"]["rows"][0]["installations"] == 1
        assert result["capabilities"]["rows"][0]["runs"] == 2
