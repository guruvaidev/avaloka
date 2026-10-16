"""The Postgres backend.

SQLite is what the service uses to run and be tested with nothing installed.
Postgres is what it uses in production, because one Cloud Run instance is not
the only thing that will ever write to this and SQLite does not survive that.

The interface is identical to :class:`telemetry_service.store.Store`, so
choosing between them is a connection string rather than a code change:

    TELEMETRY_DATABASE_URL=postgresql://user:pass@host/telemetry

The two properties that mattered in SQLite matter more here, because several
instances write concurrently:

* **Deduplication** is a primary key with ``ON CONFLICT DO NOTHING``, so two
  instances receiving the same redelivered batch cannot both insert.
* **No sender identity.** There is still no column that could hold an IP
  address or a hostname. The schema cannot store what it has no column for.
  (The payload can hold a rotating per-analyst pseudonym; see ``store.py``.)
"""

from __future__ import annotations

import json
import logging
import threading

from telemetry_service.store import StoreResult

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id     TEXT PRIMARY KEY,
    event_type   TEXT NOT NULL,
    install_id   TEXT,
    received_at  DOUBLE PRECISION NOT NULL,
    payload      JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS events_received_at ON events (received_at);
CREATE INDEX IF NOT EXISTS events_install_id  ON events (install_id);
CREATE INDEX IF NOT EXISTS events_type        ON events (event_type);
"""


class PostgresStore:
    """Event storage on Postgres. Same interface as the SQLite store."""

    def __init__(self, dsn: str, minconn: int = 1, maxconn: int = 8) -> None:
        import psycopg2
        from psycopg2 import pool

        self.dsn = dsn
        self._lock = threading.Lock()
        self._pool = pool.ThreadedConnectionPool(minconn, maxconn, dsn)
        self._psycopg2 = psycopg2
        with self._cursor() as cursor:
            cursor.execute(SCHEMA)

    # ------------------------------------------------------------- plumbing

    class _CursorContext:
        def __init__(self, store: "PostgresStore") -> None:
            self._store = store

        def __enter__(self):
            self._connection = self._store._pool.getconn()
            self._cursor = self._connection.cursor()
            return self._cursor

        def __exit__(self, exc_type, exc, tb):
            try:
                if exc_type is None:
                    self._connection.commit()
                else:
                    self._connection.rollback()
            finally:
                self._cursor.close()
                self._store._pool.putconn(self._connection)

    def _cursor(self):
        return PostgresStore._CursorContext(self)

    def close(self) -> None:
        self._pool.closeall()

    # ---------------------------------------------------------------- write

    @staticmethod
    def batch_install_id(records: list[dict]) -> str | None:
        for record in records:
            if record.get("event_type") == "install" and record.get("install_id"):
                return record["install_id"]
        return None

    def store(self, records: list[dict], received_at: float) -> StoreResult:
        """Insert a validated batch, ignoring records already held.

        ``ON CONFLICT DO NOTHING`` rather than a check-then-insert: two
        instances handling the same redelivered batch at the same moment would
        both pass a check and both insert.
        """
        batch_install = self.batch_install_id(records)
        rows = [
            (
                record["event_id"],
                record["event_type"],
                record.get("install_id") or batch_install,
                received_at,
                json.dumps(record, separators=(",", ":")),
            )
            for record in records
        ]

        result = StoreResult()
        with self._cursor() as cursor:
            for row in rows:
                cursor.execute(
                    "INSERT INTO events "
                    "(event_id, event_type, install_id, received_at, payload) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (event_id) DO NOTHING",
                    row,
                )
                if cursor.rowcount:
                    result.stored += 1
                else:
                    result.duplicates += 1
        return result

    # ----------------------------------------------------------------- read

    def count(self) -> int:
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM events")
            return cursor.fetchone()[0]

    def event_ids(self) -> set[str]:
        with self._cursor() as cursor:
            cursor.execute("SELECT event_id FROM events")
            return {row[0] for row in cursor.fetchall()}

    def records(self, limit: int = 100) -> list[dict]:
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT payload, install_id FROM events "
                "ORDER BY received_at DESC LIMIT %s",
                (limit,),
            )
            rows = cursor.fetchall()

        records = []
        for payload, install_id in rows:
            record = payload if isinstance(payload, dict) else json.loads(payload)
            if install_id:
                record["install_id"] = install_id
            records.append(record)
        return records

    def session_records(self, older_than: float | None = None,
                        limit: int = 10 ** 9) -> list[dict]:
        """Start and stop records with their arrival time. See Store for why."""
        clause = "AND received_at < %s" if older_than is not None else ""
        params: tuple = ((older_than, limit) if older_than is not None else (limit,))
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT payload, received_at FROM events "
                "WHERE event_type IN ('agent.started', 'agent.completed') "
                f"{clause} ORDER BY received_at DESC LIMIT %s",
                params,
            )
            rows = cursor.fetchall()

        out = []
        for payload, received_at in rows:
            try:
                record = payload if isinstance(payload, dict) else json.loads(payload)
            except Exception:
                continue
            if isinstance(record, dict):
                record["received_at"] = received_at
                out.append(record)
        return out

    def columns(self) -> list[str]:
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'events'"
            )
            return [row[0] for row in cursor.fetchall()]

    # ------------------------------------------------------------ retention

    def count_older_than(self, cutoff: float) -> int:
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM events WHERE received_at < %s", (cutoff,))
            return cursor.fetchone()[0]

    def purge_older_than(self, cutoff: float) -> int:
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM events WHERE received_at < %s", (cutoff,))
            return cursor.rowcount
