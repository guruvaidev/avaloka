"""Where accepted events are kept.

SQLite by default so the service runs and is testable with nothing installed;
Postgres in production by setting a connection string. The interface is the
same either way, so the choice of database is a deployment decision rather than
a code change.

Two properties matter more than performance here:

* **Deduplication.** The client delivers at-least-once — if an acknowledgement
  is lost it re-sends the same batch — so the store must count each
  ``event_id`` once. That is enforced by a primary key rather than by a check
  in application code, so two requests racing cannot both succeed.
* **No sender identity.** There is no column for an IP address, a hostname or
  anything else identifying who sent a record. The schema cannot store what it
  has no column for.

That is a statement about the sender, not about the payload. The payload is the
validated record, and a product-analytics record carries ``actor_id``: a
per-analyst pseudonym, re-keyed every 28 days by the deployment that exports it
with a secret this service never receives. So what is stored is pseudonymous.
``schema.py`` says what that pseudonym can and cannot be linked to.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DATABASE_URL_ENV = "TELEMETRY_DATABASE_URL"
DEFAULT_SQLITE_PATH = "telemetry.db"

#: How long events are kept before the retention job removes them. The design
#: proposes 400 days — long enough for a year-over-year comparison, and one of
#: the open questions still to be confirmed.
RETENTION_DAYS = int(os.environ.get("TELEMETRY_RETENTION_DAYS", "400"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id     TEXT PRIMARY KEY,
    event_type   TEXT NOT NULL,
    install_id   TEXT,
    received_at  REAL NOT NULL,
    payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_received_at ON events (received_at);
CREATE INDEX IF NOT EXISTS events_install_id  ON events (install_id);
CREATE INDEX IF NOT EXISTS events_type        ON events (event_type);
"""


def open_store(url: str | None = None):
    """Return the right store for the configured database.

    A Postgres connection string selects Postgres; anything else is treated as
    a SQLite path. This is the whole of "swapping the database is a connection
    string" — the two classes present the same interface, so nothing above this
    line knows or cares which it got.
    """
    configured = url or os.environ.get(DATABASE_URL_ENV)
    if not configured:
        # Loud, because silent was dangerous: a deployment that forgot the
        # database URL came up green and wrote everything to a file inside
        # the container — gone on the next restart, and never shared with
        # the other instances.
        logger.warning(
            "%s is not set; using a local SQLite file (%s). Fine for one "
            "process on one machine, wrong for any real deployment.",
            DATABASE_URL_ENV, DEFAULT_SQLITE_PATH,
        )
    url = configured or DEFAULT_SQLITE_PATH
    if url.startswith(("postgres://", "postgresql://")):
        from telemetry_service.postgres import PostgresStore

        logger.info("Using the Postgres store.")
        return PostgresStore(dsn=url)
    return Store(path=url)


@dataclass
class StoreResult:
    stored: int = 0
    duplicates: int = 0

    @property
    def total(self) -> int:
        return self.stored + self.duplicates


class Store:
    """Event storage. Thread-safe, and idempotent on ``event_id``."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(path or os.environ.get(DATABASE_URL_ENV) or DEFAULT_SQLITE_PATH)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # ------------------------------------------------------------------ write

    @staticmethod
    def batch_install_id(records: list[dict]) -> str | None:
        """The installation a batch came from, taken from its install record.

        The client sends the machine description once per upload rather than on
        every event, so the individual events carry no identifier of their own.
        Attributing them to the batch's install is what makes it possible to
        tell a thousand events from one machine apart from one event from a
        thousand machines — the single distinction the whole feature exists to
        make.
        """
        for record in records:
            if record.get("event_type") == "install" and record.get("install_id"):
                return record["install_id"]
        return None

    def store(self, records: list[dict], received_at: float) -> StoreResult:
        """Insert a validated batch, ignoring records already held.

        ``received_at`` is passed in rather than read from the clock so the
        whole batch shares one timestamp and the caller stays testable.
        """
        result = StoreResult()
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

        with self._lock:
            for row in rows:
                cursor = self._connection.execute(
                    "INSERT OR IGNORE INTO events "
                    "(event_id, event_type, install_id, received_at, payload) "
                    "VALUES (?, ?, ?, ?, ?)",
                    row,
                )
                # rowcount is 0 when the primary key already existed, which is
                # exactly the at-least-once redelivery case.
                if cursor.rowcount:
                    result.stored += 1
                else:
                    result.duplicates += 1
            self._connection.commit()

        return result

    # ------------------------------------------------------------------- read

    def count(self) -> int:
        with self._lock:
            return self._connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def event_ids(self) -> set[str]:
        with self._lock:
            return {row[0] for row in self._connection.execute("SELECT event_id FROM events")}

    def records(self, limit: int = 100) -> list[dict]:
        """Stored records, each carrying the installation it was attributed to.

        ``install_id`` comes from the column rather than the payload: an
        individual event does not carry one, because the client sends the
        machine description once per batch. The column holds the resolved
        answer, so reporting reads it from there.
        """
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload, install_id FROM events ORDER BY received_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        records = []
        for payload, install_id in rows:
            record = json.loads(payload)
            if install_id:
                record["install_id"] = install_id
            records.append(record)
        return records

    def session_records(self, older_than: float | None = None,
                        limit: int = 10 ** 9) -> list[dict]:
        """Start and stop records, each carrying when it was received.

        The arrival time is what the session reports need and the payload does
        not carry. Without it, a start whose completion simply has not been
        uploaded yet is indistinguishable from one whose process was killed —
        so a healthy run in progress gets reported as a crash. The reports use
        it to ignore anything too recent to have had its completion delivered.

        Separate from ``records()`` rather than an extra flag on it, so the six
        reports that do not care are unaffected.
        """
        clause = "AND received_at < ?" if older_than is not None else ""
        params: tuple = ((older_than, limit) if older_than is not None else (limit,))
        with self._lock:
            rows = self._connection.execute(
                f"SELECT payload, received_at FROM events "
                f"WHERE event_type IN ('agent.started', 'agent.completed') {clause} "
                f"ORDER BY received_at DESC LIMIT ?",
                params,
            ).fetchall()

        out = []
        for payload, received_at in rows:
            try:
                record = json.loads(payload)
            except Exception:
                continue          # a row we cannot parse is not worth a crash
            if isinstance(record, dict):
                record["received_at"] = received_at
                out.append(record)
        return out

    def columns(self) -> list[str]:
        """The stored columns. Used by a test asserting no identity is kept."""
        with self._lock:
            return [row[1] for row in self._connection.execute("PRAGMA table_info(events)")]

    # -------------------------------------------------------------- retention

    def count_older_than(self, cutoff: float) -> int:
        """How many events the retention job would remove. Deletes nothing."""
        with self._lock:
            return self._connection.execute(
                "SELECT COUNT(*) FROM events WHERE received_at < ?", (cutoff,)
            ).fetchone()[0]

    def purge_older_than(self, cutoff: float) -> int:
        """Delete events received before ``cutoff``. Returns how many went.

        A retention policy that is written down but never runs is not a policy.
        This is called by a scheduled job; it is a plain method so it can be
        tested without waiting four hundred days.
        """
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM events WHERE received_at < ?", (cutoff,)
            )
            self._connection.commit()
            return cursor.rowcount
