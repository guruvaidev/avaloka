"""An in-memory Supabase/PostgREST fake wide enough for the job ledger, the
chat-note outbox and notification_service: select/insert/upsert/update with
eq/is_/in_/order/limit, a unique constraint on job_chat_notes.job_id, and
``update … is_("notified_at","null")`` returning only the rows it changed."""

from __future__ import annotations

import threading
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


def _value(row: Dict[str, Any], column: str) -> Any:
    if "->>" in column:
        col, key = column.split("->>", 1)
        val = (row.get(col) or {}).get(key)
        return None if val is None else str(val)
    return row.get(column)


class _Query:
    def __init__(self, db: "FakeDB", table: str):
        self.db, self.table = db, table
        self.op, self.payload, self.filters = "select", None, []
        self._limit, self._count, self._order, self._conflict = None, False, None, None

    def select(self, *_cols, count=None):
        self.op, self._count = "select", count == "exact"
        return self

    def insert(self, row):
        self.op, self.payload = "insert", row
        return self

    def upsert(self, row, on_conflict=None):
        self.op, self.payload, self._conflict = "upsert", row, on_conflict or "id"
        return self

    def update(self, values):
        self.op, self.payload = "update", values
        return self

    def eq(self, column, value):
        self.filters.append(lambda r, c=column, v=value: _value(r, c) == v)
        return self

    def is_(self, column, value):
        assert value == "null"
        self.filters.append(lambda r, c=column: r.get(c) is None)
        return self

    def in_(self, column, values):
        vs = set(values)
        self.filters.append(lambda r, c=column: _value(r, c) in vs)
        return self

    def order(self, column, desc=False):
        self._order = (column, desc)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def execute(self):
        with self.db.lock:
            self.db.calls.append((self.table, self.op))
            if self.table in self.db.fail_tables or (self.table, self.op) in self.db.fail_ops:
                raise RuntimeError(f"supabase down ({self.table}.{self.op})")
            rows = self.db.tables.setdefault(self.table, [])
            if self.op in ("insert", "upsert"):
                payload = dict(self.payload)
                if self.table in self.db.missing_columns_tables:
                    bad = set(payload) & self.db.missing_columns
                    if bad:
                        raise RuntimeError(f"PGRST204: Could not find the '{sorted(bad)[0]}' column of '{self.table}' in the schema cache")
                if self.table == "job_chat_notes" and any(r.get("job_id") == payload.get("job_id") for r in rows):
                    raise RuntimeError('duplicate key value violates unique constraint "job_chat_notes_job_id_key" (23505)')
                if self.op == "upsert":
                    key = self._conflict
                    for r in rows:
                        if r.get(key) == payload.get(key):
                            r.update(payload)
                            return SimpleNamespace(data=[dict(r)], count=None)
                saved = {"id": payload.get("id") or str(uuid.uuid4()), **payload}
                if self.table == "notifications":
                    saved.setdefault("read_at", None)
                if self.table == "job_chat_notes":
                    saved.setdefault("delivered_at", None)
                    saved.setdefault("batch_id", None)
                rows.append(saved)
                return SimpleNamespace(data=[dict(saved)], count=None)
            matched = [r for r in rows if all(f(r) for f in self.filters)]
            if self.op == "update":
                if self.table in self.db.missing_columns_tables:
                    bad = set(self.payload) & self.db.missing_columns
                    if bad:
                        raise RuntimeError(f"PGRST204: Could not find the '{sorted(bad)[0]}' column of '{self.table}' in the schema cache")
                for r in matched:
                    r.update(self.payload)
                return SimpleNamespace(data=[dict(r) for r in matched], count=None)
            count = len(matched) if self._count else None
            if self._order:
                col, desc = self._order
                matched = sorted(matched, key=lambda r: r.get(col) or "", reverse=desc)
            if self._limit is not None:
                matched = matched[: self._limit]
            return SimpleNamespace(data=[dict(r) for r in matched], count=count)


class FakeDB:
    AUTH_UID = "auth-uid-1111"
    PROFILE_ID = "profile-aaaa-0001"
    THREAD_ID = "thread-1234-5678"
    ANALYSIS_ID = "9a0b1c2d-0000-4000-8000-000000000001"

    def __init__(self):
        self.tables: Dict[str, List[Dict[str, Any]]] = {
            "profiles": [{"id": self.PROFILE_ID, "user_id": self.AUTH_UID}],
            "analyses": [{"id": self.ANALYSIS_ID, "thread_id": self.THREAD_ID}],
            "notifications": [], "ml_jobs": [], "job_chat_notes": [], "analysis_messages": [],
        }
        self.calls: List = []
        self.fail_tables: set = set()
        self.fail_ops: set = set()
        self.missing_columns_tables: set = set()
        self.missing_columns: set = {"thread_id", "analysis_id", "resource_type", "resource_id", "label",
                                     "fail_reason", "notified_at", "notification_id", "origin"}
        self.lock = threading.RLock()

    def table(self, name):
        return _Query(self, name)

    @property
    def notifications(self):
        return self.tables["notifications"]

    @property
    def jobs(self):
        return self.tables["ml_jobs"]

    @property
    def notes(self):
        return self.tables["job_chat_notes"]

    @property
    def messages(self):
        return self.tables["analysis_messages"]
