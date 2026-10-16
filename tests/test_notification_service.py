"""Tests for app/services/notification_service.py (design doc D4).

No Supabase needed: a small in-memory fake replaces the client, so these run
offline and in CI. Run with:

    pytest tests/test_notification_service.py -q
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from app.services import notification_service as ns

AUTH_UID = "auth-uid-1111"
PROFILE_ID = "profile-aaaa-0001"
OTHER_PROFILE = "profile-bbbb-0002"
THREAD_ID = "thread-1234-5678"
ANALYSIS_ID = "analysis-9999"


# ---------------------------------------------------------------------------
# Fake Supabase client: just enough of the postgrest query builder
# ---------------------------------------------------------------------------

def _value(row: Dict[str, Any], column: str) -> Any:
    """Column value, including the jsonb arrow form used for the dedupe check."""
    if "->>" in column:
        col, key = column.split("->>", 1)
        val = (row.get(col) or {}).get(key)
        return None if val is None else str(val)
    return row.get(column)


class _Query:
    def __init__(self, db: "FakeSupabase", table: str):
        self.db = db
        self.table = table
        self.op = "select"
        self.payload: Any = None
        self.filters: List = []
        self._limit: Optional[int] = None
        self._count = False

    # builder methods
    def select(self, *_cols, count=None):
        self.op = "select"
        self._count = count == "exact"
        return self

    def insert(self, row):
        self.op, self.payload = "insert", row
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

    def order(self, column, desc=False):
        self._order = (column, desc)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def execute(self):
        self.db.calls.append((self.table, self.op))
        if self.table in self.db.fail_tables or self.op in self.db.fail_ops:
            raise RuntimeError(f"supabase down ({self.table}.{self.op})")
        rows = self.db.tables.setdefault(self.table, [])
        if self.op == "insert":
            saved = {"id": f"n-{len(rows) + 1}", "read_at": None, **self.payload}
            rows.append(saved)
            return SimpleNamespace(data=[saved], count=None)
        matched = [r for r in rows if all(f(r) for f in self.filters)]
        if self.op == "update":
            for r in matched:
                r.update(self.payload)
            return SimpleNamespace(data=list(matched), count=None)
        count = len(matched) if self._count else None
        if getattr(self, "_order", None):
            col, desc = self._order
            matched = sorted(matched, key=lambda r: r.get(col) or "", reverse=desc)
        if self._limit is not None:
            matched = matched[: self._limit]
        return SimpleNamespace(data=matched, count=count)


class FakeSupabase:
    def __init__(self):
        self.tables: Dict[str, List[Dict[str, Any]]] = {
            "profiles": [{"id": PROFILE_ID, "user_id": AUTH_UID}],
            "analyses": [{"id": ANALYSIS_ID, "thread_id": THREAD_ID}],
            "notifications": [],
        }
        self.calls: List = []
        self.fail_tables: set = set()
        self.fail_ops: set = set()

    def table(self, name):
        return _Query(self, name)

    @property
    def notifications(self):
        return self.tables["notifications"]


@pytest.fixture
def db(monkeypatch):
    fake = FakeSupabase()
    monkeypatch.setattr(ns, "_client", lambda: fake)
    ns._profile_cache.clear()
    yield fake
    ns._profile_cache.clear()


def run(coro):
    return asyncio.run(coro)


def notify(**overrides):
    kwargs = dict(user_id=AUTH_UID, ntype="analysis.completed", title="Analysis completed",
                  body="“Create age brackets” on healthcare_dataset.csv",
                  thread_id=THREAD_ID, dataset_id="ds-1", ref="turn-1")
    kwargs.update(overrides)
    user_id, ntype, title = kwargs.pop("user_id"), kwargs.pop("ntype"), kwargs.pop("title")
    body = kwargs.pop("body")
    return run(ns.create_notification(user_id, ntype, title, body, **kwargs))


# ---------------------------------------------------------------------------
# create_notification
# ---------------------------------------------------------------------------

def test_auth_uid_is_translated_to_profile_id(db):
    row = notify()
    assert row is not None
    saved = db.notifications[0]
    assert saved["recipient_id"] == PROFILE_ID      # profiles.id, not the auth uid
    assert saved["actor_id"] == PROFILE_ID
    assert AUTH_UID not in (saved["recipient_id"], saved["actor_id"])


def test_row_has_the_columns_the_bell_reads(db):
    notify()
    saved = db.notifications[0]
    assert saved["type"] == "analysis.completed"
    assert saved["title"] == "Analysis completed"
    assert saved["resource_type"] == "analysis"
    assert saved["metadata"] == {"ref": "turn-1", "thread_id": THREAD_ID, "dataset_id": "ds-1"}


def test_link_points_to_the_analysis_when_it_exists(db):
    notify()
    saved = db.notifications[0]
    assert saved["link"] == f"/analysis?aid={ANALYSIS_ID}"
    assert saved["resource_id"] == ANALYSIS_ID


def test_link_falls_back_when_no_analysis_row(db):
    notify(thread_id="thread-without-analysis", ref="turn-2")
    saved = db.notifications[0]
    assert saved["link"] == "/analysis"
    assert "resource_id" not in saved


def test_same_ref_twice_writes_one_row(db):
    first = notify(ref="turn-dup")
    second = notify(ref="turn-dup")
    assert first is not None
    assert second is None
    assert len(db.notifications) == 1


def test_different_refs_write_separate_rows(db):
    notify(ref="turn-a")
    notify(ref="turn-b")
    assert len(db.notifications) == 2


def test_same_ref_for_another_user_is_not_a_duplicate(db):
    db.tables["profiles"].append({"id": OTHER_PROFILE, "user_id": "auth-uid-2222"})
    notify(ref="turn-shared")
    notify(user_id="auth-uid-2222", ref="turn-shared")
    assert {r["recipient_id"] for r in db.notifications} == {PROFILE_ID, OTHER_PROFILE}


def test_unknown_type_is_rejected_without_touching_supabase(db):
    assert notify(ntype="analysis.done") is None
    assert db.notifications == []
    assert db.calls == []


@pytest.mark.parametrize("ntype", sorted(ns.TYPES))
def test_every_supported_type_is_written(db, ntype):
    assert notify(ntype=ntype, ref=f"ref-{ntype}") is not None
    assert db.notifications[-1]["type"] == ntype


def test_no_profile_row_writes_nothing(db):
    assert notify(user_id="auth-uid-unknown") is None
    assert db.notifications == []


def test_missing_user_id_writes_nothing(db):
    assert notify(user_id=None) is None
    assert notify(user_id="") is None
    assert db.notifications == []


def test_profile_lookup_is_cached(db):
    notify(ref="r1")
    notify(ref="r2")
    profile_queries = [c for c in db.calls if c[0] == "profiles"]
    assert len(profile_queries) == 1


def test_long_title_and_body_are_clipped(db):
    notify(title="T" * 500, body="B " * 2000, ref="long")
    saved = db.notifications[0]
    assert len(saved["title"]) <= ns._MAX_TITLE
    assert len(saved["body"]) <= ns._MAX_BODY
    assert saved["title"].endswith("…")


def test_whitespace_in_body_is_collapsed(db):
    notify(body="line one\n\n   line   two", ref="ws")
    assert db.notifications[0]["body"] == "line one line two"


def test_extra_metadata_is_kept_and_none_values_dropped(db):
    notify(ref="x", dataset_id=None, extra={"rows": 55500, "skip": None})
    meta = db.notifications[0]["metadata"]
    assert meta["rows"] == 55500
    assert "skip" not in meta and "dataset_id" not in meta


# ---------------------------------------------------------------------------
# Never raises: a failed notification must not fail the work it reports on
# ---------------------------------------------------------------------------

def test_insert_failure_returns_none_instead_of_raising(db):
    db.fail_ops.add("insert")
    assert notify() is None
    assert db.notifications == []


def test_profile_lookup_failure_returns_none(db):
    db.fail_tables.add("profiles")
    assert notify() is None


def test_analysis_lookup_failure_still_writes_with_fallback_link(db):
    db.fail_tables.add("analyses")
    assert notify() is not None
    assert db.notifications[0]["link"] == "/analysis"


def test_dedupe_check_failure_still_writes(db):
    original_table = db.table

    def flaky_table(name):
        q = original_table(name)
        if name == "notifications":
            real_execute = q.execute

            def execute():
                if q.op == "select":
                    raise RuntimeError("select failed")
                return real_execute()
            q.execute = execute
        return q

    db.table = flaky_table
    assert notify() is not None
    assert len(db.notifications) == 1


def test_client_not_configured_returns_none(monkeypatch):
    def broken_client():
        raise RuntimeError("SUPABASE_URL not set")
    monkeypatch.setattr(ns, "_client", broken_client)
    ns._profile_cache.clear()
    assert notify() is None


# ---------------------------------------------------------------------------
# list / mark read
# ---------------------------------------------------------------------------

def _seed(db, n_unread=2, n_read=1, recipient=PROFILE_ID):
    for i in range(n_unread):
        db.notifications.append({"id": f"u{i}-{recipient}", "recipient_id": recipient,
                                 "created_at": f"2026-10-09T10:0{i}:00Z", "read_at": None})
    for i in range(n_read):
        db.notifications.append({"id": f"r{i}-{recipient}", "recipient_id": recipient,
                                 "created_at": f"2026-10-08T10:0{i}:00Z",
                                 "read_at": "2026-10-08T11:00:00Z"})


def test_list_returns_only_own_rows_newest_first_with_unread_count(db):
    _seed(db)
    _seed(db, n_unread=3, n_read=0, recipient=OTHER_PROFILE)
    items, unread = run(ns.list_notifications(AUTH_UID))
    assert {i["recipient_id"] for i in items} == {PROFILE_ID}
    assert len(items) == 3
    assert unread == 2
    assert [i["created_at"] for i in items] == sorted((i["created_at"] for i in items), reverse=True)


def test_list_unread_only(db):
    _seed(db)
    items, unread = run(ns.list_notifications(AUTH_UID, unread_only=True))
    assert len(items) == 2 and all(i["read_at"] is None for i in items)
    assert unread == 2


def test_list_limit_is_clamped(db):
    _seed(db, n_unread=5, n_read=0)
    items, _ = run(ns.list_notifications(AUTH_UID, limit=2))
    assert len(items) == 2
    items, _ = run(ns.list_notifications(AUTH_UID, limit=0))   # 0 -> default 20
    assert len(items) == 5


def test_list_for_unknown_user_is_empty(db):
    _seed(db)
    assert run(ns.list_notifications("auth-uid-unknown")) == ([], 0)


def test_mark_read_own_row(db):
    _seed(db, n_unread=1, n_read=0)
    nid = db.notifications[0]["id"]
    assert run(ns.mark_read(AUTH_UID, nid)) is True
    assert db.notifications[0]["read_at"] is not None


def test_mark_read_already_read_counts_as_success(db):
    _seed(db, n_unread=0, n_read=1)
    assert run(ns.mark_read(AUTH_UID, db.notifications[0]["id"])) is True


def test_mark_read_someone_elses_row_is_refused(db):
    _seed(db, n_unread=1, n_read=0, recipient=OTHER_PROFILE)
    nid = db.notifications[0]["id"]
    assert run(ns.mark_read(AUTH_UID, nid)) is False
    assert db.notifications[0]["read_at"] is None


def test_mark_all_read_only_touches_own_unread(db):
    _seed(db, n_unread=2, n_read=1)
    _seed(db, n_unread=2, n_read=0, recipient=OTHER_PROFILE)
    assert run(ns.mark_all_read(AUTH_UID)) == 2
    others = [r for r in db.notifications if r["recipient_id"] == OTHER_PROFILE]
    assert all(r["read_at"] is None for r in others)


def test_mark_all_read_unknown_user_is_zero(db):
    assert run(ns.mark_all_read("auth-uid-unknown")) == 0