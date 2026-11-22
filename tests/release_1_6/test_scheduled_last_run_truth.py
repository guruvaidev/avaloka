"""A scheduled task that has never run must not report a last-run time.

celery's ScheduleEntry seeds last_run_at with the creation time (RedBeat keeps
it as the schedule anchor), and AvalokaEntry.save() persists it. GET /tasks and
GET /tasks/{id}/info used to return that anchor as last_run_at, so the
Scheduled Analysis table showed "Last Run 0s ago" next to "Ran 0 times".
"""
from __future__ import annotations

import datetime

import pytest
from celery.schedules import schedule as interval
from fastapi.testclient import TestClient

import app.api.server as server
from app.core.celery_app import AvalokaEntry, celery_app

try:
    from tests.server_harness import SESSIONS, client, make_auth_headers  # noqa: F401
except ImportError:
    from server_harness import SESSIONS, client, make_auth_headers  # noqa: F401

RAN_AT = datetime.datetime(2026, 10, 30, 9, 0, tzinfo=datetime.timezone.utc)


def _fresh_entry() -> AvalokaEntry:
    """A just-created entry exactly as the scheduler builds it (never dispatched)."""
    return AvalokaEntry(
        name="t1",
        task="app.core.celery_app.run_scheduled",
        schedule=interval(datetime.timedelta(minutes=1)),
        args=[{"session_id": "sess-lastrun", "user_id": "lastrun-user"}],
        app=celery_app,
    )


@pytest.fixture
def wired(monkeypatch):
    """Session owning task t1; RedBeat and the run store are faked per test."""
    state = {"entry": _fresh_entry(), "runs": []}
    monkeypatch.setattr(
        server.AvalokaEntry, "from_key", staticmethod(lambda *_a, **_k: state["entry"])
    )
    monkeypatch.setattr(server, "_discover_session_tasks", lambda *_a: [])

    async def no_runs(_task_id):
        return []

    monkeypatch.setattr(server, "_load_scheduled_runs", no_runs)
    monkeypatch.setattr(
        server.AvalokaScheduler,
        "get_task_ids_with_timestamp",
        staticmethod(lambda *_a, **_k: list(state["runs"])),
    )
    SESSIONS["sess-lastrun"] = {"user_id": "lastrun-user", "tasks": ["t1"]}
    headers = make_auth_headers("lastrun-user")
    headers["X-Avaloka-Session"] = "sess-lastrun"
    yield state, headers
    SESSIONS.pop("sess-lastrun", None)


def test_fresh_entry_really_carries_a_creation_time_anchor():
    # The premise of the bug: without this, the tests below would be vacuous.
    entry = _fresh_entry()
    assert entry.total_run_count == 0
    assert entry.last_run_at is not None


def test_task_list_reports_no_last_run_for_a_never_run_task(client: TestClient, wired):
    _state, headers = wired
    resp = client.get("/tasks", headers=headers)
    assert resp.status_code == 200, resp.text
    (task,) = resp.json()
    assert task["total_run_count"] == 0
    assert task["last_run_at"] is None


def test_task_list_reports_the_real_last_run_once_it_ran(client: TestClient, wired):
    state, headers = wired
    entry = state["entry"]
    entry.total_run_count = 1
    entry.last_run_at = RAN_AT
    resp = client.get("/tasks", headers=headers)
    (task,) = resp.json()
    assert task["total_run_count"] == 1
    assert task["last_run_at"] == RAN_AT.isoformat()


def test_task_info_reports_no_last_run_for_a_never_run_task(client: TestClient, wired):
    _state, headers = wired
    resp = client.get("/tasks/t1/info", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_run_count"] == 0
    assert body["last_run_at"] is None


def test_task_info_reports_the_real_last_run_once_it_ran(client: TestClient, wired):
    state, headers = wired
    state["runs"] = [("celery-id-1", RAN_AT)]
    state["entry"].total_run_count = 1
    state["entry"].last_run_at = RAN_AT
    body = client.get("/tasks/t1/info", headers=headers).json()
    assert body["total_run_count"] == 1
    assert body["last_run_at"] == RAN_AT.isoformat()
