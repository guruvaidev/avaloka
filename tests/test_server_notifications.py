"""Tests for the D4 notification code in app/api/server.py.

Covers:
  * _remember_pending_turn: the in-process copy that answers polls when Redis
    is down (a late "running" never overwrites "done"; the copy is capped).
  * GET /threads/{id}/pending-turn: falls back to that copy, per user.
  * GET/POST /api/notifications*: auth, results, errors.

The endpoints are called as plain async functions with auth and storage
patched, so no server, Redis or Supabase is needed. server.py is imported
once; if it cannot be imported in this environment (missing env or services
at import time) the whole file is skipped with the reason.

    pytest tests/test_server_notifications.py -q
"""

from __future__ import annotations

import asyncio

import pytest

try:
    from app.api import server
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"app.api.server could not be imported here: {exc}", allow_module_level=True)

USER = "auth-uid-1111"
OTHER_USER = "auth-uid-2222"
THREAD = "thread-abc"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def mem(monkeypatch):
    """A fresh, empty in-process pending-turn store."""
    store = {}
    monkeypatch.setattr(server, "_PENDING_TURN_MEM", store)
    return store


@pytest.fixture
def as_user(monkeypatch):
    monkeypatch.setattr(server, "_resolve_user_id", lambda req: USER)


@pytest.fixture
def anonymous(monkeypatch):
    monkeypatch.setattr(server, "_resolve_user_id", lambda req: None)


# ---------------------------------------------------------------------------
# _remember_pending_turn
# ---------------------------------------------------------------------------

def test_remember_records_running_then_done(mem):
    server._remember_pending_turn(THREAD, USER, server._pending_turn_running("turn-1"))
    assert mem[THREAD]["status"] == "running" and mem[THREAD]["user_id"] == USER
    server._remember_pending_turn(THREAD, USER, server._pending_turn_done("turn-1", {"messages": ["hi"]}))
    assert mem[THREAD]["status"] == "done"
    assert mem[THREAD]["id"] == "turn-1"


def test_late_running_does_not_overwrite_done(mem):
    server._remember_pending_turn(THREAD, USER, server._pending_turn_done("turn-1", {}))
    server._remember_pending_turn(THREAD, USER, server._pending_turn_running("turn-1"))
    assert mem[THREAD]["status"] == "done"


def test_new_turn_replaces_a_finished_one(mem):
    server._remember_pending_turn(THREAD, USER, server._pending_turn_done("turn-1", {}))
    server._remember_pending_turn(THREAD, USER, server._pending_turn_running("turn-2"))
    assert mem[THREAD]["id"] == "turn-2" and mem[THREAD]["status"] == "running"


def test_error_keeps_the_kind(mem):
    server._remember_pending_turn(THREAD, USER, server._pending_turn_running("t", kind="training"))
    server._remember_pending_turn(THREAD, USER, server._pending_turn_error("t", "boom"))
    assert mem[THREAD]["status"] == "error"
    assert mem[THREAD]["kind"] == "training"
    assert mem[THREAD]["message"] == "boom"


def test_remember_ignores_empty_thread(mem):
    server._remember_pending_turn("", USER, server._pending_turn_running("t"))
    assert mem == {}


def test_remember_never_raises(mem):
    def broken(cur):
        raise ValueError("bad mutator")
    server._remember_pending_turn(THREAD, USER, broken)
    assert mem == {}


def test_remember_drops_oldest_beyond_the_cap(mem, monkeypatch):
    monkeypatch.setattr(server, "_PENDING_TURN_MEM_MAX", 3)
    for i in range(5):
        server._remember_pending_turn(f"thread-{i}", USER, server._pending_turn_running(f"t{i}"))
    assert list(mem) == ["thread-2", "thread-3", "thread-4"]


# ---------------------------------------------------------------------------
# GET /threads/{id}/pending-turn  (memory fallback)
# ---------------------------------------------------------------------------

@pytest.fixture
def redis_down(monkeypatch):
    async def no_session(*a, **k):
        return None
    monkeypatch.setattr(server, "get_thread_session", no_session)
    monkeypatch.setattr(server, "get_session", no_session)


@pytest.fixture
def redis_running(monkeypatch):
    """Redis holds the session but still says 'running' (the done write failed)."""
    async def thread_session(thread_id):
        return "sid-1"

    async def session(sid):
        return {"user_id": USER, "pending_turn": {"id": "turn-1", "status": "running", "kind": "analysis"}}

    monkeypatch.setattr(server, "get_thread_session", thread_session)
    monkeypatch.setattr(server, "get_session", session)


def poll(deferred_id=None):
    return run(server.get_pending_turn(None, THREAD, deferred_id=deferred_id))


def test_poll_answers_from_memory_when_redis_is_down(as_user, mem, redis_down):
    server._remember_pending_turn(THREAD, USER, server._pending_turn_done("turn-1", {"messages": ["answer"]}))
    out = poll("turn-1")
    assert out["status"] == "done"
    assert out["result"]["messages"] == ["answer"]


def test_poll_prefers_finished_memory_over_stale_running(as_user, mem, redis_running):
    server._remember_pending_turn(THREAD, USER, server._pending_turn_done("turn-1", {"messages": ["answer"]}))
    assert poll("turn-1")["status"] == "done"


def test_poll_reports_running_when_nothing_finished(as_user, mem, redis_running):
    assert poll("turn-1") == {"status": "running", "kind": "analysis", "queued": False}


def test_poll_does_not_leak_another_users_turn(monkeypatch, mem, redis_down):
    server._remember_pending_turn(THREAD, OTHER_USER, server._pending_turn_done("turn-1", {}))
    monkeypatch.setattr(server, "_resolve_user_id", lambda req: USER)
    with pytest.raises(server.HTTPException) as err:
        poll("turn-1")
    assert err.value.status_code == 404


def test_poll_superseded_by_a_newer_turn(as_user, mem, redis_down):
    server._remember_pending_turn(THREAD, USER, server._pending_turn_running("turn-2"))
    assert poll("turn-1") == {"status": "superseded"}


def test_poll_error_has_a_default_message(as_user, mem, redis_down):
    server._remember_pending_turn(THREAD, USER, server._pending_turn_error("turn-1", ""))
    out = poll("turn-1")
    assert out["status"] == "error" and out["message"]


def test_poll_requires_auth(anonymous, mem, redis_down):
    with pytest.raises(server.HTTPException) as err:
        poll()
    assert err.value.status_code == 401


# ---------------------------------------------------------------------------
# /api/notifications routes
# ---------------------------------------------------------------------------

@pytest.fixture
def service(monkeypatch):
    """Patch the notification service the routes call; records the arguments."""
    state = {"calls": [], "fail": False, "mark_ok": True}

    async def list_notifications(user_id, unread_only=False, limit=20):
        state["calls"].append(("list", user_id, unread_only, limit))
        if state["fail"]:
            raise RuntimeError("supabase down")
        return [{"id": "n-1", "read_at": None}], 1

    async def mark_all_read(user_id):
        state["calls"].append(("read-all", user_id))
        if state["fail"]:
            raise RuntimeError("supabase down")
        return 3

    async def mark_read(user_id, notification_id):
        state["calls"].append(("read", user_id, notification_id))
        if state["fail"]:
            raise RuntimeError("supabase down")
        return state["mark_ok"]

    ns = server.notification_service
    monkeypatch.setattr(ns, "list_notifications", list_notifications)
    monkeypatch.setattr(ns, "mark_all_read", mark_all_read)
    monkeypatch.setattr(ns, "mark_read", mark_read)
    return state


def test_list_returns_items_and_unread_count(as_user, service):
    out = run(server.get_notifications(None, unread=True, limit=5))
    assert out == {"items": [{"id": "n-1", "read_at": None}], "unread_count": 1}
    assert service["calls"] == [("list", USER, True, 5)]


def test_read_all_returns_count(as_user, service):
    assert run(server.read_all_notifications(None)) == {"updated": 3}


def test_read_one(as_user, service):
    assert run(server.read_notification(None, "n-1")) == {"ok": True}
    assert service["calls"] == [("read", USER, "n-1")]


def test_read_someone_elses_notification_is_404(as_user, service):
    service["mark_ok"] = False
    with pytest.raises(server.HTTPException) as err:
        run(server.read_notification(None, "n-x"))
    assert err.value.status_code == 404


@pytest.mark.parametrize("call", [
    lambda: server.get_notifications(None, unread=False, limit=20),
    lambda: server.read_all_notifications(None),
    lambda: server.read_notification(None, "n-1"),
])
def test_routes_require_auth(anonymous, service, call):
    with pytest.raises(server.HTTPException) as err:
        run(call())
    assert err.value.status_code == 401
    assert service["calls"] == []


@pytest.mark.parametrize("call", [
    lambda: server.get_notifications(None, unread=False, limit=20),
    lambda: server.read_all_notifications(None),
    lambda: server.read_notification(None, "n-1"),
])
def test_routes_return_503_when_storage_fails(as_user, service, call):
    service["fail"] = True
    with pytest.raises(server.HTTPException) as err:
        run(call())
    assert err.value.status_code == 503