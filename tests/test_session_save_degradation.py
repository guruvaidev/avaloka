"""save_session under a Redis brownout.

The reported failure: get_session timed out at 00:32:12 and opened the cache
circuit; save_session ran at 00:32:25 anyway, spent ~13s inside redis-py's
connect/retry budget, raised, and lost the turn's state.

Two things were wrong, and they are opposites: the write neither reported its
failures into the breaker nor respected one that was already open.
"""

import asyncio
import time

import pytest
import redis.exceptions

from app.services import session_service as ss


@pytest.fixture(autouse=True)
def _reset_breaker(monkeypatch):
    monkeypatch.setattr(ss, "CACHE_DEGRADED_UNTIL", 0.0, raising=False)
    monkeypatch.setattr(ss, "_last_cache_warn", 0.0, raising=False)
    yield
    ss.CACHE_DEGRADED_UNTIL = 0.0


class FakeCache:
    """A Redis stand-in whose `set` can hang or fail like the real one."""

    def __init__(self, *, hang: float = 0.0, raises: Exception = None):
        self.hang = hang
        self.raises = raises
        self.sets = 0

    async def set(self, key, value, ex=None):
        self.sets += 1
        if self.hang:
            await asyncio.sleep(self.hang)
        if self.raises:
            raise self.raises

    async def sadd(self, key, *members):
        return None


SESSION = {"user_id": "u1", "dataset_id": "ds1", "foo": "bar"}


def _save(sid, data):
    """Drive the coroutine the way the rest of the suite does."""
    return asyncio.run(ss.save_session(sid, data))


# --------------------------------------------------------------------------- #
# Reporting: a failing write must open the circuit
# --------------------------------------------------------------------------- #

def test_a_failed_write_opens_the_circuit(monkeypatch):
    """Before this, only reads reported failures, so a brownout first seen by a
    write stayed invisible to get_session -- which then paid full timeouts to
    rediscover it."""
    monkeypatch.setattr(ss, "cache", FakeCache(
        raises=redis.exceptions.TimeoutError("Timeout reading from localhost:6380")))

    assert ss._cache_unhealthy() is False
    assert _save("sid-1", SESSION) is False
    assert ss._cache_unhealthy() is True, "a failed write left the breaker closed"


def test_a_successful_write_leaves_the_circuit_closed(monkeypatch):
    monkeypatch.setattr(ss, "cache", FakeCache())
    assert _save("sid-1", SESSION) is True
    assert ss._cache_unhealthy() is False


# --------------------------------------------------------------------------- #
# Respecting: an open circuit must shorten the write, not skip it
# --------------------------------------------------------------------------- #

def test_a_write_is_still_attempted_while_degraded(monkeypatch):
    """Skipping a write is not graceful degradation, it is silent data loss.
    Redis may well have recovered since the breaker opened."""
    monkeypatch.setattr(ss, "cache", cache := FakeCache())
    ss.CACHE_DEGRADED_UNTIL = time.time() + 30

    assert _save("sid-1", SESSION) is True
    assert cache.sets > 0, "the write was skipped rather than attempted"


def test_a_degraded_write_fails_fast(monkeypatch):
    """The reported 13s stall. redis-py's own budget cannot bound this, so the
    deadline lives here."""
    monkeypatch.setattr(ss, "cache", FakeCache(hang=30))
    monkeypatch.setattr(ss, "SESSION_SAVE_TIMEOUT_DEGRADED", 0.2)
    ss.CACHE_DEGRADED_UNTIL = time.time() + 30

    started = time.monotonic()
    assert _save("sid-1", SESSION) is False
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"degraded write took {elapsed:.1f}s; expected a fast failure"


def test_a_healthy_write_gets_the_longer_budget(monkeypatch):
    """A slow-but-successful write still beats a lost one, so the healthy path
    is not held to the degraded deadline."""
    monkeypatch.setattr(ss, "cache", FakeCache(hang=0.3))
    monkeypatch.setattr(ss, "SESSION_SAVE_TIMEOUT", 5.0)
    monkeypatch.setattr(ss, "SESSION_SAVE_TIMEOUT_DEGRADED", 0.05)

    assert _save("sid-1", SESSION) is True


# --------------------------------------------------------------------------- #
# Logging: infrastructure noise vs. real bugs
# --------------------------------------------------------------------------- #

def test_a_brownout_logs_without_a_traceback(monkeypatch, caplog):
    """Twelve frames of redis-py do not explain a brownout. The interesting
    fact is that a session was lost."""
    monkeypatch.setattr(ss, "cache", FakeCache(
        raises=redis.exceptions.TimeoutError("Timeout reading from localhost:6380")))

    with caplog.at_level("DEBUG", logger="avaloka.session"):
        _save("sid-1", SESSION)

    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errors, "losing session state must still be an ERROR"
    assert not any(r.exc_info for r in errors), "expected no traceback for a brownout"
    assert "NOT persisted" in errors[0].getMessage()


def test_an_unexpected_error_keeps_its_traceback(monkeypatch, caplog):
    """A bug in _jsonify or a bad key type is not a brownout; do not hide it."""
    monkeypatch.setattr(ss, "cache", FakeCache(raises=ValueError("programming error")))

    with caplog.at_level("DEBUG", logger="avaloka.session"):
        assert _save("sid-1", SESSION) is False

    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any(r.exc_info for r in errors), "a real bug lost its traceback"


# --------------------------------------------------------------------------- #
# Unchanged behaviour
# --------------------------------------------------------------------------- #

def test_a_session_without_user_id_is_still_refused(monkeypatch):
    monkeypatch.setattr(ss, "cache", cache := FakeCache())
    assert _save("sid-1", {"foo": "bar"}) is False
    assert cache.sets == 0


def test_no_cache_configured_is_still_false(monkeypatch):
    monkeypatch.setattr(ss, "cache", None)
    assert _save("sid-1", SESSION) is False
