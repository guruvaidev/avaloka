"""
Durable session memory when hydration did not happen.

tests/test_memory_staged_retrieval.py holds the two cases PR #420 was caught on
(PR420-M1, PR420-M2): a fresh process whose first request connects Redis.
Moving hydration after the connect stage fixes those. These are the cases that
ordering alone does not reach -- Redis still not connected when the request
runs, or unreadable when the write-through fires -- where the only thing
standing between an in-process list and the durable copy is
`_persist_session_memory` itself. `_hydration_failed` never covered them: it
records reads that failed, not reads that were never attempted.

Real MemoryOrchestrator (built by __init__, nothing connected), with only the
four clients stubbed.
"""

import threading

import pytest

from app.services.memory_plane import MemoryOrchestrator

SID = "sess-cold-start"
PREF = "Always report revenue in INR"
OLD_HINT = "Fiscal year starts in April"
LEARNED = "User compares quarters year over year"
K_PREFS = f"ctxmem:{SID}:prefs"
K_HINTS = f"ctxmem:{SID}:hints"


class ColdRedis:
    """A Redis holding a session written before the restart."""

    def __init__(self, connectable=True):
        self.kv = {K_PREFS: [PREF], K_HINTS: [OLD_HINT, PREF]}
        self.connectable = connectable
        self.fail_reads = False

    def connect(self):
        if not self.connectable:
            raise ConnectionError("redis still starting")
        return True

    def get_json(self, key, strict=False):
        if self.fail_reads:
            if strict:
                raise RuntimeError("simulated redis outage")
            return None
        return self.kv.get(key)

    def set_json(self, key, value, ttl_seconds=None):
        self.kv[key] = list(value)
        return True

    def search_schema_semantically(self, user_id, query):
        return None


class Unreachable:
    """Postgres / Chroma / Milvus: not up, so retrieval skips their stages."""

    def connect(self):
        raise ConnectionError("not reachable in this test")


@pytest.fixture(autouse=True)
def _clean_memory():
    MemoryOrchestrator.clear_all_sessions()
    yield
    MemoryOrchestrator.clear_all_sessions()


def _cold_orchestrator(monkeypatch, redis):
    """A process that has just started: nothing connected, nothing in memory."""
    orch = MemoryOrchestrator(llm_client=object())
    monkeypatch.setattr(orch, "redis", redis)
    for name in ("postgres", "chroma", "milvus"):
        monkeypatch.setattr(orch, name, Unreachable())
    monkeypatch.setattr(orch, "_extract_context_with_llm",
                        lambda query, past_queries: {"new_hints": [LEARNED]})
    assert not any(orch._store_ready.values())
    return orch


def _finish_background_learning():
    for t in threading.enumerate():
        if t.name == "memory-learn":
            t.join(10)
            assert not t.is_alive()


def test_redis_not_connected_during_the_request_still_cannot_erase(monkeypatch):
    """Redis misses the connect stage (slow start) but is writable by the time
    background learning persists. Nothing was hydrated; nothing may be lost."""
    redis = ColdRedis(connectable=False)
    orch = _cold_orchestrator(monkeypatch, redis)

    orch._retrieve_memory_internal("revenue by quarter", SID, state={"user_id": "u1"})
    _finish_background_learning()

    assert redis.kv[K_PREFS] == [PREF]
    assert OLD_HINT in redis.kv[K_HINTS] and PREF in redis.kv[K_HINTS]

    # And once Redis connects, the next request serves the preference even
    # though the session already exists in process.
    redis.connectable = True
    result = orch._retrieve_memory_internal("and by region?", SID, state={"user_id": "u1"})
    _finish_background_learning()
    assert PREF in result["memory_hints"]
    assert OLD_HINT in result["accumulated_memory_hints"]
    assert redis.kv[K_PREFS] == [PREF]


def test_write_through_is_skipped_while_the_durable_copy_is_unreadable(monkeypatch):
    redis = ColdRedis()
    orch = _cold_orchestrator(monkeypatch, redis)

    # Learned in process without any hydration, then Redis cannot be read.
    MemoryOrchestrator._session_hints_store.setdefault(SID, []).append(LEARNED)
    redis.fail_reads = True
    orch._persist_session_memory(SID)
    assert redis.kv[K_PREFS] == [PREF] and redis.kv[K_HINTS] == [OLD_HINT, PREF]

    redis.fail_reads = False
    orch._persist_session_memory(SID)
    assert redis.kv[K_PREFS] == [PREF]
    assert redis.kv[K_HINTS] == [OLD_HINT, PREF, LEARNED]


def test_new_preference_stays_newest_after_merging_with_the_durable_copy(monkeypatch):
    redis = ColdRedis()
    orch = _cold_orchestrator(monkeypatch, redis)

    newer = "Round percentages to one decimal"
    assert orch.store_user_preference(newer, SID) is True
    assert redis.kv[K_PREFS] == [PREF, newer]     # newest last: it wins a reserved slot
