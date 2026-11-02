"""
1.6: threads that share one session do not share what they learned, and
memory stored before the thread scope is not served to any thread.

On 1.6 a thread is bound to the session it was created in (POST /threads ->
bind_thread_session), and every thread a browser opens on a dataset shares that
session. The memory plane kept learned hints, explicit preferences and past
queries per session (in process and in Redis under ctxmem:<session>:...), and
Chroma signatures and Milvus insights per user, so a hint learned in one
conversation was served to every other one. tests/test_memory_thread_scope.py
(ported from 2.0 #580) covers the user-keyed stores; this file covers the
session-keyed ones, which 2.0's tests do not reach because 2.0 threads do not
share a session.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

try:  # pragma: no cover
    import cachetools  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _ct = types.ModuleType("cachetools")

    class _TTLCache(dict):
        def __init__(self, maxsize=None, ttl=None):
            super().__init__()

    _ct.TTLCache = _TTLCache
    sys.modules["cachetools"] = _ct

from app.services.memory_plane import MemoryOrchestrator

USER = "u-16"
SESSION = "sess-shared"
TRANSFER_HINT = "Transfer the patients table to warehouse as patients_copy"
TASK_HINT = "Return the work_type with the highest stroke rate"
TASK_Q = "which work_type has the highest stroke rate?"
OTHER_Q = "average age of patients with stroke"


class _DictRedis:
    """The ctxmem write-through: JSON values by key."""

    def __init__(self):
        self.kv = {}
        self.reads = []

    def connect(self):
        return True

    def get_json(self, key, strict=False):
        self.reads.append(key)
        return self.kv.get(key)

    def set_json(self, key, value, ttl_seconds=None):
        self.kv[key] = value
        return True

    def delete_key(self, key):
        self.kv.pop(key, None)

    def search_schema_semantically(self, user_id, query, dataset_id=None):
        return None


class _Empty:
    """Chroma/Milvus/Postgres with nothing stored."""

    _vector_dim = 8

    def connect(self):
        return True

    def get_user_signature(self, key):
        return None

    def update_user_signature(self, key, sig):
        return None

    def search_similar_insights(self, key, query_vector=None, top_k=None, **_):
        return []

    def insert_insight(self, *a, **k):
        return None

    def check_artifact_exists(self, *a, **k):
        return False


def _learner(query, past_queries):
    if "work_type" in str(query):
        return {"new_hints": [TASK_HINT], "logic_signature": ""}
    return {"new_hints": [], "logic_signature": ""}


def _orchestrator():
    mo = MemoryOrchestrator.__new__(MemoryOrchestrator)
    mo._explicit_llm = object()
    mo._init_runtime_state()
    mo.redis = _DictRedis()
    mo.chroma = mo.milvus = mo.postgres = _Empty()
    mo._embed = MagicMock(return_value=[0.0] * 8)
    mo._extract_context_with_llm = _learner
    return mo


def _turn(mo, query, thread):
    state = {"user_id": USER, "session_id": SESSION, "thread_id": thread}
    result = mo._retrieve_memory_internal(query, session_id=SESSION, dataset_id="unknown_dataset",
                                          state=state)
    assert MemoryOrchestrator.wait_for_learning(timeout=300), "background learning never finished"
    return result


def _served(memory):
    return " ".join(list(memory.get("memory_hints") or [])
                    + list(memory.get("accumulated_memory_hints") or [])
                    + [memory.get("session_logic_signature") or ""])


@pytest.fixture(autouse=True)
def _memory_isolation():
    MemoryOrchestrator.clear_all_sessions()
    yield
    MemoryOrchestrator.wait_for_learning(timeout=300)
    MemoryOrchestrator.clear_all_sessions()


def test_a_hint_learned_in_one_thread_is_not_served_to_another_thread_of_the_same_session():
    mo = _orchestrator()
    _turn(mo, TASK_Q, "thread-a")
    _turn(mo, TASK_Q, "thread-a")
    assert TASK_HINT in _served(_turn(mo, TASK_Q, "thread-a")), "precondition: thread A keeps what it learned"

    other = _turn(mo, OTHER_Q, "thread-b")
    assert TASK_HINT not in _served(other), f"thread B was served thread A's hint: {other}"


def test_an_explicit_preference_stays_in_its_thread_as_a_session_hint(monkeypatch):
    """Stored through the planner's entry point. The in-process preference
    store is per thread too; what crosses threads is the Milvus copy, and only
    if it is a style preference (memory_scope)."""
    import app.services.memory_plane as mp
    mo = _orchestrator()
    monkeypatch.setattr(mp, "_orchestrator", mo)
    mp.store_user_preference(TASK_HINT, {"user_id": USER, "session_id": SESSION,
                                         "thread_id": "thread-a"})
    assert TASK_HINT in _served(_turn(mo, OTHER_Q, "thread-a")), "precondition: thread A has it"
    other = _turn(mo, OTHER_Q, "thread-b")
    assert TASK_HINT not in _served(other), other


def test_session_keyed_memory_stored_before_the_fix_is_not_served_to_any_thread():
    """Pre-fix rows: Redis ctxmem under the bare session id. They carry no
    thread, so no thread is given them -- not even one of that session."""
    mo = _orchestrator()
    mo.redis.kv[f"ctxmem:{SESSION}:hints"] = [TASK_HINT, TRANSFER_HINT]
    mo.redis.kv[f"ctxmem:{SESSION}:prefs"] = [TASK_HINT]

    for thread in ("thread-a", "thread-b"):
        memory = _turn(mo, OTHER_Q, thread)
        assert any(k.startswith("ctxmem:") for k in mo.redis.reads), \
            "precondition: the Redis copy was consulted"
        served = _served(memory)
        assert TASK_HINT not in served, f"{thread} was served pre-fix session memory: {memory}"
        assert TRANSFER_HINT not in served, f"{thread} was served pre-fix session memory: {memory}"
