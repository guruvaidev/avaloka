"""Staged memory retrieval: what PR #420 changed in app/services/memory_plane.py.

The PR replaced one try-block and one shared breaker with per-stage timeouts,
connect-once-per-process, a single query embedding, and background learning.
It shipped with no tests, and it broke 13 existing ones
(tests/test_memory_top3_relevance.py) because they build the orchestrator with
``__new__`` and the new per-instance state lives only in ``__init__``.

The tests pin the behaviour the PR's own header promises, so the next rewrite
cannot quietly lose it. The last section holds the four cases (PR420-M1..M4)
that were confirmed by running the merged code (73a67a48) and carried as
xfail(strict=True) until fixed; all four are fixed and unmarked. M4 was only
half the PR's doing -- see the note on it.

No store, no LLM, no network: every tier is a MagicMock and the embedding is a
stub, so a pass here is a statement about the orchestration logic only.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import pytest

import app.services.memory_plane as mp
from app.services.memory_plane import MemoryOrchestrator

QUERY = "average fare by borough"
DOMAIN = "trips(fare, borough, pickup_at)"


def _orchestrator(llm=None, ready: bool = True) -> MemoryOrchestrator:
    """Every tier mocked. ``ready`` is what a completed warm-up leaves behind;
    ``ready=False`` is a fresh process whose first request arrives first."""
    orch = MemoryOrchestrator.__new__(MemoryOrchestrator)
    orch._explicit_llm = llm
    orch._init_runtime_state()
    orch._store_ready = {name: ready for name in MemoryOrchestrator._STORES}
    for name in MemoryOrchestrator._STORES:
        setattr(orch, name, MagicMock(name=name))
    orch.postgres.check_artifact_exists.return_value = False
    orch.redis.search_schema_semantically.return_value = None
    orch.redis.get_json.return_value = None
    orch.chroma.get_user_signature.return_value = None
    orch.milvus.search_similar_insights.return_value = []
    orch._embed = MagicMock(return_value=[0.0] * 8)
    return orch


def _durable_redis(orch: MemoryOrchestrator) -> dict:
    """Back the mocked Redis JSON calls with a dict, i.e. a Redis that survived
    the restart the test is simulating."""
    data: dict = {}
    orch.redis.get_json.side_effect = lambda key, strict=False: data.get(key)
    orch.redis.set_json.side_effect = (
        lambda key, value, ttl_seconds=None: data.__setitem__(key, list(value))
    )
    return data


def _wait_for_learning() -> None:
    """Block until background learning has landed.

    A join on the learner threads themselves (MemoryOrchestrator
    .wait_for_learning), not a poll: it returns the moment they finish, however
    loaded the machine is. The ceiling is not a tuning knob -- it only turns a
    learner that never returns into a failure instead of a hung suite.
    """
    assert MemoryOrchestrator.wait_for_learning(timeout=300), "background learning never finished"


@pytest.fixture(autouse=True)
def _memory_isolation():
    MemoryOrchestrator.clear_all_sessions()
    yield
    _wait_for_learning()
    MemoryOrchestrator.clear_all_sessions()


# ---------------------------------------------------------------------------
# What the PR promises: a bad store costs only its own hints
# ---------------------------------------------------------------------------


def test_a_failing_store_keeps_the_hints_from_the_others():
    orch = _orchestrator()
    orch.postgres.check_artifact_exists.side_effect = RuntimeError("postgres down")
    orch.redis.search_schema_semantically.return_value = DOMAIN

    out = orch._retrieve_memory_internal(QUERY, session_id="s-fail")

    assert out["memory_hints"] == [f"Domain insight: {DOMAIN}"]
    assert out["prior_artifact_found"] is False


def test_a_failing_stage_marks_only_its_own_store_for_reconnect():
    orch = _orchestrator()
    orch.chroma.get_user_signature.side_effect = RuntimeError("chroma reset")

    orch._retrieve_memory_internal(QUERY, session_id="s-mark")

    assert orch._store_ready == {
        "postgres": True, "redis": True, "chroma": False, "milvus": True,
    }


def test_a_slow_stage_is_skipped_and_the_rest_is_kept(monkeypatch):
    monkeypatch.setattr(mp, "_STAGE_TIMEOUT", 0.1)
    orch = _orchestrator()
    orch.redis.search_schema_semantically.return_value = DOMAIN
    orch.milvus.search_similar_insights.side_effect = lambda *a, **k: time.sleep(1.0) or []

    start = time.monotonic()
    out = orch._retrieve_memory_internal(QUERY, session_id="s-slow")
    elapsed = time.monotonic() - start

    assert f"Domain insight: {DOMAIN}" in out["memory_hints"]
    assert elapsed < 0.9, f"retrieval waited {elapsed:.2f}s for a stage it should have skipped"


def test_a_failing_mcp_server_does_not_mark_redis_for_reconnect():
    """The hot-load runs after a Redis cache miss, but its failure is the MCP
    server's: every other Redis stage answered."""
    module = types.ModuleType("app.services.mcp_cache_loader")

    def fetch_schema_sync(url, key, dataset_id):
        raise ConnectionError("mcp server down")

    module.fetch_schema_sync = fetch_schema_sync
    orch = _orchestrator()
    with patch.dict(sys.modules, {"app.services.mcp_cache_loader": module}):
        orch._retrieve_memory_internal(
            QUERY, session_id="s-mcp-down", dataset_id="trips",
            state={"input_data_type": "postgres"},
        )

    assert orch._store_ready["redis"] is True


def test_a_hung_store_holds_one_thread_not_one_per_request(monkeypatch):
    """A stage abandoned on timeout is still running. Until it lets go, later
    requests skip that stage outright -- no second thread on the same hung
    store, and no second wait for its timeout."""
    monkeypatch.setattr(mp, "_STAGE_TIMEOUT", 0.3)
    release = threading.Event()
    orch = _orchestrator()
    orch.redis.search_schema_semantically.return_value = DOMAIN
    orch.milvus.search_similar_insights.side_effect = lambda *a, **k: release.wait(20) and []

    try:
        orch._retrieve_memory_internal(QUERY, session_id="s-leak")
        start = time.monotonic()
        for _ in range(3):
            out = orch._retrieve_memory_internal(QUERY, session_id="s-leak")
            assert out["memory_hints"] == [f"Domain insight: {DOMAIN}"]
        later = time.monotonic() - start
        calls = orch.milvus.search_similar_insights.call_count
        assert calls == 1, f"{calls} threads parked on one hung store"
        assert later < 2 * mp._STAGE_TIMEOUT, f"three later requests waited {later:.2f}s on a store known to be hung"
    finally:
        release.set()

    for t in threading.enumerate():
        if t.name == "memory-stage":
            t.join(5)
    orch.milvus.search_similar_insights.side_effect = None
    orch.milvus.search_similar_insights.return_value = [{"content": "back"}]
    out = orch._retrieve_memory_internal(QUERY, session_id="s-leak")
    assert "back" in out["memory_hints"], "the stage must run again once the hung call has returned"


def test_connections_are_opened_once_per_process_not_once_per_request():
    orch = _orchestrator(ready=False)

    orch._retrieve_memory_internal(QUERY, session_id="s-conn")
    orch._retrieve_memory_internal(QUERY, session_id="s-conn")

    for name in MemoryOrchestrator._STORES:
        assert getattr(orch, name).connect.call_count == 1, f"{name} was reconnected"


def test_connect_does_not_queue_behind_a_connection_already_in_progress():
    """A hung warm-up used to hold one lock and stall every request's connect."""
    orch = _orchestrator(ready=False)
    orch._store_locks["milvus"].acquire()
    try:
        start = time.monotonic()
        assert orch._connect_store("milvus") is False
        assert time.monotonic() - start < 0.5
    finally:
        orch._store_locks["milvus"].release()
    orch.milvus.connect.assert_not_called()


def test_the_query_is_embedded_once_even_on_the_legacy_session_read():
    orch = _orchestrator()

    orch._retrieve_memory_internal(
        QUERY, session_id="s-embed", state={"user_id": "user-7"},
    )

    assert orch.milvus.search_similar_insights.call_count == 2, (
        "expected the user-scoped read and the legacy session-scoped read"
    )
    assert orch._embed.call_count == 1


def test_the_query_is_not_embedded_when_milvus_is_not_connected():
    orch = _orchestrator(ready=False)
    orch.milvus.connect.side_effect = RuntimeError("milvus down")

    orch._retrieve_memory_internal(QUERY, session_id="s-noembed")

    orch._embed.assert_not_called()
    orch.milvus.search_similar_insights.assert_not_called()


# ---------------------------------------------------------------------------
# Learning moved off the request path
# ---------------------------------------------------------------------------


def test_retrieval_returns_before_the_llm_extraction_finishes():
    release = threading.Event()
    orch = _orchestrator(llm=object())

    def slow_extract(query, past_queries):
        release.wait(5.0)
        return {"new_hints": ["user reports in EUR"], "logic_signature": ""}

    orch._extract_context_with_llm = slow_extract
    try:
        start = time.monotonic()
        out = orch._retrieve_memory_internal(QUERY, session_id="s-bg")
        elapsed = time.monotonic() - start
    finally:
        release.set()

    assert elapsed < 2.0, f"retrieval blocked on the LLM for {elapsed:.2f}s"
    assert out["memory_hints"] == []
    assert out["new_hints_this_context"] == []


def test_a_hint_learned_in_the_background_reaches_the_next_turn():
    orch = _orchestrator(llm=object())
    orch._extract_context_with_llm = lambda query, past: {
        "new_hints": ["user reports in EUR"], "logic_signature": "prefers polars",
    }

    orch._retrieve_memory_internal(QUERY, session_id="s-next")
    _wait_for_learning()
    # Checked before the second turn, which starts a learning thread of its own.
    orch.milvus.insert_insight.assert_called_once()
    orch.chroma.update_user_signature.assert_called_once_with("s-next", "prefers polars")

    out = orch._retrieve_memory_internal(QUERY, session_id="s-next")

    assert "user reports in EUR" in out["memory_hints"]


def test_waiting_for_learning_returns_when_the_learner_does_not_before():
    """The wait is a join on the learner, so it cannot return early and cannot
    be late: it reports False while the learner is held, True once released."""
    release = threading.Event()
    orch = _orchestrator(llm=object())

    def held(query, past):
        release.wait(60)
        return {"new_hints": ["user reports in EUR"], "logic_signature": ""}

    orch._extract_context_with_llm = held
    orch._retrieve_memory_internal(QUERY, session_id="s-wait")
    try:
        assert MemoryOrchestrator.wait_for_learning(timeout=0.2) is False
        assert "s-wait" not in MemoryOrchestrator._session_hints_store
    finally:
        release.set()

    assert MemoryOrchestrator.wait_for_learning(timeout=300) is True
    assert MemoryOrchestrator._session_hints_store["s-wait"] == ["user reports in EUR"]
    assert not MemoryOrchestrator._learning, "finished learners must not accumulate"


def test_background_learning_never_raises_into_the_process():
    orch = _orchestrator(llm=object())

    def boom(query, past):
        raise RuntimeError("provider 500")

    orch._extract_context_with_llm = boom
    orch._learn_in_background(QUERY, [], "s-boom", "s-boom")


# ---------------------------------------------------------------------------
# MCP schema hot-load is for database sources only
# ---------------------------------------------------------------------------


def _fake_mcp_loader(calls: list):
    module = types.ModuleType("app.services.mcp_cache_loader")

    def fetch_schema_sync(url, key, dataset_id):
        calls.append(dataset_id)
        return {"tables": {"trips": ["fare", "borough"]}}

    module.fetch_schema_sync = fetch_schema_sync
    return module


def test_a_csv_upload_does_not_wait_on_the_mcp_server():
    calls: list = []
    orch = _orchestrator()
    with patch.dict(sys.modules, {"app.services.mcp_cache_loader": _fake_mcp_loader(calls)}):
        orch._retrieve_memory_internal(
            QUERY, session_id="s-csv", dataset_id="trips.csv",
            state={"input_data_type": "csv"},
        )
    assert calls == []


@pytest.mark.parametrize("state", [
    {"input_data_type": "postgres"},
    {"active_db_source_table": "trips"},
    {"active_db_customer_id": "cust-1"},
])
def test_a_database_source_hot_loads_its_schema(state):
    calls: list = []
    orch = _orchestrator()
    with patch.dict(sys.modules, {"app.services.mcp_cache_loader": _fake_mcp_loader(calls)}):
        out = orch._retrieve_memory_internal(
            QUERY, session_id="s-db", dataset_id="trips", state=state,
        )
    assert calls == ["trips"]
    assert any("hot-loaded via MCP" in hint for hint in out["memory_hints"])
    orch.redis.set_schema.assert_called_once()


# ---------------------------------------------------------------------------
# Durable write-through: one session's read-merge-write is not interleaved,
# and the lock every retrieval takes is not held across Redis
# ---------------------------------------------------------------------------


def test_the_in_process_lock_is_not_held_across_redis():
    """_store_lock is taken by every retrieval. Redis calls block for up to the
    socket timeout, so holding it across them stalls every other session."""
    orch = _orchestrator()
    held_during = []

    def record(*args, **kwargs):
        free = MemoryOrchestrator._store_lock.acquire(blocking=False)
        if free:
            MemoryOrchestrator._store_lock.release()
        held_during.append(not free)

    orch.redis.get_json.side_effect = record
    orch.redis.set_json.side_effect = record
    MemoryOrchestrator._session_hints_store["s-lock"] = ["a hint"]

    orch._persist_session_memory("s-lock")

    assert len(held_during) == 4, "expected two durable reads and two writes"
    assert not any(held_during), "_store_lock was held during a Redis call"


def test_two_writers_for_one_session_do_not_interleave():
    """Each write-through reads the durable copy, merges, and writes. If a
    second writer reads between the first one's read and write, whichever
    writes last decides -- on a copy that is already out of date."""
    orch = _orchestrator()
    events = []
    first_read = threading.Event()
    hold = threading.Event()

    def get_json(key, strict=False):
        events.append(("read", threading.current_thread().name))
        if not first_read.is_set():
            first_read.set()
            hold.wait(5)  # park the first writer between its read and its write
        return None

    orch.redis.get_json.side_effect = get_json
    orch.redis.set_json.side_effect = (
        lambda key, value, ttl_seconds=None: events.append(("write", threading.current_thread().name)))
    MemoryOrchestrator._session_hints_store["s-two"] = ["a hint"]

    writers = [threading.Thread(target=orch._persist_session_memory, args=("s-two",), name=f"w{i}")
               for i in range(2)]
    writers[0].start()
    assert first_read.wait(5)
    writers[1].start()
    time.sleep(0.2)  # give the second writer every chance to barge in
    in_between = list(events)
    hold.set()
    for w in writers:
        w.join(5)

    assert in_between == [("read", "w0")], f"second writer ran inside the first one's window: {in_between}"
    assert [who for _, who in events] == ["w0"] * 4 + ["w1"] * 4


# ---------------------------------------------------------------------------
# Controls for the defect tests below
#
# Each defect test differs from its control in ONE input. If a control ever
# fails, the matching defect test is failing because of these stubs, not
# because of the defect it names -- fix the stubs before trusting either.
# ---------------------------------------------------------------------------


def _restart(session: str, **kwargs):
    orch = _orchestrator(**kwargs)
    durable = _durable_redis(orch)
    durable[mp._k_ctxmem_prefs(session)] = ["always report in EUR"]
    durable[mp._k_ctxmem_hints(session)] = ["always report in EUR"]
    return orch, durable


def test_control_a_durable_preference_is_served_when_redis_is_already_connected():
    orch, _ = _restart("s-ctl-serve", ready=True)

    out = orch._retrieve_memory_internal(QUERY, session_id="s-ctl-serve")

    assert out["memory_hints"] == ["always report in EUR"]


def test_control_learning_keeps_durable_preferences_when_redis_is_already_connected():
    orch, durable = _restart("s-ctl-keep", llm=object(), ready=True)
    orch._extract_context_with_llm = lambda query, past: {
        "new_hints": ["asked about fares"], "logic_signature": "",
    }

    orch._retrieve_memory_internal(QUERY, session_id="s-ctl-keep")
    _wait_for_learning()

    assert durable[mp._k_ctxmem_prefs("s-ctl-keep")] == ["always report in EUR"]
    assert durable[mp._k_ctxmem_hints("s-ctl-keep")] == ["always report in EUR", "asked about fares"]


def _three_hung_stages(monkeypatch, breaker: float, session: str) -> dict:
    """Shipped stage timeouts divided by 20; only the breaker varies."""
    monkeypatch.setattr(mp, "_CIRCUIT_BREAKER_TIMEOUT", breaker)
    monkeypatch.setattr(mp, "_STAGE_TIMEOUT", 3.0 / 20)
    monkeypatch.setattr(mp, "_CONNECT_TIMEOUT", 5.0 / 20)
    monkeypatch.setattr(mp, "_EMBED_TIMEOUT", 4.0 / 20)

    orch = _orchestrator()
    assert orch.store_user_preference("always report in EUR", session) is True

    def hang(*args, **kwargs):
        time.sleep(1.0)

    orch.postgres.check_artifact_exists.side_effect = hang
    orch.redis.search_schema_semantically.side_effect = hang
    orch.milvus.search_similar_insights.side_effect = hang
    return orch.retrieve_memory(QUERY, session_id=session)


def test_control_three_hung_stages_keep_the_preference_under_a_long_enough_breaker(monkeypatch):
    out = _three_hung_stages(monkeypatch, breaker=5.0, session="s-ctl-hung")

    assert out["memory_hints"] == ["always report in EUR"]


# ---------------------------------------------------------------------------
# Defects confirmed on the merged PR, all since fixed
# ---------------------------------------------------------------------------


# Was DEFECT PR420-M1 (data loss), fixed: hydration was gated on
# _store_ready['redis'], checked BEFORE the connect stage, so the first request
# of a fresh process loaded no durable preferences. It now runs after connect.
def test_a_remembered_preference_is_served_on_the_first_turn_after_a_restart():
    orch = _orchestrator(ready=False)
    durable = _durable_redis(orch)
    durable[mp._k_ctxmem_prefs("s-restart")] = ["always report in EUR"]
    durable[mp._k_ctxmem_hints("s-restart")] = ["always report in EUR"]

    out = orch._retrieve_memory_internal(QUERY, session_id="s-restart")

    assert out["memory_hints"] == ["always report in EUR"]


# Was DEFECT PR420-M2 (data loss), fixed: after a skipped hydration, background
# learning wrote its own in-process view over Redis -- ctxmem:<session>:prefs
# became []. _persist_session_memory now merges with the durable copy first.
# tests/test_memory_cold_start_hydration.py covers the case this one cannot:
# Redis still not connected when the request runs.
def test_background_learning_does_not_erase_durable_preferences_after_a_restart():
    orch = _orchestrator(llm=object(), ready=False)
    orch._extract_context_with_llm = lambda query, past: {
        "new_hints": ["asked about fares"], "logic_signature": "",
    }
    durable = _durable_redis(orch)
    durable[mp._k_ctxmem_prefs("s-erase")] = ["always report in EUR"]
    durable[mp._k_ctxmem_hints("s-erase")] = ["always report in EUR"]

    orch._retrieve_memory_internal(QUERY, session_id="s-erase")
    _wait_for_learning()

    assert durable[mp._k_ctxmem_prefs("s-erase")] == ["always report in EUR"]
    assert "always report in EUR" in durable[mp._k_ctxmem_hints("s-erase")]


# Was DEFECT PR420-M3 (LL-06 regression), fixed: memory_context_unavailable
# was `not self.llm` and nothing else, so with every store failing to connect
# and an LLM configured it reported False -- 'memory ran and found nothing',
# which tests/test_memory_plane.py calls the silent degrade that hid this
# subsystem. It is now also True when no store answered a single stage.
def test_memory_says_so_when_no_store_could_be_reached():
    orch = _orchestrator(llm=object(), ready=False)
    orch._extract_context_with_llm = lambda query, past: {}
    for name in MemoryOrchestrator._STORES:
        getattr(orch, name).connect.side_effect = RuntimeError(f"{name} down")

    out = orch._retrieve_memory_internal(QUERY, session_id="s-down")

    assert out["memory_hints"] == []
    assert out["memory_context_unavailable"] is True


def test_memory_is_available_when_one_store_answers():
    """The other side of M3: one store down is a degraded turn, not an outage."""
    orch = _orchestrator(llm=object(), ready=False)
    orch._extract_context_with_llm = lambda query, past: {}
    for name in ("postgres", "chroma", "milvus"):
        getattr(orch, name).connect.side_effect = RuntimeError(f"{name} down")

    out = orch._retrieve_memory_internal(QUERY, session_id="s-one-up")

    assert out["memory_context_unavailable"] is False


# Was marked DEFECT PR420-M4. The marker ran two things together, and only one
# of them was PR #420's:
#
#   * #420's: the breaker dropped from 20s to 8s while three store stages get
#     3s each, so three slow stores (9s) trip it. At #420's parent the same
#     delays stayed inside 20s. (Two slow stores, 6s, sit on the edge.) Fixed
#     by giving the stages a shared budget that ends before the breaker:
#     test_however_many_stores_hang_retrieval_returns_inside_the_breaker.
#   * NOT #420's: a breaker that does fire returned the empty payload, dropping
#     even the explicit preferences held in process memory, which need no
#     store. That was so before the PR as well. Fixed too:
#     test_a_tripped_breaker_still_serves_what_the_process_holds.
#
# This test passes if either fix holds, which is why each has its own test.
# Timeouts are the 8.0s-era defaults divided by 20.
def test_hung_stores_do_not_cost_the_preferences_held_in_process(monkeypatch):
    out = _three_hung_stages(monkeypatch, breaker=8.0 / 20, session="s-hung")

    assert out["memory_hints"] == ["always report in EUR"]


def test_a_tripped_breaker_still_serves_what_the_process_holds(monkeypatch):
    """The backstop itself. The test above is normally satisfied by the stage
    budget, so this one forces the breaker: retrieval hangs outside any stage.
    An explicit preference needs no store and must survive, and the payload
    must still say the stores were not consulted."""
    monkeypatch.setattr(mp, "_CIRCUIT_BREAKER_TIMEOUT", 0.2)
    release = threading.Event()
    orch = _orchestrator()
    assert orch.store_user_preference("always report in EUR", "s-tripped") is True
    orch._retrieve_memory_internal = lambda *a, **k: release.wait(20)

    try:
        out = orch.retrieve_memory(QUERY, session_id="s-tripped")
    finally:
        release.set()

    assert out["memory_hints"] == ["always report in EUR"]
    assert out["accumulated_memory_hints"] == ["always report in EUR"]
    assert out["memory_context_unavailable"] is True


def test_however_many_stores_hang_retrieval_returns_inside_the_breaker(monkeypatch):
    """The relationship that replaces a magic number: the stage timeouts add up
    to more than the breaker, so the stages must share a budget that ends
    before it. Every store stage hangs here; retrieval still returns on its
    own, inside the breaker window, rather than being discarded by it."""
    breaker = 2.0
    monkeypatch.setattr(mp, "_CIRCUIT_BREAKER_TIMEOUT", breaker)
    monkeypatch.setattr(mp, "_STAGE_TIMEOUT", 0.8)
    release = threading.Event()
    orch = _orchestrator()

    def hang(*args, **kwargs):
        release.wait(20)

    orch.postgres.check_artifact_exists.side_effect = hang
    orch.redis.search_schema_semantically.side_effect = hang
    orch.chroma.get_user_signature.side_effect = hang
    orch.milvus.search_similar_insights.side_effect = hang
    hung_stages = 4
    assert hung_stages * mp._STAGE_TIMEOUT > breaker, "precondition: the stages alone outlast the breaker"

    try:
        start = time.monotonic()
        out = orch._retrieve_memory_internal(QUERY, session_id="s-budget")
        elapsed = time.monotonic() - start
    finally:
        release.set()

    assert out is not None
    assert elapsed < breaker, f"retrieval took {elapsed:.2f}s; the {breaker}s breaker would have discarded it"
    assert elapsed >= 2 * mp._STAGE_TIMEOUT, "the budget must not cut short stages it has room for"


def test_the_shipped_budget_survives_a_cold_connect_and_a_slow_store():
    """On the SHIPPED numbers: a first request that waits out the whole connect
    timeout and then meets the slowest single stage must still have budget
    left, or one bad store costs the stores behind it. Fails for a breaker of
    8.0 (7.2s of budget against 5.5 + 4.0)."""
    budget = mp._CIRCUIT_BREAKER_TIMEOUT * (1.0 - mp._BREAKER_HEADROOM)
    cold_connect = mp._CONNECT_TIMEOUT + 0.5
    slowest_stage = max(mp._STAGE_TIMEOUT, mp._EMBED_TIMEOUT, mp._MCP_SCHEMA_TIMEOUT)

    assert budget > cold_connect + slowest_stage, (
        f"stage budget {budget:.1f}s (breaker {mp._CIRCUIT_BREAKER_TIMEOUT}s) does not cover a cold "
        f"connect ({cold_connect}s) plus the slowest stage ({slowest_stage}s)"
    )
