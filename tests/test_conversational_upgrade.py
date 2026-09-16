"""
Tests for the conversational & reasoning upgrade
(feature/conversational-reasoning-upgrade).

Covers, phase by phase:
  1a. Durable thread history — THREAD_META chat history hydrates from /
      persists to Redis (via an async fake cache), surviving a simulated
      process restart.
  1b. Durable preferences — store_user_preference writes through to Redis
      (via the sync redis_client fallback) and hydrates after a simulated
      restart, so "remember ..." survives a redeploy.
  2a. Pin-aware truncation — _limit_messages keeps SystemMessage /
      avaloka_pinned messages outside the sliding MAX_CONTEXT_TURNS window.
  2b. Compound-request loop-back — the planner's infra keyword fast path no
      longer re-fires after provisioning, so "deploy X then analyze Y"
      reaches the LLM planner for the analysis half.
  3.  Adaptive reasoning — select_reasoning_effort scales with query
      complexity, and plan_etl_job forwards reasoning_effort to the LLM
      only when the configured model supports it.
"""

import asyncio
import json
import sys
import types

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ---------------------------------------------------------------------------
# Venv-isolation: app.services.memory_plane transitively imports sqlalchemy
# (via the Layer 3 postgres client). Stub it ONLY when it is missing so this
# suite runs in minimal dev venvs while using the real package in CI. (Same
# precedent as tests/test_store_user_preference.py.)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    import sqlalchemy  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _sa = types.ModuleType("sqlalchemy")
    _sa_orm = types.ModuleType("sqlalchemy.orm")
    _sa_sql = types.ModuleType("sqlalchemy.sql")
    for _name in ("create_engine", "Column", "String", "Text", "DateTime"):
        setattr(_sa, _name, MagicMock())
    _sa_orm.declarative_base = lambda: type("Base", (), {})
    _sa_orm.sessionmaker = MagicMock()
    _sa_sql.func = MagicMock()
    _sa.orm = _sa_orm
    _sa.sql = _sa_sql
    sys.modules["sqlalchemy"] = _sa
    sys.modules["sqlalchemy.orm"] = _sa_orm
    sys.modules["sqlalchemy.sql"] = _sa_sql

import app.api.helpers as helpers
from app.api.helpers import (
    THREAD_META,
    _compact_messages,
    _is_pinned_message,
    _jsonable_to_lc_msgs,
    _k_thread_history,
    _lc_msgs_to_jsonable,
    _limit_messages,
    delete_thread_history,
    hydrate_thread_history,
    persist_thread_history,
)
from app.services import session_service
import app.agents.planner as planner
from app.agents.planner import plan_etl_job, select_reasoning_effort
from app.graph.etl_state import ETLState
from app.services.memory_plane import MemoryOrchestrator, _orchestrator
from app.services.db.redis_client import redis_client


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------

class FakeAsyncCache:
    """Minimal async ICache double backed by a dict."""

    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def delete(self, key):
        self.store.pop(key, None)


@pytest.fixture(autouse=True)
def _clean_state():
    """Isolate THREAD_META, memory stores, and the redis fallback per test."""
    THREAD_META.clear()
    MemoryOrchestrator.clear_all_sessions()
    redis_client._fallback_kv.clear()
    saved_cache = session_service.cache
    yield
    session_service.cache = saved_cache
    THREAD_META.clear()
    MemoryOrchestrator.clear_all_sessions()
    redis_client._fallback_kv.clear()


@pytest.fixture
def fake_cache():
    cache = FakeAsyncCache()
    session_service.cache = cache
    return cache


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Phase 1a — durable thread history
# ---------------------------------------------------------------------------

def test_history_serialization_round_trip():
    msgs = [
        SystemMessage(content="pinned rules", additional_kwargs={"avaloka_pinned": True}),
        HumanMessage(content="show revenue by month"),
        AIMessage(content="here is the table"),
    ]
    restored = _jsonable_to_lc_msgs(json.loads(json.dumps(_lc_msgs_to_jsonable(msgs))))
    assert [type(m) for m in restored] == [SystemMessage, HumanMessage, AIMessage]
    assert [m.content for m in restored] == [m.content for m in msgs]
    assert restored[0].additional_kwargs.get("avaloka_pinned") is True
    assert not restored[1].additional_kwargs.get("avaloka_pinned")


def test_persist_then_restart_then_hydrate(fake_cache):
    tid = "t-restart"
    THREAD_META[tid] = {
        "lc_msgs": [HumanMessage(content="hello"), AIMessage(content="hi there")],
        "metadata": {},
    }
    _run(persist_thread_history(tid))
    assert _k_thread_history(tid) in fake_cache.store

    # Simulate a process restart: in-process cache wiped, Redis survives.
    THREAD_META.clear()
    _run(hydrate_thread_history(tid))

    restored = THREAD_META[tid]["lc_msgs"]
    assert [m.content for m in restored] == ["hello", "hi there"]
    assert isinstance(restored[0], HumanMessage)
    assert isinstance(restored[1], AIMessage)


def test_hydrate_does_not_clobber_live_history(fake_cache):
    tid = "t-live"
    fake_cache.store[_k_thread_history(tid)] = json.dumps(
        [{"role": "human", "content": "stale from redis"}]
    )
    THREAD_META[tid] = {"lc_msgs": [HumanMessage(content="fresh in-process")], "metadata": {}}
    _run(hydrate_thread_history(tid))
    assert THREAD_META[tid]["lc_msgs"][0].content == "fresh in-process"


def test_hydrate_without_cache_is_noop():
    session_service.cache = None
    tid = "t-nocache"
    _run(hydrate_thread_history(tid))
    assert tid not in THREAD_META
    # persist/delete must not raise either
    _run(persist_thread_history(tid))
    _run(delete_thread_history(tid))


def test_delete_thread_history_removes_key(fake_cache):
    tid = "t-del"
    THREAD_META[tid] = {"lc_msgs": [HumanMessage(content="x")], "metadata": {}}
    _run(persist_thread_history(tid))
    assert _k_thread_history(tid) in fake_cache.store
    _run(delete_thread_history(tid))
    assert _k_thread_history(tid) not in fake_cache.store


def test_hydrate_survives_corrupt_payload(fake_cache):
    tid = "t-corrupt"
    fake_cache.store[_k_thread_history(tid)] = "{not json"
    _run(hydrate_thread_history(tid))  # must not raise
    assert tid not in THREAD_META


# ---------------------------------------------------------------------------
# Phase 1b — durable preferences
# ---------------------------------------------------------------------------

PREF = "The user's favorite column is 'reordered'"


def test_preference_write_through_and_restart_hydration():
    sid = "sess-durable"
    assert _orchestrator.store_user_preference(PREF, sid) is True

    # Write-through landed in the (fallback) redis store.
    raw = redis_client.get_json(f"ctxmem:{sid}:prefs")
    assert raw == [PREF]

    # Simulate a restart: in-process TTLCaches wiped, redis keys survive.
    kv_backup = dict(redis_client._fallback_kv)
    MemoryOrchestrator._past_queries_store.clear()
    MemoryOrchestrator._session_hints_store.clear()
    MemoryOrchestrator._session_preferences_store.clear()
    redis_client._fallback_kv.update(kv_backup)

    _orchestrator._hydrate_session_memory(sid)
    assert PREF in MemoryOrchestrator._session_hints_store.get(sid, [])
    assert PREF in MemoryOrchestrator._session_preferences_store.get(sid, [])

    # And the top-hints contract still reserves a slot for it post-restart.
    top = _orchestrator._compose_top_hints(sid, ["a", "b", "c", "d"])
    assert PREF in top


def test_store_after_restart_does_not_clobber_prior_preferences():
    sid = "sess-append"
    _orchestrator.store_user_preference("pref one", sid)
    kv_backup = dict(redis_client._fallback_kv)
    MemoryOrchestrator._session_hints_store.clear()
    MemoryOrchestrator._session_preferences_store.clear()
    redis_client._fallback_kv.update(kv_backup)

    # New process stores a second preference; the first must be hydrated
    # back before the write-through rewrites the redis list.
    _orchestrator.store_user_preference("pref two", sid)
    prefs = redis_client.get_json(f"ctxmem:{sid}:prefs")
    assert prefs == ["pref one", "pref two"]


def test_clear_all_sessions_also_clears_durable_copies():
    sid = "sess-clear"
    _orchestrator.store_user_preference(PREF, sid)
    assert redis_client.get_json(f"ctxmem:{sid}:prefs")
    MemoryOrchestrator.clear_all_sessions()
    assert redis_client.get_json(f"ctxmem:{sid}:prefs") is None


# ---------------------------------------------------------------------------
# Phase 2a — pin-aware truncation
# ---------------------------------------------------------------------------

def test_limit_messages_unchanged_without_pins():
    msgs = [HumanMessage(content=str(i)) for i in range(20)]
    out = _limit_messages(msgs, 12)
    assert len(out) == 12
    assert out[0].content == "8"
    assert out[-1].content == "19"


def test_limit_messages_keeps_pinned_outside_window():
    pinned = SystemMessage(content="user prefers weekly aggregates")
    tagged = AIMessage(content="pinned ai", additional_kwargs={"avaloka_pinned": True})
    msgs = [pinned, tagged] + [HumanMessage(content=str(i)) for i in range(20)]
    out = _limit_messages(msgs, 12)
    assert pinned in out and tagged in out
    assert len(out) == 14  # 2 pinned + 12 window
    assert out[0] is pinned and out[1] is tagged
    assert out[2].content == "8"


def test_is_pinned_message():
    assert _is_pinned_message(SystemMessage(content="x"))
    assert _is_pinned_message(HumanMessage(content="x", additional_kwargs={"avaloka_pinned": True}))
    assert not _is_pinned_message(HumanMessage(content="x"))


def test_compact_then_limit_pipeline_preserves_pins():
    pinned = SystemMessage(content="pin")
    msgs = [pinned] + [HumanMessage(content=str(i)) for i in range(30)]
    out = _limit_messages(_compact_messages(msgs), 12)
    assert pinned in out


# ---------------------------------------------------------------------------
# Phase 2b — compound-request loop-back
# ---------------------------------------------------------------------------

COMPOUND = "deploy this on gcp and then analyze average sales by region"


def _tool_call_message(name, arguments):
    return AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {"id": "t0", "type": "function", "function": {"name": name, "arguments": arguments}}
            ]
        },
    )


def _make_state(content, **kw):
    base = dict(
        messages=[HumanMessage(content=content)],
        planner_definition={},
        memory_hints=[],
        session_id="test-session",
    )
    base.update(kw)
    return ETLState(**base)


@patch("app.agents.planner.llm")
def test_first_pass_still_routes_to_infra(mock_llm):
    # llm patched so the test is independent of GROQ_API_KEY_PLANNING_AGENT
    # (with llm=None plan_etl_job returns a stub before the fast path).
    # The fast path itself returns before any LLM call.
    mock_llm.invoke.return_value = _tool_call_message(
        "respond_to_user", '{"response_text": "ok"}'
    )
    state = _make_state(COMPOUND)
    result = plan_etl_job(state)
    assert result.get("infrastructure_request") == {"type": "gcp", "app_type": "python-docker"}


@patch("app.agents.planner.llm")
def test_loop_back_reaches_llm_for_analysis_half(mock_llm):
    """After provisioning, the fast path must NOT re-fire; the LLM plans the rest."""
    mock_llm.invoke.return_value = _tool_call_message(
        "respond_to_user",
        '{"response_text": "Infrastructure is ready - now analyzing average sales by region."}',
    )
    state = _make_state(
        COMPOUND,
        infrastructure_provisioned={"status": "provisioned", "type": "gcp"},
        infrastructure_request={"type": "gcp", "app_type": "python-docker"},
    )
    result = plan_etl_job(state)

    assert mock_llm.invoke.called, "loop-back turn must reach the LLM planner"
    ai = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
    assert ai, "the analysis half must produce a response for the user"


@patch("app.agents.planner.llm")
def test_provisioned_state_skips_keyword_hijack_even_without_request(mock_llm):
    mock_llm.invoke.return_value = _tool_call_message(
        "respond_to_user", '{"response_text": "ok"}'
    )
    state = _make_state(COMPOUND, infrastructure_provisioned={"status": "provisioned"})
    plan_etl_job(state)
    assert mock_llm.invoke.called


# ---------------------------------------------------------------------------
# Phase 3 — adaptive reasoning
# ---------------------------------------------------------------------------

def test_effort_low_for_smalltalk_and_short_lookups():
    assert select_reasoning_effort("hi") == "low"
    assert select_reasoning_effort("hi there") == "low"
    assert select_reasoning_effort("thanks") == "low"
    assert select_reasoning_effort("") == "low"
    assert select_reasoning_effort("row count?") == "low"


def test_effort_low_markers_match_whole_words_not_prefixes():
    # 'highlight'/'normalize'/'note' must NOT hit the hi/no low-effort
    # markers via prefix matching (review finding): these analytical asks
    # must never be classified 'low'.
    assert select_reasoning_effort("highlight the top anomalies and explain why") == "high"
    assert select_reasoning_effort("normalize revenue and compare it to costs") != "low"
    assert select_reasoning_effort("note the outliers and explain the trend") != "low"


def test_effort_high_for_multistep_analysis():
    # Compound infra+analysis ask: at least medium, never low.
    assert select_reasoning_effort(COMPOUND) in ("medium", "high")
    assert (
        select_reasoning_effort(
            "compare Q3 vs Q2 revenue by region and explain why the anomaly happened"
        )
        == "high"
    )


def test_effort_ignores_analysis_context_augmentation():
    # The server appends "[Analysis context] ..." metadata to the stored
    # human message; the gate must judge only the user's actual ask.
    augmented = (
        "how many rows are there?\n[Analysis context] dataset=orders "
        "fidelity=portfolio_samples selected_sample=random_baseline "
        "columns=order_id,product_id,add_to_cart_order,reordered and then "
        "compare and explain everything about it in extreme detail" * 3
    )
    assert select_reasoning_effort(augmented) == "low"


def test_effort_medium_for_single_analytical_marker():
    assert select_reasoning_effort("explain the schema of this dataset please") in (
        "medium",
        "high",
    )


@patch("app.agents.planner.PLANNER_SUPPORTS_REASONING", True)
@patch("app.agents.planner.llm")
def test_reasoning_effort_forwarded_when_supported(mock_llm):
    mock_llm.invoke.return_value = _tool_call_message(
        "respond_to_user", '{"response_text": "ok"}'
    )
    plan_etl_job(_make_state(COMPOUND, infrastructure_provisioned={"status": "provisioned"}))
    assert mock_llm.invoke.called
    _, kwargs = mock_llm.invoke.call_args
    assert kwargs.get("reasoning_effort") == select_reasoning_effort(COMPOUND)
    assert kwargs.get("reasoning_effort") in ("low", "medium", "high")


@patch("app.agents.planner.PLANNER_SUPPORTS_REASONING", False)
@patch("app.agents.planner.llm")
def test_reasoning_effort_omitted_when_unsupported(mock_llm):
    mock_llm.invoke.return_value = _tool_call_message(
        "respond_to_user", '{"response_text": "ok"}'
    )
    plan_etl_job(_make_state("show me the first rows"))
    assert mock_llm.invoke.called
    _, kwargs = mock_llm.invoke.call_args
    assert "reasoning_effort" not in kwargs


def test_default_planner_model_is_reasoning_capable():
    # Guard against silent regressions of the model default.
    assert planner.PLANNER_MODEL.lower().startswith(planner._REASONING_MODEL_PREFIXES)


@patch("app.agents.planner.llm")
def test_reasoning_trace_surfaced_to_state(mock_llm):
    # reasoning_format="parsed" puts the deliberation in
    # additional_kwargs.reasoning_content; the planner must copy it into
    # state["reasoning_trace"] so the API can show a thinking panel.
    msg = _tool_call_message("respond_to_user", '{"response_text": "ok"}')
    msg.additional_kwargs["reasoning_content"] = "The user wants a comparison, so I will..."
    mock_llm.invoke.return_value = msg
    result = plan_etl_job(_make_state("compare reorder rates and explain the difference"))
    assert result.get("reasoning_trace") == "The user wants a comparison, so I will..."


# ---------------------------------------------------------------------------
# Review-fix regressions: failed hydration must never destroy durable copies
# ---------------------------------------------------------------------------

class FlakyCache(FakeAsyncCache):
    """FakeAsyncCache whose GETs can be made to fail on demand."""

    def __init__(self):
        super().__init__()
        self.fail_gets = 0

    async def get(self, key):
        if self.fail_gets > 0:
            self.fail_gets -= 1
            raise ConnectionError("simulated redis blip")
        return await super().get(key)


def test_failed_hydrate_does_not_erase_durable_history():
    cache = FlakyCache()
    session_service.cache = cache
    tid = "t-blip"
    # Durable history exists from before the "restart".
    cache.store[_k_thread_history(tid)] = json.dumps(
        [{"role": "human", "content": "old turn"}, {"role": "ai", "content": "old reply"}]
    )
    # Restarted process: empty entry, first hydrate hits a Redis blip.
    THREAD_META[tid] = {"lc_msgs": [], "metadata": {}}
    cache.fail_gets = 1
    _run(hydrate_thread_history(tid))
    assert THREAD_META[tid].get("_history_unsynced") is True

    # The request proceeds and appends a new turn, then persists.
    THREAD_META[tid]["lc_msgs"].append(HumanMessage(content="new turn"))
    _run(persist_thread_history(tid))

    # Redis recovered by persist time -> durable copy must be MERGED, not
    # replaced by the partial post-restart view.
    saved = json.loads(cache.store[_k_thread_history(tid)])
    contents = [m["content"] for m in saved]
    assert "old turn" in contents and "old reply" in contents and "new turn" in contents
    assert not THREAD_META[tid].get("_history_unsynced")


def test_persist_skips_write_while_redis_unreadable():
    cache = FlakyCache()
    session_service.cache = cache
    tid = "t-still-down"
    cache.store[_k_thread_history(tid)] = json.dumps([{"role": "human", "content": "precious"}])
    THREAD_META[tid] = {"lc_msgs": [], "metadata": {}}
    cache.fail_gets = 2  # hydrate fails AND persist's re-read fails
    _run(hydrate_thread_history(tid))
    THREAD_META[tid]["lc_msgs"].append(HumanMessage(content="new"))
    _run(persist_thread_history(tid))
    # Durable copy untouched.
    assert json.loads(cache.store[_k_thread_history(tid)]) == [
        {"role": "human", "content": "precious"}
    ]


def test_concurrent_append_wins_over_stale_hydrate_snapshot():
    tid = "t-race"

    class InterleavingCache(FakeAsyncCache):
        async def get(self, key):
            # Simulate a concurrent request appending a turn while this
            # hydrate coroutine is suspended on the Redis read.
            THREAD_META.setdefault(tid, {"lc_msgs": [], "metadata": {}})
            THREAD_META[tid]["lc_msgs"].append(HumanMessage(content="fresh concurrent turn"))
            return json.dumps([{"role": "human", "content": "stale snapshot"}])

    session_service.cache = InterleavingCache()
    THREAD_META[tid] = {"lc_msgs": [], "metadata": {}}
    _run(hydrate_thread_history(tid))
    contents = [m.content for m in THREAD_META[tid]["lc_msgs"]]
    assert contents == ["fresh concurrent turn"]  # stale snapshot must not clobber


class StubMemRedis:
    """Sync stand-in for redis_client with controllable failure."""

    def __init__(self):
        self.kv = {}
        self.fail_reads = 0

    def get_json(self, key, strict=False):
        if self.fail_reads > 0:
            self.fail_reads -= 1
            if strict:
                raise RuntimeError("simulated redis outage")
            return None
        return self.kv.get(key)

    def set_json(self, key, value, ttl_seconds=None):
        self.kv[key] = value
        return True


def test_memory_persist_refuses_overwrite_after_failed_hydration(monkeypatch):
    sid = "sess-blip"
    stub = StubMemRedis()
    stub.kv[f"ctxmem:{sid}:prefs"] = ["Use INR"]
    stub.kv[f"ctxmem:{sid}:hints"] = ["Use INR"]
    monkeypatch.setattr(_orchestrator, "redis", stub)

    # Fresh process, Redis down during hydration.
    stub.fail_reads = 2
    _orchestrator._hydrate_session_memory(sid)
    assert sid in MemoryOrchestrator._hydration_failed

    # A hint gets generated in-process and the code tries to write through
    # while Redis is STILL unreadable: the durable copy must survive.
    MemoryOrchestrator._session_hints_store.setdefault(sid, []).append("new hint")
    stub.fail_reads = 2
    _orchestrator._persist_session_memory(sid)
    assert stub.kv[f"ctxmem:{sid}:prefs"] == ["Use INR"]

    # Once Redis recovers, persist first re-hydrates (merging the durable
    # preference back) and only then writes.
    stub.fail_reads = 0
    _orchestrator._persist_session_memory(sid)
    assert "Use INR" in stub.kv[f"ctxmem:{sid}:prefs"]
    assert sid not in MemoryOrchestrator._hydration_failed


def test_memory_hydrates_each_store_independently(monkeypatch):
    sid = "sess-partial"
    stub = StubMemRedis()
    stub.kv[f"ctxmem:{sid}:prefs"] = ["Prefer medians"]
    monkeypatch.setattr(_orchestrator, "redis", stub)

    # Hints store already has this session in-process (e.g. TTL skew),
    # prefs store does not — prefs must still hydrate from Redis.
    MemoryOrchestrator._session_hints_store[sid] = ["some hint"]
    _orchestrator._hydrate_session_memory(sid)
    assert "Prefer medians" in MemoryOrchestrator._session_preferences_store.get(sid, [])
    assert MemoryOrchestrator._session_hints_store[sid] == ["some hint"]  # untouched
