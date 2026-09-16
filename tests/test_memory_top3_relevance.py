"""
Regression tests: the top-3 memory-hint contract must select by RELEVANCE,
not by append order.

Bug (Avaloka issue tracker): `_compose_top_hints` filled the non-preference
slots with the most recently appended hints (`[-3:]`). Because LLM-extracted
hints are appended LAST in `_retrieve_memory_internal`, generic extractions
evicted the query-relevant hints — the [L3] artifact-reuse hint, the domain
schema insight, and the Milvus similarity results — from the top-3 the
planner receives.

Fix: hints carry a provenance tier; slots fill in relevance order
artifact > domain > similar > llm, leftover slots taking the most recent
session-history hints. Explicit user preferences keep their reserved slots,
and a flat-list input keeps the legacy tail behavior (see
test_store_user_preference.py::test_compose_top_hints_without_preferences_matches_legacy).
"""

import json
import sys
import types

import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Venv-isolation: stub packages ONLY when missing so this suite runs in
# minimal dev venvs while using the real packages in CI. (Same precedent as
# tests/test_store_user_preference.py.)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    import cachetools  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _ct = types.ModuleType("cachetools")

    class _TTLCache(dict):
        def __init__(self, maxsize=None, ttl=None):
            super().__init__()

    _ct.TTLCache = _TTLCache
    sys.modules["cachetools"] = _ct

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

from app.services.memory_plane import MemoryOrchestrator

QUERY = "plot monthly revenue by region"
SESSION = "sess-top3-relevance"

L3_MARKER = "[L3] Prior artifact"
DOMAIN_HINT = "Domain insight: orders(order_id, user_id, region, amount, created_at)"
MILVUS_HINT = "User previously aggregated monthly revenue by region"
NOISE_HINTS = [
    "noise: user typed a query",
    "noise: conversation is ongoing",
    "noise: data may be tabular",
]
PREFERENCE = "The user's favorite column is 'reordered'"


def _stubbed_orchestrator(with_artifact=True, with_schema=True,
                          milvus_hits=(MILVUS_HINT,), llm_hints=NOISE_HINTS):
    """MemoryOrchestrator with every DB layer mocked; __init__ skipped so no
    real clients are constructed."""
    mo = MemoryOrchestrator.__new__(MemoryOrchestrator)
    mo._explicit_llm = object() if llm_hints is not None else None

    mo.postgres = MagicMock()
    mo.postgres.check_artifact_exists.return_value = with_artifact
    mo.postgres.get_artifact.return_value = {
        "status": "success",
        "chart_url": "https://storage.googleapis.com/charts/monthly_revenue.png",
    } if with_artifact else None

    mo.redis = MagicMock()
    mo.redis.search_schema_semantically.return_value = (
        "orders(order_id, user_id, region, amount, created_at)" if with_schema else None
    )
    mo.redis.get_json.return_value = None  # nothing durable to hydrate

    mo.chroma = MagicMock()
    mo.chroma.get_user_signature.return_value = None

    mo.milvus = MagicMock()
    mo.milvus.search_similar_insights.return_value = [
        {"content": h} for h in milvus_hits
    ]

    if llm_hints is not None:
        mo._extract_context_with_llm = lambda query, past_queries: {
            "new_hints": list(llm_hints),
            "logic_signature": None,
        }
    return mo


@pytest.fixture(autouse=True)
def _memory_isolation():
    MemoryOrchestrator.clear_all_sessions()
    yield
    MemoryOrchestrator.clear_all_sessions()


# ── The reported bug: LLM noise must not evict relevant hints ────────────────


def test_llm_noise_does_not_evict_relevant_hints():
    mo = _stubbed_orchestrator()
    result = mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="unknown_dataset")

    top = result["memory_hints"]
    assert len(top) == 3
    assert any(L3_MARKER in h for h in top), "artifact-reuse hint must survive"
    assert DOMAIN_HINT in top, "domain schema hint must survive"
    assert MILVUS_HINT in top, "similarity-ranked insight must survive"
    assert not any(h in NOISE_HINTS for h in top), "unranked LLM hints must not fill relevant slots"


def test_relevance_order_is_artifact_then_domain_then_similar():
    mo = _stubbed_orchestrator()
    result = mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="unknown_dataset")

    top = result["memory_hints"]
    assert L3_MARKER in top[0]
    assert top[1] == DOMAIN_HINT
    assert top[2] == MILVUS_HINT


def test_accumulated_hints_still_carry_everything():
    """The full accumulated list is a separate contract — noise stays visible there."""
    mo = _stubbed_orchestrator()
    result = mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="unknown_dataset")

    for h in NOISE_HINTS:
        assert h in result["accumulated_memory_hints"]


def test_stale_session_noise_does_not_outrank_fresh_relevant_hints():
    """Second turn: turn-1 noise sits in the session store; fresh query-matched
    hints must still win the top-3."""
    mo = _stubbed_orchestrator()
    mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="unknown_dataset")
    result = mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="unknown_dataset")

    top = result["memory_hints"]
    assert any(L3_MARKER in h for h in top)
    assert DOMAIN_HINT in top
    assert not any(h in NOISE_HINTS for h in top)


# ── Preference reserved slots must keep working alongside tiering ────────────


def test_preference_keeps_reserved_slot_over_relevant_hints():
    mo = _stubbed_orchestrator()
    assert mo.store_user_preference(PREFERENCE, SESSION) is True
    result = mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="unknown_dataset")

    top = result["memory_hints"]
    assert len(top) == 3
    assert PREFERENCE in top, "explicit preference must hold its reserved slot"
    # The two fill slots go to the highest tiers.
    assert any(L3_MARKER in h for h in top)
    assert DOMAIN_HINT in top


def test_three_preferences_leave_no_fill_slots():
    mo = _stubbed_orchestrator()
    for i in range(3):
        mo.store_user_preference(f"pref-{i}", SESSION)
    result = mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="unknown_dataset")

    assert result["memory_hints"] == ["pref-0", "pref-1", "pref-2"]


# ── Degradation: without relevant sources, keep the legacy recency contract ──


def test_degrades_to_recent_session_hints_without_relevant_sources():
    mo = _stubbed_orchestrator(with_artifact=False, with_schema=False,
                               milvus_hits=(), llm_hints=None)
    MemoryOrchestrator._session_hints_store[SESSION] = [f"old-{i}" for i in range(5)]
    result = mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="unknown_dataset")

    assert result["memory_hints"] == ["old-2", "old-3", "old-4"]


def test_compose_top_hints_flat_list_keeps_legacy_tail():
    mo = _stubbed_orchestrator()
    assert mo._compose_top_hints("sess-flat", ["a", "b"]) == ["a", "b"]
    assert mo._compose_top_hints("sess-flat", ["a", "b", "c", "d"]) == ["b", "c", "d"]


# ── Direct unit coverage of the tier composition ─────────────────────────────


def test_compose_top_hints_tiered_order_and_truncation():
    mo = _stubbed_orchestrator()
    tiered = {
        "artifact": ["art-1"],
        "domain": ["dom-1", "dom-2"],
        "similar": ["sim-1"],
        "llm": ["llm-1"],
        "session": ["old-1", "old-2"],
    }
    assert mo._compose_top_hints("sess-tiered", tiered) == ["art-1", "dom-1", "dom-2"]


def test_compose_top_hints_session_fills_leftover_slots_newest_first():
    mo = _stubbed_orchestrator()
    tiered = {"domain": ["dom-1"], "session": ["old-1", "old-2", "old-3"]}
    assert mo._compose_top_hints("sess-leftover", tiered) == ["dom-1", "old-2", "old-3"]


def test_compose_top_hints_dedups_across_tiers():
    mo = _stubbed_orchestrator()
    tiered = {"similar": ["dup", "sim-2"], "llm": ["dup", "llm-2"], "session": []}
    assert mo._compose_top_hints("sess-dedup", tiered) == ["dup", "sim-2", "llm-2"]


# ── Edge cases: alternate hint sources, dedup semantics, failure paths ───────


def test_mcp_hot_loaded_schema_ranks_as_domain_tier():
    """Schema cache miss + real dataset_id → MCP hot-load; the result must
    rank as a domain hint, not fall to the bottom of the list."""
    payload = {"tables": {"orders": ["order_id", "region", "amount"]}}
    fake_loader = types.ModuleType("app.services.mcp_cache_loader")
    fake_loader.fetch_schema_sync = lambda url, key, ds: dict(payload)

    mo = _stubbed_orchestrator(with_schema=False)
    with patch.dict(sys.modules, {"app.services.mcp_cache_loader": fake_loader}):
        result = mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="nyc_taxi")

    assert f"Domain insight (hot-loaded via MCP): {json.dumps(payload)}" in result["memory_hints"]
    mo.redis.set_schema.assert_called_once()  # persisted to L1 for next request


def test_l3_metadata_without_known_fields_still_ranks_as_artifact():
    """The raw-summary fallback branch must carry the artifact tier too."""
    mo = _stubbed_orchestrator()
    mo.postgres.get_artifact.return_value = {"unrecognised_field": "x"}
    result = mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="unknown_dataset")

    assert any("[L3] Prior artifact detected" in h for h in result["memory_hints"])


def test_duplicate_llm_hint_not_double_stored():
    """LLM re-emitting a hint Milvus already surfaced: one copy in the
    accumulated list, no duplicate persisted to the session store, and the
    raw extraction report stays unfiltered."""
    mo = _stubbed_orchestrator(llm_hints=[MILVUS_HINT, "genuinely new fact"])
    result = mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="unknown_dataset")

    assert result["accumulated_memory_hints"].count(MILVUS_HINT) == 1
    stored = MemoryOrchestrator._session_hints_store.get(SESSION, [])
    assert MILVUS_HINT not in stored, "only genuinely new hints persist to the session store"
    assert "genuinely new fact" in stored
    assert result["new_hints_this_context"] == [MILVUS_HINT, "genuinely new fact"]


def test_internal_failure_returns_safe_empty_payload():
    """Any layer blowing up trips the breaker: empty hints, flagged unavailable."""
    mo = _stubbed_orchestrator()
    mo.postgres.connect.side_effect = RuntimeError("postgres down")
    result = mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="unknown_dataset")

    assert result["memory_hints"] == []
    assert result["accumulated_memory_hints"] == []
    assert result["memory_context_unavailable"] is True


def test_retrieve_memory_wrapper_preserves_relevance_contract():
    """The public timeout wrapper must return the same tiered top-3."""
    mo = _stubbed_orchestrator()
    result = mo.retrieve_memory(QUERY, session_id=SESSION, dataset_id="unknown_dataset")

    top = result["memory_hints"]
    assert any(L3_MARKER in h for h in top)
    assert DOMAIN_HINT in top
    assert MILVUS_HINT in top


def test_four_preferences_newest_three_win_with_tiered_sources():
    """Preference overflow keeps the legacy newest-3 contract under tiering."""
    mo = _stubbed_orchestrator()
    for i in range(4):
        mo.store_user_preference(f"pref-{i}", SESSION)
    result = mo._retrieve_memory_internal(QUERY, session_id=SESSION, dataset_id="unknown_dataset")

    assert result["memory_hints"] == ["pref-1", "pref-2", "pref-3"]
