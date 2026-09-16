"""Guards for the memory plane — the learning loop.

Every one of these fails on the pre-fix tree. They exist because the loop was
written, deployed nowhere, and then silently aborted by its own timeout: the
only symptom was memory_hints=None, which is indistinguishable from "nothing
has been learned yet".
"""
from __future__ import annotations

import os

import pytest


# ── the circuit breaker must be longer than a retrieval takes ──────────────

def test_circuit_breaker_allows_a_real_retrieval() -> None:
    """3.0s aborted EVERY call; a healthy retrieval measures 4.4-5.8s.

    The breaker exists to stop a hung backend stalling a turn. Set below the
    normal completion time it stops everything, forever, and reports the same
    empty payload it would report if the system had simply learned nothing.
    """
    from app.services import memory_plane

    assert memory_plane._CIRCUIT_BREAKER_TIMEOUT >= 10.0, (
        f"breaker is {memory_plane._CIRCUIT_BREAKER_TIMEOUT}s; a successful "
        f"retrieval measured 4.4-5.8s on a healthy deployment, so anything "
        f"near that aborts every call")


# ── the embedding model must not need the network ─────────────────────────

def test_embedding_model_is_local() -> None:
    """Baked into the image at build; a query-time fetch is a defect.

    Reaching huggingface.co on first query is slow enough to trip the breaker
    and impossible in an air-gapped install.
    """
    from app.services.embedding_utils import embed_text

    vector = embed_text("total revenue by region", target_dim=384)
    assert vector is not None, "embedding returned None"
    assert len(vector) == 384


def test_hf_home_points_at_the_baked_cache() -> None:
    home = os.environ.get("HF_HOME", "")
    if not home:
        pytest.skip("HF_HOME unset — running outside the shipped image")
    assert os.path.isdir(home), f"HF_HOME={home} does not exist in this image"


# ── retrieval reports honestly ────────────────────────────────────────────

def test_retrieve_memory_returns_the_documented_shape() -> None:
    from app.services.memory_plane import retrieve_memory

    out = retrieve_memory("total revenue by region",
                          {"session_id": "pytest-shape", "user_prompt": "total revenue by region"})
    assert out is not None
    for key in ("memory_hints", "memory_context_unavailable"):
        assert key in out, f"{key} missing from the retrieval payload"


def test_unavailable_is_a_bool_not_absent() -> None:
    """LL-06: a memory tier that is down must SAY so.

    An absent field reads as "no memory needed"; False reads as "memory ran
    and found nothing"; True reads as "memory could not answer". Only the last
    two are honest states, and the caller cannot tell them apart if the field
    is simply missing.
    """
    from app.services.memory_plane import retrieve_memory

    out = retrieve_memory("anything", {"session_id": "pytest-honest", "user_prompt": "anything"})
    assert isinstance(out.get("memory_context_unavailable"), bool), (
        "memory_context_unavailable must be True or False, never absent — a "
        "silent degrade is what made this subsystem invisible for so long")
