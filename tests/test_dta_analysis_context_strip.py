"""The chat endpoint appends a '[Analysis context] …' metadata suffix to every user
message (server.py). It describes how the ANALYSIS was sampled — not a transform. If
it reaches the DTA code-gen, the LLM reads its "fidelity"/"sample" words as an
instruction and emits df.sample(fraction=…), silently dropping ~half the rows.

These tests lock in that the DTA strips the marker before any consumer sees it.
"""
from app.agents.data_transfer_agent.daft_coder import (
    strip_runtime_analysis_context,
    sanitize_user_prompt_for_transformations,
)

# The real suffix shape emitted by app/api/server.py (double newline after join).
_CONTEXT = (
    "\n\n[Analysis context] Current fidelity: portfolio_samples; "
    "Effective fidelity for this request: quick_sample; "
    "Selected sample: random_baseline; "
    "Resolved analysis source: quick_sample; Dataset size bytes: 12345"
)

# The reported demo prompt — a transfer WITHOUT a filter, so it passes guard #1.
_USER = (
    "Transfer from sunny_test2 to sunny_test7. "
    "Source table adult_income, destination table adult_income_dest."
)


def test_strip_removes_the_analysis_context_suffix():
    assert strip_runtime_analysis_context(_USER + _CONTEXT) == _USER
    # Idempotent + no-op when the marker is absent.
    assert strip_runtime_analysis_context(_USER) == _USER
    # Single-newline / lowercase variants are also stripped.
    assert strip_runtime_analysis_context("do X\n[analysis context] fidelity: x") == "do X"


def test_strip_none_and_empty():
    assert strip_runtime_analysis_context(None) == ""
    assert strip_runtime_analysis_context("") == ""


class _CapturingLLM:
    """Records what the sanitiser actually hands the coder LLM."""
    def __init__(self):
        self.seen = None

    def invoke(self, messages):
        self.seen = messages[-1].content

        class _R:
            content = "keep all rows and columns"
        return _R()


def test_sanitiser_never_shows_fidelity_metadata_to_the_llm():
    llm = _CapturingLLM()
    sanitize_user_prompt_for_transformations(_USER + _CONTEXT, llm)
    assert llm.seen is not None
    low = llm.seen.lower()
    # The words that fooled the LLM into df.sample(...) must be gone.
    for poison in ("analysis context", "fidelity", "random_baseline", "sample"):
        assert poison not in low, f"{poison!r} leaked to the coder LLM: {llm.seen!r}"


def test_sanitiser_passthrough_also_strips_when_no_llm():
    # llm=None → passthrough path must still return the cleaned prompt.
    out = sanitize_user_prompt_for_transformations(_USER + _CONTEXT, None)
    assert out == _USER
    assert "[Analysis context]" not in out
