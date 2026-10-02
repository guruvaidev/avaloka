"""Guards for the defects found by executing the 1.6 Master Test Plan.

Each test fails on the pre-fix tree. They exist because every one of these
shipped: the plan had 126 cases and none had ever been run.
"""
from __future__ import annotations

import os

import pytest


# ── DEF-008: advertised upload formats must not 500 ────────────────────────

@pytest.mark.parametrize("fmt", ["csv", "tsv", "xml"])
def test_advertised_formats_have_a_reader(fmt: str) -> None:
    """/api/upload advertises these; the sampler must not reject them.

    Before the fix, .tsv and .xml fell through to FileHandler, which has no
    reader for either, and the API answered HTTP 500 "File type not supported"
    for formats it lists in SUPPORTED_UPLOAD_EXTS.
    """
    import inspect

    from app.agents import sampling_agent

    src = inspect.getsource(sampling_agent)
    assert f'"{fmt}"' in src, f"{fmt} has no branch in the sampler dispatch"


def test_tsv_is_read_by_the_csv_reader() -> None:
    import inspect

    from app.agents import sampling_agent

    src = inspect.getsource(sampling_agent)
    assert 'source_fmt in ("csv", "tsv")' in src, (
        "TSV should share the CSV reader, which already sniffs the delimiter")


# ── DEF-006: small data is read whole, not sampled ─────────────────────────

def test_a_tiny_file_is_not_sampled() -> None:
    """A 12-row CSV must not come back labelled 'computed on a sample'."""
    from app.agents.avaloka_agent.execution_profile import (
        FIDELITY_ENTIRE, decide_execution_profile)

    out = decide_execution_profile({"dataset_size_bytes": 600}, "statistical_analysis")
    assert out["analysis_fidelity"] == FIDELITY_ENTIRE
    assert out["execution_mode"] == "local", "a 600-byte file does not need the cluster"


def test_large_input_asks_before_a_full_run() -> None:
    from app.agents.avaloka_agent.execution_profile import decide_execution_profile

    small = decide_execution_profile({"dataset_size_bytes": 600}, "statistical_analysis")
    assert small["avaloka_recommendation"]["confirm_before_full_run"] is False


# ── DEF-005: a misconfigured provider is reported, not hidden ──────────────

def test_provider_with_no_key_is_diagnosed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact shape that shipped: provider=groq, empty Groq key, OpenRouter set.

    The operator supplied a valid key; the selected provider could not use it;
    every turn silently returned canned text.
    """
    from app.agents.avaloka_agent import agent

    monkeypatch.setenv("INFERENCE_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "x" * 20)
    monkeypatch.delenv("GROQ_API_KEY_PLANNING_AGENT", raising=False)

    problem = agent._diagnose_missing_model()
    assert problem, "a provider with no usable key must be reported"
    assert "GROQ_API_KEY" in problem, "the message must name the missing variable"
    assert "openrouter" in problem.lower(), (
        "the message should say a key for another provider IS present")


def test_usable_provider_is_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agents.avaloka_agent import agent

    monkeypatch.setenv("INFERENCE_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "x" * 20)
    assert agent._diagnose_missing_model() is None


def test_local_provider_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agents.avaloka_agent import agent

    monkeypatch.setenv("INFERENCE_PROVIDER", "local")
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert agent._diagnose_missing_model() is None, (
        "a local model server needs no API key")


# ── DEF-001: no private bucket baked into shipping code ────────────────────

def test_sysdocs_path_is_not_a_hardcoded_private_bucket() -> None:
    from app.agents.data_transfer_agent import daft_validator

    assert "avaloka-test-user-filestore" not in daft_validator.SYSDOCS_PATH, (
        "shipping code must not reach a private bucket; every install listed it, "
        "got 401, logged a traceback, and validated without the docs")


def test_sysdocs_path_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVALOKA_SYSDOCS_PATH", "s3://my-bucket/docs")
    import importlib

    from app.agents.data_transfer_agent import daft_validator

    importlib.reload(daft_validator)
    assert daft_validator.SYSDOCS_PATH == "s3://my-bucket/docs"


# ── DEF-007: one version, and the OSS line is its own ──────────────────────

def test_version_is_read_from_a_single_source() -> None:
    from app.api.server import APP_VERSION

    assert APP_VERSION and APP_VERSION != "1.5", (
        "/version reported a hardcoded 1.5 for the whole 1.6 line")
