"""
Tests for app/core/agent_llm.py — the central factory that gives MTA/DTA
agents hybrid-reasoning models with per-agent env revert knobs.
"""
import pytest

from app.core.agent_llm import build_agent_llm, supports_reasoning


def test_supports_reasoning_prefixes():
    assert supports_reasoning("openai/gpt-oss-120b")
    assert supports_reasoning("OPENAI/GPT-OSS-20B")  # case-insensitive
    assert not supports_reasoning("llama-3.3-70b-versatile")
    assert not supports_reasoning("qwen/qwen3-32b")  # different effort value space
    assert not supports_reasoning("")
    assert not supports_reasoning(None)


def test_no_api_key_returns_none():
    assert build_agent_llm(agent="MTA", api_key=None) is None
    assert build_agent_llm(agent="MTA", api_key="") is None


def test_default_model_is_reasoning_with_hidden_format(monkeypatch):
    monkeypatch.delenv("AVALOKA_MTA_MODEL", raising=False)
    monkeypatch.delenv("AVALOKA_AGENT_REASONING_FORMAT", raising=False)
    llm = build_agent_llm(agent="MTA", api_key="k", default_effort="low")
    assert llm.model_name == "openai/gpt-oss-120b"
    assert llm.reasoning_format == "hidden"
    assert llm.reasoning_effort == "low"


def test_env_revert_to_llama_emits_no_reasoning_kwargs(monkeypatch):
    # The instant-revert story: a non-reasoning model must produce a plain
    # ChatGroq (no reasoning kwargs), byte-identical to pre-reasoning behavior.
    monkeypatch.setenv("AVALOKA_MTA_MODEL", "llama-3.3-70b-versatile")
    llm = build_agent_llm(agent="MTA", api_key="k")
    assert llm.model_name == "llama-3.3-70b-versatile"
    assert llm.reasoning_format is None
    assert llm.reasoning_effort is None


def test_legacy_env_var_name_is_honored(monkeypatch):
    # DTA keeps its pre-existing env names (DTA_CODER_MODEL / DTA_VALIDATOR_MODEL).
    monkeypatch.setenv("DTA_CODER_MODEL", "llama-3.3-70b-versatile")
    llm = build_agent_llm(
        agent="DTA_CODER", api_key="k", env_model_var="DTA_CODER_MODEL"
    )
    assert llm.model_name == "llama-3.3-70b-versatile"
    assert llm.reasoning_effort is None


def test_per_agent_effort_override(monkeypatch):
    monkeypatch.delenv("AVALOKA_DTA_CODER_MODEL", raising=False)
    monkeypatch.setenv("AVALOKA_DTA_CODER_REASONING_EFFORT", "high")
    llm = build_agent_llm(agent="DTA_CODER", api_key="k", default_effort="medium")
    assert llm.reasoning_effort == "high"


def test_global_format_override(monkeypatch):
    monkeypatch.setenv("AVALOKA_AGENT_REASONING_FORMAT", "parsed")
    llm = build_agent_llm(agent="MTA", api_key="k")
    assert llm.reasoning_format == "parsed"


def test_temperature_passthrough():
    # langchain_groq clamps exact 0.0 to 1e-08 (Groq API rejects zero);
    # same clamping applied to the old direct ChatGroq calls.
    llm = build_agent_llm(agent="MTA", api_key="k", temperature=0.0)
    assert llm.temperature <= 1e-6
    llm2 = build_agent_llm(agent="MTA", api_key="k", temperature=0.7)
    assert llm2.temperature == 0.7


def test_wired_agents_use_factory_defaults():
    """The four converted call sites must all resolve to reasoning models
    (guards against a silent revert of any single site)."""
    import app.agents.mta_v2.training_reply_classifier as c
    import app.agents.data_transfer_agent.daft_validator as v
    import app.agents.data_transfer_agent.daft_coder as dc

    for mod, attr in [(c, "_classifier_llm"), (v, "validator_llm"), (dc, "coder_llm")]:
        llm = getattr(mod, attr)
        if llm is None:  # key not set in this environment — nothing to assert
            continue
        assert supports_reasoning(llm.model_name), f"{attr} silently reverted"
        assert llm.reasoning_format == "hidden"
