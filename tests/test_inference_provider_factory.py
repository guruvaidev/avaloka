"""
Tests for app/core/inference.py — the provider-agnostic chat-model factory that
abstracts the agent layer away from Groq (local OpenAI-spec model in k8s, Groq,
OpenRouter, Bedrock, Vertex, Azure).

Guarantee under test: Groq stays the default and byte-identical, and every other
provider is a pure env switch that returns the right LangChain chat model (or
None when unconfigured, preserving each agent's deterministic fallback).
"""
import importlib.util

import pytest

from app.core import inference
from app.core.inference import (
    build_chat_model,
    resolve_provider,
    _resolve_model,
)


@pytest.fixture(autouse=True)
def _clear_inference_env(monkeypatch):
    """Every test starts from a clean, provider-unset environment."""
    for key in list(__import__("os").environ):
        if key.startswith("INFERENCE_") or key.startswith("AVALOKA_") or key in (
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_API_BASE",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "GOOGLE_CLOUD_PROJECT",
            "GCP_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT_LARGE",
        ):
            monkeypatch.delenv(key, raising=False)


def _mk(**over):
    base = dict(
        role="planning",
        agent="PLANNER",
        groq_model="openai/gpt-oss-120b",
        groq_api_key="gk",
        tier="large",
        temperature=0.2,
    )
    base.update(over)
    return build_chat_model(**base)


# --------------------------------------------------------------------------
# provider resolution
# --------------------------------------------------------------------------
def test_default_provider_is_groq():
    assert resolve_provider(role="planning", agent="PLANNER") == "groq"


def test_provider_precedence_agent_over_role_over_global(monkeypatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "groq")
    monkeypatch.setenv("INFERENCE_PROVIDER_CODING", "openrouter")
    monkeypatch.setenv("AVALOKA_CODER_PROVIDER", "local")
    assert resolve_provider(role="planning", agent="X") == "groq"       # global
    assert resolve_provider(role="coding", agent="X") == "openrouter"   # per-role
    assert resolve_provider(role="coding", agent="CODER") == "local"    # per-agent


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ollama", "local"),
        ("vllm", "local"),
        ("aws", "bedrock"),
        ("gcp", "vertex"),
        ("vertexai", "vertex"),
        ("azure-openai", "azure"),
        ("", "groq"),
    ],
)
def test_provider_aliases(monkeypatch, raw, expected):
    monkeypatch.setenv("INFERENCE_PROVIDER", raw)
    assert resolve_provider(role="planning", agent="X") == expected


# --------------------------------------------------------------------------
# groq path — the byte-identical default
# --------------------------------------------------------------------------
def test_groq_default_returns_chatgroq_with_reasoning():
    llm = _mk(reasoning_effort="medium", reasoning_format="parsed")
    assert type(llm).__name__ == "ChatGroq"
    assert llm.model_name == "openai/gpt-oss-120b"
    assert llm.reasoning_effort == "medium"
    assert llm.reasoning_format == "parsed"


def test_groq_no_key_returns_none():
    assert _mk(groq_api_key=None) is None
    assert _mk(groq_api_key="") is None


def test_groq_non_reasoning_model_has_no_reasoning_kwargs():
    # A non-gpt-oss model must emit a plain ChatGroq even if effort is passed.
    llm = _mk(groq_model="llama-3.3-70b-versatile", reasoning_effort="high")
    assert llm.model_name == "llama-3.3-70b-versatile"
    assert llm.reasoning_effort is None
    assert llm.reasoning_format is None


def test_groq_env_model_override(monkeypatch):
    monkeypatch.setenv("AVALOKA_PLANNER_MODEL", "llama-3.3-70b-versatile")
    llm = _mk(env_model_var="AVALOKA_PLANNER_MODEL", reasoning_effort="medium")
    assert llm.model_name == "llama-3.3-70b-versatile"


# --------------------------------------------------------------------------
# local / openai-spec providers
# --------------------------------------------------------------------------
def test_local_returns_chatopenai_at_incluster_default(monkeypatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "local")
    llm = _mk()
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "qwen2.5:3b-instruct"
    assert str(llm.openai_api_base) == "http://avaloka-local-llm:11434/v1"


def test_local_base_url_and_model_env_override(monkeypatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "local")
    monkeypatch.setenv("INFERENCE_LOCAL_BASE_URL", "http://my-llm:8000/v1")
    monkeypatch.setenv("INFERENCE_LOCAL_MODEL_LARGE", "qwen-local")
    llm = _mk()
    assert llm.model_name == "qwen-local"
    assert str(llm.openai_api_base) == "http://my-llm:8000/v1"


def test_openrouter_requires_key(monkeypatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")
    assert _mk() is None  # no OPENROUTER_API_KEY
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_key")
    llm = _mk()
    assert type(llm).__name__ == "ChatOpenAI"
    assert str(llm.openai_api_base) == "https://openrouter.ai/api/v1"
    assert llm.model_name == "qwen/qwen-2.5-72b-instruct"


# --------------------------------------------------------------------------
# cloud providers
# --------------------------------------------------------------------------
# langchain-aws and langchain-google-vertexai live in requirements-cloud-llm.txt,
# not requirements.txt: langchain-google-vertexai needs google-cloud-storage<3
# while the data plane needs >=3.4, and pinning both makes the image
# unbuildable. app/core/inference.py imports them lazily and disables only these
# two providers when they are missing -- which is the case on a default install
# and in CI. The "returns None when unconfigured" half of the contract holds
# either way and is asserted unconditionally; only the half that needs the
# package to exist is skipped, so `pip install -r requirements-cloud-llm.txt`
# still runs it.
_HAS_BEDROCK = importlib.util.find_spec("langchain_aws") is not None
_HAS_VERTEX = importlib.util.find_spec("langchain_google_vertexai") is not None


def test_bedrock_without_a_region_is_disabled(monkeypatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "bedrock")
    assert _mk() is None


@pytest.mark.skipif(
    not _HAS_BEDROCK,
    reason="langchain-aws is not installed (optional: requirements-cloud-llm.txt)",
)
def test_bedrock_builds_with_region(monkeypatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "bedrock")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    llm = _mk()
    assert type(llm).__name__ == "ChatBedrockConverse"


@pytest.mark.skipif(
    not _HAS_VERTEX,
    reason=("langchain-google-vertexai is not installed "
            "(optional: requirements-cloud-llm.txt)"),
)
def test_vertex_builds_with_project(monkeypatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "vertex")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    llm = _mk()
    assert type(llm).__name__ == "ChatVertexAI"


def test_azure_unconfigured_returns_none(monkeypatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "azure")
    assert _mk() is None


# --------------------------------------------------------------------------
# model tier resolution
# --------------------------------------------------------------------------
def test_resolve_model_tiers_and_overrides(monkeypatch):
    assert _resolve_model("openrouter", "CODER", "large") == "qwen/qwen-2.5-72b-instruct"
    assert _resolve_model("openrouter", "CODER", "small") == "qwen/qwen-2.5-7b-instruct"
    monkeypatch.setenv("INFERENCE_OPENROUTER_MODEL_SMALL", "some/small")
    assert _resolve_model("openrouter", "CODER", "small") == "some/small"
    monkeypatch.setenv("AVALOKA_CODER_MODEL_OPENROUTER", "special/model")
    assert _resolve_model("openrouter", "CODER", "small") == "special/model"


# --------------------------------------------------------------------------
# integration: build_agent_llm role passthrough
# --------------------------------------------------------------------------
def test_build_agent_llm_honors_role_provider(monkeypatch):
    from app.core.agent_llm import build_agent_llm

    monkeypatch.setenv("INFERENCE_PROVIDER_CODING", "local")
    llm = build_agent_llm(agent="DTA_CODER", api_key="k", role="coding")
    assert type(llm).__name__ == "ChatOpenAI"
    # A planning-role agent stays on the default groq backend.
    planning = build_agent_llm(agent="MTA", api_key="k", role="planning")
    assert type(planning).__name__ == "ChatGroq"
