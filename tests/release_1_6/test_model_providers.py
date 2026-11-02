"""1.6 release suite: model providers (plan cases MP-03, MP-04, MP-11, MP-12).

Every test names the plan case it certifies; see
docs/test-reports/1.6-traceability.csv. All hermetic: clients are constructed,
never called, and the agent-import checks run in a subprocess so a module-level
LLM built under one provider never leaks into another test.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

from app.core import inference
from app.core.inference import build_chat_model

REPO = Path(__file__).resolve().parents[2]

# Every env var that can steer provider or model choice. Cleared so the host
# environment (a developer's real keys, a CI secret) cannot decide a result.
_STEERING_PREFIXES = ("INFERENCE_", "AVALOKA_", "AZURE_OPENAI_")
_STEERING_KEYS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE", "GROQ_API_KEY",
    "GROQ_API_KEY_PLANNING_AGENT", "GROQ_API_KEY_CODING_AGENT", "GEMINI_API_KEY",
    "AWS_REGION", "AWS_DEFAULT_REGION", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT",
    "KUBERNETES_SERVICE_HOST",
)


def _is_steering(key: str) -> bool:
    return key.startswith(_STEERING_PREFIXES) or key in _STEERING_KEYS


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for key in list(os.environ):
        if _is_steering(key):
            monkeypatch.delenv(key, raising=False)


# Module-level LLM clients built at import time, as an operator's deployment
# really constructs them. (module, attribute)
_IMPORT_TIME_CLIENTS = [
    ("app.agents.planner", "llm"),
    ("app.agents.coder", "coder_llm"),
    ("app.agents.validator", "validator_llm"),
    ("app.agents.summarizer", "summarizer_llm"),
    ("app.agents.result_narrator", "narrator_llm"),
    ("app.agents.profiling_agent", "profiling_llm"),
    ("app.agents.visualization_agent", "viz_llm"),
]

_PROBE = r"""
import importlib, json, sys
out = {}
for mod, attr in json.loads(sys.argv[1]):
    try:
        obj = getattr(importlib.import_module(mod), attr)
    except Exception as exc:  # report, never hide, an import-time failure
        out[mod] = {"error": f"{type(exc).__name__}: {exc}"}
        continue
    out[mod] = {
        "class": type(obj).__name__,
        "model": getattr(obj, "model_name", None) or getattr(obj, "model", None),
        "base_url": str(getattr(obj, "openai_api_base", None) or ""),
        "has_fallbacks": hasattr(obj, "fallbacks"),
    }
print("PROBE" + json.dumps(out))
"""


def _import_agents_under(env_overrides: dict) -> dict:
    """Import the agent modules in a fresh interpreter under *env_overrides*."""
    env = {k: v for k, v in os.environ.items() if not _is_steering(k)}
    env.update(env_overrides)
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(_IMPORT_TIME_CLIENTS)],
        env=env, cwd=str(REPO), capture_output=True, text=True, timeout=600,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE")), None)
    assert line, f"probe produced no result (rc={proc.returncode}):\n{proc.stderr[-2000:]}"
    return json.loads(line[len("PROBE"):])


# --------------------------------------------------------------------------- #
# MP-01: Groq is the default provider, in code and in the Helm chart
# --------------------------------------------------------------------------- #
def _helm_default_provider() -> str:
    """inference.provider from the chart's values.yaml (top-level `inference:` block)."""
    import yaml

    values = yaml.safe_load((REPO / "deploy/helm/avaloka/values.yaml").read_text(encoding="utf-8"))
    return str(values["inference"]["provider"]).strip().lower()


def test_mp01_default_provider_is_groq_when_nothing_is_configured(monkeypatch):
    """MP-01: with no provider and no key configured, every agent role resolves to
    Groq (owner decision 2026-10-31), and a Groq key alone keeps it on Groq even
    when an OpenRouter key is also present."""
    for role, agent in (("planning", "PLANNER"), ("coding", "CODER"), ("viz", "VIZ")):
        assert inference.resolve_provider(role=role, agent=agent) == inference.GROQ, (role, agent)

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert inference.resolve_provider(role="planning", agent="PLANNER") == inference.GROQ


def test_mp01_code_default_matches_the_helm_chart_default():
    """MP-01: the code's unconfigured default is the provider the Helm chart ships,
    so a chart install and a bare install can never pick different providers."""
    helm = _helm_default_provider()
    assert helm == inference.GROQ, f"chart ships {helm!r}"
    assert inference.resolve_provider(role="planning", agent="PLANNER") == helm


# --------------------------------------------------------------------------- #
# MP-03: local tier sends nothing off the box
# --------------------------------------------------------------------------- #
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_on_box(url: str) -> bool:
    """An in-cluster service name (no dots) or a loopback address."""
    host = urlparse(url).hostname or ""
    return host in _LOCAL_HOSTS or (host != "" and "." not in host)


def test_mp03_local_provider_builds_every_agent_against_an_on_box_endpoint():
    """MP-03: with INFERENCE_PROVIDER=local every agent's client targets the
    local/in-cluster endpoint and carries no off-box (OpenRouter) fallback."""
    result = _import_agents_under({"INFERENCE_PROVIDER": "local"})

    errors = {m: r["error"] for m, r in result.items() if "error" in r}
    assert not errors, f"agent import failed under the local provider: {errors}"
    for mod, info in result.items():
        assert info["class"] == "ChatOpenAI", f"{mod}: {info}"
        assert _is_on_box(info["base_url"]), f"{mod} would call {info['base_url']!r}"
        assert not info["has_fallbacks"], f"{mod} carries an off-box fallback"


def test_mp03_local_provider_ignores_a_stray_openrouter_key_for_the_primary(monkeypatch):
    """MP-03: the local provider's primary client is on-box even when an
    OpenRouter key happens to be present in the environment."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "local")
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm = build_chat_model(role="coding", agent="CODER", groq_model="openai/gpt-oss-120b",
                           groq_api_key=None, tier="large")
    assert type(llm).__name__ == "ChatOpenAI"
    assert _is_on_box(str(llm.openai_api_base)), llm.openai_api_base


# --------------------------------------------------------------------------- #
# MP-04: local model defaults name a model the shipped manifests serve
# --------------------------------------------------------------------------- #
def _served_models() -> set[str]:
    """Model ids the shipped deploy/inference manifests actually serve."""
    served: set[str] = set()
    ollama = (REPO / "deploy/inference/ollama-cpu.yaml").read_text(encoding="utf-8")
    m = re.search(r'-\s*name:\s*MODEL\s*\n\s*value:\s*"?([^"\n]+)"?', ollama)
    assert m, "ollama-cpu.yaml no longer declares MODEL"
    served.add(m.group(1).strip())
    vllm = (REPO / "deploy/inference/vllm-gpu.yaml").read_text(encoding="utf-8")
    m = re.search(r'"--served-model-name"\s*\n\s*-\s*"([^"]+)"', vllm)
    assert m, "vllm-gpu.yaml no longer declares --served-model-name"
    served.add(m.group(1).strip())
    return served


def test_mp04_local_provider_defaults_are_served_by_a_shipped_manifest():
    """MP-04: the local provider's default models (both tiers) are model ids a
    shipped serving manifest actually serves."""
    served = _served_models()
    for tier, model in inference._PROVIDER_MODEL_DEFAULTS[inference.LOCAL].items():
        assert model in served, f"local {tier} default {model!r} not in {sorted(served)}"


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "MP-04: the cluster-profile local fallback default, google/gemma-4-31b-it "
    "(app/core/model_fallback.py LOCAL_MODEL_CLUSTER), is an OpenRouter catalogue "
    "id that neither shipped manifest serves. The laptop default is the served "
    "DEFAULT_LOCAL_MODEL since the Gemma laptop fallback was replaced."))
def test_mp04_local_fallback_defaults_are_served_by_a_shipped_manifest():
    """MP-04: the model the local fallback tier asks for, in both deployment
    profiles, is one a shipped serving manifest actually serves."""
    from app.core import model_fallback

    served = _served_models()
    for profile in ("cluster", "laptop"):
        model = model_fallback.local_fallback_model(profile=profile)
        assert model in served, f"{profile} fallback {model!r} not in {sorted(served)}"


# --------------------------------------------------------------------------- #
# MP-11: Azure provider construction
# --------------------------------------------------------------------------- #
def test_mp11_azure_builds_a_client_for_the_configured_endpoint_and_deployment(monkeypatch):
    """MP-11: INFERENCE_PROVIDER=azure builds an Azure client bound to the
    configured endpoint and the tier's deployment name."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "azure")
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_LARGE", "deploy-large")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_SMALL", "deploy-small")

    large = build_chat_model(role="coding", agent="CODER", groq_model="openai/gpt-oss-120b",
                             groq_api_key=None, tier="large")
    small = build_chat_model(role="planning", agent="SUMMARIZER", groq_model="openai/gpt-oss-20b",
                             groq_api_key=None, tier="small")

    assert type(large).__name__ == "AzureChatOpenAI"
    assert large.azure_endpoint == "https://example-resource.openai.azure.com"
    assert large.deployment_name == "deploy-large"
    assert small.deployment_name == "deploy-small"


def test_mp11_azure_per_agent_deployment_beats_the_tier_default(monkeypatch):
    """MP-11: AVALOKA_<AGENT>_AZURE_DEPLOYMENT selects that agent's deployment."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "azure")
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_LARGE", "deploy-large")
    monkeypatch.setenv("AVALOKA_CODER_AZURE_DEPLOYMENT", "coder-only")

    llm = build_chat_model(role="coding", agent="CODER", groq_model="openai/gpt-oss-120b",
                           groq_api_key=None, tier="large")
    assert llm.deployment_name == "coder-only"


# --------------------------------------------------------------------------- #
# MP-12: the Groq model gate does not block a non-Groq provider
# --------------------------------------------------------------------------- #
def test_mp12_openrouter_provider_is_not_blocked_by_the_groq_gate():
    """MP-12: selecting OpenRouter (with an OpenRouter-only model for the coder)
    imports every agent without UnsupportedModelError, and the coder really uses
    the OpenRouter model."""
    result = _import_agents_under({
        "INFERENCE_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "test-key",
        "AVALOKA_CODER_MODEL_OPENROUTER": "anthropic/claude-sonnet-4",
    })

    errors = {m: r["error"] for m, r in result.items() if "error" in r}
    assert not errors, f"agent import failed under OpenRouter: {errors}"
    coder = result["app.agents.coder"]
    assert coder["model"] == "anthropic/claude-sonnet-4", coder
    assert urlparse(coder["base_url"]).hostname == "openrouter.ai", coder


# --------------------------------------------------------------------------- #
# MP-12 sub-case (owner decision 2026-10-31): under a non-Groq provider the
# Groq-path model variable is ignored with a warning, and the Groq gate's error
# names the real alternatives.
# --------------------------------------------------------------------------- #
def test_mp12_groq_path_model_variable_is_ignored_with_a_warning_under_openrouter(monkeypatch, caplog):
    """MP-12: INFERENCE_PROVIDER=openrouter with AVALOKA_CODER_MODEL set resolves the
    coder to its table model and logs that the Groq-path variable was ignored,
    instead of raising UnsupportedModelError."""
    import logging

    from app.core import model_config as mc

    monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("AVALOKA_CODER_MODEL", "gpt-4o")
    with caplog.at_level(logging.WARNING, logger="app.core.model_config"):
        assert mc.resolve("coder") == mc.DEFAULT_MODELS["coder"]
    warned = [r.getMessage() for r in caplog.records if "AVALOKA_CODER_MODEL" in r.getMessage()]
    assert warned and "ignored" in warned[0] and "AVALOKA_CODER_MODEL_OPENROUTER" in warned[0], warned


def test_mp12_agents_import_under_openrouter_with_a_stray_groq_path_variable():
    """MP-12: every agent imports under INFERENCE_PROVIDER=openrouter even when the
    Groq-path AVALOKA_CODER_MODEL names a non-Groq model (it used to abort startup)."""
    result = _import_agents_under({
        "INFERENCE_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "test-key",
        "AVALOKA_CODER_MODEL": "gpt-4o",
    })
    errors = {m: r["error"] for m, r in result.items() if "error" in r}
    assert not errors, f"agent import failed: {errors}"
    assert urlparse(result["app.agents.coder"]["base_url"]).hostname == "openrouter.ai"


def test_mp12_groq_gate_error_names_the_real_alternatives():
    """MP-12: on the Groq path, a non-Groq model fails with a message that points at
    INFERENCE_PROVIDER / AVALOKA_<AGENT>_MODEL_<PROVIDER>, not the stale claim that the
    agent builds a ChatGroq client directly."""
    from app.core import model_config as mc

    with pytest.raises(mc.UnsupportedModelError) as err:
        mc.check_model_supported("coder", "gpt-4o")
    message = str(err.value)
    assert "INFERENCE_PROVIDER" in message
    assert "AVALOKA_CODER_MODEL_<PROVIDER>" in message
    assert "builds a ChatGroq client directly" not in message
