"""Central model configuration and backend-compatibility guarding."""

from __future__ import annotations

import pytest

from app.core.model_config import (DEFAULT_MODELS, UnsupportedModelError,
                                   check_model_supported, describe_all,
                                   env_var_for, model_for, provider_for_model,
                                   resolve)


# --------------------------------------------------------------------------- #
# The 8B models are gone from the two places they mattered
# --------------------------------------------------------------------------- #

def test_validator_no_longer_runs_on_an_8b_model():
    """The gate deciding whether generated code is sound was on llama-3.1-8b."""
    assert "8b" not in DEFAULT_MODELS["validator"].lower()


def test_summarizer_no_longer_runs_on_an_8b_model():
    """Every downstream agent trusts this JSON contract."""
    assert "8b" not in DEFAULT_MODELS["summarizer"].lower()


def test_validator_defaults_to_compound():
    assert DEFAULT_MODELS["validator"].startswith(("groq/compound", "compound"))


def test_no_agent_is_left_on_an_instant_tier_model():
    offenders = {a: m for a, m in DEFAULT_MODELS.items() if "instant" in m.lower()}
    assert not offenders, f"instant-tier models on correctness paths: {offenders}"


#: Verified live against this project's Groq account on 2026-08-17 via
#: GET /openai/v1/models plus a one-token completion per model. The entire
#: Llama line had already been removed from the account, which is why six
#: agents were pointing at models that returned 404.
LIVE_GROQ_CATALOGUE = {
    "allam-2-7b",
    "canopylabs/orpheus-arabic-saudi", "canopylabs/orpheus-v1-english",
    "groq/compound", "groq/compound-mini",
    "meta-llama/llama-prompt-guard-2-22m", "meta-llama/llama-prompt-guard-2-86m",
    "openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3.6-27b",
    "whisper-large-v3", "whisper-large-v3-turbo",
}


def test_every_default_model_exists_in_the_live_catalogue():
    """The regression that shipped: agents pointing at deleted models.

    Pinned to a snapshot rather than a network call so the suite stays
    offline and deterministic. scripts/ops/verify_models.py does the live
    check; run it when a provider deprecates something.
    """
    dead = {a: m for a, m in DEFAULT_MODELS.items() if m not in LIVE_GROQ_CATALOGUE}
    assert not dead, (
        f"agents point at models absent from the verified Groq catalogue: {dead}. "
        "Run scripts/ops/verify_models.py against a live key."
    )


def test_no_default_references_the_removed_llama_line():
    dead = {a: m for a, m in DEFAULT_MODELS.items() if m.lower().startswith("llama-")}
    assert not dead, f"llama-* models were removed from this Groq account: {dead}"


# --------------------------------------------------------------------------- #
# Resolution and overrides
# --------------------------------------------------------------------------- #

def test_defaults_are_used_when_no_env_is_set(monkeypatch):
    monkeypatch.delenv("AVALOKA_CODER_MODEL", raising=False)
    choice = model_for("coder")
    assert choice.model == DEFAULT_MODELS["coder"]
    assert choice.is_default is True


def test_env_overrides_the_default(monkeypatch):
    monkeypatch.setenv("AVALOKA_CODER_MODEL", "llama-3.3-70b-versatile")
    choice = model_for("coder")
    assert choice.model == "llama-3.3-70b-versatile"
    assert choice.is_default is False


def test_blank_env_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("AVALOKA_CODER_MODEL", "   ")
    assert model_for("coder").is_default is True


def test_legacy_planner_env_var_still_works(monkeypatch):
    """planner.py shipped AVALOKA_PLANNER_MODEL; breaking it breaks deployments."""
    assert env_var_for("planner") == "AVALOKA_PLANNER_MODEL"
    monkeypatch.setenv("AVALOKA_PLANNER_MODEL", "openai/gpt-oss-20b")
    assert model_for("planner").model == "openai/gpt-oss-20b"


def test_legacy_dta_coder_env_var_still_works():
    assert env_var_for("dta_coder") == "DTA_CODER_MODEL"


def test_unknown_agent_is_an_error_listing_known_agents():
    with pytest.raises(KeyError, match="coder"):
        model_for("nonexistent_agent")


# --------------------------------------------------------------------------- #
# The guard: models Groq cannot serve
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("model,provider", [
    ("gpt-5", "openai"),
    ("gpt-5-mini", "openai"),
    ("gpt-4o", "openai"),
    ("o3-mini", "openai"),
    ("claude-sonnet-4", "anthropic"),
    ("gemini-2.0-flash", "vertex"),
])
def test_foreign_models_are_identified_with_their_provider(model, provider):
    assert provider_for_model(model) == provider


@pytest.mark.parametrize("model", [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
    "groq/compound",
    "compound-beta",
    "qwen-2.5-32b",
    "gemma2-9b-it",
])
def test_groq_hosted_models_need_no_other_provider(model):
    assert provider_for_model(model) is None


def test_gpt_oss_is_recognised_as_groq_not_openai():
    """The confusing case: openai/gpt-oss-* IS on Groq, the hosted GPT line is not."""
    assert provider_for_model("openai/gpt-oss-120b") is None
    assert provider_for_model("gpt-5") == "openai"


def test_configuring_gpt5_today_fails_with_an_actionable_message():
    """A 404 midway through an analysis is a bad way to learn this."""
    with pytest.raises(UnsupportedModelError) as exc:
        check_model_supported("coder", "gpt-5")
    message = str(exc.value)
    assert "not Groq" in message
    assert "build_chat_model" in message      # points at the fix
    assert "AVALOKA_CODER_MODEL" in message   # names the var to change


def test_the_guard_is_inert_for_a_non_groq_backend():
    """Once provider routing exists, GPT-5 becomes legitimate."""
    check_model_supported("coder", "gpt-5", backend="openai")  # must not raise


def test_groq_models_pass_the_guard():
    for agent, model in DEFAULT_MODELS.items():
        check_model_supported(agent, model)   # every default must be serveable


def test_resolve_validates_as_well_as_resolves(monkeypatch):
    monkeypatch.setenv("AVALOKA_CODER_MODEL", "gpt-5")
    with pytest.raises(UnsupportedModelError):
        resolve("coder")


def test_resolve_returns_the_model_id(monkeypatch):
    monkeypatch.delenv("AVALOKA_VALIDATOR_MODEL", raising=False)
    assert resolve("validator") == DEFAULT_MODELS["validator"]


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def test_describe_all_covers_every_agent():
    described = describe_all()
    assert set(described) == set(DEFAULT_MODELS)
    assert all(v["serveable_by_backend"] for v in described.values())


def test_describe_all_flags_a_misconfigured_agent(monkeypatch):
    monkeypatch.setenv("AVALOKA_CODER_MODEL", "gpt-5")
    described = describe_all()
    assert described["coder"]["serveable_by_backend"] is False
    assert described["coder"]["requires_provider"] == "openai"
    assert described["coder"]["source"] == "AVALOKA_CODER_MODEL"


def test_describe_all_is_json_serialisable():
    import json
    json.dumps(describe_all())
