"""Model-matrix track: registration, key guards, and adapter contract."""

from __future__ import annotations

import pytest

from benchmarks.adapters import get_adapter
from benchmarks.core.registry import get_suite


def test_suite_registers_the_full_roster():
    tasks, configs = get_suite("track7_model_matrix")
    t6_tasks, _ = get_suite("track6_conversational")
    # the conversational probes, unchanged: same ids, same order
    assert [t.id for t in tasks] == [t.id for t in t6_tasks]
    assert len(tasks) == 10
    from benchmarks.tracks.model_matrix.track import MATRIX
    assert len(configs) == len(MATRIX)
    providers = {c.extra["provider"] for c in configs}
    assert providers == {"groq", "openrouter", "ollama"}


def test_config_ids_are_unique():
    """A duplicate id silently overwrites a row in the scorecard record."""
    _, configs = get_suite("track7_model_matrix")
    ids = [c.id for c in configs]
    assert len(ids) == len(set(ids))


def test_roster_spans_the_price_range_it_claims_to():
    """The track's question is 'what does the cheap tier give up'. Both ends
    must be present or the table cannot answer it."""
    from benchmarks.core.pricing import price_for
    _, configs = get_suite("track7_model_matrix")
    priced = [price_for(c.extra["provider"], c.extra["model"]) for c in configs]
    out = sorted(p.completion_usd_per_1m for p in priced if p.is_priced)
    assert out[0] < 0.2, "no budget-tier model in the roster"
    assert out[-1] >= 25.0, "no frontier-tier model in the roster"


def test_same_model_appears_on_two_providers_as_a_control():
    """gpt-oss-120b on Groq and OpenRouter: a divergence is serving, not model."""
    _, configs = get_suite("track7_model_matrix")
    by_model = {}
    for c in configs:
        by_model.setdefault(c.extra["model"], set()).add(c.extra["provider"])
    assert by_model["openai/gpt-oss-120b"] == {"groq", "openrouter"}


def test_every_config_names_its_model_in_both_places():
    """Reporting reads config.llm; the adapter reads extra['model']."""
    _, configs = get_suite("track7_model_matrix")
    for c in configs:
        assert c.llm == c.extra["model"]


def test_missing_key_is_a_tagged_config_error_not_a_zero(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_PLANNING_AGENT", raising=False)
    tasks, configs = get_suite("track7_model_matrix")
    groq_cfg = next(c for c in configs if c.extra["provider"] == "groq")
    out = get_adapter("llm_direct").run_mission(tasks[0], groq_cfg)
    assert out.error and "sut_start_error" in out.error


def test_unknown_provider_is_rejected():
    import dataclasses
    tasks, configs = get_suite("track7_model_matrix")
    bad = dataclasses.replace(configs[0], extra={**configs[0].extra, "provider": "nope"})
    out = get_adapter("llm_direct").run_mission(tasks[0], bad)
    assert out.error and "unknown provider" in out.error


def test_ollama_needs_no_key(monkeypatch):
    """Local serving must not demand credentials; it fails on connection instead."""
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")   # nothing listens here
    import importlib

    import benchmarks.adapters.llm_direct as mod
    importlib.reload(mod)
    tasks, configs = get_suite("track7_model_matrix")
    local = next(c for c in configs if c.extra["provider"] == "ollama")
    import dataclasses
    fast = dataclasses.replace(local, extra={**local.extra, "timeout_s": 2})
    out = mod.LLMDirect().run_mission(tasks[0], fast)
    assert out.error and "provider_error" in out.error   # connection, not credentials
    monkeypatch.delenv("OLLAMA_URL")
    importlib.reload(mod)
