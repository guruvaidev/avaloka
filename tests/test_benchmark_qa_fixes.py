"""Regressions for the defects QA found in the cost-accounting work.

Each test names the finding it pins. The unifying property: **a zero must
always be attributable.** A $0.00 cost, a 0.00 quality score and an empty
answer each have several possible causes, and the harness is only trustworthy
if the artifact says which one occurred.
"""

from __future__ import annotations

import dataclasses
import io
import json

import pytest

import benchmarks.adapters.llm_direct as llm_direct
from benchmarks.core.config import RunConfig
from benchmarks.core.metrics import aggregate
from benchmarks.core.pricing import (
    SOURCE_CATALOG, SOURCE_FREE_TIER, SOURCE_PROXY, SOURCE_UNPRICED,
    SOURCE_USAGE_MISSING, parse_catalog, price_for, validate_models,
)
from benchmarks.core.registry import get_suite
from benchmarks.core.reporting import render_scorecard, results_to_record
from benchmarks.core.spend import (
    RemainingBudget, SpendLedger, estimate_suite_cost, fetch_remaining_budget,
)
from benchmarks.core.task import MissionOutcome, TrialResult


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture
def stub_http(monkeypatch):
    def _install(payload):
        def _urlopen(request, timeout=None):
            return _FakeResponse(json.dumps(payload).encode())
        monkeypatch.setattr(llm_direct.urllib.request, "urlopen", _urlopen)
    return _install


def _cfg(config_id):
    _, configs = get_suite("track7_model_matrix")
    return next(c for c in configs if c.id == config_id)


def _task():
    return get_suite("track7_model_matrix")[0][0]


def _ok(prompt=1000, completion=500, content="a real answer", finish="stop"):
    return {"id": "gen-1", "choices": [{"message": {"content": content},
                                        "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion}}


# ═══ F1 · the scorecard must distinguish "scored zero" from "never ran" ════

def _record_with(n_errors, n_trials=3, cost=0.0, source=SOURCE_CATALOG):
    trials = [
        TrialResult(task_id="conv_probe", track="conversational",
                    config_id="m_or_claude_opus5", trial_index=i,
                    quality={"pass": 0.0, "score": 0.0},
                    outcome=MissionOutcome(answer=None, cost_usd=cost,
                                           cost_source=source,
                                           error="provider_error: HTTP 429"
                                           if i < n_errors else None))
        for i in range(n_trials)
    ]
    return results_to_record([aggregate(trials)], [_cfg("m_or_claude_opus5")],
                             suite="track7_model_matrix")


def test_all_trials_errored_is_marked_not_measured_on_the_scorecard():
    """The rate-limited-Opus scenario: 0.00 everywhere, and it must LOOK wrong."""
    card = render_scorecard(_record_with(n_errors=3))
    assert "NOT MEASURED" in card
    assert "never successfully measured" in card
    assert "exclude the whole config" in card


def test_a_genuine_zero_is_not_marked_not_measured():
    """A model that really did score 0.00 must not be excused as an outage."""
    card = render_scorecard(_record_with(n_errors=0, cost=0.02))
    assert "NOT MEASURED" not in card
    assert "🛑" not in card


def test_partial_errors_are_counted_but_not_written_off():
    card = render_scorecard(_record_with(n_errors=1))
    assert "NOT MEASURED" not in card
    assert "| 1 |" in card or "1/3" in card


def test_scorecard_surfaces_the_error_count_and_price_provenance():
    card = render_scorecard(_record_with(n_errors=0, cost=0.02))
    assert "Errors" in card and "Price src" in card
    assert SOURCE_CATALOG in card


def test_an_aborted_run_says_so_at_the_top():
    """Budget exhaustion truncates the sweep; later configs are absent, not bad."""
    record = _record_with(n_errors=0, cost=0.02)
    record["meta"] = {"spend": {"spent_usd": 5.0, "calls": 120, "ceiling_usd": 5.0,
                                "unpriced_calls": 0},
                      "aborted_reason": "spend ceiling exceeded: $5.0012"}
    card = render_scorecard(record)
    assert "RUN INCOMPLETE" in card
    assert "must not be published as a model comparison" in card
    assert "Observed spend" in card


# ═══ F1a · truncation is a harness setting, not a model property ═══════════

def test_truncated_answer_is_an_error_not_a_score(stub_http, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    stub_http(_ok(content="half an ans", finish="length"))
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))
    assert out.error and "truncation_error" in out.error
    assert out.cost_usd > 0, "it was still billed, and that must be recorded"
    assert out.artifacts["finish_reason"] == "length"


def test_reasoning_model_that_returns_no_content_is_an_error_not_a_zero(stub_http, monkeypatch):
    """content=null with tokens billed: graded 0.00 with n_errors=0 before the fix."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    stub_http(_ok(content=None, finish="stop"))
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))
    assert out.error and "empty_answer_error" in out.error
    assert out.completion_tokens == 500
    assert out.cost_usd > 0


def test_a_normal_answer_is_still_not_an_error(stub_http, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    stub_http(_ok())
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))
    assert out.error is None and out.answer == "a real answer"


# ═══ F1b · content as a list of parts ═════════════════════════════════════

def test_list_content_is_flattened_to_text(stub_http, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    stub_http(_ok(content=[{"type": "text", "text": "part one "},
                           {"type": "text", "text": "part two"}]))
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))
    assert out.answer == "part one part two"
    assert isinstance(out.answer, str), "evaluators reject non-str and score 0"
    assert out.error is None


# ═══ F2 · missing usage must not masquerade as a measured zero ═════════════

def test_missing_usage_is_tagged_usage_missing_not_catalog(stub_http, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    stub_http({"id": "g", "choices": [{"message": {"content": "hi"},
                                       "finish_reason": "stop"}]})
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))
    assert out.cost_source == SOURCE_USAGE_MISSING
    assert out.cost_source != SOURCE_CATALOG, "a $0 tagged 'catalog' reads as measured"
    assert out.cost_usd == 0.0


def test_a_real_zero_usage_report_is_still_catalog(stub_http, monkeypatch):
    """'the provider said zero' and 'the provider said nothing' must differ."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    stub_http(_ok(prompt=0, completion=0))
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))
    assert out.cost_source == SOURCE_CATALOG


def test_usage_missing_calls_are_counted_blind_by_the_ledger():
    ledger = SpendLedger(ceiling_usd=1.0)
    ledger.charge(0.0, cost_source=SOURCE_USAGE_MISSING)
    ledger.charge(0.0, cost_source=SOURCE_UNPRICED)
    ledger.charge(0.0)                      # MissionOutcome's "unknown" default
    ledger.charge(0.01, cost_source=SOURCE_CATALOG)
    assert ledger.blind_calls == 3
    assert ledger.usage_missing_calls == 1
    assert ledger.unpriced_calls == 2       # includes the "unknown" default


# ═══ F3 · tiered long-prompt pricing ══════════════════════════════════════

def test_snapshot_retains_tiered_overrides():
    from benchmarks.core.pricing import load_snapshot
    tiered = [m for m in load_snapshot()["data"] if (m["pricing"] or {}).get("overrides")]
    assert len(tiered) > 50, "overrides were pruned out of the snapshot again"


def test_long_prompts_are_priced_at_the_higher_tier():
    """qwen3.7-flash: $0.13/1M completion below 32k, $0.40/1M above."""
    price = price_for("openrouter", "qwen/qwen3.7-flash")
    assert price.tiers, "no tiers parsed"
    short = price.cost_usd(1_000, 1_000)
    long = price.cost_usd(50_000, 1_000)
    per_prompt_token_short = short / 1_000
    per_prompt_token_long = long / 50_000
    assert price.rates_for(1_000) != price.rates_for(50_000)
    assert long > short


def test_the_highest_applicable_tier_wins():
    price = price_for("openrouter", "qwen/qwen3.7-flash")
    thresholds = [t[0] for t in price.tiers]
    assert thresholds == sorted(thresholds), "tiers must be threshold-ordered"
    top = max(thresholds)
    assert price.rates_for(top + 1) == price.rates_for(top)


def test_models_without_tiers_are_unaffected():
    price = price_for("openrouter", "anthropic/claude-opus-5")
    assert price.tiers == ()
    assert price.cost_usd(1_000_000, 0) == pytest.approx(5.0)


def test_a_malformed_tier_is_dropped_not_guessed():
    table = parse_catalog({"data": [{"id": "v/m", "pricing": {
        "prompt": "0.000001", "completion": "0.000002",
        "overrides": [{"min_prompt_tokens": 1000, "prompt": "oops"},
                      {"min_prompt_tokens": 2000, "prompt": "0.000004",
                       "completion": "0.000008"}]}}]})
    assert table["v/m"].tiers == ((2000, 4.0, 8.0),)


# ═══ F4 · :free tiers must not win efficiency rankings ════════════════════

def test_free_slug_is_tagged_free_tier_and_excluded_from_ranking():
    free = price_for("openrouter", "google/gemma-4-31b-it:free")
    paid = price_for("openrouter", "google/gemma-4-31b-it")
    assert free.source == SOURCE_FREE_TIER
    assert free.is_priced is False, "a $0 price divides by zero in $/quality"
    assert paid.is_priced is True


def test_free_tier_still_counts_as_existing_for_slug_validation():
    """It is a real model; it just cannot be priced against."""
    resolved = validate_models([("openrouter", "google/gemma-4-31b-it:free")])
    assert resolved["google/gemma-4-31b-it:free"] == SOURCE_FREE_TIER


def test_no_matrix_row_is_a_free_tier():
    from benchmarks.tracks.model_matrix.track import MATRIX
    for _, provider, model in MATRIX:
        assert price_for(provider, model).source != SOURCE_FREE_TIER, model


# ═══ P2 · provider is required, because it selects who gets billed ════════

def test_a_config_without_a_provider_is_rejected_not_sent_to_groq():
    """QA proved this posted claude-opus-5 to api.groq.com with the real key."""
    cfg = dataclasses.replace(_cfg("m_or_claude_opus5"), extra={})
    out = llm_direct.LLMDirect().run_mission(_task(), cfg)
    assert out.error and "sut_start_error" in out.error
    assert "required" in out.error


def test_require_provider_raises_for_suite_build_time_validation():
    cfg = RunConfig(id="broken", sut="llm_direct", llm="anthropic/claude-opus-5")
    with pytest.raises(ValueError) as exc:
        llm_direct.require_provider(cfg)
    assert "broken" in str(exc.value) and "which account is billed" in str(exc.value)


def test_unknown_provider_still_rejected():
    cfg = dataclasses.replace(_cfg("m_or_claude_opus5"),
                              extra={"provider": "nope", "model": "x"})
    out = llm_direct.LLMDirect().run_mission(_task(), cfg)
    assert out.error and "unknown provider" in out.error


def test_a_providerless_config_is_treated_as_billable_by_the_cloud_gate():
    """Fail closed: withhold it rather than run it on an unknown account."""
    from benchmarks.adapters import config_requires_cloud
    cfg = dataclasses.replace(_cfg("m_or_claude_opus5"), extra={})
    assert config_requires_cloud(cfg) is True


def test_every_matrix_config_declares_its_provider():
    _, configs = get_suite("track7_model_matrix")
    for c in configs:
        assert llm_direct.require_provider(c) == c.extra["provider"]


# ═══ The $5/day account cap ═══════════════════════════════════════════════

def test_budget_probe_without_a_key_is_unavailable_not_a_guess(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    budget = fetch_remaining_budget()
    assert budget.source == "unavailable" and budget.known is False


def test_budget_probe_parses_the_live_shape(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    payload = {"data": {"limit": 5, "usage": 0.0139, "limit_remaining": 4.9861,
                        "limit_reset": "daily"}}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse(json.dumps(payload).encode()))
    budget = fetch_remaining_budget()
    assert budget.known
    assert budget.remaining_usd == pytest.approx(4.9861)
    assert budget.limit_usd == 5.0 and budget.reset_period == "daily"


def test_budget_probe_never_raises_and_never_echoes_the_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRETVALUE")

    def _boom(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr("urllib.request.urlopen", _boom)

    budget = fetch_remaining_budget()
    assert budget.source == "unavailable"
    assert "SECRETVALUE" not in budget.detail
    assert "OSError" in budget.detail


def test_remaining_budget_derives_the_remainder_when_not_reported(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse(
            json.dumps({"data": {"limit": 5, "usage": 1.25}}).encode()))
    assert fetch_remaining_budget().remaining_usd == pytest.approx(3.75)


def test_default_ceiling_never_exceeds_one_days_allowance():
    from benchmarks.core.spend import DEFAULT_MAX_SPEND_USD
    assert DEFAULT_MAX_SPEND_USD <= 5.0


def test_full_sweep_fits_inside_the_daily_cap():
    """If this ever fails, the roster grew past what the account can run."""
    tasks, configs = get_suite("track7_model_matrix")
    assert estimate_suite_cost(tasks, configs).high_usd < 5.0
