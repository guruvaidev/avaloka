"""End-to-end cost accounting through the llm_direct adapter.

The provider's ``usage`` block is the only measured input; everything in the
Cost column is derived from it. These tests pin that derivation, and pin the
behaviour when the provider gives us nothing to derive from.

No network: the HTTP call is stubbed with a canned OpenAI-shaped payload, which
is also the only honest way to assert on cost without spending money to do it.
"""

from __future__ import annotations

import dataclasses
import io
import json

import pytest

import benchmarks.adapters.llm_direct as llm_direct
from benchmarks.core.pricing import (
    SOURCE_CATALOG, SOURCE_LOCAL, SOURCE_PROXY, SOURCE_UNPRICED,
    SOURCE_USAGE_MISSING,
)
from benchmarks.core.registry import get_suite


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _payload(prompt_tokens=1000, completion_tokens=500, content="an answer"):
    return {
        "id": "gen-abc123",
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


@pytest.fixture
def stub_http(monkeypatch):
    """Replace the urlopen call; capture the request the adapter would send."""
    sent = {}

    def _fake_urlopen(request, timeout=None):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data.decode())
        sent["headers"] = dict(request.headers)
        return _FakeResponse(json.dumps(sent.pop("_payload", _payload())).encode())

    def _install(payload):
        def _urlopen(request, timeout=None):
            sent["url"] = request.full_url
            sent["body"] = json.loads(request.data.decode())
            sent["headers"] = dict(request.headers)
            return _FakeResponse(json.dumps(payload).encode())
        monkeypatch.setattr(llm_direct.urllib.request, "urlopen", _urlopen)
        return sent

    return _install


def _cfg(config_id):
    _, configs = get_suite("track7_model_matrix")
    return next(c for c in configs if c.id == config_id)


def _task():
    tasks, _ = get_suite("track7_model_matrix")
    return tasks[0]


# --------------------------------------------------------------------------- #

def test_usage_split_is_captured_not_collapsed_to_a_total(stub_http, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    stub_http(_payload(1000, 500))
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))

    assert out.prompt_tokens == 1000
    assert out.completion_tokens == 500
    assert out.tokens == 1500


def test_cost_prices_input_and_output_at_their_own_rates(stub_http, monkeypatch):
    """opus-5 is $5/1M in, $25/1M out: 1000 in + 500 out = $0.005 + $0.0125."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    stub_http(_payload(1000, 500))
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))

    assert out.cost_usd == pytest.approx(0.0175)
    assert out.cost_source == SOURCE_CATALOG


def test_cheap_model_costs_proportionally_less_for_identical_usage(stub_http, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    stub_http(_payload(1000, 500))
    cheap = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_deepseek_v4_flash"))
    stub_http(_payload(1000, 500))
    dear = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))

    assert cheap.cost_usd > 0
    # The gap is read from the live catalog snapshot, which moves: on the
    # 2026-10-17 refresh deepseek-v4-flash rose to $0.364/1M completion and
    # the ratio to Opus 5 fell to ~96x. The test's claim is "orders of
    # magnitude apart", not a particular day's quote, so the floor is 50x.
    assert dear.cost_usd > cheap.cost_usd * 50


def test_groq_row_reports_a_proxy_price_not_a_catalog_one(stub_http, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    stub_http(_payload(1000, 500))
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_groq_gptoss120b"))

    assert out.cost_source == SOURCE_PROXY
    assert out.cost_usd > 0


def test_unpriced_model_reports_zero_cost_and_flags_it(stub_http, monkeypatch):
    """groq/compound has no quote anywhere; $0.00 must not read as 'free'."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    stub_http(_payload(1000, 500))
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_groq_compound"))

    assert out.cost_usd == 0.0
    assert out.cost_source == SOURCE_UNPRICED
    assert out.prompt_tokens == 1000    # tokens are still measured


def test_local_model_is_zero_and_tagged_local(stub_http):
    stub_http(_payload(1000, 500))
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_local_ollama"))

    assert out.cost_usd == 0.0
    assert out.cost_source == SOURCE_LOCAL


def test_missing_usage_block_yields_zero_tokens_not_a_guess(stub_http, monkeypatch):
    """Some providers omit usage. We must not estimate one and call it measured.

    QA's F2: the original version asserted only ``cost_usd == 0.0`` and so
    locked in a $0.00 tagged ``catalog`` — a zero that reads as trustworthy.
    The provenance assertion is the part that makes this test the fix.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    stub_http({"id": "g1", "choices": [{"message": {"content": "hi"}}]})
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))

    assert out.prompt_tokens == 0 and out.completion_tokens == 0
    assert out.cost_usd == 0.0
    assert out.cost_source != SOURCE_CATALOG
    assert out.cost_source == SOURCE_USAGE_MISSING
    assert out.answer == "hi"


def test_total_tokens_is_derived_when_only_the_split_is_returned(stub_http, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    stub_http({"id": "g1", "choices": [{"message": {"content": "hi"}}],
               "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))
    assert out.tokens == 10


def test_artifacts_carry_the_price_and_a_reconcilable_generation_id(stub_http, monkeypatch):
    """A cost figure you cannot audit against the provider's bill is a claim."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    stub_http(_payload(1000, 500))
    out = llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))

    assert out.artifacts["generation_id"] == "gen-abc123"
    assert out.artifacts["price"]["prompt_usd_per_1m"] == pytest.approx(5.0)
    assert out.artifacts["price"]["cost_source"] == SOURCE_CATALOG
    assert out.artifacts["usage_raw"]["completion_tokens"] == 500


def test_failed_call_costs_nothing_and_is_not_silently_scored(stub_http, monkeypatch):
    """QA's F8: the 'not silently scored' half needs the RUNNER, not the adapter.

    An adapter-level assertion cannot show what a human ends up reading, which
    is exactly how F1 survived. This drives the full runner -> aggregate ->
    scorecard path and asserts on the published artifact.
    """
    from benchmarks.core.reporting import render_scorecard, results_to_record
    from benchmarks.core.runner import BenchmarkRunner

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = _cfg("m_or_claude_opus5")

    out = llm_direct.LLMDirect().run_mission(_task(), cfg)
    assert out.error and "sut_start_error" in out.error
    assert out.cost_usd == 0.0

    result = BenchmarkRunner().run_task(_task(), dataclasses.replace(cfg, trials=2),
                                        sut=llm_direct.LLMDirect())
    assert result.stats["n_errors"] == 2
    assert result.stats["pass_at_1"] == 0.0

    card = render_scorecard(results_to_record([result], [cfg],
                                              suite="track7_model_matrix"))
    assert "NOT MEASURED" in card, "a 0.00 from pure failure must be flagged"


def test_the_request_actually_targets_the_configured_model(stub_http, monkeypatch):
    """Guards against pricing one model while calling another."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    sent = stub_http(_payload())
    llm_direct.LLMDirect().run_mission(_task(), _cfg("m_or_claude_opus5"))
    assert sent["body"]["model"] == "anthropic/claude-opus-5"
    assert "openrouter.ai" in sent["url"]


# --------------------------------------------------------------------------- #
# Aggregation and reporting carry the new fields through
# --------------------------------------------------------------------------- #

def test_aggregate_summarises_the_token_split_and_cost_provenance():
    from benchmarks.core.metrics import aggregate
    from benchmarks.core.task import MissionOutcome, TrialResult

    trials = [
        TrialResult(task_id="t", track="k", config_id="c", trial_index=i,
                    quality={"score": 1.0},
                    outcome=MissionOutcome(answer="a", cost_usd=0.01, tokens=150,
                                           prompt_tokens=100, completion_tokens=50,
                                           cost_source=SOURCE_CATALOG))
        for i in range(3)
    ]
    stats = aggregate(trials).stats
    assert stats["tuple"]["prompt_tokens"]["mean"] == pytest.approx(100)
    assert stats["tuple"]["completion_tokens"]["mean"] == pytest.approx(50)
    assert stats["total_cost_usd"] == pytest.approx(0.03)
    assert stats["cost_sources"] == [SOURCE_CATALOG]


def test_mixed_provenance_is_reported_not_flattened():
    from benchmarks.core.metrics import aggregate
    from benchmarks.core.task import MissionOutcome, TrialResult

    trials = [
        TrialResult(task_id="t", track="k", config_id="c", trial_index=0,
                    quality={"score": 1.0},
                    outcome=MissionOutcome(answer="a", cost_source=SOURCE_CATALOG)),
        TrialResult(task_id="t", track="k", config_id="c", trial_index=1,
                    quality={"score": 1.0},
                    outcome=MissionOutcome(answer="a", cost_source=SOURCE_UNPRICED)),
    ]
    assert aggregate(trials).stats["cost_sources"] == [SOURCE_CATALOG, SOURCE_UNPRICED]


def test_record_serialisation_includes_the_new_outcome_fields():
    """reporting.asdict(trial) must not drop them."""
    from benchmarks.core.reporting import results_to_record
    from benchmarks.core.metrics import aggregate
    from benchmarks.core.task import MissionOutcome, TrialResult

    trial = TrialResult(task_id="t", track="k", config_id="m_or_claude_opus5",
                        trial_index=0, quality={"score": 1.0},
                        outcome=MissionOutcome(answer="a", cost_usd=0.02,
                                               prompt_tokens=10, completion_tokens=5,
                                               cost_source=SOURCE_CATALOG))
    record = results_to_record([aggregate([trial])], [_cfg("m_or_claude_opus5")],
                               suite="track7_model_matrix")
    serialised = record["results"]["m_or_claude_opus5"]["tasks"]["t"]["trials"][0]
    assert serialised["outcome"]["prompt_tokens"] == 10
    assert serialised["outcome"]["cost_source"] == SOURCE_CATALOG
