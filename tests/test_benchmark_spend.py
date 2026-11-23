"""Spend guardrails: the estimate before, the ceiling during.

The roster reaches $25/1M output tokens. The failure this file exists to
prevent is the one you cannot undo — discovering the cost after the calls have
been made.
"""

from __future__ import annotations

import dataclasses

import pytest

from benchmarks.adapters import config_requires_cloud, get_adapter, requires_cloud
from benchmarks.core.config import RunConfig
from benchmarks.core.registry import get_suite
from benchmarks.core.runner import BenchmarkRunner
from benchmarks.core.spend import (
    DEFAULT_MAX_SPEND_USD, SpendCeilingExceeded, SpendLedger, estimate_suite_cost,
    estimate_tokens,
)
from benchmarks.core.task import MissionOutcome


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #

def test_estimate_covers_every_config_and_bands_the_answer():
    tasks, configs = get_suite("track7_model_matrix")
    est = estimate_suite_cost(tasks, configs)
    assert len(est.configs) == len(configs)
    assert est.calls == len(tasks) * sum(c.trials for c in configs)
    assert 0 < est.low_usd < est.high_usd, "the band must be a band"


def test_high_bound_assumes_the_full_max_tokens_allowance():
    """The top of the band has to be a ceiling, not a forecast, to be useful."""
    tasks, configs = get_suite("track7_model_matrix")
    opus = next(c for c in configs if c.extra["model"] == "anthropic/claude-opus-5")
    est = estimate_suite_cost(tasks, [opus]).configs[0]
    assert est.max_completion_tokens == 400 * len(tasks) * opus.trials


def test_frontier_model_is_estimated_far_above_the_budget_tier():
    """If the estimate cannot separate $25/1M from $0.17/1M it is not doing anything."""
    tasks, configs = get_suite("track7_model_matrix")
    by_id = {c.id: c for c in configs}
    opus = estimate_suite_cost(tasks, [by_id["m_or_claude_opus5"]]).high_usd
    cheap = estimate_suite_cost(tasks, [by_id["m_or_deepseek_v4_flash"]]).high_usd
    assert opus > cheap * 50


def test_unpriced_config_is_excluded_from_totals_and_named():
    """A hole in the data must be visible, not averaged into a comforting zero."""
    tasks, configs = get_suite("track7_model_matrix")
    compound = next(c for c in configs if c.extra["model"] == "groq/compound")
    est = estimate_suite_cost(tasks, [compound])
    assert est.high_usd == 0.0
    assert est.unpriced_configs == ["m_groq_compound"]
    assert "UNPRICED" in est.render()


def test_local_config_costs_nothing_and_is_not_flagged_unpriced():
    tasks, configs = get_suite("track7_model_matrix")
    local = next(c for c in configs if c.extra["provider"] == "ollama")
    est = estimate_suite_cost(tasks, [local])
    assert est.high_usd == 0.0
    assert est.unpriced_configs == []


def test_estimate_scales_linearly_with_trials():
    tasks, configs = get_suite("track7_model_matrix")
    one = next(c for c in configs if c.id == "m_or_claude_opus5")
    a = estimate_suite_cost(tasks, [dataclasses.replace(one, trials=1)]).high_usd
    b = estimate_suite_cost(tasks, [dataclasses.replace(one, trials=4)]).high_usd
    assert b == pytest.approx(a * 4)


def test_estimate_prices_the_real_prompt_not_just_the_mission():
    """The system prompt carries the evidence blob and dominates input tokens."""
    tasks, configs = get_suite("track7_model_matrix")
    one = dataclasses.replace(next(iter(configs)), trials=1)
    est = estimate_suite_cost(tasks[:1], [one]).configs[0]
    assert est.prompt_tokens > estimate_tokens(tasks[0].mission)


def test_render_is_readable_and_ordered_most_expensive_first():
    tasks, configs = get_suite("track7_model_matrix")
    text = estimate_suite_cost(tasks, configs).render()
    assert "Total:" in text and "ceiling" in text
    assert text.index("m_or_claude_opus5") < text.index("m_or_gemma3_4b")


# --------------------------------------------------------------------------- #
# The ceiling
# --------------------------------------------------------------------------- #

def test_ledger_accumulates():
    ledger = SpendLedger(ceiling_usd=1.0)
    ledger.charge(0.25)
    ledger.charge(0.25)
    assert ledger.spent_usd == pytest.approx(0.5)
    assert ledger.remaining_usd == pytest.approx(0.5)
    assert ledger.calls == 2


def test_ledger_raises_the_moment_it_crosses():
    ledger = SpendLedger(ceiling_usd=0.10)
    ledger.charge(0.09)
    with pytest.raises(SpendCeilingExceeded) as exc:
        ledger.charge(0.02)
    assert exc.value.ceiling_usd == 0.10
    assert "--max-spend-usd" in str(exc.value)


def test_ledger_with_no_ceiling_never_raises():
    ledger = SpendLedger(ceiling_usd=None)
    ledger.charge(1_000.0)
    assert ledger.remaining_usd is None


def test_ledger_counts_unpriced_calls_separately():
    """Spend of $0 from 50 unpriced calls is not the same as $0 from 50 free ones."""
    ledger = SpendLedger(ceiling_usd=1.0)
    ledger.charge(0.0, cost_source="unpriced")
    ledger.charge(0.0, cost_source="local_zero")
    assert ledger.unpriced_calls == 1 and ledger.calls == 2


def test_default_ceiling_is_set_rather_than_unlimited():
    assert SpendLedger().ceiling_usd == DEFAULT_MAX_SPEND_USD
    assert DEFAULT_MAX_SPEND_USD > 0


# --------------------------------------------------------------------------- #
# Runner integration
# --------------------------------------------------------------------------- #

class _CostlySUT:
    key = "costly"
    requires_cloud = False

    def run_mission(self, task, config):
        return MissionOutcome(answer="ok", cost_usd=0.04, cost_source="catalog")


def test_runner_aborts_the_sweep_and_keeps_partial_results(monkeypatch):
    """10 tasks x 2 trials x $0.04 = $0.80 wanted; a $0.10 ceiling stops it at $0.12."""
    import benchmarks.core.runner as runner_mod
    tasks, configs = get_suite("track7_model_matrix")
    monkeypatch.setattr(runner_mod, "get_adapter", lambda key: _CostlySUT())

    ledger = SpendLedger(ceiling_usd=0.10)
    runner = BenchmarkRunner(ledger=ledger)
    results = runner.run_suite(tasks, [dataclasses.replace(configs[0], trials=2)])

    assert runner.aborted_reason and "ceiling" in runner.aborted_reason
    assert ledger.spent_usd == pytest.approx(0.12)     # 3 charges, then the raise
    assert len(results) < len(tasks)                   # partial, not complete
    assert ledger.spent_usd < len(tasks) * 2 * 0.04    # the sweep did NOT finish


def test_a_run_inside_the_ceiling_completes_normally(monkeypatch):
    import benchmarks.core.runner as runner_mod
    tasks, configs = get_suite("track7_model_matrix")
    monkeypatch.setattr(runner_mod, "get_adapter", lambda key: _CostlySUT())

    ledger = SpendLedger(ceiling_usd=10.0)
    runner = BenchmarkRunner(ledger=ledger)
    results = runner.run_suite(tasks, [dataclasses.replace(configs[0], trials=2)])

    assert runner.aborted_reason is None
    assert len(results) == len(tasks) == 10
    assert ledger.spent_usd == pytest.approx(0.80)     # 10 tasks x 2 trials x $0.04


def test_runner_without_a_ledger_is_unchanged():
    """Every pre-existing caller constructs BenchmarkRunner(progress=...) only."""
    runner = BenchmarkRunner()
    assert runner.ledger is None


# --------------------------------------------------------------------------- #
# The local-first gate must be per-config, not per-adapter
# --------------------------------------------------------------------------- #

def test_hosted_providers_are_gated_behind_allow_cloud():
    _, configs = get_suite("track7_model_matrix")
    for c in configs:
        if c.extra["provider"] in ("groq", "openrouter"):
            assert config_requires_cloud(c), f"{c.id} would spend money by default"


def test_local_ollama_is_not_gated():
    _, configs = get_suite("track7_model_matrix")
    local = next(c for c in configs if c.extra["provider"] == "ollama")
    assert not config_requires_cloud(local)


def test_flat_adapter_flag_alone_would_have_been_wrong():
    """Documents why config_requires_cloud exists at all."""
    assert requires_cloud("llm_direct") is True     # correct for the paid rows
    _, configs = get_suite("track7_model_matrix")
    local = next(c for c in configs if c.extra["provider"] == "ollama")
    assert config_requires_cloud(local) is False    # and wrong for the free one


def test_adapters_without_the_hook_fall_back_to_the_class_attribute():
    cfg = RunConfig(id="x", sut="local_swarm")
    assert config_requires_cloud(cfg) == requires_cloud("local_swarm")
