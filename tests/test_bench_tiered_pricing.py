"""DAB spend caps and reported cost price OpenRouter's long-prompt tier.

The rc7 Haiku cell reported $1.9816 because every call was priced at Haiku
5.5's flat $0.10/$0.50 per MTok. OpenRouter bills calls with 100k+ prompt tokens
at $0.50/$2.50; re-priced, the cell's own logged usage costs $4.4836, above the
$3.00 ceiling the run enforced against the flat figure. Hermetic: no model call,
no network; the tier comes from the vendored catalog snapshot.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from avaloka.data_agent import (DataMissionRequest, DataQueryMission, MissionLimits,
                                ModelResponse, ModelToolCall, ModelUsage,
                                ToolObservation)
from benchmarks.core.spend import SpendCeilingExceeded, SpendLedger

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "avaloka_dab_agent_under_test", ROOT / "scripts" / "avaloka_dab_agent.py")
agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent)

HAIKU = "openrouter/anthropic/claude-haiku-5.5"
RESULTS = ROOT / "benchmarks" / "results"
RC7_HAIKU = RESULTS / "2.0-rc7" / "dab" / "rc7-openrouter-claude-haiku55.json"
CORRECTION = RESULTS / "cost-corrections" / "2026-10-31-haiku-long-prompt-tier.json"

#: Eight rc7 generations, billed cost read back from OpenRouter's
#: /api/v1/generation on 2026-10-31: (id, prompt, completion, billed USD).
BILLED_GENERATIONS = [
    ("gen-1792455351-0VUvhfNVDHfAQ3JX8bB9", 115767, 781, 0.059836),
    ("gen-1792455357-YdGtm4DxZqERoPEzJ2Qz", 156425, 838, 0.0803075),
    ("gen-1792455364-8aViduMjZR37M1Y1l2kS", 157005, 921, 0.080805),
    ("gen-1792455370-C8rrQeezchLrVBm73yZW", 157522, 727, 0.0805785),
    ("gen-1792455206-7ekZ0SkRgVxGwWfqkcpt", 2913, 278, 0.0004303),
    ("gen-1792455209-6As0fQu1ykvwJXcv9FRr", 3979, 454, 0.0006249),
    ("gen-1792455211-LLjKdahBlyVP3LcwUEmx", 5258, 746, 0.0008988),
    ("gen-1792455215-mnEc65f8D5oPxGXHmrbk", 6304, 665, 0.0009629),
]


def _operator_rates(monkeypatch, input_rate="0.10", output_rate="0.50", **extra):
    """The rc7 run's environment: flat catalog rates, no tier configured."""
    for name in ("AVALOKA_DAB_PRICING_THRESHOLD_TOKENS",
                 "AVALOKA_DAB_INPUT_USD_PER_MTOK_OVER_THRESHOLD",
                 "AVALOKA_DAB_OUTPUT_USD_PER_MTOK_OVER_THRESHOLD",
                 "AVALOKA_DAB_CACHED_INPUT_USD_PER_MTOK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AVALOKA_DAB_PRICING_VERSION", "test")
    monkeypatch.setenv("AVALOKA_DAB_INPUT_USD_PER_MTOK", input_rate)
    monkeypatch.setenv("AVALOKA_DAB_OUTPUT_USD_PER_MTOK", output_rate)
    for name, value in extra.items():
        monkeypatch.setenv(name, value)


def _policy(monkeypatch, model=HAIKU, **env):
    _operator_rates(monkeypatch, **env)
    return agent._with_catalog_tier(agent._pricing(), model)


def _cost(policy, prompt, completion, cached=0):
    return policy.cost(ModelUsage(prompt_tokens=prompt, completion_tokens=completion,
                                  total_tokens=prompt + completion,
                                  cached_prompt_tokens=cached))


# --- the tier boundary -------------------------------------------------------

def test_haiku_at_99999_prompt_tokens_prices_flat(monkeypatch):
    policy = _policy(monkeypatch)
    assert _cost(policy, 99_999, 1_000) == pytest.approx(
        (99_999 * 0.10 + 1_000 * 0.50) / 1e6)


@pytest.mark.parametrize("prompt", [100_000, 100_001, 157_522])
def test_haiku_at_100000_prompt_tokens_and_above_prices_tiered(monkeypatch, prompt):
    policy = _policy(monkeypatch)
    assert _cost(policy, prompt, 1_000) == pytest.approx(
        (prompt * 0.50 + 1_000 * 2.50) / 1e6)


def test_gpt6_luna_tier_starts_at_272k_prompt_tokens(monkeypatch):
    policy = _policy(monkeypatch, model="openrouter/openai/gpt-6-luna")
    assert _cost(policy, 271_999, 1_000) == pytest.approx(
        (271_999 * 0.10 + 1_000 * 0.50) / 1e6)
    assert _cost(policy, 272_000, 1_000) == pytest.approx(
        (272_000 * 0.20 + 1_000 * 0.75) / 1e6)


def test_cached_prompt_tokens_above_the_threshold_use_the_tier_cache_rate(monkeypatch):
    policy = _policy(monkeypatch, AVALOKA_DAB_CACHED_INPUT_USD_PER_MTOK="0.01")
    # 150k prompt, 50k of it cached: 100k x 0.50 + 50k x 0.05 + 1k x 2.50
    assert _cost(policy, 150_000, 1_000, cached=50_000) == pytest.approx(
        (100_000 * 0.50 + 50_000 * 0.05 + 1_000 * 2.50) / 1e6)


def test_an_explicit_operator_tier_is_kept(monkeypatch):
    policy = _policy(monkeypatch,
                     AVALOKA_DAB_PRICING_THRESHOLD_TOKENS="49999",
                     AVALOKA_DAB_INPUT_USD_PER_MTOK_OVER_THRESHOLD="1.0",
                     AVALOKA_DAB_OUTPUT_USD_PER_MTOK_OVER_THRESHOLD="3.0")
    assert _cost(policy, 60_000, 1_000) == pytest.approx((60_000 * 1.0 + 1_000 * 3.0) / 1e6)


def test_a_groq_model_is_not_priced_from_the_openrouter_catalog(monkeypatch):
    # Groq bills gpt-oss-120b flat; OpenRouter's sheet is not Groq's.
    policy = _policy(monkeypatch, model="groq/openai/gpt-oss-120b",
                     input_rate="0.15", output_rate="0.60")
    assert policy.threshold_tokens is None
    assert _cost(policy, 150_000, 1_000) == pytest.approx((150_000 * 0.15 + 600) / 1e6)


def test_the_tiered_price_matches_what_openrouter_billed(monkeypatch):
    policy = _policy(monkeypatch)
    for generation_id, prompt, completion, billed in BILLED_GENERATIONS:
        assert _cost(policy, prompt, completion) == pytest.approx(billed, abs=5e-7), \
            generation_id


# --- the caps trip on the tiered figure --------------------------------------

class _Provider:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, messages, tools, *, timeout_s):
        return self.responses.pop(0)


class _Tools:
    def specs(self):
        return [{"type": "function",
                 "function": {"name": "query_db", "parameters": {"type": "object"}}}]

    def execute(self, name, arguments, *, call_id):
        return ToolObservation(True, "result_key=r; result=[1]", result_key="r",
                               rows_read=1, bytes_read=8)

    def close(self):
        pass


def _long_prompt_trial(policy, max_cost_usd):
    """Two queries and an answer, each call 120k prompt / 500 completion tokens.

    Flat: $0.01225 a call, $0.03675 for the trial. Tiered: $0.06125 a call,
    $0.18375 for the trial when no cap stops it.
    """
    usage = ModelUsage(prompt_tokens=120_000, completion_tokens=500, total_tokens=120_500)
    responses = [
        ModelResponse(f"r{i}", None, [ModelToolCall(f"q{i}", "query_db", {"query": "x"})],
                      usage)
        for i in (1, 2)
    ] + [ModelResponse("r3", None, [ModelToolCall(
        "a", "return_answer", {"answer": "1", "evidence_refs": ["q1", "q2"]})], usage)]
    return DataQueryMission(_Provider(responses), _Tools(), policy).run(DataMissionRequest(
        task_id="dab:fixture:1", question="?", database_description="fixture",
        model=HAIKU, limits=MissionLimits(max_cost_usd=max_cost_usd)))


def test_the_per_trial_cap_trips_on_the_tiered_cost(monkeypatch):
    _operator_rates(monkeypatch)
    flat = _long_prompt_trial(agent._pricing(), max_cost_usd=0.05)
    assert flat.termination_reason != "budget_exceeded"   # control: under the cap flat
    assert flat.cost_usd == pytest.approx(0.03675)

    tiered = _long_prompt_trial(_policy(monkeypatch), max_cost_usd=0.05)
    assert tiered.termination_reason == "budget_exceeded"
    assert tiered.cost_usd == pytest.approx(0.06125)


def test_the_run_ceiling_trips_on_the_tiered_cost(monkeypatch):
    # Eight such trials under a $0.40 run ceiling: $0.294 flat; tiered, each
    # trial is $0.18375 and the third takes the run to $0.55125.
    _operator_rates(monkeypatch)
    flat_ledger = SpendLedger(ceiling_usd=0.40)
    for _ in range(8):
        flat_ledger.charge(_long_prompt_trial(agent._pricing(), 2.0).cost_usd,
                           cost_source="operator_rate")
    assert flat_ledger.spent_usd == pytest.approx(0.294)

    tiered_ledger = SpendLedger(ceiling_usd=0.40)
    with pytest.raises(SpendCeilingExceeded):
        for _ in range(8):
            tiered_ledger.charge(_long_prompt_trial(_policy(monkeypatch), 2.0).cost_usd,
                                 cost_source="operator_rate")
    assert tiered_ledger.calls == 3                       # stopped at the third trial
    assert tiered_ledger.spent_usd == pytest.approx(0.55125)


# --- regression: the rc7 Haiku cell, re-priced from its own logged usage ------

def _trials(record):
    results = json.loads(record.read_text())
    for config in results["results"].values():
        for task_id, task in config["tasks"].items():
            for trial in task["trials"]:
                yield task_id, trial["outcome"]


def _model_events(outcome):
    for event in outcome.get("trajectory") or []:
        if event.get("kind") == "model" and event.get("usage"):
            yield event


def _reprice(record, policy):
    """The record's own logged per-call usage, priced by ``policy``."""
    cell = {"reported": 0.0, "total": 0.0, "calls": 0, "tier_calls": 0, "per_trial": {}}
    for task_id, outcome in _trials(record):
        cell["reported"] += outcome["cost_usd"]
        trial_cost = 0.0
        for event in _model_events(outcome):
            usage = event["usage"]
            cell["calls"] += 1
            cell["tier_calls"] += usage["prompt_tokens"] > 99_999
            trial_cost += _cost(policy, usage["prompt_tokens"], usage["completion_tokens"],
                                usage.get("cached_prompt_tokens", 0))
        cell["per_trial"][task_id] = trial_cost
        cell["total"] += trial_cost
    return cell


def test_rc7_haiku_cell_reprices_to_the_tiered_total(monkeypatch):
    cell = _reprice(RC7_HAIKU, _policy(monkeypatch))

    assert (cell["calls"], cell["tier_calls"]) == (452, 43)
    assert cell["reported"] == pytest.approx(1.9816, abs=5e-5)   # the flat figure rc7 published
    assert cell["total"] == pytest.approx(4.4836, abs=5e-5)
    assert cell["total"] > 3.00                                   # over that run's ceiling
    # No trial reached the $2 per-trial cap at the real price either.
    per_trial = cell["per_trial"]
    worst = max(per_trial, key=per_trial.get)
    assert worst == "dab:DEPS_DEV_V1:2"
    assert per_trial[worst] == pytest.approx(0.9018, abs=5e-5) and per_trial[worst] < 2.0


def test_the_billed_generations_are_the_rc7_records_own_calls():
    logged = {event["response_id"]: event["usage"]
              for _, outcome in _trials(RC7_HAIKU) for event in _model_events(outcome)}
    for generation_id, prompt, completion, _ in BILLED_GENERATIONS:
        usage = logged[generation_id]
        assert (usage["prompt_tokens"], usage["completion_tokens"]) == (prompt, completion)


def test_the_cost_correction_sidecar_matches_the_records(monkeypatch):
    """The published correction is recomputed, not copied: drift fails here."""
    sidecar = json.loads(CORRECTION.read_text())
    policy = _policy(monkeypatch)
    for path in sidecar["applies_to"]:
        record = ROOT / path
        claimed = sidecar["cells"][record.stem]
        cell = _reprice(record, policy)
        assert (cell["calls"], cell["tier_calls"]) == (claimed["calls"], claimed["calls_at_tier"])
        assert cell["reported"] == pytest.approx(claimed["reported_usd"], abs=1e-6)
        assert cell["total"] == pytest.approx(claimed["tiered_usd"], abs=1e-6)
        assert (cell["total"] > claimed["run_ceiling_usd"]) == claimed["over_run_ceiling_at_tiered"]
        worst = max(cell["per_trial"], key=cell["per_trial"].get)
        assert worst == claimed["max_trial"]["task_id"]
        assert cell["per_trial"][worst] == pytest.approx(claimed["max_trial"]["tiered_usd"],
                                                         abs=1e-6)
        assert cell["per_trial"][worst] < claimed["per_trial_cap_usd"]
