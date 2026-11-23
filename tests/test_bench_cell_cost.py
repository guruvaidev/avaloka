"""Per-cell DAB cost from provider numbers, not from the shared key's credit.

Haiku (OpenRouter) is costed from each call's generation id; gpt-oss-120b
(Groq) from logged usage times the Groq rate table. Hermetic: the generation
lookup is a stub and the model logs are written to tmp_path.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dab_cell_cost", ROOT / "scripts" / "bench" / "dab_cell_cost.py")
cost = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cost)

RATES = json.loads((ROOT / "benchmarks" / "catalog" / "provider_rates.json").read_text())


def _results(tmp_path, model, calls_by_task):
    tasks = {}
    for task_id, calls in calls_by_task.items():
        log = tmp_path / f"{task_id.replace(':', '_')}.jsonl"
        log.write_text("".join(json.dumps(c) + "\n" for c in calls))
        tasks[task_id] = {"trials": [{"outcome": {"artifacts": {"model_log": str(log)}}}]}
    return {"results": {"c": {"config": {"llm": model}, "tasks": tasks}}}


def test_groq_cost_is_usage_times_the_groq_rate(tmp_path):
    results = _results(tmp_path, "groq/openai/gpt-oss-120b", {
        "dab:a:1": [{"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 100_000,
                               "cached_prompt_tokens": 400_000}}],
        "dab:a:2": [{"usage": {"prompt_tokens": 2_000, "completion_tokens": 0}}],
    })

    report = cost.cell_cost(results, rates=RATES)

    # 600k x 0.15 + 400k x 0.075 + 100k x 0.60, per 1M tokens
    assert report["trials"]["dab:a:1"] == pytest.approx(0.09 + 0.03 + 0.06)
    assert report["trials"]["dab:a:2"] == pytest.approx(0.0003)
    assert report["uncached_upper_bound_usd"] == pytest.approx(0.15 + 0.06 + 0.0003)
    assert report["cost_source"] == "groq_usage_x_rate"
    assert report["unknown"] == []


def test_openrouter_cost_sums_generation_ids_and_reports_unknowns(tmp_path):
    results = _results(tmp_path, "openrouter/anthropic/claude-haiku-5.5", {
        "dab:b:1": [{"response_id": "gen-1"}, {"response_id": "gen-2"}],
        "dab:b:2": [{"response_id": "gen-missing"}, {"response_id": None}],
    })
    billed = {"gen-1": 0.0125, "gen-2": 0.5}
    seen = []

    def fetch(generation_id):
        seen.append(generation_id)
        return billed.get(generation_id)

    cache = {}
    report = cost.cell_cost(results, rates=RATES, fetch=fetch, cache=cache)

    assert report["trials"] == {"dab:b:1": 0.5125, "dab:b:2": 0.0}
    assert report["total_usd"] == pytest.approx(0.5125)
    assert report["cost_source"] == "openrouter_generation_ids"
    assert len(report["unknown"]) == 2              # never silently zero
    assert cache == billed
    # A second pass reuses the cache and only retries what is still unknown.
    cost.cell_cost(results, rates=RATES, fetch=fetch, cache=cache)
    assert seen.count("gen-1") == 1 and seen.count("gen-missing") == 2
