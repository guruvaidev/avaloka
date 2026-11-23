#!/usr/bin/env python3
"""What one DAB cell cost, from the provider's own numbers.

usage: dab_cell_cost.py <results.json> [--cache <generation-cache.json>]

The OpenRouter key is shared with other jobs, so a credit delta is not one
cell's cost. Instead:

* ``openrouter/`` cells: every model call's ``response_id`` (logged in each
  trial's ``model_calls.jsonl``) is an OpenRouter generation id; its billed
  ``total_cost`` comes from ``GET /api/v1/generation?id=``. The key is read from
  ``OPENROUTER_API_KEY`` and sent only in the Authorization header.
* ``groq/`` cells: Groq has no per-generation cost lookup, so the cost is the
  logged usage times the Groq entry in ``benchmarks/catalog/provider_rates.json``
  (cached prompt tokens at the cached rate; ``uncached_upper_bound_usd`` prices
  them at the full input rate). Calls the provider refused (HTTP 400) log no
  usage and are not counted.

Prints one JSON object: per-trial and total cost, and anything left unknown.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

RATES = Path(__file__).resolve().parents[2] / "benchmarks" / "catalog" / "provider_rates.json"
GENERATION_URL = "https://openrouter.ai/api/v1/generation?id="


def _trials(results: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    for config in (results.get("results") or {}).values():
        model = str((config.get("config") or {}).get("llm") or "")
        for task_id, task in (config.get("tasks") or {}).items():
            for trial in task.get("trials") or []:
                yield model, task_id, trial


def _model_calls(trial: dict[str, Any]) -> list[dict[str, Any]]:
    log = ((trial.get("outcome") or {}).get("artifacts") or {}).get("model_log")
    if not log or not Path(log).is_file():
        return []
    return [json.loads(line) for line in Path(log).read_text().splitlines() if line.strip()]


def groq_call_cost(usage: dict[str, Any], rate: dict[str, Any]) -> tuple[float, float]:
    """(cost, upper bound with cached tokens at the full input rate), USD."""
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = min(int(usage.get("cached_prompt_tokens") or 0), prompt)
    completion = int(usage.get("completion_tokens") or 0)
    in_rate = float(rate["input_usd_per_mtok"])
    cached_rate = float(rate.get("cached_input_usd_per_mtok", in_rate))
    out_rate = float(rate["output_usd_per_mtok"])
    cost = ((prompt - cached) * in_rate + cached * cached_rate + completion * out_rate) / 1e6
    upper = (prompt * in_rate + completion * out_rate) / 1e6
    return cost, upper


def openrouter_generation_cost(generation_id: str, key: str) -> float | None:
    request = urllib.request.Request(GENERATION_URL + generation_id,
                                     headers={"Authorization": f"Bearer {key}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response).get("data") or {}
            value = data.get("total_cost")
            return float(value) if value is not None else None
        except Exception:  # noqa: BLE001 - generation stats lag; retried, then unknown
            time.sleep(2 * (attempt + 1))
    return None


def cell_cost(results: dict[str, Any], *, rates: dict[str, Any],
              fetch: Callable[[str], float | None] | None = None,
              cache: dict[str, float] | None = None) -> dict[str, Any]:
    cache = {} if cache is None else cache
    per_trial: dict[str, float] = {}
    total = upper = 0.0
    unknown: list[str] = []
    model_seen = ""
    for model, task_id, trial in _trials(results):
        model_seen = model
        provider, _, model_id = model.partition("/")
        trial_cost = 0.0
        for call in _model_calls(trial):
            if provider == "groq":
                cost, bound = groq_call_cost(call.get("usage") or {},
                                             rates["groq"][model_id])
                trial_cost += cost
                upper += bound
            elif provider == "openrouter":
                generation_id = call.get("response_id")
                if not generation_id:
                    unknown.append(f"{task_id}: call without a generation id")
                    continue
                if generation_id not in cache:
                    value = fetch(generation_id) if fetch else None
                    if value is None:
                        unknown.append(f"{task_id}: {generation_id}")
                        continue
                    cache[generation_id] = value
                trial_cost += cache[generation_id]
                upper += cache[generation_id]
            else:
                raise ValueError(f"no cost source for provider of {model}")
        per_trial[task_id] = round(trial_cost, 6)
        total += trial_cost
    return {
        "model": model_seen,
        "cost_source": ("openrouter_generation_ids" if model_seen.startswith("openrouter/")
                        else "groq_usage_x_rate"),
        "total_usd": round(total, 6),
        "uncached_upper_bound_usd": round(upper, 6),
        "trials": per_trial,
        "unknown": unknown,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--cache", type=Path, default=None,
                        help="JSON map of generation id -> USD, read and updated")
    parser.add_argument("--rates", type=Path, default=RATES)
    args = parser.parse_args(argv)
    results = json.loads(args.results.read_text())
    cache = json.loads(args.cache.read_text()) if args.cache and args.cache.is_file() else {}
    key = os.environ.get("OPENROUTER_API_KEY", "")
    fetch = (lambda gid: openrouter_generation_cost(gid, key)) if key else None
    report = cell_cost(results, rates=json.loads(args.rates.read_text()),
                       fetch=fetch, cache=cache)
    if args.cache:
        args.cache.write_text(json.dumps(cache, indent=0, sort_keys=True))
    json.dump(report, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0 if not report["unknown"] else 3


if __name__ == "__main__":
    sys.exit(main())
