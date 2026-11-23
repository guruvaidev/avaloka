#!/usr/bin/env python3
"""Refuse a benchmark cell whose operator rate is not the provider's rate.

usage: check_operator_rate.py <model> <input_usd_per_mtok> <output_usd_per_mtok>
                              [--catalog-dir benchmarks/catalog]

The harness charges each trial at the operator rate (AVALOKA_DAB_*_USD_PER_MTOK)
and stops at the spend ceiling, so a wrong rate is a wrong cap. The rate is
checked against the provider that bills the call:

* ``groq/<id>``  -> ``provider_rates.json["groq"][<id>]`` (Groq's own pricing
  page, with its source and date). No fallback to the OpenRouter catalog: the
  same id can carry a different price there.
* ``openrouter/<id>`` or a bare ``<id>`` -> ``openrouter_models.json``
  (USD per token, as the API returns it).

Exit 0 on an exact match, 1 on a mismatch or a missing entry. Stdlib only, so
``run_cell.sh`` can run it with ``python3 -I``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "benchmarks" / "catalog"


def expected_rate(model: str, catalog_dir: Path) -> tuple[float, float, str]:
    """(input, output) USD per 1M tokens for ``model`` and where they came from.

    Raises LookupError when the billing provider has no entry for the model.
    """
    provider, _, rest = model.partition("/")
    if provider == "groq":
        table = json.loads((catalog_dir / "provider_rates.json").read_text())
        entry = (table.get("groq") or {}).get(rest)
        if not entry:
            raise LookupError(f"no Groq rate entry for {rest} in provider_rates.json")
        return (float(entry["input_usd_per_mtok"]), float(entry["output_usd_per_mtok"]),
                f"groq table ({entry.get('source_url')}, {entry.get('checked_at')})")
    model_id = rest if provider == "openrouter" else model
    data = json.loads((catalog_dir / "openrouter_models.json").read_text())
    items = data.get("data") if isinstance(data, dict) and "data" in data else data
    if isinstance(items, dict):
        items = list(items.values())
    entry = next((x for x in items if isinstance(x, dict) and x.get("id") == model_id), None)
    if entry is None:
        raise LookupError(f"model {model_id} not in the OpenRouter catalog")
    pricing = entry["pricing"]
    source = "openrouter catalog"
    if isinstance(data, dict) and data.get("fetched_at"):
        source += f" ({data['fetched_at']})"
    return float(pricing["prompt"]) * 1e6, float(pricing["completion"]) * 1e6, source


def check(model: str, given_in: float, given_out: float, catalog_dir: Path) -> tuple[bool, str]:
    try:
        want_in, want_out, source = expected_rate(model, catalog_dir)
    except (LookupError, OSError, ValueError, KeyError) as exc:
        return False, f"rate check {model}: REFUSED ({exc})"
    ok = abs(want_in - given_in) < 1e-9 and abs(want_out - given_out) < 1e-9
    return ok, (f"rate check {model}: {source} {want_in:g}/{want_out:g} per MTok, "
                f"given {given_in:g}/{given_out:g} -> {'OK' if ok else 'MISMATCH'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("input_usd_per_mtok", type=float)
    parser.add_argument("output_usd_per_mtok", type=float)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args(argv)
    ok, line = check(args.model, args.input_usd_per_mtok, args.output_usd_per_mtok,
                     args.catalog_dir)
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
