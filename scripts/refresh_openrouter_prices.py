#!/usr/bin/env python3
"""Re-fetch the OpenRouter catalog and rewrite the vendored price snapshot.

Run this when a price moves, a model is added, or a slug in a track stops
resolving:

    python scripts/refresh_openrouter_prices.py            # refresh + validate
    python scripts/refresh_openrouter_prices.py --check    # validate only, no write

``--check`` is the CI shape: it exits non-zero when the live catalog no longer
contains every model id the tracks reference, or when a price has moved, without
touching the working tree. That turns "a model was deprecated" from a mid-run
HTTP 400 into a failing build.

No API key is required — the models endpoint is public.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.core.pricing import (  # noqa: E402
    OPENROUTER_MODELS_URL, SNAPSHOT_PATH, catalog, fetch_catalog, load_snapshot,
    parse_catalog,
)

#: Fields worth keeping. The full payload is ~690KB of descriptions we never
#: read; the pruned form is ~100KB and diffs legibly when a price changes.
KEEP_PRICING = ("prompt", "completion", "input_cache_read", "input_cache_write",
                "internal_reasoning", "request", "image", "web_search", "overrides")


def _matrix_pairs():
    """(provider, model) for every row of every track that names a model."""
    from benchmarks.tracks.model_matrix.track import MATRIX
    return [(provider, model) for _, provider, model in MATRIX]


def prune(payload):
    rows = []
    for m in sorted(payload.get("data") or [], key=lambda x: x.get("id") or ""):
        pricing = {k: v for k, v in (m.get("pricing") or {}).items() if k in KEEP_PRICING}
        rows.append({"id": m.get("id"), "name": m.get("name"),
                     "context_length": m.get("context_length"), "pricing": pricing})
    return {
        "source_url": OPENROUTER_MODELS_URL,
        "fetched_at": datetime.date.today().isoformat(),
        "note": ("Pruned snapshot: id/name/context_length/pricing only. Prices are "
                 "USD PER TOKEN as returned by the API; multiply by 1e6 for $/1M. "
                 "Regenerate with scripts/refresh_openrouter_prices.py."),
        "model_count": len(rows),
        "data": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="validate against the live catalog without writing")
    ap.add_argument("--out", default=SNAPSHOT_PATH)
    args = ap.parse_args(argv)

    print(f"Fetching {OPENROUTER_MODELS_URL} ...")
    live_payload = fetch_catalog()
    live = parse_catalog(live_payload)
    print(f"  {len(live)} models with a parseable price.")

    pairs = _matrix_pairs()
    missing = [f"{p}:{m}" for p, m in pairs if p == "openrouter" and m not in live]
    if missing:
        print("\nFAIL — model ids referenced by a track are NOT in the live catalog:",
              file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("\nThe slug is wrong or the model was deprecated. Fix MATRIX before "
              "running the suite; a bad slug becomes an HTTP 400 mid-sweep, after "
              "the earlier configs have already been billed.", file=sys.stderr)
        return 1

    # Price drift against what we have vendored.
    try:
        current = catalog()
    except Exception:  # noqa: BLE001 - first-ever fetch has no snapshot
        current = {}
    drift = []
    for provider, model in pairs:
        old, new = current.get(model), live.get(model)
        if old and new and (abs(old.prompt_usd_per_1m - new.prompt_usd_per_1m) > 1e-9
                            or abs(old.completion_usd_per_1m - new.completion_usd_per_1m) > 1e-9):
            drift.append((model, old, new))
    if drift:
        print("\nPrice drift since the vendored snapshot:")
        for model, old, new in drift:
            print(f"  {model}: {old.prompt_usd_per_1m:.4f}/{old.completion_usd_per_1m:.4f}"
                  f"  ->  {new.prompt_usd_per_1m:.4f}/{new.completion_usd_per_1m:.4f}  $/1M")

    if args.check:
        if drift:
            print("\nFAIL — snapshot is stale. Re-run without --check.", file=sys.stderr)
            return 1
        print("\nOK — every track model resolves and no price has moved.")
        return 0

    with open(args.out, "w") as fh:
        json.dump(prune(live_payload), fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"\nWrote {args.out} ({len(live)} priced models). "
          f"{len(drift)} price(s) changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
