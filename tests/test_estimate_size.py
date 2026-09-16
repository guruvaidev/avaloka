"""
Test: Dataset Size Estimation

Tests _estimate_dataset_size in isolation for any local file or cloud URI.
Prints the estimated row count, file size, sizing tier, and how long the estimate took.

Usage:
    python tests/test_estimate_size.py --path data/myfile.csv
    python tests/test_estimate_size.py --path gs://my-bucket/events.csv
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.sampling_agent_daft import (
    _estimate_dataset_size,
    calculate_adaptive_sample_sizes,
)
from app.agents.sampling_async import (
    _determine_tier,
    _should_run_background,
    TIER_TINY, TIER_SMALL, TIER_LARGE, TIER_HUGE, TIER_MASSIVE,
)


TIER_DESCRIPTIONS = {
    TIER_TINY:    "< 1k rows — full profile runs instantly",
    TIER_SMALL:   "1k–100k rows — full profile runs fast",
    TIER_LARGE:   "100k–1M rows — quick preview + background job",
    TIER_HUGE:    "1M–10M rows — quick preview + background job",
    TIER_MASSIVE: "> 10M rows — quick preview + long background job",
}

TIER_COLORS = {
    "TINY":    "🟢",
    "SMALL":   "🟢",
    "LARGE":   "🟡",
    "HUGE":    "🟠",
    "MASSIVE": "🔴",
}


def fmt_rows(n: Optional[int]) -> str:
    if n is None:
        return "unknown"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def fmt_time(s: float) -> str:
    if s < 1:
        return f"{s*1000:.1f}ms"
    return f"{s:.3f}s"


def estimate_one(path: str, source_type: str = "csv") -> dict:
    """Run size estimation and return a dict of all computed values."""
    t0 = time.perf_counter()
    estimated_rows, size_mb = _estimate_dataset_size(path, source_type)
    elapsed = time.perf_counter() - t0

    is_multi = any(c in path for c in ("*", "?")) or path.endswith("/")
    tier = _determine_tier(estimated_rows, is_multi)
    needs_background = _should_run_background(tier)

    display_size, portfolio_size = calculate_adaptive_sample_sizes(estimated_rows)

    return {
        "path":                   path,
        "source_type":            source_type,
        "estimated_rows":         estimated_rows,
        "size_mb":                size_mb,
        "tier":                   tier,
        "needs_background":       needs_background,
        "display_rows":           display_size,
        "portfolio_rows":         portfolio_size,
        "estimation_elapsed_s":   elapsed,
    }


def print_result(r: dict) -> None:
    W = 72
    icon = TIER_COLORS.get(r["tier"], "⚫")
    tier = r["tier"]
    desc = TIER_DESCRIPTIONS.get(tier, "")

    print(f"\n{'═'*W}")
    print(f"  {icon}  {tier}  —  {desc}")
    print(f"{'═'*W}")
    print(f"  Path        : {r['path']}")
    print(f"  Source type : {r['source_type']}")
    print()
    print(f"  Estimated rows : {fmt_rows(r['estimated_rows'])}  ({r['estimated_rows']:,})")
    if r["size_mb"]:
        print(f"  File size      : {r['size_mb']:.2f} MB")
    else:
        print(f"  File size      : unknown (cloud)")
    print()

    if r["needs_background"]:
        print(f"  ⚡ Quick preview : {r['display_rows']:,} rows  ← UI sees this immediately")
        print(f"  🔄 Background job: {r['portfolio_rows']:,} rows per portfolio sample")
    else:
        print(f"  ⚡ Full profile  : {r['display_rows']:,} rows  ← no background job needed")

    print()
    print(f"  ⏱  Estimation time : {fmt_time(r['estimation_elapsed_s'])}")
    print(f"{'─'*W}")


def main():
    parser = argparse.ArgumentParser(
        description="Show size estimate, tier, and sample sizes for a file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_estimate_size.py --path data/events.csv
  python tests/test_estimate_size.py --path gs://bucket/big.parquet --source-type parquet
  python tests/test_estimate_size.py --path data/a.csv --path gs://bucket/b.csv
""",
    )
    parser.add_argument(
        "--path", "-p",
        action="append",
        required=True,
        metavar="PATH",
        help="Local file path or cloud URI (gs://, s3://). Repeat for multiple files.",
    )
    parser.add_argument(
        "--source-type", "-t",
        default="csv",
        choices=["csv", "parquet", "json", "avro"],
        help="File format (default: csv)",
    )

    args = parser.parse_args()
    paths = args.path

    is_cloud = lambda p: any(p.startswith(x) for x in ("gs://", "s3://", "az://", "http://", "https://"))

    import os
    for p in paths:
        if not is_cloud(p) and not os.path.exists(p):
            print(f"❌  File not found: {p}")
            sys.exit(1)

    results = []
    for p in paths:
        print(f"\n⏳  Estimating size for: {p}")
        try:
            r = estimate_one(p, args.source_type)
            results.append(r)
            print_result(r)
        except Exception as exc:
            print(f"❌  Failed: {exc}")
            import traceback; traceback.print_exc()

    if len(results) > 1:
        print(f"\n{'═'*72}")
        print(f"  SUMMARY — {len(results)} paths")
        print(f"{'═'*72}")
        print(f"  {'Path':<45} {'Tier':<8} {'Est. rows':>12}  {'Est. time':>10}")
        print(f"  {'─'*45} {'─'*8} {'─'*12}  {'─'*10}")
        for r in results:
            name = Path(r["path"]).name[:44]
            print(
                f"  {name:<45} "
                f"{TIER_COLORS.get(r['tier'],'')} {r['tier']:<6} "
                f"{fmt_rows(r['estimated_rows']):>12}  "
                f"{fmt_time(r['estimation_elapsed_s']):>10}"
            )
        print()


if __name__ == "__main__":
    main()
