"""Run the Avaloka benchmark from the command line.

    python -m avaloka.benchmark run                      # synthetic tasks
    python -m avaloka.benchmark run --kaggle             # + downloaded Kaggle datasets
    python -m avaloka.benchmark run --family data_science
    python -m avaloka.benchmark run --out ./bench --json scorecard.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from avaloka.benchmark.runner import BenchmarkRunner
from avaloka.benchmark.spec import Scorecard
from avaloka.benchmark.suite import all_tasks


def _print_scorecard(card: Scorecard) -> None:
    print("\n" + "=" * 72)
    print(f"Avaloka benchmark — {len(card.results)} runs, "
          f"{sum(1 for r in card.results if r.passed)} passed")
    print("=" * 72)
    for r in card.results:
        mark = "PASS" if r.passed else ("SKIP" if r.error and "unavailable" in r.error else "FAIL")
        print(f"  [{mark}] {r.task:38s} {r.source_scheme:14s} "
              f"score={r.score:5.2f}  ({r.duration_s:5.1f}s)")
        if not r.passed:
            for c in r.checks:
                if not c.passed:
                    print(f"        ✗ {c.name}: {c.detail}")
            if r.error and "unavailable" not in r.error:
                print(f"        ! {r.error.strip().splitlines()[-1]}")
    print("-" * 72)
    for fam, score in card.by_family().items():
        print(f"  {fam:20s} {score:5.2f}")
    print(f"  {'OVERALL':20s} {card.overall_score:5.2f}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(prog="avaloka.benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run the benchmark suite.")
    run.add_argument("--kaggle", action="store_true", help="Include Kaggle-dataset tasks.")
    run.add_argument("--family", choices=["data_engineering", "data_science"],
                     help="Only run one family.")
    run.add_argument("--out", default=None, help="Artifacts directory (default: a temp dir).")
    run.add_argument("--json", default=None, help="Write the scorecard JSON here.")
    run.add_argument("--quiet", action="store_true", help="Suppress per-task progress.")
    args = parser.parse_args()

    tasks = all_tasks(include_kaggle=args.kaggle)
    if args.family:
        tasks = [t for t in tasks if t.family == args.family]

    base = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="avaloka-bench-"))
    runner = BenchmarkRunner(base, verbose=not args.quiet)
    print(f"Running {len(tasks)} tasks → {base}")
    card = runner.run(tasks)
    _print_scorecard(card)

    if args.json:
        Path(args.json).write_text(json.dumps(card.as_dict(), indent=2))
        print(f"\nScorecard written to {args.json}")

    # Non-zero exit if any non-skip run failed, so CI can gate on it.
    hard_failures = [r for r in card.failures()
                     if not (r.error and "unavailable" in r.error)]
    raise SystemExit(1 if hard_failures else 0)


if __name__ == "__main__":
    main()
