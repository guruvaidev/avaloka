"""Run the Avaloka benchmark from the command line.

    python -m avaloka.benchmark run                      # synthetic tasks
    python -m avaloka.benchmark run --kaggle             # + downloaded Kaggle datasets
    python -m avaloka.benchmark run --family data_science
    python -m avaloka.benchmark run --out ./bench --json scorecard.json
    python -m avaloka.benchmark run --html scorecard.html
    python -m avaloka.benchmark run --edgar              # + one real SEC filing
    python -m avaloka.benchmark serve --port 8765        # look at it in a browser

Hermetic by default: with every provider key unset, ``run`` calls no network
and no model. ``--kaggle`` and ``--edgar`` are the only flags that reach out.
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
                    tag = "[guard] " if c.category == "guard" else ""
                    print(f"        ✗ {tag}{c.name}: {c.detail}")
            if r.error and "unavailable" not in r.error:
                print(f"        ! {r.error.strip().splitlines()[-1]}")
    print("-" * 72)
    for fam, score in card.by_family().items():
        print(f"  {fam:20s} {score:5.2f}")
    print(f"  {'OVERALL':20s} {card.overall_score:5.2f}")
    # Never one undifferentiated "N passed": a guard is green the day it is
    # written, so it is not evidence that an assertion holds.
    print("-" * 72)
    counts = card.by_category()
    for cat, label, verb in (("assertion", "assertions", "passed"), ("guard", "guards", "held")):
        got = f"{counts[cat]['passed']}/{counts[cat]['total']}"
        print(f"  {label:20s} {got:>7s} {verb}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(prog="avaloka.benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run the benchmark suite.")
    run.add_argument("--kaggle", action="store_true", help="Include Kaggle-dataset tasks.")
    run.add_argument("--edgar", action="store_true",
                     help="Include the real SEC EDGAR filing task (network; "
                          "needs AVALOKA_EDGAR_USER_AGENT).")
    run.add_argument("--family", choices=["data_engineering", "data_science",
                                          "format_comprehension", "conversational_scorers"],
                     help="Only run one family.")
    run.add_argument("--out", default=None, help="Artifacts directory (default: a temp dir).")
    run.add_argument("--json", default=None, help="Write the scorecard JSON here.")
    run.add_argument("--html", default=None,
                     help="Write a self-contained HTML scorecard here.")
    run.add_argument("--live", action="store_true",
                     help="Run the conversational family against a real provider "
                          "(local model first, then Groq; never Anthropic).")
    run.add_argument("--quiet", action="store_true", help="Suppress per-task progress.")

    srv = sub.add_parser("serve", help="Serve an HTML scorecard on localhost.")
    srv.add_argument("--scorecard", default="scorecard.json",
                     help="The scorecard JSON (or .html) to serve.")
    srv.add_argument("--port", type=int, default=8765, help="Port (default 8765).")
    args = parser.parse_args()

    if args.cmd == "serve":
        from avaloka.benchmark.serve import serve
        serve(Path(args.scorecard), port=args.port)
        return

    tasks = all_tasks(include_kaggle=args.kaggle, include_edgar=args.edgar)
    if args.family:
        tasks = [t for t in tasks if t.family == args.family]

    base = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="avaloka-bench-"))
    runner = BenchmarkRunner(base, verbose=not args.quiet, edgar=args.edgar)
    print(f"Running {len(tasks)} tasks → {base}")
    card = runner.run(tasks)

    # The conversational family is a scored line in the same scorecard rather
    # than a separate pytest-only pass/fail, which is what made conversational
    # quality invisible here.
    if args.family in (None, "conversational_scorers"):
        from avaloka.benchmark.conversational import conversational_results
        for result in conversational_results(live=args.live):
            card.add(result)

    _print_scorecard(card)

    if args.json:
        Path(args.json).write_text(json.dumps(card.as_dict(), indent=2))
        print(f"\nScorecard written to {args.json}")
    if args.html:
        from avaloka.benchmark.report import write_html
        write_html(card.as_dict(), Path(args.html))
        print(f"HTML scorecard written to {args.html}")

    # Non-zero exit if any non-skip run failed, so CI can gate on it.
    hard_failures = [r for r in card.failures()
                     if not (r.error and "unavailable" in r.error)]
    raise SystemExit(1 if hard_failures else 0)


if __name__ == "__main__":
    main()
