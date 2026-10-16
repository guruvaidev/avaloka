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

from avaloka.benchmark.known_defects import KNOWN_DEFECTS, gate
from avaloka.benchmark.runner import BenchmarkRunner
from avaloka.benchmark.spec import Scorecard
from avaloka.benchmark.suite import all_tasks


def _print_scorecard(card: Scorecard) -> None:
    print("\n" + "=" * 72)
    print(f"Avaloka benchmark — {len(card.results)} runs, "
          f"{sum(1 for r in card.results if r.passed)} passed")
    print("=" * 72)
    for r in card.results:
        mark = ("PASS" if r.passed else "SKIP" if r.error and "unavailable" in r.error
                else "KNOWN" if r.task in KNOWN_DEFECTS else "FAIL")
        print(f"  [{mark}] {r.task:38s} {r.source_scheme:14s} "
              f"score={r.score:5.2f}  ({r.duration_s:5.1f}s)")
        if not r.passed:
            for c in r.checks:
                if not c.passed:
                    tag = "[guard] " if c.category == "guard" else ""
                    print(f"        ✗ {tag}{c.name}: {c.detail}")
            if r.task in KNOWN_DEFECTS:
                print(f"        known defect: {KNOWN_DEFECTS[r.task]}")
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


def _print_gate(verdict, *, strict: bool) -> None:
    """Say why the exit code is what it is, so a green run is never mistaken
    for a run in which everything passed."""
    print(f"  known defects failing  {len(verdict.known_failing):>3d}  "
          "(declared in avaloka/benchmark/known_defects.py; still counted as failures above)")
    print(f"  new failures           {len(verdict.new_failures):>3d}")
    for label in verdict.new_failures:
        print(f"        ✗ {label}")
    print(f"  listed but passing     {len(verdict.listed_but_passing):>3d}")
    for label in verdict.listed_but_passing:
        print(f"        ✗ {label} passes: remove it from KNOWN_DEFECTS")
    failed = (verdict.new_failures or verdict.known_failing) if strict else not verdict.ok
    print(f"  exit {1 if failed else 0}{' (--strict: any failure)' if strict else ''}")
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
    run.add_argument("--strict", action="store_true",
                     help="Exit 1 on any failed run, including the known defects "
                          "declared in avaloka/benchmark/known_defects.py.")

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
    verdict = gate(card.results)
    _print_gate(verdict, strict=args.strict)

    if args.json:
        document = card.as_dict()
        document["gate"] = {**verdict.as_dict(), "strict": args.strict}
        Path(args.json).write_text(json.dumps(document, indent=2))
        print(f"\nScorecard written to {args.json}")
    if args.html:
        from avaloka.benchmark.report import write_html
        write_html(card.as_dict(), Path(args.html))
        print(f"HTML scorecard written to {args.html}")

    # Non-zero exit on a failure that is not a declared known defect, or on a
    # declared one that now passes (strict, as xfail is); --strict: on any.
    if args.strict:
        raise SystemExit(1 if verdict.new_failures or verdict.known_failing else 0)
    raise SystemExit(0 if verdict.ok else 1)


if __name__ == "__main__":
    main()
