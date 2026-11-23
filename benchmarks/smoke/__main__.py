"""Run the smoke suite. Intended as a CI gate and a pre-pilot check.

    python -m benchmarks.smoke              # human-readable
    python -m benchmarks.smoke --json       # machine-readable
    python -m benchmarks.smoke --strict     # warnings also fail

Exit codes: 0 safe to run · 1 a CRITICAL check failed · 2 the suite itself
could not run. The third is deliberately distinct: "the checker is broken" and
"the system is broken" call for different responses, and conflating them is how
a red gate gets ignored.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Avaloka pre-flight smoke checks")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--strict", action="store_true", help="warnings fail too")
    parser.add_argument("--surface", action="append",
                        choices=["harness", "prompts", "loops", "graph"],
                        help="limit to one surface (repeatable)")
    args = parser.parse_args(argv)

    try:
        from benchmarks.smoke import run_all
        report = run_all()
    except Exception as exc:  # noqa: BLE001
        print(f"smoke suite could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    results = report.results
    if args.surface:
        results = [r for r in results if r.surface in set(args.surface)]

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        tty = sys.stdout.isatty()
        G, R, Y, D, B, O = (("\033[32m", "\033[31m", "\033[33m", "\033[2m",
                             "\033[1m", "\033[0m") if tty else ("",) * 6)
        width = max((len(r.check) for r in results), default=10)
        surface = None
        for r in results:
            if r.surface != surface:
                surface = r.surface
                print(f"\n{B}{surface}{O}")
            if r.passed:
                mark = f"{G}pass{O}"
            elif r.severity.value == "critical":
                mark = f"{R}FAIL{O}"
            else:
                mark = f"{Y}warn{O}"
            print(f"  {mark}  {r.check:<{width}}  {D}{r.detail}{O}")
        print(f"\n{B}{report.summary()}{O}")

    failed_warnings = [r for r in results if not r.passed and r.severity.value != "critical"]
    blocking = [r for r in results if not r.passed and r.severity.value == "critical"]
    if blocking or (args.strict and failed_warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
