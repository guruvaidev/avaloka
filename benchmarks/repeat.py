"""Run a suite N times and report the spread, not a single sample.

Track 6 scores move between runs on unchanged code: the same six probes
measured 0.82, then 0.73, then 0.70, with individual probes swinging by 0.6.
Any one of those numbers looks like a result and none of them is reproducible.

A single run is a sample from a distribution. Publishing it as *the* score is
the same class of error the scorecard's NOT MEASURED guard exists to prevent --
a number that looks solid and is not. This module makes the distribution the
unit of reporting: mean, min, max and spread per probe, over N runs.

    python -m benchmarks.repeat track6_conversational --runs 5 --config t6_avaloka_reply

Nothing here interprets the numbers. It refuses to print a mean without the
spread beside it, which is the only editorial decision it makes.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from typing import Dict, List


def _probe_scores(path: str, config_id: str) -> Dict[str, float]:
    with open(path) as fh:
        record = json.load(fh)
    block = (record.get("results") or {}).get(config_id) or {}
    out: Dict[str, float] = {}
    for task_id, task in (block.get("tasks") or {}).items():
        stats = task.get("stats") or {}
        # n_errors == n_trials means the config was never measured; a 0.0 from
        # that is an absence of data, and averaging it in would manufacture a
        # low score out of a crash.
        n_err = stats.get("n_errors") or 0
        n_tri = stats.get("n_trials") or 0
        if n_tri and n_err >= n_tri:
            continue
        out[task_id] = float(stats.get("pass_at_1") or 0.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("suite")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--config", default="t6_avaloka_reply",
                        help="config id whose probes are aggregated")
    parser.add_argument("--out", default=None, help="write the aggregate as JSON")
    args = parser.parse_args()

    per_run: List[Dict[str, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(1, args.runs + 1):
            target = os.path.join(tmp, f"run{i}")
            print(f"  run {i}/{args.runs} ...", flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "benchmarks.run", "run", args.suite,
                 "--out", target],
                capture_output=True, text=True)
            path = os.path.join(target, f"{args.suite}.json")
            if result.returncode != 0 or not os.path.exists(path):
                tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
                print(f"    run {i} produced no results: {' | '.join(tail)}")
                continue
            per_run.append(_probe_scores(path, args.config))

    if not per_run:
        print("No run produced results; nothing to aggregate.")
        return 2

    probes = sorted({p for run in per_run for p in run})
    print(f"\n  {args.suite} · config {args.config} · {len(per_run)} run(s)\n")
    print(f"  {'probe':<40}{'mean':>7}{'min':>7}{'max':>7}{'spread':>8}")
    aggregate: Dict[str, Dict[str, float]] = {}
    for probe in probes:
        values = [run[probe] for run in per_run if probe in run]
        if not values:
            continue
        mean = statistics.fmean(values)
        spread = max(values) - min(values)
        aggregate[probe] = {"mean": mean, "min": min(values), "max": max(values),
                            "spread": spread, "n": len(values)}
        flag = "  <-- unstable" if spread >= 0.4 else ""
        print(f"  {probe:<40}{mean:>7.2f}{min(values):>7.2f}{max(values):>7.2f}"
              f"{spread:>8.2f}{flag}")

    means = [statistics.fmean([run[p] for run in per_run if p in run])
             for p in probes if any(p in run for run in per_run)]
    overall = statistics.fmean(means) if means else 0.0
    run_totals = [statistics.fmean(list(run.values())) for run in per_run if run]
    overall_spread = (max(run_totals) - min(run_totals)) if len(run_totals) > 1 else 0.0
    print(f"\n  OVERALL  mean {overall:.2f}   run-to-run spread {overall_spread:.2f}"
          f"   over {len(per_run)} run(s)")
    if overall_spread >= 0.1:
        print("  Report this as a range, not a point. A single run of this suite "
              "is not reproducible.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"suite": args.suite, "config": args.config,
                       "runs": len(per_run), "per_probe": aggregate,
                       "overall_mean": overall,
                       "overall_spread": overall_spread,
                       "run_means": run_totals}, fh, indent=2)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
