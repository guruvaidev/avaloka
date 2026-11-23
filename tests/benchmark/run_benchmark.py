#!/usr/bin/env python
"""
Standalone benchmark runner / report generator.

Runs the full DE + DS task catalog (and optionally swarm scenarios) against the
configured Avaloka client and emits a JSON + Markdown scorecard. Use this for
regression gating in CI or to compare two clients (reference vs the real CLI).

Examples
--------
    # Score the built-in reference client (offline, synthetic data)
    python -m tests.benchmark.run_benchmark --out report

    # Score the real CLI's Python entrypoint
    AVALOKA_BENCH_CLIENT=avaloka:benchmark_client \
        python -m tests.benchmark.run_benchmark --include-swarm --out report

    # Real Kaggle data + heavy DL
    AVALOKA_BENCH_KAGGLE=1 AVALOKA_BENCH_HEAVY=1 \
        python -m tests.benchmark.run_benchmark --heavy --out report
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow `python tests/benchmark/run_benchmark.py` as well as `-m`.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.benchmark.adapters import get_client  # noqa: E402
from tests.benchmark.catalog import ALL_TASKS, SWARM_SCENARIOS  # noqa: E402
from tests.benchmark.harness import prepare_source, run_task  # noqa: E402
from tests.benchmark.registry import get as get_dataset  # noqa: E402
from tests.benchmark.scoring import Scorecard  # noqa: E402
from tests.benchmark.swarm import MainAvaloka, WorkerAvaloka  # noqa: E402


def _run_swarm(scenario, workdir):
    handle, source, _ = prepare_source(scenario.dataset_key, "local", workdir)
    workers = []
    for i in range(scenario.n_workers):
        fail = set(scenario.inject_failure_kinds) if (i == 0 and scenario.inject_failure_kinds) else set()
        workers.append(WorkerAvaloka(f"avaloka-{i}", fail_kinds=fail, max_retries=2))
    swarm = MainAvaloka(workers)
    spec = get_dataset(scenario.dataset_key)
    cols = [c.name for c in spec.synthetic_columns if c.name != scenario.target]
    plan = swarm.plan(scenario.goal, target=scenario.target,
                      task_kind=scenario.task_kind, feature_columns=cols,
                      n_model_candidates=scenario.n_model_candidates,
                      heavy=scenario.heavy)
    return swarm.run(plan, source.uri)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Avaloka DE/DS benchmark runner")
    ap.add_argument("--out", default="avaloka_benchmark_report",
                    help="output basename (writes .json and .md)")
    ap.add_argument("--heavy", action="store_true", help="include heavy DL tasks")
    ap.add_argument("--include-swarm", action="store_true",
                    help="also run swarm scenarios")
    ap.add_argument("--source", default="local",
                    help="source kind for non-multisource tasks")
    args = ap.parse_args(argv)

    client = get_client()
    card = Scorecard(client=getattr(client, "name", "?"),
                     meta={"heavy": args.heavy})

    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        for task in ALL_TASKS:
            if task.heavy and not args.heavy:
                continue
            for sk in task.source_kinds if task.source_kinds != ["local"] else [args.source]:
                r = run_task(task, client, sk, workdir)
                card.add(r)
                print(f"[{'PASS' if r.passed else 'FAIL'}] {task.task_id}[{sk}] "
                      f"score={r.score:.2f}")

        if args.include_swarm:
            for sc in SWARM_SCENARIOS:
                if sc.heavy and not args.heavy:
                    continue
                res = _run_swarm(sc, workdir)
                status = "PASS" if (res.consensus.get("selected_model_path")
                                    and res.aggregated.get("ready")) else "FAIL"
                card.meta.setdefault("swarm", {})[sc.scenario_id] = {
                    "ok": status == "PASS",
                    "consensus": res.consensus,
                    "endpoint": res.aggregated.get("endpoint"),
                }
                print(f"[{status}] swarm:{sc.scenario_id}")

    json_path = Path(f"{args.out}.json")
    md_path = Path(f"{args.out}.md")
    json_path.write_text(card.to_json())
    md_path.write_text(card.to_markdown())
    print(f"\nSuite score: {card.suite_score:.3f}  pass-rate: {card.pass_rate:.1%}")
    print(f"Wrote {json_path} and {md_path}")
    # Non-zero exit if anything failed -> usable as a CI gate.
    return 0 if card.pass_rate >= 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
