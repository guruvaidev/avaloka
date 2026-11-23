"""Command-line entrypoint for the Avaloka benchmark suite."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List

from .core.checkpoint import TrialJournal
from .core.config import RunConfig
from .core.registry import get_suite, list_suites
from .core.reporting import (render_scorecard, repository_metadata,
                             results_to_record, write_json, write_text)
from .core.runner import BenchmarkRunner
from .core.spend import (DEFAULT_MAX_SPEND_USD, SpendLedger,
                         estimate_suite_cost, fetch_remaining_budget)


def _cmd_list(_args: argparse.Namespace) -> int:
    print("Available benchmark suites:\n")
    for info in list_suites():
        print(f"  [{info.kind:8}] {info.name}")
        print(f"             {info.description}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        tasks, configs = get_suite(args.suite)
    except Exception as exc:
        print(f"Cannot build suite {args.suite!r}: {exc}", file=sys.stderr)
        return 2

    if args.trials:
        configs = [_with_trials(config, args.trials) for config in configs]
    if args.config:
        wanted_configs = set(args.config)
        configs = [config for config in configs if config.id in wanted_configs]
        if not configs:
            print(f"No configs matched {args.config}", file=sys.stderr)
            return 2
    if args.dataset:
        wanted_datasets = set(args.dataset)
        tasks = [task for task in tasks if task.dataset.id in wanted_datasets]
    if args.task:
        wanted_tasks = set(args.task)
        tasks = [task for task in tasks if task.id in wanted_tasks]
    if args.tag:
        wanted_tags = set(args.tag)
        tasks = [task for task in tasks if wanted_tags.intersection(task.tags)]
    if args.database_type:
        wanted_types = set(args.database_type)
        tasks = [task for task in tasks if wanted_types.intersection(
            set(task.params.get("database_types", ())))]
    if not tasks:
        print("No tasks matched the requested filters.", file=sys.stderr)
        return 2

    # Local-first gate: billable configs are skipped unless explicitly allowed.
    # Evaluated per CONFIG, not per adapter: llm_direct serves one free local
    # provider and two paid hosted ones, and a per-adapter flag cannot be right
    # for both.
    if not args.allow_cloud:
        from .adapters import config_requires_cloud
        kept, skipped = [], []
        for config in configs:
            (skipped if config_requires_cloud(config) else kept).append(config)
        for config in skipped:
            print(f"  SKIP {config.id}: sut={config.sut!r} needs a hosted model or cloud "
                  "runtime; pass --allow-cloud to run it.", file=sys.stderr)
        configs = kept
        if not configs:
            print("All configs require cloud; nothing to run locally. Re-run with "
                  "--allow-cloud once credentials are configured.", file=sys.stderr)
            return 0

    # Cost gate. Price the sweep before spending anything, and refuse to start
    # if the ceiling could not cover it — an estimate you are shown after the
    # run is an invoice.
    estimate = estimate_suite_cost(tasks, configs)
    print("\n" + estimate.render() + "\n")

    if args.estimate_only:
        return 0

    ceiling = None if args.no_spend_limit else float(args.max_spend_usd)

    # Live remaining budget beats any local assumption. The account is capped
    # at $5/day; a sweep planned against a stale number exhausts it and every
    # config after that point fails, which reads as "the expensive models score
    # zero" unless you know the run died.
    needs_openrouter = any(c.extra.get("provider") == "openrouter" for c in configs)
    if needs_openrouter and not args.skip_budget_check:
        budget = fetch_remaining_budget()
        if budget.known:
            print(f"OpenRouter key: ${budget.remaining_usd:.4f} remaining of "
                  f"${budget.limit_usd:.2f} ({budget.reset_period}); "
                  f"used ${budget.usage_usd or 0:.4f}.")
            if estimate.high_usd > budget.remaining_usd:
                print(f"REFUSING TO START: worst-case ${estimate.high_usd:.4f} exceeds "
                      f"the ${budget.remaining_usd:.4f} REMAINING on the OpenRouter key "
                      f"({budget.reset_period} limit). Exhausting it mid-sweep makes "
                      f"every later config fail, which is indistinguishable from a bad "
                      f"score. Narrow with --config/--trials, or wait for the reset.",
                      file=sys.stderr)
                return 3
            # Never plan to spend more than is actually left.
            ceiling = min(ceiling, budget.remaining_usd) if ceiling is not None \
                else budget.remaining_usd
        else:
            print(f"WARNING: could not read the live OpenRouter budget "
                  f"({budget.detail}). Falling back to the local ceiling; the "
                  f"account's daily cap may already be spent.", file=sys.stderr)

    if ceiling is not None and estimate.high_usd > ceiling:
        print(f"REFUSING TO START: worst-case ${estimate.high_usd:.4f} exceeds the "
              f"${ceiling:.2f} ceiling. Raise it with --max-spend-usd, narrow the "
              f"sweep with --config/--trials, or accept the risk with "
              f"--no-spend-limit.", file=sys.stderr)
        return 3

    output_dir = Path(args.out) if args.out else None
    journal = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        journal = TrialJournal(
            output_dir / f"{args.suite}.trials.jsonl",
            suite=args.suite,
            resume=not args.no_resume,
        )

    ledger = SpendLedger(ceiling_usd=ceiling)
    runner = BenchmarkRunner(progress=lambda message: print(message, flush=True),
                             journal=journal, ledger=ledger)
    print(f"Running suite {args.suite!r}: {len(tasks)} tasks x {len(configs)} configs")
    results = runner.run_suite(tasks, configs)

    print(f"\nObserved spend: ${ledger.spent_usd:.4f} over {ledger.calls} calls"
          + (f" ({ledger.blind_calls} unpriced/usage-missing — the ceiling could "
             f"not see these)" if ledger.blind_calls else ""))
    if runner.aborted_reason:
        print(f"RUN ABORTED: {runner.aborted_reason}", file=sys.stderr)
    if not results:
        print("No results produced.", file=sys.stderr)
        return 3

    record = results_to_record(
        results, configs, suite=args.suite,
        meta={**repository_metadata(Path.cwd()),
              "spend": ledger.as_dict(), "estimate": estimate.as_dict(),
              "aborted_reason": runner.aborted_reason},
    )
    scorecard = render_scorecard(record)
    print("\n" + scorecard)

    if output_dir:
        json_path = output_dir / f"{args.suite}.json"
        md_path = output_dir / f"{args.suite}.md"
        write_json(record, json_path)
        write_text(scorecard + "\n", md_path)
        print(f"\nWrote {json_path}, {md_path}, and {journal.path}")
    return 0


def _with_trials(config: RunConfig, trials: int) -> RunConfig:
    return replace(config, trials=trials)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmarks.run",
                                     description="Avaloka benchmark suite runner")
    commands = parser.add_subparsers(dest="cmd", required=True)

    list_parser = commands.add_parser("list", help="list available suites")
    list_parser.set_defaults(func=_cmd_list)

    run_parser = commands.add_parser("run", help="run a suite")
    run_parser.add_argument("suite", help="suite name (see `list`)")
    run_parser.add_argument("--trials", type=int, default=None,
                            help="override trials/config (default: suite's own)")
    run_parser.add_argument("--config", action="append",
                            help="only run these config ids (repeatable)")
    run_parser.add_argument("--dataset", action="append",
                            help="only run tasks using this dataset id (repeatable)")
    run_parser.add_argument("--database-type", action="append",
                            choices=("postgres", "mongo", "sqlite", "duckdb"),
                            help="only run tasks using this database family (repeatable)")
    run_parser.add_argument("--tag", action="append",
                            help="only run tasks with any supplied tag (repeatable)")
    run_parser.add_argument("--task", action="append",
                            help="only run this exact task id (repeatable)")
    run_parser.add_argument("--out", default=None,
                            help="directory for report and resumable trial journal")
    run_parser.add_argument("--no-resume", action="store_true",
                            help="ignore compatible trials already present in the journal")
    run_parser.add_argument("--allow-cloud", action="store_true",
                            help="allow hosted-model/cloud SUTs (default: local only)")
    run_parser.add_argument("--estimate-only", action="store_true",
                            help="price the sweep and exit without calling any provider")
    run_parser.add_argument("--max-spend-usd", type=float, default=DEFAULT_MAX_SPEND_USD,
                            help=(f"hard ceiling on observed spend; the run aborts with "
                                  f"partial results when crossed "
                                  f"(default: {DEFAULT_MAX_SPEND_USD})"))
    run_parser.add_argument("--no-spend-limit", action="store_true",
                            help="disable the spend ceiling entirely (you are on your own)")
    run_parser.add_argument("--skip-budget-check", action="store_true",
                            help=("do not query OpenRouter for the live remaining daily "
                                  "budget (offline use; the local ceiling still applies)"))
    run_parser.set_defaults(func=_cmd_run)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
