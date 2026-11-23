"""Trial execution, budget enforcement, evaluation, spend ceiling, and durable resume.

The runner is deliberately dumb about *what* is being measured — it only knows how
to (a) invoke a system-under-test adapter once per trial, (b) time it, (c) hand the
answer to the task's evaluator for quality, and (d) aggregate. All benchmark-specific
logic lives in the task's ``mission``/``evaluator`` and the adapter.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import replace
from typing import Any, Callable, Iterable, List, Optional

from ..adapters import get_adapter
from ..adapters.base import SystemUnderTest
from .checkpoint import TrialJournal
from .config import Budget, RunConfig
from .metrics import aggregate
from .spend import SpendCeilingExceeded, SpendLedger
from .task import (BenchmarkTask, FailureKind, MissionOutcome, TaskResult,
                   TrialResult, classify_failure)

ProgressHook = Callable[[str], None]


class BenchmarkRunner:
    """Runs trials, resumes them from a journal, and refuses to spend past a ceiling.

    ``ledger`` and ``journal`` are both optional so every existing caller keeps
    working unchanged. With a ledger, each freshly executed trial is charged to
    it and a :class:`~benchmarks.core.spend.SpendCeilingExceeded` stops the sweep
    with whatever results are already in hand. Halting mid-suite is the point — a
    partial scorecard you can read beats a complete one you cannot afford.
    Trials replayed from the journal were paid for by an earlier run and are not
    charged again.
    """

    #: Set when a sweep stopped early. ``None`` on a complete run.
    aborted_reason: Optional[str] = None

    def __init__(self, progress: Optional[ProgressHook] = None,
                 journal: Optional[TrialJournal] = None,
                 ledger: Optional[SpendLedger] = None):
        self._progress = progress or (lambda _msg: None)
        self._journal = journal
        self.ledger = ledger

    def _log(self, msg: str) -> None:
        self._progress(msg)

    def run_task(self, task: BenchmarkTask, config: RunConfig,
                 sut: Optional[SystemUnderTest] = None) -> TaskResult:
        sut = sut or get_adapter(config.sut)
        trials: List[TrialResult] = []
        for index in range(config.trials):
            # _trial is an INT. Adapters use it as a seed salt and do arithmetic on it
            # directly (benchmarks/adapters/baselines.py::_rng), so passing str(i)
            # raised TypeError inside every baseline adapter, which the broad except
            # in _run_one turned into a legitimate-looking 0.0 score.
            trial_config = replace(config, extra={**config.extra, "_trial": index})
            cached = self._journal.evaluated(task, trial_config, index) \
                if self._journal else None
            if cached is not None:
                trials.append(cached)
                self._log(f"  [{config.id}] {task.id} trial {index + 1}/{config.trials} "
                          "RESUME evaluated")
                continue

            outcome = self._journal.execution(task, trial_config, index) \
                if self._journal else None
            fresh = outcome is None
            if fresh:
                outcome = self._run_one(task, trial_config, sut, index)
                self._enforce_budget(outcome, config.budget)
                if self._journal:
                    self._journal.record_execution(task, trial_config, index, outcome)
            else:
                self._log(f"  [{config.id}] {task.id} trial {index + 1}/{config.trials} "
                          "RESUME execution; evaluating")

            quality = self._evaluate(task, outcome)
            trial = TrialResult(
                task_id=task.id, track=task.track, config_id=config.id,
                trial_index=index, quality=quality, outcome=outcome,
            )
            trials.append(trial)
            if self._journal:
                self._journal.record_evaluated(task, trial_config, trial)

            cost = "unknown" if outcome.cost_usd is None else f"${outcome.cost_usd:.4f}"
            score = quality.get("score", quality.get("pass", 0.0))
            self._log(f"  [{config.id}] {task.id} trial {index + 1}/{config.trials} "
                      f"{'ERR' if outcome.error else 'ok'} q={score:.3f} "
                      f"{cost} {outcome.latency_s:.2f}s")
            if self.ledger is not None and fresh:
                # Charged AFTER the trial is recorded (and journaled), so the
                # trial that trips the ceiling is still in the partial results.
                self.ledger.charge(outcome.cost_usd, cost_source=outcome.cost_source)
        return aggregate(trials)

    def _evaluate(self, task: BenchmarkTask, outcome: MissionOutcome) -> dict[str, float]:
        # A budget violation is a *cost* verdict, not an answer verdict: the mission
        # still produced an answer and we must grade it, otherwise the published
        # (Quality, Cost, Latency, Data Scanned) 4-tuple cannot express "correct but
        # too expensive" and tightening a budget looks like a quality regression.
        # Any other error means there is no trustworthy answer to grade.
        if outcome.error and outcome.error_kind is not FailureKind.BUDGET:
            outcome.error_kind = outcome.error_kind or classify_failure(outcome.error)
            return {"pass": 0.0, "score": 0.0}
        try:
            raw = task.evaluate(outcome.answer)
            if not isinstance(raw, dict):
                raise TypeError(f"evaluator returned {type(raw).__name__}, expected dict")
            quality: dict[str, float] = {}
            details: dict[str, Any] = {}
            for key, value in raw.items():
                if isinstance(value, bool):
                    quality[key] = float(value)
                elif isinstance(value, (int, float)):
                    quality[key] = float(value)
                else:
                    details[key] = value
            if details:
                outcome.artifacts.setdefault("evaluation", {}).update(details)
            return quality
        except Exception:
            trace = "evaluator_error:\n" + traceback.format_exc()
            # Preserve any budget verdict already recorded on the outcome.
            outcome.error = f"{outcome.error}\n{trace}" if outcome.error else trace
            outcome.error_kind = FailureKind.EVALUATOR
            return {"pass": 0.0, "score": 0.0}

    def _run_one(self, task: BenchmarkTask, config: RunConfig,
                 sut: SystemUnderTest, trial_index: int) -> MissionOutcome:
        start = time.perf_counter()
        try:
            outcome = sut.run_mission(task, config)
        except Exception:
            return MissionOutcome(
                answer=None,
                latency_s=time.perf_counter() - start,
                error="sut_start_error:\n" + traceback.format_exc(),
                error_kind=FailureKind.SUT_START,
            )
        if not outcome.latency_s:
            outcome.latency_s = time.perf_counter() - start
        if outcome.error and outcome.error_kind is None:
            outcome.error_kind = classify_failure(outcome.error)
        return outcome

    @staticmethod
    def _enforce_budget(outcome: MissionOutcome, budget: Budget) -> None:
        """Apply runner-level ceilings and persist limits, actuals, and violations."""
        violations: list[str] = []
        if (budget.max_cost_usd is not None and outcome.cost_usd is not None
                and outcome.cost_usd > budget.max_cost_usd):
            violations.append(
                f"cost_usd={outcome.cost_usd:.6f}>{budget.max_cost_usd:.6f}"
            )
        if (budget.max_latency_s is not None
                and outcome.latency_s > budget.max_latency_s):
            violations.append(
                f"latency_s={outcome.latency_s:.6f}>{budget.max_latency_s:.6f}"
            )
        if budget.max_tokens is not None and outcome.tokens > budget.max_tokens:
            violations.append(f"tokens={outcome.tokens}>{budget.max_tokens}")
        if (budget.max_data_scanned_ratio is not None
                and outcome.data_scanned_ratio is not None
                and outcome.data_scanned_ratio > budget.max_data_scanned_ratio):
            violations.append(
                "data_scanned_ratio="
                f"{outcome.data_scanned_ratio:.6f}>{budget.max_data_scanned_ratio:.6f}"
            )

        outcome.artifacts["budget"] = {
            "limits": {
                "max_cost_usd": budget.max_cost_usd,
                "max_latency_s": budget.max_latency_s,
                "max_tokens": budget.max_tokens,
                "max_data_scanned_ratio": budget.max_data_scanned_ratio,
            },
            "actual": {
                "cost_usd": outcome.cost_usd,
                "latency_s": outcome.latency_s,
                "tokens": outcome.tokens,
                "data_scanned_ratio": outcome.data_scanned_ratio,
            },
            "violations": violations,
        }
        if violations:
            detail = ", ".join(violations)
            if outcome.error:
                outcome.error += f"\nbudget_exceeded: {detail}"
            else:
                outcome.error = f"budget_exceeded: {detail}"
            # Only claim the failure taxonomy slot when nothing else already owns it.
            # A run that crashed and *then* overran its latency ceiling is a crash;
            # filing it under budget_exceeded hides the root cause on the scorecard.
            if outcome.error_kind is None:
                outcome.error_kind = FailureKind.BUDGET

    def run_suite(self, tasks: Iterable[BenchmarkTask],
                  configs: Iterable[RunConfig]) -> List[TaskResult]:
        tasks, configs = list(tasks), list(configs)
        results: List[TaskResult] = []
        for config in configs:
            sut = get_adapter(config.sut)
            self._log(f"config {config.id} (sut={config.sut}, sampling={config.sampling_mode}, "
                      f"llm={config.llm}) over {len(tasks)} tasks x {config.trials} trials")
            for task in tasks:
                try:
                    results.append(self.run_task(task, config, sut=sut))
                except SpendCeilingExceeded as exc:
                    self._log(f"  ABORT {exc}")
                    self.aborted_reason = str(exc)
                    return results
        return results
