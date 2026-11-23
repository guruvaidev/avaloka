"""Trustworthiness tests for checkpointing, budgets, and evaluator diagnostics."""

from __future__ import annotations

import json
from dataclasses import replace

from benchmarks.core.checkpoint import TrialJournal, _json_safe
from benchmarks.core.config import Budget, RunConfig
from benchmarks.core.runner import BenchmarkRunner
from benchmarks.core.task import (BenchmarkTask, DatasetSpec, FailureKind,
                                  MissionOutcome)


class CountingSUT:
    key = "counting"

    def __init__(self, *, cost: float = 0.1) -> None:
        self.calls = 0
        self.cost = cost

    def run_mission(self, task, config):
        self.calls += 1
        return MissionOutcome(answer={"value": 7}, cost_usd=self.cost,
                              latency_s=0.01, tokens=10)


def _task():
    def evaluate(_task, answer):
        return {"pass": answer["value"] == 7, "score": 1.0,
                "validator_reason": "exact value matched"}

    return BenchmarkTask(
        id="reasoning:checkpoint:1",
        track="reasoning",
        mission="Return the supported value.",
        dataset=DatasetSpec(id="fixture", fingerprint="fixture-v1"),
        evaluator=evaluate,
    )


def test_journal_resumes_an_evaluated_trial_without_rerunning(tmp_path):
    task = _task()
    config = RunConfig(id="resume", sut="counting", trials=1)
    journal_path = tmp_path / "trials.jsonl"
    first_sut = CountingSUT()
    first = BenchmarkRunner(
        journal=TrialJournal(journal_path, suite="reasoning")
    ).run_task(task, config, sut=first_sut)
    assert first_sut.calls == 1
    assert first.trials[0].outcome.artifacts["evaluation"]["validator_reason"] \
        == "exact value matched"

    second_sut = CountingSUT()
    second = BenchmarkRunner(
        journal=TrialJournal(journal_path, suite="reasoning")
    ).run_task(task, config, sut=second_sut)
    assert second_sut.calls == 0
    assert second.stats["pass_at_1"] == 1.0


def test_increasing_trial_count_resumes_existing_trial_and_runs_only_new_ones(tmp_path):
    task = _task()
    journal_path = tmp_path / "trials.jsonl"
    one = RunConfig(id="extend", sut="counting", trials=1)
    BenchmarkRunner(
        journal=TrialJournal(journal_path, suite="reasoning")
    ).run_task(task, one, sut=CountingSUT())

    extended_sut = CountingSUT()
    extended = replace(one, trials=3)
    result = BenchmarkRunner(
        journal=TrialJournal(journal_path, suite="reasoning")
    ).run_task(task, extended, sut=extended_sut)
    assert extended_sut.calls == 2
    assert result.stats["n_trials"] == 3


def test_changed_effective_config_does_not_reuse_a_trial(tmp_path):
    task = _task()
    journal_path = tmp_path / "trials.jsonl"
    journal = TrialJournal(journal_path, suite="reasoning")
    base = RunConfig(id="resume", sut="counting", trials=1,
                     extra={"max_iterations": 10})
    BenchmarkRunner(journal=journal).run_task(task, base, sut=CountingSUT())
    changed_sut = CountingSUT()
    BenchmarkRunner(
        journal=TrialJournal(journal_path, suite="reasoning")
    ).run_task(task, replace(base, extra={"max_iterations": 11}), sut=changed_sut)
    assert changed_sut.calls == 1


def test_budget_violation_is_distinct_from_answer_failure():
    config = RunConfig(
        id="budget", sut="counting", trials=1,
        budget=Budget(max_cost_usd=0.5),
    )
    result = BenchmarkRunner().run_task(_task(), config, sut=CountingSUT(cost=0.75))
    outcome = result.trials[0].outcome
    assert outcome.error_kind is FailureKind.BUDGET
    assert outcome.artifacts["budget"]["violations"]
    assert result.stats["failure_kinds"] == {"budget_exceeded": 1}


def test_budget_violation_still_grades_the_answer():
    """Over budget is a cost verdict, not a quality verdict.

    Zeroing quality here would make "correct but too expensive" indistinguishable
    from "wrong", and would show a fake quality regression whenever a budget is
    tightened.
    """
    config = RunConfig(
        id="budget-quality", sut="counting", trials=1,
        budget=Budget(max_cost_usd=0.5),
    )
    result = BenchmarkRunner().run_task(_task(), config, sut=CountingSUT(cost=0.75))
    trial = result.trials[0]
    assert trial.quality["score"] == 1.0
    assert trial.quality["pass"] == 1.0
    assert trial.outcome.error_kind is FailureKind.BUDGET
    assert trial.outcome.artifacts["budget"]["violations"]


def test_budget_does_not_mask_an_earlier_failure_kind():
    """A crash that also overran its ceiling is a crash, not a budget violation."""

    class CrashingSUT:
        key = "crashing"

        def run_mission(self, task, config):
            raise RuntimeError("connection refused talking to provider")

    config = RunConfig(
        id="crash-and-overrun", sut="crashing", trials=1,
        budget=Budget(max_latency_s=0.0),
    )
    result = BenchmarkRunner().run_task(_task(), config, sut=CrashingSUT())
    outcome = result.trials[0].outcome
    assert outcome.error_kind is FailureKind.SUT_START
    assert "budget_exceeded" in outcome.error
    assert result.stats["failure_kinds"] == {"sut_start_error": 1}


def test_journal_skips_records_written_by_a_different_harness_version(tmp_path):
    """An unknown field must not make the whole journal unloadable."""
    task = _task()
    config = RunConfig(id="forward-compat", sut="counting", trials=1)
    journal_path = tmp_path / "trials.jsonl"
    BenchmarkRunner(
        journal=TrialJournal(journal_path, suite="reasoning")
    ).run_task(task, config, sut=CountingSUT())

    record = json.loads(journal_path.read_text(encoding="utf-8").splitlines()[-1])
    record["outcome"]["field_added_by_a_later_release"] = 1
    journal_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    journal = TrialJournal(journal_path, suite="reasoning")
    assert journal.skipped_records == 0
    assert journal.evaluated(task, replace(config, extra={"_trial": 0}), 0) is not None


def test_journal_survives_a_torn_final_line(tmp_path):
    """A kill -9 mid-append must cost one trial, not every completed trial."""
    task = _task()
    config = RunConfig(id="torn", sut="counting", trials=1)
    journal_path = tmp_path / "trials.jsonl"
    BenchmarkRunner(
        journal=TrialJournal(journal_path, suite="reasoning")
    ).run_task(task, config, sut=CountingSUT())

    intact = journal_path.read_text(encoding="utf-8")
    journal_path.write_text(intact + intact[:40], encoding="utf-8")

    journal = TrialJournal(journal_path, suite="reasoning")
    assert journal.skipped_records == 1
    # The complete records before the torn line are still resumable.
    resumed_sut = CountingSUT()
    BenchmarkRunner(journal=journal).run_task(task, config, sut=resumed_sut)
    assert resumed_sut.calls == 0


def test_scorecard_config_redacts_secret_extras():
    config = RunConfig(
        id="redact", sut="counting",
        extra={"api_key": "do-not-write", "max_iterations": 25,
               "max_output_tokens": 8192,
               "provider": {"access_token": "nested-secret"}, "_trial": 4},
    )
    published = config.scorecard_config()["extra"]
    assert published == {
        "api_key": "<redacted>",
        "max_iterations": 25,
        "max_output_tokens": 8192,
        "provider": {"access_token": "<redacted>"},
    }
    assert "do-not-write" not in config.fingerprint()
    assert "nested-secret" not in config.fingerprint()


def test_checkpoint_set_values_have_deterministic_order():
    assert _json_safe({"b", "a", "c"}) == ["a", "b", "c"]



def test_resumed_trials_are_not_charged_to_the_spend_ledger_again(tmp_path):
    """A trial replayed from the journal was paid for by the run that wrote it.

    The interesting case is an execution journaled but never evaluated (the
    process died in between): resume re-evaluates it without calling the SUT,
    and charging it again would trip the ceiling on money already accounted for.
    """
    from benchmarks.core.spend import SpendLedger

    class Killed(BaseException):
        """Escapes the runner's ``except Exception``, like a kill mid-evaluation."""

    task = _task()
    dying = replace(task, evaluator=lambda _t, _a: (_ for _ in ()).throw(Killed()))
    config = RunConfig(id="resume-spend", sut="counting", trials=1)
    journal_path = tmp_path / "trials.jsonl"

    first_ledger = SpendLedger(ceiling_usd=None)
    try:
        BenchmarkRunner(journal=TrialJournal(journal_path, suite="reasoning"),
                        ledger=first_ledger).run_task(dying, config, sut=CountingSUT(cost=0.1))
    except Killed:
        pass
    else:  # pragma: no cover - the setup itself must hold
        raise AssertionError("evaluator did not interrupt the first run")

    second_sut = CountingSUT(cost=0.1)
    second_ledger = SpendLedger(ceiling_usd=0.05)   # below one fresh trial's cost
    result = BenchmarkRunner(journal=TrialJournal(journal_path, suite="reasoning"),
                             ledger=second_ledger).run_task(task, config, sut=second_sut)
    assert second_sut.calls == 0, "execution should have been replayed, not re-run"
    assert result.stats["pass_at_1"] == 1.0
    assert second_ledger.calls == 0
    assert second_ledger.spent_usd == 0.0

    # And a genuinely fresh trial IS charged, so the zero above means something.
    fresh_ledger = SpendLedger(ceiling_usd=None)
    BenchmarkRunner(ledger=fresh_ledger).run_task(task, config, sut=CountingSUT(cost=0.1))
    assert fresh_ledger.calls == 1
    assert abs(fresh_ledger.spent_usd - 0.1) < 1e-9
