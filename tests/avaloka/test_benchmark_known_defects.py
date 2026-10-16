"""The known-defect ledger changes the exit code, strictly, and nothing else.

Three ways it could go wrong, each pinned here: a listed failure still failing
the run (the gate stays red for everyone, so the next real regression looks
the same as the eleven known ones); an unlisted failure passing the run (the
ledger became a mute button); a fixed defect staying listed (its regression
would then be hidden). The scorecard's own counts must not move at all.
"""

import json

import pytest

from avaloka.benchmark import __main__ as entry
from avaloka.benchmark import datasets as ds
from avaloka.benchmark.known_defects import KNOWN_DEFECTS, gate
from avaloka.benchmark.spec import BenchmarkTask, CheckResult, DatasetSpec, TaskResult
from avaloka.benchmark.suite import all_tasks


def _result(task, passed, error=None):
    return TaskResult(task=task, family="data_engineering", kind="analyze",
                      source_scheme="file", error=error,
                      checks=[CheckResult(name="c", passed=passed)])


def test_each_run_lands_in_exactly_one_bucket():
    known = {"known_red": "x", "known_green": "y"}
    verdict = gate([_result("known_red", False), _result("known_green", True),
                    _result("new_red", False), _result("plain_green", True)], known)
    assert verdict.known_failing == ["known_red (file)"]
    assert verdict.listed_but_passing == ["known_green (file)"]
    assert verdict.new_failures == ["new_red (file)"]
    assert not verdict.ok


def test_only_listed_failures_is_ok_and_a_skip_counts_for_nothing():
    known = {"known_red": "x", "skipped_listed": "y"}
    verdict = gate([_result("known_red", False), _result("fine", True),
                    _result("skipped_listed", False, error="dataset unavailable"),
                    _result("skipped_unlisted", False, error="kaggle unavailable")], known)
    assert verdict.ok
    assert verdict.known_failing == ["known_red (file)"]
    assert verdict.new_failures == [] and verdict.listed_but_passing == []


def test_every_listed_task_exists():
    """A stale name would never match a run, so it could never be caught as
    passing: the strictness rule only works for names that are real tasks."""
    from avaloka.benchmark.conversational import conversational_results
    names = {t.name for t in all_tasks()} | {r.task for r in conversational_results(live=False)}
    stale = sorted(set(KNOWN_DEFECTS) - names)
    assert not stale, f"KNOWN_DEFECTS lists tasks the suite does not have: {stale}"


def test_every_listed_task_fails_today(tmp_path):
    """The ledger is exactly the live defects. Run on the format_comprehension
    family, where ten of the eleven live; the eleventh is checked by name."""
    from avaloka.benchmark.runner import BenchmarkRunner
    listed = [t for t in all_tasks() if t.name in KNOWN_DEFECTS]
    card = BenchmarkRunner(tmp_path, verbose=False).run(listed)
    passing = [r.task for r in card.results if r.passed]
    assert not passing, f"listed in KNOWN_DEFECTS but passing: {passing}"
    assert len(listed) == len(KNOWN_DEFECTS) - 1   # + conv_avaloka_behaviour_is_unmeasured


def _good():
    return BenchmarkTask(name="good", family="data_engineering", kind="analyze", goal="x",
                         dataset=DatasetSpec("churn", "synthetic", builder=ds.build_churn))


def _broken(name):
    def builder(dest):
        raise RuntimeError("boom")
    return BenchmarkTask(name=name, family="data_engineering", kind="analyze", goal="x",
                         dataset=DatasetSpec(name, "synthetic", builder=builder))


def _tasks(labels):
    """'name (scheme)' -> 'name'. A run whose dataset cannot be built has no scheme."""
    return [label.split(" (")[0] for label in labels]


def _run_main(tmp_path, monkeypatch, tasks, known, *extra):
    monkeypatch.setattr(entry, "all_tasks", lambda include_kaggle=False, include_edgar=False: tasks)
    monkeypatch.setattr(entry, "KNOWN_DEFECTS", known)
    monkeypatch.setattr("avaloka.benchmark.known_defects.KNOWN_DEFECTS", known)
    monkeypatch.setattr(entry, "gate", lambda results: gate(results, known))
    out = tmp_path / "card.json"
    monkeypatch.setattr("sys.argv", ["avaloka.benchmark", "run", "--quiet",
                                     "--family", "data_engineering",
                                     "--out", str(tmp_path / "out"), "--json", str(out), *extra])
    with pytest.raises(SystemExit) as stop:
        entry.main()
    return stop.value.code, json.loads(out.read_text())


def test_a_listed_failure_exits_0_and_is_still_counted_as_a_failure(tmp_path, monkeypatch, capsys):
    code, card = _run_main(tmp_path, monkeypatch, [_broken("known"), _good()], {"known": "boom"})
    assert code == 0
    assert card["n_runs"] == 2 and card["n_passed"] == 1        # the count does not move
    assert card["results"][0]["passed"] is False
    assert _tasks(card["gate"]["known_failing"]) == ["known"] and card["gate"]["ok"] is True
    shown = capsys.readouterr().out
    assert "[KNOWN] known" in shown and "known defects failing    1" in shown


def test_strict_restores_exit_1_on_any_failure(tmp_path, monkeypatch):
    code, card = _run_main(tmp_path, monkeypatch, [_broken("known"), _good()],
                           {"known": "boom"}, "--strict")
    assert code == 1 and card["gate"]["strict"] is True


def test_an_unlisted_failure_still_exits_1(tmp_path, monkeypatch):
    code, card = _run_main(tmp_path, monkeypatch, [_broken("known"), _broken("new"), _good()],
                           {"known": "boom"})
    assert code == 1 and _tasks(card["gate"]["new_failures"]) == ["new"]


def test_a_listed_task_that_passes_exits_1(tmp_path, monkeypatch, capsys):
    code, card = _run_main(tmp_path, monkeypatch, [_good()], {"good": "was broken"})
    assert code == 1 and _tasks(card["gate"]["listed_but_passing"]) == ["good"]
    assert "remove it from KNOWN_DEFECTS" in capsys.readouterr().out
