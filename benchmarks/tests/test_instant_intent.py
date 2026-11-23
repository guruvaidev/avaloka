"""The instant_intent track: discoverable, scored field by field, replayable.

The evaluator tests are pure stdlib. The two SUT tests import ``avaloka``
(pandas) when they run the mission and skip where it is absent, as the
harness's own tests do for optional dependencies.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from benchmarks.core.registry import get_suite, list_suites
from benchmarks.core.runner import BenchmarkRunner
from benchmarks.core.task import BenchmarkTask, DatasetSpec
from benchmarks.tracks.instant_intent.track import evaluate_intent

_EXPECTED = {"recipe": "trend", "params": {"measure": "revenue", "time": "order_date"}, "flags": ["causal_question"]}


def _task(expected=_EXPECTED) -> BenchmarkTask:
    return BenchmarkTask(id="t", track="instant_intent", mission="q", dataset=DatasetSpec(id="d"),
                         evaluator=evaluate_intent, reference=expected, params={"columns": {}})


def test_track_is_discoverable():
    assert "instant_intent" in {s.name for s in list_suites()}


def test_evaluator_scores_each_field_exactly():
    exact = {"recipe": "trend", "params": {"time": "order_date", "measure": "revenue"},
             "flags": ["causal_question"], "parse_failure": False}
    q = evaluate_intent(_task(), exact)
    assert q["pass"] == 1.0 and q["recipe_match"] == 1.0 and q["params_match"] == 1.0
    assert q["flag_precision"] == 1.0 and q["flag_recall"] == 1.0 and q["parse_failure"] == 0.0

    wrong_params = dict(exact, params={"measure": "cost"})
    q = evaluate_intent(_task(), wrong_params)
    assert q["recipe_match"] == 1.0 and q["params_match"] == 0.0 and q["pass"] == 0.0

    over_flagged = dict(exact, flags=["causal_question", "ambiguous"])
    q = evaluate_intent(_task(), over_flagged)
    assert q["flag_precision"] == 0.5 and q["flag_recall"] == 1.0 and q["pass"] == 1.0

    missed_flag = dict(exact, flags=[])
    q = evaluate_intent(_task(), missed_flag)
    assert q["flag_recall"] == 0.0 and q["flag_precision"] == 1.0 and q["pass"] == 0.0


def test_evaluator_counts_a_parse_failure():
    q = evaluate_intent(_task(), {"recipe": None, "params": {}, "flags": [], "parse_failure": True})
    assert q["parse_failure"] == 1.0 and q["pass"] == 0.0 and q["score"] == 0.0
    assert evaluate_intent(_task(), None)["parse_failure"] == 1.0


def test_list_order_of_a_many_param_does_not_matter():
    expected = {"recipe": "correlate", "params": {"measures": ["revenue", "cost"]}, "flags": []}
    q = evaluate_intent(_task(expected), {"recipe": "correlate", "params": {"measures": ["cost", "revenue"]},
                                          "flags": [], "parse_failure": False})
    assert q["params_match"] == 1.0


def test_heuristic_config_runs_over_every_golden():
    pytest.importorskip("pandas")
    tasks, configs = get_suite("instant_intent")
    assert len(tasks) >= 100 and len({t.id for t in tasks}) == len(tasks)
    heuristic = [c for c in configs if c.sut == "intent_heuristic"]
    results = BenchmarkRunner().run_suite(tasks, heuristic)
    assert len(results) == len(tasks)
    assert all(r.stats["n_errors"] == 0 for r in results)
    recipe_acc = sum(r.stats["quality"]["recipe_match"]["mean"] for r in results) / len(results)
    assert recipe_acc >= 0.90
    assert all(r.stats["quality"]["parse_failure"]["mean"] == 0.0 for r in results)


def test_replay_config_rescores_recordings_and_counts_missing_ones(tmp_path):
    pytest.importorskip("pandas")
    tasks, configs = get_suite("instant_intent")
    replay = replace([c for c in configs if c.sut == "intent_replay"][0],
                     extra={"replies_dir": str(tmp_path)})
    first, second, third = tasks[:3]
    (tmp_path / f"{first.id}.json").write_text(json.dumps({"reply": json.dumps({
        "recipe": first.reference["recipe"], "params": first.reference["params"],
        "flags": first.reference["flags"], "confidence": 0.9})}))
    (tmp_path / f"{second.id}.json").write_text(json.dumps({"reply": '{"recipe": "forecast", "params": {}, '
                                                                     '"flags": [], "confidence": 1}'}))
    results = BenchmarkRunner().run_suite([first, second, third], [replay])
    by_id = {r.task_id: r for r in results}
    assert by_id[first.id].stats["pass_at_1"] == 1.0
    assert by_id[second.id].stats["quality"]["parse_failure"]["mean"] == 1.0
    assert by_id[third.id].stats["quality"]["parse_failure"]["mean"] == 1.0
    assert all(r.stats["n_errors"] == 0 for r in results)
    assert by_id[first.id].stats["n_cost_unknown"] == 1          # a replay does not invent a cost
