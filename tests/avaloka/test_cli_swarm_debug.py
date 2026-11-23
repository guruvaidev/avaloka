"""``avaloka coordinate --swarm N`` and the ``-v`` / ``--debug`` narration.

The narration must be a pure function of the trace records, and the swarm
size must be honoured within the edition cap and clamped above it with one
sentence that names the cap and the edition. The cap is read from the code
that enforces it, never typed into an assertion.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from avaloka.cli import app
from avaloka.coordinator import MainAvaloka, load_coordination
from avaloka.swarm_limits import UNLIMITED, limits_from_env
from avaloka.swarm_narration import (DEBUG, HANDLERS, QUIET, VERBOSE,
                                     SwarmNarrative, narrate)
from avaloka.swarm_trace import SwarmTraceRecorder


@pytest.fixture(autouse=True)
def _no_edition(monkeypatch):
    monkeypatch.delenv("AVALOKA_EDITION", raising=False)


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    """The records of one real two-clone analysis — the fireflies, not a stub."""
    from tests.avaloka.conftest import write_churn_csv

    base = tmp_path_factory.mktemp("story")
    data = write_churn_csv(base / "churn.csv")
    recorder = SwarmTraceRecorder()
    MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(data), goal="what drives churn", output_dir=base / "fleet", workers=2,
        recorder=recorder)
    return recorder.records


# --------------------------------------------------------------------------- #
# The narrator is a pure function of the records
# --------------------------------------------------------------------------- #

def test_folding_one_record_at_a_time_equals_narrating_the_list(records):
    story = SwarmNarrative(DEBUG)
    streamed = [line for r in records for line in story.feed(r)]
    assert streamed == narrate(records, DEBUG) == narrate(records, DEBUG)
    assert narrate([], DEBUG) == [] and narrate(records, QUIET) == []


def test_every_record_type_the_coordinator_writes_has_a_sentence(records):
    assert {r["type"] for r in records} <= set(HANDLERS)
    for r in records:
        assert narrate([r], DEBUG), f"{r['type']} produced no sentence"


def test_the_lines_are_plain_language_from_the_real_events(records):
    lines = narrate(records, DEBUG)
    text = "\n".join(lines)
    for line in lines:
        assert "{" not in line and "}" not in line and "Traceback" not in line
    # The sampling specialist's own reason, verbatim from her step record.
    sampling = next(r for r in records
                    if r["type"] == "clone.step.completed" and r["firefly"] == "sampling_specialist")
    assert sampling["output"] in text
    sharded = next(r for r in records if r["type"] == "swarm.sharded")
    assert f"{sharded['total_rows']:,} rows × {len(sharded['columns'])} columns" in text
    merged = next(r for r in records if r["type"] == "swarm.merged")
    assert f"→ {merged['verdict'].upper()}" in text
    finished = next(r for r in records if r["type"] == "swarm.finished")
    hours = finished["economics"]["estimated_manual_effort_hours"]
    # A heuristic, and said to be one (W4): never "measured work" of "a human team".
    assert f"Manual effort: about {hours:.1f} hours, a rough heuristic estimate, not measured" in text
    assert "human team" not in text
    assert "No model was called" in text  # model_cost_usd == 0 on this run
    # The hours figure says what it rests on: the rated minutes of the steps.
    steps = [r for r in records if r["type"] == "clone.step.completed"]
    assert f"each of the {len(steps)} specialist steps is rated at, added up" in text
    assert abs(sum(r["manual_minutes"] for r in steps) / 60 - hours) < 0.05
    assert "Ava-00" in text and "Ava-01" in text


def test_a_changed_record_changes_the_sentence(records):
    requested = dict(records[0])
    assert "what drives churn" in narrate([requested], VERBOSE)[0]
    requested["goal"] = "why do accounts lapse"
    assert "why do accounts lapse" in narrate([requested], VERBOSE)[0]
    assert narrate([{"type": "agent.step", "seq": 1}], DEBUG) == []  # unknown type: no guess


def test_verbose_tells_the_team_story_and_debug_adds_the_steps(records):
    verbose, debug = narrate(records, VERBOSE), narrate(records, DEBUG)
    assert len(verbose) < len(debug)
    assert not any("→ step" in line or "← " in line for line in verbose)
    steps = [r for r in records if r["type"] == "clone.step.completed"]
    assert sum("← " in line for line in debug) == len(steps)
    assert sum("→ step" in line for line in debug) == len(steps)


def test_a_failed_clone_is_told_without_a_traceback():
    failed = {"type": "clone.returned", "clone_id": "ava-00", "status": "failed",
              "duration_ms": 12, "error_class": "KeyError",
              "error": "'target'\nTraceback (most recent call last):\n  File ..."}
    (line,) = narrate([failed], VERBOSE)
    assert "came back without a result" in line and "KeyError" in line
    assert "Traceback" not in line and "File" not in line


# --------------------------------------------------------------------------- #
# --swarm: honoured within the cap, clamped above it, with one sentence
# --------------------------------------------------------------------------- #

def _run(churn_csv, out, *args):
    return CliRunner().invoke(app, ["coordinate", str(churn_csv), "--goal", "what drives churn",
                                    "-o", str(out), *args])


def test_swarm_above_the_cap_is_clamped_and_said_once(churn_csv, tmp_path):
    limits = limits_from_env()
    cap = limits.max_clones_per_mission
    assert cap is not None, "this build has no cap; the test needs one"
    result = _run(churn_csv, tmp_path / "f", "--swarm", str(cap + 3), "--quiet")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    sentence = (f"Swarm limited to {cap} clones on {limits.label}: "
                f"you asked for {cap + 3}, so {cap} will run.")
    assert text.count(sentence) == 1, text
    assert load_coordination(tmp_path / "f")["workers"] == cap


def test_swarm_within_the_cap_is_honoured_without_a_warning(churn_csv, tmp_path):
    cap = limits_from_env().max_clones_per_mission
    for n in sorted({1, cap}):
        out = tmp_path / f"f{n}"
        result = _run(churn_csv, out, "--swarm", str(n), "--quiet")
        assert result.exit_code == 0, result.output
        assert "Swarm limited" not in result.output
        report = load_coordination(out)
        assert report["workers"] == n and report["execution"]["swarm"]["limited"] is False


def test_workers_is_an_alias_of_swarm_and_may_not_disagree(churn_csv, tmp_path):
    a = _run(churn_csv, tmp_path / "a", "--swarm", "1", "--quiet")
    b = _run(churn_csv, tmp_path / "b", "--workers", "1", "--quiet")
    assert a.exit_code == b.exit_code == 0
    assert load_coordination(tmp_path / "a")["workers"] == load_coordination(tmp_path / "b")["workers"] == 1
    c = _run(churn_csv, tmp_path / "c", "--swarm", "1", "--workers", "2")
    assert c.exit_code == 2 and "disagree" in c.output


# --------------------------------------------------------------------------- #
# The CLI prints the story from the records it streams, and only when asked
# --------------------------------------------------------------------------- #

def test_debug_output_states_the_effective_size_and_the_cap(churn_csv, tmp_path):
    limits = limits_from_env()
    cap = limits.max_clones_per_mission
    out = tmp_path / "f"
    result = _run(churn_csv, out, "--swarm", str(cap + 1), "--debug")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert (f"Team size: {cap} clones will run — you asked for {cap + 1}, "
            f"and {limits.label} allows {cap} per mission.") in text
    # Every printed sentence is one the records produce — nothing else is said.
    streamed = [json.loads(l) for l in (out / "swarm_trace.jsonl").read_text().splitlines()]
    expected = narrate(streamed, DEBUG)
    printed = [l.split("› ", 1)[1].strip() for l in result.output.splitlines() if "› " in l]
    # Rich may wrap a long sentence; compare on whitespace-normalised text.
    assert " ".join(" ".join(printed).split()) == " ".join(" ".join(expected).split())
    assert any("← sampling specialist" in l for l in expected)


def test_without_a_flag_nothing_is_narrated(churn_csv, tmp_path):
    result = _run(churn_csv, tmp_path / "f", "--swarm", "1")
    assert result.exit_code == 0, result.output
    assert "› " not in result.output
    assert "Converged verdict" in result.output


# --------------------------------------------------------------------------- #
# The understanding records (feat/2.0-advanced-telemetry): narrated when present
# --------------------------------------------------------------------------- #

def test_agent_understanding_tells_the_decision_not_the_conclusion_again():
    planner = {"type": "agent.understanding", "clone_id": "ava-00", "agent": "analysis_planner",
               "outcome": "ok", "concluded": "Planned task='binary_classification' ...",
               "decided": {"task": "binary_classification", "task_reason": "Target has exactly 2 classes.",
                           "split": "stratified", "split_reason": "Stratified split preserves class balance.",
                           "findings": ["tenure is the strongest signal.", "contract matters.", "region does not."]},
               "verification": None, "model": {"calls": 0}}
    lines = narrate([planner], DEBUG)
    assert lines == [
        "ava-00's analysis planner chose the task binary_classification because Target has exactly 2 "
        "classes; split stratified: Stratified split preserves class balance.",
        "ava-00 found: tenure is the strongest signal; contract matters (and 1 more).",
    ]
    assert "Planned task" not in " ".join(lines)  # the conclusion was told from the step record
    validator = {"type": "agent.understanding", "clone_id": "ava-01", "agent": "validator", "outcome": "ok",
                 "decided": {"verdict": "warn", "blocking_issues": []},
                 "verification": {"checks_run": 3, "passed": 2, "warned": 1, "failed": 0, "retries": 0}}
    assert narrate([validator], DEBUG) == ["ava-01's validator ran 3 checks: 2 passed, 1 warned, 0 failed."]
    one = {**validator, "verification": {"checks_run": 1, "passed": 1, "warned": 0, "failed": 0}}
    assert narrate([one], DEBUG) == ["ava-01's validator ran 1 check: 1 passed, 0 warned, 0 failed."]
    assert narrate([validator], VERBOSE) == []
    assert narrate([{**validator, "outcome": "failed"}], DEBUG) == []


def test_swarm_understanding_gives_the_team_totals_at_verbose():
    rec = {"type": "swarm.understanding", "clones": 2, "steps_total": 16,
           "verification": {"checks_run": 2, "passed": 2, "warned": 0, "failed": 0, "verdict": "pass"},
           "merge": {"verdict": "pass", "agreement": "unanimous"}, "model": {"calls": 0}}
    assert narrate([rec], VERBOSE) == [
        "Across the team: 16 specialist steps across 2 clones; 2 checks run (2 passed, 0 warned, 0 failed); "
        "the clones were unanimous on PASS."]
    assert narrate([{"type": "swarm.understanding"}], VERBOSE) == []


def test_the_optimizer_and_feature_engineer_are_narrated_by_agent_and_named_by_clone_name():
    optimizer = {"type": "agent.understanding", "clone_id": "ava-01", "clone_name": "Ava-01",
                 "agent": "workload_optimizer", "role": "planning", "outcome": "ok",
                 "decided": {"strategy": "filter-then-project", "n_filter_factors": 2,
                             "n_pushdown_filters": 2, "n_projection_columns": 5,
                             "estimated_rows_after_filters": 12345, "predicate_order": ["a", "b"]}}
    assert narrate([optimizer], DEBUG) == [
        "Ava-01's workload optimizer chose the filter-then-project path: 2 filters pushed down, "
        "5 columns projected, about 12,345 rows after filtering."]
    features = {"type": "agent.understanding", "clone_id": "ava-01", "clone_name": "Ava-01",
                "agent": "feature_engineer", "role": "feature_engineering", "outcome": "ok",
                "decided": {"n_steps": 4, "operations": ["one_hot", "scale", "bin", "interact"],
                            "recipe_hash": "abc", "diagnostics": ["x"]}}
    assert narrate([features], DEBUG) == [
        "Ava-01's feature engineer applied 4 feature steps (one_hot, scale, bin and 1 more); 1 diagnostic."]
    # A planner-role record that is not the planner says nothing about a task.
    assert "chose the task" not in " ".join(narrate([optimizer], DEBUG))
    untouched = {**features, "decided": {"n_steps": 0, "operations": []}}
    assert narrate([untouched], DEBUG) == ["Ava-01's feature engineer left the columns as they were."]


def test_a_local_model_that_cost_nothing_is_still_reported_as_called():
    """gemma on Ollama costs $0, so the cost figure alone said "No model was called"."""
    reporter = {"type": "agent.understanding", "clone_id": "ava-00", "clone_name": "Ava-00",
                "agent": "reporter", "outcome": "ok", "decided": {},
                "model": {"provider": "local", "model_id": "gemma4:e4b", "calls": 1}}
    finished = {"type": "swarm.finished", "status": "done", "duration_ms": 44_200,
                "economics": {"estimated_manual_effort_hours": 25.5, "model_cost_usd": 0.0}}
    lines = narrate([reporter, {**reporter, "clone_id": "ava-01", "clone_name": "Ava-01"}, finished], DEBUG)
    assert lines[0] == "Ava-00's reporter made 1 call to gemma4:e4b on local."
    assert lines[-1].endswith("The reporter made 2 calls to gemma4:e4b; every number above was computed "
                              "from the data, not by a model.")
    assert "No model was called" not in lines[-1]
    # Without understanding records the cost figure is still the (only) evidence.
    assert narrate([finished], DEBUG)[-1].endswith("No model was called: every number above was computed from the data.")
