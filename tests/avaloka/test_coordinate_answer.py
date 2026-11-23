"""``avaloka coordinate`` answers the goal with one merged answer.

Each clone answers the goal over her shard; the fleet's answer is merged
from the sufficient statistics behind those answers (avaloka.goal_merge), so
it equals what one pass over the whole file computes: sums add, counts add,
means are weighted by n. A median cannot be merged from partials and must
say it is approximate. The answer names what the goal asked about, cites
each clone's share by row range, and reaches the terminal and the
executive report.

The fixture is built so a mean of the shard means is wrong: the first shard
holds 400 Obesity rows billed near 100, the second 100 billed near 300, so
the weighted mean is about 140 and the unweighted one about 200; and most
missing amounts sit in the second shard, so the column's own mean is
weighted unevenly as well.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from avaloka.cli import app
from avaloka.coordinator import MainAvaloka
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze
from avaloka.swarm_limits import UNLIMITED

GROUP_GOAL = "average billing amount by medical condition"


@pytest.fixture(autouse=True)
def _no_edition(monkeypatch):
    monkeypatch.delenv("AVALOKA_EDITION", raising=False)


@pytest.fixture(scope="module")
def billing_csv(tmp_path_factory):
    rng = np.random.default_rng(11)

    def half(counts: dict[str, tuple[int, float]]) -> pd.DataFrame:
        parts = [pd.DataFrame({"Medical Condition": name,
                               "Billing Amount": rng.normal(level, 5.0, n).round(3)})
                 for name, (n, level) in counts.items()]
        return pd.concat(parts).sample(frac=1.0, random_state=int(rng.integers(1 << 30)))

    first = half({"Obesity": (400, 100.0), "Diabetes": (100, 120.0), "Asthma": (60, 80.0),
                  "Cancer": (40, 60.0)})
    second = half({"Obesity": (100, 300.0), "Diabetes": (300, 125.0), "Asthma": (140, 85.0),
                   "Cancer": (60, 65.0)})
    frame = pd.concat([first, second], ignore_index=True)
    # Gaps fall mostly in the second shard, so the shards' non-null counts
    # differ and an unweighted mean of their column means is wrong too.
    frame.loc[[5, 17, *range(650, 800)], "Billing Amount"] = np.nan
    frame.insert(0, "Patient", [f"p{i:05d}" for i in range(len(frame))])
    frame["Age"] = rng.integers(18, 90, len(frame))
    path = tmp_path_factory.mktemp("billing") / "billing.csv"
    frame.to_csv(path, index=False)
    return path


def _single_pass(source, goal, out):
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal=goal, data_source=str(source),
                         output_dir=out, budget=Budget())
    run_analyze(ctx)
    return json.loads((out / "goal_analysis.json").read_text())


def _fleet(source, goal, out):
    return MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(source), goal=goal, output_dir=out, workers=2, kind=MissionKind.ANALYZE)


def _without_goal(text: str, goal: str) -> str:
    """The answer with the quoted goal taken out, so an echo of the goal
    cannot pass for an answer that names what it asked about."""
    return text.replace(f"“{goal}”", "")


# --------------------------------------------------------------------------- #
# (a) the merged figures equal one pass over the whole file
# --------------------------------------------------------------------------- #

def test_group_mean_is_weighted_by_rows_and_matches_a_single_pass(billing_csv, tmp_path):
    single = _single_pass(billing_csv, GROUP_GOAL, tmp_path / "single")
    cr = _fleet(billing_csv, GROUP_GOAL, tmp_path / "fleet")
    assert single["kind"] == cr.answer["kind"] == "group_comparison"
    want = {g["group"]: g["value"] for g in single["metrics"]["groups"]}
    got = {g["group"]: g["value"] for g in cr.answer["metrics"]["groups"]}
    assert set(got) == set(want) == {"Obesity", "Diabetes", "Asthma", "Cancer"}
    for group, value in want.items():
        assert got[group] == pytest.approx(value, rel=1e-9, abs=1e-6), group
    # The fixture is only a test of weighting if the shortcut is wrong on it.
    shard_means = [json.loads((tmp_path / "fleet" / f"sub-avaloka-{i:02d}" / "mission"
                               / "goal_analysis.json").read_text()) for i in range(2)]
    obesity = [next(g["value"] for g in s["metrics"]["groups"] if g["group"] == "Obesity")
               for s in shard_means]
    assert abs(sum(obesity) / 2 - want["Obesity"]) > 25
    assert cr.answer["exact"] is True and cr.answer["status"] == "answered"


@pytest.mark.parametrize("goal, statistic", [
    ("total billing amount", "sum"),
    ("average billing amount", "mean"),
])
def test_sum_and_mean_match_a_single_pass(billing_csv, tmp_path, goal, statistic):
    single = _single_pass(billing_csv, goal, tmp_path / "single")
    cr = _fleet(billing_csv, goal, tmp_path / "fleet")
    want = single["metrics"]["Billing Amount"]
    got = cr.answer["metrics"]["Billing Amount"]
    assert got["non_null"] == want["non_null"]
    assert got[statistic] == pytest.approx(want[statistic], rel=1e-9, abs=1e-6)
    assert cr.answer["exact"] is True
    if statistic == "mean":
        # The fixture only tests weighting if the shortcut is wrong on it.
        shard = [json.loads((tmp_path / "fleet" / f"sub-avaloka-{i:02d}" / "mission"
                             / "goal_analysis.json").read_text())["metrics"]["Billing Amount"]
                 for i in range(2)]
        assert shard[0]["non_null"] != shard[1]["non_null"]
        assert abs(sum(s["mean"] for s in shard) / 2 - want["mean"]) > 1


# --------------------------------------------------------------------------- #
# (b) the answer names what the goal asked about
# --------------------------------------------------------------------------- #

def test_the_answer_names_the_measure_the_group_and_the_leading_group(billing_csv, tmp_path):
    cr = _fleet(billing_csv, GROUP_GOAL, tmp_path / "fleet")
    answer = _without_goal(cr.answer["headline"], GROUP_GOAL)
    # The goal is lower case; the answer names the columns as the data does.
    assert "Billing Amount" in answer and "Medical Condition" in answer
    assert "Obesity" in answer
    report = (tmp_path / "fleet" / "executive_report.md").read_text()
    assert "Billing Amount" in _without_goal(report, GROUP_GOAL) and "Obesity" in report
    written = json.loads((tmp_path / "fleet" / "goal_analysis.json").read_text())
    assert written["headline"] == cr.answer["headline"]


# --------------------------------------------------------------------------- #
# (c) each clone's contribution is cited, by the rows she was given
# --------------------------------------------------------------------------- #

def test_each_clone_is_cited_by_row_range_in_the_terminal_and_the_report(billing_csv, tmp_path):
    out = tmp_path / "fleet"
    result = CliRunner().invoke(app, ["coordinate", str(billing_csv), "--goal", GROUP_GOAL,
                                      "--swarm", "2", "--quiet", "--output", str(out)])
    assert result.exit_code == 0, result.output
    shards = [s["n_rows"] for s in json.loads((out / "coordination.json").read_text())["shards"]]
    assert len(shards) == 2 and sum(shards) == 1200
    report = (out / "executive_report.md").read_text()
    start = 0
    for i, n in enumerate(shards):
        cite = f"Ava-{i:02d} (rows {start:,}–{start + n - 1:,})"
        assert cite in result.output, cite
        assert cite in report, cite
        start += n
    # The terminal states the answer, not only a profile and a verdict.
    assert "Answer" in result.output and "Obesity" in result.output
    assert re.search(r"Billing Amount by Medical Condition", result.output)


def test_verbose_narration_carries_the_answer_and_the_shares(billing_csv, tmp_path):
    result = CliRunner().invoke(app, ["coordinate", str(billing_csv), "--goal", GROUP_GOAL,
                                      "--swarm", "2", "-v", "--quiet", "--output",
                                      str(tmp_path / "fleet")])
    assert result.exit_code == 0, result.output
    narrated = [line for line in result.output.splitlines() if line.lstrip().startswith("›")]
    assert any("Answer to" in line and "Obesity" in line for line in narrated)
    assert sum(bool(re.search(r"Ava-0[01] \(rows ", line)) for line in narrated) == 2


# --------------------------------------------------------------------------- #
# (d) a statistic that does not merge says so
# --------------------------------------------------------------------------- #

def test_a_median_is_labelled_approximate_not_passed_off_as_exact(billing_csv, tmp_path):
    goal = "median billing amount"
    single = _single_pass(billing_csv, goal, tmp_path / "single")
    cr = _fleet(billing_csv, goal, tmp_path / "fleet")
    assert cr.answer["exact"] is False
    assert [a["statistic"] for a in cr.answer["approximate"]] == ["median of Billing Amount"]
    assert "median=" in cr.answer["headline"] and "(approximate)" in cr.answer["headline"]
    report = (tmp_path / "fleet" / "executive_report.md").read_text()
    assert "Approximate — median of Billing Amount" in report and "Exact." not in report
    # It is an approximation, and on this fixture a visibly different one.
    assert cr.answer["metrics"]["Billing Amount"]["median"] != pytest.approx(
        single["metrics"]["Billing Amount"]["median"], abs=1e-6)


def test_a_goal_with_no_figure_still_answers_in_words_and_cites_every_clone(billing_csv, tmp_path):
    # "What drives Obesity?" used to be this goal; a driver ranking now
    # merges (tests/avaloka/test_swarm_drivers.py), so it has a figure.
    goal = "tell me something interesting"
    cr = _fleet(billing_csv, goal, tmp_path / "fleet")
    assert cr.answer["status"] == "limited" and cr.answer["exact"] is False
    assert goal in cr.answer["headline"]
    assert [c["clone"] for c in cr.answer["contributions"]] == ["Ava-00", "Ava-01"]
