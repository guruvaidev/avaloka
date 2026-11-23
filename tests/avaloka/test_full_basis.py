"""No figure claims a basis it does not have (2.0 CLI re-run, F5).

1. ``coordinate --swarm 2`` on a 4,000,000-row CSV: each clone answered
   "average revenue by region" over all 2,000,000 rows of her shard (#542's
   full pass), but handed back sufficient statistics of her 50,000-row
   working sample, so the merge reweighted sample means and named south
   highest where the full data has central -- and the fleet said PASS.
2. Single-file ``analyze``: the Statistician's mean of the 50,000-row sample
   was printed directly above "Computed over all 2,964,624 rows".
3. A workbook's "all 541,910 rows" were the rows of one of its two sheets.
4. Avro/ORC with every column typed as a string: age, avg_glucose_level and
   bmi were labels, so the driver ranking left age out.

The sampling threshold and sample size are lowered so a gate-sized file is
sampled exactly as a 2,000,000-row shard is.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import avaloka.goal_analysis as goal_analysis
from avaloka.cli import app
from avaloka.coordinator import MainAvaloka
from avaloka.fireflies import sampling_specialist
from avaloka.goal_merge import merge_goal_answers
from avaloka.io.loader import load_dataset
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze
from avaloka.swarm_limits import UNLIMITED

GOAL = "average revenue by region"
SHARD_ROWS = 3_000
THRESHOLD, SAMPLE = 1_000, 300


@pytest.fixture(autouse=True)
def _small_sampling(monkeypatch):
    """A 3,000-row shard is "large": it crosses the threshold and its
    working frame is a 300-row sample, as a 2,000,000-row shard's is 50,000."""
    monkeypatch.delenv("AVALOKA_EDITION", raising=False)
    monkeypatch.setattr(sampling_specialist, "SAMPLE_THRESHOLD_ROWS", THRESHOLD)
    monkeypatch.setattr(sampling_specialist, "DEFAULT_SAMPLE_ROWS", SAMPLE)


@pytest.fixture(scope="module")
def regions_csv(tmp_path_factory):
    """Five regions whose means are close, with heavy noise, so a 300-row
    sample's ranking and means differ visibly from the full data's."""
    rng = np.random.default_rng(5)
    n = 2 * SHARD_ROWS
    region = rng.choice(["north", "south", "east", "west", "central"], n)
    level = pd.Series(region).map({"north": 100.0, "south": 100.5, "east": 99.0,
                                   "west": 99.5, "central": 101.0}).to_numpy()
    frame = pd.DataFrame({
        "order_id": np.arange(n),
        "region": region,
        "units": rng.integers(1, 9, n),
        "revenue": (level + rng.normal(0, 40, n)).round(4),
    })
    path = tmp_path_factory.mktemp("regions") / "regions.csv"
    frame.to_csv(path, index=False)
    return path


def _truth(path) -> dict[str, float]:
    return pd.read_csv(path).groupby("region")["revenue"].mean().to_dict()


def _fleet(source, out, goal=GOAL):
    return MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(source), goal=goal, output_dir=out, workers=2, kind=MissionKind.ANALYZE)


# --------------------------------------------------------------------------- #
# (a) a sampled shard is still answered, and merged, over every row
# --------------------------------------------------------------------------- #

def test_the_merged_means_and_winner_equal_one_pass_over_the_full_data(regions_csv, tmp_path):
    truth = _truth(regions_csv)
    cr = _fleet(regions_csv, tmp_path / "fleet")
    # Both clones did sample: the test is only one of the full path if so.
    assert [s.sampling.get("sampled") for s in cr.shards] == [True, True]
    assert [s.sampling.get("n_working_rows") for s in cr.shards] == [SAMPLE, SAMPLE]
    got = {g["group"]: g["value"] for g in cr.answer["metrics"]["groups"]}
    assert set(got) == set(truth)
    for region, mean in truth.items():
        assert got[region] == pytest.approx(mean, rel=1e-9, abs=1e-6), region
    winner = max(truth, key=truth.get)
    assert cr.answer["metrics"]["groups"][0]["group"] == winner
    assert f"highest {winner}" in cr.answer["headline"]
    assert cr.answer["sampled"] is False and cr.answer["exact"] is True
    assert "estimated from samples" not in cr.answer["headline"]
    assert cr.answer["headline"].endswith("(computed over every row of the 2 shards).")
    # Each clone's counts are her whole shard's, not her sample's.
    assert sum(g["n"] for g in cr.answer["metrics"]["groups"]) == 2 * SHARD_ROWS


def test_the_fixture_would_catch_a_sample_based_merge(regions_csv):
    """The sample's means are visibly wrong on this fixture, or (a) proves nothing."""
    frame = pd.read_csv(regions_csv)
    halves = [frame.iloc[:SHARD_ROWS], frame.iloc[SHARD_ROWS:]]
    sample = pd.concat([h.sample(frac=SAMPLE / SHARD_ROWS, random_state=42) for h in halves])
    estimated = sample.groupby("region")["revenue"].mean()
    truth = pd.Series(_truth(regions_csv))
    assert (estimated - truth).abs().max() > 1.0


# --------------------------------------------------------------------------- #
# (b) a merge of samples is labelled an estimate and never earns PASS
# --------------------------------------------------------------------------- #

@pytest.fixture
def _no_full_pass(monkeypatch):
    """The clones cannot compute over their shards (as above
    FULL_PASS_MAX_ROWS), so their answers are the samples'."""
    monkeypatch.setattr(goal_analysis, "FULL_PASS_MAX_ROWS", 100)


def test_a_merge_of_samples_says_estimated_and_is_at_most_warn(regions_csv, tmp_path, _no_full_pass):
    cr = _fleet(regions_csv, tmp_path / "fleet")
    clone_verdicts = [s.verdict for s in cr.shards]
    assert "pass" in clone_verdicts, "the cap is only tested if a clone would have passed"
    assert cr.answer["sampled"] is True and cr.answer["exact"] is False
    assert (f"estimated from samples of {SAMPLE:,} of {SHARD_ROWS:,} rows per clone"
            in cr.answer["headline"])
    assert cr.verdict != "pass"
    written = json.loads((tmp_path / "fleet" / "coordination.json").read_text())
    assert written["verdict"] == cr.verdict and written["answer"]["sampled"] is True
    report = (tmp_path / "fleet" / "executive_report.md").read_text()
    assert "estimated from samples" in report and "Exact." not in report


def test_the_terminal_says_why_a_sampled_merge_is_not_pass(regions_csv, tmp_path, _no_full_pass):
    result = CliRunner().invoke(app, ["coordinate", str(regions_csv), "--goal", GOAL, "--swarm", "2",
                                      "--quiet", "--output", str(tmp_path / "fleet")])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "Converged verdict: PASS" not in flat
    assert "estimated from samples of" in flat
    assert "not reconciled against a computation over every row" in flat


def test_a_full_pass_basis_is_not_read_as_a_sample():
    """``basis`` is a dict whose key is "sampled": the text search for
    "sample" matched the key, and every merged answer read as a sample's."""
    def clone(name, rows, sampled):
        analysis = {"kind": "group_comparison", "status": "answered", "findings": ["x"],
                    "basis": {"sampled": sampled, "rows_used": 10, "source_rows": 10, "full_pass": False}}
        partials = {"kind": "group_comparison", "group_column": "g", "value_column": "v",
                    "aggregation": "mean", "rows": 10, "truncated": False,
                    "groups": {"a": {"rows": 10, "n": 10, "sum": 20.0}}}
        return {"name": name, "rows": rows, "analysis": analysis, "partials": partials}

    full = merge_goal_answers("mean v by g", [clone("Ava-00", [0, 9], False), clone("Ava-01", [10, 19], False)])
    assert full["sampled"] is False and full["exact"] is True
    half = merge_goal_answers("mean v by g", [clone("Ava-00", [0, 9], False), clone("Ava-01", [10, 19], True)])
    assert half["sampled"] is True and half["exact"] is False


# --------------------------------------------------------------------------- #
# (c) one basis per figure in a single-file summary
# --------------------------------------------------------------------------- #

def _analyze(source, goal, out):
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal=goal, data_source=str(source),
                         output_dir=out, budget=Budget())
    return run_analyze(ctx)


def test_no_sample_figure_sits_under_a_computed_over_all_rows_claim(regions_csv, tmp_path):
    result = _analyze(regions_csv, GOAL, tmp_path / "single")
    bb = result.context.blackboard
    assert bb["sampling"]["sampled"] is True
    assert bb["goal_analysis"]["basis"]["full_pass"] is True
    assert bb.get("statistical_result") is not None, "the Statistician must run for this to test anything"
    findings = bb["findings"]
    claim = next(i for i, f in enumerate(findings) if f.startswith("Computed over all"))
    stat = next(i for i, f in enumerate(findings) if "valid rows" in f)
    # Every row is in memory, so the Statistician's figure is the full mean
    # too (F6), and it sits among the figures the claim closes.
    full_mean = pd.read_csv(regions_csv)["revenue"].mean()
    assert bb["statistical_result"].estimate == pytest.approx(full_mean, rel=1e-12)
    assert stat < claim, findings
    assert "computed over every row" in findings[stat]
    # Every line the claim closes is a full-data figure.
    for line in findings[:claim]:
        assert "sample" not in line.lower(), line


def test_a_sample_figure_is_labelled_and_never_above_a_full_claim(regions_csv, tmp_path, monkeypatch):
    """Only the goal is answered in full (the Statistician cannot see every
    row): its figure is the sample's, labelled, and printed below the claim."""
    from avaloka.fireflies.analytical_specialists import Statistician
    monkeypatch.setattr(Statistician, "_population", lambda self, plan: None)
    result = _analyze(regions_csv, GOAL, tmp_path / "single")
    bb = result.context.blackboard
    findings = bb["findings"]
    claim = next(i for i, f in enumerate(findings) if f.startswith("Computed over all"))
    stat = next(i for i, f in enumerate(findings) if "valid rows" in f)
    full_mean = pd.read_csv(regions_csv)["revenue"].mean()
    assert bb["statistical_result"].estimate != pytest.approx(full_mean, abs=1e-3)
    assert stat > claim, f"the sample's figure is printed above the full-data claim: {findings}"
    total = int(bb["sampling"]["n_input_rows"])
    assert f"sample of {SAMPLE:,} of {total:,} rows" in findings[stat]
    for line in findings[:claim]:
        assert "sample" not in line.lower(), line


# --------------------------------------------------------------------------- #
# (d) a workbook's basis names the sheet, and the sheets not read
# --------------------------------------------------------------------------- #

def test_a_workbook_basis_names_the_sheet_and_the_other_sheets(regions_csv, tmp_path):
    frame = pd.read_csv(regions_csv)
    first, second = frame.iloc[:1_500], frame.iloc[1_500:1_700]
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        first.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        second.to_excel(writer, sheet_name="Year 2010-2011", index=False)
    result = _analyze(path, GOAL, tmp_path / "book")
    findings = result.context.blackboard["findings"]
    line = next(f for f in findings if f.startswith("Computed over all"))
    assert line.startswith("Computed over all 1,500 rows of sheet 'Year 2009-2010'"), line
    assert "1 other sheet not analysed ('Year 2010-2011' with 200 rows)" in line, line
    basis = result.context.blackboard["goal_analysis"]["basis"]
    assert basis["sheet"] == "Year 2009-2010"
    assert basis["other_sheets"] == [{"name": "Year 2010-2011", "rows": 200}]


# --------------------------------------------------------------------------- #
# (e) Avro and ORC numbers written as strings are numbers
# --------------------------------------------------------------------------- #

def _stroke_strings(n: int = 600) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(3)
    age = rng.uniform(5, 85, n).round(1)
    stroke = (rng.random(n) < 1 / (1 + np.exp(-(age - 70) / 6))).astype(int)
    bmi = rng.normal(28, 6, n).round(1).astype(str).astype(object)
    gaps = rng.random(n) < 0.05
    bmi[gaps] = "N/A"
    frame = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], n),
        "age": age.astype(str),
        "avg_glucose_level": rng.normal(105, 30, n).round(2).astype(str),
        "bmi": bmi,
        "stroke": stroke.astype(str),
    })
    return frame, int(gaps.sum())


def _write(frame: pd.DataFrame, path) -> None:
    if path.suffix == ".avro":
        fastavro = pytest.importorskip("fastavro")
        schema = {"type": "record", "name": "row",
                  "fields": [{"name": c, "type": "string"} for c in frame.columns]}
        with path.open("wb") as fh:
            fastavro.writer(fh, fastavro.parse_schema(schema), frame.to_dict("records"))
    else:
        orc = pytest.importorskip("pyarrow.orc")
        import pyarrow as pa
        orc.write_table(pa.Table.from_pandas(frame, preserve_index=False), str(path))


@pytest.mark.parametrize("suffix", [".avro", ".orc"])
def test_string_typed_numbers_are_numeric_and_rank_as_drivers(suffix, tmp_path):
    frame, gaps = _stroke_strings()
    path = tmp_path / f"stroke{suffix}"
    _write(frame, path)
    handle = load_dataset(str(path))
    for column in ("age", "avg_glucose_level", "bmi"):
        assert pd.api.types.is_float_dtype(handle.frame[column]), (column, handle.frame[column].dtype)
    assert int(handle.frame["bmi"].isna().sum()) == gaps
    assert handle.read["coerced"]["bmi"] == {"kind": "numeric", "cells": gaps, "words": ["N/A"]}
    assert handle.frame["gender"].dtype == object or isinstance(handle.frame["gender"].dtype, pd.StringDtype)
    result = _analyze(path, "what predicts stroke?", tmp_path / "mission")
    findings = result.context.blackboard["findings"]
    assert any(re.match(r"age: r=", f) for f in findings), findings
