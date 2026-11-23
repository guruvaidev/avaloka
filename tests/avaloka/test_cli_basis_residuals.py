"""Every number the CLI prints states a true basis (2.0 CLI re-check, F6).

1. The "Full-dataset option" panel offered "the exact answer over all
   181,115 rows" of an online_retail workbook whose analysed sheet holds
   541,910 rows and the workbook 1,067,371: 181,115 was the planner's
   newline count of the zip, printed as the count.
2. ``analyze`` on taxi printed the Statistician's mean of the 50,000-row
   sample (18.0796) while every one of the 2,964,624 rows was in memory
   (18.1751 over all of them).
3. ``coordinate`` printed "100,000/4,000,000 rows in the shard missions'
   working frames · 2 sampled shards" directly under an answer computed
   over every row of both shards.

The sampling threshold and sample size are lowered so a gate-sized file is
sampled exactly as the 2,964,624-row taxi file is.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import avaloka.goal_analysis as goal_analysis
from avaloka import workload
from avaloka.cli import _execution_line, app
from avaloka.fireflies import sampling_specialist
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze
from avaloka.statistics import analytic_mean_interval

THRESHOLD, SAMPLE = 1_000, 300


@pytest.fixture(autouse=True)
def _small_sampling(monkeypatch):
    monkeypatch.delenv("AVALOKA_EDITION", raising=False)
    monkeypatch.setattr(sampling_specialist, "SAMPLE_THRESHOLD_ROWS", THRESHOLD)
    monkeypatch.setattr(sampling_specialist, "DEFAULT_SAMPLE_ROWS", SAMPLE)


@pytest.fixture(scope="module")
def fares(tmp_path_factory):
    """6,000 skewed fares: a 300-row sample's mean is visibly off the full one."""
    rng = np.random.default_rng(11)
    n = 6_000
    frame = pd.DataFrame({
        "trip_id": np.arange(n),
        "payment_type": rng.choice(["card", "cash", "dispute"], n),
        "fare_amount": rng.lognormal(2.6, 0.9, n).round(2),
    })
    path = tmp_path_factory.mktemp("fares") / "fares.csv"
    frame.to_csv(path, index=False)
    return path


def _flat(output: str) -> str:
    """Terminal output with panel borders and wrapping removed."""
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", output).split())


# --------------------------------------------------------------------------- #
# (b) the Full-dataset panel uses the reader's count, and names the sheet
# --------------------------------------------------------------------------- #

@pytest.fixture
def workbook(fares, tmp_path):
    frame = pd.read_csv(fares)
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        frame.iloc[:1_500].to_excel(writer, sheet_name="Year 2010-2011", index=False)
        frame.iloc[1_500:1_700].to_excel(writer, sheet_name="Year 2009-2010", index=False)
    return path


def test_the_workbook_panel_shows_the_sheet_count_and_name_not_the_estimate(workbook, tmp_path,
                                                                             monkeypatch):
    # Route the workbook to the sampled lane, as a 541,910-row sheet is.
    monkeypatch.setattr(workload, "ONLINE_MAX_ROWS", 0)
    plan = workload.route(str(workbook))
    assert plan.recommend_batch and plan.estimated, "a workbook's peek is only an estimate"
    assert plan.n_rows != 1_500, "the estimate must differ from the count, or this proves nothing"
    result = CliRunner().invoke(app, ["analyze", str(workbook), "--goal", "average fare_amount",
                                      "--quiet", "--output", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    panel = flat[flat.index("Full-dataset option"):]
    assert ("This was the sampled live read of sheet 'Year 2010-2011' (1 other sheet not "
            "analysed: 'Year 2009-2010' with 200 rows). For the exact answer over all 1,500 rows "
            "of sheet 'Year 2010-2011', run it as a Ray swarm") in panel, panel
    # The planner's figure appears only as an estimate, anywhere in the run.
    estimate = f"{plan.n_rows:,}"
    said = len(re.findall(rf"(?<![\d,]){re.escape(estimate)} rows", flat))
    assert said >= 1, "the quiet header states the planner's figure"
    assert flat.count(f"about {estimate} rows (estimated)") == said, flat


def test_an_estimate_with_no_reader_count_is_said_as_one():
    from avaloka.cli import _full_dataset_scope
    ctx = SimpleNamespace(blackboard={})
    wl = workload.classify(181_115, 50 * 1024 * 1024, estimated=True)
    assert _full_dataset_scope(ctx, wl).endswith("over every row (about 181,115 rows (estimated))")
    assert "about 181,115 rows (estimated)" in wl.spoken and "~" not in wl.spoken


# --------------------------------------------------------------------------- #
# (a) the overall mean is the full data's when the full frame is in memory
# --------------------------------------------------------------------------- #

def _analyze(source, goal, out):
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal=goal, data_source=str(source),
                         output_dir=out, budget=Budget())
    return run_analyze(ctx)


def test_the_overall_mean_equals_pandas_on_the_full_data(fares, tmp_path):
    result = _analyze(fares, "average fare_amount by payment_type", tmp_path / "m")
    bb = result.context.blackboard
    assert bb["sampling"]["sampled"] is True and bb["sampling"]["n_working_rows"] == SAMPLE
    full = pd.read_csv(fares)["fare_amount"]
    stat = bb["statistical_result"]
    assert stat.estimate == pytest.approx(full.mean(), rel=1e-12)
    assert stat.sample_size == len(full)
    assert stat.confidence_interval == pytest.approx(analytic_mean_interval(full))
    assert bb["statistical_basis"] == {"full_pass": True, "sampled": False, "rows_used": len(full)}
    line = next(f for f in bb["findings"] if f.startswith("mean of fare_amount"))
    assert f"{full.mean():.4f} over {len(full):,} valid rows (computed over every row; 95% CI" in line
    assert "sample" not in line
    # The validator reconciled it against the same rows.
    checks = {c["name"]: c for c in bb["validation"]["checks"]}
    assert checks["goal_reconciliation"]["status"] == "pass", checks["goal_reconciliation"]


def test_the_fixture_sample_mean_is_visibly_not_the_full_mean(fares):
    full = pd.read_csv(fares)["fare_amount"]
    sample = full.sample(SAMPLE, random_state=42)
    assert abs(sample.mean() - full.mean()) > 0.2


def test_a_true_sample_keeps_its_sample_label(fares, tmp_path, monkeypatch):
    """Above FULL_PASS_MAX_ROWS the frame is not answered in full: the figure
    stays the sample's, and says so."""
    monkeypatch.setattr(goal_analysis, "FULL_PASS_MAX_ROWS", 100)
    result = _analyze(fares, "average fare_amount by payment_type", tmp_path / "s")
    bb = result.context.blackboard
    assert bb["statistical_basis"]["sampled"] is True and bb["statistical_basis"]["full_pass"] is False
    assert bb["statistical_result"].sample_size <= SAMPLE
    line = next(f for f in bb["findings"] if f.startswith("mean of fare_amount"))
    assert f"sample of {SAMPLE:,} of 6,000 rows" in line, line


def test_the_analytic_interval_is_the_normal_interval():
    values = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    half = 1.959963984540054 * values.std(ddof=1) / np.sqrt(len(values))
    assert analytic_mean_interval(values) == pytest.approx((values.mean() - half, values.mean() + half))


# --------------------------------------------------------------------------- #
# (c) the coordinate Execution line names each basis
# --------------------------------------------------------------------------- #

def _cr(*, answer_sampled, shard_samples, rows=4_000_000):
    shards = [SimpleNamespace(sampling={"sampled": n is not None, "n_working_rows": n or rows // 2})
              for n in shard_samples]
    return SimpleNamespace(execution={"parallelism": len(shards)}, shards=shards, n_rows=rows,
                           answer={"status": "answered", "sampled": answer_sampled,
                                   "basis": {"sampled": answer_sampled}})


def test_an_every_row_answer_with_sampled_profiling_says_both():
    line = _flat(_execution_line(_cr(answer_sampled=False, shard_samples=[50_000, 50_000])))
    assert line == ("Execution: [bold]2 concurrent sub-Avalokas[/bold] · answer: every row of "
                    "2 shards (4,000,000 rows) · profiling: samples of 50,000 rows per shard")
    assert "sampled shards" not in line and "working frames" not in line


def test_an_estimated_answer_is_not_called_every_row():
    line = _execution_line(_cr(answer_sampled=True, shard_samples=[50_000, 50_000]))
    assert "every row" not in line and "answer: estimated from samples" in line


def test_unsampled_shards_profile_every_row():
    line = _execution_line(_cr(answer_sampled=False, shard_samples=[None, None]))
    assert line.endswith("answer: every row of 2 shards (4,000,000 rows) · profiling: every row")


def test_the_terminal_execution_line_under_an_every_row_answer(fares, tmp_path):
    result = CliRunner().invoke(app, ["coordinate", str(fares), "--goal",
                                      "average fare_amount by payment_type", "--swarm", "2",
                                      "--quiet", "--output", str(tmp_path / "fleet")])
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "(computed over every row of the 2 shards)" in flat, flat
    assert (f"Execution: 2 concurrent sub-Avalokas · answer: every row of 2 shards (6,000 rows) "
            f"· profiling: samples of {SAMPLE:,} rows per shard") in flat, flat
    assert "sampled shards" not in flat and "working frames" not in flat
