"""A workbook's own sheet, named as a sheet, is not a column claim.

integration/2.0's Benchmark Scorecard (pull-request pipeline, blocking) exited 1
on xlsx_multi_sheet_data_on_sheet_two: both narration checks flagged
``technical_report.html: quoted 'Usage'``. The sentence is Data Scout's
"Analysed sheet 'Usage' only; 2 other sheet(s) ... were not read" (167a375f,
reached integration/2.0 in #595). 'Usage' is the workbook's sheet and the
sentence says so; the same convention writes "of sheet 'Year 2009-2010'" in
goal_analysis and cli. The checks read every quoted token as a column. They now
skip one only when it is a sheet of the graded workbook *and* its clause calls
it a sheet -- everything else is graded as before.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from avaloka.benchmark.comprehension import (score_narration_against_ground_truth,
                                             score_narration_columns)
from avaloka.benchmark.spec import Expectations

COLUMNS = ["Account", "Plan", "Seats", "Active users", "Monthly cost"]
SHEETS = {"analysed": "Usage", "others": [{"name": "Cover", "rows": 4},
                                         {"name": "Reference data", "rows": 3}]}


def _mission(tmp_path: Path, sentence: str, sheets=SHEETS) -> Path:
    out = tmp_path / "mission"
    out.mkdir()
    (out / "technical_report.html").write_text(
        f"<html><body><h2>Data quality issues</h2><table><tr><td>{sentence}</td></tr>"
        "</table></body></html>")
    (out / "data_quality.json").write_text(json.dumps(
        {"dataset": {"n_rows": 30, "structure": {"sheets": sheets} if sheets else {}}}))
    return out


def _grade(out: Path) -> dict[str, tuple[bool, str]]:
    exp = Expectations(narration_names_only_real_columns=True,
                       narration_columns_from_ground_truth=True,
                       expected_columns_resolved=list(COLUMNS))
    frame = pd.DataFrame(columns=COLUMNS)
    results = score_narration_columns(out, exp, frame) + score_narration_against_ground_truth(out, exp)
    return {r.name: (r.passed, r.detail) for r in results}


def test_the_data_scout_sentence_naming_the_analysed_sheet_is_not_a_column_claim(tmp_path):
    # The exact text of the failing run's technical_report.html.
    out = _mission(tmp_path, "Analysed sheet &#39;Usage&#39; only; 2 other sheet(s) hold 7 rows "
                             "and were not read: Cover (4 rows), Reference data (3 rows).")
    for name, (passed, detail) in _grade(out).items():
        assert passed, f"{name}: {detail}"


def test_goal_analysis_and_cli_sheet_phrasing_is_recognised_too(tmp_path):
    out = _mission(tmp_path, "Computed over all 30 rows of sheet &#39;Usage&#39;; 2 other sheets "
                             "not analysed (&#39;Cover&#39; with 4 rows, &#39;Reference data&#39; "
                             "with 3 rows). This was the sampled live read of sheet &#39;Usage&#39;.")
    for name, (passed, detail) in _grade(out).items():
        assert passed, f"{name}: {detail}"


def test_a_sheet_name_used_as_a_column_is_still_flagged(tmp_path):
    # No sheet label in the clause: the report is treating 'Usage' as a column.
    out = _mission(tmp_path, "&#39;Usage&#39; and &#39;Seats&#39; are strongly associated.")
    for name, (passed, detail) in _grade(out).items():
        assert not passed and "quoted 'Usage'" in detail, f"{name}: {detail}"


def test_a_labelled_name_the_workbook_does_not_have_is_still_flagged(tmp_path):
    out = _mission(tmp_path, "Analysed sheet &#39;Revenue&#39; only.")
    for name, (passed, detail) in _grade(out).items():
        assert not passed and "quoted 'Revenue'" in detail, f"{name}: {detail}"


def test_the_label_does_not_carry_across_a_sentence(tmp_path):
    out = _mission(tmp_path, "Analysed one sheet. &#39;Usage&#39; drives cost.")
    for name, (passed, detail) in _grade(out).items():
        assert not passed and "quoted 'Usage'" in detail, f"{name}: {detail}"


def test_a_source_that_is_not_a_workbook_exempts_nothing(tmp_path):
    out = _mission(tmp_path, "Analysed sheet &#39;Usage&#39; only.", sheets=None)
    for name, (passed, detail) in _grade(out).items():
        assert not passed and "quoted 'Usage'" in detail, f"{name}: {detail}"
