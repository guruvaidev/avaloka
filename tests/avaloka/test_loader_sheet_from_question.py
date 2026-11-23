"""The question picks the sheet, when the question settles it.

Benchmark task ``xlsx_decoy_sheet_outweighs_the_answer`` (a declared known
defect until this change): ``_read_excel`` read the sheet with the most
non-empty cells, whatever was asked. A reference tab longer and denser than
the summary someone asks about wins, the mission profiles and answers off the
wrong table, and nothing looks wrong -- the wrong sheet reads perfectly.

The rule now (``loader._choose_sheet``): a sheet named by the caller wins;
otherwise, with a question, the one sheet whose own words (name, title and
header rows, first-column labels) share the most words with the question;
otherwise, as before, the sheet with the most cells. A tie, or a question that
names nothing a sheet has, decides nothing. Which rule decided is recorded and
said in the quality report.

The mission-level tests at the top drive the existing entry points (an
analyze mission, the benchmark runner) and fail on the base by assertion. The
API-level tests below them use the new ``question``/``sheet`` arguments.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("openpyxl", reason="writing the workbooks needs openpyxl")

from avaloka.benchmark.datasets_spreadsheet import (  # noqa: E402
    ANSWER_SHEET,
    DECOY_SHEET,
    build_decoy_sheet_outweighs_the_answer,
    build_sec_financial_report,
)
from avaloka.fireflies.data_scout import _loader_issues  # noqa: E402
from avaloka.io.loader import load_dataset  # noqa: E402
from avaloka.mission.budget import Budget  # noqa: E402
from avaloka.mission.context import MissionContext, MissionKind  # noqa: E402
from avaloka.missions import run_analyze  # noqa: E402

OPERATIONS = "CONSOLIDATED STATEMENTS OF OPER"
BALANCE = "Consolidated Balance Sheets"


def _mission(tmp_path, source, goal):
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal=goal, data_source=str(source),
                         output_dir=tmp_path / "bundle", budget=Budget(limit_usd=None))
    run_analyze(ctx)
    return ctx


# --------------------------------------------------------------------------
# Through the existing entry points
# --------------------------------------------------------------------------
def test_an_analyze_mission_reads_the_sheet_the_question_is_about(tmp_path):
    book = build_decoy_sheet_outweighs_the_answer(tmp_path / "decoy.xlsx")
    ctx = _mission(tmp_path, book, "What is the total Revenue?")

    sheets = ctx.blackboard["profile"]["dataset"]["structure"]["sheets"]
    assert sheets["analysed"] == ANSWER_SHEET
    assert list(ctx.blackboard["frame"].columns) == ["Quarter", "Revenue", "Headcount"]
    assert ctx.blackboard["goal_analysis"]["metrics"]["Revenue"]["sum"] == pytest.approx(1_777_000.0)


def test_the_benchmark_grades_the_sheet_the_mission_read(tmp_path):
    from avaloka.benchmark.runner import BenchmarkRunner
    from avaloka.benchmark.suite import spreadsheet_tasks

    task = next(t for t in spreadsheet_tasks() if t.name == "xlsx_decoy_sheet_outweighs_the_answer")
    (result,) = BenchmarkRunner(tmp_path, verbose=False).run([task]).results
    failed = [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.passed, failed


def test_a_question_that_names_nothing_keeps_the_old_choice(tmp_path):
    book = build_decoy_sheet_outweighs_the_answer(tmp_path / "decoy.xlsx")
    ctx = _mission(tmp_path, book, "Profile this workbook")
    assert ctx.blackboard["profile"]["dataset"]["structure"]["sheets"]["analysed"] == DECOY_SHEET


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------
def test_the_choice_and_its_reason_are_recorded_and_said(tmp_path):
    book = build_decoy_sheet_outweighs_the_answer(tmp_path / "decoy.xlsx")
    handle = load_dataset(str(book), question="What is the total Revenue?")

    assert handle.read["sheet_name"] == ANSWER_SHEET
    assert handle.structure["sheets"]["chosen_by"] == {"by": "question", "matched": ["revenue"]}
    detail = next(i["detail"] for i in _loader_issues(handle)
                  if i["kind"] == "other_sheets_not_analysed")
    assert "because the question names revenue" in detail


def test_without_a_question_the_largest_sheet_is_read_and_that_is_said(tmp_path):
    book = build_decoy_sheet_outweighs_the_answer(tmp_path / "decoy.xlsx")
    handle = load_dataset(str(book))

    assert handle.read["sheet_name"] == DECOY_SHEET
    assert handle.structure["sheets"]["chosen_by"] == {"by": "size", "matched": []}
    detail = next(i["detail"] for i in _loader_issues(handle)
                  if i["kind"] == "other_sheets_not_analysed")
    assert "most filled cells" in detail


def test_aggregation_words_alone_choose_nothing(tmp_path):
    # "total" says what to compute, not what about: the smaller sheet has a
    # Total row, and that must not make it the sheet the question is about.
    path = tmp_path / "two.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Item": ["a", "Total"], "Value": [1, 1]}).to_excel(
            writer, sheet_name="Small", index=False)
        pd.DataFrame({"Item": list("abcdefgh"), "Value": range(8)}).to_excel(
            writer, sheet_name="Large", index=False)
    handle = load_dataset(str(path), question="What is the total?")
    assert handle.read["sheet_name"] == "Large"
    assert handle.structure["sheets"]["chosen_by"]["by"] == "size"


def test_a_tie_falls_back_to_size(tmp_path):
    path = tmp_path / "tie.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Region": ["n"], "Revenue": [1]}).to_excel(writer, sheet_name="A", index=False)
        pd.DataFrame({"Region": list("nsew"), "Revenue": range(4)}).to_excel(
            writer, sheet_name="B", index=False)
    handle = load_dataset(str(path), question="Revenue by Region")
    assert handle.read["sheet_name"] == "B"
    assert handle.structure["sheets"]["chosen_by"]["by"] == "size"


def test_row_labels_count_so_a_statement_question_finds_its_statement(tmp_path):
    book = build_sec_financial_report(tmp_path / "10k.xlsx")
    income = load_dataset(str(book), question='What is the total Dec. 31, 2023 where Line item = "Net income"?')
    assets = load_dataset(str(book), question="What were Total assets at Dec. 31, 2023?")

    assert income.read["sheet_name"] == OPERATIONS
    assert assets.read["sheet_name"] == BALANCE


def test_a_named_sheet_wins_and_an_unknown_one_is_refused(tmp_path):
    book = build_decoy_sheet_outweighs_the_answer(tmp_path / "decoy.xlsx")
    handle = load_dataset(str(book), question="What is the total Revenue?", sheet=DECOY_SHEET)
    assert handle.read["sheet_name"] == DECOY_SHEET
    assert handle.structure["sheets"]["chosen_by"]["by"] == "named"

    with pytest.raises(ValueError, match="has no sheet 'Nope'.*Quarterly summary"):
        load_dataset(str(book), sheet="Nope")


def test_a_swarm_splits_the_sheet_the_question_is_about(tmp_path):
    from avaloka.io.sharding import write_shards

    book = build_decoy_sheet_outweighs_the_answer(tmp_path / "decoy.xlsx")
    plan = write_shards(book, [tmp_path / "s0", tmp_path / "s1"], question="What is the total Revenue?")
    assert plan.total_rows == 4
    assert plan.columns == ["Quarter", "Revenue", "Headcount"]


def test_other_formats_ignore_the_question(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("Region,Revenue\nn,1\ns,2\n")
    assert load_dataset(str(path), question="What is the total Revenue?").frame.shape == (2, 2)
