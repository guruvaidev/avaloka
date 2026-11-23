"""PR-A: the spreadsheet tasks grade what they ask, and only that.

Two scorer defects made the planted-answer checks unpassable on a perfect read:

* the task goal was the generic "find the table and profile it", which
  goal_analysis answers with an overview and no figure; and
* the totals and subtotals answers carried no ``value``.

Fixing that must not let a misread pass. On those two workbooks the totals row
read as data *equals* the true sum and becomes the column max, so the answer now
declares its statistic and only a ``sum`` key path counts.

Each case runs the real mission twice: on a correct read (the workbook with the
offending part removed) the answer checks pass; on a misread they fail. The
misread was the benchmark's own workbook while the CLI reader got it wrong;
since PR-E and PR-D1 fixed the reader, each misread is built (a sheet removed,
or the reader's table-end stages switched off) so the scorer's bite is still
shown.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from avaloka.benchmark.datasets_spreadsheet import (ANSWER_SHEET, DECOY_SHEET,
                                                    SPREADSHEET_DATASETS)
from avaloka.benchmark.runner import BenchmarkRunner
from avaloka.benchmark.suite import spreadsheet_tasks

openpyxl = pytest.importorskip("openpyxl")


def _task(name: str, builder=None):
    task = next(t for t in spreadsheet_tasks() if t.name == f"xlsx_{name}")
    if builder is not None:
        task = dataclasses.replace(task, dataset=dataclasses.replace(task.dataset, builder=builder))
    return task


def _checks(tmp_path: Path, task) -> dict[str, tuple[bool, str]]:
    card = BenchmarkRunner(tmp_path / task.name).run([task])
    (result,) = card.results
    assert result.error is None, result.error
    return {c.name: (c.passed, c.detail) for c in result.checks}


def _without_last_row(name):
    builder = SPREADSHEET_DATASETS[name][0]

    def build(path: Path) -> Path:
        out = builder(path)
        book = openpyxl.load_workbook(out)
        sheet = book.active
        sheet.delete_rows(sheet.max_row)
        book.save(out)
        return out
    return build


def _answer_sheet_only(path: Path) -> Path:
    out = SPREADSHEET_DATASETS["decoy_sheet_outweighs_the_answer"][0](path)
    book = openpyxl.load_workbook(out)
    del book[DECOY_SHEET]
    book.save(out)
    return out


def test_the_tasks_with_a_planted_answer_ask_its_question():
    goals = {t.name: t.goal for t in spreadsheet_tasks()}
    assert goals["xlsx_totals_row_at_bottom"] == "What is the total Revenue?"
    assert goals["xlsx_decoy_sheet_outweighs_the_answer"] == "What is the total Revenue?"
    assert goals["xlsx_parenthesised_negatives"] == "What is the total Movement?"
    # A task with nothing to answer keeps the generic goal.
    assert goals["xlsx_merged_header_cells"].startswith("Find the table")


def test_totals_answer_passes_on_a_correct_read(tmp_path):
    checks = _checks(tmp_path, _task("totals_row_at_bottom", _without_last_row("totals_row_at_bottom")))
    assert checks["row_count"][0], checks["row_count"][1]
    for name in ("answer_value", "answer_bound"):
        assert checks[name][0], f"{name}: {checks[name][1]}"


@pytest.fixture
def reader_keeps_every_row(monkeypatch):
    """A deliberately broken read: the CLI reader with its table-end stages
    switched off, so totals rows come back as records. Since PR-D1 the
    reader no longer misreads these workbooks, so the misread is made here
    rather than relied on -- the scorer must still fail it."""
    from avaloka.io import loader

    monkeypatch.setattr(loader, "_table_ends",
                        lambda frame, first: (list(range(first, first + len(frame))), [], {}))


def test_totals_answer_fails_when_the_totals_row_is_read_as_data(tmp_path, reader_keeps_every_row):
    checks = _checks(tmp_path, _task("totals_row_at_bottom"))
    passed, detail = checks["answer_value"]
    # The totals row's value equals the true sum and is the profile's max;
    # it must not count as the answer.
    assert not passed and "under a 'sum' key path" in detail, detail


def test_subtotals_answer_passes_on_a_correct_read_and_fails_on_the_misread(tmp_path):
    def records_only(path: Path) -> Path:
        out = SPREADSHEET_DATASETS["subtotals_and_grand_total"][0](path)
        book = openpyxl.load_workbook(out)
        sheet = book.active
        for row in range(sheet.max_row, 1, -1):
            label = " ".join(str(c.value or "") for c in sheet[row]).lower()
            if "subtotal" in label or "grand total" in label:
                sheet.delete_rows(row)
        book.save(out)
        return out
    good = _checks(tmp_path / "good", _task("subtotals_and_grand_total", records_only))
    from avaloka.io import loader
    with pytest.MonkeyPatch.context() as patch:
        # The broken read, as in reader_keeps_every_row: subtotals kept as records.
        patch.setattr(loader, "_table_ends",
                      lambda frame, first: (list(range(first, first + len(frame))), [], {}))
        bad = _checks(tmp_path / "bad", _task("subtotals_and_grand_total"))
    assert good["answer_value"][0] and good["answer_bound"][0], good["answer_value"][1]
    assert not bad["answer_value"][0], bad["answer_value"][1]


def _decoy_sheet_only(path: Path) -> Path:
    """A read that cannot land on the answer: the answer sheet is gone. Since
    the reader chose the sheet by the question (PR-E), the benchmark's own
    workbook is no longer misread, so the misread is built rather than relied
    on -- the scorer must still fail it."""
    out = SPREADSHEET_DATASETS["decoy_sheet_outweighs_the_answer"][0](path)
    book = openpyxl.load_workbook(out)
    del book[ANSWER_SHEET]
    book.save(out)
    return out


def test_decoy_answer_passes_when_the_answer_sheet_is_read(tmp_path):
    good = _checks(tmp_path / "good", _task("decoy_sheet_outweighs_the_answer", _answer_sheet_only))
    assert good["sheet_identified"][0], good["sheet_identified"][1]
    for name in ("answer_value", "answer_bound"):
        assert good[name][0], f"{name}: {good[name][1]}"
    bad = _checks(tmp_path / "bad", _task("decoy_sheet_outweighs_the_answer", _decoy_sheet_only))
    assert not bad["sheet_identified"][0] and not bad["answer_value"][0]
    assert ANSWER_SHEET in bad["sheet_identified"][1]
