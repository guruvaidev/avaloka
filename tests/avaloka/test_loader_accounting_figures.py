"""A workbook's "(1,234)" is the number -1234.

Benchmark task ``xlsx_parenthesised_negatives`` (a declared known defect until
this change): a ledger export writes costs as the text ``(1,234)`` beside
positives stored as real numbers. ``_mostly_numeric`` asked ``pd.to_numeric``,
which reads neither the brackets nor the thousands separators, so two thirds
of the Movement column failed MOSTLY_TYPED_SHARE, the column stayed text, and
every statistic skipped it -- or a total read the costs as positive. The upload
path has read the brackets as a minus sign since ``table_shape._as_numeric``.

What these pin:

* the workbook read types the column and gets the sign right, and says so (a
  ``accounting`` count on the coercion record, a Data Scout issue);
* a label in brackets -- ``(a)``, ``(n/a)`` -- is never a number, and a decimal
  comma ``1,2`` is not a thousands separator;
* a column whose brackets hold real labels as well stays text, as before;
* CSV is unchanged: the CLI's CSV reader is held cell for cell to the
  platform's (tests/test_cli_null_policy_parity.py), and neither reads these
  forms from a CSV.

The first four tests fail on the base loader.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("openpyxl", reason="writing the workbooks needs openpyxl")

from avaloka.benchmark.datasets_spreadsheet import (  # noqa: E402
    _ledger_net_movement,
    build_parenthesised_negatives,
)
from avaloka.fireflies.data_scout import _loader_issues  # noqa: E402
from avaloka.io.loader import load_dataset  # noqa: E402


def _workbook(tmp_path, columns: dict, name="book.xlsx"):
    path = tmp_path / name
    pd.DataFrame(columns).to_excel(path, index=False)
    return path


def test_the_benchmark_ledger_reads_with_the_right_sign(tmp_path):
    path = build_parenthesised_negatives(tmp_path / "ledger.xlsx")
    handle = load_dataset(str(path))
    movement = handle.frame["Movement"]

    assert pd.api.types.is_numeric_dtype(movement), movement.dtype
    assert movement.sum() == pytest.approx(_ledger_net_movement())
    assert movement.isna().sum() == 0
    # Closing = opening + movement on every row, which only holds if the
    # brackets were read as a minus sign.
    frame = handle.frame
    assert (frame["Opening balance"] + frame["Movement"]
            - frame["Closing balance"]).abs().max() == pytest.approx(0.0)


def test_the_read_is_recorded_and_reported(tmp_path):
    path = _workbook(tmp_path, {
        "Account": ["a", "b", "c", "d"],
        "Movement": [1200.0, "(1,234)", "(56.5)", 300.0],
    })
    handle = load_dataset(str(path))

    assert handle.frame["Movement"].tolist() == [1200.0, -1234.0, -56.5, 300.0]
    record = handle.structure["coerced_columns"]["Movement"]
    assert record["kind"] == "numeric" and record["accounting"] == 2 and record["cells"] == 0

    issues = {i["kind"]: i for i in _loader_issues(handle)}
    assert issues["accounting_figures_read_as_numbers"]["column"] == "Movement"
    assert "2 cell(s)" in issues["accounting_figures_read_as_numbers"]["detail"]
    # Nothing was read as missing, so nothing may say it was.
    assert "null_words_read_as_missing" not in issues


def test_grouped_thousands_are_numbers_and_gap_words_still_gaps(tmp_path):
    path = _workbook(tmp_path, {
        "Region": ["n", "s", "e", "w", "c"],
        "Revenue": ["1,234,567", "(2,000)", "none", 10.5, "12"],
    })
    handle = load_dataset(str(path))
    revenue = handle.frame["Revenue"]

    assert pd.api.types.is_numeric_dtype(revenue)
    assert revenue.tolist()[:2] == [1234567.0, -2000.0]
    assert revenue.isna().sum() == 1
    record = handle.structure["coerced_columns"]["Revenue"]
    assert record["cells"] == 1 and record["words"] == ["none"] and record["accounting"] == 2


def test_labels_in_brackets_and_decimal_commas_are_not_numbers(tmp_path):
    path = _workbook(tmp_path, {
        "Note": ["(a)", "(b)", "(n/a)", "(c)", "(d)"],
        "Euro": ["1,2", "3,4", "5,6", "7,8", "9,0"],
        "Movement": ["(1,234)", "(2,345)", "(3,456)", "(4,567)", "(5,678)"],
    })
    frame = load_dataset(str(path)).frame

    assert frame["Note"].tolist() == ["(a)", "(b)", "(n/a)", "(c)", "(d)"]
    assert frame["Euro"].tolist() == ["1,2", "3,4", "5,6", "7,8", "9,0"]
    assert frame["Movement"].tolist() == [-1234.0, -2345.0, -3456.0, -4567.0, -5678.0]


def test_brackets_beside_real_labels_stay_text(tmp_path):
    # Two real labels in five cells: under MOSTLY_TYPED_SHARE, so the column is
    # text and its cells are kept exactly as typed.
    path = _workbook(tmp_path, {
        "Status": ["(1,234)", "pending", "(2,000)", "held", 15.0],
    })
    frame = load_dataset(str(path)).frame
    assert not pd.api.types.is_numeric_dtype(frame["Status"])
    assert "(1,234)" in frame["Status"].tolist() and "pending" in frame["Status"].tolist()


def test_csv_is_unchanged(tmp_path):
    path = tmp_path / "ledger.csv"
    path.write_text('Account,Movement\na,1200\nb,"(1,234)"\nc,"(56.5)"\nd,300\n')
    frame = load_dataset(str(path)).frame
    assert not pd.api.types.is_numeric_dtype(frame["Movement"])
    assert "(1,234)" in frame["Movement"].tolist()
