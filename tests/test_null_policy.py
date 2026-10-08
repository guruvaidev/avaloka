"""The null-word load policy (file_handler/null_policy.py) and its first user,
the Excel connector.

A cell that says ``None`` or ``N/A`` is a label in a text column and a gap in a
typed one. Three things are pinned here, in order of how badly they fail:

  1. Table detection is unchanged. A sheet read for the policy must look, to
     header and table detection, exactly as it did before -- otherwise a filler
     row of ``N/A`` stops being blank and a table boundary moves.
  2. Numeric, boolean and date typing is unchanged.
  3. Text columns keep their null-like labels, through to the converted CSV.

Workbooks are written with pandas/openpyxl into tmp_path; no fixtures on disk.
"""

from __future__ import annotations

import csv

import pandas as pd
import pandas.testing as pdt
import pytest

from file_handler import null_policy
from file_handler.excel_connector import (
    ExcelConnector,
    _read_sheet,
    export_workbook_sheets_to_csv,
)
from file_handler.table_shape import extract_tables, keep_null_word_labels

WORDS = ["None", "null", "nan", "NaN", "NA", "N/A", "n/a", "NULL"]


def _workbook(tmp_path, rows, name="book.xlsx", sheet="data"):
    path = tmp_path / name
    pd.DataFrame(rows).to_excel(path, index=False, header=False, sheet_name=sheet)
    return str(path)


# ---------------------------------------------------------------------------
# The word list and the one decision
# ---------------------------------------------------------------------------


def test_the_word_list_is_exactly_what_pandas_reads_as_missing():
    """NULL_WORDS is a literal so a pandas upgrade cannot change the policy
    unnoticed. If this fails, pandas changed its list: decide, then update."""
    from pandas._libs.parsers import STR_NA_VALUES

    assert null_policy.NULL_WORDS == set(STR_NA_VALUES) - {""}


@pytest.mark.parametrize("value,expected", [
    ("None", True), ("N/A", True), ("nan", True), ("#N/A", True),
    ("NONE", False), (" NA ", False), ("", False), ("Card", False),
    (None, False), (float("nan"), False), (0, False),
])
def test_a_null_word_is_an_exact_string_match(value, expected):
    assert null_policy.is_null_word(value) is expected


@pytest.mark.parametrize("cells,reader_typed,expected", [
    (["Card", "Cash"], False, True),      # text column: labels stay
    (["Card", "Cash"], True, False),      # the reader typed it: gaps stay gaps
    ([1.0, 2.0], False, False),           # no real string anywhere
    ([True, False], False, False),        # booleans are typed, not text
    ([], False, False),                   # only null-like words: nothing says "label"
    (["  "], False, False),               # whitespace is not a real string
    ([1, "abc"], False, True),            # mixed, but it holds real text
])
def test_the_decision(cells, reader_typed, expected):
    assert null_policy.keeps_null_words(cells, reader_typed) is expected


# ---------------------------------------------------------------------------
# 1. Table detection sees what it always saw
# ---------------------------------------------------------------------------

SHEETS = {
    "plain table with labels in a text column": [
        ["region", "method", "amount"], ["East", "Card", 10], ["West", "N/A", 20], ["North", "None", 30]],
    "a column of N/A between two tables": [
        ["region", "amount", "N/A", "team", "units"], ["East", 10, "N/A", "A", 1],
        ["West", 20, "N/A", "B", 2], ["North", 30, "N/A", "C", 3]],
    "a row of N/A inside a table": [
        ["region", "amount"], ["East", 10], ["West", 20], ["N/A", "N/A"], ["team", "units"], ["A", 1], ["B", 2]],
    "title, a filler row of N/A, then the header": [
        ["Quarterly report", None, None], ["N/A", "N/A", "N/A"], ["region", "method", "amount"],
        ["East", "Card", 10], ["West", "Cash", 20]],
    "N/A in the header row": [
        ["region", "N/A", "amount"], ["East", "x", 10], ["West", "y", 20], ["North", "z", 30]],
    "numeric column with NA cells": [
        ["region", "amount"], ["East", 10], ["West", "NA"], ["North", 30], ["South", "n/a"]],
    "currency text with N/A": [
        ["region", "price"], ["East", "$1,200"], ["West", "N/A"], ["North", "$300"]],
    "rows of NA above the table": [
        ["NA", "NA"], ["NA", "NA"], ["region", "amount"], ["East", 10], ["West", 20]],
    "two stacked tables split by blank rows": [
        ["region", "amount"], ["East", 10], ["West", 20], [None, None], [None, None],
        ["team", "note"], ["A", "None"], ["B", "ok"]],
}


@pytest.mark.parametrize("name", sorted(SHEETS))
@pytest.mark.parametrize("header", [None, 0])
def test_the_blanked_read_is_the_read_pandas_gave_before(tmp_path, name, header):
    """Everything about structure rests on this: what table detection is handed
    is cell-for-cell and dtype-for-dtype the old default read."""
    path = _workbook(tmp_path, SHEETS[name])

    blanked, _words = _read_sheet(path, "data", header=header)

    pdt.assert_frame_equal(blanked, pd.read_excel(path, sheet_name="data", header=header))


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_no_table_boundary_or_header_moves(tmp_path, name):
    path = _workbook(tmp_path, SHEETS[name])
    before = extract_tables(pd.read_excel(path, sheet_name="data", header=None))

    after = ExcelConnector(path).load_all_tables(normalize=False)

    assert [(list(t["dataframe"].columns), t["dataframe"].shape, t["shape_report"]) for t in after] == [
        (list(frame.columns), frame.shape, report) for frame, report in before
    ]


def test_a_naive_uncoerced_read_would_have_moved_them(tmp_path):
    """Why _read_sheet returns a blanked frame at all. If this stops holding,
    table detection no longer depends on those cells being blank and the
    blanking can go."""
    path = _workbook(tmp_path, SHEETS["title, a filler row of N/A, then the header"])
    naive = pd.read_excel(path, sheet_name="data", header=None, keep_default_na=False, na_values=[""])

    (frame, report), = extract_tables(naive)

    assert report["header_row"] is None and list(frame.columns) == ["column_1", "column_2", "column_3"]


# ---------------------------------------------------------------------------
# 2. Typed columns keep their typing and their gaps
# ---------------------------------------------------------------------------


def _single(tmp_path, rows):
    return ExcelConnector(_workbook(tmp_path, rows)).load_dataframe()


@pytest.mark.parametrize("word", WORDS)
def test_a_null_word_in_a_numeric_column_is_missing(tmp_path, word):
    df = _single(tmp_path, [["region", "amount"], ["East", 10], ["West", word], ["North", 30]])

    assert pd.api.types.is_numeric_dtype(df["amount"])
    assert df["amount"].isna().tolist() == [False, True, False]
    assert df["amount"].sum() == 40


def test_a_currency_column_written_as_text_is_still_numeric(tmp_path):
    df = _single(tmp_path, [["region", "price"], ["East", "$1,200"], ["West", "N/A"], ["North", "$300"]])

    assert pd.api.types.is_numeric_dtype(df["price"])
    assert df["price"].tolist()[0] == 1200 and pd.isna(df["price"].tolist()[1])


def test_a_null_word_in_a_boolean_column_is_missing(tmp_path):
    """The connector's normalisation turns a boolean column with a gap into
    1.0 / NaN / 0.0, as it did before this policy. What matters here is that
    the gap is still a gap and no 'NA' label appears in a typed column."""
    df = _single(tmp_path, [["region", "active"], ["East", True], ["West", "NA"], ["North", False]])

    values = df["active"].tolist()
    assert values[0] == 1.0 and values[2] == 0.0 and pd.isna(values[1])
    assert pd.api.types.is_numeric_dtype(df["active"])


def test_a_null_word_in_a_date_column_is_missing(tmp_path):
    df = _single(tmp_path, [["region", "day"], ["East", pd.Timestamp("2024-01-05")],
                            ["West", "N/A"], ["North", pd.Timestamp("2024-03-01")]])

    assert pd.api.types.is_datetime64_any_dtype(df["day"])
    assert df["day"].isna().tolist() == [False, True, False]


def test_a_column_of_nothing_but_null_words_stays_missing(tmp_path):
    """Nothing in the column says they are labels, so it is what it was before:
    a named column with no values."""
    df = _single(tmp_path, [["region", "note", "amount"], ["East", "N/A", 10], ["West", "None", 20]])

    assert list(df.columns) == ["region", "note", "amount"]
    assert df["note"].isna().all()


# ---------------------------------------------------------------------------
# 3. Text columns keep their labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("word", WORDS)
def test_a_null_word_in_a_text_column_is_a_label(tmp_path, word):
    df = _single(tmp_path, [["method", "amount"], ["Card", 10], [word, 20], ["Cash", 30]])

    assert df["method"].tolist() == ["Card", word, "Cash"]
    assert df.groupby("method")["amount"].sum().to_dict() == {"Card": 10, word: 20, "Cash": 30}


def test_an_empty_cell_in_a_text_column_is_still_missing(tmp_path):
    df = _single(tmp_path, [["method", "amount"], ["Card", 10], [None, 20], ["None", 30]])

    assert df["method"].tolist()[0] == "Card" and df["method"].tolist()[2] == "None"
    assert pd.isna(df["method"].tolist()[1])


def test_labels_land_on_their_own_cells_in_an_offset_table(tmp_path):
    """The table starts on row 3 and column B, has a blank row inside it, and
    sits next to a second table. A label must come back in the cell it left."""
    rows = [
        [None, "Sales by method", None, None, None, None],
        [None, None, None, None, None, None],
        [None, "method", "amount", None, "team", "lead"],
        [None, "Card", "NA", None, "A", "None"],
        [None, None, None, None, "B", "Ann"],
        [None, "None", 20, None, "C", "N/A"],
        [None, "N/A", 30, None, "D", "Bo"],
    ]
    tables = ExcelConnector(_workbook(tmp_path, rows)).load_all_tables()

    first, second = (t["dataframe"] for t in tables)
    assert first["method"].tolist() == ["Card", "None", "N/A"]
    assert first["amount"].isna().tolist() == [True, False, False]
    assert second["team"].tolist() == ["A", "B", "C", "D"]
    assert second["lead"].tolist() == ["None", "Ann", "N/A", "Bo"]


def test_a_row_of_nothing_but_null_words_is_still_dropped_as_blank(tmp_path):
    """Known limit, pinned so it is a decision and not a surprise. Table
    detection must keep seeing these cells as blank, and to it a row holding
    only null-like words IS a blank row -- so a record whose label is 'N/A' and
    whose every other cell is empty or a null-like word is dropped, as it was
    before. One real value anywhere in the row keeps it."""
    rows = [["method", "amount"], ["Card", 10], ["N/A", "NA"], ["None", None], ["Cash", 30]]

    df = _single(tmp_path, rows)

    assert df["method"].tolist() == ["Card", "Cash"]


def test_labels_reach_every_way_of_reading_the_sheet(tmp_path):
    path = _workbook(tmp_path, [["method", "amount"], ["Card", 10], ["None", 20], ["N/A", 30]])
    connector = ExcelConnector(path)

    assert [row[0] for row in connector.load_data()] == ["Card", "None", "N/A"]
    assert connector.load_dataframe(normalize=False)["method"].tolist() == ["Card", "None", "N/A"]
    (table,) = connector.load_all_tables()
    assert table["dataframe"]["method"].tolist() == ["Card", "None", "N/A"]


def test_labels_survive_into_the_converted_csv(tmp_path):
    """The loss this fixes was permanent: the CSV written here is the file every
    later step reads, and the label was already an empty cell in it.

    Read back with the csv module on purpose. pd.read_csv still drops these
    words until the CSV readers adopt the policy (null_policy KNOWN LIMITS)."""
    path = tmp_path / "book.xlsx"
    pd.DataFrame({
        "method": ["Card", "None", "N/A", None],
        "amount": [10, 20, "NA", 40],
    }).to_excel(path, index=False)

    (export,), _notes = export_workbook_sheets_to_csv(str(path), tmp_path / "out")

    with open(export.csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["method"] for r in rows] == ["Card", "None", "N/A", ""]
    assert [r["amount"] for r in rows] == ["10.0", "20.0", "", "40.0"]


def test_keep_null_word_labels_leaves_a_frame_with_no_words_alone():
    frame = pd.DataFrame({"a": ["x", None], "b": [1, 2]})
    words = pd.DataFrame({"a": [None, None], "b": [None, None]})

    assert keep_null_word_labels(frame, words) is frame
