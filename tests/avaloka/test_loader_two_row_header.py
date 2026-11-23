"""A workbook header written on two rows names its columns from both.

Two benchmark tasks were declared known defects until this change
(xlsx_merged_header_cells, xlsx_stacked_two_row_header). ``_read_excel``
searched for one header row, so a group label merged or repeated across
columns over the labels under it was half a header: the second row became
the first record, its columns came back ``Unnamed: 2`` or ``Sales.1``, and
every numeric column was typed from a row of labels.

The upload path has read the span since file_handler/table_shape
(``detect_header_rows`` and ``_two_row_header_names``); the CLI now asks the
same question through its copy. A one-row header is left to the CLI's own
search, so its names are exactly what they were. The first three tests fail
on the reader before this change by assertion; the rest pin what must not
change.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("openpyxl", reason="writing the workbooks needs openpyxl")

from avaloka.benchmark.datasets_spreadsheet import (  # noqa: E402
    build_merged_header_cells,
    build_owner_file_shape,
    build_stacked_two_row_header,
    build_year_columns_header,
)
from avaloka.io.loader import load_dataset  # noqa: E402


def test_a_merged_group_label_names_the_columns_under_it(tmp_path):
    handle = load_dataset(str(build_merged_header_cells(tmp_path / "m.xlsx")))
    frame = handle.frame

    assert list(frame.columns) == ["Store", "Transactions Count", "Transactions Value",
                                   "Refunds Count", "Refunds Value"]
    assert len(frame) == 24
    assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns[1:])
    assert handle.read["header_row"] == 1          # the first row of the two, 1-based


def test_a_repeated_group_label_names_the_columns_under_it(tmp_path):
    frame = load_dataset(str(build_stacked_two_row_header(tmp_path / "s.xlsx"))).frame

    assert list(frame.columns) == ["Region", "Sales 2022", "Sales 2023", "Cost 2022", "Cost 2023"]
    assert len(frame) == 24
    assert frame["Sales 2022"].iloc[0] == 480_000.0


def test_a_two_row_header_and_a_totals_row_together(tmp_path):
    # The table's end is found under the second header row, not the first.
    path = build_stacked_two_row_header(tmp_path / "s.xlsx")
    import openpyxl
    book = openpyxl.load_workbook(path)
    sheet = book.active
    last = sheet.max_row
    sheet.cell(row=last + 1, column=1, value="Total")
    for column in range(2, 6):
        sheet.cell(row=last + 1, column=column,
                   value=sum(sheet.cell(row=r, column=column).value for r in range(3, last + 1)))
    book.save(path)

    handle = load_dataset(str(path))
    assert len(handle.frame) == 24
    assert handle.structure["shape"]["aggregate_rows"][0]["label"] == "Total"


def test_a_one_row_header_is_read_exactly_as_before(tmp_path):
    owner = load_dataset(str(build_owner_file_shape(tmp_path / "o.xlsx")))
    assert owner.read["header_row"] == 3
    assert list(owner.frame.columns) == ["Segment", "Start (s)", "End (s)",
                                         "Foresight (intended action)", "Hindsight (achieved)",
                                         "Optimality (1-5)"]
    years = load_dataset(str(build_year_columns_header(tmp_path / "y.xlsx"))).frame
    assert list(years.columns) == ["Region", "2021", "2022", "2023"]


def test_two_rows_of_text_are_a_header_and_a_record(tmp_path):
    # An all-text sheet: the second row is a record, not half a header,
    # because there is no number under it.
    path = tmp_path / "text.xlsx"
    pd.DataFrame({"Name": ["Ana", "Ben", "Cal"], "Team": ["Red", "Red", "Blue"],
                  "Role": ["Lead", "Dev", "Dev"]}).to_excel(path, index=False)
    frame = load_dataset(str(path)).frame
    assert list(frame.columns) == ["Name", "Team", "Role"] and len(frame) == 3
