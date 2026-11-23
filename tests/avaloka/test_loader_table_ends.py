"""A workbook's table ends where the table ends.

Four benchmark tasks were declared known defects until this change
(xlsx_totals_row_at_bottom, xlsx_subtotals_and_grand_total,
xlsx_footnote_markers_on_labels, xlsx_second_table_below). ``_read_excel``
read from the header to the last row of the sheet, so a totals row became a
thirteenth region and doubled every sum, subtotals tripled them, a footnote
block became three records with markers left on the names, and a second table
put its text into a numeric column.

The upload path has stopped at each of these since file_handler/table_shape;
the CLI now runs the same stages (its copy, avaloka/io/table_shape.py) and
says what it left out. Rows are left out only on evidence: a totals row whose
numbers add up, notes that read as notes under a blank row, a second table
with a header of its own. Every test here except the last two fails on the
reader before this change by assertion.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("openpyxl", reason="writing the workbooks needs openpyxl")

from avaloka.benchmark.datasets_spreadsheet import (  # noqa: E402
    _PLACES,
    _footnote_headcount,
    _subtotal_records,
    _totals_revenue,
    build_footnote_markers_on_labels,
    build_sec_financial_report,
    build_second_table_below,
    build_subtotals_and_grand_total,
    build_totals_row_at_bottom,
)
from avaloka.fireflies.data_scout import _loader_issues  # noqa: E402
from avaloka.io.loader import load_dataset  # noqa: E402


def _issues(handle) -> dict:
    return {i["kind"]: i for i in _loader_issues(handle)}


def test_a_totals_row_is_left_out_and_said(tmp_path):
    handle = load_dataset(str(build_totals_row_at_bottom(tmp_path / "t.xlsx")))
    frame = handle.frame

    assert len(frame) == 12 and "Total" not in frame["Region"].tolist()
    assert frame["Revenue"].sum() == pytest.approx(sum(_totals_revenue(i) for i in range(12)))
    assert handle.structure["shape"]["aggregate_rows"] == [{"row": 14, "label": "Total", "kind": "total"}]
    assert "row 14 (Total)" in _issues(handle)["aggregate_rows_left_out"]["detail"]


def test_subtotals_and_the_grand_total_are_left_out(tmp_path):
    handle = load_dataset(str(build_subtotals_and_grand_total(tmp_path / "s.xlsx")))
    frame = handle.frame

    assert len(frame) == 12
    assert sorted(frame["Area"].unique()) == ["North", "South", "West"]
    assert frame["Revenue"].sum() == pytest.approx(sum(r[3] for r in _subtotal_records()))
    kinds = [r["kind"] for r in handle.structure["shape"]["aggregate_rows"]]
    assert kinds == ["subtotal", "subtotal", "subtotal", "total"]


def test_a_footnote_block_is_left_out_and_its_markers_come_off(tmp_path):
    handle = load_dataset(str(build_footnote_markers_on_labels(tmp_path / "f.xlsx")))
    frame = handle.frame

    assert list(frame.columns) == ["Segment", "Headcount", "Attrition rate", "Notes"]
    assert len(frame) == 22
    assert frame["Segment"].tolist() == list(_PLACES[:22])
    assert frame.loc[frame["Segment"] == "Bath", "Headcount"].item() == _footnote_headcount(1)
    assert pd.api.types.is_numeric_dtype(frame["Headcount"])
    shape = handle.structure["shape"]
    assert shape["trailing_note_rows"] == 3 and shape["footnote_markers"] == ["*", "(a)", "†"]
    assert "notes_below_table_left_out" in _issues(handle)


def test_a_second_table_is_not_read_as_more_rows(tmp_path):
    handle = load_dataset(str(build_second_table_below(tmp_path / "two.xlsx")))
    frame = handle.frame

    assert len(frame) == 20
    assert pd.api.types.is_numeric_dtype(frame["Hours open"])
    assert handle.structure["shape"]["next_table_row"] == 24
    assert "starting at row 24" in _issues(handle)["second_table_not_read"]["detail"]


def test_null_word_labels_survive_the_cut(tmp_path):
    # The cut is decided on the default read and the same rows are kept from
    # the re-read that keeps null-like words as labels: a region called None
    # is still a region, and the totals row is still gone.
    path = tmp_path / "none.xlsx"
    pd.DataFrame({"Region": ["North", "None", "South", "Total"],
                  "Revenue": [10.0, 20.0, 30.0, 60.0]}).to_excel(path, index=False)
    frame = load_dataset(str(path)).frame
    assert frame["Region"].tolist() == ["North", "None", "South"]


def test_a_row_called_total_that_does_not_add_up_is_a_record(tmp_path):
    path = tmp_path / "tqm.xlsx"
    pd.DataFrame({"Department": ["Sales", "Ops", "Finance", "Total Quality"],
                  "Headcount": [10, 20, 30, 5]}).to_excel(path, index=False)
    handle = load_dataset(str(path))
    assert len(handle.frame) == 4
    assert "aggregate_rows_left_out" not in _issues(handle)
    assert "total_rows_kept_as_records" in _issues(handle)


def test_a_statement_keeps_its_subtotal_line_items(tmp_path):
    # "Total operating expenses" adds up, but with no grand total to check it
    # against it is a line item people ask for, and it stays.
    handle = load_dataset(str(build_sec_financial_report(tmp_path / "10k.xlsx")))
    frame = handle.frame
    assert len(frame) == 13
    assert "Total operating expenses" in frame["Line item"].tolist()
    assert "Cost of revenue" in frame["Line item"].tolist()


def test_a_label_ending_in_a_marker_with_no_note_is_left_as_typed(tmp_path):
    path = tmp_path / "plain.xlsx"
    pd.DataFrame({"Grade (a)": ["x", "y", "z"], "Score": [1, 2, 3]}).to_excel(path, index=False)
    handle = load_dataset(str(path))
    assert list(handle.frame.columns) == ["Grade (a)", "Score"]
    assert "shape" not in handle.structure


def test_csv_is_unchanged(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("Region,Revenue\nNorth,10\nSouth,30\nEast,20\nTotal,60\n")
    assert len(load_dataset(str(path)).frame) == 4
