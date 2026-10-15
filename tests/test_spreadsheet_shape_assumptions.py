"""A reader that guesses must say what it guessed, and must not trade shapes.

``tests/test_spreadsheet_comprehension_upload_path.py`` grades the reader
through ``ExcelConnector``. Three things it cannot see are pinned here:

* **The browser upload goes through a different function.**
  ``export_workbook_sheets_to_csv`` keeps pandas' default read whenever the
  result looks usable, so a fix in ``table_shape`` alone does not reach it. The
  same fixtures are read through the export here.
* **The shapes that already worked.** A header rule loosened for one sheet can
  quietly break another, so the four fixtures that were right before are held
  to their exact columns, row counts and sheet on both paths.
* **What the user is told.** ``table_shape.reshape`` has always recorded what
  it assumed and nothing ever read it. No reader is right about every
  hand-made spreadsheet; the difference between a wrong answer and a usable
  product is whether the assumption is on the screen.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("openpyxl", reason="the workbook builders need openpyxl")

from avaloka.benchmark.datasets_spreadsheet import SPREADSHEET_DATASETS  # noqa: E402
from file_handler.excel_connector import (  # noqa: E402
    ExcelConnector,
    describe_sheet_choice,
    export_workbook_sheets_to_csv,
)
from file_handler.table_shape import (  # noqa: E402
    describe_shape,
    detect_header_row,
    detect_header_rows,
    reshape,
)

NA = float("nan")


def frame(rows):
    return pd.DataFrame(rows)


def _build(name, tmp_path):
    builder, truth = SPREADSHEET_DATASETS[name]
    path = tmp_path / f"{name}.xlsx"
    builder(path)
    return str(path), truth


def _exported(name, tmp_path):
    """The sheet the ground truth describes, as the upload path wrote it."""
    path, truth = _build(name, tmp_path)
    exports, notes = export_workbook_sheets_to_csv(path, tmp_path / "out")
    wanted = truth.sheet_name
    export = next(e for e in exports if wanted in (None, e.sheet_name)) if wanted else exports[0]
    return pd.read_csv(export.csv_path), export, exports, notes, truth


# ---------------------------------------------------------------------------
# The shapes that already worked must still work, on both paths
# ---------------------------------------------------------------------------

#: name -> (sheet the connector lands on, columns in order, data rows).
#: Written out rather than read from the ground truth: a pin that follows the
#: fixture would follow it into a regression too.
ALREADY_WORKING = {
    "owner_file_shape": (
        "Segments",
        ["Segment", "Start (s)", "End (s)", "Foresight (intended action)",
         "Hindsight (achieved)", "Optimality (1-5)"],
        139),
    "thousands_scaling_in_title": (
        "Revenue", ["Region", "Revenue", "Direct cost"], 24),
    "parenthesised_negatives": (
        "Ledger", ["Account", "Opening balance", "Movement", "Closing balance"], 28),
    "multi_sheet_data_on_sheet_two": (
        "Usage", ["Account", "Plan", "Seats", "Monthly cost", "Active users"], 30),
}


@pytest.mark.parametrize("name", sorted(ALREADY_WORKING))
def test_a_shape_that_worked_still_reads_the_same_through_the_connector(name, tmp_path):
    sheet, columns, n_rows = ALREADY_WORKING[name]
    path, _truth = _build(name, tmp_path)
    connector = ExcelConnector(path)
    got = connector.load_dataframe()
    assert connector.selected_sheet == sheet
    assert [str(c) for c in got.columns] == columns
    assert len(got) == n_rows


@pytest.mark.parametrize("name", sorted(ALREADY_WORKING))
def test_a_shape_that_worked_still_reads_the_same_through_the_upload(name, tmp_path):
    sheet, columns, n_rows = ALREADY_WORKING[name]
    path, _truth = _build(name, tmp_path)
    exports, _notes = export_workbook_sheets_to_csv(path, tmp_path / "out")
    assert exports[0].sheet_name == sheet
    got = pd.read_csv(exports[0].csv_path)
    assert list(got.columns) == columns
    assert len(got) == n_rows == exports[0].n_rows


def test_an_ordinary_sheet_is_read_without_comment(tmp_path):
    """Nothing was assumed, so nothing is said -- or the notes become noise
    and the one that matters is not read."""
    path, _truth = _build("parenthesised_negatives", tmp_path)
    assert ExcelConnector(path).shape_notes == []
    (export,), notes = export_workbook_sheets_to_csv(path, tmp_path / "out")
    assert export.notes == () and notes == []


# ---------------------------------------------------------------------------
# The same fixes, through the function the browser upload actually calls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "footnote_markers_on_labels",
    "merged_header_cells",
    "second_table_below",
    "stacked_two_row_header",
    "year_columns_header",
])
def test_the_upload_path_reads_the_table_that_is_in_the_file(name, tmp_path):
    got, export, _exports, _notes, truth = _exported(name, tmp_path)
    assert list(got.columns) == truth.columns
    assert len(got) == truth.n_rows == export.n_rows


def test_the_ten_k_income_statement_reads_right_once_its_sheet_is_named(tmp_path):
    """Sheet choice is the only thing still wrong on the 10-K.

    The connector still lands on the balance sheet (see the comprehension
    test); named explicitly, the statement comes back whole: 13 line items,
    the footnote row gone, and ``Cost of revenue (a)`` as ``Cost of revenue``.
    """
    path, truth = _build("sec_financial_report", tmp_path)
    got = ExcelConnector(path, sheet_name=truth.sheet_name).load_dataframe()
    assert [str(c) for c in got.columns] == truth.columns
    assert len(got) == truth.n_rows
    assert list(got[truth.columns[0]]) == truth.planted_answer["line_items"]
    planted = truth.planted_answer
    row = got[got[truth.columns[0]] == planted["label"]]
    assert row.iloc[0][planted["period"]] == pytest.approx(planted["value"])


def test_footnote_markers_come_off_the_row_labels_too(tmp_path):
    got, _export, _exports, _notes, truth = _exported("footnote_markers_on_labels", tmp_path)
    assert list(got["Segment"]) == truth.planted_answer["row_labels"]


# ---------------------------------------------------------------------------
# A header is a span of rows, not an index
# ---------------------------------------------------------------------------

class TestHeaderSpan:
    def test_an_ordinary_header_is_a_span_of_one(self):
        raw = frame([["Report", NA, NA], ["a", "b", "c"], [1, 2, 3]])
        assert detect_header_rows(raw) == (1, 1)
        assert detect_header_row(raw) == 1

    def test_a_repeated_group_label_over_a_label_row_is_two_rows(self):
        raw = frame([["Region", "Sales", "Sales", "Cost", "Cost"],
                     [NA, 2022, 2023, 2022, 2023],
                     ["Leeds", 480000, 503000, 297600, 296770]])
        assert detect_header_rows(raw) == (0, 1)
        # the single-index caller is told where the header starts
        assert detect_header_row(raw) == 0

    def test_a_merged_group_label_is_carried_onto_each_column_under_it(self):
        raw = frame([["Store", "Transactions", NA, "Refunds", NA],
                     [NA, "Count", "Value", "Count", "Value"],
                     ["Leeds", 1200, 13680, 18, 217.8],
                     ["Bath", 1237, 14102.8, 23, 279.3]])
        table, report = reshape(raw)
        assert list(table.columns) == [
            "Store", "Transactions Count", "Transactions Value",
            "Refunds Count", "Refunds Value"]
        assert len(table) == 2
        assert report["header_rows"] == [0, 1] and report["header_row"] == 0

    def test_the_row_label_may_sit_on_the_second_row(self):
        raw = frame([[NA, "Transactions", NA, "Refunds", NA],
                     ["Store", "Count", "Value", "Count", "Value"],
                     ["Leeds", 1200, 13680, 18, 217.8]])
        table, _report = reshape(raw)
        assert list(table.columns)[:2] == ["Store", "Transactions Count"]

    def test_a_wide_merged_group_row_is_found_above_its_labels(self):
        """Three columns per group leaves the group row mostly blank, which is
        what a title looks like to the scan."""
        raw = frame([["Store", "Transactions", NA, NA, "Refunds", NA, NA],
                     [NA, "Count", "Value", "Avg", "Count", "Value", "Avg"],
                     ["Leeds", 1200, 13680, 11.4, 18, 217.8, 12.1]])
        table, report = reshape(raw)
        assert list(table.columns) == [
            "Store", "Transactions Count", "Transactions Value", "Transactions Avg",
            "Refunds Count", "Refunds Value", "Refunds Avg"]
        assert report["header_rows"] == [0, 1] and len(table) == 1

    # -- what must NOT become a two-row header -----------------------------

    def test_a_title_over_an_ordinary_header_stays_a_title(self):
        raw = frame([["Report", "2023-Q4", NA, NA], ["a", "b", "c", "d"], [1, 2, 3, 4]])
        table, report = reshape(raw)
        assert list(table.columns) == ["a", "b", "c", "d"]
        assert report["header_rows"] == [1]

    def test_a_title_over_repeated_labels_does_not_become_their_group(self):
        raw = frame([["Report", NA, NA, NA], ["x", "y", "x", "y"], [1, 2, 3, 4]])
        assert detect_header_rows(raw) is None


    def test_repeated_labels_over_numbers_are_still_a_data_row(self):
        raw = frame([["x", "x", "x"], [1, 2, 3], [4, 5, 6]])
        assert detect_header_rows(raw) is None

    def test_two_rows_of_an_all_text_sheet_are_not_a_header(self):
        """A banded key/value sheet repeats itself and has blanks. Without a
        number underneath there is nothing to tell labels from records."""
        raw = frame([["yes", "yes", "no"], ["no", "yes", NA], ["a", "b", "c"]])
        assert detect_header_rows(raw) is None

    def test_a_blank_header_cell_over_records_stays_one_row(self):
        raw = frame([["Name", NA, "Amount"], ["Ann", "x", 12], ["Bob", "y", 14]])
        assert detect_header_rows(raw) == (0, 0)

    def test_a_group_missing_a_label_under_one_column_is_not_a_group(self):
        raw = frame([["Store", NA, "Refunds"], [NA, "Count", "Value"], ["Leeds", 3, 4]])
        assert detect_header_rows(raw) == (0, 0)


class TestYearsAsColumnNames:
    def test_a_row_of_years_beside_a_label_is_a_header(self):
        raw = frame([["Region", 2021, 2022, 2023], ["Leeds", 310000, 331500, 354700]])
        table, report = reshape(raw)
        assert list(table.columns) == ["Region", "2021", "2022", "2023"]
        assert len(table) == 1
        assert report["numeric_header_cells"] == ["2021", "2022", "2023"]

    def test_a_float_year_is_named_as_a_person_would_say_it(self):
        raw = frame([["Region", 2021.0, 2022.0], ["Leeds", 1.5, NA]])
        assert list(reshape(raw)[0].columns) == ["Region", "2021", "2022"]

    def test_records_that_contain_years_are_not_a_header(self):
        raw = frame([["Acme", 1999, 2004], ["Bolt", 2001, 2010]])
        assert detect_header_rows(raw) is None

    def test_one_year_is_not_enough_to_call_a_row_a_header(self):
        raw = frame([["Founded", 1999], ["Employees", 250]])
        assert detect_header_rows(raw) is None

    def test_a_row_of_only_years_is_still_data(self):
        raw = frame([[2021, 2022, 2023], [4, 5, 6]])
        assert detect_header_rows(raw) is None


class TestWhatLiesUnderTheTable:
    def test_notes_under_a_blank_row_are_not_records(self):
        raw = frame([["Segment*", "Headcount"], ["Leeds*", 48], ["Bath", 61],
                     [NA, NA], ["* excludes contractors", NA]])
        table, report = reshape(raw)
        assert list(table.columns) == ["Segment", "Headcount"]
        assert list(table["Segment"]) == ["Leeds", "Bath"]
        assert report["trailing_notes"] == ["* excludes contractors"]
        assert report["footnote_markers"] == ["*"]

    def test_a_marker_no_note_defines_is_left_as_typed(self):
        raw = frame([["Plan (a)", "Seats*"], ["Basic*", 5], ["Pro (a)", 9]])
        table, report = reshape(raw)
        assert list(table.columns) == ["Plan (a)", "Seats*"]
        assert list(table["Plan (a)"]) == ["Basic*", "Pro (a)"]
        assert report["footnote_markers"] == []

    def test_a_one_cell_row_with_no_blank_row_above_it_is_a_record(self):
        raw = frame([["Name", "Amount"], ["Ann", 1], ["Bob", 2], ["Cy", NA]])
        table, report = reshape(raw)
        assert len(table) == 3 and report["trailing_note_rows"] == 0

    def test_a_row_that_reads_like_a_note_but_touches_the_table_is_a_record(self):
        """The blank row is the half of the rule the wording cannot stand in
        for: a text column may hold a sentence in its last record."""
        raw = frame([["Name", "Amount"], ["Ann", 1], ["Bob", 2],
                     ["Note: Cy joined after the period closed", NA]])
        table, report = reshape(raw)
        assert len(table) == 3 and report["trailing_note_rows"] == 0

    def test_a_full_row_under_a_blank_row_is_not_a_note(self):
        raw = frame([["Name", "Amount"], ["Ann", 1], ["Bob", 2], [NA, NA], ["Cy", 7]])
        table, _report = reshape(raw)
        assert list(table["Name"]) == ["Ann", "Bob", "Cy"]

    def test_one_cell_records_under_a_blank_row_are_not_notes(self):
        """A checklist whose last items have nothing ticked. Dropping these
        would be losing the user's rows, so a note has to read like one."""
        raw = frame([["Task", "Done"], ["a", "yes"], ["b", "no"], [NA, NA], ["c", NA], ["d", NA]])
        table, report = reshape(raw)
        assert list(table["Task"]) == ["a", "b", "c", "d"]
        assert report["trailing_note_rows"] == 0

    def test_a_source_line_is_a_note(self):
        raw = frame([["Name", "Amount"], ["Ann", 1], ["Bob", 2], [NA, NA], ["Source: ONS 2023", NA]])
        table, report = reshape(raw)
        assert len(table) == 2 and report["trailing_notes"] == ["Source: ONS 2023"]

    def test_a_hash_is_part_of_a_label_not_a_footnote(self):
        raw = frame([["Order #", "Amount"], ["A1", 1], ["A2", 2], [NA, NA],
                     ["# excludes cancelled orders and test orders", NA]])
        table, _report = reshape(raw)
        assert list(table.columns) == ["Order #", "Amount"]

    def test_a_second_table_is_not_more_rows_of_the_first(self):
        raw = frame([["Ticket", "Queue", "Hours"], ["T1", "billing", 1.5], ["T2", "ops", 4],
                     [NA, NA, NA], [NA, NA, NA],
                     ["Queue", "Target", "Owner"], ["billing", 8, "Leeds team"],
                     ["ops", 4, "Bath team"]])
        table, report = reshape(raw)
        assert list(table["Ticket"]) == ["T1", "T2"]
        assert report["rows_below_table"] == 3
        assert report["next_table_row"] == 6          # as the spreadsheet numbers it

    def test_a_narrower_second_table_is_told_by_the_numbers_under_it(self):
        raw = frame([["Ticket", "Queue", "Hours"], ["T1", "billing", 1.5], ["T2", "ops", 4],
                     [NA, NA, NA], [NA, NA, NA],
                     ["Queue", "Target", NA], ["billing", 8, NA], ["ops", 4, NA]])
        table, report = reshape(raw)
        assert list(table["Ticket"]) == ["T1", "T2"] and report["rows_below_table"] == 3

    def test_a_gap_followed_by_more_records_is_one_table(self):
        raw = frame([["Ticket", "Hours"], ["T1", 1.5], [NA, NA], [NA, NA], ["T2", 4], ["T3", 6]])
        table, report = reshape(raw)
        assert list(table["Ticket"]) == ["T1", "T2", "T3"]
        assert report["rows_below_table"] == 0

    def test_a_gap_in_a_table_of_text_is_still_one_table(self):
        """Every record of an all-text table looks like a header. Nothing here
        says a new table started, so no row may be left out."""
        raw = frame([["Name", "City"], ["Ann", "Leeds"], ["Bob", "Bath"], [NA, NA], [NA, NA],
                     ["Cy", "York"], ["Di", "Hull"]])
        table, report = reshape(raw)
        assert list(table["Name"]) == ["Ann", "Bob", "Cy", "Di"]
        assert report["rows_below_table"] == 0


# ---------------------------------------------------------------------------
# Totals rows: the shape that leaves no trace
# ---------------------------------------------------------------------------
# Right names, right types, a frame that prints cleanly -- and every sum is
# double. Nothing has to be repaired, so nothing used to be said.

AGGREGATES = ["totals_row_at_bottom", "subtotals_and_grand_total"]


def _through_the_connector(name, tmp_path):
    path, truth = _build(name, tmp_path)
    connector = ExcelConnector(path)
    return connector.load_dataframe(), connector.shape_notes, truth


def _through_the_upload(name, tmp_path):
    got, export, _exports, _notes, truth = _exported(name, tmp_path)
    return got, list(export.notes), truth


@pytest.mark.parametrize("read", [_through_the_connector, _through_the_upload],
                         ids=["connector", "upload"])
@pytest.mark.parametrize("name", AGGREGATES)
def test_a_totals_row_is_not_counted_as_data(name, read, tmp_path):
    got, _notes, truth = read(name, tmp_path)
    planted = truth.planted_answer
    assert len(got) == truth.n_rows
    # the three statistics a totals row corrupts, each against the file
    assert got["Revenue"].sum() == pytest.approx(planted["sum"])
    assert got["Revenue"].max() == pytest.approx(planted["max"])
    assert got["Units"].sum() == planted["units_sum"]


@pytest.mark.parametrize("read", [_through_the_connector, _through_the_upload],
                         ids=["connector", "upload"])
def test_subtotal_rows_do_not_become_groups(read, tmp_path):
    got, _notes, truth = read("subtotals_and_grand_total", tmp_path)
    assert sorted(got["Area"].unique()) == sorted(truth.planted_answer["groups"])


@pytest.mark.parametrize("read", [_through_the_connector, _through_the_upload],
                         ids=["connector", "upload"])
def test_the_user_is_told_which_row_was_left_out_and_why(read, tmp_path):
    _got, notes, _truth = read("totals_row_at_bottom", tmp_path)
    (said,) = notes
    assert "Row 14 ('Total') equals the sum of the rows above it" in said
    assert "left out of the data" in said


@pytest.mark.parametrize("read", [_through_the_connector, _through_the_upload],
                         ids=["connector", "upload"])
def test_each_subtotal_left_out_is_named(read, tmp_path):
    _got, notes, _truth = read("subtotals_and_grand_total", tmp_path)
    said = " ".join(notes)
    assert "4 rows were read as subtotals and a grand total" in said
    for row in ("row 7 ('North subtotal')", "row 12 ('South subtotal')",
                "row 16 ('West subtotal')", "row 17 ('Grand Total')"):
        assert row in said


SALES = [["Region", "Units", "Revenue"],
         ["Leeds", 100, 1000.0], ["Bath", 200, 2000.0],
         ["York", 300, 3000.0], ["Hull", 400, 4000.0]]


def _with_last_row(*last):
    return frame(SALES + [list(last)])


class TestWhatCountsAsATotal:
    """The arithmetic decides; the label is only a hint. Every case below is a
    row that must stay, except the ones that say otherwise."""

    def test_the_minimal_case(self):
        table, report = reshape(_with_last_row("Total", 1000, 10000.0))
        assert len(table) == 4 and table["Revenue"].sum() == 10000.0
        assert report["aggregate_rows"] == [{"row": 6, "label": "Total", "kind": "total"}]

    @pytest.mark.parametrize("label", [
        "TOTAL", "Total:", "Grand Total", "Sub-total", "Totals", "Sum",
        "Gesamt", "Gesamtsumme", "Summe", "Totaal", "Somme", "Итого", "合计"])
    def test_the_label_may_be_shouted_punctuated_or_not_english(self, label):
        table, _report = reshape(_with_last_row(label, 1000, 10000.0))
        assert len(table) == 4

    def test_a_row_merely_called_total_is_data(self):
        """A department, not a sum. Its numbers are its own."""
        table, report = reshape(_with_last_row("Total Quality Management", 55, 640.0))
        assert list(table["Region"])[-1] == "Total Quality Management"
        assert len(table) == 5 and report["aggregate_rows"] == []

    def test_a_row_called_total_in_the_middle_is_data(self):
        raw = frame(SALES[:3] + [["Total Quality Management", 55, 640.0]] + SALES[3:])
        table, report = reshape(raw)
        assert len(table) == 5 and report["aggregate_rows"] == []

    def test_a_total_that_does_not_add_up_is_kept_and_said_to_be(self):
        table, report = reshape(_with_last_row("Total", 999, 12345.0))
        assert len(table) == 5
        assert report["labelled_total_rows_kept"] == [{"row": 6, "label": "Total"}]
        said = " ".join(describe_shape(report, list(table.columns)))
        assert "Row 6 ('Total') is labelled like a total" in said
        assert "kept as data" in said and "count those figures twice" in said

    def test_one_agreeing_column_is_enough_under_a_label(self):
        """A totals row often carries an average beside its sums."""
        raw = frame([["Region", "Units", "Price"], ["Leeds", 100, 9.5], ["Bath", 200, 10.5],
                     ["York", 300, 11.0], ["Total", 600, 10.33]])
        table, _report = reshape(raw)
        assert len(table) == 3

    def test_an_unlabelled_row_is_a_total_only_if_every_number_adds_up(self):
        table, report = reshape(_with_last_row(NA, 1000, 10000.0))
        assert len(table) == 4
        assert report["aggregate_rows"] == [{"row": 6, "label": "", "kind": "total"}]

    def test_an_unlabelled_row_with_one_number_out_is_data(self):
        table, _report = reshape(_with_last_row("Wick", 1000, 4321.0))
        assert len(table) == 5

    def test_one_agreeing_number_and_no_label_is_a_coincidence(self):
        raw = frame([["Step", "Count"], ["a", 1], ["b", 2], ["c", 3], ["d", 6]])
        assert len(reshape(raw)[0]) == 4

    def test_a_column_of_zeros_is_not_evidence(self):
        raw = frame([["Region", "Returns"], ["Leeds", 0], ["Bath", 0], ["York", 0],
                     ["Total Quality Management", 0]])
        assert len(reshape(raw)[0]) == 4

    def test_a_total_typed_to_the_penny_still_adds_up(self):
        raw = frame([["Item", "Cost"], ["a", 0.105], ["b", 0.105], ["c", 0.105], ["Total", 0.32]])
        assert len(reshape(raw)[0]) == 3

    def test_a_total_out_by_more_than_rounding_does_not(self):
        raw = frame([["Item", "Cost"], ["a", 10.0], ["b", 20.0], ["c", 30.0], ["Total", 60.5]])
        assert len(reshape(raw)[0]) == 4

    def test_formatted_numbers_are_added_as_numbers(self):
        raw = frame([["Account", "Movement"], ["a", "1,200"], ["b", "(200)"], ["c", "500"],
                     ["Total", "1,500"]])
        table, _report = reshape(raw)
        assert list(table["Account"]) == ["a", "b", "c"]

    def test_a_table_too_short_to_tell_is_left_alone(self):
        raw = frame([["Region", "Units"], ["Leeds", 100], ["Total", 100]])
        assert len(reshape(raw)[0]) == 2

    def test_a_statement_keeps_the_totals_that_are_its_line_items(self):
        """In a statement a subtotal is a fact people ask for, and nothing
        checks it: no grand total closes the groups."""
        raw = frame([["Line item", "FY2023", "FY2022"], ["Sales", 100, 90], ["Services", 50, 40],
                     ["Total revenue", 150, 130], ["Rent", 40, 35], ["Wages", 60, 55],
                     ["Total costs", 100, 90], ["Net income", 50, 40]])
        table, report = reshape(raw)
        assert len(table) == 7 and report["aggregate_rows"] == []
        assert "Total revenue" in list(table["Line item"])

    def test_subtotals_with_no_grand_total_are_left_in__a_known_gap(self):
        """NOT the right answer: the two subtotal rows are counted as data and
        Units sums to 2000, not 1000. They are left because without a grand
        total nothing distinguishes this from the statement above, and taking
        line items out of a statement is the worse mistake. What the reader
        does instead is say so. If this starts failing because the rows are
        now excluded, that is progress -- re-point the assertions."""
        raw = frame([["Area", "Store", "Units"],
                     ["North", "Leeds", 100], ["North", "York", 200], ["North subtotal", NA, 300],
                     ["South", "Bath", 300], ["South", "Hull", 400], ["South subtotal", NA, 700]])
        table, report = reshape(raw)
        assert len(table) == 6 and table["Units"].sum() == 2000
        assert [r["label"] for r in report["labelled_total_rows_kept"]] == [
            "North subtotal", "South subtotal"]
        assert "count those figures twice" in " ".join(describe_shape(report))

    GROUPED = [["Area", "Store", "Units"],
               ["North", "Leeds", 100], ["North", "York", 200], ["North subtotal", NA, 300],
               ["South", "Bath", 300], ["South", "Hull", 400], ["South subtotal", NA, 700]]

    def test_a_grand_total_that_does_not_check_takes_nothing_out(self):
        """The subtotals add up, but the row that should confirm them does
        not. Unconfirmed, they are what they are in a statement: rows."""
        table, report = reshape(frame(self.GROUPED + [["Grand Total", NA, 1234]]))
        assert len(table) == 7 and report["aggregate_rows"] == []
        assert len(report["labelled_total_rows_kept"]) == 3

    def test_a_record_called_total_inside_a_group_is_not_a_subtotal(self):
        """Arithmetic, not the label, even for rows in the middle."""
        rows = [["Area", "Dept", "Units"],
                ["North", "Sales", 100], ["North", "Total Quality Management", 200],
                ["North subtotal", NA, 300],
                ["South", "Sales", 300], ["South", "Ops", 400], ["South subtotal", NA, 700],
                ["Grand Total", NA, 1000]]
        table, report = reshape(frame(rows))
        assert "Total Quality Management" in list(table["Dept"])
        assert len(table) == 4 and table["Units"].sum() == 1000
        assert [r["kind"] for r in report["aggregate_rows"]] == ["subtotal", "subtotal", "total"]

    def test_a_last_group_with_no_subtotal_is_still_checked_by_the_grand_total(self):
        table, report = reshape(frame(self.GROUPED[:6] + [["Grand Total", NA, 1000]]))
        assert len(table) == 4 and table["Units"].sum() == 1000
        assert [r["label"] for r in report["aggregate_rows"]] == ["North subtotal", "Grand Total"]

    def test_a_balance_sheet_is_told_its_totals_were_kept(self, tmp_path):
        """Six of its seventeen rows are totals of other rows, in a structure
        no single check confirms. They stay -- they are what gets asked about
        -- and the reader says a column sum will count them twice."""
        path, _truth = _build("sec_financial_report", tmp_path)
        connector = ExcelConnector(path, sheet_name="Consolidated Balance Sheets")
        assert len(connector.load_dataframe()) == 17
        kept = [r["label"] for r in connector.shape_report["labelled_total_rows_kept"]]
        assert "Total assets" in kept and "Total liabilities" in kept
        assert "count those figures twice" in " ".join(connector.shape_notes)

    def test_a_word_common_in_the_data_is_not_reported_row_by_row(self):
        rows = [["Metric", "Value"]] + [[f"Total sales {i}", i * 3 + 1] for i in range(40)]
        table, report = reshape(frame(rows))
        assert len(table) == 40 and report["labelled_total_rows_kept"] == []
        assert describe_shape(report, list(table.columns)) == []


def test_null_word_labels_stay_on_their_own_rows_when_a_total_is_removed(tmp_path):
    """The upload restores cells like 'None' by position. Taking a row out
    must not slide them onto a neighbour."""
    path = tmp_path / "plans.xlsx"
    rows = [["Plan", "Tier", "Seats"], ["a", "None", 1], ["b", "Gold", 2], ["c", "None", 3],
            ["d", "Gold", 4], ["Total", "None", 10]]
    pd.DataFrame(rows[1:], columns=rows[0]).to_excel(path, index=False)
    (export,), _notes = export_workbook_sheets_to_csv(str(path), tmp_path / "out")
    got = pd.read_csv(export.csv_path, keep_default_na=False)
    assert list(got["Plan"]) == ["a", "b", "c", "d"]
    assert list(got["Tier"]) == ["None", "Gold", "None", "Gold"]


# ---------------------------------------------------------------------------
# What the user is told
# ---------------------------------------------------------------------------

def _said(name, tmp_path):
    path, _truth = _build(name, tmp_path)
    return " ".join(ExcelConnector(path).shape_notes)


def test_the_header_row_and_the_rows_above_it_are_named(tmp_path):
    said = _said("thousands_scaling_in_title", tmp_path)
    assert "Row 2 was taken as the header" in said
    # the skipped title is quoted: it is where "£ thousands" lives
    assert "Revenue by region (£ thousands)" in said
    assert "not read as data" in said


def test_a_two_row_header_is_said_to_be_one(tmp_path):
    said = _said("stacked_two_row_header", tmp_path)
    assert "Rows 1 and 2 were read together as one header" in said
    assert "Cost 2023" in said


def test_years_taken_as_names_are_said_to_be(tmp_path):
    said = _said("year_columns_header", tmp_path)
    assert "'2021', '2022', '2023'" in said and "column names, not as values" in said


def test_notes_and_markers_removed_are_said_to_be(tmp_path):
    said = _said("footnote_markers_on_labels", tmp_path)
    assert "The last 3 rows under the table were read as notes" in said
    assert "* excludes contractors and agency staff" in said
    assert "Footnote markers '*', '(a)', '†' were removed" in said


def test_rows_left_out_are_counted_and_located(tmp_path):
    said = _said("second_table_below", tmp_path)
    assert "A second table starts at row 24" in said
    assert "9 rows are not part of this dataset" in said


def test_made_up_column_names_are_admitted():
    table, report = reshape(frame([[1, 2, 3], [4, 5, 6]]))
    said = " ".join(describe_shape(report, list(table.columns)))
    assert "No header row was found" in said
    assert "column_1 to column_3" in said


def test_a_blank_header_cell_is_named_and_said():
    table, report = reshape(frame([["a", NA, "c"], [1, 2, 3], [4, 5, 6]]))
    assert report["placeholder_names"] == ["column_2"]
    assert "'column_2'" in " ".join(describe_shape(report, list(table.columns)))


def test_the_upload_carries_the_assumptions_with_the_sheet(tmp_path):
    """They travel on the export, per sheet, so the server can put each one on
    the dataset it describes (``UploadedDatasetOut.notes``)."""
    _got, export, _exports, notes, _truth = _exported("merged_header_cells", tmp_path)
    assert any("read together as one header" in n for n in export.notes)
    # the workbook-level notes keep their old meaning: skipped or capped sheets
    assert notes == []


def test_a_multi_sheet_upload_says_which_sheet_leads_and_why(tmp_path):
    path, _truth = _build("decoy_sheet_outweighs_the_answer", tmp_path)
    exports, _notes = export_workbook_sheets_to_csv(path, tmp_path / "out")
    (said,) = describe_sheet_choice(exports)
    assert "'Reference data' is listed first only because it has the most rows (40)" in said
    assert "has not seen your question" in said
    assert "Quarterly summary" in said            # the one the answer is on
    assert describe_sheet_choice(exports[:1]) == []


def test_the_sampling_response_carries_the_assumptions(tmp_path):
    from app.agents.sampling_agent import sample_data_from_source

    path, truth = _build("stacked_two_row_header", tmp_path)
    result = sample_data_from_source(path, "xlsx", stratify_by=None)
    assert result.get("error") is None, result.get("error")
    assert result["schema"] == truth.columns
    assert any("read together as one header" in w for w in result["warnings"])


def test_the_assumptions_cannot_be_an_analytics_field():
    """They name the user's columns and sheets and quote their notes.

    The standing rule is that none of that is ever transmitted. The analytics
    permit-lists fail closed, so this holds as long as nobody adds one of
    these keys -- which is what the assertion is for.
    """
    from app.analytics import allowlist, schema

    carriers = {"notes", "ingest_notes", "warnings", "shape_report", "shape_notes",
                "sheet_name", "source_workbook"}
    assert not carriers & set(schema.CLIENT_FIELDS)
    assert not carriers & set(allowlist.STORED_KEYS)
    assert not carriers & set(allowlist.EXPORT_KEYS)
    assert allowlist.safe({key: ["Row 2 was taken as the header."] for key in carriers}) == {}
