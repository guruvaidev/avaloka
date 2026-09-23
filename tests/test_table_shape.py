"""Finding the table inside a spreadsheet.

Every case here is a shape someone actually sends: a title on row 0, a blank
row, the table indented a column to the right, two tables side by side. Read
with pandas' default header those become columns named ``Unnamed: 13``, and the
insights built on top are fluent sentences about columns that do not exist:

    Shows the distribution of Unnamed: 13, highlighting spread and outliers.
"""
import pandas as pd
import pytest

from file_handler.table_shape import (detect_header_row, extract_tables,
                                      needs_reshaping, reshape,
                                      split_column_blocks)

NA = None


def frame(rows):
    return pd.DataFrame(rows)


class TestHeaderDetection:
    def test_finds_a_header_below_a_blank_row(self):
        raw = frame([
            [NA, NA, NA],
            ["Parameter", "Value", "Unit"],
            ["Volume", 500, "hours"],
            ["Domains", 10, "count"],
        ])
        assert detect_header_row(raw) == 1

    def test_finds_a_header_below_a_title_and_a_blank(self):
        raw = frame([
            ["Q3 Pricing — internal", NA, NA],
            [NA, NA, NA],
            ["Role", "Rate", "Region"],
            ["Engineer", 120, "US"],
            ["Analyst", 90, "EU"],
        ])
        assert detect_header_row(raw) == 2

    def test_a_one_cell_title_is_not_a_header(self):
        raw = frame([["Report", NA, NA], ["a", "b", "c"], [1, 2, 3]])
        assert detect_header_row(raw) == 1

    def test_a_row_of_numbers_is_data_not_a_header(self):
        raw = frame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert detect_header_row(raw) is None

    def test_repeated_labels_are_a_data_row(self):
        raw = frame([["x", "x", "x"], [1, 2, 3], [4, 5, 6]])
        assert detect_header_row(raw) is None

    def test_a_sentence_is_a_value_not_a_header(self):
        """Key/value sheets have no header; naming columns after one record
        both loses the row and names everything after one arbitrary answer."""
        raw = frame([
            ["Language", "Not specified in the source correspondence.", NA],
            ["Region", "APAC", NA],
            ["Owner", "Centific", NA],
        ])
        assert detect_header_row(raw) is None

    def test_returns_none_for_an_empty_frame(self):
        assert detect_header_row(pd.DataFrame()) is None


class TestReshape:
    def test_the_reported_case(self):
        """Blank row 0, header on row 1, whole table indented one column."""
        raw = frame([
            [NA, NA, NA, NA],
            [NA, "Parameter", "Value", "Unit"],
            [NA, "Total Target Volume", 500, "hours"],
            [NA, "Number of Domains", 10, "domains"],
        ])
        out, report = reshape(raw)
        assert list(out.columns) == ["Parameter", "Value", "Unit"]
        assert report["dropped_empty_columns"] == 1
        assert report["header_row"] == 1        # blank row 0, header row 1
        assert len(out) == 2
        assert out.iloc[0]["Parameter"] == "Total Target Volume"

    def test_no_unnamed_columns_survive(self):
        raw = frame([[NA, NA], [NA, NA], ["a", "b"], [1, 2]])
        out, _ = reshape(raw)
        assert not any(str(c).startswith("Unnamed") for c in out.columns)

    def test_blank_header_cells_get_sayable_names(self):
        raw = frame([["Role", NA, "Region"], ["Eng", 1, "US"]])
        out, _ = reshape(raw)
        assert list(out.columns) == ["Role", "column_2", "Region"]

    def test_duplicate_header_names_are_disambiguated(self):
        raw = frame([["Cost", "Cost", "Total"], [1, 2, 3], [4, 5, 6]])
        out, _ = reshape(raw)
        assert len(set(out.columns)) == len(out.columns)

    def test_headerless_sheet_keeps_every_row(self):
        """No header means no row is consumed as one."""
        raw = frame([
            ["Language", "Not specified in the source correspondence.", NA],
            ["Region", "APAC", NA],
            ["Owner", "Centific", NA],
        ])
        out, report = reshape(raw)
        assert report["synthesized_names"] is True
        assert len(out) == 3

    def test_an_ordinary_table_is_left_alone(self):
        raw = frame([["a", "b"], [1, 2], [3, 4]])
        out, report = reshape(raw)
        assert list(out.columns) == ["a", "b"]
        assert report["dropped_empty_columns"] == 0
        assert len(out) == 2


class TestBlocks:
    def test_splits_tables_that_sit_side_by_side(self):
        """The reported sheet had Parameter|Value beside a Management Efforts
        block, and read as one frame the header described only the left half."""
        raw = frame([
            ["Parameter", "Value", NA, "Metric", "Amount"],
            ["Volume", 500, NA, "Man hrs", 1500],
            ["Domains", 10, NA, "Month", 1],
        ])
        blocks = split_column_blocks(raw)
        assert len(blocks) == 2

    def test_extracts_each_block_as_its_own_table(self):
        raw = frame([
            ["Parameter", "Value", NA, "Metric", "Amount"],
            ["Volume", 500, NA, "Man hrs", 1500],
            ["Domains", 10, NA, "Month", 1],
        ])
        tables = extract_tables(raw)
        assert len(tables) == 2
        assert list(tables[0][0].columns) == ["Parameter", "Value"]
        assert list(tables[1][0].columns) == ["Metric", "Amount"]

    def test_a_single_table_stays_one_table(self):
        raw = frame([["a", "b"], [1, 2], [3, 4]])
        assert len(extract_tables(raw)) == 1


class TestNeedsReshaping:
    def test_detects_the_symptom_on_a_loaded_frame(self):
        df = pd.DataFrame({"Unnamed: 0": [1], "Unnamed: 1": [2]})
        assert needs_reshaping(df) is True

    def test_a_named_frame_is_fine(self):
        assert needs_reshaping(pd.DataFrame({"role": [1], "rate": [2]})) is False

    def test_empty_is_not_a_symptom(self):
        assert needs_reshaping(pd.DataFrame()) is False
