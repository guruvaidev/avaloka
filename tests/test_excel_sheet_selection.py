"""
tests/test_excel_sheet_selection.py

XLSX ingestion fixes — sheet selection + type fidelity.

Covers the client-reported bug "CSV analysis is better than XLSX for the
same data / same prompt", which decomposed into:

  BUG 1 — pd.read_excel() defaults to sheet 0, so multi-sheet workbooks were
          silently analyzed on the FIRST sheet with no way to pick another
          and no warning about the sheets that were dropped.
  BUG 2 — the sampling paths rebuilt the DataFrame from load_data() (a
          list-of-lists round trip) which stringified dates and collapsed
          numeric columns to object dtype, so the profiler saw weakly-typed
          data for XLSX while CSV stayed clean.
  BUG 3 — the production Daft path dropped any sheet option entirely.

All fixtures are generated on the fly — no repo data files required.
Only pandas + openpyxl are needed; the sampling-agent tests are skipped
automatically when pyspark isn't importable.
"""

import datetime

import pandas as pd
import pytest

from file_handler.excel_connector import ExcelConnector
from file_handler.handler import FileHandler


# ---------------------------------------------------------------------------
# Fixtures — a multi-sheet workbook shaped like the client's:
# a small decoy first sheet, the real data sheet elsewhere.
# ---------------------------------------------------------------------------

DATA_ROWS = 20

def _data_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(1, DATA_ROWS + 1),
            "amount": [round(10.5 * i, 2) for i in range(1, DATA_ROWS + 1)],
            "created": [
                datetime.datetime(2026, 1, 1) + datetime.timedelta(days=i)
                for i in range(DATA_ROWS)
            ],
            "category": ["alpha" if i % 2 else "beta" for i in range(DATA_ROWS)],
        }
    )


@pytest.fixture()
def multi_sheet_xlsx(tmp_path):
    """Workbook: decoy 'Notes' first, real 'Data' second, small 'Summary' last."""
    path = tmp_path / "workbook.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"note": ["cover", "sheet", "decoy"]}).to_excel(
            writer, sheet_name="Notes", index=False
        )
        _data_frame().to_excel(writer, sheet_name="Data", index=False)
        pd.DataFrame({"total": [1, 2]}).to_excel(
            writer, sheet_name="Summary", index=False
        )
    return str(path)


@pytest.fixture()
def single_sheet_xlsx(tmp_path):
    path = tmp_path / "single.xlsx"
    _data_frame().to_excel(path, sheet_name="Only", index=False)
    return str(path)


@pytest.fixture()
def messy_xlsx(tmp_path):
    """Blank-ish cells and numeric-looking strings, to test normalization."""
    path = tmp_path / "messy.xlsx"
    pd.DataFrame(
        {
            "code": ["1", "2", "   ", "4"],   # numeric strings + whitespace cell
            "label": ["a", "  ", "c", "d"],   # whitespace-only cell
        }
    ).to_excel(path, sheet_name="Sheet1", index=False)
    return str(path)


# ---------------------------------------------------------------------------
# BUG 1 — sheet selection
# ---------------------------------------------------------------------------

class TestSheetSelection:
    def test_list_sheets_returns_all_in_order(self, multi_sheet_xlsx):
        conn = ExcelConnector(multi_sheet_xlsx)
        assert conn.list_sheets() == ["Notes", "Data", "Summary"]

    def test_default_auto_picks_largest_sheet_not_first(self, multi_sheet_xlsx):
        """REGRESSION: pd.read_excel() default silently analyzed sheet 0
        ('Notes', 3 rows) instead of the real data sheet."""
        conn = ExcelConnector(multi_sheet_xlsx)
        assert conn.selected_sheet == "Data"
        assert conn.get_row_count() == DATA_ROWS

    def test_explicit_sheet_by_name(self, multi_sheet_xlsx):
        conn = ExcelConnector(multi_sheet_xlsx, sheet_name="Summary")
        assert conn.selected_sheet == "Summary"
        assert conn.get_columns() == ["total"]

    def test_explicit_sheet_by_index(self, multi_sheet_xlsx):
        conn = ExcelConnector(multi_sheet_xlsx, sheet_name=0)
        assert conn.selected_sheet == "Notes"
        assert conn.get_row_count() == 3

    def test_unknown_sheet_raises_and_names_available(self, multi_sheet_xlsx):
        conn = ExcelConnector(multi_sheet_xlsx, sheet_name="NoSuchSheet")
        with pytest.raises(ValueError) as exc:
            _ = conn.selected_sheet
        # The error must help the caller recover: name the bad sheet and list options
        assert "NoSuchSheet" in str(exc.value)
        assert "Data" in str(exc.value)

    def test_single_sheet_workbook_trivially_selected(self, single_sheet_xlsx):
        conn = ExcelConnector(single_sheet_xlsx)
        assert conn.selected_sheet == "Only"
        assert conn.get_row_count() == DATA_ROWS

    def test_filehandler_forwards_sheet_name(self, multi_sheet_xlsx):
        handler = FileHandler(multi_sheet_xlsx, "xlsx", sheet_name="Notes")
        assert handler.connector.selected_sheet == "Notes"
        assert handler.get_columns() == ["note"]


# ---------------------------------------------------------------------------
# BUG 2 — type fidelity
# ---------------------------------------------------------------------------

class TestTypeFidelity:
    def test_load_dataframe_preserves_dtypes(self, multi_sheet_xlsx):
        """REGRESSION: the list-of-lists round trip stringified dates and
        collapsed numerics to object, degrading downstream profiling."""
        df = ExcelConnector(multi_sheet_xlsx).load_dataframe()
        assert pd.api.types.is_integer_dtype(df["id"])
        assert pd.api.types.is_float_dtype(df["amount"])
        assert pd.api.types.is_datetime64_any_dtype(df["created"])

    def test_legacy_load_data_stringified_dates(self, multi_sheet_xlsx):
        """Documents the legacy behavior load_dataframe() exists to avoid:
        rebuilding a frame from load_data() loses the datetime dtype."""
        conn = ExcelConnector(multi_sheet_xlsx)
        legacy_df = pd.DataFrame(conn.load_data(), columns=conn.get_columns())
        assert not pd.api.types.is_datetime64_any_dtype(legacy_df["created"])
        assert isinstance(legacy_df["created"].iloc[0], str)

    def test_blank_cells_normalized_to_na_and_numeric_coerced(self, messy_xlsx):
        """Mirrors the CSV path's normalization: whitespace-only -> NA,
        numeric-looking object columns coerced to numbers."""
        df = ExcelConnector(messy_xlsx).load_dataframe()
        assert pd.api.types.is_numeric_dtype(df["code"])
        assert df["code"].isna().sum() == 1          # the whitespace cell
        assert df["label"].isna().sum() == 1         # whitespace-only -> NA
        assert df["code"].dropna().tolist() == [1, 2, 4]

    def test_normalize_false_returns_raw_sheet(self, messy_xlsx):
        df = ExcelConnector(messy_xlsx).load_dataframe(normalize=False)
        # Untouched: still text (object on pandas < 3, str dtype on pandas >= 3),
        # no numeric coercion applied.
        assert not pd.api.types.is_numeric_dtype(df["code"])
        assert df["code"].iloc[0] == "1"


# ---------------------------------------------------------------------------
# End-to-end parity — the client's actual complaint
# ---------------------------------------------------------------------------

class TestCsvXlsxParity:
    def test_same_table_yields_equivalent_frames(self, tmp_path):
        """The same table saved as CSV and as a sheet in a multi-sheet
        workbook must reach the analyzer as the same data."""
        table = _data_frame()

        csv_path = tmp_path / "table.csv"
        table.to_csv(csv_path, index=False)

        xlsx_path = tmp_path / "table.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame({"note": ["decoy"]}).to_excel(writer, sheet_name="Cover", index=False)
            table.to_excel(writer, sheet_name="Table", index=False)

        from_csv = pd.read_csv(csv_path, parse_dates=["created"])
        from_xlsx = ExcelConnector(str(xlsx_path), sheet_name="Table").load_dataframe()

        assert list(from_csv.columns) == list(from_xlsx.columns)
        assert len(from_csv) == len(from_xlsx)
        pd.testing.assert_frame_equal(
            from_csv, from_xlsx, check_dtype=False, check_datetimelike_compat=True
        )


# ---------------------------------------------------------------------------
# Sampling agent integration (skipped when pyspark isn't available)
# ---------------------------------------------------------------------------

class TestSamplingAgentExcel:
    @pytest.fixture(autouse=True)
    def _needs_pyspark(self):
        pytest.importorskip("pyspark")

    def test_sample_reports_selected_sheet_and_warns_on_skipped(self, multi_sheet_xlsx):
        from app.agents.sampling_agent import sample_data_from_source

        result = sample_data_from_source(multi_sheet_xlsx, "xlsx", stratify_by=None)

        assert result.get("error") is None, result.get("error")
        assert result["sheets"] == ["Notes", "Data", "Summary"]
        assert result["selected_sheet"] == "Data"
        assert result["schema"] == ["id", "amount", "created", "category"]
        # Skipped sheets must be surfaced, never silent
        assert len(result["warnings"]) == 1
        for skipped in ("Notes", "Summary"):
            assert skipped in result["warnings"][0]

    def test_sample_honors_explicit_sheet_name(self, multi_sheet_xlsx):
        from app.agents.sampling_agent import sample_data_from_source

        result = sample_data_from_source(
            multi_sheet_xlsx, "xlsx", stratify_by=None, sheet_name="Summary"
        )

        assert result.get("error") is None, result.get("error")
        assert result["selected_sheet"] == "Summary"
        assert result["schema"] == ["total"]

    def test_single_sheet_has_no_warnings(self, single_sheet_xlsx):
        from app.agents.sampling_agent import sample_data_from_source

        result = sample_data_from_source(single_sheet_xlsx, "xlsx", stratify_by=None)

        assert result.get("error") is None, result.get("error")
        assert result["selected_sheet"] == "Only"
        assert result["warnings"] == []
