"""
tests/test_xlsx_multisheet_export.py

Multi-sheet Excel uploads become one dataset per sheet.

Follow-up to test_excel_sheet_selection.py: picking a better single sheet is
not enough — the client expectation is that Avaloka knows about EVERY sheet
in an uploaded workbook. /api/upload now expands a workbook into sibling
datasets (one per non-empty sheet), each materialized as a CSV so the whole
downstream pipeline (sampling, codegen, execution) runs on the CSV-native
path.

Unit tests here need only pandas + openpyxl. The /api/upload tests reuse the
fakes from test_server_integration and are skipped automatically when the
server's dependency stack (fastapi, daft, langchain, ...) is not importable.
"""

import io

import pandas as pd
import pytest

from file_handler.excel_connector import export_workbook_sheets_to_csv


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _df(rows: int, prefix: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{prefix}id": range(1, rows + 1),
            f"{prefix}amount": [round(10.5 * i, 2) for i in range(1, rows + 1)],
            f"{prefix}category": ["alpha" if i % 2 else "beta" for i in range(rows)],
        }
    )


def _write_workbook(path, sheets: dict) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)


def _workbook_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    _write_workbook(buf, sheets)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Unit: export_workbook_sheets_to_csv
# ---------------------------------------------------------------------------

def test_export_orders_largest_sheet_first_and_skips_empty(tmp_path):
    wb = tmp_path / "wb.xlsx"
    _write_workbook(
        wb,
        {
            "Notes": pd.DataFrame({"note": ["cover", "decoy", "text"]}),
            "Data": _df(20),
            "Blank": pd.DataFrame(),
        },
    )

    exports, notes = export_workbook_sheets_to_csv(str(wb), tmp_path / "out")

    assert [e.sheet_name for e in exports] == ["Data", "Notes"]
    assert exports[0].n_rows == 20 and exports[1].n_rows == 3

    # Sheet CSV round-trips with columns and dtypes usable downstream.
    out = pd.read_csv(exports[0].csv_path)
    assert list(out.columns) == ["id", "amount", "category"]
    assert len(out) == 20
    assert pd.api.types.is_numeric_dtype(out["amount"])

    assert any("Blank" in n and "empty" in n.lower() for n in notes)


def test_export_max_sheets_keeps_largest_and_notes_dropped(tmp_path):
    wb = tmp_path / "wb.xlsx"
    _write_workbook(
        wb,
        {"S1": _df(2), "S2": _df(8), "S3": _df(5), "S4": _df(1)},
    )

    exports, notes = export_workbook_sheets_to_csv(str(wb), tmp_path / "out", max_sheets=2)

    assert [e.sheet_name for e in exports] == ["S2", "S3"]
    dropped_note = next(n for n in notes if "Not imported" in n)
    assert "S1" in dropped_note and "S4" in dropped_note
    # Dropped sheets leave no CSV files behind.
    remaining = {e.csv_path for e in exports}
    on_disk = {str(p) for p in (tmp_path / "out").glob("*.csv")}
    assert on_disk == remaining


def test_export_single_sheet_workbook(tmp_path):
    wb = tmp_path / "wb.xlsx"
    _write_workbook(wb, {"Only": _df(5)})

    exports, notes = export_workbook_sheets_to_csv(str(wb), tmp_path / "out")

    assert len(exports) == 1
    assert exports[0].sheet_name == "Only"
    assert exports[0].n_rows == 5
    assert notes == []


def test_export_all_sheets_empty_returns_no_exports(tmp_path):
    wb = tmp_path / "wb.xlsx"
    _write_workbook(wb, {"A": pd.DataFrame(), "B": pd.DataFrame()})

    exports, notes = export_workbook_sheets_to_csv(str(wb), tmp_path / "out")

    assert exports == []
    assert len(notes) == 2
