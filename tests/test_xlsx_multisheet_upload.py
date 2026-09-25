"""
tests/test_xlsx_multisheet_upload.py

/api/upload expands a multi-sheet Excel workbook into one dataset per sheet.

Each sheet is materialized as a CSV so the whole downstream pipeline
(sampling, codegen, execution) runs on the CSV-native path, and every sheet
becomes a sibling dataset in the upload's group session (multi-dataset chat).

Reuses the in-memory fakes (blob store, redis, sessions, langgraph stubs)
from test_server_integration; the entire module is skipped when the server's
dependency stack (fastapi, daft, langchain, ...) is not importable. The
pandas-only unit tests for the sheet exporter live in
tests/test_xlsx_multisheet_export.py so they run everywhere.
"""

import io
import os
import sys

import pandas as pd
import pytest

# pytest >= 9 defaults to --import-mode=importlib, which no longer puts this
# directory on sys.path — make the sibling test-module import explicit.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_tsi = pytest.importorskip(
    "test_server_integration",
    reason="app.api.server dependency stack not importable in this environment",
    exc_type=ImportError,
)

client = _tsi.client  # pytest fixture, collected via module namespace
make_auth_headers = _tsi.make_auth_headers
SESSIONS = _tsi.SESSIONS

XLSX_CT = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _df(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(1, rows + 1),
            "amount": [round(10.5 * i, 2) for i in range(1, rows + 1)],
            "category": ["alpha" if i % 2 else "beta" for i in range(rows)],
        }
    )


def _workbook_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


def _post_workbook(client, sheets: dict, filename: str = "book.xlsx", field: str = "file"):
    return client.post(
        "/api/upload",
        files={field: (filename, _workbook_bytes(sheets), XLSX_CT)},
        data={},
        headers=make_auth_headers("u-xlsx"),
    )


def _dataset_session(dataset_id: str) -> dict:
    for sess in SESSIONS.values():
        if sess.get("dataset_id") == dataset_id:
            return sess
    raise AssertionError(f"no session found for dataset {dataset_id}")


def test_multisheet_upload_creates_one_dataset_per_sheet(client):
    resp = _post_workbook(client, {"Notes": pd.DataFrame({"note": ["x", "y"]}), "Data": _df(20)})
    assert resp.status_code == 200, resp.text
    out = resp.json()

    datasets = out.get("datasets")
    assert datasets and len(datasets) == 2

    # Primary (top-level legacy fields) = largest sheet.
    assert out["dataset_id"] == datasets[0]["dataset_id"]
    assert datasets[0]["sheet_name"] == "Data"
    assert datasets[1]["sheet_name"] == "Notes"
    # The top-level schema comes from the sampler, and this module reuses
    # test_server_integration's fakes -- which stub sample_data_from_source to a
    # fixed {"a": int, "b": int} for every input. Asserting the workbook's real
    # column names here was asserting the stub, not the upload. The column
    # round-trip belongs to the pandas-only path and is covered there:
    # tests/test_xlsx_multisheet_export.py asserts ["id", "amount", "category"].
    # What this layer owns is that the *primary* sheet is the one that got
    # sampled, and that it produced a schema and rows at all.
    assert out["schema"], "primary sheet should be sampled and carry a schema"
    assert out["samples"], "primary sheet should produce sample rows"

    # Aliases stay unique and filenames identify the sheet.
    assert len({d["alias"] for d in datasets}) == 2
    assert all(d["source_workbook"] == "book.xlsx" for d in datasets)
    assert "[Data]" in datasets[0]["filename"] and "[Notes]" in datasets[1]["filename"]

    # Every sheet dataset is a CSV-native session with provenance recorded.
    for d in datasets:
        sess = _dataset_session(d["dataset_id"])
        assert sess["input_data_type"] == "csv"
        assert sess["sheet_name"] == d["sheet_name"]
        assert sess["source_workbook"] == "book.xlsx"
        assert sess["object_name"].endswith(".csv")

    # Group session indexes both sheet datasets (multi-dataset chat).
    group = SESSIONS[out["session_id"]]
    assert set(group["dataset_ids"]) == {d["dataset_id"] for d in datasets}


def test_single_sheet_workbook_stays_single_dataset(client):
    resp = _post_workbook(client, {"Only": _df(5)}, filename="single.xlsx")
    assert resp.status_code == 200, resp.text
    out = resp.json()

    assert not out.get("datasets")
    sess = _dataset_session(out["dataset_id"])
    assert sess["input_data_type"] == "csv"
    assert sess["sheet_name"] == "Only"
    assert sess["source_workbook"] == "single.xlsx"
    # Single-sheet workbooks keep the original filename.
    assert sess["filename"] == "single.xlsx"


def test_workbook_with_only_empty_sheets_is_rejected(client):
    resp = _post_workbook(client, {"Empty": pd.DataFrame()})
    assert resp.status_code == 422
    assert "no non-empty sheets" in resp.json()["detail"]


def test_files_field_workbook_returns_multiupload_with_sheets(client):
    resp = _post_workbook(
        client,
        {"Small": pd.DataFrame({"k": [1]}), "Big": _df(9)},
        field="files",
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()

    # MultiUploadResponse shape: no top-level dataset_id.
    assert "dataset_id" not in out
    assert [d["sheet_name"] for d in out["datasets"]] == ["Big", "Small"]


def test_plain_csv_upload_unchanged(client):
    resp = client.post(
        "/api/upload",
        files={"file": ("plain.csv", b"a,b\n1,2\n3,4\n", "text/csv")},
        data={},
        headers=make_auth_headers("u-xlsx"),
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert not out.get("datasets")
    sess = _dataset_session(out["dataset_id"])
    assert sess["sheet_name"] is None
    assert sess["source_workbook"] is None
