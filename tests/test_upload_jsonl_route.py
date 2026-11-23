"""jsonl and ndjson are accepted by POST /api/upload itself, not only by the
sampler underneath it.

#508 removed JSON Lines from the capabilities document because the upload
route refused it (415) while the CLI loader read it. Declaring the format
enabled again is earned here, on the route: the fixture goes through the
multipart gate, the real sampler, and comes back with the dataset's columns
and the statistics contract in the session.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api.server as server

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import SESSIONS, client, make_auth_headers  # noqa: F401
except ImportError:
    from server_harness import SESSIONS, client, make_auth_headers  # noqa: F401

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "formats"
TRUTH = json.loads((FIXTURES / "truth.json").read_text())


@pytest.mark.parametrize("name,content_type", [
    ("sales.jsonl", "application/jsonl"),
    ("sales.ndjson", "application/x-ndjson"),
])
def test_json_lines_pass_the_upload_gate_and_are_profiled(client: TestClient, monkeypatch, name, content_type):
    # The harness installs a two-column fake sampler; this test is about the
    # real reader behind the route, so put the real one back.
    from app.agents.sampling_agent_daft import sample_with_profiling
    monkeypatch.setattr(server, "sample_with_profiling", sample_with_profiling)
    resp = client.post(
        "/api/upload",
        files={"file": (name, (FIXTURES / name).read_bytes(), content_type)},
        data={},
        headers=make_auth_headers("fmt-jsonl"),
    )
    assert resp.status_code == 200, resp.text
    upload = resp.json()
    session = SESSIONS[upload["session_id"]]
    assert session["input_data_type"] == name.rsplit(".", 1)[1]
    assert list(session["uploaded_csv_columns"]) == TRUTH["columns"]
    assert session["file_size_bytes"] == (FIXTURES / name).stat().st_size
    # The statistics contract the understanding agent reads, as stored by the route.
    stats = server._profile_meta[upload["dataset_id"]]["sample_statistics"]
    assert stats["data_shape"] == {"rows": TRUTH["n_rows"], "columns": len(TRUTH["columns"])}
    assert set(stats["column_statistics"]) == set(TRUTH["columns"])
    assert stats["column_statistics"]["units_sold"]["type"] == "numeric"
    assert stats["source"]["format"] == name.rsplit(".", 1)[1]
    assert stats["source"]["rows_exact"] is True
    assert stats["source"]["partitions"] == [
        {"bytes": (FIXTURES / name).stat().st_size, "rows": TRUTH["n_rows"]}]


def test_a_format_outside_the_set_is_still_refused_by_name(client: TestClient):
    resp = client.post(
        "/api/upload",
        files={"file": ("table.iceberg", b"\x00\x01", "application/octet-stream")},
        data={},
        headers=make_auth_headers("fmt-refused"),
    )
    assert resp.status_code == 415
    assert ".jsonl" in resp.text and ".iceberg" in resp.text
