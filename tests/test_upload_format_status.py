"""/api/capabilities says, per format, whether an upload is enabled or deferred.

``upload_formats`` was a flat list of what POST /api/upload accepts. A UI
could not tell from it why a table layout it knows about (Iceberg, Delta)
is missing, so it hardcoded its own accept list and the two drifted: the shipped modal offered four extensions of the server's nine.
``upload_format_status`` is additive and sits beside the list: every accepted
extension is ``enabled`` with statistics, every deferral carries its reason.
"""
from __future__ import annotations

import json

import pytest

from app.api.ui_contract import api_capabilities


def test_capabilities_without_a_status_map_still_emit_the_key():
    features = api_capabilities(upload_formats=["csv"])["features"]
    assert features["upload_formats"] == ["csv"]
    assert features["upload_format_status"] == {}


def test_status_entries_pass_through_sorted_and_copied():
    status = {"xml": {"status": "enabled", "statistics": True, "reason": None},
              "mongodb": {"status": "deferred", "statistics": False, "reason": "a live connection"}}
    features = api_capabilities(upload_formats=["xml"], upload_format_status=status)["features"]
    assert list(features["upload_format_status"]) == ["mongodb", "xml"]
    assert features["upload_format_status"]["mongodb"]["reason"] == "a live connection"
    features["upload_format_status"]["xml"]["status"] = "mutated"
    assert status["xml"]["status"] == "enabled", "the contract must not alias the caller's dict"


def test_the_server_status_covers_every_accepted_format_and_names_each_deferral():
    server = pytest.importorskip("app.api.server")
    status = server.upload_format_status()
    from app.agents.sampling_agent_daft import COMPLEX_SOURCE_TYPES
    for ext in server.SUPPORTED_UPLOAD_EXTS:
        # The bounded complex formats (PDF, images, video) return rows but no
        # statistics contract; every tabular format produces it.
        assert status[ext] == {
            "status": "enabled", "statistics": ext not in COMPLEX_SOURCE_TYPES, "reason": None,
        }, ext
    deferred = {ext for ext, entry in status.items() if entry["status"] == "deferred"}
    assert deferred == set(server.DEFERRED_UPLOAD_FORMATS)
    assert deferred.isdisjoint(server.SUPPORTED_UPLOAD_EXTS), "a format cannot be both"
    for ext in deferred:
        assert len(status[ext]["reason"]) >= 40, f"{ext}: the reason must say why, not just that"
    # The formats this change enabled are accepted, with the statistics path.
    assert {"jsonl", "ndjson"} <= server.SUPPORTED_UPLOAD_EXTS
    assert {"iceberg", "delta", "mcap"} <= deferred
    # A connection is not a format: the contract describes what app/ takes
    # (#508), so mongodb is named nowhere in it.
    assert "mongodb" not in status
    # The whole document stays JSON-serialisable for the route.
    json.dumps(api_capabilities(upload_formats=server.SUPPORTED_UPLOAD_EXTS,
                                upload_format_status=status))
