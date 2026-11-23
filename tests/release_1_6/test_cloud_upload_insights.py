"""1.6 release suite: UI-B14, Auto Insights for a dataset registered from
cloud storage.

The bug: datasets brought in from a cloud connection showed no Auto Insights,
while local uploads did. The insights are the visualization config built at
registration time; the UI renders whatever the response and the session carry.
The chart builder is replaced by a recording stub so the test certifies the
wiring (is it built from the sampled rows, returned, and stored), not the LLM.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

import app.api.server as server

try:
    from tests.server_harness import SESSIONS, client, make_auth_headers  # noqa: F401
except ImportError:  # pragma: no cover
    from server_harness import SESSIONS, client, make_auth_headers  # noqa: F401

CHART = {"id": "c1", "type": "bar", "title": "amount by region",
         "encodings": {"x": {"field": "region"}, "y": {"field": "amount"}}}


def test_uib14_cloud_registered_dataset_gets_auto_insights(client: TestClient, monkeypatch):
    """UI-B14: registering an object from a cloud connection builds Auto
    Insights from its sample, returns them, and stores them on the session,
    exactly as a local upload does."""
    calls = []

    def recording_viz(**kw):
        calls.append(kw)
        return {"visualization_status": "ready", "charts": [dict(CHART, dataset=kw.get("dataset_id"))]}

    monkeypatch.setattr(server, "build_visualization_config_from_sample", recording_viz, raising=False)

    resp = client.post(
        "/api/register-existing-storage",
        json={"storage_uri": "s3://example-bucket/exports", "key": "foo.csv", "connection_id": "conn-uib14"},
        headers=make_auth_headers("uib14-user"),
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()

    assert len(calls) == 1, "insights must be built once for the registered object"
    assert calls[0]["dataset_id"] == out["dataset_id"]
    assert calls[0]["sample_rows"], "insights are built from the sampled rows"

    assert out["visualization_status"] == "ready"
    charts = (out.get("visualization_config") or {}).get("charts") or []
    assert [(c["id"], c.get("dataset")) for c in charts] == [("c1", out["dataset_id"])]

    stored = SESSIONS[out["session_id"]]
    stored_viz = server._maybe_json_load(stored.get("visualization_config")) or stored.get("visualization_config") or {}
    assert [c["id"] for c in (stored_viz.get("charts") or [])] == ["c1"]
    assert stored.get("visualization_status") == "ready"
