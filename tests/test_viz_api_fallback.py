"""The API half of the single-row-summary chart contract.

For a result that is one row with several columns the visualization node
returns ``visualization_config={}`` and ``visualization_status=None`` on
purpose (tests/test_visualization_grounding.py pins that half). The point of
the falsy pair is what send_message does with it in app/api/server.py:

    primary_viz_config = final.get("visualization_config") or visualization_configs.get(primary_dsid) or {}
    primary_viz_status = final.get("visualization_status") or visualization_statuses.get(primary_dsid)

-- the reply keeps the charts the session got at upload time. Nothing tested
that expression: replace the ``or`` with a plain ``.get(..., default)`` and the
node test stays green while the user's charts vanish from the reply.

These tests drive the real handler through TestClient with a fake graph, so
they fail when that fallback is broken. The fixtures come from
tests/test_server_integration.py; this file is separate because that one is
auto-marked ``integration`` by its filename (tests/conftest.py) and so never
runs in the hermetic gate. No LLM, no store, no network.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

import app.api.server as server
from tests.test_server_integration import (  # noqa: F401  (client is a fixture)
    FakeGraph,
    _do_upload,
    _quiet_send,
    client,
    make_auth_headers,
)

#: What the upload-time auto-viz stored on the session.
UPLOAD_CONFIG = {"visualization_status": "ready", "charts": [{"id": "upload_overview"}]}


def _chart_ids(config: Any) -> list:
    """The reply serialises the config through the response model, which adds
    its own None-valued keys; the charts are what identify whose config it is."""
    return [c.get("id") for c in ((config or {}).get("charts") or [])]


class _GraphReturning(FakeGraph):
    """A graph whose final state carries the given chart config and status."""

    def __init__(self, config: Any, status: Optional[str]):
        self._config, self._status = config, status

    def invoke(self, state_in: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = super().invoke(state_in, config)
        out["visualization_config"] = self._config
        out["visualization_status"] = self._status
        return out


@pytest.fixture()
def uploaded(client, monkeypatch):
    """A dataset whose session holds UPLOAD_CONFIG, ready for a chat turn."""
    monkeypatch.setattr(
        server, "build_visualization_config_from_sample", lambda **kw: dict(UPLOAD_CONFIG), raising=False
    )
    _quiet_send(monkeypatch)
    user = "viz-fallback"
    upload = _do_upload(client, user_id=user)
    assert _chart_ids(upload["visualization_config"]) == ["upload_overview"], (
        "the upload did not store the config under test"
    )

    def send(graph) -> Dict[str, Any]:
        monkeypatch.setattr(server, "GRAPH", graph, raising=False)
        resp = client.post(
            f"/threads/{upload['thread_id']}/messages",
            json={"content": "average fare and tip", "metadata": {"dataset_id": upload["dataset_id"]}},
            headers=make_auth_headers(user),
        )
        assert resp.status_code == 200, resp.text
        return resp.json()

    return upload, send


@pytest.mark.parametrize("config,status", [
    ({}, None),   # exactly what the node returns for a single-row summary
    ({}, ""),     # FakeGraph's own empty turn
    (None, None),
])
def test_a_turn_with_no_chart_config_keeps_the_sessions_upload_charts(uploaded, config, status):
    upload, send = uploaded

    body = send(_GraphReturning(config, status))

    assert _chart_ids(body["visualization_config"]) == ["upload_overview"]
    assert body["visualization_status"] == "ready"
    artifacts = server.THREAD_META[upload["thread_id"]]["artifacts"]
    assert _chart_ids(artifacts["visualization_config"]) == ["upload_overview"]
    assert artifacts["visualization_status"] == "ready"


def test_a_turn_with_its_own_chart_config_replaces_the_upload_charts(uploaded):
    """The other side of the `or`: without this, the test above would also pass
    if the API ignored the graph and always served the session's config."""
    _, send = uploaded
    turn_config = {"visualization_status": "ready", "charts": [{"id": "from_this_turn"}]}

    body = send(_GraphReturning(turn_config, "ready"))

    assert _chart_ids(body["visualization_config"]) == ["from_this_turn"]
    assert body["visualization_status"] == "ready"


def test_a_turn_status_replaces_the_session_status(uploaded):
    """The status line falls back independently of the config line."""
    _, send = uploaded
    turn_config = {"visualization_status": "skipped: no data available", "charts": []}

    body = send(_GraphReturning(turn_config, "skipped: no data available"))

    assert body["visualization_status"] == "skipped: no data available"
