"""
L5 plan suites E2.10 (error hygiene), E2.11 (CORS / private-network preflight)
and E2.12 (SSE streaming contract).

Hermetic legs (default, no deployment needed):
  * error-body hygiene for 422 and 404 responses
  * source-level pin of the Celery traceback leak (E2.10b defect)
  * absence of a global ``Exception`` handler
  * CORS header behaviour for a hostile Origin (E2.11 defect) plus a structural
    pin of the middleware kwargs themselves
  * the private-network-access preflight decision (E2.11a)
  * the four ``text/event-stream`` sites and the two routes that own them
  * auth-before-stream on ``GET /tasks/{task_id}/status``
  * SSE frame grammar driven through ``POST /threads/{thread_id}/messages``

Deployed-only:
  * client-disconnect leak observation (needs a real, long-lived stream)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from tests.e2e.conftest import REPO_ROOT, UNAUTH_MESSAGE, requires_deployed

SERVER_PY = REPO_ROOT / "app" / "api" / "server.py"

HOSTILE_ORIGIN = "https://evil.example"

# Markers that prove an internal detail escaped into a client-visible body.
_LEAK_MARKERS = (
    "Traceback (most recent call last)",
    "site-packages",
    "/usr/lib/python",
    "/app/app/api/server.py",
)

_ROUTE_DECORATOR_RE = re.compile(r'^@app\.(get|post|put|patch|delete)\(\s*"(?P<path>[^"]+)"')
_EVENT_STREAM_RE = re.compile(r'media_type\s*=\s*"text/event-stream"')

EXPECTED_STREAM_ROUTES: Set[str] = {
    "POST /threads/{thread_id}/messages",
    "GET /tasks/{task_id}/status",
}


def _server_source_lines() -> List[str]:
    return SERVER_PY.read_text(encoding="utf-8").splitlines()


def _event_stream_sites() -> List[Tuple[int, str]]:
    """Every StreamingResponse(media_type='text/event-stream') site as (line, 'METHOD /path')."""
    lines = _server_source_lines()
    sites: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        if not _EVENT_STREAM_RE.search(line):
            continue
        owner = "<module-level>"
        for back in range(idx, -1, -1):
            m = _ROUTE_DECORATOR_RE.match(lines[back].strip())
            if m:
                owner = f"{m.group(1).upper()} {m.group('path')}"
                break
        sites.append((idx + 1, owner))
    return sites


def _client_traceback_sites() -> List[str]:
    """Lines that put a raw ``traceback`` key into a payload returned or yielded to a client."""
    hits: List[str] = []
    for lineno, line in enumerate(_server_source_lines(), 1):
        stripped = line.strip()
        if '"traceback"' not in stripped:
            continue
        if stripped.startswith(("return ", "yield ")) or "json.dumps" in stripped:
            hits.append(f"server.py:{lineno}: {stripped[:110]}")
    return hits


def _assert_no_internals(body: str) -> None:
    for marker in _LEAK_MARKERS:
        assert marker not in body, f"error body leaks {marker!r}: {body[:400]}"
    assert str(REPO_ROOT) not in body, f"error body leaks the repo path: {body[:400]}"


def _server_app() -> Any:
    try:
        import app.api.server as server
    except Exception as exc:  # pragma: no cover - only when app deps are absent
        pytest.skip(f"cannot import app.api.server for structural inspection: {exc}")
    return server.app


def _cors_kwargs() -> Dict[str, Any]:
    from starlette.middleware.cors import CORSMiddleware

    for mw in _server_app().user_middleware:
        if getattr(mw, "cls", None) is CORSMiddleware:
            return dict(getattr(mw, "kwargs", {}) or {})
    pytest.fail("CORSMiddleware is not installed on the app")


def _pin_hermetic_blob_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-point storage at the in-memory fake: the app lifespan swaps in a real cloud client at startup, so an upload issued through the harness would otherwise hit the network."""
    import app.api.server as server
    from app.services import storage_service
    from tests import test_server_integration as si

    store = si.FakeBlobStore(bucket="fake-bucket", prefix="test")

    def _resolve(uri: str, object_name: Optional[str]) -> Tuple[Any, str]:
        return store, object_name or uri.rsplit("/", 1)[-1]

    monkeypatch.setattr(storage_service, "blob_store", store, raising=False)
    monkeypatch.setattr(storage_service, "_store_and_key_from_uri", _resolve, raising=False)
    monkeypatch.setattr(server, "_store_and_key_from_uri", _resolve, raising=False)


def _upload(api: Any, user: str) -> Dict[str, Any]:
    resp = api.post(
        "/api/upload",
        user=user,
        files={"file": ("hygiene.csv", b"a,b\n1,2\n3,4\n", "text/csv")},
        data={},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# E2.10 -- error hygiene
# ---------------------------------------------------------------------------


def test_e2_10a_invalid_body_returns_422_with_field_detail_only(api: Any) -> None:
    """Malformed request body -> 422 naming the offending field, with no traceback or host paths."""
    resp = api.post(
        "/threads/thread-does-not-exist/messages",
        user="hygiene-422",
        json={"role": "user"},
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, list) and detail, resp.text
    assert any("content" in [str(p) for p in item.get("loc", [])] for item in detail), resp.text
    _assert_no_internals(resp.text)


def test_e2_10a_unknown_id_returns_404_without_internals(api: Any) -> None:
    """Unknown thread id with a valid token -> 404 whose body carries no traceback or host paths."""
    resp = api.get("/threads/thread-does-not-exist/code", user="hygiene-404")
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "Unknown thread_id"
    _assert_no_internals(resp.text)


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E2.10 (P1): failed/cancelled Celery tasks return the raw Python traceback to "
        "clients - app/api/server.py:4587,4591 (GET /tasks/{task_id}/result/{index}) and "
        "app/api/server.py:2151,2153 (task SSE stream). Remove this xfail when fixed."
    ),
)
def test_e2_10b_failed_task_payload_omits_raw_traceback() -> None:
    """A failed task's client payload must carry a sanitized message, never a raw traceback; asserted source-level because the hermetic fakes cannot drive Celery to FAILURE."""
    assert _client_traceback_sites() == []


def test_e2_10c_no_global_exception_handler_is_registered() -> None:
    """No override for bare Exception exists, so unhandled errors surface as FastAPI's opaque 500."""
    handlers = _server_app().exception_handlers
    assert Exception not in handlers, (
        "a global Exception handler now exists; error-hygiene expectations must be re-derived: "
        f"{sorted(str(k) for k in handlers)}"
    )


# ---------------------------------------------------------------------------
# E2.11 -- CORS
# ---------------------------------------------------------------------------


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E2.11 (P0): wildcard CORS combined with credentials - app/api/config.py:45 sets "
        "allow_origins=['*'] and app/api/server.py:708-715 installs CORSMiddleware with "
        "allow_credentials=True, so Starlette hands every Origin a credential-allowed response. "
        "Remove this xfail when fixed."
    ),
)
def test_e2_11_hostile_origin_is_not_credential_allowed(api: Any) -> None:
    """A hostile Origin must not receive access-control-allow-credentials: true next to an echoed or wildcard allow-origin."""
    resp = api.get("/version", headers={"Origin": HOSTILE_ORIGIN})
    allow_origin = resp.headers.get("access-control-allow-origin")
    allow_credentials = resp.headers.get("access-control-allow-credentials")
    assert not (
        allow_credentials == "true" and allow_origin in {HOSTILE_ORIGIN, "*"}
    ), f"allow-origin={allow_origin!r} allow-credentials={allow_credentials!r}"


def test_e2_11_pins_wildcard_credentials_config() -> None:
    """Pins the middleware kwargs themselves: allow_origins contains '*' while allow_credentials is True."""
    from app.api.config import allow_origins

    assert "*" in allow_origins
    kwargs = _cors_kwargs()
    assert "*" in kwargs["allow_origins"]
    assert kwargs["allow_credentials"] is True


@pytest.mark.parametrize(
    "origin, expect_pna_grant",
    [
        pytest.param(HOSTILE_ORIGIN, False, id="hostile-origin"),
        pytest.param("https://app.avaloka.ai", False, id="plausible-origin"),
        pytest.param("*", True, id="literal-star-origin"),
    ],
)
def test_e2_11a_pins_private_network_preflight_requires_literal_origin_match(
    api: Any, origin: str, expect_pna_grant: bool
) -> None:
    """Pins the undecided call: the PNA middleware (server.py:719-740) grants Access-Control-Allow-Private-Network only when Origin is literally a member of allow_origins, which holds '*', so no browser origin ever qualifies -- should '*' satisfy the PNA check?"""
    resp = api.options(
        "/version",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    granted = resp.headers.get("access-control-allow-private-network") == "true"
    assert granted is expect_pna_grant
    if expect_pna_grant:
        assert resp.status_code == 204
        assert resp.headers.get("access-control-allow-origin") == origin


# ---------------------------------------------------------------------------
# E2.12 -- SSE streaming contract
# ---------------------------------------------------------------------------


def test_e2_12_pins_event_stream_site_inventory() -> None:
    """Exactly four text/event-stream sites exist and they belong to only the two streaming routes."""
    sites = _event_stream_sites()
    assert len(sites) == 4, sites
    assert {owner for _, owner in sites} == EXPECTED_STREAM_ROUTES, sites


def test_e2_12_task_status_requires_auth_before_streaming(api: Any) -> None:
    """GET /tasks/{task_id}/status without a token -> 401 JSON; the stream is never opened."""
    resp = api.get("/tasks/task-does-not-exist/status")
    assert resp.status_code == 401, resp.text
    assert "text/event-stream" not in resp.headers.get("content-type", "")
    assert resp.json()["detail"] == UNAUTH_MESSAGE


def test_e2_12_message_stream_emits_valid_sse_frames(
    api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With stream=true the chat route answers text/event-stream whose frames are blank-line separated data: lines closed by a done event."""
    if api.mode != "hermetic":
        pytest.skip(
            "a deployed chat turn runs real LLM agents with unbounded latency; the framing "
            "contract is asserted hermetically over the fake graph"
        )
    _pin_hermetic_blob_store(monkeypatch)
    upload = _upload(api, "sse-user")
    body = {
        "content": "hello world",
        "metadata": {"dataset_id": upload["dataset_id"]},
        "stream": True,
    }
    with api.stream(
        "POST",
        f"/threads/{upload['thread_id']}/messages",
        user="sse-user",
        json=body,
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"
        payload = b"".join(resp.iter_bytes()).decode("utf-8")

    assert payload.endswith("\n\n")
    frames = [f for f in payload.split("\n\n") if f]
    assert len(frames) >= 2, payload[:400]
    for frame in frames:
        lines = frame.split("\n")
        assert all(line.startswith(("event: ", "data: ", ":")) for line in lines), frame[:200]
        assert any(line.startswith("data: ") for line in lines), frame[:200]
    assert frames[-1].startswith("event: done"), frames[-1][:200]


def test_e2_12_client_disconnect_does_not_leak_the_stream(api: Any) -> None:
    """Deployed-only: aborting mid-stream must let the generator unwind so no worker or connection stays pinned."""
    requires_deployed(
        "client-disconnect leak needs a real long-lived task stream: schedule a task, open "
        "GET /tasks/{task_id}/status, abort the socket after the first frame, then observe that "
        "the server's active connection count drops and the 1s task_sse_iter poll loop "
        "(app/api/server.py:2131) stops instead of running to task completion"
    )
    pytest.skip("no scripted long-running task fixture exists against a deployment yet")
