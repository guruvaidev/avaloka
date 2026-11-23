"""D5: the upload route must not wait for the LLM profile.

On a local model a ~50 MB upload took over 600 s because the semantic profile
and the model-written Auto Insights ran inside the request. They now run as a
background job raced against a short soft deadline; past it the response says
profiling_status="pending" and the job lands on the dataset session.

The job keeps running after its response, so these tests use
shared_loop_client (one event loop across requests, as under uvicorn).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
        PROFILE_RESULT, SESSIONS, _install_sampler, _rows, _wait_for,
        client, make_auth_headers, server, shared_loop_client,
    )
except ImportError:
    from server_harness import (  # noqa: F401
        PROFILE_RESULT, SESSIONS, _install_sampler, _rows, _wait_for,
        client, make_auth_headers, server, shared_loop_client,
    )

CSV = b"a,b\n1,2\n3,4\n5,6\n"
STATS = {"column_statistics": {"a": {"type": "numeric"}}, "data_shape": {"rows": 3, "columns": 2}}
LLM_CHARTS = [{"title": "written with the profile"}]
DATA_ONLY_CHARTS = [{"title": "data only"}]


def _install_llm_stubs(
    monkeypatch, *, delay_s: float, release: Optional[threading.Event] = None,
    profile_error: Optional[Exception] = None,
) -> List[Dict[str, Any]]:
    """Profile + insight-model stubs that take `delay_s` (or until `release`)."""
    calls: List[Dict[str, Any]] = []

    def _wait():
        if release is not None:
            release.wait(delay_s)
        elif delay_s:
            time.sleep(delay_s)

    def profile_stub(**kw):
        calls.append({"step": "profile"})
        _wait()
        if profile_error is not None:
            raise profile_error
        return {"full_profiling_result": dict(PROFILE_RESULT), "profiling_status": "full_profile"}

    def viz_stub(**kw):
        if not kw.get("use_llm", True):
            calls.append({"step": "data_only_viz"})
            return {"visualization_status": "ready", "charts": list(DATA_ONLY_CHARTS)}
        calls.append({"step": "llm_viz", "profiling_result": kw.get("profiling_result")})
        _wait()
        return {"visualization_status": "ready", "charts": list(LLM_CHARTS)}

    monkeypatch.setattr(server, "profile_full", profile_stub, raising=False)
    monkeypatch.setattr(server, "build_visualization_config_from_sample", viz_stub, raising=False)
    return calls


def _install_side_effect_recorders(monkeypatch) -> Dict[str, list]:
    seen: Dict[str, list] = {"notifications": [], "upserts": []}

    async def notify(user_id, ntype, title, body=None, **kw):
        seen["notifications"].append({"type": ntype, "title": title, "body": body, **kw})

    def upsert(**kw):
        seen["upserts"].append(kw)

    monkeypatch.setattr(server.notification_service, "create_notification", notify)
    monkeypatch.setattr(server, "upsert_profile", upsert, raising=False)
    return seen


def _upload(c, user_id: str) -> Dict[str, Any]:
    resp = c.post(
        "/api/upload",
        files={"file": ("test.csv", CSV, "text/csv")},
        headers=make_auth_headers(user_id),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _profile(c, dsid: str, user_id: str) -> Dict[str, Any]:
    resp = c.get(f"/api/datasets/{dsid}/profile", headers=make_auth_headers(user_id))
    assert resp.status_code == 200, resp.text
    return resp.json()


def _wait_landed(c, dsid: str, user_id: str) -> Dict[str, Any]:
    assert _wait_for(
        lambda: _profile(c, dsid, user_id)["profiling_status"] != "pending", timeout_s=5.0
    ), "the deferred profile never landed"
    assert _wait_for(lambda: not server._upload_profile_tasks, timeout_s=5.0)
    return _profile(c, dsid, user_id)


# (a) ------------------------------------------------------------------------

def test_upload_returns_in_under_5s_while_a_30s_llm_runs(shared_loop_client, monkeypatch):
    c = shared_loop_client
    release = threading.Event()
    calls = _install_llm_stubs(monkeypatch, delay_s=30.0, release=release)
    _install_side_effect_recorders(monkeypatch)
    _install_sampler(monkeypatch, {"random_baseline": _rows(3)}, sample_statistics=STATS)
    user = "deferred-slow-llm"

    try:
        t0 = time.monotonic()
        out = _upload(c, user)
        elapsed = time.monotonic() - t0
        status_now = _profile(c, out["dataset_id"], user)
    finally:
        release.set()
        _wait_for(lambda: not server._upload_profile_tasks, timeout_s=10.0)

    assert elapsed < 5.0, f"upload took {elapsed:.1f}s with a 30 s LLM"
    assert out["profiling_status"] == "pending"
    assert out["profiling_result"] is None
    # The data-only charts are in the response, ready to render.
    assert out["visualization_status"] == "ready"
    assert out["visualization_config"]["charts"] == DATA_ONLY_CHARTS
    assert status_now["profiling_status"] == "pending"
    # The profile did start: it is deferred, not skipped.
    assert {"step": "profile"} in calls


# (b) ------------------------------------------------------------------------

def test_deferred_profile_lands_and_is_retrievable(shared_loop_client, monkeypatch):
    c = shared_loop_client
    monkeypatch.setattr(server, "UPLOAD_PROFILE_SYNC_DEADLINE_S", 0.05)
    calls = _install_llm_stubs(monkeypatch, delay_s=0.3)
    seen = _install_side_effect_recorders(monkeypatch)
    _install_sampler(monkeypatch, {"random_baseline": _rows(3)}, sample_statistics=STATS)
    user = "deferred-lands"

    out = _upload(c, user)
    landed = _wait_landed(c, out["dataset_id"], user)
    preview = c.get(
        f"/datasets/{out['dataset_id']}/preview",
        params={"session_id": out["session_id"]},
        headers=make_auth_headers(user),
    )

    assert out["profiling_status"] == "pending"
    assert landed["profiling_status"] == "full_profile"
    assert landed["profiling_error"] is None
    assert landed["profiling_result"] == PROFILE_RESULT
    # The charts were rewritten with the profile once it landed.
    assert landed["visualization_config"]["charts"] == LLM_CHARTS
    llm_viz = [x for x in calls if x["step"] == "llm_viz"]
    assert llm_viz and llm_viz[0]["profiling_result"] == PROFILE_RESULT

    sess = SESSIONS[out["session_id"]]
    assert sess["profiling_status"] == "full_profile"
    assert sess["full_profiling_result"] == PROFILE_RESULT
    assert server._profile_meta[out["dataset_id"]]["full_profiling_result"] == PROFILE_RESULT

    assert preview.status_code == 200, preview.text
    assert preview.json()["profiling_status"] == "full_profile"
    assert preview.json()["full_profiling_result"] == PROFILE_RESULT

    assert [u["dataset_id"] for u in seen["upserts"]] == [out["dataset_id"]]
    assert [n["type"] for n in seen["notifications"]] == ["insights.generated"]


# (c) ------------------------------------------------------------------------

def test_deferred_profile_failure_is_a_visible_failed_state(shared_loop_client, monkeypatch):
    c = shared_loop_client
    monkeypatch.setattr(server, "UPLOAD_PROFILE_SYNC_DEADLINE_S", 0.05)
    _install_llm_stubs(monkeypatch, delay_s=0.3, profile_error=RuntimeError("model endpoint refused"))
    seen = _install_side_effect_recorders(monkeypatch)
    _install_sampler(monkeypatch, {"random_baseline": _rows(3)}, sample_statistics=STATS)
    user = "deferred-fails"

    out = _upload(c, user)
    landed = _wait_landed(c, out["dataset_id"], user)

    assert out["profiling_status"] == "pending"
    assert landed["profiling_status"] == "failed"
    assert landed["profiling_result"] is None
    msg = landed["profiling_error"]
    assert msg and "could not be built" in msg
    # Plain words, not the exception text.
    assert "RuntimeError" not in msg and "refused" not in msg
    # The charts survive a failed profile.
    assert landed["visualization_status"] == "ready"

    sess = SESSIONS[out["session_id"]]
    assert sess["profiling_status"] == "failed"
    assert sess["profiling_error"] == msg
    assert seen["upserts"] == []
    assert [n["type"] for n in seen["notifications"]] == ["insights.failed"]
    assert msg in seen["notifications"][0]["body"]


def test_fast_profile_is_still_returned_inline_and_route_is_owner_scoped(client, monkeypatch):
    _install_llm_stubs(monkeypatch, delay_s=0.0)
    seen = _install_side_effect_recorders(monkeypatch)
    _install_sampler(monkeypatch, {"random_baseline": _rows(3)}, sample_statistics=STATS)

    out = _upload(client, "owner")
    # A profile that beats the soft deadline is in the response, as before.
    assert out["profiling_status"] == "full_profile"
    assert out["profiling_result"] == PROFILE_RESULT
    assert out["visualization_config"]["charts"] == LLM_CHARTS
    assert seen["notifications"] == []  # nothing was deferred, nothing to announce

    assert _profile(client, out["dataset_id"], "owner")["profiling_status"] == "full_profile"
    other = client.get(
        f"/api/datasets/{out['dataset_id']}/profile", headers=make_auth_headers("someone-else")
    )
    assert other.status_code == 404
