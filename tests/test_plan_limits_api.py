"""Plan-aware upload limits at the API: GET /api/limits, the 413 from the
upload handler, and the early Content-Length refusal (with CORS headers).
 
Moved out of test_server_integration.py, which the CI gate deselects
(`-m "not integration"`), so these run on every pipeline. The plan logic
itself is unit-tested in tests/test_plan_limits.py.
"""
from fastapi.testclient import TestClient
 
import app.api.server as server
 
try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
    SESSIONS, make_auth_headers, client, _do_upload,
    )
except ImportError:
    from server_harness import (  # noqa: F401
    SESSIONS, make_auth_headers, client, _do_upload,
    )
 
 
 
def _plan(monkeypatch, plan):
    monkeypatch.setattr(server, "resolve_plan", lambda user_id: plan, raising=False)
 
 
def test_limits_requires_auth(client: TestClient):
    assert client.get("/api/limits").status_code == 401
 
 
def test_limits_reports_the_callers_plan(client: TestClient, monkeypatch):
    _plan(monkeypatch, "free")
    resp = client.get("/api/limits", headers=make_auth_headers("lim-user"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["plan"] == "free"
    assert body["limits"]["max_total_bytes"] == 100 * 1024 * 1024
    assert body["limits"]["batch_jobs"] is False
 
 
def test_free_upload_over_the_total_is_refused_with_an_upgrade_offer(client: TestClient, monkeypatch):
    _plan(monkeypatch, "free")
    monkeypatch.setenv("AVALOKA_FREE_MAX_TOTAL_BYTES", "20")
    monkeypatch.setenv("AVALOKA_UPLOAD_CL_SLACK_BYTES", str(1 << 20))  # let it reach the handler
    files = {"file": ("big.csv", b"a,b\n" + b"1,2\n" * 20, "text/csv")}
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("free-user"))
    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert detail["code"] == "plan_limit_exceeded"
    assert detail["plan"] == "free"
    assert detail["upgrade_available"] is True
    assert SESSIONS == {}
 
 
def test_free_upload_under_the_total_is_accepted(client: TestClient, monkeypatch):
    _plan(monkeypatch, "free")
    out = _do_upload(client, user_id="free-ok")
    assert out["dataset_id"]
 
 
def test_several_free_files_together_over_the_total_are_refused(client: TestClient, monkeypatch):
    _plan(monkeypatch, "free")
    monkeypatch.setenv("AVALOKA_FREE_MAX_TOTAL_BYTES", "30")
    monkeypatch.setenv("AVALOKA_UPLOAD_CL_SLACK_BYTES", str(1 << 20))
    files = [
        ("files", ("a.csv", b"a,b\n1,2\n3,4\n", "text/csv")),   # 12 bytes
        ("files", ("b.csv", b"a,b\n5,6\n7,8\n", "text/csv")),   # 12
        ("files", ("c.csv", b"a,b\n9,9\n8,8\n", "text/csv")),   # 12 -> 36 > 30
    ]
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("free-multi"))
    assert resp.status_code == 413
    assert resp.json()["detail"]["reason"] == "total_size"
 
 
def test_the_same_upload_is_accepted_on_professional(client: TestClient, monkeypatch):
    _plan(monkeypatch, "professional")
    monkeypatch.setenv("AVALOKA_FREE_MAX_TOTAL_BYTES", "20")  # must not apply to paid plans
    out = _do_upload(client, user_id="pro-user")
    assert out["dataset_id"]
 
 
def test_an_over_limit_content_length_is_refused_before_the_body_is_processed(client: TestClient, monkeypatch):
    _plan(monkeypatch, "free")
    monkeypatch.setenv("AVALOKA_FREE_MAX_TOTAL_BYTES", "20")
    monkeypatch.setenv("AVALOKA_UPLOAD_CL_SLACK_BYTES", "0")
    sampled = []
    monkeypatch.setattr(server, "sample_with_profiling",
                        lambda **kw: sampled.append(kw) or {"error": "should not run"}, raising=False)
    files = {"file": ("big.csv", b"a,b\n" + b"1,2\n" * 50, "text/csv")}
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("cl-user"))
    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] == "plan_limit_exceeded"
    assert sampled == [], "the upload handler must not run"
 
 
def test_the_early_refusal_keeps_cors_headers(client: TestClient, monkeypatch):
    _plan(monkeypatch, "free")
    monkeypatch.setenv("AVALOKA_FREE_MAX_TOTAL_BYTES", "20")
    monkeypatch.setenv("AVALOKA_UPLOAD_CL_SLACK_BYTES", "0")
    monkeypatch.setattr(server, "allow_origins", ["https://app.example"], raising=False)
    files = {"file": ("big.csv", b"a,b\n" + b"1,2\n" * 50, "text/csv")}
    headers = make_auth_headers("cors-user")
    headers["Origin"] = "https://app.example"
    resp = client.post("/api/upload", files=files, headers=headers)
    assert resp.status_code == 413
    assert resp.headers.get("access-control-allow-origin") == "https://app.example"
 