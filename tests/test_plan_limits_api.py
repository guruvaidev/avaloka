"""Plan-aware upload limits at the API: GET /api/limits, the 413 from the
upload handler, and the early Content-Length refusal (with CORS headers).
 
Moved out of test_server_unit.py while that file was still named
test_server_integration.py and deselected by the CI gate
(`-m "not integration"`). It runs in the gate now; these stayed here. The plan logic
itself is unit-tested in tests/test_plan_limits.py.
"""
import asyncio
import os
import resource
import sys

import httpx
import pytest
from fastapi.testclient import TestClient
 
import app.api.server as server
import app.core.plan_limits as pl
 
try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
    SESSIONS, make_auth_headers, client, _do_upload,
    )
except ImportError:
    from server_harness import (  # noqa: F401
    SESSIONS, make_auth_headers, client, _do_upload,
    )
 
 
 
MB = 1024 * 1024


@pytest.fixture(autouse=True)
def _hosted_unless_set(monkeypatch):
    """These are hosted-plan tests unless a test says otherwise; a developer
    shell exporting AVALOKA_EDITION=oss must not turn them into self-hosted ones."""
    monkeypatch.delenv("AVALOKA_EDITION", raising=False)
    monkeypatch.delenv("AVALOKA_PLAN_LIMITS", raising=False)
    monkeypatch.delenv("AVALOKA_DEFAULT_PLAN", raising=False)
    monkeypatch.delenv("AVALOKA_SELF_HOSTED_MAX_UPLOAD_BYTES", raising=False)
    monkeypatch.delenv("AVALOKA_DEPLOYMENT", raising=False)
    pl.clear_plan_cache()


def _plan(monkeypatch, plan):
    monkeypatch.setattr(server, "resolve_plan", lambda user_id: plan, raising=False)


def _self_hosted_oss(monkeypatch):
    """The release-test server: AVALOKA_EDITION=oss, laptop profile, local storage,
    the real plan resolver (the harness stubs it to 'professional')."""
    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    monkeypatch.setenv("AVALOKA_DEPLOYMENT_PROFILE", "laptop")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setattr(server, "resolve_plan", pl.resolve_plan, raising=False)
 
 
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
 
 
# ---- self-hosted: never refused on plan grounds -----------------------------


_BOUNDARY = "avaloka-streamed-upload"
_CHUNK = 8 * MB


def _csv_chunk() -> bytes:
    """About 8 MB of whole CSV rows; the streamed body repeats this one object."""
    rows = bytearray()
    i = 0
    while len(rows) < _CHUNK:
        rows += f"{i},north,{i % 977}.25,{'x' * 40}\n".encode()
        i += 1
    return bytes(rows)


class _StreamedCsvUpload:
    """A multipart body of one ``size_bytes`` CSV, generated piece by piece.

    The whole body never exists in memory: the same 8 MB chunk is yielded
    again and again, and ``sent`` counts what the server actually pulled.
    Content-Length is exact, so the pre-body middleware sees the real size.
    """

    def __init__(self, size_bytes: int, filename: str = "big.csv"):
        self.chunk = _csv_chunk()
        self.n_chunks = -(-size_bytes // len(self.chunk))
        self.head = (f"--{_BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; "
                     f"filename=\"{filename}\"\r\nContent-Type: text/csv\r\n\r\n"
                     "id,region,amount,note\n").encode()
        self.tail = f"\r\n--{_BOUNDARY}--\r\n".encode()
        self.file_bytes = len(b"id,region,amount,note\n") + self.n_chunks * len(self.chunk)
        self.content_length = len(self.head) + self.n_chunks * len(self.chunk) + len(self.tail)
        self.sent = 0

    async def body(self):
        for piece in [self.head, *([self.chunk] * self.n_chunks), self.tail]:
            self.sent += len(piece)
            yield piece

    def post(self, headers):
        """POST it to /api/upload through httpx's ASGI transport, which hands the
        app one chunk per receive() (TestClient would read the body whole first)."""
        async def go():
            transport = httpx.ASGITransport(app=server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver",
                                         timeout=None) as ac:
                return await ac.post("/api/upload", content=self.body(), headers={
                    **headers,
                    "Content-Type": f"multipart/form-data; boundary={_BOUNDARY}",
                    "Content-Length": str(self.content_length),
                })
        return asyncio.run(go())


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024   # Linux reports KiB


def _store_sizes_only(monkeypatch):
    """The harness's FakeBlobStore keeps every object's bytes in a dict; for a
    300 MB upload record the size instead, so the test measures the server."""
    stored = {}

    def put_file(path, key):
        stored[key] = os.path.getsize(path)
        return f"gs://fake-bucket/test/{key}"

    monkeypatch.setattr(server.storage_service.blob_store, "put_file", put_file)
    return stored


def test_a_300_mb_upload_is_accepted_on_oss_self_hosted(client: TestClient, monkeypatch, tmp_path):
    """Defect D5 at its real size: a 300 MB CSV (3x Enterprise's 100 MB per file,
    1.5x its 200 MB total) posted to an oss self-hosted server is accepted.

    Nothing about the limit is stubbed: the plan resolver is the real one and
    the bound is this machine's free disk and memory. The body is streamed, so
    the test holds 8 MB of it at a time, and the peak-RSS check shows the
    server did not buffer it whole either."""
    _self_hosted_oss(monkeypatch)
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path / "filestore"))
    stored = _store_sizes_only(monkeypatch)
    upload = _StreamedCsvUpload(300 * MB)
    assert upload.file_bytes >= 300 * MB

    limits = pl.limits_for(pl.resolve_plan("oss-user"))
    assert limits.self_hosted and limits.plan == "oss"
    assert limits.max_file_bytes >= upload.file_bytes, (
        f"this machine cannot hold a 300 MB upload ({limits.bound_note}); the test needs more room")

    # Warm up with a small streamed upload first, so modules the first request
    # imports lazily are not counted against the 300 MB one.
    assert _StreamedCsvUpload(1).post(make_auth_headers("oss-user")).status_code == 200
    stored.clear()
    rss_before = _peak_rss_bytes()
    resp = upload.post(make_auth_headers("oss-user"))
    rss_growth = _peak_rss_bytes() - rss_before

    assert resp.status_code == 200, resp.text[:400]
    assert resp.json()["dataset_id"]
    assert upload.sent == upload.content_length, "the server must read the whole body"
    assert list(stored.values()) == [upload.file_bytes], "the full file reached the store"
    # Holding the body whole would need at least its own size. Python-level
    # allocations peak near 25 MB here (tracemalloc); the rest of the RSS growth
    # is native buffers, which stay well under the body.
    assert rss_growth < upload.content_length, (
        f"peak RSS grew {rss_growth / MB:.0f} MB for a {upload.content_length / MB:.0f} MB body")


def test_a_hosted_free_plan_still_refuses_300_mb_before_reading_it(client: TestClient, monkeypatch):
    """Hosted plans keep their limits, at their real defaults (no env override):
    the same 300 MB body is refused for Free from its Content-Length, before
    the body is read, with the upgrade offer the UI already handles."""
    _plan(monkeypatch, "free")
    upload = _StreamedCsvUpload(300 * MB)
    resp = upload.post(make_auth_headers("free-user"))
    assert resp.status_code == 413, resp.text[:400]
    detail = resp.json()["detail"]
    assert detail["code"] == "plan_limit_exceeded" and detail["plan"] == "free"
    assert detail["limit"] == 100 * MB and detail["upgrade_available"] is True
    assert upload.sent < 2 * _CHUNK, f"{upload.sent} bytes were read before refusing"
    assert SESSIONS == {}


def test_oss_self_hosted_reports_its_own_plan_and_bound(client: TestClient, monkeypatch):
    _self_hosted_oss(monkeypatch)
    resp = client.get("/api/limits", headers=make_auth_headers("oss-user"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["plan"] == "oss"
    assert body["limits"]["self_hosted"] is True
    assert body["limits"]["bound_by"] in ("disk", "memory", "env")
    assert body["limits"]["max_file_bytes"] > 100 * MB  # not Enterprise's number


def test_oss_self_hosted_refusal_names_the_resource_and_the_edition(client: TestClient, monkeypatch):
    """When the machine really is short, the 413 says so in plain language, with the
    plan name following the edition and no upgrade offer."""
    _self_hosted_oss(monkeypatch)
    monkeypatch.setenv("AVALOKA_UPLOAD_DISK_RESERVE_BYTES", "0")
    monkeypatch.setattr(pl, "_upload_roots", lambda: [pl.Path("/srv/avaloka")])
    monkeypatch.setattr(pl, "_free_disk_bytes", lambda p: 40)   # 20 bytes per upload
    monkeypatch.setenv("AVALOKA_UPLOAD_CL_SLACK_BYTES", str(1 << 20))
    files = {"file": ("big.csv", b"a,b\n" + b"1,2\n" * 20, "text/csv")}
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("oss-user"))
    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert detail["code"] == "resource_limit_exceeded"
    assert detail["plan"] == "oss"
    assert detail["limit"] == 20 and detail["upgrade_available"] is False
    assert "This server can take 20 bytes per file" in detail["message"]
    assert "40 bytes of disk is free under /srv/avaloka" in detail["message"]
    assert "plan" not in detail["message"].lower()
    assert SESSIONS == {}


def test_a_hosted_enterprise_plan_still_refuses_over_its_cap(client: TestClient, monkeypatch):
    """Hosted limits are untouched: Enterprise refuses above AVALOKA_MAX_UPLOAD_FILE_BYTES
    with the plan name and code the UI already handles."""
    _plan(monkeypatch, "enterprise")
    monkeypatch.setenv("AVALOKA_MAX_UPLOAD_FILE_BYTES", "50")
    monkeypatch.setenv("AVALOKA_UPLOAD_CL_SLACK_BYTES", str(1 << 20))
    files = {"file": ("big.csv", b"a,b\n" + b"1,2\n" * 20, "text/csv")}
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("ent-user"))
    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert detail["code"] == "plan_limit_exceeded" and detail["plan"] == "enterprise"
    assert detail["reason"] == "file_size" and detail["limit"] == 50
    assert "The Enterprise plan allows 50 bytes per file." in detail["message"]
    assert pl.limits_for("enterprise").max_file_bytes == 50
    monkeypatch.delenv("AVALOKA_MAX_UPLOAD_FILE_BYTES")
    assert pl.limits_for("enterprise").max_file_bytes == 100 * MB


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
 