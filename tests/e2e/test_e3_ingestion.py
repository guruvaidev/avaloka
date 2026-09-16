"""
Plan suite E3 -- data ingestion: upload format matrix, connector matrix, upload
limits, multi-file semantics, filename hygiene and alternate destination inputs.

Everything in this module runs hermetically (in-process app over the fake blob
store / cache / graph from tests/test_server_integration.py). Two legs are
deployed-only and skip otherwise:

  * the byte/count limit cases, which monkeypatch the server's module-level
    limit constants instead of generating 100MB payloads -- meaningless against
    a remote process, so they run hermetically and skip when deployed;
  * the E3.13 tenant-isolation leg, which needs a real storage backend to prove
    whether a client-supplied destination URI can escape the caller's tenant.

Two corrections the ingestion plan gets wrong, both encoded below:

  * delta and iceberg are NOT accepted by POST /api/upload -- they are not in
    SUPPORTED_UPLOAD_EXTS and get 415 (app/api/server.py:299, 958-964);
  * the upload whitelist and the file_handler connector table are DIFFERENT
    sets. tsv and orc pass the upload gate but have no connector, which is the
    defect pinned by E3.01c.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import pytest

import app.api.server as server
from app.api.helpers import _alias_from_filename, _filename_ext
from file_handler import handler as fh
from file_handler.handler import FileHandler

from tests.e2e.conftest import ApiClient, requires_deployed

CSV_BYTES = b"a,b\n1,2\n3,4\n"

# app/api/server.py:299 -- the upload gate's whitelist, frozen here so a format
# silently added or dropped breaks this suite.
EXPECTED_UPLOAD_EXTS = {"csv", "tsv", "json", "xml", "parquet", "avro", "orc", "xls", "xlsx"}

# file_handler/handler.py:36-52 -- connectors whose __init__ touches nothing.
LAZY_CONNECTORS: Dict[str, str] = {
    "csv": "CSVConnector",
    "excel": "ExcelConnector",
    "xlsx": "ExcelConnector",
    "xls": "ExcelConnector",
    "avro": "AvroConnector",
    "json": "JSONConnector",
    "xml": "XMLConnector",
}

# These open a real table/catalog in __init__, so dispatch is checked with a
# recording stand-in rather than a real Delta/Iceberg/Parquet fixture.
EAGER_CONNECTORS: Dict[str, str] = {
    "parquet": "ParquetConnector",
    "delta": "DeltaConnector",
    "iceberg": "IcebergConnector",
}

ALL_CONNECTOR_CLASSES: Tuple[str, ...] = (
    "CSVConnector",
    "ExcelConnector",
    "AvroConnector",
    "JSONConnector",
    "ParquetConnector",
    "DeltaConnector",
    "IcebergConnector",
    "XMLConnector",
)


def _require_hermetic(api: ApiClient, reason: str) -> None:
    if api.mode != "hermetic":
        pytest.skip(f"{reason} (only observable in the in-process hermetic app)")


@pytest.fixture(autouse=True)
def fake_store(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Re-bind storage_service.blob_store to the fake the harness's URI resolver already returns.

    The harness installs the fake store before building the TestClient, but entering the client
    runs the app lifespan, and app/api/config.py:66 calls storage_service.init_blob_store() which
    replaces it with the real backend. server._store_and_key_from_uri stays faked, so writes go to
    the real bucket and the read-back looks in the fake -- every upload 500s on "Failed to create
    working copy from storage". Calling the patched resolver hands back the very store the harness
    built, so writes and reads agree again.

    The same lifespan leaves _schedule_small_file_persistence live, which posts profiles to the
    real Supabase project named in .env; it is stubbed out here so ingestion tests stay offline.
    """
    if "api" not in request.fixturenames:
        yield None
        return

    api: ApiClient = request.getfixturevalue("api")
    if api.mode != "hermetic":
        yield None
        return

    from app.services import storage_service

    store, _ = server._store_and_key_from_uri("gs://fake-bucket/fake-key", None)
    assert hasattr(store, "objects"), "harness no longer resolves URIs to the in-memory fake store"
    monkeypatch.setattr(storage_service, "blob_store", store, raising=False)
    monkeypatch.setattr(
        server, "_schedule_small_file_persistence", lambda **kw: None, raising=False
    )
    yield store


def _upload_single(
    api: ApiClient,
    filename: str,
    payload: bytes = CSV_BYTES,
    *,
    user: str = "e3-user",
    content_type: str = "application/octet-stream",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, str]] = None,
):
    return api.post(
        "/api/upload",
        user=user,
        headers=headers,
        files={"file": (filename, payload, content_type)},
        data=data or {},
    )


def _upload_many(
    api: ApiClient,
    items: List[Tuple[str, bytes]],
    *,
    user: str = "e3-user",
    content_type: str = "text/csv",
):
    files = [("files", (name, body, content_type)) for name, body in items]
    return api.post("/api/upload", user=user, files=files, data={})


def _blob_keys(store: Any) -> Set[str]:
    return set(store.objects.keys())


def _make_recorder(class_name: str) -> Any:
    return type(class_name, (), {"__init__": lambda self, *a, **kw: None})


def _stub_all_connectors(monkeypatch: pytest.MonkeyPatch) -> None:
    for class_name in ALL_CONNECTOR_CLASSES:
        monkeypatch.setattr(fh, class_name, _make_recorder(class_name), raising=True)


# ---------------------------------------------------------------------------
# E3.01a -- the upload-endpoint format matrix
# ---------------------------------------------------------------------------


def test_e3_01a_supported_upload_exts_frozen() -> None:
    """SUPPORTED_UPLOAD_EXTS is exactly the nine whitelisted formats; delta/iceberg are absent."""
    assert server.SUPPORTED_UPLOAD_EXTS == EXPECTED_UPLOAD_EXTS
    assert "delta" not in server.SUPPORTED_UPLOAD_EXTS
    assert "iceberg" not in server.SUPPORTED_UPLOAD_EXTS


@pytest.mark.parametrize("ext", sorted(EXPECTED_UPLOAD_EXTS))
def test_e3_01a_accepted_extension_passes_the_gate(api: ApiClient, ext: str) -> None:
    """Every whitelisted extension gets past the 415 gate (later sampling failures are a different contract)."""
    resp = _upload_single(api, f"sample.{ext}")
    assert resp.status_code != 415, resp.text


@pytest.mark.parametrize("ext", ["delta", "iceberg", "exe", "py", "xyz"])
def test_e3_01a_unlisted_extension_rejected_415(api: ApiClient, ext: str) -> None:
    """Anything outside the whitelist -- including delta/iceberg -- is rejected 415 with a supported-list hint."""
    resp = _upload_single(api, f"sample.{ext}")
    assert resp.status_code == 415, resp.text
    detail = resp.json()["detail"]
    assert f".{ext} is not supported yet" in detail
    assert ".csv" in detail


def test_e3_01a_extensionless_unknown_content_type_rejected_415(api: ApiClient) -> None:
    """No extension and an unmapped content-type resolves to no ext -> 415 'this file type'."""
    resp = _upload_single(api, "payload", content_type="application/x-tar")
    assert resp.status_code == 415, resp.text
    assert "this file type is not supported yet" in resp.json()["detail"]


@pytest.mark.parametrize("ext", ["delta", "iceberg", "exe"])
def test_e3_01a_register_existing_storage_enforces_same_whitelist(api: ApiClient, ext: str) -> None:
    """POST /api/register-existing-storage applies SUPPORTED_UPLOAD_EXTS to the object key too."""
    resp = api.post(
        "/api/register-existing-storage",
        user="e3-register",
        json={
            "storage_uri": "gs://external-bucket/some-prefix",
            "key": f"foo.{ext}",
            "connection_id": "conn-1",
            "schema_json": None,
        },
    )
    assert resp.status_code == 415, resp.text
    assert f".{ext} is not supported yet" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# E3.01b -- the connector matrix is a different set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("file_type,class_name", sorted(LAZY_CONNECTORS.items()))
def test_e3_01b_connector_matrix_lazy_types(file_type: str, class_name: str) -> None:
    """get_connector dispatches csv/excel/xlsx/xls/avro/json/xml to their real connector classes."""
    connector = FileHandler.get_connector("does-not-need-to-exist", file_type)
    assert connector is not None
    assert type(connector).__name__ == class_name


@pytest.mark.parametrize("file_type,class_name", sorted(EAGER_CONNECTORS.items()))
def test_e3_01b_connector_matrix_eager_types(
    monkeypatch: pytest.MonkeyPatch, file_type: str, class_name: str
) -> None:
    """parquet/delta/iceberg also dispatch; their readers open real tables so dispatch is checked via a stand-in."""
    monkeypatch.setattr(fh, class_name, _make_recorder(class_name), raising=True)
    connector = FileHandler.get_connector("does-not-need-to-exist", file_type, iceberg_properties={})
    assert connector is not None
    assert type(connector).__name__ == class_name


@pytest.mark.parametrize("file_type", ["tsv", "orc", "exe", "sqlite", ""])
def test_e3_01b_connector_matrix_has_no_connector(file_type: str) -> None:
    """tsv/orc (and anything unknown) fall off the end of get_connector and return None."""
    assert FileHandler.get_connector("does-not-need-to-exist", file_type) is None


def test_e3_01b_filehandler_raises_documented_error_without_connector(tmp_path: Path) -> None:
    """FileHandler turns a missing connector into TypeError "<x> files aren't supported" (handler.py:25-27)."""
    sample = tmp_path / "sample.tsv"
    sample.write_bytes(b"a\tb\n1\t2\n")
    with pytest.raises(TypeError) as exc:
        FileHandler(str(sample), "tsv")
    assert "aren't supported" in str(exc.value)


# ---------------------------------------------------------------------------
# E3.01c -- DEFECT: the two matrices disagree
# ---------------------------------------------------------------------------


def _upload_exts_without_connector(monkeypatch: pytest.MonkeyPatch) -> Set[str]:
    _stub_all_connectors(monkeypatch)
    return {
        ext
        for ext in server.SUPPORTED_UPLOAD_EXTS
        if FileHandler.get_connector(f"x.{ext}", ext, iceberg_properties={}) is None
    }


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E3.01c (P1): tsv/orc are accepted by /api/upload (app/api/server.py:299) but "
        "file_handler has no connector for them (file_handler/handler.py:36-52) - the upload 200s "
        "and sampling later raises TypeError \"files aren't supported\" (handler.py:25-27). "
        "Remove this xfail when fixed."
    ),
)
def test_e3_01c_every_accepted_upload_ext_has_a_connector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intended: every extension the upload gate accepts must be readable by file_handler.get_connector."""
    assert _upload_exts_without_connector(monkeypatch) == set()


def test_e3_01c_pins_the_current_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the exact size of the gap so a partial fix (or a new orphan format) is noticed."""
    assert _upload_exts_without_connector(monkeypatch) == {"tsv", "orc"}


# ---------------------------------------------------------------------------
# E3.02 -- multi-file upload semantics
# ---------------------------------------------------------------------------


def test_e3_02_pins_multi_file_all_register(api: ApiClient) -> None:
    """files[] is supported: N good files become N sibling datasets under one group session and one thread."""
    resp = _upload_many(api, [("alpha.csv", CSV_BYTES), ("beta.csv", CSV_BYTES)], user="e3-multi")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"]
    datasets = body["datasets"]
    assert len(datasets) == 2
    assert len({d["dataset_id"] for d in datasets}) == 2
    assert [d["alias"] for d in datasets] == ["alpha", "beta"]


def test_e3_02_pins_all_or_nothing_failure(api: ApiClient) -> None:
    """Pins the undecided semantic: one bad file aborts the whole batch with a single 5xx and no per-file report.

    Decision pinned: POST /api/upload is all-or-nothing (server.py:1417-1438 rolls back every
    created tmp upload, work dir and session and re-raises). There is deliberately no partial
    success payload, so a client cannot learn which of its files landed. If per-file partial
    reporting is ever adopted, this test must be rewritten, not deleted.
    """
    resp = _upload_many(api, [("good.csv", CSV_BYTES), ("broken.csv", b"")], user="e3-multi-fail")
    assert resp.status_code >= 500, resp.text
    body = resp.json()
    assert "datasets" not in body
    assert "detail" in body


def test_e3_02_pins_single_file_field_keeps_flat_response(api: ApiClient) -> None:
    """The legacy `file` field still returns the flat UploadResponse, not MultiUploadResponse."""
    resp = _upload_single(api, "solo.csv", user="e3-single", content_type="text/csv")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dataset_id"]
    assert body["session_id"]
    assert "schema" in body and "rows_sampled" in body


# ---------------------------------------------------------------------------
# E3.03 -- upload limits
# ---------------------------------------------------------------------------


def test_e3_03_limit_defaults() -> None:
    """The three ingestion limits default to 10 files / 100MB per file / 200MB per request."""
    assert server.MAX_UPLOAD_FILES == 10
    assert server.MAX_UPLOAD_FILE_BYTES == 100 * 1024 * 1024
    assert server.MAX_UPLOAD_TOTAL_BYTES == 200 * 1024 * 1024


def test_e3_03_over_per_file_limit_413(api: ApiClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file one byte over MAX_UPLOAD_FILE_BYTES is rejected 413 while streaming."""
    _require_hermetic(api, "patching server byte limits")
    monkeypatch.setattr(server, "MAX_UPLOAD_FILE_BYTES", len(CSV_BYTES) - 1, raising=False)
    resp = _upload_single(api, "big.csv", user="e3-limit")
    assert resp.status_code == 413, resp.text
    assert "exceeds max size" in resp.json()["detail"]


def test_e3_03_exactly_at_per_file_limit_not_rejected(
    api: ApiClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The comparison is strict '>', so a file exactly at MAX_UPLOAD_FILE_BYTES is accepted."""
    _require_hermetic(api, "patching server byte limits")
    monkeypatch.setattr(server, "MAX_UPLOAD_FILE_BYTES", len(CSV_BYTES), raising=False)
    resp = _upload_single(api, "exact.csv", user="e3-limit-exact")
    assert resp.status_code != 413, resp.text


def test_e3_03_over_total_limit_413(api: ApiClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-file sizes may pass while the running total trips MAX_UPLOAD_TOTAL_BYTES -> 413."""
    _require_hermetic(api, "patching server byte limits")
    monkeypatch.setattr(server, "MAX_UPLOAD_FILE_BYTES", 1 << 20, raising=False)
    monkeypatch.setattr(server, "MAX_UPLOAD_TOTAL_BYTES", len(CSV_BYTES), raising=False)
    resp = _upload_many(api, [("one.csv", CSV_BYTES), ("two.csv", CSV_BYTES)], user="e3-limit-total")
    assert resp.status_code == 413, resp.text
    assert "Total upload size exceeds limit" in resp.json()["detail"]


def test_e3_03_too_many_files_413(api: ApiClient) -> None:
    """More than MAX_UPLOAD_FILES (10) files is rejected 413 before any byte is read."""
    items = [(f"f{i}.csv", CSV_BYTES) for i in range(server.MAX_UPLOAD_FILES + 1)]
    resp = _upload_many(api, items, user="e3-limit-count")
    assert resp.status_code == 413, resp.text
    assert "Too many files" in resp.json()["detail"]


def test_e3_03_exactly_max_files_not_rejected(api: ApiClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The file-count check is also strict '>', so exactly MAX_UPLOAD_FILES is accepted."""
    _require_hermetic(api, "patching the server file-count limit")
    monkeypatch.setattr(server, "MAX_UPLOAD_FILES", 2, raising=False)
    resp = _upload_many(api, [("a.csv", CSV_BYTES), ("b.csv", CSV_BYTES)], user="e3-limit-count-exact")
    assert resp.status_code != 413, resp.text


def test_e3_03_no_file_at_all_422(api: ApiClient) -> None:
    """Neither `file` nor `files` present -> 422, not a 5xx."""
    resp = api.post("/api/upload", user="e3-nofile", data={"schema_json": ""}, files=[])
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# E3.03a -- DEFECT: the 0-byte file
# ---------------------------------------------------------------------------


# DEFECT E3.03a (P0) is fixed: both "no samples" sites in POST /api/upload now
# raise 400 instead of 500, so an empty upload names the caller's mistake.
def test_e3_03a_zero_byte_upload_is_client_error(api: ApiClient) -> None:
    """Intended: a 0-byte upload is the caller's error and must return a clean 4xx, never a 5xx."""
    resp = _upload_single(api, "empty.csv", payload=b"", user="e3-empty")
    assert 400 <= resp.status_code < 500, f"{resp.status_code}: {resp.text}"


def test_e3_03a_pins_current_zero_byte_response(api: ApiClient) -> None:
    """Pins the 0-byte contract. Was 500 while E3.03a was open; now 400."""
    resp = _upload_single(api, "empty.csv", payload=b"", user="e3-empty-pin")
    assert resp.status_code == 400, resp.text
    assert "Empty Dataset Provided" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# E3.11 -- filename hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("../../etc/passwd.csv", "csv"),
        ("..\\..\\windows\\system32\\evil.CSV", "csv"),
        ("/absolute/path/data.Parquet", "parquet"),
        ("archive.tar.gz", "gz"),
        ("no-extension", ""),
        (".hidden", ""),
        ("trailing.", ""),
        (None, ""),
        ("", ""),
    ],
)
def test_e3_11_filename_ext_normalises(filename: Optional[str], expected: str) -> None:
    """_filename_ext takes only the final suffix, lowercased, dot-stripped, and never a path component."""
    assert _filename_ext(filename) == expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("../../etc/passwd.csv", "passwd"),
        ("..\\..\\evil.csv", "evil"),
        ("/absolute/path/sales-2024.csv", "sales_2024"),
        ("my file (final).csv", "my_file__final"),
        ("____.csv", "dataset"),
        ("...csv", "dataset"),
        ("", "dataset"),
        ("données_日本.csv", "données_日本"),
        ("a" * 255 + ".csv", "a" * 255),
    ],
)
def test_e3_11_alias_from_filename_sanitises(filename: str, expected: str) -> None:
    """_alias_from_filename keeps only alnum/underscore from the stem and falls back to 'dataset'."""
    assert _alias_from_filename(filename) == expected


def test_e3_11_alias_never_contains_path_separators() -> None:
    """No sanitised alias can carry a separator or dot-dot into a later path join."""
    for hostile in [
        "../../etc/passwd.csv",
        "..\\..\\boot.ini.csv",
        "/etc/shadow.csv",
        "a/../../b.csv",
        "%2e%2e%2fetc%2fpasswd.csv",
    ]:
        alias = _alias_from_filename(hostile)
        assert "/" not in alias and "\\" not in alias and ".." not in alias
        assert alias == "".join(ch for ch in alias if ch.isalnum() or ch == "_")


def test_e3_11_duplicate_filenames_get_distinct_aliases(api: ApiClient) -> None:
    """Two files with the same name in one batch are de-duplicated deterministically (alias, alias_2)."""
    resp = _upload_many(api, [("data.csv", CSV_BYTES), ("data.csv", CSV_BYTES)], user="e3-dupe")
    assert resp.status_code == 200, resp.text
    aliases = [d["alias"] for d in resp.json()["datasets"]]
    assert aliases == ["data", "data_2"]


@pytest.mark.parametrize(
    "filename",
    [
        "../../../../etc/passwd.csv",
        "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts.csv",
        "/etc/shadow.csv",
        "données_日本.csv",
        "a" * 200 + ".csv",
    ],
)
def test_e3_11_hostile_filename_writes_nothing_outside_tmp_root(
    api: ApiClient, fake_store: Any, filename: str
) -> None:
    """A hostile upload filename never becomes a path component: bytes land at TMP_ROOT/upload_<uuid>.<ext>."""
    _require_hermetic(api, "inspecting the server's TMP_ROOT on disk")

    tmp_root = Path(server.TMP_ROOT).resolve()
    sandbox = tmp_root.parent
    before = {p for p in sandbox.rglob("*") if p.is_file()}
    escape_targets = [tmp_root, *list(tmp_root.parents)[:4]]
    basename = Path(filename.replace("\\", "/")).name
    before_escapes = {d: set(p.name for p in d.iterdir()) for d in escape_targets if d.is_dir()}

    resp = _upload_many(api, [(filename, CSV_BYTES)], user="e3-hostile")
    assert resp.status_code == 200, resp.text

    created = {p for p in sandbox.rglob("*") if p.is_file()} - before
    assert created, "upload wrote nothing at all -- the escape check would be vacuous"
    for path in created:
        assert tmp_root in path.resolve().parents, f"{path} escaped TMP_ROOT"

    for directory, names in before_escapes.items():
        new_names = set(p.name for p in directory.iterdir()) - names
        assert basename not in new_names, f"{basename} materialised in {directory}"

    alias = resp.json()["datasets"][0]["alias"]
    assert "/" not in alias and "\\" not in alias and ".." not in alias

    for key in _blob_keys(fake_store):
        assert ".." not in key
        assert not key.startswith("/")


# ---------------------------------------------------------------------------
# E3.13 -- client-chosen destinations (no plan coverage)
# ---------------------------------------------------------------------------


def test_e3_13_pins_x_storage_uri_header_selects_destination(api: ApiClient, fake_store: Any) -> None:
    """Pins that the X-Storage-URI request header is honoured as the write destination (server.py:886).

    Open tenant-boundary question this test deliberately does NOT answer: nothing between
    _resolve_user_id and the put_file call ties the caller's identity to the URI they supplied,
    so a caller can name any bucket/prefix the service account can reach. Hermetically the fake
    store cannot prove a cross-tenant write, so this leg only asserts that the header controls
    the object key; the isolation leg is test_e3_13_header_cannot_cross_tenant_boundary.
    """
    _require_hermetic(api, "inspecting the fake blob store's keys")

    before = _blob_keys(fake_store)
    _upload_single(
        api,
        "routed.csv",
        user="e3-header",
        headers={"X-Storage-URI": "gs://attacker-bucket/attacker-prefix"},
    )
    written = _blob_keys(fake_store) - before

    assert written, "the header path wrote no object at all"
    assert all(key.startswith("attacker-prefix/") for key in written), written


def test_e3_13_pins_backend_bucket_prefix_form_fields_build_destination(
    api: ApiClient, fake_store: Any
) -> None:
    """Pins that backend/bucket/prefix form fields build the destination URI client-side (server.py:887-915)."""
    _require_hermetic(api, "inspecting the fake blob store's keys")

    before = _blob_keys(fake_store)
    _upload_single(
        api,
        "routed.csv",
        user="e3-form",
        data={"backend": "gcs", "bucket": "chosen-bucket", "prefix": "chosen-prefix"},
    )
    written = _blob_keys(fake_store) - before

    assert written, "the form-field path wrote no object at all"
    assert all(key.startswith("chosen-prefix/") for key in written), written


def test_e3_13_unknown_backend_rejected_400(api: ApiClient) -> None:
    """An unrecognised `backend` form value is a clean 400 naming the three supported backends."""
    resp = _upload_single(api, "routed.csv", user="e3-backend", data={"backend": "ftp"})
    assert resp.status_code == 400, resp.text
    assert "Unknown backend" in resp.json()["detail"]


def test_e3_13_header_cannot_cross_tenant_boundary(api: ApiClient) -> None:
    """Deployed-only: a caller-supplied X-Storage-URI must not write into another tenant's bucket."""
    requires_deployed(
        "proving a cross-tenant write needs a real storage backend; the hermetic fake "
        "resolves every URI to the same in-memory store"
    )
    resp = _upload_single(
        api,
        "routed.csv",
        user="e3-tenant",
        headers={"X-Storage-URI": "gs://some-other-tenants-bucket/data"},
    )
    assert resp.status_code in (400, 403), resp.text
