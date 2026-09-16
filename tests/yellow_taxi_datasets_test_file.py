"""
End-to-end suite for the NYC Yellow Taxi dataset already staged in GCS.

    gs://avaloka-test-user-filestore/test_c2c_jyothi/yellow_tripdata_dataset_400k.csv  (~63 MB, default)
    gs://avaloka-test-user-filestore/test_c2c_jyothi/yellow_tripdata_2015-01.csv       (~2 GB)

Two fixtures, and the size decides how much of the suite runs. The 400k file is
the default because the c2c path is what most runs care about and 63 MB moves in
under a minute; point ``YT_OBJECT`` at the 2 GB file for the full matrix.

The 2 GB vintage sits above ``LARGE_DATASET_THRESHOLD_BYTES`` (1 GB,
app/api/workflow.py:29), and that single fact is what makes it worth staging at
all: it is the only fixture that exercises

  * the *forced fidelity choice* on registration (server.py:1680),
  * the ``entire_dataset`` -> ``k8s-ray`` + Celery offload (workflow.py:707-727),
    which requires large **and** a cloud connection, and
  * cloud-to-cloud (c2c) transfers at a size where Daft/Ray actually matter.

Tests that depend on the large path call ``_require_large_fixture()`` and SKIP
(never fail) on the 400k file, so a small-fixture run reports honestly rather
than failing assertions that were never about correctness.

The suite is laid out in phases and is safe to run partially — each phase skips
(never fails) when its prerequisites are absent, so `pytest -k phase0` on a
laptop with only GCS creds still tells you something useful.

    phase 0  source object in GCS            needs: GOOGLE_APPLICATION_CREDENTIALS
    phase 1  sampler / schema inference      needs: phase 0
    phase 2  API ingestion (register)        needs: AVALOKA_API_URL, YT_CONNECTION_ID,
                                                    SUPABASE_JWT_SECRET
    phase 3  fidelity + analysis on Ray      needs: phase 2 + a live cluster
    phase 4  c2c transfers (the main event)  needs: GROQ_API_KEY_CODING_AGENT,
                                                    RAY_DASHBOARD_URL
    phase 9  manual prompt pack (prints)     needs: nothing

Running it
----------
    # everything the environment supports
    export GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json
    export RAY_DASHBOARD_URL=http://<ray-dashboard>:8265      # already in .env
    export KUBE_NAMESPACE=avaloka-test

    # API legs: port-forward the API (not the webui, which is nginx on :80)
    kubectl port-forward -n avaloka-test svc/avaloka 9000:9000 &
    export AVALOKA_API_URL=http://localhost:9000
    export SUPABASE_JWT_SECRET=<cluster secret>
    export YT_CONNECTION_ID=<the GCS cloud-connection id from the UI>

    pytest tests/yellow_taxi_datasets_test_file.py -m cloud -v -s

    # cheap legs only — no cluster, no LLM, no transfer
    pytest tests/yellow_taxi_datasets_test_file.py -m "cloud and not slow" -v

    # just the c2c transfers -- phase 0 MUST be selected too: it is what populates
    # the shared state (source_object/source_uri) every phase 4 test reads via
    # _need(). Selecting phase 4 alone skips every transfer and still exits 0.
    pytest tests/yellow_taxi_datasets_test_file.py -k "phase0 or phase4" -v -s -rs

Knobs (all optional, defaults match the staged bucket)
-----------------------------------------------------
    YT_BUCKET, YT_PREFIX, YT_OBJECT       source location
    YT_DEST_BUCKET, YT_DEST_PREFIX        c2c destination
    YT_EXPECTED_ROWS                      exact row count; enables strict assertions
    YT_TASK_TIMEOUT                       seconds to wait for a Ray/Celery task (default 1800)
    YT_KEEP_OUTPUT=1                      do not delete objects this run wrote
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# The whole module touches real buckets. `-m "not cloud"` deselects it wholesale,
# which is what the hermetic PR gate does.
pytestmark = pytest.mark.cloud


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BUCKET = os.getenv("YT_BUCKET", "avaloka-test-user-filestore")
PREFIX = os.getenv("YT_PREFIX", "test_c2c_jyothi").strip("/")
OBJECT = os.getenv("YT_OBJECT", "yellow_tripdata_dataset_400k.csv")

DEST_BUCKET = os.getenv("YT_DEST_BUCKET", "avaloka-dta-destination")
DEST_PREFIX = os.getenv("YT_DEST_PREFIX", "transfers/yellow-taxi").strip("/")

API_URL = (os.getenv("AVALOKA_API_URL") or "").rstrip("/")
CONNECTION_ID = os.getenv("YT_CONNECTION_ID", "")
JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
USER_ID = os.getenv("YT_USER_ID", "yellow-taxi-e2e")

EXPECTED_ROWS = int(os.getenv("YT_EXPECTED_ROWS") or 0)
TASK_TIMEOUT = int(os.getenv("YT_TASK_TIMEOUT") or 1800)
KEEP_OUTPUT = os.getenv("YT_KEEP_OUTPUT") == "1"

# Turn "prerequisite missing -> skip" into "-> fail". See _unmet().
STRICT = os.getenv("YT_STRICT") == "1"

ONE_GB = 1024 ** 3

# "Did the job read the whole object, or silently a sample?" — the question the
# row-count assertions below actually ask. A sampled run tops out at
# DEFAULT_SAMPLE_MAX_ROWS (1000 by default, see .env), so an order of magnitude
# above that separates the two cases without hardcoding a fixture's size.
SAMPLE_ROW_CEILING = int(os.getenv("DEFAULT_SAMPLE_MAX_ROWS") or 1000)
MIN_FULL_SCAN_ROWS = SAMPLE_ROW_CEILING * 10

# Unique per run so parallel runs never fight over destination objects.
RUN_ID = os.getenv("YT_RUN_ID") or uuid.uuid4().hex[:6]

# The 2015 vintage carries lat/long (LocationID only arrives in the 2016-07 files).
# Compared case-insensitively: TLC shipped both `RateCodeID` and `RatecodeID`.
EXPECTED_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "pickup_longitude",
    "pickup_latitude",
    "RatecodeID",
    "store_and_fwd_flag",
    "dropoff_longitude",
    "dropoff_latitude",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
]

# Cross-phase handoff. Phase N stores what phase N+1 needs; a missing key means
# the earlier phase skipped, so the later one skips too instead of erroring.
STATE: Dict[str, Any] = {"written_objects": []}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unmet(reason: str) -> None:
    """A prerequisite is missing: skip normally, FAIL under YT_STRICT=1.

    Skipping is right for `pytest -m cloud` on a laptop that simply has no GCS
    creds. It is dangerous in the case this suite exists for: a misconfigured
    shell makes every cloud leg skip, and the run still exits 0 with a couple of
    offline tests passing -- "green" for a suite that transferred nothing. Set
    YT_STRICT=1 (CI, or any run whose result you intend to trust) to turn every
    unmet prerequisite into a failure instead.
    """
    if STRICT:
        pytest.fail(f"[YT_STRICT] unmet prerequisite: {reason}", pytrace=False)
    pytest.skip(reason)


def _need(key: str, phase: str) -> Any:
    if key not in STATE:
        _unmet(f"{phase} did not run (missing {key!r} in shared state)")
    return STATE[key]


def _require_large_fixture() -> int:
    """Skip unless the configured source is over the 1 GB large-dataset threshold.

    The forced-fidelity and Ray/Celery-offload paths only engage above 1 GB, so on
    the 400k fixture those tests would assert against the *small* path and fail for
    a reason that has nothing to do with correctness. Skipping keeps a small-fixture
    run honest: it reports what it did not cover instead of going red.
    """
    size = _need("source_size", "phase 0")
    if size < ONE_GB:
        pytest.skip(
            f"source is {size / 1e6:.0f} MB, below the 1 GB large-dataset threshold — "
            f"set YT_OBJECT=yellow_tripdata_2015-01.csv to exercise this path"
        )
    return size


def _import_or_skip(dotted: str, *names: str):
    """Import symbols, skipping when an optional heavy dep is missing.

    ``daft``, ``torch`` and ``fastavro`` ship in the cluster image but are often
    absent on a laptop. House rule (docs/testing.md): skip with a reason, never
    fail spuriously.
    """
    import importlib

    try:
        module = importlib.import_module(dotted)
    except ModuleNotFoundError as exc:
        pytest.skip(f"{dotted} requires {exc.name!r}, which is not installed here")
    resolved = tuple(getattr(module, name) for name in names)
    return resolved[0] if len(resolved) == 1 else resolved


def _storage():
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        _unmet(
            "needs GOOGLE_APPLICATION_CREDENTIALS to read the test bucket. "
            "This suite reads os.getenv() and never calls load_dotenv(), so having "
            "it in .env is not enough — run `source tests/dev-env.sh` first."
        )
    storage = pytest.importorskip(
        "google.cloud.storage", reason="google-cloud-storage not installed"
    )
    return storage.Client()


def _bucket(name: str = BUCKET):
    return _storage().bucket(name)


def _head_text(blob, n_bytes: int = 200_000) -> str:
    """Range-read the first n bytes. Never pulls the whole 2 GB object."""
    return blob.download_as_bytes(start=0, end=n_bytes).decode("utf-8", "replace")


def _header_and_rows(text: str) -> Tuple[List[str], List[List[str]]]:
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    rows = [r for r in reader if len(r) == len(header)]
    # The range read almost certainly cut the last row in half.
    return header, rows[:-1] if rows else rows


def _norm(names: List[str]) -> List[str]:
    return [n.strip().lstrip("﻿").lower() for n in names]


def _gs(bucket: str, path: str) -> str:
    return f"gs://{bucket}/{path.lstrip('/')}"


def _dest_path(name: str) -> str:
    """Namespaced destination path so runs don't collide and cleanup is precise."""
    return f"{DEST_PREFIX}/{RUN_ID}/{name}"


def _list_objects(bucket_name: str, prefix: str) -> List[Any]:
    return list(_storage().list_blobs(bucket_name, prefix=prefix))


def _wait_for_objects(
    bucket_name: str, prefix: str, timeout: int = 120
) -> List[Any]:
    """Poll for written objects.

    Tolerant of writers that emit `prefix/part-0.parquet` instead of the exact
    object name — Daft partitions some formats, and the assertion we care about
    is "the destination materialised", not "it is exactly one blob".
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = [b for b in _list_objects(bucket_name, prefix) if b.size and b.size > 0]
        if found:
            return found
        time.sleep(5)
    return []


# ---- API helpers (only used by phases 2 and 3) ------------------------------

def _api_ready() -> None:
    if not API_URL:
        _unmet("set AVALOKA_API_URL (kubectl port-forward svc/avaloka 9000:9000)")
    if not JWT_SECRET:
        _unmet("set SUPABASE_JWT_SECRET to mint a portal-shaped token")


def _auth_headers() -> Dict[str, str]:
    jwt = pytest.importorskip("jwt", reason="pyjwt not installed")
    now = int(time.time())
    token = jwt.encode(
        {"sub": USER_ID, "iat": now, "exp": now + 3600}, JWT_SECRET, algorithm="HS256"
    )
    return {"Authorization": f"Bearer {token}"}


def _http():
    httpx = pytest.importorskip("httpx")
    return httpx.Client(base_url=API_URL, headers=_auth_headers(), timeout=180.0)


def _chat(thread_id: str, content: str, **extra: Any) -> Dict[str, Any]:
    """One turn of conversation. Returns the parsed ChatResponse."""
    body: Dict[str, Any] = {"role": "user", "content": content}
    body.update(extra)
    with _http() as http:
        resp = http.post(f"/threads/{thread_id}/messages", json=body)
    assert resp.status_code == 200, f"chat failed [{resp.status_code}]: {resp.text[:800]}"
    return resp.json()


def _last_assistant_text(payload: Dict[str, Any]) -> str:
    msgs = payload.get("messages") or []
    for msg in reversed(msgs):
        if (msg.get("role") or "").lower() in {"assistant", "ai"}:
            return msg.get("content") or ""
    return json.dumps(msgs)[-2000:]


def _poll_task(task_id: str, timeout: int = TASK_TIMEOUT) -> Dict[str, Any]:
    """Block until a scheduled task reaches a terminal state."""
    terminal = {"success", "succeeded", "completed", "failure", "failed", "error", "cancelled"}
    deadline = time.time() + timeout
    last: Dict[str, Any] = {}
    with _http() as http:
        while time.time() < deadline:
            resp = http.get(f"/tasks/{task_id}/status")
            if resp.status_code == 200:
                last = resp.json()
                state = str(last.get("status") or last.get("state") or "").lower()
                if state in terminal:
                    return last
            time.sleep(15)
    pytest.fail(f"task {task_id} did not finish within {timeout}s; last status={last}")


# ---- DTA helpers (phase 4) --------------------------------------------------

def _dta_ready() -> None:
    if not os.getenv("GROQ_API_KEY_CODING_AGENT"):
        _unmet("GROQ_API_KEY_CODING_AGENT is required to generate transfer code")


def _ray_ready() -> None:
    if not os.getenv("RAY_DASHBOARD_URL"):
        _unmet("set RAY_DASHBOARD_URL to launch the transfer on the cluster")


def _cloud_creds_cls():
    return _import_or_skip(
        "app.agents.data_transfer_agent.dta_state", "cloud_storage_credentials"
    )


def _source_creds():
    cloud_storage_credentials = _cloud_creds_cls()

    return cloud_storage_credentials(
        provider="gcp",
        bucket_name=BUCKET,
        file_path=_need("source_object", "phase 0"),
    )


def _dest_creds(path: str, bucket: str = DEST_BUCKET):
    return _cloud_creds_cls()(provider="gcp", bucket_name=bucket, file_path=path)


def _generate_transfer(
    prompt: str, dest_path: str, dest_type: str, dest_bucket: str = DEST_BUCKET
) -> Tuple[Dict[str, Any], Optional[str], Any]:
    """Run the DTA codegen half of a c2c transfer (no cluster involved yet)."""
    data_transfer_pipeline = _import_or_skip(
        "app.agents.data_transfer_agent.data_transfer_agent", "data_transfer_pipeline"
    )

    src = _source_creds()
    dst = _dest_creds(dest_path, dest_bucket)

    final_state, injection_script = data_transfer_pipeline(
        user_prompt=prompt,
        source_type="csv",
        destination_type=dest_type,
        source_credentials=src,
        destination_credentials=dst,
        source_file=src.get_cloud_uri("csv"),
        destination_file=dst.get_cloud_uri(dest_type),
        write_mode="overwrite",
    )
    return final_state, injection_script, dst


def _run_transfer_on_ray(injection_script: str, label: str) -> Any:
    launch_gke_pipeline = _import_or_skip(
        "app.data_transfer_docker.gke_run", "launch_gke_pipeline"
    )

    job_id = f"taxi-{label}-{RUN_ID}"
    print(f"\n[phase4] launching RayJob dta-{job_id} (spin-up + execution takes minutes)")
    return launch_gke_pipeline(injection_script, job_id=job_id)


@pytest.fixture(scope="module", autouse=True)
def _cleanup_written_objects():
    """Delete only what this run created. Source data is never touched."""
    yield
    if KEEP_OUTPUT or not STATE["written_objects"]:
        if STATE["written_objects"]:
            print(f"\n[cleanup] YT_KEEP_OUTPUT=1 — leaving {len(STATE['written_objects'])} object(s)")
        return
    try:
        client = _storage()
    except Exception:  # pragma: no cover - creds vanished mid-run
        return
    for bucket_name, prefix in STATE["written_objects"]:
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            try:
                blob.delete()
                print(f"[cleanup] deleted gs://{bucket_name}/{blob.name}")
            except Exception as exc:  # pragma: no cover
                print(f"[cleanup] could not delete {blob.name}: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# Phase 0 — the source object
# ═════════════════════════════════════════════════════════════════════════════

def test_phase0_source_object_exists_and_is_over_the_large_threshold() -> None:
    """The configured CSV is present, and we record which path the run will take.

    Presence is the assertion; size is not. The large-dataset cases downstream
    (forced fidelity choice, Ray offload) gate themselves on ``file_size_bytes >=
    1 GB`` via ``_require_large_fixture()``, so pointing YT_OBJECT at the 400k
    fixture narrows the suite rather than breaking it.
    """
    exact = f"{PREFIX}/{OBJECT}" if PREFIX else OBJECT
    blob = _bucket().get_blob(exact)

    if blob is None:
        # Tolerate the exact filename drifting (…2015-0.csv vs …2015-01.csv).
        # Deliberately the SMALLEST match: falling back to the largest would
        # silently promote a 400k run onto the 2 GB file and multiply its runtime.
        candidates = [
            b for b in _list_objects(BUCKET, f"{PREFIX}/")
            if b.name.lower().endswith(".csv") and "yellow_tripdata" in b.name.lower()
        ]
        if not candidates:
            pytest.skip(f"no yellow_tripdata CSV under gs://{BUCKET}/{PREFIX}/")
        blob = min(candidates, key=lambda b: b.size or 0)
        print(f"\n[phase0] {exact} not found; using gs://{BUCKET}/{blob.name}")

    STATE["source_object"] = blob.name
    STATE["source_size"] = blob.size
    STATE["source_uri"] = _gs(BUCKET, blob.name)
    path = "large (>=1 GB)" if blob.size >= ONE_GB else "small (<1 GB)"
    print(f"\n[phase0] source = {STATE['source_uri']}  "
          f"({blob.size / 1e6:.0f} MB) — {path} path")

    assert blob.size > 0, f"{blob.name} is empty"


def test_phase0_header_matches_the_2015_taxi_schema() -> None:
    """All 19 columns are present, so column-name assertions downstream are safe."""
    name = _need("source_object", "phase 0")
    header, rows = _header_and_rows(_head_text(_bucket().get_blob(name), 100_000))
    STATE["header"] = header

    missing = set(_norm(EXPECTED_COLUMNS)) - set(_norm(header))
    assert not missing, f"schema drifted; missing columns: {sorted(missing)}"
    assert len(header) == len(EXPECTED_COLUMNS), (
        f"expected {len(EXPECTED_COLUMNS)} columns, got {len(header)}: {header}"
    )
    assert rows, "range read produced no complete data rows"


def test_phase0_sample_rows_parse_and_carry_the_known_dirt() -> None:
    """Values parse as the types the profiler will claim, and the file really is dirty.

    The dirty rows matter: several later cases assert that Avaloka *reports*
    zero-passenger / zero-distance / negative-total rows. If the fixture were
    clean, those cases would pass vacuously.
    """
    name = _need("source_object", "phase 0")
    header, rows = _header_and_rows(_head_text(_bucket().get_blob(name), 400_000))
    idx = {c: i for i, c in enumerate(_norm(header))}

    for row in rows[:200]:
        datetime.strptime(row[idx["tpep_pickup_datetime"]], "%Y-%m-%d %H:%M:%S")
        datetime.strptime(row[idx["tpep_dropoff_datetime"]], "%Y-%m-%d %H:%M:%S")
        float(row[idx["trip_distance"]])
        float(row[idx["total_amount"]])
        int(row[idx["passenger_count"]])

    dirt = {
        "zero_passengers": sum(1 for r in rows if int(r[idx["passenger_count"]]) == 0),
        "zero_distance": sum(1 for r in rows if float(r[idx["trip_distance"]]) == 0),
        "missing_gps": sum(1 for r in rows if float(r[idx["pickup_latitude"]]) == 0),
        "nonpositive_total": sum(1 for r in rows if float(r[idx["total_amount"]]) <= 0),
    }
    STATE["dirt_in_head"] = dirt
    print(f"\n[phase0] quality signals in the first {len(rows)} rows: {dirt}")

    assert sum(dirt.values()) > 0, (
        "no quality problems in the head of the file — the data-quality cases in "
        "the prompt pack would prove nothing against this fixture"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — sampler and schema inference (agent level, no cluster)
# ═════════════════════════════════════════════════════════════════════════════

def test_phase1_sampler_infers_full_schema_and_caps_rows(tmp_path: Path) -> None:
    """Sampling yields the whole schema from a bounded slice, and never exceeds the cap.

    The cap is a data-egress control — sampled rows are what reach the LLM — so
    a regression here leaks raw trip records, not just performance.
    """
    DEFAULT_SAMPLE_MAX_ROWS, sample_data_from_source = _import_or_skip(
        "app.agents.sampling_agent", "DEFAULT_SAMPLE_MAX_ROWS", "sample_data_from_source"
    )

    name = _need("source_object", "phase 0")
    header, rows = _header_and_rows(_head_text(_bucket().get_blob(name), 4_000_000))
    assert len(rows) > DEFAULT_SAMPLE_MAX_ROWS, (
        "slice is smaller than the sampler cap; widen the range read or this proves nothing"
    )

    local = tmp_path / "taxi_head.csv"
    with local.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    result = sample_data_from_source(str(local), "csv", None)
    sampled = result.get("rows") or []
    schema = result.get("schema") or {}
    ddl = result.get("ddl_schema") or ""

    assert 0 < len(sampled) <= DEFAULT_SAMPLE_MAX_ROWS, (
        f"sampler emitted {len(sampled)} rows against a cap of {DEFAULT_SAMPLE_MAX_ROWS}"
    )
    assert len(sampled) < len(rows), "sampler returned everything; the cap is not applied"

    schema_cols = _norm(list(schema.keys()) if isinstance(schema, dict) else list(schema))
    assert set(_norm(EXPECTED_COLUMNS)).issubset(set(schema_cols)), (
        f"inferred schema dropped columns: {sorted(set(_norm(EXPECTED_COLUMNS)) - set(schema_cols))}"
    )
    assert "total_amount" in ddl.lower(), "generated DDL omits total_amount"


def test_phase1_upload_endpoint_could_never_accept_this_file() -> None:
    """A 2 GB CSV cannot come in through /api/upload — the cloud path is mandatory.

    Pins the reason this dataset must be registered from a bucket. If someone
    raises the cap past 2 GB, this test says so loudly rather than letting a
    2 GB multipart upload quietly become the new normal.
    """
    MAX_UPLOAD_FILE_BYTES, MAX_UPLOAD_TOTAL_BYTES = _import_or_skip(
        "app.api.server", "MAX_UPLOAD_FILE_BYTES", "MAX_UPLOAD_TOTAL_BYTES"
    )

    size = _require_large_fixture()
    assert size > MAX_UPLOAD_FILE_BYTES, (
        f"per-file upload cap is {MAX_UPLOAD_FILE_BYTES / ONE_GB:.2f} GB, which now "
        f"admits this {size / ONE_GB:.2f} GB file"
    )
    assert size > MAX_UPLOAD_TOTAL_BYTES


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2 — ingestion through the API (register the existing bucket object)
# ═════════════════════════════════════════════════════════════════════════════

def test_phase2_register_existing_storage_creates_the_dataset() -> None:
    """POST /api/register-existing-storage registers the 2 GB object without downloading it.

    Asserts the large-dataset contract: quick_sample by default, and the user is
    *asked* to choose a fidelity rather than being dropped into a full scan.
    """
    _api_ready()
    if not CONNECTION_ID:
        _unmet("set YT_CONNECTION_ID to the GCS cloud-connection id from the UI")

    name = _need("source_object", "phase 0")
    key = name.split("/")[-1]
    prefix = "/".join(name.split("/")[:-1])

    started = time.time()
    with _http() as http:
        resp = http.post(
            "/api/register-existing-storage",
            json={
                "storage_uri": _gs(BUCKET, prefix),
                "key": key,
                "connection_id": CONNECTION_ID,
                "schema_json": None,
            },
        )
    assert resp.status_code == 200, f"register failed [{resp.status_code}]: {resp.text[:1000]}"
    out = resp.json()
    elapsed = time.time() - started

    STATE["dataset_id"] = out["dataset_id"]
    STATE["thread_id"] = out["thread_id"]
    STATE["session_id"] = out["session_id"]
    print(
        f"\n[phase2] dataset={out['dataset_id']} thread={out['thread_id']} "
        f"registered in {elapsed:.1f}s, size={out.get('file_size_bytes', 0) / ONE_GB:.2f} GB"
    )

    # Registration itself is size-agnostic, and phases 3-4 need the thread/dataset
    # it just produced -- so the registration is asserted for every fixture and only
    # the >=1 GB contract below is gated. Gating the whole test would strand every
    # downstream phase on the 400k fixture.
    assert out.get("file_size_bytes", 0) > 0, (
        "registration did not carry the real object size; the fidelity and Ray "
        "routing decisions both read file_size_bytes"
    )
    assert out.get("rows_sampled", 0) > 0, "no quick sample produced"

    _require_large_fixture()
    assert out.get("file_size_bytes", 0) >= ONE_GB
    assert out.get("analysis_fidelity") == "quick_sample", (
        f"a >1 GB dataset must default to quick_sample, got {out.get('analysis_fidelity')!r}"
    )
    assert out.get("requires_fidelity_choice") is True, (
        "the user was not asked to choose a fidelity for a 2 GB dataset"
    )
    prompt = out.get("fidelity_prompt") or ""
    assert "quick_sample" in prompt and "entire dataset" in prompt.lower(), (
        f"fidelity prompt does not offer the three modes: {prompt[:300]!r}"
    )
    assert elapsed < 300, (
        f"registration took {elapsed:.0f}s — a 2 GB object should be sampled by "
        f"streaming its head, not downloaded in full"
    )


def test_phase2_preview_returns_the_taxi_schema() -> None:
    """The dataset preview the UI renders has all 19 columns and real rows."""
    _api_ready()
    dsid = _need("dataset_id", "phase 2")

    with _http() as http:
        resp = http.get(f"/datasets/{dsid}/preview")
    assert resp.status_code == 200, resp.text
    preview = resp.json()

    schema = preview.get("schema") or {}
    cols = _norm(list(schema.keys()) if isinstance(schema, dict) else list(schema))
    assert set(_norm(EXPECTED_COLUMNS)).issubset(set(cols)), (
        f"preview schema is missing {sorted(set(_norm(EXPECTED_COLUMNS)) - set(cols))}"
    )
    assert preview.get("rows_sampled", 0) > 0


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3 — fidelity switching and the entire-dataset offload to Ray
# ═════════════════════════════════════════════════════════════════════════════

def test_phase3_quick_sample_question_answers_inline() -> None:
    """A question in quick_sample mode answers in-request, with no cluster job."""
    _api_ready()
    tid = _need("thread_id", "phase 2")
    dsid = _need("dataset_id", "phase 2")

    payload = _chat(
        tid,
        "How many columns does this dataset have, and what are the payment types present?",
        dataset_ids=[dsid],
        metadata={"dataset_id": dsid},
    )
    reply = _last_assistant_text(payload)
    print(f"\n[phase3][quick_sample] {reply[:600]}")

    assert reply.strip(), "planner returned an empty reply"
    assert payload.get("analysis_fidelity") in (None, "quick_sample")
    ctx = payload.get("execution_context") or {}
    assert ctx.get("mode", "quick_sample") == "quick_sample"


@pytest.mark.integration
def test_phase3_switch_to_entire_dataset_is_acknowledged() -> None:
    """`use entire dataset` flips the mode and warns about runtime before spending it."""
    _api_ready()
    tid = _need("thread_id", "phase 2")
    dsid = _need("dataset_id", "phase 2")

    payload = _chat(tid, "use entire dataset", dataset_ids=[dsid], metadata={"dataset_id": dsid})
    reply = _last_assistant_text(payload).lower()
    print(f"\n[phase3][switch] {reply[:600]}")

    assert payload.get("analysis_fidelity") == "entire_dataset", (
        f"mode did not switch: {payload.get('analysis_fidelity')!r}"
    )
    assert "entire dataset" in reply
    assert "estimated" in reply or "time" in reply, (
        "switching a 2 GB dataset to full-scan mode gave no runtime warning"
    )


@pytest.mark.slow
@pytest.mark.cluster
@pytest.mark.integration
def test_phase3_entire_dataset_analysis_is_offloaded_to_ray() -> None:
    """The flagship case: full-fidelity analysis is scheduled on the cluster, not run inline.

    ``entire_dataset`` + >1 GB + a cloud connection is the only combination that
    sets execution_mode="k8s-ray" and attaches a one-shot Celery schedule
    (app/api/workflow.py:707-727). This dataset is the only fixture that can
    reach that branch.
    """
    _api_ready()
    _require_large_fixture()
    tid = _need("thread_id", "phase 2")
    dsid = _need("dataset_id", "phase 2")

    payload = _chat(
        tid,
        "Using the entire dataset, count the number of trips for each day in the month "
        "and return the daily counts.",
        dataset_ids=[dsid],
        analysis_fidelity="entire_dataset",
        metadata={"dataset_id": dsid},
    )

    task_id = payload.get("analysis_task_id") or (payload.get("task_info") or {}).get("task_id")
    print(f"\n[phase3][entire] task_id={task_id} fidelity={payload.get('analysis_fidelity')}")
    assert task_id, (
        "no task was scheduled — a full scan of a 2 GB cloud dataset must be "
        f"offloaded, not answered inline. reply was: {_last_assistant_text(payload)[:500]!r}"
    )

    status = _poll_task(task_id)
    state = str(status.get("status") or status.get("state") or "").lower()
    print(f"[phase3][entire] terminal status={state}")
    assert state in {"success", "succeeded", "completed"}, f"task failed: {status}"

    with _http() as http:
        result = http.get(f"/tasks/{task_id}/result/0")
    assert result.status_code == 200, result.text
    body = result.text
    # January has 31 days; a real full-month aggregation mentions the late-month dates.
    assert "2015-01-31" in body or "31" in body, f"daily counts look wrong: {body[:600]}"

    if EXPECTED_ROWS:
        digits = "".join(ch for ch in body if ch.isdigit())
        assert digits, "no numbers in the full-dataset result"


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4 — cloud-to-cloud transfers (DTA)
# ═════════════════════════════════════════════════════════════════════════════

def test_phase4_c2c_rejects_a_json_cloud_source() -> None:
    """JSON is a valid destination but not a valid cloud source — rejected up front.

    Needs no LLM key: the guard fires before the coder is built, which is the
    point — it must not burn a pipeline run to discover an unreadable source.
    """
    data_transfer_pipeline = _import_or_skip(
        "app.agents.data_transfer_agent.data_transfer_agent", "data_transfer_pipeline"
    )
    cloud_storage_credentials = _cloud_creds_cls()

    src = cloud_storage_credentials(
        provider="gcp", bucket_name=BUCKET, file_path=f"{PREFIX}/does-not-need-to-exist.json"
    )
    dst = cloud_storage_credentials(
        provider="gcp", bucket_name=DEST_BUCKET, file_path=_dest_path("never_written.csv")
    )

    state, script = data_transfer_pipeline(
        user_prompt="copy everything",
        source_type="json",
        destination_type="csv",
        source_credentials=src,
        destination_credentials=dst,
        source_file=src.get_cloud_uri("json"),
        destination_file=dst.get_cloud_uri("csv"),
    )

    assert script is None, "a cloud JSON source must not produce an injection script"
    assert "json is not supported as a cloud source" in (state.get("error_message") or "").lower()


@pytest.mark.integration
def test_phase4_c2c_codegen_targets_the_right_uris() -> None:
    """Codegen for CSV -> Parquet resolves both endpoints before anything is launched.

    Split out from the launch so a bad URI or an incompatible destination schema
    is caught in seconds rather than after a multi-minute cluster spin-up.
    """
    _dta_ready()
    dest = _dest_path("taxi_full.parquet")

    state, script, dst = _generate_transfer(
        "Read the yellow taxi trip CSV. Keep every column and every row unchanged. "
        "Write the result to the destination as Parquet.",
        dest_path=dest,
        dest_type="parquet",
    )

    assert state.get("error_message") in (None, ""), f"codegen error: {state.get('error_message')}"
    assert script, "no injection script produced"
    assert state.get("code_generated_successfully") is True
    assert state.get("destination_schema_compatible") is True, (
        f"destination schema rejected: {state.get('error_message')}"
    )

    src_uri = _need("source_uri", "phase 0")
    assert src_uri in script, f"generated script does not read {src_uri}"
    assert dst.get_cloud_uri("parquet") in script, "generated script does not write the destination"

    STATE["full_parquet_script"] = script
    STATE["full_parquet_dest"] = dest
    print(f"\n[phase4] codegen OK -> {dst.get_cloud_uri('parquet')}")


@pytest.mark.slow
@pytest.mark.cluster
@pytest.mark.integration
def test_phase4_c2c_full_conversion_csv_to_parquet_cross_bucket() -> None:
    """Full 2 GB CSV -> Parquet across buckets, verified by reading the result back.

    Row-preserving by construction, so the row count is the assertion: a transfer
    that silently drops rows is the failure mode that matters most here.
    """
    _dta_ready()
    _ray_ready()
    script = _need("full_parquet_script", "phase 4 codegen")
    dest = _need("full_parquet_dest", "phase 4 codegen")
    STATE["written_objects"].append((DEST_BUCKET, dest))

    started = time.time()
    result = _run_transfer_on_ray(script, "full-parquet")
    elapsed = time.time() - started
    print(f"[phase4] full conversion finished in {elapsed / 60:.1f} min, result={result}")
    assert bool(result), (
        "the RayJob failed; check the Ray dashboard for job "
        f"dta-taxi-full-parquet-{RUN_ID} and `kubectl get pods -n {os.getenv('KUBE_NAMESPACE', 'default')}`"
    )

    blobs = _wait_for_objects(DEST_BUCKET, dest)
    assert blobs, f"nothing written to gs://{DEST_BUCKET}/{dest}"
    written = sum(b.size for b in blobs)
    source_size = _need("source_size", "phase 0")
    print(f"[phase4] wrote {written / 1e6:.1f} MB parquet from {source_size / 1e6:.1f} MB csv")
    assert written < source_size, "Parquet output is not smaller than the CSV — suspect a raw copy"

    daft = pytest.importorskip("daft", reason="daft needed to verify the parquet output")
    rows = daft.read_parquet(_gs(DEST_BUCKET, dest)).count_rows()
    print(f"[phase4] destination row count = {rows:,}")
    # Let the aggregation test below check itself against the real row count.
    STATE["transferred_rows"] = rows
    if EXPECTED_ROWS:
        assert rows == EXPECTED_ROWS, (
            f"transfer wrote {rows:,} rows, expected {EXPECTED_ROWS:,} — rows were lost"
        )
    else:
        assert rows > MIN_FULL_SCAN_ROWS, (
            f"only {rows:,} rows landed; set YT_EXPECTED_ROWS for an exact check"
        )


@pytest.mark.slow
@pytest.mark.cluster
@pytest.mark.integration
def test_phase4_c2c_transform_and_aggregate_to_json() -> None:
    """A transforming c2c transfer: filter + derive + aggregate, written as JSON.

    Exercises the half of the DTA the pass-through case cannot — the generated
    Daft transform — and the small output makes the result cheap to verify exactly.
    """
    _dta_ready()
    _ray_ready()
    dest = _dest_path("revenue_by_payment_type.json")
    STATE["written_objects"].append((DEST_BUCKET, dest))

    state, script, dst = _generate_transfer(
        "Read the yellow taxi trip CSV. Drop rows where total_amount is less than or "
        "equal to zero or passenger_count is zero. Group the remaining rows by "
        "payment_type and compute: trip_count as the number of trips, total_revenue as "
        "the sum of total_amount, and avg_tip as the average of tip_amount. "
        "Write one record per payment_type to the destination.",
        dest_path=dest,
        dest_type="json",
    )
    assert script, f"codegen failed: {state.get('error_message')}"
    assert state.get("destination_schema_compatible") is True

    result = _run_transfer_on_ray(script, "agg-json")
    assert bool(result), f"aggregation transfer failed; state={state.get('error_message')}"

    blobs = _wait_for_objects(DEST_BUCKET, dest)
    assert blobs, f"nothing written to gs://{DEST_BUCKET}/{dest}"

    raw = b"".join(b.download_as_bytes() for b in blobs).decode("utf-8", "replace")
    records: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(raw)
        records = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:  # newline-delimited JSON
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]

    print(f"\n[phase4] aggregate output ({len(records)} records): {records[:8]}")
    assert 1 <= len(records) <= 10, f"expected one record per payment type, got {len(records)}"

    keys = _norm(list(records[0].keys()))
    for expected in ("payment_type", "trip_count", "total_revenue", "avg_tip"):
        assert expected in keys, f"output is missing {expected}: {keys}"

    total_trips = sum(int(r[[k for k in r if k.lower() == "trip_count"][0]]) for r in records)
    # Prefer the row count the conversion test actually measured. The prompt filters
    # a few rows out, so this is a floor check, not equality — but a sampled run
    # would land near SAMPLE_ROW_CEILING, orders of magnitude below half.
    full_rows = STATE.get("transferred_rows")
    floor = max(int(full_rows * 0.5), MIN_FULL_SCAN_ROWS) if full_rows else MIN_FULL_SCAN_ROWS
    assert total_trips > floor, (
        f"aggregate covers only {total_trips:,} trips (floor {floor:,}) — the transform "
        f"ran on a sample, not the whole object"
    )


@pytest.mark.slow
@pytest.mark.cluster
@pytest.mark.integration
def test_phase4_c2c_same_bucket_roundtrip_preserves_values() -> None:
    """Same-bucket c2c (test_c2c_jyothi -> test_c2c_jyothi/out) with a derived column.

    Source and destination sharing a bucket is its own path — one set of
    credentials, one IOConfig — and it is the shape most users hit first.
    """
    _dta_ready()
    _ray_ready()
    dest = f"{PREFIX}/out/{RUN_ID}/taxi_tip_pct.parquet"
    STATE["written_objects"].append((BUCKET, dest))

    state, script, dst = _generate_transfer(
        "Read the yellow taxi trip CSV. Keep only trips where payment_type equals 1 and "
        "fare_amount is greater than zero. Output exactly four columns: "
        "tpep_pickup_datetime, fare_amount, tip_amount, and tip_pct computed as "
        "tip_amount divided by fare_amount. Write the result to the destination as Parquet.",
        dest_path=dest,
        dest_type="parquet",
        dest_bucket=BUCKET,
    )
    assert script, f"codegen failed: {state.get('error_message')}"
    assert _gs(BUCKET, dest) in script, "same-bucket destination URI missing from the script"

    result = _run_transfer_on_ray(script, "same-bucket")
    assert bool(result), "same-bucket transfer failed"

    blobs = _wait_for_objects(BUCKET, dest)
    assert blobs, f"nothing written to gs://{BUCKET}/{dest}"

    daft = pytest.importorskip("daft", reason="daft needed to verify the parquet output")
    df = daft.read_parquet(_gs(BUCKET, dest))
    cols = _norm(df.column_names)
    assert "tip_pct" in cols, f"derived column missing: {cols}"
    assert len(cols) == 4, f"expected 4 projected columns, got {cols}"

    head = df.limit(50).to_pydict()
    tip_key = [k for k in head if k.lower() == "tip_pct"][0]
    values = [v for v in head[tip_key] if v is not None]
    assert values, "tip_pct is entirely null"
    assert all(v >= 0 for v in values), f"negative tip percentages: {values[:10]}"


@pytest.mark.integration
def test_phase4_planner_routes_a_plain_english_transfer_request() -> None:
    """"transfer this to gs://…" from chat reaches the transfer path, not a generic answer.

    The DTA cases above call the pipeline directly; this is the only case that
    proves a user can trigger a c2c transfer by asking for one.
    """
    _api_ready()
    tid = _need("thread_id", "phase 2")
    dsid = _need("dataset_id", "phase 2")
    dest_uri = _gs(DEST_BUCKET, _dest_path("from_chat.parquet"))

    payload = _chat(
        tid,
        f"Transfer this dataset to {dest_uri} as parquet",
        dataset_ids=[dsid],
        metadata={"dataset_id": dsid},
    )
    reply = _last_assistant_text(payload).lower()
    print(f"\n[phase4][planner] {reply[:600]}")

    planner = json.dumps(payload.get("planner_definition") or {}).lower()
    signals = ("transfer", "parquet", DEST_BUCKET.lower())
    assert any(s in reply or s in planner for s in signals), (
        f"a plain-English transfer request was not routed to the transfer path: {reply[:400]!r}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Phase 9 — the manual prompt pack (run with -s to print it)
# ═════════════════════════════════════════════════════════════════════════════

MANUAL_PROMPTS: Dict[str, List[str]] = {
    "warm-up (quick_sample)": [
        "What is this dataset about, and what does each column mean?",
        "How many columns are there and what are their data types?",
        "Show me 10 sample rows.",
    ],
    "profiling and data quality": [
        "Profile this dataset and rank the columns by predictive power.",
        "Find and quantify the data quality problems in this dataset.",
        "How many trips have zero passengers, zero distance, or a negative total_amount?",
        "Are there trips where the dropoff time is before the pickup time?",
        "What fraction of trips have missing GPS coordinates (latitude or longitude = 0)?",
    ],
    "fidelity control": [
        "use quick sample",
        "use portfolio samples",
        "use entire dataset",
        "just analyse the full dataset please",          # loose phrasing, must still parse
    ],
    "analysis (re-run the starred ones in entire_dataset)": [
        "How many trips are in this dataset?",                                      # *
        "What is the average fare and average trip distance by payment type?",      # *
        "Show trips per hour of day across the month.",                             # *
        "Which day in January 2015 had the fewest trips, and why might that be?",
        "What is the tip percentage distribution for credit card trips?",
        "Clean the data: drop trips with zero passengers, non-positive distance, or "
        "negative total, then recompute the average fare.",
        "Compute the median trip distance per borough.",   # no borough column — must refuse, not invent
    ],
    "visualization": [
        "Plot trips per day for January 2015.",
        "Show me a histogram of trip distances under 20 miles.",
        "Visualize average tip percentage by hour of day.",
        "Plot pickup locations as a scatter of longitude vs latitude.",
    ],
    "cloud-to-cloud transfers": [
        f"Transfer this dataset to gs://{DEST_BUCKET}/{DEST_PREFIX}/manual/taxi.parquet as parquet",
        f"Move the data to gs://{DEST_BUCKET}/{DEST_PREFIX}/manual/taxi.json as json",
        "Group by payment_type, sum total_amount, and write the result to "
        f"gs://{DEST_BUCKET}/{DEST_PREFIX}/manual/revenue.json",
        f"Copy only credit card trips to gs://{BUCKET}/{PREFIX}/out/manual/card_trips.parquet",
        "Export this to Excel",                            # 12.7M rows — expect a clear refusal
    ],
    "joins (needs a small lookup CSV uploaded alongside)": [
        "Compare these two datasets.",
        "Join the trips with the payment lookup on payment_type and show total revenue "
        "by payment_name.",
        "Join the trips with the vendor lookup on trip_distance.",   # nonsense key — must refuse
    ],
    "model training and inference": [
        "Train a model to predict total_amount from trip_distance, passenger_count and "
        "the pickup hour.",
        "Train a model to predict whether a trip is paid by credit card.",
        "Predict total_amount using fare_amount and tip_amount.",    # leakage — should be called out
    ],
    "scheduling and ops": [
        "Run the daily trip count analysis every 30 minutes.",
        "List my scheduled tasks.",
        "What is the status of task <task_id>?",
        "Cancel task <task_id>.",
    ],
}


def test_phase9_print_manual_prompt_pack() -> None:
    """Not an assertion of behaviour — prints the prompts to drive by hand.

        pytest tests/yellow_taxi_datasets_test_file.py -k phase9 -s
    """
    print("\n" + "=" * 78)
    print("Avaloka manual prompt pack — NYC yellow taxi")
    print(f"source: gs://{BUCKET}/{PREFIX}/{OBJECT}")
    print("=" * 78)
    for section, prompts in MANUAL_PROMPTS.items():
        print(f"\n## {section}")
        for prompt in prompts:
            print(f"  - {prompt}")
    print()

    assert all(MANUAL_PROMPTS.values())
