"""Owner decision D16.5 (2026-10-31): the typed prompt is never exported.

Prompts stay in the deployment's own database; only metadata leaves. These
tests hold that at the transport, not at a helper: the collector takes a real
batch over HTTP, ``export_pending`` runs with its real ``_post``, and
``urllib.request.urlopen`` -- the one network call product analytics makes --
is replaced by a recorder that gunzips and keeps every request body. A socket
guard fails the test if anything tries to reach a network at all, so no real
telemetry is ever sent.

The turn: an analyst uploads a CSV whose file name and header are probes, the
UI reports the connection and the typed question (which names the column, the
file and a free-text probe), and the question completes. None of the three
probes may appear in any outbound byte.

The positive control proves the recorder is not blind: a builder that carries
the question the way the withdrawn decision allowed puts the probe on the wire.
"""
import csv
import gzip
import inspect
import json
import socket
import urllib.request
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.analytics import allowlist, export, redact, schema
from app.analytics.collector import build_router
from tests.analytics.conftest import USER, batch, ev

ENDPOINT = "https://ingest.example.test/v1/events"

PROBE_TEXT = "ZQXPROBE7731"
PROBE_COLUMN = "zqx_probe_col_4417"
PROBE_FILE = "zqx_probe_file_9902.csv"
PROBES = (PROBE_TEXT, PROBE_COLUMN, PROBE_FILE)

#: Keys of the withdrawn question-text export path; none may ever be on the wire.
QUESTION_TEXT_KEYS = {"prompt", "prompt_export", "prompt_stripped", "prompt_redactions",
                      "prompt_names_exact", "prompt_names_common_word", "prompt_names_substring"}


class Transport:
    """Stands in for ``urllib.request.urlopen``: records, decodes, answers 200."""

    def __init__(self):
        self.requests = []

    def __call__(self, request, timeout=None):
        body = request.data or b""
        if request.get_header("Content-encoding") == "gzip":
            body = gzip.decompress(body)
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()),
                              "body": body.decode("utf-8")})

        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Response()

    @property
    def records(self):
        return [json.loads(line) for r in self.requests for line in r["body"].splitlines() if line]

    @property
    def everything(self):
        """Every outbound byte, URL and header included, case-folded."""
        return json.dumps(self.requests).casefold()


@pytest.fixture
def transport(monkeypatch):
    sent = Transport()
    monkeypatch.setattr(urllib.request, "urlopen", sent)

    def _no_network(*args, **kwargs):
        raise AssertionError(f"a real network connection was attempted: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    return sent


@pytest.fixture
def export_on(monkeypatch):
    monkeypatch.setenv("AVALOKA_ANALYTICS_EXPORT", "on")
    monkeypatch.setenv("AVALOKA_ANALYTICS_EXPORT_ENDPOINT", ENDPOINT)


@pytest.fixture
def uploaded_csv(tmp_path):
    path = tmp_path / PROBE_FILE
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([PROBE_COLUMN, "amount"])
        writer.writerows([["a", "1"], ["b", "2"]])
    with path.open(newline="") as fh:
        headers = next(csv.reader(fh))
    return path, headers


def _client(store, headers):
    # The pre-D16.5 collector took a names_provider: handing it the uploaded
    # file's headers is what made it prepare an exportable form of the question.
    # The fixed collector has no such parameter. Passing it where it exists
    # keeps this test meaningful against the old tree (the bite).
    kwargs = {}
    if "names_provider" in inspect.signature(build_router).parameters:
        kwargs["names_provider"] = lambda request, user: set(headers) | {PROBE_FILE}
    app = FastAPI()
    app.include_router(build_router(lambda r: r.headers.get("X-Test-User"), store=store, **kwargs))
    return TestClient(app)


def run_turn(store, uploaded):
    """One analyst turn as the UI reports it, ingested through the real route."""
    path, headers = uploaded
    question = (f"average {headers[0]} by month from {path.name}, "
                f"and flag rows whose note says {PROBE_TEXT}")
    assert all(p.casefold() in question.casefold() for p in PROBES)
    turn = uuid.uuid4().hex
    r = _client(store, headers).post("/analytics/events", headers=USER, json=batch(
        ev("dataset.connect_started", 0, source_kind="upload"),
        ev("dataset.connect_completed", 1, source_kind="upload", outcome="success",
           file_format="csv", column_count_bucket="1-10", row_count_bucket="1-1e2", duration_ms=40),
        ev("question.submitted", 2, turn_id=turn, input_method="typed", prompt=question),
        ev("question.completed", 3, turn_id=turn, outcome="success", duration_ms=900,
           result_kind="table"),
    ))
    assert r.status_code == 202 and r.json()["accepted"] == 4, r.json()
    return question


def assert_no_probe(sent):
    assert sent.requests, "nothing was exported: the test would pass vacuously"
    for probe in PROBES:
        assert probe.casefold() not in sent.everything, f"{probe!r} left the cluster"
    keys = {k for record in sent.records for k in record}
    assert not (keys & QUESTION_TEXT_KEYS), f"question-text keys on the wire: {keys & QUESTION_TEXT_KEYS}"


# --------------------------------------------------------------- egress test

@pytest.mark.parametrize("env", [
    {},
    {"AVALOKA_ANALYTICS_PROMPT_TEXT": "on"},
    {"AVALOKA_ANALYTICS_PROMPT_TEXT": "off"},
    {"AVALOKA_ANALYTICS_EXPORT_QUESTION_TEXT": "on"},
    {"AVALOKA_ANALYTICS": "on", "AVALOKA_TELEMETRY": "on"},
], ids=["default", "prompt-text-on", "prompt-text-off", "export-question-env", "everything-on"])
def test_no_probe_reaches_any_outbound_request(store, export_on, transport, uploaded_csv, monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    run_turn(store, uploaded_csv)
    report = export.export_pending(store)
    assert report.get("exported") == 4, report
    assert [r["url"] for r in transport.requests] == [ENDPOINT]
    assert_no_probe(transport)


def test_no_probe_leaves_even_with_the_old_question_text_switch_set(store, export_on, transport,
                                                                   uploaded_csv, monkeypatch):
    """The withdrawn path was opened by a code constant. Setting that name now
    does nothing: the path is gone, not switched off."""
    monkeypatch.setattr(export, "EXPORT_QUESTION_TEXT", True, raising=False)
    run_turn(store, uploaded_csv)
    export.export_pending(store)
    assert_no_probe(transport)


def test_prompts_stay_local(store, export_on, transport, uploaded_csv):
    """Not vacuous: the deployment does hold the question, probes and all."""
    question = run_turn(store, uploaded_csv)
    export.export_pending(store)
    local = [row["prompt"] for row in store.all_rows() if row["prompt"]]
    assert len(local) == 1 and PROBE_TEXT in local[0] and PROBE_COLUMN in local[0]
    assert local[0] == redact.scrub(question)[0]
    assert_no_probe(transport)


def test_a_client_sending_names_directly_is_refused_and_nothing_of_it_leaves(store, export_on, transport,
                                                                            uploaded_csv):
    path, headers = uploaded_csv
    r = _client(store, headers).post("/analytics/events", headers=USER, json=batch(
        ev("dataset.connect_completed", 0, source_kind="upload", outcome="success",
           column_names=headers, file_name=path.name, table_name=PROBE_TEXT),
        ev("feature.used", 1, feature="dataset_upload") if "dataset_upload" in schema.FEATURES
        else ev("page.viewed", 1, route=sorted(schema.ROUTES)[0]),
    ))
    assert r.status_code == 202 and r.json()["accepted"] == 1 and len(r.json()["rejected"]) == 1
    export.export_pending(store)
    assert_no_probe(transport)


# ---------------------------------------------------------- positive control

_TO_WIRE = export.to_wire


def _legacy_to_wire(row, install_id, secret=b""):
    """What the withdrawn decision allowed: the typed question on the record."""
    record = _TO_WIRE(row, install_id, secret)
    if row.get("prompt"):
        record["prompt"] = row["prompt"]
    return record


def test_positive_control_the_recorder_sees_a_prompt_that_is_sent(store, export_on, transport,
                                                                  uploaded_csv, monkeypatch):
    """With the old builder and no egress check, the probe is on the wire and
    the recorder finds it -- so the absence asserted above means something."""
    monkeypatch.setattr(export, "to_wire", _legacy_to_wire)
    monkeypatch.setattr(allowlist, "exportable", lambda record: record, raising=False)
    run_turn(store, uploaded_csv)
    export.export_pending(store)
    assert PROBE_TEXT.casefold() in transport.everything
    assert PROBE_COLUMN.casefold() in transport.everything
    with pytest.raises(AssertionError, match="left the cluster"):
        assert_no_probe(transport)


def test_the_egress_check_drops_a_prompt_whatever_built_the_record(store, export_on, transport,
                                                                  uploaded_csv, monkeypatch):
    """The old builder again, but this time the single egress point applies
    the allowlist: the prompt does not leave."""
    monkeypatch.setattr(export, "to_wire", _legacy_to_wire)
    run_turn(store, uploaded_csv)
    export.export_pending(store)
    assert_no_probe(transport)


# ------------------------------------------------------------ allowlist test

#: Metadata, by kind. Stated here independently of the product's allowlist so a
#: key added there that is not plainly metadata fails review in this file.
METADATA_KEYS = {
    "event_type", "event_id", "seq", "t_ms",
    "duration_ms", "first_response_ms", "active_ms", "turn_index", "retry_count", "http_status",
    "row_count_bucket", "column_count_bucket", "prompt_chars_bucket",
    "outcome", "error_class", "reason",
    "route", "from_route", "entry", "viewport", "method", "source_kind", "file_format",
    "input_method", "is_retry", "retry_of", "turn_id", "result_kind", "chart_type", "action",
    "export_format", "feedback_type", "has_comment", "surface", "feature",
    "schema_version", "ui_version", "session_id", "install_id", "actor_id",
}


def test_exported_payload_keys_are_a_subset_of_the_allowlist(store, export_on, transport, uploaded_csv):
    run_turn(store, uploaded_csv)
    export.export_pending(store)
    keys = {k for record in transport.records for k in record}
    assert keys and keys <= allowlist.EXPORT_KEYS, keys - allowlist.EXPORT_KEYS


def test_the_allowlist_is_metadata_only():
    assert allowlist.EXPORT_KEYS <= METADATA_KEYS, allowlist.EXPORT_KEYS - METADATA_KEYS
    assert not (allowlist.EXPORT_KEYS & QUESTION_TEXT_KEYS)
    assert not (allowlist.EXPORT_KEYS & schema.NEVER_ACCEPTED)
    assert not (allowlist.EXPORT_KEYS & schema.SENSITIVE_FIELDS)


@pytest.mark.parametrize("event_type", sorted(schema.EVENT_TYPES))
def test_every_event_type_exports_only_allowlisted_keys_even_with_the_old_switch_set(event_type, monkeypatch):
    monkeypatch.setattr(export, "EXPORT_QUESTION_TEXT", True, raising=False)
    required, optional = schema.EVENTS[event_type]
    row = {
        "event_id": uuid.uuid4().hex, "event_type": event_type, "session_id": uuid.uuid4().hex,
        "schema_version": schema.SCHEMA_VERSION, "ui_version": "1.6.0", "seq": 0, "t_ms": 0,
        "route": None, "turn_id": None, "outcome": None, "error_class": None, "duration_ms": None,
        "actor_id": "a" * 32, "received_at": None,
        "prompt": f"average {PROBE_COLUMN} from {PROBE_FILE} {PROBE_TEXT}",
        "prompt_export": f"average [name] from [name] {PROBE_TEXT}",
        "props": json.dumps({"input_method": "typed", "prompt_chars_bucket": "21-80",
                             "prompt_redactions": 0, "prompt_names_exact": 2,
                             "column_names": [PROBE_COLUMN], "file_name": PROBE_FILE}),
    }
    record = export.to_wire(row, uuid.uuid4().hex, b"s" * 32)
    assert set(record) <= allowlist.EXPORT_KEYS, set(record) - allowlist.EXPORT_KEYS
    assert "zqx" not in json.dumps(record).casefold()
