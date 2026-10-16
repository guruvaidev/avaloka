"""The contract between the product-analytics export and this ingest.

Two services, two schemas, stated independently on purpose. This is the test
that stops them drifting. It imports both sides -- nothing in either imports
the other -- and holds them to one rule in both directions:

* everything ``app/analytics`` can export, the ingest accepts;
* nothing else is accepted: not the stored-tier fields, not the question, not
  a name, and not the question-text path that exists in the exporter but is
  switched off.

The wire records are not written by hand. They are produced by the exporter's
own code from events posted to the real collector, so a change to what leaves
a deployment changes what is tested here without anyone editing this file.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.analytics import allowlist, config, export
from app.analytics import schema as produced
from app.analytics.collector import build_router
from app.analytics.store import Store as AnalyticsStore
from telemetry_service import schema as accepted
from telemetry_service.app import create_app
from telemetry_service.ratelimit import RateLimiter
from telemetry_service.schema import Rejected, validate, validate_batch
from telemetry_service.store import Store

REPO = Path(__file__).resolve().parents[2]
ENDPOINT = "https://ingest.example.test/v1/events"
USER = {"X-Test-User": "analyst-7"}

#: A question that names a column, a table and a file, the way analysts do.
QUESTION = "average monthly_charges by region_id from churn_customers in q3_layoffs_final.csv"
QUESTION_WORDS = ("monthly_charges", "region_id", "churn_customers", "q3_layoffs_final")

_ENV = (config.ENV_ENABLED, config.ENV_UMBRELLA, config.ENV_PROMPT_TEXT,
        config.ENV_RETENTION_DAYS, config.ENV_DB_URL, config.ENV_EXPORT,
        config.ENV_EXPORT_ENDPOINT)


# --------------------------------------------------------------------- helpers

def _sample(spec: produced.Field, last: bool):
    """A value the producer's own schema accepts for this field."""
    if spec.kind == "enum":
        return sorted(spec.values)[-1 if last else 0]
    if spec.kind == "int":
        return spec.hi if last else spec.lo
    if spec.kind == "bool":
        return last
    if spec.kind == "id":
        return uuid.uuid4().hex
    if spec.kind == "class":
        return "pandas.errors.ParserError" if last else "KeyError"
    if spec.kind == "text":
        return QUESTION
    raise AssertionError(f"a field kind this test does not know: {spec.kind}")


def _events() -> list[dict]:
    """For every event type the producer declares: one with only its required
    fields, and two with every field it may carry, at each end of its range."""
    out = []
    for name, (required, optional) in sorted(produced.EVENTS.items()):
        for fields, last in ((required, False),
                             ({**produced.COMMON_OPTIONAL, **optional, **required}, False),
                             ({**produced.COMMON_OPTIONAL, **optional, **required}, True)):
            seq = len(out)
            out.append({
                "event_id": uuid.uuid4().hex, "event_type": name, "seq": seq, "t_ms": seq * 10,
                **{key: _sample(spec, last) for key, spec in fields.items()},
            })
    return out


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(config.ENV_EXPORT, "on")
    monkeypatch.setenv(config.ENV_EXPORT_ENDPOINT, ENDPOINT)


@pytest.fixture()
def deployment():
    """A deployment's analytics store, filled through its real collector."""
    store = AnalyticsStore("sqlite:///:memory:")
    app = FastAPI()
    app.include_router(build_router(lambda r: r.headers.get("X-Test-User"), store=store))
    client = TestClient(app)
    events = _events()
    response = client.post("/analytics/events", headers=USER, json={
        "schema_version": produced.SCHEMA_VERSION, "session_id": uuid.uuid4().hex,
        "ui_version": "1.6.2", "events": events,
    })
    assert response.status_code == 202, response.text
    assert response.json()["rejected"] == [] and response.json()["accepted"] == len(events)
    return store


@pytest.fixture()
def wire(deployment) -> list[dict]:
    """What the exporter would send, captured where it hands off to the network."""
    sent: list[dict] = []

    def capture(endpoint, records, timeout=15.0):
        sent.extend(records)
        return 202

    report = export.export_pending(deployment, post=capture)
    assert report.get("exported") == len(sent) > 0, report
    return sent


def _wire_type(name: str) -> str:
    return export.WIRE_PREFIX + name


# ------------------------------------------------- what is exported is accepted

class TestEverythingExportedIsAccepted:
    def test_the_fixture_covers_every_event_type_the_producer_declares(self, wire):
        """Otherwise the acceptance test below proves less than it says."""
        assert {r["event_type"] for r in wire} == {_wire_type(n) for n in produced.EVENTS}
        assert len(produced.EVENTS) >= 12

    @pytest.mark.parametrize("name", sorted(produced.EVENTS))
    def test_every_exported_record_of_this_type_validates(self, wire, name):
        records = [r for r in wire if r["event_type"] == _wire_type(name)]
        assert len(records) == 3
        for record in records:
            assert validate(dict(record)) == record

    def test_the_whole_export_is_accepted_as_one_batch(self, wire):
        assert len(validate_batch([dict(r) for r in wire])) == len(wire)

    def test_the_exporters_own_request_is_accepted_over_http(self, deployment, tmp_path,
                                                             monkeypatch):
        """The real serialisation, not a re-implementation of it: the bytes
        ``export._post`` builds are handed to the ingest endpoint unchanged."""
        ingest_store = Store(path=tmp_path / "ingest.db")
        ingest = TestClient(create_app(store=ingest_store, limiter=RateLimiter(limit=100)))
        statuses = []

        class _Response:
            def __init__(self, status):
                self.status = status

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def urlopen(request, timeout=None):
            assert request.full_url == ENDPOINT
            response = ingest.post("/v1/events", content=request.data,
                                   headers=dict(request.header_items()))
            statuses.append((response.status_code, response.text))
            return _Response(response.status_code)

        monkeypatch.setattr(export.urllib.request, "urlopen", urlopen)
        report = export.export_pending(deployment)

        assert statuses and statuses[0][0] == 202, statuses
        assert report["exported"] == ingest_store.count() == 3 * len(produced.EVENTS)
        held = json.dumps(ingest_store.records(limit=10 ** 6))
        for word in QUESTION_WORDS:
            assert word not in held, f"{word!r} from the question reached the ingest"
        ingest_store.close()


# ------------------------------------------------ the two schemas say the same

class TestTheTwoSchemasAgree:
    """Structure, not samples: a vocabulary that gains a value on one side, or
    a range that widens, fails here even if no sample happens to use it."""

    def test_the_event_types_are_the_same_set(self):
        analytics = {t for t in accepted.EVENT_TYPES if t.startswith(accepted.ANALYTICS_PREFIX)}
        assert analytics == {_wire_type(name) for name in produced.EVENTS}
        assert accepted.ANALYTICS_PREFIX == export.WIRE_PREFIX
        assert set(accepted.ANALYTICS_EVENTS) == set(produced.EVENTS)

    def test_the_process_types_are_untouched(self):
        assert accepted.EVENT_TYPES - {_wire_type(n) for n in produced.EVENTS} == {
            "install", "agent.started", "agent.completed", "dataset"}

    @pytest.mark.parametrize("name", sorted(produced.EVENTS))
    def test_this_type_accepts_exactly_the_fields_the_producer_can_export(self, name):
        required, optional = produced.EVENTS[name]
        declared = {**produced.COMMON_OPTIONAL, **optional, **produced.COMMON_REQUIRED, **required}
        client_fields = {k: spec for k, spec in declared.items()
                         if not spec.sensitive and k != "event_type"}
        added_on_export = {"session_id", "install_id", "actor_id", "schema_version", "ui_version"}
        if any(spec.sensitive and spec.kind == "text" for spec in declared.values()):
            # The one thing said about a question: its length band.
            added_on_export = added_on_export | {"prompt_chars_bucket"}

        ours = accepted.ANALYTICS_FIELDS_BY_TYPE[_wire_type(name)]
        assert set(ours) == set(client_fields) | added_on_export
        assert accepted.KEYS_BY_TYPE[_wire_type(name)] == set(ours) | {"event_type"}

        for key, theirs in client_fields.items():
            mine = ours[key]
            assert mine.kind == theirs.kind, key
            if theirs.kind == "enum":
                assert mine.values == theirs.values, key
            if theirs.kind == "int":
                assert (mine.lo, mine.hi) == (theirs.lo, theirs.hi), key

    def test_the_patterns_are_the_same(self):
        assert accepted._OPAQUE_ID.pattern == produced._OPAQUE_ID.pattern
        assert accepted._CLASS_NAME.pattern == produced._CLASS_NAME.pattern
        assert accepted._VERSION.pattern == produced._VERSION.pattern

    def test_the_length_bands_are_the_same(self):
        bands = accepted.ANALYTICS_EVENTS["question.submitted"]["prompt_chars_bucket"]
        assert bands.values == set(produced.PROMPT_CHAR_BUCKETS)

    def test_the_accepted_keys_are_the_export_permit_list_and_no_more(self):
        ours = set().union(*(accepted.KEYS_BY_TYPE[_wire_type(n)] for n in produced.EVENTS))
        assert ours == set(allowlist.EXPORT_KEYS)

    def test_the_published_schema_says_the_same(self):
        published = json.loads((REPO / "docs/analytics/event-schema.v1.json").read_text())
        exported = published["representations"]["exported"]
        ours = set().union(*(accepted.KEYS_BY_TYPE[_wire_type(n)] for n in produced.EVENTS))
        assert set(exported["fields"]) == ours
        assert exported["event_type_prefix"] == accepted.ANALYTICS_PREFIX
        assert {_wire_type(n) for n in published["events"]} == {
            t for t in accepted.EVENT_TYPES if t.startswith(accepted.ANALYTICS_PREFIX)}


# --------------------------------------------------- what stays in the cluster

#: Everything the deployment stores that must not leave it, by the producer's
#: own lists, plus the names an analyst's question carries.
STAYS_LOCAL = sorted(
    set(allowlist.SENSITIVE_STORED)
    | (set(allowlist.STORED_KEYS) - set(allowlist.EXPORT_KEYS))
    | set(allowlist.EXPORT_TEXT_KEYS)
    | {"column_names", "table_name", "file_name", "dataset_name"}
)


class TestTheStoredTierIsRefused:
    def test_the_list_is_what_it_claims(self):
        for key in ("prompt", "prompt_export", "prompt_stripped", "column_names",
                    "table_name", "file_name"):
            assert key in STAYS_LOCAL

    @pytest.mark.parametrize("key", STAYS_LOCAL)
    def test_a_stored_only_field_is_rejected_on_every_analytics_type(self, wire, key):
        for record in wire:
            with pytest.raises(Rejected):
                validate({**record, key: QUESTION})

    def test_the_question_is_stored_locally_and_absent_from_the_wire(self, deployment, wire):
        """Not vacuous: the deployment does hold the text. It just never leaves."""
        local = json.dumps(deployment.all_rows(), default=str)
        assert "monthly_charges" in local
        sent = json.dumps(wire)
        for word in QUESTION_WORDS:
            assert word not in sent
        assert "prompt" not in {key for record in wire for key in record} - {"prompt_chars_bucket"}

    def test_every_name_the_producer_refuses_the_ingest_refuses_too(self):
        assert produced.NEVER_ACCEPTED <= accepted.REFUSED_KEYS
        assert allowlist.SENSITIVE_STORED <= accepted.REFUSED_KEYS
        assert allowlist.EXPORT_TEXT_KEYS <= accepted.REFUSED_KEYS

    def test_the_disabled_question_text_path_is_refused_if_someone_enables_it(self, monkeypatch):
        """``export.EXPORT_QUESTION_TEXT`` is False and is a constant. If it is
        ever changed without this ingest being changed too, what it would send
        is rejected here rather than stored."""
        assert export.EXPORT_QUESTION_TEXT is False
        row = {
            "event_id": uuid.uuid4().hex, "event_type": "question.submitted",
            "session_id": uuid.uuid4().hex, "schema_version": produced.SCHEMA_VERSION,
            "ui_version": None, "seq": 0, "t_ms": 0, "route": None,
            "turn_id": uuid.uuid4().hex, "outcome": None, "error_class": None,
            "duration_ms": None, "actor_id": "a" * 32,
            "received_at": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
            "prompt": QUESTION, "prompt_export": "average [name] by [name]",
            "props": json.dumps({"input_method": "typed", "prompt_chars_bucket": "81-200",
                                 "prompt_names_exact": 2}),
        }
        install = uuid.uuid4().hex

        as_shipped = export.to_wire(row, install, b"s" * 32)
        assert validate(dict(as_shipped))

        monkeypatch.setattr(export, "EXPORT_QUESTION_TEXT", True)
        enabled = export.to_wire(row, install, b"s" * 32)
        assert "prompt_stripped" in enabled, "the disabled path produced nothing to refuse"
        with pytest.raises(Rejected) as exc:
            validate(dict(enabled))
        assert "refused" in exc.value.reason


# ------------------------------------------------------- the set stays closed

class TestTheSetStaysClosed:
    def test_an_unknown_analytics_type_is_rejected(self, wire):
        for unknown in ("analytics.question.edited", "analytics.", "question.submitted",
                        "analytics.page.viewed "):
            with pytest.raises(Rejected):
                validate({**wire[0], "event_type": unknown})

    def test_an_unknown_key_is_rejected_on_every_analytics_type(self, wire):
        for record in wire:
            with pytest.raises(Rejected) as exc:
                validate({**record, "scroll_depth": 3})
            assert "scroll_depth" in exc.value.reason

    def test_a_key_from_another_type_is_rejected(self, wire):
        """`feature` belongs to feature.used; `agent` to a process record."""
        page = next(r for r in wire if r["event_type"] == "analytics.page.viewed")
        for key, value in (("feature", "auto_insights"), ("agent", "validator"),
                           ("llm_model", "some-model"), ("prompt_chars_bucket", "1-20")):
            with pytest.raises(Rejected):
                validate({**page, key: value})

    def test_a_value_outside_its_vocabulary_is_rejected(self, wire):
        """A route is a template. A concrete path can carry a share token."""
        page = next(r for r in wire if r["event_type"] == "analytics.page.viewed")
        for key, value in (("route", "/shared/8fa2c1d09e7b4a55"), ("route", QUESTION),
                           ("seq", -1), ("seq", True), ("session_id", "ravi_kumar_acme_corp"),
                           ("ui_version", "tell me about churn"), ("schema_version", "1"),
                           ("actor_id", "analyst-7@acme.example")):
            with pytest.raises(Rejected) as exc:
                validate({**page, key: value})
            assert str(value) not in exc.value.reason or isinstance(value, (int, bool))

    def test_an_error_message_cannot_travel_as_an_error_class(self, wire):
        shown = next(r for r in wire if r["event_type"] == "analytics.error.shown")
        with pytest.raises(Rejected):
            validate({**shown, "error_class": "KeyError: 'monthly_charges'"})
