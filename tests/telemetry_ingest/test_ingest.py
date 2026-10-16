"""Tests for the telemetry ingest service.

The server on its own: what it accepts, what it refuses, what it stores, and
how it behaves under abuse. The tests that wire a real client uploader to this
server belong with the uploader and arrive with it; nothing here imports a
client.
"""

from __future__ import annotations

import gzip
import json
import time

import pytest
from fastapi.testclient import TestClient

from telemetry_service.app import create_app
from telemetry_service.ratelimit import RateLimiter
from telemetry_service.retention import purge
from telemetry_service import schema
from telemetry_service.schema import Rejected, validate, validate_batch
from telemetry_service.store import Store

NEWLINE = chr(10)


# --------------------------------------------------------------------- helpers

def event(index: int = 0, **overrides) -> dict:
    record = {
        "event_id": f"{index:032x}",
        "event_type": "agent.completed",
        "agent": "data_scout",
        "outcome": "success",
        "duration_ms": 42,
    }
    record.update(overrides)
    return record


def body(records: list[dict], compress: bool = True) -> bytes:
    payload = (NEWLINE.join(json.dumps(r) for r in records) + NEWLINE).encode("utf-8")
    return gzip.compress(payload) if compress else payload


@pytest.fixture()
def store(tmp_path):
    s = Store(path=tmp_path / "telemetry.db")
    yield s
    s.close()


@pytest.fixture()
def client(store):
    return TestClient(create_app(store=store, limiter=RateLimiter(limit=1000)))


def post(client, records, compress=True, headers=None):
    return client.post(
        "/v1/events",
        content=body(records, compress),
        headers={"Content-Encoding": "gzip" if compress else "identity",
                 "Content-Type": "application/x-ndjson", **(headers or {})},
    )


# ------------------------------------------------------------------ validation

class TestSchemaRejects:
    """The server rejects rather than filters. Filtering would quietly accept a
    client that had started sending more than it should."""

    def test_a_valid_record_passes(self):
        assert validate(event())["agent"] == "data_scout"

    def test_an_unpermitted_field_is_rejected(self):
        with pytest.raises(Rejected) as exc:
            validate(event(customer_email="someone@example.com"))
        assert "customer_email" in exc.value.reason

    def test_an_error_message_is_rejected(self):
        """The client drops it; if one ever arrives, something is wrong upstream."""
        with pytest.raises(Rejected):
            validate(event(error_message="KeyError: 'monthly_charges'"))

    def test_an_unknown_event_type_is_rejected(self):
        with pytest.raises(Rejected):
            validate(event(event_type="something.new"))

    def test_a_missing_event_id_is_rejected(self):
        record = event()
        del record["event_id"]
        with pytest.raises(Rejected):
            validate(record)

    def test_an_out_of_vocabulary_outcome_is_rejected(self):
        with pytest.raises(Rejected):
            validate(event(outcome="probably_fine"))

    def test_an_overlong_string_is_rejected(self):
        with pytest.raises(Rejected):
            validate(event(agent="X" * 500))

    def test_column_type_counts_are_allowed_but_bounded(self):
        shape = {"event_id": "a" * 32, "event_type": "dataset"}
        assert validate({**shape, "column_type_counts": {"numeric": 12}})
        with pytest.raises(Rejected):
            validate({**shape, "column_type_counts": {f"t{i}": 1 for i in range(100)}})

    def test_a_key_from_another_event_type_is_rejected(self):
        """A record is checked against its own type's keys. Checking against
        every key any type may carry let `agent` ride on an install record,
        and would have let it ride on an analytics one."""
        base = {"event_id": "a" * 32}
        assert validate({**base, "event_type": "install", "os": "linux"})
        for record in [
            {**base, "event_type": "install", "agent": "validator"},
            {**base, "event_type": "dataset", "outcome": "success"},
            {**base, "event_type": "agent.completed", "os": "linux"},
            {**base, "event_type": "agent.started", "file_format": "csv"},
        ]:
            with pytest.raises(Rejected) as exc:
                validate(record)
            assert "unpermitted" in exc.value.reason

    def test_every_event_type_has_its_own_key_set(self):
        assert set(schema.KEYS_BY_TYPE) == set(schema.EVENT_TYPES)
        for keys in schema.KEYS_BY_TYPE.values():
            assert {"event_id", "event_type"} <= keys

    def test_one_bad_record_rejects_the_whole_batch(self):
        """All-or-nothing keeps the client's cursor arithmetic simple."""
        with pytest.raises(Rejected):
            validate_batch([event(0), event(1, secret_field="x"), event(2)])

    def test_an_empty_batch_is_rejected(self):
        with pytest.raises(Rejected):
            validate_batch([])

    def test_an_oversized_batch_is_rejected(self):
        with pytest.raises(Rejected):
            validate_batch([event(i) for i in range(10_001)])


# ------------------------------------------------------------ customer content

#: What the original design accepted and this one does not. Each was removed
#: from the accepted keys, not merely discouraged.
REMOVED_KEYS = {
    "prompt": "average spend per region?",
    "column_names": ["customer_id", "region"],
}

#: A value for every other refused name, shaped the way a careless client
#: would send it. Built from the list itself, so a name added there is tested
#: here without anyone remembering to.
REFUSED_SAMPLES = {
    name: REMOVED_KEYS.get(name, "x") for name in sorted(schema.REFUSED_KEYS)
}

SENTINEL = "monthly_charges_for_acme_corp"


class TestCustomerContentIsRefused:
    """Telemetry is behavioural only. Question text, column names, table
    names and file names are customer data and are never transmitted -- so the
    server must not be able to accept them. Refused means the batch is
    rejected and nothing is stored: dropping the field and keeping the rest
    would hide a client that had started sending it."""

    @pytest.mark.parametrize("key", sorted(REMOVED_KEYS))
    @pytest.mark.parametrize("event_type",
                             ["install", "agent.started", "agent.completed", "dataset"])
    def test_a_removed_key_is_rejected_on_every_event_type(self, key, event_type):
        record = {"event_id": "a" * 32, "event_type": event_type, key: REMOVED_KEYS[key]}
        with pytest.raises(Rejected) as exc:
            validate(record)
        assert key in exc.value.reason and "refused" in exc.value.reason

    @pytest.mark.parametrize("key", sorted(REMOVED_KEYS))
    def test_a_removed_key_gets_400_and_nothing_is_stored(self, client, store, key):
        """Rejected, not dropped: the good records beside it are not stored
        either, and the status is the permanent one."""
        response = post(client, [event(0), event(1, **{key: REMOVED_KEYS[key]})])
        assert response.status_code == 400
        assert key in response.json()["reason"]
        assert store.count() == 0

    @pytest.mark.parametrize("key", sorted(REFUSED_SAMPLES))
    def test_every_refused_name_is_rejected(self, key):
        with pytest.raises(Rejected) as exc:
            validate(event(**{key: REFUSED_SAMPLES[key]}))
        assert "refused" in exc.value.reason

    def test_no_refused_name_is_on_an_accepted_list(self):
        """The service refuses to start if one is; this says so before a deploy
        does."""
        assert not (schema.ALLOWED_KEYS & schema.REFUSED_KEYS)
        for name in ("prompt", "column_names"):
            assert name in schema.REFUSED_KEYS
            assert name not in schema.ALLOWED_KEYS

    def test_the_limits_that_existed_only_for_them_are_gone(self):
        """A length cap on prompts implies prompts. There is nothing to cap."""
        for name in ("MAX_PROMPT_LENGTH", "MAX_COLUMN_NAMES"):
            assert not hasattr(schema, name)

    def test_the_refused_value_is_never_echoed(self, client, store, caplog):
        """The reason names the field. It must not carry what was in it, into
        the response or into the log."""
        with caplog.at_level("WARNING"):
            response = post(client, [event(0, prompt=f"total {SENTINEL} by month")])
        assert response.status_code == 400
        assert SENTINEL not in response.text
        assert SENTINEL not in caplog.text
        assert store.count() == 0

    def test_an_error_message_cannot_travel_as_an_error_class(self):
        """`error_class` is permitted so that `error_message` need not be. That
        only holds if a message cannot be sent under the permitted name."""
        assert validate(event(error_class="KeyError"))
        assert validate(event(error_class="pandas.errors.ParserError"))
        for message in ["KeyError: 'monthly_charges'",
                        "could not open /data/q3_layoffs_final.csv",
                        "X" * 500, "", 7]:
            with pytest.raises(Rejected) as exc:
                validate(event(error_class=message))
            assert "monthly_charges" not in exc.value.reason
            assert "q3_layoffs" not in exc.value.reason

    def test_a_type_count_key_must_be_a_kind_of_column(self):
        """Counts by kind of column. The keys are a vocabulary, not a shape: a
        column called `salary` looks exactly like a kind called `numeric`, so
        only a fixed list keeps a column name out."""
        shape = {"event_id": "a" * 32, "event_type": "dataset"}
        assert validate({**shape, "column_type_counts": {"numeric": 12, "datetime": 1}})
        assert validate({**shape, "column_type_counts": {k: 1 for k in schema.COLUMN_KINDS}})
        for key in ["salary", "patient_hiv_status", "Customer Name",
                    "q3_layoffs_final.csv", "unknown", ""]:
            with pytest.raises(Rejected) as exc:
                validate({**shape, "column_type_counts": {"numeric": 3, key: 1}})
            assert key == "" or key not in exc.value.reason
        for count in [-1, True, "12", 1.5, None, schema.MAX_COLUMNS_COUNTED + 1]:
            with pytest.raises(Rejected):
                validate({**shape, "column_type_counts": {"numeric": count}})
        for not_counts in ["numeric", ["numeric"], 3]:
            with pytest.raises(Rejected):
                validate({**shape, "column_type_counts": not_counts})


# ----------------------------------------------------------- every value closed

#: The kinds of rule a value may be held to. None of them is "a string".
CLOSED_KINDS = {"enum", "int", "bool", "id", "class", "version", "release_version",
                "schema_version", "counts"}

#: What a customer's own words look like when they turn up in a field: a file
#: name, a column name, a model deployment, a path, a bucket, a question, a
#: host. Each must be rejected by every field of every event type.
CUSTOMER_SHAPED = [
    "Q3 payroll.xlsx",
    "patient_hiv_status",
    "acme-prod-gpt4-deployment",
    "/data/acme/churn_customers.csv",
    "s3://acme-finance/exports",
    "what is average salary by region",
    "db01.internal.acme.example",
    "",
]

#: The one rule that constrains form and not meaning. A class name is a dotted
#: identifier, and so are a column name and a host name; `error_class`
#: therefore rejects everything above except these two. Stated here rather than
#: hidden: closing it would need a vocabulary of exception classes.
PASSES_AS_A_CLASS_NAME = {"patient_hiv_status", "db01.internal.acme.example"}

PROCESS_FIELDS = sorted(
    (event_type, key)
    for event_type, fields in schema.PROCESS_FIELDS_BY_TYPE.items()
    for key in fields
)


def _valid(spec, last=False):
    if spec.kind == "enum":
        return sorted(spec.values)[-1 if last else 0]
    if spec.kind == "int":
        return spec.hi if last else spec.lo
    if spec.kind == "bool":
        return last
    if spec.kind == "id":
        return "0f8fad5b-d9cb-469f-a165-70867728950e" if last else "a" * 32
    if spec.kind == "class":
        return "pandas.errors.ParserError" if last else "KeyError"
    if spec.kind == "version":
        return "1.7.0-rc1" if last else "3.11"
    if spec.kind == "release_version":
        return "3.13.0rc1" if last else "3.11"
    if spec.kind == "schema_version":
        return "1.1"
    if spec.kind == "counts":
        return {name: spec.hi if last else spec.lo for name in spec.values}
    raise AssertionError(spec.kind)


class TestEveryProcessValueIsClosed:
    """A pattern constrains form, a vocabulary constrains meaning, and a name
    a customer chose is a meaning problem. Every process-telemetry value is
    held to a closed rule, and nothing is accepted merely for being short."""

    def test_no_field_of_any_type_is_free_text(self):
        """The sweep, as a test: every accepted key on every type, from both
        producers, has one of the closed kinds, and every vocabulary is
        non-empty."""
        open_fields = [
            (event_type, key, spec.kind)
            for event_type, fields in schema.FIELDS_BY_TYPE.items()
            for key, spec in fields.items()
            if spec.kind not in CLOSED_KINDS
            or (spec.kind in {"enum", "counts"} and not spec.values)
            or (spec.kind in {"int", "counts"} and not spec.lo <= spec.hi)
        ]
        assert open_fields == []
        for event_type, keys in schema.KEYS_BY_TYPE.items():
            assert keys == set(schema.FIELDS_BY_TYPE[event_type]) | {"event_type"}

    @pytest.mark.parametrize("event_type,key", PROCESS_FIELDS)
    def test_a_valid_value_is_accepted(self, event_type, key):
        spec = schema.PROCESS_FIELDS_BY_TYPE[event_type][key]
        base = {"event_id": "a" * 32, "event_type": event_type}
        for last in (False, True):
            assert validate({**base, key: _valid(spec, last)})
        if spec.kind == "enum":
            for value in spec.values:
                assert validate({**base, key: value})

    @pytest.mark.parametrize("event_type,key", PROCESS_FIELDS)
    def test_a_customer_shaped_value_is_rejected(self, event_type, key):
        spec = schema.PROCESS_FIELDS_BY_TYPE[event_type][key]
        base = {"event_id": "a" * 32, "event_type": event_type}
        for value in CUSTOMER_SHAPED:
            if spec.kind == "class" and value in PASSES_AS_A_CLASS_NAME:
                assert validate({**base, key: value})     # known, and pinned
                continue
            with pytest.raises(Rejected) as exc:
                validate({**base, key: value})
            assert not value or value not in exc.value.reason

    @pytest.mark.parametrize("event_type,key", PROCESS_FIELDS)
    def test_the_wrong_type_is_rejected(self, event_type, key):
        spec = schema.PROCESS_FIELDS_BY_TYPE[event_type][key]
        base = {"event_id": "a" * 32, "event_type": event_type}
        wrong = [None, 1.5, ["csv"], {"csv": "x"}]
        wrong += [7, True] if spec.kind not in {"int", "bool"} else ["7"]
        wrong += [True] if spec.kind == "int" else []
        wrong += [1] if spec.kind == "bool" else []
        for value in wrong:
            with pytest.raises(Rejected):
                validate({**base, key: value})

    def test_a_file_name_cannot_travel_as_a_file_format(self):
        shape = {"event_id": "a" * 32, "event_type": "dataset"}
        for fmt in ["csv", "tsv", "xlsx", "parquet", "json", "other"]:
            assert validate({**shape, "file_format": fmt})
        for name in ["Q3 payroll.xlsx", "payroll.xlsx", "q3_layoffs_final", "CSV", "excel",
                     "jsonl", ".csv"]:
            with pytest.raises(Rejected):
                validate({**shape, "file_format": name})

    def test_an_unlisted_model_is_rejected_not_coerced(self):
        """A deployment name or a self-hosted path is chosen by the customer.
        The sender maps anything unlisted to "other"; if it does not, the
        batch is refused -- quietly storing "other" would hide that it tried."""
        assert validate(event(llm_model="other", llm_provider="azure"))
        assert validate(event(llm_model="openai/gpt-oss-120b", llm_provider="groq"))
        for model in ["acme-prod-gpt4-deployment", "/models/acme/finetune-v3", "gpt-4o",
                      "Other", "openai/gpt-oss-120b "]:
            with pytest.raises(Rejected) as exc:
                validate(event(llm_model=model))
            assert model not in exc.value.reason
        for provider in ["acme-gateway", "other", "Groq", "ollama"]:
            with pytest.raises(Rejected):
                validate(event(llm_provider=provider))

    def test_an_unknown_agent_is_rejected(self):
        for agent in ["coder", "validator", "data_scout", "train_models"]:
            assert validate(event(agent=agent))
        for agent in ["acme_payroll_agent", "Coder", "firefly", "other"]:
            with pytest.raises(Rejected):
                validate(event(agent=agent))

    @pytest.mark.parametrize("event_type,key", [
        ("install", "event_id"), ("install", "install_id"),
        ("agent.started", "event_id"), ("agent.started", "session_id"),
        ("agent.completed", "event_id"), ("agent.completed", "session_id"),
        ("dataset", "event_id"), ("dataset", "session_id"),
    ])
    def test_an_id_must_be_hex_or_a_uuid(self, event_type, key):
        """Hex digits and hyphens cannot spell a name. `machine-1` and
        `ravi_kumar_acme` both passed a length check."""
        base = {"event_id": "a" * 32, "event_type": event_type}
        for good in ["a" * 32, "0f8fad5b-d9cb-469f-a165-70867728950e", "A1" * 32]:
            assert validate({**base, key: good})
        for bad in ["machine-1", "ravi_kumar_acme_corp", "s" * 32, "a" * 8, "a" * 65,
                    "acme-finance-laptop-0001", 12345678]:
            with pytest.raises(Rejected):
                validate({**base, key: bad})

    #: What interpreters and build tools actually write. Not ours to constrain.
    REAL_VERSIONS = [
        "3.11", "3.11.9", "1.6.2", "0",
        "3.13.0a2", "3.13.0b1", "3.13.0rc1",          # pre-releases
        "1.6.2.post1", "1.6.2.dev3", "1.6.2rc1.post2.dev4",
        "1.6.2+3f9a1c0", "1.6.2.dev3+g1a2b3c4.d20261007",   # local versions
        "3.12.3+",                                    # a distribution's patched build
        "1.7.0-rc1",
    ]
    #: Still not free text.
    NOT_VERSIONS = [
        "Q3 payroll.xlsx", "payroll.xlsx", "/data/acme/churn.csv", "1.0/../etc",
        "https://1.2.3.4", "s3://acme/1.0", "what version is this", "3.11 payroll",
        "db01.internal.acme.example", "acme-internal-build", "v1.6.2", "latest",
        "1.6.2 (acme fork)", "1.6.2_final", "1.2.3.4.5", "1.6.2+" + "x" * 25,
        "1." * 20 + "1", "", "١.٢",
    ]

    @pytest.mark.parametrize("key", ["avaloka_version", "python_version"])
    def test_real_world_versions_are_accepted(self, key):
        """A batch must not be thrown away because the interpreter is a
        release candidate."""
        base = {"event_id": "a" * 32, "event_type": "install"}
        for version in self.REAL_VERSIONS:
            assert validate({**base, key: version}), version

    @pytest.mark.parametrize("key", ["avaloka_version", "python_version"])
    def test_a_version_is_still_a_bounded_pattern(self, key):
        """Starts with a digit; digits, ASCII letters, dots, + and - only; no
        space, slash or underscore; at most 40 characters."""
        base = {"event_id": "a" * 32, "event_type": "install"}
        for value in self.NOT_VERSIONS:
            with pytest.raises(Rejected):
                validate({**base, key: value})

    def test_the_widest_thing_a_version_admits_is_known(self):
        """Form, not meaning: a digit, a separator and up to 24 letters, digits
        and dots pass. Recorded here so it is a stated residual and not a
        discovery."""
        base = {"event_id": "a" * 32, "event_type": "install"}
        assert validate({**base, "python_version": "1+db01.internal.acme.corp"})
        assert schema._RELEASE_VERSION.match("9999.9999.9999.9999rc999.post9999.dev99999+" + "z" * 24) is None
        assert schema._RELEASE_VERSION.match("1+" + "z" * 24)
        assert not schema._RELEASE_VERSION.match("1+" + "z" * 25)

    def test_the_ui_version_keeps_the_producers_own_pattern(self):
        """Tied to the analytics schema by the contract test, so not widened."""
        page = {"event_id": "a" * 32, "event_type": "analytics.page.viewed"}
        assert validate({**page, "ui_version": "1.7.0-rc1"})
        with pytest.raises(Rejected):
            validate({**page, "ui_version": "3.13.0rc1"})
        with pytest.raises(Rejected):
            validate({**page, "schema_version": "1.1.0"})

    def test_an_unlisted_cloud_is_other_and_never_local(self):
        base = {"event_id": "a" * 32, "event_type": "install"}
        for cloud in ["local", "gcp", "aws", "azure", "other"]:
            assert validate({**base, "cloud": cloud})
        for cloud in ["oracle", "acme-private-cloud", "Other", "acme.example", ""]:
            with pytest.raises(Rejected):
                validate({**base, "cloud": cloud})

    def test_process_buckets_are_the_process_labels_only(self):
        """The two producers band sizes differently. Each type accepts its own
        producer's labels and not the other's."""
        shape = {"event_id": "a" * 32, "event_type": "dataset"}
        assert validate({**shape, "row_count_bucket": "0-1e3", "column_count_bucket": "200+"})
        for key, other_producers in (("row_count_bucket", "1-1e2"),
                                     ("column_count_bucket", "1000+")):
            with pytest.raises(Rejected):
                validate({**shape, key: other_producers})
            with pytest.raises(Rejected):
                validate({**shape, key: "1048576"})


# ---------------------------------------------------------------------- store

class TestStore:
    def test_records_are_stored(self, store):
        result = store.store([event(0), event(1)], received_at=time.time())
        assert result.stored == 2 and store.count() == 2

    def test_redelivery_is_counted_once(self, store):
        """At-least-once delivery means the same batch can arrive twice."""
        now = time.time()
        store.store([event(0), event(1)], received_at=now)
        again = store.store([event(0), event(1)], received_at=now)
        assert again.stored == 0 and again.duplicates == 2
        assert store.count() == 2

    def test_a_partial_redelivery_stores_only_the_new(self, store):
        now = time.time()
        store.store([event(0), event(1)], received_at=now)
        result = store.store([event(1), event(2)], received_at=now)
        assert result.stored == 1 and result.duplicates == 1

    def test_there_is_nowhere_to_put_a_sender_identity(self, store):
        """The schema cannot store what it has no column for."""
        columns = set(store.columns())
        for forbidden in ["ip", "ip_address", "remote_addr", "hostname", "user", "sender"]:
            assert forbidden not in columns
        assert columns == {"event_id", "event_type", "install_id", "received_at", "payload"}


# ------------------------------------------------------------------- endpoint

class TestEndpoint:
    def test_a_good_batch_is_accepted(self, client, store):
        response = post(client, [event(0), event(1)])
        assert response.status_code == 202
        assert response.json() == {"accepted": 2, "duplicates": 0}
        assert store.count() == 2

    def test_uncompressed_bodies_are_also_accepted(self, client):
        assert post(client, [event(0)], compress=False).status_code == 202

    def test_a_redelivered_batch_is_acknowledged_not_duplicated(self, client, store):
        post(client, [event(0), event(1)])
        response = post(client, [event(0), event(1)])
        assert response.status_code == 202
        assert response.json() == {"accepted": 0, "duplicates": 2}
        assert store.count() == 2

    def test_an_unpermitted_field_gets_400_not_500(self, client, store):
        """400 tells the client this is permanent, so it steps over the batch."""
        response = post(client, [event(0, customer_email="x@y.com")])
        assert response.status_code == 400
        assert "customer_email" in response.json()["reason"]
        assert store.count() == 0

    def test_malformed_json_is_rejected(self, client):
        response = client.post("/v1/events", content=b'{"not":"closed"' + NEWLINE.encode(),
                               headers={"Content-Encoding": "identity"})
        assert response.status_code == 400

    def test_a_body_that_is_not_gzip_is_rejected(self, client):
        response = client.post("/v1/events", content=b"plain text",
                               headers={"Content-Encoding": "gzip"})
        assert response.status_code == 400

    def test_an_oversized_body_is_rejected(self, client, monkeypatch):
        import telemetry_service.app as app_module
        monkeypatch.setattr(app_module, "MAX_BODY_BYTES", 100)
        assert post(client, [event(i) for i in range(50)]).status_code == 413

    def test_a_decompression_bomb_is_refused(self, client, monkeypatch):
        import telemetry_service.app as app_module
        monkeypatch.setattr(app_module, "MAX_DECOMPRESSED_BYTES", 1000)
        bomb = gzip.compress(b"A" * 10_000_000)
        response = client.post("/v1/events", content=bomb,
                               headers={"Content-Encoding": "gzip"})
        assert response.status_code == 400

    def test_no_documentation_is_exposed(self, client):
        """A public write-only endpoint has nothing to explore."""
        for path in ["/docs", "/redoc", "/openapi.json"]:
            assert client.get(path).status_code == 404

    def test_health_reports_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"


# ---------------------------------------------------------------- rate limits

class TestRateLimiting:
    def test_a_sender_over_the_limit_gets_429(self, store):
        client = TestClient(create_app(store=store, limiter=RateLimiter(limit=2)))
        assert post(client, [event(0)]).status_code == 202
        assert post(client, [event(1)]).status_code == 202
        assert post(client, [event(2)]).status_code == 429

    def test_429_is_transient_so_the_client_retries(self, store):
        """The client treats 429 as retryable and 400 as permanent. Getting
        these the wrong way round would either lose data or block forever."""
        client = TestClient(create_app(store=store, limiter=RateLimiter(limit=1)))
        post(client, [event(0)])
        assert post(client, [event(1)]).status_code == 429

    def test_the_address_is_never_kept_in_the_clear(self):
        limiter = RateLimiter(limit=5)
        limiter.allow("203.0.113.42", now=1000.0)
        stored = json.dumps(limiter._counters)
        assert "203.0.113.42" not in stored
        assert "203.0.113" not in stored

    def test_two_processes_hash_the_same_address_differently(self):
        """The salt is per process, so the hashes are useless if obtained."""
        a, b = RateLimiter(), RateLimiter()
        a.allow("203.0.113.42", now=1000.0)
        b.allow("203.0.113.42", now=1000.0)
        assert set(a._counters) != set(b._counters)

    def test_the_window_resets(self):
        limiter = RateLimiter(limit=1, window_seconds=60)
        assert limiter.allow("1.2.3.4", now=1000.0)
        assert not limiter.allow("1.2.3.4", now=1010.0)
        assert limiter.allow("1.2.3.4", now=1070.0)

    def test_expired_windows_are_swept(self):
        limiter = RateLimiter(limit=5, window_seconds=60)
        limiter.allow("1.2.3.4", now=1000.0)
        assert limiter.tracked == 1
        limiter.sweep(now=2000.0)
        assert limiter.tracked == 0


# ----------------------------------------------------------------- retention

class TestRetention:
    def test_events_past_the_period_are_deleted(self, store):
        now = time.time()
        store.store([event(0)], received_at=now - (401 * 86_400))
        store.store([event(1)], received_at=now)
        assert store.count() == 2
        assert purge(store, days=400, now=now) == 1
        assert store.count() == 1

    def test_nothing_recent_is_deleted(self, store):
        now = time.time()
        store.store([event(i) for i in range(3)], received_at=now)
        assert purge(store, days=400, now=now) == 0
        assert store.count() == 3

    def test_it_is_safe_to_run_repeatedly(self, store):
        now = time.time()
        store.store([event(0)], received_at=now - (500 * 86_400))
        assert purge(store, days=400, now=now) == 1
        assert purge(store, days=400, now=now) == 0

    def test_a_dry_run_count_matches_what_would_go(self, store):
        now = time.time()
        store.store([event(0)], received_at=now - (500 * 86_400))
        store.store([event(1)], received_at=now)
        cutoff = now - (400 * 86_400)
        assert store.count_older_than(cutoff) == 1
        assert store.count() == 2  # counting deleted nothing


class TestInstallAttribution:
    """Events must be attributable to the installation that produced them.

    Found by running the whole pipeline end to end: every report showed one
    installation and a twelfth of the events, because agent records carry no
    install_id of their own and nothing tied them to the batch's install
    record. Every unit test passed while this was true.
    """

    def test_batch_events_inherit_the_install_id(self, store):
        now = time.time()
        store.store([
            {"event_id": "a" * 32, "event_type": "install", "install_id": "1" * 32},
            event(1),
            event(2),
        ], received_at=now)
        assert {r.get("install_id") for r in store.records()} == {"1" * 32}

    def test_a_batch_without_an_install_record_is_not_misattributed(self, store):
        now = time.time()
        store.store([event(1), event(2)], received_at=now)
        assert all(r.get("install_id") is None for r in store.records())

    def test_two_installs_stay_separate(self, store):
        now = time.time()
        store.store([{"event_id": "a" * 32, "event_type": "install",
                      "install_id": "1" * 32}, event(1)], received_at=now)
        store.store([{"event_id": "b" * 32, "event_type": "install",
                      "install_id": "2" * 32}, event(2)], received_at=now)
        by_id = {r["event_id"]: r.get("install_id") for r in store.records()}
        assert by_id[event(1)["event_id"]] == "1" * 32
        assert by_id[event(2)["event_id"]] == "2" * 32


class TestRateLimitCannotBeEvaded:
    """X-Forwarded-For is a list the caller can start and each proxy appends to.

    Reading the FIRST entry meant an attacker rotated one header and got an
    unlimited number of buckets against a public, unauthenticated endpoint.
    Measured before the fix: twenty requests against a limit of three, none
    blocked. The entry counted from the right is used instead, because that one
    was written by our own infrastructure.
    """

    def _hammer(self, store, header_for, n=20, limit=3):
        client = TestClient(create_app(store=store, limiter=RateLimiter(limit=limit)))
        return [
            client.post("/v1/events", content=body([event(i)]),
                        headers={"Content-Encoding": "gzip", **header_for(i)}).status_code
            for i in range(n)
        ]

    def test_rotating_the_header_does_not_buy_more_requests(self, store):
        codes = self._hammer(store, lambda i: {"X-Forwarded-For": f"1.2.3.{i}, 203.0.113.9"})
        assert codes.count(429) > 0, "rate limit evaded by spoofing X-Forwarded-For"

    def test_prepending_several_entries_does_not_help_either(self, store):
        codes = self._hammer(
            store, lambda i: {"X-Forwarded-For": f"evil-{i}, evil2-{i}, 203.0.113.9"})
        assert codes.count(429) > 0

    def test_no_header_still_limits(self, store):
        assert self._hammer(store, lambda i: {}).count(429) > 0

    def test_a_short_header_is_not_trusted(self, store, monkeypatch):
        """Fewer entries than expected hops means our edge did not write it."""
        import telemetry_service.app as app_module

        monkeypatch.setattr(app_module, "TRUSTED_PROXY_HOPS", 2)
        codes = self._hammer(store, lambda i: {"X-Forwarded-For": f"10.0.0.{i}"})
        assert codes.count(429) > 0

    def test_genuinely_different_senders_are_not_over_blocked(self, store):
        codes = self._hammer(store, lambda i: {"X-Forwarded-For": f"9.9.9.9, 198.51.100.{i}"})
        assert codes.count(429) == 0, "distinct senders behind one proxy were blocked"

    def test_the_address_is_never_echoed_back(self, store):
        client = TestClient(create_app(store=store, limiter=RateLimiter(limit=100)))
        response = client.post("/v1/events", content=body([event(1)]),
                               headers={"Content-Encoding": "gzip",
                                        "X-Forwarded-For": "203.0.113.7"})
        assert "203.0.113" not in response.text


class TestHostileInput:
    """It is public and unauthenticated. It has to survive being attacked."""

    def _post(self, client, raw, encoding="gzip"):
        return client.post("/v1/events", content=raw,
                           headers={"Content-Encoding": encoding}).status_code

    def test_a_decompression_bomb_is_refused(self, client):
        assert self._post(client, gzip.compress(b"A" * 50_000_000)) in {400, 413}

    def test_null_bytes_are_rejected(self, client):
        payload = b'{"a":"' + bytes([0]) + b'"}' + NEWLINE.encode()
        assert self._post(client, payload, 'identity') == 400

    def test_invalid_utf8_is_rejected(self, client):
        payload = bytes([255, 254, 0]) + b'bad' + NEWLINE.encode()
        assert self._post(client, payload, 'identity') == 400

    def test_a_huge_single_field_is_rejected(self, client):
        record = json.dumps({"event_id": "a" * 32, "event_type": "install",
                             "edition": "z" * 200_000})
        assert self._post(client, (record + NEWLINE).encode(), "identity") == 400

    def test_gzip_declared_but_not_sent_is_rejected(self, client):
        assert self._post(client, b'{"a":1}' + NEWLINE.encode(), "gzip") == 400

    def test_the_server_survives_all_of_it(self, client):
        for raw, enc in [(b"", "identity"), (b"<html>", "identity"),
                         (gzip.compress(b"A" * 10_000_000), "gzip")]:
            self._post(client, raw, enc)
        assert client.get("/health").status_code == 200


class TestAuditRegressions:
    """Each of these pins a server-side defect the adversarial audit confirmed."""
    def test_the_record_cap_fires_before_any_json_is_parsed(self, client):
        """A small compressed request decompressed into millions of lines, and
        every one was parsed into a dict BEFORE the batch cap was consulted —
        a ~30x heap amplification an unauthenticated stranger could trigger.
        The cap now fires on the raw line count."""
        from telemetry_service.schema import MAX_RECORDS_PER_BATCH
        flood = ("{}" + NEWLINE) * (MAX_RECORDS_PER_BATCH + 5)
        response = client.post("/v1/events", content=gzip.compress(flood.encode()),
                               headers={"Content-Encoding": "gzip"})
        assert response.status_code == 400
        assert "records" in response.json()["reason"]
    def test_an_event_type_that_is_not_a_string_gets_400_not_500(self, client, store):
        """A list or an object is unhashable; looking it up in the set of
        known types raised, and the request died as a 500 -- which the client
        treats as transient and retries forever."""
        for bad in ([], {}, ["install"], {"a": 1}, 7, None, True):
            response = post(client, [{"event_id": "a" * 32, "event_type": bad}])
            assert response.status_code == 400, bad
        assert store.count() == 0

    def test_an_unknown_event_type_is_not_echoed_at_length(self):
        with pytest.raises(Rejected) as exc:
            validate(event(event_type="x" * 5000))
        assert len(exc.value.reason) < 200

    def test_an_integer_in_a_vocabulary_field_is_rejected(self):
        """outcome=1 sailed past a check that outcome='nonsense' failed: the
        enum branch was unreachable for non-strings."""
        with pytest.raises(Rejected):
            validate(event(outcome=1))
    def test_control_characters_in_values_are_rejected(self):
        bad = event()
        bad["agent"] = "validator" + chr(10) + "FAKE LOG LINE"
        with pytest.raises(Rejected):
            validate(bad)
    def test_the_rejection_reason_is_capped(self):
        """Unknown field names are attacker-chosen text that lands in our logs
        and response bodies. Two hundred of them must not arrive verbatim."""
        flooded = event()
        for i in range(200):
            flooded[f"attacker_field_{i:03d}_" + "x" * 100] = 1
        with pytest.raises(Rejected) as excinfo:
            validate(flooded)
        assert len(excinfo.value.reason) < 600
        assert "+195 more" in excinfo.value.reason
    def test_the_rate_limiter_sweeps_itself(self):
        """sweep() existed from the start but only a test ever called it, so a
        public endpoint grew one table entry per distinct sender for the life
        of the process."""
        limiter = RateLimiter(limit=5, window_seconds=60)
        for i in range(50):
            limiter.allow(f"10.0.0.{i}", now=1000.0)
        assert limiter.tracked == 50
        limiter.allow("fresh", now=2000.0)   # a later window triggers the sweep
        assert limiter.tracked == 1
    def test_health_reports_degraded_when_the_store_is_dead(self, store):
        client = TestClient(create_app(store=store, limiter=RateLimiter(limit=100)))
        store.close()
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"
