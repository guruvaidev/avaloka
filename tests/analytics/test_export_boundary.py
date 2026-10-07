"""The export boundary: the single point where anything leaves the cluster.

Everything here drives the real path -- HTTP in, database, ``export_pending``
out -- and asserts on the bytes handed to the network function.

Two groups. The first proves what ships: NO question text is exported, whatever
is configured. The second keeps the disabled question-text path proven on
constructed data, by flipping ``export.EXPORT_QUESTION_TEXT`` in the test only;
those tests take the ``text_path`` fixture and describe a capability, not
production behaviour.
"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.analytics import allowlist, export, schema
from app.analytics.collector import build_router
from app.analytics.store import PROMOTED, events as events_table
from tests.analytics.conftest import USER, batch, ev
from tests.analytics.test_names import NAMES, PLANTED

ENDPOINT = "https://ingest.example.test/v1/events"


class Wire:
    def __init__(self):
        self.records = []

    def __call__(self, endpoint, records, timeout=15.0):
        self.records.extend(records)
        return 200

    @property
    def text(self):
        return json.dumps(self.records).casefold()


def make_client(store, provider):
    app = FastAPI()
    app.include_router(build_router(lambda r: r.headers.get("X-Test-User"), store=store,
                                    names_provider=provider))
    return TestClient(app)


@pytest.fixture
def export_on(monkeypatch):
    monkeypatch.setenv("AVALOKA_ANALYTICS_EXPORT", "on")
    monkeypatch.setenv("AVALOKA_ANALYTICS_EXPORT_ENDPOINT", ENDPOINT)


@pytest.fixture
def text_path(monkeypatch):
    """Open the disabled question-text path. Tests only; nothing in the product does this."""
    monkeypatch.setattr(export, "EXPORT_QUESTION_TEXT", True)


def ask(client, prompt, turn="7" * 16):
    r = client.post("/analytics/events", headers=USER, json=batch(
        ev("question.submitted", turn_id=turn, input_method="typed", prompt=prompt)))
    assert r.status_code == 202 and r.json()["accepted"] == 1


def exported(store):
    wire = Wire()
    export.export_pending(store, post=wire)
    return wire


@pytest.mark.parametrize("text,gone", PLANTED, ids=[p[0][:36] for p in PLANTED])
def test_planted_names_never_reach_the_wire_but_stay_in_the_local_store(store, export_on, text_path, text, gone):
    ask(make_client(store, lambda request, user: NAMES), text)
    wire = exported(store)
    # Masked as a name, or already as a quoted literal by the pattern pass.
    assert len(wire.records) == 1 and "[" in wire.records[0]["prompt_stripped"]
    for fragment in gone:
        assert fragment.casefold() not in wire.text
    # Tier one: the deployment's own copy keeps the question as typed.
    assert store.all_rows()[0]["prompt"] is not None


def test_the_local_store_keeps_names_and_the_wire_does_not(store, export_on, text_path):
    ask(make_client(store, lambda request, user: NAMES), "average patient_hiv_status by region")
    assert store.all_rows()[0]["prompt"] == "average patient_hiv_status by region"
    record = exported(store).records[0]
    assert record["prompt_stripped"] == "average [name] by [name]" and record["prompt_names_exact"] == 1 and record["prompt_names_common_word"] == 1
    assert "prompt" not in record and "prompt_export" not in record


@pytest.mark.parametrize("provider", [
    None,                                              # no provider wired
    lambda request, user: None,                        # schema unavailable
    lambda request, user: [],                          # analyst has no registered dataset
    lambda request, user: set(),
    lambda request, user: ["", "___"],                 # names that normalise to nothing
    lambda request, user: (_ for _ in ()).throw(RuntimeError("redis down")),
], ids=["no-provider", "unavailable", "no-datasets", "empty-set", "unusable-names", "provider-raises"])
def test_fail_closed_no_vocabulary_no_question_text(store, export_on, text_path, provider):
    ask(make_client(store, provider), "average patient_hiv_status by clinic")
    wire = exported(store)
    assert len(wire.records) == 1                       # the behavioural event still goes
    assert "prompt_stripped" not in wire.records[0]
    assert "hiv" not in wire.text and "clinic" not in wire.text and "average" not in wire.text


def test_async_provider_is_awaited(store, export_on, text_path):
    async def provider(request, user):
        return NAMES
    ask(make_client(store, provider), "average patient_hiv_status")
    assert exported(store).records[0]["prompt_stripped"] == "average [name]"


def test_provider_is_given_the_real_user_and_only_that_users_names_apply(store, export_on, text_path):
    seen = []
    def provider(request, user):
        seen.append(user)
        return NAMES
    ask(make_client(store, provider), "status")
    assert seen == [USER["X-Test-User"]]


def test_questions_captured_while_export_was_off_are_never_exported(store, monkeypatch, text_path):
    calls = []
    client = make_client(store, lambda request, user: calls.append(user) or NAMES)
    ask(client, "average patient_hiv_status by clinic")
    assert calls == []                                  # no vocabulary lookup while export is off
    monkeypatch.setenv("AVALOKA_ANALYTICS_EXPORT", "on")
    monkeypatch.setenv("AVALOKA_ANALYTICS_EXPORT_ENDPOINT", ENDPOINT)
    wire = exported(store)
    assert len(wire.records) == 1 and "prompt_stripped" not in wire.records[0] and "hiv" not in wire.text


def test_pattern_redaction_still_applies_on_the_wire(store, export_on, text_path):
    ask(make_client(store, lambda request, user: NAMES),
        "mail ravi@acme.example the region totals from q3.csv, token is hunter2")
    text = exported(store).text
    for fragment in ("ravi", "acme", "q3.csv", "hunter2", "region"):
        assert fragment not in text


# -- what ships: no question text, whatever is configured ------------------------

from tests.analytics.test_redaction import KEPT

ALL_QUESTIONS = [p[0] for p in PLANTED] + KEPT


def _content_words(question):
    return {w for w in __import__("re").findall(r"[a-z_]{4,}", question.casefold())}


def test_question_text_export_is_off_in_code():
    assert export.EXPORT_QUESTION_TEXT is False


@pytest.mark.parametrize("question", ALL_QUESTIONS, ids=[q[:36] for q in ALL_QUESTIONS])
def test_no_question_text_reaches_the_wire(store, export_on, question):
    ask(make_client(store, lambda request, user: NAMES), question)
    wire = exported(store)
    assert len(wire.records) == 1                              # the behavioural event still goes
    record = wire.records[0]
    assert {k for k in record if "prompt" in k} == {"prompt_chars_bucket"}
    assert record["prompt_chars_bucket"] in schema.PROMPT_CHAR_BUCKETS
    values = " ".join(str(v) for k, v in record.items() if k != "event_type").casefold()
    for word in _content_words(question):
        assert word not in values, f"{word!r} from the question is on the wire"
    assert store.all_rows()[0]["prompt"] is not None           # and the local copy is intact
    assert store.all_rows()[0]["prompt_export"] is None        # no exportable form is even prepared


def test_the_name_vocabulary_is_never_looked_up_in_production(store, export_on):
    calls = []
    ask(make_client(store, lambda request, user: calls.append(user) or NAMES), "average patient_hiv_status")
    assert calls == []


@pytest.mark.parametrize("env", [
    {"AVALOKA_ANALYTICS_PROMPT_TEXT": "on"},
    {"AVALOKA_ANALYTICS_EXPORT_PROMPT_TEXT": "on"},
    {"AVALOKA_ANALYTICS_EXPORT_QUESTION_TEXT": "on"},
    {"EXPORT_QUESTION_TEXT": "1"},
    {"AVALOKA_ANALYTICS": "on", "AVALOKA_TELEMETRY": "on", "AVALOKA_ANALYTICS_PROMPT_TEXT": "on"},
])
def test_no_setting_turns_question_text_export_on(store, export_on, monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ask(make_client(store, lambda request, user: NAMES), "average revenue by month")
    wire = exported(store)
    assert "revenue" not in wire.text and "prompt_stripped" not in wire.records[0]


def test_a_stored_exportable_form_is_ignored_while_the_path_is_off():
    # Even a row that somehow has prompt_export populated exports no text.
    row = _full_row("question.submitted")
    record = export.to_wire(row, "i" * 32)
    assert "prompt_stripped" not in record and "stripped question" not in json.dumps(record)


def test_the_length_band_is_coarse():
    assert len(schema.PROMPT_CHAR_BUCKETS) == 6
    assert {schema.prompt_chars_bucket(n) for n in range(0, 2000)} == set(schema.PROMPT_CHAR_BUCKETS)


# -- the two representations --------------------------------------------------

def test_export_permit_list_excludes_every_sensitive_stored_field():
    assert schema.SENSITIVE_FIELDS == {"prompt"}
    assert not (allowlist.EXPORT_KEYS & allowlist.SENSITIVE_STORED)
    assert "received_at" not in allowlist.EXPORT_KEYS
    # The only prompt-derived key in the exported representation is the length band.
    assert {k for k in allowlist.EXPORT_KEYS if "prompt" in k} == {"prompt_chars_bucket"}
    assert not (allowlist.EXPORT_KEYS & allowlist.EXPORT_TEXT_KEYS)


def test_published_schema_states_both_representations():
    reps = schema.describe()["representations"]
    assert reps["stored"]["sensitive"] == ["prompt"]
    assert "prompt" in reps["stored"]["fields"] and "prompt" not in reps["exported"]["fields"]
    assert "prompt_stripped" not in reps["exported"]["fields"]
    assert reps["exported"]["question_text"] == "never exported, in any form"
    assert schema.describe()["events"]["question.submitted"]["optional"]["prompt"]["exported"] is False


def _full_row(event_type):
    """A stored row with every column and every declared field populated with a marker."""
    required, optional = schema.EVENTS[event_type]
    props = {name: f"MARK-{name}" for name in {**required, **optional, **schema.COMMON_OPTIONAL}}
    props.update({name: f"MARK-{name}" for name in schema.NEVER_ACCEPTED})
    props.update({"prompt": "MARK-prompt-in-props", "prompt_export": "MARK-export-in-props",
                  "prompt_chars_bucket": "1-20", "prompt_redactions": 0})
    row = {c.name: f"MARK-{c.name}" for c in events_table.columns}
    row.update(event_type=event_type, props=json.dumps(props), prompt="SENSITIVE stored question",
               prompt_export="stripped question", event_id="e" * 16, actor_id="a" * 32,
               session_id="5" * 16, schema_version="1.0")
    return row


@pytest.mark.parametrize("event_type", sorted(schema.EVENT_TYPES))
def test_exported_projection_never_carries_a_sensitive_or_undeclared_field(event_type):
    record = export.to_wire(_full_row(event_type), "i" * 32)
    assert set(record) <= allowlist.EXPORT_KEYS
    assert not (set(record) & allowlist.SENSITIVE_STORED)
    assert not (set(record) & schema.NEVER_ACCEPTED)
    dumped = json.dumps(record)
    assert "SENSITIVE stored question" not in dumped and "MARK-prompt-in-props" not in dumped
    assert "MARK-export-in-props" not in dumped and "MARK-received_at" not in dumped
    assert "stripped question" not in dumped and not (set(record) & allowlist.EXPORT_TEXT_KEYS)
    required, optional = schema.EVENTS[event_type]
    declared = set(required) | set(optional) | set(schema.COMMON_REQUIRED) | set(schema.COMMON_OPTIONAL)
    server_side = {"actor_id", "schema_version", "session_id", "ui_version", "install_id",
                   "prompt_chars_bucket"}
    assert set(record) <= declared | server_side
    assert "MARK-" not in dumped        # nothing outside its vocabulary survives the projection


def test_export_never_reads_the_sensitive_column():
    class Row(dict):
        def _deny(self, key):
            if key in allowlist.SENSITIVE_STORED:
                raise AssertionError(f"export read the sensitive column {key!r}")
        def __getitem__(self, key):
            self._deny(key)
            return super().__getitem__(key)
        def get(self, key, default=None):
            self._deny(key)
            return super().get(key, default)
    export.to_wire(Row(_full_row("question.submitted")), "i" * 32)


def test_sensitive_columns_are_promoted_columns_not_hidden_in_props():
    # If a sensitive field lived in the props blob, the row-level guard above
    # could not see it being read.
    assert allowlist.SENSITIVE_STORED <= set(PROMOTED)


def test_null_prompt_export_means_no_text_even_if_prompt_is_present():
    row = _full_row("question.submitted")
    row["prompt_export"] = None
    assert "prompt_stripped" not in export.to_wire(row, "i" * 32)


# -- identity on the wire ------------------------------------------------------
# The stored actor_id is stable (a persistent identifier, local only). What is
# exported must not be: with install_id beside it, it would be a durable global
# per-person key.

import datetime as dt

from app.analytics.collector import derive_actor_id
from app.analytics.store import Store

T0 = dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)


def test_exported_actor_is_not_the_stored_actor_and_the_secret_never_leaves(store, export_on):
    ask(make_client(store, lambda request, user: NAMES), "status")
    stored = store.all_rows()[0]["actor_id"]
    wire = exported(store)
    assert wire.records[0]["actor_id"] != stored and len(wire.records[0]["actor_id"]) == 32
    assert stored not in wire.text
    assert store.identity_secret().hex() not in wire.text
    assert USER["X-Test-User"].casefold() not in wire.text


def test_exported_actor_is_stable_within_an_epoch_and_rotates_across_epochs(store):
    secret, actor = store.identity_secret(), "a" * 32
    same = export.export_actor_id(secret, actor, T0 + dt.timedelta(hours=1))
    assert export.export_actor_id(secret, actor, T0) == same
    seen = {export.export_actor_id(secret, actor, T0 + dt.timedelta(days=export.EXPORT_ACTOR_EPOCH_DAYS * k))
            for k in range(6)}
    assert len(seen) == 6
    assert export.EXPORT_ACTOR_EPOCH_DAYS <= 28


def test_same_human_in_two_deployments_gets_unrelated_ids_stored_and_exported(store):
    other = Store("sqlite:///:memory:")
    assert store.identity_secret() != other.identity_secret()
    a = derive_actor_id(store.identity_secret(), "same-user")
    b = derive_actor_id(other.identity_secret(), "same-user")
    assert a != b
    assert export.export_actor_id(store.identity_secret(), a, T0) != export.export_actor_id(other.identity_secret(), b, T0)


def test_knowing_the_user_id_and_everything_exported_does_not_yield_the_actor(store, export_on):
    ask(make_client(store, lambda request, user: NAMES), "status")
    record = exported(store).records[0]
    user = USER["X-Test-User"]
    # Every key an outsider holds: the install id and the other ids on the wire.
    for guess in (b"", record["install_id"].encode(), record["session_id"].encode(), user.encode()):
        assert derive_actor_id(guess, user) != record["actor_id"]
        assert export.export_actor_id(guess, derive_actor_id(guess, user), T0) != record["actor_id"]


@pytest.mark.parametrize("secret,actor,when", [(b"", "a" * 32, T0), (b"k", None, T0), (b"k", "a" * 32, None),
                                               (b"k", "a" * 32, "2026-10-01")])
def test_no_secret_or_no_timestamp_means_no_identity_exported(secret, actor, when):
    assert export.export_actor_id(secret, actor, when) is None
    row = _full_row("page.viewed")
    row.update(actor_id=actor, received_at=when)
    assert "actor_id" not in export.to_wire(row, "i" * 32, secret)


def test_identity_secret_is_minted_per_install_and_not_configurable():
    from app.analytics import config
    assert not hasattr(config, "ENV_ID_SECRET")
    assert len(Store("sqlite:///:memory:").identity_secret()) == 32
