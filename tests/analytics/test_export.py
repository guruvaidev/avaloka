"""The outbound adapter. The property that matters on develop-1.6: by default it sends nothing."""
import json

import pytest

from app.analytics import config, export
from app.analytics.store import to_row
from tests.analytics.conftest import ev

ENDPOINT = "https://ingest.example.test/v1/events"


class Recorder:
    def __init__(self, status=200):
        self.status, self.calls = status, []

    def __call__(self, endpoint, records, timeout=15.0):
        self.calls.append((endpoint, records))
        return self.status


def seed(store, n=3, actor="a" * 32, prompt=None):
    rows = []
    for i in range(n):
        event = ev("question.submitted", seq=i, turn_id="7" * 16, input_method="typed")
        if prompt:
            event["prompt"] = prompt
            event["prompt_export"] = prompt      # as the collector would have prepared it
        rows.append(to_row(event, schema_version="1.0", session_id="5" * 16, ui_version="1.6.0", actor_id=actor))
    store.insert_events(rows)


def enable(monkeypatch, endpoint=ENDPOINT):
    monkeypatch.setenv("AVALOKA_ANALYTICS_EXPORT", "on")
    monkeypatch.setenv("AVALOKA_ANALYTICS_EXPORT_ENDPOINT", endpoint)


def test_nothing_is_sent_by_default(store):
    seed(store)
    post = Recorder()
    assert export.export_pending(store, post=post)["exported"] == 0 and post.calls == []


@pytest.mark.parametrize("env", [
    {"AVALOKA_ANALYTICS_EXPORT": "on"},                                             # no endpoint
    {"AVALOKA_ANALYTICS_EXPORT_ENDPOINT": ENDPOINT},                                # no switch
    {"AVALOKA_ANALYTICS_EXPORT": "on", "AVALOKA_ANALYTICS_EXPORT_ENDPOINT": "http://plain.example/x"},
    {"AVALOKA_ANALYTICS_EXPORT": "on", "AVALOKA_ANALYTICS_EXPORT_ENDPOINT": ENDPOINT, "AVALOKA_ANALYTICS": "off"},
    {"AVALOKA_ANALYTICS_EXPORT": "on", "AVALOKA_ANALYTICS_EXPORT_ENDPOINT": ENDPOINT, "AVALOKA_TELEMETRY": "off"},
])
def test_export_needs_switch_and_https_endpoint_and_capture_on(store, monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    seed(store)
    post = Recorder()
    assert config.export_endpoint() is None
    export.export_pending(store, post=post)
    assert post.calls == []


def test_there_is_no_default_destination_in_the_code():
    import inspect
    source = inspect.getsource(export) + inspect.getsource(config)
    assert "https://" not in source.replace('startswith("https://")', "")


def test_enabled_export_sends_prefixed_records_and_advances_the_cursor(store, monkeypatch):
    enable(monkeypatch)
    seed(store, prompt="average revenue by region")
    post = Recorder()
    assert export.export_pending(store, post=post)["exported"] == 3
    endpoint, records = post.calls[0]
    assert endpoint == ENDPOINT
    assert {r["event_type"] for r in records} == {"analytics.question.submitted"}
    assert len(records[0]["install_id"]) == 32
    assert not [k for k in records[0] if k.startswith("prompt")] and "average revenue" not in json.dumps(records)
    assert export.export_pending(store, post=post)["reason"] == "nothing pending" and len(post.calls) == 1


def test_wire_projection_is_a_permit_list_and_rescrubs(store):
    seed(store, n=1)
    row = store.all_rows()[0]
    # A row tampered with after storage, or written by a buggy older collector.
    row["props"] = json.dumps({"input_method": "typed", "file_name": "q3_layoffs.csv", "user_id": "u-1"})
    row["prompt_export"] = "mail ravi@acme.example"
    record = export.to_wire(row, "i" * 32)
    assert "file_name" not in record and "user_id" not in record
    assert "acme" not in json.dumps(record) and record["input_method"] == "typed"
    assert "id" not in record and "received_at" not in record


def test_prompt_text_switch_applies_on_export_too(store, monkeypatch):
    seed(store, n=1, prompt="average revenue by region")
    monkeypatch.setenv("AVALOKA_ANALYTICS_PROMPT_TEXT", "off")
    assert "prompt_stripped" not in export.to_wire(store.all_rows()[0], "i" * 32)


def test_opted_out_analyst_is_never_exported(store, monkeypatch):
    enable(monkeypatch)
    seed(store, n=2, actor="a" * 32)
    seed(store, n=1, actor="b" * 32)
    # Opt-out recorded without the delete, to prove export checks for itself.
    from app.analytics import store as store_mod
    import datetime as dt
    with store.engine.begin() as conn:
        conn.execute(store_mod.optout.insert().values(actor_id="a" * 32, at=dt.datetime.now(dt.timezone.utc)))
    post = Recorder()
    export.export_pending(store, post=post)
    assert len(post.calls[0][1]) == 1
    assert post.calls[0][1][0]["actor_id"] == export.export_actor_id(
        store.identity_secret(), "b" * 32, store.all_rows()[-1]["received_at"])


@pytest.mark.parametrize("status,advances", [(500, False), (429, False), (400, True), (422, True)])
def test_cursor_on_failure(store, monkeypatch, status, advances):
    enable(monkeypatch)
    seed(store)
    export.export_pending(store, post=Recorder(status))
    retry = Recorder()
    export.export_pending(store, post=retry)
    assert (retry.calls == []) is advances


def test_export_never_raises(store, monkeypatch):
    enable(monkeypatch)
    seed(store)
    def boom(*a, **k):
        raise OSError("network down")
    assert export.export_pending(store, post=boom) == {"exported": 0, "reason": "OSError"}
