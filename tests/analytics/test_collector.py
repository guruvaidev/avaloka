"""The collector end to end, over HTTP, against a real (in-memory) database.

Assertions about privacy are made against a dump of every table, not against
the API's response: what matters is what is on disk afterwards.
"""
import datetime as dt
import json

import pytest
from sqlalchemy import select

from app.analytics import store as store_mod
from app.analytics.schema import SCHEMA_VERSION
from tests.analytics.conftest import USER, batch, eid, ev

RAW_PROMPT = "email ravi.kumar@acme.example the churn in q3_layoffs_final.csv, token is hunter2, call 07700 900123"


def dump(store) -> str:
    """Everything in every table, as one string."""
    out = []
    with store.engine.connect() as conn:
        for table in (store_mod.events, store_mod.optout, store_mod.meta):
            if table is store_mod.meta:
                rows = conn.execute(select(table).where(table.c.key != "id_secret")).all()
            else:
                rows = conn.execute(select(table)).all()
            out.extend(json.dumps([str(v) for v in r]) for r in rows)
    return "\n".join(out)


def question(turn="7" * 16, prompt=RAW_PROMPT, seq=1):
    return ev("question.submitted", seq=seq, turn_id=turn, input_method="typed", prompt=prompt)


def post(client, body, headers=USER):
    return client.post("/analytics/events", json=body, headers=headers)


# -- capture ----------------------------------------------------------------

def test_stores_events_and_reports_counts(client, store):
    r = post(client, batch(ev("session.started", entry="login"), question(),
                           ev("question.completed", seq=2, turn_id="7" * 16, outcome="success",
                              duration_ms=1200, chart_type="bar")))
    assert r.status_code == 202
    assert r.json() == {"enabled": True, "accepted": 3, "duplicates": 0, "rejected": []}
    rows = store.all_rows()
    assert [row["event_type"] for row in rows] == ["session.started", "question.submitted", "question.completed"]
    assert json.loads(rows[2]["props"])["chart_type"] == "bar"


def test_prompt_is_redacted_before_it_reaches_storage(client, store):
    post(client, batch(question()))
    stored = dump(store)
    for fragment in ("ravi.kumar", "acme.example", "q3_layoffs_final", "hunter2", "900123"):
        assert fragment not in stored
    row = store.all_rows()[0]
    assert "churn" in row["prompt"] and "[email]" in row["prompt"]
    props = json.loads(row["props"])
    assert props["prompt_redactions"] >= 4 and props["prompt_chars_bucket"] == "81-200"


def test_no_customer_identifier_is_stored(client, store):
    post(client, batch(question(prompt="average revenue by region")))
    stored = dump(store)
    assert "user-7f3a" not in stored and "acme" not in stored
    actor = store.all_rows()[0]["actor_id"]
    assert len(actor) == 32 and actor != USER["X-Test-User"]


def test_actor_is_stable_per_user_and_distinct_between_users_and_installs(client, store):
    post(client, batch(ev("page.viewed", route="/analysis")))
    post(client, batch(ev("page.viewed", route="/datasets")))
    post(client, batch(ev("page.viewed", route="/reports")), headers={"X-Test-User": "someone-else"})
    actors = [r["actor_id"] for r in store.all_rows()]
    assert actors[0] == actors[1] != actors[2]

    from app.analytics.collector import derive_actor_id
    other_install = store_mod.Store("sqlite:///:memory:")
    assert derive_actor_id(other_install.identity_secret(), USER["X-Test-User"]) != actors[0]


def test_unauthenticated_is_refused_and_nothing_is_stored(client, store):
    assert post(client, batch(question()), headers={}).status_code == 401
    assert store.count() == 0


def test_redelivery_is_deduplicated(client, store):
    body = batch(question(), ev("page.viewed", seq=2, route="/analysis"))
    assert post(client, body).json()["accepted"] == 2
    again = post(client, body).json()
    assert (again["accepted"], again["duplicates"]) == (0, 2)
    assert store.count() == 2


def test_bad_records_are_rejected_individually_and_reported(client, store):
    r = post(client, batch(
        ev("page.viewed", route="/analysis"),
        ev("page.viewed", seq=1, route="/shared/8fa2c1d0"),                 # concrete path
        {**ev("page.viewed", seq=2, route="/datasets"), "file_name": "q3.csv"},
        ev("made.up", seq=3),
    )).json()
    assert r["accepted"] == 1 and [x["index"] for x in r["rejected"]] == [1, 2, 3]
    assert "q3.csv" not in dump(store) and "8fa2c1d0" not in dump(store)


def test_client_cannot_supply_server_fields(client, store):
    r = post(client, batch({**ev("page.viewed", route="/"), "actor_id": "f" * 32})).json()
    assert r["accepted"] == 0 and store.count() == 0
    bad = {**batch(ev("page.viewed", route="/")), "actor_id": "f" * 32}
    assert post(client, bad).status_code == 400


def test_unsupported_major_version_rejects_the_batch(client, store):
    body = {**batch(ev("page.viewed", route="/")), "schema_version": "2.0"}
    assert post(client, body).status_code == 400 and store.count() == 0


def test_oversized_batch_is_refused(client, store):
    r = client.post("/analytics/events", content=b"x" * (128 * 1024 + 1), headers=USER)
    assert r.status_code == 413 and store.count() == 0


def test_rate_limit(store):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.analytics.collector import build_router
    app = FastAPI()
    # A fixed clock, so both posts land in one limiter window whatever the wall
    # clock is doing; the injected-clock test below covers the window itself.
    app.include_router(build_router(lambda req: "u1", store=store, rate_limit_per_minute=3,
                                    clock=lambda: 30.0))
    c = TestClient(app)
    assert c.post("/analytics/events", json=batch(*[ev("page.viewed", seq=i, route="/") for i in range(3)])).status_code == 202
    assert c.post("/analytics/events", json=batch(ev("page.viewed", route="/"))).status_code == 429


def test_rate_limit_window_is_per_minute_with_an_injected_clock():
    from app.analytics.collector import _RateLimiter
    now = [59.0]
    limiter = _RateLimiter(3, clock=lambda: now[0])
    assert limiter.allow("a", 3) and not limiter.allow("a", 1)
    assert limiter.allow("b", 1)                      # another analyst is unaffected
    now[0] = 61.0
    assert limiter.allow("a", 3)                      # a new minute, a new allowance


def test_store_failure_is_a_503_and_leaks_no_detail(client, store, monkeypatch):
    def boom(rows):
        raise RuntimeError("could not insert row: prompt='secret question text'")
    monkeypatch.setattr(store, "insert_events", boom)
    r = post(client, batch(question()))
    assert r.status_code == 503 and "secret question" not in r.text


def test_retention_purge_removes_old_rows_only(store):
    old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=200)
    rows = [store_mod.to_row(ev("page.viewed", route="/"), schema_version="1.0", session_id="5" * 16,
                             ui_version=None, actor_id="a" * 32, received_at=when)
            for when in (old, None)]
    store.insert_events(rows)
    assert store.purge_older_than(180) == 1 and store.count() == 1


# -- opt-out: every switch, verified by what is in the database --------------

@pytest.mark.parametrize("env", [
    {"AVALOKA_ANALYTICS": "off"}, {"AVALOKA_ANALYTICS": "0"}, {"AVALOKA_ANALYTICS": "false"},
    {"AVALOKA_ANALYTICS": " Disabled "},
    {"AVALOKA_TELEMETRY": "off"},              # the umbrella switch from PR #329
])
def test_operator_opt_out_stores_nothing(client, store, monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    r = post(client, batch(ev("session.started", entry="direct"), question()))
    assert r.status_code == 200 and r.json()["enabled"] is False and r.json()["accepted"] == 0
    assert store.count() == 0 and dump(store) == ""
    assert client.get("/analytics/config", headers=USER).json()["enabled"] is False


def test_explicit_analytics_on_overrides_the_umbrella(client, store, monkeypatch):
    monkeypatch.setenv("AVALOKA_TELEMETRY", "off")
    monkeypatch.setenv("AVALOKA_ANALYTICS", "on")
    assert post(client, batch(ev("page.viewed", route="/"))).status_code == 202 and store.count() == 1


def test_unrecognised_switch_value_does_not_silently_disable_or_enable(client, store, monkeypatch):
    monkeypatch.setenv("AVALOKA_ANALYTICS", "maybe")
    monkeypatch.setenv("AVALOKA_TELEMETRY", "off")
    post(client, batch(ev("page.viewed", route="/")))
    assert store.count() == 0          # falls through to the umbrella, which says off


def test_prompt_text_switch_keeps_behaviour_and_drops_the_text(client, store, monkeypatch):
    monkeypatch.setenv("AVALOKA_ANALYTICS_PROMPT_TEXT", "off")
    assert post(client, batch(question(prompt="average revenue by region"))).json()["accepted"] == 1
    row = store.all_rows()[0]
    assert row["prompt"] is None and "average revenue" not in dump(store)
    assert json.loads(row["props"])["prompt_chars_bucket"] == "21-80"
    assert client.get("/analytics/config", headers=USER).json()["prompt_text"] is False


def test_analyst_opt_out_deletes_history_and_blocks_new_events(client, store):
    other = {"X-Test-User": "someone-else"}
    post(client, batch(question(), ev("page.viewed", seq=2, route="/analysis")))
    post(client, batch(ev("page.viewed", route="/datasets")), headers=other)
    assert store.count() == 3

    r = client.put("/analytics/opt-out", json={"opted_out": True}, headers=USER)
    assert r.json() == {"opted_out": True, "deleted_events": 2}
    assert store.count() == 1                       # the other analyst is untouched

    blocked = post(client, batch(question(turn="0" * 16)))
    assert blocked.status_code == 200 and blocked.json()["reason"] == "opted_out"
    assert store.count() == 1
    cfg = client.get("/analytics/config", headers=USER).json()
    assert cfg["enabled"] is False and cfg["opted_out"] is True
    assert post(client, batch(ev("page.viewed", route="/")), headers=other).status_code == 202


def test_analyst_can_opt_back_in(client, store):
    client.put("/analytics/opt-out", json={"opted_out": True}, headers=USER)
    client.put("/analytics/opt-out", json={"opted_out": False}, headers=USER)
    assert post(client, batch(ev("page.viewed", route="/"))).status_code == 202 and store.count() == 1


def test_opt_out_record_holds_the_pseudonym_not_the_user(client, store):
    client.put("/analytics/opt-out", json={"opted_out": True}, headers=USER)
    assert "user-7f3a" not in dump(store) and "acme" not in dump(store)


def test_opt_out_survives_capture_being_disabled(client, store, monkeypatch):
    monkeypatch.setenv("AVALOKA_ANALYTICS", "off")
    assert client.put("/analytics/opt-out", json={"opted_out": True}, headers=USER).status_code == 200
    monkeypatch.setenv("AVALOKA_ANALYTICS", "on")
    assert post(client, batch(ev("page.viewed", route="/"))).json()["reason"] == "opted_out"
    assert store.count() == 0


def test_opt_out_requires_auth_and_a_boolean(client):
    assert client.put("/analytics/opt-out", json={"opted_out": True}).status_code == 401
    assert client.put("/analytics/opt-out", json={"opted_out": "yes"}, headers=USER).status_code == 400


@pytest.mark.parametrize("header", [{"Sec-GPC": "1"}, {"DNT": "1"}])
def test_browser_privacy_signal_is_honoured(client, store, header):
    r = post(client, batch(question()), headers={**USER, **header})
    assert r.json()["enabled"] is False and store.count() == 0
    assert client.get("/analytics/config", headers={**USER, **header}).json()["reason"] == "browser_signal"


def test_config_tells_an_enabled_client_what_to_send(client):
    cfg = client.get("/analytics/config", headers=USER).json()
    assert cfg == {"schema_version": SCHEMA_VERSION, "prompt_text": True, "enabled": True, "reason": "ok", "opted_out": False}
    assert client.get("/analytics/config").json()["enabled"] is False
