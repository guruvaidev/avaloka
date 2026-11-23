"""The understanding records' egress projection carries no customer content.

The local layer is the customer's own fleet at work and is full of their
data: column names, the file name, the goal, the reasons the agents gave,
and (planted here through a firefly's metadata) a prompt fragment and a
model name. The egress layer must carry none of it. These tests run a real
two-clone mission on a frame whose column names, file name, goal and values
are planted, project it, hand the projection to the ingest service's own
validator and then to the ingest service itself over HTTP, and check that
nothing planted travels in the events, in the POST body or in the store.

The harness is checked against itself: a projection that leaked one planted
string must go red here (``test_the_harness_bites_when_a_field_leaks``), so a
green run is a run in which the scan found nothing, not one that scanned
nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from avaloka.coordinator import MainAvaloka
from avaloka.fireflies.reporter import Reporter
from avaloka.mission.context import MissionKind
from avaloka.swarm_limits import UNLIMITED
from avaloka.swarm_trace import SwarmTraceRecorder
from avaloka.understanding_trace import (AGENT_RECORD, EGRESS_FILE, LOG_FILE, SWARM_RECORD, egress_event,
                                         egress_events, read_log)
from app.trace.telemetry import understanding_events
from telemetry_service import schema as ingest
from telemetry_service.app import create_app
from telemetry_service.ratelimit import RateLimiter
from telemetry_service.store import Store

COLUMN = "patient_hiv_status"
FILE = "acme_layoffs_q3_2026.csv"
VALUE = "987654.321"
GOAL = "who in Zanzibar-Ward-7 has patient_hiv_status"
PROMPT = "You are a data scientist; the CEO asked about the Zanzibar layoffs"
MODEL = "acme-private-finetune-v7"
PLANTED = [COLUMN, FILE, VALUE, GOAL, PROMPT, MODEL, "Zanzibar", "salary_band_x91", "churned", "ward"]


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """A real mission with the five plants in it. The prompt fragment and the
    model name travel through the Reporter's unit metadata, which is where a
    model-backed firefly would put them."""
    base = tmp_path_factory.mktemp("planted")
    rng = np.random.default_rng(3)
    n = 300
    frame = pd.DataFrame({
        COLUMN: rng.integers(0, 2, n),
        "salary_band_x91": rng.normal(70, 20, n).round(3),
        "ward": rng.choice(["Zanzibar-Ward-7", "Ward-2"], n),
        "churned": rng.integers(0, 2, n),
    })
    frame.loc[0, "salary_band_x91"] = float(VALUE)
    path = base / FILE
    frame.to_csv(path, index=False)

    original = Reporter.perform

    def perform(self):
        output, minutes, extra = original(self)
        extra = {**extra, "prompt": PROMPT, "llm_provider": "openrouter", "llm_model": MODEL,
                 "tokens_in": 321, "tokens_out": 45}
        return output, minutes, extra

    Reporter.perform = perform
    try:
        recorder = SwarmTraceRecorder()
        MainAvaloka(swarm_limits=UNLIMITED).coordinate(
            str(path), goal=GOAL, output_dir=base / "fleet", workers=2, kind=MissionKind.TRAIN,
            target="churned", metric="f1", recorder=recorder)
    finally:
        Reporter.perform = original
    return {"mission": recorder.mission(), "out": base / "fleet"}


def _scan(text: str, where: str) -> None:
    for plant in PLANTED:
        assert plant not in text, f"{plant!r} reached {where}"


def test_the_local_layer_does_carry_the_plants(planted):
    """The control's other half: if the plants never made it into the local
    records, their absence from the wire would prove nothing."""
    local = json.dumps(list(read_log(planted["out"] / LOG_FILE)), default=str)
    for plant in (COLUMN, PROMPT, MODEL, "Zanzibar"):
        assert plant in local, plant
    reporter = next(r for r in read_log(planted["out"] / LOG_FILE) if r["agent"] == "reporter")
    assert reporter["decided"]["prompt"] == PROMPT and reporter["model"]["model_id"] == MODEL
    scout = next(r for r in read_log(planted["out"] / LOG_FILE) if r["agent"] == "data_scout")
    assert COLUMN in scout["data"]["column_names"]
    # The goal is in the trace (swarm.requested, clone.dispatched) and the file
    # name in its source; a clone reads its shard, so the understanding records
    # carry the shard's size, not the file's name. Cells enter no layer at all.
    trace = json.dumps(planted["mission"], default=str)
    assert GOAL in trace and FILE in trace and VALUE not in trace and VALUE not in local


def test_nothing_planted_reaches_the_projection(planted):
    mission = planted["mission"]
    events = understanding_events(mission)
    # 2 clones x 11 steps (the ConformalCalibrator runs on every train
    # mission since #527) + the fleet record.
    assert len(events) == 23 and events[-1]["event_type"] == SWARM_RECORD
    ingest.validate_batch(events)
    _scan(json.dumps(events), "the projection")
    # Nor anything that names the mission, a clone, a step or a path.
    wire = json.dumps(events)
    for secret in [mission["mission_id"], "ava-00", "ava-01", "Ava-0", "/s0", "fleet", "sub-avaloka", "shard"]:
        assert secret not in wire, secret
    # The model name became "other"; the token counts travelled.
    reporter = [e for e in events if e.get("agent") == "reporter"]
    assert all(e["llm_model"] == "other" and e["llm_provider"] == "openrouter" for e in reporter)
    assert all(e["tokens_in"] == 321 and e["tokens_out"] == 45 for e in reporter)
    # Every key is on the closed list; the list is the ingest's.
    for event in events:
        assert set(event) <= set(ingest.PROCESS_FIELDS_BY_TYPE[event["event_type"]]) | {"event_type"}


def test_the_egress_file_the_coordinator_wrote_is_the_same_projection(planted):
    egress = json.loads((planted["out"] / EGRESS_FILE).read_text())
    ingest.validate_batch(egress["events"])
    _scan(json.dumps(egress), "the egress file")
    assert [e["event_type"] for e in egress["events"]] == [e["event_type"] for e in understanding_events(planted["mission"])]


def test_nothing_planted_reaches_the_telemetry_post_body_or_the_store(planted, tmp_path):
    """The whole path a sender would take: the batch body as JSON Lines, the
    ingest's HTTP endpoint, the store it writes."""
    install = {"event_id": "1" * 32, "event_type": "install", "install_id": "a" * 32, "avaloka_version": "2.0.0",
               "edition": "oss", "deployment": "laptop", "cloud": "local", "os": "linux", "arch": "arm64",
               "python_version": "3.11", "cpu_count": 8}
    events = understanding_events(planted["mission"])
    body = "\n".join(json.dumps(r) for r in [install, *events]) + "\n"
    _scan(body, "the POST body")
    store = Store(tmp_path / "telemetry.db")
    client = TestClient(create_app(store=store, limiter=RateLimiter(limit=1000)))
    response = client.post("/v1/events", content=body.encode(), headers={"content-type": "application/x-ndjson"})
    assert response.status_code == 202, response.text
    stored = store.records(limit=1000)
    assert len(stored) == len(events) + 1
    _scan(json.dumps(stored, default=str), "the store")
    store.close()


def test_the_ingest_refuses_a_projection_that_started_sending_content(planted):
    """The server-side half of the control: a client that leaked the column
    names, the goal or the conclusion text would have its batch rejected."""
    event = understanding_events(planted["mission"])[0]
    ingest.validate(event)
    for key, value in [("columns", [COLUMN]), ("goal", GOAL), ("prompt", PROMPT), ("output", "Profiled"),
                       ("file_name", FILE), ("mission_id", "abc"), ("row_count", 300)]:
        with pytest.raises(ingest.Rejected, match="refused"):
            ingest.validate({**event, key: value})
    with pytest.raises(ingest.Rejected, match="unpermitted"):
        ingest.validate({**event, "concluded": "Profiled 4 columns"})
    with pytest.raises(ingest.Rejected, match="refused"):
        ingest.validate({**event, "column_names": [COLUMN]})
    with pytest.raises(ingest.Rejected):  # a histogram key that is not vocabulary is not a count
        ingest.validate({**event, "column_type_counts": {COLUMN: 1}})


def test_the_harness_bites_when_a_field_leaks(planted, monkeypatch):
    """A deliberately leaky projection must fail the scan above. If it did
    not, a green run would mean the scan was not looking."""
    mission = planted["mission"]
    leaky = egress_events(mission["records"])
    scout = next(r for r in mission["records"] if r.get("type") == AGENT_RECORD and r.get("agent") == "data_scout")
    leaky[0]["column_type_counts"] = {name: 1 for name in scout["data"]["column_names"]}  # a name as a key
    with pytest.raises(AssertionError, match="patient_hiv_status"):
        _scan(json.dumps(leaky), "a leaky projection")
    with pytest.raises(ingest.Rejected):
        ingest.validate_batch(leaky)
    leaky = egress_events(mission["records"])
    leaky[-1]["concluded"] = next(r for r in mission["records"] if r.get("type") == AGENT_RECORD)["concluded"]
    with pytest.raises(ingest.Rejected, match="unpermitted"):
        ingest.validate_batch(leaky)
    # And a one-record leak of the goal straight from the local record:
    local = next(r for r in mission["records"] if r.get("type") == AGENT_RECORD and r.get("agent") == "reporter")
    assert PROMPT not in json.dumps(egress_event(local)) and GOAL not in json.dumps(egress_event(local))
    with pytest.raises(AssertionError, match="Zanzibar"):
        _scan(json.dumps({**egress_event(local), "reason": local["decided"]["prompt"]}), "a leaky projection")


def test_a_failed_step_projects_its_class_never_its_message():
    rec = SwarmTraceRecorder()
    rec.record(AGENT_RECORD, agent="data_scout", role="understanding", clone_id="ava-00", clone_index=0,
               step_id="ava-00/s01", step_no=1, outcome="failed",
               error_class="KeyError", concluded=None,
               decided={"error": f"'{COLUMN}' not in {FILE}"}, data=None, sampling=None, operations=None,
               verification=None, model={"provider": "openrouter", "model_id": MODEL, "calls": 1,
                                         "tokens_in": 1, "tokens_out": 0}, timing={"duration_ms": 5})
    events = egress_events(rec.records)
    ingest.validate_batch(events)
    event, = events
    assert event["outcome"] == "failed" and event["error_class"] == "KeyError"
    assert event["llm_model"] == "other" and event["llm_provider"] == "openrouter"
    _scan(json.dumps(events), "the projection of a failed step")
