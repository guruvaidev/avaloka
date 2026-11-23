"""The domain vocabulary never reaches telemetry.

The vocabulary (``avaloka/domains.py``) reads the dataset's domain from its
column names. Everything it produces is derived from those names: the domain
id and label, the matched columns, the reason, the hint set and the questions.
That makes all of it customer schema for the standing rule (no customer data,
column name, file name, or anything derived from them in telemetry; a label
in a model prompt is allowed, scope ruled 2026-10-10).

These tests plant distinctive column names that the vocabulary reads as
healthcare, in a file with a distinctive name, and push them through both
profilers:

* the CLI: a real two-clone mission, so the Data Scout classifies every shard
  and records the domain in its local understanding record;
* the app: the ``profile_data`` graph node, traced, so its step carries the
  profile with the domain in it.

They then collect every telemetry payload those runs can produce
(``understanding_events``, the coordinator's egress file, ``swarm_events``,
``agent_events``, the ingest POST body and the ingest store) and check that no
planted name, no file name and nothing the vocabulary derived from them is in
any of it. The control half checks the local layer *does* carry the domain, so
a clean wire is a wire that dropped it, not one that never saw it; and
``test_the_scan_bites_on_a_routed_domain`` checks the scan goes red when the
domain is routed into a payload.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from avaloka.coordinator import MainAvaloka
from avaloka.domains import VOCABULARY
from avaloka.mission.context import MissionKind
from avaloka.swarm_limits import UNLIMITED
from avaloka.swarm_trace import SwarmTraceRecorder
from avaloka.understanding_trace import AGENT_RECORD, EGRESS_FILE, EGRESS_KEYS, LOG_FILE, read_log
from app.analytics import allowlist
from app.trace.recorder import drain, traced_node
from app.trace.telemetry import agent_events, swarm_events, understanding_events
from telemetry_service import schema as ingest
from telemetry_service.app import create_app
from telemetry_service.ratelimit import RateLimiter
from telemetry_service.store import Store

TOKEN = "qzv81"
# Each name hits one healthcare cue (patient, diagnos, admission, readmit,
# length of stay), so the vocabulary is confident without a model.
COLUMNS = [f"patient_{TOKEN}_ref", f"diagnosis_{TOKEN}_code", f"admission_{TOKEN}_type",
           f"readmitted_{TOKEN}", f"length_of_stay_{TOKEN}"]
FILE = f"{TOKEN}_cardiology_ward_export.csv"
DOMAIN = VOCABULARY["healthcare"]
# What the vocabulary derives from the names: the id, the label, and every
# line of the hint set and the questions it attaches.
DERIVED = [DOMAIN.id, DOMAIN.label, *DOMAIN.sanity_checks, *DOMAIN.questions,
           *DOMAIN.units, *DOMAIN.join_keys]
PLANTED = [TOKEN, *COLUMNS, FILE, *DERIVED]


def _scan(text: str, where: str) -> None:
    low = text.lower()
    for plant in PLANTED:
        assert plant.lower() not in low, f"{plant!r} reached {where}"


@pytest.fixture(scope="module")
def cli_mission(tmp_path_factory):
    base = tmp_path_factory.mktemp("domain_planted")
    rng = np.random.default_rng(11)
    n = 240
    frame = pd.DataFrame({
        COLUMNS[0]: rng.integers(1000, 9999, n),
        COLUMNS[1]: rng.choice(["I10", "E11", "J45"], n),
        COLUMNS[2]: rng.choice(["emergency", "elective"], n),
        COLUMNS[3]: rng.integers(0, 2, n),
        COLUMNS[4]: rng.integers(1, 15, n),
    })
    path = base / FILE
    frame.to_csv(path, index=False)
    recorder = SwarmTraceRecorder()
    MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(path), goal="Describe the data", output_dir=base / "fleet", workers=2,
        kind=MissionKind.ANALYZE, recorder=recorder)
    return {"mission": recorder.mission(), "out": base / "fleet"}


@pytest.fixture(scope="module")
def app_step():
    """The ``profile_data`` node, traced, on the planted schema. The model is
    stubbed (no key, no network); the names decide the domain anyway."""
    from app.agents import profiling_agent as pa

    mp = pytest.MonkeyPatch()
    mp.setattr(pa, "profiling_llm", object())
    mp.setattr(pa, "_call_json", lambda system, user, label: {"column_explanations": {},
                                                              "predictive_values": {}})
    mp.setattr(pa, "_PROFILE_PARALLEL", False)
    # The node runs outside a turn, so it carries out whatever an earlier test
    # left parked under "unscoped" (test_coding_graph_execution_wiring parks
    # coder steps there). Only this node's own steps are under test.
    drain("unscoped")
    rows = [{COLUMNS[0]: 1234, COLUMNS[1]: "I10", COLUMNS[2]: "emergency", COLUMNS[3]: 1, COLUMNS[4]: 3}]
    try:
        node = traced_node("profile_data", pa.profiling_agent_node)
        final = node({"sample_status": "full_sample", "full_sample_rows": rows,
                      "uploaded_csv_columns": list(COLUMNS),
                      "sample_statistics": {"column_statistics": {}, "data_quality": {}},
                      "data_source_location": f"/data/{FILE}", "agent_trace": []})
    finally:
        mp.undo()
    return final


def test_the_local_layer_does_carry_the_domain(cli_mission, app_step):
    """The control: the plants reached both profilers and the vocabulary read
    them, so their absence from the wire below means something."""
    scouts = [r for r in read_log(cli_mission["out"] / LOG_FILE)
              if r.get("type") == AGENT_RECORD and r.get("agent") == "data_scout"]
    assert len(scouts) == 2
    for scout in scouts:
        assert scout["decided"]["domain"] == "healthcare"
        assert set(COLUMNS) <= set(scout["data"]["column_names"])
    assert FILE in json.dumps(cli_mission["mission"], default=str)

    assert app_step["full_profiling_result"]["domain"]["category"] == "healthcare"
    assert app_step["full_profiling_result"]["domain"]["source"] == "vocabulary"
    step, = [s for s in app_step["agent_trace"] if s.get("node") == "profile_data"]
    assert "full_profiling_result" in step["written"]


def test_no_cli_telemetry_payload_carries_a_name_or_the_domain(cli_mission):
    mission = cli_mission["mission"]
    payloads = {
        "understanding_events": understanding_events(mission),
        "swarm_events": swarm_events(mission),
        "the coordinator's egress file": json.loads((cli_mission["out"] / EGRESS_FILE).read_text()),
    }
    assert payloads["understanding_events"] and payloads["swarm_events"]
    assert any(e.get("agent") == "data_scout" for e in payloads["understanding_events"])
    for where, payload in payloads.items():
        _scan(json.dumps(payload), where)
    ingest.validate_batch(payloads["understanding_events"])
    ingest.validate_batch(payloads["swarm_events"])


def test_no_app_telemetry_payload_carries_a_name_or_the_domain(app_step):
    events = agent_events(app_step["agent_trace"])
    assert [e["agent"] for e in events] == ["profile_data", "profile_data"]
    _scan(json.dumps(events), "agent_events")
    ingest.validate_batch(events)


def test_nothing_reaches_the_ingest_body_or_its_store(cli_mission, app_step, tmp_path):
    events = [*understanding_events(cli_mission["mission"]), *swarm_events(cli_mission["mission"]),
              *agent_events(app_step["agent_trace"])]
    body = "\n".join(json.dumps(e) for e in events) + "\n"
    _scan(body, "the POST body")
    store = Store(tmp_path / "telemetry.db")
    client = TestClient(create_app(store=store, limiter=RateLimiter(limit=1000)))
    response = client.post("/v1/events", content=body.encode(), headers={"content-type": "application/x-ndjson"})
    assert response.status_code == 202, response.text
    stored = store.records(limit=1000)
    assert len(stored) == len(events)
    _scan(json.dumps(stored, default=str), "the store")
    store.close()


def test_no_permit_list_has_a_field_a_domain_could_travel_in():
    """Every closed key set on the way out: the understanding projection, the
    ingest's process fields and the analytics export. None names a domain, a
    label, a column or a file."""
    banned = re.compile(r"domain|label|column_name|columns$|file_name|file_path|hint|sanity|question")
    keysets = {f"EGRESS_KEYS[{k}]": v for k, v in EGRESS_KEYS.items()}
    keysets.update({f"ingest[{k}]": set(v) for k, v in ingest.PROCESS_FIELDS_BY_TYPE.items()})
    keysets["analytics EXPORT_KEYS"] = allowlist.EXPORT_KEYS
    for where, keys in keysets.items():
        assert not [k for k in keys if banned.search(k)], where


def test_the_scan_bites_on_a_routed_domain(cli_mission, app_step):
    """Route what the vocabulary concluded into a payload and the scan must go
    red, whichever piece is routed. If it did not, a green run above would
    mean the scan was not looking."""
    events = understanding_events(cli_mission["mission"])
    scout = next(r for r in cli_mission["mission"]["records"]
                 if r.get("type") == AGENT_RECORD and r.get("agent") == "data_scout")
    leaky = [dict(e) for e in events]
    leaky[0]["error_class"] = scout["decided"]["domain"]
    with pytest.raises(AssertionError, match="healthcare"):
        _scan(json.dumps(leaky), "a leaky projection")
    domain = app_step["full_profiling_result"]["domain"]
    for piece in (domain["label"], domain["hints"]["sanity_checks"][0], domain["questions"][0],
                  domain["matched_cues"][0]["column"]):
        with pytest.raises(AssertionError):
            _scan(json.dumps({**events[0], "reason": piece}), "a leaky projection")
    # And the ingest refuses a domain field outright.
    with pytest.raises(ingest.Rejected):
        ingest.validate({**events[0], "domain": "healthcare"})
