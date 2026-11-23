"""The telemetry projection of a swarm mission carries no customer content.

The swarm trace is the user's own fleet at work and is full of their data:
the goal, the file, every column, every clone's output. The projection must
not be. These tests run a real two-clone mission on a frame whose column
names, file name, goal and values are planted, project it with
``swarm_events``, hand the result to the ingest service's own validator, and
check that nothing planted travels.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from avaloka.coordinator import MainAvaloka
from avaloka.mission.context import MissionKind
from avaloka.swarm_limits import UNLIMITED
from avaloka.swarm_trace import SwarmTraceRecorder
from app.trace.telemetry import KNOWN_FIREFLIES, swarm_events
from telemetry_service import schema as ingest

#: What the trace is allowed to carry (the user's own fleet at work) and the
#: projection is not: column names, a file name, a category value, the goal.
IN_TRACE = [
    "patient_hiv_status", "salary_band_x91", "Zanzibar-Ward-7",
    "acme_layoffs_q3_2026.csv", "who in Zanzibar-Ward-7 has patient_hiv_status",
]
#: A planted cell value. The trace holds step outputs and metadata, not cells,
#: so it does not carry this either; the projection must not, whatever a
#: future firefly puts in its output.
PLANTED = IN_TRACE + ["987654.321"]


@pytest.fixture(scope="module")
def planted_mission(tmp_path_factory):
    base = tmp_path_factory.mktemp("planted")
    rng = np.random.default_rng(3)
    n = 300
    frame = pd.DataFrame({
        "patient_hiv_status": rng.integers(0, 2, n),
        "salary_band_x91": rng.normal(70, 20, n).round(3),
        "ward": rng.choice(["Zanzibar-Ward-7", "Ward-2"], n),
        "churned": rng.integers(0, 2, n),
    })
    frame.loc[0, "salary_band_x91"] = 987654.321
    path = base / "acme_layoffs_q3_2026.csv"
    frame.to_csv(path, index=False)
    recorder = SwarmTraceRecorder()
    MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(path), goal="who in Zanzibar-Ward-7 has patient_hiv_status", output_dir=base / "fleet",
        workers=2, kind=MissionKind.TRAIN, target="churned", metric="f1", recorder=recorder)
    return recorder.mission()


def test_projection_is_accepted_by_the_ingest_schema(planted_mission):
    events = swarm_events(planted_mission)
    assert events, "nothing projected"
    ingest.validate_batch(events)  # raises Rejected on any unapproved field or value
    completed = [e for e in events if e["event_type"] == "agent.completed"]
    assert {e["agent"] for e in completed} >= {"data_scout", "model_scientist", "validator", "reporter"}
    assert all(e["outcome"] == "success" for e in completed)
    swarm = events[-1]
    assert swarm["event_type"] == "swarm.completed"
    assert swarm["clones_requested"] == swarm["clones_granted"] == swarm["parallel"] == 2
    assert swarm["limited"] is False and swarm["kind"] == "train" and swarm["outcome"] == "success"
    assert swarm["verdict"] in {"pass", "warn", "fail"} and swarm["row_count_bucket"] == "0-1e3"
    assert "edition" not in swarm  # UNLIMITED is not an edition; nothing is invented for it
    assert [e["seq"] for e in events] == list(range(len(events)))
    starts = [e for e in events if e["event_type"] == "agent.started"]
    assert all(c["started_seq"] == s["seq"] for s, c in zip(starts, completed))


def test_projection_carries_none_of_the_planted_content(planted_mission):
    # The trace does carry it -- it is the user's own fleet at work ...
    trace_text = json.dumps(planted_mission, default=str)
    for planted in IN_TRACE:
        assert planted in trace_text, planted
    # ... and the telemetry projection carries none of it, nor anything that
    # names the mission or a clone.
    wire = json.dumps(swarm_events(planted_mission))
    for planted in PLANTED + [planted_mission["mission_id"], "Ava-00", "ava-00", "ava-01",
                              "churned", "ward", "fleet", "sub-avaloka"]:
        assert planted not in wire, planted
    for record in planted_mission["records"]:
        if record.get("step_id"):
            assert record["step_id"] not in wire


def test_a_failed_mission_reports_its_class_never_its_message():
    rec = SwarmTraceRecorder()
    rec.record("swarm.requested", kind="analyze", goal="secret goal", source="/data/acme_layoffs_q3_2026.csv",
               requested_clones=4, requested_parallel=None, granted_clones=3, granted_parallel=3,
               limited=True, message="Swarm limited to 3 clones", limits={"edition": "oss"})
    rec.record("clone.dispatched", clone_id="ava-00", name="Ava-00", shard_id=0,
               received={"rows": 10, "columns": ["patient_hiv_status"]})
    rec.record("clone.step.started", clone_id="ava-00", step_id="ava-00/s01", step_no=1, firefly="data_scout")
    rec.record("clone.step.completed", clone_id="ava-00", step_id="ava-00/s01", step_no=1, firefly="data_scout",
               outcome="failed", error_class="KeyError", error="'patient_hiv_status' not in acme_layoffs_q3_2026.csv",
               duration_ms=5, metadata={"llm_provider": "openrouter", "llm_model": "acme-private-finetune",
                                        "tokens_in": 12, "tokens_out": 3})
    rec.record("clone.returned", clone_id="ava-00", status="failed", error_class="KeyError", error="...")
    rec.record("swarm.finished", status="failed", duration_ms=9, error_class="KeyError")
    events = swarm_events(rec.mission())
    ingest.validate_batch(events)
    started, completed, swarm = events
    assert completed["outcome"] == "failed" and completed["error_class"] == "KeyError"
    assert completed["llm_provider"] == "openrouter" and completed["llm_model"] == "other"
    assert completed["tokens_in"] == 12 and completed["tokens_out"] == 3
    assert swarm["outcome"] == "failed" and swarm["limited"] is True and swarm["edition"] == "oss"
    assert swarm["clones_requested"] == 4 and swarm["clones_granted"] == 3 and "verdict" not in swarm
    wire = json.dumps(events)
    for planted in ("patient_hiv_status", "acme_layoffs", "secret goal", "acme-private-finetune",
                    "Swarm limited", "ava-00"):
        assert planted not in wire, planted


def test_a_running_mission_projects_no_swarm_completed_and_no_open_step():
    rec = SwarmTraceRecorder()
    rec.record("swarm.requested", kind="analyze", granted_clones=1, granted_parallel=1, limited=False)
    rec.record("clone.dispatched", clone_id="ava-00", name="Ava-00", shard_id=0, received={"rows": 5})
    rec.record("clone.step.started", clone_id="ava-00", step_id="ava-00/s01", step_no=1, firefly="data_scout")
    rec.record("clone.step.completed", clone_id="ava-00", step_id="ava-00/s01", step_no=1, firefly="data_scout",
               outcome="ok", duration_ms=3, metadata={})
    rec.record("clone.step.started", clone_id="ava-00", step_id="ava-00/s02", step_no=2, firefly="sampling_specialist")
    events = swarm_events(rec.mission())
    assert [e["event_type"] for e in events] == ["agent.started", "agent.completed"]
    ingest.validate_batch(events)


def test_a_firefly_the_ingest_does_not_know_is_dropped_not_renamed():
    rec = SwarmTraceRecorder()
    rec.record("clone.step.completed", clone_id="ava-00", step_id="ava-00/s01", step_no=1,
               firefly="acme_custom_firefly", outcome="ok", duration_ms=1, metadata={})
    assert swarm_events(rec.mission()) == []


def test_vocabularies_agree_with_the_ingest_service():
    """Every firefly the projection can name is one the ingest accepts, and
    every firefly a mission can run is one the projection can name."""
    from avaloka.missions import analyze_fireflies, train_fireflies

    assert KNOWN_FIREFLIES <= set(ingest.AGENTS.values), KNOWN_FIREFLIES - set(ingest.AGENTS.values)
    runnable = {cls.name for cls in analyze_fireflies() + train_fireflies(deployable=True)}
    assert runnable <= KNOWN_FIREFLIES, runnable - KNOWN_FIREFLIES
    swarm = ingest.PROCESS_FIELDS_BY_TYPE["swarm.completed"]
    assert set(swarm) == {"event_id", "clones_requested", "clones_granted", "parallel", "limited",
                          "edition", "kind", "outcome", "verdict", "duration_ms", "row_count_bucket",
                          "session_id", "seq"}
    assert not set(swarm) & ingest.REFUSED_KEYS


def test_the_ingest_refuses_the_goal_on_a_swarm_record():
    """The server-side half of the control: a client that started sending the
    goal would have its batch rejected, not quietly filtered."""
    record = {"event_type": "swarm.completed", "event_id": "a" * 32, "clones_requested": 2,
              "clones_granted": 2, "parallel": 2, "limited": False, "outcome": "success",
              "duration_ms": 1, "row_count_bucket": "0-1e3", "session_id": "b" * 32, "seq": 0}
    ingest.validate(record)
    with pytest.raises(ingest.Rejected, match="refused"):
        ingest.validate({**record, "goal": "what drives churn"})
    with pytest.raises(ingest.Rejected, match="unpermitted"):
        ingest.validate({**record, "clone_names": ["Ava-00"]})


def test_the_sizing_reason_stays_in_the_local_record():
    """``derived_from`` (2.0) is for the trace and the narration; the
    projection carries the counts it already carried and nothing else."""
    rec = SwarmTraceRecorder()
    rec.record("swarm.requested", kind="analyze", requested_clones=5, requested_parallel=None,
               granted_clones=2, granted_parallel=2, limited=True,
               message="Swarm limited to 2 clones on the free open-source edition: the data sized the fleet at 5, so 2 will run.",
               derived_from="PLANTED-REASON 120,000 rows: ceil(rows/25,000)=5; 3 row_groups -> 3",
               limits={"edition": "oss"})
    rec.record("swarm.finished", status="done", duration_ms=9, economics=None)
    events = swarm_events(rec.mission())
    ingest.validate_batch(events)
    (swarm,) = events
    assert swarm["clones_requested"] == 5 and swarm["clones_granted"] == 2 and swarm["limited"] is True
    wire = json.dumps(events)
    assert "PLANTED-REASON" not in wire and "row_groups" not in wire and "derived_from" not in wire
