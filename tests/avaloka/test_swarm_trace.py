"""The swarm trace records every clone's work separately, and survives the ways
a fleet actually runs: threads appending at once, a shard that raises, the CLI.

The recorder is stdlib-only and lives in ``avaloka/``; these tests drive it
through the real coordinator on a small frame so the records are the ones a
user would get, not hand-built ones.
"""

from __future__ import annotations

import json
import math
import threading

import numpy as np
import pytest
from typer.testing import CliRunner

import avaloka.coordinator as coordinator
from avaloka.cli import app
from avaloka.coordinator import MainAvaloka
from avaloka.mission.context import MissionKind
from avaloka.missions import analyze_fireflies
from avaloka.swarm_limits import UNLIMITED
from avaloka.swarm_trace import (SwarmTraceRecorder, _plain, clone_document,
                                 episodes, mission_document)

# The mission's own order, so the test follows the fireflies the line ships
# (1.7 adds a workload optimizer and a feature engineer to every analysis).
ANALYZE_FIREFLIES = [cls.name for cls in analyze_fireflies()]


def _coordinate(churn_csv, tmp_path, *, kind=MissionKind.ANALYZE, workers=2, **kw):
    sink_seen = []
    recorder = SwarmTraceRecorder(sink=sink_seen.append)
    main = MainAvaloka(swarm_limits=UNLIMITED)
    cr = main.coordinate(str(churn_csv), goal="what drives churn", output_dir=tmp_path / "fleet",
                         workers=workers, kind=kind, recorder=recorder, **kw)
    return cr, recorder, sink_seen


# --------------------------------------------------------------- the recorder
def test_seq_is_monotonic_and_gap_free_under_threads():
    seen = []
    rec = SwarmTraceRecorder(mission_id="m1", sink=seen.append)

    def worker(i):
        for n in range(200):
            rec.record("clone.step.started", clone_id=f"ava-{i:02d}", step_no=n)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seqs = [r["seq"] for r in rec.records]
    assert seqs == list(range(1, 1601))
    # The sink saw every record, in seq order, exactly once.
    assert [r["seq"] for r in seen] == seqs
    assert rec.records_after(1598) == rec.records[-2:]
    assert rec.records_after(0, limit=3) == rec.records[:3]
    assert all(r["mission_id"] == "m1" and r["ts"].endswith("Z") for r in rec.records)


def test_a_failing_sink_never_fails_the_run():
    def bad(_rec):
        raise RuntimeError("sink down")

    rec = SwarmTraceRecorder(sink=bad)
    rec.record("swarm.requested", kind="analyze")
    assert rec.last_seq == 1


def test_records_are_json_plain_by_construction():
    rec = SwarmTraceRecorder()
    r = rec.record("clone.step.completed", clone_id="ava-00", metadata={
        "n": np.int64(3), "f": np.float32(1.5), "arr": np.arange(2), "nan": float("nan"),
        "kind": MissionKind.TRAIN, "inf": math.inf})
    json.dumps(r, allow_nan=False)  # a strict encoder (Starlette's) must not choke
    assert r["metadata"] == {"n": 3, "f": 1.5, "arr": [0, 1], "nan": None, "kind": "train", "inf": None}
    assert _plain((1, "a")) == [1, "a"]


# ------------------------------------------------------------ the coordinator
def test_every_clone_is_dispatched_steps_and_returns(churn_csv, tmp_path):
    cr, recorder, seen = _coordinate(churn_csv, tmp_path, workers=2)
    records = recorder.records
    assert [r["seq"] for r in records] == list(range(1, len(records) + 1))
    assert [r["seq"] for r in seen] == [r["seq"] for r in records]
    types = [r["type"] for r in records]
    assert types[0] == "swarm.requested" and types[1] == "swarm.sharded"
    assert types[-2:] == ["swarm.merged", "swarm.finished"]

    requested = records[0]
    assert requested["kind"] == "analyze" and requested["goal"] == "what drives churn"
    assert requested["granted_clones"] == 2 and requested["limited"] is False
    assert requested["limits"]["edition"] == "unlimited"
    sharded = records[1]
    assert sharded["clones"] == 2 and sum(sharded["rows_per_clone"]) == sharded["total_rows"] == 400
    assert "tenure" in sharded["columns"]

    for cid in ("ava-00", "ava-01"):
        own = [r for r in records if r.get("clone_id") == cid]
        assert own[0]["type"] == "clone.dispatched" and own[-1]["type"] == "clone.returned"
        assert own[0]["received"]["rows"] in sharded["rows_per_clone"]
        assert own[0]["received"]["columns"] == sharded["columns"]
        assert own[0]["received"]["goal"] == "what drives churn"
        started = [r for r in own if r["type"] == "clone.step.started"]
        completed = [r for r in own if r["type"] == "clone.step.completed"]
        assert [r["firefly"] for r in started] == ANALYZE_FIREFLIES
        assert [r["firefly"] for r in completed] == ANALYZE_FIREFLIES
        assert [r["step_id"] for r in completed] == [f"{cid}/s{n:02d}" for n in range(1, len(ANALYZE_FIREFLIES) + 1)]
        # Each step was started before it completed, and completed from its unit.
        for s, c in zip(started, completed):
            assert s["seq"] < c["seq"] and s["step_id"] == c["step_id"]
            assert c["outcome"] == "ok" and c["output"] and c["human_role"]
            assert c["duration_ms"] >= 0 and c["manual_minutes"] > 0
        assert own[-1]["status"] == "done" and own[-1]["steps"] == len(ANALYZE_FIREFLIES)
        assert own[-1]["verdict"] in {"pass", "warn", "fail"}

    merged = records[-2]
    assert merged["rules"] == {"profile": "exact sufficient statistics", "verdict": "worst case",
                               "model": "best honest score",
                               "answer": "sufficient statistics (sums and counts add, means weighted by n)"}
    # The answer to the goal is on the merge, with each clone's share cited.
    assert merged["answer"]["headline"] and len(merged["answer"]["contributions"]) == 2
    assert merged["verdict"] == cr.verdict and merged["n_rows"] == 400
    assert {c["clone_id"] for c in merged["contributions"]} == {"ava-00", "ava-01"}
    assert not any(c["selected"] for c in merged["contributions"])  # analyze selects no model
    finished = records[-1]
    assert finished["status"] == "done" and finished["economics"]["estimated_manual_effort_hours"] > 0

    # The coordinator wrote the trace beside coordination.json and said so.
    doc = json.loads((tmp_path / "fleet" / "swarm_trace.json").read_text())
    assert doc["status"] == "done" and len(doc["records"]) == len(records)
    assert cr.trace == {"path": "swarm_trace.json", "records": len(records),
                        "mission_id": recorder.mission_id}
    assert json.loads((tmp_path / "fleet" / "coordination.json").read_text())["trace"] == cr.trace


def test_the_mission_document_is_derived_from_the_records(churn_csv, tmp_path):
    cr, recorder, _ = _coordinate(churn_csv, tmp_path, workers=3)
    doc = recorder.mission()
    assert doc["schema_version"] == "1.0" and doc["mission_id"] == recorder.mission_id
    assert doc["status"] == "done" and doc["started_at"] < doc["ended_at"]
    assert doc["kind"] == "analyze" and doc["goal"] == "what drives churn"
    assert doc["grant"]["granted_clones"] == 3 and doc["limits"]["edition"] == "unlimited"
    assert [c["clone_id"] for c in doc["clones"]] == ["ava-00", "ava-01", "ava-02"]
    for clone in doc["clones"]:
        assert clone["status"] == "done" and clone["name"] == f"Ava-{clone['shard_id']:02d}"
        assert [s["firefly"] for s in clone["steps"]] == ANALYZE_FIREFLIES
        assert all(s["status"] == "ok" and s["started_at"] <= s["ended_at"] for s in clone["steps"])
        assert clone["rows"] == clone["received"]["rows"]
    assert doc["merge"]["verdict"] == cr.verdict
    # One source of truth: the same records give the same document.
    assert mission_document(doc["records"], include_records=False) == recorder.mission(include_records=False)
    assert "records" not in recorder.mission(include_records=False)
    one = clone_document(doc, "ava-01")
    assert one["clone_id"] == "ava-01" and all(r["clone_id"] == "ava-01" for r in one["records"])
    assert clone_document(doc, "ava-09") is None


def test_train_merge_selects_exactly_one_clone_and_episodes_say_so(churn_csv, tmp_path):
    cr, recorder, _ = _coordinate(churn_csv, tmp_path, kind=MissionKind.TRAIN, workers=2,
                                  target="churned", metric="f1")
    doc = recorder.mission()
    selected = [c for c in doc["merge"]["contributions"] if c["selected"]]
    assert len(selected) == 1
    assert selected[0]["clone_id"] == f"ava-{cr.selected_model['from_shard']:02d}"
    assert doc["merge"]["selected_model"]["from_shard"] == cr.selected_model["from_shard"]

    eps = episodes(doc)
    assert [e["episode_id"] for e in eps] == ["ava-00", "ava-01"]
    assert sum(e["terminal"]["selected_by_merge"] for e in eps) == 1
    for ep in eps:
        assert ep["mission_id"] == recorder.mission_id and ep["reward"] is None
        assert ep["context"]["kind"] == "train" and ep["context"]["target"] == "churned"
        assert ep["context"]["metric"] == "f1" and "tenure" in ep["context"]["columns"]
        assert ep["context"]["goal"] == "what drives churn" and ep["context"]["rows"] == 200
        assert [s["t"] for s in ep["steps"]] == list(range(len(ep["steps"])))
        assert ep["steps"][0]["action"] == {"firefly": "data_scout", "human_role": "Data analyst"}
        assert {"output", "metadata"} == set(ep["steps"][0]["observation"])
        assert {"duration_ms", "model_cost_usd", "compute_cost_usd", "manual_minutes"} == set(ep["steps"][0]["cost"])
        assert all(s["outcome"] == "ok" for s in ep["steps"])
        assert ep["terminal"]["status"] == "done" and ep["terminal"]["score"] is not None
        assert ep["terminal"]["cost_usd"] > 0 and ep["terminal"]["verdict"] in {"pass", "warn", "fail"}
    assert "model_scientist" in [s["action"]["firefly"] for s in eps[0]["steps"]]


def test_a_shard_that_raises_is_recorded_before_the_error_propagates(churn_csv, tmp_path, monkeypatch):
    real = coordinator.run_analyze

    def flaky(ctx, on_progress):
        if "sub-avaloka-01" in str(ctx.data_source):
            on_progress("start", "data_scout")  # the step opened, the firefly then blew up
            raise RuntimeError("shard 1 exploded")
        return real(ctx, on_progress=on_progress)

    monkeypatch.setattr(coordinator, "run_analyze", flaky)
    recorder = SwarmTraceRecorder()
    with pytest.raises(RuntimeError, match="shard 1 exploded"):
        MainAvaloka(swarm_limits=UNLIMITED).coordinate(
            str(churn_csv), goal="g", output_dir=tmp_path / "fleet", workers=2, recorder=recorder)
    records = recorder.records
    failed_step = next(r for r in records if r["type"] == "clone.step.completed" and r["clone_id"] == "ava-01")
    assert failed_step["outcome"] == "failed" and failed_step["error_class"] == "RuntimeError"
    assert failed_step["step_id"] == "ava-01/s01" and failed_step["output"] is None
    returned = next(r for r in records if r["type"] == "clone.returned" and r["clone_id"] == "ava-01")
    assert returned["status"] == "failed" and returned["error_class"] == "RuntimeError"
    assert returned["steps"] == 0 and returned["seq"] > failed_step["seq"]
    assert records[-1]["type"] == "swarm.finished" and records[-1]["status"] == "failed"
    assert records[-1]["error_class"] == "RuntimeError"
    assert not any(r["type"] == "swarm.merged" for r in records)
    doc = recorder.mission()
    assert doc["status"] == "failed" and doc["error_class"] == "RuntimeError"
    assert {c["clone_id"]: c["status"] for c in doc["clones"]}["ava-01"] == "failed"
    assert episodes(doc)[1]["terminal"] == {
        "status": "failed", "verdict": None, "max_safe_deployment_level": None, "score": None,
        "selected_by_merge": False, "cost_usd": 0.0}
    # The failed trace is on disk too.
    assert json.loads((tmp_path / "fleet" / "swarm_trace.json").read_text())["status"] == "failed"


def test_a_request_refused_before_sharding_still_finishes_the_trace(churn_csv, tmp_path):
    recorder = SwarmTraceRecorder()
    with pytest.raises(ValueError, match="target column"):
        MainAvaloka(swarm_limits=UNLIMITED).coordinate(
            str(churn_csv), goal="g", output_dir=tmp_path / "fleet", workers=2,
            kind=MissionKind.TRAIN, target="nope", recorder=recorder)
    assert [r["type"] for r in recorder.records] == ["swarm.requested", "swarm.finished"]
    assert recorder.records[-1]["status"] == "failed" and recorder.records[-1]["error_class"] == "ValueError"


def test_the_coordinator_records_a_trace_even_when_none_is_given(churn_csv, tmp_path):
    cr = MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(churn_csv), goal="g", output_dir=tmp_path / "fleet", workers=2)
    doc = json.loads((tmp_path / "fleet" / "swarm_trace.json").read_text())
    assert doc["mission_id"] == cr.trace["mission_id"] and len(doc["clones"]) == 2


# ---------------------------------------------------------------------- CLI
def test_cli_writes_the_trace_and_streams_it_live(churn_csv, tmp_path):
    out = tmp_path / "fleet"
    result = CliRunner().invoke(app, ["coordinate", str(churn_csv), "--goal", "x",
                                      "--workers", "2", "-o", str(out), "--quiet"])
    assert result.exit_code == 0, result.output
    doc = json.loads((out / "swarm_trace.json").read_text())
    lines = [json.loads(line) for line in (out / "swarm_trace.jsonl").read_text().splitlines()]
    assert [r["seq"] for r in lines] == [r["seq"] for r in doc["records"]] == list(range(1, len(lines) + 1))
    assert lines[-1]["type"] == "swarm.finished" and lines[-1]["status"] == "done"
    report = json.loads((out / "coordination.json").read_text())
    assert report["trace"]["path"] == "swarm_trace.json" and report["trace"]["records"] == len(lines)
    flat = " ".join(result.output.split())  # rich wraps long paths
    assert "Swarm trace" in flat and "swarm_trace.json" in flat.replace(" ", "")


def test_cli_help_names_the_trace_files():
    result = CliRunner().invoke(app, ["coordinate", "--help"])
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "swarm_trace.json" in flat and "swarm_trace.jsonl" in flat
