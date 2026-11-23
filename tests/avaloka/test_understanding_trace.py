"""The understanding records: one per agent step, one per mission, two layers.

Unit tests per component and per record type on synthetic blackboards, then
a real two-clone mission: every completed step has its understanding record
in the stream, the local log holds the same records, the egress file holds
what the ingest accepts, and the trace still ends ``merged, finished``.
"""

from __future__ import annotations

import json

import pytest

from avaloka.coordinator import MainAvaloka
from avaloka.mission.context import MissionKind
from avaloka.swarm_limits import UNLIMITED
from avaloka.swarm_trace import SwarmTraceRecorder, mission_document
from avaloka import understanding_trace as ut
from avaloka.understanding_trace import (AGENT_RECORD, EGRESS_FILE, LOG_FILE, SWARM_RECORD,
                                         AgentUnderstanding, DataShape, ModelUsage, OperationSummary,
                                         SamplingDecision, SwarmUnderstanding, UnderstandingLog,
                                         VerificationSummary, egress_event, egress_events, read_log)
from telemetry_service import schema as ingest

PROFILE = {
    "dataset": {"n_rows": 300, "n_cols": 4, "format": "csv"},
    "columns": [
        {"name": "age", "role": "numeric", "missing_pct": 0.0, "n_unique": 60},
        {"name": "ward", "role": "categorical", "missing_pct": 0.05, "n_unique": 2},
        {"name": "note", "role": "text", "missing_pct": 0.6, "n_unique": 290},
        {"name": "flag", "role": "boolean", "missing_pct": 0.004, "n_unique": 2},
    ],
    "quality": {"score": 71, "issues": [{"kind": "high_missingness", "column": "note", "severity": "high"}]},
}


class Unit:
    """A WorkUnit stand-in: only the attributes the builder reads."""

    def __init__(self, firefly, output="did the thing", **meta):
        self.firefly = firefly
        self.human_role = "Specialist"
        self.output = output
        self.duration_seconds = 0.25
        self.manual_minutes = 30.0
        self.compute_cost_usd = 0.0002
        self.model_cost_usd = 0.0
        self.metadata = meta


def step(no, firefly):
    return {"step_id": f"ava-00/s{no:02d}", "step_no": no, "firefly": firefly}


# ------------------------------------------------------------------ buckets
@pytest.mark.parametrize("fn, value, expected", [
    (ut.row_bucket, 0, "0-1e3"), (ut.row_bucket, 999, "0-1e3"), (ut.row_bucket, 1000, "1e3-1e4"),
    (ut.row_bucket, 10 ** 8, "1e8+"), (ut.row_bucket, None, "0-1e3"),
    (ut.column_bucket, 1, "1-5"), (ut.column_bucket, 5, "5-10"), (ut.column_bucket, 250, "200+"),
    (ut.null_rate_bucket, 0.0, "0"), (ut.null_rate_bucket, 0.004, "0-1%"), (ut.null_rate_bucket, 0.05, "1-10%"),
    (ut.null_rate_bucket, 0.3, "10-50%"), (ut.null_rate_bucket, 0.9, "50%+"), (ut.null_rate_bucket, None, "0"),
    (ut.cardinality_bucket, 1, "1"), (ut.cardinality_bucket, 2, "2"), (ut.cardinality_bucket, 7, "3-10"),
    (ut.cardinality_bucket, 100, "11-100"), (ut.cardinality_bucket, 1000, "101-1000"), (ut.cardinality_bucket, 5000, "1000+"),
    (ut.size_bucket, None, None), (ut.size_bucket, 10, "<1MB"), (ut.size_bucket, 5 * 2 ** 20, "1-10MB"),
    (ut.size_bucket, 2 ** 31, "1GB+"),
    (ut.fraction_bucket, 1.0, "full"), (ut.fraction_bucket, 0.5, "50-100%"), (ut.fraction_bucket, 0.2, "10-50%"),
    (ut.fraction_bucket, 0.05, "1-10%"), (ut.fraction_bucket, 0.001, "<1%"),
])
def test_buckets_are_closed_vocabularies(fn, value, expected):
    assert fn(value) == expected


def test_every_bucket_label_is_in_its_vocabulary():
    assert {ut.row_bucket(10 ** k) for k in range(0, 10)} <= set(ut.ROW_BUCKETS)
    assert {ut.column_bucket(n) for n in range(0, 300)} <= set(ut.COLUMN_BUCKETS)
    assert {ut.null_rate_bucket(x / 100) for x in range(0, 101)} <= set(ut.NULL_RATE_BUCKETS)
    assert {ut.cardinality_bucket(n) for n in range(0, 2000)} <= set(ut.CARDINALITY_BUCKETS)
    assert {ut.size_bucket(2 ** k) for k in range(0, 40)} <= set(ut.SIZE_BUCKETS)
    assert {ut.fraction_bucket(x / 1000) for x in range(0, 1001)} <= set(ut.FRACTION_BUCKETS)


# --------------------------------------------------------------- components
def test_data_shape_turns_a_profile_into_histograms():
    shape = DataShape.from_profile(PROFILE, size_bytes=3 * 2 ** 20).as_dict()
    assert shape["rows"] == 300 and shape["columns"] == 4
    assert shape["row_count_bucket"] == "0-1e3" and shape["column_count_bucket"] == "1-5"
    assert shape["column_type_counts"] == {"numeric": 1, "categorical": 1, "text": 1, "boolean": 1}
    assert shape["null_rate_counts"] == {"0": 1, "1-10%": 1, "50%+": 1, "0-1%": 1}
    assert shape["cardinality_counts"] == {"11-100": 1, "2": 2, "101-1000": 1}
    assert shape["file_format"] == "csv" and shape["size_bucket"] == "1-10MB" and shape["quality_score"] == 71
    # The local layer keeps the names; the histograms above are what travels.
    assert shape["column_names"] == ["age", "ward", "note", "flag"]
    assert shape["issues"][0]["column"] == "note"
    assert DataShape.from_profile(None).as_dict()["rows"] is None
    assert DataShape.from_profile({"dataset": {"format": ".AVRO"}}).as_dict()["file_format"] == "other"


def test_an_unknown_column_role_is_counted_as_other_not_passed_through():
    shape = DataShape.from_profile({"columns": [{"name": "x", "role": "acme_special"}]})
    assert shape.column_type_counts == {"other": 1}


def test_sampling_decision_derives_the_fraction():
    full = SamplingDecision.from_blackboard({"sampled": False, "strategy": "none", "n_input_rows": 10,
                                             "n_working_rows": 10, "reason": "small"}).as_dict()
    assert full["fraction"] == 1.0 and full["fraction_bucket"] == "full"
    part = SamplingDecision.from_blackboard({"sampled": True, "strategy": "stratified", "n_input_rows": 400_000,
                                             "n_working_rows": 50_000}).as_dict()
    assert part["fraction"] == 0.125 and part["fraction_bucket"] == "10-50%"
    assert SamplingDecision.from_blackboard(None) is None


def test_operations_count_by_kind_and_unknown_kinds_are_other():
    ops = OperationSummary(detail=[{"op": "drop_constant_columns", "columns": ["x"]},
                                   {"op": "strip_whitespace", "columns": ["a", "b"]},
                                   {"op": "strip_whitespace", "columns": ["c"]},
                                   {"op": "acme_magic"}]).as_dict()
    assert ops["count"] == 4
    assert ops["op_counts"] == {"drop_constant_columns": 1, "strip_whitespace": 2, "other": 1}
    assert ops["detail"][0]["columns"] == ["x"]  # local layer only


def test_verification_counts_each_status():
    v = VerificationSummary(checks=[{"status": "pass"}, {"status": "warn"}, {"status": "fail"}, {"status": "pass"}],
                            retries=2, verdict="fail").as_dict()
    assert (v["checks_run"], v["passed"], v["warned"], v["failed"], v["retries"]) == (4, 2, 1, 1, 2)
    assert v["verdict"] == "fail"


def test_model_usage_from_metadata_and_from_inference():
    assert ModelUsage.from_metadata({"quality_score": 1}) is None
    m = ModelUsage.from_metadata({"llm_provider": "groq", "llm_model": "openai/gpt-oss-120b",
                                  "tokens_in": 10, "tokens_out": 5}).as_dict()
    assert m == {"provider": "groq", "model_id": "openai/gpt-oss-120b", "calls": 1, "tokens_in": 10, "tokens_out": 5}
    i = ModelUsage.from_inference([{"provider": "local", "model": "gemma4:e4b", "tokens_in": 3, "tokens_out": 1},
                                   {"provider": "local", "model": "gemma4:e4b", "tokens_in": 4, "tokens_out": 2}])
    assert i.as_dict() == {"provider": "local", "model_id": "gemma4:e4b", "calls": 2, "tokens_in": 7, "tokens_out": 3}
    assert ModelUsage.from_inference([]) is None


# ---------------------------------------------------------- agent records
def test_the_scout_records_the_shape_and_the_quality_decision(tmp_path):
    src = tmp_path / "shard.csv"
    src.write_bytes(b"x" * 2048)
    rec = AgentUnderstanding.from_firefly(step=step(1, "data_scout"), unit=Unit("data_scout"),
                                          blackboard={"profile": PROFILE}, clone_id="ava-00", clone_index=0,
                                          data_source=str(src)).as_record()
    assert rec["role"] == "understanding" and rec["outcome"] == "ok" and rec["concluded"] == "did the thing"
    assert rec["decided"] == {"quality_score": 71, "n_issues": 1, "issue_kinds": {"high_missingness": 1}}
    assert rec["data"]["size_bytes"] == 2048 and rec["data"]["size_bucket"] == "<1MB"
    assert rec["timing"] == {"duration_ms": 250, "manual_minutes": 30.0, "compute_cost_usd": 0.0002,
                             "model_cost_usd": 0.0}
    assert rec["sampling"] is None and rec["operations"] is None and rec["verification"] is None


def test_the_sampler_records_its_decision_and_the_rows_it_kept():
    bb = {"profile": PROFILE, "sampling": {"sampled": True, "strategy": "random", "n_input_rows": 300,
                                           "n_working_rows": 30, "reason": "it is big"}}
    rec = AgentUnderstanding.from_firefly(step=step(2, "sampling_specialist"), unit=Unit("sampling_specialist"),
                                          blackboard=bb).as_record()
    assert rec["role"] == "sampling"
    assert rec["sampling"]["sampled"] and rec["sampling"]["fraction"] == 0.1 and rec["sampling"]["reason"] == "it is big"
    assert rec["decided"] == {"sampled": True, "strategy": "random", "reason": "it is big"}
    assert rec["data"]["rows"] == 30  # what the sampler handed on, not what it received


def test_the_engineer_records_its_operations_and_the_shape_it_produced():
    class Frame:
        shape = (290, 3)

        def __bool__(self):
            raise ValueError("The truth value of a DataFrame is ambiguous")

    bb = {"profile": PROFILE, "transformed_frame": Frame(),
          "transformations": [{"op": "drop_duplicate_rows", "rows_removed": 10},
                              {"op": "flag_high_missingness", "columns": ["note"], "applied": False}]}
    rec = AgentUnderstanding.from_firefly(step=step(3, "data_engineer"), unit=Unit("data_engineer"),
                                          blackboard=bb).as_record()
    assert rec["role"] == "feature_engineering"
    assert rec["operations"]["op_counts"] == {"drop_duplicate_rows": 1, "flag_high_missingness": 1}
    assert rec["decided"] == {"n_transformations": 2, "applied": ["drop_duplicate_rows"],
                              "deferred": ["flag_high_missingness"]}
    assert (rec["data"]["rows"], rec["data"]["columns"]) == (290, 3)


def test_the_feature_engineer_records_its_recipe_not_the_engineers_transformations():
    class Step:
        def __init__(self, operation, inputs, output):
            self.operation, self.inputs, self.output = operation, inputs, output

    class Frame:
        """Like a DataFrame: has a shape, and refuses to be a boolean."""
        shape = (290, 6)

        def __bool__(self):
            raise ValueError("The truth value of a DataFrame is ambiguous")

    class Result:
        recipe = type("R", (), {"steps": (Step("ratio", ("a", "b"), "a_per_b"), Step("log", ("a",), "log_a"))})()
        recipe_hash = "abc123"
        diagnostics = ("kept all rows",)
        frame = Frame()

    bb = {"profile": PROFILE, "transformations": [{"op": "drop_duplicate_rows"}], "feature_result": Result()}
    rec = AgentUnderstanding.from_firefly(step=step(5, "feature_engineer"), unit=Unit("feature_engineer"),
                                          blackboard=bb).as_record()
    assert rec["role"] == "feature_engineering"
    assert rec["operations"]["op_counts"] == {"other": 2}  # recipe operations are not a wire vocabulary
    assert rec["operations"]["detail"][0] == {"op": "ratio", "columns": ["a", "b", "a_per_b"]}
    assert rec["decided"]["operations"] == ["ratio", "log"] and rec["decided"]["recipe_hash"] == "abc123"
    assert (rec["data"]["rows"], rec["data"]["columns"]) == (290, 6)
    opt = AgentUnderstanding.from_firefly(step=step(3, "workload_optimizer"), unit=Unit("workload_optimizer"),
                                          blackboard={"profile": PROFILE, "workload_optimization": {
                                              "strategy": "full_scan", "filter_factors": [{"column": "a"}],
                                              "pushdown_filters": [], "projection_columns": ["a", "b"],
                                              "predicate_order": ["a"], "estimated_rows_after_filters": 150}}).as_record()
    assert opt["role"] == "planning"
    assert opt["decided"] == {"strategy": "full_scan", "n_filter_factors": 1, "n_pushdown_filters": 0,
                              "n_projection_columns": 2, "n_indexed_columns": 0, "n_partition_columns": 0,
                              "estimated_rows_after_filters": 150, "predicate_order": ["a"]}


def test_the_planner_records_the_task_the_split_and_the_goal_analysis():
    bb = {"profile": PROFILE, "plan": {
        "task": "binary_classification", "task_reason": "two classes",
        "split": {"strategy": "stratified_train_test", "reason": "balance"},
        "candidate_algorithms": ["logistic_regression", "random_forest"],
        "human_approval_gates": ["Confirm the target."], "data_sufficiency": {"sufficient": True}},
          "goal_analysis": {"kind": "missingness", "status": "answered", "scope": "All 300 working rows",
                            "findings": ["note is 60% missing"], "metrics": {"secret": 1}}}
    rec = AgentUnderstanding.from_firefly(step=step(4, "analysis_planner"), unit=Unit("analysis_planner"),
                                          blackboard=bb).as_record()
    d = rec["decided"]
    assert rec["role"] == "planning" and d["task"] == "binary_classification" and d["split"] == "stratified_train_test"
    assert d["candidate_algorithms"] == ["logistic_regression", "random_forest"]
    assert d["goal_analysis"] == {"kind": "missingness", "status": "answered", "scope": "All 300 working rows",
                                  "findings": ["note is 60% missing"]}


def test_the_scientist_records_the_model_choice_with_its_baseline():
    bb = {"profile": PROFILE, "training_summary": {
        "selected": "random_forest", "metric": "f1", "selected_metrics": {"f1": 0.81},
        "baseline_score": 0.5, "baseline_strategy": "most_frequent", "beats_baseline": True,
        "candidates": [{}, {}, {}], "n_train": 240, "n_test": 60, "feature_columns": ["age", "ward", "flag"]}}
    rec = AgentUnderstanding.from_firefly(step=step(5, "model_scientist"), unit=Unit("model_scientist"),
                                          blackboard=bb).as_record()
    assert rec["role"] == "modeling"
    assert rec["decided"] == {"selected": "random_forest", "metric": "f1", "score": 0.81, "baseline_score": 0.5,
                              "baseline_strategy": "most_frequent", "beats_baseline": True, "n_candidates": 3,
                              "n_train": 240, "n_test": 60, "n_features": 3}


def test_the_validator_records_every_check_and_the_verdict():
    bb = {"profile": PROFILE, "validation": {
        "verdict": "warn", "max_safe_deployment_level": 2,
        "checks": [{"name": "data_sufficiency", "status": "pass", "detail": "fine"},
                   {"name": "stability", "status": "warn", "detail": "CV std/mean = 20%"}],
        "blocking_issues": [], "leakage": []}}
    rec = AgentUnderstanding.from_firefly(step=step(6, "validator"), unit=Unit("validator"), blackboard=bb).as_record()
    v = rec["verification"]
    assert rec["role"] == "verification" and (v["checks_run"], v["passed"], v["warned"], v["failed"]) == (2, 1, 1, 0)
    assert v["verdict"] == "warn" and v["checks"][1]["detail"] == "CV std/mean = 20%"
    assert rec["decided"]["max_safe_deployment_level"] == 2 and rec["decided"]["leakage_findings"] == 0


def test_a_failed_step_has_no_unit_and_says_why_by_class():
    rec = AgentUnderstanding.from_firefly(step=step(1, "data_scout"), unit=None, blackboard={},
                                          outcome="failed", error_class="KeyError").as_record()
    assert rec["outcome"] == "failed" and rec["error_class"] == "KeyError"
    assert rec["concluded"] is None and rec["data"] is None and rec["timing"]["duration_ms"] is None


def test_model_usage_comes_from_the_unit_metadata():
    rec = AgentUnderstanding.from_firefly(
        step=step(9, "reporter"), blackboard={"profile": PROFILE},
        unit=Unit("reporter", llm_provider="groq", llm_model="openai/gpt-oss-120b", tokens_in=100,
                  tokens_out=40, artifacts=5)).as_record()
    assert rec["model"] == {"provider": "groq", "model_id": "openai/gpt-oss-120b", "calls": 1,
                            "tokens_in": 100, "tokens_out": 40}
    assert rec["decided"] == {"artifacts": 5}  # the rest of the metadata, llm keys excluded


def test_an_app_trace_step_maps_onto_the_same_record():
    coder = {"kind": "agent", "node": "coder", "step_id": "s1", "seq": 3, "status": "ok", "latency_ms": 900,
             "inference": [{"provider": "groq", "model": "openai/gpt-oss-120b", "tokens_in": 500, "tokens_out": 80}],
             "role": "Writes the analysis code",
             "inputs": {"question": "avg salary by ward", "data_scope": {"rows": 1200, "columns": ["salary", "ward"]}},
             "decision": {"type": "verdict", "value": "ok", "reason": "code written"},
             "calculation": {"operations": [{"op": "aggregate", "formula": "df.groupby('ward')['salary'].mean()",
                                             "columns": ["ward", "salary"]}, {"op": "sort"}]}}
    rec = AgentUnderstanding.from_app_step(coder).as_record()
    assert rec["agent"] == "coder" and rec["role"] == "coding" and rec["concluded"] == "code written"
    assert rec["data"]["rows"] == 1200 and rec["data"]["column_names"] == ["salary", "ward"]
    assert rec["operations"]["op_counts"] == {"aggregate": 1, "sort": 1}
    assert rec["model"]["tokens_in"] == 500 and rec["timing"]["duration_ms"] == 900

    check = {"kind": "agent", "node": "validator_logical", "seq": 4, "status": "ok", "latency_ms": 10,
             "decision": {"type": "check", "check": "Logical review", "value": "fail", "reason": "wrong column"},
             "calculation": {"attempt": 2}}
    rec = AgentUnderstanding.from_app_step(check).as_record()
    assert rec["role"] == "verification" and rec["verification"]["failed"] == 1
    assert rec["verification"]["retries"] == 1 and rec["verification"]["verdict"] == "fail"
    assert AgentUnderstanding.from_app_step({"kind": "decision", "node": "route"}) is None


# ---------------------------------------------------------- swarm record
def _stream(verdicts=("warn", "pass"), selected=None):
    rec = SwarmTraceRecorder()
    rec.record("swarm.requested", kind="train", granted_clones=2, granted_parallel=2)
    rec.record("swarm.sharded", clones=2, total_rows=600, rows_per_clone=[300, 300], columns=["age", "ward", "note", "flag"])
    for i, verdict in enumerate(verdicts):
        cid = f"ava-{i:02d}"
        bb = {"profile": PROFILE}
        rec.record(AGENT_RECORD, **AgentUnderstanding.from_firefly(
            step=step(1, "data_scout"), unit=Unit("data_scout"), blackboard=bb, clone_id=cid, clone_index=i).as_record())
        bb["sampling"] = {"sampled": i == 0, "strategy": "random" if i == 0 else "none", "n_input_rows": 300,
                          "n_working_rows": 100 if i == 0 else 300}
        rec.record(AGENT_RECORD, **AgentUnderstanding.from_firefly(
            step=step(2, "sampling_specialist"), unit=Unit("sampling_specialist"), blackboard=bb,
            clone_id=cid, clone_index=i).as_record())
        bb["validation"] = {"verdict": verdict, "checks": [{"status": "pass"}, {"status": verdict}]}
        rec.record(AGENT_RECORD, **AgentUnderstanding.from_firefly(
            step=step(3, "validator"), unit=Unit("validator", llm_provider="groq", llm_model="x", tokens_in=5, tokens_out=1),
            blackboard=bb, clone_id=cid, clone_index=i).as_record())
        rec.record("clone.returned", clone_id=cid, status="done", verdict=verdict, model={"score": 0.5 + i / 10})
    merge = {"verdict": "warn", "max_safe_deployment_level": 2, "rules": {"verdict": "worst case"},
             "n_rows": 600, "contributions": [{"clone_id": "ava-00", "selected": selected == 0},
                                              {"clone_id": "ava-01", "selected": selected == 1}]}
    return rec, merge


def test_the_fleet_record_folds_the_steps_and_the_merge():
    rec, merge = _stream(selected=1)
    sw = SwarmUnderstanding.from_records(rec.records, merge=merge, duration_ms=1234).as_record()
    assert sw["clones"] == 2 and sw["parallel"] == 2 and sw["steps_total"] == 6
    assert sw["steps_per_clone"] == {"ava-00": 3, "ava-01": 3}
    assert sw["roles"] == {"understanding": 2, "sampling": 2, "verification": 2}
    assert sw["data"]["rows"] == 600 and sw["data"]["columns"] == 4
    assert sw["data"]["column_type_counts"] == {"numeric": 1, "categorical": 1, "text": 1, "boolean": 1}
    assert sw["sampling"] == {"sampled_clones": 1, "strategy_counts": {"random": 1, "none": 1},
                              "strategies": ["none", "random"]}
    assert sw["verification"] == {"checks_run": 4, "passed": 3, "warned": 1, "failed": 0, "retries": 0, "verdict": "warn"}
    assert sw["merge"]["agreement"] == "split" and sw["merge"]["selected_clone"] == "ava-01"
    assert sw["merge"]["model_selected"] is True and sw["merge"]["candidates"] == 2
    assert sw["merge"]["clone_verdicts"] == {"ava-00": "warn", "ava-01": "pass"}
    assert sw["model"] == {"provider": "groq", "model_id": "x", "calls": 2, "tokens_in": 10, "tokens_out": 2}
    assert sw["timing"] == {"duration_ms": 1234} and sw["outcome"] == "done"


def test_the_fleet_record_reads_a_recorded_merge_when_none_is_given():
    rec, merge = _stream(verdicts=("pass", "pass"))
    rec.record("swarm.merged", **merge)
    sw = SwarmUnderstanding.from_records(rec.records).as_record()
    assert sw["merge"]["verdict"] == "warn" and sw["merge"]["agreement"] == "unanimous"
    assert sw["merge"]["model_selected"] is False and sw["merge"]["selected_clone"] is None


# ------------------------------------------------------------- local layer
def test_the_log_is_appended_per_record_and_read_incrementally(tmp_path):
    path = tmp_path / LOG_FILE
    rec = SwarmTraceRecorder()
    with UnderstandingLog(path) as log:
        log.append(rec.record("swarm.requested", kind="analyze"))  # not an understanding record: skipped
        a = rec.record(AGENT_RECORD, agent="data_scout", role="understanding")
        log.append(a)
        assert [r["seq"] for r in read_log(path)] == [a["seq"]]  # visible before close: flushed per record
        b = rec.record(SWARM_RECORD, clones=1)
        log.append(b)
    assert [r["seq"] for r in read_log(path)] == [a["seq"], b["seq"]]
    assert [r["seq"] for r in read_log(path, after=a["seq"])] == [b["seq"]]
    assert list(read_log(path, after=b["seq"])) == []
    # A half-written last line is skipped, not an error.
    with open(path, "a") as fh:
        fh.write('{"seq": 99, "type": "agent.under')
    assert [r["seq"] for r in read_log(path)] == [a["seq"], b["seq"]]
    assert list(read_log(tmp_path / "missing.jsonl")) == []


def test_the_log_never_raises_into_the_mission(tmp_path):
    log = UnderstandingLog(tmp_path / "nope" / "deeper" / LOG_FILE)  # creates parents
    assert log.path.parent.is_dir()
    unwritable = UnderstandingLog(tmp_path)  # a directory: cannot be opened
    unwritable.append({"type": AGENT_RECORD, "seq": 1})  # swallowed
    unwritable.close()


# ----------------------------------------------------------- egress layer
def test_the_projection_vocabularies_are_the_ingest_vocabularies():
    assert ut.KNOWN_AGENTS == set(ingest.AGENTS.values)
    assert set(ut.ROLES) == ingest._ROLES.values
    assert set(ut.ROW_BUCKETS) == ingest._PROCESS_ROW_BUCKETS.values
    assert set(ut.COLUMN_BUCKETS) == ingest._PROCESS_COLUMN_BUCKETS.values
    assert set(ut.COLUMN_KINDS) == ingest.COLUMN_KINDS
    assert set(ut.NULL_RATE_BUCKETS) == ingest._NULL_RATE_BUCKETS
    assert set(ut.CARDINALITY_BUCKETS) == ingest._CARDINALITY_BUCKETS
    assert set(ut.SIZE_BUCKETS) == ingest._SIZE_BUCKETS.values
    assert set(ut.FRACTION_BUCKETS) == ingest._FRACTION_BUCKETS.values
    assert set(ut.SAMPLING_STRATEGIES) == ingest._SAMPLING_STRATEGIES
    assert set(ut.OP_KINDS) == ingest._OP_KINDS
    assert set(ut.FILE_FORMATS) == ingest._FILE_FORMATS.values
    assert set(ut.LLM_PROVIDERS) == ingest.LLM_PROVIDERS.values
    assert set(ut.LLM_MODELS) | {"other"} == ingest.LLM_MODELS.values
    for kind in (AGENT_RECORD, SWARM_RECORD):
        assert ut.EGRESS_KEYS[kind] == set(ingest.PROCESS_FIELDS_BY_TYPE[kind]) | {"event_type"}, kind
        assert not ut.EGRESS_KEYS[kind] & ingest.REFUSED_KEYS


def test_the_projection_is_accepted_by_the_ingest_and_numbered_from_the_batch():
    rec, merge = _stream(selected=0)
    rec.record(SWARM_RECORD, **SwarmUnderstanding.from_records(rec.records, merge=merge, duration_ms=9).as_record())
    events = egress_events(rec.records, start_seq=5, session_id="c" * 32)
    ingest.validate_batch(events)
    assert [e["event_type"] for e in events] == [AGENT_RECORD] * 6 + [SWARM_RECORD]
    assert [e["seq"] for e in events] == list(range(5, 12))  # the batch's seq, not the trace's
    assert all(e["session_id"] == "c" * 32 for e in events)
    scout = events[0]
    assert scout["agent"] == "data_scout" and scout["role"] == "understanding" and scout["outcome"] == "success"
    assert scout["row_count_bucket"] == "0-1e3" and scout["null_rate_counts"] == {"0": 1, "1-10%": 1, "50%+": 1, "0-1%": 1}
    assert "sampled" not in scout and "checks_run" not in scout  # nothing invented for a step that did neither
    sampler = events[1]
    assert sampler["sampled"] is True and sampler["sampling_strategy"] == "random" and sampler["sample_fraction_bucket"] == "10-50%"
    validator = events[2]
    assert validator["checks_run"] == 2 and validator["llm_provider"] == "groq" and validator["llm_model"] == "other"
    swarm = events[-1]
    assert swarm["clones"] == 2 and swarm["steps_per_clone_min"] == swarm["steps_per_clone_max"] == 3
    assert swarm["role_counts"] == {"understanding": 2, "sampling": 2, "verification": 2}
    assert swarm["merge_agreement"] == "split" and swarm["model_selected"] is True and swarm["candidates"] == 2
    assert swarm["sampling_strategy_counts"] == {"random": 1, "none": 1} and swarm["sampled_clones"] == 1
    for event in events:
        assert set(event) <= ut.EGRESS_KEYS[event["event_type"]]


def test_an_unknown_agent_or_a_running_record_is_dropped_not_renamed():
    assert egress_event({"type": AGENT_RECORD, "agent": "acme_custom_firefly", "role": "other"}) is None
    assert egress_event({"type": "clone.step.completed", "firefly": "data_scout"}) is None
    failed = egress_event({"type": AGENT_RECORD, "agent": "data_scout", "role": "understanding",
                           "outcome": "failed", "error_class": "KeyError: 'patient_hiv_status'"})
    assert failed["outcome"] == "failed" and failed["error_class"] == "Exception"  # not a class name: not echoed


def test_a_histogram_with_a_foreign_key_loses_that_key():
    event = egress_event({"type": AGENT_RECORD, "agent": "data_scout", "role": "understanding", "outcome": "ok",
                          "data": {"rows": 5, "columns": 1, "column_type_counts": {"numeric": 1, "salary": 1}},
                          "operations": {"count": 1, "op_counts": {"derive": 1, "df['bonus'] = ...": 1}}})
    assert event["column_type_counts"] == {"numeric": 1} and event["op_counts"] == {"derive": 1}
    ingest.validate(event)


# ------------------------------------------------------------ real mission
def _coordinate(churn_csv, tmp_path, **kw):
    recorder = SwarmTraceRecorder()
    cr = MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(churn_csv), goal="what drives churn", output_dir=tmp_path / "fleet", recorder=recorder, **kw)
    return cr, recorder


def test_a_real_mission_records_understanding_behind_every_step(churn_csv, tmp_path):
    cr, recorder = _coordinate(churn_csv, tmp_path, workers=2, kind=MissionKind.TRAIN, target="churned", metric="f1")
    records = recorder.records
    types = [r["type"] for r in records]
    assert types[-3:] == [SWARM_RECORD, "swarm.merged", "swarm.finished"]
    completed = [r for r in records if r["type"] == "clone.step.completed"]
    understood = [r for r in records if r["type"] == AGENT_RECORD]
    # One understanding record per completed step, right after it, same step.
    # The clones run on their own threads and append to one recorder, so the
    # fleet's stream interleaves them and "right after" holds within a clone:
    # in each clone's own records, the next one after a completed step is
    # that step's understanding.
    assert len({r["step_id"] for r in understood}) == len(understood)
    assert sorted(r["step_id"] for r in understood) == sorted(r["step_id"] for r in completed)
    pairs = []
    for clone_id in sorted({r["clone_id"] for r in completed}):
        mine = [r for r in records if r.get("clone_id") == clone_id]
        for i, c in enumerate(mine):
            if c["type"] != "clone.step.completed":
                continue
            u = mine[i + 1] if i + 1 < len(mine) else {}
            assert (u.get("type"), u.get("step_id")) == (AGENT_RECORD, c["step_id"])
            pairs.append((c, u))
    assert len(pairs) == len(completed) == len(understood)
    for c, u in pairs:
        assert u["seq"] > c["seq"] and u["agent"] == c["firefly"] and u["clone_id"] == c["clone_id"]
        assert u["outcome"] == "ok" and u["concluded"] == c["output"]
        assert u["timing"]["duration_ms"] == c["duration_ms"]
    roles = {r["agent"]: r["role"] for r in understood}
    assert roles == {"data_scout": "understanding", "sampling_specialist": "sampling",
                     "workload_optimizer": "planning", "data_engineer": "feature_engineering",
                     "feature_engineer": "feature_engineering", "analysis_planner": "planning",
                     "model_scientist": "modeling", "conformal_calibrator": "verification",
                     "validator": "verification", "finops": "finops", "reporter": "reporting"}
    assert all(r["clone_name"] == f"Ava-{r['clone_index']:02d}" for r in understood)
    by_agent = {r["agent"]: r for r in understood if r["clone_id"] == "ava-00"}
    scout = by_agent["data_scout"]["data"]
    assert scout["rows"] == 200 and scout["columns"] == 6 and "tenure" in scout["column_names"]
    assert sum(scout["column_type_counts"].values()) == 6 and scout["file_format"] == "csv"
    assert scout["size_bucket"] == "<1MB" and scout["size_bytes"] > 0
    assert by_agent["sampling_specialist"]["sampling"]["sampled"] is False
    assert by_agent["sampling_specialist"]["sampling"]["fraction_bucket"] == "full"
    assert by_agent["analysis_planner"]["decided"]["task"] == "binary_classification"
    assert by_agent["model_scientist"]["decided"]["selected"] and by_agent["model_scientist"]["decided"]["metric"] == "f1"
    ver = by_agent["validator"]["verification"]
    assert ver["checks_run"] >= 3 and ver["checks_run"] == ver["passed"] + ver["warned"] + ver["failed"]
    assert ver["verdict"] in {"pass", "warn", "fail"}

    assert by_agent["feature_engineer"]["operations"]["count"] == by_agent["feature_engineer"]["decided"]["n_steps"]
    assert by_agent["workload_optimizer"]["decided"]["strategy"]
    fleet = records[-3]
    # Eleven steps a clone: every train mission plans a conformal calibration
    # (#527), so the ConformalCalibrator runs ahead of the Validator.
    assert fleet["clones"] == 2 and fleet["steps_total"] == len(understood) == 22
    assert fleet["steps_per_clone"] == {"ava-00": 11, "ava-01": 11}
    assert fleet["data"]["rows"] == 400 and fleet["merge"]["verdict"] == cr.verdict
    assert fleet["merge"]["model_selected"] is True and fleet["merge"]["selected_clone"] in {"ava-00", "ava-01"}
    # The calibrator shares the verification role but runs no validation checks.
    assert by_agent["conformal_calibrator"]["verification"] is None
    assert fleet["verification"]["checks_run"] == sum(r["verification"]["checks_run"] for r in understood
                                                        if r["role"] == "verification"
                                                        and r["verification"] is not None)

    # The mission document carries the fleet's record; the derived document agrees.
    doc = recorder.mission()
    assert doc["understanding"]["steps_total"] == 22
    assert mission_document(doc["records"], include_records=False) == recorder.mission(include_records=False)

    # The local log holds exactly the understanding records, in stream order ...
    logged = list(read_log(tmp_path / "fleet" / LOG_FILE))
    assert [r["seq"] for r in logged] == [r["seq"] for r in records if r["type"] in ut.RECORD_TYPES]
    assert logged[0]["data"]["column_names"] == scout["column_names"]
    # ... and the egress file holds the projection the ingest accepts, with no column name in it.
    egress = json.loads((tmp_path / "fleet" / EGRESS_FILE).read_text())
    ingest.validate_batch(egress["events"])
    assert len(egress["events"]) == 23 and egress["events"][-1]["event_type"] == SWARM_RECORD
    assert "tenure" not in json.dumps(egress) and "Ava-0" not in json.dumps(egress)


def test_the_local_log_stays_in_seq_order_while_clones_record_together(churn_csv, tmp_path, monkeypatch):
    """understanding.jsonl is read incrementally by seq (``read_log(after=N)``),
    so it must be in seq order even when two clones record at once. Hold
    Ava-00's first understanding record after it has its seq until Ava-01
    has recorded one too: if the log is appended outside the moment the seq
    is taken, Ava-01's later seq reaches the file first."""
    import threading

    original = SwarmTraceRecorder.record
    other_recorded = threading.Event()
    held = []

    def record(self, type, **fields):  # noqa: A002 - mirrors the recorder
        rec = original(self, type, **fields)
        if type == AGENT_RECORD and rec.get("clone_id") == "ava-01":
            other_recorded.set()
        elif type == AGENT_RECORD and rec.get("clone_id") == "ava-00" and not held:
            held.append(rec["seq"])
            other_recorded.clear()
            other_recorded.wait(2.0)  # with the fix, Ava-01 cannot take a seq meanwhile
        return rec

    monkeypatch.setattr(SwarmTraceRecorder, "record", record)
    _, recorder = _coordinate(churn_csv, tmp_path, workers=2)
    assert held, "Ava-00 recorded no understanding"
    streamed = [r["seq"] for r in recorder.records if r["type"] in ut.RECORD_TYPES]
    logged = [r["seq"] for r in read_log(tmp_path / "fleet" / LOG_FILE)]
    assert logged == sorted(logged) == streamed
    after = held[0] + 1
    assert [r["seq"] for r in read_log(tmp_path / "fleet" / LOG_FILE, after=after)] == [s for s in streamed if s > after]


def test_an_understanding_bug_never_fails_the_mission(churn_csv, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("derivation bug")

    monkeypatch.setattr(AgentUnderstanding, "from_firefly", boom)
    cr, recorder = _coordinate(churn_csv, tmp_path, workers=2)
    assert cr.verdict in {"pass", "warn", "fail"}
    types = [r["type"] for r in recorder.records]
    assert AGENT_RECORD not in types and types[-2:] == ["swarm.merged", "swarm.finished"]
    assert types.count("clone.step.completed") == 16
    fleet = recorder.mission()["understanding"]
    assert fleet["steps_total"] == 0 and fleet["clones"] == 2  # folded from what there was


# ------------------------------------------------ model usage from a firefly
def test_a_narrating_firefly_records_its_model_usage(monkeypatch):
    """The Reporter is the one CLI firefly that calls a model (off the local
    execution mode). What it used lands in its unit metadata under the names
    the understanding record reads, and nothing lands when no model answered."""
    from avaloka.llm import LLMClient, usage_metadata

    assert usage_metadata(None) == {}
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert usage_metadata(LLMClient()) == {}  # no key: not available

    class Resp:
        class usage:
            input_tokens, output_tokens = 120, 30
        content = [type("B", (), {"type": "text", "text": "Two sentences."})()]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:11434")
    client = LLMClient(model="gemma4:e4b")
    client._client = type("C", (), {"messages": type("M", (), {"create": staticmethod(lambda **kw: Resp())})()})()
    assert client.complete("sys", "prompt") == "Two sentences."
    meta = usage_metadata(client)
    assert meta == {"llm_provider": "local", "llm_model": "gemma4:e4b", "llm_calls": 1,
                    "tokens_in": 120, "tokens_out": 30}
    rec = AgentUnderstanding.from_firefly(step=step(9, "reporter"), unit=Unit("reporter", **meta),
                                          blackboard={"profile": PROFILE}).as_record()
    assert rec["model"] == {"provider": "local", "model_id": "gemma4:e4b", "calls": 1, "tokens_in": 120, "tokens_out": 30}
    event = egress_event({"type": AGENT_RECORD, **rec})
    assert event["llm_provider"] == "local" and event["llm_model"] == "other"  # not a shipped default
    assert event["tokens_in"] == 120 and event["model_calls"] == 1
    ingest.validate(event)
