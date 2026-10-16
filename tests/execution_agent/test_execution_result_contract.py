"""execution_result is written, and logged, the same way on every backend.

The k8s node (`execution_agent_node`) used to write `execution_result` only on
its two pre-flight failures. Once code had actually been sent to the executor
-- success or failure -- the key was left as it came in, while the local node
(`execution_agent_node_local`) wrote it on every outcome. Readers downstream
branch on `execution_result["status"]`, so the two backends did not behave the
same after the same run.

Three things are held here:

  1. a k8s run leaves a well-formed `execution_result`, on success and failure;
  2. k8s and local write the same keys, so they cannot drift apart again;
  3. each run emits one `execution_result ...` log line that carries behaviour
     only -- no column name, file name, cell value, code or question text.

Hermetic: the executor and the local interpreter are both replaced, so there is
no cluster, no subprocess and no network.
"""

import base64
import logging
import re
from unittest import mock

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage

import app.agents.execution_agent as exec_mod
from app.agents.execution_agent import (
    execution_agent_node,
    execution_agent_node_local,
)
from app.analytics import schema

LOGGER_NAME = "app.agents.execution_agent"

# Deliberately unmistakable. If any of these reaches a log line, the search
# below cannot miss it and nothing else in the run can be mistaken for it.
COLUMN_A = "zq_patient_surname_xk"
COLUMN_B = "zq_diagnosis_code_xk"
CELL_A = "Vasquez-Okonkwo"
CELL_B = "ICDQ7731"
DATASET_STEM = "acme_oncology_roster_zq"
QUESTION = "zq which surname has the most diagnosis codes xk"
CODE = f"df.groupby('{COLUMN_A}')['{COLUMN_B}'].count().to_csv(output_file)"
STDERR = f"Traceback (most recent call last):\nKeyError: '{COLUMN_A}'"

FORBIDDEN = (COLUMN_A, COLUMN_B, CELL_A, CELL_B, DATASET_STEM, QUESTION, "groupby", "KeyError")

# The keys both nodes write when they report an outcome. (output_location is
# passed through unchanged by both, so it does not show up as written.)
STATE_KEYS = {
    "messages", "output_file_data", "output_json",
    "fresh_output_produced", "execution_stdout", "execution_stderr",
    "execution_error", "execution_result",
}


def _frame() -> pd.DataFrame:
    return pd.DataFrame({COLUMN_A: [CELL_A, "Adeyemi-Lindqvist", CELL_A],
                         COLUMN_B: [CELL_B, "ICDQ7732", "ICDQ7733"]})


def _state(**overrides):
    state = {
        "messages": [HumanMessage(content=QUESTION)],
        "coder_definition": {"code": CODE},
        # Already set, as it is after the coding subgraph: the local node only
        # copies the code here when it is missing, which is not what is compared.
        "generated_code": CODE,
        "data_source_location": f"/data/{DATASET_STEM}.csv",
        "output_location": f"/data/{DATASET_STEM}_output.csv",
        "schema": {COLUMN_A: "string", COLUMN_B: "string"},
        "infrastructure_provisioned": {
            "status": "provisioned",
            "details": [{
                "step_name": "Get infra-agent-service IP/Port",
                "status": "SUCCESS",
                "details": {"ip": "10.0.0.7", "port": "8080"},
            }],
        },
    }
    state.update(overrides)
    return state


@pytest.fixture(autouse=True)
def _no_side_channels(monkeypatch):
    """Lineage and the memory record are not what is under test."""
    monkeypatch.setattr(exec_mod, "record_analysis_lineage", lambda *a, **k: None)
    monkeypatch.setattr(exec_mod, "_dispatch_execution_memory_record", lambda *a, **k: None)


def _k8s_response(outcome: str, with_file: bool = True) -> dict:
    if outcome == "failed":
        return {"status": "error", "message": "Code execution failed", "stderr": STDERR,
                "execution_time": 0.4}
    csv = _frame().to_csv(index=False)
    payload = "data:text/csv;base64," + base64.b64encode(csv.encode()).decode()
    response = {"status": "success", "stdout": "", "stderr": "", "output": payload,
                "execution_time": 1.2}
    if with_file:
        response["output_file"] = {"filename": "etl_results_1a2b3c4d_20260101_000000.csv",
                                   "content": payload, "size": len(csv)}
    return response


def _run_k8s(outcome: str, state=None, with_file: bool = True):
    state = state if state is not None else _state()
    with mock.patch.object(exec_mod, "execute_code_on_k8s",
                           return_value=_k8s_response(outcome, with_file)):
        return state, execution_agent_node(dict(state))


def _run_local(outcome: str, tmp_path, state=None):
    out_path = tmp_path / f"{DATASET_STEM}_output.csv"
    state = state if state is not None else _state(output_location=str(out_path))

    def fake_interpreter(**kwargs):
        if outcome == "failed":
            return {"status": "error", "execution_error": "Code execution failed",
                    "execution_stdout": "", "execution_stderr": STDERR, "returncode": 1}
        _frame().to_csv(kwargs["output_location"], index=False)
        return {"status": "success", "execution_stdout": "", "execution_stderr": ""}

    with mock.patch.object(exec_mod, "execute_code_on_local", side_effect=fake_interpreter):
        return state, execution_agent_node_local(dict(state))


def _run(backend: str, outcome: str, tmp_path):
    if backend == "k8s":
        return _run_k8s(outcome)
    return _run_local(outcome, tmp_path)


# ---------------------------------------------------------------------------
# 1. A k8s run leaves execution_result behind
# ---------------------------------------------------------------------------

def test_k8s_success_writes_a_well_formed_execution_result():
    state_in, out = _run_k8s("success")

    assert "execution_result" in out, "a k8s success left execution_result unwritten"
    result = out["execution_result"]
    assert result == {
        "status": "success",
        "message": "ETL job completed successfully",
        "output_file": state_in["output_location"],
        "fresh_output": True,
        "output_row_count": 3,
    }
    assert out["fresh_output_produced"] is True
    assert out["execution_error"] is None
    assert out["output_json"] == _frame().to_dict(orient="records")
    assert out["output_file_data"]["filename"] == "etl_results_1a2b3c4d_20260101_000000.csv"


def test_k8s_success_overwrites_the_empty_result_the_api_seeds():
    """The API and the Streamlit app both start a turn with execution_result={}."""
    _, out = _run_k8s("success", state=_state(execution_result={}))

    assert out["execution_result"]["status"] == "success"


def test_k8s_runtime_failure_writes_a_failed_execution_result():
    """Not only success: a run the executor rejected used to leave no result either."""
    _, out = _run_k8s("failed", state=_state(execution_result={}))

    result = out["execution_result"]
    assert result["status"] == "failed"
    assert set(result) == {"status", "message", "returncode", "output_file"}
    assert result["output_file"] is None
    # The state keeps the detail for the repair loop; only the log line may not.
    assert COLUMN_A in result["message"]
    assert out["execution_error"] == "Code execution failed"
    assert out["fresh_output_produced"] is False
    assert out["output_file_data"] is None


def test_k8s_success_without_a_returned_file_still_carries_the_output():
    """A success with no file attached would let a reader fall back to disk."""
    _, out = _run_k8s("success", with_file=False)

    file_data = out["output_file_data"]
    assert file_data["content"] == _k8s_response("success")["output"]
    assert file_data["size"] == len(_frame().to_csv(index=False).encode())
    assert file_data["filename"].endswith("_output.csv")
    assert out["execution_result"]["status"] == "success"


def test_k8s_reply_that_is_not_a_csv_is_a_failure_not_a_success():
    """Same verdict the local node gives a run that produced no readable output."""
    with mock.patch.object(exec_mod, "execute_code_on_k8s",
                           return_value={"status": "success", "output": "<html>nope</html>"}):
        out = execution_agent_node(_state())

    assert out["execution_result"] == {
        "status": "failed",
        "message": "Code executed successfully but no valid CSV output received",
        "output_file": None,
        "fresh_output": False,
        "output_row_count": None,
    }


# --- the readers that branch on it -----------------------------------------

def test_routing_after_a_k8s_success_reaches_visualisation():
    """route_after_execution sends an empty execution_result to the end unless
    an upload preview happens to be in state, skipping the chart and narration."""
    from app.api.workflow import route_after_execution

    _, out = _run_k8s("success", state=_state(execution_result={}))

    assert route_after_execution(out) == "visualize"


def test_routing_after_a_k8s_failure_does_not_reach_visualisation():
    """With a preview in state, an unreported failure was routed as a result."""
    from app.api.workflow import route_after_execution

    state = _state(execution_result={}, uploaded_csv_preview=[[COLUMN_A, COLUMN_B]])
    _, out = _run_k8s("failed", state=state)

    assert route_after_execution(out) == "end"


def test_a_scheduled_k8s_failure_is_recorded_as_a_failure():
    """The scheduler reads a missing status as success."""
    from app.core.celery_app import _run_status_from_final

    _, out = _run_k8s("failed", state=_state(execution_result={}))

    status, _error = _run_status_from_final(out, task_type="execute")
    assert status == "failure"


# ---------------------------------------------------------------------------
# 2. k8s and local write the same contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("outcome", ["success", "failed"])
def test_k8s_and_local_write_the_same_keys(outcome, tmp_path):
    k8s_in, k8s_out = _run("k8s", outcome, tmp_path)
    local_in, local_out = _run("local", outcome, tmp_path)

    k8s_result, local_result = k8s_out["execution_result"], local_out["execution_result"]
    assert set(k8s_result) == set(local_result)
    # returncode is the one value that cannot match: the local node runs an
    # interpreter process and has its exit code, the executor pod runs the code
    # in-process and has none. The key is there on both; on k8s it is None.
    assert {k: type(v) for k, v in k8s_result.items() if k != "returncode"} == \
           {k: type(v) for k, v in local_result.items() if k != "returncode"}
    assert k8s_result["status"] == local_result["status"]

    # ... and the same keys around it. Nothing one backend reports is missing
    # from the other.
    k8s_written = {k for k in k8s_out if k not in k8s_in or k8s_out[k] is not k8s_in[k]}
    local_written = {k for k in local_out if k not in local_in or local_out[k] is not local_in[k]}
    assert k8s_written == local_written == STATE_KEYS

    for key in ("output_file_data", "output_json", "fresh_output_produced", "execution_error"):
        assert type(k8s_out[key]) is type(local_out[key]), key


def test_k8s_and_local_agree_when_there_is_no_code():
    no_code = _state(coder_definition={})

    k8s_out = execution_agent_node(dict(no_code))
    local_out = execution_agent_node_local(dict(no_code))

    assert k8s_out["execution_result"] == local_out["execution_result"]


def test_a_success_reports_the_same_rows_on_both_backends(tmp_path):
    _, k8s_out = _run("k8s", "success", tmp_path)
    _, local_out = _run("local", "success", tmp_path)

    assert k8s_out["execution_result"]["output_row_count"] == \
           local_out["execution_result"]["output_row_count"] == 3
    assert k8s_out["output_json"] == local_out["output_json"]


# ---------------------------------------------------------------------------
# 3. The log line
# ---------------------------------------------------------------------------

_CLASS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,63}$")

# Every value the line may carry. A closed vocabulary or a number: there is no
# free-text field, so there is nowhere for a name to travel.
_ALLOWED = {
    "backend": lambda v: v in {"k8s", "local"},
    "outcome": lambda v: v in schema.OUTCOMES,
    "error_class": lambda v: bool(_CLASS_NAME.match(v)),
    "duration_ms": lambda v: v.isdigit(),
    "row_count_bucket": lambda v: v in schema.ROW_BUCKETS | {"none"},
    "column_count_bucket": lambda v: v in schema.COL_BUCKETS | {"none"},
    "fresh_output": lambda v: v in {"True", "False"},
    "output_kind": lambda v: v in {"inline", "file", "none"},
}


def _result_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == LOGGER_NAME and r.getMessage().startswith("execution_result ")]


def _fields(line: str) -> dict:
    pairs = line.split(" ")[1:]
    assert all("=" in pair for pair in pairs), line
    return dict(pair.split("=", 1) for pair in pairs)


@pytest.mark.parametrize("backend", ["k8s", "local"])
@pytest.mark.parametrize("outcome", ["success", "failed"])
def test_every_run_logs_its_result_and_nothing_else(backend, outcome, tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    _run(backend, outcome, tmp_path)

    lines = _result_lines(caplog)
    assert len(lines) == 1, f"expected one execution_result line, got {lines}"
    fields = _fields(lines[0])

    assert tuple(fields) == exec_mod.EXECUTION_RESULT_LOG_KEYS
    assert set(fields) == set(_ALLOWED)
    for key, value in fields.items():
        assert _ALLOWED[key](value), f"{key}={value!r} is outside its vocabulary"

    assert fields["backend"] == backend
    assert fields["outcome"] == outcome
    if outcome == "success":
        assert fields["error_class"] == "none"
        assert fields["row_count_bucket"] == "1-1e2"
        assert fields["column_count_bucket"] == "1-10"
        assert fields["fresh_output"] == "True"
        assert fields["output_kind"] == {"k8s": "inline", "local": "file"}[backend]
    else:
        assert fields["error_class"] == "ExecutionFailed"
        assert fields["row_count_bucket"] == "none"
        assert fields["fresh_output"] == "False"
        assert fields["output_kind"] == "none"

    for forbidden in FORBIDDEN:
        assert forbidden not in lines[0], f"{forbidden!r} leaked into the result line"


@pytest.mark.parametrize("backend", ["k8s", "local"])
def test_a_successful_run_logs_no_customer_data_at_all(backend, tmp_path, caplog):
    """Not just the result line: nothing the node itself logs on a success may
    name a column, a file, a value, the code or the question.

    The interpreter is replaced here, so this covers the node and not
    execute_code_on_k8s / execute_code_on_local, which log the code they run.
    """
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    _run(backend, "success", tmp_path)

    logged = [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME]
    assert logged
    for message in logged:
        for forbidden in FORBIDDEN:
            assert forbidden not in message, f"{forbidden!r} leaked into: {message[:80]!r}"


@pytest.mark.parametrize("backend", ["k8s", "local"])
def test_a_run_with_no_code_is_logged_too(backend, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    node = execution_agent_node if backend == "k8s" else execution_agent_node_local

    node(_state(coder_definition={}))

    fields = _fields(_result_lines(caplog)[0])
    assert (fields["backend"], fields["outcome"], fields["error_class"]) == \
           (backend, "failed", "NoCode")


def test_an_unprovisioned_cluster_is_logged_too(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    def never_provisions(state):
        state["infrastructure_provisioned"] = {"status": "failed", "details": []}
        return state

    with mock.patch.object(exec_mod, "infra_agent_node", side_effect=never_provisions):
        execution_agent_node(_state(infrastructure_provisioned=None))

    fields = _fields(_result_lines(caplog)[0])
    assert (fields["outcome"], fields["error_class"]) == ("failed", "InfrastructureUnavailable")


def test_the_log_fields_stay_off_the_never_accepted_list():
    """row_count / column_count are refused by the analytics schema as exact
    values; the line carries the bands instead."""
    assert not set(exec_mod.EXECUTION_RESULT_LOG_KEYS) & schema.NEVER_ACCEPTED


@pytest.mark.parametrize("n", [-1, 0, 1, 100, 101, 1000, 10**4 + 1, 10**8, 10**8 + 1, 10**12])
def test_row_counts_always_land_in_a_declared_band(n):
    assert schema.row_count_bucket(n) in schema.ROW_BUCKETS


@pytest.mark.parametrize("n", [-1, 0, 1, 10, 11, 50, 51, 200, 201, 1000, 1001, 10**6])
def test_column_counts_always_land_in_a_declared_band(n):
    assert schema.column_count_bucket(n) in schema.COL_BUCKETS


# ---------------------------------------------------------------------------
# 4. What a node writes has to be declared, or the graph throws it away
# ---------------------------------------------------------------------------

def _graph_with(node):
    """The real ETLState in a real LangGraph: `node`, then a node that records
    what it was handed."""
    from langgraph.graph import END, StateGraph
    from app.graph.etl_state import ETLState

    seen = {}

    def downstream(state):
        seen.update(state)
        return {}

    builder = StateGraph(ETLState)
    builder.add_node("execute", node)
    builder.add_node("downstream", downstream)
    builder.set_entry_point("execute")
    builder.add_edge("execute", "downstream")
    builder.add_edge("downstream", END)
    return builder.compile(), seen


def test_langgraph_drops_a_key_that_etlstate_does_not_declare():
    """The premise, measured rather than assumed. If a LangGraph upgrade ever
    stops dropping undeclared keys this fails, and the declarations below stop
    being load-bearing."""
    graph, seen = _graph_with(lambda state: {"execution_result": {"status": "success"},
                                             "zq_not_a_declared_channel": True})

    final = graph.invoke({"messages": []})

    assert final["execution_result"] == {"status": "success"}
    assert "zq_not_a_declared_channel" not in seen
    assert "zq_not_a_declared_channel" not in final


@pytest.mark.parametrize("backend", ["k8s", "local"])
@pytest.mark.parametrize("outcome", ["success", "failed"])
def test_every_key_an_execution_node_writes_is_a_declared_channel(backend, outcome, tmp_path):
    from app.graph.etl_state import ETLState

    state_in, out = _run(backend, outcome, tmp_path)

    assert set(state_in) <= set(ETLState.__annotations__), "the fixture itself is undeclared"
    undeclared = set(out) - set(ETLState.__annotations__)
    assert not undeclared, (
        f"{sorted(undeclared)} written by the {backend} node but not declared in "
        "ETLState: LangGraph will drop them and every reader will see nothing"
    )


@pytest.mark.parametrize("outcome", ["success", "failed"])
def test_what_the_k8s_node_writes_reaches_the_next_node(outcome):
    """End to end through LangGraph, not by reading the node's return value."""
    graph, seen = _graph_with(execution_agent_node)

    with mock.patch.object(exec_mod, "execute_code_on_k8s", return_value=_k8s_response(outcome)):
        final = graph.invoke(_state())

    for key in STATE_KEYS:
        assert key in seen, f"{key} did not reach the node after execution"
        assert key in final, f"{key} is missing from the final state"
    assert seen["fresh_output_produced"] is (outcome == "success")
    assert (seen["execution_error"] is None) is (outcome == "success")
    assert seen["execution_result"]["status"] == outcome


def test_a_stale_execution_error_does_not_outlive_a_successful_run():
    """execution_error is checkpointed now, so a success has to clear it or the
    router would end a good run on the last turn's failure."""
    from app.api.workflow import route_after_execution

    graph, seen = _graph_with(execution_agent_node)
    stale = _state(execution_error="left over from the previous turn")

    with mock.patch.object(exec_mod, "execute_code_on_k8s", return_value=_k8s_response("success")):
        graph.invoke(stale)

    assert seen["execution_error"] is None
    assert route_after_execution(seen) == "visualize"
