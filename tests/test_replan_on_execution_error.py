"""A failed analysis run is diagnosed, repaired and run again, or explained plainly.

Every case here goes through the graph ``build_graph()`` registers, from the
point the coder has handed over its code: the real ``execute_locally`` (a real
subprocess run of the code on a fixture file), the real routers, the real
claim verifier. Only the coder (for a code fix), the chart and the narrator are
stubbed: no model is called, and nothing is spent.

Before this, a failed run went straight to the verifier and the turn ended
with "I couldn't complete this step"; each test fails on that base by an
assertion, not by an import.

Fixtures (tests/fixtures/replan/, synthetic):

* prices.csv          -- 40 of 200 prices are currency text ("$1,234.50")
* amounts.csv         -- 25 of 200 amounts carry thousands separators
* revenue.csv         -- 12 of 300 revenues are blank
* revenue_sparse.csv  -- 114 of 300 revenues are blank (38%)
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

FIXTURES = Path(__file__).parent / "fixtures" / "replan"

PRICE_CODE = (
    "import pandas as pd\n"
    "def main(df):\n"
    "    df['price'] = df['price'].astype(float)\n"
    "    return df.groupby('region', as_index=False)['price'].mean()\n"
)
AMOUNT_CODE = (
    "import pandas as pd\n"
    "def main(df):\n"
    "    df['net'] = df['amount'] - df['cost']\n"
    "    return df.groupby('region', as_index=False)['net'].sum()\n"
)
BAND_CODE = (
    "import pandas as pd\n"
    "def main(df):\n"
    "    df['revenue_band'] = (df['revenue'].astype(int) // 500) * 500\n"
    "    return df.groupby('revenue_band', as_index=False)['units'].sum()\n"
)
TOTAL_CODE = (
    "import pandas as pd\n"
    "def main(df):\n"
    "    total = int(df['revenue'].astype(int).sum())\n"
    "    return pd.DataFrame({'total_revenue': [total]})\n"
)
TYPO_CODE = (
    "import pandas as pd\n"
    "def main(df):\n"
    "    return df.groupby('region', as_index=False)['revnue'].sum()\n"
)
FIXED_CODE = (
    "import pandas as pd\n"
    "def main(df):\n"
    "    return df.groupby('region', as_index=False)['units'].sum()\n"
)


def _crashing(n: int) -> str:
    return (
        "import pandas as pd\n"
        "def main(df):\n"
        f"    return df.assign(ratio=1 / (0 * {n + 1}))\n"
    )


TRACEBACK_MARKERS = ("Traceback", 'File "', "ZeroDivisionError", "division by zero", "KeyError",
                     "IntCastingNaNError", "ValueError", "TypeError", "exit status")


# --------------------------------------------------------------------------- #
# Driving the registered graph
# --------------------------------------------------------------------------- #

@pytest.fixture
def graph(monkeypatch):
    """The real graph with the chart and narrator stubbed and no models configured."""
    from langgraph.checkpoint.memory import MemorySaver

    from app.agents import validator as validator_mod
    from app.api import reply_segments as rs
    from app.api import workflow

    monkeypatch.setattr(workflow, "visualization_agent_node", lambda state: {})
    monkeypatch.setattr(workflow, "result_narrator_node",
                        lambda state: {"messages": [rs.commentary_message(content="Here is the result.")]})
    monkeypatch.setattr(workflow, "coding_graph", None)
    monkeypatch.setattr(workflow, "plan_metric_actions", lambda *a, **k: None)
    monkeypatch.setattr(validator_mod, "validator_llm", None)
    for name in ("AVALOKA_REPLAN_MAX_ATTEMPTS", "AVALOKA_REPLAN_MEASURED_POLICY",
                 "AVALOKA_STRICT_LOGIC_VALIDATION"):
        monkeypatch.delenv(name, raising=False)
    return workflow, workflow.build_graph().compile(checkpointer=MemorySaver())


def _coder(monkeypatch, workflow, codes):
    """Stub the coder: the i-th call returns codes[i] (the last one repeats)."""
    calls = []

    def fake_coder(state):
        calls.append(dict(state))
        code = codes[min(len(calls) - 1, len(codes) - 1)]
        return {"generated_code": code, "retry_count": (state.get("retry_count") or 0) + 1,
                "llm_raw_response": "stub"}

    monkeypatch.setattr(workflow, "coder_node", fake_coder)
    # build_graph() builds the coding subgraph eagerly; rebuild it on the stub.
    monkeypatch.setattr(workflow, "coding_graph", None)
    return calls


def _run(graph, tmp_path, fixture: str, code: str, question: str):
    workflow, compiled = graph
    data = tmp_path / fixture
    shutil.copy(FIXTURES / fixture, data)
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    compiled.update_state(config, {
        "messages": [HumanMessage(content=question)],
        "user_prompt": question,
        "coder_definition": {"code": code},
        "data_source_location": str(data),
        "output_location": str(tmp_path / "out" / "output.csv"),
        "execution_mode": "local",
        "input_data_type": "csv",
        "agent_errors": [],
    }, as_node="code_etl")
    compiled.invoke(None, config)
    final = compiled.get_state(config).values
    return final, data


def _reply(final) -> str:
    """The reply the user reads: the last assistant message as the server presents it."""
    from app.agents.claim_verifier import presentable_reply

    last = next(m for m in reversed(final["messages"]) if isinstance(m, AIMessage))
    return presentable_reply(last.content, final.get("verification_report")) or last.content


def _rows(final):
    from app.agents.analysis_outcome import result_records

    return result_records(final) or []


def _attempts(final):
    return final.get("replan_attempts") or []


def _run_truth(final):
    from app.core.run_truth import classify_turn_with_recovery

    return classify_turn_with_recovery(final)


# --------------------------------------------------------------------------- #
# Data errors: repaired, re-run, disclosed
# --------------------------------------------------------------------------- #

def test_value_error_on_currency_text_is_repaired_and_disclosed(graph, tmp_path):
    final, data = _run(graph, tmp_path, "prices.csv", PRICE_CODE, "Average price by region?")
    reply = _reply(final)
    assert "40 values in price were numbers stored as text" in reply, reply
    assert "I read them as numbers" in reply

    raw = pd.read_csv(data, dtype=str)
    expected = (raw.assign(price=raw["price"].str.replace("[$,]", "", regex=True).astype(float))
                .groupby("region")["price"].mean())
    got = {r["region"]: float(r["price"]) for r in _rows(final)}
    assert got == pytest.approx(expected.to_dict())
    assert (FIXTURES / "prices.csv").read_bytes() == data.read_bytes(), "the source file was changed"

    truth = _run_truth(final)
    assert truth["outcome"] == "succeeded"
    [attempt] = truth["recovery"]["attempts"]
    assert attempt["error_class"] == "ValueError"
    assert (attempt["diagnosis"], attempt["diagnosed_by"], attempt["action"], attempt["outcome"]) == (
        "dtype_mismatch", "rule", "data_step", "recovered")
    assert attempt["transforms"][0]["op"] == "coerce_numeric"
    assert attempt["transforms"][0]["values_converted"] == 40
    assert truth["recovery"]["final"] == "recovered"


def test_type_error_on_a_text_column_is_repaired_and_disclosed(graph, tmp_path):
    final, data = _run(graph, tmp_path, "amounts.csv", AMOUNT_CODE, "Net amount by region?")
    reply = _reply(final)
    assert "25 values in amount were numbers stored as text" in reply, reply

    raw = pd.read_csv(data, dtype=str)
    amount = raw["amount"].str.replace(",", "", regex=False).astype(float)
    expected = (amount - raw["cost"].astype(float)).groupby(raw["region"]).sum()
    got = {r["region"]: float(r["net"]) for r in _rows(final)}
    assert got == pytest.approx(expected.to_dict())

    [attempt] = _run_truth(final)["recovery"]["attempts"]
    assert (attempt["error_class"], attempt["diagnosis"], attempt["outcome"]) == (
        "TypeError", "dtype_mismatch", "recovered")


def test_nans_in_a_numeric_column_are_imputed_with_the_median_and_disclosed(graph, tmp_path):
    final, data = _run(graph, tmp_path, "revenue.csv", BAND_CODE, "Units by revenue band?")
    raw = pd.read_csv(data)
    median = float(raw["revenue"].median())
    reply = _reply(final)
    assert (f"12 of 300 rows had no value in revenue; I filled them with the median ({median:,.2f})"
            in reply), reply

    filled = raw.assign(revenue=raw["revenue"].fillna(median))
    expected = filled.assign(revenue_band=(filled["revenue"].astype(int) // 500) * 500) \
        .groupby("revenue_band")["units"].sum()
    got = {int(float(r["revenue_band"])): int(float(r["units"])) for r in _rows(final)}
    assert got == expected.to_dict()

    [attempt] = _run_truth(final)["recovery"]["attempts"]
    assert attempt["error_class"] == "IntCastingNaNError"
    assert attempt["diagnosis"] == "missing_values" and attempt["outcome"] == "recovered"
    assert attempt["transforms"] == [{"op": "impute", "column": "revenue", "rows_total": 300,
                                      "rows_affected": 12, "method": "median", "value": median}]


# --------------------------------------------------------------------------- #
# The measured column. The shipped behaviour is to ASK (the owner's ruling;
# tests/test_replan_measured_choice.py). "exclude" and "impute" below are the
# AVALOKA_REPLAN_MEASURED_POLICY test hooks, not settings.
# --------------------------------------------------------------------------- #

def test_measured_column_exclude_hook_leaves_the_rows_out_and_says_so(graph, tmp_path, monkeypatch):
    monkeypatch.setenv("AVALOKA_REPLAN_MEASURED_POLICY", "exclude")
    final, data = _run(graph, tmp_path, "revenue.csv", TOTAL_CODE, "Total revenue?")
    reply = _reply(final)
    assert "I left out 12 of 300 rows (4.0%) that had no value in revenue" in reply, reply
    raw = pd.read_csv(data)
    assert [int(float(r["total_revenue"])) for r in _rows(final)] == [int(raw["revenue"].dropna().sum())]

    truth = _run_truth(final)
    assert (truth["outcome"], truth["reason_code"]) == ("partial", "rows_excluded")
    [attempt] = truth["recovery"]["attempts"]
    assert attempt["transforms"][0]["op"] == "exclude_rows"
    assert attempt["transforms"][0]["rows_affected"] == 12


def test_measured_column_impute_hook_fills_the_median_and_says_so(graph, tmp_path, monkeypatch):
    monkeypatch.setenv("AVALOKA_REPLAN_MEASURED_POLICY", "impute")
    final, data = _run(graph, tmp_path, "revenue.csv", TOTAL_CODE, "Total revenue?")
    raw = pd.read_csv(data)
    median = float(raw["revenue"].median())
    reply = _reply(final)
    assert f"12 of 300 rows had no value in revenue; I filled them with the median ({median:,.2f})" in reply
    assert [int(float(r["total_revenue"])) for r in _rows(final)] == [
        int(raw["revenue"].fillna(median).astype(int).sum())]
    assert _run_truth(final)["recovery"]["attempts"][0]["transforms"][0]["method"] == "median"


# --------------------------------------------------------------------------- #
# Over a limit: nothing applied, the user is asked with the counts
# --------------------------------------------------------------------------- #

def test_over_the_imputation_limit_stops_and_asks(graph, tmp_path):
    final, data = _run(graph, tmp_path, "revenue_sparse.csv", BAND_CODE, "Units by revenue band?")
    reply = _reply(final)
    assert "114 of 300 rows have no value in revenue, more than the 10% I fill in without asking" in reply, reply
    assert "Should I fill them with the median, or leave them out?" in reply
    assert "Nothing has been filled or left out yet" in reply
    assert not final.get("replan_transforms") and not final.get("replan_input_overrides")
    assert final.get("replan_final") == "needs_user"
    for marker in TRACEBACK_MARKERS:
        assert marker not in reply
    truth = _run_truth(final)
    assert (truth["outcome"], truth["reason_code"]) == ("failed", "needs_decision")
    assert truth["recovery"]["attempts"] == []


def test_over_the_exclusion_limit_stops_and_asks(graph, tmp_path, monkeypatch):
    # The exclude hook sends the measured column to the exclusion limit.
    monkeypatch.setenv("AVALOKA_REPLAN_MEASURED_POLICY", "exclude")
    final, _ = _run(graph, tmp_path, "revenue_sparse.csv", TOTAL_CODE, "Total revenue?")
    reply = _reply(final)
    assert "114 of 300 rows have no value in revenue" in reply, reply
    assert "leaving them out would drop more than the 5% of rows" in reply
    assert final.get("replan_final") == "needs_user"


# --------------------------------------------------------------------------- #
# Code errors: fixed by the coder, with the error as feedback
# --------------------------------------------------------------------------- #

def test_a_code_error_is_fixed_by_the_coder(graph, tmp_path, monkeypatch):
    workflow, _ = graph
    calls = _coder(monkeypatch, workflow, [FIXED_CODE])
    final, data = _run(graph, tmp_path, "revenue.csv", TYPO_CODE, "Units by region?")
    raw = pd.read_csv(data)
    got = {r["region"]: int(float(r["units"])) for r in _rows(final)}
    assert got == raw.groupby("region")["units"].sum().to_dict()
    assert len(calls) == 1
    feedback = calls[0].get("code_validation_feedback") or ""
    assert "KeyError" in feedback and "units" in feedback and "revenue" in feedback
    assert "revnue" in (calls[0].get("generated_code") or "")
    [attempt] = _run_truth(final)["recovery"]["attempts"]
    assert (attempt["diagnosis"], attempt["action"], attempt["outcome"]) == (
        "column_reference", "code_fix", "recovered")


# --------------------------------------------------------------------------- #
# Unrecoverable: every attempt fails, the reply is plain
# --------------------------------------------------------------------------- #

def test_exhausted_attempts_end_in_a_plain_explanation(graph, tmp_path, monkeypatch):
    workflow, _ = graph
    calls = _coder(monkeypatch, workflow, [_crashing(1), _crashing(2), _crashing(3), _crashing(4)])
    final, _ = _run(graph, tmp_path, "revenue.csv", _crashing(0), "Units per region ratio?")
    reply = _reply(final)
    assert reply.startswith("I couldn't finish this analysis. I tried 3 times:"), reply
    assert "1. The calculation stopped because there was an error in my code." in reply
    assert "3. It then stopped because" in reply
    assert "That was my last attempt" in reply
    for marker in TRACEBACK_MARKERS:
        assert marker not in reply, marker
    assert not re.search(r"\bline \d+", reply)
    assert len(calls) == 3

    truth = _run_truth(final)
    assert (truth["outcome"], truth["reason_code"]) == ("failed", "recovery_exhausted")
    attempts = truth["recovery"]["attempts"]
    assert [a["n"] for a in attempts] == [1, 2, 3]
    assert all(a["error_class"] == "ZeroDivisionError" and a["action"] == "code_fix"
               and a["outcome"] == "failed_again" for a in attempts)
    assert truth["recovery"]["final"] == "exhausted"
    assert truth["recovery"]["stopped"]["reason"] == "attempts"


def test_an_unchanged_rewrite_stops_early(graph, tmp_path, monkeypatch):
    workflow, _ = graph
    calls = _coder(monkeypatch, workflow, [_crashing(0)])
    final, _ = _run(graph, tmp_path, "revenue.csv", _crashing(0), "Units per region ratio?")
    assert len(calls) == 1
    recovery = _run_truth(final)["recovery"]
    assert recovery["stopped"]["reason"] == "code_unchanged"
    assert len(recovery["attempts"]) == 1
    assert "came back unchanged" in _reply(final)


def test_an_error_that_comes_back_stops_early(graph, tmp_path, monkeypatch):
    workflow, _ = graph
    # KeyError -> ZeroDivisionError -> KeyError again: going round in circles.
    typo_again = TYPO_CODE.replace("return", "# again\n    return")
    calls = _coder(monkeypatch, workflow, [_crashing(1), typo_again, FIXED_CODE])
    final, _ = _run(graph, tmp_path, "revenue.csv", TYPO_CODE, "Units by region?")
    assert len(calls) == 2
    recovery = _run_truth(final)["recovery"]
    assert recovery["stopped"]["reason"] == "oscillating"
    assert [a["diagnosis"] for a in recovery["attempts"]] == ["column_reference", "code_bug"]
    assert "went back to an error" in _reply(final)


# --------------------------------------------------------------------------- #
# Telemetry: no column name or value leaves the process
# --------------------------------------------------------------------------- #

PLANTED_COLUMN = "zq_planted_turnover"


def test_no_column_name_or_value_reaches_telemetry(graph, tmp_path):
    planted = tmp_path / "planted.csv"
    raw = pd.read_csv(FIXTURES / "revenue.csv").rename(columns={"revenue": PLANTED_COLUMN})
    raw.to_csv(planted, index=False)
    code = BAND_CODE.replace("'revenue'", f"'{PLANTED_COLUMN}'")
    workflow, compiled = graph
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    compiled.update_state(config, {
        "messages": [HumanMessage(content="Units by band?")], "user_prompt": "Units by band?",
        "coder_definition": {"code": code}, "data_source_location": str(planted),
        "output_location": str(tmp_path / "out" / "output.csv"), "execution_mode": "local",
        "input_data_type": "csv", "agent_errors": [],
    }, as_node="code_etl")
    compiled.invoke(None, config)
    final = compiled.get_state(config).values

    # Not vacuous: the user's own record and reply do name the column.
    assert _attempts(final), "no replan attempt was recorded"
    assert PLANTED_COLUMN in _reply(final)
    median = float(raw[PLANTED_COLUMN].median())

    from app.agents.replan import recovery_block, recovery_telemetry
    from app.trace.telemetry import agent_events

    recovery = recovery_block(final)
    assert recovery["attempts"][0]["transforms"][0]["column"] == PLANTED_COLUMN
    wire = json.dumps({"recovery": recovery_telemetry(recovery),
                       "events": agent_events(final.get("agent_trace") or [])}, default=str)
    assert PLANTED_COLUMN not in wire
    assert f"{median:,.2f}" not in wire and str(median) not in wire
    assert "planted.csv" not in wire
    assert json.loads(wire)["recovery"]["attempts"][0]["ops"] == ["impute"]


# --------------------------------------------------------------------------- #
# Scope: local only; Ray, Kubernetes and scheduled runs are not routed here
# --------------------------------------------------------------------------- #

def test_only_the_local_run_is_routed_through_replan(graph):
    workflow, compiled = graph
    edges = {(e.source, e.target) for e in compiled.get_graph().edges}
    assert ("execute_locally", "replan") in edges
    assert ("replan", "execute_locally") in edges and ("replan", "code_etl") in edges
    for node in ("execute_on_ray", "execute_on_k8s", "schedule_task", "execute_actions"):
        assert (node, "replan") not in edges, node
    # The shared router keeps its old behaviour for Ray and Kubernetes.
    failed = {"execution_result": {"status": "failed"}, "execution_error": "x"}
    assert workflow.route_after_execution(failed) == "end"


def test_the_loop_can_be_turned_off(graph, tmp_path, monkeypatch):
    monkeypatch.setenv("AVALOKA_REPLAN_MAX_ATTEMPTS", "0")
    final, _ = _run(graph, tmp_path, "prices.csv", PRICE_CODE, "Average price by region?")
    assert final["execution_result"]["status"] == "failed"
    assert not _attempts(final)
    assert "couldn't complete this step" in _reply(final)
