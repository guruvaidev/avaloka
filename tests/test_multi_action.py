"""Structured planning and deterministic multi-metric execution."""

import json

import pandas as pd

import app.agents.multi_action as multi
import app.api.workflow as workflow


class PlanModel:
    def __init__(self, payload, review=None):
        self.payload = payload
        self.review = review if review is not None else {"complete": True, "missing": []}
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        payload = self.payload if len(self.calls) == 1 else self.review
        return type("Reply", (), {"content": json.dumps(payload)})()


def _plan(actions):
    return multi.ActionPlan(
        mode="aggregate",
        actions=[
            multi.MetricAction(
                id=f"metric_{index}", operation=item[0], column=item[1],
                group_by=item[2], output_name=f"{item[0]}_{item[1] or 'records'}",
            )
            for index, item in enumerate(actions, 1)
        ],
    )


def test_model_plan_uses_schema_without_prompt_word_matching():
    model = PlanModel({
        "mode": "aggregate",
        "actions": [
            {"operation": "mean", "column": "sales", "group_by": []},
            {"operation": "median", "column": "sales", "group_by": []},
        ],
    })
    plan = multi.plan_metric_actions("Give me both central figures for sales", {"sales": "float"}, model)
    assert [action.output_name for action in plan.actions] == ["mean_sales", "median_sales"]
    assert len(model.calls) == 2


def test_incomplete_plan_review_falls_back_to_coding_agent():
    model = PlanModel({
        "mode": "aggregate",
        "actions": [
            {"operation": "mean", "column": "sales", "group_by": []},
            {"operation": "median", "column": "sales", "group_by": []},
        ],
    }, review={"complete": False, "missing": ["regional breakdown"]})
    assert multi.plan_metric_actions("Mean and median sales by region", {"sales": "float", "region": "text"}, model) is None


def test_unknown_column_asks_for_clarification():
    model = PlanModel({
        "mode": "aggregate",
        "actions": [
            {"operation": "mean", "column": "sales", "group_by": []},
            {"operation": "median", "column": "unknown", "group_by": []},
        ],
    })
    plan = multi.plan_metric_actions("Two measures", {"sales": "float"}, model)
    assert plan.mode == "clarify"
    assert "unknown" in plan.clarification


def test_complex_plan_falls_back_to_coding_agent():
    model = PlanModel({"mode": "code", "actions": []})
    assert multi.plan_metric_actions("Sales after a date filter", {"sales": "float"}, model) is None


def test_same_grouping_returns_one_table_with_both_metrics():
    frame = pd.DataFrame({"sales": [10, 20, 30, 40]})
    result = multi.execute_action_plan(frame, _plan([
        ("mean", "sales", []), ("median", "sales", []),
    ]))
    assert result.status == "complete"
    assert result.tables == [{
        "title": "Mean and median of sales",
        "rows": [{"mean_sales": 25.0, "median_sales": 25.0}],
        "action_ids": ["metric_1", "metric_2"],
        "group_by": [],
        "metric_columns": ["mean_sales", "median_sales"],
    }]


def test_different_groupings_return_two_typed_tables():
    frame = pd.DataFrame({
        "region": ["East", "East", "West", "West"],
        "product": ["A", "B", "A", "B"],
        "sales": [10, 20, 30, 40],
    })
    result = multi.execute_action_plan(frame, _plan([
        ("mean", "sales", ["region"]), ("median", "sales", ["product"]),
    ]))
    assert result.status == "complete"
    assert result.tables[0]["title"] == "Mean of sales by region"
    assert result.tables[0]["group_by"] == ["region"]
    assert result.tables[0]["metric_columns"] == ["mean_sales"]
    assert result.tables[0]["rows"] == [
        {"region": "East", "mean_sales": 15.0}, {"region": "West", "mean_sales": 35.0},
    ]
    assert result.tables[1]["title"] == "Median of sales by product"
    assert result.tables[1]["group_by"] == ["product"]
    assert result.tables[1]["metric_columns"] == ["median_sales"]
    assert result.tables[1]["rows"] == [
        {"product": "A", "median_sales": 20.0}, {"product": "B", "median_sales": 30.0},
    ]
    assert not any("__result_table" in row for table in result.tables for row in table["rows"])


def test_different_source_fields_keep_separate_grouped_tables():
    frame = pd.DataFrame({"region": ["East", "East", "West"], "product": ["A", "B", "A"]})
    result = multi.execute_action_plan(frame, _plan([
        ("count", None, ["region"]), ("count_distinct", "product", ["region"]),
    ]))
    assert len(result.tables) == 2
    assert result.tables[0]["rows"] == [
        {"region": "East", "count_records": 2},
        {"region": "West", "count_records": 1},
    ]
    assert result.tables[1]["rows"] == [
        {"region": "East", "count_distinct_product": 2},
        {"region": "West", "count_distinct_product": 1},
    ]


def test_three_different_measures_produce_three_tables():
    frame = pd.DataFrame({
        "gross_revenue": [100, 200], "net_revenue": [80, 160], "age": [20, 30],
    })
    result = multi.execute_action_plan(frame, _plan([
        ("mean", "gross_revenue", []),
        ("median", "net_revenue", []),
        ("mean", "age", []),
    ]))
    assert result.status == "complete"
    assert [table["title"] for table in result.tables] == [
        "Mean of gross revenue", "Median of net revenue", "Mean of age",
    ]
    assert [table["rows"] for table in result.tables] == [
        [{"mean_gross_revenue": 150.0}],
        [{"median_net_revenue": 120.0}],
        [{"mean_age": 25.0}],
    ]


def test_partial_result_keeps_success_and_reports_dirty_metric():
    frame = pd.DataFrame({"sales": [10, 20], "profit": ["bad", "worse"]})
    result = multi.execute_action_plan(frame, _plan([
        ("mean", "sales", []), ("median", "profit", []),
    ]))
    assert result.status == "incomplete"
    assert result.tables[0]["rows"] == [{"mean_sales": 15.0}]
    assert result.missing[0]["id"] == "metric_2"
    assert "median of profit" in result.missing[0]["label"]


def test_formatted_numbers_are_parsed_and_disclosed():
    frame = pd.DataFrame({"sales": ["$1,000", "$2,000"]})
    result = multi.execute_action_plan(frame, _plan([
        ("mean", "sales", []), ("median", "sales", []),
    ]))
    assert result.status == "complete"
    assert result.tables[0]["rows"] == [{"mean_sales": 1500.0, "median_sales": 1500.0}]
    assert len(result.notes) == 2


def test_ambiguous_decimal_separator_is_reported_instead_of_guessed():
    frame = pd.DataFrame({"sales": ["1,5", "2,5"]})
    result = multi.execute_action_plan(frame, _plan([
        ("mean", "sales", []), ("median", "sales", []),
    ]))
    assert result.status == "incomplete"
    assert result.tables == []
    assert len(result.missing) == 2


def test_failed_metric_gets_three_distinct_conversion_approaches(monkeypatch):
    calls = []

    def fail(_frame, _action, approach):
        calls.append(approach)
        raise ValueError("dirty values")

    monkeypatch.setattr(multi, "_calculate", fail)
    result = multi.execute_action_plan(pd.DataFrame({"sales": [1]}), _plan([("mean", "sales", [])]))
    assert calls == [1, 2, 3]
    assert result.status == "incomplete"


def test_structured_route_and_node_return_separate_tables(tmp_path, monkeypatch):
    source = tmp_path / "sales.csv"
    source.write_text("region,product,sales\nEast,A,10\nEast,B,20\nWest,A,30\nWest,B,40\n")
    plan = _plan([("mean", "sales", ["region"]), ("median", "sales", ["product"])])
    monkeypatch.setattr(workflow, "plan_metric_actions", lambda *_: plan)
    monkeypatch.setattr(workflow, "_detect_ambiguous_prompt_details", lambda _: None)
    state = {
        "user_prompt": "Mean sales by region and median sales by product",
        "messages": [],
        "schema": {"region": "object", "product": "object", "sales": "int64"},
        "input_data_type": "csv",
        "data_source_location": str(source),
        "output_location": str(tmp_path / "output.csv"),
    }
    coded = workflow.coding_subgraph_node(state)
    assert coded["structured_action_plan"]
    assert workflow.route_after_code(state | coded) == "execute_actions"
    output = workflow.execute_structured_actions_node(state | coded)
    assert output["multi_action_status"] == "complete"
    assert len(output["output_tables"]) == 2
    assert output["output_file_data"] is None
    assert output["output_json"] == output["output_tables"][0]["rows"]


def test_cloud_and_oversized_inputs_keep_existing_execution_path(tmp_path, monkeypatch):
    source = tmp_path / "sales.csv"
    source.write_text("sales\n10\n20\n")
    state = {"data_source_location": str(source), "input_data_type": "csv"}
    assert workflow._structured_source(state | {"execution_mode": "k8s-ray"}) is None
    monkeypatch.setattr(workflow, "STRUCTURED_ACTION_MAX_BYTES", 1)
    assert workflow._structured_source(state) is None


def test_langgraph_preserves_typed_tables(tmp_path, monkeypatch):
    from langgraph.graph import END, StateGraph
    from app.graph.etl_state import ETLState

    source = tmp_path / "sales.csv"
    source.write_text("sales\n10\n20\n")
    monkeypatch.setattr(workflow, "plan_metric_actions", lambda *_: _plan([
        ("mean", "sales", []), ("median", "sales", []),
    ]))
    monkeypatch.setattr(workflow, "_detect_ambiguous_prompt_details", lambda _: None)
    graph = StateGraph(ETLState)
    graph.add_node("code", workflow.coding_subgraph_node)
    graph.add_node("execute", workflow.execute_structured_actions_node)
    graph.set_entry_point("code")
    graph.add_conditional_edges("code", workflow.route_after_code,
                                {"execute_actions": "execute", "end": END})
    graph.add_edge("execute", END)
    final = graph.compile().invoke({
        "user_prompt": "mean and median sales", "messages": [],
        "schema": {"sales": "int64"}, "input_data_type": "csv",
        "data_source_location": str(source),
    })
    assert final["output_tables"][0]["rows"] == [{"mean_sales": 15.0, "median_sales": 15.0}]
    assert final["multi_action_status"] == "complete"


def test_unsupported_request_keeps_existing_coding_path(tmp_path, monkeypatch):
    source = tmp_path / "sales.csv"
    source.write_text("sales\n10\n20\n")

    class CodingGraph:
        def invoke(self, _state):
            return {"generated_code": "def main(df): return df", "llm_raw_response": "code ready"}

    monkeypatch.setattr(workflow, "plan_metric_actions", lambda *_: None)
    monkeypatch.setattr(workflow, "_detect_ambiguous_prompt_details", lambda _: None)
    monkeypatch.setattr(workflow, "coding_graph", CodingGraph())
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda _: b"")
    result = workflow.coding_subgraph_node({
        "user_prompt": "Filter sales then rank products", "messages": [],
        "schema": {"sales": "int64"}, "input_data_type": "csv",
        "data_source_location": str(source),
    })
    assert result["structured_action_plan"] is None
    assert result["coder_definition"]["code"] == "def main(df): return df"


def test_structured_output_survives_api_response_serialization():
    from app.api.schemas import ChatResponse
    from app.api.server import get_output_from_state

    tables = [{
        "title": "By region", "rows": [{"region": "East", "mean_sales": 25}],
        "action_ids": ["metric_1"], "group_by": ["region"],
        "metric_columns": ["mean_sales"],
    }]
    state = {
        "structured_action_plan": {"mode": "aggregate"},
        "output_tables": tables,
        "output_json": tables[0]["rows"],
        "output_file_data": None,
    }
    file_data, rows = get_output_from_state(state)
    response = ChatResponse(messages=[], output_file_data=file_data, output_json=rows,
                            output_tables=tables, multi_action_status="complete",
                            multi_action_actions=[{"id": "metric_1", "operation": "mean", "column": "sales", "group_by": ["region"]}])
    assert response.model_dump()["output_tables"] == tables
    assert response.multi_action_actions[0]["id"] == "metric_1"
    assert response.output_json == [{"region": "East", "mean_sales": 25}]
