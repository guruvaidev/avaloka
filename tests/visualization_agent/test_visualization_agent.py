
from tests._quarantine import requires_api

requires_api("app.agents.visualization_agent", "VisualizationAgent", replacement="the VisualizationAgent class was replaced by the visualization_agent_node function; repair against the node API or remove this file.")

import sys
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.visualization_agent import VisualizationAgent, visualization_agent_node
from app.graph.etl_state import ETLState


@pytest.fixture(autouse=True)
def disable_viz_llm(monkeypatch):
    """Ensure tests do not invoke the external LLM."""
    monkeypatch.setattr("app.agents.visualization_agent.viz_llm", None)


@pytest.fixture
def agent():
    return VisualizationAgent()


def _base_state(**overrides) -> ETLState:
    state: ETLState = {
        "messages": [HumanMessage(content="Generate a quick visualization")],
        "planner_definition": {"job_name": "default_job", "transformations": []},
        "ready_to_summarize": True,
        "ready_to_code": True,
        "coder_definition": {},
        "data_source_location": "/tmp/source.csv",
        "output_location": "/tmp/output.csv",
        "schema": {},
        "infrastructure_request": {},
        "infrastructure_provisioned": {},
        "execution_result": {},
        "input_data_type": "csv",
        "output_file_data": {},
        "deploy_on_k8s": False,
        "uploaded_csv_preview": [],
        "uploaded_csv_columns": [],
        "planner_graph_path": None,
        "planner_graph_base64": None,
        "planner_graph_status": None,
        "planner_graph_error": None,
        "generated_code": None,
        "coder_pseudocode": None,
        "execution_output_data": None,
        "execution_output_preview": None,
        "visualization_config": None,
        "visualization_status": None,
    }
    state.update(overrides)
    return state


def test_generate_bar_chart_metadata(agent):
    sample_data = [
        {"Region": "North", "Sales": 25000},
        {"Region": "South", "Sales": 18000},
        {"Region": "East", "Sales": 32000},
    ]
    result = agent.generate_visualizations(
        planner_def={"job_name": "sales", "transformations": []},
        execution_result={},
        sample_data=sample_data,
    )

    bar_chart = next((chart for chart in result["charts"] if chart["type"] == "bar"), None)
    assert result["status"] == "success"
    assert result["metadata"]["total_charts"] == len(result["charts"])
    assert result["metadata"]["data_rows"] == len(sample_data)
    assert bar_chart is not None
    assert bar_chart["config"]["xAxis"]["data"]
    assert bar_chart["config"]["series"][0]["data"]


def test_scatter_chart_contains_numeric_points(agent):
    sample_data = [
        {"Price": 10.5, "Quantity": 100},
        {"Price": 15.0, "Quantity": 85},
        {"Price": 8.25, "Quantity": 120},
        {"Price": 12.0, "Quantity": 95},
    ]
    result = agent.generate_visualizations(
        planner_def={"job_name": "correlations", "transformations": []},
        execution_result={},
        sample_data=sample_data,
    )

    scatter_chart = next((chart for chart in result["charts"] if chart["type"] == "scatter"), None)
    assert scatter_chart is not None
    assert scatter_chart["config"]["series"][0]["data"]
    assert all(len(point) == 2 for point in scatter_chart["config"]["series"][0]["data"])


def test_execution_result_rows_used_for_metadata(agent):
    sample_data = [{"Region": "A", "Sales": 10}]
    execution_output = [
        {"Region": "North", "Sales": 100},
        {"Region": "South", "Sales": 80},
        {"Region": "East", "Sales": 120},
        {"Region": "West", "Sales": 90},
    ]
    result = agent.generate_visualizations(
        planner_def={"job_name": "sales", "transformations": []},
        execution_result={"output_data": execution_output},
        sample_data=sample_data,
    )

    assert result["metadata"]["data_rows"] == len(execution_output)


def test_visualization_agent_node_converts_preview():
    preview = [
        ["Category", "Amount"],
        ["Hardware", "100"],
        ["Software", "200"],
        ["Services", "150"],
    ]
    state = _base_state(
        planner_definition={
            "job_name": "category_sales",
            "transformations": [{"type": "aggregation", "columns": ["Category", "Amount"]}],
        },
        uploaded_csv_preview=preview,
        uploaded_csv_columns=["Category", "Amount"],
    )

    result_state = visualization_agent_node(state)
    viz_config = result_state["visualization_config"]

    assert result_state["visualization_status"] == "success"
    assert viz_config["metadata"]["data_rows"] == len(preview) - 1
    assert viz_config["charts"]


def test_generate_visualizations_without_data_returns_error(agent):
    result = agent.generate_visualizations(
        planner_def={"job_name": "empty", "transformations": []},
        execution_result={},
        sample_data=[],
    )

    assert result["status"] == "error"
    assert "No data" in result["metadata"]["error"]
