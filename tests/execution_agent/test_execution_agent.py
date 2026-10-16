"""Unit tests for the execution agent (k8s service path).

Repaired from the quarantined original. Three of the APIs the original imported
are gone for good -- `prepare_infrastructure`, and `save_output_to_file` /
`process_execution_results` from the deleted `app.execution_helper`. What each
one *checked* was re-pointed at the function that does that job now:

  prepare_infrastructure  ->  get_service_details(), which pulls the service
                              ip/port out of the provisioning outcome, plus
                              execution_agent_node's reprovision attempt.
  process_execution_results -> execution_agent_node, which now decodes the
                              base64 CSV and attaches output_file_data itself.

`tests/test_execution_unit.py` covers much of this module. It was named
test_execution_integration.py when this file was written, and that suffix made
tests/conftest.py mark all 27 of its tests `integration`, deselecting them from
the hermetic gate (`-m "not cluster and not cloud and not integration"`). It
has since been renamed and runs there. Everything below is hermetic too -- no
cluster, no credentials, no network.
"""

import base64

import pandas as pd
import pytest
from unittest import mock

import app.agents.execution_agent as exec_mod
from app.agents.execution_agent import (
    execution_agent_node,
    execute_code_on_k8s,
    get_service_details,
)
from app.graph.etl_state import ETLState


@pytest.fixture
def provisioned_state():
    """State whose infrastructure is provisioned and advertising an ip/port."""
    return ETLState(
        messages=[],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=True,
        coder_definition={},
        infrastructure_request={},
        infrastructure_provisioned={
            "type": "gcp",
            "app_type": "python-docker",
            "status": "provisioned",
            "details": [
                {
                    "step_name": "Get infra-agent-service IP/Port",
                    "status": "SUCCESS",
                    "message": "Service infra-agent-service accessible at 34.123.456.789:8080",
                    "details": {"ip": "34.123.456.789", "port": "8080"},
                }
            ],
        },
    )


@pytest.fixture
def k8s_success_response():
    """What the in-cluster executor returns for a successful run."""
    iris_sample = pd.DataFrame(
        {
            "Species": ["setosa", "versicolor", "virginica"],
            "sepal_length": [5.1, 5.9, 6.3],
            "sepal_width": [3.5, 2.7, 2.9],
        }
    )
    csv_data = iris_sample.to_csv(index=False)
    payload = "data:text/csv;base64," + base64.b64encode(csv_data.encode()).decode()

    return {
        "status": "success",
        "message": "Code executed successfully",
        "execution_time": 1.23,
        "stdout": "Processing complete!",
        "stderr": "",
        "output": payload,
        "output_file": {
            "filename": "output.csv",
            "content": payload,
            "size": len(csv_data),
        },
    }


def test_service_details_come_from_the_provisioning_outcome(provisioned_state):
    """The ip/port the agent will dial are read out of the provisioned details.

    Repairs test_prepare_infrastructure_success: prepare_infrastructure used to
    provision and then return this triple; get_service_details now does the
    extraction half, off the same outcome structure.
    """
    result = get_service_details(provisioned_state["infrastructure_provisioned"])

    assert result["status"] == "success"
    assert result["service_ip"] == "34.123.456.789"
    assert result["service_port"] == "8080"


def test_execute_code_on_k8s():
    """Test execute_code_on_k8s function"""
    with mock.patch("requests.post") as mock_post:
        mock_post.return_value = mock.MagicMock(
            status_code=200,
            json=lambda: {"status": "success", "message": "Execution completed"},
        )

        result = execute_code_on_k8s(
            code="print('hello')",
            service_ip="192.168.1.1",
            service_port="8080",
            data_source_location="/data/input.csv",
        )

        # The service's verdict is handed back, with the measured duration added.
        assert result["status"] == "success"
        assert "Execution completed" in result["message"]
        assert "execution_time" in result

        # Exactly one request, to the /execute endpoint of the service we were given.
        mock_post.assert_called_once()
        assert mock_post.call_args.args[0] == "http://192.168.1.1:8080/execute"


def test_successful_output_is_parsed_and_offered_as_a_file(
    provisioned_state, k8s_success_response
):
    """A base64 CSV result becomes a rendered table and a downloadable file.

    Repairs test_process_execution_results. The helper that did this is gone;
    execution_agent_node decodes the data URI and populates output_file_data.
    """
    with mock.patch.object(
        exec_mod, "execute_code_on_k8s", return_value=k8s_success_response
    ):
        state = dict(provisioned_state)
        state["coder_definition"] = {"code": "print('Test code')"}
        state["data_source_location"] = "data/iris.csv"

        result_state = execution_agent_node(state)

    # The CSV was decoded into a frame and rendered, not passed through raw.
    assert len(result_state["messages"]) == 1
    rendered = result_state["messages"][-1].content
    assert "base64" not in rendered
    assert "Species" in rendered and "setosa" in rendered

    # The output file is carried out of the result for download.
    assert result_state["output_file_data"] is not None
    assert result_state["output_file_data"]["filename"] == "output.csv"
    assert result_state["output_file_data"]["size"] == len(
        base64.b64decode(k8s_success_response["output"].split("base64,")[1])
    )


def test_execution_agent_node_infra_failure():
    """Test execution_agent_node when infrastructure cannot be provisioned"""
    unprovisioned = {"status": "failed", "details": []}

    def never_provisions(state):
        state["infrastructure_provisioned"] = unprovisioned
        return state

    with mock.patch.object(
        exec_mod, "infra_agent_node", side_effect=never_provisions
    ) as mock_infra:
        state = ETLState(
            messages=[],
            planner_definition={},
            ready_to_summarize=False,
            ready_to_code=True,
            coder_definition={"code": "print('Test code')"},
            infrastructure_request={},
            infrastructure_provisioned=None,
        )

        result_state = execution_agent_node(state)

    # Provisioning was attempted before giving up.
    mock_infra.assert_called_once()

    assert "execution_result" in result_state
    assert result_state["execution_result"]["status"] == "failed"
    assert "Infrastructure service not available" in result_state["execution_result"]["message"]
    assert len(result_state["messages"]) == 1  # Should have added error message


def test_execution_agent_node_no_code():
    """Test execution_agent_node with no code provided"""
    state = ETLState(
        messages=[],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=True,
        coder_definition={},  # Empty code definition
        infrastructure_request={},
        infrastructure_provisioned=None,
    )

    result_state = execution_agent_node(state)

    assert "execution_result" in result_state
    assert result_state["execution_result"]["status"] == "failed"
    assert "No code provided" in result_state["execution_result"]["message"]
    assert len(result_state["messages"]) == 1  # Should have added error message
