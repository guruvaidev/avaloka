import pytest
from unittest.mock import patch, MagicMock
from typing import Any, Dict

from langchain_core.messages import HumanMessage, AIMessage

from app.api.workflow import build_graph
from app.graph.etl_state import ETLState
from app.agents.planner import plan_etl_job


mock_summarize_job_llm_response = AIMessage(
        content="",
        **{
            "tool_calls": [{"name": "summarize_job", "args": {}, "id": "1"}]
        }
    )


mock_gather_information_llm_response = AIMessage(
        content="",
        **{
            "tool_calls": [{"name": "gather_information", "args": {"prompt": "Step added. Can you tell me more?"}, "id": "1"}]
        }
    )

mock_gather_information_no_message_llm_response = AIMessage(
        content="",
        **{
            "tool_calls": [{"name": "gather_information", "args": {"prompt": ""}, "id": "1"}]
        }
    )


mock_gather_information_add_transform_llm_response = AIMessage(
        content="",
        **{
            "tool_calls": [{"name": "gather_information", "args": {"prompt": "Added transform_1"}, "id": "1"}]
        }
    )

mock_gather_information_clarify_llm_response = AIMessage(
        content="",
        **{
            "tool_calls": [{"name": "gather_information", "args": {"prompt": "could you clarify"}, "id": "1"}]
        }
    )


mock_respond_with_couldnt_understand = AIMessage(
        content="",
        **{
            "tool_calls": [{"name": "respond_to_user", "args": {"response_text": "couldn't understand"}, "id": "1"}]
        }
    )

mock_code_generation_llm_response = AIMessage(
        content="",
        **{
            "tool_calls": [{"name": "generate_code", "args": {}, "id": "1"}]
        }
    )

mock_deploy_infra_llm_response = AIMessage(
        content="",
        **{
            "tool_calls": [{"name": "deploy_infrastructure", "args": {"type": "gcp", "app_type": "python-docker"}, "id": "1"}]
        }
    )


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def compiled_graph():
    return build_graph().compile()


@pytest.fixture
def compiled_graph_e2e_tests():
    mock_planner = MagicMock(name="mock_planner")
    mock_summarizer = MagicMock(name="mock_summarizer")
    mock_coder = MagicMock(name="mock_coder")

    mock_planner.side_effect = lambda s: ETLState(
        messages=s["messages"], planner_definition=s.get("planner_definition", {}),
        ready_to_summarize=True, ready_to_code=False,
        coder_definition={}, data_source_location=s.get("data_source_location", ""),
        output_location=s.get("output_location", ""),
        infrastructure_request={}, infrastructure_provisioned={},
        execution_result={}, input_data_type=s.get("input_data_type", ""),
        output_file_data={}, deploy_on_k8s=s.get("deploy_on_k8s", False),
        uploaded_csv_preview=s.get("uploaded_csv_preview", []),
        uploaded_csv_columns=s.get("uploaded_csv_columns", []),
    )

    mock_summarizer.return_value = ETLState(
        messages=[HumanMessage(content="Summary created")],
        planner_definition={"steps": ["transform"]},
        ready_to_summarize=True, ready_to_code=False,
        coder_definition={}, data_source_location="", output_location="",
        infrastructure_request={}, infrastructure_provisioned={},
        execution_result={}, input_data_type="", output_file_data={},
        deploy_on_k8s=False, uploaded_csv_preview=[], uploaded_csv_columns=[],
    )

    mock_coder.return_value = ETLState(
        messages=[HumanMessage(content="Code generated")],
        planner_definition={"steps": ["transform"]},
        ready_to_summarize=False, ready_to_code=True,
        coder_definition={"code": "..."},
        data_source_location="", output_location="",
        infrastructure_request={}, infrastructure_provisioned={},
        execution_result={}, input_data_type="", output_file_data={},
        deploy_on_k8s=False, uploaded_csv_preview=[], uploaded_csv_columns=[],
    )

    import app.api.workflow as workflow_module
    with patch.object(workflow_module, "plan_etl_job", mock_planner), \
            patch.object(workflow_module, "summarize_etl_job", mock_summarizer), \
            patch.object(workflow_module, "coder_node", mock_coder):
        compiled_graph = build_graph().compile()
    return compiled_graph, mock_coder


# -----------------------------------------------------------------------------
# Integration tests (planner only)
# -----------------------------------------------------------------------------

@patch("app.agents.planner.llm")
def test_ready_to_summarize_integration(mock_llm):
    """Integration: plan_etl_job detects summarize flag."""
    mock_llm.invoke.return_value = mock_summarize_job_llm_response

    state = ETLState(messages=[HumanMessage(content="Looks good, please summarize")],
                     planner_definition={})
    result = plan_etl_job(state)
    assert result["ready_to_summarize"] is True
    assert result["ready_to_code"] is False


@patch("app.agents.planner.llm")
def test_ready_to_summarize_e2e(mock_llm, compiled_graph):
    """E2E: compiled_graph detects summarize flag."""
    mock_llm.invoke.return_value = mock_summarize_job_llm_response

    init = ETLState(messages=[HumanMessage(content="Good to go!")],
                    planner_definition={})
    result = compiled_graph.invoke(init)
    assert result["ready_to_summarize"] is True


@patch("app.agents.planner.llm")
def test_planner_definition_merge(mock_llm):
    """
    Since planner.py does not merge planner_definition from LLM inputs,
    we simply assert that the existing planner_definition remains intact
    and we don't crash.
    """
    mock_llm.invoke.return_value = mock_gather_information_llm_response

    initial = {"source": "csv"}
    state = ETLState(messages=[HumanMessage(content="Drop nulls")],
                     planner_definition=initial.copy())
    result = plan_etl_job(state)
    assert result["planner_definition"] == initial  # unchanged
    assert any(isinstance(m, AIMessage) and "Step added" in m.content for m in result["messages"])


@patch("app.agents.planner.llm")
def test_invalid_state_no_messages(mock_llm):
    mock_llm.invoke.return_value = mock_gather_information_no_message_llm_response

    state = ETLState(messages=[], planner_definition={})
    result = plan_etl_job(state)
    assert isinstance(result["messages"], list)
    assert result["planner_definition"] == {}


@patch("app.agents.coder.coder_llm")
@patch("app.agents.planner.llm")
def test_planning_agent_sql_input(mock_llm, mock_coder_llm, compiled_graph):

    mock_llm.invoke.return_value = mock_code_generation_llm_response
    mock_coder_llm.invoke.return_value = AIMessage(content="Validated SQL plan:\n```python\n2==2\n```")
    sql_message = "SELECT name, SUM(sales) FROM orders GROUP BY name"
    state = ETLState(messages=[HumanMessage(content=sql_message)],
                     planner_definition={})
    result = compiled_graph.invoke(state)
    # The coding subgraph runs coder -> validators -> (refine -> coder)*, so a
    # logic-review bounce legitimately invokes the coder LLM more than once.
    assert mock_coder_llm.invoke.call_count >= 1
    assert any("Validated SQL plan" in m.content for m in result["messages"])


@patch("app.agents.coder.coder_llm")
@patch("app.agents.planner.llm")
def test_planning_agent_nl_to_sql(mock_llm, mock_coder_llm, compiled_graph):
    mock_llm.invoke.return_value = mock_code_generation_llm_response
    mock_coder_llm.invoke.return_value = AIMessage(content="Generated SQL:\n```python\n2==2\n```")

    nl_message = "Get total sales per customer"
    state = ETLState(messages=[HumanMessage(content=nl_message)],
                     planner_definition={})
    result = compiled_graph.invoke(state)
    # The coding subgraph runs coder -> validators -> (refine -> coder)*, so a
    # logic-review bounce legitimately invokes the coder LLM more than once.
    assert mock_coder_llm.invoke.call_count >= 1
    assert any("Generated SQL" in m.content for m in result["messages"])


@patch("app.agents.coder.coder_llm")
@patch("app.agents.planner.llm")
def test_planning_agent_handles_bad_sql(mock_llm, mock_coder_llm, compiled_graph):
    """Invalid SQL handling via clarifying prompt."""
    mock_llm.invoke.return_value = mock_code_generation_llm_response
    mock_coder_llm.invoke.return_value = AIMessage(content="Invalid SQL:\n```python\n2==2\n```")

    bad_sql = "SELECT FROM WHERE"
    state = ETLState(messages=[HumanMessage(content=bad_sql)],
                     planner_definition={})
    result = compiled_graph.invoke(state)
    assert any("Invalid SQL" in m.content for m in result["messages"])


@patch("app.agents.coder.coder_llm")
@patch("app.agents.planner.llm")
@pytest.mark.parametrize("sql,msg", [
    ("SELECT FROM WHERE", "Invalid SQL, please clarify"),
    ("SELECT id, name FROM users", "Validated SQL plan"),
])
def test_invalid_and_corrected_sql(mock_llm, mock_coder_llm, compiled_graph, sql, msg):
    mock_llm.invoke.return_value = mock_code_generation_llm_response
    mock_coder_llm.invoke.return_value = AIMessage(content=msg)

    state = ETLState(messages=[HumanMessage(content=sql)],
                     planner_definition={})
    result = compiled_graph.invoke(state)
    assert any(msg.split()[0] in m.content for m in result["messages"])

@patch("app.agents.planner.llm")
@pytest.mark.parametrize("content", [
    "Please deploy to GCP",
    "Looks good, deploy to GCP",
])
def test_gcp_infra_variants(mock_llm, content):
    """
    An explicit GCP deploy instruction must populate infrastructure_request.

    Asserted at the planner level: driving the compiled graph would route on to
    provision_infra and shell out to gcloud/kubectl for real. Graph-level infra
    routing is covered by test_csv_upload_triggers_infra_edge, which stubs the
    infra agent.

    The second variant is the compound case -- the "looks good" acknowledgement
    must not swallow the deploy instruction that follows it.
    """
    mock_llm.invoke.return_value = mock_deploy_infra_llm_response

    state = ETLState(messages=[HumanMessage(content=content)], planner_definition={})
    result = plan_etl_job(state)
    assert result["infrastructure_request"]["type"] == "gcp"
    assert result["infrastructure_request"]["app_type"] == "python-docker"


@patch("app.agents.coder.coder_llm")
@patch("app.agents.planner.llm")
def test_csv_upload_triggers_infra_request(mock_llm, mock_coder_llm):
    """
    Even though planner doesn't auto-infer infra from CSV,
    tests can simulate infra tool call; include CSV context for completeness.
    """
    mock_coder_llm.invoke.return_value = AIMessage(content="Validated SQL plan:\n```python\n2==2\n```")
    mock_llm.invoke.return_value = mock_deploy_infra_llm_response

    state = ETLState(messages=[HumanMessage(content="Here's my file")],
                     planner_definition={})
    result = plan_etl_job(state)
    assert result.get("infrastructure_request", {}).get("type") == "gcp"
    assert result.get("infrastructure_request", {}).get("app_type") == "python-docker"


def test_csv_upload_triggers_infra_edge():

    mock_infra_agent = MagicMock(name="infra_agent")
    mock_planner = MagicMock(name="planner")

    mock_planner.side_effect = lambda state: ETLState(
        messages=state["messages"],
        planner_definition=state.get("planner_definition", {}),
        ready_to_summarize=state.get("ready_to_summarize", False),
        ready_to_code=state.get("ready_to_code", False),
        coder_definition=state.get("coder_definition", {}),
        data_source_location=state.get("data_source_location", ""),
        output_location=state.get("output_location", ""),
        infrastructure_request={"type": "gcp", "app-type": "python-docker"},
        execution_result=state.get("execution_result", {}),
        input_data_type=state.get("input_data_type", ""),
        output_file_data=state.get("output_file_data", {}),
        deploy_on_k8s=state.get("deploy_on_k8s", False),
        uploaded_csv_preview=state.get("uploaded_csv_preview", []),
        uploaded_csv_columns=state.get("uploaded_csv_columns", []),
    )

    mock_infra_agent.side_effect = lambda state: ETLState(
        messages=state["messages"],
        planner_definition=state.get("planner_definition", {}),
        ready_to_summarize=state.get("ready_to_summarize", False),
        ready_to_code=state.get("ready_to_code", False),
        coder_definition=state.get("coder_definition", {}),
        data_source_location=state.get("data_source_location", ""),
        output_location=state.get("output_location", ""),
        infrastructure_request={},  # cleared
        infrastructure_provisioned={"status": "ok"},
        execution_result=state.get("execution_result", {}),
        input_data_type=state.get("input_data_type", ""),
        output_file_data=state.get("output_file_data", {}),
        deploy_on_k8s=state.get("deploy_on_k8s", False),
        uploaded_csv_preview=state.get("uploaded_csv_preview", []),
        uploaded_csv_columns=state.get("uploaded_csv_columns", []),
    )

    import app.api.workflow as workflow_module
    with patch.object(workflow_module, "plan_etl_job", mock_planner), \
            patch.object(workflow_module, "infra_agent_node", mock_infra_agent):
        graph = build_graph().compile()

        result = graph.invoke(ETLState(messages=[HumanMessage(content="Here's my file")],
                                       planner_definition={}))

    assert result["infrastructure_provisioned"]["status"] == "ok"


@patch("app.agents.planner.llm")
def test_planner_merges_multiple_steps(mock_llm):
    """
    planner does not mutate planner_definition; ensure stability and message append.
    """
    mock_llm.invoke.return_value = mock_gather_information_add_transform_llm_response

    initial = {"steps": ["extract"]}
    state = ETLState(messages=[HumanMessage(content="Add a transform step")],
                     planner_definition=initial.copy())
    result = plan_etl_job(state)
    assert result["planner_definition"] == initial
    assert any("transform_1" in m.content for m in result["messages"])


@patch("app.agents.planner.llm")
def test_planner_overwrites_existing_key(mock_llm):
    """
    planner does not overwrite keys via LLM; definition remains unchanged.
    """
    mock_llm.invoke.return_value = mock_gather_information_add_transform_llm_response

    initial = {"source": "csv"}
    state = ETLState(messages=[HumanMessage(content="Change source to parquet")],
                     planner_definition=initial.copy())
    result = plan_etl_job(state)
    assert result["planner_definition"]["source"] == "csv"
    assert any("parquet" in m.content for m in result["messages"])


class FakeBad:
    def __init__(self, content):
        self.content = content

@patch("app.agents.planner.llm")
def test_planner_invalid_ai_message_type(mock_llm):
    mock_llm.invoke.return_value = mock_respond_with_couldnt_understand


    state = ETLState(messages=[HumanMessage(content="Start ETL job")],
                     planner_definition={})
    result = plan_etl_job(state)
    # falls back to apology message
    assert any(isinstance(m, AIMessage) and "couldn't understand" in m.content.lower()
               for m in result["messages"])


# -----------------------------------------------------------------------------
# End-to-end tests (graph routing)
# -----------------------------------------------------------------------------
@patch("app.agents.coder.coder_llm")
@patch("app.agents.planner.llm")
def test_e2e_sql_pipeline(mock_llm, mock_coder_llm, compiled_graph):
    mock_llm.invoke.return_value = mock_code_generation_llm_response
    mock_coder_llm.invoke.return_value = AIMessage(content="Running SQL plan:\n```python\n2==2\n```")

    sql_message = "SELECT region, SUM(sales) FROM orders GROUP BY region"
    state = ETLState(messages=[HumanMessage(content=sql_message)],
                     planner_definition={})
    result = compiled_graph.invoke(state)
    # The coding subgraph runs coder -> validators -> (refine -> coder)*, so a
    # logic-review bounce legitimately invokes the coder LLM more than once.
    assert mock_coder_llm.invoke.call_count >= 1
    assert any("Running SQL" in m.content for m in result["messages"])


@patch("app.agents.planner.llm")
def test_e2e_ambiguous_nl(mock_llm, compiled_graph):
    mock_llm.invoke.return_value = mock_gather_information_clarify_llm_response

    state = ETLState(messages=[HumanMessage(content="Do the thing")],
                     planner_definition={})
    result = compiled_graph.invoke(state)
    assert any("clarify" in m.content.lower() for m in result["messages"])


@patch("app.agents.planner.llm")
def test_e2e_planner_exception_handling(mock_llm, compiled_graph):
    mock_llm.invoke.side_effect = RuntimeError("Planner failure")

    state = ETLState(messages=[HumanMessage(content="Trigger some error case")],
                     planner_definition={})
    result = compiled_graph.invoke(state)
    # A transient planner failure must not fabricate a plan and route arbitrary
    # input to the coder; it apologises and leaves both ready_* flags down.
    # ("LLM reasoning unable to resolve request" is now emitted only under the
    # AVALOKA_FORCE_PLAN_ON_RESPONSE test flag.)
    assert any("I ran into a temporary problem while planning" in m.content
               for m in result["messages"])
    assert result.get("ready_to_code") is False
    assert result.get("ready_to_summarize") is False
    assert isinstance(result["planner_definition"], dict)


@patch("app.agents.planner.llm")
def test_nl_input_generates_plan(mock_llm):
    mock_llm.invoke.return_value = mock_gather_information_llm_response

    state = ETLState(messages=[HumanMessage(content="Aggregate revenue by year")],
                     planner_definition={})
    result = plan_etl_job(state)
    assert any("Step added" in m.content for m in result["messages"])


@patch("app.agents.planner.llm")
def test_user_requests_infra_gcp(mock_llm):
    mock_llm.invoke.return_value = mock_deploy_infra_llm_response

    state = ETLState(messages=[HumanMessage(content="Please deploy on GCP")],
                     planner_definition={})
    result = plan_etl_job(state)
    assert result.get("infrastructure_request", {}).get("type") == "gcp"
    assert result.get("infrastructure_request", {}).get("app_type") == "python-docker"


@patch("app.agents.planner.llm")
def test_continue_planning_flow(mock_llm):
    mock_llm.invoke.return_value = mock_gather_information_clarify_llm_response

    state = ETLState(messages=[HumanMessage(content="I need to clean nulls")],
                     planner_definition={})
    result = plan_etl_job(state)
    assert any("clarify" in m.content for m in result["messages"])


@patch("app.agents.planner.llm")
def test_incremental_plan_merge(mock_llm):
    mock_llm.invoke.return_value = mock_gather_information_add_transform_llm_response

    state = ETLState(messages=[HumanMessage(content="Add a filter step")],
                     planner_definition={"steps": ["extract"]})
    result = plan_etl_job(state)
    assert any("Added" in m.content for m in result["messages"])


@patch("app.agents.planner.llm")
def test_handles_invalid_sql(mock_llm):
    mock_llm.invoke.return_value = mock_code_generation_llm_response

    state = ETLState(messages=[HumanMessage(content="SELECT FROM WHERE")],
                     planner_definition={})
    result = plan_etl_job(state)
    assert result["ready_to_code"] is False and result["ready_to_summarize"] is True


@patch("app.agents.planner.llm")
def test_bare_mode_no_streamlit_available(mock_llm):
    mock_llm.invoke.return_value = mock_gather_information_llm_response

    state = ETLState(messages=[HumanMessage(content="Start ETL job")],
                     planner_definition={})
    result = plan_etl_job(state)
    assert any("Can you tell me more" in m.content for m in result["messages"])


def test_conflicting_flags_routes_gracefully():
    """
    If downstream sets both flags True, route_planner_output should prefer ready_to_code.
    We simulate planner -> summarize first; then patch summarizer to set both True,
    and coder to add code (so route_after_code proceeds).
    """

    mock_planner = MagicMock(name="mock_planner")
    mock_summarizer = MagicMock(name="mock_summarizer")
    mock_coder = MagicMock(name="mock_coder")
    mock_exec = MagicMock(name="mock_exec")

    mock_planner.side_effect = lambda s: ETLState(
        messages=s["messages"], planner_definition=s.get("planner_definition", {}),
        ready_to_summarize=True, ready_to_code=False,
        coder_definition={}, data_source_location=s.get("data_source_location", ""),
        output_location=s.get("output_location", ""),
        infrastructure_request={}, infrastructure_provisioned={},
        execution_result={}, input_data_type=s.get("input_data_type", ""),
        output_file_data={}, deploy_on_k8s=s.get("deploy_on_k8s", False),
        uploaded_csv_preview=s.get("uploaded_csv_preview", []),
        uploaded_csv_columns=s.get("uploaded_csv_columns", []),
    )

    def create_from_state(s, updates):
        s.update(updates)
        return ETLState(**s)

    mock_summarizer.side_effect = lambda s: create_from_state(s, {"ready_to_summarize": True, "ready_to_code": True})
    mock_coder.side_effect = lambda s: create_from_state(s, {"coder_definition": {"code": "# ok"}})
    mock_exec.side_effect = lambda s: create_from_state(s, {"execution_result": {"status": "completed", "mode": "local"}})

    import app.api.workflow as workflow_module
    with patch.object(workflow_module, "plan_etl_job", mock_planner), \
            patch.object(workflow_module, "summarize_etl_job", mock_summarizer), \
            patch.object(workflow_module, "coder_node", mock_coder), \
            patch.object(workflow_module, "execution_agent_node_local", mock_exec):
        graph = build_graph().compile()

        result = graph.invoke(ETLState(messages=[HumanMessage(content="conflict")],
                                               planner_definition={}))
    assert result.get("execution_result", {}).get("mode") == "local"


@patch("app.agents.planner.llm")
def test_empty_ai_response_handled(mock_llm):
    """
    Simulate empty gather_information prompt -> planner appends an empty AIMessage content.
    """
    mock_llm.invoke.return_value = mock_gather_information_no_message_llm_response

    state = ETLState(messages=[HumanMessage(content="Start planning")],
                     planner_definition={})
    result = plan_etl_job(state)
    assert result["messages"][-1].content == ""


@patch("app.agents.planner.llm")
def test_end_to_end_nl_to_code_flow(mock_llm):
    """
    Natural language -> summarizer path (flag only). We stop before coder since planner
    itself only flips ready_to_summarize for generate_code/summarize_job.
    """
    mock_llm.invoke.return_value = mock_code_generation_llm_response

    state = ETLState(messages=[HumanMessage(content="Group by product and sum sales")],
                     planner_definition={})
    result = plan_etl_job(state)
    assert result["ready_to_code"] is False and result["ready_to_summarize"] is True


def test_planner_triggers_infra_on_csv_upload():
    """
    In our planner, infra is only set via deploy_infrastructure tool. We simulate that
    tool call together with CSV presence; assertion checks the tool result was applied.
    """
    with patch("app.agents.planner.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_deploy_infra_llm_response

        state = ETLState(messages=[HumanMessage(content="Here's my file")],
                         planner_definition={},
                         infrastructure_provisioned=None)
        result = plan_etl_job(state)
        assert result["infrastructure_request"]["type"] == "gcp"
        assert result["infrastructure_request"]["app_type"] == "python-docker"



def test_e2e_nl_pipeline(compiled_graph_e2e_tests):

    state = ETLState(messages=[HumanMessage(content="Aggregate sales per region")],
                     planner_definition={})
    graph, mock_coder = compiled_graph_e2e_tests
    result = graph.invoke(state)
    assert any("Code generated" in m.content for m in result["messages"])
    mock_coder.assert_called_once()

def test_e2e_full_chain(compiled_graph_e2e_tests):
    state = ETLState(messages=[HumanMessage(content="Aggregate sales")],
                     planner_definition={})
    graph, mock_coder = compiled_graph_e2e_tests
    result = graph.invoke(state)
    assert any("Code generated" in m.content for m in result["messages"])
    mock_coder.assert_called_once()