import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from app.agents.planner import plan_etl_job, PlannerOutput, ToolCall, NoParams, DeployInfrastructureParams, GatherInformationParams, SuggestAnalysisParams
from app.graph.etl_state import ETLState

@patch("app.agents.planner.llm")
def test_planner_generate_code(mock_llm):
    """Tests that the planner correctly calls the 'generate_code' tool."""
    mock_llm.invoke.return_value = PlannerOutput(
        tool_call=ToolCall(name="generate_code", parameters=NoParams())
    )
    state = ETLState(messages=[HumanMessage(content="generate code for me")])
    result = plan_etl_job(state)
    assert result["ready_to_summarize"] is True
    assert result["ready_to_code"] is False

@patch("app.agents.planner.llm")
def test_planner_deploy_infrastructure(mock_llm):
    """Tests that the planner correctly calls the 'deploy_infrastructure' tool."""
    mock_llm.invoke.return_value = PlannerOutput(
        tool_call=ToolCall(name="deploy_infrastructure", parameters=DeployInfrastructureParams(platform="gcp", app_type="python-docker"))
    )
    state = ETLState(messages=[HumanMessage(content="deploy on gcp")])
    result = plan_etl_job(state)
    assert result["infrastructure_request"]["type"] == "gcp"
    assert result["infrastructure_request"]["app_type"] == "python-docker"

@patch("app.agents.planner.llm")
def test_planner_summarize_job(mock_llm):
    """Tests that the planner correctly calls the 'summarize_job' tool."""
    mock_llm.invoke.return_value = PlannerOutput(
        tool_call=ToolCall(name="summarize_job", parameters=NoParams())
    )
    state = ETLState(messages=[HumanMessage(content="all set")])
    result = plan_etl_job(state)
    assert result["ready_to_summarize"] is True
    assert result["ready_to_code"] is False

@patch("app.agents.planner.llm")
def test_planner_gather_information(mock_llm):
    """Tests that the planner correctly calls the 'gather_information' tool."""
    mock_llm.invoke.return_value = PlannerOutput(
        tool_call=ToolCall(name="gather_information", parameters=GatherInformationParams(prompt="What is the data source?"))
    )
    state = ETLState(messages=[HumanMessage(content="I want to do some analysis")])
    result = plan_etl_job(state)
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "What is the data source?"

@patch("app.agents.planner.llm")
def test_planner_suggest_analysis(mock_llm):
    """Tests that the planner correctly calls the 'suggest_analysis' tool."""
    mock_llm.invoke.return_value = PlannerOutput(
        tool_call=ToolCall(name="suggest_analysis", parameters=SuggestAnalysisParams(suggestions=["average salary by job title"]))
    )
    state = ETLState(messages=[HumanMessage(content="give me some ideas")])
    result = plan_etl_job(state)
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][-1], AIMessage)
    assert "average salary by job title" in result["messages"][-1].content

def test_planner_schedule_task():
    """Tests that the planner correctly calls the 'schedule_task' tool."""
    state = ETLState(messages=[HumanMessage(content="Group by region then sum the Revenue and schedule it to run every minute")])
    result = plan_etl_job(state)
    assert result.get("task_schedule") is not None
    assert result["task_schedule"]["schedule_type"] == "repetitive"
    assert result["ready_to_summarize"] is False
    assert result["ready_to_code"] is True

def test_planner_schedule_relative_task():
    """Tests that the planner correctly calls the 'schedule_task' tool."""
    state = ETLState(messages=[HumanMessage(content="Group by region then sum the Revenue and schedule it to run 5 minutes from now")])
    result = plan_etl_job(state)
    assert result.get("task_schedule") is not None
    assert result["task_schedule"]["schedule_type"] == "relative"
    assert result["task_schedule"].get("second", -1) == 300
    assert result["ready_to_summarize"] is False
    assert result["ready_to_code"] is True


def test_planner_schedule_absolute_task():
    """Tests that the planner correctly calls the 'schedule_task' tool."""
    state = ETLState(messages=[HumanMessage(content="Group by region then sum the Revenue and schedule it to Feb 18, 2027 at 9:09 PM")])

    result = plan_etl_job(state)
    assert result.get("task_schedule") is not None

    task_schedule = result["task_schedule"]

    assert task_schedule["schedule_type"] == "absolute"
    assert int(task_schedule["month"]) == 2
    assert int(task_schedule["day_of_month"]) == 18
    assert int(task_schedule["hour"]) == 21
    assert int(task_schedule["minute"]) == 9
    assert result["ready_to_summarize"] is False
    assert result["ready_to_code"] is True

def test_planner_task_status():
    """Tests that the planner correctly calls the 'task_status' tool."""
    state = ETLState(
        messages=[HumanMessage(content="what is the status of task scheduled-task-id")],
        task_list=["scheduled-task-id", "scheduled-task-id2"]
    )
    result = plan_etl_job(state)

    task_operation = result.get("task_operation")
    assert task_operation is not None
    assert task_operation["operation"] == "STATUS"
    assert task_operation["task_id"] == "scheduled-task-id"
    assert result["ready_to_summarize"] is False
    assert result["ready_to_code"] is False

def test_planner_task_info():
    """Tests that the planner correctly calls the 'task_info' tool."""
    state = ETLState(
        messages=[HumanMessage(content="give me information on task scheduled-task-id")],
        task_list=["scheduled-task-id", "scheduled-task-id2"]
    )
    result = plan_etl_job(state)

    task_operation = result.get("task_operation")
    assert task_operation is not None
    assert task_operation["operation"] == "INFO"
    assert task_operation["task_id"] == "scheduled-task-id"
    assert result["ready_to_summarize"] is False
    assert result["ready_to_code"] is False

def test_planner_retrieve_result():
    """Tests that the planner correctly calls the 'retrieve_result' tool."""
    state = ETLState(
        messages=[HumanMessage(content="give me the result from the 2nd run of the scheduled-task-id")],
        task_list=["scheduled-task-id", "scheduled-task-id2"]
    )
    result = plan_etl_job(state)

    task_operation = result.get("task_operation")
    assert task_operation is not None
    assert task_operation["operation"] == "RETRIEVE_RESULT"
    assert task_operation["task_id"] == "scheduled-task-id"
    assert task_operation["result_index"] == 1
    assert result["ready_to_summarize"] is False
    assert result["ready_to_code"] is False

def test_planner_list_tasks():
    """Tests that the planner correctly calls the 'list_tasks' tool."""
    state = ETLState(
        messages=[HumanMessage(content="what is the task list")],
        task_list=["scheduled-task-id", "scheduled-task-id2"]
    )
    result = plan_etl_job(state)

    task_operation = result.get("task_operation")
    assert task_operation is not None
    assert task_operation["operation"] == "LIST"

def test_planner_train_model():
    """Tests that the planner correctly calls the 'train_model' tool."""
    state = ETLState(
        messages=[HumanMessage(content="start training")]
    )
    result = plan_etl_job(state)

    assert result["ready_to_summarize"] is False
    assert result["ready_to_code"] is False
    assert result["enable_training"] is True

def test_planner_stub_medical_condition(monkeypatch):
    """Ensure stub planner handles medical condition percentage prompt."""
    monkeypatch.setenv("AVALOKA_FORCE_PLAN_ON_RESPONSE", "1")
    prompt = "Calculate the percentage of total patients for each medical condition to understand relative prevalence."
    state = ETLState(
        messages=[HumanMessage(content=prompt)],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=False,
        coder_definition={},
        data_source_location="app/sample_data/test-input-data_24f9bef2-4cd6-4771-92c4-833550791330.csv",
        output_location="app/sample_data/output_medical_conditions.csv",
        schema={},
        infrastructure_request={},
        infrastructure_provisioned={},
        execution_result={},
        input_data_type="csv",
        output_file_data={},
        deploy_on_k8s=False,
        uploaded_csv_preview=[
            ["Name", "Medical Condition"],
            ["Alice", "Diabetes"],
            ["Bob", "Cancer"],
        ],
        uploaded_csv_columns=["Name", "Medical Condition"],
        planner_graph_path=None,
        planner_graph_base64=None,
        planner_graph_status=None,
        planner_graph_error=None,
        generated_code=None,
        coder_pseudocode=None,
    )

    result = plan_etl_job(state)
    plan = result["plan"]

    assert "percentage" in plan.lower()
    assert "medical condition" in plan.lower()
    assert result["ready_to_code"] is True

def test_planner_train_plan1():
    """Tests that the planner correctly calls the 'train_model' tool."""
    state = ETLState(
        messages=[HumanMessage(content="make an initial training plan to predict transmission_type")]
    )
    result = plan_etl_job(state)
    assert result["enable_training"] is True

def test_planner_train_plan2():
    """Tests that the planner correctly calls the 'train_model' or `respond_to_user` tool."""
    state = ETLState(
        messages=[HumanMessage(content="Use an unsupervised `KMeans` algorithm to cluster all 32 million rows into 100 behavioral segments based on raw integer IDs.")]
    )
    result = plan_etl_job(state)

    assert result["enable_training"] is True or len(state["messages"]) > 1