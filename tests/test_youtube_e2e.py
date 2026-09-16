import os
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from langchain_core.messages import HumanMessage

from app.api.workflow import build_graph
from app.graph.etl_state import ETLState
from app.services.db.milvus_client import milvus_client


@pytest.fixture(scope="session")
def compiled_graph():
    return build_graph().compile()


@pytest.fixture(scope="session")
def youtube_dataset_path():
    path = Path(__file__).parent.parent / "app" / "sample_data" / "Global_YouTube_Statistics.csv"
    if not path.exists():
        pytest.skip(f"Dataset {path} unavailable.")
    return path


@pytest.fixture(scope="session")
def youtube_schema(youtube_dataset_path):
    df = pd.read_csv(youtube_dataset_path, nrows=0)
    return {col: "object" for col in df.columns}


# ---------------------------------------------------------------------------
# Phase 5.2: Core Pipeline Tests
# ---------------------------------------------------------------------------
@pytest.mark.e2e
def test_e2e_youtube_aggregation_target(compiled_graph, youtube_dataset_path, tmp_path):
    """
    Test Phase 5.2: Core Pipeline Integration (Planner & Coder).
    Group YouTube channels by category and calculate sum of video views.
    """
    output_location = tmp_path / "youtube_output.csv"
    
    # We explicitly tell it the output path in the prompt or state so it knows
    prompt = f"Group the Global YouTube Statistics by 'category' and calculate the average of 'subscribers'."
    
    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        plan="",
        ready_to_summarize=False,
        ready_to_code=False,
        data_source_location=str(youtube_dataset_path),
        output_location=str(output_location),
        infrastructure_request=None,
        infrastructure_provisioned=None,
        session_id="test_youtube_e2e_session_1"
    )

    final_state = compiled_graph.invoke(initial_state, config={"recursion_limit": 50})
    
    # 1. Assert successful execution
    exec_result = final_state.get("execution_result", {})
    assert exec_result.get("status") in ["success", "completed"], f"Execution failed: {exec_result}"
    
    # Wait, the execution_agent_node_local creates `output_file_data` if successful mapping is there.
    # Given we might be running the real LLM planner & coder during this test, we must check if output was created.
    assert "output_file_data" in final_state and final_state["output_file_data"] is not None

    # Verify the output data can be parsed
    output_df = pd.read_csv(output_location)
    assert not output_df.empty, "Output CSV should not be empty"
    # Should have category and subscribers or subscribers_mean since the prompt asked for average.
    columns_list = [c.lower() for c in output_df.columns]
    assert "category" in columns_list, f"Expected 'category' column, got: {columns_list}"

# ---------------------------------------------------------------------------
# Phase 5.3: Memory Validation
# ---------------------------------------------------------------------------
@pytest.mark.e2e
@patch('app.services.milvus_recorder.record_execution_to_milvus.delay')
def test_e2e_youtube_milvus_trigger(mock_record_delay, compiled_graph, youtube_dataset_path, tmp_path):
    """
    Test Phase 5.3: Memory Validation.
    Ensure Layer 4 (Milvus) tasks are correctly triggered.
    """
    output_location = tmp_path / "youtube_output_milvus.csv"
    prompt = f"Find the top 5 YouTube channels with the highest video views."
    session_id = "test_youtube_milvus_session_1"
    
    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        plan="",
        ready_to_summarize=False,
        ready_to_code=False,
        data_source_location=str(youtube_dataset_path),
        output_location=str(output_location),
        infrastructure_request=None,
        infrastructure_provisioned=None,
        session_id=session_id
    )

    final_state = compiled_graph.invoke(initial_state, config={"recursion_limit": 50})
    
    exec_result = final_state.get("execution_result", {})
    assert exec_result.get("status") in ["success", "completed"]
    
    # Verify the celery background task was triggered
    assert mock_record_delay.called, "record_execution_to_milvus.delay was not called!"
    
    # Verify the correct session ID was passed
    args, kwargs = mock_record_delay.call_args
    assert session_id in args or kwargs.get("session_id") == session_id

# ---------------------------------------------------------------------------
# Phase 5.4: Remote / Cloud Execution Workflows
# ---------------------------------------------------------------------------
@pytest.mark.e2e
def test_e2e_youtube_cloud_routing(compiled_graph, youtube_dataset_path, tmp_path):
    """
    Test Phase 5.4: Remote Cloud Execution Workflows.
    Verify K8s/Ray execution parameters build properly with a dryrun.
    """
    # Using dryrun to test parameterization logic without live K8s cluster
    os.environ["RAYJOB_DRYRUN"] = "1"
    
    # We use a mocked cloud path here so it triggers the Ray execution agent
    fake_cloud_path = "s3://avaloka-fake-bucket/youtube/Global_YouTube_Statistics.csv"
    
    prompt = f"Calculate the total video views by country."
    
    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        plan="",
        ready_to_summarize=False,
        ready_to_code=False,
        data_source_location_cloud=fake_cloud_path,
        execution_mode="k8s-ray",
        ray_namespace="test-ray-ns",
        session_id="test_youtube_cloud_session"
    )

    final_state = compiled_graph.invoke(initial_state, config={"recursion_limit": 50})
    
    exec_result = final_state.get("execution_result", {})
    assert exec_result.get("status") == "dryrun", "Expected Ray job dryrun state"
    assert exec_result.get("namespace") == "test-ray-ns"
    
    rayjob_yaml = exec_result.get("rayjob_yaml")
    assert rayjob_yaml is not None, "RayJob YAML should have been generated"
    assert fake_cloud_path in rayjob_yaml, "Cloud URI should be embedded in the submitted script"
    
    # Reset env
    os.environ.pop("RAYJOB_DRYRUN", None)
