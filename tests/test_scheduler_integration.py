import os
from pathlib import Path
import time
from datetime import datetime, timezone

import pytest

pytest.importorskip("celery")
pytest.importorskip("redis")

from langchain_core.messages import HumanMessage, AIMessage
from app.graph.etl_state import ETLState
from app.core.celery_app import AvalokaScheduler, celery_app
from app.agents.scheduler import task_scheduler_node
from redbeat.schedulers import ensure_conf
from dotenv import load_dotenv

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _redbeat_conf():
    """Configure RedBeat lazily, once the module is actually selected to run.

    Doing this at import time runs it during collection - before the
    `integration` marker can deselect the module - so a hermetic run aborts here
    with InvalidSpecError whenever an earlier test module has already swapped
    celery_app for a mock.
    """
    load_dotenv()
    ensure_conf(celery_app)


SAMPLE_PREVIEW = [
    ['name', 'job_title', 'age', 'location', 'salary'],
    ['Clint Eastwood', 'Manager', '75', 'Los Angeles', '10000'],
    ['Eric Bana', 'Producer', '50', 'Los Angeles', '8000']
]

def _base_state(tmp_output: str, prompt: str) -> ETLState:
    return ETLState(
        messages=[HumanMessage(content=prompt)],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=False,
        coder_definition={
            "code": "print('hello world')"
        },
        data_source_location=os.path.abspath("app/sample_data/salaries.csv"),
        output_location=tmp_output,
        schema={"OrderID": "int64"},
        infrastructure_request={},
        infrastructure_provisioned={},
        execution_result={},
        input_data_type="csv",
        output_file_data={},
        deploy_on_k8s=False,
        uploaded_csv_preview=SAMPLE_PREVIEW,
        uploaded_csv_columns=SAMPLE_PREVIEW[0],
    )


@pytest.mark.integration
def test_task_create_absolute(monkeypatch, tmp_path):
    output_path = tmp_path / "real_schedule.csv"
    state = _base_state(str(output_path), "...")
    state["task_schedule"] = {
        "task_type": "execute",
        "schedule_type": "absolute",
        "month": 2,
        "day_of_month": 18,
        "hour": 21,
        "minute": 9
    }

    final_state = task_scheduler_node(state)

    assert (final_state.get("task_info") or {}).get("task_id") is not None
    assert len(final_state["task_list"]) == 1

@pytest.mark.integration
def test_task_create_relative(monkeypatch, tmp_path):
    output_path = tmp_path / "real_schedule.csv"
    state = _base_state(str(output_path), "...")
    state["task_schedule"] = {
        "task_type": "training",
        "schedule_type": "relative",
        "second": 300,
    }

    final_state = task_scheduler_node(state)

    assert (final_state.get("task_info") or {}).get("task_id") is not None
    assert len(final_state["task_list"]) == 1

@pytest.mark.integration
def test_task_create_repetitive(monkeypatch, tmp_path):
    output_path = tmp_path / "real_schedule.csv"
    state = _base_state(str(output_path), "...")
    state["task_schedule"] = {
        "task_type": "repetitve",
        "schedule_type": "relative",
        "minute": "*/15"
    }

    final_state = task_scheduler_node(state)

    assert (final_state.get("task_info") or {}).get("task_id") is not None
    assert len(final_state["task_list"]) == 1

@pytest.mark.integration
def test_task_cancel(monkeypatch, tmp_path):
    output_path = tmp_path / "real_schedule.csv"
    state = _base_state(str(output_path), "")
    state["task_schedule"] = {
        "task_type": "training",
        "schedule_type": "relative",
        "second": 10,
    }
    new_state = task_scheduler_node(state)

    task_id = (new_state.get("task_info") or {}).get("task_id")
    assert task_id is not None
    assert len(new_state["task_list"]) == 1

    new_state["task_operation"] = {
        "operation": "CANCEL",
        "task_id": task_id
    }
    final_state = task_scheduler_node(new_state)
    assert len(final_state["task_list"]) == 0

@pytest.mark.integration
def test_task_result_status(monkeypatch, tmp_path):
    output_path = tmp_path / "real_schedule.csv"
    state = _base_state(str(output_path), "")
    state["task_schedule"] = {
        "task_type": "training",
        "schedule_type": "relative",
        "second": 10,
    }
    new_state = task_scheduler_node(state)

    task_id = (new_state.get("task_info") or {}).get("task_id")
    assert task_id is not None
    assert len(new_state["task_list"]) == 1

    new_state["task_operation"] = {
        "operation": "STATUS",
        "task_id": task_id
    }
    final_state = task_scheduler_node(new_state.copy())
    
    assert final_state.get("output_file_data") is not None
    assert len(final_state.get("messages", [])) > len(new_state.get("messages", []))

@pytest.mark.integration
def test_task_info(monkeypatch, tmp_path):
    output_path = tmp_path / "real_schedule.csv"
    state = _base_state(str(output_path), "")
    state["task_schedule"] = {
        "task_type": "training",
        "schedule_type": "relative",
        "second": 10,
    }
    new_state = task_scheduler_node(state)

    task_id = (new_state.get("task_info") or {}).get("task_id")
    assert task_id is not None
    assert len(new_state["task_list"]) == 1

    new_state["task_operation"] = {
        "operation": "INFO",
        "task_id": task_id
    }
    final_state = task_scheduler_node(new_state.copy())
    
    assert final_state.get("output_file_data") is not None
    assert len(final_state.get("messages", [])) > len(new_state.get("messages", []))