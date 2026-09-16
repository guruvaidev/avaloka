"""Regression tests: inference requests must route to the MTA deterministically
when a trained model exists, not only via the enable_training flag.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.api.workflow as wf

MODEL_STATE = {"training_result": {"model_type": "xgboost"}, "training_completed": True}


def _state(prompt, **extra):
    state = {"messages": [HumanMessage(content=prompt)]}
    state.update(extra)
    return state


@pytest.mark.parametrize("prompt", [
    "predict the species for SepalLength 5.1",
    "run inference on the new rows",
    "use the model on this batch",
    "can you infer labels for the uploaded file?",
])
def test_inference_request_with_model_routes_to_mta(prompt):
    assert wf.route_planner_output(_state(prompt, **MODEL_STATE)) == "train_models"


def test_inference_request_beats_stale_summarize_flag():
    state = _state(
        'run inference with {"sepal_length": 5.1, "sepal_width": 3.5}',
        ready_to_summarize=True,
        **MODEL_STATE,
    )

    assert wf.route_planner_output(state) == "train_models"


def test_bare_classify_phrasing_is_not_hijacked():
    # "classify" alone is conversational-collision-prone; only explicit
    # predict/inference phrasing may divert a turn to the MTA.
    result = wf.route_planner_output(
        _state("can you classify which features matter most?", **MODEL_STATE))
    assert result != "train_models"


def test_pending_infra_request_beats_inference_detection():
    state = _state("deploy this on gcp and then predict churn", **MODEL_STATE)
    state["infrastructure_request"] = {"type": "gcp", "app_type": "python-docker"}
    assert wf.route_planner_output(state) == "provision_infra"


def test_inference_phrasing_without_a_model_does_not_route_to_mta():
    result = wf.route_planner_output(_state("predict the species for this row"))
    assert result != "train_models"


def test_ordinary_analysis_with_model_present_is_not_hijacked():
    result = wf.route_planner_output(_state("average sales per region", **MODEL_STATE))
    assert result != "train_models"


def test_ready_to_code_still_handles_unstructured_prediction_analysis():
    state = _state("predict totals per region", ready_to_code=True, plan="1. x", **MODEL_STATE)
    assert wf.route_planner_output(state) == "prepare_code"


def test_structured_inference_beats_stale_ready_to_code():
    state = _state(
        'can you run inference with {"Pclass": 0, "Sex": 0, "Age": 0}',
        ready_to_code=True,
        plan="1. stale transform plan",
        **MODEL_STATE,
    )
    assert wf.route_planner_output(state) == "train_models"


def test_named_value_inference_beats_stale_ready_to_code():
    state = _state(
        (
            "Run inference using these values: Pclass 2, male, age 125, "
            "SibSp 5, Parch 1, fare 18, embarked 2, title Master, "
            "family size 6, not alone, and cabin not known."
        ),
        ready_to_code=True,
        plan="1. stale transform plan",
        **MODEL_STATE,
    )
    assert wf.route_planner_output(state) == "train_models"


def test_enable_training_flag_still_routes_to_mta():
    assert wf.route_planner_output(_state("anything", enable_training=True)) == "train_models"


def test_bare_yes_without_service_followup_is_not_hijacked():
    state = {
        "messages": [
            HumanMessage(content="average sales per region"),
            AIMessage(content="Do you want a chart?"),
            HumanMessage(content="yes"),
        ],
        **MODEL_STATE,
    }

    assert wf.route_planner_output(state) != "train_models"


def test_inference_service_tasks_route_to_scheduler():
    start_state = {
        "task_schedule": {"task_type": "start_inference"},
        "configure_inference_service_scheduled": True,
    }
    stop_state = {
        "task_schedule": {"task_type": "stop_inference"},
        "stop_inference_service_scheduled": True,
    }

    assert wf.route_after_training(start_state) == "schedule_task"
    assert wf.route_after_training(stop_state) == "schedule_task"
