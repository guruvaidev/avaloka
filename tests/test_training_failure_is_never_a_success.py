"""A failed step is never followed by a success claim, and a lost upload says so.

From the 2.0 web e2e: a training turn whose trainer died ("No such file", then
skops' untrusted types) showed "Training stopped because of an unclassified
error" and, right under it, the execution prompt "I trained Stroke Prediction
Model to predict `stroke`." -- then "That's my answer to ...". And the "No
such file" itself was a session input under TMP_ROOT deleted by recreating
the API container: the user needed to hear "upload it again", not "retry".
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
from langchain_core.messages import HumanMessage

from app.api import chat_voice
from app.agents.mta_v2.failure_diagnostics import build_training_failure

PLAN = {"model_name": "Stroke Prediction Model", "model_type": "classification",
        "data_config": {"target_column": "stroke"}}


def _failed_final(error: str) -> dict:
    return {
        "training_plan": PLAN,
        "training_completed": False,
        "training_result": {"status": "error", "error": error,
                            "failure": build_training_failure(error)},
    }


def _steps(final):
    return chat_voice.prompts_after_turn(final, question="Train a model to predict stroke")


def test_a_failed_training_turn_says_it_failed_and_claims_nothing():
    prompts = _steps(_failed_final(
        "[Errno 2] No such file or directory: '/var/tmp/avaloka/s1/input_1.csv'"))
    text = " ".join(p["message"] for p in prompts)
    assert "I trained" not in text
    assert "That's my answer" not in text
    assert prompts[-1]["message"].startswith("I don't have an answer")
    ex = next(p for p in prompts if p["step"] == "execution")
    assert ex["message"].startswith("I couldn't train a classification model to predict `stroke`")
    assert "/var/tmp" not in ex["message"]               # raw error text holds paths
    assert ex["quick_replies"][0]["send"] == "Retry training"


def test_a_result_without_a_classified_failure_is_still_a_failure():
    final = {"training_plan": PLAN, "training_result": {"status": "error", "error": "boom"}}
    ex = next(p for p in _steps(final) if p["step"] == "execution")
    assert "I trained" not in ex["message"] and "couldn't train" in ex["message"]


def test_a_successful_training_turn_still_says_what_it_trained():
    final = {"training_plan": PLAN, "training_completed": True,
             "training_result": {"status": "success", "mlflow_run_id": "r1"},
             "evaluation_beats_baseline": True}
    prompts = _steps(final)
    ex = next(p for p in prompts if p["step"] == "execution")
    assert ex["message"].startswith("I trained a classification model to predict `stroke`.")
    # The name is the training planner model's words (c-strict).
    assert ex["message"].endswith("My planner named it Stroke Prediction Model. (commentary — not checked)")
    assert prompts[-1]["step"] == "result"


def test_a_failed_or_blocked_code_run_is_not_followed_by_an_answer():
    """She still offers a next step -- but not as "That's my answer"."""
    for final in (
        {"execution_result": {"status": "failed", "message": "KeyError: 'x'"}},
        {"contract_violations": ["writes outside the work dir"]},
        _failed_final("boom"),
    ):
        prompts = {p["step"]: p for p in chat_voice.prompts_after_turn(final, question="avg billing")}
        assert "execution" in prompts, final
        result = prompts["result"]["message"]
        assert "That's my answer" not in result, (final, result)
        # The stable part only: other fixes extend the sentence after "yet".
        assert result.startswith("I don't have an answer to “avg billing” yet"), result


# ------------------------------------------------------------- lost upload
def _state(dataset_uri: str) -> dict:
    plan = {
        "model_type": "classification", "model_name": "Stroke Prediction Model",
        "hyperparameter_config": {"epochs": 1}, "ray_config": None,
        "execution_backend": "local", "execution_dataset_uri": dataset_uri,
        "data_config": {"dataset_uri": dataset_uri, "feature_columns": ["age", "bmi"],
                        "target_column": "stroke"},
    }
    return {
        "user_id": "u", "session_id": "s", "messages": [HumanMessage(content="train")],
        "training_plan": plan, "training_result": None, "training_scheduled": False,
        "enable_training": True, "skip_to_training": False, "training_task": None,
        "data_source_location_cloud": None, "data_source_location_local": dataset_uri,
        "data_source_location": dataset_uri, "dataset_size_bytes": 100,
        # What Redis still held after the container was recreated: the schema,
        # so the column check passed without the file.
        "uploaded_csv_columns": ["age", "bmi", "stroke"],
    }


def test_a_session_input_deleted_by_a_restart_asks_for_the_upload_again(tmp_path):
    from app.agents.mta_v2.agent import ModelTrainingAgent

    gone = tmp_path / "var-tmp-avaloka" / "s1" / "null_words_three_sheets [2022].csv"
    trainer = MagicMock()
    with patch("app.agents.mta_v2.agent.LocalTrainer", return_value=trainer):
        result = ModelTrainingAgent()._execute_training(_state(str(gone)))

    trainer.train.assert_not_called()                    # no doomed run, no unclassified error
    message = result["messages"][-1].content
    assert "is no longer on the server" in message
    assert "upload the file again" in message
    assert "null_words_three_sheets [2022].csv" in message
    assert str(tmp_path) not in message                  # the name, not the server path
    assert result["training_completed"] is False


def test_an_input_that_is_there_trains_normally(tmp_path):
    """The chat turn enqueues the training (it never trains in the request);
    the worker's pass over the same state then trains on the file. Neither
    pass reports the present input as lost or the run as failed."""
    from app.agents.mta_v2.agent import ModelTrainingAgent
    from app.core import ml_store

    present = tmp_path / "sheets [2022].csv"             # brackets are not a glob here
    pd.DataFrame({"age": [1, 2, 3, 4], "bmi": [1.0, 2.0, 3.0, 4.0],
                  "stroke": [0, 1, 0, 1]}).to_csv(present, index=False)
    trainer = MagicMock()
    trainer.train.return_value = {"mlflow_run_id": "r1", "final_accuracy": 0.9}
    store = ml_store.InMemoryMLStore()
    ml_store.set_ml_store(store)
    try:
        with patch("app.agents.mta_v2.agent.LocalTrainer", return_value=trainer):
            state = _state(str(present))
            before = len(state["messages"])
            queued = ModelTrainingAgent()._execute_training(state)

            # The chat turn: a queued job, no training, nothing said about a failure.
            trainer.train.assert_not_called()
            assert queued["training_scheduled"] is True
            assert (queued.get("task_schedule") or {}).get("task_type") == "training"
            assert len(queued["messages"]) == before
            assert not queued.get("agent_errors")
            job = store.get_job(queued["ml_job_id"])
            assert (job.kind, job.status) == ("training", "queued")

            # The worker's pass: it trains on the file that is there.
            result = ModelTrainingAgent()._execute_training(queued)
        trainer.train.assert_called_once()
        assert trainer.train.call_args.args[0]["data_config"]["dataset_uri"] == str(present)
        assert result["training_completed"] is True
        message = result["messages"][-1].content
        assert "is no longer on the server" not in message
        assert "couldn't train" not in message
        assert store.get_job(queued["ml_job_id"]).status == "succeeded"
    finally:
        ml_store.set_ml_store(None)
