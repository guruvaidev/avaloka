"""
Integration tests for app.agents.mta_v2.agent — ModelTrainingAgent

Tests the agent's full execution paths by mocking only external dependencies
(LLM, RayTrainer, LocalTrainer, InferenceInterface) and verifying that the
correct state transitions occur end-to-end.

Scenarios:
  1. generate_training_plan flow — LLM responds with plan + AI summary
  2. ask_more_detail_to_update_training_plan flow
  3. update_training_plan flow
  4. generate_code_repo + review_code_repo + execute_training (local) — confirmation flow
  5. execute_training with no training plan — returns error message
  6. execute_training with ray_config set — schedules job (training_scheduled=True)
  7. execute_training when training_scheduled=True (already scheduled) — runs RayTrainer
  8. execute_inference — happy path
  9. execute_inference — missing training_result guard
  10. execute_inference — model_detail not found guard
  11. Exception inside execute() — returns apology message
  12. Two tool calls in one LLM response (generate_code_repo → execute_training)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides):
    base = {
        "user_id": "u1",
        "session_id": "s1",
        "messages": [HumanMessage(content="Train a neural network on iris.csv")],
        "training_plan": None,
        "training_result": None,
        "training_scheduled": False,
        "enable_training": True,
        "skip_to_training": False,
        "training_task": None,
        "data_source_location_cloud": None,
        "data_source_location_local": None,
        "data_source_location": "/tmp/iris.csv",
        "dataset_size_bytes": 1024,  # small → LocalTrainer
    }
    base.update(overrides)
    return base


def _make_local_training_plan():
    return {
        "model_type": "classification",
        "model_name": "iris_clf",
        "model_description": "Iris flower classifier",
        "model_version": "v1.0",
        "hyperparameter_config": {
            "learning_rate": 0.001,
            "epochs": 5,
            "batch_size": 32,
            "optimizer_name": "adam",
            "random_seed": 42,
            "early_stopping_patience": 5,
            "hidden_layer_sizes": [64, 32],
            "activation": "relu",
            "dropout_rate": 0.2,
            "batch_norm": True,
        },
        "ray_config": None,  # None → LocalTrainer
        "data_config": {
            "dataset_uri": "/tmp/iris.csv",
            "feature_columns": ["sepal_length", "sepal_width"],
            "target_column": "species",
        },
    }


def _make_ray_training_plan():
    plan = _make_local_training_plan()
    plan["ray_config"] = {"num_workers": 2, "use_gpu": False}
    plan["data_config"]["dataset_uri"] = "gs://bucket/iris.csv"
    return plan


def _mock_llm_with_tools(*tool_names):
    """Return a mock LLM that produces the given tool calls in one response."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.tool_calls = [{"name": name} for name in tool_names]
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def _mock_llm_plain_text(text="Here is the plan."):
    """Return mock LLM that returns a plain AIMessage (no tool calls)."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.tool_calls = []
    mock_response.content = text
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ---------------------------------------------------------------------------
# Flow 1: generate_training_plan
# ---------------------------------------------------------------------------

class TestGenerateTrainingPlanFlow:
    def test_state_has_training_plan_after_flow(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        plan = _make_local_training_plan()
        summary_llm = MagicMock()
        summary_llm.invoke.return_value = MagicMock(content="Here is your plan. Confirm?")

        agent = ModelTrainingAgent()
        state = _make_state()

        with patch("app.agents.mta_v2.agent.llm", _mock_llm_with_tools("generate_training_plan")), \
             patch.object(agent, "_extract_training_plan", return_value=plan), \
             patch("app.agents.mta_v2.agent.llm", _mock_llm_with_tools("generate_training_plan")) as mock_outer:
            # also patch the inner llm used by _generate_training_plan
            with patch("app.agents.mta_v2.agent.llm") as mock_llm:
                mock_llm.invoke.side_effect = [
                    # First call: tool selection
                    MagicMock(tool_calls=[{"name": "generate_training_plan"}]),
                    # Second call: plan summary
                    MagicMock(content="Here is your initial training plan. Confirm to proceed.", tool_calls=[]),
                ]
                with patch.object(agent, "_extract_training_plan", return_value=plan):
                    result = agent.execute(state)

        assert result.get("training_plan") == plan

    def test_ai_message_appended(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        plan = _make_local_training_plan()
        agent = ModelTrainingAgent()
        state = _make_state()

        with patch("app.agents.mta_v2.agent.llm") as mock_llm:
            mock_llm.invoke.side_effect = [
                MagicMock(tool_calls=[{"name": "generate_training_plan"}]),
                MagicMock(content="Plan generated. Awaiting confirmation.", tool_calls=[]),
            ]
            with patch.object(agent, "_extract_training_plan", return_value=plan):
                result = agent.execute(state)

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1


# ---------------------------------------------------------------------------
# Flow 2: ask_more_detail_to_update_training_plan
# ---------------------------------------------------------------------------

class TestAskMoreDetailFlow:
    def test_ai_question_appended(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        agent = ModelTrainingAgent()
        state = _make_state()

        with patch("app.agents.mta_v2.agent.llm") as mock_llm:
            mock_llm.invoke.side_effect = [
                MagicMock(tool_calls=[{"name": "ask_more_detail_to_update_training_plan"}]),
                MagicMock(content="Could you clarify the target column?", tool_calls=[]),
            ]
            result = agent.execute(state)

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert any("clarify" in m.content.lower() or "could" in m.content.lower() for m in ai_msgs)


# ---------------------------------------------------------------------------
# Flow 3: update_training_plan
# ---------------------------------------------------------------------------

class TestUpdateTrainingPlanFlow:
    def test_training_plan_updated(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        old_plan = _make_local_training_plan()
        new_plan = dict(old_plan)
        new_plan["model_name"] = "iris_v2"

        agent = ModelTrainingAgent()
        state = _make_state(training_plan=old_plan)

        with patch("app.agents.mta_v2.agent.llm") as mock_llm:
            mock_llm.invoke.side_effect = [
                MagicMock(tool_calls=[{"name": "update_training_plan"}]),
                MagicMock(content="Plan updated with new model name.", tool_calls=[]),
            ]
            with patch.object(agent, "_extract_training_plan", return_value=new_plan):
                result = agent.execute(state)

        assert result["training_plan"]["model_name"] == "iris_v2"


# ---------------------------------------------------------------------------
# Flow 4: execute_training — local (no ray_config)
# ---------------------------------------------------------------------------

class TestExecuteTrainingLocal:
    def test_training_result_stored_in_state(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        plan = _make_local_training_plan()
        mock_result = {
            "mlflow_run_id": "run-abc",
            "model_name": "iris_clf",
            "final_accuracy": 0.95,
            "final_f1_score": 0.94,
            "num_epochs_trained": 5,
        }

        agent = ModelTrainingAgent()
        state = _make_state(training_plan=plan)

        mock_local_trainer = MagicMock()
        mock_local_trainer.train.return_value = mock_result

        with patch("app.agents.mta_v2.agent.llm") as mock_llm, \
             patch("app.agents.mta_v2.agent.LocalTrainer", return_value=mock_local_trainer):
            mock_llm.invoke.side_effect = [
                MagicMock(tool_calls=[{"name": "execute_training"}]),
                MagicMock(content="Training completed. Accuracy: 95%", tool_calls=[]),
            ]
            result = agent.execute(state)

        assert result["training_result"] == mock_result
        mock_local_trainer.train.assert_called_once_with(plan)

    def test_error_when_no_training_plan(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        agent = ModelTrainingAgent()
        state = _make_state(training_plan=None)

        with patch("app.agents.mta_v2.agent.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(tool_calls=[{"name": "execute_training"}])
            result = agent.execute(state)

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "plan" in last_msg.content.lower() or "sorry" in last_msg.content.lower()


# ---------------------------------------------------------------------------
# Flow 5: execute_training — Ray path scheduling
# ---------------------------------------------------------------------------

class TestExecuteTrainingRay:
    def test_schedules_training_when_not_yet_scheduled(self):
        """First call with ray_config and training_scheduled=False → sets task_schedule."""
        from app.agents.mta_v2.agent import ModelTrainingAgent

        plan = _make_ray_training_plan()
        agent = ModelTrainingAgent()
        state = _make_state(training_plan=plan, training_scheduled=False)

        with patch("app.agents.mta_v2.agent.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(tool_calls=[{"name": "execute_training"}])
            result = agent.execute(state)

        assert result.get("training_scheduled") is True
        assert result.get("task_schedule") is not None
        assert result["task_schedule"]["task_type"] == "training"

    def test_runs_ray_trainer_when_already_scheduled(self):
        """Second call with training_scheduled=True → RayTrainer.train() is called."""
        from app.agents.mta_v2.agent import ModelTrainingAgent

        plan = _make_ray_training_plan()
        mock_result = {
            "mlflow_run_id": "ray-run-xyz",
            "dashboard_url": "http://localhost:8265",
            "final_accuracy": 0.91,
        }
        agent = ModelTrainingAgent()
        state = _make_state(training_plan=plan, training_scheduled=True)

        mock_ray_trainer = MagicMock()
        mock_ray_trainer.train.return_value = mock_result

        with patch("app.agents.mta_v2.agent.llm") as mock_llm, \
             patch("app.agents.mta_v2.agent.RayTrainer", return_value=mock_ray_trainer):
            mock_llm.invoke.return_value = MagicMock(content="Training done.", tool_calls=[])
            result = agent.execute(state)

        mock_ray_trainer.train.assert_called_once_with(plan)
        assert result["training_result"] == mock_result


# ---------------------------------------------------------------------------
# Flow 6: execute_inference
# ---------------------------------------------------------------------------

class TestExecuteInference:
    def test_inference_happy_path(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        plan = _make_local_training_plan()
        training_result = {"mlflow_run_id": "run-abc", "final_accuracy": 0.95}
        mock_model_detail = {
            "feature_names": ["sepal_length", "sepal_width"],
            "target_column": "species",
            "num_features": 2,
            "num_classes": 3,
        }
        inference_result = {"prediction": "setosa", "probabilities": {"setosa": 0.9}}

        agent = ModelTrainingAgent()
        state = _make_state(training_plan=plan, training_result=training_result)

        mock_inferencer = MagicMock()
        mock_inferencer.get_model_details.return_value = mock_model_detail
        mock_inferencer.predict.return_value = inference_result

        with patch("app.agents.mta_v2.agent.llm") as mock_llm, \
             patch("app.agents.mta_v2.agent.InferenceInterface", return_value=mock_inferencer):
            mock_llm.invoke.side_effect = [
                MagicMock(tool_calls=[{"name": "execute_inference"}]),
                MagicMock(content='{"sepal_length": 5.1, "sepal_width": 3.5}', tool_calls=[]),
                MagicMock(content="Prediction: setosa", tool_calls=[]),
            ]
            result = agent.execute(state)

        mock_inferencer.predict.assert_called_once()

    def test_inference_blocked_without_training_result(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        agent = ModelTrainingAgent()
        state = _make_state(training_result=None)

        with patch("app.agents.mta_v2.agent.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(tool_calls=[{"name": "execute_inference"}])
            result = agent.execute(state)

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "train" in last_msg.content.lower() or "sorry" in last_msg.content.lower()

    def test_inference_blocked_when_mlflow_run_id_missing(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        agent = ModelTrainingAgent()
        # training_result exists but no mlflow_run_id
        state = _make_state(training_result={"status": "ok"})

        with patch("app.agents.mta_v2.agent.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(tool_calls=[{"name": "execute_inference"}])
            result = agent.execute(state)

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)

    def test_inference_blocked_when_model_detail_not_found(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        training_result = {"mlflow_run_id": "bad-run"}
        agent = ModelTrainingAgent()
        state = _make_state(training_result=training_result)

        mock_inferencer = MagicMock()
        mock_inferencer.get_model_details.return_value = None  # not found

        with patch("app.agents.mta_v2.agent.llm") as mock_llm, \
             patch("app.agents.mta_v2.agent.InferenceInterface", return_value=mock_inferencer):
            mock_llm.invoke.return_value = MagicMock(tool_calls=[{"name": "execute_inference"}])
            result = agent.execute(state)

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "detail" in last_msg.content.lower() or "sorry" in last_msg.content.lower()


# ---------------------------------------------------------------------------
# Flow 7: Multi-tool response (generate_code_repo + execute_training)
# ---------------------------------------------------------------------------

class TestMultiToolResponse:
    def test_two_tools_called_in_sequence(self):
        """LLM returns two tool calls; both handlers should be invoked in order."""
        from app.agents.mta_v2.agent import ModelTrainingAgent

        plan = _make_local_training_plan()
        agent = ModelTrainingAgent()
        state = _make_state(training_plan=plan)

        call_order = []

        original_gcr = agent._generate_code_repo
        original_et = agent._execute_training

        def fake_gcr(s):
            call_order.append("generate_code_repo")
            return original_gcr(s)

        mock_result = {"mlflow_run_id": "run-m", "final_accuracy": 0.9}
        mock_local_trainer = MagicMock()
        mock_local_trainer.train.return_value = mock_result

        def fake_et(s):
            call_order.append("execute_training")
            return original_et(s)

        with patch("app.agents.mta_v2.agent.llm") as mock_llm, \
             patch.object(agent, "_generate_code_repo", side_effect=fake_gcr), \
             patch.object(agent, "_execute_training", side_effect=fake_et), \
             patch("app.agents.mta_v2.agent.LocalTrainer", return_value=mock_local_trainer):
            mock_llm.invoke.side_effect = [
                MagicMock(tool_calls=[
                    {"name": "generate_code_repo"},
                    {"name": "execute_training"},
                ]),
                MagicMock(content="Training completed.", tool_calls=[]),
            ]
            agent.execute(state)

        assert call_order == ["generate_code_repo", "execute_training"]


# ---------------------------------------------------------------------------
# Flow 8: Exception handling inside execute()
# ---------------------------------------------------------------------------

class TestExecuteExceptionHandling:
    def test_exception_returns_apology_message(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        agent = ModelTrainingAgent()
        state = _make_state()

        with patch("app.agents.mta_v2.agent.llm") as mock_llm:
            mock_llm.invoke.side_effect = RuntimeError("Unexpected LLM error")
            result = agent.execute(state)

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "sorry" in last_msg.content.lower() or "wrong" in last_msg.content.lower()

    def test_state_training_flags_cleared_on_exception(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        agent = ModelTrainingAgent()
        state = _make_state(enable_training=True, skip_to_training=True)

        with patch("app.agents.mta_v2.agent.llm") as mock_llm:
            mock_llm.invoke.side_effect = RuntimeError("boom")
            result = agent.execute(state)

        assert result["enable_training"] is False
        assert result["skip_to_training"] is False


# ---------------------------------------------------------------------------
# Misc: tool_calls from response attr variants
# ---------------------------------------------------------------------------

class TestToolCallAttrVariants:
    def test_tool_call_singular_attr(self):
        """Agent handles responses with 'tool_call' (singular) attribute."""
        from app.agents.mta_v2.agent import ModelTrainingAgent

        plan = _make_local_training_plan()
        agent = ModelTrainingAgent()
        state = _make_state(training_plan=plan)

        # Response with 'tool_call' (singular), not 'tool_calls' (plural)
        mock_response = MagicMock(spec=[])  # no attributes by default
        mock_response.tool_call = {"name": "generate_code_repo"}

        with patch("app.agents.mta_v2.agent.llm") as mock_llm, \
             patch.object(agent, "_generate_code_repo", return_value=state.copy()) as mock_gcr:
            mock_llm.invoke.return_value = mock_response
            agent.execute(state)

        mock_gcr.assert_called_once()
