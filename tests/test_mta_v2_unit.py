"""
Unit tests for app.agents.mta_v2.agent — ModelTrainingAgent

Covers:
- parse_tool_call  (string, nested-dict, flat-dict, TypeError)
- _normalize_messages  (BaseMessage pass-through, dict variants, fallback)
- _is_local_data_source  (local path, small file, remote schemes)
- _finish_mta  (state flag reset)
- _generate_code_repo / _review_code_repo  (no-op stubs)
- execute()  when LLM is disabled (no api key)
- execute()  when LLM returns no tool_calls
- execute()  routes each known tool name to the correct handler
- TOOLS list completeness
- SYSTEM_PROMPT contains all tool names
"""

from __future__ import annotations

import json
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Helpers: build a minimal ETLState
# ---------------------------------------------------------------------------

def _make_state(**overrides):
    base = {
        "user_id": "user-1",
        "session_id": "session-1",
        "messages": [HumanMessage(content="Train a classifier on iris.csv")],
        "training_plan": None,
        "training_result": None,
        "training_scheduled": False,
        "enable_training": True,
        "skip_to_training": False,
        "training_task": None,
        "task_schedule": None,
        "data_source_location_cloud": None,
        "data_source_location_local": None,
        "data_source_location": None,
        "dataset_size_bytes": None,
    }
    base.update(overrides)
    return base


def _sample_training_plan(**overrides):
    plan = {
        "model_type": "classification",
        "model_name": "my_model",
        "model_description": "Predict passenger survival.",
        "model_version": "v1.0",
        "hyperparameter_config": {
            "learning_rate": 0.001,
            "epochs": 30,
            "batch_size": 32,
            "optimizer_name": "adam",
            "random_seed": 42,
            "early_stopping_patience": 5,
            "hidden_layer_sizes": [128, 64],
            "activation": "relu",
            "dropout_rate": 0.2,
            "batch_norm": True,
        },
        "ray_config": None,
        "data_config": {
            "dataset_uri": "/tmp/train.csv",
            "feature_columns": [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked",
                "Title",
                "FamilySize",
                "IsAlone",
                "CabinKnown",
            ],
            "target_column": "Survived",
        },
    }
    plan.update(overrides)
    return plan


# ---------------------------------------------------------------------------
# Import agent under test (patches module-level LLM so no real API key needed)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_llm():
    """Disable the module-level LLM so every test starts with a known mock."""
    with patch("app.agents.mta_v2.agent.llm", None):
        yield


# ---------------------------------------------------------------------------
# Tests: constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_tools_list_has_expected_names(self):
        from app.agents.mta_v2.agent import TOOLS
        names = {t["function"]["name"] for t in TOOLS}
        assert "generate_training_plan" in names
        assert "update_training_plan" in names
        assert "ask_more_detail_to_update_training_plan" in names
        assert "generate_code_repo" in names
        assert "review_code_repo" in names
        assert "execute_training" in names
        assert "execute_inference" in names

    def test_system_prompt_contains_all_tool_names(self):
        from app.agents.mta_v2.agent import SYSTEM_PROMPT, TOOLS
        for tool in TOOLS:
            assert tool["function"]["name"] in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tests: parse_tool_call
# ---------------------------------------------------------------------------

class TestParseToolCall:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def test_string_input_returned_as_is(self):
        assert self.agent.parse_tool_call("generate_training_plan") == "generate_training_plan"

    def test_nested_dict_with_function_key(self):
        tool_call = {"function": {"name": "execute_training"}}
        assert self.agent.parse_tool_call(tool_call) == "execute_training"

    def test_flat_dict_with_name_key(self):
        tool_call = {"name": "execute_inference"}
        assert self.agent.parse_tool_call(tool_call) == "execute_inference"

    def test_unsupported_type_raises_type_error(self):
        with pytest.raises(TypeError):
            self.agent.parse_tool_call(12345)

    def test_none_name_in_nested_dict(self):
        tool_call = {"function": {}}
        result = self.agent.parse_tool_call(tool_call)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _normalize_messages
# ---------------------------------------------------------------------------

class TestNormalizeMessages:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def test_passthrough_base_message(self):
        msg = HumanMessage(content="hello")
        result = self.agent._normalize_messages([msg])
        assert result == [msg]

    def test_dict_human_role(self):
        result = self.agent._normalize_messages([{"role": "user", "content": "hi"}])
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "hi"

    def test_dict_ai_role(self):
        result = self.agent._normalize_messages([{"role": "assistant", "content": "ok"}])
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "ok"

    def test_dict_type_human(self):
        result = self.agent._normalize_messages([{"type": "human", "content": "hello"}])
        assert isinstance(result[0], HumanMessage)

    def test_dict_type_ai(self):
        result = self.agent._normalize_messages([{"type": "ai", "content": "response"}])
        assert isinstance(result[0], AIMessage)

    def test_dict_unknown_role_falls_back_to_human(self):
        result = self.agent._normalize_messages([{"role": "system", "content": "sys"}])
        assert isinstance(result[0], HumanMessage)

    def test_list_content_with_text_field(self):
        result = self.agent._normalize_messages([
            {"role": "user", "content": [{"text": "hello from list"}]}
        ])
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "hello from list"

    def test_list_content_plain_strings(self):
        result = self.agent._normalize_messages([
            {"role": "user", "content": ["first string"]}
        ])
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "first string"

    def test_non_dict_non_message_coerced_to_human(self):
        result = self.agent._normalize_messages([42])
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "42"

    def test_mixed_list(self):
        msgs = [
            HumanMessage(content="a"),
            {"role": "assistant", "content": "b"},
            "not a dict",
        ]
        result = self.agent._normalize_messages(msgs)
        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert isinstance(result[2], HumanMessage)

    def test_empty_list(self):
        assert self.agent._normalize_messages([]) == []


# ---------------------------------------------------------------------------
# Tests: training plan markdown
# ---------------------------------------------------------------------------

class TestTrainingPlanMarkdown:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def test_default_training_plan_is_compact(self):
        content = self.agent._format_training_plan_response(_sample_training_plan())

        assert content.startswith("## Training Plan Ready")
        assert "**Model:** Classification (`my_model`, v1.0)" in content
        assert "**Target:** `Survived`" in content
        assert "**Features:** 11 selected columns" in content
        assert "**Training:** 30 epochs, Adam optimizer, learning rate `0.001`, batch size `32`" in content
        assert "Training has not started." in content
        assert "**Start training with this plan, or describe any changes you want to make.**" in content
        assert "Dataset URI" not in content
        assert "Feature Columns" not in content
        assert "Execution" not in content

    def test_full_training_plan_includes_all_details(self):
        content = self.agent._format_training_plan_response(_sample_training_plan(), full=True)

        assert content.startswith("## Full Training Plan")
        assert "- **Trainer / Architecture:** DynamicMLP (PyTorch)" in content
        assert "- **Dataset URI:** `/tmp/train.csv`" in content
        assert "- **Feature Columns:** ['Pclass', 'Sex', 'Age'" in content
        assert "- **Hidden Layer Sizes:** [128, 64]" in content
        assert ".pth" in content
        assert ".onnx" in content
        assert "**Start training with this plan, or describe any changes you want to make.**" in content
        assert "XGBoost" not in content
        assert "Gradient Boosting" not in content
        assert "grid search" not in content.lower()
        assert ".pkl" not in content
        assert "Execution Backend" not in content
        assert "Execution Decision" not in content

    def test_successful_training_report_points_to_configurations(self):
        content = self.agent._format_training_result_report(
            {
                "status": "success",
                "mlflow_run_id": "run-123",
                "model_name": "my_model",
                "model_type": "classification",
            },
            use_local=False,
        )

        assert "Configurations tab" in content
        assert "run-123" in content

    def test_failed_training_report_shows_actions_without_diagnostics(self):
        content = self.agent._format_training_result_report(
            {
                "status": "error",
                "error": "ImagePullBackOff: secret registry details",
            },
            use_local=False,
        )

        assert "Model training could not be completed." in content
        assert "IMAGE_PULL_FAILED" in content
        assert "configured training image name and tag" in content
        assert "ImagePullBackOff" not in content
        assert "secret registry details" not in content

    def test_full_updated_training_plan_after_edit_uses_actual_plan(self):
        state = _make_state(
            messages=[HumanMessage(content="Change epochs to 10 and show the full training plan")],
            training_plan=_sample_training_plan(),
            training_completed=False,
            training_plan_reply_action="update",
        )

        result = self.agent.execute(state)

        content = result["messages"][-1].content
        assert content.startswith("## Full Updated Training Plan")
        assert "- **Epochs:** 10" in content
        assert "- **Trainer / Architecture:** DynamicMLP (PyTorch)" in content
        assert "XGBoost" not in content
        assert ".pkl" not in content

    def test_execute_returns_full_plan_without_training(self):
        state = _make_state(
            messages=[HumanMessage(content="Can I see the full training plan?")],
            training_plan=_sample_training_plan(),
            training_completed=False,
            training_plan_reply_action="unclear",
        )

        with patch.object(self.agent, "_execute_training") as execute_training, \
             patch("app.agents.mta_v2.agent.llm") as mock_llm:
            result = self.agent.execute(state)

        execute_training.assert_not_called()
        mock_llm.invoke.assert_not_called()
        assert result["messages"][-1].content.startswith("## Full Training Plan")


# ---------------------------------------------------------------------------
# Tests: inference JSON extraction and validation
# ---------------------------------------------------------------------------

class TestInferenceInput:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def _titanic_feature_names(self):
        return [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Title",
            "FamilySize",
            "IsAlone",
            "CabinKnown",
        ]

    def test_extracts_exact_json_from_chat_prompt(self):
        prompt = (
            'can you run inference with {"Pclass": 0, "Sex": 0, "Age": 0}'
            "\n\n[Analysis context] ignored"
        )

        assert self.agent._extract_inference_json(prompt) == {
            "Pclass": 0,
            "Sex": 0,
            "Age": 0,
        }

    def test_extracts_named_titanic_inference_values(self):
        prompt = (
            "Run inference using these values: Pclass 2, male, age 125, "
            "SibSp 5, Parch 1, fare 18, embarked 2, title Master, "
            "family size 6, not alone, and cabin not known."
        )

        assert self.agent._extract_named_inference_values(prompt, self._titanic_feature_names()) == {
            "Pclass": 2,
            "Sex": "male",
            "Age": 125,
            "SibSp": 5,
            "Parch": 1,
            "Fare": 18,
            "Embarked": 2,
            "Title": "Master",
            "FamilySize": 6,
            "IsAlone": 0,
            "CabinKnown": 0,
        }

    def test_extracts_repeated_named_inference_requests_as_batch(self):
        prompt = (
            "Run inference with: Pclass=1 Sex=female Age=29 SibSp=0 Parch=0 Embarked=S  "
            "Run inference with: Pclass=3 Sex=female Age=12 SibSp=0 Parch=0 Embarked=S"
        )

        assert self.agent._extract_named_inference_values(
            prompt,
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"],
        ) == [
            {
                "Pclass": 1,
                "Sex": "female",
                "Age": 29,
                "SibSp": 0,
                "Parch": 0,
                "Embarked": "S",
            },
            {
                "Pclass": 3,
                "Sex": "female",
                "Age": 12,
                "SibSp": 0,
                "Parch": 0,
                "Embarked": "S",
            },
        ]

    def test_accepts_batch_json(self):
        payload = self.agent._extract_inference_json(
            'predict [{"f1": 1, "f2": 2}, {"f1": 3, "f2": 4}]'
        )

        assert len(payload) == 2
        assert self.agent._inference_input_error(payload, ["f1", "f2"]) is None

    def test_allows_missing_model_features(self):
        error = self.agent._inference_input_error({"f1": 1}, ["f1", "f2"])

        assert error is None

    def test_full_inference_without_values_asks_for_sample_row(self):
        state = _make_state(
            messages=[HumanMessage(content="can you give me the full inference")],
            training_result={"mlflow_run_id": "run-123"},
            training_completed=True,
        )
        inferencer = MagicMock()
        inferencer.get_model_details.return_value = {"feature_names": ["f1", "f2"]}

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.llm", None):
            result = self.agent.execute(state)

        inferencer.predict.assert_not_called()
        assert result["task_schedule"] is None
        assert "Please send one row" in result["messages"][-1].content
        assert "f1, f2" in result["messages"][-1].content

    def test_batch_json_uses_local_predict_batch(self):
        payload = [{"f1": 1}, {"f1": 2}]
        state = _make_state(
            messages=[HumanMessage(content=f"run inference with {json.dumps(payload)}")],
            training_result={"mlflow_run_id": "run-123"},
        )
        inferencer = MagicMock()
        inferencer.get_model_details.return_value = {"feature_names": ["f1"]}
        inferencer.predict_batch.return_value = [
            {"prediction": "0"},
            {"prediction": "1"},
        ]

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.llm", None):
            result = self.agent._execute_inference(state)

        inferencer.predict_batch.assert_called_once_with(
            mlflow_run_id="run-123",
            rows=payload,
        )
        assert result["messages"][-1].content == (
            "Inference completed for 2 rows:\n"
            "- Row 1: Prediction: 0.\n"
            "- Row 2: Prediction: 1."
        )

    def test_batch_json_shows_each_rows_prediction_and_confidence(self):
        payload = [{"f1": 1}, {"f1": 2}]
        state = _make_state(
            messages=[HumanMessage(content=f"run inference with {json.dumps(payload)}")],
            training_result={"mlflow_run_id": "run-123"},
        )
        inferencer = MagicMock()
        inferencer.get_model_details.return_value = {"feature_names": ["f1"]}
        inferencer.predict_batch.return_value = [
            {"prediction": "0", "probabilities": {"0": 0.8, "1": 0.2}},
            {"prediction": "1", "probabilities": {"0": 0.1, "1": 0.9}},
        ]

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.llm", None):
            result = self.agent._execute_inference(state)

        assert result["messages"][-1].content == (
            "Inference completed for 2 rows:\n"
            "- Row 1: Prediction: 0. Confidence: 80%.\n"
            "- Row 2: Prediction: 1. Confidence: 90%."
        )

    def test_dataset_uri_uses_local_dataset_inference_without_service_followup(self):
        payload = {"dataset_uri": "/tmp/input.csv"}
        state = _make_state(
            messages=[HumanMessage(content=f"run inference with {json.dumps(payload)}")],
            training_result={"mlflow_run_id": "run-123"},
        )
        inferencer = MagicMock()
        inferencer.get_model_details.return_value = {"feature_names": ["f1"]}
        inferencer.inference_local.return_value = {
            "output_uri": "gs://bucket/predictions.csv"
        }

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.llm", None):
            result = self.agent._execute_inference(state)

        inferencer.inference_local.assert_called_once_with(
            mlflow_run_id="run-123",
            dataset_uri="/tmp/input.csv",
        )
        assert result["messages"][-1].content == (
            "Inference completed. Predictions were saved to `gs://bucket/predictions.csv`."
        )

    def test_numeric_string_json_values_are_sent_as_encoded_numbers(self):
        payload = {
            "Pclass": 2,
            "Sex": "1",
            "Age": 48,
            "SibSp": 0,
            "Parch": 1,
            "Fare": 18,
            "Embarked": "0",
            "Title": 0,
            "FamilySize": 0,
            "IsAlone": 0,
            "CabinKnown": 0,
        }
        expected_payload = {
            **payload,
            "Sex": 1,
            "Embarked": 0,
        }
        state = _make_state(
            messages=[HumanMessage(content=f"can you run inference with {json.dumps(payload)}")],
            training_result={"mlflow_run_id": "run-123"},
        )
        inferencer = MagicMock()
        inferencer.get_model_details.return_value = {"feature_names": list(payload)}
        inferencer.predict.return_value = {"prediction": "1", "probabilities": {"1": 0.95}}
        summary_llm = MagicMock()
        summary_llm.invoke.return_value = MagicMock(content="Prediction: 1")

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.llm", summary_llm):
            result = self.agent._execute_inference(state)

        inferencer.predict.assert_called_once_with(mlflow_run_id="run-123", row=expected_payload)
        summary_llm.invoke.assert_not_called()
        assert result["messages"][-1].content == "Prediction: 1. Confidence: 95%."

    def test_partial_json_payload_is_sent_to_local_inference(self):
        payload = {
            "Pclass": 2,
            "Sex": "male",
            "Age": 48,
        }
        state = _make_state(
            messages=[HumanMessage(content=f"can you run inference with {json.dumps(payload)}")],
            training_result={"mlflow_run_id": "run-123"},
        )
        inferencer = MagicMock()
        inferencer.get_model_details.return_value = {
            "feature_names": ["Pclass", "Sex", "Age", "Fare", "Embarked"],
            "target_column": "Survived",
            "class_names": ["0", "1"],
        }
        inferencer.predict.return_value = {"prediction": "0", "probabilities": {"0": 0.87}}
        summary_llm = MagicMock()

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.llm", summary_llm):
            result = self.agent._execute_inference(state)

        inferencer.predict.assert_called_once_with(mlflow_run_id="run-123", row=payload)
        summary_llm.invoke.assert_not_called()
        assert result["messages"][-1].content == (
            "Prediction: Survived = 0 (did not survive). Confidence: 87%."
        )

    def test_chat_json_is_sent_unchanged_to_local_inference(self):
        payload = {
            "Pclass": 0,
            "Sex": 0,
            "Age": 0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 0,
            "Embarked": 0,
            "Title": 0,
            "FamilySize": 0,
            "IsAlone": 0,
            "CabinKnown": 0,
        }
        state = _make_state(
            messages=[HumanMessage(content=f"can you run inference with {json.dumps(payload)}")],
            training_result={"mlflow_run_id": "run-123"},
        )
        model_detail = {"feature_names": list(payload)}
        inference_result = {"prediction": "1", "probabilities": {"1": 0.99}}

        inferencer = MagicMock()
        inferencer.get_model_details.return_value = model_detail
        inferencer.predict.return_value = inference_result
        summary_llm = MagicMock()
        summary_llm.invoke.return_value = MagicMock(content="Prediction: 1")

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.llm", summary_llm):
            result = self.agent._execute_inference(state)

        inferencer.predict.assert_called_once_with(mlflow_run_id="run-123", row=payload)
        summary_llm.invoke.assert_not_called()
        assert result["messages"][-1].content == "Prediction: 1. Confidence: 99%."

    def test_planner_format_warning_is_removed_before_inference_summary(self):
        payload = {"f1": "1"}
        planner_warning = (
            "The inference request contains fields that do not match the expected "
            "format. Please provide the values again."
        )
        state = _make_state(
            messages=[
                HumanMessage(content=f"can you run inference with {json.dumps(payload)}"),
                AIMessage(content=planner_warning),
            ],
            training_result={"mlflow_run_id": "run-123"},
        )
        inferencer = MagicMock()
        inferencer.get_model_details.return_value = {"feature_names": ["f1"]}
        inferencer.predict.return_value = {"prediction": "1"}
        summary_llm = MagicMock()
        summary_llm.invoke.return_value = MagicMock(content="Prediction: 1")

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.llm", summary_llm):
            result = self.agent._execute_inference(state)

        inferencer.predict.assert_called_once_with(mlflow_run_id="run-123", row={"f1": 1})
        summary_llm.invoke.assert_not_called()
        assert all(getattr(message, "content", None) != planner_warning for message in result["messages"])
        assert result["messages"][-1].content == "Prediction: 1."

    def test_named_value_prompt_is_sent_to_local_inference(self):
        expected_payload = {
            "Pclass": 2,
            "Sex": "male",
            "Age": 125,
            "SibSp": 5,
            "Parch": 1,
            "Fare": 18,
            "Embarked": 2,
            "Title": "Master",
            "FamilySize": 6,
            "IsAlone": 0,
            "CabinKnown": 0,
        }
        state = _make_state(
            messages=[
                HumanMessage(
                    content=(
                        "Run inference using these values: Pclass 2, male, age 125, "
                        "SibSp 5, Parch 1, fare 18, embarked 2, title Master, "
                        "family size 6, not alone, and cabin not known."
                    )
                )
            ],
            training_result={"mlflow_run_id": "run-123"},
        )
        inferencer = MagicMock()
        inferencer.get_model_details.return_value = {
            "feature_names": self._titanic_feature_names(),
            "target_column": "Survived",
            "class_names": ["0", "1"],
        }
        inferencer.predict.return_value = {"prediction": "0", "probabilities": {"0": 0.98}}
        summary_llm = MagicMock()
        summary_llm.invoke.return_value = MagicMock(content="Prediction: 0")

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.llm", summary_llm):
            result = self.agent._execute_inference(state)

        inferencer.predict.assert_called_once_with(mlflow_run_id="run-123", row=expected_payload)
        summary_llm.invoke.assert_not_called()
        assert result["messages"][-1].content == (
            "Prediction: Survived = 0 (did not survive). Confidence: 98%."
        )

    def test_repeated_named_value_prompts_use_batch_inference_and_show_both_results(self):
        rows = [
            {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Embarked": "S"},
            {"Pclass": 3, "Sex": "female", "Age": 12, "SibSp": 0, "Parch": 0, "Embarked": "S"},
        ]
        state = _make_state(
            messages=[
                HumanMessage(
                    content=(
                        "Run inference with: Pclass=1 Sex=female Age=29 SibSp=0 Parch=0 Embarked=S  "
                        "Run inference with: Pclass=3 Sex=female Age=12 SibSp=0 Parch=0 Embarked=S"
                    )
                )
            ],
            training_result={"mlflow_run_id": "run-123"},
        )
        inferencer = MagicMock()
        inferencer.get_model_details.return_value = {
            "feature_names": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"],
            "target_column": "Fare",
        }
        inferencer.predict_batch.return_value = [
            {"prediction": 57.924725},
            {"prediction": 0.476585},
        ]

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.llm", None):
            result = self.agent._execute_inference(state)

        inferencer.predict_batch.assert_called_once_with(
            mlflow_run_id="run-123",
            rows=rows,
        )
        assert result["messages"][-1].content == (
            "Inference completed for 2 rows:\n"
            "- Row 1: Prediction: Fare = 57.924725.\n"
            "- Row 2: Prediction: Fare = 0.476585."
        )

    def test_compact_summary_decodes_class_index_from_model_metadata(self):
        summary = self.agent._format_compact_inference_summary(
            {
                "predicted_class_index": 1,
                "probabilities": {"0": 0.2, "1": 0.8},
            },
            {
                "target_column": "Survived",
                "class_names": ["0", "1"],
            },
        )

        assert summary == "Prediction: Survived = 1 (survived). Confidence: 80%."

    def test_detailed_inference_request_uses_expanded_report(self):
        payload = {"f1": 1}
        state = _make_state(
            messages=[HumanMessage(content=f"Give me a full inference report for {json.dumps(payload)}")],
            training_result={"mlflow_run_id": "run-123"},
        )
        inferencer = MagicMock()
        inferencer.get_model_details.return_value = {"feature_names": ["f1"]}
        inferencer.predict.return_value = {
            "prediction": "1",
            "raw_output": [2.0, -1.0],
            "probabilities": {"0": 0.1, "1": 0.9},
        }
        summary_llm = MagicMock()
        summary_llm.invoke.return_value = MagicMock(content="Detailed inference report")

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.llm", summary_llm):
            result = self.agent._execute_inference(state)

        inferencer.predict.assert_called_once_with(mlflow_run_id="run-123", row=payload)
        summary_llm.invoke.assert_called_once()
        assert result["messages"][-1].content == "Detailed inference report"

    def test_execute_routes_inference_payload_without_scheduling(self):
        state = _make_state(
            messages=[HumanMessage(content='Can you run inference with {"f1": 1}')],
            training_result={"mlflow_run_id": "run-123"},
            training_completed=True,
            task_schedule={
                "task_type": "execute",
                "schedule_type": "relative",
                "second": 0,
            },
        )

        with patch.object(self.agent, "_execute_inference", return_value=state) as mock_execute, \
             patch("app.agents.mta_v2.agent.llm") as mock_llm:
            result = self.agent.execute(state)

        mock_execute.assert_called_once_with(state)
        mock_llm.invoke.assert_not_called()
        assert result["task_schedule"] is None
        assert result["inference_scheduled"] is False

    def test_execute_routes_named_value_inference_without_tool_selection(self):
        state = _make_state(
            messages=[
                HumanMessage(
                    content=(
                        "Run inference using these values: Pclass 2, male, age 125, "
                        "SibSp 5, Parch 1, fare 18, embarked 2, title Master, "
                        "family size 6, not alone, and cabin not known."
                    )
                )
            ],
            training_result={"mlflow_run_id": "run-123"},
            training_completed=True,
        )

        with patch.object(self.agent, "_execute_inference", return_value=state) as mock_execute, \
             patch("app.agents.mta_v2.agent.llm") as mock_llm:
            result = self.agent.execute(state)

        mock_execute.assert_called_once_with(state)
        mock_llm.invoke.assert_not_called()
        assert result["task_schedule"] is None

    def test_execute_inference_uses_local_predict(self):
        payload = {"f1": 1}
        state = _make_state(
            messages=[HumanMessage(content=f"predict with {json.dumps(payload)}")],
            training_result={"mlflow_run_id": "run-123"},
        )
        inferencer = MagicMock()
        inferencer.get_model_details.return_value = {"feature_names": ["f1"]}
        inferencer.predict.return_value = {"prediction": "1"}
        summary_llm = MagicMock()
        summary_llm.invoke.return_value = MagicMock(content="Prediction: 1")

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.llm", summary_llm):
            result = self.agent._execute_inference(state)

        inferencer.predict.assert_called_once_with(mlflow_run_id="run-123", row=payload)
        summary_llm.invoke.assert_not_called()
        assert result["messages"][-1].content == "Prediction: 1."

    def test_execute_inference_uses_configured_gateway_metadata(self):
        payload = {"f1": 1}
        state = _make_state(
            messages=[HumanMessage(content=f"predict with {json.dumps(payload)}")],
            training_result={"mlflow_run_id": "run-123"},
        )
        inferencer = MagicMock()
        inferencer.get_model_details.return_value = {
            "feature_names": ["f1"],
            "inference_service_details": {
                "gateway_url": "https://gateway.example.dev"
            },
        }
        service_manager = MagicMock()
        service_manager.inference.return_value = {"prediction": "1"}

        with patch("app.agents.mta_v2.agent.InferenceInterface", return_value=inferencer), \
             patch("app.agents.mta_v2.agent.InferenceServiceManager", return_value=service_manager), \
             patch("app.agents.mta_v2.agent.llm", None):
            result = self.agent._execute_inference(state)

        service_manager.inference.assert_called_once_with(
            mlflow_run_id="run-123",
            input_data=payload,
        )
        inferencer.predict.assert_not_called()
        assert result["messages"][-1].content == "Prediction: 1."

    def test_gateway_tools_are_exposed(self):
        from app.agents.mta_v2.agent import SYSTEM_PROMPT, TOOLS

        tool_names = {tool["function"]["name"] for tool in TOOLS}
        assert "configure_inference_service" in tool_names
        assert "stop_inference_service" in tool_names
        assert "configured service" in SYSTEM_PROMPT

    def test_configure_inference_service_schedules_then_configures(self):
        state = _make_state(
            messages=[HumanMessage(content="configure the inference service")],
            training_result={"mlflow_run_id": "run-123"},
        )

        scheduled = self.agent._configure_inference_service(state)
        assert scheduled["configure_inference_service_scheduled"] is True
        assert scheduled["task_schedule"]["task_type"] == "start_inference"

        manager = MagicMock()
        manager.configure_inference_service.return_value = {
            "backend": "rayserve",
            "endpoint": "http://avaloka-inference-serve-svc:8000",
        }
        with patch("app.agents.mta_v2.agent.InferenceServiceManager", return_value=manager):
            configured = self.agent._configure_inference_service(scheduled)

        manager.configure_inference_service.assert_called_once_with("run-123")
        assert configured["configure_inference_service_scheduled"] is False
        assert configured["inference_service_details"]["backend"] == "rayserve"


class TestInferenceEncoding:
    def test_categorical_labels_and_encoded_numeric_values_are_allowed(self):
        import pandas as pd
        from app.agents.mta_v2.inference import InferenceInterface
        from app.agents.mta_v2.model import ModelConfig

        cfg = ModelConfig(
            input_size=1,
            output_size=2,
            hidden_sizes=[4],
            feature_names=["Sex"],
            preprocessing={
                "categorical_features": {
                    "Sex": {
                        "mapping": {"female": 0, "male": 1},
                        "unknown_value": -1,
                    }
                }
            },
        )

        assert InferenceInterface._encode_feature_value("Sex", "male", cfg) == 1.0
        assert InferenceInterface._encode_feature_value("Sex", "1", cfg) == 1.0
        assert InferenceInterface._encode_feature_value("Sex", 1, cfg) == 1.0
        assert InferenceInterface._encode_feature_value("Sex", "unknown", cfg) == -1.0

        prepared = InferenceInterface._preprocess_input_dataframe(
            pd.DataFrame({"Sex": ["male", "1", 1, "unknown", None]}),
            cfg,
        )

        assert prepared["Sex"].tolist() == [1.0, 1.0, 1.0, -1.0, -1.0]

    def test_raw_pii_is_pseudonymised_before_inference_encoding(self, monkeypatch):
        from app.agents.mta_v2.inference import InferenceInterface
        from app.agents.mta_v2.model import ModelConfig
        from app.agents.pii_agent import pseudonym

        monkeypatch.setenv("AVALOKA_PII_KEY", "test-only-key")
        token = pseudonym("alice@example.com", salt="customer_email")
        cfg = ModelConfig(
            input_size=2,
            output_size=2,
            hidden_sizes=[4],
            feature_names=["customer_email"],
            preprocessing={
                "version": 3,
                "pii_transformations": {
                    "customer_email": {
                        "kind": "email",
                        "strategy": "pseudonymise",
                    }
                },
                "categorical_features": {
                    "customer_email": {
                        "encoding": "one_hot",
                        "categories": [token],
                        "output_columns": ["email_known"],
                        "unknown_column": "email_unknown",
                    }
                },
                "imputation": {
                    "customer_email": {"fill_value": "__MISSING__"}
                },
                "feature_scaling": {},
                "model_feature_names": ["email_known", "email_unknown"],
            },
        )

        assert InferenceInterface._model_feature_values(
            {"customer_email": "alice@example.com"}, cfg
        ) == [1.0, 0.0]
        assert "alice@example.com" not in str(cfg.preprocessing)


# ---------------------------------------------------------------------------
# Tests: _is_local_data_source
# ---------------------------------------------------------------------------

class TestIsLocalDataSource:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def test_empty_uri_is_local(self):
        assert self.agent._is_local_data_source("") is True
        assert self.agent._is_local_data_source(None) is True

    def test_local_file_path_is_local(self):
        with patch.object(self.agent, "_get_file_size", return_value=1024):
            assert self.agent._is_local_data_source("/data/train.csv") is True

    def test_small_file_below_threshold_is_local(self):
        with patch.object(self.agent, "_get_file_size", return_value=1024):
            assert self.agent._is_local_data_source("gs://bucket/data.csv") is True

    def test_large_gcs_uri_is_remote(self):
        large = 20 * 1024 * 1024  # 20 MB
        with patch.object(self.agent, "_get_file_size", return_value=large):
            assert self.agent._is_local_data_source("gs://bucket/data.csv") is False

    def test_large_s3_uri_is_remote(self):
        large = 20 * 1024 * 1024
        with patch.object(self.agent, "_get_file_size", return_value=large):
            assert self.agent._is_local_data_source("s3://bucket/data.parquet") is False

    def test_large_http_uri_is_remote(self):
        large = 20 * 1024 * 1024
        with patch.object(self.agent, "_get_file_size", return_value=large):
            assert self.agent._is_local_data_source("https://example.com/data.csv") is False

    def test_postgresql_uri_is_remote(self):
        large = 20 * 1024 * 1024
        with patch.object(self.agent, "_get_file_size", return_value=large):
            assert self.agent._is_local_data_source("postgresql://user:pw@host/db") is False

    def test_file_size_none_uses_state_size(self):
        """When _get_file_size returns None, falls back to dataset_size_bytes kwarg."""
        with patch.object(self.agent, "_get_file_size", return_value=None):
            result = self.agent._is_local_data_source(
                "gs://bucket/data.csv",
                dataset_size_bytes=5 * 1024 * 1024,  # below threshold
            )
            assert result is True


# ---------------------------------------------------------------------------
# Tests: _get_file_size
# ---------------------------------------------------------------------------

class TestGetFileSize:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def test_local_existing_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\n1,2\n")
        size = self.agent._get_file_size(str(f))
        assert size == f.stat().st_size

    def test_local_missing_file_returns_none(self):
        result = self.agent._get_file_size("/nonexistent/path/file.csv")
        assert result is None

    def test_http_with_content_length(self):
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "12345"}
        with patch("app.agents.mta_v2.agent.requests.head", return_value=mock_resp):
            size = self.agent._get_file_size("http://example.com/data.csv")
        assert size == 12345

    def test_http_without_content_length_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.headers = {}
        with patch("app.agents.mta_v2.agent.requests.head", return_value=mock_resp):
            size = self.agent._get_file_size("http://example.com/data.csv")
        assert size is None

    def test_exception_returns_none(self):
        with patch("app.agents.mta_v2.agent.requests.head", side_effect=Exception("timeout")):
            result = self.agent._get_file_size("http://example.com/data.csv")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: latest generated output as training data
# ---------------------------------------------------------------------------

class TestLatestOutputTrainingData:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def _write_titanic_files(self, tmp_path):
        original = tmp_path / "train.csv"
        original.write_text(
            "Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,Survived\n"
            "3,male,22,1,0,7.25,S,0\n"
            "1,female,38,1,0,71.2833,C,1\n",
            encoding="utf-8",
        )
        latest = tmp_path / "output.csv"
        latest.write_text(
            "Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,Title,FamilySize,IsAlone,CabinKnown,Survived\n"
            "3,male,22,1,0,7.25,S,Mr,2,0,0,0\n"
            "1,female,38,1,0,71.2833,C,Mrs,2,0,1,1\n",
            encoding="utf-8",
        )
        return str(original), str(latest)

    def _state_with_latest_output(self, original, latest, message):
        return _make_state(
            messages=[HumanMessage(content=message)],
            data_source_location=original,
            data_source_location_local=original,
            latest_output_location=latest,
            latest_output_location_local=latest,
            latest_output_columns=[
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked",
                "Title",
                "FamilySize",
                "IsAlone",
                "CabinKnown",
                "Survived",
            ],
            latest_output_row_count=2,
            latest_output_is_trainable=True,
        )

    def test_explicit_output_prompt_uses_latest_output(self, tmp_path):
        original, latest = self._write_titanic_files(tmp_path)
        state = self._state_with_latest_output(
            original,
            latest,
            "Using the currently active modified output.csv dataset, train a classifier to predict Survived.",
        )

        assert self.agent._resolve_training_data_location(state) == latest

    def test_plain_current_dataset_prompt_does_not_use_latest_output(self, tmp_path):
        original, latest = self._write_titanic_files(tmp_path)
        state = self._state_with_latest_output(
            original,
            latest,
            "Using the current dataset, train a classifier to predict Survived.",
        )

        assert self.agent._resolve_training_data_location(state) == original

    def test_missing_requested_columns_switch_to_latest_output(self, tmp_path):
        original, latest = self._write_titanic_files(tmp_path)
        state = self._state_with_latest_output(
            original,
            latest,
            "Train a classification model to predict Survived with the requested feature columns.",
        )
        data_config = {
            "dataset_uri": original,
            "feature_columns": [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked",
                "Title",
                "FamilySize",
                "IsAlone",
                "CabinKnown",
            ],
            "target_column": "Survived",
        }

        self.agent._validate_training_data_config(data_config, state)

        assert data_config["dataset_uri"] == latest
        assert "Title" in data_config["feature_columns"]


# ---------------------------------------------------------------------------
# Tests: _finish_mta
# ---------------------------------------------------------------------------

class TestFinishMta:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def test_clears_training_flags(self):
        state = _make_state(
            training_scheduled=True,
            enable_training=True,
            skip_to_training=True,
            training_task={"some": "task"},
        )
        result = self.agent._finish_mta(state)
        assert result["training_scheduled"] is False
        assert result["enable_training"] is False
        assert result["skip_to_training"] is False
        assert result["training_task"] is None

    def test_does_not_mutate_original_state(self):
        state = _make_state(training_scheduled=True, enable_training=True)
        self.agent._finish_mta(state)
        assert state["training_scheduled"] is True  # original unchanged


# ---------------------------------------------------------------------------
# Tests: stub actions (_generate_code_repo, _review_code_repo)
# ---------------------------------------------------------------------------

class TestStubActions:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def test_generate_code_repo_returns_state_copy(self):
        state = _make_state()
        result = self.agent._generate_code_repo(state)
        assert result is not state
        assert result["user_id"] == state["user_id"]

    def test_review_code_repo_returns_state_copy(self):
        state = _make_state()
        result = self.agent._review_code_repo(state)
        assert result is not state
        assert result["session_id"] == state["session_id"]


# ---------------------------------------------------------------------------
# Tests: execute() — LLM disabled
# ---------------------------------------------------------------------------

class TestExecuteLlmDisabled:
    def test_returns_apology_message_when_llm_is_none(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        agent = ModelTrainingAgent()
        state = _make_state()
        # llm is already patched to None by the autouse fixture
        result = agent.execute(state)
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "unavailable" in last_msg.content.lower() or "sorry" in last_msg.content.lower()

    def test_training_flags_cleared_when_llm_is_none(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        agent = ModelTrainingAgent()
        state = _make_state(enable_training=True, skip_to_training=True, training_scheduled=False)
        result = agent.execute(state)
        assert result["enable_training"] is False


# ---------------------------------------------------------------------------
# Tests: execute() — LLM enabled but returns no tool_calls
# ---------------------------------------------------------------------------

class TestExecuteNoToolCalls:
    def test_apology_when_no_tool_calls(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_llm.invoke.return_value = mock_response

        agent = ModelTrainingAgent()
        state = _make_state()

        with patch("app.agents.mta_v2.agent.llm", mock_llm):
            result = agent.execute(state)

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "sorry" in last_msg.content.lower()


# ---------------------------------------------------------------------------
# Tests: execute() — tool routing
# ---------------------------------------------------------------------------

class TestExecuteToolRouting:
    """
    Verify that execute() calls the correct private handler for each tool name.
    """

    def _run_with_tool(self, tool_name: str, extra_state: dict | None = None):
        from app.agents.mta_v2.agent import ModelTrainingAgent

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"name": tool_name}]
        mock_llm.invoke.return_value = mock_response

        agent = ModelTrainingAgent()
        handler_name = {
            "generate_training_plan": "_generate_training_plan",
            "ask_more_detail_to_update_training_plan": "_ask_more_detail_to_update_training_plan",
            "update_training_plan": "_update_training_plan",
            "generate_code_repo": "_generate_code_repo",
            "review_code_repo": "_review_code_repo",
            "execute_training": "_execute_training",
            "execute_inference": "_execute_inference",
        }[tool_name]

        state = _make_state(**(extra_state or {}))
        returned_state = _make_state(messages=state["messages"].copy())

        with patch("app.agents.mta_v2.agent.llm", mock_llm), \
             patch.object(agent, handler_name, return_value=returned_state) as mock_handler:
            agent.execute(state)

        mock_handler.assert_called_once()

    def test_routes_generate_training_plan(self):
        self._run_with_tool("generate_training_plan")

    def test_routes_ask_more_detail(self):
        self._run_with_tool("ask_more_detail_to_update_training_plan")

    def test_routes_update_training_plan(self):
        self._run_with_tool("update_training_plan")

    def test_routes_generate_code_repo(self):
        self._run_with_tool("generate_code_repo")

    def test_routes_review_code_repo(self):
        self._run_with_tool("review_code_repo")

    def test_routes_execute_training(self):
        self._run_with_tool(
            "execute_training",
            {
                "training_plan": {"data_config": {"dataset_uri": "/tmp/train.csv"}},
                "skip_to_training": True,
            },
        )

    def test_routes_execute_inference(self):
        self._run_with_tool("execute_inference")

    def test_unknown_tool_skipped(self):
        """Unknown tool names should not raise; state is returned with flags cleared."""
        from app.agents.mta_v2.agent import ModelTrainingAgent

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"name": "nonexistent_tool"}]
        mock_llm.invoke.return_value = mock_response

        agent = ModelTrainingAgent()
        state = _make_state()

        with patch("app.agents.mta_v2.agent.llm", mock_llm):
            result = agent.execute(state)

        # Should not raise and should return a state dict
        assert isinstance(result, dict)

    def test_training_scheduled_bypasses_llm(self):
        """When training_scheduled=True execute() should call _execute_training directly."""
        from app.agents.mta_v2.agent import ModelTrainingAgent

        mock_llm = MagicMock()
        agent = ModelTrainingAgent()
        state = _make_state(training_scheduled=True)
        returned_state = _make_state(messages=state["messages"].copy(), training_scheduled=False)

        with patch("app.agents.mta_v2.agent.llm", mock_llm), \
             patch.object(agent, "_execute_training", return_value=returned_state) as mock_et:
            agent.execute(state)

        mock_et.assert_called_once()
        mock_llm.invoke.assert_not_called()

    def test_confirmed_ray_training_preserves_schedule_for_graph_router(self):
        """A confirmed Ray plan must reach schedule_task instead of being cleared."""
        from app.agents.mta_v2.agent import ModelTrainingAgent

        agent = ModelTrainingAgent()
        plan = _sample_training_plan(
            ray_config={"num_workers": 2, "use_gpu": False},
            data_config={
                "dataset_uri": "gs://bucket/large.csv",
                "feature_columns": ["feature"],
                "target_column": "target",
            },
        )
        state = _make_state(
            training_plan=plan,
            training_plan_reply_action="confirm",
            skip_to_training=True,
        )
        scheduled_state = {
            **state,
            "training_scheduled": True,
            "task_schedule": {
                "task_type": "training",
                "schedule_type": "relative",
                "second": 0,
                "max_runs": 1,
            },
        }

        with patch.object(
            agent,
            "_execute_training",
            return_value=scheduled_state,
        ) as mock_execute:
            result = agent.execute(state)

        mock_execute.assert_called_once()
        assert result["training_scheduled"] is True
        assert result["task_schedule"]["task_type"] == "training"
