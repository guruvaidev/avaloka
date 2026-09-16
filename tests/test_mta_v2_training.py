"""
Training-focused tests for app.agents.mta_v2.agent — ModelTrainingAgent

Covers the full training lifecycle with lightweight in-process data,
focusing on:
  - _execute_training with LocalTrainer (no Ray)
  - _execute_training with RayTrainer (cloud path)
  - _execute_training guard: no training plan
  - _execute_training guard: ray_config set but not yet scheduled
  - LocalTrainer.train() with a real CSV on disk (no mocking the trainer itself)
  - _execute_inference with model detail
  - _execute_inference guards (missing training_result, missing model detail)
  - _finish_mta state cleanup
  - _extract_ray_config defaults vs. from-state values
  - _extract_hyperparameter_config defaults
"""

from __future__ import annotations

import os
import csv
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_csv(tmp_path):
    """Write a tiny classification CSV to a temp directory; return its path."""
    np.random.seed(0)
    n = 80
    X = np.random.randn(n, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["target"] = y
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def _make_state(**overrides):
    base = {
        "user_id": "train-user",
        "session_id": "train-session",
        "messages": [HumanMessage(content="Train on my data")],
        "training_plan": None,
        "training_result": None,
        "training_scheduled": False,
        "enable_training": True,
        "skip_to_training": False,
        "training_task": None,
        "data_source_location_cloud": None,
        "data_source_location_local": None,
        "data_source_location": None,
        "dataset_size_bytes": None,
    }
    base.update(overrides)
    return base


def _make_local_plan(dataset_uri: str) -> dict:
    return {
        "model_type": "classification",
        "model_name": "test_clf",
        "model_description": "Unit test classifier",
        "model_version": "v1.0",
        "hyperparameter_config": {
            "learning_rate": 0.01,
            "epochs": 2,
            "batch_size": 16,
            "optimizer_name": "adam",
            "random_seed": 42,
            "early_stopping_patience": 2,
            "hidden_layer_sizes": [32, 16],
            "activation": "relu",
            "dropout_rate": 0.1,
            "batch_norm": False,
        },
        "ray_config": None,  # LocalTrainer
        "data_config": {
            "dataset_uri": dataset_uri,
            "feature_columns": ["f1", "f2", "f3"],
            "target_column": "target",
        },
    }


def _make_ray_plan(dataset_uri: str = "gs://bucket/data.csv") -> dict:
    plan = _make_local_plan(dataset_uri)
    plan["ray_config"] = {"num_workers": 2, "use_gpu": False, "memory_per_worker": "4Gi", "cpu_per_worker": 1}
    return plan


def test_default_features_exclude_identifier_like_columns():
    from app.agents.mta_v2.agent import ModelTrainingAgent

    features = ModelTrainingAgent._default_feature_columns(
        ["PassengerId", "customer_id", "UUID", "Pclass", "Age", "fluid", "Survived"],
        "Survived",
    )

    assert features == ["Pclass", "Age", "fluid"]


def test_default_features_keep_repeated_product_id_but_exclude_order_id():
    from app.agents.mta_v2.agent import ModelTrainingAgent

    sample = pd.DataFrame({
        "order_id": [1, 1, 2, 2, 3, 3],
        "product_id": [42, 7, 42, 9, 42, 7],
        "add_to_cart_order": [1, 2, 1, 2, 1, 2],
        "reordered": [0, 1, 1, 0, 1, 0],
    })

    features = ModelTrainingAgent._default_feature_columns(
        list(sample.columns), "reordered", sample
    )

    assert features == ["product_id", "add_to_cart_order"]


def test_sanitized_plan_uses_sample_to_keep_repeated_entity_id():
    from app.agents.mta_v2.agent import ModelTrainingAgent

    agent = ModelTrainingAgent()
    columns = ["order_id", "product_id", "add_to_cart_order", "reordered"]
    rows = [
        {"order_id": 1, "product_id": 42, "add_to_cart_order": 1, "reordered": 0},
        {"order_id": 1, "product_id": 7, "add_to_cart_order": 2, "reordered": 1},
        {"order_id": 2, "product_id": 42, "add_to_cart_order": 1, "reordered": 1},
        {"order_id": 2, "product_id": 9, "add_to_cart_order": 2, "reordered": 0},
    ]
    data_config = {
        "dataset_uri": "gs://bucket/instacart.csv",
        "feature_columns": ["order_id", "product_id", "add_to_cart_order"],
        "target_column": "reordered",
        # Simulate a name-only LLM decision from an older plan.
        "excluded_identifier_columns": ["order_id", "product_id"],
    }
    state = _make_state(
        messages=[HumanMessage(content="Use these features")],
        quick_sample_rows=rows,
    )
    with patch.object(agent, "_columns_for_training_data", return_value=columns), \
         patch.object(agent, "_maybe_switch_data_config_to_latest_output", return_value=columns):
        agent._sanitize_training_data_config(data_config, state)

    assert data_config["feature_columns"] == ["product_id", "add_to_cart_order"]
    assert data_config["excluded_identifier_columns"] == ["order_id"]


def test_sanitized_plan_excludes_identifier_selected_by_llm():
    from app.agents.mta_v2.agent import ModelTrainingAgent

    agent = ModelTrainingAgent()
    columns = ["PassengerId", "Pclass", "Age", "Survived"]
    data_config = {
        "dataset_uri": "/tmp/titanic.csv",
        "feature_columns": ["PassengerId", "Pclass", "Age"],
        "target_column": "Survived",
    }
    state = _make_state(messages=[HumanMessage(content="Use all other columns")])
    with patch.object(agent, "_columns_for_training_data", return_value=columns), \
         patch.object(agent, "_maybe_switch_data_config_to_latest_output", return_value=columns):
        agent._sanitize_training_data_config(data_config, state)

    assert data_config["feature_columns"] == ["Pclass", "Age"]
    assert data_config["excluded_identifier_columns"] == ["PassengerId"]

    plan = _make_local_plan("/tmp/titanic.csv")
    plan["data_config"] = data_config
    rendered = agent._format_training_plan_response(plan)
    assert "Excluded identifier columns:** `PassengerId`" in rendered


def test_training_pii_guard_detects_sensitive_values_and_requires_key(monkeypatch):
    from app.agents.mta_v2.agent import ModelTrainingAgent
    from app.agents.pii_agent import MissingPIIKey

    monkeypatch.delenv("AVALOKA_PII_KEY", raising=False)
    agent = ModelTrainingAgent()
    data_config = {
        "dataset_uri": "gs://bucket/customers.csv",
        "feature_columns": ["contact_field", "spend"],
        "target_column": "churned",
    }
    state = _make_state(quick_sample_rows=[
        {"contact_field": "alice@example.com", "spend": 10.0, "churned": 0},
        {"contact_field": "bob@example.com", "spend": 20.0, "churned": 1},
    ])

    with pytest.raises(MissingPIIKey, match="contact_field"):
        agent._validate_training_pii_key(data_config, state)

    assert data_config["pii_key_required_columns"] == ["contact_field"]


def test_training_pii_guard_allows_sensitive_values_when_key_is_set(monkeypatch):
    from app.agents.mta_v2.agent import ModelTrainingAgent

    monkeypatch.setenv("AVALOKA_PII_KEY", "test-only-key")
    agent = ModelTrainingAgent()
    data_config = {
        "dataset_uri": "gs://bucket/customers.csv",
        "feature_columns": ["customer_email", "spend"],
        "target_column": "churned",
    }
    state = _make_state(quick_sample_rows=[
        {"customer_email": "alice@example.com", "spend": 10.0, "churned": 0},
    ])

    agent._validate_training_pii_key(data_config, state)

    assert data_config["pii_key_required_columns"] == ["customer_email"]
    assert data_config["pii_transformations"] == {
        "customer_email": {"kind": "email", "strategy": "pseudonymise"}
    }


def test_ray_job_refuses_scheduled_sensitive_training_if_key_was_removed(monkeypatch):
    from app.agents.mta_v2.ray_trainer import RayTrainer
    from app.agents.pii_agent import MissingPIIKey

    monkeypatch.delenv("AVALOKA_PII_KEY", raising=False)
    plan = _make_ray_plan()
    plan["data_config"]["pii_key_required_columns"] = ["customer_email"]

    with pytest.raises(MissingPIIKey, match="customer_email"):
        RayTrainer("user", "session")._build_job_env(plan)


def test_ray_job_receives_pii_contract_but_not_the_secret(monkeypatch):
    import json
    from app.agents.mta_v2.ray_trainer import RayTrainer

    monkeypatch.setenv("AVALOKA_PII_KEY", "test-only-key")
    plan = _make_ray_plan()
    plan["data_config"]["pii_key_required_columns"] = ["customer_email"]
    plan["data_config"]["pii_transformations"] = {
        "customer_email": {"kind": "email", "strategy": "pseudonymise"}
    }

    job_env = RayTrainer("user", "session")._build_job_env(plan)

    assert json.loads(job_env["PII_TRANSFORMATIONS"]) == {
        "customer_email": {"kind": "email", "strategy": "pseudonymise"}
    }
    assert "AVALOKA_PII_KEY" not in job_env
    assert "test-only-key" not in str(job_env)


def test_execute_training_surfaces_missing_pii_key_to_the_user(tmp_path, monkeypatch):
    from app.agents.mta_v2.agent import ModelTrainingAgent

    monkeypatch.delenv("AVALOKA_PII_KEY", raising=False)
    dataset = tmp_path / "customers.csv"
    rows = [
        {"customer_email": "alice@example.com", "spend": 10.0, "churned": 0},
        {"customer_email": "bob@example.com", "spend": 20.0, "churned": 1},
    ]
    pd.DataFrame(rows).to_csv(dataset, index=False)
    plan = _make_local_plan(str(dataset))
    plan["data_config"] = {
        "dataset_uri": str(dataset),
        "feature_columns": ["customer_email", "spend"],
        "target_column": "churned",
    }
    plan["execution_backend"] = "local"
    plan["execution_dataset_uri"] = str(dataset)
    state = _make_state(
        training_plan=plan,
        data_source_location=str(dataset),
        data_source_location_local=str(dataset),
        quick_sample_rows=rows,
    )

    result = ModelTrainingAgent()._execute_training(state)

    message = result["messages"][-1].content
    assert "PII_KEY_MISSING" in message
    assert "AVALOKA_PII_KEY" in message
    assert result["training_completed"] is False


# ---------------------------------------------------------------------------
# Tests: _execute_training — no training plan
# ---------------------------------------------------------------------------

class TestExecuteTrainingNoплан:
    def test_returns_error_message_if_no_plan(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        agent = ModelTrainingAgent()
        state = _make_state(training_plan=None)

        with patch("app.agents.mta_v2.agent.llm", None):
            result = agent._execute_training(state)

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "plan" in last_msg.content.lower() or "sorry" in last_msg.content.lower()

    def test_finishes_mta_flags_if_no_plan(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        agent = ModelTrainingAgent()
        state = _make_state(training_plan=None, enable_training=True, skip_to_training=True)

        with patch("app.agents.mta_v2.agent.llm", None):
            result = agent._execute_training(state)

        assert result["enable_training"] is False
        assert result["skip_to_training"] is False


# ---------------------------------------------------------------------------
# Tests: _execute_training — LocalTrainer path (mocked trainer)
# ---------------------------------------------------------------------------

class TestExecuteTrainingLocalMocked:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def _build_mock_llm(self, text="Training completed."):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=text, tool_calls=[])
        return mock_llm

    def test_training_result_stored(self, tmp_csv):
        plan = _make_local_plan(tmp_csv)
        expected_result = {
            "mlflow_run_id": "run-local-1",
            "final_accuracy": 0.85,
            "final_f1_score": 0.83,
            "num_epochs_trained": 2,
        }

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = expected_result

        state = _make_state(training_plan=plan)
        with patch("app.agents.mta_v2.agent.llm", self._build_mock_llm()), \
             patch("app.agents.mta_v2.agent.LocalTrainer", return_value=mock_trainer):
            result = self.agent._execute_training(state)

        assert result["training_result"] == expected_result
        mock_trainer.train.assert_called_once()
        submitted_plan = mock_trainer.train.call_args.args[0]
        assert {key: submitted_plan[key] for key in plan} == plan
        assert submitted_plan["execution_backend"] == "local"
        assert submitted_plan["execution_dataset_uri"] == tmp_csv
        assert isinstance(submitted_plan["execution_dataset_size_bytes"], int)
        assert submitted_plan["execution_dataset_size_bytes"] > 0
        assert submitted_plan["execution_reason"]

    def test_integrity_and_evaluation_reports_are_copied_into_mta_state(self, tmp_csv):
        plan = _make_local_plan(tmp_csv)
        integrity = {
            "safe_to_train": True,
            "rows": 80,
            "checked_columns": 3,
            "n_findings": 0,
            "n_blockers": 0,
            "findings": [],
        }
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "mlflow_run_id": "run-with-integrity",
            "final_accuracy": 0.85,
            "integrity_report": integrity,
            "evaluation_report": {
                "primary_metric": "accuracy",
                "beats_baseline": True,
            },
        }

        with patch("app.agents.mta_v2.agent.LocalTrainer", return_value=mock_trainer):
            result = self.agent._execute_training(_make_state(training_plan=plan))

        assert result["integrity_report"] == integrity
        assert result["integrity_safe_to_train"] is True
        assert result["evaluation_report"]["primary_metric"] == "accuracy"
        assert result["evaluation_beats_baseline"] is True

    def test_trainer_receives_correct_user_and_session(self, tmp_csv):
        plan = _make_local_plan(tmp_csv)
        state = _make_state(training_plan=plan, user_id="alice", session_id="sess-99")

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"mlflow_run_id": "r1", "final_accuracy": 0.9}

        with patch("app.agents.mta_v2.agent.llm", self._build_mock_llm()), \
             patch("app.agents.mta_v2.agent.LocalTrainer") as MockLocalTrainer:
            MockLocalTrainer.return_value = mock_trainer
            self.agent._execute_training(state)

        MockLocalTrainer.assert_called_once_with(user_id="alice", session_id="sess-99")

    def test_training_scheduled_set_false_after_local_run(self, tmp_csv):
        plan = _make_local_plan(tmp_csv)
        state = _make_state(training_plan=plan, training_scheduled=False)

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"mlflow_run_id": "r-x", "final_accuracy": 0.88}

        with patch("app.agents.mta_v2.agent.llm", self._build_mock_llm()), \
             patch("app.agents.mta_v2.agent.LocalTrainer", return_value=mock_trainer):
            result = self.agent._execute_training(state)

        assert result.get("training_scheduled") is False

    def test_deterministic_report_appended_after_local_training(self, tmp_csv):
        plan = _make_local_plan(tmp_csv)
        state = _make_state(training_plan=plan)

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"mlflow_run_id": "r-ai", "final_accuracy": 0.9}

        with patch("app.agents.mta_v2.agent.llm", self._build_mock_llm()), \
             patch("app.agents.mta_v2.agent.LocalTrainer", return_value=mock_trainer):
            result = self.agent._execute_training(state)

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1
        report = ai_msgs[-1].content
        assert "Local Training Run Report" in report
        assert "MLflow Run ID:** r-ai" in report
        assert "Final Accuracy:** 0.9" in report


# ---------------------------------------------------------------------------
# Tests: _execute_training — RayTrainer path (scheduling + execution)
# ---------------------------------------------------------------------------

class TestExecuteTrainingRayMocked:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def test_first_call_sets_task_schedule(self):
        """ray_config present + training_scheduled=False → returns task_schedule."""
        plan = _make_ray_plan()
        state = _make_state(training_plan=plan, training_scheduled=False)

        with patch("app.agents.mta_v2.agent.llm", None):
            result = self.agent._execute_training(state)

        assert result["training_scheduled"] is True
        assert result["task_schedule"]["task_type"] == "training"
        assert result["task_schedule"]["max_runs"] == 1

    def test_second_call_runs_ray_trainer(self):
        """training_scheduled=True → RayTrainer.train() is invoked."""
        plan = _make_ray_plan()
        mock_result = {
            "mlflow_run_id": "ray-run-1",
            "dashboard_url": "http://localhost:8265",
            "final_accuracy": 0.89,
        }

        mock_ray_trainer = MagicMock()
        mock_ray_trainer.train.return_value = mock_result

        state = _make_state(training_plan=plan, training_scheduled=True)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Ray training done.", tool_calls=[])

        with patch("app.agents.mta_v2.agent.llm", mock_llm), \
             patch("app.agents.mta_v2.agent.RayTrainer", return_value=mock_ray_trainer):
            result = self.agent._execute_training(state)

        mock_ray_trainer.train.assert_called_once()
        submitted_plan = mock_ray_trainer.train.call_args.args[0]
        assert {key: submitted_plan[key] for key in plan} == plan
        assert submitted_plan["execution_backend"] == "ray"
        assert submitted_plan["execution_dataset_uri"] == "gs://bucket/data.csv"
        assert submitted_plan["execution_dataset_size_bytes"] is None
        assert submitted_plan["execution_reason"]
        assert result["training_result"] == mock_result

    def test_ray_trainer_receives_correct_ids(self):
        plan = _make_ray_plan()
        state = _make_state(
            training_plan=plan,
            training_scheduled=True,
            user_id="bob",
            session_id="s-ray",
        )
        mock_result = {"mlflow_run_id": "rr", "dashboard_url": "http://x", "final_accuracy": 0.8}
        mock_ray_trainer = MagicMock()
        mock_ray_trainer.train.return_value = mock_result
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Done.", tool_calls=[])

        with patch("app.agents.mta_v2.agent.llm", mock_llm), \
             patch("app.agents.mta_v2.agent.RayTrainer") as MockRayTrainer:
            MockRayTrainer.return_value = mock_ray_trainer
            self.agent._execute_training(state)

        MockRayTrainer.assert_called_once_with(user_id="bob", session_id="s-ray")

    def test_training_scheduled_reset_after_ray_run(self):
        plan = _make_ray_plan()
        mock_ray_trainer = MagicMock()
        mock_ray_trainer.train.return_value = {
            "mlflow_run_id": "r", "dashboard_url": "http://d", "final_accuracy": 0.7
        }
        state = _make_state(training_plan=plan, training_scheduled=True)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Done.", tool_calls=[])

        with patch("app.agents.mta_v2.agent.llm", mock_llm), \
             patch("app.agents.mta_v2.agent.RayTrainer", return_value=mock_ray_trainer):
            result = self.agent._execute_training(state)

        assert result.get("training_scheduled") is False


# ---------------------------------------------------------------------------
# Tests: _execute_inference
# ---------------------------------------------------------------------------

class TestExecuteInferenceAgent:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def _build_mock_llm_multi(self, *contents):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content=c, tool_calls=[]) for c in contents
        ]
        return mock_llm

    def test_predict_called_with_extracted_input(self):
        training_result = {"mlflow_run_id": "run-inf", "final_accuracy": 0.95}
        model_detail = {
            "feature_names": ["f1", "f2"],
            "target_column": "target",
            "num_features": 2,
            "num_classes": 2,
        }
        inference_result = {"prediction": "1", "probabilities": {"0": 0.2, "1": 0.8}}

        mock_inferencer = MagicMock()
        mock_inferencer.get_model_details.return_value = model_detail
        mock_inferencer.predict.return_value = inference_result

        state = _make_state(training_result=training_result)
        input_json = '{"f1": 1.0, "f2": 2.5}'
        llm_mock = self._build_mock_llm_multi(input_json, "Prediction is 1.")

        with patch("app.agents.mta_v2.agent.llm", llm_mock), \
             patch("app.agents.mta_v2.agent.InferenceInterface", return_value=mock_inferencer):
            result = self.agent._execute_inference(state)

        mock_inferencer.predict.assert_called_once()
        call_kwargs = mock_inferencer.predict.call_args
        assert call_kwargs.kwargs.get("mlflow_run_id") == "run-inf" or \
               call_kwargs[1].get("mlflow_run_id") == "run-inf" or \
               "run-inf" in str(call_kwargs)

    def test_error_without_training_result(self):
        state = _make_state(training_result=None)
        with patch("app.agents.mta_v2.agent.llm", None):
            result = self.agent._execute_inference(state)

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "train" in last_msg.content.lower() or "sorry" in last_msg.content.lower()

    def test_error_when_mlflow_run_id_missing(self):
        state = _make_state(training_result={"status": "ok"})  # no mlflow_run_id
        with patch("app.agents.mta_v2.agent.llm", None):
            result = self.agent._execute_inference(state)

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)

    def test_error_when_model_detail_none(self):
        training_result = {"mlflow_run_id": "bad-run"}
        state = _make_state(training_result=training_result)

        mock_inferencer = MagicMock()
        mock_inferencer.get_model_details.return_value = None

        with patch("app.agents.mta_v2.agent.llm", None), \
             patch("app.agents.mta_v2.agent.InferenceInterface", return_value=mock_inferencer):
            result = self.agent._execute_inference(state)

        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "detail" in last_msg.content.lower() or "sorry" in last_msg.content.lower()

    def test_mta_flags_cleared_on_inference_error(self):
        state = _make_state(training_result=None, enable_training=True, skip_to_training=True)
        with patch("app.agents.mta_v2.agent.llm", None):
            result = self.agent._execute_inference(state)

        assert result["enable_training"] is False
        assert result["skip_to_training"] is False


# ---------------------------------------------------------------------------
# Tests: _extract_ray_config
# ---------------------------------------------------------------------------

class TestExtractRayConfig:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def test_returns_defaults_when_no_plan(self):
        state = _make_state(training_plan=None)
        config = self.agent._extract_ray_config(state)
        assert config["num_workers"] == 4
        assert config["use_gpu"] is False
        assert config["memory_per_worker"] == "16Gi"
        assert config["cpu_per_worker"] == 4

    def test_returns_stored_config_from_plan(self):
        plan = _make_ray_plan()
        plan["ray_config"] = {
            "num_workers": 8,
            "use_gpu": True,
            "memory_per_worker": "32Gi",
            "cpu_per_worker": 8,
        }
        state = _make_state(training_plan=plan)
        config = self.agent._extract_ray_config(state)
        assert config["num_workers"] == 8
        assert config["use_gpu"] is True


# ---------------------------------------------------------------------------
# Tests: LocalTrainer.train() with a real CSV (end-to-end lightweight)
# ---------------------------------------------------------------------------

class TestLocalTrainerTrain:
    """
    Run LocalTrainer.train() on a tiny synthetic CSV to verify the end-to-end
    training pipeline.  MLflow is patched to use a temp directory.
    """

    @pytest.fixture(autouse=True)
    def allow_mlflow_file_store(self, monkeypatch):
        """Explicitly opt these isolated tests into MLflow's file backend."""
        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        monkeypatch.setenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "")

    def test_train_returns_metrics_dict(self, tmp_csv, tmp_path):
        from app.agents.mta_v2.local_trainer import LocalTrainer

        mlflow_dir = str(tmp_path / "mlruns")
        plan = _make_local_plan(tmp_csv)

        with patch.dict(os.environ, {
            "MLFLOW_TRACKING_URI": mlflow_dir,
            "MLFLOW_EXPERIMENT_NAME": "test-local-trainer",
        }):
            trainer = LocalTrainer(user_id="tester", session_id="s1")
            result = trainer.train(plan)

        # Must return a dict with these keys
        assert isinstance(result, dict)
        for key in ("mlflow_run_id", "final_accuracy", "num_epochs_trained", "num_features"):
            assert key in result, f"Missing key: {key}"
        assert result["integrity_report"]["safe_to_train"] is True
        assert result["evaluation_report"]["primary_metric"] == "accuracy"
        assert isinstance(result["evaluation_report"]["beats_baseline"], bool)

    def test_train_blocks_exact_copied_source_records(self, tmp_path):
        from app.agents.mta_v2.local_trainer import LocalTrainer

        frame = pd.DataFrame({
            "PassengerId": list(range(1, 7)) * 2,
            "f1": [1, 2, 3, 4, 5, 6] * 2,
            "f2": [6, 2, 5, 1, 3, 4] * 2,
            "target": [0, 1, 0, 1, 1, 0] * 2,
        })
        duplicate_csv = tmp_path / "duplicate-training-rows.csv"
        frame.to_csv(duplicate_csv, index=False)
        plan = _make_local_plan(str(duplicate_csv))
        plan["data_config"]["feature_columns"] = ["PassengerId", "f1", "f2"]

        with patch.dict(os.environ, {
            "MLFLOW_TRACKING_URI": str(tmp_path / "mlruns-duplicates"),
            "MLFLOW_EXPERIMENT_NAME": "test-duplicate-training-rows",
        }):
            result = LocalTrainer(user_id="tester", session_id="duplicates").train(plan)

        assert result["status"] == "error"
        assert "Duplicate training rows detected" in result["error"]

    def test_train_blocks_unique_identifier_feature_as_final_guard(self, tmp_csv, tmp_path):
        from app.agents.mta_v2.local_trainer import LocalTrainer

        frame = pd.read_csv(tmp_csv)
        frame.insert(0, "PassengerId", range(1, len(frame) + 1))
        identifier_csv = tmp_path / "identifier-feature.csv"
        frame.to_csv(identifier_csv, index=False)
        plan = _make_local_plan(str(identifier_csv))
        plan["data_config"]["feature_columns"].insert(0, "PassengerId")

        with patch.dict(os.environ, {
            "MLFLOW_TRACKING_URI": str(tmp_path / "mlruns-identifier"),
            "MLFLOW_EXPERIMENT_NAME": "test-identifier-feature",
        }):
            result = LocalTrainer(user_id="tester", session_id="identifier").train(plan)

        assert result["status"] == "error"
        assert "Identifier-like feature detected" in result["error"]
        assert "PassengerId" in result["error"]

    def test_train_blocks_sensitive_feature_without_pii_key(self, tmp_path, monkeypatch):
        from app.agents.mta_v2.local_trainer import LocalTrainer

        monkeypatch.delenv("AVALOKA_PII_KEY", raising=False)
        frame = pd.DataFrame({
            "customer_email": [f"customer{i}@example.com" for i in range(20)],
            "spend": np.linspace(10.0, 100.0, 20),
            "target": [0, 1] * 10,
        })
        dataset = tmp_path / "sensitive-training.csv"
        frame.to_csv(dataset, index=False)
        plan = _make_local_plan(str(dataset))
        plan["data_config"]["feature_columns"] = ["customer_email", "spend"]

        with patch.dict(os.environ, {
            "MLFLOW_TRACKING_URI": str(tmp_path / "mlruns-pii"),
            "MLFLOW_EXPERIMENT_NAME": "test-pii-guard",
        }):
            result = LocalTrainer(user_id="tester", session_id="pii").train(plan)

        assert result["status"] == "error"
        assert "AVALOKA_PII_KEY is not set" in result["error"]
        assert "customer_email" in result["error"]

    def test_train_blocks_a_feature_that_copies_the_target(self, tmp_csv, tmp_path):
        from app.agents.mta_v2.local_trainer import LocalTrainer

        frame = pd.read_csv(tmp_csv)
        frame["target_leak"] = frame["target"]
        leaking_csv = tmp_path / "leaking.csv"
        frame.to_csv(leaking_csv, index=False)
        plan = _make_local_plan(str(leaking_csv))
        plan["data_config"]["feature_columns"].append("target_leak")

        with patch.dict(os.environ, {
            "MLFLOW_TRACKING_URI": str(tmp_path / "mlruns-leak"),
            "MLFLOW_EXPERIMENT_NAME": "test-target-leakage",
        }):
            result = LocalTrainer(user_id="tester", session_id="leak").train(plan)

        assert result["status"] == "error"
        assert "Target leakage detected" in result["error"]
        assert "target_leak" in result["error"]

    def test_train_uses_provided_user_id(self, tmp_csv, tmp_path):
        from app.agents.mta_v2.local_trainer import LocalTrainer

        mlflow_dir = str(tmp_path / "mlruns2")
        plan = _make_local_plan(tmp_csv)

        with patch.dict(os.environ, {
            "MLFLOW_TRACKING_URI": mlflow_dir,
            "MLFLOW_EXPERIMENT_NAME": "test-local-trainer2",
        }):
            trainer = LocalTrainer(user_id="user-xyz", session_id="s2")
            result = trainer.train(plan)

        assert result.get("user_id") == "user-xyz"

    def test_train_with_classification_plan(self, tmp_csv, tmp_path):
        from app.agents.mta_v2.local_trainer import LocalTrainer

        mlflow_dir = str(tmp_path / "mlruns3")
        plan = _make_local_plan(tmp_csv)
        plan["model_type"] = "classification"

        with patch.dict(os.environ, {
            "MLFLOW_TRACKING_URI": mlflow_dir,
            "MLFLOW_EXPERIMENT_NAME": "test-local-clf",
        }):
            trainer = LocalTrainer(user_id="ux", session_id="sx")
            result = trainer.train(plan)

        assert result.get("model_type") in ("classification", "clf", None) or \
               result.get("num_classes") is not None

    def test_train_failure_when_dataset_missing(self, tmp_path):
        from app.agents.mta_v2.local_trainer import LocalTrainer

        plan = _make_local_plan("/nonexistent/ghost.csv")
        trainer = LocalTrainer(user_id="u", session_id="s")

        with patch.dict(os.environ, {
            "MLFLOW_TRACKING_URI": str(tmp_path / "mlruns-fail"),
        }):
            result = trainer.train(plan)

        assert result["status"] == "error"
        assert result["model_name"] == plan["model_name"]
        assert result["model_type"] == plan["model_type"]
        assert result["error"]


# ---------------------------------------------------------------------------
# Tests: _finish_mta state cleanup
# ---------------------------------------------------------------------------

class TestFinishMtaTraining:
    def setup_method(self):
        from app.agents.mta_v2.agent import ModelTrainingAgent
        self.agent = ModelTrainingAgent()

    def test_all_flags_cleared(self):
        state = _make_state(
            training_scheduled=True,
            enable_training=True,
            skip_to_training=True,
            training_task={"id": "t1"},
        )
        result = self.agent._finish_mta(state)
        assert result["training_scheduled"] is False
        assert result["enable_training"] is False
        assert result["skip_to_training"] is False
        assert result["training_task"] is None

    def test_messages_preserved(self):
        msgs = [HumanMessage(content="hello"), AIMessage(content="world")]
        state = _make_state(messages=msgs, training_scheduled=True)
        result = self.agent._finish_mta(state)
        assert result["messages"] == msgs

    def test_training_result_preserved(self):
        tr = {"mlflow_run_id": "preserved", "final_accuracy": 0.9}
        state = _make_state(training_result=tr, training_scheduled=True)
        result = self.agent._finish_mta(state)
        assert result["training_result"] == tr
