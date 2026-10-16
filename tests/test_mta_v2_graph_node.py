"""
Tests for the graph's model-training entry point.

train_models (app/api/workflow.py) calls model_training_agent_node, a thin
wrapper in app/agents/model_training_agent.py around the mta_v2
ModelTrainingAgent. These drive that wrapper rather than the agent class, so
the route a user request actually takes is what is under test.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.model_training_agent import ModelTrainingAgent, model_training_agent_node


class TestGraphNode:
    @pytest.fixture(autouse=True)
    def training_csv(self, tmp_path):
        """A tiny binary-classification CSV on local disk."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
        df["target"] = y
        self.temp_dir = str(tmp_path)
        self.data_path = os.path.join(self.temp_dir, "test_data.csv")
        df.to_csv(self.data_path, index=False)

    def test_mta_agent_exposes_the_graph_entry_point(self):
        """The agent must be constructible and drivable from the graph.

        This used to assert execute_training/communication/mlflow_manager.
        app.agents.model_training_agent now re-exports the mta_v2 agent, which
        has none of them: its entry point is execute(state), and that is what
        model_training_agent_node calls.
        """
        mta = ModelTrainingAgent()

        assert callable(getattr(mta, "execute", None))
        assert callable(getattr(mta, "parse_tool_call", None))

    def _node_state(self, **overrides):
        """A graph state shaped like the one train_models receives."""
        state = {
            "user_id": "training-test-user",
            "session_id": "training-test-session",
            "messages": [HumanMessage(content="Train a classifier on my data")],
            "training_plan": None,
            "training_result": None,
            "training_scheduled": False,
            "enable_training": True,
            "skip_to_training": False,
            "training_task": None,
            "data_source_location_local": self.data_path,
        }
        state.update(overrides)
        return state

    def test_confirmed_plan_trains_through_the_graph_node(self, monkeypatch):
        """A confirmed plan must actually train when the graph node is called.

        This replaces test_mta_execute_training_mock, which called the v1
        execute_training(task, data_info) on the mta_v2 agent. That method does
        not exist there, so the call raised AttributeError and the except
        branch passed on the substring "execute_training" appearing in the
        class dict (it matches the private _execute_training). Nothing trained.

        The live route is model_training_agent_node(state) -> execute(state),
        and a local CSV runs in-process through LocalTrainer with no LLM.
        """
        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        monkeypatch.setenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", os.path.join(self.temp_dir, "mlruns"))
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-mta-graph-node")

        plan = {
            "model_type": "classification",
            "model_name": "graph_node_clf",
            "model_description": "model built by the training tests",
            "model_version": "1",
            "hyperparameter_config": {
                "learning_rate": 0.01,
                "epochs": 2,
                "batch_size": 16,
                "optimizer_name": "adam",
                "random_seed": 42,
                "early_stopping_patience": 2,
                "hidden_layer_sizes": [16, 8],
                "activation": "relu",
                "dropout_rate": 0.1,
                "batch_norm": False,
            },
            "ray_config": None,
            "data_config": {
                "dataset_uri": self.data_path,
                "feature_columns": ["feature1", "feature2", "feature3"],
                "target_column": "target",
            },
        }
        state = self._node_state(training_plan=plan, skip_to_training=True)

        # No LLM: a confirmed plan must not need one to start training.
        with patch("app.agents.mta_v2.agent.llm", None):
            result = model_training_agent_node(state)

        training_result = result["training_result"]
        assert training_result.get("status") != "error", training_result.get("error")
        assert training_result["mlflow_run_id"]
        assert training_result["model_type"] == "classification"
        assert training_result["num_features"] == 3
        assert training_result["num_classes"] == 2
        assert training_result["num_epochs_trained"] >= 1
        assert 0.0 <= training_result["final_accuracy"] <= 1.0

        assert result["training_completed"] is True
        assert result["mlflow_run_id"] == training_result["mlflow_run_id"]
        assert isinstance(result["messages"][-1], AIMessage)
        # _finish_mta hands the turn back to the graph with training switched off.
        assert result["enable_training"] is False
        assert result["skip_to_training"] is False

    def test_node_turns_a_failed_turn_into_a_reply(self):
        """A turn that blows up inside the agent must come back as a reply.

        This replaces the "invalid state" half of test_mta_integration_failures,
        which called mta.model_training_agent_node(...) -- a module function,
        not a method -- and then accepted the resulting AttributeError as the
        expected failure.

        What the live agent does with a turn it cannot process: execute()
        catches the error, appends one apology, and clears the training flags
        so the graph does not route straight back into train_models.
        """
        invalid_state = {
            "messages": [],
            "ready_to_train": False,
            "training_task": None,
            "enable_training": True,
            "invalid_key": "should_cause_issues",
        }
        broken_llm = MagicMock()
        broken_llm.invoke.side_effect = RuntimeError("provider exploded")

        with patch("app.agents.mta_v2.agent.llm", broken_llm):
            result = model_training_agent_node(invalid_state)

        broken_llm.invoke.assert_called_once()
        assert [type(message) for message in result["messages"]] == [AIMessage]
        assert result["messages"][-1].content == (
            "Sorry, something went wrong while processing your request. Please try again."
        )
        assert result["enable_training"] is False
        assert result["skip_to_training"] is False
        assert result["training_task"] is None
        assert not result.get("training_completed")
        assert result.get("training_result") is None
        # Keys the agent does not own pass through untouched.
        assert result["invalid_key"] == "should_cause_issues"
        assert result["ready_to_train"] is False
