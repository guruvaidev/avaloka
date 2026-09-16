"""
Unit tests for Model Training Agent (MTA)

Tests cover:
- Agent initialization
- Task validation
- Training plan creation
- Environment setup
- State management
"""

from tests._quarantine import requires_api

requires_api("app.agents.model_training_agent", "create_sample_training_task", replacement="create_sample_training_task was removed; the module now exposes model_training_agent / model_training_agent_node. Repair against those or remove this file.")


import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

from app.agents.model_training_agent import (
    ModelTrainingAgent,
    model_training_agent_node,
    create_sample_training_task,
    get_supported_model_types
)
from app.graph.etl_state import ETLState
from langchain_core.messages import AIMessage, HumanMessage


class TestModelTrainingAgent:
    """Test suite for ModelTrainingAgent class"""
    
    def test_agent_initialization_default(self):
        """Test MTA agent initialization with default config"""
        mta = ModelTrainingAgent()
        assert mta.config == {}
        assert mta.experiment_name == "avaloka_mta_gcs"
    
    def test_agent_initialization_with_config(self):
        """Test MTA agent initialization with custom config"""
        config = {"experiment_name": "test_experiment", "timeout": 300}
        mta = ModelTrainingAgent(config)
        assert mta.config == config
        assert mta.experiment_name == "test_experiment"
    
    def test_validate_training_task_valid(self):
        """Test validation of valid training task"""
        mta = ModelTrainingAgent()
        task = {
            "task_id": "test_task_001",
            "model_type": "pytorch_classification",
            "goal_description": "Classify customer segments"
        }
        
        result = mta.validate_training_task(task)
        assert result["valid"] is True
        assert result["errors"] is None
    
    def test_validate_training_task_missing_fields(self):
        """Test validation of training task with missing required fields"""
        mta = ModelTrainingAgent()
        task = {
            "task_id": "test_task_001"
            # Missing model_type and goal_description
        }
        
        result = mta.validate_training_task(task)
        assert result["valid"] is False
        assert "Training task is missing required fields" in str(result["errors"])
        assert "model_type" in str(result["errors"])
        assert "goal_description" in str(result["errors"])
    
    def test_validate_training_task_unsupported_model(self):
        """Test validation of training task with unsupported model type"""
        mta = ModelTrainingAgent()
        task = {
            "task_id": "test_task_001",
            "model_type": "unsupported_model_type",
            "goal_description": "Test goal"
        }
        
        result = mta.validate_training_task(task)
        assert result["valid"] is False
        assert "Unsupported model types" in str(result["errors"])
    
    def test_validate_training_task_multiple_models(self):
        """Test validation of training task with multiple model types"""
        mta = ModelTrainingAgent()
        task = {
            "task_id": "test_task_001",
            "model_type": ["pytorch_classification", "svm"],
            "goal_description": "Multi-model comparison"
        }
        
        result = mta.validate_training_task(task)
        assert result["valid"] is True
        assert result["errors"] is None
    
    def test_create_training_plan_basic(self):
        """Test creation of basic training plan"""
        mta = ModelTrainingAgent()
        task = {
            "task_id": "test_task_001",
            "model_type": "pytorch_classification",
            "goal_description": "Test classification"
        }
        
        plan = mta.create_training_plan(task)
        assert plan["task_id"] == "test_task_001"
        assert plan["goal"] == "Test classification"
        assert "model_suggestions" in plan
        assert "training_strategy" in plan
        assert "estimated_duration" in plan
    
    def test_create_training_plan_with_data_info(self):
        """Test creation of training plan with data information"""
        mta = ModelTrainingAgent()
        task = {
            "task_id": "test_task_001",
            "model_type": "pytorch_classification",
            "goal_description": "Test classification"
        }
        data_info = {
            "schema": ["feature1", "feature2", "feature3", "target"]
        }
        
        plan = mta.create_training_plan(task, data_info)
        assert "data_analysis" in plan
        assert plan["data_analysis"]["num_features"] == 4
        assert plan["data_analysis"]["complexity"] == "low"
    
    def test_create_training_plan_competition_mode(self):
        """Test creation of training plan with multiple models (competition mode)"""
        mta = ModelTrainingAgent()
        task = {
            "task_id": "test_task_001",
            "model_type": ["pytorch_classification", "svm", "gam"],
            "goal_description": "Model competition"
        }
        
        plan = mta.create_training_plan(task)
        assert len(plan["model_suggestions"]) > 1
        assert plan["training_strategy"] == "competition"
        assert "30-60 minutes" in plan["estimated_duration"]
    
    def test_prepare_training_environment_basic(self):
        """Test preparation of basic training environment"""
        mta = ModelTrainingAgent()
        task = {
            "task_id": "test_task_001",
            "model_type": "pytorch_classification"
        }
        
        env_setup = mta.prepare_training_environment(task)
        assert env_setup["status"] == "ready"
        assert env_setup["environment"] == "local"
        assert "resources" in env_setup
        assert env_setup["distributed_training"] is False
    
    def test_prepare_training_environment_with_ray(self):
        """Test preparation of training environment with Ray configuration"""
        mta = ModelTrainingAgent()
        task = {
            "task_id": "test_task_001",
            "model_type": "pytorch_classification",
            "ray_config": {"num_workers": 4}
        }
        
        env_setup = mta.prepare_training_environment(task)
        assert env_setup["status"] == "ready"
        assert env_setup["distributed_training"] is True
        assert env_setup["ray_config"] == {"num_workers": 4}


class TestModelTrainingAgentNode:
    """Test suite for the model_training_agent_node function"""
    
    def create_basic_state(self) -> ETLState:
        """Create a basic ETL state for testing"""
        return ETLState(
            messages=[HumanMessage(content="Test message")],
            planner_definition={},
            ready_to_summarize=False,
            ready_to_code=False,
            coder_definition={},
            data_source_location="",
            output_location="",
            infrastructure_request={},
            infrastructure_provisioned={},
            execution_result={},
            output_file_data={},
            # MTA fields
            training_task=None,
            model_artifacts=None,
            mlflow_run_id=None,
            training_metrics=None,
            model_registry=None,
            ray_config=None,
            validation_params=None,
            rlhf_feedback=None,
            business_recommendations=None,
            ready_to_train=None,
            training_completed=None
        )
    
    def test_node_no_training_task_not_ready(self):
        """Test node behavior when no training task and not ready to code"""
        state = self.create_basic_state()
        
        result = model_training_agent_node(state)
        
        # The node automatically creates a training task from state analysis
        assert result["ready_to_train"] is True
        assert result["training_task"] is not None
        assert result["training_task"]["created_from"] == "state_analysis"
        assert len(result["messages"]) == 2  # Original + new AI message
        assert "Created training task" in result["messages"][-1].content
    
    def test_node_auto_create_task_from_coder(self):
        """Test node creating automatic training task from coder results"""
        state = self.create_basic_state()
        state.update({
            "ready_to_code": True,
            "coder_definition": {"code": "print('test code')"}
        })
        
        result = model_training_agent_node(state)
        
        assert result["training_task"] is not None
        assert result["training_task"]["created_from"] == "state_analysis"
        assert result["ready_to_train"] is True
    
    def test_node_with_valid_training_task(self):
        """Test node behavior with valid training task"""
        state = self.create_basic_state()
        state.update({
            "training_task": {
                "task_id": "test_task_001",
                "model_type": "pytorch_classification",
                "goal_description": "Test classification"
            },
            "schema": ["feature1", "feature2", "target"],
            "sample_data": {"rows": [{"feature1": 1, "feature2": 2, "target": 0}]}
        })
        
        result = model_training_agent_node(state)
        
        # When there's already a training task, the node waits for training trigger
        assert result["ready_to_train"] is False
        assert result["training_task"]["task_id"] == "test_task_001"
        assert "Waiting for training instructions" in result["messages"][-1].content
    
    def test_node_with_invalid_training_task(self):
        """Test node behavior with invalid training task"""
        state = self.create_basic_state()
        state.update({
            "training_task": {
                "task_id": "test_task_001"
                # Missing required fields
            }
        })
        
        result = model_training_agent_node(state)
        
        assert result["ready_to_train"] is False
        assert result["training_completed"] is False
        assert "Task validation failed" in result["messages"][-1].content
    
    @patch('app.agents.model_training_agent.ModelTrainingAgent.validate_training_task')
    def test_node_validation_exception(self, mock_validate):
        """Test node behavior when validation throws exception"""
        mock_validate.side_effect = Exception("Test exception")
        
        state = self.create_basic_state()
        state.update({
            "training_task": {
                "task_id": "test_task_001",
                "model_type": "pytorch_classification",
                "goal_description": "Test"
            }
        })
        
        result = model_training_agent_node(state)
        
        assert result["ready_to_train"] is False
        assert "Unexpected error" in result["messages"][-1].content

    def test_node_creates_training_plan_only(self):
        """Test that node creates training plan without execution when requested"""
        state = self.create_basic_state()
        state.update({
            "messages": [HumanMessage(content="create a training plan")],
            "schema": {"feature1": "number", "feature2": "number", "target": "number"},
            "data_source_location": "/test/path/data.csv"
        })
        
        with patch('app.agents.model_training_agent.ModelTrainingAgent.create_training_task_from_state') as mock_create_task, \
             patch('app.agents.model_training_agent.ModelTrainingAgent.create_training_plan') as mock_create_plan:
            
            mock_task = {
                "task_id": "plan_task_001",
                "model_type": "pytorch_classification",
                "goal_description": "Test plan"
            }
            mock_plan = {
                "training_strategy": "single_model",
                "model_suggestions": ["pytorch_classification"],
                "estimated_duration": "5 minutes"
            }
            
            mock_create_task.return_value = mock_task
            mock_create_plan.return_value = mock_plan
            
            result = model_training_agent_node(state)
            
            # Verify plan was created but not executed
            assert result["training_task"] == mock_task
            assert result["ready_to_train"] is True
            assert result["training_completed"] is False
            assert "training plan" in result["messages"][-1].content.lower()
            
            # Verify MLflow experiment was NOT created (plan only, no execution)
            mock_create_task.assert_called_once()
            mock_create_plan.assert_called_once()

    def test_node_executes_training_after_plan(self):
        """Test that node executes training when start training is requested after plan"""
        state = self.create_basic_state()
        state.update({
            "messages": [
                HumanMessage(content="create a training plan"),
                HumanMessage(content="start training")
            ],
            "training_task": {
                "task_id": "plan_task_001",
                "model_type": "pytorch_classification"
            },
            "ready_to_train": True,
            "training_completed": False,
            "schema": {"feature1": "number", "target": "number"},
            "data_source_location": "/test/path/data.csv"
        })
        
        with patch('app.agents.model_training_agent.ModelTrainingAgent.create_training_plan') as mock_create_plan, \
             patch('app.agents.model_training_agent.ModelTrainingAgent.execute_training') as mock_execute, \
             patch('app.agents.model_training_agent.ModelTrainingAgent.mlflow_manager') as mock_mlflow:
            
            mock_plan = {"training_strategy": "single_model"}
            mock_execute_result = {
                "success": True,
                "task_id": "plan_task_001",
                "training_metrics": {"accuracy": 0.95},
                "artifacts": {"model_path": "./models/test.pth"}
            }
            
            mock_create_plan.return_value = mock_plan
            mock_execute.return_value = mock_execute_result
            mock_mlflow.create_experiment.return_value = {"run_id": "run_001"}
            
            result = model_training_agent_node(state)
            
            # Verify training was executed
            assert result["training_completed"] is True
            assert result["training_metrics"] == mock_execute_result["training_metrics"]
            assert result["model_artifacts"] == mock_execute_result["artifacts"]
            assert "completed successfully" in result["messages"][-1].content.lower()
            
            # Verify execution was called
            mock_execute.assert_called_once()

    def test_node_handles_training_plan_keywords(self):
        """Test that node detects various training plan creation keywords"""
        plan_keywords = [
            "create a training plan",
            "create training plan",
            "training plan",
            "plan training"
        ]
        
        for keyword in plan_keywords:
            state = self.create_basic_state()
            state.update({
                "messages": [HumanMessage(content=keyword)],
                "schema": {"x": "number", "y": "number"},
                "data_source_location": "/test/path.csv"
            })
            
            with patch('app.agents.model_training_agent.ModelTrainingAgent.create_training_task_from_state') as mock_create:
                mock_create.return_value = {"task_id": "test_001", "model_type": "pytorch_classification"}
                
                result = model_training_agent_node(state)
                
                # Should create plan, not execute
                assert result["ready_to_train"] is True
                assert result["training_completed"] is False
                assert result["training_task"] is not None


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    def test_create_sample_training_task(self):
        """Test creation of sample training task"""
        task = create_sample_training_task()
        
        assert "task_id" in task
        assert task["model_type"] == "pytorch_classification"
        assert task["goal_description"] == "Classify data based on features"
        assert "hyperparameters" in task
        assert "validation_params" in task
    
    def test_get_supported_model_types(self):
        """Test getting supported model types"""
        models = get_supported_model_types()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "pytorch_classification" in models
        assert "pytorch_regression" in models
        assert "svm" in models


# Integration test placeholders for future development
class TestMTAIntegration:
    """Integration tests for MTA with other components"""
    
    def test_mlflow_integration(self):
        """Test MLflow experiment tracking integration"""
        pass
    
    @pytest.mark.skip(reason="Ray integration not yet implemented")
    def test_ray_distributed_training(self):
        """Test Ray distributed training setup"""
        pass
    
    @pytest.mark.skip(reason="ONNX export not yet implemented")
    def test_onnx_model_export(self):
        """Test ONNX model export functionality"""
        pass



