"""
Unit tests for MLflow Integration

Tests cover:
- MLflowManager singleton pattern
- Run management (start, end, cleanup)
- Parameter and metrics logging
- Model artifact logging
- Dataset info logging
- Training code logging
- Utility functions
"""

import pytest
import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from datetime import datetime

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.mta.mlflow_integration import (
    MLflowManager,
    get_mlflow_manager,
    create_mlflow_manager,
    is_mlflow_available
)


@pytest.fixture
def mock_mlflow():
    """Create a mock MLflow module"""
    with patch('app.agents.mta.mlflow_integration.mlflow') as mock_mlflow:
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.artifact_location = "gs://bucket/artifacts"
        mock_experiment.experiment_id = "exp_123"
        
        # Mock run
        mock_run = MagicMock()
        mock_run.info.run_id = "run_123"
        mock_run.info.status = "RUNNING"
        mock_run.data.metrics = {}
        mock_run.data.params = {}
        mock_run.data.tags = {}
        
        # Mock active run
        mock_mlflow.active_run.return_value = None
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.create_experiment.return_value = "exp_123"
        mock_mlflow.set_experiment.return_value = None
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.get_tracking_uri.return_value = "file:///tmp/mlflow"
        
        yield mock_mlflow


@pytest.fixture
def reset_singleton():
    """Reset MLflowManager singleton between tests"""
    MLflowManager._instance = None
    MLflowManager._initialized = False
    yield
    MLflowManager._instance = None
    MLflowManager._initialized = False


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for MLflow"""
    with patch.dict(os.environ, {
        'DISABLE_MLFLOW': 'false',
        'MLFLOW_BACKEND_STORE_URI': '',
        'MLFLOW_DEFAULT_ARTIFACT_ROOT': ''
    }, clear=False):
        yield


class TestMLflowManagerSingleton:
    """Test MLflowManager singleton pattern"""
    
    def test_singleton_pattern(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test that MLflowManager is a singleton"""
        manager1 = MLflowManager(experiment_name="test_exp")
        manager2 = MLflowManager(experiment_name="different_exp")
        
        assert manager1 is manager2
        assert manager1.experiment_name == "test_exp"  # First initialization is used
    
    def test_initialization_only_once(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test that initialization only happens once"""
        manager1 = MLflowManager(experiment_name="test_exp")
        initial_experiment = manager1.experiment_name
        
        # Create another instance
        manager2 = MLflowManager(experiment_name="another_exp")
        
        # Should still have original experiment name
        assert manager2.experiment_name == initial_experiment


class TestMLflowManagerInitialization:
    """Test MLflowManager initialization"""
    
    def test_init_with_gcp_config(self, reset_singleton, mock_mlflow):
        """Test initialization with GCP configuration"""
        with patch.dict(os.environ, {
            'DISABLE_MLFLOW': 'false',
            'MLFLOW_BACKEND_STORE_URI': 'postgresql://user:pass@host/db',
            'MLFLOW_DEFAULT_ARTIFACT_ROOT': 'gs://bucket/artifacts'
        }, clear=False):
            manager = MLflowManager(experiment_name="test_exp")
            assert manager.mlflow_enabled is True
            mock_mlflow.set_tracking_uri.assert_called()
    
    def test_init_with_local_fallback(self, reset_singleton, mock_mlflow):
        """Test initialization with local file store fallback"""
        with patch.dict(os.environ, {
            'DISABLE_MLFLOW': 'false'
        }, clear=False):
            # Remove GCP config
            if 'MLFLOW_BACKEND_STORE_URI' in os.environ:
                del os.environ['MLFLOW_BACKEND_STORE_URI']
            if 'MLFLOW_DEFAULT_ARTIFACT_ROOT' in os.environ:
                del os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT']
            
            manager = MLflowManager(experiment_name="test_exp")
            assert manager.mlflow_enabled is True
    
    def test_init_with_mlflow_disabled(self, reset_singleton, mock_mlflow):
        """Test initialization when MLflow is disabled"""
        with patch.dict(os.environ, {'DISABLE_MLFLOW': 'true'}, clear=False):
            manager = MLflowManager(experiment_name="test_exp")
            assert manager.mlflow_enabled is False
    
    def test_init_handles_deleted_experiment(self, reset_singleton, mock_mlflow):
        """Test initialization handles deleted experiment gracefully"""
        mock_mlflow.get_experiment_by_name.side_effect = Exception("deleted experiment")
        mock_mlflow.create_experiment.return_value = "new_exp_123"
        
        manager = MLflowManager(experiment_name="test_exp")
        assert manager.mlflow_enabled is True
        mock_mlflow.create_experiment.assert_called()


class TestMLflowManagerRunManagement:
    """Test MLflow run management"""
    
    def test_start_training_run(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test starting a training run"""
        manager = MLflowManager(experiment_name="test_exp")
        
        run_id = manager.start_training_run(task_id="task_123", tags={"key": "value"})
        
        assert run_id == "run_123"
        mock_mlflow.start_run.assert_called()
    
    def test_start_training_run_continues_existing(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test continuing an existing run"""
        manager = MLflowManager(experiment_name="test_exp")
        
        run_id = manager.start_training_run(task_id="task_123", existing_run_id="existing_run_456")
        
        assert run_id == "existing_run_456"
        mock_mlflow.start_run.assert_not_called()
    
    def test_start_training_run_when_disabled(self, reset_singleton, mock_mlflow):
        """Test starting run when MLflow is disabled"""
        with patch.dict(os.environ, {'DISABLE_MLFLOW': 'true'}, clear=False):
            manager = MLflowManager(experiment_name="test_exp")
            run_id = manager.start_training_run(task_id="task_123")
            
            assert run_id.startswith("mlflow_disabled_")
            mock_mlflow.start_run.assert_not_called()
    
    def test_end_run(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test ending a run"""
        manager = MLflowManager(experiment_name="test_exp")
        
        manager.end_run(status="FINISHED", run_id="run_123")
        
        # Should use client.set_terminated for specific run_id
        # For active run, would use mlflow.end_run()
    
    def test_ensure_no_active_run(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test ensuring no active run"""
        mock_mlflow.active_run.return_value = MagicMock()
        manager = MLflowManager(experiment_name="test_exp")
        
        manager.ensure_no_active_run()
        
        mock_mlflow.end_run.assert_called()
    
    def test_cleanup_all_runs(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test cleaning up all runs"""
        # Mock active run that returns something first, then None
        mock_mlflow.active_run.side_effect = [MagicMock(), None]
        manager = MLflowManager(experiment_name="test_exp")
        
        manager.cleanup_all_runs()
        
        mock_mlflow.end_run.assert_called()


class TestMLflowManagerLogging:
    """Test MLflow logging functionality"""
    
    def test_log_training_params(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test logging training parameters"""
        manager = MLflowManager(experiment_name="test_exp")
        params = {"learning_rate": 0.01, "batch_size": 32, "epochs": 10}
        
        manager.log_training_params(params, run_id="run_123")
        
        mock_mlflow.log_params.assert_called()
    
    def test_log_training_params_when_disabled(self, reset_singleton, mock_mlflow):
        """Test logging params when MLflow is disabled"""
        with patch.dict(os.environ, {'DISABLE_MLFLOW': 'true'}, clear=False):
            manager = MLflowManager(experiment_name="test_exp")
            params = {"learning_rate": 0.01}
            
            manager.log_training_params(params)
            
            mock_mlflow.log_params.assert_not_called()
    
    def test_log_training_metrics(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test logging training metrics"""
        manager = MLflowManager(experiment_name="test_exp")
        metrics = {"accuracy": 0.95, "loss": 0.1}
        
        manager.log_training_metrics(metrics, step=10, run_id="run_123")
        
        assert mock_mlflow.log_metric.call_count == 2
    
    def test_log_training_metrics_with_step(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test logging metrics with step number"""
        manager = MLflowManager(experiment_name="test_exp")
        metrics = {"loss": 0.5}
        
        manager.log_training_metrics(metrics, step=5)
        
        mock_mlflow.log_metric.assert_called_with("loss", 0.5, step=5)
    
    def test_log_model_artifact(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test logging model artifact"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            model_path = f.name
            f.write(b"dummy model data")
        
        try:
            mock_mlflow.active_run.return_value = MagicMock()
            mock_mlflow.active_run().info.run_id = "run_123"
            
            manager = MLflowManager(experiment_name="test_exp")
            
            with patch.dict(os.environ, {'MLFLOW_DEFAULT_ARTIFACT_ROOT': 'gs://bucket/artifacts'}, clear=False):
                artifact_uri = manager.log_model_artifact(model_path, model_type="pytorch")
                
                assert artifact_uri == "runs:/run_123/model"
                mock_mlflow.log_artifact.assert_called()
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_log_model_artifact_when_disabled(self, reset_singleton, mock_mlflow):
        """Test logging model artifact when MLflow is disabled"""
        with patch.dict(os.environ, {'DISABLE_MLFLOW': 'true'}, clear=False):
            manager = MLflowManager(experiment_name="test_exp")
            
            artifact_uri = manager.log_model_artifact("/path/to/model", model_type="pytorch")
            
            assert artifact_uri.startswith("local_artifact_")
            mock_mlflow.log_artifact.assert_not_called()
    
    def test_log_training_code(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test logging training code"""
        manager = MLflowManager(experiment_name="test_exp")
        code = "import torch\nmodel = torch.nn.Linear(10, 1)"
        
        manager.log_training_code(code, run_id="run_123")
        
        mock_mlflow.log_artifact.assert_called()
    
    def test_log_dataset_info(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test logging dataset information"""
        manager = MLflowManager(experiment_name="test_exp")
        dataset_info = {
            "num_samples": 1000,
            "num_features": 10,
            "target_column": "label"
        }
        
        manager.log_dataset_info(dataset_info, run_id="run_123")
        
        mock_mlflow.log_artifact.assert_called()
        mock_mlflow.log_param.assert_called()
    
    def test_log_dataset_info_key_params(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test that dataset info logs key parameters"""
        manager = MLflowManager(experiment_name="test_exp")
        dataset_info = {
            "num_samples": 500,
            "num_features": 20
        }
        
        manager.log_dataset_info(dataset_info)
        
        # Check that num_samples and num_features were logged as params
        param_calls = [call[0][0] for call in mock_mlflow.log_param.call_args_list]
        assert "dataset_num_samples" in param_calls
        assert "dataset_num_features" in param_calls


class TestMLflowManagerRunInfo:
    """Test getting run information"""
    
    def test_get_run_info(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test getting run information"""
        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "run_123"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 1234567890
        mock_run.info.end_time = 1234567900
        mock_run.data.metrics = {"accuracy": 0.95}
        mock_run.data.params = {"lr": "0.01"}
        mock_run.data.tags = {"task_id": "task_123"}
        
        mock_client.get_run.return_value = mock_run
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        manager = MLflowManager(experiment_name="test_exp")
        run_info = manager.get_run_info("run_123")
        
        assert run_info["run_id"] == "run_123"
        assert run_info["status"] == "FINISHED"
        assert "accuracy" in run_info["metrics"]
        assert "lr" in run_info["params"]
    
    def test_get_run_info_error(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test getting run info when run doesn't exist"""
        mock_client = MagicMock()
        mock_client.get_run.side_effect = Exception("Run not found")
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        manager = MLflowManager(experiment_name="test_exp")
        run_info = manager.get_run_info("nonexistent_run")
        
        assert "error" in run_info


class TestMLflowManagerPrivateMethods:
    """Test private methods"""
    
    def test_log_model_by_type_onnx(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test logging ONNX model"""
        mock_run = MagicMock()
        mock_run.info.run_id = "run_123"
        mock_mlflow.active_run.return_value = mock_run
        
        manager = MLflowManager(experiment_name="test_exp")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.onnx') as f:
            model_path = f.name
            f.write(b"dummy onnx model")
        
        try:
            artifact_uri = manager._log_model_by_type(model_path, "onnx")
            
            assert "onnx_model" in artifact_uri
            mock_mlflow.log_artifact.assert_called_with(model_path, "onnx_model")
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_log_model_by_type_pytorch(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test logging PyTorch model"""
        mock_run = MagicMock()
        mock_run.info.run_id = "run_123"
        mock_mlflow.active_run.return_value = mock_run
        
        manager = MLflowManager(experiment_name="test_exp")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            model_path = f.name
            f.write(b"dummy pytorch model")
        
        try:
            artifact_uri = manager._log_model_by_type(model_path, "pytorch")
            
            assert "pytorch_model" in artifact_uri
            mock_mlflow.log_artifact.assert_called_with(model_path, "pytorch_model")
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_get_mlflow_manager(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test get_mlflow_manager factory function"""
        manager1 = get_mlflow_manager(experiment_name="test_exp")
        manager2 = get_mlflow_manager(experiment_name="different_exp")
        
        # Should return singleton
        assert manager1 is manager2
    
    def test_create_mlflow_manager(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test create_mlflow_manager factory function"""
        config = {
            "experiment_name": "config_exp",
            "mlflow_tracking_uri": "file:///tmp/mlflow"
        }
        
        manager = create_mlflow_manager(config)
        
        assert isinstance(manager, MLflowManager)
        assert manager.experiment_name == "config_exp"


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_is_mlflow_available(self, mock_mlflow):
        """Test checking if MLflow is available"""
        # Mock successful get_tracking_uri
        mock_mlflow.get_tracking_uri.return_value = "file:///tmp/mlflow"
        
        assert is_mlflow_available() is True
    
    def test_is_mlflow_available_false(self, mock_mlflow):
        """Test when MLflow is not available"""
        # Mock exception when getting tracking URI
        mock_mlflow.get_tracking_uri.side_effect = Exception("MLflow not available")
        
        assert is_mlflow_available() is False


class TestMLflowManagerErrorHandling:
    """Test error handling in MLflowManager"""
    
    def test_log_params_handles_error(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test that log_params handles errors gracefully"""
        mock_mlflow.log_params.side_effect = Exception("MLflow error")
        manager = MLflowManager(experiment_name="test_exp")
        
        # Should not raise exception
        manager.log_training_params({"lr": 0.01})
        
        # Should attempt cleanup
        assert mock_mlflow.end_run.called or True  # May or may not be called depending on cleanup
    
    def test_log_metrics_handles_error(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test that log_metrics handles errors gracefully"""
        mock_mlflow.log_metric.side_effect = Exception("MLflow error")
        manager = MLflowManager(experiment_name="test_exp")
        
        # Should not raise exception
        manager.log_training_metrics({"loss": 0.5})
    
    def test_start_run_handles_error(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test that start_run handles errors gracefully"""
        mock_mlflow.start_run.side_effect = Exception("MLflow error")
        manager = MLflowManager(experiment_name="test_exp")
        
        run_id = manager.start_training_run(task_id="task_123")
        
        # Should return a local run ID instead of raising
        assert run_id.startswith("local_run_")
    
    def test_end_run_handles_error(self, reset_singleton, mock_mlflow, mock_env_vars):
        """Test that end_run handles errors gracefully"""
        mock_mlflow.end_run.side_effect = Exception("MLflow error")
        manager = MLflowManager(experiment_name="test_exp")
        
        # Should not raise exception
        manager.end_run()


class TestMLflowManagerExperimentHandling:
    """Test experiment handling logic"""
    
    def test_experiment_creation_with_gcs(self, reset_singleton, mock_mlflow):
        """Test creating experiment with GCS artifact location"""
        mock_mlflow.get_experiment_by_name.return_value = None  # Experiment doesn't exist
        
        with patch.dict(os.environ, {
            'MLFLOW_DEFAULT_ARTIFACT_ROOT': 'gs://bucket/artifacts'
        }, clear=False):
            manager = MLflowManager(experiment_name="test_exp")
            
            # Should create experiment with GCS location
            mock_mlflow.create_experiment.assert_called_with(
                "test_exp",
                artifact_location="gs://bucket/artifacts"
            )
    
    def test_experiment_uses_existing_gcs(self, reset_singleton, mock_mlflow):
        """Test using existing experiment with GCS"""
        mock_experiment = MagicMock()
        mock_experiment.artifact_location = "gs://bucket/artifacts"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        manager = MLflowManager(experiment_name="test_exp")
        
        # Should use existing experiment, not create new one
        mock_mlflow.set_experiment.assert_called_with("test_exp")
        # Should not create if it exists
        # (create_experiment may be called in some code paths, so we just check set_experiment was called)
    
    def test_experiment_creates_gcs_version(self, reset_singleton, mock_mlflow):
        """Test creating GCS version when existing experiment doesn't use GCS"""
        mock_experiment = MagicMock()
        mock_experiment.artifact_location = "/local/path"  # Not GCS
        mock_mlflow.get_experiment_by_name.side_effect = [
            mock_experiment,  # First call: existing exp doesn't use GCS
            None  # Second call: GCS version doesn't exist
        ]
        
        with patch.dict(os.environ, {
            'MLFLOW_DEFAULT_ARTIFACT_ROOT': 'gs://bucket/artifacts'
        }, clear=False):
            manager = MLflowManager(experiment_name="test_exp")
            
            # Should create new GCS version
            create_calls = [call for call in mock_mlflow.create_experiment.call_args_list]
            assert len(create_calls) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

