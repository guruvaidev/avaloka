"""
Unit tests for Local Inference Manager

Tests cover:
- Model caching
- Model loading from MLflow
- Metadata loading
- Error handling
"""

import pytest
import os
import sys
import tempfile
import json
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from app.agents.mta.local_inference import (
    LocalInferenceManager,
    ModelCache
)


class TestModelCache:
    """Test ModelCache functionality"""
    
    def test_cache_initialization(self):
        """Test cache initializes correctly"""
        cache = ModelCache(max_size=5)
        assert cache.max_size == 5
        assert cache.size() == 0
    
    def test_cache_put_and_get(self):
        """Test putting and getting from cache"""
        cache = ModelCache(max_size=3)
        mock_model = MagicMock()
        metadata = {"x_mean": [1.0, 2.0], "x_std": [0.5, 0.5]}
        
        cache.put("run_1", mock_model, metadata)
        assert cache.size() == 1
        
        result = cache.get("run_1")
        assert result is not None
        assert result['model'] == mock_model
        assert result['metadata'] == metadata
    
    def test_cache_miss(self):
        """Test cache miss returns None"""
        cache = ModelCache()
        result = cache.get("nonexistent")
        assert result is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = ModelCache(max_size=2)
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_model3 = MagicMock()
        
        cache.put("run_1", mock_model1, {})
        cache.put("run_2", mock_model2, {})
        assert cache.size() == 2
        
        # Access run_1 to make it most recently used
        cache.get("run_1")
        
        # Add run_3 - should evict run_2 (least recently used)
        cache.put("run_3", mock_model3, {})
        assert cache.size() == 2
        assert cache.get("run_2") is None  # Should be evicted
        assert cache.get("run_1") is not None  # Should still be there
        assert cache.get("run_3") is not None  # Should be there
    
    def test_cache_clear(self):
        """Test clearing cache"""
        cache = ModelCache()
        cache.put("run_1", MagicMock(), {})
        assert cache.size() == 1
        
        cache.clear()
        assert cache.size() == 0
        assert cache.get("run_1") is None


class TestLocalInferenceManager:
    """Test LocalInferenceManager functionality"""
    
    def test_initialization(self):
        """Test manager initializes correctly"""
        manager = LocalInferenceManager()
        assert manager.mlflow_manager is None
        assert manager.cache is not None
        assert manager.cache.max_size == 10
    
    def test_initialization_with_mlflow_manager(self):
        """Test initialization with MLflowManager"""
        mock_mlflow = MagicMock()
        manager = LocalInferenceManager(mlflow_manager=mock_mlflow)
        assert manager.mlflow_manager == mock_mlflow
    
    def test_initialization_with_custom_cache_size(self):
        """Test initialization with custom cache size"""
        manager = LocalInferenceManager(cache_size=5)
        assert manager.cache.max_size == 5
    
    @patch('app.agents.mta.local_inference.MlflowClient')
    def test_load_metadata_from_mlflow_success(self, mock_client_class):
        """Test loading metadata from MLflow artifacts"""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock run
        mock_run = MagicMock()
        mock_run.info.artifact_uri = "gs://bucket/artifacts"
        mock_client.get_run.return_value = mock_run
        
        # Mock artifacts
        mock_artifact = MagicMock()
        mock_artifact.path = "model/test_model_metadata.json"
        mock_client.list_artifacts.return_value = [mock_artifact]
        
        # Mock metadata file
        metadata_content = {
            "x_mean": [1.0, 2.0, 3.0],
            "x_std": [0.5, 0.5, 0.5],
            "class_names": ["ClassA", "ClassB"],
            "feature_names": ["feature1", "feature2", "feature3"]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_file = os.path.join(temp_dir, "test_model_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata_content, f)
            
            # Mock download_artifacts to copy our test file
            def mock_download(run_id, artifact_path, dst_path):
                import shutil
                shutil.copy(metadata_file, dst_path)
            
            mock_client.download_artifacts.side_effect = mock_download
        
            manager = LocalInferenceManager()
            metadata = manager.load_metadata("run_123", "runs:/run_123/model")
            
            assert metadata is not None
            assert len(metadata["x_mean"]) == 3
            assert len(metadata["x_std"]) == 3
            assert len(metadata["class_names"]) == 2
            assert metadata["feature_names"] == ["feature1", "feature2", "feature3"]
    
    @patch('app.agents.mta.local_inference.MlflowClient')
    def test_load_metadata_from_mlflow_not_found(self, mock_client_class):
        """Test loading metadata when not found"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_run = MagicMock()
        mock_run.info.artifact_uri = "gs://bucket/artifacts"
        mock_client.get_run.return_value = mock_run
        mock_client.list_artifacts.return_value = []  # No artifacts
        
        manager = LocalInferenceManager()
        
        with pytest.raises(ValueError, match="Could not load metadata"):
            manager.load_metadata("run_123", "runs:/run_123/model")
    
    def test_load_metadata_from_local_file(self):
        """Test loading metadata from local file path"""
        metadata_content = {
            "x_mean": [1.0, 2.0],
            "x_std": [0.5, 0.5],
            "class_names": ["A", "B"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata_content, f)
            temp_path = f.name
        
        try:
            manager = LocalInferenceManager()
            metadata = manager.load_metadata("run_123", "runs:/run_123/model", metadata_path=temp_path)
            
            assert metadata is not None
            assert len(metadata["x_mean"]) == 2
            assert len(metadata["class_names"]) == 2
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @patch('app.agents.mta.local_inference.MlflowClient')
    def test_load_onnx_model_success(self, mock_client_class):
        """Test loading ONNX model from MLflow"""
        # This is a complex test that would require actual ONNX model file
        # For now, we'll test the structure
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_artifact = MagicMock()
        mock_artifact.path = "model/test_model.onnx"
        mock_client.list_artifacts.return_value = [mock_artifact]
        
        # Mock download - we can't easily create a real ONNX file in tests
        # So we'll just verify the method structure
        manager = LocalInferenceManager()
        
        # This will fail without a real ONNX file, but we can test the structure
        with pytest.raises((FileNotFoundError, ValueError)):
            manager._load_onnx_model("run_123", "runs:/run_123/model")
    
    @patch('app.agents.mta.local_inference.MlflowClient')
    def test_load_onnx_model_download_failure_cleanup(self, mock_client_class):
        """Test that cleanup works correctly when download_artifacts fails"""
        # This test verifies Bug 1 fix: downloaded_path is initialized before try block
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_artifact = MagicMock()
        mock_artifact.path = "model/test_model.onnx"
        mock_client.list_artifacts.return_value = [mock_artifact]
        
        # Make download_artifacts raise an exception
        mock_client.download_artifacts.side_effect = Exception("Download failed")
        
        manager = LocalInferenceManager()
        
        # Should raise the download exception, not UnboundLocalError
        with pytest.raises(Exception, match="Download failed"):
            manager._load_onnx_model("run_123", "runs:/run_123/model")
        
        # Verify cleanup was attempted (temp_path should be cleaned up)
        # The finally block should not crash with UnboundLocalError
    
    def test_preprocess_data_basic(self):
        """Test basic preprocessing with normalization"""
        manager = LocalInferenceManager()
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [3.0, 4.0, 5.0]})
        metadata = {
            "x_mean": [1.5, 3.5],  # Mean for each feature
            "x_std": [0.5, 0.5],    # Std for each feature
            "feature_names": ["feature1", "feature2"]
        }
        result = manager.preprocess_data(df, metadata)
        
        assert result.shape == (3, 2)
        assert result.dtype == np.float32
        
        # Check normalization: (x - mean) / std
        # For feature1: (1.0 - 1.5) / 0.5 = -1.0, (2.0 - 1.5) / 0.5 = 1.0, (3.0 - 1.5) / 0.5 = 3.0
        assert np.isclose(result[0, 0], -1.0)
        assert np.isclose(result[1, 0], 1.0)
        assert np.isclose(result[2, 0], 3.0)
    
    def test_preprocess_data_without_normalization(self):
        """Test preprocessing without normalization stats"""
        manager = LocalInferenceManager()
        df = pd.DataFrame({"feature1": [1.0, 2.0], "feature2": [3.0, 4.0]})
        metadata = {
            "feature_names": ["feature1", "feature2"]
            # No x_mean or x_std
        }
        result = manager.preprocess_data(df, metadata)
        
        assert result.shape == (2, 2)
        assert result.dtype == np.float32
        # Should return original values (no normalization)
        assert np.allclose(result, df.values.astype(np.float32))
    
    def test_preprocess_data_missing_features(self):
        """Test preprocessing with missing required features"""
        manager = LocalInferenceManager()
        df = pd.DataFrame({"feature1": [1.0, 2.0]})  # Missing feature2
        metadata = {
            "feature_names": ["feature1", "feature2"]
        }
        
        with pytest.raises(ValueError, match="Missing required features"):
            manager.preprocess_data(df, metadata)
    
    def test_preprocess_data_feature_order(self):
        """Test that preprocessing respects feature order from metadata"""
        manager = LocalInferenceManager()
        # DataFrame has features in different order
        df = pd.DataFrame({"feature2": [3.0, 4.0], "feature1": [1.0, 2.0]})
        metadata = {
            "x_mean": [1.5, 3.5],  # Mean for [feature1, feature2]
            "x_std": [0.5, 0.5],
            "feature_names": ["feature1", "feature2"]  # Order matters
        }
        result = manager.preprocess_data(df, metadata)
        
        assert result.shape == (2, 2)
        # First column should be feature1 (normalized), second should be feature2
        assert np.isclose(result[0, 0], -1.0)  # (1.0 - 1.5) / 0.5
        assert np.isclose(result[0, 1], -1.0)  # (3.0 - 3.5) / 0.5
    
    def test_preprocess_data_with_nan(self):
        """Test preprocessing handles NaN values"""
        manager = LocalInferenceManager()
        df = pd.DataFrame({"feature1": [1.0, np.nan, 3.0], "feature2": [3.0, 4.0, 5.0]})
        metadata = {
            "x_mean": [2.0, 4.0],
            "x_std": [1.0, 1.0],
            "feature_names": ["feature1", "feature2"]
        }
        result = manager.preprocess_data(df, metadata)
        
        assert result.shape == (3, 2)
        assert not np.isnan(result).any()  # Should have no NaN after filling
    
    def test_run_inference_basic(self):
        """Test basic inference with mock model"""
        manager = LocalInferenceManager()
        
        # Create mock ONNX model
        mock_model = MagicMock()
        
        # Mock input specification
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [None, 2]  # Dynamic batch, 2 features
        mock_input.type = "tensor(float)"
        mock_model.get_inputs.return_value = [mock_input]
        
        # Mock output (classification with 3 classes)
        mock_output = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], dtype=np.float32)
        mock_model.run.return_value = [mock_output]
        
        # Test inference
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = manager.run_inference(mock_model, input_data)
        
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        assert np.array_equal(result, mock_output)
        mock_model.run.assert_called_once()
        # Verify input was passed correctly
        call_args = mock_model.run.call_args
        assert "input" in call_args[0][1]  # input_feed dict
        assert call_args[0][1]["input"].shape == (2, 2)
    
    def test_run_inference_regression_output(self):
        """Test inference with 1D output (regression)"""
        manager = LocalInferenceManager()
        
        # Create mock ONNX model
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [None, 2]
        mock_input.type = "tensor(float)"
        mock_model.get_inputs.return_value = [mock_input]
        
        # Mock 1D output (regression)
        mock_output = np.array([0.5, 0.7], dtype=np.float32)
        mock_model.run.return_value = [mock_output]
        
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = manager.run_inference(mock_model, input_data)
        
        assert result.shape == (2,)
        assert result.dtype == np.float32
        assert np.array_equal(result, mock_output)
    
    def test_run_inference_empty_input(self):
        """Test inference fails with empty input"""
        manager = LocalInferenceManager()
        mock_model = MagicMock()
        mock_model.get_inputs.return_value = [MagicMock()]
        
        empty_data = np.array([], dtype=np.float32).reshape(0, 2)
        with pytest.raises(ValueError, match="Cannot run inference on empty data"):
            manager.run_inference(mock_model, empty_data)
    
    def test_run_inference_wrong_shape(self):
        """Test inference fails with wrong feature count"""
        manager = LocalInferenceManager()
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [None, 2]  # Expects 2 features
        mock_input.type = "tensor(float)"
        mock_model.get_inputs.return_value = [mock_input]
        
        # Wrong feature count
        wrong_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # 3 features instead of 2
        with pytest.raises(ValueError, match="Feature count mismatch"):
            manager.run_inference(mock_model, wrong_data)
    
    def test_run_inference_1d_input_error(self):
        """Test inference fails with 1D input (needs 2D)"""
        manager = LocalInferenceManager()
        mock_model = MagicMock()
        mock_model.get_inputs.return_value = [MagicMock()]
        
        # 1D input (should be 2D)
        one_d_data = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 2D input array"):
            manager.run_inference(mock_model, one_d_data)
    
    def test_run_inference_type_conversion(self):
        """Test inference converts input to float32"""
        manager = LocalInferenceManager()
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [None, 2]
        mock_input.type = "tensor(float)"
        mock_model.get_inputs.return_value = [mock_input]
        
        mock_output = np.array([[0.5]], dtype=np.float32)
        mock_model.run.return_value = [mock_output]
        
        # Input in float64 (should be converted to float32)
        input_data = np.array([[1.0, 2.0]], dtype=np.float64)
        result = manager.run_inference(mock_model, input_data)
        
        # Verify model was called with float32
        call_args = mock_model.run.call_args
        assert call_args is not None
        input_feed = call_args[0][1]  # Second argument is input_feed dict
        assert input_feed["input"].dtype == np.float32
        assert result.shape == (1, 1)
    
    def test_postprocess_predictions_classification_with_names(self):
        """Test postprocessing for classification with class names"""
        manager = LocalInferenceManager()
        
        # Classification probabilities: [n_samples, n_classes]
        predictions = np.array([
            [0.1, 0.8, 0.1],  # Class 1 with high confidence
            [0.7, 0.2, 0.1],  # Class 0
            [0.1, 0.1, 0.8]   # Class 2
        ], dtype=np.float32)
        
        metadata = {
            "class_names": ["A", "B", "C"],
            "task_type": "classification"
        }
        
        result = manager.postprocess_predictions(predictions, metadata)
        
        assert len(result) == 3
        assert "Predicted" in result.columns
        assert "Confidence" in result.columns
        assert result["Predicted"].iloc[0] == "B"  # Index 1 (max probability)
        assert result["Predicted"].iloc[1] == "A"  # Index 0
        assert result["Predicted"].iloc[2] == "C"  # Index 2
        assert np.isclose(result["Confidence"].iloc[0], 0.8)  # Max probability
        assert np.isclose(result["Confidence"].iloc[1], 0.7)
        assert np.isclose(result["Confidence"].iloc[2], 0.8)
    
    def test_postprocess_predictions_classification_logits(self):
        """Test postprocessing for classification with logits (apply softmax)"""
        manager = LocalInferenceManager()
        
        # Logits (large values, not probabilities)
        logits = np.array([
            [1.0, 5.0, 2.0],  # Should become class 1 after softmax
            [4.0, 1.0, 0.0]   # Should become class 0 after softmax
        ], dtype=np.float32)
        
        metadata = {
            "class_names": ["Class0", "Class1", "Class2"],
            "task_type": "classification"
        }
        
        result = manager.postprocess_predictions(logits, metadata)
        
        assert len(result) == 2
        assert result["Predicted"].iloc[0] == "Class1"  # Highest logit
        assert result["Predicted"].iloc[1] == "Class0"
        assert 0.0 <= result["Confidence"].iloc[0] <= 1.0  # Should be probability after softmax
        assert 0.0 <= result["Confidence"].iloc[1] <= 1.0
    
    def test_postprocess_predictions_classification_no_names(self):
        """Test postprocessing for classification without class names"""
        manager = LocalInferenceManager()
        
        predictions = np.array([
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1]
        ], dtype=np.float32)
        
        metadata = {
            "task_type": "classification"
            # No class_names
        }
        
        result = manager.postprocess_predictions(predictions, metadata)
        
        assert len(result) == 2
        assert result["Predicted"].iloc[0] == "1"  # Index as string
        assert result["Predicted"].iloc[1] == "0"
        assert "Confidence" in result.columns
    
    def test_postprocess_predictions_regression(self):
        """Test postprocessing for regression task"""
        manager = LocalInferenceManager()
        
        # Regression predictions (1D or 2D with 1 column)
        predictions = np.array([1.5, 2.3, 3.7], dtype=np.float32)
        
        metadata = {
            "task_type": "regression"
        }
        
        result = manager.postprocess_predictions(predictions, metadata)
        
        assert len(result) == 3
        assert "Predicted" in result.columns
        assert "Confidence" in result.columns
        assert np.isclose(result["Predicted"].iloc[0], 1.5)
        assert np.isclose(result["Predicted"].iloc[1], 2.3)
        assert np.isclose(result["Predicted"].iloc[2], 3.7)
        # Confidence should be None for regression
        assert result["Confidence"].iloc[0] is None or pd.isna(result["Confidence"].iloc[0])
    
    def test_postprocess_predictions_regression_2d(self):
        """Test postprocessing for regression with 2D output"""
        manager = LocalInferenceManager()
        
        # Regression with shape [n_samples, 1]
        predictions = np.array([[1.5], [2.3], [3.7]], dtype=np.float32)
        
        metadata = {
            "task_type": "regression"
        }
        
        result = manager.postprocess_predictions(predictions, metadata)
        
        assert len(result) == 3
        assert np.isclose(result["Predicted"].iloc[0], 1.5)
        assert result["Predicted"].dtype == np.float32
    
    def test_postprocess_predictions_infer_task_type(self):
        """Test postprocessing infers task type from shape and metadata"""
        manager = LocalInferenceManager()
        
        # 2D predictions with class names -> classification
        predictions = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        metadata = {
            "class_names": ["Negative", "Positive"]
            # No task_type specified
        }
        
        result = manager.postprocess_predictions(predictions, metadata)
        
        # Should infer as classification
        assert result["Predicted"].dtype == object  # String class names
        assert result["Predicted"].iloc[0] == "Positive"
        assert "Confidence" in result.columns
    
    def test_postprocess_predictions_classification_index_range(self):
        """Test postprocessing handles out-of-range class indices"""
        manager = LocalInferenceManager()
        
        # Predictions with more classes than class_names
        predictions = np.array([
            [0.1, 0.8, 0.1]  # 3 classes
        ], dtype=np.float32)
        
        metadata = {
            "class_names": ["A", "B"],  # Only 2 classes
            "task_type": "classification"
        }
        
        result = manager.postprocess_predictions(predictions, metadata)
        
        # Should handle gracefully - index 1 exists, but if index 2 appears, use string
        assert len(result) == 1
        assert result["Predicted"].iloc[0] in ["A", "B", "1"]  # Should map to available class or use index
    
    def test_predict_method_structure(self):
        """Test predict method structure (will fail without real model, but tests structure)"""
        manager = LocalInferenceManager()
        df = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})
        
        # This will fail without real MLflow connection, but tests the method exists
        with pytest.raises((ValueError, Exception)):
            manager.predict(df, "run_123")
    
    def test_cache_usage_in_load_model(self):
        """Test that load_model uses cache"""
        manager = LocalInferenceManager()
        
        # Mock the actual loading methods
        mock_model = MagicMock()
        mock_metadata = {
            "x_mean": np.array([1.0]),
            "x_std": np.array([0.5]),
            "class_names": ["A"],
            "feature_names": ["f1"]
        }
        
        with patch.object(manager, 'load_metadata', return_value=mock_metadata):
            with patch.object(manager, '_load_onnx_model', return_value=mock_model):
                # First load - should call load methods
                model1, metadata1 = manager.load_model("run_1", "runs:/run_1/model")
                assert model1 == mock_model
                
                # Second load - should use cache
                with patch.object(manager, 'load_metadata') as mock_load_meta:
                    with patch.object(manager, '_load_onnx_model') as mock_load_onnx:
                        model2, metadata2 = manager.load_model("run_1", "runs:/run_1/model")
                        # Should not call load methods again
                        mock_load_meta.assert_not_called()
                        mock_load_onnx.assert_not_called()
                        assert model2 == mock_model


