"""
Unit tests for ONNX Inference Job Script

Tests cover:
- Metadata loading from MLflow
- CSV loading from GCS and local paths
- Prediction logic with ONNX models
- Error handling
- Environment variable validation
"""

import pytest
import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the inference job module
from app.agents.mta.inference_docker import inference_job


class TestMetadataLoading:
    """Test metadata loading functionality"""
    
    @patch('app.agents.mta.inference_docker.inference_job.mlflow')
    def test_load_metadata_from_mlflow_with_metadata_file(self, mock_mlflow):
        """Test loading metadata from explicit metadata file"""
        # Setup mock metadata
        metadata = {
            "x_mean": [1.0, 2.0, 3.0, 4.0],
            "x_std": [0.5, 0.6, 0.7, 0.8],
            "class_names": ["setosa", "versicolor", "virginica"],
            "feature_names": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        }
        
        # Create temp metadata file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            metadata_path = f.name
        
        try:
            x_mean, x_std, class_names, feature_names = inference_job.load_metadata_from_mlflow(
                "runs:/test_run/model",
                metadata_path=metadata_path
            )
            
            assert len(x_mean) == 4
            assert len(x_std) == 4
            assert len(class_names) == 3
            assert feature_names == ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
            assert np.allclose(x_mean, [1.0, 2.0, 3.0, 4.0])
            assert np.allclose(x_std, [0.5, 0.6, 0.7, 0.8])
        finally:
            os.unlink(metadata_path)
    
    @patch('app.agents.mta.inference_docker.inference_job.mlflow')
    def test_load_metadata_missing_stats(self, mock_mlflow):
        """Test error when metadata is missing normalization stats"""
        metadata = {
            "class_names": ["setosa", "versicolor"]
            # Missing x_mean and x_std
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            metadata_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Metadata missing normalization stats"):
                inference_job.load_metadata_from_mlflow(
                    "runs:/test_run/model",
                    metadata_path=metadata_path
                )
        finally:
            os.unlink(metadata_path)
    
    @patch('app.agents.mta.inference_docker.inference_job.mlflow')
    def test_load_metadata_mismatch_stats(self, mock_mlflow):
        """Test error when x_mean and x_std have different lengths"""
        metadata = {
            "x_mean": [1.0, 2.0, 3.0],
            "x_std": [0.5, 0.6],  # Different length
            "class_names": ["setosa", "versicolor"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            metadata_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Mismatch"):
                inference_job.load_metadata_from_mlflow(
                    "runs:/test_run/model",
                    metadata_path=metadata_path
                )
        finally:
            os.unlink(metadata_path)


class TestCSVLoading:
    """Test CSV loading from GCS and local paths"""
    
    def test_load_csv_from_local_path(self):
        """Test loading CSV from local file path"""
        # Create test CSV
        df_test = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_test.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            df_loaded = inference_job.load_csv_from_gcs(csv_path)
            assert df_loaded.shape == (3, 3)
            assert list(df_loaded.columns) == ["feature1", "feature2", "target"]
        finally:
            os.unlink(csv_path)
    
    @patch('app.agents.mta.inference_docker.inference_job.storage')
    def test_load_csv_from_gcs(self, mock_storage):
        """Test loading CSV from GCS"""
        # Setup mock GCS client
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        mock_storage.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Create test CSV content
        csv_content = "feature1,feature2,target\n1,4,0\n2,5,1\n3,6,0\n"
        
        # Mock download to write CSV content to temp file
        def mock_download(filename):
            with open(filename, 'w') as f:
                f.write(csv_content)
        
        mock_blob.download_to_filename.side_effect = mock_download
        
        # Test loading
        gcs_path = "gs://test-bucket/test.csv"
        df = inference_job.load_csv_from_gcs(gcs_path)
        
        assert df.shape == (3, 3)
        assert list(df.columns) == ["feature1", "feature2", "target"]
        mock_blob.download_to_filename.assert_called_once()


class TestPrediction:
    """Test prediction functionality"""
    
    @patch('app.agents.mta.inference_docker.inference_job.ort')
    @patch('app.agents.mta.inference_docker.inference_job.mlflow')
    def test_predict_with_classification(self, mock_mlflow, mock_ort):
        """Test prediction for classification task"""
        # Setup test data
        df = pd.DataFrame({
            "SepalLengthCm": [5.1, 4.9, 5.0],
            "SepalWidthCm": [3.5, 3.0, 3.2],
            "PetalLengthCm": [1.4, 1.4, 1.5],
            "PetalWidthCm": [0.2, 0.2, 0.2]
        })
        
        # Setup metadata
        metadata = {
            "x_mean": [5.0, 3.0, 1.5, 0.2],
            "x_std": [0.5, 0.5, 0.5, 0.5],
            "class_names": ["setosa", "versicolor", "virginica"],
            "feature_names": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        }
        
        # Mock ONNX runtime
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [None, 4]  # Dynamic batch size
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [MagicMock()]
        
        # Mock inference output (logits for 3 samples, 3 classes)
        mock_logits = np.array([
            [2.0, 0.5, 0.1],  # Sample 1: class 0 (setosa)
            [0.3, 2.5, 0.2],  # Sample 2: class 1 (versicolor)
            [0.1, 0.2, 2.8]   # Sample 3: class 2 (virginica)
        ], dtype=np.float32)
        
        mock_session.run.return_value = [mock_logits]
        mock_ort.InferenceSession.return_value = mock_session
        
        # Mock MLflow artifact download
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            metadata_path = f.name
        
        # Mock MLflow client
        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.info.artifact_uri = "gs://bucket/artifacts"
        mock_client.get_run.return_value = mock_run
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        # Create ONNX file first
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as onnx_file:
            onnx_path = onnx_file.name
            # Write dummy ONNX file (just needs to exist)
            onnx_file.write(b"dummy onnx content")
        
        # Mock list_artifacts to return an artifact with .onnx extension
        mock_artifact = MagicMock()
        mock_artifact.path = "model.onnx"
        mock_client.list_artifacts.return_value = [mock_artifact]
        
        # Mock download_artifacts to return the ONNX path
        def mock_download_artifacts(run_id=None, artifact_path=None):
            # Return the ONNX path for any artifact download
            return onnx_path
        
        mock_mlflow.artifacts.download_artifacts.side_effect = mock_download_artifacts
        
        try:
            # Run prediction
            df_result = inference_job.predict(
                df,
                "runs:/test_run/model",
                metadata_path=metadata_path
            )
            
            # Check results
            assert "Predicted" in df_result.columns
            assert "Confidence" in df_result.columns
            assert len(df_result) == 3
            assert df_result["Predicted"].iloc[0] == "setosa"
            assert df_result["Predicted"].iloc[1] == "versicolor"
            assert df_result["Predicted"].iloc[2] == "virginica"
            
            # Check confidence values are between 0 and 1
            assert all(0 <= conf <= 1 for conf in df_result["Confidence"])
            
        finally:
            if os.path.exists(metadata_path):
                os.unlink(metadata_path)
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)
    
    @patch('app.agents.mta.inference_docker.inference_job.mlflow')
    def test_predict_missing_columns(self, mock_mlflow):
        """Test error when CSV is missing required columns"""
        df = pd.DataFrame({
            "SepalLengthCm": [5.1, 4.9],
            "SepalWidthCm": [3.5, 3.0]
            # Missing PetalLengthCm and PetalWidthCm
        })
        
        metadata = {
            "x_mean": [5.0, 3.0, 1.5, 0.2],
            "x_std": [0.5, 0.5, 0.5, 0.5],
            "class_names": ["setosa", "versicolor"],
            "feature_names": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            metadata_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                inference_job.predict(df, "runs:/test_run/model", metadata_path=metadata_path)
        finally:
            os.unlink(metadata_path)
    
    @patch('app.agents.mta.inference_docker.inference_job.mlflow')
    def test_predict_feature_count_mismatch(self, mock_mlflow):
        """Test error when feature count doesn't match"""
        df = pd.DataFrame({
            "feature1": [1, 2],
            "feature2": [3, 4]
            # Only 2 features, but model expects 4
        })
        
        metadata = {
            "x_mean": [1.0, 2.0, 3.0, 4.0],  # 4 features
            "x_std": [0.5, 0.5, 0.5, 0.5],
            "class_names": ["class1", "class2"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f)
            metadata_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Feature count mismatch"):
                inference_job.predict(df, "runs:/test_run/model", metadata_path=metadata_path)
        finally:
            os.unlink(metadata_path)


class TestCSVUpload:
    """Test CSV upload to GCS"""
    
    @patch('app.agents.mta.inference_docker.inference_job.storage')
    @patch('app.agents.mta.inference_docker.inference_job.os.remove')
    def test_upload_csv_to_gcs(self, mock_remove, mock_storage):
        """Test uploading DataFrame to GCS"""
        # Setup mock GCS client
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        mock_storage.Client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Create test DataFrame
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "predicted": ["A", "B", "A"]
        })
        
        # Upload
        gcs_path = "gs://test-bucket/output.csv"
        inference_job.upload_csv_to_gcs(df, gcs_path)
        
        # Verify upload was called
        mock_blob.upload_from_filename.assert_called_once()
        # Verify the uploaded file path was passed
        uploaded_file = mock_blob.upload_from_filename.call_args[0][0]
        # The file should exist (created by the function, not deleted because we mocked os.remove)
        assert os.path.exists(uploaded_file)
        # Verify the file contains the correct data
        df_loaded = pd.read_csv(uploaded_file)
        assert df_loaded.shape == df.shape
        assert list(df_loaded.columns) == list(df.columns)
        # Verify the data matches
        pd.testing.assert_frame_equal(df_loaded[["feature1", "predicted"]], df)
        
        # Verify cleanup was attempted
        mock_remove.assert_called_once_with(uploaded_file)
        
        # Cleanup
        if os.path.exists(uploaded_file):
            os.unlink(uploaded_file)
    
    def test_upload_csv_to_local_path(self):
        """Test saving DataFrame to local path"""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "predicted": ["A", "B", "A"]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            local_path = f.name
        
        try:
            inference_job.upload_csv_to_gcs(df, local_path)
            assert os.path.exists(local_path)
            df_loaded = pd.read_csv(local_path)
            assert df_loaded.shape == df.shape
        finally:
            if os.path.exists(local_path):
                os.unlink(local_path)


class TestMainFunction:
    """Test main() function with mocked environment"""
    
    @patch.dict(os.environ, {
        'MLFLOW_TRACKING_URI': 'http://localhost:5000',
        'MODEL_URI': 'runs:/test_run/model',
        'INPUT_PATH': '/tmp/test_input.csv',
        'OUTPUT_PATH': '/tmp/test_output.csv'
    })
    @patch('app.agents.mta.inference_docker.inference_job.load_csv_from_gcs')
    @patch('app.agents.mta.inference_docker.inference_job.predict')
    @patch('app.agents.mta.inference_docker.inference_job.upload_csv_to_gcs')
    @patch('app.agents.mta.inference_docker.inference_job.mlflow')
    def test_main_success(self, mock_mlflow, mock_upload, mock_predict, mock_load):
        """Test successful inference job execution"""
        # Setup mocks
        df_input = pd.DataFrame({"feature1": [1, 2, 3]})
        df_output = pd.DataFrame({
            "feature1": [1, 2, 3],
            "Predicted": ["A", "B", "A"],
            "Confidence": [0.9, 0.8, 0.9]
        })
        
        mock_load.return_value = df_input
        mock_predict.return_value = df_output
        
        # Run main - sys.exit raises SystemExit
        with pytest.raises(SystemExit) as exc_info:
            inference_job.main()
        assert exc_info.value.code == 0
        
        # Verify calls
        mock_load.assert_called_once()
        mock_predict.assert_called_once()
        mock_upload.assert_called_once()
    
    @patch.dict(os.environ, {
        'MLFLOW_TRACKING_URI': 'http://localhost:5000',
        # Missing MODEL_URI
        'INPUT_PATH': '/tmp/test_input.csv',
        'OUTPUT_PATH': '/tmp/test_output.csv'
    }, clear=False)
    def test_main_missing_model_uri(self):
        """Test error when MODEL_URI is missing"""
        # sys.exit raises SystemExit, so we need to catch it
        with pytest.raises(SystemExit) as exc_info:
            inference_job.main()
        assert exc_info.value.code == 1
    
    @patch.dict(os.environ, {
        'MLFLOW_TRACKING_URI': 'http://localhost:5000',
        'MODEL_URI': 'runs:/test_run/model',
        # Missing INPUT_PATH
        'OUTPUT_PATH': '/tmp/test_output.csv'
    }, clear=False)
    def test_main_missing_input_path(self):
        """Test error when INPUT_PATH is missing"""
        # sys.exit raises SystemExit, so we need to catch it
        with pytest.raises(SystemExit) as exc_info:
            inference_job.main()
        assert exc_info.value.code == 1
    
    @patch.dict(os.environ, {
        'MLFLOW_TRACKING_URI': 'http://localhost:5000',
        'MODEL_URI': 'runs:/test_run/model',
        'INPUT_PATH': '/tmp/test_input.csv',
        'OUTPUT_PATH': '/tmp/test_output.csv'
    }, clear=False)
    @patch('app.agents.mta.inference_docker.inference_job.load_csv_from_gcs')
    @patch('app.agents.mta.inference_docker.inference_job.predict')
    @patch('app.agents.mta.inference_docker.inference_job.upload_csv_to_gcs')
    @patch('app.agents.mta.inference_docker.inference_job.mlflow')
    def test_main_prediction_error(self, mock_mlflow, mock_upload, mock_predict, mock_load):
        """Test error handling when prediction fails"""
        # Setup mocks
        df_input = pd.DataFrame({"feature1": [1, 2, 3]})
        mock_load.return_value = df_input
        mock_predict.side_effect = ValueError("Prediction failed")
        
        # Run main - should exit with error code 1
        with pytest.raises(SystemExit) as exc_info:
            inference_job.main()
        assert exc_info.value.code == 1

