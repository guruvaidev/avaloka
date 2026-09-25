"""
Unit tests for Temporary Inference Manager

Tests cover:
- Inference request detection
- CSV extraction from state
- Path validation
- TemporaryInferenceManager methods
- End-to-end inference workflow
"""

import pytest
import os
import sys
import tempfile
import pandas as pd
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from io import StringIO

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.mta.temporary_inference import (
    detect_inference_request,
    detect_inference_mode,
    extract_csv_from_state,
    validate_inference_path,
    TemporaryInferenceManager,
    parse_direct_input
)
from app.graph.etl_state import ETLState
from langchain_core.messages import HumanMessage, AIMessage


class TestDetectInferenceRequest:
    """Test inference request detection"""
    
    def test_no_training_completed(self):
        """Test that detection fails if training not completed"""
        messages = [HumanMessage(content="run inference on this data")]
        state = ETLState({"training_completed": False})
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is False
        assert path is None
        assert inference_mode is None
    
    def test_no_mlflow_run_id(self):
        """Test that detection fails if no MLflow run ID"""
        messages = [HumanMessage(content="run inference on this data")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": None
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is False
        assert path is None
        assert inference_mode is None
    
    def test_no_inference_keywords(self):
        """Test that detection fails without inference keywords"""
        messages = [HumanMessage(content="hello world")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123"
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is False
        assert path is None
        assert inference_mode is None
    
    def test_detects_with_keyword(self):
        """Test detection with inference keywords"""
        messages = [HumanMessage(content="run inference on this data")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "uploaded_csv_preview": [["col1", "col2"], ["val1", "val2"]]
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is True
        assert path == "uploaded"
        assert inference_mode in ["local", "k8s"]  # Should detect a mode
    
    def test_detects_various_keywords(self):
        """Test detection with various inference keywords"""
        keywords = [
            "run inference",
            "run predictions",
            "predict",
            "test model",
            "use model",
            "apply model",
            "make predictions",
            "generate predictions"
        ]
        
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "uploaded_csv_preview": [["col1"], ["val1"]]
        })
        
        for keyword in keywords:
            messages = [HumanMessage(content=f"{keyword} on this data")]
            is_request, path, inference_mode = detect_inference_request(messages, state)
            assert is_request is True, f"Failed to detect keyword: {keyword}"
            assert inference_mode in ["local", "k8s"]  # Should detect a mode
    
    def test_detects_absolute_path(self):
        """Test detection with absolute file path"""
        messages = [HumanMessage(content='run inference on "/tmp/test.csv"')]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123"
        })
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\nval1,val2\n")
            temp_path = f.name
        
        try:
            # Replace /tmp/test.csv with actual temp path in message
            messages = [HumanMessage(content=f'run inference on "{temp_path}"')]
            is_request, path, inference_mode = detect_inference_request(messages, state)
            assert is_request is True
            assert path == temp_path
            assert inference_mode in ["local", "k8s"]  # Should detect a mode
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_detects_data_source_reference(self):
        """Test detection with data_source_location reference"""
        messages = [HumanMessage(content="run inference on the original data")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "data_source_location": "/path/to/data.csv"
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is True
        assert path == "data_source"
        assert inference_mode in ["local", "k8s"]  # Should detect a mode
    
    def test_detects_output_reference(self):
        """Test detection with output_location reference"""
        messages = [HumanMessage(content="run inference on the output")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "output_location": "/path/to/output.csv"
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is True
        assert path == "output"
        assert inference_mode in ["local", "k8s"]  # Should detect a mode
    
    def test_detects_execution_output_reference(self):
        """Test detection with execution_output_data reference"""
        messages = [HumanMessage(content="run inference on the result")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "execution_output_data": pd.DataFrame({"col1": [1, 2]})
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is True
        assert path == "execution_output"
        assert inference_mode in ["local", "k8s"]  # Should detect a mode
    
    def test_empty_messages(self):
        """Test detection with empty messages"""
        messages = []
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123"
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is False
        assert path is None
        assert inference_mode is None
    
    def test_detects_direct_input(self):
        """Test detection with direct input pattern"""
        messages = [HumanMessage(content="SepalLengthCm 5.1 SepalWidthCm 3.5 PetalLengthCm 1.4 PetalWidthCm 0.2")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123"
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is True
        assert path == "direct_input"
        assert inference_mode in ["local", "k8s"]  # Should detect a mode
    
    def test_detects_direct_input_with_keyword(self):
        """Test detection with direct input and inference keyword"""
        messages = [HumanMessage(content="predict SepalLengthCm 5.1 SepalWidthCm 3.5 PetalLengthCm 1.4 PetalWidthCm 0.2")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123"
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is True
        assert path == "direct_input"
        assert inference_mode in ["local", "k8s"]  # Should detect a mode
    
    def test_detects_dataset_mention_by_filename(self):
        """Test detection of dataset mention by filename"""
        messages = [HumanMessage(content="run inference on Iris_info.csv")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "datasets_context": [
                {
                    "dataset_id": "ds_123",
                    "filename": "Iris_info.csv",
                    "alias": "iris_info",
                    "data_source_location": "/tmp/iris.csv"
                }
            ]
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is True
        assert state.get("active_dataset_id") == "ds_123"
        assert inference_mode in ["local", "k8s"]
    
    def test_detects_dataset_mention_by_alias(self):
        """Test detection of dataset mention by alias"""
        messages = [HumanMessage(content="inference using iris_info")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "datasets_context": [
                {
                    "dataset_id": "ds_456",
                    "filename": "Iris_info.csv",
                    "alias": "iris_info",
                    "data_source_location": "/tmp/iris.csv"
                }
            ]
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is True
        assert state.get("active_dataset_id") == "ds_456"
    
    def test_detects_dataset_mention_without_extension(self):
        """Test detection of dataset mention without file extension"""
        messages = [HumanMessage(content="predict on Iris_info")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "datasets_context": [
                {
                    "dataset_id": "ds_789",
                    "filename": "Iris_info.csv",
                    "alias": "iris_info",
                    "data_source_location": "/tmp/iris.csv"
                }
            ]
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        assert is_request is True
        assert state.get("active_dataset_id") == "ds_789"


class TestDetectInferenceMode:
    """Test inference mode detection from user intent"""
    
    def test_detects_k8s_keywords(self):
        """Test detection of k8s mode from batch/scale keywords"""
        k8s_keywords = [
            "schedule batch inference",
            "run inference at scale",
            "deploy inference for production",
            "automated recurring inference",
            "periodic inference job",
            "large-scale inference"
        ]
        
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123"
        })
        
        for keyword in k8s_keywords:
            messages = [HumanMessage(content=keyword)]
            mode = detect_inference_mode(messages, state)
            assert mode == "k8s", f"Failed to detect k8s mode for: {keyword}"
    
    def test_detects_local_keywords(self):
        """Test detection of local mode from quick/test keywords"""
        local_keywords = [
            "test model quickly",
            "run inference now",
            "interactive inference",
            "direct inference",
            "instant prediction"
        ]
        
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123"
        })
        
        for keyword in local_keywords:
            messages = [HumanMessage(content=keyword)]
            mode = detect_inference_mode(messages, state)
            assert mode == "local", f"Failed to detect local mode for: {keyword}"
    
    def test_defaults_to_local(self):
        """Test that default mode is local when no keywords detected"""
        messages = [HumanMessage(content="run inference")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123"
        })
        
        mode = detect_inference_mode(messages, state)
        assert mode == "local"
    
    def test_k8s_keyword_overrides_local(self):
        """Test that k8s keywords take precedence over local keywords"""
        messages = [HumanMessage(content="schedule batch inference now")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123"
        })
        
        mode = detect_inference_mode(messages, state)
        assert mode == "k8s"
    
    def test_state_based_detection(self):
        """Test detection based on state indicators"""
        # Test with scheduled_task in state
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "scheduled_task": True
        })
        messages = [HumanMessage(content="run inference")]
        mode = detect_inference_mode(messages, state)
        assert mode == "k8s"
        
        # Test with batch_inference in state
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "batch_inference": True
        })
        mode = detect_inference_mode(messages, state)
        assert mode == "k8s"


class TestParseDirectInput:
    """Test direct input parsing"""
    
    def test_parse_valid_input(self):
        """Test parsing valid direct input string"""
        input_string = "SepalLengthCm 5.1 SepalWidthCm 3.5 PetalLengthCm 1.4 PetalWidthCm 0.2"
        df = parse_direct_input(input_string)
        
        assert df is not None
        assert len(df) == 1
        assert "SepalLengthCm" in df.columns
        assert "SepalWidthCm" in df.columns
        assert "PetalLengthCm" in df.columns
        assert "PetalWidthCm" in df.columns
        assert df.iloc[0]["SepalLengthCm"] == 5.1
        assert df.iloc[0]["SepalWidthCm"] == 3.5
    
    def test_parse_invalid_input_odd_tokens(self):
        """Test parsing fails with odd number of tokens"""
        input_string = "SepalLengthCm 5.1 SepalWidthCm"
        df = parse_direct_input(input_string)
        assert df is None
    
    def test_parse_invalid_input_non_numeric(self):
        """Test parsing fails with non-numeric values"""
        input_string = "SepalLengthCm abc SepalWidthCm 3.5"
        df = parse_direct_input(input_string)
        assert df is None
    
    def test_parse_empty_input(self):
        """Test parsing empty input"""
        df = parse_direct_input("")
        assert df is None


class TestExtractCSVFromState:
    """Test CSV extraction from state"""
    
    def test_extract_from_direct_input(self):
        """Test extraction from direct input string"""
        input_string = "SepalLengthCm 5.1 SepalWidthCm 3.5 PetalLengthCm 1.4 PetalWidthCm 0.2"
        state = ETLState({})
        df = extract_csv_from_state(state, "direct_input", direct_input_string=input_string)
        
        assert df is not None
        assert len(df) == 1
        assert "SepalLengthCm" in df.columns
    
    def test_extract_from_explicit_path(self):
        """Test extraction from explicit file path"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\nval1,val2\nval3,val4\n")
            temp_path = f.name
        
        try:
            state = ETLState({})
            df = extract_csv_from_state(state, temp_path)
            assert df is not None
            assert len(df) == 2
            assert list(df.columns) == ["col1", "col2"]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_extract_from_data_source_location(self):
        """Test extraction from data_source_location"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\nval1,val2\n")
            temp_path = f.name
        
        try:
            state = ETLState({
                "data_source_location": temp_path
            })
            df = extract_csv_from_state(state, "data_source")
            assert df is not None
            assert len(df) == 1
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_extract_from_output_location(self):
        """Test extraction from output_location"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\nval1,val2\n")
            temp_path = f.name
        
        try:
            state = ETLState({
                "output_location": temp_path
            })
            df = extract_csv_from_state(state, "output")
            assert df is not None
            assert len(df) == 1
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_extract_from_execution_output_dataframe(self):
        """Test extraction from execution_output_data (DataFrame)"""
        df_original = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        state = ETLState({
            "execution_output_data": df_original
        })
        
        df = extract_csv_from_state(state, "execution_output")
        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ["col1", "col2"]
    
    def test_extract_from_execution_output_dict(self):
        """Test extraction from execution_output_data (dict)"""
        state = ETLState({
            "execution_output_data": {"col1": [1, 2], "col2": [3, 4]}
        })
        
        df = extract_csv_from_state(state, "execution_output")
        assert df is not None
        assert len(df) == 2
    
    def test_extract_from_uploaded_csv_preview(self):
        """Test extraction from uploaded_csv_preview"""
        state = ETLState({
            "uploaded_csv_preview": [
                ["col1", "col2"],
                ["val1", "val2"],
                ["val3", "val4"]
            ],
            "uploaded_csv_columns": ["col1", "col2"]
        })
        
        df = extract_csv_from_state(state, "uploaded")
        assert df is not None
        assert len(df) == 2  # Excluding header row
        assert list(df.columns) == ["col1", "col2"]
    
    def test_extract_from_uploaded_csv_preview_with_duplicate_headers(self):
        """Test extraction from uploaded_csv_preview with duplicate header rows"""
        # Simulate the bug where headers are duplicated as data rows
        state = ETLState({
            "uploaded_csv_preview": [
                ["SepalLeng", "SepalWid", "PetalLeng", "PetalWidt", "Species"],
                ["SepalLeng", "SepalWid", "PetalLeng", "PetalWidt", "Species"],  # Duplicate header
                ["SepalLeng", "SepalWid", "PetalLeng", "PetalWidt", "Species"],  # Duplicate header
                ["5.1", "3.5", "1.4", "0.2", "setosa"],  # Actual data
                ["4.9", "3.0", "1.4", "0.2", "setosa"],  # Actual data
            ],
            "uploaded_csv_columns": ["SepalLeng", "SepalWid", "PetalLeng", "PetalWidt", "Species"]
        })
        
        df = extract_csv_from_state(state, "uploaded")
        assert df is not None
        assert len(df) == 2  # Should only have 2 data rows, headers filtered out
        assert list(df.columns) == ["SepalLeng", "SepalWid", "PetalLeng", "PetalWidt", "Species"]
        assert df.iloc[0]["SepalLeng"] == "5.1"  # First actual data row
        assert df.iloc[1]["SepalLeng"] == "4.9"  # Second actual data row
    
    def test_extract_from_sample_data(self):
        """Test extraction from sample_data"""
        csv_content = "col1,col2\nval1,val2\nval3,val4\n"
        state = ETLState({
            "sample_data": csv_content
        })
        
        df = extract_csv_from_state(state, "uploaded")
        assert df is not None
        assert len(df) == 2
        assert list(df.columns) == ["col1", "col2"]
    
    def test_extract_none_when_no_data(self):
        """Test extraction returns None when no data available"""
        state = ETLState({})
        df = extract_csv_from_state(state, None)
        assert df is None
    
    def test_extract_none_when_path_not_exists(self):
        """Test extraction returns None when path doesn't exist"""
        state = ETLState({})
        df = extract_csv_from_state(state, "/nonexistent/path.csv")
        assert df is None
    
    def test_extract_from_selected_dataset_by_data_source_location(self):
        """Test extraction from selected dataset using data_source_location"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\nval1,val2\nval3,val4\n")
            temp_path = f.name
        
        try:
            state = ETLState({
                "active_dataset_id": "ds_123",
                "datasets_context": [
                    {
                        "dataset_id": "ds_123",
                        "filename": "test.csv",
                        "alias": "test",
                        "data_source_location": temp_path
                    }
                ]
            })
            
            df = extract_csv_from_state(state, None)
            assert df is not None
            assert len(df) == 2
            assert list(df.columns) == ["col1", "col2"]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_extract_from_selected_dataset_by_preview(self):
        """Test extraction from selected dataset using preview data"""
        state = ETLState({
            "active_dataset_id": "ds_456",
            "datasets_context": [
                {
                    "dataset_id": "ds_456",
                    "filename": "test2.csv",
                    "alias": "test2",
                    "preview": [
                        ["col1", "col2"],
                        ["val1", "val2"],
                        ["val3", "val4"]
                    ],
                    "columns": ["col1", "col2"]
                }
            ]
        })
        
        df = extract_csv_from_state(state, None)
        assert df is not None
        assert len(df) == 2
        assert list(df.columns) == ["col1", "col2"]
    
    def test_extract_falls_back_when_selected_dataset_not_found(self):
        """Test extraction falls back to uploaded_csv_preview when selected dataset not found"""
        state = ETLState({
            "active_dataset_id": "nonexistent",
            "datasets_context": [
                {
                    "dataset_id": "ds_123",
                    "filename": "other.csv"
                }
            ],
            "uploaded_csv_preview": [
                ["col1", "col2"],
                ["val1", "val2"]
            ],
            "uploaded_csv_columns": ["col1", "col2"]
        })
        
        df = extract_csv_from_state(state, None)
        # Should fall back to uploaded_csv_preview
        assert df is not None
        assert len(df) == 1
        assert list(df.columns) == ["col1", "col2"]


class TestValidateInferencePath:
    """Test path validation"""
    
    def test_validate_existing_csv_file(self):
        """Test validation of existing CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\nval1,val2\n")
            temp_path = f.name
        
        try:
            is_valid, error = validate_inference_path(temp_path)
            assert is_valid is True
            assert error is None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_validate_nonexistent_file(self):
        """Test validation fails for nonexistent file"""
        is_valid, error = validate_inference_path("/nonexistent/file.csv")
        assert is_valid is False
        assert "not found" in error.lower()
    
    def test_validate_directory(self):
        """Test validation fails for directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            is_valid, error = validate_inference_path(temp_dir)
            assert is_valid is False
            assert "not a file" in error.lower()
    
    def test_validate_wrong_extension(self):
        """Test validation fails for non-CSV/TSV files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some text")
            temp_path = f.name
        
        try:
            is_valid, error = validate_inference_path(temp_path)
            assert is_valid is False
            assert "csv or tsv" in error.lower()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_validate_with_allowed_directories(self):
        """Test validation with allowed directories restriction"""
        with tempfile.TemporaryDirectory() as allowed_dir:
            csv_path = os.path.join(allowed_dir, "test.csv")
            with open(csv_path, 'w') as f:
                f.write("col1\nval1\n")
            
            # Should pass when in allowed directory
            is_valid, error = validate_inference_path(csv_path, [allowed_dir])
            assert is_valid is True
            
            # Should fail when outside allowed directory
            with tempfile.TemporaryDirectory() as other_dir:
                other_csv = os.path.join(other_dir, "test.csv")
                with open(other_csv, 'w') as f:
                    f.write("col1\nval1\n")
                
                is_valid, error = validate_inference_path(other_csv, [allowed_dir])
                assert is_valid is False
                assert "outside allowed directories" in error.lower()


class TestTemporaryInferenceManager:
    """Test TemporaryInferenceManager class"""
    
    @pytest.fixture
    def mock_blob_store(self):
        """Create a mock blob store"""
        blob_store = MagicMock()
        blob_store.put_file.return_value = "gs://bucket/tmp-inference/inputs/run_123/test_input.csv"
        blob_store.get_file.return_value = None  # Just downloads, doesn't return
        blob_store._prefix = ''  # Set as empty string, not MagicMock
        return blob_store
    
    @pytest.fixture
    def mock_k8s_manager(self):
        """Create a mock Kubernetes Job Manager"""
        k8s_manager = MagicMock()
        k8s_manager.create_inference_job.return_value = {
            "created": True,
            "job_name": "temp-inference-run-123-20240101",
            "namespace": "temp-inference"
        }
        k8s_manager.wait_for_completion.return_value = {
            "status": "Succeeded",
            "completion_time": "2024-01-01T12:00:00Z"
        }
        k8s_manager.get_job_logs.return_value = "Job completed successfully"
        return k8s_manager
    
    @pytest.fixture
    def mock_mlflow_manager(self):
        """Create a mock MLflow Manager"""
        mlflow_manager = MagicMock()
        mlflow_manager.tracking_uri = "postgresql://mlflow_user:pass@host:5432/mlflow_db"
        return mlflow_manager
    
    @pytest.fixture
    def manager(self, mock_blob_store, mock_k8s_manager, mock_mlflow_manager):
        """Create TemporaryInferenceManager instance with mocks"""
        with patch('app.agents.mta.temporary_inference.KubernetesJobManager', return_value=mock_k8s_manager):
            manager = TemporaryInferenceManager(
                mlflow_manager=mock_mlflow_manager,
                blob_store=mock_blob_store,
                k8s_namespace="temp-inference",
                gcs_bucket="test-bucket",
                gcs_prefix="tmp-inference"
            )
            manager.k8s_manager = mock_k8s_manager  # Override with our mock
            return manager
    
    def test_manager_initialization(self, manager):
        """Test manager initialization"""
        assert manager.k8s_namespace == "temp-inference"
        assert manager.gcs_bucket == "test-bucket"
        assert manager.gcs_prefix == "tmp-inference"
        assert manager.k8s_manager is not None
    
    def test_stage_input_csv(self, manager, mock_blob_store):
        """Test staging input CSV to GCS"""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        run_id = "run_123"
        
        gcs_path = manager.stage_input_csv(df, run_id)
        
        assert gcs_path.startswith("gs://")
        assert run_id in gcs_path
        mock_blob_store.put_file.assert_called_once()
    
    def test_stage_input_csv_filters_header_rows(self, manager, mock_blob_store):
        """Test that stage_input_csv filters out rows that match headers"""
        # Create DataFrame where some rows match column names (simulating duplicate headers)
        df = pd.DataFrame({
            "col1": ["col1", "col1", "val1", "val2"],  # First two rows match header
            "col2": ["col2", "col2", "val3", "val4"]
        })
        run_id = "run_123"
        
        # Capture the file path and content before it's deleted
        uploaded_file_paths = []
        uploaded_file_contents = []
        
        def capture_put_file(file_path, *args, **kwargs):
            file_path_str = str(file_path)
            uploaded_file_paths.append(file_path_str)
            # Read and store the file content before it might be deleted
            if os.path.exists(file_path_str):
                with open(file_path_str, 'r') as f:
                    uploaded_file_contents.append(f.read())
            # Return the expected GCS path
            return f"gs://bucket/tmp-inference/inputs/{run_id}/test_input.csv"
        
        mock_blob_store.put_file.side_effect = capture_put_file
        
        gcs_path = manager.stage_input_csv(df, run_id)
        
        # Should succeed and filter out header rows
        assert gcs_path.startswith("gs://")
        assert run_id in gcs_path
        mock_blob_store.put_file.assert_called_once()
        
        # Verify the uploaded file doesn't contain header rows
        # We captured the file content before it was deleted
        assert len(uploaded_file_paths) > 0, "put_file should have been called with a file path"
        assert len(uploaded_file_contents) > 0, "File content should have been captured"
        
        # Parse the captured CSV content to verify it was correctly filtered
        from io import StringIO
        uploaded_df = pd.read_csv(StringIO(uploaded_file_contents[0]))
        assert len(uploaded_df) == 2, "Should only have 2 data rows after filtering header rows"
        assert uploaded_df.iloc[0]["col1"] == "val1"
        assert uploaded_df.iloc[1]["col1"] == "val2"
    
    def test_stage_input_csv_raises_on_only_headers(self, manager):
        """Test that stage_input_csv raises error if DataFrame contains only header rows"""
        # Create DataFrame where all rows match column names
        df = pd.DataFrame({
            "col1": ["col1", "col1", "col1"],
            "col2": ["col2", "col2", "col2"]
        })
        run_id = "run_123"
        
        with pytest.raises(ValueError, match="DataFrame contains only header rows"):
            manager.stage_input_csv(df, run_id)
    
    def test_stage_input_csv_no_blob_store(self):
        """Test staging fails without blob_store"""
        manager = TemporaryInferenceManager(blob_store=None)
        df = pd.DataFrame({"col1": [1, 2]})
        
        with pytest.raises(ValueError, match="blob_store not configured"):
            manager.stage_input_csv(df, "run_123")
    
    def test_retrieve_output_csv(self, manager, mock_blob_store):
        """Test retrieving output CSV from GCS"""
        # Create a temporary CSV file to simulate downloaded file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,Predicted,Confidence\n1,ClassA,0.95\n2,ClassB,0.87\n")
            temp_path = f.name
        
        try:
            # Mock get_file to write to temp_path
            def mock_get_file(object_name, local_path):
                import shutil
                shutil.copy(temp_path, str(local_path))
            
            mock_blob_store.get_file.side_effect = mock_get_file
            
            gcs_path = "gs://bucket/tmp-inference/outputs/run_123/output.csv"
            df = manager.retrieve_output_csv(gcs_path)
            
            assert df is not None
            assert len(df) == 2
            assert "Predicted" in df.columns
            assert "Confidence" in df.columns
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_retrieve_output_csv_no_blob_store(self):
        """Test retrieval fails without blob_store"""
        manager = TemporaryInferenceManager(blob_store=None)
        
        with pytest.raises(ValueError, match="blob_store not configured"):
            manager.retrieve_output_csv("gs://bucket/path.csv")
    
    def test_create_inference_job(self, manager, mock_k8s_manager):
        """Test creating inference job"""
        run_id = "run_123"
        model_uri = "runs:/run_123/model"
        input_gcs_path = "gs://bucket/tmp-inference/inputs/run_123/input.csv"
        mlflow_tracking_uri = "postgresql://mlflow_user:pass@host:5432/mlflow_db"
        
        result = manager.create_inference_job(
            run_id=run_id,
            model_uri=model_uri,
            input_gcs_path=input_gcs_path,
            mlflow_tracking_uri=mlflow_tracking_uri
        )
        
        assert result["created"] is True
        assert "job_name" in result
        assert "output_gcs_path" in result
        mock_k8s_manager.create_inference_job.assert_called_once()
    
    def test_create_inference_job_no_k8s_manager(self):
        """Test creating job fails without k8s_manager"""
        manager = TemporaryInferenceManager()
        manager.k8s_manager = None
        
        with pytest.raises(ValueError, match="Kubernetes Job Manager not initialized"):
            manager.create_inference_job(
                run_id="run_123",
                model_uri="runs:/run_123/model",
                input_gcs_path="gs://bucket/input.csv",
                mlflow_tracking_uri="postgresql://..."
            )
    
    def test_format_predictions_message(self, manager):
        """Test formatting predictions message"""
        df = pd.DataFrame({
            "Predicted": ["ClassA", "ClassB", "ClassA"],
            "Confidence": [0.95, 0.87, 0.92]
        })
        
        message = manager.format_predictions_message(df)
        
        assert "Inference Results" in message
        assert "Total samples: 3" in message
        assert "ClassA" in message
        assert "ClassB" in message
        assert "Average confidence" in message
    
    def test_format_predictions_message_with_job_status(self, manager):
        """Test formatting with job status"""
        df = pd.DataFrame({
            "Predicted": ["ClassA"],
            "Confidence": [0.95]
        })
        job_status = {
            "status": "Succeeded",
            "completion_time": "2024-01-01T12:00:00Z"
        }
        
        message = manager.format_predictions_message(df, job_status)
        
        assert "Job Status" in message
        assert "Succeeded" in message
        assert "Completed at" in message


class TestHandleInferenceRequest:
    """Test end-to-end inference request handling"""
    
    @pytest.fixture
    def mock_blob_store(self):
        """Create a mock blob store"""
        blob_store = MagicMock()
        blob_store.put_file.return_value = "gs://bucket/tmp-inference/inputs/run_123/test_input.csv"
        blob_store._prefix = ''  # Set as empty string, not MagicMock
        
        # Mock get_file to create a temporary CSV
        def mock_get_file(object_name, local_path):
            with open(local_path, 'w') as f:
                f.write("col1,Predicted,Confidence\n1,ClassA,0.95\n2,ClassB,0.87\n")
        
        blob_store.get_file.side_effect = mock_get_file
        return blob_store
    
    @pytest.fixture
    def mock_k8s_manager(self):
        """Create a mock Kubernetes Job Manager"""
        k8s_manager = MagicMock()
        k8s_manager.create_inference_job.return_value = {
            "created": True,
            "job_name": "temp-inference-run-123-20240101",
            "namespace": "temp-inference"
        }
        k8s_manager.wait_for_completion.return_value = {
            "status": "Succeeded",
            "completion_time": "2024-01-01T12:00:00Z"
        }
        k8s_manager.get_job_logs.return_value = "Job completed successfully"
        return k8s_manager
    
    @pytest.fixture
    def manager(self, mock_blob_store, mock_k8s_manager):
        """Create TemporaryInferenceManager with mocks"""
        with patch('app.agents.mta.temporary_inference.KubernetesJobManager', return_value=mock_k8s_manager):
            manager = TemporaryInferenceManager(
                blob_store=mock_blob_store,
                k8s_namespace="temp-inference",
                gcs_bucket="test-bucket"
            )
            manager.k8s_manager = mock_k8s_manager
            return manager
    
    @patch.dict(os.environ, {"MLFLOW_BACKEND_STORE_URI": "postgresql://mlflow_user:pass@host:5432/mlflow_db"})
    @patch('app.agents.mta.temporary_inference.LOCAL_INFERENCE_AVAILABLE', False)
    def test_handle_inference_request_success(self, manager):
        """Test successful inference request handling"""
        messages = [HumanMessage(content="schedule batch inference on this data")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "uploaded_csv_preview": [
                ["col1", "col2"],
                ["val1", "val2"],
                ["val3", "val4"]
            ],
            "uploaded_csv_columns": ["col1", "col2"]
        })
        
        result = manager.handle_inference_request(state, messages)
        
        assert result["success"] is True
        assert "message" in result
        assert "predictions_df" in result
        assert "Inference Results" in result["message"]
    
    def test_handle_inference_request_no_detection(self, manager):
        """Test handling when no inference request detected"""
        messages = [HumanMessage(content="hello world")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123"
        })
        
        result = manager.handle_inference_request(state, messages)
        
        assert result["success"] is False
        assert "No inference request detected" in result["error"]
    
    def test_handle_inference_request_no_csv_data(self, manager):
        """Test handling when no CSV data available"""
        # Use a path that doesn't exist to trigger CSV extraction failure
        messages = [HumanMessage(content='run inference on "/nonexistent/file.csv"')]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123"
        })
        
        result = manager.handle_inference_request(state, messages)
        
        assert result["success"] is False
        # Should detect request (path mentioned) but fail on CSV extraction
        error_msg = result.get("error", "") or ""
        message = result.get("message") or ""
        # Either detection fails (no CSV) or extraction fails (file not found)
        assert ("No CSV data found" in error_msg or "No CSV data found" in message or 
                "File not found" in error_msg or "File not found" in message or
                "No inference request detected" in error_msg)
    
    def test_handle_inference_request_no_mlflow_run(self, manager):
        """Test handling when no MLflow run ID"""
        messages = [HumanMessage(content="run inference on uploaded data")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": None,  # No run ID
            "uploaded_csv_preview": [["col1"], ["val1"]],
            "uploaded_csv_columns": ["col1"]
        })
        
        result = manager.handle_inference_request(state, messages)
        
        assert result["success"] is False
        # Detection should fail because no mlflow_run_id
        assert "No inference request detected" in result["error"] or "No trained model found" in result.get("error", "")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_handle_inference_request_no_mlflow_uri(self, manager):
        """Test handling when MLflow tracking URI not configured"""
        messages = [HumanMessage(content="run inference")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "uploaded_csv_preview": [["col1"], ["val1"]],
            "uploaded_csv_columns": ["col1"]
        })
        
        result = manager.handle_inference_request(state, messages)
        
        assert result["success"] is False
        assert "MLflow tracking URI not configured" in result["error"]
    
    def test_handle_inference_request_job_failure(self, manager, mock_k8s_manager):
        """Test handling when Kubernetes job fails"""
        # Disable local inference to force k8s mode
        manager.local_inference = None
        mock_k8s_manager.create_inference_job.return_value = {
            "created": False,
            "error": "Failed to create job"
        }
        
        messages = [HumanMessage(content="schedule batch inference on uploaded data")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "uploaded_csv_preview": [["col1"], ["val1"]],
            "uploaded_csv_columns": ["col1"]
        })
        
        with patch.dict(os.environ, {"MLFLOW_BACKEND_STORE_URI": "postgresql://..."}):
            result = manager.handle_inference_request(state, messages)
        
        assert result["success"] is False
        assert "Failed to create" in result["error"] or "create inference job" in result["error"]
    
    @patch('app.agents.mta.temporary_inference.LOCAL_INFERENCE_AVAILABLE', False)
    def test_handle_inference_request_job_timeout(self, manager, mock_k8s_manager):
        """Test handling when job times out"""
        mock_k8s_manager.wait_for_completion.return_value = {
            "status": "Timeout"
        }
        
        messages = [HumanMessage(content="schedule batch inference")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "uploaded_csv_preview": [["col1"], ["val1"]],
            "uploaded_csv_columns": ["col1"]
        })
        
        with patch.dict(os.environ, {"MLFLOW_BACKEND_STORE_URI": "postgresql://..."}):
            result = manager.handle_inference_request(state, messages)
        
        assert result["success"] is False
        assert "timed out" in result["error"].lower()
    
    @patch('app.agents.mta.temporary_inference.LOCAL_INFERENCE_AVAILABLE', False)
    def test_handle_inference_request_job_failed(self, manager, mock_k8s_manager):
        """Test handling when job fails"""
        mock_k8s_manager.wait_for_completion.return_value = {
            "status": "Failed"
        }
        mock_k8s_manager.get_job_logs.return_value = "Error: Model not found"
        
        messages = [HumanMessage(content="schedule batch inference")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "uploaded_csv_preview": [["col1"], ["val1"]],
            "uploaded_csv_columns": ["col1"]
        })
        
        with patch.dict(os.environ, {"MLFLOW_BACKEND_STORE_URI": "postgresql://..."}):
            result = manager.handle_inference_request(state, messages)
        
        assert result["success"] is False
        assert "failed" in result["error"].lower()
        assert "logs" in result
    
    def test_handle_inference_request_invalid_dataset_id(self, manager):
        """Test error handling when selected dataset ID doesn't exist"""
        messages = [HumanMessage(content="run inference on nonexistent_dataset.csv")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "active_dataset_id": "nonexistent_id",
            "datasets_context": [
                {
                    "dataset_id": "ds_123",
                    "filename": "valid.csv",
                    "alias": "valid"
                }
            ]
        })
        
        result = manager.handle_inference_request(state, messages)
        
        assert result["success"] is False
        assert "not found" in result["error"].lower() or "not found" in result.get("message", "").lower()
        assert "Available datasets" in result.get("message", "")
        assert "ds_123" in result.get("message", "")
    
    def test_handle_inference_request_dataset_data_unavailable(self, manager):
        """Test error handling when selected dataset exists but data is unavailable"""
        messages = [HumanMessage(content="run inference on test.csv")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "active_dataset_id": "ds_456",
            "datasets_context": [
                {
                    "dataset_id": "ds_456",
                    "filename": "test.csv",
                    "alias": "test",
                    # Missing data_source_location, preview, and columns
                },
                {
                    "dataset_id": "ds_789",
                    "filename": "other.csv",
                    "alias": "other",
                    "data_source_location": "/tmp/other.csv"
                }
            ]
        })
        
        result = manager.handle_inference_request(state, messages)
        
        assert result["success"] is False
        assert "could not extract" in result["error"].lower() or "could not extract" in result.get("message", "").lower()
        # Should mention other available datasets
        assert "other available datasets" in result.get("message", "").lower() or "Other available datasets" in result.get("message", "")
    
    def test_handle_inference_request_after_training_completed(self, manager):
        """Test inference request after training is completed (real-world scenario)"""
        # Simulate the exact scenario: training completed, user says "Run inference immediately"
        messages = [HumanMessage(content="Run inference immediately")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "active_dataset_id": "ds_iris",
            "datasets_context": [
                {
                    "dataset_id": "ds_iris",
                    "filename": "Iris_info.csv",
                    "alias": "iris_info",
                    "preview": [
                        ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"],
                        ["5.1", "3.5", "1.4", "0.2", "setosa"],
                        ["4.9", "3.0", "1.4", "0.2", "setosa"],
                        ["4.7", "3.2", "1.3", "0.2", "setosa"]
                    ],
                    "columns": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]
                }
            ]
        })
        
        # manager.local_inference is a real LocalInferenceManager -- the fixture
        # replaces the Kubernetes job manager, not this one. Assigning
        # `.predict.return_value` therefore set an attribute on a bound method
        # and mocked nothing, then raised AttributeError for exactly that
        # reason. Replace the method itself.
        if manager.local_inference:
            manager.local_inference.predict = MagicMock(return_value=pd.DataFrame({
                "Predicted": ["setosa", "setosa", "setosa"],
                "Confidence": [0.95, 0.92, 0.88]
            }))
        
        with patch.dict(os.environ, {"MLFLOW_BACKEND_STORE_URI": "postgresql://..."}):
            result = manager.handle_inference_request(state, messages)
        
        # Should succeed with inference, NOT attempt training
        # Even though dataset has only 3 samples (too small for training), it's fine for inference
        assert result["success"] is True or "inference" in result.get("message", "").lower()
        # Should NOT contain training-related errors
        assert "Dataset too small for ML training" not in result.get("message", "")
        assert "training" not in result.get("error", "").lower()
    
    def test_detect_inference_request_run_inference_immediately(self):
        """Test that 'Run inference immediately' is detected as inference request"""
        messages = [HumanMessage(content="Run inference immediately")]
        state = ETLState({
            "training_completed": True,
            "mlflow_run_id": "run_123",
            "uploaded_csv_preview": [["col1"], ["val1"]],
            "uploaded_csv_columns": ["col1"]
        })
        
        is_request, path, inference_mode = detect_inference_request(messages, state)
        
        assert is_request is True, "Should detect 'Run inference immediately' as inference request"
        assert inference_mode == "local", "Should detect 'immediately' as local mode"
        assert path in ["uploaded", None], "Should detect uploaded data or auto-detect"


class TestInferenceMode:
    """Test inference manager initialization (both managers initialized)"""
    
    @patch('app.agents.mta.temporary_inference.LOCAL_INFERENCE_AVAILABLE', True)
    @patch('app.agents.mta.temporary_inference.KubernetesJobManager')
    @patch('app.agents.mta.temporary_inference.LocalInferenceManager')
    def test_init_both_managers_available(self, mock_local_class, mock_k8s_class):
        """Test that both managers are initialized when available"""
        mock_mlflow = Mock()
        mock_local_instance = Mock()
        mock_k8s_instance = Mock()
        mock_local_class.return_value = mock_local_instance
        mock_k8s_class.return_value = mock_k8s_instance
        
        manager = TemporaryInferenceManager(mlflow_manager=mock_mlflow)
        
        # Both managers should be initialized
        assert manager.local_inference is not None
        assert manager.k8s_manager is not None
        mock_local_class.assert_called_once_with(mlflow_manager=mock_mlflow)
        mock_k8s_class.assert_called_once()
    
    @patch('app.agents.mta.temporary_inference.LOCAL_INFERENCE_AVAILABLE', False)
    @patch('app.agents.mta.temporary_inference.KubernetesJobManager')
    @patch('app.agents.mta.temporary_inference.LocalInferenceManager')
    def test_init_local_unavailable(self, mock_local_class, mock_k8s_class):
        """Test initialization when local inference is not available"""
        mock_mlflow = Mock()
        mock_k8s_instance = Mock()
        mock_k8s_class.return_value = mock_k8s_instance
        
        manager = TemporaryInferenceManager(mlflow_manager=mock_mlflow)
        
        # Only k8s manager should be initialized
        assert manager.local_inference is None
        assert manager.k8s_manager is not None
        mock_local_class.assert_not_called()
        mock_k8s_class.assert_called_once()
    
    @patch('app.agents.mta.temporary_inference.LOCAL_INFERENCE_AVAILABLE', True)
    @patch('app.agents.mta.temporary_inference.KubernetesJobManager')
    @patch('app.agents.mta.temporary_inference.LocalInferenceManager')
    def test_init_local_fails_gracefully(self, mock_local_class, mock_k8s_class):
        """Test that local manager initialization failure doesn't prevent k8s initialization"""
        mock_mlflow = Mock()
        mock_k8s_instance = Mock()
        mock_k8s_class.return_value = mock_k8s_instance
        
        # Make LocalInferenceManager raise an exception
        mock_local_class.side_effect = Exception("Local inference init failed")
        
        manager = TemporaryInferenceManager(mlflow_manager=mock_mlflow)
        
        # Local should be None, but k8s should still be initialized
        assert manager.local_inference is None
        assert manager.k8s_manager is not None
        mock_k8s_class.assert_called_once()
    
    @patch('app.agents.mta.temporary_inference.LOCAL_INFERENCE_AVAILABLE', True)
    @patch('app.agents.mta.temporary_inference.KubernetesJobManager')
    @patch('app.agents.mta.temporary_inference.LocalInferenceManager')
    def test_init_k8s_fails_gracefully(self, mock_local_class, mock_k8s_class):
        """Test that k8s manager initialization failure doesn't prevent local initialization"""
        mock_mlflow = Mock()
        mock_local_instance = Mock()
        mock_local_class.return_value = mock_local_instance
        
        # Make KubernetesJobManager raise an exception
        mock_k8s_class.side_effect = Exception("K8s inference init failed")
        
        manager = TemporaryInferenceManager(mlflow_manager=mock_mlflow)
        
        # K8s should be None, but local should still be initialized
        assert manager.local_inference is not None
        assert manager.k8s_manager is None
        mock_local_class.assert_called_once_with(mlflow_manager=mock_mlflow)
    
    @patch('app.agents.mta.temporary_inference.LOCAL_INFERENCE_AVAILABLE', True)
    @patch('app.agents.mta.temporary_inference.KubernetesJobManager')
    @patch('app.agents.mta.temporary_inference.LocalInferenceManager')
    def test_init_independent_of_env_var(self, mock_local_class, mock_k8s_class):
        """Test that initialization doesn't depend on INFERENCE_MODE env var"""
        mock_mlflow = Mock()
        mock_local_instance = Mock()
        mock_k8s_instance = Mock()
        mock_local_class.return_value = mock_local_instance
        mock_k8s_class.return_value = mock_k8s_instance
        
        # Test with INFERENCE_MODE set (should be ignored)
        with patch.dict(os.environ, {"INFERENCE_MODE": "k8s"}, clear=False):
            manager = TemporaryInferenceManager(mlflow_manager=mock_mlflow)
        
        # Both managers should still be initialized regardless of env var
        assert manager.local_inference is not None
        assert manager.k8s_manager is not None
        mock_local_class.assert_called_once()
        mock_k8s_class.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

