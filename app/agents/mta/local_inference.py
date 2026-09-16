"""
Local Inference Manager for Model Training Agent

Provides on-demand local inference without requiring Kubernetes.
Loads models from MLflow, caches them in memory, and runs inference locally.
"""

import logging
import os
import shutil
import threading
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
import tempfile
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
import onnxruntime as ort

from app.agents.mta.mlflow_integration import MLflowManager

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Thread-safe LRU cache for ONNX models and metadata.
    
    Caches models by run_id to avoid reloading from MLflow on every request.
    """
    
    def __init__(self, max_size: int = 10):
        """
        Initialize model cache.
        
        Args:
            max_size: Maximum number of models to cache (default: 10)
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        logger.info(f"ModelCache initialized with max_size={max_size}")
    
    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached model and metadata for a run_id.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary with 'model' (ONNX session) and 'metadata' (dict), or None if not cached
        """
        with self._lock:
            if run_id in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(run_id)
                logger.debug(f"Cache hit for run_id: {run_id}")
                return self._cache[run_id].copy()
            logger.debug(f"Cache miss for run_id: {run_id}")
            return None
    
    def put(self, run_id: str, model: ort.InferenceSession, metadata: Dict[str, Any]):
        """
        Cache a model and its metadata.
        
        Args:
            run_id: MLflow run ID
            model: ONNX Runtime InferenceSession
            metadata: Model metadata dictionary
        """
        with self._lock:
            # Remove oldest entry if cache is full
            if len(self._cache) >= self.max_size and run_id not in self._cache:
                oldest_run_id = next(iter(self._cache))
                del self._cache[oldest_run_id]
                logger.info(f"Cache full, evicted run_id: {oldest_run_id}")
            
            self._cache[run_id] = {
                'model': model,
                'metadata': metadata
            }
            # Move to end (most recently used)
            self._cache.move_to_end(run_id)
            logger.info(f"Cached model for run_id: {run_id} (cache size: {len(self._cache)})")
    
    def clear(self):
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            logger.info("Model cache cleared")
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


class LocalInferenceManager:
    """
    Manages local inference for trained models.
    
    Loads models from MLflow on-demand, caches them, and runs inference locally
    without requiring Kubernetes infrastructure.
    """
    
    def __init__(self, mlflow_manager: Optional[MLflowManager] = None, cache_size: int = 10):
        """
        Initialize Local Inference Manager.
        
        Args:
            mlflow_manager: MLflowManager instance (optional, will create if None)
            cache_size: Maximum number of models to cache (default: 10)
        """
        self.mlflow_manager = mlflow_manager
        self.cache = ModelCache(max_size=cache_size)
        logger.info("LocalInferenceManager initialized")
    
    def load_model(self, run_id: str, model_uri: str) -> Tuple[ort.InferenceSession, Dict[str, Any]]:
        """
        Load ONNX model and metadata from MLflow.
        
        Args:
            run_id: MLflow run ID
            model_uri: MLflow model URI (e.g., "runs:/run_id/model")
            
        Returns:
            Tuple of (ONNX InferenceSession, metadata dictionary)
        """
        # Check cache first
        cached = self.cache.get(run_id)
        if cached:
            logger.info(f"Using cached model for run_id: {run_id}")
            return cached['model'], cached['metadata']
        
        logger.info(f"Loading model from MLflow: run_id={run_id}, model_uri={model_uri}")
        
        # Load metadata first
        metadata = self.load_metadata(run_id, model_uri)
        
        # Load ONNX model
        model = self._load_onnx_model(run_id, model_uri)
        
        # Cache the model
        self.cache.put(run_id, model, metadata)
        
        return model, metadata
    
    def load_metadata(self, run_id: str, model_uri: str, metadata_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load metadata (normalization stats, class names, feature names) from MLflow.
        
        Args:
            run_id: MLflow run ID
            model_uri: MLflow model URI
            metadata_path: Optional path to metadata JSON file (local or GCS)
            
        Returns:
            Dictionary with x_mean, x_std, class_names, feature_names
        """
        metadata = None
        
        # Priority 1: Try to load from provided metadata_path
        if metadata_path:
            try:
                if metadata_path.startswith("gs://"):
                    # Download from GCS
                    metadata = self._load_metadata_from_gcs(metadata_path)
                else:
                    # Local file path
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    logger.info(f"Loaded metadata from local path: {metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        
        # Priority 2: Download metadata from MLflow artifacts
        if not metadata:
            metadata = self._load_metadata_from_mlflow(run_id, model_uri)
        
        if not metadata:
            raise ValueError("Could not load metadata from MLflow. Please ensure metadata file exists in model artifacts.")
        
        # Extract and validate required fields
        x_mean = np.array(metadata.get("x_mean", []), dtype=np.float32)
        x_std = np.array(metadata.get("x_std", []), dtype=np.float32)
        class_names = metadata.get("class_names", [])
        feature_names = metadata.get("feature_names")
        
        # Also check nested locations
        if not feature_names and "model_info" in metadata:
            model_info = metadata.get("model_info", {})
            if isinstance(model_info, dict):
                data_info = model_info.get("data_info", {})
                if isinstance(data_info, dict):
                    feature_names = data_info.get("feature_names")
        
        result = {
            "x_mean": x_mean,
            "x_std": x_std,
            "class_names": class_names,
            "feature_names": feature_names,
            "raw_metadata": metadata  # Keep full metadata for reference
        }
        
        logger.info(f"Loaded metadata: x_mean shape={len(x_mean)}, class_names={len(class_names)}, feature_names={feature_names is not None}")
        return result
    
    def _load_metadata_from_mlflow(self, run_id: str, model_uri: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata from MLflow artifacts.
        
        This method replicates the logic from inference_docker/inference_job.py
        to ensure compatibility.
        """
        try:
            client = MlflowClient()
            run = client.get_run(run_id)
            artifact_uri = run.info.artifact_uri
            
            logger.info(f"Artifact URI: {artifact_uri}")
            
            # Try to find metadata file in artifacts
            try:
                artifacts = client.list_artifacts(run_id, "model")
                logger.info(f"Found {len(artifacts)} artifacts in model/ folder")
            except Exception as e:
                logger.warning(f"Could not list artifacts in model/ folder: {e}")
                try:
                    artifacts = client.list_artifacts(run_id)
                    logger.info(f"Found {len(artifacts)} artifacts at root level")
                except Exception as e2:
                    logger.warning(f"Could not list artifacts at root level: {e2}")
                    artifacts = []
            
            if artifacts:
                # Priority: Find model-specific metadata file (*_model_metadata.json)
                model_specific_metadata = None
                generic_metadata = None
                
                for artifact in artifacts:
                    artifact_path = artifact.path
                    logger.debug(f"Found artifact: {artifact_path}")
                    
                    # Skip generic metadata.json - we prefer model-specific metadata
                    if artifact_path == "metadata.json" or artifact_path == "model/metadata.json":
                        generic_metadata = artifact_path
                        continue
                    
                    # Look for model-specific metadata (ends with _model_metadata.json)
                    if artifact_path.endswith("_model_metadata.json"):
                        model_specific_metadata = artifact_path
                        logger.info(f"Found model-specific metadata: {artifact_path}")
                        break
                
                # Use model-specific metadata if found, otherwise try alternatives
                metadata_path_to_use = model_specific_metadata
                if not metadata_path_to_use:
                    logger.debug("Model-specific metadata (*_model_metadata.json) not found. Looking for alternative...")
                    # Fallback: look for any _metadata.json that's not the generic one
                    for artifact in artifacts:
                        artifact_path = artifact.path
                        if artifact_path.endswith("_metadata.json") and artifact_path != "metadata.json" and artifact_path != "model/metadata.json":
                            metadata_path_to_use = artifact_path
                            logger.info(f"Found alternative metadata file: {artifact_path}")
                            break
                
                # Try generic metadata.json as last resort
                if not metadata_path_to_use and generic_metadata:
                    logger.warning("Using generic metadata.json as last resort")
                    metadata_path_to_use = generic_metadata
                
                if metadata_path_to_use:
                    # Try different path constructions for download (matching inference_job.py logic)
                    download_paths = []
                    
                    # If path already starts with "model/", use as-is
                    if metadata_path_to_use.startswith("model/"):
                        download_paths.append(metadata_path_to_use)
                    else:
                        # Try with "model/" prefix
                        download_paths.append(f"model/{metadata_path_to_use}")
                        # Try without prefix (in case it's at root)
                        download_paths.append(metadata_path_to_use)
                    
                    # Try different download paths using client.download_artifacts
                    # client.download_artifacts(run_id, artifact_path, dst_path) downloads to dst_path directory
                    metadata_content = None
                    for download_path in download_paths:
                        try:
                            logger.info(f"Attempting to download metadata from: {download_path}")
                            # Create temporary directory for download
                            with tempfile.TemporaryDirectory() as temp_dir:
                                # Download artifact - client.download_artifacts downloads to the directory
                                # It preserves the artifact path structure in the destination directory
                                client.download_artifacts(run_id, download_path, temp_dir)
                                
                                # Find the downloaded file
                                # client.download_artifacts creates subdirectories matching the artifact path
                                # So "model/metadata.json" becomes temp_dir/model/metadata.json
                                possible_paths = [
                                    os.path.join(temp_dir, download_path),  # Full path with subdirs
                                    os.path.join(temp_dir, os.path.basename(download_path)),  # Just filename
                                ]
                                
                                # If download_path contains subdirectories, also try those
                                if "/" in download_path:
                                    possible_paths.append(os.path.join(temp_dir, *download_path.split("/")))
                                
                                # Also search recursively in temp_dir for any _metadata.json or metadata.json
                                downloaded_file = None
                                for possible_path in possible_paths:
                                    if os.path.exists(possible_path) and os.path.isfile(possible_path):
                                        downloaded_file = possible_path
                                        break
                                
                                # If not found in expected paths, search recursively
                                if not downloaded_file:
                                    for root, dirs, files in os.walk(temp_dir):
                                        for file in files:
                                            if file.endswith("_metadata.json") or file == "metadata.json":
                                                downloaded_file = os.path.join(root, file)
                                                break
                                        if downloaded_file:
                                            break
                                
                                if downloaded_file and os.path.exists(downloaded_file):
                                    # Read the file content before temp_dir is deleted
                                    with open(downloaded_file, "r") as f:
                                        metadata_content = json.load(f)
                                    logger.info(f"Successfully downloaded and loaded metadata from: {download_path}")
                                    break
                                else:
                                    logger.debug(f"Downloaded file not found in temp directory for: {download_path}")
                        except Exception as e:
                            logger.debug(f"Failed to download from {download_path}: {e}")
                            continue
                    
                    if metadata_content:
                        return metadata_content
                    else:
                        logger.warning(f"Failed to download metadata artifact from any path: {download_paths}")
            
            logger.warning("No suitable metadata file found in artifacts")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load metadata from MLflow: {e}", exc_info=True)
            return None
    
    def _load_metadata_from_gcs(self, gcs_path: str) -> Dict[str, Any]:
        """Load metadata from GCS path."""
        try:
            from google.cloud import storage
            path_parts = gcs_path.replace("gs://", "").split("/", 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1] if len(path_parts) > 1 else ""
            
            if not blob_name:
                raise ValueError(f"Invalid GCS path: {gcs_path}")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_path = f.name
            
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.download_to_filename(temp_path)
                
                with open(temp_path, "r") as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from GCS: {gcs_path}")
                return metadata
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logger.error(f"Failed to load metadata from GCS: {e}")
            raise
    
    def _load_onnx_model(self, run_id: str, model_uri: str) -> ort.InferenceSession:
        """
        Load ONNX model from MLflow artifacts.
        
        Args:
            run_id: MLflow run ID
            model_uri: MLflow model URI
            
        Returns:
            ONNX Runtime InferenceSession
        """
        try:
            client = MlflowClient()
            
            # List artifacts to find ONNX model
            try:
                artifacts = client.list_artifacts(run_id, "model")
            except Exception:
                artifacts = client.list_artifacts(run_id)
            
            # Find ONNX model file
            onnx_path = None
            for artifact in artifacts:
                if artifact.path.endswith(".onnx"):
                    # Prefer model-specific ONNX files
                    if "model.onnx" in artifact.path or "_model.onnx" in artifact.path:
                        onnx_path = artifact.path
                        break
                    elif onnx_path is None:
                        # Keep as fallback
                        onnx_path = artifact.path
            
            if not onnx_path:
                raise ValueError(f"No ONNX model found in artifacts for run_id: {run_id}")
            
            # Download ONNX model to temp file
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                temp_path = f.name
            
            # Initialize downloaded_path to None to prevent UnboundLocalError in finally block
            downloaded_path = None
            
            try:
                client.download_artifacts(run_id, onnx_path, os.path.dirname(temp_path))
                # MLflow downloads to a subdirectory, find the actual file
                downloaded_path = os.path.join(os.path.dirname(temp_path), onnx_path)
                if not os.path.exists(downloaded_path):
                    # Try with just the filename
                    downloaded_path = os.path.join(os.path.dirname(temp_path), os.path.basename(onnx_path))
                
                if not os.path.exists(downloaded_path):
                    raise FileNotFoundError(f"Downloaded ONNX model not found at: {downloaded_path}")
                
                # Load ONNX model
                session = ort.InferenceSession(downloaded_path)
                logger.info(f"Loaded ONNX model: {onnx_path}, input names: {[inp.name for inp in session.get_inputs()]}")
                return session
            finally:
                # Clean up temp files
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if downloaded_path and os.path.exists(downloaded_path) and downloaded_path != temp_path:
                    os.remove(downloaded_path)
                    
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess input data for inference.
        
        Applies the same preprocessing as during training:
        1. Select features based on feature_names from metadata
        2. Normalize using x_mean and x_std from metadata
        3. Convert to numpy array with float32 dtype
        
        Args:
            df: Input DataFrame
            metadata: Model metadata with x_mean, x_std, feature_names
            
        Returns:
            Preprocessed numpy array ready for model inference
        """
        logger.info(f"Preprocessing data: input shape {df.shape}")
        
        # Extract feature names from metadata
        feature_names = metadata.get("feature_names")
        if feature_names is None:
            # Fallback: use all numeric columns
            logger.warning("feature_names not found in metadata, using all numeric columns")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found in input DataFrame")
            feature_names = numeric_cols
        else:
            # Ensure feature_names is a list
            if not isinstance(feature_names, list):
                feature_names = list(feature_names)
        
        # Select features from DataFrame
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features in input DataFrame: {missing_features}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Extract feature columns
        features = df[feature_names].copy()
        
        # Check for non-numeric values
        for col in features.columns:
            if not pd.api.types.is_numeric_dtype(features[col]):
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                    logger.warning(f"Converted non-numeric column '{col}' to numeric (NaN values may result)")
                except Exception as e:
                    raise ValueError(f"Cannot convert column '{col}' to numeric: {e}")
        
        # Check for NaN values after conversion
        if features.isna().any().any():
            nan_cols = features.columns[features.isna().any()].tolist()
            logger.warning(f"Found NaN values in columns: {nan_cols}. Filling with column mean.")
            features = features.fillna(features.mean())
        
        # Convert to numpy array
        features_array = features.values.astype(np.float32)
        
        # Apply normalization if x_mean and x_std are available
        x_mean = metadata.get("x_mean")
        x_std = metadata.get("x_std")
        
        if x_mean is not None and x_std is not None:
            # Convert to numpy arrays if they're lists
            if isinstance(x_mean, list):
                x_mean = np.array(x_mean, dtype=np.float32)
            if isinstance(x_std, list):
                x_std = np.array(x_std, dtype=np.float32)
            
            # Validate shapes match
            if len(x_mean) != features_array.shape[1]:
                raise ValueError(
                    f"Feature count mismatch: metadata has {len(x_mean)} features, "
                    f"but input has {features_array.shape[1]} features"
                )
            
            if len(x_std) != features_array.shape[1]:
                raise ValueError(
                    f"Feature count mismatch: metadata has {len(x_std)} features, "
                    f"but input has {features_array.shape[1]} features"
                )
            
            # Normalize: (x - mean) / std
            # Handle zero std (avoid division by zero)
            x_std_safe = np.where(x_std == 0, 1.0, x_std)
            features_array = (features_array - x_mean) / x_std_safe
            
            logger.info(f"Applied normalization: mean shape={x_mean.shape}, std shape={x_std.shape}")
        else:
            logger.warning("x_mean or x_std not found in metadata, skipping normalization")
        
        logger.info(f"Preprocessed data shape: {features_array.shape}, dtype: {features_array.dtype}")
        return features_array
    
    def run_inference(self, model: ort.InferenceSession, preprocessed_data: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed data using ONNX Runtime.
        
        Handles batch processing, input validation, and error handling.
        
        Args:
            model: ONNX InferenceSession
            preprocessed_data: Preprocessed input data (shape: [n_samples, n_features])
            
        Returns:
            Predictions array (shape depends on task type)
                - Classification: [n_samples, n_classes] (logits or probabilities)
                - Regression: [n_samples] or [n_samples, 1] (raw predictions)
        """
        logger.info(f"Running inference: input shape={preprocessed_data.shape}, dtype={preprocessed_data.dtype}")
        
        # Validate input
        if preprocessed_data.size == 0:
            raise ValueError("Cannot run inference on empty data")
        
        if preprocessed_data.ndim != 2:
            raise ValueError(
                f"Expected 2D input array [n_samples, n_features], got shape {preprocessed_data.shape}"
            )
        
        # Get model input specification
        model_inputs = model.get_inputs()
        if not model_inputs:
            raise ValueError("Model has no input specifications")
        
        # Use first input (most models have single input)
        input_spec = model_inputs[0]
        input_name = input_spec.name
        input_shape = input_spec.shape
        input_type = input_spec.type
        
        logger.debug(
            f"Model input: name={input_name}, shape={input_shape}, type={input_type}"
        )
        
        # Prepare input data - ensure correct shape
        # ONNX models may have dynamic batch dimension (None or -1)
        # Handle shape compatibility
        batch_size, n_features = preprocessed_data.shape
        
        # Validate feature count matches model input
        if len(input_shape) >= 2:
            expected_features = input_shape[-1]
            if expected_features is not None and expected_features != -1 and expected_features != n_features:
                raise ValueError(
                    f"Feature count mismatch: input has {n_features} features, "
                    f"but model expects {expected_features} features"
                )
        
        # Ensure input is contiguous and correct dtype
        if not preprocessed_data.flags['C_CONTIGUOUS']:
            preprocessed_data = np.ascontiguousarray(preprocessed_data)
        
        # Convert to correct dtype if needed
        # ONNX models typically expect float32
        if preprocessed_data.dtype != np.float32:
            logger.debug(f"Converting input dtype from {preprocessed_data.dtype} to float32")
            preprocessed_data = preprocessed_data.astype(np.float32)
        
        try:
            # Run inference
            # model.run(output_names=None, input_feed={input_name: data})
            # Returns list of output arrays
            input_feed = {input_name: preprocessed_data}
            logger.debug(f"Running model with input feed: {input_name} shape={preprocessed_data.shape}")
            
            outputs = model.run(None, input_feed)
            
            if not outputs or len(outputs) == 0:
                raise ValueError("Model returned no outputs")
            
            # Get first output (most models have single output)
            predictions = outputs[0]
            
            # Validate output shape
            if predictions.ndim == 0:
                raise ValueError(f"Model returned scalar output, expected array")
            
            # Log output information
            logger.info(
                f"Inference completed: output shape={predictions.shape}, "
                f"dtype={predictions.dtype}, output range=[{predictions.min():.3f}, {predictions.max():.3f}]"
            )
            
            # Handle 1D outputs - ensure consistent shape
            if predictions.ndim == 1:
                # For classification, might need to reshape to [n_samples, 1]
                # For regression, [n_samples] is fine
                if len(predictions) == batch_size:
                    logger.debug(f"Model returned 1D output: {predictions.shape}")
                else:
                    raise ValueError(
                        f"Output size mismatch: expected {batch_size} predictions, "
                        f"got {len(predictions)}"
                    )
            elif predictions.ndim == 2:
                # For classification: [n_samples, n_classes]
                # For regression: [n_samples, 1]
                if predictions.shape[0] != batch_size:
                    raise ValueError(
                        f"Batch size mismatch: expected {batch_size} samples, "
                        f"got {predictions.shape[0]} in output"
                    )
                logger.debug(f"Model returned 2D output: {predictions.shape}")
            else:
                logger.warning(
                    f"Unexpected output shape: {predictions.shape}, expected 1D or 2D"
                )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise ValueError(f"Failed to run inference: {str(e)}") from e
    
    def postprocess_predictions(self, predictions: np.ndarray, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Postprocess predictions (map class indices to names, add confidence).
        
        Handles both classification and regression tasks:
        - Classification: Maps class indices to names, computes confidence from probabilities
        - Regression: Uses raw predictions, no confidence (or placeholder)
        
        Args:
            predictions: Raw predictions from model
                - Classification: [n_samples, n_classes] (logits or probabilities)
                - Regression: [n_samples] or [n_samples, 1] (raw values)
            metadata: Model metadata with class_names, task_type
            
        Returns:
            DataFrame with 'Predicted' and 'Confidence' columns
        """
        logger.info(f"Postprocessing predictions: shape={predictions.shape}, dtype={predictions.dtype}")
        
        # Determine task type from metadata or prediction shape
        task_type = metadata.get("task_type") or metadata.get("raw_metadata", {}).get("task_type")
        if not task_type:
            # Infer from prediction shape and class_names
            class_names = metadata.get("class_names", [])
            if predictions.ndim == 2 and predictions.shape[1] > 1 and len(class_names) > 0:
                task_type = "classification"
            elif predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
                task_type = "regression"
            else:
                # Default to classification if we have class names, else regression
                task_type = "classification" if len(class_names) > 0 else "regression"
                logger.warning(f"Could not determine task_type from metadata, inferring as: {task_type}")
        
        logger.info(f"Task type: {task_type}")
        
        if task_type == "classification":
            # Classification: map indices to names, compute confidence
            class_names = metadata.get("class_names", [])
            
            # Handle nested class_names location
            if not class_names:
                raw_metadata = metadata.get("raw_metadata", {})
                if isinstance(raw_metadata, dict):
                    class_info = raw_metadata.get("class_info", {})
                    if isinstance(class_info, dict):
                        class_names = class_info.get("class_names", [])
            
            # Ensure predictions are 2D for classification
            if predictions.ndim == 1:
                # Single class output - might be regression or binary classification
                logger.warning("1D predictions for classification task - assuming binary classification")
                predictions = predictions.reshape(-1, 1)
            
            # Apply softmax if needed (check if values are logits)
            # Logits typically have large range, probabilities are in [0, 1] and sum to 1
            if predictions.ndim == 2 and predictions.shape[1] > 1:
                # Check if values look like logits (large range, not summing to 1)
                row_sums = np.sum(predictions, axis=1)
                max_val = np.max(predictions)
                min_val = np.min(predictions)
                
                # If values are not in [0, 1] range or don't sum to ~1, they're likely logits
                is_logits = (max_val > 10.0 or min_val < -10.0) or not np.allclose(row_sums, 1.0, atol=0.1)
                
                if is_logits:
                    logger.debug("Applying softmax to convert logits to probabilities")
                    # Apply softmax: exp(x) / sum(exp(x))
                    exp_predictions = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
                    probabilities = exp_predictions / np.sum(exp_predictions, axis=1, keepdims=True)
                else:
                    probabilities = predictions
            else:
                # Single class - assume probabilities
                probabilities = predictions
            
            # Get predicted class indices (argmax)
            predicted_indices = np.argmax(probabilities, axis=1)
            
            # Get confidence (max probability)
            if probabilities.ndim == 2 and probabilities.shape[1] > 1:
                confidence = np.max(probabilities, axis=1)
            else:
                # Binary classification or single output
                confidence = probabilities.flatten()
            
            # Map class indices to names
            if class_names and len(class_names) > 0:
                # Ensure class_names is a list
                if not isinstance(class_names, list):
                    class_names = list(class_names)
                
                # Validate indices are within range
                max_index = max(predicted_indices) if len(predicted_indices) > 0 else 0
                if max_index >= len(class_names):
                    logger.warning(
                        f"Predicted class index {max_index} exceeds class_names length {len(class_names)}. "
                        f"Using index as string."
                    )
                    predicted_classes = [class_names[idx] if idx < len(class_names) else str(idx) 
                                       for idx in predicted_indices]
                else:
                    predicted_classes = [class_names[idx] for idx in predicted_indices]
                
                logger.info(f"Mapped {len(predicted_classes)} predictions to class names")
            else:
                # No class names available - use indices as strings
                logger.warning("No class_names in metadata - using class indices as predictions")
                predicted_classes = [str(idx) for idx in predicted_indices]
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'Predicted': predicted_classes,
                'Confidence': confidence.astype(np.float32)
            })
            
            logger.info(
                f"Postprocessing complete: {len(result_df)} predictions, "
                f"confidence range=[{confidence.min():.3f}, {confidence.max():.3f}]"
            )
            
        else:
            # Regression: use raw predictions, no confidence
            logger.info("Regression task - using raw predictions")
            
            # Flatten to 1D if needed
            if predictions.ndim == 2:
                if predictions.shape[1] == 1:
                    predictions = predictions.flatten()
                else:
                    logger.warning(f"Unexpected 2D regression output shape: {predictions.shape}, using first column")
                    predictions = predictions[:, 0]
            
            # Create result DataFrame
            # Create Confidence column as array of None values matching predictions length
            # This avoids ValueError when mixing scalar None with array-like predictions
            n_samples = len(predictions)
            result_df = pd.DataFrame({
                'Predicted': predictions.astype(np.float32),
                'Confidence': [None] * n_samples  # No confidence for regression - array of None
            })
            
            logger.info(
                f"Postprocessing complete: {len(result_df)} regression predictions, "
                f"range=[{predictions.min():.3f}, {predictions.max():.3f}]"
            )
        
        return result_df
    
    def predict(self, df: pd.DataFrame, run_id: str, model_uri: Optional[str] = None) -> pd.DataFrame:
        """
        Main inference method - runs complete inference pipeline.
        
        Args:
            df: Input DataFrame
            run_id: MLflow run ID
            model_uri: Optional model URI (defaults to f"runs:/{run_id}/model")
            
        Returns:
            DataFrame with predictions
        """
        if model_uri is None:
            model_uri = f"runs:/{run_id}/model"
        
        # Load model and metadata
        model, metadata = self.load_model(run_id, model_uri)
        
        # Preprocess data
        preprocessed = self.preprocess_data(df, metadata)
        
        # Run inference
        predictions = self.run_inference(model, preprocessed)
        
        # Postprocess
        result_df = self.postprocess_predictions(predictions, metadata)
        
        return result_df


