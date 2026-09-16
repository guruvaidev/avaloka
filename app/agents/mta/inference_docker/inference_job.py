"""
ONNX Inference Job Script for Kubernetes

This script runs inside a Docker container to perform inference on ONNX models.
It reads configuration from environment variables and outputs predictions to GCS.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import onnxruntime as ort
import mlflow
from pathlib import Path
from google.cloud import storage
from google.api_core import exceptions as google_exceptions
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_metadata_from_mlflow(model_uri: str, metadata_path: str = None):
    """
    Load normalization statistics and class names from MLflow model artifacts.
    
    Args:
        model_uri: MLflow model URI (e.g., "runs:/run_42/model")
        metadata_path: Optional path to metadata JSON file (local or GCS path)
    
    Returns:
        Tuple of (x_mean, x_std, class_names, feature_names)
    """
    metadata = None
    
    # Priority 1: Try to load from provided metadata_path (if it's a GCS path or local file)
    if metadata_path:
        try:
            # Check if it's a GCS path
            if metadata_path.startswith("gs://"):
                # Download from GCS
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                temp_path = temp_file.name
                temp_file.close()
                
                try:
                    path_parts = metadata_path.replace("gs://", "").split("/", 1)
                    bucket_name = path_parts[0]
                    blob_name = path_parts[1] if len(path_parts) > 1 else ""
                    
                    # Validate that blob_name is non-empty
                    if not blob_name or not blob_name.strip():
                        raise ValueError(
                            f"Invalid GCS metadata path: '{metadata_path}'. "
                            f"GCS paths must include a blob path after the bucket name (format: gs://bucket/path/to/file). "
                            f"Received path only contains bucket: '{bucket_name}'"
                        )
                    
                    storage_client = storage.Client()
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    blob.download_to_filename(temp_path)
                    
                    with open(temp_path, "r") as f:
                        metadata = json.load(f)
                    logger.info(f"Loaded metadata from GCS: {metadata_path}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                # Local file path
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from local path: {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
    
    # Priority 2: Download metadata from MLflow artifacts
    if not metadata:
        try:
            # Parse model_uri to extract run_id
            # Format: "runs:/run_id/model" or "runs:/run_id"
            if model_uri.startswith("runs:/"):
                run_id = model_uri.replace("runs:/", "").split("/")[0]
                logger.info(f"Extracting run_id from model_uri: {run_id}")
            else:
                raise ValueError(f"Invalid model_uri format: {model_uri}. Expected format: 'runs:/run_id/model'")
            
            # Get run info to access artifacts
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            artifact_uri = run.info.artifact_uri
            
            logger.info(f"Artifact URI: {artifact_uri}")
            
            # Try to download metadata from model artifacts
            # Priority: Use model-specific metadata file (*_model_metadata.json), skip generic metadata.json
            try:
                # Try listing artifacts in model/ folder first
                artifacts = None
                try:
                    artifacts = client.list_artifacts(run_id, "model")
                    logger.info(f"Found {len(artifacts)} artifacts in model/ folder")
                except Exception as e:
                    logger.warning(f"Could not list artifacts in model/ folder: {e}")
                    # Try listing at root level
                    try:
                        artifacts = client.list_artifacts(run_id)
                        logger.info(f"Found {len(artifacts)} artifacts at root level")
                    except Exception as e2:
                        logger.warning(f"Could not list artifacts at root level: {e2}")
                
                if artifacts:
                    # Priority 1: Find model-specific metadata file (*_model_metadata.json)
                    model_specific_metadata = None
                    generic_metadata = None
                    
                    for artifact in artifacts:
                        artifact_path = artifact.path
                        logger.debug(f"Found artifact: {artifact_path}")
                        
                        # Skip generic metadata.json - we only want model-specific metadata
                        if artifact_path == "metadata.json" or artifact_path == "model/metadata.json":
                            generic_metadata = artifact_path
                            continue
                        
                        # Look for model-specific metadata (ends with _model_metadata.json)
                        if artifact_path.endswith("_model_metadata.json"):
                            model_specific_metadata = artifact_path
                            logger.info(f"Found model-specific metadata: {artifact_path}")
                            break
                    
                    # Use model-specific metadata if found, otherwise skip generic
                    metadata_path_to_use = model_specific_metadata
                    if not metadata_path_to_use:
                        logger.warning("Model-specific metadata (*_model_metadata.json) not found. Looking for alternative...")
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
                        # Try different path constructions for download
                        download_paths = []
                        
                        # If path already starts with "model/", use as-is
                        if metadata_path_to_use.startswith("model/"):
                            download_paths.append(metadata_path_to_use)
                        else:
                            # Try with "model/" prefix
                            download_paths.append(f"model/{metadata_path_to_use}")
                            # Try without prefix (in case it's at root)
                            download_paths.append(metadata_path_to_use)
                        
                        metadata_file = None
                        for download_path in download_paths:
                            try:
                                logger.info(f"Attempting to download metadata from: {download_path}")
                                metadata_file = mlflow.artifacts.download_artifacts(
                                    run_id=run_id,
                                    artifact_path=download_path
                                )
                                if os.path.exists(metadata_file):
                                    logger.info(f"Successfully downloaded metadata from: {download_path}")
                                    break
                                else:
                                    logger.warning(f"Downloaded file does not exist: {metadata_file}")
                                    metadata_file = None
                            except Exception as e:
                                logger.debug(f"Failed to download from {download_path}: {e}")
                                metadata_file = None
                                continue
                        
                        if metadata_file and os.path.exists(metadata_file):
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                            logger.info(f"Successfully loaded metadata from: {metadata_path_to_use}")
                        else:
                            logger.warning(f"Could not download metadata file from any path: {download_paths}")
                            
                            # Fallback: Try downloading directly from GCS if artifact_uri is a GCS path
                            if artifact_uri and artifact_uri.startswith("gs://") and metadata_path_to_use:
                                try:
                                    logger.info(f"Attempting direct GCS download from artifact URI: {artifact_uri}")
                                    # Construct full GCS path
                                    # artifact_uri is like: gs://bucket/run_id/artifacts
                                    # We need: gs://bucket/run_id/artifacts/model/filename
                                    if metadata_path_to_use.startswith("model/"):
                                        gcs_path = f"{artifact_uri}/{metadata_path_to_use}"
                                    else:
                                        gcs_path = f"{artifact_uri}/model/{metadata_path_to_use}"
                                    
                                    logger.info(f"Downloading metadata directly from GCS: {gcs_path}")
                                    # Use the existing load_csv_from_gcs function logic
                                    import tempfile
                                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                                    temp_path = temp_file.name
                                    temp_file.close()
                                    
                                    path_parts = gcs_path.replace("gs://", "").split("/", 1)
                                    bucket_name = path_parts[0]
                                    blob_name = path_parts[1] if len(path_parts) > 1 else ""
                                    
                                    # Validate that blob_name is non-empty
                                    if not blob_name or not blob_name.strip():
                                        logger.warning(
                                            f"Invalid GCS metadata path: '{gcs_path}'. "
                                            f"GCS paths must include a blob path after the bucket name. "
                                            f"Skipping GCS metadata download."
                                        )
                                    else:
                                        storage_client = storage.Client()
                                        bucket = storage_client.bucket(bucket_name)
                                        blob = bucket.blob(blob_name)
                                        blob.download_to_filename(temp_path)
                                        
                                        if os.path.exists(temp_path):
                                            with open(temp_path, "r") as f:
                                                metadata = json.load(f)
                                            logger.info(f"Successfully loaded metadata from GCS: {gcs_path}")
                                            os.remove(temp_path)
                                        else:
                                            logger.warning(f"Downloaded GCS file does not exist: {temp_path}")
                                except Exception as e:
                                    logger.warning(f"Failed to download metadata directly from GCS: {e}")
                    else:
                        logger.warning("No suitable metadata file found in artifacts")
                else:
                    logger.warning("No artifacts found to search for metadata")
                    
                    # Last resort: Try to find metadata files directly in GCS if artifact_uri is GCS
                    if artifact_uri and artifact_uri.startswith("gs://") and not metadata:
                        try:
                            logger.info("Attempting to find metadata files directly in GCS...")
                            path_parts = artifact_uri.replace("gs://", "").split("/", 1)
                            bucket_name = path_parts[0]
                            prefix = f"{path_parts[1]}/model/" if len(path_parts) > 1 else "model/"
                            
                            storage_client = storage.Client()
                            bucket = storage_client.bucket(bucket_name)
                            
                            # List blobs with _model_metadata.json pattern
                            blobs = bucket.list_blobs(prefix=prefix)
                            metadata_blob = None
                            for blob in blobs:
                                if blob.name.endswith("_model_metadata.json"):
                                    metadata_blob = blob
                                    logger.info(f"Found metadata file in GCS: {blob.name}")
                                    break
                            
                            if metadata_blob:
                                import tempfile
                                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                                temp_path = temp_file.name
                                temp_file.close()
                                
                                metadata_blob.download_to_filename(temp_path)
                                if os.path.exists(temp_path):
                                    with open(temp_path, "r") as f:
                                        metadata = json.load(f)
                                    logger.info(f"Successfully loaded metadata from GCS: {metadata_blob.name}")
                                    os.remove(temp_path)
                        except Exception as e:
                            logger.warning(f"Failed to search for metadata files directly in GCS: {e}")
                    
            except Exception as e:
                logger.warning(f"Could not list or download artifacts: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Fallback: Try to load model using MLflow Python function API and extract metadata
            # This handles cases where the model might be loaded differently
            if not metadata:
                try:
                    logger.info("Attempting to load model using MLflow Python function API as fallback...")
                    # Use mlflow.pyfunc directly since mlflow is already imported at module level
                    mlflow_model = mlflow.pyfunc.load_model(model_uri)
                    
                    # Check if model was loaded successfully
                    if mlflow_model is None:
                        logger.warning("MLflow model loaded as None - skipping metadata extraction from model object")
                    else:
                        # Try to access metadata if available
                        if hasattr(mlflow_model, 'metadata') and mlflow_model.metadata is not None:
                            try:
                                # Try to get model_uuid or other metadata
                                model_path = mlflow_model.metadata.model_uuid if hasattr(mlflow_model.metadata, 'model_uuid') else None
                                logger.info(f"Extracted model path from MLflow model metadata: {model_path}")
                                
                                # If we have a model path, try to load metadata from artifacts
                                # This is a fallback - the artifact-based approach above should work
                                # But if it didn't, we can try using the model's metadata
                                pass  # Continue to try artifact-based approach
                            except AttributeError as e:
                                logger.warning(f"Could not access model metadata: {e}")
                        else:
                            logger.warning("MLflow model does not have accessible metadata attribute")
                except Exception as e:
                    logger.warning(f"Failed to load model using MLflow Python function API: {e}")
                    # Continue - we'll try other approaches or raise error at the end
                    
        except Exception as e:
            logger.error(f"Failed to load metadata from MLflow artifacts: {e}")
            # Don't raise immediately - check if we have metadata from other sources
            if not metadata:
                raise
    
    if not metadata:
        raise ValueError("Could not load metadata from MLflow. Please ensure metadata file exists in model artifacts.")
    
    # Extract normalization stats
    x_mean = np.array(metadata.get("x_mean", []), dtype=np.float32)
    x_std = np.array(metadata.get("x_std", []), dtype=np.float32)
    
    # Extract class names
    class_names = metadata.get("class_names", [])
    
    # Extract feature names
    feature_names = None
    if "feature_names" in metadata:
        feature_names = metadata["feature_names"]
        logger.info(f"Found feature_names in metadata: {feature_names}")
    elif "model_info" in metadata:
        model_info = metadata.get("model_info", {})
        if isinstance(model_info, dict):
            data_info = model_info.get("data_info", {})
            if isinstance(data_info, dict):
                feature_names = data_info.get("feature_names")
                if feature_names:
                    logger.info(f"Found feature_names in model_info.data_info: {feature_names}")
    
    # Also check for feature names in other possible locations
    if not feature_names:
        # Try checking for column names or input features
        if "columns" in metadata:
            feature_names = metadata.get("columns")
            logger.info(f"Found columns in metadata: {feature_names}")
        elif "input_features" in metadata:
            feature_names = metadata.get("input_features")
            logger.info(f"Found input_features in metadata: {feature_names}")
        elif "feature_columns" in metadata:
            feature_names = metadata.get("feature_columns")
            logger.info(f"Found feature_columns in metadata: {feature_names}")
    
    if not feature_names:
        logger.warning(f"Feature names not found in metadata. Available keys: {list(metadata.keys())}")
        # Log a sample of metadata for debugging
        logger.debug(f"Metadata sample (first 500 chars): {str(metadata)[:500]}")
    
    # Validate
    if len(x_mean) == 0 or len(x_std) == 0:
        raise ValueError(f"Metadata missing normalization stats (x_mean, x_std)")
    
    if len(x_mean) != len(x_std):
        raise ValueError(f"Mismatch: x_mean has {len(x_mean)} features, x_std has {len(x_std)}")
    
    return x_mean, x_std, class_names, feature_names


def load_csv_from_gcs(gcs_path: str, local_path: str = None):
    """
    Download CSV from GCS and load into DataFrame.
    
    Args:
        gcs_path: GCS path (gs://bucket/path/to/file.csv)
        local_path: Optional local path to save file
    
    Returns:
        DataFrame
    """
    if not gcs_path or not isinstance(gcs_path, str):
        raise ValueError("gcs_path must be a non-empty string")
    
    if not gcs_path.startswith("gs://"):
        # Path doesn't start with gs:// - try to construct full GCS URI
        # First check if it's actually a local file that exists
        if os.path.exists(gcs_path):
            logger.info(f"Loading CSV from local path: {gcs_path}")
            return pd.read_csv(gcs_path)
        
        # Not a local file - try to construct GCS URI from environment or default bucket
        # Check for GCS bucket in environment variables
        gcs_bucket = os.getenv("GCS_BUCKET") or os.getenv("MLFLOW_BUCKET")
        if gcs_bucket:
            # Construct full GCS URI
            full_gcs_path = f"gs://{gcs_bucket}/{gcs_path.lstrip('/')}"
            logger.info(f"Constructed GCS path from bucket env var: {full_gcs_path}")
            gcs_path = full_gcs_path
        else:
            # Last resort: try to infer bucket from MLflow tracking URI if it's a GCS path
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
            if mlflow_uri.startswith("gs://"):
                # Extract bucket from MLflow URI as fallback
                mlflow_bucket = mlflow_uri.replace("gs://", "").split("/")[0]
                full_gcs_path = f"gs://{mlflow_bucket}/{gcs_path.lstrip('/')}"
                logger.info(f"Constructed GCS path from MLflow URI: {full_gcs_path}")
                gcs_path = full_gcs_path
            else:
                # Cannot determine bucket - raise error with helpful message
                raise ValueError(
                    f"Cannot load CSV: path '{gcs_path}' is not a full GCS URI (gs://bucket/path) "
                    f"and is not a local file. Please provide a full GCS URI or ensure GCS_BUCKET "
                    f"environment variable is set."
                )
    
    # Parse GCS path
    path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""
    
    # Validate that blob_name is non-empty
    if not blob_name or not blob_name.strip():
        raise ValueError(
            f"Invalid GCS path: '{gcs_path}'. "
            f"GCS paths must include a blob path after the bucket name (format: gs://bucket/path/to/file). "
            f"Received path only contains bucket: '{bucket_name}'"
        )
    
    # Download to local temp file
    if not local_path:
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        local_path = temp_file.name
        temp_file.close()  # Explicitly close to avoid file handle leak
    
    # Download from GCS
    logger.info(f"Downloading CSV from GCS: {gcs_path}")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
    except (google_exceptions.Forbidden, PermissionError) as e:
        # Handle GCS permission errors specifically
        error_msg = str(e)
        logger.error(
            f"GCS access denied (403 Forbidden) when downloading: {gcs_path}\n"
            f"Bucket: {bucket_name}, Blob: {blob_name}\n"
            f"Exception type: {type(e).__name__}\n"
            f"This usually means:\n"
            f"  1. The Kubernetes pod's service account doesn't have GCS read permissions\n"
            f"  2. The service account needs 'Storage Object Viewer' role for reading\n"
            f"  3. Check the service account attached to the pod and its IAM permissions\n"
            f"  4. Ensure the service account has access to bucket: {bucket_name}\n"
            f"  5. Verify Workload Identity is configured if using GKE\n"
            f"Error details: {error_msg}"
        )
        raise PermissionError(
            f"GCS access denied: Cannot download {gcs_path}. "
            f"The pod's service account needs 'Storage Object Viewer' role on bucket '{bucket_name}'. "
            f"Please check IAM permissions for the service account."
        ) from e
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        # Check for permission errors in string representation
        if "403" in error_msg or "Forbidden" in error_msg or "Permission denied" in error_msg.lower():
            logger.error(
                f"GCS access denied (403 Forbidden) when downloading: {gcs_path}\n"
                f"Bucket: {bucket_name}, Blob: {blob_name}\n"
                f"Exception type: {error_type}\n"
                f"This usually means:\n"
                f"  1. The Kubernetes pod's service account doesn't have GCS read permissions\n"
                f"  2. The service account needs 'Storage Object Viewer' role for reading\n"
                f"  3. Check the service account attached to the pod and its IAM permissions\n"
                f"  4. Ensure the service account has access to bucket: {bucket_name}\n"
                f"Error details: {error_msg}"
            )
            raise PermissionError(
                f"GCS access denied: Cannot download {gcs_path}. "
                f"The pod's service account needs 'Storage Object Viewer' role on bucket '{bucket_name}'. "
                f"Please check IAM permissions for the service account."
            ) from e
        else:
            # Re-raise other errors as-is
            logger.error(f"Unexpected error downloading from GCS: {error_type}: {error_msg}")
            raise
    
    # Load CSV
    df = pd.read_csv(local_path)
    logger.info(f"Loaded CSV with shape: {df.shape}")
    
    return df


def upload_csv_to_gcs(df: pd.DataFrame, gcs_path: str):
    """
    Upload DataFrame as CSV to GCS.
    
    Args:
        df: DataFrame to upload
        gcs_path: GCS path (gs://bucket/path/to/file.csv)
    """
    if not gcs_path.startswith("gs://"):
        # Assume local path
        logger.info(f"Saving CSV to local path: {gcs_path}")
        df.to_csv(gcs_path, index=False)
        return
    
    # Parse GCS path
    path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""
    
    # Validate that blob_name is non-empty
    if not blob_name or not blob_name.strip():
        raise ValueError(
            f"Invalid GCS path: '{gcs_path}'. "
            f"GCS paths must include a blob path after the bucket name (format: gs://bucket/path/to/file). "
            f"Received path only contains bucket: '{bucket_name}'"
        )
    
    # Save to local temp file first
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_path = temp_file.name
    df.to_csv(temp_path, index=False)
    temp_file.close()
    
    try:
        # Upload to GCS
        logger.info(f"Uploading CSV to GCS: {gcs_path}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(temp_path)
        logger.info(f"Successfully uploaded to: {gcs_path}")
    except (google_exceptions.Forbidden, PermissionError) as e:
        # Handle GCS permission errors specifically
        error_msg = str(e)
        logger.error(
            f"GCS access denied (403 Forbidden) when uploading: {gcs_path}\n"
            f"Bucket: {bucket_name}, Blob: {blob_name}\n"
            f"Exception type: {type(e).__name__}\n"
            f"This usually means:\n"
            f"  1. The Kubernetes pod's service account doesn't have GCS write permissions\n"
            f"  2. The service account needs 'Storage Object Creator' or 'Storage Object Admin' role\n"
            f"  3. Check the service account attached to the pod and its IAM permissions\n"
            f"  4. Ensure the service account has write access to bucket: {bucket_name}\n"
            f"  5. Verify Workload Identity is configured if using GKE\n"
            f"Error details: {error_msg}"
        )
        raise PermissionError(
            f"GCS access denied: Cannot upload {gcs_path}. "
            f"The pod's service account needs 'Storage Object Creator' role on bucket '{bucket_name}'. "
            f"Please check IAM permissions for the service account."
        ) from e
    except Exception as e:
        error_msg = str(e)
        # Check for permission errors in string representation
        if "403" in error_msg or "Forbidden" in error_msg or "Permission denied" in error_msg.lower():
            logger.error(
                f"GCS access denied (403 Forbidden) when uploading: {gcs_path}\n"
                f"Bucket: {bucket_name}, Blob: {blob_name}\n"
                f"Exception type: {type(e).__name__}\n"
                f"This usually means:\n"
                f"  1. The Kubernetes pod's service account doesn't have GCS write permissions\n"
                f"  2. The service account needs 'Storage Object Creator' or 'Storage Object Admin' role\n"
                f"  3. Check the service account attached to the pod and its IAM permissions\n"
                f"  4. Ensure the service account has write access to bucket: {bucket_name}\n"
                f"Error details: {error_msg}"
            )
            raise PermissionError(
                f"GCS access denied: Cannot upload {gcs_path}. "
                f"The pod's service account needs 'Storage Object Creator' role on bucket '{bucket_name}'. "
                f"Please check IAM permissions for the service account."
            ) from e
        else:
            # Re-raise other errors as-is
            logger.error(f"Unexpected error uploading to GCS: {type(e).__name__}: {error_msg}")
            raise
        logger.info(f"Successfully uploaded to: {gcs_path}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def predict(df: pd.DataFrame, 
            model_uri: str,
            metadata_path: str = None,
            feature_cols: list = None):
    """
    Run inference on DataFrame using ONNX model from MLflow.
    
    Args:
        df: Input DataFrame
        model_uri: MLflow model URI (e.g., "runs:/run_42/model")
        metadata_path: Optional path to metadata JSON file
        feature_cols: List of feature column names (auto-detected if None)
    
    Returns:
        DataFrame with predictions added
    """
    # Load metadata
    x_mean, x_std, class_names, metadata_feature_names = load_metadata_from_mlflow(model_uri, metadata_path)
    
    logger.info(f"Loaded metadata: x_mean shape={len(x_mean)}, feature_names={metadata_feature_names}")
    logger.info(f"CSV columns: {list(df.columns)}")
    logger.info(f"CSV shape: {df.shape}")
    logger.info(f"CSV dtypes: {df.dtypes.to_dict()}")
    
    # Determine feature columns
    if feature_cols is None:
        # Priority 1: Use feature names from metadata (most accurate)
        if metadata_feature_names and len(metadata_feature_names) == len(x_mean):
            feature_cols = metadata_feature_names.copy()
            logger.info(f"Using feature columns from metadata: {feature_cols}")
        else:
            # Priority 2: Auto-detect from CSV
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            logger.info(f"Found numeric columns in CSV: {numeric_cols}")
            
            # Exclude target/label columns and ID columns
            exclude = ['target', 'Target', 'label', 'Label', 'Species', 'species', 'Predicted', 'Confidence',
                      'Id', 'id', 'ID', 'index', 'Index', 'idx', 'Idx']
            feature_cols = [col for col in numeric_cols if col not in exclude]
            logger.info(f"Auto-detected feature columns (after exclusions): {feature_cols}")
            
            # If we still have no features and metadata_feature_names is available, try using it
            if len(feature_cols) == 0 and metadata_feature_names and len(metadata_feature_names) == len(x_mean):
                logger.warning("No features detected from CSV, but metadata has feature names. Attempting to use metadata feature names...")
                # Check if any metadata feature names exist in CSV
                matching_cols = [col for col in metadata_feature_names if col in df.columns]
                if len(matching_cols) > 0:
                    logger.info(f"Found {len(matching_cols)} matching columns from metadata: {matching_cols}")
                    feature_cols = matching_cols
                else:
                    # Use metadata feature names anyway and let the missing columns handler deal with it
                    logger.warning(f"No matching columns found. Using metadata feature names as-is: {metadata_feature_names}")
                    feature_cols = metadata_feature_names.copy()
            
            # If still no features, try including all numeric columns (even excluded ones) as last resort
            if len(feature_cols) == 0 and len(numeric_cols) > 0:
                logger.warning(f"No features after exclusions. Using all numeric columns (including potentially excluded ones): {numeric_cols}")
                feature_cols = numeric_cols.copy()
            
            # If still no features and we know how many we need, try to match by position
            if len(feature_cols) == 0 and len(x_mean) > 0:
                logger.warning(f"Still no features detected. Model expects {len(x_mean)} features.")
                logger.warning(f"Attempting to use first {len(x_mean)} numeric columns by position...")
                if len(numeric_cols) >= len(x_mean):
                    feature_cols = numeric_cols[:len(x_mean)]
                    logger.info(f"Using first {len(x_mean)} numeric columns: {feature_cols}")
                elif len(df.columns) >= len(x_mean):
                    # Try using columns, but exclude ID columns
                    candidate_cols = [col for col in df.columns if col not in exclude]
                    if len(candidate_cols) >= len(x_mean):
                        feature_cols = candidate_cols[:len(x_mean)]
                        logger.warning(f"Using first {len(x_mean)} non-excluded columns (may need type conversion): {feature_cols}")
                    else:
                        # Last resort: use first N columns but skip ID columns
                        feature_cols = []
                        for col in df.columns:
                            if col not in exclude and len(feature_cols) < len(x_mean):
                                feature_cols.append(col)
                        logger.warning(f"Using {len(feature_cols)} non-excluded columns: {feature_cols}")
    
    # Handle missing columns - check if we need to add dummy columns
    missing_cols = [col for col in feature_cols if col not in df.columns]
    
    # If we have missing columns, try to handle them intelligently
    if missing_cols:
        logger.warning(f"Missing columns in CSV: {missing_cols}")
        if metadata_feature_names:
            logger.info(f"Model expects these features: {metadata_feature_names}")
        
        # Check if missing columns are ID-like columns that can be safely added
        id_like_cols = ['Id', 'id', 'ID', 'index', 'Index']
        can_add_dummy = all(
            any(col.lower() == id_col.lower() for id_col in id_like_cols) 
            for col in missing_cols
        )
        
        if can_add_dummy:
            logger.info(f"Adding dummy {missing_cols} column(s) with sequential values")
            for col in missing_cols:
                df[col] = range(len(df))
            logger.info(f"✓ Added dummy columns. Updated feature columns: {feature_cols}")
        else:
            # For non-ID columns, we can't safely add dummy values
            raise ValueError(
                f"Missing required columns in CSV: {missing_cols}.\n"
                f"Model was trained with {len(x_mean)} features.\n"
                f"Expected features: {metadata_feature_names if metadata_feature_names else 'Unknown (check metadata)'}\n"
                f"CSV has: {list(df.columns)}\n\n"
                f"Solutions:\n"
                f"  1. Add missing columns to your CSV\n"
                f"  2. Retrain model without the problematic columns"
            )
    
    # Validate feature count matches
    if len(feature_cols) == 0:
        raise ValueError(
            f"Could not detect any feature columns from CSV.\n"
            f"Model expects {len(x_mean)} features.\n"
            f"CSV columns: {list(df.columns)}\n"
            f"CSV dtypes: {df.dtypes.to_dict()}\n"
            f"Model expects features: {metadata_feature_names if metadata_feature_names else 'Unknown (check metadata)'}\n\n"
            f"Possible issues:\n"
            f"  1. CSV has no numeric columns\n"
            f"  2. All numeric columns are excluded (target/label columns)\n"
            f"  3. Column names don't match metadata feature names\n"
            f"  4. CSV structure is unexpected"
        )
    
    if len(feature_cols) != len(x_mean):
        # If we're missing exactly 1 feature and have an Id column, try including it
        if len(feature_cols) == len(x_mean) - 1 and 'Id' in df.columns:
            logger.warning(f"Feature count mismatch: CSV has {len(feature_cols)} features, model expects {len(x_mean)}. "
                         f"Attempting to include 'Id' column as the missing feature.")
            try:
                # Try converting Id to numeric
                df['Id'] = pd.to_numeric(df['Id'], errors='coerce')
                if df['Id'].notna().all():
                    feature_cols.append('Id')
                    logger.info(f"Successfully included 'Id' column. Updated features: {feature_cols}")
                else:
                    logger.warning("Could not convert 'Id' column to numeric")
            except Exception as e:
                logger.warning(f"Failed to include 'Id' column: {e}")
        
        # If still mismatched, raise error
        if len(feature_cols) != len(x_mean):
            # Provide more helpful error message
            error_msg = (
                f"Feature count mismatch: CSV has {len(feature_cols)} features, "
                f"model expects {len(x_mean)} features.\n"
                f"CSV features: {feature_cols}\n"
                f"CSV columns (all): {list(df.columns)}\n"
            )
            
            if metadata_feature_names:
                error_msg += (
                    f"Model expects features: {metadata_feature_names}\n"
                    f"Missing features: {[f for f in metadata_feature_names if f not in feature_cols]}\n"
                )
            else:
                error_msg += (
                    f"Model expects: Unknown (check metadata) - feature names not available in metadata\n"
                    f"Metadata has normalization stats for {len(x_mean)} features, but feature names are missing.\n"
                )
            
            error_msg += (
                f"\nSolutions:\n"
                f"  1. Ensure CSV has {len(x_mean)} numeric feature columns\n"
                f"  2. Check that column names match the training data\n"
                f"  3. If model was trained with an ID column, ensure your CSV has a matching ID column\n"
                f"  4. Verify the uploaded CSV matches the training data structure"
            )
            
            raise ValueError(error_msg)
    
    # Extract features - handle non-numeric columns by converting to float
    logger.info(f"Extracting features from columns: {feature_cols}")
    try:
        X = df[feature_cols].values.astype(np.float32)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert features to float32, attempting conversion: {e}")
        # Try converting each column individually
        X_converted = []
        for col in feature_cols:
            try:
                X_converted.append(pd.to_numeric(df[col], errors='coerce').values.astype(np.float32))
            except Exception as col_e:
                raise ValueError(
                    f"Could not convert column '{col}' to numeric. "
                    f"Column dtype: {df[col].dtype}, Sample values: {df[col].head(3).tolist()}. "
                    f"Error: {col_e}"
                )
        X = np.column_stack(X_converted)
        logger.info(f"Successfully converted features to float32")
    
    # Apply normalization: (X - mean) / std
    if x_mean.ndim == 1:
        X_norm = (X - x_mean) / x_std
    else:
        X_norm = (X - x_mean) / x_std
    
    # Load ONNX model from MLflow
    logger.info(f"Loading ONNX model from MLflow: {model_uri}")
    
    # MLflow stores ONNX models in artifacts
    # We need to download the ONNX model file
    try:
        # Parse model_uri to extract run_id
        # Format: "runs:/run_id/model" or "runs:/run_id"
        if model_uri.startswith("runs:/"):
            run_id = model_uri.replace("runs:/", "").split("/")[0]
            logger.info(f"Extracting run_id from model_uri: {run_id}")
        else:
            raise ValueError(f"Invalid model_uri format: {model_uri}. Expected format: 'runs:/run_id/model'")
        
        # Get run info to access artifacts
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        
        logger.info(f"Artifact URI: {artifact_uri}")
        
        # Try to find ONNX model in model/ folder
        onnx_model_path = None
        onnx_path_to_use = None  # Store detected path for GCS fallback
        
        # List artifacts in model/ folder to find ONNX model
        try:
            artifacts = client.list_artifacts(run_id, "model")
            
            # Priority: Find model-specific ONNX file (matches the metadata pattern)
            # Look for files like: mta_data_2025-12-03_20_00_43_020168_model.onnx
            onnx_candidates = []
            for artifact in artifacts:
                if artifact.path.endswith(".onnx"):
                    onnx_candidates.append(artifact.path)
            
            # Prefer model-specific ONNX files (contain model name pattern)
            # These typically match the metadata file pattern
            preferred_onnx = None
            fallback_onnx = None
            
            for candidate in onnx_candidates:
                # list_artifacts returns paths relative to "model/" prefix
                # So paths are like "mta_data_2025-12-03_20_00_43_020168_model.onnx"
                
                # Check if it's a model-specific file (contains model name pattern)
                # Pattern: *model.onnx or matches metadata file name pattern
                if "model.onnx" in candidate or "_model.onnx" in candidate:
                    preferred_onnx = candidate
                    break
                else:
                    # Keep as fallback
                    if not fallback_onnx:
                        fallback_onnx = candidate
            
            # Use preferred ONNX file, or fallback if no preferred found
            onnx_path_to_use = preferred_onnx or fallback_onnx
            
            if onnx_path_to_use:
                # Try different path constructions for download
                download_paths = []
                
                # If path already starts with "model/", use as-is
                if onnx_path_to_use.startswith("model/"):
                    download_paths.append(onnx_path_to_use)
                else:
                    # Try with "model/" prefix
                    download_paths.append(f"model/{onnx_path_to_use}")
                    # Try without prefix (in case it's at root)
                    download_paths.append(onnx_path_to_use)
                
                for download_path in download_paths:
                    try:
                        logger.info(f"Attempting to download ONNX model from: {download_path}")
                        onnx_model_path = mlflow.artifacts.download_artifacts(
                            run_id=run_id,
                            artifact_path=download_path
                        )
                        if os.path.exists(onnx_model_path):
                            logger.info(f"Successfully downloaded ONNX model from: {download_path}")
                            break
                        else:
                            logger.warning(f"Downloaded file does not exist: {onnx_model_path}")
                            onnx_model_path = None
                    except Exception as e:
                        logger.debug(f"Failed to download from {download_path}: {e}")
                        onnx_model_path = None
                        continue
            else:
                logger.warning("No ONNX model file found in model artifacts")
                
        except Exception as e:
            logger.warning(f"Could not list artifacts in model/ folder: {e}")
            # Fallback: try direct paths
            possible_paths = [
                "model/model.onnx",
            ]
            for path in possible_paths:
                try:
                    onnx_model_path = mlflow.artifacts.download_artifacts(
                        run_id=run_id,
                        artifact_path=path
                    )
                    if os.path.exists(onnx_model_path):
                        logger.info(f"Found ONNX model using fallback path: {path}")
                        break
                except Exception:
                    continue
        
        # If still not found, try downloading directly from GCS
        if (not onnx_model_path or not os.path.exists(onnx_model_path)) and artifact_uri and artifact_uri.startswith("gs://"):
            try:
                logger.info("Attempting to find ONNX model directly in GCS...")
                path_parts = artifact_uri.replace("gs://", "").split("/", 1)
                bucket_name = path_parts[0]
                
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                
                # If we have a detected path, try constructing the full GCS path first
                if onnx_path_to_use:
                    try:
                        prefix = f"{path_parts[1]}/model/" if len(path_parts) > 1 else "model/"
                        if onnx_path_to_use.startswith("model/"):
                            gcs_path = f"{artifact_uri}/{onnx_path_to_use}"
                        else:
                            gcs_path = f"{artifact_uri}/model/{onnx_path_to_use}"
                        
                        logger.info(f"Attempting direct GCS download from: {gcs_path}")
                        gcs_path_parts = gcs_path.replace("gs://", "").split("/", 1)
                        gcs_bucket_name = gcs_path_parts[0]
                        gcs_blob_name = gcs_path_parts[1] if len(gcs_path_parts) > 1 else ""
                        
                        # Validate that blob_name is non-empty
                        if not gcs_blob_name or not gcs_blob_name.strip():
                            logger.warning(
                                f"Invalid GCS path for ONNX model: '{gcs_path}'. "
                                f"GCS paths must include a blob path after the bucket name. "
                                f"Skipping direct GCS download, will try MLflow artifacts instead."
                            )
                        else:
                            gcs_bucket = storage_client.bucket(gcs_bucket_name)
                            gcs_blob = gcs_bucket.blob(gcs_blob_name)
                            
                            if gcs_blob.exists():
                                import tempfile
                                temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.onnx', delete=False)
                                temp_path = temp_file.name
                                temp_file.close()
                                
                                gcs_blob.download_to_filename(temp_path)
                                if os.path.exists(temp_path):
                                    onnx_model_path = temp_path
                                    logger.info(f"Successfully downloaded ONNX model from GCS: {gcs_path}")
                                else:
                                    logger.warning(f"Downloaded file does not exist: {temp_path}")
                            else:
                                logger.warning(f"ONNX model blob does not exist at: {gcs_path}")
                    except Exception as e:
                        logger.debug(f"Failed to download from constructed GCS path: {e}")
                
                # If still not found, search for .onnx files in GCS
                if not onnx_model_path or not os.path.exists(onnx_model_path):
                    prefix = f"{path_parts[1]}/model/" if len(path_parts) > 1 else "model/"
                    
                    # List blobs with .onnx extension
                    blobs = bucket.list_blobs(prefix=prefix)
                    onnx_blob = None
                    for blob in blobs:
                        if blob.name.endswith(".onnx"):
                            # Prefer model-specific ONNX files
                            if "_model.onnx" in blob.name or "model.onnx" in blob.name:
                                onnx_blob = blob
                                logger.info(f"Found ONNX model in GCS: {blob.name}")
                                break
                            elif not onnx_blob:
                                # Keep as fallback
                                onnx_blob = blob
                                logger.info(f"Found ONNX model in GCS (fallback): {blob.name}")
                    
                    if onnx_blob:
                        import tempfile
                        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.onnx', delete=False)
                        temp_path = temp_file.name
                        temp_file.close()
                        
                        onnx_blob.download_to_filename(temp_path)
                        if os.path.exists(temp_path):
                            onnx_model_path = temp_path
                            logger.info(f"Successfully downloaded ONNX model from GCS: {onnx_blob.name}")
            except Exception as e:
                logger.warning(f"Failed to download ONNX model directly from GCS: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
        
        if not onnx_model_path or not os.path.exists(onnx_model_path):
            raise FileNotFoundError(
                f"ONNX model not found in MLflow artifacts for {model_uri}.\n"
                f"Artifact URI: {artifact_uri}\n"
                f"Checked model/ folder and GCS direct download.\n"
                f"Please ensure the ONNX model file exists in the model artifacts."
            )
        
        logger.info(f"Found ONNX model at: {onnx_model_path}")
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_model_path)
        
        # Get input name (should be "input" based on our export)
        input_tensor = session.get_inputs()[0]
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        logger.info(f"Model input name: {input_name}")
        logger.info(f"Model input shape: {input_shape}")
        logger.info(f"Model output name: {session.get_outputs()[0].name}")
        logger.info(f"Model output shape: {session.get_outputs()[0].shape}")
        
        # Check if model has fixed batch size (batch dimension is 1 or None)
        is_fixed_batch = input_shape and len(input_shape) > 0 and input_shape[0] == 1
        
        # Run inference
        logger.info(f"Running inference on {len(X_norm)} samples...")
        
        if is_fixed_batch:
            # Model expects batch size of 1, process samples one at a time
            logger.warning(f"Model has fixed batch size of 1. Processing {len(X_norm)} samples individually (this may be slow).")
            all_logits = []
            
            for i in range(len(X_norm)):
                # Extract single sample and ensure shape is (1, num_features)
                sample = X_norm[i:i+1]  # Keep batch dimension
                if sample.ndim == 1:
                    sample = sample.reshape(1, -1)
                outputs = session.run(None, {input_name: sample})
                all_logits.append(outputs[0])
            
            # Concatenate all results
            logits = np.concatenate(all_logits, axis=0)
        else:
            # Model supports dynamic batch size, process all at once
            outputs = session.run(None, {input_name: X_norm})
            logits = outputs[0]
        
        # Apply softmax to get probabilities
        # Numerical stability: subtract max before exp
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # Get predictions
        pred_idx = probs.argmax(axis=1)
        
        # Map to class names if available
        if class_names and len(class_names) > 0:
            pred_labels = [class_names[i] for i in pred_idx]
        else:
            # Regression or no class names
            pred_labels = pred_idx
        
        confidence = probs.max(axis=1)
        
        # Add predictions to dataframe
        df["Predicted"] = pred_labels
        df["Confidence"] = confidence
        
        # Add per-class probabilities if classification
        if len(class_names) > 0:
            for i, class_name in enumerate(class_names):
                df[f"Prob_{class_name}"] = probs[:, i]
        
        logger.info(f"✓ Inference completed. Predictions added to DataFrame.")
        logger.info(f"Class distribution:\n{df['Predicted'].value_counts()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to run inference: {e}")
        raise


def main():
    """Main entry point for inference job"""
    # Read environment variables
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_uri = os.getenv("MODEL_URI")  # e.g., "runs:/run_42/model"
    input_path = os.getenv("INPUT_PATH")  # GCS path or local path
    output_path = os.getenv("OUTPUT_PATH")  # GCS path or local path
    metadata_path = os.getenv("METADATA_PATH")  # Optional metadata file path
    
    # Validate required environment variables
    if not mlflow_tracking_uri:
        logger.error("MLFLOW_TRACKING_URI environment variable not set")
        sys.exit(1)
    
    if not model_uri:
        logger.error("MODEL_URI environment variable not set")
        sys.exit(1)
    
    if not input_path:
        logger.error("INPUT_PATH environment variable not set")
        sys.exit(1)
    
    if not output_path:
        logger.error("OUTPUT_PATH environment variable not set")
        sys.exit(1)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
    
    try:
        # Load input CSV
        logger.info("Loading input CSV...")
        df = load_csv_from_gcs(input_path)
        
        # Run inference
        logger.info("Running inference...")
        df_with_predictions = predict(df, model_uri, metadata_path)
        
        # Save results
        logger.info("Saving predictions...")
        upload_csv_to_gcs(df_with_predictions, output_path)
        
        logger.info("✓ Inference job completed successfully!")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Inference job failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

