"""
MLflow Integration for Model Training Agent

Provides experiment tracking, model logging, and artifact management
following the existing patterns in the Avaloka project.
"""

import logging
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import os
from app.utils import flatten_params
import tempfile
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Configure MLflow globally at module import time
def configure_mlflow_globally():
    """Configure MLflow tracking URI globally to use GCP Cloud SQL and GCS"""
    try:
        # ALWAYS force DISABLE_MLFLOW to False - override any external setting
        os.environ["DISABLE_MLFLOW"] = "false"
        disable_mlflow = "false"
        
        if disable_mlflow == "true":
            return
            
        # Check for GCP MLflow configuration
        gcp_tracking_uri = os.getenv("MLFLOW_BACKEND_STORE_URI")
        gcp_artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT")
        
        if gcp_tracking_uri and gcp_artifact_root:
            # Use GCP Cloud SQL and GCS
            mlflow.set_tracking_uri(gcp_tracking_uri)
            # Set the artifact root for GCS
            os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = gcp_artifact_root
            
            # Configure MLflow to use GCS for artifacts
            from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
            logger.info(f"MLflow configured globally to use GCP: {gcp_tracking_uri}")
            logger.info(f"Artifact root: {gcp_artifact_root}")
            
            # Safely set the experiment, handling deleted experiments
            try:
                # Check if experiment exists and if it's using GCS
                existing_exp = mlflow.get_experiment_by_name("avaloka_mta_gcs")
                if existing_exp:
                    # If experiment exists but doesn't use GCS, create a new one with GCS suffix
                    if gcp_artifact_root and not existing_exp.artifact_location.startswith('gs://'):
                        logger.warning(f"Existing experiment 'avaloka_mta_gcs' not using GCS. Creating new experiment with GCS...")
                        new_experiment_name = "avaloka_mta_gcs_v2"
                        try:
                            # Check if GCS version already exists
                            gcs_exp = mlflow.get_experiment_by_name(new_experiment_name)
                            if gcs_exp:
                                mlflow.set_experiment(new_experiment_name)
                                logger.info(f"Using existing GCS experiment: {new_experiment_name}")
                            else:
                                # Create new experiment with GCS
                                exp_id = mlflow.create_experiment(new_experiment_name, artifact_location=gcp_artifact_root)
                                mlflow.set_experiment(new_experiment_name)
                                logger.info(f"Created new GCS experiment '{new_experiment_name}' with ID: {exp_id}")
                        except Exception as create_error:
                            logger.error(f"Could not create GCS experiment: {create_error}")
                            # Fall back to existing experiment
                            mlflow.set_experiment("avaloka_mta_gcs")
                            logger.info(f"Using existing experiment: avaloka_mta_gcs")
                    else:
                        # Experiment exists and uses GCS (or no GCS configured)
                        mlflow.set_experiment("avaloka_mta_gcs")
                        logger.info(f"Experiment 'avaloka_mta_gcs' set as active")
                else:
                    # Experiment doesn't exist, create new one
                    if gcp_artifact_root:
                        exp_id = mlflow.create_experiment("avaloka_mta_gcs", artifact_location=gcp_artifact_root)
                        mlflow.set_experiment("avaloka_mta_gcs")
                        logger.info(f"Created new experiment 'avaloka_mta_gcs' with GCS location: {gcp_artifact_root}")
                    else:
                        exp_id = mlflow.create_experiment("avaloka_mta_gcs")
                        mlflow.set_experiment("avaloka_mta_gcs")
                        logger.info(f"Created new experiment 'avaloka_mta_gcs' with ID: {exp_id}")
                        
            except Exception as e:
                if "deleted experiment" in str(e).lower():
                    logger.warning(f"Experiment 'avaloka_mta_gcs' is deleted. Creating new experiment...")
                    try:
                        # Create a new experiment
                        exp_id = mlflow.create_experiment("avaloka_mta_gcs", artifact_location=gcp_artifact_root)
                        mlflow.set_experiment("avaloka_mta_gcs")
                        logger.info(f"Created new experiment 'avaloka_mta_gcs' with ID: {exp_id}")
                    except Exception as create_error:
                        logger.error(f"Could not create new experiment: {create_error}")
                        raise e
                else:
                    logger.error(f"Could not configure MLflow globally: {e}")
                    raise
        else:
            # Fallback to local file store
            tracking_dir = os.path.join(os.getcwd(), "mlflow_runs")
            os.makedirs(tracking_dir, exist_ok=True)
            mlflow.set_tracking_uri(f"file://{tracking_dir}")
            logger.info(f"MLflow configured globally to use local storage: {tracking_dir}")
    except Exception as e:
        logger.warning(f"Could not configure MLflow globally: {e}")

# Configure MLflow immediately when module is imported
configure_mlflow_globally()



class MLflowManager:
    """Manages MLflow experiment tracking for the MTA - Singleton pattern"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, experiment_name: str = "avaloka_mta", tracking_uri: Optional[str] = None):
        """
        Create or return the singleton instance
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (defaults to local file store)
        """
        if cls._instance is None:
            cls._instance = super(MLflowManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, experiment_name: str = "avaloka_mta", tracking_uri: Optional[str] = None):
        """Initialize only once (Singleton pattern)"""
        if self._initialized:
            return
            
        self._initialized = True
        self.experiment_name = experiment_name
        # Check if MLflow is disabled via environment variable
        disable_mlflow = os.getenv("DISABLE_MLFLOW", "false").lower()
        self.mlflow_enabled = disable_mlflow != "true"
        
        if not self.mlflow_enabled:
            logger.info("MLflow tracking disabled via DISABLE_MLFLOW environment variable")
            return
            
        try:
            # Set tracking URI
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                # Check for GCP configuration first
                gcp_tracking_uri = os.getenv("MLFLOW_BACKEND_STORE_URI")
                gcp_artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT")
                
                if gcp_tracking_uri and gcp_artifact_root:
                    # Use GCP Cloud SQL and GCS
                    mlflow.set_tracking_uri(gcp_tracking_uri)
                    # Set the artifact root for GCS
                    os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = gcp_artifact_root
                    logger.info(f"Using GCP MLflow: {gcp_tracking_uri}")
                    logger.info(f"Artifact root: {gcp_artifact_root}")
                    
                    # Ensure MLflow uses GCS for artifacts
                    from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
                else:
                    # Fallback to local file store
                    tracking_dir = os.path.join(os.getcwd(), "mlflow_runs")
                    os.makedirs(tracking_dir, exist_ok=True)
                    mlflow.set_tracking_uri(f"file://{tracking_dir}")
                    logger.info(f"Using local MLflow storage: {tracking_dir}")
            
            # Set or create experiment, handling deleted experiments
            try:
                # Get GCS artifact root for the experiment
                gcp_artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT")
                artifact_location = gcp_artifact_root if gcp_artifact_root else None
                
                # Check if experiment exists and if it's using GCS
                existing_exp = mlflow.get_experiment_by_name(experiment_name)
                if existing_exp:
                    # If experiment exists but doesn't use GCS, create a new one with GCS suffix
                    if artifact_location and not existing_exp.artifact_location.startswith('gs://'):
                        logger.warning(f"Existing experiment '{experiment_name}' not using GCS. Creating new experiment with GCS...")
                        new_experiment_name = f"{experiment_name}_gcs"
                        try:
                            # Check if GCS version already exists
                            gcs_exp = mlflow.get_experiment_by_name(new_experiment_name)
                            if gcs_exp:
                                mlflow.set_experiment(new_experiment_name)
                                logger.info(f"Using existing GCS experiment: {new_experiment_name}")
                            else:
                                # Create new experiment with GCS
                                exp_id = mlflow.create_experiment(new_experiment_name, artifact_location=artifact_location)
                                mlflow.set_experiment(new_experiment_name)
                                logger.info(f"Created new GCS experiment '{new_experiment_name}' with ID: {exp_id}")
                        except Exception as create_error:
                            logger.error(f"Could not create GCS experiment: {create_error}")
                            # Fall back to existing experiment
                            mlflow.set_experiment(experiment_name)
                            logger.info(f"Using existing experiment: {experiment_name}")
                    else:
                        # Experiment exists and uses GCS (or no GCS configured)
                        mlflow.set_experiment(experiment_name)
                        logger.info(f"Using existing experiment: {experiment_name}")
                else:
                    # Experiment doesn't exist, create new one
                    if artifact_location:
                        exp_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
                        mlflow.set_experiment(experiment_name)
                        logger.info(f"Created new experiment '{experiment_name}' with GCS location: {artifact_location}")
                    else:
                        exp_id = mlflow.create_experiment(experiment_name)
                        mlflow.set_experiment(experiment_name)
                        logger.info(f"Created new experiment '{experiment_name}' with ID: {exp_id}")
                    
            except Exception as exp_error:
                if "deleted experiment" in str(exp_error).lower():
                    logger.warning(f"Experiment '{experiment_name}' is deleted. Creating new experiment...")
                    try:
                        # Create a new experiment with GCS artifact location
                        exp_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
                        mlflow.set_experiment(experiment_name)
                        logger.info(f"Created new experiment '{experiment_name}' with ID: {exp_id} and GCS location: {artifact_location}")
                    except Exception as create_error:
                        logger.error(f"Could not create new experiment: {create_error}")
                        raise exp_error
                else:
                    raise exp_error
        except Exception as e:
            logger.warning(f"Could not initialize MLflow: {e}. Disabling MLflow tracking.")
            self.mlflow_enabled = False
    
    def start_training_run(self, task_id: str, tags: Optional[Dict[str, str]] = None, existing_run_id: Optional[str] = None) -> str:
        """
        Start a new MLflow run for training or continue an existing run
        
        Args:
            task_id: Unique identifier for the training task
            tags: Optional tags for the run
            existing_run_id: If provided, continue this existing run instead of starting new one
            
        Returns:
            MLflow run ID
        """
        if not self.mlflow_enabled:
            return f"mlflow_disabled_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        try:
            # If we have an existing run ID, use it
            if existing_run_id:
                logger.info(f"Continuing existing MLflow run: {existing_run_id}")
                return existing_run_id
            
            # Ensure no active runs before starting new one
            self.cleanup_all_runs()
            
            run_tags = {
                "task_id": task_id,
                "agent": "model_training_agent",
                "timestamp": datetime.now().isoformat(),
                "avaloka_version": "1.0.0"
            }
            
            if tags:
                run_tags.update(tags)
            
            # Start MLflow run
            run = mlflow.start_run(
                run_name=f"mta_task_{task_id}",
                tags=run_tags
            )
            
            logger.info(f"Started new MLflow run: {run.info.run_id}")
            return run.info.run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            # Return a dummy run ID if MLflow is not available
            return f"local_run_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def log_training_params(self, params: Dict[str, Any], run_id: Optional[str] = None):
        """
        Log training parameters to MLflow
        
        Args:
            params: Dictionary of parameters to log
            run_id: MLflow run ID (uses active run if None)
        """
        if not self.mlflow_enabled:
            logger.info(f"MLflow disabled, skipping parameter logging: {len(params)} parameters")
            return
            
        try:
            # Flatten nested parameters for MLflow
            flat_params = flatten_params(params)
            
            # Ensure we have a clean run context
            self.cleanup_all_runs()
            
            # Check if we're already in a run context
            if mlflow.active_run() is not None:
                # Already in a run, just log directly
                mlflow.log_params(flat_params)
            elif run_id:
                # Start specific run
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_params(flat_params)
            else:
                # Start new run
                with mlflow.start_run():
                    mlflow.log_params(flat_params)
                
            logger.info(f"Logged {len(flat_params)} parameters to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log parameters to MLflow: {e}")
            # Try to clean up and continue
            try:
                self.cleanup_all_runs()
            except:
                pass
    
    def log_training_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, run_id: Optional[str] = None):
        """
        Log training metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time series metrics
            run_id: MLflow run ID (uses active run if None)
        """
        if not self.mlflow_enabled:
            logger.info(f"MLflow disabled, skipping metrics logging: {len(metrics)} metrics")
            return
            
        try:
            # Ensure we have a clean run context
            self.cleanup_all_runs()
            
            # Check if we're already in a run context
            if mlflow.active_run() is not None:
                # Already in a run, just log directly
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value, step=step)
            elif run_id:
                # Start specific run
                with mlflow.start_run(run_id=run_id):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(metric_name, value, step=step)
            else:
                # Start new run
                with mlflow.start_run():
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(metric_name, value, step=step)
            
            logger.info(f"Logged {len(metrics)} metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")
            # Try to clean up and continue
            try:
                self.cleanup_all_runs()
            except:
                pass
    
    def log_model_artifact(self, model_path: str, model_type: str = "pytorch", run_id: Optional[str] = None) -> str:
        """
        Log model artifact to MLflow
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (pytorch, sklearn, etc.)
            run_id: MLflow run ID (uses active run if None)
            
        Returns:
            Artifact URI
        """
        if not self.mlflow_enabled:
            logger.info(f"MLflow disabled, skipping model artifact logging: {model_path}")
            return f"local_artifact_{model_path}"
            
        try:
            # Ensure GCP artifact root is set
            gcp_artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT")
            if gcp_artifact_root:
                os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = gcp_artifact_root
                logger.info(f"Using GCP artifact root for model logging: {gcp_artifact_root}")
                
                # Get the current run ID
                if run_id:
                    current_run_id = run_id
                else:
                    current_run = mlflow.active_run()
                    if not current_run:
                        raise Exception("No active MLflow run")
                    current_run_id = current_run.info.run_id
                
                # Use MLflow's built-in artifact logging with GCS
                # Check if we're already in a run context
                if mlflow.active_run() and mlflow.active_run().info.run_id == current_run_id:
                    # Already in the correct run, just log directly
                    artifact_uri = mlflow.log_artifact(model_path, "model")
                    logger.info(f"Model logged to GCS via MLflow: {artifact_uri}")
                else:
                    # Start the run context
                    with mlflow.start_run(run_id=current_run_id):
                        # Log the artifact using MLflow's standard method
                        artifact_uri = mlflow.log_artifact(model_path, "model")
                        logger.info(f"Model logged to GCS via MLflow: {artifact_uri}")
                
                return f"runs:/{current_run_id}/model"
            else:
                # Fallback to local logging
                if run_id:
                    with mlflow.start_run(run_id=run_id):
                        artifact_uri = self._log_model_by_type(model_path, model_type)
                else:
                    artifact_uri = self._log_model_by_type(model_path, model_type)
                
                logger.info(f"Logged model artifact: {artifact_uri}")
                return artifact_uri
            
        except Exception as e:
            logger.error(f"Failed to log model artifact: {e}")
            return f"local://{model_path}"
    
    def log_training_code(self, code: str, run_id: Optional[str] = None):
        """
        Log training code as an artifact
        
        Args:
            code: Python code used for training
            run_id: MLflow run ID (uses active run if None)
        """
        if not self.mlflow_enabled:
            logger.info("MLflow disabled, skipping training code logging")
            return
            
        try:
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(temp_path, "code")
            else:
                mlflow.log_artifact(temp_path, "code")
            
            # Clean up temp file
            os.unlink(temp_path)
            
            logger.info("Logged training code to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log training code: {e}")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any], run_id: Optional[str] = None):
        """
        Log dataset information as parameters and artifacts
        
        Args:
            dataset_info: Information about the dataset
            run_id: MLflow run ID (uses active run if None)
        """
        if not self.mlflow_enabled:
            logger.info("MLflow disabled, skipping dataset info logging")
            return
            
        try:
            # Log dataset info as JSON artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(dataset_info, f, indent=2)
                temp_path = f.name
            
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(temp_path, "dataset")
                    # Also log key dataset parameters
                    if "num_samples" in dataset_info:
                        mlflow.log_param("dataset_num_samples", dataset_info["num_samples"])
                    if "num_features" in dataset_info:
                        mlflow.log_param("dataset_num_features", dataset_info["num_features"])
            else:
                mlflow.log_artifact(temp_path, "dataset")
                if "num_samples" in dataset_info:
                    mlflow.log_param("dataset_num_samples", dataset_info["num_samples"])
                if "num_features" in dataset_info:
                    mlflow.log_param("dataset_num_features", dataset_info["num_features"])
            
            # Clean up temp file
            os.unlink(temp_path)
            
            logger.info("Logged dataset information to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log dataset info: {e}")
    
    def end_run(self, status: str = "FINISHED", run_id: Optional[str] = None):
        """
        End the current MLflow run
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
            run_id: MLflow run ID (ends active run if None)
        """
        if not self.mlflow_enabled:
            logger.info(f"MLflow disabled, skipping run end: {status}")
            return
            
        try:
            if run_id:
                # End specific run
                client = mlflow.tracking.MlflowClient()
                client.set_terminated(run_id, status)
            else:
                # End active run
                mlflow.end_run(status)
            
            logger.info(f"Ended MLflow run with status: {status}")
            
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    def ensure_no_active_run(self):
        """
        Ensure no MLflow run is active by ending any active run
        """
        try:
            if mlflow.active_run() is not None:
                logger.info("Ending active MLflow run")
                mlflow.end_run()
        except Exception as e:
            logger.warning(f"Could not end active run: {e}")
    
    def cleanup_all_runs(self):
        """
        Clean up all active MLflow runs
        """
        try:
            # End any active run multiple times to ensure cleanup
            while mlflow.active_run() is not None:
                logger.info("Cleaning up active MLflow run")
                mlflow.end_run()
                # Small delay to ensure cleanup
                import time
                time.sleep(0.1)
        except Exception as e:
            logger.warning(f"Could not clean up MLflow runs: {e}")
    
    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """
        Get information about a specific run
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Run information dictionary
        """
        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            
            return {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "tags": dict(run.data.tags)
            }
            
        except Exception as e:
            logger.error(f"Failed to get run info: {e}")
            return {"error": str(e)}
    
    
    def _log_model_by_type(self, model_path: str, model_type: str) -> str:
        """Log model based on its type"""
        try:
            # Get the current run
            current_run = mlflow.active_run()
            if not current_run:
                raise Exception("No active MLflow run")
            
            # Determine artifact directory based on model type
            if model_type.lower() == "onnx":
                artifact_dir = "onnx_model"
            elif model_type.lower() == "pytorch":
                artifact_dir = "pytorch_model"
            else:
                artifact_dir = "model"
            
            # Log the artifact (we're already in a run context)
            mlflow.log_artifact(model_path, artifact_dir)
            
            # Return the artifact URI - this will be GCS if properly configured
            artifact_uri = f"runs:/{current_run.info.run_id}/{artifact_dir}"
            
            # Log additional info about where the artifact is stored
            gcp_artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT")
            if gcp_artifact_root and gcp_artifact_root.startswith("gs://"):
                logger.info(f"{model_type.upper()} model logged to GCS: {gcp_artifact_root}/{current_run.info.run_id}/artifacts/{artifact_dir}/")
            else:
                logger.info(f"{model_type.upper()} model logged locally: {artifact_uri}")
            
            return artifact_uri
            
        except Exception as e:
            logger.error(f"Failed to log {model_type} model artifact: {e}")
            return f"local://{model_path}"

    def get_all_run_infos(self):
        """Get a list of all run information dictionaries in the current experiment"""
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            if not experiment:
                raise Exception(f"Experiment '{self.experiment_name}' not found")
            
            runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
            run_infos = []
            for run in runs:
                run_infos.append({
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params),
                    "tags": dict(run.data.tags)
                })
            return {
                "run_infos": run_infos
            }
        except Exception as e:
            logger.error(f"Failed to get all run infos: {e}")
            return {"error": str(e)}

    def get_run_model_artifact_details(self, run_id: str):
        """Get details about model artifacts logged in a specific run"""
        client = mlflow.tracking.MlflowClient()
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
            generic_metadata = None
            
            for artifact in artifacts:
                artifact_path = artifact.path
                logger.debug(f"Found artifact: {artifact_path}")
                
                # Skip generic metadata.json - we prefer model-specific metadata
                if artifact_path == "metadata.json" or artifact_path == "model/metadata.json":
                    generic_metadata = artifact_path
                    break
            
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
                                        if file == "metadata.json":
                                            downloaded_file = os.path.join(root, file)
                                            break
                                    if downloaded_file:
                                        break
                            
                            if downloaded_file and os.path.exists(downloaded_file):
                                # Read the file content before temp_dir is deleted
                                with open(downloaded_file, "r") as f:
                                    metadata_content: dict = json.load(f)
                                logger.info(f"Successfully downloaded and loaded metadata from: {download_path}")
                                break
                            else:
                                logger.debug(f"Downloaded file not found in temp directory for: {download_path}")
                    except Exception as e:
                        logger.debug(f"Failed to download from {download_path}: {e}")
                        continue
                
                if metadata_content:
                    params = client.get_run(run_id).data.params
                    metadata_content.update(params)
                    metadata_content["run_id"] = run_id
                    return metadata_content
                else:
                    logger.warning(f"Failed to download metadata artifact from any path: {download_paths}")
        
        logger.warning("No suitable metadata file found in artifacts")
        return {
            "error": "No metadata file found in artifacts"
        }

    def get_run_artifact_details(self, run_id: str):
        """Get details about artifacts logged in a specific run"""
        client = mlflow.tracking.MlflowClient()
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
                                    metadata_content: dict = json.load(f)
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
        return {
            "error": "No metadata file found in artifacts"
        }
    
    def hard_reset_mlflow_storage(self):
        """
        🚨 DANGER: Completely deletes ALL MLflow data.
        
        - Drops and recreates PostgreSQL database
        - Deletes all artifacts from GCS bucket
        
        Requires:
            - psycopg2-binary
            - google-cloud-storage
            - GOOGLE_APPLICATION_CREDENTIALS set
        """

        import psycopg2
        from google.cloud import storage
        from urllib.parse import urlparse

        logger.warning("⚠️ HARD RESET OF MLFLOW STORAGE INITIATED")

        # ==============================
        # 1️⃣ RESET POSTGRES BACKEND
        # ==============================
        try:
            backend_uri = os.getenv("MLFLOW_BACKEND_STORE_URI")
            if not backend_uri:
                raise Exception("MLFLOW_BACKEND_STORE_URI not set")

            parsed = urlparse(backend_uri)

            db_name = parsed.path.lstrip("/")
            db_user = parsed.username
            db_password = parsed.password
            db_host = parsed.hostname
            db_port = parsed.port or 5432

            logger.info(f"Connecting to PostgreSQL at {db_host}:{db_port}")

            conn = psycopg2.connect(
                dbname="postgres",
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port,
            )
            conn.autocommit = True
            cursor = conn.cursor()

            # Terminate active connections
            cursor.execute(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{db_name}';
            """)

            logger.info(f"Dropping database: {db_name}")
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name};")

            logger.info(f"Recreating database: {db_name}")
            cursor.execute(f"CREATE DATABASE {db_name};")

            cursor.close()
            conn.close()

            logger.info("PostgreSQL backend reset complete.")

        except Exception as e:
            logger.error(f"Failed to reset PostgreSQL: {e}")
            raise

        # ==============================
        # 2️⃣ CLEAR GCS ARTIFACTS
        # ==============================
        try:
            artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT")
            if not artifact_root or not artifact_root.startswith("gs://"):
                raise Exception("Invalid GCS artifact root")

            bucket_name = artifact_root.replace("gs://", "").split("/")[0]

            logger.info(f"Connecting to GCS bucket: {bucket_name}")

            client = storage.Client()
            bucket = client.bucket(bucket_name)

            blobs = list(bucket.list_blobs())
            logger.info(f"Deleting {len(blobs)} artifact objects...")

            for blob in blobs:
                blob.delete()

            logger.info("GCS artifacts cleared.")

        except Exception as e:
            logger.error(f"Failed to clear GCS bucket: {e}")
            raise

        logger.warning("🔥 MLflow HARD RESET COMPLETE.")

    def delete_all_runs(self):
        """Delete all runs in the current experiment"""
        if not self.mlflow_enabled:
            logger.info("MLflow disabled, skipping run deletion")
            return
            
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            if not experiment:
                logger.warning(f"Experiment '{self.experiment_name}' not found, cannot delete runs")
                return
            
            runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
            for run in runs:
                client.delete_run(run.info.run_id)
                logger.info(f"Deleted MLflow run: {run.info.run_id}")
        except Exception as e:
            logger.error(f"Failed to delete MLflow runs: {e}")

    def delete_run(self, run_id: str):
        """Delete a specific MLflow run by ID"""
        if not self.mlflow_enabled:
            logger.info(f"MLflow disabled, skipping run deletion: {run_id}")
            return
            
        try:
            client = mlflow.tracking.MlflowClient()
            client.delete_run(run_id)
            logger.info(f"Deleted MLflow run: {run_id}")
        except Exception as e:
            logger.error(f"Failed to delete MLflow run {run_id}: {e}")

# Singleton factory function for getting MLflow manager
def get_mlflow_manager(experiment_name: str = "avaloka_mta", tracking_uri: Optional[str] = None) -> MLflowManager:
    """
    Get the singleton MLflow manager instance
    
    Args:
        experiment_name: Name of the experiment (only used on first creation)
        tracking_uri: MLflow tracking URI (only used on first creation)
        
    Returns:
        Singleton MLflowManager instance
    """
    return MLflowManager(experiment_name=experiment_name, tracking_uri=tracking_uri)

# Backward compatibility alias
def create_mlflow_manager(config: Dict[str, Any]) -> MLflowManager:
    """
    Create MLflow manager from configuration (backward compatibility)
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MLflowManager singleton instance
    """
    experiment_name = config.get("experiment_name", "avaloka_mta")
    tracking_uri = config.get("mlflow_tracking_uri")
    
    return get_mlflow_manager(experiment_name=experiment_name, tracking_uri=tracking_uri)


# Utility functions
def is_mlflow_available() -> bool:
    """Check if MLflow is available and properly configured"""
    try:
        mlflow.get_tracking_uri()
        return True
    except Exception:
        return False

