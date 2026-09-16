"""
Kubernetes Job Manager for Temporary Inference

Manages Kubernetes Jobs for running ONNX inference in isolated containers.
Uses the kubernetes Python client library for direct API access.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from kubernetes import config, client
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


class KubernetesJobManager:
    """Manages Kubernetes Jobs for inference workloads"""
    
    def __init__(self, 
                 namespace: str = None,
                 kubeconfig_path: str = None,
                 in_cluster: bool = None):
        """
        Initialize Kubernetes Job Manager.
        
        Args:
            namespace: Kubernetes namespace (default: from env or 'default')
            kubeconfig_path: Path to kubeconfig file (None = auto-detect)
            in_cluster: Whether running in-cluster (None = auto-detect)
        """
        self.namespace = namespace or os.getenv("K8S_NAMESPACE", "default")
        
        # Auto-detect if running in-cluster
        if in_cluster is None:
            in_cluster = os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token")
        
        try:
            if in_cluster:
                logger.info("Loading in-cluster Kubernetes config")
                config.load_incluster_config()
            else:
                logger.info(f"Loading Kubernetes config from: {kubeconfig_path or 'default location'}")
                if kubeconfig_path:
                    config.load_kube_config(config_file=kubeconfig_path)
                else:
                    config.load_kube_config()
            
            # Initialize API clients
            self.batch_api = client.BatchV1Api()
            self.core_api = client.CoreV1Api()
            
            # Ensure namespace exists (create if it doesn't)
            self._ensure_namespace_exists()
            
            logger.info(f"Kubernetes Job Manager initialized (namespace: {self.namespace})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    def _ensure_namespace_exists(self):
        """
        Ensure the namespace exists, create it if it doesn't.
        Skips creation for 'default' namespace (always exists).
        """
        if self.namespace == "default":
            # Default namespace always exists
            return
        
        try:
            # Check if namespace exists
            try:
                self.core_api.read_namespace(name=self.namespace)
                logger.info(f"Namespace '{self.namespace}' already exists")
                return
            except ApiException as e:
                if e.status == 404:
                    # Namespace doesn't exist, create it
                    logger.info(f"Namespace '{self.namespace}' not found, creating it...")
                    namespace_metadata = client.V1ObjectMeta(name=self.namespace)
                    namespace_body = client.V1Namespace(metadata=namespace_metadata)
                    
                    self.core_api.create_namespace(body=namespace_body)
                    logger.info(f"✓ Namespace '{self.namespace}' created successfully")
                else:
                    # Other error, log and re-raise
                    logger.error(f"Error checking namespace '{self.namespace}': {e}")
                    raise
        except Exception as e:
            logger.warning(f"Failed to ensure namespace '{self.namespace}' exists: {e}")
            logger.warning(f"Will attempt to use namespace anyway. If it fails, please create it manually:")
            logger.warning(f"  kubectl create namespace {self.namespace}")
            # Don't raise - let the job creation fail with a clearer error if namespace is required
    
    def create_inference_job(self,
                            job_name: str,
                            image: str,
                            model_uri: str,
                            input_path: str,
                            output_path: str,
                            mlflow_tracking_uri: str,
                            metadata_path: str = None,
                            resource_requests: Dict[str, str] = None,
                            resource_limits: Dict[str, str] = None,
                            service_account: str = None,
                            labels: Dict[str, str] = None,
                            gcs_bucket: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a Kubernetes Job for inference.
        
        Args:
            job_name: Unique name for the job
            image: Docker image for inference container
            model_uri: MLflow model URI (e.g., "runs:/run_42/model")
            input_path: GCS or local path to input CSV
            output_path: GCS or local path for output CSV
            mlflow_tracking_uri: MLflow tracking server URI
            metadata_path: Optional path to metadata JSON file
            resource_requests: Resource requests (e.g., {"cpu": "500m", "memory": "1Gi"})
            resource_limits: Resource limits (e.g., {"cpu": "2", "memory": "4Gi"})
            service_account: Service account for GCS access (optional)
            labels: Additional labels for the job
        
        Returns:
            Dictionary with job info: {"job_name": ..., "namespace": ..., "created": True/False}
        """
        try:
            # Set default resource requests/limits if not provided
            # Use very low defaults to improve schedulability in resource-constrained clusters
            # Inference jobs are typically CPU-light (mostly I/O bound for loading models and data)
            # Can be overridden via environment variables: INFERENCE_CPU_REQUEST, INFERENCE_MEMORY_REQUEST, etc.
            if not resource_requests:
                # Try to get from environment variables, otherwise use minimal defaults
                cpu_request = os.getenv("INFERENCE_CPU_REQUEST", "100m")
                memory_request = os.getenv("INFERENCE_MEMORY_REQUEST", "256Mi")
                resource_requests = {"cpu": cpu_request, "memory": memory_request}
            if not resource_limits:
                # Limits can be higher to allow burst performance when resources are available
                cpu_limit = os.getenv("INFERENCE_CPU_LIMIT", "2")
                memory_limit = os.getenv("INFERENCE_MEMORY_LIMIT", "2Gi")
                resource_limits = {"cpu": cpu_limit, "memory": memory_limit}
            
            # Prepare environment variables
            env_vars = [
                client.V1EnvVar(name="MLFLOW_TRACKING_URI", value=mlflow_tracking_uri),
                client.V1EnvVar(name="MODEL_URI", value=model_uri),
                client.V1EnvVar(name="INPUT_PATH", value=input_path),
                client.V1EnvVar(name="OUTPUT_PATH", value=output_path),
            ]
            
            # Extract and pass GCS bucket name - this is critical for the inference job
            # to construct full GCS URIs if the input_path doesn't have gs:// prefix
            # Priority: explicit parameter > extract from input_path > environment variables
            bucket_name = None
            if gcs_bucket:
                # Use explicitly provided bucket name (highest priority)
                bucket_name = gcs_bucket
            elif input_path.startswith("gs://"):
                # Extract bucket from full GCS URI
                bucket_name = input_path.replace("gs://", "").split("/")[0]
            elif os.getenv("GCS_BUCKET"):
                # Use environment variable if path is not a full URI
                bucket_name = os.getenv("GCS_BUCKET")
            elif os.getenv("GCS_INFERENCE_BUCKET"):
                # Try inference-specific bucket env var
                bucket_name = os.getenv("GCS_INFERENCE_BUCKET")
            
            # Always pass GCS_BUCKET if we have it - this helps inference job handle paths without gs:// prefix
            # CRITICAL: This is required for the inference job to construct full GCS URIs from partial paths
            if bucket_name:
                env_vars.append(client.V1EnvVar(name="GCS_BUCKET", value=bucket_name))
                logger.info(f"✓ Passing GCS_BUCKET={bucket_name} to inference job pod")
            else:
                # This is a critical error - the pod will fail if it receives a path without gs:// prefix
                error_msg = (
                    f"CRITICAL: GCS_BUCKET not available for inference job! "
                    f"Input path: '{input_path}'. "
                    f"gcs_bucket parameter: {gcs_bucket}, "
                    f"env GCS_BUCKET: {os.getenv('GCS_BUCKET')}, "
                    f"env GCS_INFERENCE_BUCKET: {os.getenv('GCS_INFERENCE_BUCKET')}. "
                    f"The pod will fail if the input_path doesn't have gs:// prefix."
                )
                logger.error(error_msg)
                # Don't raise here - let the pod fail with a clear error message
                # But log it as an error so it's visible
            
            if metadata_path:
                env_vars.append(client.V1EnvVar(name="METADATA_PATH", value=metadata_path))
            
            # Prepare resource requirements
            resources = client.V1ResourceRequirements(
                requests=resource_requests,
                limits=resource_limits
            )
            
            # Create container spec
            container = client.V1Container(
                name="inference",
                image=image,
                image_pull_policy="Always",
                env=env_vars,
                resources=resources
            )
            
            # Get service account from parameter, environment variable
            # Priority: parameter > GCS_WRITER > K8S_SERVICE_ACCOUNT > INFERENCE_SERVICE_ACCOUNT
            # With Workload Identity enabled, the service account should be annotated
            # to bind to a GCP service account with GCS permissions
            if not service_account:
                service_account = (
                    os.getenv("GCS_WRITER") or  # Check for gcs-writer first
                    os.getenv("K8S_SERVICE_ACCOUNT") or 
                    os.getenv("INFERENCE_SERVICE_ACCOUNT")
                )
            
            # Create pod template
            pod_spec = client.V1PodSpec(
                restart_policy="Never",
                containers=[container]
            )
            
            # Only set service_account_name if provided and it exists
            # If not provided or doesn't exist, Kubernetes will use the default service account for the namespace
            if service_account and service_account.strip():
                service_account = service_account.strip()
                # Check if service account exists in the namespace
                try:
                    self.core_api.read_namespaced_service_account(
                        name=service_account,
                        namespace=self.namespace
                    )
                    pod_spec.service_account_name = service_account
                    logger.info(f"Using Kubernetes service account: {service_account} in namespace {self.namespace}")
                except ApiException as e:
                    if e.status == 404:
                        logger.warning(
                            f"Service account '{service_account}' not found in namespace '{self.namespace}'. "
                            f"Using default service account instead. "
                            f"To use a custom service account, create it first: "
                            f"kubectl create serviceaccount {service_account} -n {self.namespace}"
                        )
                        # Don't set service_account_name - use default
                    else:
                        logger.warning(
                            f"Error checking service account '{service_account}': {e}. "
                            f"Using default service account instead."
                        )
                        # Don't set service_account_name - use default
            else:
                # No service account specified - use default
                logger.info(f"Using default service account for namespace {self.namespace} (no service account specified)")
            
            pod_template = client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={
                        "app": "avaloka-temp-inference",
                        "job-name": job_name,
                        **(labels or {})
                    }
                ),
                spec=pod_spec
            )
            
            # Create job spec
            # TTL (Time To Live) for automatic cleanup of completed/failed jobs
            # This requires the TTL controller to be enabled in Kubernetes (TTLAfterFinished feature gate)
            # If TTL controller is not enabled, jobs will not be automatically cleaned up
            ttl_env = os.getenv("INFERENCE_JOB_TTL_SECONDS", "600")
            logger.debug(f"Reading INFERENCE_JOB_TTL_SECONDS: '{ttl_env}' (type: {type(ttl_env)})")
            try:
                # Handle empty string or None
                if not ttl_env or ttl_env.strip() == "":
                    ttl_seconds = 600  # Default: 10 minutes
                    logger.debug("TTL env var is empty, using default: 600 seconds")
                else:
                    ttl_seconds = int(ttl_env)
                    logger.debug(f"TTL parsed successfully: {ttl_seconds} seconds")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid INFERENCE_JOB_TTL_SECONDS value: '{ttl_env}'. "
                    f"Using default: 600 seconds (10 minutes). Error: {e}"
                )
                ttl_seconds = 600  # Default: 10 minutes
            
            if ttl_seconds <= 0:
                # TTL of 0 or negative means no automatic cleanup
                logger.info("TTL disabled (INFERENCE_JOB_TTL_SECONDS <= 0), jobs will not be automatically cleaned up")
                job_spec = client.V1JobSpec(
                    template=pod_template,
                    backoff_limit=2  # Retry up to 2 times on failure
                )
            else:
                job_spec = client.V1JobSpec(
                    template=pod_template,
                    backoff_limit=2,  # Retry up to 2 times on failure
                    ttl_seconds_after_finished=ttl_seconds
                )
                logger.info(
                    f"Job TTL set to {ttl_seconds} seconds ({ttl_seconds // 60} minutes) - "
                    f"jobs will be automatically cleaned up after completion. "
                    f"(INFERENCE_JOB_TTL_SECONDS={ttl_env})"
                )
            
            # Create job metadata
            job_metadata = client.V1ObjectMeta(
                name=job_name,
                labels={
                    "app": "avaloka-temp-inference",
                    "created-by": "model-training-agent",
                    **(labels or {})
                }
            )
            
            # Create job
            job = client.V1Job(
                api_version="batch/v1",
                kind="Job",
                metadata=job_metadata,
                spec=job_spec
            )
            
            # Submit job to Kubernetes
            logger.info(f"Creating Kubernetes Job: {job_name} in namespace {self.namespace}")
            created_job = self.batch_api.create_namespaced_job(
                namespace=self.namespace,
                body=job
            )
            
            logger.info(f"✓ Job created successfully: {created_job.metadata.name}")
            
            return {
                "job_name": job_name,
                "namespace": self.namespace,
                "created": True,
                "uid": created_job.metadata.uid
            }
            
        except ApiException as e:
            logger.error(f"Failed to create Kubernetes Job: {e}")
            if e.status == 409:  # Conflict - job already exists
                logger.warning(f"Job {job_name} already exists")
                return {
                    "job_name": job_name,
                    "namespace": self.namespace,
                    "created": False,
                    "error": "Job already exists"
                }
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating Kubernetes Job: {e}")
            raise
    
    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """
        Get current status of a Kubernetes Job.
        
        Args:
            job_name: Name of the job
        
        Returns:
            Dictionary with job status information
        """
        try:
            job = self.batch_api.read_namespaced_job(
                name=job_name,
                namespace=self.namespace
            )
            
            status = job.status
            
            # Determine overall status
            if status.succeeded:
                job_status = "Succeeded"
            elif status.failed:
                job_status = "Failed"
            elif status.active:
                job_status = "Running"
            else:
                job_status = "Pending"
            
            return {
                "job_name": job_name,
                "namespace": self.namespace,
                "status": job_status,
                "active": status.active or 0,
                "succeeded": status.succeeded or 0,
                "failed": status.failed or 0,
                "start_time": status.start_time.isoformat() if status.start_time else None,
                "completion_time": status.completion_time.isoformat() if status.completion_time else None,
                "conditions": [
                    {
                        "type": c.type,
                        "status": c.status,
                        "reason": c.reason,
                        "message": c.message
                    } for c in (status.conditions or [])
                ]
            }
            
        except ApiException as e:
            if e.status == 404:
                return {
                    "job_name": job_name,
                    "namespace": self.namespace,
                    "status": "NotFound",
                    "error": "Job not found"
                }
            logger.error(f"Failed to get job status: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting job status: {e}")
            raise
    
    def wait_for_completion(self, 
                           job_name: str,
                           timeout: int = 3600,
                           poll_interval: int = 5) -> Dict[str, Any]:
        """
        Wait for a job to complete (succeed or fail).
        
        Args:
            job_name: Name of the job
            timeout: Maximum time to wait in seconds (default: 1 hour)
            poll_interval: Time between status checks in seconds (default: 5)
        
        Returns:
            Final job status dictionary
        """
        start_time = time.time()
        logger.info(f"Waiting for job {job_name} to complete (timeout: {timeout}s)")
        
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"Timeout waiting for job {job_name} after {elapsed:.0f}s")
                return {
                    "job_name": job_name,
                    "status": "Timeout",
                    "error": f"Job did not complete within {timeout} seconds"
                }
            
            # Get current status
            status = self.get_job_status(job_name)
            
            if status["status"] in ["Succeeded", "Failed", "NotFound"]:
                logger.info(f"Job {job_name} completed with status: {status['status']}")
                return status
            
            # Log progress
            if elapsed % 30 == 0:  # Log every 30 seconds
                logger.info(f"Job {job_name} still running... (elapsed: {elapsed:.0f}s)")
            
            # Wait before next check
            time.sleep(poll_interval)
    
    def get_job_logs(self, job_name: str, tail_lines: int = 100) -> str:
        """
        Get logs from the inference job pod.
        
        Args:
            job_name: Name of the job
            tail_lines: Number of lines to retrieve (default: 100)
        
        Returns:
            Log output as string
        """
        try:
            # Find pods for this job
            label_selector = f"job-name={job_name}"
            pods = self.core_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector
            )
            
            if not pods.items:
                return f"No pods found for job {job_name}"
            
            # Get logs from the first pod (should only be one for a Job)
            pod = pods.items[0]
            pod_name = pod.metadata.name
            
            logger.info(f"Retrieving logs from pod: {pod_name}")
            
            logs = self.core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                tail_lines=tail_lines
            )
            
            return logs
            
        except ApiException as e:
            if e.status == 404:
                return f"Pod not found for job {job_name}"
            logger.error(f"Failed to get job logs: {e}")
            return f"Error retrieving logs: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error getting job logs: {e}")
            return f"Error retrieving logs: {str(e)}"
    
    def delete_job(self, job_name: str, wait: bool = True) -> Dict[str, Any]:
        """
        Delete a Kubernetes Job.
        
        Args:
            job_name: Name of the job
            wait: Whether to wait for deletion to complete
        
        Returns:
            Dictionary with deletion status
        """
        try:
            # Delete job with propagation policy to also delete pods
            delete_options = client.V1DeleteOptions(
                propagation_policy="Background"
            )
            
            logger.info(f"Deleting Kubernetes Job: {job_name}")
            self.batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=delete_options
            )
            
            if wait:
                # Wait for deletion
                max_wait = 60  # 1 minute
                start_time = time.time()
                
                while time.time() - start_time < max_wait:
                    try:
                        self.batch_api.read_namespaced_job(
                            name=job_name,
                            namespace=self.namespace
                        )
                        time.sleep(2)
                    except ApiException as e:
                        if e.status == 404:
                            logger.info(f"✓ Job {job_name} deleted successfully")
                            return {
                                "job_name": job_name,
                                "deleted": True
                            }
                        raise
                
                logger.warning(f"Job {job_name} deletion timed out")
                return {
                    "job_name": job_name,
                    "deleted": False,
                    "warning": "Deletion timed out"
                }
            else:
                return {
                    "job_name": job_name,
                    "deleted": True
                }
            
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"Job {job_name} not found (may already be deleted)")
                return {
                    "job_name": job_name,
                    "deleted": False,
                    "error": "Job not found"
                }
            logger.error(f"Failed to delete job: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting job: {e}")
            raise
    
    def list_jobs(self, label_selector: str = None) -> List[Dict[str, Any]]:
        """
        List all inference jobs.
        
        Args:
            label_selector: Optional label selector (e.g., "app=avaloka-temp-inference")
        
        Returns:
            List of job information dictionaries
        """
        try:
            if not label_selector:
                label_selector = "app=avaloka-temp-inference"
            
            jobs = self.batch_api.list_namespaced_job(
                namespace=self.namespace,
                label_selector=label_selector
            )
            
            result = []
            for job in jobs.items:
                status = self.get_job_status(job.metadata.name)
                result.append({
                    "job_name": job.metadata.name,
                    "namespace": self.namespace,
                    "status": status["status"],
                    "created": job.metadata.creation_timestamp.isoformat() if job.metadata.creation_timestamp else None
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise

