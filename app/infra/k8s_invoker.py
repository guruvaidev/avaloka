# app/infra/k8s_invoker.py
import subprocess
import os
import time
import boto3
from botocore.exceptions import ClientError
from kubernetes import config, client
from app.core import cloud_config

def _probe(command: list, **kwargs) -> subprocess.CompletedProcess:
    """Run a read-only pre-flight check that must never raise.

    The probe sites below all branch on returncode/stdout, and every one of
    them shells out to a CLI that may simply not be installed (gcloud, aws,
    kubectl, helm). A bare subprocess.run raises FileNotFoundError in that
    case, and since these run inside a LangGraph node the exception tears down
    the entire graph invocation instead of producing a FAILED step the caller
    can report. Return a synthetic non-zero result instead, matching how
    _run_command already treats a missing binary.
    """
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("shell", False)
    try:
        return subprocess.run(command, **kwargs)
    except FileNotFoundError as e:
        return subprocess.CompletedProcess(
            args=command,
            returncode=127,
            stdout="",
            stderr=f"Command '{command[0]}' not found. Ensure it's installed and in PATH. ({e})",
        )


def _run_command(command: list, step_name: str, env: dict | None = None) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, shell=False, env=env)
        print ("----------")
        print (command)
        print (result)
        print ("----------")
        return {
            "step_name": step_name,
            "status": "SUCCESS",
            "message": f"Command '{' '.join(command)}' executed successfully.",
            "details": result.stdout.strip()
        }
    except subprocess.CalledProcessError as e:
        return {
            "step_name": step_name,
            "status": "FAILED",
            "message": f"Command '{' '.join(command)}' failed.",
            "details": f"Stdout: {e.stdout.strip()}\nStderr: {e.stderr.strip()}"
        }
    except FileNotFoundError:
        return {
            "step_name": step_name,
            "status": "FAILED",
            "message": f"Command '{command[0]}' not found. Ensure it's installed and in PATH.",
            "details": ""
        }

def _kubectl() -> str:
    return os.getenv("KUBECTL_PATH", "kubectl")

def _helm() -> str:
    return os.getenv("HELM_PATH", "helm")

def provision_cluster(platform: str) -> dict:
    """Provisions a K8s cluster (GKE or EKS) or skips if it exists."""
    print(f"Attempting to provision {platform} K8s cluster...")
    
    if platform == "gcp":
        project_id = os.environ.get("GCP_PROJECT_ID")
        cluster_name = os.environ.get("GCP_CLUSTER_NAME", "test-gke-cluster")
        zone = os.environ.get("GCP_ZONE", "us-central1-c")

        if not project_id:
            return {
                "step_name": f"Provision {platform} Cluster",
                "status": "FAILED",
                "message": "GCP_PROJECT_ID environment variable not set.",
                "details": ""
            }

        # Check if the cluster already exists
        check_cmd = ["gcloud", "container", "clusters", "describe", cluster_name, "--project", project_id, "--zone", zone]
        check_result = _probe(check_cmd)
        if check_result.returncode == 0:
            print(f"GKE cluster {cluster_name} already exists. Skipping creation.")
            return {
                "step_name": f"Provision {platform} Cluster",
                "status": "SKIPPED",
                "message": f"GKE cluster {cluster_name} already exists.",
                "details": check_result.stdout.strip()
            }
        else:
            print(f"Creating GKE cluster {cluster_name}...")
            create_cmd = [
                "gcloud", "container", "clusters", "create", cluster_name,
                "--project", project_id,
                "--zone", zone,
                "--num-nodes", "1"
            ]
            return _run_command(create_cmd, f"Create {platform} Cluster")

    elif platform == "aws":
        aws_region = os.environ.get("AWS_REGION")
        eks_cluster_role_arn = os.environ.get("AWS_EKS_CLUSTER_ROLE_ARN")
        aws_subnet_ids = os.environ.get("AWS_SUBNET_IDS")
        aws_security_group_ids = os.environ.get("AWS_SECURITY_GROUP_IDS")
        cluster_name = os.environ.get("AWS_CLUSTER_NAME", "test-eks-cluster")

        if not all([aws_region, eks_cluster_role_arn, aws_subnet_ids, aws_security_group_ids]):
            return {
                "step_name": f"Provision {platform} Cluster",
                "status": "FAILED",
                "message": "Missing AWS_REGION, AWS_EKS_CLUSTER_ROLE_ARN, AWS_SUBNET_IDS, or AWS_SECURITY_GROUP_IDS environment variables.",
                "details": ""
            }
        
        # Assume the IAM role. The temporary credentials go only into the env of
        # the aws CLI children below, never os.environ: that made them
        # process-wide for every later request (NEW-RS-02).
        try:
            sts_client = boto3.client('sts', region_name=aws_region)
            assumed_role_object = sts_client.assume_role(
                RoleArn=eks_cluster_role_arn,
                RoleSessionName="EKSClusterProvisionSession"
            )
            credentials = assumed_role_object['Credentials']
            aws_env = {
                **os.environ,
                'AWS_ACCESS_KEY_ID': credentials['AccessKeyId'],
                'AWS_SECRET_ACCESS_KEY': credentials['SecretAccessKey'],
                'AWS_SESSION_TOKEN': credentials['SessionToken'],
                'AWS_DEFAULT_REGION': aws_region,
            }
        except ClientError as e:
            return {
                "step_name": f"Assume IAM Role for {platform}",
                "status": "FAILED",
                "message": f"Failed to assume IAM role: {e}",
                "details": str(e)
            }

        # Check if the EKS cluster already exists
        check_cmd = ["aws", "eks", "describe-cluster", "--name", cluster_name, "--region", aws_region]
        check_result = _probe(check_cmd, env=aws_env)

        if check_result.returncode == 0:
            print(f"EKS cluster {cluster_name} already exists. Skipping creation.")
            return {
                "step_name": f"Provision {platform} Cluster",
                "status": "SKIPPED",
                "message": f"EKS cluster {cluster_name} already exists.",
                "details": check_result.stdout.strip()
            }
        else:
            print(f"Creating EKS cluster {cluster_name}...")
            create_cmd = [
                "aws", "eks", "create-cluster",
                "--name", cluster_name,
                "--region", aws_region,
                "--version", "1.28", # Specify a Kubernetes version
                "--role-arn", eks_cluster_role_arn,
                "--resources-vpc-config", f"subnetIds={aws_subnet_ids},securityGroupIds={aws_security_group_ids}"
            ]
            create_outcome = _run_command(create_cmd, f"Create {platform} Cluster", env=aws_env)
            if create_outcome["status"] == "FAILED":
                return create_outcome
            
            # Wait for cluster to become active
            wait_cmd = ["aws", "eks", "wait", "cluster-active", "--name", cluster_name, "--region", aws_region]
            return _run_command(wait_cmd, f"Wait for {platform} Cluster Active", env=aws_env)
    else:
        return {
            "step_name": f"Provision Cluster",
            "status": "FAILED",
            "message": f"Unsupported platform: {platform}",
            "details": "Only 'gcp' and 'aws' are supported."
        }

def configure_kubectl(platform: str) -> dict:
    """Configures kubectl for the provisioned cluster."""
    print(f"Configuring kubectl for {platform} cluster...")
    if platform == "gcp":
        project_id = os.environ.get("GCP_PROJECT_ID")
        cluster_name = os.environ.get("GCP_CLUSTER_NAME", "test-gke-cluster")
        zone = os.environ.get("GCP_ZONE", "us-central1-c")
        cmd = [
            "gcloud", "container", "clusters", "get-credentials", cluster_name,
            "--project", project_id,
            "--zone", zone
        ]
    elif platform == "aws":
        aws_region = os.environ.get("AWS_REGION")
        cluster_name = os.environ.get("AWS_CLUSTER_NAME", "test-eks-cluster")
        cmd = [
            "aws", "eks", "update-kubeconfig",
            "--name", cluster_name,
            "--region", aws_region
        ]
        role_arn = os.environ.get("AWS_EKS_CLUSTER_ROLE_ARN")
        if role_arn:
            # The cluster's creator is the assumed role, whose credentials no
            # longer sit in os.environ; the kubeconfig's get-token assumes it.
            cmd += ["--role-arn", role_arn]
    else:
        return {
            "step_name": "Configure Kubectl",
            "status": "FAILED",
            "message": f"Unsupported platform: {platform}",
            "details": ""
        }
    
    outcome = _run_command(cmd, "Configure Kubectl")
    if outcome["status"] == "SUCCESS":
        # Wait for kubectl to be able to connect to the cluster
        for _ in range(60): # 5 minutes timeout
            try:
                subprocess.run(["kubectl", "get", "nodes"], check=True, capture_output=True, shell=False)
                outcome["details"] += "\nkubectl connected to cluster."
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                time.sleep(5)
        else:
            outcome["status"] = "FAILED"
            outcome["message"] = "kubectl could not connect to the cluster after multiple retries."
            outcome["details"] = "Timed out waiting for kubectl to connect."
    return outcome

def run_cluster_health_checks() -> dict:
    """Runs basic K8s cluster health checks."""
    print("Running K8s cluster health checks...")
    # Using kubectl get nodes to check connectivity and node status
    cmd = ["kubectl", "get", "nodes"]
    outcome = _run_command(cmd, "Cluster Health Checks")
    if outcome["status"] == "SUCCESS" and " Ready" in outcome["details"]:
        outcome["message"] = "Cluster health checks passed. Nodes are ready."
    elif outcome["status"] == "SUCCESS":
        outcome["status"] = "WARNING"
        outcome["message"] = "Cluster health checks completed, but not all nodes are 'Ready'."
    return outcome


def ensure_namespace(namespace: str) -> dict:
    """Ensure the namespace exists (idempotent)."""
    if not namespace:
        return {
            "step_name": "Ensure Namespace",
            "status": "FAILED",
            "message": "Namespace not provided.",
            "details": ""
        }

    # Check if namespace exists
    check_cmd = ["kubectl", "get", "namespace", namespace]
    check = _probe(check_cmd)
    if check.returncode == 0:
        return {
            "step_name": f"Ensure Namespace {namespace}",
            "status": "SKIPPED",
            "message": f"Namespace '{namespace}' already exists.",
            "details": check.stdout.strip()
        }

    # Create namespace
    create_cmd = ["kubectl", "create", "namespace", namespace]
    return _run_command(create_cmd, f"Create Namespace {namespace}")


def kuberay_crds_installed() -> bool:
    """Best-effort check: RayJob CRD present."""
    cmd = ["kubectl", "get", "crd", "rayjobs.ray.io"]
    r = _probe(cmd)
    return r.returncode == 0

# def _operator_running(ns: str, deployment: str) -> bool:
#     # best-effort: deployment exists and has available replicas
#     r = subprocess.run(
#         [_kubectl(), "-n", ns, "get", "deployment", deployment, "-o",
#          "jsonpath={.status.availableReplicas}"],
#         capture_output=True, text=True, shell=False
#     )
#     if r.returncode != 0:
#         return False
#     try:
#         return int((r.stdout or "0").strip() or "0") > 0
#     except Exception:
#         return False


def _operator_running(ns: str, deployment: str) -> bool:
    # Check provided namespace first, then kuberay-system (common alternative)
    for check_ns in [ns, "kuberay-system"]:
        r = _probe(
            [_kubectl(), "-n", check_ns, "get", "deployment", deployment, "-o",
             "jsonpath={.status.availableReplicas}"],
        )
        if r.returncode == 0:
            try:
                if int((r.stdout or "0").strip() or "0") > 0:
                    return True
            except Exception:
                pass
    return False

def _version_tuple(v: str) -> tuple:
    import re as _re
    return tuple(int(x) for x in _re.findall(r"\d+", (v or "").split("-")[0])[:3])


def _installed_kuberay_version(ns: str, release: str):
    """The kuberay-operator chart version Helm installed as ``release``, or None."""
    import json as _json
    r = _probe([_helm(), "list", "-n", ns, "-o", "json", "--filter", f"^{release}$"])
    if r.returncode != 0:
        return None
    try:
        rows = _json.loads(r.stdout or "[]")
    except ValueError:
        return None
    for row in rows:
        chart = str(row.get("chart") or "")
        if chart.startswith("kuberay-operator-"):
            return chart[len("kuberay-operator-"):]
    return None


def ensure_kuberay_installed() -> dict:
    """
    Ensure KubeRay operator/CRDs exist.
    Priority:
      1) If CRD + operator ready -> SKIPPED
      2) If KUBERAY_MANIFEST_PATH exists -> kubectl apply
      3) else -> helm repo add/update + helm upgrade --install + wait
    """
    ns = os.environ.get("KUBERAY_OPERATOR_NAMESPACE", "kuberay-operator")
    release = os.environ.get("KUBERAY_HELM_RELEASE", "kuberay-operator")
    chart = os.environ.get("KUBERAY_HELM_CHART", "kuberay/kuberay-operator")

    from app.infra.ray_manager import KUBERAY_OPERATOR_VERSION, upgrade_kuberay_crds

    # If CRDs exist AND operator seems running, skip -- unless Helm installed an
    # operator older than the one Ray token auth needs (ray_manager), which
    # would be refused by every token-authenticated dashboard it polls.
    if kuberay_crds_installed() and _operator_running(ns, release):
        installed = _installed_kuberay_version(ns, release)
        if installed is None or _version_tuple(installed) >= _version_tuple(KUBERAY_OPERATOR_VERSION):
            return {
                "step_name": "Ensure KubeRay Installed",
                "status": "SKIPPED",
                "message": "KubeRay already installed (CRD present + operator running).",
                "details": ""
            }

    # Preferred: apply local manifest if provided
    manifest_path = os.environ.get("KUBERAY_MANIFEST_PATH")
    if manifest_path and os.path.exists(manifest_path):
        outcome = _run_command([_kubectl(), "apply", "-f", manifest_path], "Install KubeRay (kubectl apply)")
        if outcome["status"] != "FAILED":
            _run_command([_kubectl(), "wait", "--for=condition=Established", "crd/rayjobs.ray.io", "--timeout=120s"],
                         "Wait for RayJob CRD")
        # final check
        ok = kuberay_crds_installed()
        return {
            "step_name": "Ensure KubeRay Installed",
            "status": "SUCCESS" if ok else "FAILED",
            "message": "KubeRay installed via manifest." if ok else "Applied manifest but CRD rayjobs.ray.io not found.",
            "details": outcome.get("details", "")
        }

    # Helm fallback
    helm_check = _probe([_helm(), "version"])
    if helm_check.returncode != 0:
        return {
            "step_name": "Ensure KubeRay Installed",
            "status": "FAILED",
            "message": "KubeRay not installed and helm not available.",
            "details": "Provide KUBERAY_MANIFEST_PATH (preferred) or install helm."
        }

    # Add repo (Ray docs)
    repo_add = _probe(
        [_helm(), "repo", "add", "kuberay", "https://ray-project.github.io/kuberay-helm/"],
    )
    if repo_add.returncode != 0 and "already exists" not in (repo_add.stderr or "").lower():
        return {
            "step_name": "Ensure KubeRay Installed",
            "status": "FAILED",
            "message": "Failed to add kuberay helm repo.",
            "details": f"{repo_add.stdout}\n{repo_add.stderr}"
        }

    _run_command([_helm(), "repo", "update"], "Helm repo update")

    crds = upgrade_kuberay_crds()
    if crds["status"] == "FAILED":
        return crds
    install = _run_command(
        [_helm(), "upgrade", "--install", release, chart, "--version", KUBERAY_OPERATOR_VERSION,
         "-n", ns, "--create-namespace", "--wait", "--timeout", "180s"],
        "Install KubeRay (helm)"
    )
    if install["status"] == "FAILED":
        return install

    # Wait for CRD and operator rollout
    _run_command([_kubectl(), "wait", "--for=condition=Established", "crd/rayjobs.ray.io", "--timeout=120s"],
                 "Wait for RayJob CRD")
    _run_command([_kubectl(), "-n", ns, "rollout", "status", f"deployment/{release}", "--timeout=180s"],
                 "Wait for KubeRay operator rollout")

    ok = kuberay_crds_installed()
    return {
        "step_name": "Ensure KubeRay Installed",
        "status": "SUCCESS" if ok else "FAILED",
        "message": "KubeRay installed and validated." if ok else "KubeRay install ran, but CRD not found.",
        "details": install.get("details", "")
    }


def deploy_application(app_type: str, platform: str) -> dict:
    """Deploys the specified application type to the K8s cluster."""
    print(f"Deploying {app_type} application...")
    if app_type == "python-docker":
        # This used to docker-build app/infra/python_app, push it to
        # gcr.io/<project>/infra-agent-python-app and kubectl-apply a manifest
        # that exposed its unauthenticated /execute through a LoadBalancer with
        # the default ServiceAccount token mounted. The hardened service ships
        # with the Helm chart instead; nothing is built or applied at runtime.
        return locate_execute_service()
    elif app_type == "spark":
        # Placeholder for Spark deployment logic
        # This would typically involve installing a Spark operator or deploying Spark via Helm.
        # Example: helm install spark-operator stable/spark-operator
        # Or kubectl apply -f spark-application.yaml
        return {
            "step_name": "Deploy Spark Cluster",
            "status": "SKIPPED", # Change to SUCCESS/FAILED based on actual implementation
            "message": "Spark deployment is a placeholder. Implement actual Helm/kubectl commands here.",
            "details": ""
        }
    elif app_type == "kafka":
        # Placeholder for Kafka deployment logic
        # This would typically involve installing a Kafka operator (e.g., Strimzi) or deploying Kafka via Helm.
        # Example: helm install kafka bitnami/kafka
        return {
            "step_name": "Deploy Kafka Cluster",
            "status": "SKIPPED", # Change to SUCCESS/FAILED based on actual implementation
            "message": "Kafka deployment is a placeholder. Implement actual Helm/kubectl commands here.",
            "details": ""
        }
    else:
        return {
            "step_name": "Deploy Application",
            "status": "FAILED",
            "message": f"Unsupported application type: {app_type}",
            "details": "Only 'python-docker', 'spark', 'kafka' are supported."
        }

def verify_application_status(app_type: str) -> dict:
    """Verifies the status of the deployed application."""
    print(f"Verifying {app_type} application status...")
    if app_type == "python-docker":
        located = locate_execute_service()
        if located["status"] != "SUCCESS":
            return located
        return check_execute_service_health(located["details"])
    elif app_type == "spark":
        # Placeholder for Spark application status verification
        return {
            "step_name": "Verify Spark Application Status",
            "status": "SKIPPED",
            "message": "Spark application status verification is a placeholder.",
            "details": ""
        }
    elif app_type == "kafka":
        # Placeholder for Kafka application status verification
        return {
            "step_name": "Verify Kafka Application Status",
            "status": "SKIPPED",
            "message": "Kafka application status verification is a placeholder.",
            "details": ""
        }
    else:
        return {
            "step_name": "Verify Application Status",
            "status": "FAILED",
            "message": f"Unsupported application type: {app_type}",
            "details": ""
        }

EXECUTE_SERVICE_STEP = "Locate execute service"


def locate_execute_service() -> dict:
    """The python-docker step: find the chart-deployed ``/execute`` service.

    SUCCESS when the deployment configured ``AVALOKA_EXECUTE_SERVICE_URL`` and
    ``EXECUTE_SERVICE_TOKEN`` (Helm: ``executeService.enabled``), FAILED with
    the fix otherwise. Never builds an image or applies a manifest.
    """
    from app.executor.execute_service import TOKEN_ENV, URL_ENV, configured_host_port, configured_url

    where = configured_host_port()
    if where and (os.getenv(TOKEN_ENV) or "").strip():
        return {
            "step_name": EXECUTE_SERVICE_STEP,
            "status": "SUCCESS",
            "message": f"Execute service configured at {configured_url()}",
            "details": {"ip": where[0], "port": where[1]},
        }
    missing = [n for n, v in ((URL_ENV, where), (TOKEN_ENV, (os.getenv(TOKEN_ENV) or "").strip())) if not v]
    return {
        "step_name": EXECUTE_SERVICE_STEP,
        "status": "FAILED",
        "message": (
            "The execute service is not configured (missing " + ", ".join(missing) + "). "
            "It is deployed by the Helm chart: install or upgrade with executeService.enabled=true."
        ),
        "details": "",
    }


def check_execute_service_health(where: dict) -> dict:
    """GET /health on the execute service (no token needed; it runs nothing)."""
    import urllib.request

    url = f"http://{where['ip']}:{where['port']}/health"
    try:
        with urllib.request.urlopen(url, timeout=10) as res:
            ok = res.status == 200
    except Exception as exc:  # noqa: BLE001 - reported as the step's outcome
        return {"step_name": "Verify execute service", "status": "FAILED",
                "message": f"Execute service health check failed: {exc}", "details": ""}
    return {"step_name": "Verify execute service", "status": "SUCCESS" if ok else "FAILED",
            "message": "Execute service is healthy." if ok else "Execute service health check failed.",
            "details": where}


def get_service_ip_and_port(service_name: str) -> dict:
    """Retrieves the external IP and port of a Kubernetes service."""
    print(f"Getting service IP and port for {service_name}...")
    # Wait for the LoadBalancer IP to be available
    for _ in range(120): # 10 minutes timeout
        cmd = ["kubectl", "get", "service", service_name, "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"]
        print (cmd)
        result = _probe(cmd)
        print (result)
        ip = result.stdout.strip()
        if ip:
            port_cmd = ["kubectl", "get", "service", service_name, "-o", "jsonpath={.spec.ports[0].port}"]
            port_result = _probe(port_cmd)
            port = port_result.stdout.strip()
            if port:
                return {
                    "step_name": f"Get {service_name} IP/Port",
                    "status": "SUCCESS",
                    "message": f"Service {service_name} accessible at {ip}:{port}",
                    "details": {"ip": ip, "port": port}
                }
        time.sleep(5)
    
    return {
        "step_name": f"Get {service_name} IP/Port",
        "status": "FAILED",
        "message": f"Timed out waiting for external IP for service {service_name}.",
        "details": ""
    }

def cleanup_resources(platform: str) -> dict:
    """Cleans up K8s cluster resources."""
    print(f"Cleaning up {platform} K8s resources...")
    cluster_name = ""
    if platform == "gcp":
        project_id = os.environ.get("GCP_PROJECT_ID")
        cluster_name = os.environ.get("GCP_CLUSTER_NAME", "test-gke-cluster")
        zone = os.environ.get("GCP_ZONE", "us-central1-c")
        if not project_id:
            return {
                "step_name": f"Cleanup {platform} Cluster",
                "status": "SKIPPED",
                "message": "GCP_PROJECT_ID environment variable not set, skipping cleanup.",
                "details": ""
            }
        # Check if the cluster exists before trying to delete it
        check_cmd = ["gcloud", "container", "clusters", "describe", cluster_name, "--project", project_id, "--zone", zone]
        check_result = _probe(check_cmd)
        if check_result.returncode != 0:
            return {
                "step_name": f"Cleanup {platform} Cluster",
                "status": "SKIPPED",
                "message": f"GKE cluster {cluster_name} does not exist, skipping deletion.",
                "details": ""
            }
        cmd = [
            "gcloud", "container", "clusters", "delete", cluster_name,
            "--project", project_id,
            "--zone", zone,
            "--quiet"
        ]
        return _run_command(cmd, f"Delete {platform} Cluster")

    elif platform == "aws":
        aws_region = os.environ.get("AWS_REGION")
        cluster_name = os.environ.get("AWS_CLUSTER_NAME", "test-eks-cluster")
        if not aws_region:
            return {
                "step_name": f"Cleanup {platform} Cluster",
                "status": "SKIPPED",
                "message": "AWS_REGION environment variable not set, skipping cleanup.",
                "details": ""
            }
        # Check if the EKS cluster exists before trying to delete it
        check_cmd = ["aws", "eks", "describe-cluster", "--name", cluster_name, "--region", aws_region]
        check_result = _probe(check_cmd)
        if check_result.returncode != 0:
            return {
                "step_name": f"Cleanup {platform} Cluster",
                "status": "SKIPPED",
                "message": f"EKS cluster {cluster_name} does not exist, skipping deletion.",
                "details": ""
            }
        cmd = [
            "aws", "eks", "delete-cluster",
            "--name", cluster_name,
            "--region", aws_region
        ]
        delete_outcome = _run_command(cmd, f"Delete {platform} Cluster")
        if delete_outcome["status"] == "FAILED":
            return delete_outcome
        
        # Wait for cluster to be deleted
        wait_cmd = ["aws", "eks", "wait", "cluster-deleted", "--name", cluster_name, "--region", aws_region]
        return _run_command(wait_cmd, f"Wait for {platform} Cluster Deleted")
    else:
        return {
            "step_name": "Cleanup Resources",
            "status": "SKIPPED",
            "message": f"Unsupported platform: {platform}, skipping cleanup.",
            "details": ""
        }