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


def _run_command(command: list, step_name: str) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, shell=False)
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
        
        # Assume IAM role and set temporary credentials (as done in test_eks_deployment.py)
        try:
            sts_client = boto3.client('sts', region_name=aws_region)
            assumed_role_object = sts_client.assume_role(
                RoleArn=eks_cluster_role_arn,
                RoleSessionName="EKSClusterProvisionSession"
            )
            credentials = assumed_role_object['Credentials']
            os.environ['AWS_ACCESS_KEY_ID'] = credentials['AccessKeyId']
            os.environ['AWS_SECRET_ACCESS_KEY'] = credentials['SecretAccessKey']
            os.environ['AWS_SESSION_TOKEN'] = credentials['SessionToken']
            os.environ["AWS_DEFAULT_REGION"] = aws_region # Configure AWS CLI default region
        except ClientError as e:
            return {
                "step_name": f"Assume IAM Role for {platform}",
                "status": "FAILED",
                "message": f"Failed to assume IAM role: {e}",
                "details": str(e)
            }

        # Check if the EKS cluster already exists
        check_cmd = ["aws", "eks", "describe-cluster", "--name", cluster_name, "--region", aws_region]
        check_result = _probe(check_cmd)

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
            create_outcome = _run_command(create_cmd, f"Create {platform} Cluster")
            if create_outcome["status"] == "FAILED":
                return create_outcome
            
            # Wait for cluster to become active
            wait_cmd = ["aws", "eks", "wait", "cluster-active", "--name", cluster_name, "--region", aws_region]
            return _run_command(wait_cmd, f"Wait for {platform} Cluster Active")
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

    # If CRDs exist AND operator seems running, skip
    if kuberay_crds_installed() and _operator_running(ns, release):
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

    install = _run_command(
        [_helm(), "upgrade", "--install", release, chart, "-n", ns, "--create-namespace", "--wait", "--timeout", "180s"],
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
        project_id = cloud_config.project_id()
        image_repo = os.environ.get(
            "INFRA_AGENT_IMAGE_REPO",
            f"gcr.io/{project_id}/infra-agent-python-app"
        )
        image_tag  = os.environ.get("INFRA_AGENT_IMAGE_TAG", "latest")
        full_image = f"{image_repo}:{image_tag}"

        # 1) Build
        dockerfile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python_app')
        build_outcome = _run_command(
            ["docker", "build", "-t", f"infra-agent-python-app:{image_tag}", dockerfile_path],
            "Build Python Docker Image"
        )
        if build_outcome["status"] == "FAILED":
            return build_outcome

        # 2) Auth (configurable registry host)
        registry_host = full_image.split("/")[0]
        _run_command(
            ["gcloud", "auth", "configure-docker", registry_host, "--quiet"],
            "Authenticate GCR"
        )

        # 3) Tag + Push
        _run_command(
            ["docker", "tag", f"infra-agent-python-app:{image_tag}", full_image],
            "Tag Docker Image"
        )
        push_outcome = _run_command(["docker", "push", full_image], "Push Docker Image")
        if push_outcome["status"] == "FAILED":
            return push_outcome

        # 4) Deploy (skip if exists)
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'config', 'infra-agent-deployment.yaml'
        )
        check_result = _probe(
            ["kubectl", "get", "deployment", "infra-agent", "-o", "jsonpath='{.metadata.name}'"],
        )
        if check_result.returncode == 0 and "infra-agent" in check_result.stdout:
            return {
                "step_name": "Deploy Python Docker Application",
                "status": "SKIPPED",
                "message": "Deployment 'infra-agent' already exists.",
                "details": ""
            }

        return _run_command(
            ["kubectl", "apply", "-f", config_path],
            "Deploy Python Docker Application"
        )
    
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
        # Wait for the deployment to be ready
        wait_cmd = ["kubectl", "wait", "--for=condition=available", "deployment/infra-agent", "--timeout=300s"]
        outcome = _run_command(wait_cmd, "Wait for Python App Deployment Ready")
        if outcome["status"] == "FAILED":
            return outcome

        # Check that the deployment is available
        get_cmd = ["kubectl", "get", "deployment", "infra-agent"]
        get_outcome = _run_command(get_cmd, "Get Python App Deployment Status")
        if get_outcome["status"] == "SUCCESS" and "1/1" in get_outcome["details"]:
            get_outcome["message"] = "Python application deployment verified: 1/1 replicas available."
        elif get_outcome["status"] == "SUCCESS":
            get_outcome["status"] = "WARNING"
            get_outcome["message"] = "Python application deployment not fully ready."
        return get_outcome
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