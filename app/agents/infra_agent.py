# app/agents/infra_agent.py
import os
import subprocess 
from typing import Literal, List, Dict, Any
from app.graph.etl_state import ETLState
from app.infra import k8s_invoker

def infra_agent_node(state: ETLState) -> ETLState:
    """
    Provisions an infrastructure agent that provisions resources and reports detailed outcomes.
    """
    print("Infra Agent: Requesting infrastructure...")

    # try:
    #     subprocess.run(
    #         ["kubectl", "get", "nodes"],
    #         check=True,
    #         capture_output=True,
    #         text=True,
    #         shell=False,
    #     )
    #     kubectl_ready = True
    # except Exception:
    #     kubectl_ready = False

    try:
        p = subprocess.run(
            ["kubectl", "get", "nodes"],
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        kubectl_ready = True
        kubectl_nodes = p.stdout
    except Exception as e:
        kubectl_ready = False
        kubectl_nodes = str(e)
    
    
    infra_request = state.get("infrastructure_request") or {}
    platform = infra_request.get("type") or infra_request.get("platform")
    app_type = infra_request.get("app_type", "python-docker")

    execution_mode = state.get("execution_mode")
    use_ray = (execution_mode == "k8s-ray")

    # Resolve provider if type is k8s-ray
    if platform == "k8s-ray":
        platform = infra_request.get("provider") or os.getenv("DEFAULT_INFRA_PLATFORM", "gcp")

    platform = str(platform).lower() if platform else None
    if not platform or platform not in ["gcp", "aws"]:
        platform = os.getenv("DEFAULT_INFRA_PLATFORM", "gcp")

    #  Define ray_ns EARLY so it always exists (even if we fail before the ray block)
    ray_ns = None
    if use_ray:
        ray_ns = (
            (state.get("infrastructure_request") or {}).get("ray_namespace")
            or state.get("ray_namespace")
            # or os.getenv("RAY_NAMESPACE", "ray-jobs")
            or os.getenv("RAY_NAMESPACE", "ray-training")
        )

    provisioning_outcomes: List[Dict[str, Any]] = []
    overall_status = "provisioned"

    # Step 1 + 2: Provision cluster + configure kubectl
    if kubectl_ready:
        provisioning_outcomes.append({
            "step_name": "Provision Cluster",
            "status": "SKIPPED",
            "message": "kubectl already configured; skipping cloud cluster provisioning",
            "details": ""
        })
        provisioning_outcomes.append({
            "step_name": "Configure Kubectl",
            "status": "SKIPPED",
            "message": "kubectl already configured; skipping kubectl configuration",
            "details": ""
        })
    else:
        outcome = k8s_invoker.provision_cluster(platform)
        provisioning_outcomes.append(outcome)
        if outcome["status"] == "FAILED":
            overall_status = "failed"

        if overall_status == "provisioned":
            outcome = k8s_invoker.configure_kubectl(platform)
            provisioning_outcomes.append(outcome)
            if outcome["status"] == "FAILED":
                overall_status = "failed"

    # Step 3: Health checks
    if overall_status == "provisioned":
        outcome = k8s_invoker.run_cluster_health_checks()
        provisioning_outcomes.append(outcome)
        if outcome["status"] == "FAILED":
            overall_status = "failed"

    #  Ray prerequisites ONLY (namespace + kuberay)
    if overall_status == "provisioned" and use_ray:
        outcome = k8s_invoker.ensure_namespace(ray_ns)
        provisioning_outcomes.append(outcome)
        if outcome["status"] == "FAILED":
            overall_status = "failed"

        # if overall_status == "provisioned":
        #     outcome = k8s_invoker.ensure_kuberay_installed()
        #     provisioning_outcomes.append(outcome)
        #     if outcome["status"] == "FAILED":
        #         overall_status = "failed"
        if overall_status == "provisioned":
            outcome = k8s_invoker.ensure_kuberay_installed()
            # GKE manages Ray CRDs via kube-addon-manager — treat conflict as SKIPPED
            if outcome["status"] == "FAILED" and (
                "conflict" in str(outcome.get("details", "")).lower()
                or "kube-addon-manager" in str(outcome.get("details", "")).lower()
            ):
                outcome["status"] = "SKIPPED"
                outcome["message"] = "KubeRay CRDs already managed by GKE kube-addon-manager — skipping"
            provisioning_outcomes.append(outcome)
            if outcome["status"] == "FAILED":
                overall_status = "failed"

    #  Non-ray path: deploy app + verify + service IP
    if overall_status == "provisioned" and not use_ray:
        outcome = k8s_invoker.deploy_application(app_type, platform)
        provisioning_outcomes.append(outcome)
        if outcome["status"] == "FAILED":
            overall_status = "failed"

    if overall_status == "provisioned" and not use_ray:
        outcome = k8s_invoker.verify_application_status(app_type)
        provisioning_outcomes.append(outcome)
        if outcome["status"] == "FAILED":
            overall_status = "failed"

    if overall_status == "provisioned" and not use_ray:
        outcome = k8s_invoker.get_service_ip_and_port("infra-agent-service")
        provisioning_outcomes.append(outcome)
        if outcome["status"] == "FAILED":
            overall_status = "failed"

    infrastructure_details = {
        "type": platform,
        "app_type": app_type,
        "status": overall_status,
        "details": provisioning_outcomes,
        "kubectl_nodes": kubectl_nodes,
    }

    #  Return full updated state (safe default)
    new_state = dict(state)
    new_state["infrastructure_request"] = state.get("infrastructure_request")
    new_state["infrastructure_provisioned"] = infrastructure_details
    if use_ray:
        new_state["ray_namespace"] = ray_ns

    return new_state
