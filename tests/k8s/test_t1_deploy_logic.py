"""T1 — Deploy logic, no cluster (< 30 s).

Mocks ``deploy_stack.run_command`` and asserts on command vectors + rendered
manifests. Real ``helm lint``/``helm template`` are run where helm is present.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tests.k8s.conftest import (
    CHART_DIR, OVERLAY_DIR, KIND_CLUSTER_YAML, RAY_VERSION,
    helm, has, requires_helm,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OVERLAYS = ["gke", "eks", "aks", "minikube", "onprem"]


def _capture_deploy_argv(**kwargs):
    """Call deploy_avaloka with run_command mocked; return the argv it built."""
    from app.infra import deploy_stack

    with patch.object(deploy_stack, "run_command") as rc:
        rc.return_value = {"step_name": "x", "status": "SUCCESS", "message": "", "details": ""}
        deploy_stack.deploy_avaloka(**kwargs)
        assert rc.called, "deploy_avaloka did not invoke run_command"
        return rc.call_args.args[0]


# --------------------------------------------------------------------------- T1.1
def test_t1_1_deploy_avaloka_base_argv(monkeypatch):
    for k in ("GROQ_API_KEY_PLANNING_AGENT", "GROQ_API_KEY_CODING_AGENT",
              "SUPABASE_JWT_SECRET", "GCS_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    argv = _capture_deploy_argv(namespace="avaloka-test")
    joined = " ".join(argv)
    assert argv[:4] == ["helm", "upgrade", "--install", "avaloka"]
    assert "--namespace" in argv and "avaloka-test" in argv
    assert "--create-namespace" in argv
    assert "--wait" in argv and "--timeout" in argv and "5m" in argv
    assert "ray.connectExisting=false" in joined
    assert "config.mtaRayDashboardUrl=http://avaloka-raycluster-head-svc:8265" in joined
    assert "config.mtaRayNamespace=avaloka-test" in joined
    assert "config.rayApiServerAddress=http://avaloka-raycluster-head-svc:8265" in joined


# --------------------------------------------------------------------------- T1.2
def test_t1_2_connect_mode_sets_ray_address(monkeypatch):
    for k in ("GROQ_API_KEY_PLANNING_AGENT", "GROQ_API_KEY_CODING_AGENT",
              "SUPABASE_JWT_SECRET", "GCS_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    argv = _capture_deploy_argv(connect_existing=True, ray_address="ray://h:10001")
    joined = " ".join(argv)
    assert "ray.connectExisting=true" in joined
    assert "ray.address=ray://h:10001" in joined


# --------------------------------------------------------------------------- T1.3
def test_t1_3_cloud_image_overrides_present(monkeypatch):
    for k in ("GROQ_API_KEY_PLANNING_AGENT", "GROQ_API_KEY_CODING_AGENT",
              "SUPABASE_JWT_SECRET", "GCS_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    argv = _capture_deploy_argv(
        image_repository="gcr.io/p/avaloka-api", image_tag="v9", image_pull_policy="Always",
    )
    joined = " ".join(argv)
    assert "image.repository=gcr.io/p/avaloka-api" in joined
    assert "image.tag=v9" in joined
    assert "image.pullPolicy=Always" in joined


# --------------------------------------------------------------------------- T1.4
def test_t1_4_no_empty_groq_flag_when_unset(monkeypatch):
    for k in ("GROQ_API_KEY_PLANNING_AGENT", "GROQ_API_KEY_CODING_AGENT",
              "SUPABASE_JWT_SECRET", "GCS_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    joined = " ".join(_capture_deploy_argv())
    # Not even an empty one — the flag must be absent entirely.
    assert "secrets.groqPlanningKey" not in joined
    assert "secrets.groqCodingKey" not in joined


def test_t1_4_pii_key_is_forwarded_to_helm(monkeypatch):
    monkeypatch.setenv("AVALOKA_PII_KEY", "test-pii-key")

    joined = " ".join(_capture_deploy_argv())
    assert "secrets.piiKey=test-pii-key" in joined


def test_t1_4_empty_pii_key_is_not_forwarded_to_helm(monkeypatch):
    monkeypatch.setenv("AVALOKA_PII_KEY", "")

    joined = " ".join(_capture_deploy_argv())
    assert "secrets.piiKey" not in joined


def test_t1_4_external_supabase_reaches_api_and_webui(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "public-anon-key")

    joined = " ".join(_capture_deploy_argv())
    assert "config.supabaseUrl=https://example.supabase.co" in joined
    assert "webui.supabaseUrl=https://example.supabase.co" in joined
    assert "webui.supabaseAnonKey=public-anon-key" in joined


def test_t1_4_local_bootstrap_can_enable_bundled_supabase():
    joined = " ".join(_capture_deploy_argv(provider="local", supabase=True))
    assert "supabase.enabled=true" in joined


def test_t1_4_mlflow_http_endpoint_is_passed_to_helm(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://avaloka-mlflow:5000")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "http://avaloka-mlflow:5000")
    monkeypatch.setenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "s3://avaloka/mlflow-artifacts")

    joined = " ".join(_capture_deploy_argv())
    assert "mlflow.trackingUri=http://avaloka-mlflow:5000" in joined
    assert "mlflow.registryUri=http://avaloka-mlflow:5000" in joined
    assert "mlflow.defaultArtifactRoot=s3://avaloka/mlflow-artifacts" in joined


def test_t1_4_gcp_bucket_configures_gcs_for_app_and_mlflow(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "dev-project")
    monkeypatch.setenv("GCS_BUCKET", "dev-artifacts")

    joined = " ".join(_capture_deploy_argv(provider="gcp", minio=False))
    assert "minio.enabled=false" in joined
    assert "config.gcsBucket=dev-artifacts" in joined
    assert "mlflow.artifactsDestination=gs://dev-artifacts/mlflow-artifacts" in joined
    assert "config.gcpProjectId=dev-project" in joined
    assert (
        "serviceAccount.annotations.iam\\.gke\\.io/gcp-service-account="
        "avaloka@dev-project.iam.gserviceaccount.com"
    ) in joined


def test_t1_4_explicit_shared_ray_settings_override_defaults(monkeypatch):
    monkeypatch.setenv("MTA_RAY_DASHBOARD_URL", "http://shared-ray.ray.svc:8265")
    monkeypatch.setenv("MTA_RAY_NAMESPACE", "ray")

    joined = " ".join(_capture_deploy_argv(namespace="avaloka"))
    assert "config.mtaRayDashboardUrl=http://shared-ray.ray.svc:8265" in joined
    assert "config.mtaRayNamespace=ray" in joined


@requires_helm
def test_t1_4_gcp_project_identity_is_rendered_for_all_envfrom_workloads():
    rendered = helm(
        "template", "avaloka", str(CHART_DIR),
        "--set-string", "config.gcpProjectId=dev-project",
    )
    docs = [d for d in yaml.safe_load_all(rendered.stdout) if d]
    config = next(d for d in docs if d.get("kind") == "ConfigMap")
    assert config["data"]["CLOUD_PROVIDER"] == "gcp"
    assert config["data"]["GCP_PROJECT_ID"] == "dev-project"
    assert config["data"]["GOOGLE_CLOUD_PROJECT"] == "dev-project"


# --------------------------------------------------------------------------- T1.5
def test_t1_5_factory_providers():
    from app.infra.providers.factory import get_provider, SUPPORTED_PLATFORMS
    from app.infra.providers.local_kind import LocalKindProvider
    from app.infra.providers.gcp_gke import GcpGkeProvider
    from app.infra.providers.aws_eks import AwsEksProvider
    from app.infra.providers.azure_aks import AzureAksProvider

    assert SUPPORTED_PLATFORMS == ("local", "gcp", "aws", "azure")
    assert isinstance(get_provider("local"), LocalKindProvider)
    assert isinstance(get_provider("gcp"), GcpGkeProvider)
    assert isinstance(get_provider("aws"), AwsEksProvider)
    assert isinstance(get_provider("azure"), AzureAksProvider)
    with pytest.raises(ValueError):
        get_provider("k3s-unknown")


# --------------------------------------------------------------------------- T1.6
@pytest.mark.parametrize("provider,expected", [
    ("gcp", None), ("aws", None), ("azure", None), ("local", None),
])
def test_t1_6_service_type_default(provider, expected):
    """Cloud overlays use ClusterIP behind their Gateway unless explicitly overridden."""
    from app.infra import cluster_bootstrap

    captured = {}

    def fake_provision(args):
        captured["args"] = args
        return 0

    with patch.object(cluster_bootstrap, "provision", side_effect=fake_provision):
        rc = cluster_bootstrap.main(["--provider", provider, "--mode", "provision"])
    assert rc == 0
    assert captured["args"].service_type == expected


# --------------------------------------------------------------------------- T1.7
@requires_helm
def test_t1_7_helm_lint_and_template_all_overlays():
    lint = helm("lint", str(CHART_DIR), check=False)
    assert lint.returncode == 0, f"helm lint failed:\n{lint.stdout}\n{lint.stderr}"
    # base render
    base = helm("template", "avaloka", str(CHART_DIR), check=False)
    assert base.returncode == 0, base.stderr
    list(yaml.safe_load_all(base.stdout))  # valid YAML
    # every overlay
    for ov in OVERLAYS:
        f = OVERLAY_DIR / f"values-{ov}.yaml"
        assert f.exists(), f"missing overlay {f}"
        r = helm("template", "avaloka", str(CHART_DIR), "-f", str(f), check=False)
        assert r.returncode == 0, f"{ov} template failed:\n{r.stderr}"
        list(yaml.safe_load_all(r.stdout))


@requires_helm
def test_t1_7_external_supabase_does_not_reuse_bundled_service_key():
    rendered = helm(
        "template", "avaloka", str(CHART_DIR),
        "--set-string", "config.supabaseUrl=https://example.supabase.co",
    )
    docs = [d for d in yaml.safe_load_all(rendered.stdout) if d]
    secret = next(
        d for d in docs
        if d.get("kind") == "Secret"
        and "SUPABASE_SERVICE_ROLE_KEY" in (d.get("stringData") or {})
    )
    assert secret["stringData"]["SUPABASE_SERVICE_ROLE_KEY"] == ""
    assert secret["stringData"]["SUPABASE_JWT_SECRET"] == ""


# --------------------------------------------------------------------------- T1.8
@requires_helm
def test_t1_8_api_port_probes_and_nodeport(rendered_chart):
    docs = [d for d in yaml.safe_load_all(rendered_chart) if d]
    deploys = [d for d in docs if d.get("kind") == "Deployment"]
    api = next(d for d in deploys
              if d["metadata"]["labels"].get("app.kubernetes.io/component") == "api"
              or d["metadata"]["name"] == "avaloka")
    container = api["spec"]["template"]["spec"]["containers"][0]
    assert container["ports"][0]["containerPort"] == 9000
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"

    # chart NodePort matches a kind-cluster extraPortMappings.containerPort.
    # The api Service is the one whose selector targets the api component.
    svc = next(d for d in docs if d.get("kind") == "Service"
               and d["spec"]["selector"].get("app.kubernetes.io/component") == "api")
    node_port = svc["spec"]["ports"][0].get("nodePort")
    assert node_port == 30085
    kind_cfg = yaml.safe_load(KIND_CLUSTER_YAML.read_text())
    mapped = {m["containerPort"] for n in kind_cfg["nodes"]
              for m in n.get("extraPortMappings", [])}
    assert node_port in mapped, f"chart NodePort {node_port} not in kind extraPortMappings {mapped}"


# --------------------------------------------------------------------------- T1.9
def test_t1_9_single_source_ray_version():
    """D2 guard: ray image tag == both CR rayVersion == requirements ray pin == 2.58.0."""
    dockerfile = (REPO_ROOT / "deploy/docker/Dockerfile.ray").read_text()
    m = re.search(r"FROM rayproject/ray:([0-9.]+)-py311", dockerfile)
    assert m, "could not find ray base image tag"
    img_tag = m.group(1)

    # Both CR files are multi-document; pick the RayCluster / RayService doc.
    rc_docs = [d for d in yaml.safe_load_all((REPO_ROOT / "deploy/helm/ray/raycluster.yaml").read_text()) if d]
    rs_docs = [d for d in yaml.safe_load_all((REPO_ROOT / "deploy/helm/ray/rayservice.yaml").read_text()) if d]
    rc = next(d for d in rc_docs if d.get("kind") == "RayCluster")
    rs = next(d for d in rs_docs if d.get("kind") == "RayService")
    rc_ver = rc["spec"]["rayVersion"]
    rs_ver = rs["spec"]["rayClusterConfig"]["rayVersion"]

    req = (REPO_ROOT / "requirements.txt").read_text()
    # Match whatever extras are pinned, not one hard-coded set. This asked for
    # `ray[ml]` specifically, and `ray[ml]` stopped existing when that extra was
    # found not to be real -- so the guard could no longer find the pin it was
    # guarding, and failed on its own assert instead of comparing versions.
    rm = re.search(r"^ray(?:\[[^\]]*\])?==([0-9.]+)", req, re.M)
    assert rm, "could not find a ray pin in requirements.txt"
    req_ver = rm.group(1)

    assert img_tag == rc_ver == rs_ver == req_ver == RAY_VERSION, (
        f"ray version mismatch: image={img_tag} raycluster={rc_ver} "
        f"rayservice={rs_ver} requirements={req_ver} expected={RAY_VERSION}"
    )


# --------------------------------------------------------------------------- T1.10
@requires_helm
def test_t1_10_rbac_rayservices_and_ray_serve_url(rendered_chart):
    """D3 guard: rbac grants rayservices patch verbs; ConfigMap carries RAY_SERVE_URL."""
    docs = [d for d in yaml.safe_load_all(rendered_chart) if d]
    role = next(d for d in docs if d.get("kind") == "Role")
    ray_rules = [r for r in role["rules"] if "ray.io" in r.get("apiGroups", [])]
    assert ray_rules, "no ray.io rules in Role"
    rule = ray_rules[0]
    assert "rayservices" in rule["resources"]
    assert "patch" in rule["verbs"] and "update" in rule["verbs"]

    cm = next(d for d in docs if d.get("kind") == "ConfigMap")
    assert "RAY_SERVE_URL" in cm["data"]
    assert "REDIS_URL" in cm["data"]


# --------------------------------------------------------------------------- T1.11
@requires_helm
def test_t1_11_local_chart_manages_mlflow_with_minio(rendered_chart):
    docs = [d for d in yaml.safe_load_all(rendered_chart) if d]
    mlflow = next(
        d for d in docs
        if d.get("kind") == "Deployment"
        and d["metadata"]["labels"].get("app.kubernetes.io/component") == "mlflow"
    )
    service = next(d for d in docs if d.get("kind") == "Service" and d["metadata"]["name"] == "avaloka-mlflow")
    assert service["spec"]["type"] == "ClusterIP"

    container = mlflow["spec"]["template"]["spec"]["containers"][0]
    command = container["args"][0]
    assert "--artifacts-destination" in command
    assert "--default-artifact-root" not in command
    env = {entry["name"]: entry for entry in container["env"]}
    assert env["MLFLOW_ARTIFACTS_DESTINATION"]["value"] == "s3://avaloka/mlflow-artifacts"
    assert env["MLFLOW_S3_ENDPOINT_URL"]["value"] == "http://avaloka-minio:9000"
    assert env["AWS_ACCESS_KEY_ID"]["valueFrom"]["secretKeyRef"]["name"] == "avaloka-minio"

    config = next(d for d in docs if d.get("kind") == "ConfigMap" and d["metadata"]["name"] == "avaloka-config")
    assert config["data"]["MLFLOW_TRACKING_URI"] == "http://avaloka-mlflow:5000"
    assert config["data"]["MLFLOW_REGISTRY_URI"] == "http://avaloka-mlflow:5000"
    assert "MLFLOW_DEFAULT_ARTIFACT_ROOT" not in config["data"]


@requires_helm
def test_t1_11_gke_chart_manages_mlflow_with_gcs_and_no_minio():
    rendered = helm(
        "template", "avaloka", str(CHART_DIR),
        "-f", str(OVERLAY_DIR / "values-gke.yaml"),
        check=False,
    )
    assert rendered.returncode == 0, rendered.stderr
    docs = [d for d in yaml.safe_load_all(rendered.stdout) if d]

    # The chart deliberately retains the MinIO PVC when MinIO is disabled so a
    # later local re-enable does not discard persisted artifacts. GKE must not
    # deploy a MinIO workload or expose a MinIO service, but that dormant claim
    # is safe to render.
    assert not any(
        d.get("kind") in {"Deployment", "Service", "Job"}
        and d.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component") == "minio"
        for d in docs
    )
    mlflow = next(
        d for d in docs
        if d.get("kind") == "Deployment"
        and d["metadata"]["labels"].get("app.kubernetes.io/component") == "mlflow"
    )
    container = mlflow["spec"]["template"]["spec"]["containers"][0]
    env = {entry["name"]: entry for entry in container["env"]}
    assert env["MLFLOW_ARTIFACTS_DESTINATION"]["value"] == (
        "gs://avaloka-user-filestore/mlflow-artifacts"
    )
    assert "MLFLOW_S3_ENDPOINT_URL" not in env
    assert "AWS_ACCESS_KEY_ID" not in env


def test_t1_11_bootstrap_creates_chart_before_ray_pods():
    import inspect
    from app.infra import cluster_bootstrap

    source = inspect.getsource(cluster_bootstrap.provision)
    assert source.index("deploy_stack.deploy_avaloka") < source.index("ray_manager.apply_ray_cluster")
