# app/infra/deploy_stack.py
"""Build/deploy the avaloka application stack onto a Kubernetes cluster.

Responsibilities:
  * build the avaloka + ray images and (for local kind) side-load them;
  * ``helm upgrade --install`` the avaloka chart (connect-or-provision Ray wiring);
  * optionally install data-stack dependencies (postgres/kafka/milvus/neo4j/
    opensearch) from ``manifests/helm-values`` — all opt-in, disabled by default.
"""
from __future__ import annotations

import os
from typing import List, Optional

from app.infra.providers.base import run_command
from app.infra.providers.local_kind import LocalKindProvider

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEPLOY = os.path.join(_REPO_ROOT, "deploy")

# Nothing in app.infra imports app.core.settings, so without this the repo .env
# is invisible and every key has to be exported by hand before `make up`.
# Precedence: shell export > .env.local (gitignored) > .env
try:  # pragma: no cover - never fail the deploy over dotenv
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_REPO_ROOT, ".env.local"), override=False)
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)
except Exception:  # noqa: BLE001
    pass
AVALOKA_CHART = os.path.join(_DEPLOY, "helm", "avaloka")
DOCKER_DIR = os.path.join(_DEPLOY, "docker")
HELM_VALUES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifests", "helm-values")

API_IMAGE = "avaloka-api:latest"      # FastAPI backend — the chart's primary workload (R1)
RAY_IMAGE = "avaloka-ray:latest"      # Ray head/worker + Serve image
UI_IMAGE = "avaloka:latest"           # optional Streamlit UI (built only when with_ui)
WEBUI_IMAGE = "avaloka-ui:latest"    # 1.6 TanStack SSR UI (chart 'webui.enabled')
FUNCTIONS_IMAGE = "avaloka-functions:latest"  # Supabase Edge Functions (chart 'supabase.functions.enabled')
# Back-compat alias for callers that referenced the old UI image constant.
AVALOKA_IMAGE = UI_IMAGE

# Optional data-stack components -> (helm repo name, repo url, chart, release name).
DATA_STACK = {
    "postgres": ("bitnami", "https://charts.bitnami.com/bitnami", "bitnami/postgresql", "avaloka-postgres"),
    "kafka": ("bitnami", "https://charts.bitnami.com/bitnami", "bitnami/kafka", "avaloka-kafka"),
    "milvus": ("milvus", "https://zilliztech.github.io/milvus-helm/", "milvus/milvus", "avaloka-milvus"),
    "neo4j": ("neo4j", "https://helm.neo4j.com/neo4j", "neo4j/neo4j", "avaloka-neo4j"),
    "opensearch": ("opensearch", "https://opensearch-project.github.io/helm-charts/", "opensearch/opensearch", "avaloka-opensearch"),
}


def build_images(
    load_into_kind: bool = False,
    kind_cluster: Optional[str] = None,
    with_ui: bool = False,
    with_webui: bool = True,
    with_functions: bool = True,
) -> List[dict]:
    """Build the avaloka images; optionally side-load them into kind.

    Builds the API image (chart's primary workload) and the Ray image by default.
    The Streamlit UI image (``avaloka:latest``) is built only when ``with_ui`` — the
    UI is opt-in (chart ``ui.enabled``), so most deploys don't need it.
    """
    outcomes: List[dict] = []
    built: List[str] = []
    outcomes.append(run_command(
        ["docker", "build", "-f", os.path.join(DOCKER_DIR, "Dockerfile.api"), "-t", API_IMAGE, _REPO_ROOT],
        "Build avaloka-api image",
    ))
    built.append(API_IMAGE)
    outcomes.append(run_command(
        ["docker", "build", "-f", os.path.join(DOCKER_DIR, "Dockerfile.ray"), "-t", RAY_IMAGE, _REPO_ROOT],
        "Build avaloka-ray image",
    ))
    built.append(RAY_IMAGE)
    if with_webui:
        outcomes.append(run_command(
            ["docker", "build", "-f", os.path.join(_REPO_ROOT, "ui", "Dockerfile"),
             "-t", WEBUI_IMAGE, os.path.join(_REPO_ROOT, "ui")],
            "Build avaloka-ui (1.6 SSR) image",
        ))
        built.append(WEBUI_IMAGE)
    if with_functions:
        # supabase.functions.enabled defaults on and nothing else builds this,
        # so without it the pod sits in ImagePullBackOff and helm --wait times out.
        outcomes.append(run_command(
            ["docker", "build", "-f", os.path.join(DOCKER_DIR, "Dockerfile.functions"),
             "-t", FUNCTIONS_IMAGE, _REPO_ROOT],
            "Build avaloka-functions (Supabase Edge Functions) image",
        ))
        built.append(FUNCTIONS_IMAGE)
    if with_ui:
        outcomes.append(run_command(
            ["docker", "build", "-f", os.path.join(DOCKER_DIR, "Dockerfile.avaloka"), "-t", UI_IMAGE, _REPO_ROOT],
            "Build avaloka UI (Streamlit) image",
        ))
        built.append(UI_IMAGE)
    if load_into_kind and not any(o["status"] == "FAILED" for o in outcomes):
        provider = LocalKindProvider(cluster_name=kind_cluster) if kind_cluster else LocalKindProvider()
        for img in built:
            outcomes.append(provider.load_image(img))
    return outcomes


def deploy_avaloka(
    namespace: str = "default",
    connect_existing: bool = False,
    ray_address: str = "",
    provider: str = "local",
    groq_planning_key: Optional[str] = None,
    groq_coding_key: Optional[str] = None,
    jwt_secret: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    service_type: Optional[str] = None,
    image_repository: Optional[str] = None,
    image_tag: Optional[str] = None,
    image_pull_policy: Optional[str] = None,
    minio: Optional[bool] = None,
    supabase: Optional[bool] = None,
) -> dict:
    """helm upgrade --install the avaloka chart.

    Image overrides let cloud (GKE/EKS) point at a registry image (e.g.
    gcr.io/<project>/avaloka) while local kind uses the side-loaded image.

    ``minio`` deploys the in-cluster S3-compatible object store and points the
    app's storage backend at it. This is what gives a local/on-prem cluster a
    shared data plane: Ray workers land on arbitrary nodes and must read what the
    API wrote, and kind/Docker Desktop have no ReadWriteMany storage class to
    share a volume with. None leaves the chart default (off).
    """
    cmd = [
        "helm", "upgrade", "--install", "avaloka", AVALOKA_CHART,
        "--namespace", namespace, "--create-namespace",
        "--set", f"ray.connectExisting={'true' if connect_existing else 'false'}",
        "--wait", "--timeout", "5m",
    ]

    if provider == "gcp":
        cmd += ["-f", os.path.join(AVALOKA_CHART, "values", "values-gke.yaml")]
        project_id = os.environ.get("GCP_PROJECT_ID", "").strip()
        if project_id:
            gcp_service_account = (
                os.environ.get("GCP_SERVICE_ACCOUNT", "").strip()
                or f"avaloka@{project_id}.iam.gserviceaccount.com"
            )
            cmd += [
                "--set-string",
                "serviceAccount.annotations.iam\\.gke\\.io/gcp-service-account="
                f"{gcp_service_account}",
                "--set-string", f"config.gcpProjectId={project_id}",
            ]
    elif provider == "aws":
        cmd += ["-f", os.path.join(AVALOKA_CHART, "values", "values-eks.yaml")]
    elif provider == "azure":
        cmd += ["-f", os.path.join(AVALOKA_CHART, "values", "values-aks.yaml")]

    if connect_existing and ray_address:
        cmd += ["--set-string", f"ray.address={ray_address}"]
    if service_type:
        cmd += ["--set", f"service.type={service_type}"]
    if image_repository:
        cmd += ["--set-string", f"image.repository={image_repository}"]
    if image_tag:
        cmd += ["--set-string", f"image.tag={image_tag}"]
    if image_pull_policy:
        cmd += ["--set", f"image.pullPolicy={image_pull_policy}"]
    if minio is not None:
        cmd += ["--set", f"minio.enabled={'true' if minio else 'false'}"]
    if supabase is not None:
        cmd += ["--set", f"supabase.enabled={'true' if supabase else 'false'}"]
    if supabase:
        # The chart ships no Supabase credentials: a usable secret in a public
        # repository is a forgeable auth stack, and the published demo keys let
        # anyone mint a service_role token. Forward them from the environment
        # instead, so they can live in an env file outside git.
        #
        # Nothing is defaulted here. If they are unset the chart's credential
        # guard fails the render with an actionable message, which is better
        # than standing up an auth stack whose keys are public.
        for flag, env_var in (
            ("supabase.jwtSecret", "SUPABASE_JWT_SECRET"),
            ("supabase.anonKey", "SUPABASE_ANON_KEY"),
            ("supabase.serviceKey", "SUPABASE_SERVICE_KEY"),
        ):
            value = (os.environ.get(env_var) or "").strip()
            if value:
                cmd += ["--set-string", f"{flag}={value}"]

    # Provision mode installs ``deploy/helm/ray/raycluster.yaml`` immediately
    # after this Helm release. Point all Ray Jobs clients at that shared,
    # in-cluster dashboard by default so scheduled MTA training does not fall
    # back to creating a separate GKE namespace and RayCluster for every run.
    # Explicit environment variables still win for externally managed clusters.
    default_ray_dashboard = (
        "" if connect_existing else "http://avaloka-raycluster-head-svc:8265"
    )
    ray_dashboard = (
        os.environ.get("RAY_DASHBOARD_URL", "").strip()
        or default_ray_dashboard
    )
    ray_dashboard_in_cluster = (
        os.environ.get("RAY_DASHBOARD_URL_INCLUSTER", "").strip()
        or default_ray_dashboard
    )
    ray_api_server = (
        os.environ.get("RAY_API_SERVER_ADDRESS", "").strip()
        or default_ray_dashboard
    )
    mta_ray_dashboard = (
        os.environ.get("MTA_RAY_DASHBOARD_URL", "").strip()
        or default_ray_dashboard
    )
    ray_values = (
        ("config.rayDashboardUrl", ray_dashboard),
        ("config.rayDashboardUrlInCluster", ray_dashboard_in_cluster),
        ("config.rayApiServerAddress", ray_api_server),
        ("config.mtaRayDashboardUrl", mta_ray_dashboard),
    )
    for value_path, value in ray_values:
        if value:
            cmd += ["--set-string", f"{value_path}={value}"]
    if mta_ray_dashboard:
        mta_namespace = (
            os.environ.get("MTA_RAY_NAMESPACE", "").strip() or namespace
        )
        cmd += ["--set-string", f"config.mtaRayNamespace={mta_namespace}"]

    # Prefer env vars so keys never land in shell history when not passed explicitly.
    pk = groq_planning_key if groq_planning_key is not None else (os.environ.get("GROQ_API_KEY_PLANNING_AGENT", "") or os.environ.get("GROQ_API_KEY", ""))
    ck = groq_coding_key if groq_coding_key is not None else (os.environ.get("GROQ_API_KEY_CODING_AGENT", "") or os.environ.get("GROQ_API_KEY", ""))
    if pk:
        cmd += ["--set-string", f"secrets.groqPlanningKey={pk}"]
    if ck:
        cmd += ["--set-string", f"secrets.groqCodingKey={ck}"]
    pii_key = os.environ.get("AVALOKA_PII_KEY", "").strip()
    if pii_key:
        cmd += ["--set-string", f"secrets.piiKey={pii_key}"]
    # Optional OpenRouter backup (app/core/model_fallback.py). Same hygiene as the
    # Groq keys: read from the env by default, and only emit the flag when set —
    # an empty value arms nothing and would just add a confusing Secret key.
    ork = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if ork:
        cmd += ["--set-string", f"secrets.openrouterKey={ork}"]
    # The API validates HS256 JWTs against SUPABASE_JWT_SECRET; inject it so in-cluster
    # auth works. Only emit the flag when a value is present (T1.4-style hygiene).
    js = jwt_secret if jwt_secret is not None else os.environ.get("SUPABASE_JWT_SECRET", "")
    if js:
        cmd += ["--set-string", f"secrets.jwtSecret={js}"]
    # The API derives its JWKS URL from this, and tokens are ES256, so it must
    # match the project the UI signs in to or every request verifies as anonymous.
    su = os.environ.get("SUPABASE_URL", "").strip()
    if su:
        cmd += [
            "--set-string", f"config.supabaseUrl={su}",
            "--set-string", f"webui.supabaseUrl={su}",
        ]
    sak = os.environ.get("SUPABASE_ANON_KEY", "").strip()
    if sak:
        cmd += ["--set-string", f"webui.supabaseAnonKey={sak}"]
    # Same project's service-role key; the webui needs it as
    # PRIMARY_SUPABASE_SERVICE_ROLE_KEY.
    srk = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if srk:
        cmd += ["--set-string", f"secrets.supabaseServiceRoleKey={srk}"]
    gb = (gcs_bucket if gcs_bucket is not None else os.environ.get("GCS_BUCKET", "")).strip()
    if gb:
        cmd += ["--set-string", f"config.gcsBucket={gb}"]
        if provider == "gcp":
            cmd += ["--set-string", f"mlflow.artifactsDestination=gs://{gb}/mlflow-artifacts"]

    # Ray and API clients must use MLflow's HTTP endpoint for runs:/ model
    # resolution. A raw PostgreSQL backend URI cannot serve registry/artifact APIs.
    inference_env_overrides = (
        ("INFERENCE_BACKEND", "config.inferenceBackend"),
        ("MLFLOW_TRACKING_URI", "mlflow.trackingUri"),
        ("MLFLOW_REGISTRY_URI", "mlflow.registryUri"),
        ("MLFLOW_DEFAULT_ARTIFACT_ROOT", "mlflow.defaultArtifactRoot"),
        ("MLFLOW_ARTIFACTS_DESTINATION", "mlflow.artifactsDestination"),
    )
    for env_name, value_path in inference_env_overrides:
        value = os.environ.get(env_name, "").strip()
        if value:
            cmd += ["--set-string", f"{value_path}={value}"]
    return run_command(cmd, "Deploy avaloka (Helm)")


def deploy_data_stack(components: List[str], namespace: str = "default") -> List[dict]:
    """Install optional data-stack components from manifests/helm-values (opt-in)."""
    outcomes: List[dict] = []
    for comp in components:
        if comp not in DATA_STACK:
            outcomes.append({
                "step_name": f"Deploy {comp}", "status": "SKIPPED",
                "message": f"Unknown data-stack component '{comp}'. Known: {', '.join(DATA_STACK)}.",
                "details": "",
            })
            continue
        repo_name, repo_url, chart, release = DATA_STACK[comp]
        run_command(["helm", "repo", "add", repo_name, repo_url], f"Add {repo_name} repo")
        run_command(["helm", "repo", "update", repo_name], f"Update {repo_name} repo")
        values_file = os.path.join(HELM_VALUES_DIR, f"{comp}-values.yaml")
        cmd = ["helm", "upgrade", "--install", release, chart, "--namespace", namespace, "--create-namespace"]
        if os.path.exists(values_file) and os.path.getsize(values_file) > 0:
            cmd += ["-f", values_file]
        outcomes.append(run_command(cmd, f"Deploy {comp}"))
    return outcomes


if __name__ == "__main__":  # pragma: no cover
    print("deploy_stack: use build_images(), deploy_avaloka(), deploy_data_stack([...])")
