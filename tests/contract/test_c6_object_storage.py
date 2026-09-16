"""C6 — in-cluster object storage (MinIO) contract.

Clusterless: values are parsed with ``yaml.safe_load``, templates are scanned
line-wise, and the app-side legs import the real modules.

The failure this suite exists to prevent is the quiet one. A local install with
no shared data plane looks completely healthy — the API accepts an upload, writes
it to its own pod filesystem, and only later does a Ray worker on another node
fail to find the file. Likewise an S3 client with no ``endpoint_url`` does not
error at startup: it resolves the bucket against real AWS and returns
NoSuchBucket, which reads as a missing dataset rather than a misconfiguration.

So the pins here are: MinIO exists and is durable; the chart wires the endpoint
and credentials everywhere that reads ``s3://`` (API, LangGraph, Ray head and
workers); and a real cloud backend always wins over it.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import re
from unittest import mock

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHART_ROOT = REPO_ROOT / "deploy" / "helm" / "avaloka"
TEMPLATE_DIR = CHART_ROOT / "templates"
OVERLAY_DIR = CHART_ROOT / "values"
RAY_CLUSTER = REPO_ROOT / "deploy" / "helm" / "ray" / "raycluster.yaml"
RAY_SERVICE = REPO_ROOT / "deploy" / "helm" / "ray" / "rayservice.yaml"
KIND_CLUSTER = REPO_ROOT / "deploy" / "clusters" / "kind-cluster.yaml"

MINIO_TEMPLATE = TEMPLATE_DIR / "minio.yaml"
LOADER_COPIES = (
    REPO_ROOT / "app" / "agents" / "mta_v2" / "loader" / "url.py",
    REPO_ROOT / "app" / "agents" / "mta_v2" / "inference_service_image" / "loader" / "url.py",
)

# Overlays for clusters with no cloud bucket — these need the in-cluster store.
LOCAL_OVERLAYS = ("values-minikube.yaml", "values-onprem.yaml")
# Overlays with real cloud dataset storage — MinIO must not displace it. GKE also
# replaces the MLflow artifact store with GCS; the other cloud overlays retain
# their current artifact-store configuration.
CLOUD_OVERLAYS = ("values-gke.yaml", "values-eks.yaml", "values-aks.yaml")


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_values(path: pathlib.Path) -> dict:
    data = yaml.safe_load(_read(path))
    return data if isinstance(data, dict) else {}


@pytest.fixture(scope="module")
def base_values() -> dict:
    return _load_values(CHART_ROOT / "values.yaml")


# ------------------------------------------------------------------ the store
def test_c6_01_minio_template_exists_and_is_values_gated() -> None:
    assert MINIO_TEMPLATE.is_file(), f"missing {MINIO_TEMPLATE}"
    head = _read(MINIO_TEMPLATE).lstrip().splitlines()[0]
    assert head.startswith("{{- if .Values.minio.enabled }}"), (
        f"minio.yaml is not gated on .Values.minio.enabled: {head!r}"
    )


def test_c6_02_minio_is_on_by_default_for_mlflow(base_values: dict) -> None:
    """The base/local chart gives MLflow a MinIO artifact store by default."""
    assert (base_values.get("minio") or {}).get("enabled") is True


@pytest.mark.parametrize("kind", ["Deployment", "Service", "PersistentVolumeClaim", "Secret", "Job"])
def test_c6_03_minio_ships_a_complete_object(kind: str) -> None:
    """Store, network identity, durable volume, credentials, and bucket creation."""
    assert f"kind: {kind}" in _read(MINIO_TEMPLATE), f"minio.yaml renders no {kind}"


def test_c6_04_minio_storage_is_durable_and_not_rolling(base_values: dict) -> None:
    """A PVC, and Recreate — a RWO volume cannot be held by two pods at once.

    With the default RollingUpdate the replacement pod waits forever on a volume
    the outgoing pod still has mounted, and the upgrade hangs rather than fails.
    """
    text = _read(MINIO_TEMPLATE)
    assert "kind: PersistentVolumeClaim" in text
    assert "emptyDir" not in text, "MinIO data on emptyDir would lose datasets on reschedule"
    assert re.search(r"strategy:\s*\n\s*(#.*\n\s*)*type:\s*Recreate", text), (
        "MinIO uses the default RollingUpdate against a ReadWriteOnce volume"
    )
    size = ((base_values.get("minio") or {}).get("persistence") or {}).get("size")
    assert size, "no minio.persistence.size configured"


def test_c6_05_bucket_creation_is_idempotent() -> None:
    """The bucket hook survives re-runs and races with a still-starting MinIO."""
    text = _read(MINIO_TEMPLATE)
    assert "--ignore-existing" in text, "mc mb would fail the hook on an existing bucket"
    assert "post-install,post-upgrade" in text, "bucket hook does not run on upgrade"
    assert "until mc alias set" in text, "hook does not wait for MinIO to accept connections"


# ------------------------------------------------------------- chart wiring
@pytest.mark.parametrize("overlay", LOCAL_OVERLAYS)
def test_c6_06_local_overlays_enable_the_object_store(overlay: str) -> None:
    """Local/on-prem clusters have no cloud bucket, so they must ship MinIO.

    They also have no ReadWriteMany storage class, so a shared volume is not an
    alternative — object storage is the only way Ray workers on other nodes can
    read what the API wrote.
    """
    assert (_load_values(OVERLAY_DIR / overlay).get("minio") or {}).get("enabled") is True, (
        f"{overlay} ships no shared data plane"
    )


@pytest.mark.parametrize("overlay", CLOUD_OVERLAYS)
def test_c6_07_cloud_overlays_keep_their_own_dataset_storage(overlay: str) -> None:
    """A cloud overlay keeps GCS/S3/Azure for datasets instead of adopting MinIO."""
    values = _load_values(OVERLAY_DIR / overlay)
    backend = ((values.get("config") or {}).get("storageBackend") or "").lower()
    assert backend in {"gcs", "s3", "azure"}, (
        f"{overlay} sets storageBackend={backend!r}; a cloud install must use cloud storage"
    )


def test_c6_07_gke_replaces_minio_with_gcs_for_mlflow() -> None:
    values = _load_values(OVERLAY_DIR / "values-gke.yaml")
    bucket = (values.get("config") or {}).get("gcsBucket")
    assert bucket
    assert (values.get("minio") or {}).get("enabled") is False
    assert (values.get("mlflow") or {}).get("artifactsDestination") == (
        f"gs://{bucket}/mlflow-artifacts"
    )


def test_c6_08_cloud_storage_backend_wins_over_minio() -> None:
    """The helper adopts MinIO only when no real cloud backend is configured.

    Someone enabling MinIO on a cloud cluster for a side use must not silently
    redirect the app's datasets into it.
    """
    helpers = _read(TEMPLATE_DIR / "_helpers.tpl")
    assert 'define "avaloka.minioIsStorageBackend"' in helpers
    guard = helpers[helpers.index('define "avaloka.minioIsStorageBackend"'):]
    guard = guard[: guard.index("{{- end -}}")]
    assert 'has $backend (list "" "local")' in guard, (
        "MinIO adoption is not restricted to the local/unset storage backend"
    )


def test_c6_09_configmap_publishes_the_s3_endpoint_under_both_names() -> None:
    """S3_ENDPOINT_URL for the app, AWS_ENDPOINT_URL for botocore itself."""
    text = _read(TEMPLATE_DIR / "configmap.yaml")
    for key in ("S3_ENDPOINT_URL", "AWS_ENDPOINT_URL", "S3_BUCKET", "STORAGE_BACKEND"):
        assert key in text, f"configmap.yaml never sets {key}"
    assert 'include "avaloka.storageBackend"' in text, (
        "STORAGE_BACKEND is hardcoded rather than resolved through the MinIO helper"
    )


def test_c6_10_minio_credentials_reach_s3_clients_as_aws_env() -> None:
    """MinIO's root credentials are published as the standard AWS env names.

    boto3, s3fs and pyarrow all read those, which is what lets the app code stay
    unaware that it is talking to MinIO rather than AWS.
    """
    text = _read(TEMPLATE_DIR / "secret.yaml")
    assert "AWS_ACCESS_KEY_ID" in text and "AWS_SECRET_ACCESS_KEY" in text
    assert 'include "avaloka.minioIsStorageBackend"' in text, (
        "secret.yaml does not switch credentials on whether MinIO backs the store"
    )


@pytest.mark.parametrize("group", ["head", "worker"])
def test_c6_11_ray_pods_inherit_the_storage_wiring(group: str) -> None:
    """Ray head and workers read the same s3:// URIs, so they need the same env.

    A worker without S3_ENDPOINT_URL resolves the bucket against real AWS and
    fails with NoSuchBucket — which surfaces as a missing dataset, not as the
    configuration error it is.
    """
    spec = yaml.safe_load_all(_read(RAY_CLUSTER))
    cluster = next(d for d in spec if d and d.get("kind") == "RayCluster")
    if group == "head":
        containers = cluster["spec"]["headGroupSpec"]["template"]["spec"]["containers"]
    else:
        pod_spec = cluster["spec"]["workerGroupSpecs"][0]["template"]["spec"]
        containers = pod_spec["containers"]

    if group == "head":
        pod_spec = cluster["spec"]["headGroupSpec"]["template"]["spec"]

    assert pod_spec.get("serviceAccountName") == "avaloka", (
        f"ray {group} does not use the Workload Identity-enabled avaloka KSA"
    )

    env_from = containers[0].get("envFrom") or []
    sources = {
        (ref["name"], key)
        for entry in env_from
        for key, ref in entry.items()
        if key in {"configMapRef", "secretRef"}
    }
    assert ("avaloka-config", "configMapRef") in sources, f"ray {group} does not read avaloka-config"
    assert ("avaloka-secrets", "secretRef") in sources, f"ray {group} does not read avaloka-secrets"
    # Ray must remain usable without the avaloka chart installed alongside it.
    assert all(
        ref.get("optional") is True
        for entry in env_from
        for key, ref in entry.items()
        if key in {"configMapRef", "secretRef"}
    ), f"ray {group} hard-requires the avaloka ConfigMap/Secret"


@pytest.mark.parametrize("group", ["head", "worker"])
def test_c6_11_ray_serve_pods_use_workload_identity_service_account(group: str) -> None:
    service = yaml.safe_load(_read(RAY_SERVICE))
    config = service["spec"]["rayClusterConfig"]
    if group == "head":
        pod_spec = config["headGroupSpec"]["template"]["spec"]
    else:
        pod_spec = config["workerGroupSpecs"][0]["template"]["spec"]
    assert pod_spec.get("serviceAccountName") == "avaloka"


def test_c6_12_kind_maps_the_minio_console_port() -> None:
    """The console NodePort the minikube/kind overlay picks is reachable from the host."""
    console_port = (
        ((_load_values(OVERLAY_DIR / "values-minikube.yaml").get("minio") or {}).get("service") or {})
        .get("consoleNodePort")
    )
    assert console_port, "values-minikube.yaml sets no MinIO console NodePort"
    kind_cfg = yaml.safe_load(_read(KIND_CLUSTER))
    mapped = {
        m.get("containerPort")
        for node in kind_cfg.get("nodes", [])
        for m in (node.get("extraPortMappings") or [])
    }
    assert console_port in mapped, (
        f"MinIO console NodePort {console_port} is not in kind extraPortMappings {sorted(mapped)}"
    )


# ------------------------------------------------------------------ app side
def test_c6_13_settings_reads_an_s3_endpoint_override() -> None:
    """Settings exposes s3_endpoint_url from either env name.

    Settings' fields are dataclass defaults evaluated at import, so each case
    reloads the module. load_dotenv is stubbed out for the duration: the module
    calls it at import, and a developer's local .env would otherwise leak values
    into the cleared-environment case below.
    """
    import app.core.settings as settings_mod

    def _reload_with(env: dict, *, clear: bool):
        with mock.patch.dict(os.environ, env, clear=clear), \
             mock.patch.object(settings_mod, "load_dotenv", lambda *a, **k: False):
            return importlib.reload(settings_mod).Settings()

    try:
        assert _reload_with({"S3_ENDPOINT_URL": "http://minio:9000"}, clear=False).s3_endpoint_url == (
            "http://minio:9000"
        )
        # AWS_ENDPOINT_URL alone must work: that is the name botocore itself reads,
        # so a deployment may well set only it.
        assert _reload_with({"AWS_ENDPOINT_URL": "http://other:9000"}, clear=True).s3_endpoint_url == (
            "http://other:9000"
        )
        # Unset on both names means real AWS S3, not a bogus endpoint.
        assert _reload_with({}, clear=True).s3_endpoint_url == ""
    finally:
        importlib.reload(settings_mod)


def test_c6_14_blob_store_factory_forwards_the_endpoint() -> None:
    """init_blob_store passes endpoint_url through — without it MinIO is unreachable."""
    source = _read(REPO_ROOT / "app" / "services" / "storage_service.py")
    constructions = re.findall(r"S3BlobStore\((?:[^()]|\([^()]*\))*\)", source, re.DOTALL)
    assert constructions, "no S3BlobStore construction found in storage_service.py"
    for call in constructions:
        assert "endpoint_url" in call, (
            "an S3BlobStore is built without endpoint_url, so it always targets AWS:\n"
            f"{call}"
        )


def test_c6_15_ray_loader_passes_the_endpoint_to_fsspec_and_pyarrow() -> None:
    """Both S3 read paths honour the endpoint override.

    pyarrow's S3FileSystem does not read AWS_ENDPOINT_URL the way botocore does,
    so it needs endpoint_override passed explicitly or Ray reads go to AWS.
    """
    for path in LOADER_COPIES:
        text = _read(path)
        assert "endpoint_url" in text, f"{path.name}: fsspec options carry no endpoint_url"
        assert "endpoint_override" in text, (
            f"{path.name}: pyarrow S3FileSystem gets no endpoint_override, so it ignores MinIO"
        )


def test_c6_16_loader_copies_stay_identical() -> None:
    """The two loader/url.py copies are byte-identical.

    The inference image vendors its own copy; if they drift, a dataset that loads
    in the agent fails in the deployed inference service.
    """
    first, second = (_read(p) for p in LOADER_COPIES)
    assert first == second, (
        "app/agents/mta_v2/loader/url.py and the inference_service_image copy have diverged"
    )
