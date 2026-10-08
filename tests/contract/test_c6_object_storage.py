"""C6 — in-cluster object storage (SeaweedFS) contract.

Clusterless: values are parsed with ``yaml.safe_load``, templates are scanned
line-wise, and the app-side legs import the real modules. The last section
(c6_17 onwards) renders the chart with ``helm template`` and asserts on the parsed
objects, because the two properties that make the store safe -- S3 credential
enforcement and who may reach the S3 port -- are properties of the render, and a
scan of template text cannot fail on them.

The failure this suite exists to prevent is the quiet one. A local install with
no shared data plane looks completely healthy — the API accepts an upload, writes
it to its own pod filesystem, and only later does a Ray worker on another node
fail to find the file. Likewise an S3 client with no ``endpoint_url`` does not
error at startup: it resolves the bucket against real AWS and returns
NoSuchBucket, which reads as a missing dataset rather than a misconfiguration.

So the pins here are: the in-cluster store exists and is durable; the chart wires
the endpoint and credentials everywhere that reads ``s3://`` (API, LangGraph, Ray
head and workers); and a real cloud backend always wins over it.

The store is SeaweedFS as of the MinIO migration -- MinIO's images stopped being
pullable from every public registry, including by digest. ``minio.yaml`` is
retained in-tree and values-gated for one release as the rollback path, so the
MinIO-shaped pins below are kept pointing at it deliberately rather than deleted:
they are what says the rollback path still renders a complete store.

Two pins are specific to SeaweedFS rather than inherited from MinIO. Its bucket
hook has to treat ``BucketAlreadyOwnedByYou`` as success, because re-creating an
existing bucket raises there where MinIO's ``mc mb --ignore-existing`` was silent.
And ``weed server -s3`` listens on eleven ports of which only the S3 port checks
credentials, so the NetworkPolicy restricting ingress to that port is part of the
contract, not a hardening nicety -- without it the migration would reintroduce
the unauthenticated-write exposure it was performed to escape.
"""
from __future__ import annotations

import base64
import importlib
import json
import os
import pathlib
import re
import shutil
import subprocess
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

SEAWEEDFS_TEMPLATE = TEMPLATE_DIR / "seaweedfs.yaml"
#: Retained for one release as the rollback path; no longer the default store.
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
@pytest.mark.parametrize(
    "template,gate",
    [
        (SEAWEEDFS_TEMPLATE, "{{- if .Values.seaweedfs.enabled }}"),
        (MINIO_TEMPLATE, "{{- if .Values.minio.enabled }}"),
    ],
    ids=["seaweedfs", "minio"],
)
def test_c6_01_object_store_template_exists_and_is_values_gated(
    template: pathlib.Path, gate: str
) -> None:
    """Each store template gates its workloads on its own .Values toggle.

    This used to require the gate on line 1, which failed once the template
    opened with a `$fullname :=` assignment -- a formatting detail, not a
    gating one.

    minio.yaml keeps its PersistentVolumeClaim above the gate deliberately:
    turning MinIO off must not delete the bucket it was holding, and with MinIO
    now off by default that claim is what stops `helm upgrade` from reaping an
    existing install's data. seaweedfs.yaml puts its claim inside the gate and
    relies on helm.sh/resource-policy: keep instead, so a cluster that never ran
    it does not carry a stray claim. Either way no *workload* escapes the gate,
    which is what this asserts.
    """
    assert template.is_file(), f"missing {template}"
    lines = _read(template).splitlines()

    idx = next((i for i, line in enumerate(lines)
                if line.lstrip().startswith(gate)), None)
    assert idx is not None, f"{template.name} has no `{gate}` gate"

    workloads = {"Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob", "Pod"}
    ungated = [
        line.split(":", 1)[1].strip()
        for line in lines[:idx]
        if line.startswith("kind:") and line.split(":", 1)[1].strip() in workloads
    ]
    assert not ungated, (
        f"{template.name} emits {ungated} before its gate, so they deploy even "
        f"with the toggle off"
    )


def test_c6_02_seaweedfs_is_on_by_default_and_minio_is_the_rollback(
    base_values: dict,
) -> None:
    """The base/local chart gives MLflow a SeaweedFS artifact store by default.

    MinIO must be off: it is retained only so an on-prem install can roll back,
    and its images are no longer pullable from any public registry. Enabling both
    is refused at render time -- they would claim the same role.
    """
    assert (base_values.get("seaweedfs") or {}).get("enabled") is True
    assert (base_values.get("minio") or {}).get("enabled") is False


@pytest.mark.parametrize(
    "kind",
    ["Deployment", "Service", "PersistentVolumeClaim", "Secret", "Job", "NetworkPolicy"],
)
def test_c6_03_seaweedfs_ships_a_complete_object(kind: str) -> None:
    """Store, network identity, durable volume, credentials, bucket creation, and
    the ingress restriction that makes the store safe to run at all."""
    assert f"kind: {kind}" in _read(SEAWEEDFS_TEMPLATE), (
        f"seaweedfs.yaml renders no {kind}"
    )


@pytest.mark.parametrize("kind", ["Deployment", "Service", "PersistentVolumeClaim", "Secret", "Job"])
def test_c6_03_minio_rollback_path_stays_complete(kind: str) -> None:
    """The retained MinIO path still renders a whole store, so rollback works.

    No NetworkPolicy here: MinIO serves S3 on one port and authenticates it,
    which is exactly the property SeaweedFS lacks and needs a policy to recover.
    """
    assert f"kind: {kind}" in _read(MINIO_TEMPLATE), f"minio.yaml renders no {kind}"


@pytest.mark.parametrize(
    "template,values_key",
    [(SEAWEEDFS_TEMPLATE, "seaweedfs"), (MINIO_TEMPLATE, "minio")],
    ids=["seaweedfs", "minio"],
)
def test_c6_04_object_store_storage_is_durable_and_not_rolling(
    base_values: dict, template: pathlib.Path, values_key: str
) -> None:
    """A PVC, and Recreate — a RWO volume cannot be held by two pods at once.

    With the default RollingUpdate the replacement pod waits forever on a volume
    the outgoing pod still has mounted, and the upgrade hangs rather than fails.

    For SeaweedFS this is sharper than a hung upgrade. The filer's embedded
    LevelDB takes an exclusive lock on the data directory, so an overlapping pod
    does not wait -- it exits 255 and crash-loops, refusing S3 connections while
    it does. Recreate is the mitigation and must not be relaxed.
    """
    text = _read(template)
    assert "kind: PersistentVolumeClaim" in text
    assert "emptyDir" not in text, (
        f"{template.name}: data on emptyDir would lose datasets on reschedule"
    )
    assert re.search(r"strategy:\s*\n\s*(#.*\n\s*)*type:\s*Recreate", text), (
        f"{template.name} uses the default RollingUpdate against a ReadWriteOnce volume"
    )
    size = ((base_values.get(values_key) or {}).get("persistence") or {}).get("size")
    assert size, f"no {values_key}.persistence.size configured"


def test_c6_05_bucket_creation_is_idempotent() -> None:
    """The bucket hook survives re-runs and races with a still-starting store.

    SeaweedFS needs an extra step MinIO did not. `mc mb --ignore-existing` was
    silent about an existing bucket; `aws s3api create-bucket` is not -- SeaweedFS
    raises BucketAlreadyOwnedByYou (HTTP 409), which is the AWS-standard
    behaviour everywhere except us-east-1. So the hook has to catch that code, or
    every `helm upgrade` fails on the second install.
    """
    text = _read(SEAWEEDFS_TEMPLATE)
    assert "post-install,post-upgrade" in text, "bucket hook does not run on upgrade"
    assert "BucketAlreadyOwnedByYou" in text, (
        "the bucket hook does not treat an existing bucket as success, so it "
        "fails on every upgrade after the first"
    )
    assert re.search(r"until aws .*list-buckets", text), (
        "hook does not wait for the store to accept connections"
    )
    # The idempotent branch must not swallow real failures.
    assert "exit 1" in text, (
        "the bucket hook catches every error, so an unreachable endpoint would "
        "report success"
    )


def test_c6_05_minio_rollback_bucket_hook_is_idempotent() -> None:
    """The retained MinIO hook keeps the property it always had."""
    text = _read(MINIO_TEMPLATE)
    assert "--ignore-existing" in text, "mc mb would fail the hook on an existing bucket"
    assert "post-install,post-upgrade" in text, "bucket hook does not run on upgrade"
    assert "until mc alias set" in text, "hook does not wait for MinIO to accept connections"


def test_c6_05_seaweedfs_ingress_is_restricted_to_the_s3_port(base_values: dict) -> None:
    """`weed server -s3` exposes eleven ports; only the S3 port checks credentials.

    The other ten enforce nothing. The filer in particular serves a browsable
    directory UI and ACCEPTS WRITES with no credentials, which is functionally the
    MinIO CVE this migration was performed to escape -- so shipping SeaweedFS
    without this policy would relocate the vulnerability rather than remove it.

    Verified live on kind (kindnet enforces it): from a co-tenant pod, the filer,
    master, volume and metrics ports all returned HTTP 000 against the store's POD
    IP while the S3 port returned 403. Restricting the Service is NOT a substitute
    and this test does not accept one -- pod IPs are directly routable, and with
    only the S3 port published those ports still answered 200 on the pod IP.
    """
    text = _read(SEAWEEDFS_TEMPLATE)
    assert "kind: NetworkPolicy" in text, (
        "seaweedfs.yaml ships no NetworkPolicy; the unauthenticated filer, master "
        "and volume ports are reachable from any pod in the cluster"
    )
    policy = text[text.index("kind: NetworkPolicy"):]
    assert "Ingress" in policy, "the policy declares no Ingress policyType"
    # The allowed port must be the configured S3 port, not a literal that can
    # drift away from it.
    assert re.search(r"-\s*port:\s*\{\{\s*\.Values\.seaweedfs\.port\s*\}\}", policy), (
        "the NetworkPolicy does not key its allowed port on .Values.seaweedfs.port"
    )
    # An empty podSelector would make this a cluster-wide policy by accident.
    assert re.search(r"podSelector:\s*\n\s+matchLabels:", policy), (
        "the policy has no podSelector matchLabels, so it would apply to every pod"
    )
    # It must not be switchable off: that is the whole point.
    assert "networkPolicy" not in (base_values.get("seaweedfs") or {}), (
        "the NetworkPolicy gained a values toggle; it is mandatory, because "
        "without it the store accepts unauthenticated writes"
    )


# ------------------------------------------------------------- chart wiring
@pytest.mark.parametrize("overlay", LOCAL_OVERLAYS)
def test_c6_06_local_overlays_enable_the_object_store(overlay: str) -> None:
    """Local/on-prem clusters have no cloud bucket, so they must ship MinIO.

    They also have no ReadWriteMany storage class, so a shared volume is not an
    alternative — object storage is the only way Ray workers on other nodes can
    read what the API wrote.
    """
    assert (_load_values(OVERLAY_DIR / overlay).get("seaweedfs") or {}).get("enabled") is True, (
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
    # BOTH gates, not just MinIO's: seaweedfs defaults to enabled, so omitting it
    # here would ship an in-cluster object store onto a cluster that has GCS.
    assert (values.get("seaweedfs") or {}).get("enabled") is False
    assert (values.get("minio") or {}).get("enabled") is False
    assert (values.get("mlflow") or {}).get("artifactsDestination") == (
        f"gs://{bucket}/mlflow-artifacts"
    )


def test_c6_08_cloud_storage_backend_wins_over_the_in_cluster_store() -> None:
    """The helper adopts the in-cluster store only when no cloud backend is set.

    Someone enabling it on a cloud cluster for a side use must not silently
    redirect the app's datasets into it.

    Renamed from avaloka.minioIsStorageBackend with the SeaweedFS migration: the
    guard never had anything to do with MinIO specifically, and a helper called
    `minioIs...` returning true for a SeaweedFS install would read as a bug at
    every call site.
    """
    helpers = _read(TEMPLATE_DIR / "_helpers.tpl")
    assert 'define "avaloka.objectStoreIsStorageBackend"' in helpers
    guard = helpers[helpers.index('define "avaloka.objectStoreIsStorageBackend"'):]
    guard = guard[: guard.index("{{- end -}}")]
    assert 'has $backend (list "" "local")' in guard, (
        "adoption is not restricted to the local/unset storage backend"
    )


def test_c6_09_configmap_publishes_the_s3_endpoint_under_both_names() -> None:
    """S3_ENDPOINT_URL for the app, AWS_ENDPOINT_URL for botocore itself."""
    text = _read(TEMPLATE_DIR / "configmap.yaml")
    for key in ("S3_ENDPOINT_URL", "AWS_ENDPOINT_URL", "S3_BUCKET", "STORAGE_BACKEND"):
        assert key in text, f"configmap.yaml never sets {key}"
    assert 'include "avaloka.storageBackend"' in text, (
        "STORAGE_BACKEND is hardcoded rather than resolved through the MinIO helper"
    )


def test_c6_10_store_credentials_reach_s3_clients_as_aws_env() -> None:
    """The store's credentials are published as the standard AWS env names.

    boto3, s3fs and pyarrow all read those, which is what lets the app code stay
    unaware of which S3-compatible store is behind the endpoint -- and is why this
    migration changed no application code at all.
    """
    text = _read(TEMPLATE_DIR / "secret.yaml")
    assert "AWS_ACCESS_KEY_ID" in text and "AWS_SECRET_ACCESS_KEY" in text
    assert 'include "avaloka.objectStoreIsStorageBackend"' in text, (
        "secret.yaml does not switch credentials on whether the in-cluster store "
        "backs the app's object storage"
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


# test_c6_12_kind_maps_the_minio_console_port was DELETED by the SeaweedFS
# migration rather than adapted, because its subject no longer exists: SeaweedFS
# ships no console, so there is no console NodePort for values-minikube.yaml to
# pick and no port for kind-cluster.yaml to map. The test asserted a real
# property of a real feature; the feature is gone, and a rewritten version would
# have had to assert something else to stay green, which is how a pin becomes
# decoration. Rather than weaken it, the whole premise is retired.
#
# This is a genuine UX regression and worth recording as one -- "browse the
# bucket from the laptop" now means:
#   kubectl port-forward svc/avaloka-seaweedfs 8333:8333
#   aws --endpoint-url http://localhost:8333 s3 ls s3://avaloka/
# (MinIO's community console had itself been gutted in 2025, so what the port
# mapping was originally added for was already mostly gone.)
#
# What replaces it as a pin is test_c6_05_seaweedfs_ingress_is_restricted_to_the
# _s3_port: the reason there is no second published port is the same reason
# there is no console.
def test_c6_12_kind_maps_no_object_store_console_port() -> None:
    """The removed console NodePort stays removed on both sides.

    A stale 30092 mapping in kind-cluster.yaml would be harmless but misleading,
    and a consoleNodePort left in values-minikube.yaml would be silently ignored
    by a chart that no longer reads it.
    """
    minikube = _load_values(OVERLAY_DIR / "values-minikube.yaml")
    for key in ("seaweedfs", "minio"):
        service = ((minikube.get(key) or {}).get("service") or {})
        assert "consoleNodePort" not in service, (
            f"values-minikube.yaml still sets {key}.service.consoleNodePort; "
            "no in-cluster store in this chart serves a console"
        )
    kind_cfg = yaml.safe_load(_read(KIND_CLUSTER))
    mapped = {
        m.get("containerPort")
        for node in kind_cfg.get("nodes", [])
        for m in (node.get("extraPortMappings") or [])
    }
    assert 30092 not in mapped, (
        "kind-cluster.yaml still maps 30092 (the old MinIO console NodePort) to "
        "the host, where nothing listens any more"
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


# ------------------------------------------------- the rendered store (helm)
# Everything above reads template TEXT. That cannot pin the two properties that
# make this store safe to run, because both are properties of what the chart
# RENDERS: a line can sit in a template inside a conditional that never fires,
# and a regex over the text passes on it all the same.
#
# Before these tests existed, deleting `-s3.config=/etc/seaweedfs/s3.json` from
# the container args -- which turns S3 credential checking off ENTIRELY -- left
# the suite byte-identical: no file under tests/ mentioned s3.config or s3.json.
# So these render the chart with the real helm binary and assert on the parsed
# objects.
#
# They are still not the real test. The real test is an unauthenticated S3
# request against a live cluster returning 403; that needs a cluster and belongs
# in the `cluster` tier. What is pinned here is the necessary condition a render
# can see.
S3_CONFIG_FLAG = "-s3.config="
S3_IDENTITIES_KEY = "s3.json"

#: Renders that ship the in-cluster store: the default values, and each overlay
#: for a cluster with no cloud bucket.
STORE_RENDERS = ("default",) + LOCAL_OVERLAYS


def _helm_template(*extra: str) -> list[dict]:
    """`helm template` the chart and return the parsed objects.

    A missing helm binary is a FAILURE under CI, not a skip. A skipped security
    test is indistinguishable from a passing one in a green pipeline, which is
    the exact defect these tests were written to close -- so the only place this
    is allowed to skip is a developer machine with no helm on it.
    """
    helm = shutil.which("helm")
    if helm is None:
        message = (
            "helm is not installed, so the rendered-chart contract (S3 credential "
            "enforcement, object-store NetworkPolicy) cannot be checked"
        )
        if os.environ.get("CI") or os.environ.get("AVALOKA_REQUIRE_HELM"):
            pytest.fail(message + "; install helm in this pipeline")
        pytest.skip(message)
    result = subprocess.run(
        [helm, "template", "avaloka", str(CHART_ROOT), *extra],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"helm template failed:\n{result.stderr}"
    return [doc for doc in yaml.safe_load_all(result.stdout) if isinstance(doc, dict)]


@pytest.fixture(scope="module", params=STORE_RENDERS)
def store_render(request: pytest.FixtureRequest) -> list[dict]:
    if request.param == "default":
        return _helm_template()
    return _helm_template("-f", str(OVERLAY_DIR / request.param))


def _pod_template(obj: dict) -> dict | None:
    kind = obj.get("kind")
    if kind in {"Deployment", "StatefulSet", "DaemonSet", "Job", "ReplicaSet"}:
        return obj["spec"]["template"]
    if kind == "CronJob":
        return obj["spec"]["jobTemplate"]["spec"]["template"]
    if kind == "Pod":
        return obj
    return None


def _workloads(docs: list[dict]) -> list[tuple[str, dict]]:
    """(name, pod template) for every rendered object that creates pods."""
    return [
        (f"{doc['kind']}/{doc['metadata']['name']}", template)
        for doc in docs
        if (template := _pod_template(doc)) is not None
    ]


def _pod_labels(template: dict) -> dict:
    return (template.get("metadata") or {}).get("labels") or {}


def _containers(template: dict) -> list[dict]:
    spec = template["spec"]
    return list(spec.get("initContainers") or []) + list(spec.get("containers") or [])


def _by_kind(docs: list[dict], kind: str) -> dict[str, dict]:
    return {d["metadata"]["name"]: d for d in docs if d.get("kind") == kind}


def _secret_value(secret: dict, key: str) -> str | None:
    if key in (secret.get("stringData") or {}):
        return secret["stringData"][key]
    if key in (secret.get("data") or {}):
        return base64.b64decode(secret["data"][key]).decode("utf-8")
    return None


def _store_pod(docs: list[dict]) -> tuple[str, dict]:
    """The pod template that actually runs `weed server`."""
    stores = [
        (name, template)
        for name, template in _workloads(docs)
        if any((c.get("args") or [None])[0] == "server" and "seaweedfs" in c.get("image", "")
               for c in template["spec"]["containers"])
    ]
    assert len(stores) == 1, (
        f"expected exactly one rendered workload running `weed server`, found "
        f"{[name for name, _ in stores]}"
    )
    return stores[0]


def _selector_matches(selector: dict | None, labels: dict) -> bool:
    """Kubernetes LabelSelector semantics. An empty selector matches everything."""
    selector = selector or {}
    for key, value in (selector.get("matchLabels") or {}).items():
        if labels.get(key) != value:
            return False
    for expr in selector.get("matchExpressions") or []:
        key, operator, values = expr["key"], expr["operator"], expr.get("values") or []
        if operator == "In" and labels.get(key) not in values:
            return False
        if operator == "NotIn" and labels.get(key) in values:
            return False
        if operator == "Exists" and key not in labels:
            return False
        if operator == "DoesNotExist" and key in labels:
            return False
    return True


def test_c6_17_rendered_store_enforces_s3_credentials(store_render: list[dict]) -> None:
    """The rendered `weed server` is started with an identities file that exists.

    `weed server -s3` with no `-s3.config` serves the S3 API to anyone: every
    request is treated as an admin. So the argument is the control, and three
    things have to line up in the RENDER for it to mean anything:

    1. the container args carry `-s3.config=<path>`;
    2. `<path>` is a file projected from a Secret this chart renders -- a flag
       pointing at a path nothing mounts makes weed exit, and one pointing at an
       emptyDir or ConfigMap is not this chart's credentials;
    3. that Secret's identities file names real credentials, and none of them is
       the `anonymous` identity, which SeaweedFS treats as "no signature needed".

    CONTROL (run when this test was written): deleting the `-s3.config` line from
    templates/seaweedfs.yaml fails this test on every render, and so does
    removing the s3-config volumeMount. See the PR for the recorded output.
    """
    name, template = _store_pod(store_render)
    container = next(c for c in template["spec"]["containers"]
                     if (c.get("args") or [None])[0] == "server")
    args = [str(a) for a in container["args"]]

    assert "-s3" in args, f"{name}: `weed server` is not started with the S3 gateway"
    config_args = [a for a in args if a.startswith(S3_CONFIG_FLAG)]
    assert len(config_args) == 1, (
        f"{name}: expected exactly one {S3_CONFIG_FLAG}<path> argument, got "
        f"{config_args or 'none'}. Without it SeaweedFS performs NO S3 credential "
        f"check: any client that can reach the port is an admin. args={args}"
    )
    config_path = pathlib.PurePosixPath(config_args[0][len(S3_CONFIG_FLAG):])
    assert config_path.is_absolute(), f"{name}: -s3.config is not an absolute path: {config_path}"

    # (2) the path resolves, through a volumeMount, to a key of a rendered Secret.
    volumes = {v["name"]: v for v in template["spec"].get("volumes") or []}
    secrets = _by_kind(store_render, "Secret")
    resolved = None
    for mount in container.get("volumeMounts") or []:
        mount_path = pathlib.PurePosixPath(mount["mountPath"])
        if mount_path != config_path and mount_path not in config_path.parents:
            continue
        volume = volumes.get(mount["name"]) or {}
        source = volume.get("secret")
        assert source, (
            f"{name}: {config_path} is served by volume {mount['name']!r}, which is "
            f"not a Secret ({sorted(set(volume) - {'name'})}); the identities file "
            "holds the secret key in cleartext"
        )
        relative = (
            pathlib.PurePosixPath(mount["subPath"]) if mount_path == config_path and mount.get("subPath")
            else config_path.relative_to(mount_path)
        )
        secret = secrets.get(source["secretName"])
        assert secret is not None, (
            f"{name}: mounts Secret {source['secretName']!r}, which this chart does not render"
        )
        # With `items`, only the listed keys are projected, at the listed paths.
        # Without it every key appears under its own name.
        projection = (
            {item["path"]: item["key"] for item in source["items"]}
            if source.get("items") else
            {key: key for key in {**(secret.get("data") or {}), **(secret.get("stringData") or {})}}
        )
        key = projection.get(str(relative))
        assert key is not None, (
            f"{name}: Secret {source['secretName']!r} is mounted at {mount_path} but "
            f"projects {sorted(projection)}, so {config_path} does not exist in the pod"
        )
        resolved = (source["secretName"], key, secret)
        break
    assert resolved is not None, (
        f"{name}: -s3.config points at {config_path}, but no volumeMount provides "
        f"it; mounts are {[m['mountPath'] for m in container.get('volumeMounts') or []]}"
    )
    secret_name, key, secret = resolved
    assert key == S3_IDENTITIES_KEY, (
        f"{name}: {config_path} is Secret key {key!r}, expected {S3_IDENTITIES_KEY!r}"
    )

    # (3) the file is a usable, non-anonymous identities document.
    raw = _secret_value(secret, key)
    assert raw, f"Secret {secret_name!r} has an empty {key}"
    identities = (json.loads(raw).get("identities") or [])
    assert identities, f"Secret {secret_name!r}: {key} declares no identities"
    for identity in identities:
        assert str(identity.get("name", "")).lower() != "anonymous", (
            f"Secret {secret_name!r}: {key} declares an `anonymous` identity, which "
            "SeaweedFS grants to requests carrying no signature at all"
        )
        credentials = identity.get("credentials") or []
        assert credentials, f"identity {identity.get('name')!r} has no credentials"
        for credential in credentials:
            assert credential.get("accessKey") and credential.get("secretKey"), (
                f"identity {identity.get('name')!r} has a blank access or secret key"
            )

    # ...and they are the credentials the clients are actually handed, or the
    # store would be authenticated and unusable.
    client_key = _secret_value(secret, "AWS_ACCESS_KEY_ID")
    accepted = {c["accessKey"] for i in identities for c in i["credentials"]}
    assert client_key in accepted, (
        f"Secret {secret_name!r} hands clients AWS_ACCESS_KEY_ID={client_key!r}, "
        f"which is not an identity in {key} ({sorted(accepted)})"
    )


def _store_endpoint_markers(docs: list[dict]) -> set[str]:
    """Strings that mean "this workload is pointed at the in-cluster store"."""
    services = _by_kind(docs, "Service")
    _, store = _store_pod(docs)
    markers = set()
    for service in services.values():
        selector = service["spec"].get("selector") or {}
        if selector and _selector_matches({"matchLabels": selector}, _pod_labels(store)):
            for port in service["spec"]["ports"]:
                markers.add(f"{service['metadata']['name']}:{port['port']}")
    assert markers, "no rendered Service selects the store pod"
    return markers


def _s3_clients(docs: list[dict]) -> dict[str, dict]:
    """Rendered workloads wired to the store, derived from the render itself.

    A workload is a client if any container is handed the store's Service address
    -- directly as an env value, or through an envFrom ConfigMap that carries it.
    Deriving this rather than listing it is the point: a hand-written list here
    would agree with the hand-written list in the NetworkPolicy by construction.
    """
    markers = _store_endpoint_markers(docs)
    configmaps = _by_kind(docs, "ConfigMap")

    def _mentions(value: object) -> bool:
        return any(marker in str(value) for marker in markers)

    carrying = {
        name for name, cm in configmaps.items()
        if any(_mentions(v) for v in (cm.get("data") or {}).values())
    }
    clients = {}
    for name, template in _workloads(docs):
        for container in _containers(template):
            direct = any(_mentions(e.get("value", "")) for e in container.get("env") or [])
            direct = direct or any(_mentions(part) for part in
                                   (container.get("command") or []) + (container.get("args") or []))
            via_config = any(
                (entry.get("configMapRef") or {}).get("name") in carrying
                for entry in container.get("envFrom") or []
            )
            if direct or via_config:
                clients[name] = _pod_labels(template)
    return clients


def test_c6_18_rendered_network_policy_admits_only_the_s3_clients(
    store_render: list[dict],
) -> None:
    """The policy names WHO may reach the S3 port, and names every real client.

    An ingress rule with `ports` and no `from` admits every pod in every
    namespace. Both directions are pinned, because either mistake is silent:

    * too loose -- a rule with no `from`, or a peer that selects everything --
      looks identical on a cluster to a tight one until someone tests from a pod
      that should have been refused;
    * too tight -- a real client left off the list -- does not fail at install
      either. It fails later as a dataset that cannot be read or an MLflow
      artifact that cannot be written.

    The client set is derived from the render (who is handed the store's address),
    not restated here.
    """
    name, store = _store_pod(store_render)
    store_labels = _pod_labels(store)
    policies = [
        p for p in _by_kind(store_render, "NetworkPolicy").values()
        if _selector_matches(p["spec"].get("podSelector"), store_labels)
    ]
    assert len(policies) == 1, (
        f"expected exactly one NetworkPolicy selecting {name}, found "
        f"{[p['metadata']['name'] for p in policies]}"
    )
    spec = policies[0]["spec"]
    assert spec["podSelector"], "an empty podSelector applies the policy to every pod"
    assert "Ingress" in (spec.get("policyTypes") or [])

    s3_port = next(p["containerPort"] for c in store["spec"]["containers"]
                   for p in c.get("ports") or [] if p.get("name") == "s3")
    rules = spec.get("ingress") or []
    assert rules, "the policy has no ingress rule, so nothing can reach the store at all"

    peers: list[dict] = []
    for rule in rules:
        ports = [(p.get("port"), p.get("protocol", "TCP")) for p in rule.get("ports") or []]
        assert ports == [(s3_port, "TCP")], (
            f"an ingress rule opens {ports or 'ALL ports'}; only TCP {s3_port} checks "
            "credentials -- the filer, master and volume ports accept unauthenticated writes"
        )
        assert rule.get("from"), (
            f"an ingress rule on port {s3_port} has no `from`, so it admits every pod "
            "in every namespace"
        )
        peers.extend(rule["from"])

    for peer in peers:
        assert peer.get("podSelector") or peer.get("namespaceSelector") or peer.get("ipBlock"), (
            f"empty NetworkPolicy peer {peer!r}"
        )
        if "ipBlock" in peer:
            assert peer["ipBlock"]["cidr"] not in {"0.0.0.0/0", "::/0"}, (
                "an ipBlock peer admits every address"
            )
        else:
            # `podSelector: {}` is every pod in the namespace; a namespaceSelector
            # with no podSelector is every pod in the selected namespaces.
            assert peer.get("podSelector"), (
                f"peer {peer!r} selects pods without restricting which ones"
            )

    def _admitted(labels: dict) -> bool:
        # The chart renders into one namespace, so only same-namespace peers
        # (no namespaceSelector) can admit a workload from this render.
        return any(
            "namespaceSelector" not in peer and "podSelector" in peer
            and _selector_matches(peer["podSelector"], labels)
            for peer in peers
        )

    clients = _s3_clients(store_render)
    assert clients, "no rendered workload is wired to the store; the derivation is broken"
    locked_out = sorted(n for n, labels in clients.items() if not _admitted(labels))
    assert not locked_out, (
        f"{locked_out} are configured with the store's address but the "
        f"NetworkPolicy does not admit them, so their S3 calls would time out. "
        "Add their app.kubernetes.io/component to the `from` list in "
        "templates/seaweedfs.yaml"
    )

    # Everything else this chart renders must be refused -- otherwise the list is
    # decoration and the policy is still "any pod in the release".
    strangers = sorted(
        n for n, template in _workloads(store_render)
        if n not in clients and _pod_labels(template) != store_labels
        and _admitted(_pod_labels(template))
    )
    assert not strangers, (
        f"{strangers} have no S3 configuration but are admitted to the store's S3 port"
    )

    # Ray head and worker pods are created by KubeRay outside this chart and read
    # the same s3:// URIs (test_c6_11). They carry none of the release's labels,
    # so they are admitted by the label KubeRay puts on every pod it creates.
    assert _admitted({"ray.io/node-type": "worker", "ray.io/cluster": "avaloka-raycluster"}), (
        "Ray workers are not admitted to the object store, so they cannot read datasets"
    )
    assert not _admitted({"app": "some-other-tenant"}), (
        "an unrelated pod in the release namespace is admitted to the S3 port"
    )


def test_c6_18_bucket_hook_is_not_a_backend_of_the_store_service(
    store_render: list[dict],
) -> None:
    """The hook pod must not carry the store's own labels.

    It did: with `component: seaweedfs` the short-lived aws-cli pod matched the
    Service selector and was a backend for port 8333, where it listens on
    nothing, and it could not be distinguished from the store in a `from` clause.
    """
    _, store = _store_pod(store_render)
    service_selectors = [
        s["spec"]["selector"] for s in _by_kind(store_render, "Service").values()
        if s["spec"].get("selector")
        and _selector_matches({"matchLabels": s["spec"]["selector"]}, _pod_labels(store))
    ]
    for workload, template in _workloads(store_render):
        if template is store:
            continue
        for selector in service_selectors:
            assert not _selector_matches({"matchLabels": selector}, _pod_labels(template)), (
                f"{workload} matches the object store's Service selector {selector}, "
                "so the Service load-balances S3 requests to a pod that is not the store"
            )


def test_c6_18_network_policy_follows_a_relocated_ray_namespace() -> None:
    """config.mtaRayNamespace moves Ray out of the release namespace.

    A bare podSelector only matches pods in the policy's own namespace, so
    without a namespaceSelector peer the relocated Ray cluster is locked out.
    """
    docs = _helm_template("--set-string", "config.mtaRayNamespace=ray-elsewhere")
    policy = _by_kind(docs, "NetworkPolicy")["avaloka-seaweedfs-s3-only"]
    peers = [peer for rule in policy["spec"]["ingress"] for peer in rule["from"]]
    cross = [p for p in peers if "namespaceSelector" in p]
    assert cross, "no peer admits Ray pods from the configured mtaRayNamespace"
    assert cross[0]["namespaceSelector"] == {
        "matchLabels": {"kubernetes.io/metadata.name": "ray-elsewhere"}
    }
    assert _selector_matches(cross[0]["podSelector"], {"ray.io/node-type": "head"})
    assert not _selector_matches(cross[0]["podSelector"], {"app": "anything"})


def test_c6_19_extra_ingress_peers_are_additive_only() -> None:
    """seaweedfs.extraIngressFrom appends peers; it cannot remove the built-in ones."""
    docs = _helm_template(
        "--set", "seaweedfs.extraIngressFrom[0].podSelector.matchLabels.app=notebook",
    )
    policy = _by_kind(docs, "NetworkPolicy")["avaloka-seaweedfs-s3-only"]
    (rule,) = policy["spec"]["ingress"]
    assert {"podSelector": {"matchLabels": {"app": "notebook"}}} in rule["from"]
    assert len(rule["from"]) >= 3, "the extra peer replaced the built-in client list"
    assert [p["port"] for p in rule["ports"]] == [8333]
