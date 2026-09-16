"""C2 — clusterless chart invariants (plan suites E1, E2.07, E6.08/E6.09, E7.09, E11.10/E11.12, E14.06/E14.10).

Everything here is a pure filesystem/YAML assertion over ``deploy/``: no cluster,
no kubectl, no helm binary. Helm templates are scanned line-wise / by regex
(``yaml.safe_load`` chokes on ``{{ }}``); the values files and ``Chart.yaml``
carry no templating and are parsed with ``yaml.safe_load``. Templates are located
by scanning the whole chart tree for their component label rather than by
filename, so a moved template cannot silently pass.

Runs identically in every mode — nothing here is deployed-only. The single
mode-dependent leg is E1.05 (webui/local-Supabase templates), which lives only on
branch ``ui-k8s-deploy-1.5.2`` and skips when those templates are absent.
"""
from __future__ import annotations

import pathlib
import re
from typing import Iterator, Optional

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEPLOY_ROOT = REPO_ROOT / "deploy"
CHART_ROOT = DEPLOY_ROOT / "helm" / "avaloka"
CHART_VALUES = CHART_ROOT / "values.yaml"
OVERLAY_DIR = CHART_ROOT / "values"
RAY_MANIFEST_DIR = DEPLOY_ROOT / "helm" / "ray"
DEPLOY_README = DEPLOY_ROOT / "README.md"
LEGACY_CHART_DIRS = (DEPLOY_ROOT / "avaloka", DEPLOY_ROOT / "avaloka-router")

# Overlays intended for a real cluster; minikube is the dev/local overlay.
NON_DEV_OVERLAYS = ("values-gke.yaml", "values-eks.yaml", "values-aks.yaml", "values-onprem.yaml")

MUTABLE_TAGS = frozenset({"latest", "main", "master", "dev", "develop", "stable", "edge", ""})

# Workloads the chart ships everywhere, and the additions the UI merge brings in.
# Split so the pin holds on both branches without losing its E6.08 evidence value.
BASE_CHART_COMPONENTS = frozenset({"api", "langgraph", "redis", "chroma", "postgres", "ui"})
UI_MERGE_CHART_COMPONENTS = frozenset(
    {"webui", "supabase-db", "supabase-auth", "supabase-rest", "supabase-kong"}
)
# `gateway` labels the cloud load-balancer objects (Gateway, HTTPRoute and the
# provider policy CRDs), not a pod. It runs nothing in-cluster, so it is pinned
# apart from the workload sets above — the executor check below still applies.
INGRESS_CHART_COMPONENTS = frozenset({"gateway"})
# Components that shipped without ever being added to a pin: `mcp` (multi-tenant
# MCP server) and `supabase-functions` (edge runtime). Recorded here so the
# inventory check is meaningful again rather than permanently red.
UNPINNED_LEGACY_COMPONENTS = frozenset({"mcp", "supabase-functions"})
# Shared model/data plane: local installs use MinIO while cloud overlays may use
# managed object storage; MLflow provides the tracking and model-registry API.
STORAGE_CHART_COMPONENTS = frozenset({"minio", "mlflow"})

_CELERY_WORKER_RE = re.compile(r"celery[^\n]{0,120}\b(worker|beat)\b", re.IGNORECASE)
_NETWORK_POLICY_RE = re.compile(r"^\s*kind:\s*NetworkPolicy\s*$", re.MULTILINE)
_AUTOMOUNT_OFF_RE = re.compile(r"^\s*automountServiceAccountToken:\s*false\s*$", re.MULTILINE)
_SHARED_SA_RE = re.compile(r'serviceAccountName:\s*\{\{\s*include\s+"avaloka\.serviceAccountName"')
_DEPRECATION_RE = re.compile(r"deprecat|legacy|do not use|no longer maintained", re.IGNORECASE)


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _files_under(root: pathlib.Path) -> Iterator[pathlib.Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _chart_files() -> list[pathlib.Path]:
    return list(_files_under(CHART_ROOT))


def _deploy_files() -> list[pathlib.Path]:
    return list(_files_under(DEPLOY_ROOT))


def _grep_deploy(pattern: str) -> list[str]:
    """Repo-relative paths of every file under deploy/ containing `pattern` (case-insensitive)."""
    rx = re.compile(pattern, re.IGNORECASE)
    return [
        str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        for path in _deploy_files()
        if rx.search(_read(path))
    ]


def _template_for_component(component: str) -> Optional[pathlib.Path]:
    """Chart file carrying `app.kubernetes.io/component: <component>` on a workload."""
    needle = f"app.kubernetes.io/component: {component}"
    for path in _chart_files():
        if needle in _read(path):
            return path
    return None


def _chart_components() -> set[str]:
    rx = re.compile(r"^\s*app\.kubernetes\.io/component:\s*(\S+)\s*$", re.MULTILINE)
    found: set[str] = set()
    for path in _chart_files():
        found.update(rx.findall(_read(path)))
    return found


def _load_values(path: pathlib.Path) -> dict:
    data = yaml.safe_load(_read(path))
    return data if isinstance(data, dict) else {}


def _is_immutable_image(repository: str, tag: str) -> bool:
    if "@sha256:" in repository or "@sha256:" in tag:
        return True
    return tag.strip().lower() not in MUTABLE_TAGS


# --------------------------------------------------------------------------- E1
def test_e1_01_real_chart_is_present_and_identified() -> None:
    """deploy/helm/avaloka is the real chart: Chart.yaml (plain YAML) names it `avaloka` and templates exist."""
    assert CHART_ROOT.is_dir(), f"real chart missing at {CHART_ROOT}"
    chart = _load_values(CHART_ROOT / "Chart.yaml")
    assert chart.get("name") == "avaloka", f"unexpected chart name: {chart.get('name')!r}"
    assert chart.get("apiVersion") == "v2"
    templates = sorted(p.name for p in (CHART_ROOT / "templates").glob("*.yaml"))
    for required in ("configmap.yaml", "deployment.yaml", "secret.yaml", "service.yaml"):
        assert required in templates, f"{required} missing from the chart; got {templates}"


def test_e1_02_pins_chart_workload_inventory() -> None:
    """Pins the exact set of component labels the chart ships (line-scan of every chart file).

    Pins the workload set behind E6.08: the chart runs API + langgraph + redis +
    chroma + postgres + ui, plus the webui/Supabase stack once the UI merge is in
    the tree - and no task-executing component in either case. Adding a component
    must update this pin.
    """
    components = _chart_components()
    assert BASE_CHART_COMPONENTS <= components, (
        f"chart lost a core workload: missing {sorted(BASE_CHART_COMPONENTS - components)}"
    )
    unexpected = (
        components
        - BASE_CHART_COMPONENTS
        - UI_MERGE_CHART_COMPONENTS
        - INGRESS_CHART_COMPONENTS
        - UNPINNED_LEGACY_COMPONENTS
        - STORAGE_CHART_COMPONENTS
    )
    assert not unexpected, (
        f"chart gained unpinned workload(s) {sorted(unexpected)}; decide whether they belong "
        f"and update the *_CHART_COMPONENTS pins at the top of this module"
    )
    executors = {c for c in components if re.search(r"celery|worker|beat|scheduler", c)}
    assert not executors, (
        f"a task-executing workload {sorted(executors)} now ships in the chart - E6.08 is no "
        f"longer a defect; re-check test_e6_08_chart_runs_a_celery_worker and drop its xfail"
    )


@pytest.mark.parametrize(
    "overlay", ["values-gke.yaml", "values-eks.yaml", "values-aks.yaml", "values-minikube.yaml", "values-onprem.yaml"]
)
def test_e1_03_values_overlay_parses_and_sets_image(overlay: str) -> None:
    """Every shipped overlay is loadable YAML and re-points image.repository for its target."""
    path = OVERLAY_DIR / overlay
    if not path.is_file():
        pytest.skip(f"overlay {overlay} not present at {path}")
    values = _load_values(path)
    image = values.get("image") or {}
    assert image.get("repository"), f"{overlay} does not set image.repository"


@pytest.mark.parametrize("component", ["webui", "supabase"])
def test_e1_05_branch_only_templates_are_values_gated(component: str) -> None:
    """The webui / local-Supabase templates, when present, are opt-in behind a .Values toggle."""
    matches = [p for p in _chart_files() if component in p.name.lower()]
    if not matches:
        pytest.skip(
            f"chart template for {component!r} is absent on this branch - it exists only on "
            "branch ui-k8s-deploy-1.5.2 (covered by the UI-wiring suite there)"
        )
    for path in matches:
        head = _read(path).lstrip().splitlines()[0]
        assert head.startswith("{{- if .Values"), f"{path.name} is not values-gated: {head!r}"


# --------------------------------------------------------------------------- E2.07
def test_e2_07_no_insecure_auth_escape_hatch_under_deploy() -> None:
    """AVALOKA_ALLOW_INSECURE_AUTH appears in no values file, template or overlay under deploy/ (recursive text scan)."""
    offenders = _grep_deploy(r"AVALOKA_ALLOW_INSECURE_AUTH")
    assert offenders == [], f"auth bypass flag reachable from a deploy artifact: {offenders}"


# --------------------------------------------------------------------------- E6.08
@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E6.08 (P0): no chart template runs a Celery worker or beat/RedBeat scheduler "
        "(deploy/helm/avaloka/templates/ ships only api/langgraph/redis/chroma/postgres/ui), so "
        "nothing executes background tasks in-cluster: .delay() enqueues are swallowed and RedBeat "
        "entries never fire, making /api/datasets/*/background-task-status, Milvus episodic writes "
        "and scheduled entire-dataset executions vacuous in k8s. Remove this xfail when fixed."
    ),
)
def test_e6_08_chart_runs_a_celery_worker() -> None:
    """Some chart template must launch a celery worker/beat process (regex scan of every chart file)."""
    runners = [
        str(p.relative_to(CHART_ROOT)).replace("\\", "/")
        for p in _chart_files()
        if _CELERY_WORKER_RE.search(_read(p))
    ]
    assert runners, "no chart template starts a celery worker or beat scheduler"


def test_e6_08_pins_no_celery_broker_configured() -> None:
    """CELERY_REDIS_URL is set by no chart file — the broker app/core/celery_app.py:23-29 reads is unset in-cluster.

    Paired with test_e6_08_chart_runs_a_celery_worker: wiring the broker without a
    worker (or a worker without the broker) makes the pair disagree, which is the
    signal to re-review the background-task story.
    """
    offenders = [
        str(p.relative_to(CHART_ROOT)).replace("\\", "/")
        for p in _chart_files()
        if "CELERY_REDIS_URL" in _read(p)
    ]
    assert offenders == [], f"CELERY_REDIS_URL is configured but no worker runs: {offenders}"


# --------------------------------------------------------------------------- E6.09
@pytest.mark.parametrize(
    "component,expected", [("redis", "none"), ("chroma", "none"), ("postgres", "emptyDir")]
)
def test_e6_09_pins_stateful_component_persistence(component: str, expected: str) -> None:
    """Pins that redis/chroma mount no volume at all and postgres uses emptyDir (line scan of the owning template).

    Pins the Day-0 decision that in-cluster state is ephemeral. App-side mitigation:
    chat history and user preferences write through to Redis keys, so an API-pod
    restart is survivable — a REDIS-pod restart is not.
    """
    path = _template_for_component(component)
    assert path is not None, f"no chart template declares component {component!r}"
    text = _read(path)
    assert "persistentVolumeClaim" not in text and "volumeClaimTemplates" not in text
    if expected == "none":
        assert not re.search(r"^\s*volumes:\s*$", text, re.MULTILINE), (
            f"{path.name} now mounts volumes — update this pin deliberately"
        )
    else:
        assert re.search(r"^\s*emptyDir:\s*\{\}\s*$", text, re.MULTILINE), (
            f"{path.name} no longer uses emptyDir — update this pin deliberately"
        )


def test_e6_09_pins_which_templates_claim_durable_storage() -> None:
    """Only the deliberately-durable templates carry a PVC; nothing uses a StatefulSet.

    The Day-0 decision was that all in-cluster state is ephemeral. Two templates
    now opt out on purpose, and both are the kind of state whose loss is not a
    cache miss but data loss:

      * ``mcp.yaml``    — the DTA connection registry;
      * ``minio.yaml``  — uploaded datasets and trained model artifacts, and the
        shared data plane Ray workers read from on local/on-prem clusters.

    redis/chroma/postgres stay ephemeral — that leg is pinned separately above.
    Any *other* template claiming a volume is a new decision, not an accident.
    """
    with_pvc = {
        str(p.relative_to(CHART_ROOT)).replace("\\", "/")
        for p in _chart_files()
        if re.search(r"^\s*kind:\s*PersistentVolumeClaim\s*$", _read(p), re.MULTILINE)
    }
    assert with_pvc == {"templates/mcp.yaml", "templates/minio.yaml"}, (
        f"the set of templates claiming durable storage changed: {sorted(with_pvc)} — "
        "re-review E6.09 and update this pin deliberately"
    )

    statefulsets = [
        str(p.relative_to(CHART_ROOT)).replace("\\", "/")
        for p in _chart_files()
        if re.search(r"^\s*kind:\s*StatefulSet\s*$", _read(p), re.MULTILINE)
    ]
    assert statefulsets == [], f"chart gained a StatefulSet — re-review E6.09: {statefulsets}"


# --------------------------------------------------------------------------- E7.09
def test_e7_09_mlflow_backend_is_chart_managed() -> None:
    """Fresh installs include an HTTP MLflow server with a remote artifact store."""
    template = CHART_ROOT / "templates" / "mlflow.yaml"
    assert template.is_file(), "chart has no MLflow Deployment"
    text = _read(template)
    for token in (
        "kind: Deployment",
        "kind: Service",
        "MLFLOW_BACKEND_STORE_URI",
        "MLFLOW_S3_ENDPOINT_URL",
        "--artifacts-destination",
    ):
        assert token in text, f"mlflow.yaml is missing {token}"
    assert "--default-artifact-root" not in text, (
        "proxied artifact mode must use --artifacts-destination without a direct-client root"
    )
    assert "MLFLOW_TRACKING_URI" in _read(CHART_ROOT / "templates" / "configmap.yaml")
    assert "MLFLOW_BACKEND_STORE_URI" in _read(CHART_ROOT / "templates" / "secret.yaml")


# --------------------------------------------------------------------------- E11.10
@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E11.10 (P0): the chart ships zero NetworkPolicy manifests (none anywhere under "
        "deploy/), so Ray workers executing LLM-generated user code have unrestricted network "
        "access to redis (no auth), postgres, chroma, supabase-db and the internet - a data-plane "
        "bypass that route-level auth never sees. Remove this xfail when fixed."
    ),
)
def test_e11_10_chart_ships_default_deny_networkpolicy() -> None:
    """The chart must ship a default-deny NetworkPolicy (empty podSelector, both policyTypes)."""
    policies = [p for p in _chart_files() if _NETWORK_POLICY_RE.search(_read(p))]
    assert policies, "no NetworkPolicy manifest in the chart"
    default_deny = [
        p for p in policies
        if re.search(r"^\s*podSelector:\s*\{\}\s*$", _read(p), re.MULTILINE)
        and "Ingress" in _read(p) and "Egress" in _read(p)
    ]
    assert default_deny, f"NetworkPolicies exist but none is default-deny: {[p.name for p in policies]}"


# --------------------------------------------------------------------------- E11.12
@pytest.mark.defect
@pytest.mark.parametrize("component", ["langgraph", "ui"])
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E11.12 (P1): the langgraph pod (deploy/helm/avaloka/templates/langgraph.yaml:27) and "
        "the Streamlit ui pod (deploy/helm/avaloka/templates/ui.yaml:23) run under the SAME "
        "ServiceAccount the API uses, whose Role grants ray.io rayjobs/rayservices create+delete and "
        "pods/log read (rbac.yaml:15-19), and automountServiceAccountToken is never disabled - so "
        "generated code executing in those pods inherits a token that can schedule arbitrary "
        "containers. Remove this xfail when fixed."
    ),
)
def test_e11_12_untrusted_pods_do_not_inherit_the_ray_privileged_token(component: str) -> None:
    """The langgraph/ui pods use a distinct no-privilege SA or set automountServiceAccountToken: false."""
    path = _template_for_component(component)
    assert path is not None, f"no chart template declares component {component!r}"
    text = _read(path)
    assert not _SHARED_SA_RE.search(text) or _AUTOMOUNT_OFF_RE.search(text), (
        f"{path.name} runs under the shared privileged ServiceAccount with an auto-mounted token"
    )


def test_e11_12_pins_api_role_grants_ray_job_lifecycle() -> None:
    """Pins that the chart Role grants ray.io CRUD + pods/log read (line scan of the Role template).

    Pins the Day-0 decision that the API may schedule and delete RayJobs/RayServices.
    Narrowing the Role is the fix path for E11.12; widening it must be deliberate.
    """
    path = next((p for p in _chart_files() if re.search(r"^\s*kind:\s*Role\s*$", _read(p), re.MULTILINE)), None)
    assert path is not None, "chart ships no Role"
    text = _read(path)
    assert 'apiGroups: ["ray.io"]' in text
    assert '"rayjobs"' in text and '"rayservices"' in text
    assert '"create"' in text and '"delete"' in text
    assert '"pods/log"' in text


# --------------------------------------------------------------------------- E14.06
def test_e14_06_pins_legacy_charts_ship_beside_the_real_chart() -> None:
    """Pins that deploy/avaloka and deploy/avaloka-router still exist next to deploy/helm/avaloka.

    Pins the undecided cleanup: deleting the legacy charts (or keeping them) is a
    product call; this test just makes their presence explicit.
    """
    present = [str(d.relative_to(REPO_ROOT)).replace("\\", "/") for d in LEGACY_CHART_DIRS if d.is_dir()]
    assert present == ["deploy/avaloka", "deploy/avaloka-router"]


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E14.06 (P2): deploy/README.md documents the legacy chart first as a live install path "
        "(deploy/avaloka/run.sh at README.md:6, before any mention of the real chart deploy/helm/avaloka) "
        "with no deprecation marker, so an operator following the README installs the superseded stack. "
        "Remove this xfail when fixed."
    ),
)
def test_e14_06_readme_leads_with_the_real_chart() -> None:
    """deploy/README.md either mentions the real chart before the legacy one or marks the legacy path deprecated."""
    assert DEPLOY_README.is_file(), f"missing {DEPLOY_README}"
    text = _read(DEPLOY_README)
    legacy_at = text.find("deploy/avaloka/run.sh")
    real_at = text.find("helm/avaloka")
    assert legacy_at != -1 and real_at != -1, "README no longer references both charts — revisit this test"
    leads_with_real = real_at < legacy_at
    deprecation_banner = bool(_DEPRECATION_RE.search(text[:legacy_at]))
    assert leads_with_real or deprecation_banner, (
        "README presents the legacy chart first with no deprecation marker"
    )


# --------------------------------------------------------------------------- E14.10
@pytest.mark.defect
@pytest.mark.parametrize("overlay", list(NON_DEV_OVERLAYS))
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E14.10 (P2): every non-dev overlay under deploy/helm/avaloka/values/ pins image.tag: "
        "latest (values-gke.yaml:7, values-eks.yaml:6, values-aks.yaml:6, values-onprem.yaml:6), and the "
        "base values.yaml does the same for avaloka-api and the UI image - a cluster pull tomorrow can "
        "run different code than this plan certified. Remove this xfail when fixed."
    ),
)
def test_e14_10_non_dev_overlay_pins_an_immutable_image(overlay: str) -> None:
    """Cluster-targeted overlays must pin the image by digest or an immutable tag, never a floating one."""
    path = OVERLAY_DIR / overlay
    if not path.is_file():
        pytest.skip(f"overlay {overlay} not present at {path}")
    image = _load_values(path).get("image") or {}
    repository = str(image.get("repository", ""))
    tag = str(image.get("tag", ""))
    assert _is_immutable_image(repository, tag), f"{overlay} floats on {repository}:{tag}"


def test_e14_10_pins_ray_manifests_use_mutable_image_tags() -> None:
    """Pins that both KubeRay manifests reference avaloka-ray:latest (regex scan of deploy/helm/ray/).

    Pins the undecided decision that the Ray runtime image is unpinned; the overlay
    leg (test_e14_10_non_dev_overlay_pins_an_immutable_image) covers the app image,
    and this pin makes the Ray side fail loudly when someone pins one but not the other.
    """
    if not RAY_MANIFEST_DIR.is_dir():
        pytest.skip(f"no KubeRay manifests at {RAY_MANIFEST_DIR}")
    manifests = sorted(RAY_MANIFEST_DIR.glob("*.yaml"))
    assert manifests, f"no KubeRay manifests under {RAY_MANIFEST_DIR}"
    for path in manifests:
        images = re.findall(r"^\s*(?:- )?image:\s*(\S+)\s*$", _read(path), re.MULTILINE)
        assert images, f"{path.name} declares no container image"
        assert set(images) == {"avaloka-ray:latest"}, f"{path.name} image set changed: {sorted(set(images))}"


# ── Avaloka's own images must be pullable from somewhere ───────────────────

def test_avaloka_images_are_registry_qualified():
    """A bare `avaloka-api:latest` resolves to Docker Hub, where it does not exist.

    That worked on kind -- `make images` side-loads onto the node and
    pullPolicy IfNotPresent then never pulls -- and failed on every cluster
    that cannot side-load, with ImagePullBackOff. kind is exactly the
    environment in which the problem is invisible, which is why it survived to
    the release branch.

    Only Avaloka's own images are checked. Third-party references like
    `chromadb/chroma` or `supabase/postgres` are namespaced Docker Hub images
    and pull perfectly well unqualified.
    """
    from pathlib import Path

    import yaml

    values = yaml.safe_load(
        (Path(__file__).parent.parent.parent / "deploy/helm/avaloka/values.yaml").read_text()
    )

    refs: list[tuple[str, str]] = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for key in ("repository", "image"):
                if isinstance(node.get(key), str):
                    refs.append((path or key, node[key]))
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(values)
    assert refs, "no image references found; the walk is wrong, not the chart"

    ours = [(where, ref) for where, ref in refs
            if ref.split(":")[0].split("/")[-1].startswith("avaloka")]
    assert ours, "expected the chart to reference at least one Avaloka image"

    unqualified = [f"{where}: {ref}" for where, ref in ours
                   if "." not in ref.split("/")[0]]
    assert not unqualified, (
        "Avaloka images must name a registry, or they resolve to Docker Hub and "
        "fail with ImagePullBackOff on any cluster that cannot side-load: "
        + "; ".join(unqualified)
    )
