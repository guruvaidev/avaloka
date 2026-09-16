"""Suite C3 — release / documentation coherence (plan suite E14, E14.07 CI wiring)
plus the hermetic-invariant marker audit that plan sections 3 and 5 depend on.

Every case here is a pure source-text contract over the checked-in repo: it reads
VERSION, CHANGELOG.md, deploy/helm/avaloka/**, bitbucket-pipelines.yml, pytest.ini
and the tests/ tree. Nothing imports the app, starts a server or touches a cluster,
so the whole module runs hermetically and there is no deployed-only leg.

Cases decorated ``@pytest.mark.defect`` with ``xfail(strict=True)`` encode defects
that exist in the tree today; the assertion states the intended behaviour, so the
day the defect is fixed the xfail turns into an xpass and must be deleted.
"""

from __future__ import annotations

import configparser
import pathlib
import re
import subprocess
from typing import Dict, List, Tuple

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_TEMPLATES_DIR = REPO_ROOT / "deploy" / "helm" / "avaloka" / "templates"

_APP_VERSION_DEFAULT_RE = re.compile(
    r"""APP_VERSION\s*=\s*os\.getenv\(\s*["']APP_VERSION["']\s*,\s*["']([^"']+)["']\s*\)"""
)
_CHANGELOG_HEADING_RE = re.compile(r"^##\s*\[([^\]]+)\]", re.M)
_ROADMAP_HEADING_RE = re.compile(r"^###\s+(Roadmap|Unreleased)\s*$", re.M | re.I)
_NEXT_HEADING_RE = re.compile(r"^#{2,3}\s+\S", re.M)
_TEST_FUNC_RE = re.compile(r"^\s*(?:async\s+)?def\s+test_\w+", re.M)


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")


def _normalize_version(raw: str) -> str:
    parts = [p for p in raw.strip().strip("v").split(".") if p != ""]
    while len(parts) < 3:
        parts.append("0")
    return ".".join(parts[:3])


def _changelog_newest_version() -> str:
    headings = _CHANGELOG_HEADING_RE.findall(_read("CHANGELOG.md"))
    assert headings, "CHANGELOG.md has no '## [version]' heading at all"
    return headings[0].strip()


def _chart() -> Dict[str, object]:
    return yaml.safe_load(_read("deploy/helm/avaloka/Chart.yaml"))


def _server_app_version_default() -> str:
    match = _APP_VERSION_DEFAULT_RE.search(_read("app/api/server.py"))
    assert match, "app/api/server.py no longer defines APP_VERSION via os.getenv with a literal default"
    return match.group(1)


def _version_surfaces() -> Dict[str, str]:
    chart = _chart()
    return {
        "CHANGELOG.md newest heading": _changelog_newest_version(),
        "Chart.yaml appVersion": str(chart["appVersion"]),
        "Chart.yaml version": str(chart["version"]),
        "VERSION file": _read("VERSION").strip(),
        "app/api/server.py APP_VERSION default (served by GET /version)": _server_app_version_default(),
    }


def _roadmap_section() -> str:
    text = _read("CHANGELOG.md")
    match = _ROADMAP_HEADING_RE.search(text)
    assert match, "CHANGELOG.md has no '### Roadmap' or '### Unreleased' section"
    rest = text[match.end():]
    nxt = _NEXT_HEADING_RE.search(rest)
    return rest[: nxt.start()] if nxt else rest


def _pipelines_text() -> str:
    return _read("bitbucket-pipelines.yml")


def _pipelines_yaml() -> Dict[str, object]:
    return yaml.safe_load(_pipelines_text())


def _script_lines(node: object) -> List[str]:
    lines: List[str] = []
    if isinstance(node, str):
        lines.append(node)
    elif isinstance(node, list):
        for item in node:
            lines.extend(_script_lines(item))
    elif isinstance(node, dict):
        for key, value in node.items():
            if key == "script":
                lines.extend(_script_lines(value))
            else:
                lines.extend(_script_lines(value))
    return lines


def _all_pipeline_script_lines() -> List[str]:
    return _script_lines(_pipelines_yaml())


def _template_sources() -> Dict[str, str]:
    if not _TEMPLATES_DIR.is_dir():
        return {}
    return {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/"): path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(_TEMPLATES_DIR.rglob("*"))
        if path.is_file()
    }


def _tracked_test_files() -> List[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "tests/"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tracked = [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        tracked = [
            str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            for p in (REPO_ROOT / "tests").rglob("*.py")
        ]
    return [f for f in tracked if re.fullmatch(r"tests/(?:.+/)?test_[^/]+\.py", f)]


_DECLARED_MARKERS: Tuple[str, ...] = ("cloud", "integration", "cluster", "kuberay", "slow")
_MARKER_RE = re.compile(r"pytest\.mark\.(?:%s)\b" % "|".join(_DECLARED_MARKERS))

_INFRA_SIGNALS: Dict[str, re.Pattern] = {
    "boto3-aws-client": re.compile(r"^\s*(?:import|from)\s+boto3\b", re.M),
    "docker-build": re.compile(r"[\"']docker[\"']\s*,\s*[\"']build[\"']|docker\s+build\b"),
    "load_dotenv-real-credentials": re.compile(r"^\s*load_dotenv\s*\(", re.M),
    "localhost-service-port": re.compile(
        r"(?:localhost|127\.0\.0\.1):(?:5432|3306|6379|19530|8080|8081|9092|27017)"
    ),
    "real-vector-db-client": re.compile(
        r"^\s*from\s+app\.services\.db\.milvus_client\b|MilvusClient\s*\(|chromadb\.HttpClient\s*\(", re.M
    ),
    "gke-job-launch": re.compile(r"launch_gke_pipeline\s*\("),
}

_KNOWN_UNMARKED_INFRA_TESTS: Tuple[str, ...] = (
    "tests/infra/test_eks_deployment.py",
    "tests/test_e2e_gke_groupby_sum.py",
    "tests/test_mcp_server_integration.py",
    "tests/test_youtube_e2e.py",
)


def _unmarked_infra_test_files() -> Dict[str, List[str]]:
    offenders: Dict[str, List[str]] = {}
    for rel in _tracked_test_files():
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        hits = sorted(name for name, pattern in _INFRA_SIGNALS.items() if pattern.search(src))
        if hits and not _MARKER_RE.search(src):
            offenders[rel] = hits
    return offenders


def _pytest_ini() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read_string(_read("pytest.ini"))
    return parser


def _declared_marker_names() -> List[str]:
    raw = _pytest_ini().get("pytest", "markers")
    return [line.split(":", 1)[0].strip() for line in raw.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# E14.01 — the version story must be one story
# ---------------------------------------------------------------------------

_PINNED_VERSION_SURFACES: Dict[str, str] = {
    "CHANGELOG.md newest heading": "0.2",
    "Chart.yaml appVersion": "0.2.0",
    "Chart.yaml version": "0.1.0",
    "VERSION file": "0.2.0",
    "app/api/server.py APP_VERSION default (served by GET /version)": "1.2",
}


@pytest.mark.parametrize("surface,expected", sorted(_PINNED_VERSION_SURFACES.items()))
def test_e14_01_pins_current_version_surfaces(surface: str, expected: str) -> None:
    """Each release-version surface still carries the exact value recorded at audit time."""
    assert _version_surfaces()[surface] == expected


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E14.01 (P0): the release version is five-way inconsistent - VERSION='0.2.0', "
        "CHANGELOG.md:7 '## [0.2]', deploy/helm/avaloka/Chart.yaml version=0.1.0 / appVersion='0.2.0', "
        "and GET /version serves APP_VERSION default '1.2' (app/api/server.py:296) because APP_VERSION "
        "is set nowhere in deploy/ or .env, while the release branch claims 1.5.2. "
        "Remove this xfail when fixed."
    ),
)
def test_e14_01_version_surfaces_agree() -> None:
    """VERSION, Chart version/appVersion, the newest CHANGELOG heading and the served APP_VERSION are one version."""
    surfaces = _version_surfaces()
    distinct = {_normalize_version(value) for value in surfaces.values()}
    assert len(distinct) == 1, f"version surfaces disagree: {surfaces}"


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E14.01a (P1): no Helm template injects APP_VERSION from .Chart.AppVersion, so the "
        "version served by GET /version can never match the deployed artifact "
        "(deploy/helm/avaloka/templates/configmap.yaml has no APP_VERSION key). "
        "Remove this xfail when fixed."
    ),
)
def test_e14_01a_chart_injects_app_version_from_chart_appversion() -> None:
    """A chart template wires APP_VERSION from .Chart.AppVersion so /version cannot drift from the artifact."""
    templates = _template_sources()
    assert templates, f"no Helm templates found under {_TEMPLATES_DIR}"
    injecting = {
        name: src
        for name, src in templates.items()
        if re.search(r"^\s*(?:-?\s*name:\s*)?APP_VERSION\s*:?", src, re.M)
    }
    assert injecting, f"no template under {_TEMPLATES_DIR} declares APP_VERSION; searched {sorted(templates)}"
    assert any(
        ".Chart.AppVersion" in src for src in injecting.values()
    ), f"APP_VERSION is declared but not sourced from .Chart.AppVersion: {sorted(injecting)}"


# ---------------------------------------------------------------------------
# E14.02 — CHANGELOG Roadmap must not list shipped features
# ---------------------------------------------------------------------------

_ROADMAP_CLAIMS: Tuple[Tuple[str, str, str], ...] = (
    (
        "Azure AKS provider",
        "app/infra/providers/azure_aks.py",
        "DEFECT E14.02 (P1): CHANGELOG.md:45 lists 'Azure AKS provider' under Roadmap but "
        "app/infra/providers/azure_aks.py ships, is registered in app/infra/providers/factory.py:21 and "
        "deploy/helm/avaloka/values/values-aks.yaml exists. Remove this xfail when fixed.",
    ),
    (
        "CLI remote mode",
        "app/interfaces/cli/main.py",
        "DEFECT E14.02 (P1): CHANGELOG.md:46 lists 'CLI remote mode (--endpoint)' under Roadmap but it is "
        "implemented at app/interfaces/cli/main.py:58-73 and :145 and covered by tests/k8s/test_t4_cli.py. "
        "Remove this xfail when fixed.",
    ),
)


def test_e14_02_pins_roadmap_section_names_the_drifted_entries() -> None:
    """The CHANGELOG Roadmap section currently names both already-shipped features, anchoring the drift."""
    section = _roadmap_section()
    assert "Azure AKS provider" in section
    assert "CLI remote mode" in section and "--endpoint" in section


@pytest.mark.defect
@pytest.mark.parametrize(
    "phrase,implementation",
    [pytest.param(p, i, id=p, marks=pytest.mark.xfail(strict=True, reason=r)) for p, i, r in _ROADMAP_CLAIMS],
)
def test_e14_02_roadmap_does_not_list_implemented_features(phrase: str, implementation: str) -> None:
    """No feature named under CHANGELOG Roadmap has a shipped implementation file."""
    implemented = (REPO_ROOT / implementation).exists()
    listed_as_roadmap = phrase in _roadmap_section()
    assert not (
        implemented and listed_as_roadmap
    ), f"{phrase!r} is listed under Roadmap but {implementation} exists"


# ---------------------------------------------------------------------------
# E14.07 / E12.06 — CI must actually enforce something
# ---------------------------------------------------------------------------


def test_e14_07_pins_ci_triggers_are_branch_pushes_only() -> None:
    """bitbucket-pipelines.yml currently defines branch-push pipelines only, with no PR/nightly/weekly tiers."""
    pipelines = _pipelines_yaml()["pipelines"]
    assert set(pipelines) == {"branches"}, f"unexpected pipeline tiers: {sorted(pipelines)}"
    assert set(pipelines["branches"]) == {
        "feature/context-memory",
        "feature/*",
        "bugfix/*",
        "hotfix/*",
        "release/*",
    }


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E14.07 (P0): the CI test step is 'pytest --continue-on-collection-errors tests/ || true' "
        "(bitbucket-pipelines.yml, step &test) - the trailing '|| true' makes a fully red suite report green, "
        "so CI enforces nothing. Remove this xfail when fixed."
    ),
)
def test_e14_07_ci_test_step_does_not_swallow_failures() -> None:
    """No CI script line runs pytest with its exit status discarded."""
    swallowed = [
        line.strip()
        for line in _all_pipeline_script_lines()
        if "pytest" in line and re.search(r"\|\|\s*true|\|\|\s*:", line)
    ]
    assert not swallowed, f"CI discards pytest failures: {swallowed}"


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E14.07 (P0): bitbucket-pipelines.yml has no 'pull-requests' pipeline and no step that runs "
        "the hermetic marker filter -m 'not cluster and not cloud and not integration', so nothing gates a "
        "merge. Remove this xfail when fixed."
    ),
)
def test_e14_07_ci_has_pull_request_pipeline_with_hermetic_marker_filter() -> None:
    """A pull-requests pipeline runs pytest deselecting the cluster/cloud/integration markers."""
    pipelines = _pipelines_yaml()["pipelines"]
    assert "pull-requests" in pipelines, f"no pull-requests pipeline; tiers present: {sorted(pipelines)}"
    lines = _script_lines(pipelines["pull-requests"])
    filtered = [
        line
        for line in lines
        if "pytest" in line and "-m" in line and "not cluster" in line and "not cloud" in line
    ]
    assert filtered, f"pull-requests pipeline runs no hermetic marker filter; scripts: {lines}"


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E12.06 (P1): the plan marks the image-size budget 'AUTO (CI)', but bitbucket-pipelines.yml "
        "never builds an image and has no size gate at all. Remove this xfail when fixed."
    ),
)
def test_e12_06_ci_enforces_an_image_size_budget() -> None:
    """CI builds the runtime image and fails when it exceeds a declared size budget."""
    text = _pipelines_text()
    builds = re.search(r"docker\s+(?:build|buildx)\b", text)
    assert builds, "no docker build step in bitbucket-pipelines.yml"
    gate = re.search(r"IMAGE_SIZE|MAX_IMAGE|image\s+size|\{\{\s*\.Size\s*\}\}|docker\s+image\s+inspect", text, re.I)
    assert gate, "bitbucket-pipelines.yml has no image-size gate"


# ---------------------------------------------------------------------------
# Hermetic-invariant marker audit (plan sections 3 and 5)
# ---------------------------------------------------------------------------


def test_pytest_ini_markers_are_stable() -> None:
    """pytest.ini still declares the five selection markers section 3 filters on, plus `defect`."""
    declared = _declared_marker_names()
    assert declared[: len(_DECLARED_MARKERS)] == list(_DECLARED_MARKERS)
    assert set(declared) - set(_DECLARED_MARKERS) <= {"defect"}, (
        "a new marker was added to pytest.ini; decide whether the hermetic filter in "
        "section 3 and CI must also deselect it, then update this pin"
    )


def test_root_level_test_scripts_are_not_collectable_pins_testpaths_scoping() -> None:
    """Pins the decision to leave root test.py / test_milvus_integration.py in place and exclude them via testpaths=tests."""
    assert _pytest_ini().get("pytest", "testpaths").split() == ["tests"]
    for name in ("test.py", "test_milvus_integration.py"):
        script = REPO_ROOT / name
        assert script.is_file(), f"{name} no longer sits at the repo root; update this pin"
        assert not str(script).startswith(str(REPO_ROOT / "tests"))


@pytest.mark.parametrize("offender", _KNOWN_UNMARKED_INFRA_TESTS)
def test_hermetic_marker_audit_detector_flags_known_offenders(offender: str) -> None:
    """The conservative infrastructure detector still recognises each known unmarked infra-touching test file."""
    detected = _unmarked_infra_test_files()
    assert offender in detected, (
        f"{offender} is no longer flagged; either it gained a marker (then drop it from "
        f"_KNOWN_UNMARKED_INFRA_TESTS) or the detector regressed. Currently flagged: {sorted(detected)}"
    )


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT MARKER-AUDIT (P0): infrastructure-touching tests carry no marker, so "
        "-m 'not cluster and not cloud and not integration' does not deselect them and the plan's "
        "'hermetic suite green' entry criterion is unenforceable. Known offenders: "
        "tests/test_e2e_gke_groupby_sum.py (docker build + GKE), tests/test_youtube_e2e.py (real LLM/Milvus), "
        "tests/infra/test_eks_deployment.py (boto3 + load_dotenv of the root .env), "
        "tests/test_mcp_server_integration.py (live services on localhost:5432/8080/8081). "
        "Remove this xfail when fixed."
    ),
)
def test_hermetic_marker_audit() -> None:
    """Every test file touching real infrastructure carries one of the declared skip markers."""
    offenders = _unmarked_infra_test_files()
    rendered = "\n".join(f"  {rel}: {', '.join(hits)}" for rel, hits in sorted(offenders.items()))
    assert not offenders, f"{len(offenders)} unmarked infrastructure-touching test files:\n{rendered}"


# ---------------------------------------------------------------------------
# Plan-referenced artifacts that exist in no branch and no history
# ---------------------------------------------------------------------------

_PLAN_REFERENCED_ARTIFACTS: Tuple[Tuple[str, str], ...] = (
    ("E3.04", "tests/test_bomb_logic.py"),
    ("E12.02", "profile_nyc_taxi.py"),
    ("E6.05", "demo_top3_hints.py"),
    ("E6.07", "demo_circuit_breaker.py"),
    ("E8.05", "demo_mcp_tarpit.py"),
    ("plan-index", "K8S_TEST_PLAN.md"),
    ("plan-index", "RUNBOOK-k8s-deploy-1.5.2.md"),
    ("plan-index", "DEMO_RUNBOOK.md"),
)


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT PLAN-REFS (P1): the test plan cites artifacts that exist in no branch and no git history "
        "(test_bomb_logic.py, profile_nyc_taxi.py, demo_top3_hints.py, demo_circuit_breaker.py, "
        "demo_mcp_tarpit.py, K8S_TEST_PLAN.md, RUNBOOK-k8s-deploy-1.5.2.md, DEMO_RUNBOOK.md); "
        "either commit them or rewrite the cases. Remove this xfail when fixed."
    ),
)
@pytest.mark.parametrize(
    "case_id,artifact", [pytest.param(c, a, id=f"{c}:{a}") for c, a in _PLAN_REFERENCED_ARTIFACTS]
)
def test_plan_referenced_artifacts_exist(case_id: str, artifact: str) -> None:
    """Every file the test plan names as evidence is actually present in the repo."""
    assert (REPO_ROOT / artifact).exists(), f"plan case {case_id} cites missing artifact {artifact}"


# ---------------------------------------------------------------------------
# E6.05 / E6.07 — the plan's memory exit criteria have real anchors
# ---------------------------------------------------------------------------


def test_e6_05_top3_hint_relevance_regression_suite_is_merged() -> None:
    """The top-3 hint relevance fix and its regression suite are merged, so E6.05's exit criterion is anchored."""
    suite = _read("tests/test_memory_top3_relevance.py")
    assert len(_TEST_FUNC_RE.findall(suite)) >= 10
    plane = _read("app/services/memory_plane.py")
    assert '_HINT_TIER_ORDER = ("artifact", "domain", "similar", "llm")' in plane
    assert "def _compose_top_hints(" in plane


def test_e6_07_memory_circuit_breaker_is_merged() -> None:
    """The memory circuit breaker is merged app code with a 3.0s default and live coverage, anchoring E6.07."""
    plane = _read("app/services/memory_plane.py")
    assert re.search(r'MEMORY_CIRCUIT_BREAKER_TIMEOUT["\']\s*,\s*["\']3\.0["\']', plane)
    semantics = _read("tests/test_memory_semantics.py")
    assert "MEMORY_CIRCUIT_BREAKER_TIMEOUT" in semantics
    assert len(_TEST_FUNC_RE.findall(semantics)) >= 1
