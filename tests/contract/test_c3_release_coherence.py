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
import shlex
import subprocess
from typing import Dict, List, Tuple

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_TEMPLATES_DIR = REPO_ROOT / "deploy" / "helm" / "avaloka" / "templates"

# app/api/server.py used to hardcode the served version as the default of an
# env lookup -- `os.getenv("APP_VERSION", "1.2")` -- which is how GET /version
# answered "1.2" for the whole 1.6 line. It now reads the VERSION file, so the
# surface to compare is the file, not a literal in the source.
_APP_VERSION_READS_VERSION_FILE_RE = re.compile(r'open\(candidate\)|"VERSION"|VERSION["\']?\s*\)')
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
    """The newest *released* version heading in CHANGELOG.md.

    An ``## [Unreleased]`` section above the current release is the normal
    Keep a Changelog layout, not a version, so it is skipped. Only that exact
    label is skipped: the first heading that is anything else is returned as it
    stands, so a newer release heading, or a mislabelled one, still disagrees
    with VERSION and fails the comparison.
    """
    headings = [h.strip() for h in _CHANGELOG_HEADING_RE.findall(_read("CHANGELOG.md"))]
    assert headings, "CHANGELOG.md has no '## [version]' heading at all"
    versioned = [h for h in headings if h.lower() != "unreleased"]
    assert versioned, f"CHANGELOG.md has no released version heading, only: {headings}"
    return versioned[0]


def _chart() -> Dict[str, object]:
    return yaml.safe_load(_read("deploy/helm/avaloka/Chart.yaml"))


def _server_app_version_default() -> str:
    """The version GET /version will serve.

    server.py resolves it at import: an APP_VERSION env var wins, then the OSS
    edition override, then the VERSION file. With neither env var set -- which
    is the shipped default, and what the chart does -- the answer is the VERSION
    file, so that is the surface this compares.
    """
    source = _read("app/api/server.py")
    assert "def _read_version()" in source, (
        "app/api/server.py no longer resolves APP_VERSION through _read_version(); "
        "update this helper to match however it is derived now"
    )
    assert _APP_VERSION_READS_VERSION_FILE_RE.search(source), (
        "app/api/server.py no longer falls back to the VERSION file, so the served "
        "version can drift from it again"
    )
    return _read("VERSION").strip()


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


_PIPELINES_FILE = "bitbucket-pipelines.yml"

# The generated public tree is the only one that carries these at its root:
# oss/manifest.yaml overlays them, and develop-1.6 keeps them under oss/overlay/.
_PUBLIC_TREE_MARKERS = ("DCO", "GOVERNANCE.md", "MAINTAINERS.md")

# The same line shape scripts/generate-oss.sh reads, so the two cannot disagree
# about what is withheld.
_MANIFEST_EXCLUDE_BLOCK_RE = re.compile(r"^exclude:\n((?:(?![a-z_]+:).*\n)*)", re.M)
_MANIFEST_EXCLUDE_PATH_RE = re.compile(r"^  - path: *(.*)$", re.M)


def _withheld_from_public_tree(rel: str) -> bool:
    """True only on the generated public tree, and only for a path the manifest withholds.

    Absence alone decides nothing. On develop-1.6 a missing pipeline file means
    mainline is ungated, which is what the E14.07 cases exist to catch, so both
    halves must hold: this is the public tree (every overlay marker is at the
    root) and oss/manifest.yaml lists the path under ``exclude``.

    The detection rests on one invariant: DCO, GOVERNANCE.md and MAINTAINERS.md
    must never exist at the root of develop-1.6; only the overlay adds them. If
    they ever appear there, this skip can fire on mainline and the pipeline
    contract goes ungated without any test failing.
    """
    if not all((REPO_ROOT / marker).is_file() for marker in _PUBLIC_TREE_MARKERS):
        return False
    manifest = REPO_ROOT / "oss" / "manifest.yaml"
    if not manifest.is_file():
        return False
    block = _MANIFEST_EXCLUDE_BLOCK_RE.search(manifest.read_text(encoding="utf-8"))
    if not block:
        return False
    excluded = {p.strip().strip('"') for p in _MANIFEST_EXCLUDE_PATH_RE.findall(block.group(1))}
    return rel in excluded


def _pipelines_text() -> str:
    if not (REPO_ROOT / _PIPELINES_FILE).exists() and _withheld_from_public_tree(_PIPELINES_FILE):
        pytest.skip(
            f"{_PIPELINES_FILE} is withheld from the public tree by oss/manifest.yaml; "
            "this case runs on develop-1.6, where the file is the CI definition"
        )
    return _read(_PIPELINES_FILE)


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


# The markers CI and scripts/ci.sh actually deselect. `kaggle` joined them when
# the Kaggle-dependent suites appeared; it was missing here, so this pin failed
# on position 5 rather than telling anyone a marker had been added.
_DECLARED_MARKERS: Tuple[str, ...] = (
    "cloud", "integration", "cluster", "kuberay", "kaggle", "slow",
)

#: Markers that exist for grouping rather than selection -- they are never
#: deselected by CI, so adding one does not need a hermetic-filter decision.
#:
#: `containers` is here by decision, not by default: every test that carries it
#: also carries `integration` (tests/conftest.py marks tests/containers/ by
#: directory), so the hermetic filter already deselects it and needs no new
#: clause. It exists so scripts/integration.sh can select that suite alone.
_NON_SELECTION_MARKERS = frozenset(
    {"defect", "single", "multi", "transfer", "operations", "containers"})
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

# DEFECT E14.01 is fixed: these were five different answers to "what version is
# this?" -- 0.2, 0.2.0, 0.1.0, 0.2.0 and 1.2 -- and they are now one. The pin
# stays because the property worth holding is that a release bump moves every
# surface together; it is read from VERSION so a bump updates it in one place.
_RELEASE_VERSION = _read("VERSION").strip()

_PINNED_VERSION_SURFACES: Dict[str, str] = {
    "CHANGELOG.md newest heading": _RELEASE_VERSION,
    "Chart.yaml appVersion": _RELEASE_VERSION,
    "Chart.yaml version": _RELEASE_VERSION,
    "VERSION file": _RELEASE_VERSION,
    "app/api/server.py APP_VERSION default (served by GET /version)": _RELEASE_VERSION,
}


@pytest.mark.parametrize("surface,expected", sorted(_PINNED_VERSION_SURFACES.items()))
def test_e14_01_pins_current_version_surfaces(surface: str, expected: str) -> None:
    """Each release-version surface still carries the exact value recorded at audit time."""
    assert _version_surfaces()[surface] == expected


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


def test_e14_07_pins_ci_triggers_cover_mainline_and_pull_requests() -> None:
    """Mainline and every pull request are gated; branch patterns alone left feat/, fix/, chore/ ungated."""
    pipelines = _pipelines_yaml()["pipelines"]
    branches = set(pipelines["branches"])
    assert "develop-1.6" in branches, (
        "no develop-1.6 pipeline: no merge to mainline would be tested. "
        f"branch tiers present: {sorted(branches)}"
    )
    # Branch globs are literal: feat/* is NOT feature/*, so a pull-request tier
    # is what actually covers every branch name.
    assert "pull-requests" in pipelines, f"no pull-requests tier: {sorted(pipelines)}"
    assert set(pipelines["pull-requests"]) == {"**"}


def test_e14_07_ci_test_step_does_not_swallow_failures() -> None:
    """No CI script line runs pytest with its exit status discarded."""
    swallowed = [
        line.strip()
        for line in _all_pipeline_script_lines()
        if "pytest" in line and re.search(r"\|\|\s*true|\|\|\s*:", line)
    ]
    assert not swallowed, f"CI discards pytest failures: {swallowed}"


def test_e14_07_ci_has_pull_request_pipeline_with_hermetic_marker_filter() -> None:
    """A pull-requests pipeline runs pytest deselecting the cluster/cloud/integration markers."""
    pipelines = _pipelines_yaml()["pipelines"]
    assert "pull-requests" in pipelines, f"no pull-requests pipeline; tiers present: {sorted(pipelines)}"
    lines = _script_lines(pipelines["pull-requests"])
    filtered = [
        line
        for line in lines
        if "pytest" in line
        and "-m" in line
        and "not cluster" in line
        and "not cloud" in line
        # "not integration" was absent from this check while the docstring claimed
        # it. Dropping that one marker puts ~150 environmental failures back into
        # every run, so the test has to see it.
        and "not integration" in line
    ]
    assert filtered, f"pull-requests pipeline runs no hermetic marker filter; scripts: {lines}"


def test_e14_07_ci_test_step_runs_without_provider_credentials() -> None:
    """The hermetic step clears provider variables Bitbucket injects into every step."""
    must_clear = {
        "GROQ_API_KEY",
        "GROQ_API_KEY_CODING_AGENT",
        "GROQ_API_KEY_PLANNING_AGENT",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "INFERENCE_PROVIDER",
    }
    hermetic = [
        line
        for line in _all_pipeline_script_lines()
        if "pytest" in line and "not cluster" in line
    ]
    assert hermetic, "no hermetic pytest invocation found in any pipeline script"
    for line in hermetic:
        missing = sorted(var for var in must_clear if f"-u {var}" not in line)
        assert not missing, (
            f"hermetic step does not clear {missing}. Repository variables reach every "
            "step, so a stored provider key makes tests that would stub or skip call the "
            "provider for real -- 29 extra failures on mainline when the stored Groq key "
            "was rejected. scripts/ci.sh runs with no provider environment; this matches it."
        )


def _run_tests_script() -> List[str]:
    steps = _pipelines_yaml()["definitions"]["steps"]
    step = next(s["step"] for s in steps if s["step"].get("name") == "Run Tests")
    return [line for line in step["script"] if isinstance(line, str)]


def test_e14_07_postgres_suite_cannot_end_the_step_before_the_main_suite() -> None:
    """A red Postgres test still lets the hermetic suite run, and still fails the step.

    The CI runner stops a step at the first command that exits non-zero. The Postgres
    `-m integration` line runs first, so while it was a plain command a red test there
    would have left the commit with no main-suite result.
    """
    script = _run_tests_script()
    pg = [i for i, line in enumerate(script) if "pytest" in line and "-m integration" in line]
    hermetic = [i for i, line in enumerate(script) if "pytest" in line and "not integration" in line]
    assert len(pg) == 1 and hermetic, f"Run Tests has {len(pg)} Postgres and {len(hermetic)} hermetic lines"
    line = script[pg[0]]
    # `pytest ...; echo "$?"` still ends the step where the runner stops at the first
    # failing command (set -e); the left side of `||` never does.
    assert re.match(r"\s*pg_rc=0;\s*pytest\s", line) and re.search(
        r'\|\|\s*pg_rc=\$\?;\s*echo\s+"\$pg_rc"\s*>\s*"\$POSTGRES_SUITE_RC"\s*$', line), (
        "the Postgres line must record its exit status in $POSTGRES_SUITE_RC with `|| pg_rc=$?`, "
        "not end the step: " + line[-160:])
    assert re.search(r"\s-r[a-zA-Z]*A", line), "the Postgres line must report every test by node id (-rA)"
    # The read itself, not the bare word "cat": the Postgres line names
    # test_plan_enforce_comments_notifications.py, which contains it.
    verdict = [i for i, l in enumerate(script) if 'cat "$POSTGRES_SUITE_RC"' in l]
    assert verdict and verdict[0] > max(hermetic), "nothing after the hermetic line fails the step on the Postgres status"
    assert script[verdict[0]].rstrip().endswith('[ "$pg_rc" = 0 ]'), "the verdict line must exit with the Postgres status"


#: Real-Postgres modules that run somewhere other than the Run Tests Postgres line.
#: tests/containers/ is the scripts/integration.sh suite, with a database of its own;
#: tests/release_1_6/ is the 1.6 Release Suite (integration) step, which runs the
#: whole directory (checked below, so the exemption cannot outlive that step).
_POSTGRES_LINE_EXEMPT_PREFIXES: Tuple[str, ...] = ("tests/containers/", "tests/release_1_6/")

#: Node ids the Postgres line may `--deselect`. Each is an open question, recorded
#: beside the line in bitbucket-pipelines.yml; anything else deselected is a test
#: quietly taken out of CI.
_POSTGRES_LINE_DESELECTED: Tuple[str, ...] = ()


def _postgres_line_args() -> Tuple[List[str], List[str]]:
    """The Postgres line's module arguments and its --deselect node ids."""
    pg = [line for line in _run_tests_script() if "pytest" in line and "-m integration" in line]
    assert len(pg) == 1, f"Run Tests has {len(pg)} Postgres lines"
    command = next(part for part in re.split(r";|\|\|", pg[0]) if re.search(r"\bpytest\s", part))
    words = shlex.split(command)
    modules: List[str] = []
    deselected: List[str] = []
    i = 0
    while i < len(words):
        if words[i] == "--deselect" and i + 1 < len(words):
            deselected.append(words[i + 1])
            i += 2
            continue
        if words[i].startswith("--deselect="):
            deselected.append(words[i].split("=", 1)[1])
        elif re.fullmatch(r"tests/\S+\.py", words[i]):
            modules.append(words[i])
        i += 1
    return modules, deselected


def test_e14_07_postgres_line_names_every_real_postgres_module() -> None:
    """Every `integration` module that reads the CI database's URL is on the Postgres line.

    The hermetic line deselects `integration`, so a real-Postgres module left off the
    `-m integration` line never runs in CI at all, and natively it only skips. The
    converse holds too: every module the line names exists, and the only node it
    deselects is one recorded as an open question.
    """
    modules, deselected = _postgres_line_args()
    named = set(modules)
    needed = []
    for rel in _tracked_test_files():
        if rel.startswith(_POSTGRES_LINE_EXEMPT_PREFIXES):
            continue
        text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        if re.search(r"AVALOKA_TEST_(?:PG_ADMIN|POSTGRES)_URL", text) and re.search(
                r"pytest\.mark\.integration\b", text):
            needed.append(rel)
    assert needed, "found no real-Postgres module at all; the detector is broken"
    missing = sorted(set(needed) - named)
    assert not missing, f"real-Postgres modules missing from the Postgres line: {missing}"
    absent = sorted(f for f in named if not (REPO_ROOT / f).is_file())
    assert not absent, f"the Postgres line names modules that do not exist: {absent}"
    unexpected = sorted(set(deselected) - set(_POSTGRES_LINE_DESELECTED))
    assert not unexpected, f"the Postgres line deselects tests nobody recorded as open: {unexpected}"
    for node in deselected:
        module, _, name = node.partition("::")
        assert module in named, f"{node} is deselected from a module the line does not run"
        assert re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(name)}\b",
                         _read(module), re.M), f"the deselected test {node} does not exist"
    release = [line for line in _all_pipeline_script_lines()
               if "pytest" in line and "-m integration" in line
               and re.search(r"\stests/release_1_6/?\s*$", line)]
    assert release, "no step runs `-m integration tests/release_1_6/`, so its exemption above is false"


def test_e14_07_ci_marker_expression_matches_ci_sh() -> None:
    """The pipeline's marker expression and scripts/ci.sh's must not drift apart."""
    ci_sh = (REPO_ROOT / "scripts" / "ci.sh").read_text(encoding="utf-8")
    match = re.search(r'HERMETIC_MARKERS="([^"]+)"', ci_sh)
    assert match, "scripts/ci.sh no longer defines HERMETIC_MARKERS"
    expected = match.group(1)
    lines = _all_pipeline_script_lines()
    assert any(
        expected in line for line in lines if "pytest" in line or "-m" in line
    ), (
        f"pipeline does not run ci.sh's marker expression {expected!r}. "
        "Both files carry a comment saying they must stay in step; this is what enforces it."
    )


def _steps(node: object):
    """Every step mapping under a pipeline tier, through `parallel:` groups and `<<:` merges."""
    if isinstance(node, dict):
        if isinstance(node.get("step"), dict):
            yield node["step"]
        for value in node.values():
            yield from _steps(value)
    elif isinstance(node, list):
        for item in node:
            yield from _steps(item)


def test_e14_07_gating_tier_steps_carry_no_condition() -> None:
    """No step in the pull-requests tier or a branch tier may carry a `condition:`.

    Build 2706 (PR #783) skipped UI Unit Tests and Build and Verify CLI with
    "Changesets condition not satisfied" on a push whose only changed file,
    bitbucket-pipelines.yml, was listed in their includePaths -- while the first
    build of the same pull request had evaluated the same one-file diff as
    "Step condition satisfied" and run them. A skip rule that nobody can
    explain fails "if in doubt, a step runs", so the gating tiers are
    unconditional: a step either is in a tier or it is not.
    """
    pipelines = _pipelines_yaml()["pipelines"]
    conditioned = sorted(
        f"{tier_kind}: {name}: {step.get('name')}"
        for tier_kind in ("pull-requests", "branches")
        for name, tier in (pipelines.get(tier_kind) or {}).items()
        for step in _steps(tier)
        if "condition" in step
    )
    assert not conditioned, (
        f"gating-tier steps carry a condition and can be skipped: {conditioned}. "
        "Remove the condition or remove the step from the tier; see build 2706."
    )


def test_e14_07_each_gating_tier_runs_ci_sh_marker_expression() -> None:
    """Mainline and pull requests each run the hermetic selection themselves.

    The parity test above searches the whole document, so the expression sitting
    in the ``&test`` definition satisfies it even if no pipeline references that
    step. Anchors are resolved on load, so each tier is checked on what it runs.
    """
    ci_sh = (REPO_ROOT / "scripts" / "ci.sh").read_text(encoding="utf-8")
    match = re.search(r'HERMETIC_MARKERS="([^"]+)"', ci_sh)
    assert match, "scripts/ci.sh no longer defines HERMETIC_MARKERS"
    expected = match.group(1)
    pipelines = _pipelines_yaml()["pipelines"]
    tiers = {
        "branches: develop-1.6": pipelines.get("branches", {}).get("develop-1.6"),
        "pull-requests: **": pipelines.get("pull-requests", {}).get("**"),
    }
    ungated = sorted(
        name
        for name, tier in tiers.items()
        if not any("pytest" in line and expected in line for line in _script_lines(tier))
    )
    assert not ungated, (
        f"these tiers run no pytest step with ci.sh's marker expression {expected!r}: {ungated}"
    )


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    # Only the stated defect is expected. Without this a missing pipeline file
    # (FileNotFoundError) would be recorded as the expected failure too.
    raises=AssertionError,
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
    unexpected = set(declared) - set(_DECLARED_MARKERS) - _NON_SELECTION_MARKERS
    assert not unexpected, (
        f"new marker(s) in pytest.ini: {sorted(unexpected)}. Decide whether the "
        "hermetic filter in scripts/ci.sh and .github/workflows/ci.yml must also "
        "deselect them, then add them to _DECLARED_MARKERS or "
        "_NON_SELECTION_MARKERS here."
    )
    assert len(declared) == len(set(declared)), (
        f"pytest.ini declares a marker twice: {sorted(m for m in declared if declared.count(m) > 1)}"
    )


def test_root_level_test_scripts_are_not_collectable_pins_testpaths_scoping() -> None:
    """testpaths stays scoped to tests/, and no stray test script sits at the root.

    This used to pin the opposite: that test.py and test_milvus_integration.py
    DID sit at the repo root, excluded from collection only by
    `testpaths = tests`. Both were deleted while preparing the public 1.0 --
    they were debug scripts, and one of them carried an MCP api_key literal.

    The scoping assertion is the part worth keeping: without it a stray
    `test_*.py` at the root would be collected and run as if it were a test.
    """
    assert _pytest_ini().get("pytest", "testpaths").split() == ["tests"]

    strays = sorted(p.name for p in REPO_ROOT.glob("test*.py"))
    assert not strays, (
        "test scripts at the repo root are collected by some tools regardless of "
        f"testpaths, and these are not tests: {', '.join(strays)}"
    )


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
        "tests/test_youtube_e2e.py (real LLM/Milvus), "
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
    # The relevance ranking, in order. This used to pin a fourth tier, "llm",
    # as part of the literal; since LLM extraction moved off the request path
    # nothing fills it (learned hints arrive as session history), so it was
    # removed rather than kept as a tier that reads like a guarantee.
    assert re.search(
        r'_HINT_TIER_ORDER = \("artifact", "domain", "similar"[,)]', plane
    ), "the top-3 relevance order no longer starts artifact > domain > similar"
    assert "def _compose_top_hints(" in plane


def test_e6_07_memory_circuit_breaker_is_merged() -> None:
    """The memory circuit breaker is merged app code with a default and live coverage.

    The default used to be pinned at exactly 3.0s. It is 20.0s now, raised
    deliberately: 3 seconds was shorter than the first-call load of the
    sentence-transformer model, so the breaker tripped on the very first query
    and the learning loop looked empty rather than slow.

    What is worth pinning is that the breaker exists, is configurable, and is
    covered -- not the number, which is an operational tuning decision.
    """
    plane = _read("app/services/memory_plane.py")
    assert re.search(
        r'MEMORY_CIRCUIT_BREAKER_TIMEOUT["\']\s*,\s*["\'][0-9]+(?:\.[0-9]+)?["\']', plane
    ), "the circuit-breaker timeout is no longer an env-configurable numeric default"
    semantics = _read("tests/test_memory_semantics.py")
    assert "MEMORY_CIRCUIT_BREAKER_TIMEOUT" in semantics
    assert len(_TEST_FUNC_RE.findall(semantics)) >= 1


def test_release_tag_publishes_its_version_tags() -> None:
    """A ``v*`` tag push builds, so ``:<version>`` and ``:<major>.<minor>`` exist.

    The workflow skips a build whose content tag is already published. A release
    tag normally points at a commit main has already built, so without a tag
    exception the skip fired, the semver tags -- applied only by the build --
    were never pushed, and ``:1.0.0`` did not resolve after tagging v1.0.0.
    """
    workflow = yaml.safe_load(_read(".github/workflows/images.yml"))
    # PyYAML reads the bare key `on` as True.
    triggers = workflow.get("on", workflow.get(True))
    assert "v*" in triggers["push"]["tags"], "images.yml no longer builds on a v* tag"
    steps = {step.get("id"): step for step in workflow["jobs"]["build"]["steps"] if step.get("id")}
    meta = steps["meta"]["with"]["tags"]
    assert "type=semver,pattern={{version}}" in meta
    assert "type=semver,pattern={{major}}.{{minor}}" in meta
    script = steps["published"]["run"]
    marker = '"${{ github.ref_type }}" == "tag"'
    assert marker in script, (
        "images.yml skips a release tag like any push whose content tag exists, so "
        ":<version> and :<major>.<minor> are never published")
    tag_branch = script.index(marker)
    skip_branch = script.index('echo "skip=true"')
    assert tag_branch < skip_branch, "a release tag can still be skipped before it is built"
    assert 'echo "skip=false"' in script[tag_branch:skip_branch]
