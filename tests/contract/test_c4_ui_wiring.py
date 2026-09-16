"""Suite C4 - plan suite E9 (frontend wiring) plus the ui-k8s-deploy-1.5.2 merge surface.

Pure filesystem/text contract checks: no app import, no network, no tests/e2e fixtures.

The ui/ tree and the webui/supabase half of the Helm chart live on branch
ui-k8s-deploy-1.5.2 (merged via PR #237), NOT on feature/k8s-deploy-1.5.2. Every test
resolves its target through _merge_file()/_merge_dir(), which skip with a reason naming
that branch when the target is missing. On this branch the module therefore collects
and reports ALL SKIPPED; it does real work as soon as ui/ is present.

To run it against the merge surface without disturbing this checkout:

    git worktree add ../avaloka-ui ui-k8s-deploy-1.5.2
    venv/Scripts/python.exe -m pytest ../avaloka-ui/tests/contract/test_c4_ui_wiring.py -q -rxX

(or simply check out ui-k8s-deploy-1.5.2 and run the same path). The xfail-marked cases
below are the P0/P1 defects that ship broken on that branch today.
"""

from __future__ import annotations

import base64
import binascii
import json
import pathlib
import re
from typing import Any, Iterable

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
UI_BRANCH = "ui-k8s-deploy-1.5.2"

_CHART = "deploy/helm/avaloka"
_VALUES_FILES = (
    f"{_CHART}/values.yaml",
    f"{_CHART}/values/values-gke.yaml",
    f"{_CHART}/values/values-eks.yaml",
    f"{_CHART}/values/values-aks.yaml",
    f"{_CHART}/values/values-onprem.yaml",
    f"{_CHART}/values/values-minikube.yaml",
)
_CLOUD_OVERLAYS = (
    f"{_CHART}/values/values-gke.yaml",
    f"{_CHART}/values/values-eks.yaml",
    f"{_CHART}/values/values-aks.yaml",
    f"{_CHART}/values/values-onprem.yaml",
)

_DEMO_JWT_SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"
_DEMO_JWT_ISSUER = "supabase-demo"


def _merge_surface() -> pathlib.Path:
    ui_dir = REPO_ROOT / "ui"
    if not ui_dir.is_dir():
        pytest.skip(f"ui/ merge surface present only on {UI_BRANCH} (looked for {ui_dir})")
    return ui_dir


def _merge_file(rel: str) -> pathlib.Path:
    _merge_surface()
    path = REPO_ROOT / rel
    if not path.is_file():
        pytest.skip(f"{rel} present only on {UI_BRANCH}")
    return path


def _merge_dir(rel: str) -> pathlib.Path:
    _merge_surface()
    path = REPO_ROOT / rel
    if not path.is_dir():
        pytest.skip(f"{rel}/ present only on {UI_BRANCH}")
    return path


def _ui_file(rel: str) -> pathlib.Path:
    return _merge_file(f"ui/{rel}")


def _text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


_LOCATION_RE = re.compile(r"^\s*location\s+~\*?\s+(?P<regex>\S+)\s*\{", re.M)
_API_CALL_RE = re.compile(r"\$\{API_BASE\}(/[^`'\"\s?]*)")
_TS_INTERP_RE = re.compile(r"\$\{[^}]*\}")
_INJECTED_READ_RE = re.compile(r"\binjected\.([A-Z][A-Z0-9_]*)")
_RUNTIME_ENV_TYPE_RE = re.compile(r"type RuntimeEnv\s*=\s*\{(?P<body>.*?)\};", re.S)
_ENV_OBJECT_RE = re.compile(r"window\._env_\s*=\s*\{(?P<body>.*?)^\};", re.S | re.M)
_OBJECT_KEY_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*\??\s*:", re.M)
_EXPORT_CONST_RE = re.compile(r"^export const (?P<name>[A-Z][A-Z0-9_]*)\s*=(?P<rhs>.*?);", re.S | re.M)
_IT_CASE_RE = re.compile(r"^[ \t]*it\(", re.M)
_SPEC_SUFFIX_RE = re.compile(r"\.(test|spec)\.tsx?$")


def _proxy_location_regexes(template: str) -> list[str]:
    regexes: list[str] = []
    for match in _LOCATION_RE.finditer(template):
        body = template[match.end() :]
        following = _LOCATION_RE.search(body)
        if following:
            body = body[: following.start()]
        if "proxy_pass" in body:
            regexes.append(match.group("regex"))
    return regexes


def _api_base_paths(source: str) -> set[str]:
    paths: set[str] = set()
    for raw in _API_CALL_RE.findall(source):
        path = _TS_INTERP_RE.sub("x", raw).rstrip("/") or "/"
        paths.add(path)
    return paths


def _first_string_literal(expression: str) -> str:
    for literal in re.findall(r'"([^"]*)"', expression):
        if literal:
            return literal
    return ""


def _runtime_fallbacks(source: str) -> dict[str, str]:
    return {
        match.group("name"): _first_string_literal(match.group("rhs"))
        for match in _EXPORT_CONST_RE.finditer(source)
    }


def _jwt_issuer(token: str) -> str:
    parts = token.split(".")
    if len(parts) != 3:
        return ""
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload)).get("iss", "")
    except (binascii.Error, ValueError, AttributeError):
        return ""


def _load_values(rel: str) -> dict[str, Any]:
    return yaml.safe_load(_text(_merge_file(rel))) or {}


def _dig(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    node: Any = mapping
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _spec_files() -> list[pathlib.Path]:
    src = _merge_dir("ui/src")
    return sorted(p for p in src.rglob("*") if p.is_file() and _SPEC_SUFFIX_RE.search(p.name))


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E9.14a (P0): ui/deploy/nginx.conf.template proxies only "
        "^/(health|version|threads|api|docs|openapi.json) so the SPA's root-level calls "
        "/tasks/*/result/*, /datasets/*/preview and /buckets/list fall through to the SPA "
        "fallback and return index.html instead of JSON. Remove this xfail when fixed."
    ),
)
def test_e9_14a_nginx_allowlist_covers_every_api_call() -> None:
    """Every API_BASE-relative path the SPA fetches is matched by an nginx proxy location."""
    template = _text(_ui_file("deploy/nginx.conf.template"))
    sources = "\n".join(
        _text(_ui_file(rel))
        for rel in ("src/services/backendApi.ts", "src/services/assistantApi.ts")
    )
    matchers = [re.compile(regex) for regex in _proxy_location_regexes(template)]
    assert matchers, "nginx.conf.template declares no regex location containing proxy_pass"

    called = _api_base_paths(sources)
    assert called, "no ${API_BASE} fetch literals found - the path extractor is stale"

    uncovered = sorted(path for path in called if not any(m.search(path) for m in matchers))
    assert not uncovered, (
        f"paths the SPA calls but nginx never proxies: {uncovered} "
        f"(proxy locations: {[m.pattern for m in matchers]})"
    )


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E9.15a (P0): ui/docker-entrypoint.d/10-env-config.sh writes only "
        "SUPABASE_URL/SUPABASE_ANON_KEY/API_BASE, so window._env_.MCP_API_BASE and "
        "MCP_TOOL_BASE are never set and ui/src/lib/runtime-config.ts always falls back to "
        "the hardcoded ngrok hosts - the Helm webui.mcp.* values are inert. "
        "Remove this xfail when fixed."
    ),
)
def test_e9_15a_env_config_emits_every_runtime_key() -> None:
    """The entrypoint writes exactly the window._env_ keys runtime-config.ts reads."""
    runtime = _text(_ui_file("src/lib/runtime-config.ts"))
    script = _text(_ui_file("docker-entrypoint.d/10-env-config.sh"))

    declared = set(_INJECTED_READ_RE.findall(runtime))
    type_block = _RUNTIME_ENV_TYPE_RE.search(runtime)
    if type_block:
        declared |= set(_OBJECT_KEY_RE.findall(type_block.group("body")))
    assert declared, "runtime-config.ts exposes no window._env_ keys - the extractor is stale"

    emitted_block = _ENV_OBJECT_RE.search(script)
    assert emitted_block, "10-env-config.sh no longer writes a window._env_ object literal"
    emitted = set(_OBJECT_KEY_RE.findall(emitted_block.group("body")))

    assert emitted == declared, (
        f"runtime keys read by the SPA but never injected: {sorted(declared - emitted)}; "
        f"injected but unused: {sorted(emitted - declared)}"
    )


def test_e9_16_index_html_loads_env_config_before_bundle() -> None:
    """index.html loads /env-config.js ahead of the module bundle so window._env_ exists."""
    html = _text(_ui_file("index.html"))
    env_script = re.search(r"<script[^>]+src=[\"']/env-config\.js[\"'][^>]*>", html)
    module_script = re.search(r"<script[^>]+type=[\"']module[\"'][^>]*>", html)

    assert env_script, "index.html does not load /env-config.js - window._env_ is never defined"
    assert module_script, "index.html loads no type=module bundle"
    assert env_script.start() < module_script.start(), (
        "the app bundle is loaded before /env-config.js, so runtime-config.ts reads an "
        "empty window._env_ and silently uses its hardcoded defaults"
    )


def test_e9_16_pins_hardcoded_production_fallbacks() -> None:
    """runtime-config.ts still ships non-empty cloud fallbacks when window._env_ is absent.

    Pins the undecided product decision: an unconfigured container currently phones home to
    a real cloud Supabase project and the ngrok MCP hosts instead of failing loudly. Change
    this test only together with a deliberate decision to fail closed.
    """
    fallbacks = _runtime_fallbacks(_text(_ui_file("src/lib/runtime-config.ts")))

    assert fallbacks.get("SUPABASE_URL", "").endswith(".supabase.co")
    assert fallbacks.get("SUPABASE_ANON_KEY", "").startswith("eyJ")
    assert "ngrok" in fallbacks.get("API_BASE", "")
    assert "ngrok" in fallbacks.get("MCP_API_BASE", "")
    assert "ngrok" in fallbacks.get("MCP_TOOL_BASE", "")


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E9.17 (P0): ui/package.json has no \"test\" script and no "
        "vitest/@testing-library/jsdom devDependencies, so the 11 merged *.test.* specs and "
        "ui/vitest.config.ts are unrunnable at branch tip. Remove this xfail when fixed."
    ),
)
def test_e9_17_vitest_harness_is_wired() -> None:
    """Committed vitest specs come with a test script and the vitest/RTL/jsdom devDeps."""
    specs = _spec_files()
    if not specs:
        pytest.skip("no *.test.*/*.spec.* files under ui/src - nothing to wire")

    package = json.loads(_text(_ui_file("package.json")))
    scripts = package.get("scripts", {})
    dev_deps = package.get("devDependencies", {})

    missing_scripts = [name for name in ("test",) if name not in scripts]
    missing_deps = [name for name in ("vitest", "jsdom") if name not in dev_deps]
    if not any(name.startswith("@testing-library/") for name in dev_deps):
        missing_deps.append("@testing-library/*")

    assert not missing_scripts and not missing_deps, (
        f"{len(specs)} spec files are committed but package.json lacks "
        f"scripts={missing_scripts} devDependencies={missing_deps}"
    )


def test_e9_17_pins_vitest_spec_inventory() -> None:
    """The merged vitest surface is 11 spec files / 37 it() cases behind ui/vitest.config.ts.

    Pins the inventory so the suite cannot silently shrink (or grow unnoticed) while it stays
    unrunnable; update the counts deliberately when specs are added or deleted.
    """
    _merge_file("ui/vitest.config.ts")
    _merge_file("ui/src/test/setup.ts")
    specs = _spec_files()
    cases = sum(len(_IT_CASE_RE.findall(_text(path))) for path in specs)

    assert len(specs) == 11, [path.relative_to(REPO_ROOT).as_posix() for path in specs]
    assert cases == 37


def test_e9_18_chart_embeds_every_ui_migration() -> None:
    """The chart's embedded migration set covers ui/supabase/migrations (or documents the subset)."""
    chart_dir = _merge_dir(f"{_CHART}/files/supabase-migrations")
    ui_dir = _merge_dir("ui/supabase/migrations")

    chart_names = {path.name for path in chart_dir.glob("*.sql")}
    ui_names = {path.name for path in ui_dir.glob("*.sql")}
    assert ui_names, "ui/supabase/migrations holds no .sql files"

    documented_subset = [path.name for path in chart_dir.iterdir() if path.suffix.lower() != ".sql"]
    missing = sorted(ui_names - chart_names)

    assert not missing or documented_subset, (
        f"{len(missing)} UI migrations are not embedded in the chart and no subset marker "
        f"documents the omission; first missing: {missing[:3]}"
    )


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E9.18a (P0): the supabase-migrate Job runs psql with ON_ERROR_STOP=0 and "
        "'|| echo ... ignored' (deploy/helm/avaloka/templates/supabase.yaml), so the Job "
        "reports success even when every migration fails. Remove this xfail when fixed."
    ),
)
def test_e9_18a_migrate_job_does_not_mask_sql_errors() -> None:
    """The supabase-migrate Job fails the release when a migration errors."""
    template = _text(_merge_file(f"{_CHART}/templates/supabase.yaml"))
    psql_lines = [line.strip() for line in template.splitlines() if "psql" in line and "-f" in line]
    assert psql_lines, "supabase.yaml no longer invokes psql with -f - the extractor is stale"

    masking = [
        line
        for line in psql_lines
        if "ON_ERROR_STOP=1" not in line or "||" in line
    ]
    assert not masking, f"migration psql invocations swallow failures: {masking}"


@pytest.mark.parametrize("values_rel", _VALUES_FILES)
def test_e9_19_shipped_values_never_pair_demo_secret_with_enabled_supabase(values_rel: str) -> None:
    """No shipped values file turns on in-cluster Supabase while keeping the public demo secret."""
    values = _load_values(values_rel)
    supabase = values.get("supabase") or {}
    if not supabase.get("enabled"):
        return

    assert supabase.get("jwtSecret") != _DEMO_JWT_SECRET, (
        f"{values_rel} enables Supabase with the well-known public demo JWT secret; "
        "templates/secret.yaml adopts it as the API's SUPABASE_JWT_SECRET, so anyone can "
        "forge valid HS256 tokens for every endpoint"
    )
    for key in ("anonKey", "serviceKey"):
        token = supabase.get(key) or ""
        assert _jwt_issuer(token) != _DEMO_JWT_ISSUER, (
            f"{values_rel} enables Supabase with the published demo {key}"
        )


@pytest.mark.parametrize("values_rel", _CLOUD_OVERLAYS)
def test_e9_19_cloud_overlays_do_not_enable_in_cluster_supabase(values_rel: str) -> None:
    """Cloud/on-prem overlays leave supabase.enabled off and bring their own managed Supabase."""
    supabase = _load_values(values_rel).get("supabase") or {}
    assert not supabase.get("enabled"), (
        f"{values_rel} enables the bundled demo Supabase; cloud installs must point at a "
        "managed project with real keys"
    )


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E9.19 (P0, security): deploy/helm/avaloka/values.yaml ships the public demo "
        "supabase.jwtSecret/anonKey/serviceKey as chart defaults and templates/secret.yaml "
        "adopts jwtSecret as the API's SUPABASE_JWT_SECRET whenever supabase.enabled=true, so "
        "one --set makes every HS256 token forgeable. Remove this xfail when fixed."
    ),
)
def test_e9_19a_chart_defaults_do_not_ship_public_demo_credentials() -> None:
    """Chart defaults carry no publicly-known Supabase secret that secret.yaml can adopt."""
    secret_template = _text(_merge_file(f"{_CHART}/templates/secret.yaml"))
    if "supabase.jwtSecret" not in secret_template:
        pytest.skip("templates/secret.yaml no longer derives SUPABASE_JWT_SECRET from supabase.jwtSecret")

    supabase = _load_values(f"{_CHART}/values.yaml").get("supabase") or {}
    assert supabase.get("jwtSecret") != _DEMO_JWT_SECRET, (
        "values.yaml defaults supabase.jwtSecret to the public self-hosted demo secret; "
        "it must be empty (or generated) so enabling Supabase cannot hand out a known "
        "API signing key"
    )
    for key in ("anonKey", "serviceKey"):
        assert _jwt_issuer(supabase.get(key) or "") != _DEMO_JWT_ISSUER, (
            f"values.yaml defaults supabase.{key} to the published demo token"
        )


@pytest.mark.parametrize(
    ("component", "value_path"),
    [
        ("webui", ("webui", "service", "nodePort")),
        ("supabase-kong", ("supabase", "kong", "nodePort")),
    ],
)
def test_e9_20_nodeports_match_kind_extra_port_mappings(
    component: str, value_path: tuple[str, ...]
) -> None:
    """Each chart NodePort is mapped to the host by deploy/clusters/kind-cluster.yaml."""
    node_port = _dig(_load_values(f"{_CHART}/values.yaml"), value_path)
    assert isinstance(node_port, int), f"values.yaml has no int at {'.'.join(value_path)}"

    cluster = yaml.safe_load(_text(_merge_file("deploy/clusters/kind-cluster.yaml"))) or {}
    mapped = {
        mapping.get("containerPort")
        for node in cluster.get("nodes", [])
        for mapping in (node.get("extraPortMappings") or [])
    }
    assert node_port in mapped, (
        f"{component} NodePort {node_port} is not in kind extraPortMappings {sorted(p for p in mapped if p)}; "
        "the T6 port-forward path hides this, but the browser reaches the UI over the NodePort"
    )
