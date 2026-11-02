"""Install is autonomous from .env: generated secrets are created, persisted and reused.

The owner's rule after #647: "make the encryption and install autonomous from the
.env file, and don't hard-fail for manual steps". DB_ENCRYPTION_KEY,
MCP_SERVICE_TOKEN and AVALOKA_PII_KEY are generated when unset, empty or a
published value, saved to .env (the release Secret on Helm), and reused on every
later start. A service refuses to start only when, after that, it still has no
key. docs/INSTALL.md's parameter table and .env.example list the same names.

Every value here is generated per test; none is real, and no assertion prints one.
The tests reach the code through the installer's entry point (a subprocess), the
services' startup and the chart, so on a tree without the change they fail on
their assertions, not on an import.
"""
from __future__ import annotations

import base64
import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import app.api.cloud_connections as cc

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "deploy" / "helm" / "avaloka"
GENERATED = ("DB_ENCRYPTION_KEY", "MCP_SERVICE_TOKEN", "AVALOKA_PII_KEY")
PUBLISHED = "avaloka-local-dev-encryption-key"


def _env_values(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"^\s*([A-Z_][A-Z0-9_]*)=(.*)$", line)
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


def _provision_cli(env_file: Path, extra_env=None):
    """The installer's step (scripts/install.sh runs exactly this)."""
    env = {k: v for k, v in os.environ.items() if k not in GENERATED}
    env.update(extra_env or {})
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run([sys.executable, "-m", "app.core.provisioning", "--env-file", str(env_file)],
                          capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=60)


def _usable(v):
    return bool(v) and len(v) >= 32 and v.lower() != PUBLISHED


def _start(app):
    """Run the app's startup; return the exception it raised, or None."""
    try:
        with TestClient(app):
            return None
    except Exception as exc:  # noqa: BLE001 - whatever startup raised is the answer
        return exc


# ── (a) a fresh install generates and persists every generated secret ───────

def test_a_fresh_install_generates_and_saves_every_generated_secret(tmp_path):
    env_file = tmp_path / ".env"
    out = _provision_cli(env_file)
    values = _env_values(env_file)
    missing = [n for n in GENERATED if not _usable(values.get(n))]
    assert out.returncode == 0 and not missing, f"not generated into .env: {missing} (rc={out.returncode})"
    assert oct(env_file.stat().st_mode & 0o777) == "0o600", "the .env holding the secrets is not mode 600"
    printed = out.stdout + out.stderr
    assert not any(values[n] in printed for n in GENERATED), "a generated secret was printed"
    assert "generated and saved to" in printed


def test_a_fresh_service_start_generates_and_saves_the_secrets(tmp_path, monkeypatch):
    """No installer run at all: the first service start provisions the same .env."""
    import app.mcp_server.multi_tenant_mcp_server as mts
    env_file = tmp_path / ".env"
    monkeypatch.setenv("AVALOKA_ENV_FILE", str(env_file))
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setenv("AVALOKA_AUTO_PROVISION", "1")
    for n in GENERATED:  # empty, not deleted: monkeypatch then restores what the start sets
        monkeypatch.setenv(n, "")
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", None, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    monkeypatch.setattr(mts, "CUSTOMERS_FILE", str(tmp_path / "customers.json"))
    err = _start(mts.app)
    assert err is None, f"the MCP server did not start: {type(err).__name__}"
    values = _env_values(env_file)
    missing = [n for n in GENERATED if not _usable(values.get(n))]
    assert not missing, f"the first start did not generate and save: {missing}"
    assert cc.DB_ENCRYPTION_KEY == values["DB_ENCRYPTION_KEY"], "the service is not using the key it saved"


# ── (b) a second start reuses what was saved ────────────────────────────────

def test_a_second_run_reuses_the_saved_secrets(tmp_path):
    env_file = tmp_path / ".env"
    _provision_cli(env_file)
    first = _env_values(env_file)
    out = _provision_cli(env_file)
    second = _env_values(env_file)
    assert first and all(_usable(first.get(n)) for n in GENERATED), "nothing was generated the first time"
    assert all(second.get(n) == first.get(n) for n in GENERATED), "a second run regenerated a saved secret"
    assert "kept the value already in" in out.stdout


def test_a_second_service_start_reuses_the_saved_key(tmp_path, monkeypatch):
    import app.mcp_server.multi_tenant_mcp_server as mts
    env_file = tmp_path / ".env"
    saved = {n: "saved-" + secrets.token_hex(24) for n in GENERATED}
    env_file.write_text("".join(f"{k}={v}\n" for k, v in saved.items()))
    monkeypatch.setenv("AVALOKA_ENV_FILE", str(env_file))
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setenv("AVALOKA_AUTO_PROVISION", "1")
    for n in GENERATED:  # empty, not deleted: monkeypatch then restores what the start sets
        monkeypatch.setenv(n, "")
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", None, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    monkeypatch.setattr(mts, "CUSTOMERS_FILE", str(tmp_path / "customers.json"))
    err = _start(mts.app)
    assert err is None, f"the MCP server did not start: {type(err).__name__}"
    assert _env_values(env_file) == saved, "a start rewrote the saved secrets"
    assert cc.DB_ENCRYPTION_KEY == saved["DB_ENCRYPTION_KEY"], "the service did not adopt the saved key"


# ── (c) a published value is replaced, with a warning ───────────────────────

def test_a_published_value_is_replaced_with_a_warning(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"DB_ENCRYPTION_KEY={PUBLISHED}\nMCP_SERVICE_TOKEN=<generate-with-openssl-rand-hex-32>\n")
    out = _provision_cli(env_file)
    values = _env_values(env_file)
    assert _usable(values.get("DB_ENCRYPTION_KEY")), "the published DB_ENCRYPTION_KEY was kept"
    assert _usable(values.get("MCP_SERVICE_TOKEN")), "the published MCP_SERVICE_TOKEN placeholder was kept"
    assert "WARNING" in out.stdout and "published" in out.stdout, "no warning that a published value was replaced"
    assert PUBLISHED not in env_file.read_text()


def test_a_service_started_with_a_published_key_replaces_it(tmp_path, monkeypatch, caplog):
    import app.mcp_server.multi_tenant_mcp_server as mts
    caplog.set_level("WARNING")
    env_file = tmp_path / ".env"
    monkeypatch.setenv("AVALOKA_ENV_FILE", str(env_file))
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setenv("AVALOKA_AUTO_PROVISION", "1")
    for n in GENERATED:  # empty, not deleted: monkeypatch then restores what the start sets
        monkeypatch.setenv(n, "")
    monkeypatch.setenv("DB_ENCRYPTION_KEY", PUBLISHED)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", PUBLISHED, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    monkeypatch.setattr(mts, "CUSTOMERS_FILE", str(tmp_path / "customers.json"))
    err = _start(mts.app)
    assert err is None, f"the MCP server did not start: {type(err).__name__}"
    assert _usable(cc.DB_ENCRYPTION_KEY), "the service kept running on the published key"
    assert _env_values(env_file).get("DB_ENCRYPTION_KEY") == cc.DB_ENCRYPTION_KEY
    assert "published" in caplog.text, "no warning that the published key was replaced"
    assert cc.DB_ENCRYPTION_KEY not in caplog.text, "the replacement key was logged"


# ── (d) still no key after provisioning: refuse, and say how to fix it ──────

@pytest.mark.parametrize("why", ["provisioning off", "kubernetes pod", "read-only env file"])
def test_with_no_key_after_provisioning_the_service_refuses_with_the_fix(tmp_path, monkeypatch, why):
    import app.mcp_server.multi_tenant_mcp_server as mts
    env_file = tmp_path / ".env"
    monkeypatch.setenv("AVALOKA_ENV_FILE", str(env_file))
    monkeypatch.setenv("AVALOKA_AUTO_PROVISION", "0" if why == "provisioning off" else "1")
    if why == "kubernetes pod":
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    else:
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    if why == "read-only env file":
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(0o500)
        monkeypatch.setenv("AVALOKA_ENV_FILE", str(ro / ".env"))
    for n in GENERATED:  # empty, not deleted: monkeypatch then restores what the start sets
        monkeypatch.setenv(n, "")
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", None, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    monkeypatch.setattr(mts, "CUSTOMERS_FILE", str(tmp_path / "customers.json"))
    if why == "read-only env file" and os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    try:
        with TestClient(mts.app):
            err = None
    except Exception as exc:  # noqa: BLE001
        err = exc
    finally:
        (tmp_path / "ro").exists() and (tmp_path / "ro").chmod(0o700)
    assert err is not None, f"the MCP server started with no DB_ENCRYPTION_KEY ({why})"
    assert "DB_ENCRYPTION_KEY" in str(err) and "python -m app.core.provisioning" in str(err), \
        "the refusal does not say how to fix it"


# ── (e) Helm: lookup reuses the existing Secret for every generated secret ──

def _helm(chart, *args):
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")
    out = subprocess.run([helm, "template", "t", str(chart), *args], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-600:]
    return out.stdout


def _secret(text):
    for d in yaml.safe_load_all(text):
        if d and d.get("kind") == "Secret" and "DB_ENCRYPTION_KEY" in (d.get("stringData") or {}):
            return d["stringData"]
    return {}


def test_the_chart_generates_every_generated_secret():
    a, b = _secret(_helm(CHART)), _secret(_helm(CHART))
    missing = [n for n in GENERATED if not _usable(a.get(n))]
    assert not missing, f"the chart renders empty or short: {missing}"
    assert all(a[n] != b[n] for n in GENERATED), "a generated secret renders the same on every install"


def test_the_chart_reuses_an_existing_secret_on_upgrade(tmp_path):
    """helm template cannot reach a cluster: lookup is stubbed with the Secret an
    install left behind; the rest of the chart renders as written."""
    chart = tmp_path / "avaloka"
    shutil.copytree(CHART, chart)
    helpers = chart / "templates" / "_helpers.tpl"
    text = helpers.read_text()
    lookup = 'lookup "v1" "Secret" $ctx.Release.Namespace (include "avaloka.secretName" $ctx)'
    assert lookup in text, "the chart has no lookup of the release Secret for generated secrets"
    kept = {n: "kept-" + secrets.token_hex(20) for n in GENERATED}
    data = " ".join(f'"{n}" "{base64.b64encode(v.encode()).decode()}"' for n, v in kept.items())
    helpers.write_text(text.replace(lookup, f'(dict "data" (dict {data}))'))
    rendered = _secret(_helm(chart))
    assert {n: rendered.get(n) for n in GENERATED} == kept, "an upgrade rotated a generated secret"


def test_the_chart_replaces_a_published_secret_and_a_published_existing_one(tmp_path):
    rendered = _secret(_helm(CHART, "--set-string", f"secrets.dbEncryptionKey={PUBLISHED}",
                             "--set-string", "secrets.mcpServiceToken=changeme"))
    assert _usable(rendered.get("DB_ENCRYPTION_KEY")) and rendered.get("MCP_SERVICE_TOKEN") != "changeme", \
        "the chart rendered a published value"
    chart = tmp_path / "avaloka"
    shutil.copytree(CHART, chart)
    helpers = chart / "templates" / "_helpers.tpl"
    lookup = 'lookup "v1" "Secret" $ctx.Release.Namespace (include "avaloka.secretName" $ctx)'
    old = base64.b64encode(PUBLISHED.encode()).decode()
    helpers.write_text(helpers.read_text().replace(lookup, f'(dict "data" (dict "DB_ENCRYPTION_KEY" "{old}"))'))
    upgraded = _secret(_helm(chart))
    assert _usable(upgraded.get("DB_ENCRYPTION_KEY")), "an upgrade kept a published key from the old Secret"


# ── the one install guide and .env.example list the same parameters ─────────

def _example_names():
    names = []
    for line in (ROOT / ".env.example").read_text().splitlines():
        m = re.match(r"^#?\s*([A-Z][A-Z0-9_]+)=", line)
        if m and m.group(1) not in names:
            names.append(m.group(1))
    return names


def _guide_rows():
    text = (ROOT / "docs" / "INSTALL.md").read_text()
    m = re.search(r"### Every parameter\n(.*?)(?:\n## |\Z)", text, re.S)
    assert m, "docs/INSTALL.md has no 'Every parameter' table"
    rows = {}
    for line in m.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 5 and re.fullmatch(r"`[A-Z][A-Z0-9_]+`", cells[0]):
            rows[cells[0].strip("`")] = cells
    return rows


def test_the_install_guide_table_and_env_example_list_the_same_parameters():
    rows, names = _guide_rows(), _example_names()
    assert sorted(set(names) - set(rows)) == [], "in .env.example but not in the install guide's table"
    assert sorted(set(rows) - set(names)) == [], "in the install guide's table but not in .env.example"


def test_the_guide_marks_exactly_the_generated_secrets_as_generated():
    rows = _guide_rows()
    generated = sorted(n for n, cells in rows.items() if "generated automatically" in cells[3])
    assert generated == sorted(GENERATED), f"the table's generated rows are {generated}"
    example = (ROOT / ".env.example").read_text()
    for n in GENERATED:
        assert re.search(rf"^{n}=\s*$", example, re.M), f".env.example ships a value for generated {n}"
