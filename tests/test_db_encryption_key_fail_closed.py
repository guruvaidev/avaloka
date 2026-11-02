"""DB_ENCRYPTION_KEY fails closed only when it cannot be provisioned; the Helm
chart generates a private one and keeps it across upgrades.

CHANGED (owner, after #647): the services no longer hard-fail on an empty or
published key. They provision it (generate, save to .env, reuse) and refuse to
start only when, after provisioning, there is still none -- provisioning
switched off (AVALOKA_AUTO_PROVISION=0), or a Kubernetes pod, which never
generates its own. The startup tests below run in that state; the
self-provisioning itself is tested in tests/test_install_configuration.py. The
chart now replaces a published value instead of failing the render.

Every sealed credential (customer database passwords at rest and in transit,
chat-typed passwords, cloud/MCP/GitHub tokens) is AES-GCM under this key. The
chart used to ship "avaloka-local-dev-encryption-key" as the bundled-Supabase
default, and the services ran with no key at all.

The tests reach the code only through what 1.6 already had (the apps' startup,
the API lifespan, the chart), so on a tree without the fix they fail on their
assertions, not on an import.
"""
from __future__ import annotations

import asyncio
import base64
import secrets
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import app.api.cloud_connections as cc

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "deploy" / "helm" / "avaloka"
PUBLISHED = "avaloka-local-dev-encryption-key"


def _set_key(monkeypatch, value, provisioning=False):
    """Set the key, and (by default) leave the service no way to provision one."""
    monkeypatch.setenv("AVALOKA_AUTO_PROVISION", "1" if provisioning else "0")
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", value, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    if value is None:
        monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
    else:
        monkeypatch.setenv("DB_ENCRYPTION_KEY", value)


def _startup_error(app):
    """The exception the app's startup raised, or None when it started."""
    try:
        with TestClient(app):
            return None
    except Exception as exc:  # noqa: BLE001 - whatever startup raised is the answer
        return exc


# ── the services' startup ───────────────────────────────────────────────────

@pytest.mark.parametrize("value", [None, "", "   ", PUBLISHED, PUBLISHED.upper()],
                         ids=["unset", "empty", "blank", "published", "published-case"])
def test_the_mcp_server_refuses_to_start_when_no_key_can_be_provisioned(monkeypatch, tmp_path, value):
    import app.mcp_server.multi_tenant_mcp_server as mts
    monkeypatch.setattr(mts, "CUSTOMERS_FILE", str(tmp_path / "customers.json"))
    _set_key(monkeypatch, value)
    err = _startup_error(mts.app)
    assert err is not None, "the MCP server started with no usable DB_ENCRYPTION_KEY and no way to provision one"
    msg = str(err)
    assert "DB_ENCRYPTION_KEY" in msg and "python -m app.core.provisioning" in msg, \
        "the startup error does not say how to set the key"


def test_the_mcp_server_starts_on_a_private_key(monkeypatch, tmp_path):
    import app.mcp_server.multi_tenant_mcp_server as mts
    monkeypatch.setattr(mts, "CUSTOMERS_FILE", str(tmp_path / "customers.json"))
    _set_key(monkeypatch, "private-" + secrets.token_hex(24))
    assert _startup_error(mts.app) is None


@pytest.mark.parametrize("value", [None, PUBLISHED], ids=["unset", "published"])
def test_the_onboarding_service_refuses_to_start_when_no_key_can_be_provisioned(monkeypatch, value):
    import app.mcp_server.customer_dbs as cdb
    _set_key(monkeypatch, value)
    err = _startup_error(cdb.app)
    assert err is not None and "DB_ENCRYPTION_KEY" in str(err), \
        "the onboarding service started with no usable DB_ENCRYPTION_KEY and no way to provision one"


class _PastTheKeyCheck(BaseException):
    """Raised by the lifespan step after the key check: startup got past it."""


@pytest.mark.parametrize("value", [None, PUBLISHED], ids=["unset", "published"])
def test_the_api_refuses_to_start_when_no_key_can_be_provisioned(monkeypatch, value):
    import app.api.server as server
    import app.agents.sampling_persistence as sp
    _set_key(monkeypatch, value)
    monkeypatch.setattr(server, "_validate_auth_config", lambda: None, raising=False)

    def _next_step():
        raise _PastTheKeyCheck()

    monkeypatch.setattr(sp, "warn_on_supabase_credential_mismatch", _next_step, raising=False)

    async def _start():
        async with server.server_lifespan(server.app):
            pass

    try:
        asyncio.run(_start())
        err = None
    except _PastTheKeyCheck:
        err = None
    except Exception as exc:  # noqa: BLE001
        err = exc
    assert err is not None and "DB_ENCRYPTION_KEY" in str(err) and "python -m app.core.provisioning" in str(err), \
        "the API got past startup with no usable DB_ENCRYPTION_KEY and no way to provision one"


# ── the chart ───────────────────────────────────────────────────────────────

def _helm(chart, *args, check=True):
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")
    out = subprocess.run([helm, "template", "t", str(chart), *args], capture_output=True, text=True, timeout=120)
    if check:
        assert out.returncode == 0, out.stderr[-600:]
    return out


def _docs(text):
    return [d for d in yaml.safe_load_all(text) if d]


def _secret_key(text):
    for d in _docs(text):
        if d.get("kind") == "Secret" and "DB_ENCRYPTION_KEY" in (d.get("stringData") or {}):
            return d["stringData"]["DB_ENCRYPTION_KEY"]
    return None


def test_the_chart_generates_a_private_key_when_none_is_given():
    a, b = _secret_key(_helm(CHART).stdout), _secret_key(_helm(CHART).stdout)
    assert a and len(a) >= 32, "the chart renders an empty or short DB_ENCRYPTION_KEY"
    assert a != b, "the chart renders a fixed DB_ENCRYPTION_KEY, not a generated one"
    assert a.lower() != PUBLISHED


def test_the_chart_keeps_the_existing_key_across_upgrades(tmp_path):
    """helm template cannot reach a cluster, so lookup is stubbed with the Secret an
    install left behind; everything else in the template runs as written."""
    # CHANGED: the lookup lives in the avaloka.generatedSecret helper (_helpers.tpl),
    # shared by every generated secret, instead of inline in secret.yaml.
    chart = tmp_path / "avaloka"
    shutil.copytree(CHART, chart)
    helpers = chart / "templates" / "_helpers.tpl"
    text = helpers.read_text()
    lookup = 'lookup "v1" "Secret" $ctx.Release.Namespace (include "avaloka.secretName" $ctx)'
    assert lookup in text, "the chart does not read the release's existing Secret for DB_ENCRYPTION_KEY"
    kept = "kept-" + secrets.token_hex(20)
    stub = f'(dict "data" (dict "DB_ENCRYPTION_KEY" "{base64.b64encode(kept.encode()).decode()}"))'
    helpers.write_text(text.replace(lookup, stub))
    assert _secret_key(_helm(chart).stdout) == kept, "an upgrade rotates DB_ENCRYPTION_KEY"
    # An explicit value still wins over what the Secret held.
    pinned = "pinned-" + secrets.token_hex(20)
    assert _secret_key(_helm(chart, "--set-string", f"secrets.dbEncryptionKey={pinned}").stdout) == pinned


@pytest.mark.parametrize("flag", ["secrets.dbEncryptionKey", "supabase.functions.dbEncryptionKey"])
def test_the_chart_replaces_a_published_key(flag):
    # CHANGED: the render used to fail on a published value; it now generates one instead.
    out = _helm(CHART, "--set-string", f"{flag}={PUBLISHED}")
    key = _secret_key(out.stdout)
    assert key and key.lower() != PUBLISHED and len(key) >= 32, "the chart rendered a published DB_ENCRYPTION_KEY"


def test_bundled_supabase_functions_and_mcp_use_the_release_key():
    out = _helm(CHART, "--set", "supabase.enabled=true", "--set", "supabase.requireOwnCredentials=false").stdout
    rendered_published = PUBLISHED in out  # a boolean: a failure must not print the manifests
    assert not rendered_published, "a published DB_ENCRYPTION_KEY is rendered into the release"
    consumers = {}
    for d in _docs(out):
        if d.get("kind") != "Deployment":
            continue
        for c in d["spec"]["template"]["spec"].get("containers") or []:
            for e in c.get("env") or []:
                if e.get("name") == "DB_ENCRYPTION_KEY":
                    consumers[d["metadata"]["name"]] = e
    for name in ("t-avaloka-supabase-functions", "t-avaloka-mcp"):
        match = next((e for n, e in consumers.items() if n.endswith(name.split("t-avaloka-")[1])), None)
        assert match is not None, f"{name} does not get DB_ENCRYPTION_KEY"
        ref = (match.get("valueFrom") or {}).get("secretKeyRef") or {}
        assert ref.get("key") == "DB_ENCRYPTION_KEY" and "value" not in match, \
            f"{name} does not read DB_ENCRYPTION_KEY from the release Secret"
