"""A container never generates its own secrets; it adopts its env file's, or refuses.

1.6's provisioning (d0737d14) let a service generate its secrets
(DB_ENCRYPTION_KEY, MCP_SERVICE_TOKEN, EXECUTE_SERVICE_TOKEN, AVALOKA_PII_KEY)
at startup wherever it was not in Kubernetes. Inside a Docker container that writes the key into the container's
own filesystem: two containers of one deployment get two keys, and each loses
its key when the container is replaced. Port of 2.x's rule
(fix/2.x-encryption-key-self-provision 396baf8e): a container adopts what its
env file holds and never generates unless AVALOKA_AUTO_PROVISION=1; Kubernetes
pods never generate.

The container is simulated the way the code detects one (``/.dockerenv``).
Every value here is generated per test; none is real, and no assertion prints one.
"""
from __future__ import annotations

import os
import re
import secrets
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api.cloud_connections as cc
from app.core.provisioning import GENERATED_SECRETS

GENERATED = tuple(GENERATED_SECRETS)  # every secret provisioning knows, so a new one is covered


def _env_values(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"^\s*([A-Z_][A-Z0-9_]*)=(.*)$", line)
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


def _start(app):
    try:
        with TestClient(app):
            return None
    except Exception as exc:  # noqa: BLE001 - whatever startup raised is the answer
        return exc


@pytest.fixture
def mcp_start(tmp_path, monkeypatch):
    """The MCP server's startup, in a simulated Docker container with no secrets set."""
    import app.mcp_server.multi_tenant_mcp_server as mts

    real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists",
                        lambda p: True if str(p) == "/.dockerenv" else real_exists(p))
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    env_file = tmp_path / ".env"
    monkeypatch.setenv("AVALOKA_ENV_FILE", str(env_file))
    for n in GENERATED:  # empty, not deleted: monkeypatch then restores what the start sets
        monkeypatch.setenv(n, "")
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", None, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    monkeypatch.setattr(mts, "CUSTOMERS_FILE", str(tmp_path / "customers.json"))
    return mts, env_file


def test_a_container_without_a_key_writes_none_and_refuses_with_the_fix(mcp_start, monkeypatch):
    mts, env_file = mcp_start
    monkeypatch.delenv("AVALOKA_AUTO_PROVISION", raising=False)
    err = _start(mts.app)
    written = sorted(n for n in GENERATED if _env_values(env_file).get(n))
    assert not written, f"a container generated and wrote {written} into its own filesystem"
    assert err is not None, "a container with no key started anyway"
    msg = str(err)
    assert "DB_ENCRYPTION_KEY" in msg and "env_file" in msg, "the refusal does not say how to give the container its key"


def test_a_container_adopts_every_secret_its_env_file_holds(mcp_start, monkeypatch):
    mts, env_file = mcp_start
    monkeypatch.delenv("AVALOKA_AUTO_PROVISION", raising=False)
    saved = {n: "shared-" + secrets.token_hex(24) for n in GENERATED}
    env_file.write_text("".join(f"{k}={v}\n" for k, v in saved.items()))
    err = _start(mts.app)
    assert err is None, f"a container with its env file's key did not start: {type(err).__name__}"
    assert cc.DB_ENCRYPTION_KEY == saved["DB_ENCRYPTION_KEY"], "the container did not adopt the shared key"
    assert all(os.environ.get(n) == saved[n] for n in GENERATED), "a shared secret was not adopted"
    assert _env_values(env_file) == saved, "the container rewrote the shared env file"


def test_a_container_generates_only_when_opted_in(mcp_start, monkeypatch):
    mts, env_file = mcp_start
    monkeypatch.setenv("AVALOKA_AUTO_PROVISION", "1")
    err = _start(mts.app)
    assert err is None, f"an opted-in container did not start: {type(err).__name__}"
    assert all(len(_env_values(env_file).get(n, "")) >= 32 for n in GENERATED), "opt-in did not generate"


def test_a_kubernetes_pod_never_generates_even_when_opted_in(mcp_start, monkeypatch):
    mts, env_file = mcp_start
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("AVALOKA_AUTO_PROVISION", "1")
    err = _start(mts.app)
    assert not _env_values(env_file), "a Kubernetes pod generated its own secret"
    assert err is not None, "a pod with no key started anyway"


def test_on_a_host_the_first_start_still_generates(tmp_path, monkeypatch):
    """1.6's existing rule for non-container runs is unchanged."""
    import app.mcp_server.multi_tenant_mcp_server as mts
    real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists",
                        lambda p: False if str(p) in ("/.dockerenv", "/run/.containerenv") else real_exists(p))
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.delenv("AVALOKA_AUTO_PROVISION", raising=False)
    env_file = tmp_path / ".env"
    monkeypatch.setenv("AVALOKA_ENV_FILE", str(env_file))
    for n in GENERATED:
        monkeypatch.setenv(n, "")
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", None, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    monkeypatch.setattr(mts, "CUSTOMERS_FILE", str(tmp_path / "customers.json"))
    assert _start(mts.app) is None
    assert all(len(_env_values(env_file).get(n, "")) >= 32 for n in GENERATED), "a host start did not generate"


def test_adopted_secrets_never_reach_a_model_written_script(mcp_start, monkeypatch):
    """What a container adopts from its env file stays out of the executor (#662's allowlist)."""
    from app.executor.env import scrubbed_env
    mts, env_file = mcp_start
    monkeypatch.delenv("AVALOKA_AUTO_PROVISION", raising=False)
    saved = {n: "shared-" + secrets.token_hex(24) for n in GENERATED}
    env_file.write_text("".join(f"{k}={v}\n" for k, v in saved.items()))
    assert _start(mts.app) is None
    script_env = scrubbed_env()
    leaked = sorted(n for n in GENERATED if n in script_env or saved[n] in script_env.values())
    assert not leaked, f"an adopted secret reaches a model-written script: {leaked}"
