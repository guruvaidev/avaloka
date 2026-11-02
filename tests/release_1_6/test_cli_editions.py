"""1.6 release suite: CLI, MCP and editions (plan sheet 8).

Every test names the plan case it certifies and drives the shipped entrypoint
(the install --check script, the MCP credential endpoints with the real
AES-GCM cipher, the prompt-suite runner, the edition resolver) rather than a
re-implementation of it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# CM-01: install --check reports the edition and every capability.
# --------------------------------------------------------------------------- #

def _run_check_edition(tmp_path: Path, **extra_env: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("AVALOKA_") and k not in ("CLOUD_PROVIDER",)}
    env.update({"AVALOKA_HOME": str(tmp_path), "PYTHONPATH": str(REPO)})
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_edition.py")],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120,
    )


def test_cm01_check_edition_reports_oss_and_every_capability(tmp_path):
    """CM-01: `install.sh --check` (scripts/check_edition.py) exits 0 and reports the edition and every capability with its state."""
    from app.core.editions import GATE_REASON

    proc = _run_check_edition(tmp_path)

    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    lines = out.splitlines()
    assert any(l.split() == ["edition", "oss"] for l in lines), out
    assert any(l.split() == ["deployment", "self_hosted"] for l in lines), out
    reported = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("on", "off") and parts[1] in GATE_REASON:
            reported[parts[1]] = parts[0]
    assert set(reported) == set(GATE_REASON), sorted(set(GATE_REASON) - set(reported))


def test_cm01_check_edition_never_prints_the_licence_token(tmp_path):
    """CM-01: the --check report never echoes the licence token, even when the token is rejected."""
    token = "test-licence-tok"  # deliberately not a valid licence
    proc = _run_check_edition(tmp_path, AVALOKA_LICENSE_KEY=token)

    assert proc.returncode == 0, proc.stderr
    assert token not in proc.stdout
    assert token not in proc.stderr
    assert "rejected" in proc.stdout  # the token was read and refused, not ignored


# --------------------------------------------------------------------------- #
# CM-06: MCP connection credentials round-trip through the real cipher.
# --------------------------------------------------------------------------- #

from tests.server_harness import client, make_auth_headers  # noqa: E402,F401  (pytest fixture)


@pytest.fixture
def real_cipher(monkeypatch):
    import app.api.cloud_connections as cc

    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", "test-db-enc-key")
    monkeypatch.setattr(cc, "_AESGCM_KEY", None)
    yield cc
    monkeypatch.setattr(cc, "_AESGCM_KEY", None)


def test_cm06_mcp_credentials_round_trip_with_the_real_cipher(client, real_cipher, monkeypatch):
    """CM-06: encrypting MCP connection credentials stores only ciphertext at rest, and decrypting that stored row recovers the original values."""
    import app.api.server as server

    row: Dict[str, Any] = {"id": "conn-1", "user_id": "mcp-owner", "customer_id": None}
    monkeypatch.setattr(server, "_mcp_conn_select", lambda cid: dict(row), raising=False)
    monkeypatch.setattr(server, "_mcp_conn_update", lambda cid, patch: row.update(patch), raising=False)
    api_key, username = "test-api-key-1", "user-1"

    enc = client.post("/api/mcp-connections/conn-1/encrypt",
                      json={"apiKey": api_key, "username": username},
                      headers=make_auth_headers("mcp-owner"))
    assert enc.status_code == 200, enc.text
    assert sorted(enc.json()["encrypted"]) == ["api_key", "username"]

    stored = json.dumps(row)
    assert api_key not in stored and username not in stored, "plaintext kept at rest"
    assert row["api_key"] == "" and row["username"] == ""
    assert row["api_key_ciphertext"] and row["api_key_iv"]

    dec = client.post("/api/mcp-connections/conn-1/decrypt", headers=make_auth_headers("mcp-owner"))
    assert dec.status_code == 200, dec.text
    assert dec.json() == {"apiKey": api_key, "username": username}


def test_cm06_ciphertext_does_not_open_under_another_key(client, real_cipher, monkeypatch):
    """CM-06: stored credentials are bound to DB_ENCRYPTION_KEY; under a different key decrypt refuses (409) instead of returning garbage."""
    import app.api.server as server

    row: Dict[str, Any] = {"id": "conn-2", "user_id": "mcp-owner", "customer_id": None}
    monkeypatch.setattr(server, "_mcp_conn_select", lambda cid: dict(row), raising=False)
    monkeypatch.setattr(server, "_mcp_conn_update", lambda cid, patch: row.update(patch), raising=False)
    assert client.post("/api/mcp-connections/conn-2/encrypt", json={"apiKey": "test-api-key-1"},
                       headers=make_auth_headers("mcp-owner")).status_code == 200

    monkeypatch.setattr(real_cipher, "DB_ENCRYPTION_KEY", "other-db-enc-key")
    monkeypatch.setattr(real_cipher, "_AESGCM_KEY", None)
    dec = client.post("/api/mcp-connections/conn-2/decrypt", headers=make_auth_headers("mcp-owner"))
    assert dec.status_code == 409, dec.text


# --------------------------------------------------------------------------- #
# CM-08: the prompt-suite inventories the runner executes are intact.
# --------------------------------------------------------------------------- #

_SUITE_SIZES = {"v4": 86, "v2": 96, "heavy": 19}


@pytest.mark.parametrize("suite,expected", sorted(_SUITE_SIZES.items()))
def test_cm08_prompt_suite_runner_lists_every_item(suite, expected):
    """CM-08: the prompt-suite runner (run_suite.py --list) enumerates the full suite (v4: 86 items, v2: 96, heavy: 19)."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "tests" / "prompt_suite" / "run_suite.py"), "--suite", suite, "--list"],
        capture_output=True, text=True, cwd=str(REPO), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    counts = [int(l.rsplit(None, 2)[-2]) for l in proc.stdout.splitlines() if l.rstrip().endswith(" items")]
    assert counts, proc.stdout
    assert sum(counts) == expected, (suite, counts)


def test_cm08_v4_items_are_runnable_and_checkable():
    """CM-08: every v4 item has a unique id, something to execute (a prompt or an op) and an expectation to check the result against."""
    suite = json.loads((REPO / "tests" / "prompt_suite" / "suites" / "v4_suite.json").read_text())
    items = [i for b in suite["batches"] for i in b["items"]]

    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)) == _SUITE_SIZES["v4"]
    assert [i["id"] for i in items if not (i.get("prompt") or i.get("op"))] == []
    assert [i["id"] for i in items if not i.get("expect")] == []


# --------------------------------------------------------------------------- #
# CM-12: commercial capabilities are absent from this build, not just switched off.
# --------------------------------------------------------------------------- #

def test_cm12_commercial_package_is_not_on_disk():
    """CM-12: the commercial package is genuinely absent from this build (not importable, no directory), so commercial_build() is False."""
    from app.core.editions import commercial_build

    assert importlib.util.find_spec("avaloka_commercial") is None
    assert not list(REPO.glob("**/avaloka_commercial/__init__.py"))
    assert commercial_build() is False


def test_cm12_commercial_capabilities_refused_for_the_commercial_reason():
    """CM-12: with no injection, even an enterprise edition resolves every COMMERCIAL capability off, and explain() gives the commercial reason."""
    from app.core.editions import GATE_REASON, GateReason, explain, resolve_capabilities

    commercial = sorted(n for n, r in GATE_REASON.items() if r is GateReason.COMMERCIAL)
    assert commercial, "no COMMERCIAL capabilities declared"

    # The editions each capability is really sold on, read from the resolved
    # matrix of a commercial build rather than from what explain() says.
    def sold_on(name):
        return [e.title() for e in ("professional", "enterprise")
                if getattr(resolve_capabilities("avaloka_hosted", e, has_commercial_build=True), name)]

    # Owner's rule: what is Enterprise-only. Everything else commercial is both.
    enterprise_only = {"org_structure", "sso_audit"}
    for name in commercial:
        assert sold_on(name) == (["Enterprise"] if name in enterprise_only else ["Professional", "Enterprise"]), name

    for deployment in ("self_hosted", "avaloka_hosted"):
        caps = resolve_capabilities(deployment, "enterprise")  # discovers the build from disk
        for name in commercial:
            assert getattr(caps, name) is False, (deployment, name)
            # The whole sentence, so a wrong, extra or missing edition name fails.
            assert explain(name, caps) == f"{name} is available in Avaloka {' and '.join(sold_on(name))}.", (deployment, name)
