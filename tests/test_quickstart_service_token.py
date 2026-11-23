"""The quickstart gives every service-token consumer one generated MCP_SERVICE_TOKEN.

The token is the API's service identity on the onboarding /customers routes,
and the only identity that reaches a database registered before owners were
recorded (#618). It fails closed when unset, so a stock quickstart without it
refused every transfer from such a legacy database. These tests pin both
halves: ``scripts/quickstart.sh`` generates it into ``.env.quickstart``
(mode 600, never printed), and ``docker-compose.quickstart.yml`` hands that one
value to api, langgraph and mcp, and to nothing else.

``ensure_env`` runs for real, from a copy of the script, against a temporary
checkout; ``docker`` is a stub that reports no old volumes. No stack starts.
"""

import os
import pathlib
import re
import shutil
import stat
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.quickstart.yml"
SCRIPT = ROOT / "scripts" / "quickstart.sh"
CONSUMERS = ("api", "langgraph", "mcp")
VAR = re.compile(r"\$\{([A-Z0-9_]+)(?::?[-?][^}]*)?\}")


def _ensure_env(checkout: pathlib.Path, env_file_text: str = None):
    """Run the real ensure_env of a copy of quickstart.sh rooted at ``checkout``."""
    (checkout / "scripts").mkdir(parents=True, exist_ok=True)
    text = SCRIPT.read_text("utf-8")
    dispatch = text.rindex('case "${1:-}" in')
    (checkout / "scripts" / "quickstart.sh").write_text(text[:dispatch] + "ensure_env\n", "utf-8")
    bindir = checkout / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "docker"
    stub.write_text("#!/bin/sh\nexit 0\n", "utf-8")
    stub.chmod(0o755)
    if env_file_text is not None:
        (checkout / ".env.quickstart").write_text(env_file_text, "utf-8")
        (checkout / ".env.quickstart").chmod(0o600)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"}
    env.pop("MCP_SERVICE_TOKEN", None)
    return subprocess.run(["bash", str(checkout / "scripts" / "quickstart.sh")],
                          cwd=checkout, env=env, capture_output=True, text=True, timeout=60)


def _env_values(path: pathlib.Path):
    out = {}
    for line in path.read_text("utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k] = v
    return out


@pytest.fixture
def checkout(tmp_path):
    if not (shutil.which("bash") and shutil.which("openssl")):
        pytest.skip("bash and openssl are needed to run quickstart.sh's ensure_env")
    return tmp_path / "qs"


def test_quickstart_generates_the_token_privately(checkout):
    run = _ensure_env(checkout)
    assert run.returncode == 0, run.stderr
    env_file = checkout / ".env.quickstart"
    token = _env_values(env_file).get("MCP_SERVICE_TOKEN", "")
    assert re.fullmatch(r"[0-9a-f]{64}", token), "MCP_SERVICE_TOKEN missing or not 32 random bytes"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert token not in run.stdout and token not in run.stderr, "the token was printed"
    # Reused, not rotated, on the next run.
    assert _ensure_env(checkout).returncode == 0
    assert _env_values(env_file)["MCP_SERVICE_TOKEN"] == token


def test_an_install_from_before_the_token_gets_one_and_keeps_its_secrets(checkout):
    old = "INFERENCE_PROVIDER=local\nDB_ENCRYPTION_KEY=" + "ab" * 32 + "\n"
    run = _ensure_env(checkout, env_file_text=old)
    assert run.returncode == 0, run.stderr
    values = _env_values(checkout / ".env.quickstart")
    assert values["DB_ENCRYPTION_KEY"] == "ab" * 32
    assert re.fullmatch(r"[0-9a-f]{64}", values.get("MCP_SERVICE_TOKEN", ""))


def test_compose_hands_one_token_to_every_consumer_and_no_one_else(checkout):
    run = _ensure_env(checkout)
    assert run.returncode == 0, run.stderr
    values = _env_values(checkout / ".env.quickstart")
    services = yaml.safe_load(COMPOSE.read_text("utf-8"))["services"]

    def resolved(service):
        raw = (services[service].get("environment") or {}).get("MCP_SERVICE_TOKEN")
        if raw is None:
            return None
        # Compose interpolation: ${VAR}, ${VAR:-x}, ${VAR:?msg} against .env.quickstart.
        return VAR.sub(lambda m: values.get(m.group(1), ""), str(raw))

    tokens = {s: resolved(s) for s in CONSUMERS}
    assert all(tokens.values()), f"a consumer has no MCP_SERVICE_TOKEN: {sorted(s for s, t in tokens.items() if not t)}"
    assert set(tokens.values()) == {values["MCP_SERVICE_TOKEN"]}, "consumers would hold different tokens"
    # Required, not defaulted: compose refuses to start without it rather than
    # starting the stack with the service identity silently off.
    for s in CONSUMERS:
        assert ":?" in str(services[s]["environment"]["MCP_SERVICE_TOKEN"])
    others = [s for s in services if s not in CONSUMERS and resolved(s) is not None]
    assert others == [], f"the token reaches services that do not use it: {others}"


CHART = ROOT / "deploy" / "helm" / "avaloka"


def _chart_token(*args):
    out = subprocess.run(["helm", "template", "t", str(CHART), *args],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    secrets = [d for d in yaml.safe_load_all(out.stdout)
               if isinstance(d, dict) and d.get("kind") == "Secret"
               and "MCP_SERVICE_TOKEN" in (d.get("stringData") or {})]
    assert len(secrets) == 1, "MCP_SERVICE_TOKEN must be in exactly one rendered Secret"
    return secrets[0]["stringData"]["MCP_SERVICE_TOKEN"]


def test_chart_never_renders_an_empty_token():
    if not shutil.which("helm"):
        pytest.skip("helm is not installed")
    generated = _chart_token()
    assert generated and len(generated) >= 32, "the chart rendered MCP_SERVICE_TOKEN empty"
    pinned = "pin" + "0" * 40
    assert _chart_token("--set-string", f"secrets.mcpServiceToken={pinned}") == pinned
