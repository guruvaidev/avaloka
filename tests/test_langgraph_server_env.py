"""The LangGraph server's own environment starts `langgraph dev` and serves the thread registry.

requirements.txt no longer carries the LangGraph CLI: its langgraph-api 0.5.x caps
cryptography below the API's. The server is installed from
requirements-langgraph-server.txt instead, and in CI it runs in a venv of its own
(scripts/langgraph_server_tests.sh). This module is the proof, on every pull
request, that the file still builds a server that starts and answers what the API
asks of it: GET /ok, and POST /threads then /threads/search.

It imports the CLI, so tests/conftest.py marks it ``langgraph_server``. The main
environment deselects that marker and the server venv selects it. If the CLI is
missing where it is selected, it fails under CI rather than skipping.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import secrets
import socket
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _need_cli() -> None:
    have = (importlib.util.find_spec("langgraph_cli") is not None
            and importlib.util.find_spec("langgraph_api") is not None)
    if have:
        return
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
        pytest.fail("langgraph-cli[inmem] is not installed in this environment; CI is set, so this "
                    "is a failure rather than a skip (scripts/langgraph_server_tests.sh installs it)")
    pytest.skip("langgraph-cli[inmem] is not installed (requirements-langgraph-server.txt)")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    _need_cli()
    tmp = tmp_path_factory.mktemp("lgenv")
    (tmp / "graph.py").write_text(
        "from typing import TypedDict\nfrom langgraph.graph import StateGraph, END\n"
        "class S(TypedDict):\n    x: int\n"
        "g = StateGraph(S)\ng.add_node('n', lambda s: {'x': 1})\ng.set_entry_point('n')\ng.add_edge('n', END)\n"
        "graph = g.compile()\n")
    # The repository's langgraph.json, with only the graph swapped (the real one
    # loads embedding models) and .env dropped; relative module paths made absolute.
    cfg = json.loads((ROOT / "langgraph.json").read_text())
    cfg["graphs"] = {"avaloka": f"{tmp / 'graph.py'}:graph"}
    cfg.pop("env", None)
    for key in ("http", "http_server"):
        app = (cfg.get(key) or {}).get("app", "")
        if isinstance(app, str) and app.startswith("./"):
            cfg[key]["app"] = str(ROOT / app[2:])
    (tmp / "langgraph.json").write_text(json.dumps(cfg))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    # A service token is set and sent on every call. Harmless where the server
    # has no token gate; required where it has one.
    token = secrets.token_hex(24)
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "TMPDIR") if k in os.environ}
    env.update({"LANGGRAPH_SERVICE_TOKEN": token, "PYTHONPATH": str(ROOT), "AVALOKA_AUTO_PROVISION": "0",
                "LANGSMITH_TRACING": "false", "LANGGRAPH_CLI_NO_ANALYTICS": "1",
                "LANGGRAPH_NO_VERSION_CHECK": "true", "PYTHONDONTWRITEBYTECODE": "1"})
    log = tmp / "server.log"
    with log.open("w") as fh:
        proc = subprocess.Popen(
            [sys.executable, "-c", "from langgraph_cli.cli import cli; cli()", "dev",
             "--config", str(tmp / "langgraph.json"), "--host", "127.0.0.1", "--port", str(port),
             "--no-browser", "--no-reload"],
            cwd=str(tmp), env=env, stdout=fh, stderr=subprocess.STDOUT)
    import httpx

    url = f"http://127.0.0.1:{port}"
    up = False
    deadline = time.time() + 180
    while time.time() < deadline and proc.poll() is None:
        try:
            if httpx.get(url + "/ok", timeout=2).status_code == 200:
                up = True
                break
        except httpx.HTTPError:
            pass
        time.sleep(1)
    yield {"url": url if up else None, "auth": {"Authorization": f"Bearer {token}"}, "log": log}
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def _tail(log: pathlib.Path) -> str:
    try:
        return log.read_text(errors="replace")[-1500:]
    except OSError:
        return "(no log)"


def test_the_server_environment_starts_langgraph_dev(server):
    assert server["url"], f"langgraph dev did not answer /ok within 180 s:\n{_tail(server['log'])}"


def test_the_server_serves_the_thread_registry_the_api_uses(server):
    import httpx

    assert server["url"], f"langgraph dev did not start:\n{_tail(server['log'])}"
    mark = secrets.token_hex(6)
    created = httpx.post(server["url"] + "/threads", headers=server["auth"],
                         json={"metadata": {"probe": mark}}, timeout=15)
    assert created.status_code == 200, created.status_code
    tid = created.json()["thread_id"]
    found = httpx.post(server["url"] + "/threads/search", headers=server["auth"],
                       json={"metadata": {"probe": mark}}, timeout=15)
    assert found.status_code == 200 and [t["thread_id"] for t in found.json()] == [tid]
