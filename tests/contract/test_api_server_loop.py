"""The API is served on the asyncio event loop, never uvicorn's default uvloop.

Under uvloop the 1.6 API process segfaults (exit 139, a NULL read callback in
libuv ``uv__read``). The graph runs in an executor thread, and
``avaloka_agent._load_redis_context`` does ``asyncio.run`` there against the
process-wide ``redis.asyncio`` pool the request handlers use. A pooled
connection belongs to the server loop. When the worker's read fails, redis-py
closes that uvloop transport from the worker thread, while the server thread is
inside ``uv_run`` without the GIL. ``--loop asyncio`` turns that crash into a
Python exception. Every place the API command is defined must keep it until the
cross-loop use is gone.

Pure source-text checks: nothing imports the app or starts a server.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
from typing import List, Tuple

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_API_TARGET = "app.api.server:app"
_API_DOCKERFILES = (
    "deploy/docker/Dockerfile.api",
    "deploy/avaloka/charts/avaloka-backend/docker/Dockerfile.api",
)
# Where a command line that serves the API can live outside app/.
_SCAN_GLOBS = (
    "deploy/**/*",
    "scripts/**/*",
    "docker-compose*.yml",
    "Makefile",
    "CONTRIBUTING.md",
    "README*.md",
    "docs/**/*.md",
    "docs/**/*.html",
)
_PINNED_RE = re.compile(r'--loop[ =]asyncio|"--loop",\s*"asyncio"|\'--loop\',\s*\'asyncio\'')


def _cmd_argv(dockerfile: str) -> List[str]:
    """The exec-form argv of the Dockerfile's last CMD."""
    text = (REPO_ROOT / dockerfile).read_text(encoding="utf-8")
    cmds = re.findall(r"^CMD\s+(\[.*\])\s*$", text, re.M)
    assert cmds, f"{dockerfile} has no exec-form CMD"
    return json.loads(cmds[-1])


@pytest.mark.parametrize("dockerfile", _API_DOCKERFILES)
def test_api_dockerfile_cmd_runs_the_asyncio_loop(dockerfile):
    argv = _cmd_argv(dockerfile)
    assert _API_TARGET in argv, f"{dockerfile} CMD no longer serves {_API_TARGET}: {argv}"
    assert "--loop" in argv, (
        f"{dockerfile} CMD has no --loop, so uvicorn picks uvloop and the API "
        f"segfaults on the cross-loop Redis read: {argv}")
    loop = argv[argv.index("--loop") + 1] if argv.index("--loop") + 1 < len(argv) else None
    assert loop == "asyncio", f"{dockerfile} CMD runs --loop {loop!r}, not asyncio: {argv}"


def test_server_main_runs_the_asyncio_loop():
    """`python -m app.api.server` must not fall back to uvloop either."""
    tree = ast.parse((REPO_ROOT / "app/api/server.py").read_text(encoding="utf-8"))
    runs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "uvicorn"
        and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == _API_TARGET
    ]
    assert runs, "app/api/server.py no longer calls uvicorn.run('app.api.server:app', ...)"
    for call in runs:
        loop = {k.arg: k.value for k in call.keywords}.get("loop")
        assert isinstance(loop, ast.Constant) and loop.value == "asyncio", (
            "uvicorn.run in app/api/server.py must pass loop='asyncio' "
            f"(line {call.lineno})")


def _api_command_lines() -> List[Tuple[str, int, str]]:
    found = []
    seen = set()
    for pattern in _SCAN_GLOBS:
        for path in REPO_ROOT.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") and not stripped.startswith("#!"):
                    continue  # a comment that names the command does not run it
                if "uvicorn" in line and _API_TARGET in line:
                    found.append((str(path.relative_to(REPO_ROOT)), n, stripped))
    return found


def test_every_api_command_line_pins_the_asyncio_loop():
    lines = _api_command_lines()
    # Not vacuous: the scan must at least see both Dockerfile CMDs.
    hit_files = {f for f, _, _ in lines}
    assert set(_API_DOCKERFILES) <= hit_files, (
        f"the scan no longer finds the API Dockerfile CMDs; it saw {sorted(hit_files)}")
    unpinned = [f"{f}:{n}: {s}" for f, n, s in lines if not _PINNED_RE.search(s)]
    assert not unpinned, (
        "these lines start the API without --loop asyncio (uvicorn then picks "
        "uvloop):\n" + "\n".join(unpinned))


@pytest.mark.parametrize("template", [
    "deploy/helm/avaloka/templates/deployment.yaml",
    "deploy/avaloka/charts/avaloka-backend/templates/deployment.yaml",
])
def test_chart_api_container_keeps_the_image_cmd(template):
    """The charts run the image CMD. A command/args override on the API
    container would bypass the pinned loop unless it pins it too."""
    text = (REPO_ROOT / template).read_text(encoding="utf-8")
    live = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    starts_api = [ln.strip() for ln in live if "uvicorn" in ln or _API_TARGET in ln]
    if starts_api:
        # A YAML list spreads the argv over lines, so look for the pair anywhere.
        body = "\n".join(live)
        assert re.search(r"--loop[ =\"',\s-]*asyncio", body), (
            f"{template} starts the API itself without --loop asyncio: {starts_api}")
