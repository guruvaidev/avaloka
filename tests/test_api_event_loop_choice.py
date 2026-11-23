"""The API runs on the stdlib event loop, not uvicorn[standard]'s uvloop.

uvloop 0.23 segfaulted the API (exit 139, main thread in the loop) five times
in 2.5 h of e2e, because worker threads reached objects of the main loop. The
Redis instance of that is fixed in app/core/cache.py; the stdlib loop is what
turns the next instance into an exception instead of a dead process. Every way
the image starts the API must therefore say ``--loop asyncio``: uvicorn picks
uvloop whenever it is importable, and uvicorn[standard] installs it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

DOCKERFILES = [
    "deploy/docker/Dockerfile.api",
    "deploy/avaloka/charts/avaloka-backend/docker/Dockerfile.api",
]


def _cmd(dockerfile: Path) -> list:
    cmds = [line for line in dockerfile.read_text().splitlines() if line.startswith("CMD ")]
    assert len(cmds) == 1, f"{dockerfile}: expected one CMD, found {len(cmds)}"
    return json.loads(cmds[0][len("CMD "):])


@pytest.mark.parametrize("rel", DOCKERFILES)
def test_api_image_cmd_pins_the_stdlib_loop(rel):
    argv = _cmd(ROOT / rel)
    assert argv[:2] == ["uvicorn", "app.api.server:app"]
    assert "--loop" in argv, f"{rel}: no --loop, so uvicorn picks uvloop"
    assert argv[argv.index("--loop") + 1] == "asyncio"


def test_release_stack_and_local_runner_use_the_same_loop():
    stack = (ROOT / "scripts/release_test/api_stack.sh").read_text()
    for launch in re.findall(r"uvicorn app\.api\.server:app[^\n|]*", stack):
        assert "--loop asyncio" in launch, launch
    server = (ROOT / "app/api/server.py").read_text()
    runner = server[server.index('if __name__ == "__main__":'):]
    assert 'loop="asyncio"' in runner
