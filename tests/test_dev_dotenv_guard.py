"""A developer's .env, and their exported SUPABASE_* variables, never reach a test.

App modules call load_dotenv() when imported -- most with an explicit
<checkout>/.env path, some bare. On a developer machine that file holds real
credentials, so an unstubbed path (_schedule_small_file_persistence,
resolve_plan, the MCP connection lookups) would talk to a real Supabase
project. tests/conftest.py stops python-dotenv loading anything from inside
the checkout and starts the session with no SUPABASE_* variables.

The real checkout's .env is never touched. Each case builds a small checkout
under tmp_path -- this repository's tests/conftest.py and pytest.ini, an app
package with a module that loads .env exactly as app modules do, and a .env
holding a fake SUPABASE_URL -- and runs pytest there in a child process. The
probe test inside reports what it saw.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FAKE_URL = "https://fake.example"

# Loads .env the two ways app modules do: app/agents/coder.py and friends with
# an explicit project-root path, app/api/config.py and friends bare. Then a
# file outside the checkout, which must still load: the guard is not a blanket
# off switch for python-dotenv.
_PROBE_MODULE = textwrap.dedent(
    """
    import os
    from dotenv import load_dotenv as _load, dotenv_values

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _load(os.path.join(project_root, ".env"))
    _load()
    _load(os.environ["PROBE_OUTSIDE_ENV"])
    ON_DISK = dotenv_values(os.path.join(project_root, ".env"))
    """
)

_PROBE_TESTS = textwrap.dedent(
    """
    import json, os

    NAMES = ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY", "PROBE_OUTSIDE_LOADED")

    def _report(key, data):
        path = os.environ["PROBE_OUT"]
        seen = json.load(open(path)) if os.path.exists(path) else {}
        seen[key] = data
        json.dump(seen, open(path, "w"))

    def test_session_view():
        import app.probe_env as probe
        _report("session", {
            "env": {name: os.environ.get(name) for name in NAMES},
            "on_disk_url": probe.ON_DISK.get("SUPABASE_URL"),
        })

    def test_opted_in(real_supabase_env):
        _report("opted_in", {name: os.environ.get(name) for name in NAMES})
    """
)


def _checkout(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "checkout"
    (root / "tests").mkdir(parents=True)
    (root / "app" / "core").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "tests" / "conftest.py", root / "tests" / "conftest.py")
    shutil.copy(REPO_ROOT / "pytest.ini", root / "pytest.ini")
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "test_probe.py").write_text(_PROBE_TESTS)
    # Whatever app package files the conftest imports, as this tree has them.
    for rel in ("app/__init__.py", "app/core/thirdparty_optout.py"):
        if (REPO_ROOT / rel).is_file():
            shutil.copy(REPO_ROOT / rel, root / rel)
    (root / "app" / "probe_env.py").write_text(_PROBE_MODULE)
    (root / ".env").write_text(f"SUPABASE_URL={FAKE_URL}\nSUPABASE_SERVICE_ROLE_KEY=fake-service-role\n")
    return root


def _run(tmp_path: pathlib.Path, **extra_env: str) -> dict:
    root = _checkout(tmp_path)
    outside = tmp_path / "outside.env"
    outside.write_text("PROBE_OUTSIDE_LOADED=yes\n")
    out = tmp_path / "probe.json"
    env = {k: v for k, v in os.environ.items() if "SUPABASE" not in k.upper()}
    env.update(
        PYTHONPATH=str(root),
        PROBE_OUT=str(out),
        PROBE_OUTSIDE_ENV=str(outside),
        SUPABASE_ANON_KEY="exported-in-the-shell",  # a developer's shell export
        **extra_env,
    )
    env.pop("PROBE_OUTSIDE_LOADED", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "tests/test_probe.py"],
        cwd=str(root), env=env, capture_output=True, text=True, timeout=600,
    )
    seen = json.loads(out.read_text()) if out.exists() else {}
    seen["_output"] = (proc.stdout + proc.stderr)[-3000:]
    return seen


def test_the_session_never_sees_the_checkouts_dotenv(tmp_path):
    seen = _run(tmp_path)
    assert "session" in seen, seen["_output"]
    session = seen["session"]
    # Controls: the .env really is there, and dotenv still loads a file from
    # outside the checkout -- so a None below means "refused", not "absent".
    assert session["on_disk_url"] == FAKE_URL
    assert session["env"]["PROBE_OUTSIDE_LOADED"] == "yes"
    assert session["env"]["SUPABASE_URL"] is None, (
        "the test session loaded SUPABASE_URL from the checkout's .env")
    assert session["env"]["SUPABASE_SERVICE_ROLE_KEY"] is None


def test_the_session_drops_exported_supabase_variables(tmp_path):
    seen = _run(tmp_path)
    assert "session" in seen, seen["_output"]
    assert seen["session"]["env"]["SUPABASE_ANON_KEY"] is None, (
        "a SUPABASE_* variable exported in the shell reached the test session")


def test_a_test_opts_back_in_with_the_fixture(tmp_path):
    """real_supabase_env restores the shell's values -- and never the .env's."""
    seen = _run(tmp_path)
    assert "opted_in" in seen, seen["_output"]
    assert seen["opted_in"]["SUPABASE_ANON_KEY"] == "exported-in-the-shell"
    assert seen["opted_in"]["SUPABASE_URL"] is None


def test_a_whole_session_opts_back_in_explicitly(tmp_path):
    seen = _run(tmp_path, AVALOKA_TEST_REAL_SUPABASE="1")
    assert "session" in seen, seen["_output"]
    assert seen["session"]["env"]["SUPABASE_ANON_KEY"] == "exported-in-the-shell"
    # Opting in keeps the shell's variables; it still never reads the .env.
    assert seen["session"]["env"]["SUPABASE_URL"] is None
