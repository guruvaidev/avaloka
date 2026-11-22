"""No conftest may quietly change the process environment when it is imported.

pytest imports every conftest during collection, before any test runs, so an
os.environ write at import time is a write for the WHOLE session: every test
collected anywhere, in any directory, runs with it. tests/e2e/conftest.py once
set MEMORY_WARMUP=0 that way "for the e2e pack", which in a gate run meant for
all 7900 tests.

The root tests/conftest.py is the one sanctioned place: it sets the session's
test environment before any app module is imported, and the keys it may touch
are pinned below, so a new one is a decision rather than a side effect. Every
other conftest must leave os.environ exactly as it found it; per-test settings
belong in a fixture with monkeypatch.

Each conftest is imported in its own subprocess, so nothing here can leak into
this session either.
"""

import json
import os
import pathlib
import subprocess
import sys
from typing import List

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT_CONFTEST = REPO / "tests" / "conftest.py"

# The keys tests/conftest.py sets or removes at import, on purpose.
ROOT_CONFTEST_KEYS = {
    "CHROMA_HOST",
    "CHROMA_PORT",
    "AVALOKA_LINEAGE_DB_URL",
    "AVALOKA_LINEAGE",
    "DB_ENCRYPTION_KEY",
    "AVALOKA_ENV_FILE",
}

# Set by a third-party library when it is imported, not by any conftest: sklearn
# sets both OpenMP flags on import (see the test below that proves it). A
# conftest that imports sklearn changes them as a side effect it cannot avoid.
LIBRARY_SET_KEYS = {"KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK"}

_PROBE = r"""
import importlib.util, json, os, sys

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

root, target = sys.argv[1], sys.argv[2]
if target != root:
    load(root, "tests.conftest")  # what pytest has always imported first
before = dict(os.environ)
load(target, "conftest_under_probe")
after = dict(os.environ)
# Names only: values are never printed.
print(json.dumps(sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))))
"""


def _keys_changed_by(conftest: pathlib.Path) -> List[str]:
    env = {k: v for k, v in os.environ.items() if k not in ROOT_CONFTEST_KEYS}
    env["PYTHONPATH"] = str(REPO)
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, str(ROOT_CONFTEST), str(conftest)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"importing {conftest} failed:\n{done.stderr[-1500:]}"
    return [k for k in json.loads(done.stdout.strip().splitlines()[-1]) if k not in LIBRARY_SET_KEYS]


def test_the_library_set_keys_really_come_from_a_library_import():
    """Keeps the exemption honest: a bare `import sklearn` sets exactly these keys."""
    probe = ("import os, json; b = dict(os.environ); import sklearn; "
             "print(json.dumps(sorted(k for k in os.environ if os.environ.get(k) != b.get(k))))")
    env = {k: v for k, v in os.environ.items() if k not in LIBRARY_SET_KEYS}
    done = subprocess.run([sys.executable, "-c", probe], env=env,
                          capture_output=True, text=True, timeout=300)
    if done.returncode != 0 and "No module named 'sklearn'" in done.stderr:
        pytest.skip("sklearn is not installed here, so nothing sets these keys")
    assert done.returncode == 0, done.stderr[-800:]
    assert set(json.loads(done.stdout.strip().splitlines()[-1])) == LIBRARY_SET_KEYS


def _other_conftests() -> List[pathlib.Path]:
    found = sorted(p for p in (REPO / "tests").rglob("conftest.py")
                   if "node_modules" not in p.parts and p != ROOT_CONFTEST)
    return found


def test_the_root_conftest_touches_only_its_pinned_keys():
    changed = _keys_changed_by(ROOT_CONFTEST)
    # Two of its writes are unconditional; seeing them proves the probe works.
    assert {"AVALOKA_LINEAGE_DB_URL", "AVALOKA_ENV_FILE"} <= set(changed), changed
    assert set(changed) <= ROOT_CONFTEST_KEYS, (
        f"tests/conftest.py now sets {sorted(set(changed) - ROOT_CONFTEST_KEYS)} at import, "
        "for the whole session. Pin it in ROOT_CONFTEST_KEYS if that is intended.")


def test_there_are_other_conftests_to_check():
    # Without this, a discovery bug would make the check below pass on nothing.
    assert len(_other_conftests()) >= 5, _other_conftests()


@pytest.mark.parametrize(
    "conftest", _other_conftests(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_other_conftest_changes_the_environment_at_import(conftest: pathlib.Path):
    changed = _keys_changed_by(conftest)
    assert changed == [], (
        f"{conftest.relative_to(REPO)} changes {changed} in os.environ when imported, which "
        "applies to every test in the session. Use a fixture with monkeypatch.setenv.")
