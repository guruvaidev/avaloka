"""Which paths tests/conftest.py marks `integration`, and which it must not.

The marker decides what the hermetic gate never runs, so a rule that matches
too much removes tests from the gate without anyone deciding to. Two things are
pinned here: the rule is anchored at tests/, and no file can claim to be an
integration test by name while sitting in the gate's blind spot unnoticed.
"""

import pathlib
import subprocess
import sys

import pytest

from tests.conftest import _wants_integration_marker

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("path", [
    "tests/test_scheduler_integration.py",
    "tests/e2e/test_e2_auth.py",
    "tests/k8s/helpers/test_anything.py",
    "tests/infra/test_ray_manager.py",
    "tests/containers/test_lineage_postgres.py",
    "tests\\e2e\\test_e2_auth.py",
])
def test_integration_suites_are_marked(path):
    assert _wants_integration_marker(path) is True


@pytest.mark.parametrize("path", [
    "tests/test_server_unit.py",
    "tests/test_lineage.py",
    # Outside tests/: a suffix or a directory name is not enough.
    "app/agents/mta/mlflow_integration.py",
    "app/agents/mta/test_mlflow_integration.py",
    "vendor/somepkg/tests/e2e/test_theirs.py",
    "docs/examples/tests/k8s/test_example.py",
    "integration_tests/e2e/test_thing_integration.py",
    # Under tests/, but not one of the suites: the directory rule is a prefix.
    "tests/avaloka/tests/e2e/test_nested.py",
    "tests/not_e2e/test_thing.py",
])
def test_everything_else_is_left_in_the_gate(path):
    assert _wants_integration_marker(path) is False


def test_no_file_outside_tests_is_caught_by_the_old_unanchored_rule():
    """The rule used to be `"tests/e2e/" in path or path.endswith("_integration.py")`.
    If a collectable test file outside tests/ ever matches that, anchoring the
    rule changed what the gate runs, and that should be a decision."""
    old_dirs = ("tests/e2e/", "tests/k8s/", "tests/infra/")
    caught = []
    for path in REPO.rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(("tests/", ".git/", "ui/", "node_modules/", ".venv/", "venv/")):
            continue
        if not (path.name.startswith("test_") or path.name.endswith("_test.py")):
            continue
        if any(seg in rel for seg in old_dirs) or rel.endswith("_integration.py"):
            caught.append(rel)
    assert caught == []


# ---------------------------------------------------------------------------
# `containers` never travels without `integration`
# ---------------------------------------------------------------------------
# tests/contract/test_c3_release_coherence.py lists `containers` as a marker the
# hermetic filter need not name, on the strength of one claim: every test that
# carries it also carries `integration`, so `not integration` already removes
# it. Nothing else enforces that. A `containers` test without `integration`
# would be selected by the gate and fail there on a missing database, for a
# reason nobody would guess from the failure. These two tests are the
# enforcement, and they inspect collected markers rather than reading files.


def test_no_containers_test_is_in_a_run_without_also_being_integration(request):
    """Whatever this session selected: in the hermetic gate that is everything
    `not integration`, so a `containers` test found here has leaked into it."""
    leaked = [item.nodeid for item in request.session.items
              if item.get_closest_marker("containers")
              and not item.get_closest_marker("integration")]
    assert leaked == [], (
        f"{len(leaked)} test(s) carry `containers` without `integration`, so the hermetic "
        f"gate selects them: {leaked[:5]}. Put them under tests/containers/ (marked by "
        "directory in tests/conftest.py) or mark them `integration` as well.")


def _collect(marker_expression: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/containers", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-m", marker_expression],
        cwd=REPO, capture_output=True, text=True, timeout=300)


def test_the_container_suite_is_all_integration_when_collected():
    """Collect tests/containers/ for real and ask pytest which markers stuck."""
    marked = _collect("containers")
    collected = [line for line in marked.stdout.splitlines() if "::" in line]
    # Without this the check below passes on an empty or uncollectable suite.
    assert marked.returncode == 0 and collected, (
        f"`-m containers` selected nothing from tests/containers/ (exit {marked.returncode}):\n"
        f"{marked.stdout[-800:]}{marked.stderr[-800:]}")

    stray = _collect("containers and not integration")
    stray_ids = [line for line in stray.stdout.splitlines() if "::" in line]
    # pytest exits 5 when the expression selects no tests.
    assert stray.returncode == 5 and stray_ids == [], (
        f"tests carrying `containers` without `integration` (exit {stray.returncode}): "
        f"{stray_ids[:5]}\n{stray.stdout[-400:]}")
