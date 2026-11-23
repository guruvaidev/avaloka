"""Which paths tests/conftest.py marks `integration`, and which it must not.

The marker decides what the hermetic gate never runs, so a rule that matches
too much removes tests from the gate without anyone deciding to. Two things are
pinned here: the rule is anchored at tests/, and no file can claim to be an
integration test by name while sitting in the gate's blind spot unnoticed.
"""

import pathlib
import subprocess
import sys
from typing import List

import pytest

from tests.conftest import _wants_integration_marker

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("path", [
    "tests/test_scheduler_integration.py",
    "tests/k8s/helpers/test_anything.py",
    "tests/infra/test_ray_manager.py",
    "tests/containers/test_lineage_postgres.py",
    "tests\\k8s\\test_t2_cluster_deployment.py",
])
def test_integration_suites_are_marked(path):
    assert _wants_integration_marker(path) is True


@pytest.mark.parametrize("path", [
    "tests/test_server_unit.py",
    "tests/test_lineage.py",
    # The e2e pack is hermetic by default and gated; see tests/conftest.py.
    "tests/e2e/test_e2_auth.py",
    "tests\\e2e\\test_e2_auth.py",
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


# ---------------------------------------------------------------------------
# The e2e pack is in the hermetic gate -- and only its hermetic part
# ---------------------------------------------------------------------------
# From 2026-09-24 to this change, tests/e2e/ was marked `integration` by
# directory, so the gate selected none of it and 500+ hermetic tests rotted.
# These pin both halves: the gate's expression selects the pack, and the files
# that need cloud data or a model provider stay out of it.

HERMETIC_EXPRESSION = "not cluster and not cloud and not integration"
# Files whose every test needs something outside the process.
E2E_OUT_OF_GATE = ("tests/e2e/test_e3_real_dataset_gcs.py", "tests/e2e/test_e5_golden_missions.py")


def _collect_e2e(marker_expression: str) -> List[str]:
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/e2e", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-m", marker_expression],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, f"collection failed ({done.returncode}):\n{done.stdout[-800:]}{done.stderr[-800:]}"
    return [line for line in done.stdout.splitlines() if "::" in line]


def test_the_hermetic_gate_selects_the_e2e_pack():
    selected = _collect_e2e(HERMETIC_EXPRESSION)
    # 520 at the time of writing; the floor catches the directory being
    # deselected wholesale again, not ordinary growth or pruning.
    assert len(selected) >= 400, f"the hermetic gate selects only {len(selected)} e2e tests"
    files = {nodeid.split("::")[0] for nodeid in selected}
    for name in ("test_e2_auth.py", "test_e2_contract.py", "test_e2_hygiene.py",
                 "test_e3_ingestion.py", "test_e7_inference.py"):
        assert f"tests/e2e/{name}" in files, f"tests/e2e/{name} is not in the hermetic gate"


def test_the_out_of_process_e2e_files_stay_out_of_the_gate():
    selected = _collect_e2e(HERMETIC_EXPRESSION)
    leaked = [nodeid for nodeid in selected if nodeid.startswith(E2E_OUT_OF_GATE)]
    assert leaked == [], f"tests that need cloud data or a provider are in the gate: {leaked[:5]}"
    # And they exist: an empty file would make the check above pass vacuously.
    assert _collect_e2e("cloud or integration"), "no cloud/integration e2e tests collected at all"
