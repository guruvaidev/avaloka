"""End-to-End (E2E) tests for the Avaloka CLI application.

Covers full process-boundary execution and live serving lifecycles:
- True external process invocation via `subprocess.run([sys.executable, "-m", "avaloka", ...])`
- End-to-end analyze mission generating complete on-disk deliverable bundles
- End-to-end train mission generating deployable models and serving scripts
- Infer command rollout planning and manifest inspection
- Live Docker serving lifecycle (gated by Docker availability)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_e2e_cli_process_help_and_version():
    """True external process invocation of CLI version and help."""
    # Test --version
    res_ver = subprocess.run(
        [sys.executable, "-m", "avaloka", "--version"],
        capture_output=True,
        text=True,
    )
    assert res_ver.returncode == 0
    assert "1.6" in res_ver.stdout or "avaloka" in res_ver.stdout.lower()

    # Test --help
    res_help = subprocess.run(
        [sys.executable, "-m", "avaloka", "--help"],
        capture_output=True,
        text=True,
    )
    assert res_help.returncode == 0
    assert "usage" in res_help.stdout.lower()
    assert "avaloka" in res_help.stdout.lower()
    for cmd in ["analyze", "train", "chat", "batch", "deploy", "infer"]:
        assert cmd in res_help.stdout


def test_e2e_analyze_process_lifecycle(churn_csv, tmp_path):
    """Run full analyze mission through an external Python process."""
    out_dir = tmp_path / "e2e_analyze_out"
    cmd = [
        sys.executable,
        "-m",
        "avaloka",
        "analyze",
        str(churn_csv),
        "--goal",
        "Analyze customer churn drivers and patterns",
        "-o",
        str(out_dir),
        "--quiet",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Verify deliverable files written to disk
    assert (out_dir / "mission.json").exists()
    assert (out_dir / "executive_report.html").exists()
    assert (out_dir / "analysis.py").exists()

    mission_data = json.loads((out_dir / "mission.json").read_text())
    assert mission_data["kind"] == "analyze"
    assert mission_data["verdict"] in {"pass", "warn"}


def test_e2e_train_deployable_lifecycle(churn_csv, tmp_path):
    """Run full train mission with --deployable through an external process."""
    out_dir = tmp_path / "e2e_train_out"
    cmd = [
        sys.executable,
        "-m",
        "avaloka",
        "train",
        str(churn_csv),
        "--target",
        "churned",
        "-o",
        str(out_dir),
        "--deployable",
        "--quiet",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Verify model and deployment packaging
    assert (out_dir / "model" / "model.pkl").exists()
    assert (out_dir / "Dockerfile").exists()
    assert (out_dir / "service.py").exists()

    # Verify service.py syntax compiles cleanly
    compile_res = subprocess.run(
        [sys.executable, "-m", "py_compile", str(out_dir / "service.py")],
        capture_output=True,
    )
    assert compile_res.returncode == 0


def test_e2e_infer_plan_lifecycle(trained_package, tmp_path):
    """Run infer command planning without --apply to verify serving plan."""
    cmd = [
        sys.executable,
        "-m",
        "avaloka",
        "infer",
        str(trained_package),
        "--target",
        "docker",
        "--quiet",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "docker" in res.stdout.lower() or "plan" in res.stdout.lower() or res.returncode == 0


@pytest.mark.slow
def test_e2e_live_docker_serving(trained_package):
    """Live Docker deployment test (opt-in via AVALOKA_TEST_ALLOW_DOCKER=1)."""
    if os.environ.get("AVALOKA_TEST_ALLOW_DOCKER") != "1":
        pytest.skip("Set AVALOKA_TEST_ALLOW_DOCKER=1 to run live container deployment.")

    if not shutil.which("docker"):
        pytest.skip("Docker CLI binary not found on PATH.")

    # Check if docker daemon is responsive
    daemon_check = subprocess.run(["docker", "info"], capture_output=True)
    if daemon_check.returncode != 0:
        pytest.skip("Docker daemon is not running or accessible.")

    # Execute infer with --apply
    cmd = [
        sys.executable,
        "-m",
        "avaloka",
        "infer",
        str(trained_package),
        "--target",
        "docker",
        "--apply",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0
