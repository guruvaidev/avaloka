"""Shared fixtures/constants for the k8s integration test tiers (T1–T5)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = REPO_ROOT / "deploy" / "helm" / "avaloka"
OVERLAY_DIR = CHART_DIR / "values"
KIND_CLUSTER_YAML = REPO_ROOT / "deploy" / "clusters" / "kind-cluster.yaml"

# The single source of truth the whole stack must agree on (D2 / R4).
RAY_VERSION = "2.58.0"


def has(tool: str) -> bool:
    return shutil.which(tool) is not None


def helm(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a helm command from the repo root."""
    return subprocess.run(
        ["helm", *args], cwd=str(REPO_ROOT),
        capture_output=True, text=True, check=check,
    )


requires_helm = pytest.mark.skipif(not has("helm"), reason="helm not installed")
requires_kubectl = pytest.mark.skipif(not has("kubectl"), reason="kubectl not installed")


@pytest.fixture(scope="session")
def rendered_chart() -> str:
    """`helm template` output for the default (kind) values — rendered once."""
    if not has("helm"):
        pytest.skip("helm not installed")
    return helm("template", "avaloka", str(CHART_DIR)).stdout
