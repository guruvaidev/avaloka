# tests/infra/test_local_kind_deployment.py
"""End-to-end-ish local deployment test against a kind cluster.

Skipped automatically unless `kind`, `kubectl` and `helm` are on PATH and the
RUN_KIND_E2E=1 env var is set (the full flow builds images and is slow). This
keeps CI green by default while giving a real local smoke test on demand.
"""
import os
import shutil

import pytest

REQUIRED = ["kind", "kubectl", "helm", "docker"]
_missing = [t for t in REQUIRED if shutil.which(t) is None]

pytestmark = pytest.mark.skipif(
    _missing or os.environ.get("RUN_KIND_E2E") != "1",
    reason=f"kind e2e disabled (set RUN_KIND_E2E=1; missing tools: {_missing})",
)


def test_local_kind_bootstrap_provision_and_teardown():
    """Provision a local kind cluster + KubeRay + avaloka, then tear it down."""
    from app.infra import cluster_bootstrap

    rc = cluster_bootstrap.main([
        "--mode", "provision", "--provider", "local", "--skip-serve",
    ])
    try:
        assert rc == 0, "bootstrap provision should succeed"

        import subprocess
        pods = subprocess.run(["kubectl", "get", "pods"], capture_output=True, text=True)
        assert "avaloka" in pods.stdout, pods.stdout
    finally:
        from app.infra import cloud_provisioner
        cloud_provisioner.teardown("local")
