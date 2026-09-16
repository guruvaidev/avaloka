"""
Integration tests for RayJob on GKE cluster.

Supports TWO submission modes (auto-detected):
  DIRECT mode  — RAY_DASHBOARD_URL is set  → uses Ray Job Submission API (no kubectl needed)
  KUBERAY mode — RAY_DASHBOARD_URL not set → uses kubectl apply RayJob YAML

Requirements (cluster tests):
  - kubectl configured + pointing at ray-gke-cluster
  - ray-training namespace exists
  - KubeRay CRD installed (KUBERAY mode only)

Run ALL tests:
    pytest tests/test_ray_integration.py -v -s

Run only no-cluster tests:
    pytest tests/test_ray_integration.py -v -s -m "not cluster"

Run cloud tests (needs TEST_CLOUD_URI + TEST_CONNECTION_ID):
    pytest tests/test_ray_integration.py -v -s -m cloud

Env vars:
    RAY_DASHBOARD_URL     — set this to enable Direct mode (e.g. http://34.132.134.104:8265)
    TEST_RAY_NAMESPACE    — default: ray-training
    TEST_RAY_WORKERS      — default: 3
    TEST_RAY_TIMEOUT_S    — default: 300
    TEST_CLOUD_URI        — e.g. gs://avaloka-test-user-filestore/test-input-data/2019-Nov.csv
    TEST_CONNECTION_ID    — Supabase connection_id for cloud tests
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import textwrap
from typing import Any, Dict, Optional

import pytest


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RAY_NAMESPACE       = os.getenv("TEST_RAY_NAMESPACE", "ray-training")
RAY_WORKERS         = int(os.getenv("TEST_RAY_WORKERS", "3"))
RAY_TIMEOUT_S       = int(os.getenv("TEST_RAY_TIMEOUT_S", "300"))
TEST_CLOUD_URI      = os.getenv("TEST_CLOUD_URI", "")
TEST_CONNECTION_ID  = os.getenv("TEST_CONNECTION_ID", "")
RAY_DASHBOARD_URL   = os.getenv("RAY_DASHBOARD_URL", "").strip()

# True when running against a live Ray cluster via Job Submission API
DIRECT_MODE = bool(RAY_DASHBOARD_URL)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _kubectl(*args, timeout_s: int = 120) -> tuple[int, str, str]:
    import shutil
    kubectl = shutil.which("kubectl") or "kubectl"
    proc = subprocess.run(
        [kubectl, *args],
        capture_output=True, text=True, timeout=timeout_s, check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _unique_job_name(prefix: str = "test-integ") -> str:
    return f"{prefix}-{int(time.time())}"


def _get_worker_pods(rayjob_name: str, namespace: str = RAY_NAMESPACE) -> list[str]:
    """Return list of worker pod names for a RayJob (KubeRay mode only)."""
    rc, out, _ = _kubectl(
        "get", "pods", "-n", namespace,
        "-l", f"app={rayjob_name},ray-node-type=worker",
        "-o", "jsonpath={.items[*].metadata.name}",
        timeout_s=60,
    )
    if rc != 0 or not out.strip():
        return []
    return out.strip().split()


def _delete_rayjob(rayjob_name: str, namespace: str = RAY_NAMESPACE) -> None:
    """Best-effort cleanup. In Direct mode cleans up nothing (jobs auto-expire)."""
    if not DIRECT_MODE:
        _kubectl("delete", "rayjob", rayjob_name, "-n", namespace,
                 "--ignore-not-found=true", timeout_s=60)
        _kubectl("delete", "configmap", f"{rayjob_name}-code", "-n", namespace,
                 "--ignore-not-found=true", timeout_s=60)


def _cluster_available() -> tuple[bool, str]:
    """Check kubectl + namespace availability."""
    rc, out, err = _kubectl("get", "nodes", timeout_s=30)
    if rc != 0:
        return False, f"kubectl get nodes failed: {err.strip()}"

    rc2, _, _ = _kubectl("get", "namespace", RAY_NAMESPACE, timeout_s=30)
    if rc2 != 0:
        return False, f"Namespace {RAY_NAMESPACE} not found."

    if not DIRECT_MODE:
        rc3, _, _ = _kubectl("get", "crd", "rayjobs.ray.io", timeout_s=30)
        if rc3 != 0:
            return False, "KubeRay CRD not installed (required for KubeRay mode)."

    return True, out


def _run_minimal_job(rayjob_name: str) -> "RayJobOutcome":
    """
    Submit MINIMAL_SCRIPT via run_rayjob_from_yaml which auto-selects
    Direct mode (RAY_DASHBOARD_URL set) or KubeRay mode (kubectl apply).
    """
    from app.agents.execution_agent import render_rayjob_yaml, BASE_RAYJOB_YAML
    from app.infra.ray_job_runner import run_rayjob_from_yaml

    yaml_text = render_rayjob_yaml(
        base_yaml=BASE_RAYJOB_YAML,
        name=rayjob_name,
        namespace=RAY_NAMESPACE,
        workers=RAY_WORKERS,
        data_uri="",
        script_text=MINIMAL_SCRIPT,
        cloud_secret_name="cloud-creds",
        output_artifact_uri="",
        output_metrics_uri="",
    )

    return run_rayjob_from_yaml(
        yaml_text=yaml_text,
        namespace=RAY_NAMESPACE,
        rayjob_name=rayjob_name,
        timeout_s=RAY_TIMEOUT_S,
    )


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def cluster_ready():
    """
    Session-scoped, NOT autouse — only tests that need a cluster request this.
    In Direct mode: only checks kubectl nodes (no CRD check).
    In KubeRay mode: checks kubectl + namespace + CRD.
    """
    ok, msg = _cluster_available()
    if not ok:
        pytest.skip(f"Cluster not available: {msg}")
    mode = "DIRECT (RAY_DASHBOARD_URL)" if DIRECT_MODE else "KUBERAY (kubectl apply)"
    print(f"\n✅ Cluster ready. Namespace={RAY_NAMESPACE}  Mode={mode}")
    return True


@pytest.fixture
def require_cluster(cluster_ready):
    return cluster_ready


@pytest.fixture
def ray_job_name(request, require_cluster):
    """Generate unique name + auto-cleanup after test."""
    prefix = getattr(request, "param", "test-integ")
    name = _unique_job_name(prefix)
    yield name
    print(f"\n  [cleanup] {name}")
    _delete_rayjob(name)


# ─────────────────────────────────────────────
# MINIMAL TEST SCRIPT
# Works in both Direct mode and KubeRay mode.
# In Direct mode: ray.init(address="auto") connects to existing cluster.
# ─────────────────────────────────────────────

MINIMAL_SCRIPT = textwrap.dedent("""\
import os
import time
import ray
from ray.util import get_node_ip_address

ray.init(address="auto")

# Wait for at least 1 worker (Direct mode may have variable workers)
deadline = time.time() + 120
while True:
    alive = [n for n in ray.nodes() if n.get("Alive")]
    workers = [n for n in alive if float((n.get("Resources") or {}).get("CPU", 0)) > 0]
    ips = sorted({n.get("NodeManagerAddress") for n in workers if n.get("NodeManagerAddress")})
    print("alive_worker_ips =", ips, "count =", len(workers))
    if len(workers) >= 1:
        break
    if time.time() > deadline:
        raise RuntimeError(f"No workers joined within 120s. ips={ips}")
    time.sleep(3)

num_workers = len(workers)
print(f"cluster_resources = {ray.cluster_resources()}")

@ray.remote(num_cpus=0.1)
def where_am_i(i):
    from ray.util import get_node_ip_address
    return get_node_ip_address()

results = ray.get([where_am_i.remote(i) for i in range(9)])
nodes_used = sorted(set(results))
print("nodes_used =", nodes_used)
print("num_nodes =", len(nodes_used))
print(f"METRIC worker_count={num_workers}")
print(f"METRIC runtime_s=1.0")
print(f"METRIC bytes_processed=0")
print("INTEGRATION_TEST_PASSED=true")
""")


# ─────────────────────────────────────────────
# TEST 5 — DRYRUN (NO cluster needed)
# ─────────────────────────────────────────────

class TestExecutionAgentNodeRayDryrun:
    """No cluster required — validates YAML rendering only."""

    def test_dryrun_renders_valid_yaml(self):
        from app.agents.execution_agent import execution_agent_node_ray

        state = {
            "ray_namespace": RAY_NAMESPACE,
            "ray_workers": RAY_WORKERS,
            "ray_timeout_s": 60,
            "data_source_location_cloud": "s3://test-bucket/test.csv",
            "connection_id": "test-conn-id",
            "dataset_id": "ds-integ-001",
            "coder_definition": {"code": "import os\nprint('hello')"},
            "messages": [],
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("RAYJOB_DRYRUN", "1")
            result = execution_agent_node_ray(state)

        er = result["execution_result"]
        assert er["status"] == "dryrun"
        assert "rayjob_yaml" in er

        yaml_text = er["rayjob_yaml"]
        assert "kind: RayJob" in yaml_text
        assert "kind: ConfigMap" in yaml_text
        assert RAY_NAMESPACE in yaml_text
        assert str(RAY_WORKERS) in yaml_text
        assert "s3://test-bucket/test.csv" in yaml_text
        assert "__DATA_SOURCE_URI__" not in yaml_text
        assert "__CLOUD_SECRET_NAME__" not in yaml_text

        leftovers = re.findall(r"__([A-Z0-9_]+)__", yaml_text)
        assert leftovers == [], f"Leftover placeholders: {leftovers}"
        print(f"\n  ✅ DRYRUN YAML valid, {len(yaml_text)} chars")

    def test_dryrun_uses_ray_training_namespace(self):
        from app.agents.execution_agent import execution_agent_node_ray

        state = {
            "ray_namespace": "ray-training",
            "ray_workers": 2,
            "ray_timeout_s": 60,
            "data_source_location_cloud": "gs://bucket/file.csv",
            "connection_id": "test-conn",
            "dataset_id": "ds-ns-test",
            "coder_definition": {"code": "print('ns test')"},
            "messages": [],
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("RAYJOB_DRYRUN", "1")
            result = execution_agent_node_ray(state)

        yaml_text = result["execution_result"]["rayjob_yaml"]
        assert "ray-training" in yaml_text
        assert "ray-jobs" not in yaml_text
        print("  ✅ namespace=ray-training confirmed in rendered YAML")

    def test_dryrun_interactive_profile_reduces_resources(self):
        from app.agents.execution_agent import execution_agent_node_ray

        state = {
            "ray_namespace": "ray-training",
            "ray_workers": 3,
            "ray_timeout_s": 60,
            "data_source_location_cloud": "gs://bucket/file.csv",
            "connection_id": "test-conn",
            "dataset_id": "ds-profile-test",
            "ray_execution_profile": "interactive_sample_analysis",
            "coder_definition": {"code": "print('profile test')"},
            "messages": [],
        }

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("RAYJOB_DRYRUN", "1")
            result = execution_agent_node_ray(state)

        er = result["execution_result"]
        assert er["status"] == "dryrun"
        assert er["worker_count"] == 1, \
            f"Interactive profile must pin workers to 1, got {er['worker_count']}"
        print("  ✅ interactive_sample_analysis profile pins workers=1")


# ─────────────────────────────────────────────
# TEST 6 — Cloud URI tests
# ─────────────────────────────────────────────

@pytest.mark.cloud
@pytest.mark.cluster
class TestCloudUriRead:

    @pytest.fixture(autouse=True)
    def require_cloud_env(self, require_cluster):
        if not TEST_CLOUD_URI:
            pytest.skip("Set TEST_CLOUD_URI env var to run cloud tests.")
        if not TEST_CONNECTION_ID:
            pytest.skip("Set TEST_CONNECTION_ID env var to run cloud tests.")

    def test_rayjob_reads_from_cloud_uri(self, ray_job_name):
        from app.agents.execution_agent import execution_agent_node_ray

        state = {
            "ray_namespace": RAY_NAMESPACE,
            "ray_workers": RAY_WORKERS,
            "ray_timeout_s": RAY_TIMEOUT_S,
            "data_source_location_cloud": TEST_CLOUD_URI,
            "connection_id": TEST_CONNECTION_ID,
            "dataset_id": "ds-cloud-integ",
            "ray_job_name": ray_job_name,
            "coder_definition": {
                "code": textwrap.dedent("""\
                    import os
                    print("cloud read test")
                    print("ROWS:", len(df))
                    print("COLS:", list(df.columns))
                """)
            },
            "messages": [],
        }

        result = execution_agent_node_ray(state)
        er = result["execution_result"]

        print(f"\n  {json.dumps({k: v for k, v in er.items() if k != 'logs'}, indent=2)}")
        print(f"  Log tail:\n{(er.get('logs') or '')[-400:]}")

        assert er["status"] in ("success", "succeeded"), \
            f"Expected success/succeeded, got {er['status']}. Logs:\n{er.get('logs', '')[-1000:]}"

        metrics = er.get("metrics") or {}
        assert metrics.get("bytes_processed", 0) > 0, \
            f"bytes_processed must be > 0 for real cloud data. metrics={metrics}"
        assert er.get("runtime_s", 0) > 0
        print(f"  ✅ bytes_processed={metrics.get('bytes_processed')}")

    def test_artifact_written_to_cloud(self, ray_job_name):
        from app.agents.execution_agent import execution_agent_node_ray

        state = {
            "ray_namespace": RAY_NAMESPACE,
            "ray_workers": RAY_WORKERS,
            "ray_timeout_s": RAY_TIMEOUT_S,
            "data_source_location_cloud": TEST_CLOUD_URI,
            "connection_id": TEST_CONNECTION_ID,
            "dataset_id": "ds-artifact-integ",
            "ray_job_name": ray_job_name,
            "coder_definition": {"code": "import os\nprint('artifact test')"},
            "messages": [],
        }

        result = execution_agent_node_ray(state)
        er = result["execution_result"]

        assert er["status"] in ("success", "succeeded")
        assert er.get("artifact_uri"), f"artifact_uri must be set. Got: {er}"
        assert "://" in er["artifact_uri"]
        print(f"  ✅ artifact_uri = {er['artifact_uri']}")


# ─────────────────────────────────────────────
# TEST 7 — Partitioned read
# ─────────────────────────────────────────────

@pytest.mark.cloud
@pytest.mark.cluster
class TestPartitionedRead:

    @pytest.fixture(autouse=True)
    def require_cloud_env(self, require_cluster):
        if not TEST_CLOUD_URI or not TEST_CONNECTION_ID:
            pytest.skip("Set TEST_CLOUD_URI and TEST_CONNECTION_ID to run partitioned read tests.")

    def test_partitions_spread_across_nodes(self, ray_job_name):
        from app.agents.execution_agent import execution_agent_node_ray

        state = {
            "ray_namespace": RAY_NAMESPACE,
            "ray_workers": 3,
            "ray_timeout_s": RAY_TIMEOUT_S,
            "data_source_location_cloud": TEST_CLOUD_URI,
            "connection_id": TEST_CONNECTION_ID,
            "dataset_id": "ds-partition-test",
            "ray_job_name": ray_job_name,
            "coder_definition": {"code": "print('partition test')"},
            "messages": [],
        }

        result = execution_agent_node_ray(state)
        er = result["execution_result"]
        assert er["status"] in ("success", "succeeded")

        logs = er.get("logs", "")
        assert "NUM NODES" in logs
        assert "Partition 1" in logs
        assert "Partition 2" in logs
        assert "Partition 3" in logs
        print(f"\n✅ Partitioned read verified.")


# ─────────────────────────────────────────────
# TEST 8 — Node resilience (NO cluster needed)
# ─────────────────────────────────────────────

class TestNodeResilience:
    """Tests that validate wrapper script content — no cluster required."""

    def test_max_retries_set_on_tasks(self):
        from app.agents.execution_agent import _build_ray_wrapper
        wrapper = _build_ray_wrapper("gs://test/file.csv", "", "")
        assert "max_retries=3" in wrapper, "max_retries must be set for node failure resilience"
        print("✅ max_retries=3 confirmed in wrapper")

    def test_wrapper_contains_worker_wait_loop(self):
        from app.agents.execution_agent import _build_ray_wrapper
        wrapper = _build_ray_wrapper("gs://test/file.csv", "", "")
        assert "deadline" in wrapper
        assert "ray.nodes()" in wrapper
        assert "workers" in wrapper
        print("✅ Worker wait loop present in wrapper")

    def test_wrapper_writes_artifact_json(self):
        from app.agents.execution_agent import _build_ray_footer
        footer = _build_ray_footer("gs://test/file.csv")
        assert "artifact" in footer
        assert "output_rows" in footer
        assert "_write_json" in footer
        print("✅ Artifact write confirmed in footer")

    def test_interactive_sample_wrapper_reads_dataframe(self):
        from app.agents.execution_agent import _build_interactive_sample_wrapper
        wrapper = _build_interactive_sample_wrapper(
            "gs://test/file.csv", "gs://test/artifact.json", "gs://test/metrics.json"
        )
        assert "_read_dataframe" in wrapper
        assert "DATA_SOURCE_URI" in wrapper
        assert "INTERACTIVE_SAMPLE_ANALYSIS_START" in wrapper
        print("✅ Interactive sample wrapper reads dataframe correctly")

    def test_interactive_sample_footer_writes_artifact(self):
        from app.agents.execution_agent import _build_interactive_sample_footer
        footer = _build_interactive_sample_footer()
        assert "_write_json" in footer
        assert "INTERACTIVE_SAMPLE_ANALYSIS_DONE" in footer
        assert "output_rows" in footer
        print("✅ Interactive sample footer writes artifact and metrics")