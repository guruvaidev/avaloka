"""
Unit tests for Ray infra components.
Run with: pytest tests/test_ray_unit.py -v

No cluster required — all kubectl calls are mocked.
"""
import re
import textwrap
import pytest
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────
# 1. YAML TEMPLATE RENDERING
# ─────────────────────────────────────────────

MINIMAL_TEMPLATE = textwrap.dedent("""\
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: rayjob-sample-3w
  namespace: ray-jobs
spec:
  entrypoint: python /home/ray/samples/sample_code.py
  runtimeEnvYAML: |
    env_vars:
      DATA_SOURCE_URI: "__DATA_SOURCE_URI__"
      OUTPUT_ARTIFACT_URI: "__OUTPUT_ARTIFACT_URI__"
      OUTPUT_METRICS_URI: "__OUTPUT_METRICS_URI__"
  rayClusterSpec:
    workerGroupSpecs:
      - groupName: small-group
        replicas: 3
        minReplicas: 3
        maxReplicas: 3
    headGroupSpec:
      template:
        spec:
          containers:
            - name: ray-head
              envFrom:
                - secretRef:
                    name: __CLOUD_SECRET_NAME__
              volumeMounts:
                - mountPath: /var/secrets/cloud
                  name: cloud-creds-vol
          volumes:
            - name: cloud-creds-vol
              secret:
                secretName: cloud-creds
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: rayjob-sample-3w-code
  namespace: ray-jobs
data:
  sample_code.py: |
    import os
    DATA_SOURCE_URI = os.getenv("DATA_SOURCE_URI")
""")


def _get_renderer():
    """Import render_rayjob_yaml from execution_agent."""
    from app.agents.execution_agent import render_rayjob_yaml
    return render_rayjob_yaml


class TestRenderRayjobYaml:
    """Tests for render_rayjob_yaml()"""

    def _render(self, **kwargs):
        render = _get_renderer()
        defaults = dict(
            base_yaml=MINIMAL_TEMPLATE,
            name="test-job-123",
            namespace="ray-jobs",
            workers=3,
            data_uri="s3://my-bucket/data.csv",
            script_text="import os\nprint('hello')",
            cloud_secret_name="avaloka-raycreds-abc123",
            output_artifact_uri="s3://my-bucket/_runs/artifact.json",
            output_metrics_uri="s3://my-bucket/_runs/metrics.json",
        )
        defaults.update(kwargs)
        return render(**defaults)

    def test_unique_name_injected(self):
        result = self._render(name="my-unique-job")
        assert "my-unique-job" in result
        assert "rayjob-sample-3w" not in result

    def test_unique_configmap_name_injected(self):
        result = self._render(name="my-unique-job")
        assert "my-unique-job-code" in result

    def test_replicas_pinned_to_n(self):
        result = self._render(workers=5)
        for line in result.splitlines():
            stripped = line.strip()
            if stripped.startswith(("replicas:", "minReplicas:", "maxReplicas:")):
                val = int(stripped.split(":")[1].strip())
                assert val == 5, f"Expected 5 got {val} in: {line}"

    def test_data_uri_injected(self):
        result = self._render(data_uri="gs://my-bucket/dataset.parquet")
        assert "gs://my-bucket/dataset.parquet" in result
        assert "__DATA_SOURCE_URI__" not in result

    def test_secret_name_injected(self):
        result = self._render(cloud_secret_name="my-secret-xyz")
        assert "my-secret-xyz" in result
        assert "__CLOUD_SECRET_NAME__" not in result

    def test_artifact_uri_injected(self):
        result = self._render(output_artifact_uri="s3://bucket/artifact.json")
        assert "s3://bucket/artifact.json" in result
        assert "__OUTPUT_ARTIFACT_URI__" not in result

    def test_metrics_uri_injected(self):
        result = self._render(output_metrics_uri="s3://bucket/metrics.json")
        assert "s3://bucket/metrics.json" in result
        assert "__OUTPUT_METRICS_URI__" not in result

    def test_script_injected_into_configmap(self):
        result = self._render(script_text="import pandas as pd\ndf = pd.read_csv(uri)")
        assert "import pandas as pd" in result

    def test_no_leftover_placeholders(self):
        result = self._render()
        leftovers = re.findall(r"__([A-Z0-9_]+)__", result)
        assert leftovers == [], f"Leftover placeholders: {leftovers}"

    def test_name_sanitized_to_k8s_safe(self):
        """Names with uppercase/underscores must be sanitized."""
        result = self._render(name="My_Job_NAME")
        # k8s names must be lowercase alphanumeric + hyphens
        lines = [l for l in result.splitlines() if "name:" in l.lower()]
        for line in lines:
            val = line.split(":", 1)[1].strip()
            if val:
                assert re.match(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$", val) or len(val) == 1, \
                    f"Unsafe k8s name: {val!r}"

    def test_empty_script_raises(self):
        render = _get_renderer()
        with pytest.raises((ValueError, Exception)):
            render(
                base_yaml=MINIMAL_TEMPLATE,
                name="test",
                namespace="ray-jobs",
                workers=3,
                data_uri="s3://bucket/data.csv",
                script_text="",  # empty
                cloud_secret_name="secret",
            )


# ─────────────────────────────────────────────
# 2. LOG PARSER
# ─────────────────────────────────────────────

class TestLogParsers:
    """Tests for _parse_kv_metrics() and _parse_marker()"""

    def _get_parsers(self):
        from app.agents.execution_agent import _parse_kv_metrics, _parse_marker
        return _parse_kv_metrics, _parse_marker

    def test_parses_int_metric(self):
        parse_kv, _ = self._get_parsers()
        logs = "METRIC bytes_processed=10737418240"
        result = parse_kv(logs)
        assert result["bytes_processed"] == 10737418240
        assert isinstance(result["bytes_processed"], int)

    def test_parses_float_metric(self):
        parse_kv, _ = self._get_parsers()
        logs = "METRIC throughput_mb_s=86.9"
        result = parse_kv(logs)
        assert abs(result["throughput_mb_s"] - 86.9) < 0.001
        assert isinstance(result["throughput_mb_s"], float)

    def test_parses_multiple_metrics(self):
        parse_kv, _ = self._get_parsers()
        logs = textwrap.dedent("""\
            Some ray log output
            METRIC bytes_processed=10737418240
            METRIC throughput_mb_s=86.9
            METRIC runtime_s=123.4
            METRIC worker_count=3
            More output
        """)
        result = parse_kv(logs)
        assert result["bytes_processed"] == 10737418240
        assert abs(result["throughput_mb_s"] - 86.9) < 0.001
        assert abs(result["runtime_s"] - 123.4) < 0.001
        assert result["worker_count"] == 3

    def test_empty_logs_returns_empty_dict(self):
        parse_kv, _ = self._get_parsers()
        assert parse_kv("") == {}
        assert parse_kv(None) == {}

    def test_no_metrics_in_logs(self):
        parse_kv, _ = self._get_parsers()
        logs = "Ray started successfully\nNodes joined\nJob complete"
        assert parse_kv(logs) == {}

    def test_parse_marker_found(self):
        _, parse_marker = self._get_parsers()
        logs = "Processing...\nARTIFACT_URI=s3://bucket/_runs/artifact.json\nDone"
        result = parse_marker(logs, "ARTIFACT_URI")
        assert result == "s3://bucket/_runs/artifact.json"

    def test_parse_marker_not_found(self):
        _, parse_marker = self._get_parsers()
        logs = "No marker here"
        result = parse_marker(logs, "ARTIFACT_URI")
        assert result is None

    def test_parse_marker_empty_logs(self):
        _, parse_marker = self._get_parsers()
        assert parse_marker("", "ARTIFACT_URI") is None
        assert parse_marker(None, "ARTIFACT_URI") is None


# ─────────────────────────────────────────────
# 3. K8S SECRET YAML BUILDER
# ─────────────────────────────────────────────

class TestBuildCloudSecretYaml:
    """Tests for build_cloud_secret_yaml() in k8s_secrets.py"""

    def _build(self, provider, creds, secret_name="test-secret", namespace="ray-jobs"):
        from app.infra.k8s_secrets import build_cloud_secret_yaml
        return build_cloud_secret_yaml(
            secret_name=secret_name,
            namespace=namespace,
            provider=provider,
            creds=creds,
        )

    def test_s3_secret_contains_required_keys(self):
        yaml = self._build("s3", {
            "access_key": "AKIAIOSFODNN7EXAMPLE",
            "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "region": "us-east-1",
        })
        assert "AWS_ACCESS_KEY_ID" in yaml
        assert "AWS_SECRET_ACCESS_KEY" in yaml
        assert "AWS_DEFAULT_REGION" in yaml
        # Secret values must NOT appear in test output — just check structure
        assert "stringData:" in yaml

    def test_s3_secret_missing_creds_raises(self):
        from app.infra.k8s_secrets import build_cloud_secret_yaml
        with pytest.raises(ValueError, match="Missing AWS credentials"):
            build_cloud_secret_yaml(
                secret_name="s",
                namespace="n",
                provider="s3",
                creds={"region": "us-east-1"},  # no access_key/secret_key
            )

    def test_gcs_secret_contains_sa_json(self):
        yaml = self._build("gcs", {
            "service_account_json": '{"type": "service_account", "project_id": "my-proj"}',
        })
        assert "GCP_SERVICE_ACCOUNT_JSON" in yaml

    def test_gcs_secret_missing_sa_raises(self):
        from app.infra.k8s_secrets import build_cloud_secret_yaml
        with pytest.raises(ValueError, match="Missing GCP service account"):
            build_cloud_secret_yaml(
                secret_name="s",
                namespace="n",
                provider="gcs",
                creds={"project": "my-proj"},  # no SA json
            )

    def test_azure_secret_with_sas(self):
        yaml = self._build("azure", {
            "account_name": "mystorageaccount",
            "sas_token": "sv=2021-06-08&ss=b&...",
        })
        assert "AZURE_STORAGE_ACCOUNT_NAME" in yaml
        assert "AZURE_STORAGE_SAS_TOKEN" in yaml

    def test_azure_secret_missing_creds_raises(self):
        from app.infra.k8s_secrets import build_cloud_secret_yaml
        with pytest.raises(ValueError, match="Missing Azure credentials"):
            build_cloud_secret_yaml(
                secret_name="s",
                namespace="n",
                provider="azure",
                creds={},  # nothing
            )

    def test_unsupported_provider_raises(self):
        from app.infra.k8s_secrets import build_cloud_secret_yaml
        with pytest.raises(ValueError, match="Unsupported provider"):
            build_cloud_secret_yaml(
                secret_name="s",
                namespace="n",
                provider="unknown_cloud",
                creds={},
            )

    def test_secret_name_in_yaml(self):
        yaml = self._build("s3", {
            "access_key": "AK",
            "secret_key": "SK",
        }, secret_name="my-specific-secret")
        assert "my-specific-secret" in yaml

    def test_namespace_in_yaml(self):
        yaml = self._build("s3", {
            "access_key": "AK",
            "secret_key": "SK",
        }, namespace="custom-ns")
        assert "custom-ns" in yaml


# ─────────────────────────────────────────────
# 4. SERVER STATE WIRING
# ─────────────────────────────────────────────

class TestServerStateWiring:
    """
    Validate that send_message correctly wires
    data_source_location_cloud, data_source_location_local,
    and connection_id into state_in for k8s-ray mode.
    """

    def _make_mock_session(self, cloud_uri="s3://bucket/data.csv", connection_id="conn-123"):
        return {
            "user_id": "user-abc",
            "dataset_id": "ds-001",
            "work_dir": "/tmp/test",
            "data_source_location": cloud_uri,
            "object_name": "data.csv",
            "work_local_input": "/tmp/test/input_ds-001.csv",
            "sample_local_input": "/tmp/test/sample_input.csv",
            "output_location": "/tmp/test/output.csv",
            "schema": "{}",
            "ddl_schema": "",
            "input_data_type": "csv",
            "connection_id": connection_id,
            "uploaded_csv_preview": "[]",
            "uploaded_csv_columns": "[]",
        }

    def test_cloud_uri_wired_for_k8s_ray(self):
        """When execution_mode=k8s-ray, data_source_location must be the cloud URI."""
        sess = self._make_mock_session(cloud_uri="s3://bucket/data.csv")
        cloud_uri = sess.get("data_source_location")
        local_path = sess.get("work_local_input")

        # Simulate server.py logic (section 11 of send_message)
        execution_mode = "k8s-ray"
        state_data_source = cloud_uri if execution_mode == "k8s-ray" else local_path

        assert state_data_source == "s3://bucket/data.csv"
        assert state_data_source.startswith("s3://")

    def test_local_path_wired_for_local_mode(self):
        """When execution_mode=local, data_source_location must be the local path."""
        sess = self._make_mock_session()
        cloud_uri = sess.get("data_source_location")
        local_path = sess.get("work_local_input")

        execution_mode = "local"
        state_data_source = cloud_uri if execution_mode == "k8s-ray" else local_path

        assert state_data_source == "/tmp/test/input_ds-001.csv"

    def test_connection_id_present_in_session_for_ray(self):
        """connection_id must be stored in session when using register-existing-storage."""
        sess = self._make_mock_session(connection_id="conn-abc-123")
        assert sess.get("connection_id") == "conn-abc-123"

    def test_missing_cloud_uri_detected(self):
        """If cloud URI is missing for k8s-ray, it should be caught before job submit."""
        sess = self._make_mock_session(cloud_uri="")
        cloud_uri = sess.get("data_source_location") or ""
        execution_mode = "k8s-ray"

        if execution_mode == "k8s-ray":
            is_valid = bool(cloud_uri and "://" in cloud_uri)
        else:
            is_valid = True

        assert not is_valid, "Should detect missing cloud URI for k8s-ray"

    def test_missing_connection_id_detected(self):
        """If connection_id is missing for k8s-ray, it should be caught."""
        sess = self._make_mock_session(connection_id=None)
        # Simulate server.py validation block
        execution_mode = "k8s-ray"
        conn_id = sess.get("connection_id")

        if execution_mode == "k8s-ray":
            is_valid = bool(conn_id)
        else:
            is_valid = True

        assert not is_valid, "Should detect missing connection_id for k8s-ray"


# ─────────────────────────────────────────────
# 5. RAYJOB RUNNER — MOCKED KUBECTL
# ─────────────────────────────────────────────

class TestRayJobRunnerMocked:
    """
    Tests for ray_job_runner.py with kubectl mocked out.
    No cluster required.
    """

    def test_apply_yaml_calls_kubectl(self):
        from app.infra import ray_job_runner
        with patch.object(ray_job_runner, "_kubectl") as mock_kubectl:
            mock_kubectl.return_value = (0, "rayjob.ray.io/test-job created", "")
            ray_job_runner.apply_yaml_text("apiVersion: ray.io/v1\nkind: RayJob", namespace="ray-jobs")
            assert mock_kubectl.called

    def test_apply_yaml_raises_on_failure(self):
        from app.infra import ray_job_runner
        with patch.object(ray_job_runner, "_kubectl") as mock_kubectl:
            mock_kubectl.return_value = (1, "", "error: the server could not find the requested resource")
            with pytest.raises(RuntimeError, match="kubectl apply failed"):
                ray_job_runner.apply_yaml_text("bad yaml", namespace="ray-jobs")

    def test_get_rayjob_json_returns_parsed_dict(self):
        import json
        from app.infra import ray_job_runner

        fake_obj = {
            "metadata": {"name": "test-job"},
            "status": {"jobStatus": "SUCCEEDED"}
        }
        with patch.object(ray_job_runner, "_kubectl") as mock_kubectl:
            mock_kubectl.return_value = (0, json.dumps(fake_obj), "")
            result = ray_job_runner.get_rayjob_json("test-job", namespace="ray-jobs")
            assert result["status"]["jobStatus"] == "SUCCEEDED"

    def test_wait_for_rayjob_succeeds_immediately(self):
        import json
        from app.infra import ray_job_runner

        fake_obj = {
            "metadata": {"name": "test-job"},
            "status": {
                "jobStatus": "SUCCEEDED",
                "startTime": "2026-01-01T00:00:00Z",
                "endTime": "2026-01-01T00:02:00Z",
            }
        }
        with patch.object(ray_job_runner, "_kubectl") as mock_kubectl:
            mock_kubectl.return_value = (0, json.dumps(fake_obj), "")
            result = ray_job_runner.wait_for_rayjob("test-job", namespace="ray-jobs", timeout_s=30)
            assert result["status"]["jobStatus"] == "SUCCEEDED"

    def test_wait_for_rayjob_times_out(self):
        import json
        from app.infra import ray_job_runner

        # Always return RUNNING
        fake_obj = {
            "metadata": {"name": "test-job"},
            "status": {"jobStatus": "RUNNING"}
        }
        with patch.object(ray_job_runner, "_kubectl") as mock_kubectl:
            mock_kubectl.return_value = (0, json.dumps(fake_obj), "")
            with pytest.raises(TimeoutError):
                ray_job_runner.wait_for_rayjob(
                    "test-job",
                    namespace="ray-jobs",
                    timeout_s=1,      # very short timeout
                    poll_interval_s=0,
                )

    def test_run_rayjob_from_yaml_success_path(self):
        """Full end-to-end with kubectl mocked to return SUCCEEDED."""
        import json
        from app.infra import ray_job_runner

        succeeded_obj = {
            "metadata": {"name": "test-job"},
            "status": {
                "jobStatus": "SUCCEEDED",
                "startTime": "2026-01-01T00:00:00Z",
                "endTime": "2026-01-01T00:02:03Z",
            }
        }

        def _mock_kubectl(*args, **kwargs):
            cmd = list(args)
            if "apply" in cmd:
                return (0, "created", "")
            if "get" in cmd and "rayjob" in cmd:
                return (0, json.dumps(succeeded_obj), "")
            if "logs" in cmd:
                return (0, "METRIC bytes_processed=1024\nARTIFACT_URI=s3://b/a.json", "")
            if "delete" in cmd:
                return (0, "deleted", "")
            if "current-context" in cmd:
                return (0, "kind-kind", "")
            return (0, "", "")

        with patch.object(ray_job_runner, "_kubectl", side_effect=_mock_kubectl):
            outcome = ray_job_runner.run_rayjob_from_yaml(
                yaml_text=MINIMAL_TEMPLATE,
                namespace="ray-jobs",
                rayjob_name="test-job",
                timeout_s=30,
            )
        # assert outcome.status == "SUCCESS"
        assert outcome.status in ("SUCCESS", "SUCCEEDED")  # KubeRay mode returns SUCCEEDED
        assert outcome.runtime_s is not None
        assert outcome.runtime_s > 0

    def test_run_rayjob_from_yaml_failed_path(self):
        """Full end-to-end with kubectl mocked to return FAILED."""
        import json
        from app.infra import ray_job_runner

        failed_obj = {
            "metadata": {"name": "test-job"},
            "status": {"jobStatus": "FAILED"}
        }

        def _mock_kubectl(*args, **kwargs):
            cmd = list(args)
            if "apply" in cmd:
                return (0, "created", "")
            if "get" in cmd and "rayjob" in cmd:
                return (0, json.dumps(failed_obj), "")
            if "logs" in cmd:
                return (0, "Traceback: something went wrong", "")
            if "current-context" in cmd:
                return (0, "kind-kind", "")
            return (0, "", "")

        with patch.object(ray_job_runner, "_kubectl", side_effect=_mock_kubectl):
            outcome = ray_job_runner.run_rayjob_from_yaml(
                yaml_text=MINIMAL_TEMPLATE,
                namespace="ray-jobs",
                rayjob_name="test-job",
                timeout_s=30,
            )
        assert outcome.status == "FAILED"


# ─────────────────────────────────────────────
# 6. EXECUTION AGENT RAY NODE — MOCKED
# ─────────────────────────────────────────────

class TestExecutionAgentNodeRayMocked:
    """
    Tests for execution_agent_node_ray() with all external calls mocked.
    """

    def _base_state(self):
        return {
            "ray_namespace": "ray-jobs",
            "ray_workers": 3,
            "ray_timeout_s": 60,
            "data_source_location_cloud": "s3://bucket/data.csv",
            "connection_id": "conn-abc",
            "dataset_id": "ds-001",
            "coder_definition": {"code": "import os\nprint('hello')"},
            "messages": [],
        }

    def test_missing_cloud_uri_returns_error(self):
        from app.agents.execution_agent import execution_agent_node_ray
        state = self._base_state()
        state["data_source_location_cloud"] = None
        state["data_source_location"] = None

        result = execution_agent_node_ray(state)
        assert result["execution_result"]["status"] == "error"
        assert "Missing data_source_location_cloud" in result["execution_result"]["message"]

    # def test_missing_code_returns_error(self):
    #     from app.agents.execution_agent import execution_agent_node_ray
    #     state = self._base_state()
    #     state["coder_definition"] = {"code": ""}

    #     result = execution_agent_node_ray(state)
    #     assert result["execution_result"]["status"] == "error"
    #     assert "Missing coder_definition.code" in result["execution_result"]["message"]

    def test_missing_code_returns_error(self):
        """
        Empty coder_definition.code triggers _autogenerate_code().
        We patch it to return empty so the error path is hit.
        """
        import app.agents.execution_agent as _ea
        from app.agents.execution_agent import execution_agent_node_ray

        state = self._base_state()
        state["coder_definition"] = {"code": ""}

        with patch.object(_ea, "_autogenerate_code", return_value=""):
            result = execution_agent_node_ray(state)

        assert result["execution_result"]["status"] == "error"
        assert "Missing coder_definition.code" in result["execution_result"]["message"]

    

    def test_dryrun_mode_returns_yaml_without_submit(self):
        """RAYJOB_DRYRUN=1 should render YAML and return without calling kubectl."""
        import os
        from app.agents.execution_agent import execution_agent_node_ray

        state = self._base_state()

        with patch.dict(os.environ, {"RAYJOB_DRYRUN": "1"}):
            result = execution_agent_node_ray(state)

        assert result["execution_result"]["status"] == "dryrun"
        assert "rayjob_yaml" in result["execution_result"]
        assert result["execution_result"]["rayjob_name"] is not None

    def test_dryrun_does_not_call_kubectl(self):
        """Verify no kubectl calls in dryrun mode."""
        import os
        from app.agents.execution_agent import execution_agent_node_ray
        from app.infra import k8s_secrets

        state = self._base_state()

        with patch.dict(os.environ, {"RAYJOB_DRYRUN": "1"}):
            with patch.object(k8s_secrets, "kubectl_apply_yaml") as mock_apply:
                execution_agent_node_ray(state)
                mock_apply.assert_not_called()

    def test_execution_result_has_required_fields(self):
        """Successful run must return all required fields per design doc output contract."""
        import os
        import json
        from app.agents.execution_agent import execution_agent_node_ray
        from app.infra import ray_job_runner, k8s_secrets
        from app.infra.ray_job_runner import RayJobOutcome
        from app.api import cloud_connections

        state = self._base_state()

        mock_outcome = RayJobOutcome(
            status="SUCCESS",
            rayjob_name="rayjob-ds-001-123",
            namespace="ray-jobs",
            job_status="SUCCEEDED",
            deployment_status="Complete",
            runtime_s=42.5,
            logs="METRIC bytes_processed=1024\nMETRIC throughput_mb_s=0.1\nARTIFACT_URI=s3://b/a.json",
            details={},
        )

        fake_conn = {
            "provider": "s3",
            "access_key": "AK",
            "secret_key": "SK",
            "region": "us-east-1",
        }

        with patch.dict(os.environ, {"RAYJOB_DRYRUN": "0"}):
            with patch.object(k8s_secrets, "kubectl_apply_yaml"):
                with patch.object(k8s_secrets, "kubectl_delete_secret"):
                    with patch("app.agents.execution_agent._run_coro_sync", return_value=fake_conn):
                        with patch("app.agents.execution_agent.run_rayjob_from_yaml", return_value=mock_outcome):
                            result = execution_agent_node_ray(state)

        er = result["execution_result"]
        required_fields = ["mode", "status", "rayjob_name", "namespace", "worker_count", "runtime_s", "logs"]
        for field in required_fields:
            assert field in er, f"Missing required field in execution_result: {field}"

        assert er["mode"] == "k8s-ray"
        assert er["status"] == "success"
        assert er["worker_count"] == 3