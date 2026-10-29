"""DIRECT-mode Ray submission (app/infra/ray_job_runner.py), Oct 15 fixes.

Issue 3  runtimeEnvYAML / sample_code.py literal blocks parsed by the YAML rule,
         so pip packages and env vars reach the job (was: "No module named 'daft'").
Issue 3  packages the script imports are always installed.
Issue 5  an unreachable dashboard fails in seconds, not ~2 minutes.
Issue 6  a retried submission keeps the GCP credentials.
"""
import json
import sys
import types

import pytest

from app.infra import ray_job_runner as rjr


# A RayJob manifest shaped like the one gke_run renders: a ConfigMap carrying the
# script, then the RayJob whose runtimeEnvYAML block is followed by SIBLING keys
# (rayClusterSpec, ...). The old regex ran on into those keys.
RAYJOB_YAML = """apiVersion: v1
kind: ConfigMap
metadata:
  name: dta-test-code
data:
  sample_code.py: |
    import daft
    import os

    def main():
        print("Starting data pipeline execution...")
---
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: dta-test
spec:
  entrypoint: python /home/ray/samples/sample_code.py
  runtimeEnvYAML: |
    pip:
      - pandas
      - daft
      - "google-cloud-storage>=2"
    env_vars:
      JOB_ID: "dta-test"
      SECRET: "abc\\\\def\\"ghi"
  rayClusterSpec:
    headGroupSpec:
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray:2.49.2
"""

SA_JSON = json.dumps({"type": "service_account", "project_id": "p", "private_key": "k", "client_email": "a@b.c"})


# ---------------------------------------------------------------- _literal_block

def test_literal_block_stops_at_sibling_key():
    body = rjr._literal_block(RAYJOB_YAML, "runtimeEnvYAML")
    assert body.startswith("pip:")
    assert "rayClusterSpec" not in body
    assert "headGroupSpec" not in body


def test_literal_block_with_crlf_line_endings():
    body = rjr._literal_block(RAYJOB_YAML.replace("\n", "\r\n"), "runtimeEnvYAML")
    assert body is not None
    assert "rayClusterSpec" not in body
    assert "- daft" in body


def test_literal_block_accepts_strip_and_keep_indicators():
    for indicator in ("|-", "|+"):
        text = f"a:\n  key: {indicator}\n    line one\n    line two\n  other: 1\n"
        assert rjr._literal_block(text, "key") == "line one\nline two"


def test_literal_block_keeps_blank_lines_inside_and_drops_trailing_ones():
    text = "key: |\n  one\n\n  two\n\n\nnext: 1\n"
    assert rjr._literal_block(text, "key") == "one\n\ntwo"


def test_literal_block_missing_key_returns_none():
    assert rjr._literal_block("a: 1\n", "runtimeEnvYAML") is None
    assert rjr._literal_block("", "runtimeEnvYAML") is None
    assert rjr._literal_block(None, "runtimeEnvYAML") is None


# ------------------------------------------------- _extract_runtime_env_from_yaml

def test_runtime_env_reads_pip_and_env_vars_from_full_rayjob():
    env = rjr._extract_runtime_env_from_yaml(RAYJOB_YAML)
    assert env["pip"] == ["pandas", "daft", "google-cloud-storage>=2"]
    assert set(env["env_vars"]) == {"JOB_ID", "SECRET"}


def test_runtime_env_unescapes_quoted_secret():
    # A secret containing a backslash and a double quote must reach Ray as-is.
    env = rjr._extract_runtime_env_from_yaml(RAYJOB_YAML)
    assert env["env_vars"]["SECRET"] == 'abc\\def"ghi'


def test_runtime_env_same_result_with_crlf():
    assert rjr._extract_runtime_env_from_yaml(RAYJOB_YAML.replace("\n", "\r\n")) == \
        rjr._extract_runtime_env_from_yaml(RAYJOB_YAML)


def test_runtime_env_empty_when_block_absent():
    assert rjr._extract_runtime_env_from_yaml("kind: RayJob\nspec:\n  entrypoint: x\n") == {}


def test_runtime_env_values_are_strings():
    yaml_text = "spec:\n  runtimeEnvYAML: |\n    env_vars:\n      RETRIES: 3\n      DEBUG: true\n  other: 1\n"
    env = rjr._extract_runtime_env_from_yaml(yaml_text)
    assert env["env_vars"] == {"RETRIES": "3", "DEBUG": "True"}


# ------------------------------------------------------ _extract_script_from_yaml

def test_script_extracted_and_dedented():
    script = rjr._extract_script_from_yaml(RAYJOB_YAML)
    assert script.startswith("import daft\nimport os")
    assert '    print("Starting data pipeline execution...")' in script
    assert "apiVersion" not in script


def test_script_stops_at_sibling_key_without_document_separator():
    # The old regex only stopped because a '---' line followed the block.
    yaml_text = (
        "data:\n"
        "  sample_code.py: |\n"
        "    import daft\n"
        "    print(1)\n"
        "  helper.txt: |\n"
        "    not python\n"
    )
    assert rjr._extract_script_from_yaml(yaml_text) == "import daft\nprint(1)"


def test_script_missing_raises():
    with pytest.raises(ValueError, match="sample_code.py"):
        rjr._extract_script_from_yaml("kind: RayJob\n")


# ------------------------------------------------------- _ensure_script_packages

@pytest.fixture
def pinned_versions(monkeypatch):
    import importlib.metadata as md

    def fake_version(dist):
        if dist == "daft":
            return "0.6.1"
        raise md.PackageNotFoundError(dist)

    monkeypatch.setattr(md, "version", fake_version)


def test_adds_daft_pinned_to_backend_version(pinned_versions):
    assert rjr._ensure_script_packages(["gcsfs"], "import daft\n") == ["gcsfs", "daft==0.6.1"]


def test_from_import_counts_as_import(pinned_versions):
    assert "daft==0.6.1" in rjr._ensure_script_packages([], "from daft import col\n")


def test_unpinned_when_backend_lacks_package(pinned_versions):
    assert rjr._ensure_script_packages([], "import gcsfs\n") == ["gcsfs"]


@pytest.mark.parametrize("listed", ["daft", "daft==0.5.1", "daft>=0.4", "daft[gcp]", "Daft"])
def test_not_added_twice_whatever_the_specifier(pinned_versions, listed):
    assert rjr._ensure_script_packages([listed], "import daft\n") == [listed]


def test_nothing_added_when_script_does_not_import(pinned_versions):
    assert rjr._ensure_script_packages(["gcsfs"], "import pandas\n# daft mentioned in a comment\n") == ["gcsfs"]


def test_handles_none_inputs(pinned_versions):
    assert rjr._ensure_script_packages(None, None) == []


def test_switch_turns_auto_packages_off(pinned_versions, monkeypatch):
    monkeypatch.setenv("RAY_ADD_MISSING_SCRIPT_PACKAGES", "0")
    assert rjr._ensure_script_packages(["gcsfs"], "import daft\n") == ["gcsfs"]


# --------------------------------------------------------- _preflight_dashboard

class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_preflight_passes_when_dashboard_answers(monkeypatch):
    import urllib.request
    calls = []

    def answer(url, timeout):
        calls.append((url, timeout))
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", answer)
    monkeypatch.setenv("RAY_DASHBOARD_CONNECT_TIMEOUT_S", "7")
    rjr._preflight_dashboard("http://127.0.0.1:18265/")
    assert calls == [("http://127.0.0.1:18265/api/version", 7.0)]


def test_preflight_http_error_still_means_reachable(monkeypatch):
    import urllib.error
    import urllib.request

    def not_found(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", not_found)
    rjr._preflight_dashboard("http://127.0.0.1:18265")      # no exception


def test_preflight_raises_clear_error_when_unreachable(monkeypatch):
    import urllib.error
    import urllib.request

    def refuse(url, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(ConnectionError, match="Ray cluster unreachable at http://10.0.0.1:8265"):
        rjr._preflight_dashboard("http://10.0.0.1:8265")


# ------------------------------------------------ _run_ray_job_direct (fake Ray)

class _FakeRay:
    """Stands in for ray.job_submission; records what the runner submits."""

    def __init__(self, collide_first=False, final="SUCCEEDED"):
        self.events = []
        self.submits = []
        self.collide_first = collide_first
        self.final = final
        fake = self

        class JobStatus:
            SUCCEEDED = "SUCCEEDED"
            FAILED = "FAILED"
            STOPPED = "STOPPED"

        class JobSubmissionClient:
            def __init__(self, url):
                fake.events.append(("client", url))

            def submit_job(self, **kw):
                fake.submits.append(kw)
                if fake.collide_first and len(fake.submits) == 1:
                    raise RuntimeError(f"Job with submission_id {kw['submission_id']} already exists")
                return kw["submission_id"]

            def get_job_status(self, submission_id):
                return fake.final

            def get_job_logs(self, submission_id):
                return "AVALOKA_RESULT rows_inserted=20"

        self.module = types.ModuleType("ray.job_submission")
        self.module.JobStatus = JobStatus
        self.module.JobSubmissionClient = JobSubmissionClient

    def install(self, monkeypatch):
        ray_pkg = types.ModuleType("ray")
        ray_pkg.job_submission = self.module
        monkeypatch.setitem(sys.modules, "ray", ray_pkg)
        monkeypatch.setitem(sys.modules, "ray.job_submission", self.module)
        return self


@pytest.fixture
def no_preflight(monkeypatch):
    monkeypatch.setattr(rjr, "_preflight_dashboard", lambda url: None)


def _run(**kw):
    args = dict(
        script_text="import daft\nprint('hi')\n",
        runtime_env={"pip": ["pandas"], "env_vars": {"JOB_ID": "j"}},
        job_id="dta-dta-test",
        dashboard_url="http://127.0.0.1:18265",
        poll_interval_s=0,
        cloud_creds={"gcp_service_account_json": SA_JSON, "dest_gcp_service_account_json": SA_JSON},
    )
    args.update(kw)
    return rjr._run_ray_job_direct(**args)


def test_retry_after_id_collision_keeps_credentials_and_packages(monkeypatch, no_preflight, pinned_versions):
    ray = _FakeRay(collide_first=True).install(monkeypatch)
    out = _run()

    assert out.status == "SUCCEEDED"
    first, retry = ray.submits
    assert retry["submission_id"].startswith("dta-dta-test-")
    assert retry["submission_id"] != first["submission_id"]
    # The bug: the retry rebuilt env_vars without the GCP keys.
    assert retry["runtime_env"]["env_vars"] == first["runtime_env"]["env_vars"]
    assert retry["runtime_env"]["env_vars"]["GOOGLE_APPLICATION_CREDENTIALS"] == "gcp_sa.json"
    assert "DEST_GCP_SA_JSON" in retry["runtime_env"]["env_vars"]
    assert retry["runtime_env"]["pip"] == first["runtime_env"]["pip"]


def test_submitted_env_has_template_vars_creds_and_daft(monkeypatch, no_preflight, pinned_versions):
    ray = _FakeRay().install(monkeypatch)
    _run()
    (sub,) = ray.submits
    assert sub["runtime_env"]["pip"] == ["pandas", "daft==0.6.1"]
    assert sub["runtime_env"]["env_vars"]["JOB_ID"] == "j"
    assert sub["runtime_env"]["env_vars"]["GOOGLE_APPLICATION_CREDENTIALS"] == "gcp_sa.json"
    assert sub["entrypoint"] == "python sample_code.py"


def test_other_submit_errors_are_not_retried(monkeypatch, no_preflight, pinned_versions):
    ray = _FakeRay().install(monkeypatch)

    def boom(self, **kw):
        ray.submits.append(kw)
        raise RuntimeError("runtime env setup failed")

    monkeypatch.setattr(ray.module.JobSubmissionClient, "submit_job", boom)
    with pytest.raises(RuntimeError, match="runtime env setup failed"):
        _run()
    assert len(ray.submits) == 1


def test_failed_job_reported_as_failed(monkeypatch, no_preflight, pinned_versions):
    _FakeRay(final="FAILED").install(monkeypatch)
    assert _run().status == "FAILED"


def test_preflight_runs_before_client_is_created(monkeypatch, pinned_versions):
    ray = _FakeRay().install(monkeypatch)
    monkeypatch.setattr(rjr, "_preflight_dashboard", lambda url: ray.events.append(("preflight", url)))
    _run()
    assert ray.events[:2] == [("preflight", "http://127.0.0.1:18265"), ("client", "http://127.0.0.1:18265")]


# ------------------------------------------------- run_rayjob_from_yaml (end to end)

def test_end_to_end_yaml_to_submission(monkeypatch, no_preflight, pinned_versions):
    ray = _FakeRay().install(monkeypatch)
    out = rjr.run_rayjob_from_yaml(
        RAYJOB_YAML, rayjob_name="dta-test", dashboard_url="http://127.0.0.1:18265",
        poll_interval_s=0, cloud_creds={"gcp_service_account_json": SA_JSON},
    )
    assert out.status == "SUCCEEDED"
    (sub,) = ray.submits
    # Template packages arrive (daft already listed, so not added twice).
    assert sub["runtime_env"]["pip"] == ["pandas", "daft", "google-cloud-storage>=2"]
    assert sub["runtime_env"]["env_vars"]["SECRET"] == 'abc\\def"ghi'


def test_unreachable_dashboard_returns_error_outcome_without_submitting(monkeypatch, pinned_versions):
    import urllib.request
    ray = _FakeRay().install(monkeypatch)

    def refuse(url, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    out = rjr.run_rayjob_from_yaml(RAYJOB_YAML, rayjob_name="dta-test",
                                   dashboard_url="http://34.132.134.104:8265")
    assert out.status == "ERROR"
    assert "Ray cluster unreachable" in out.details["error"]
    assert ray.submits == []
    assert ("client", "http://34.132.134.104:8265") not in ray.events