# """
# RayJob runner utility:
# - Apply a RayJob YAML to Kubernetes
# - Wait for completion (SUCCEEDED / FAILED)
# - Collect logs
# - Return structured outcome for ETLState

# Works with:
# - Local kind/minikube clusters
# - Any K8s cluster reachable via current kubeconfig context

# Requirements:
# - kubectl must be on PATH (or set KUBECTL_PATH env var)
# - KubeRay operator + RayJob CRDs installed in the cluster
# """

# from __future__ import annotations

# import json
# import os
# import re
# import shutil
# import subprocess
# import tempfile
# import time
# from dataclasses import dataclass
# from datetime import datetime, timezone
# from typing import Any, Dict, Optional, Tuple


# # ----------------------------
# # Config / helpers
# # ----------------------------

# def _which_kubectl() -> str:
#     """Return kubectl executable path. Raises if not found."""
#     explicit = os.getenv("KUBECTL_PATH")
#     if explicit and os.path.exists(explicit):
#         return explicit

#     path = shutil.which("kubectl")
#     if not path:
#         raise RuntimeError(
#             "kubectl not found on PATH. Install kubectl or set KUBECTL_PATH to the full path."
#         )
#     return path


# def _run_cmd(cmd: list[str], timeout_s: int = 120) -> Tuple[int, str, str]:
#     """
#     Run a command with timeout. Returns (returncode, stdout, stderr).
#     Never uses shell=True (safer on Windows).
#     """
#     proc = subprocess.run(
#         cmd,
#         capture_output=True,
#         text=True,
#         timeout=timeout_s,
#         check=False,
#     )
#     return proc.returncode, proc.stdout or "", proc.stderr or ""


# def _kubectl(*args: str, timeout_s: int = 120) -> Tuple[int, str, str]:
#     kubectl = _which_kubectl()
#     return _run_cmd([kubectl, *args], timeout_s=timeout_s)


# def _now_utc() -> datetime:
#     return datetime.now(timezone.utc)


# def _parse_rfc3339(dt_str: Optional[str]) -> Optional[datetime]:
#     """
#     Parse common Kubernetes RFC3339 timestamps like: 2026-01-29T13:24:51Z
#     """
#     if not dt_str or not isinstance(dt_str, str):
#         return None
#     s = dt_str.strip()
#     # handle trailing 'Z'
#     if s.endswith("Z"):
#         s = s[:-1] + "+00:00"
#     try:
#         return datetime.fromisoformat(s)
#     except Exception:
#         return None


# def _extract_rayjob_name_from_yaml(yaml_text: str) -> Optional[str]:
#     """
#     Best-effort extraction of RayJob metadata.name from YAML.
#     Supports multi-doc YAML; finds the first doc with kind: RayJob.
#     """
#     if not yaml_text:
#         return None

#     # Find blocks that look like a RayJob doc
#     # We'll locate "kind: RayJob" and then search forward for "metadata:\n  name:"
#     kind_idx = [m.start() for m in re.finditer(r"(?m)^\s*kind:\s*RayJob\s*$", yaml_text)]
#     for idx in kind_idx:
#         tail = yaml_text[idx: idx + 2000]  # scan next chunk
#         m = re.search(r"(?ms)^\s*metadata:\s*\n(?:\s+.*\n)*?\s+name:\s*([A-Za-z0-9\-\.]+)\s*$", tail)
#         if m:
#             return m.group(1).strip()

#     # Fallback: first occurrence of "metadata: ... name:" anywhere
#     m2 = re.search(r"(?ms)^\s*metadata:\s*\n(?:\s+.*\n)*?\s+name:\s*([A-Za-z0-9\-\.]+)\s*$", yaml_text)
#     if m2:
#         return m2.group(1).strip()

#     return None


# def _get_current_context() -> str:
#     rc, out, err = _kubectl("config", "current-context", timeout_s=30)
#     if rc != 0:
#         return f"(unknown context; kubectl error: {err.strip()})"
#     return out.strip()


# # ----------------------------
# # RayJob status parsing
# # ----------------------------

# def _rayjob_status_fields(rayjob_obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
#     """
#     Return (job_status, deployment_status, start_time, end_time) if present.
#     KubeRay RayJob CRD fields can vary by version.
#     """
#     status = rayjob_obj.get("status") or {}

#     job_status = status.get("jobStatus") or status.get("status")  # common variants
#     deployment_status = status.get("jobDeploymentStatus") or status.get("deploymentStatus")

#     start_time = status.get("startTime") or status.get("submissionTime")
#     end_time = status.get("endTime") or status.get("completionTime")

#     # Some versions nest under "rayJobStatus"
#     if not job_status and isinstance(status.get("rayJobStatus"), dict):
#         job_status = status["rayJobStatus"].get("jobStatus")

#     return job_status, deployment_status, start_time, end_time


# def _is_terminal(job_status: Optional[str]) -> bool:
#     if not job_status:
#         return False
#     s = str(job_status).upper()
#     return s in {"SUCCEEDED", "FAILED", "STOPPED", "TERMINATED"}


# def _is_success(job_status: Optional[str]) -> bool:
#     return str(job_status).upper() == "SUCCEEDED"


# # ----------------------------
# # Public API
# # ----------------------------

# @dataclass
# class RayJobOutcome:
#     status: str  # SUCCESS | FAILED | TIMEOUT | ERROR
#     rayjob_name: str
#     namespace: str
#     job_status: Optional[str]
#     deployment_status: Optional[str]
#     runtime_s: Optional[float]
#     logs: str
#     details: Dict[str, Any]


# def apply_yaml_text(yaml_text: str, namespace: str = "default") -> None:
#     """
#     Apply YAML text to the cluster.
#     """
#     if not yaml_text.strip():
#         raise ValueError("Empty YAML text")

#     # Write to temp file and kubectl apply
#     with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
#         f.write(yaml_text)
#         tmp_path = f.name

#     try:
#         rc, out, err = _kubectl("apply", "-n", namespace, "-f", tmp_path, timeout_s=120)
#         if rc != 0:
#             raise RuntimeError(f"kubectl apply failed:\nSTDOUT:\n{out}\nSTDERR:\n{err}")
#     finally:
#         try:
#             os.remove(tmp_path)
#         except Exception:
#             pass


# def get_rayjob_json(rayjob_name: str, namespace: str = "default") -> Dict[str, Any]:
#     rc, out, err = _kubectl("get", "rayjob", rayjob_name, "-n", namespace, "-o", "json", timeout_s=60)
#     if rc != 0:
#         raise RuntimeError(f"kubectl get rayjob failed for {rayjob_name} in ns={namespace}: {err.strip()}")
#     return json.loads(out)


# def wait_for_rayjob(
#     rayjob_name: str,
#     namespace: str = "default",
#     timeout_s: int = 1800,
#     poll_interval_s: int = 5,
# ) -> Dict[str, Any]:
#     """
#     Poll RayJob status until terminal or timeout. Returns final RayJob object (json).
#     """
#     start = time.time()
#     last_status = None

#     while True:
#         obj = get_rayjob_json(rayjob_name, namespace=namespace)
#         job_status, deployment_status, _st, _et = _rayjob_status_fields(obj)

#         # Only print status changes if you add logging later; keep runner quiet by default
#         if job_status != last_status:
#             last_status = job_status

#         if _is_terminal(job_status):
#             return obj

#         if (time.time() - start) > timeout_s:
#             raise TimeoutError(f"RayJob {rayjob_name} did not reach terminal state within {timeout_s}s")

#         time.sleep(poll_interval_s)


# def fetch_rayjob_logs(
#     rayjob_name: str,
#     namespace: str = "default",
#     max_bytes: int = 200_000,
# ) -> str:
#     """
#     Fetch logs for a RayJob run.

#     Order:
#     0) Your template: pods labeled app=<rayjob_name> (head+worker)
#     1) Other templates: pods labeled job-name=<rayjob_name>
#     2) Fallback: find "submitter" pod by labels (preferred), else by name prefix "<rayjob_name>-"
#        but EXCLUDE head/worker using labels (ray-node-type=head/worker)
#     """

#     def _truncate(logs: str) -> str:
#         if len(logs) > max_bytes:
#             return "[truncated]\n" + logs[-max_bytes:]
#         return logs

    
#     # 1) Common alternative: pods labeled job-name=<rayjob_name>
#     rc1, out1, err1 = _kubectl(
#         "logs",
#         "-n", namespace,
#         "-l", f"job-name={rayjob_name}",
#         "--all-containers=true",
#         "--tail", "5000",
#         timeout_s=180,
#     )
#     if rc1 == 0 and (out1 or "").strip():
#         return _truncate(out1)

#     # 2) Fallback: list pods and find submitter pod
#     rc2, pods_json, err2 = _kubectl("get", "pods", "-n", namespace, "-o", "json", timeout_s=60)
#     if rc2 != 0 or not (pods_json or "").strip():
#         msg = (err2 or err1 or err0 or "").strip() or "unable to list pods"
#         return f"[log-fetch-failed] {msg}"

#     try:
#         obj = json.loads(pods_json)
#         items = obj.get("items", [])

#         candidates = []
#         for it in items:
#             meta = it.get("metadata", {}) or {}
#             name = meta.get("name", "") or ""
#             labels = meta.get("labels", {}) or {}

#             # Skip ray cluster pods (head/worker) using LABELS (reliable)
#             if labels.get("ray-node-type") in {"head", "worker"}:
#                 continue

#             # Prefer label match if present (different KubeRay versions use different keys)
#             label_job = (
#                 labels.get("ray.io/rayjob-name")
#                 or labels.get("rayjob-name")
#                 or labels.get("ray-job-name")
#             )
#             if label_job == rayjob_name:
#                 candidates.append(name)
#                 continue

#             # Fallback: name prefix (older/varied patterns)
#             if name.startswith(rayjob_name + "-"):
#                 candidates.append(name)

#         if not candidates:
#             msg = (err1 or err0 or "").strip() or "no submitter pod found"
#             return f"[log-fetch-failed] {msg}"

#         candidates.sort()
#         submitter_pod = candidates[-1]

#         rc3, out3, err3 = _kubectl(
#             "logs",
#             "-n", namespace,
#             submitter_pod,
#             "--all-containers=true",
#             "--tail", "5000",
#             timeout_s=180,
#         )
#         if rc3 != 0:
#             return f"[log-fetch-failed] {(err3 or out3 or '').strip()}"

#         return _truncate(out3)

#     except Exception as e:
#         return f"[log-fetch-failed] {str(e)}"


# def cleanup_rayjob_code_configmap(rayjob_name: str, namespace: str) -> None:
#     """Best-effort cleanup for the code ConfigMap created by the RayJob YAML template."""
#     cm_name = f"{rayjob_name}-code"
#     # --ignore-not-found keeps this safe even if it was already deleted
#     _kubectl("delete", "configmap", cm_name, "-n", namespace, "--ignore-not-found=true", timeout_s=60)


# def run_rayjob_from_yaml(
#     yaml_text: str,
#     namespace: str = "default",
#     rayjob_name: Optional[str] = None,
#     timeout_s: int = 1800,
#     poll_interval_s: int = 5,
# ) -> RayJobOutcome:
#     """
#     End-to-end: apply yaml → wait → logs → outcome.
#     If rayjob_name not provided, tries to extract from YAML.
#     """
#     ctx = _get_current_context()
#     name = rayjob_name or _extract_rayjob_name_from_yaml(yaml_text)
#     if not name:
#         raise ValueError("rayjob_name not provided and could not be extracted from YAML")

#     started = _now_utc()
#     try:
#         apply_yaml_text(yaml_text, namespace=namespace)

#         final_obj = wait_for_rayjob(
#             rayjob_name=name,
#             namespace=namespace,
#             timeout_s=timeout_s,
#             poll_interval_s=poll_interval_s,
#         )

#         job_status, deployment_status, st_str, et_str = _rayjob_status_fields(final_obj)
#         st = _parse_rfc3339(st_str) or started
#         et = _parse_rfc3339(et_str) or _now_utc()
#         runtime_s = max(0.0, (et - st).total_seconds())

#         logs = fetch_rayjob_logs(name, namespace=namespace)
#         cleanup_rayjob_code_configmap(name, namespace)


#         if _is_success(job_status):
#             return RayJobOutcome(
#                 status="SUCCESS",
#                 rayjob_name=name,
#                 namespace=namespace,
#                 job_status=job_status,
#                 deployment_status=deployment_status,
#                 runtime_s=runtime_s,
#                 logs=logs,
#                 details={"kube_context": ctx},
#             )

#         return RayJobOutcome(
#             status="FAILED",
#             rayjob_name=name,
#             namespace=namespace,
#             job_status=job_status,
#             deployment_status=deployment_status,
#             runtime_s=runtime_s,
#             logs=logs,
#             details={"kube_context": ctx, "rayjob": final_obj},
#         )

#     except TimeoutError as e:
#         logs = fetch_rayjob_logs(name, namespace=namespace)
#         cleanup_rayjob_code_configmap(name, namespace)
#         return RayJobOutcome(
#             status="TIMEOUT",
#             rayjob_name=name,
#             namespace=namespace,
#             job_status=None,
#             deployment_status=None,
#             runtime_s=None,
#             logs=logs,
#             details={"error": str(e), "kube_context": ctx},
#         )
#     except Exception as e:
#         # Include context + best-effort logs
#         logs = ""
#         try:
#             logs = fetch_rayjob_logs(name, namespace=namespace)
#             cleanup_rayjob_code_configmap(name, namespace)
#         except Exception:
#             pass
#         return RayJobOutcome(
#             status="ERROR",
#             rayjob_name=name,
#             namespace=namespace,
#             job_status=None,
#             deployment_status=None,
#             runtime_s=None,
#             logs=logs,
#             details={"error": str(e), "kube_context": ctx},
#         )








"""
RayJob runner utility.

Two submission modes (auto-selected):
  1. DIRECT  — Ray Job Submission API  (RAY_DASHBOARD_URL is set)
               Submits script directly to a running Ray cluster via HTTP.
               No KubeRay / kubectl required.
  2. KUBERAY — kubectl apply RayJob YAML (RAY_DASHBOARD_URL not set)
               Requires KubeRay operator + RayJob CRDs in the cluster.

Requirements (DIRECT mode):
  - pip install "ray[default]"
  - RAY_DASHBOARD_URL env var set, e.g. http://136.119.229.5:8265

Requirements (KUBERAY mode):
  - kubectl on PATH (or KUBECTL_PATH env var)
  - KubeRay operator installed in the cluster
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from uuid import uuid4
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def _pinned_submission_address(dashboard_url: str):
    """Force Ray's job-submission client to use ``dashboard_url``.

    RAY_API_SERVER_ADDRESS and RAY_ADDRESS both override the address passed to
    JobSubmissionClient (ray/dashboard/utils.py::get_address_for_submission_client),
    so a deployment that sets either for a *different* cluster silently hijacks
    every submission. Pin them while the client is constructed, then restore.
    """
    keys = ("RAY_API_SERVER_ADDRESS", "RAY_ADDRESS")
    previous = {k: os.environ.get(k) for k in keys}
    hijackers = {k: v for k, v in previous.items() if v and v != dashboard_url}
    if hijackers:
        logger.info(
            "[ray_job_runner] Pinning submission to %s (overriding %s)",
            dashboard_url,
            ", ".join(f"{k}={v}" for k, v in hijackers.items()),
        )
    os.environ["RAY_API_SERVER_ADDRESS"] = dashboard_url
    os.environ.pop("RAY_ADDRESS", None)
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _coerce_sa_json(value) -> Optional[str]:
    """Resolve a GCP service-account key to its JSON *content* string.

    ``value`` may be raw JSON text, an already-parsed dict, or a PATH to a key file
    (e.g. the value of ``GOOGLE_APPLICATION_CREDENTIALS``). Surrounding quotes — a
    common Windows ``set VAR="..."`` artefact — are stripped. Returns compact JSON,
    or ``None`` when the value can't be resolved to valid JSON, so callers never
    ship a bogus key (writing a file path AS the key silently drops the Ray job onto
    the cluster's default identity → 'Caller does not have storage.objects.get').
    """
    if not value:
        return None
    if isinstance(value, dict):
        try:
            return json.dumps(value)
        except Exception:
            return None
    s = str(value).strip().strip('"').strip("'").strip()
    if not s:
        return None
    if os.path.isfile(s):  # a path to a key file → read its contents
        try:
            with open(s, "r", encoding="utf-8") as _f:
                s = _f.read().strip()
        except OSError as _e:
            logger.warning("[ray_job_runner] Could not read SA key file %s: %s", value, _e)
            return None
    # Try the string as-is first (valid JSON parses, incl. proper \n escapes); only
    # then fall back to de-escaping literal "\n" (env vars sometimes double-escape
    # the private_key). Doing the replace first would corrupt already-valid JSON.
    for candidate in (s, s.replace("\\n", "\n")):
        try:
            return json.dumps(json.loads(candidate))
        except Exception:
            continue
    return None


# ----------------------------
# Config / helpers
# ----------------------------

def _which_kubectl() -> str:
    explicit = os.getenv("KUBECTL_PATH")
    if explicit and os.path.exists(explicit):
        return explicit
    path = shutil.which("kubectl")
    if not path:
        raise RuntimeError(
            "kubectl not found on PATH. Install kubectl or set KUBECTL_PATH."
        )
    return path


def _run_cmd(cmd: list[str], timeout_s: int = 120) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s, check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _kubectl(*args: str, timeout_s: int = 120) -> Tuple[int, str, str]:
    return _run_cmd([_which_kubectl(), *args], timeout_s=timeout_s)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_rfc3339(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str or not isinstance(dt_str, str):
        return None
    s = dt_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _extract_rayjob_name_from_yaml(yaml_text: str) -> Optional[str]:
    if not yaml_text:
        return None
    kind_idx = [m.start() for m in re.finditer(r"(?m)^\s*kind:\s*RayJob\s*$", yaml_text)]
    for idx in kind_idx:
        tail = yaml_text[idx: idx + 2000]
        m = re.search(r"(?ms)^\s*metadata:\s*\n(?:\s+.*\n)*?\s+name:\s*([A-Za-z0-9\-\.]+)\s*$", tail)
        if m:
            return m.group(1).strip()
    m2 = re.search(r"(?ms)^\s*metadata:\s*\n(?:\s+.*\n)*?\s+name:\s*([A-Za-z0-9\-\.]+)\s*$", yaml_text)
    if m2:
        return m2.group(1).strip()
    return None


def _extract_script_from_yaml(yaml_text: str) -> str:
    """
    Extract the sample_code.py contents from the ConfigMap block in the YAML.
    The block looks like:
      data:
        sample_code.py: |
            import os
            ...
    """
    m = re.search(r"(?ms)^\s*sample_code\.py:\s*\|\s*\n((?:[ \t]+.*\n?)*)", yaml_text)
    if not m:
        raise ValueError("Could not extract sample_code.py from YAML ConfigMap block")

    raw = m.group(1)
    # De-indent: find minimum indentation and strip it
    lines = raw.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        raise ValueError("sample_code.py block is empty")
    min_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
    dedented = "\n".join(l[min_indent:] if len(l) >= min_indent else l for l in lines)
    return dedented.strip()


# def _extract_runtime_env_from_yaml(yaml_text: str) -> dict:
#     """
#     Extract runtimeEnvYAML pip packages and env_vars from the RayJob YAML.
#     Returns a dict suitable for ray JobSubmissionClient runtime_env.
#     """
#     import yaml as _yaml

#     m = re.search(r"(?ms)runtimeEnvYAML:\s*\|\s*\n((?:[ \t]+.*\n?)*)", yaml_text)
#     if not m:
#         return {}

#     raw = m.group(1)
#     lines = raw.split("\n")
#     non_empty = [l for l in lines if l.strip()]
#     if not non_empty:
#         return {}
#     min_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
#     dedented = "\n".join(l[min_indent:] if len(l) >= min_indent else l for l in lines)

#     try:
#         parsed = _yaml.safe_load(dedented) or {}
#     except Exception:
#         return {}

#     runtime_env = {}
#     if "pip" in parsed:
#         runtime_env["pip"] = parsed["pip"]
#     if "env_vars" in parsed:
#         runtime_env["env_vars"] = {
#             k: str(v) for k, v in (parsed["env_vars"] or {}).items()
#         }
#     return runtime_env

def _extract_runtime_env_from_yaml(yaml_text: str) -> dict:
    """
    Extract pip packages and env_vars from the runtimeEnvYAML block only.
    Scoped to the literal block under ``runtimeEnvYAML: |`` so that K8s
    spec entries (name:, containerPort:, etc.) are never captured.
    """
    runtime_env = {}

    # Step 1: isolate the runtimeEnvYAML literal-block content.
    # It starts after "runtimeEnvYAML: |" and contains all subsequent
    # lines that are indented deeper than the key itself.
    block_match = re.search(
        r"runtimeEnvYAML:\s*\|\s*\n((?:[ \t]+.*\n|[ \t]*\n)*)", yaml_text
    )
    if not block_match:
        logger.warning("[ray_job_runner] No runtimeEnvYAML block found in YAML")
        return runtime_env

    env_yaml = block_match.group(1)

    # Step 2: extract pip packages from within the isolated block
    pip_block = re.search(r"pip:\s*\n((?:[ \t]+-[ \t]+\S.*\n?)*)", env_yaml)
    if pip_block:
        pkgs = []
        for line in pip_block.group(1).splitlines():
            m = re.match(r"[ \t]+-[ \t]+(\S+)", line)
            if m:
                pkgs.append(m.group(1))
        if pkgs:
            runtime_env["pip"] = pkgs

    # Step 3: extract env_vars from within the isolated block
    env_vars = {}
    env_block = re.search(r"env_vars:\s*\n((?:[ \t]+\S.*\n?)*)", env_yaml)
    if env_block:
        for line in env_block.group(1).splitlines():
            m = re.match(r'[ \t]+([A-Z_][A-Z0-9_]*):\s*["\']?(.*?)["\']?\s*$', line)
            if m:
                env_vars[m.group(1)] = m.group(2)
    if env_vars:
        runtime_env["env_vars"] = env_vars

    logger.info("[ray_job_runner] Extracted runtime_env: pip=%s env_vars=%s",
                runtime_env.get("pip", []),
                list(runtime_env.get("env_vars", {}).keys()))
    return runtime_env


def _get_current_context() -> str:
    rc, out, err = _kubectl("config", "current-context", timeout_s=30)
    if rc != 0:
        return f"(unknown context; kubectl error: {err.strip()})"
    return out.strip()


# ----------------------------
# RayJob status parsing (KubeRay)
# ----------------------------

def _rayjob_status_fields(rayjob_obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    status = rayjob_obj.get("status") or {}
    job_status = status.get("jobStatus") or status.get("status")
    deployment_status = status.get("jobDeploymentStatus") or status.get("deploymentStatus")
    start_time = status.get("startTime") or status.get("submissionTime")
    end_time = status.get("endTime") or status.get("completionTime")
    if not job_status and isinstance(status.get("rayJobStatus"), dict):
        job_status = status["rayJobStatus"].get("jobStatus")
    return job_status, deployment_status, start_time, end_time


def _is_terminal(job_status: Optional[str]) -> bool:
    if not job_status:
        return False
    return str(job_status).upper() in {"SUCCEEDED", "FAILED", "STOPPED", "TERMINATED"}


def _is_success(job_status: Optional[str]) -> bool:
    return str(job_status).upper() == "SUCCEEDED"


# ----------------------------
# Public API dataclass
# ----------------------------

@dataclass
class RayJobOutcome:
    status: str          # SUCCEEDED | FAILED | TIMEOUT | ERROR
    rayjob_name: str
    namespace: str
    job_status: Optional[str]
    deployment_status: Optional[str]
    runtime_s: Optional[float]
    logs: str
    details: Dict[str, Any]


# ============================================================
# DIRECT MODE — Ray Job Submission API
# ============================================================

def _run_ray_job_direct(
    script_text: str,
    runtime_env: dict,
    job_id: str,
    dashboard_url: str,
    timeout_s: int = 1800,
    poll_interval_s: int = 5,
    cloud_creds: Optional[Dict[str, Any]] = None,
) -> RayJobOutcome:
    """
    Submit a Python script to a running Ray cluster via the Job Submission API.
    No kubectl / KubeRay required.
    """
    try:
        from ray.job_submission import JobSubmissionClient, JobStatus
    except ImportError:
        raise RuntimeError(
            'ray[default] not installed. Run: pip install "ray[default]"'
        )

    logger.info("[ray_job_runner] DIRECT mode: submitting to %s  job_id=%s", dashboard_url, job_id)

    # Write script to a temp file — Ray will upload it as a working-dir file
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "sample_code.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_text)

        # Ray's get_address_for_submission_client() OVERRIDES the address passed
        # here with RAY_API_SERVER_ADDRESS (then RAY_ADDRESS) when either is set —
        # see ray/dashboard/utils.py: "address is always overridden by the
        # RAY_ADDRESS environment variable". With those pointing at a different
        # cluster, every job silently ran there instead: the target dashboard
        # reported 0 jobs while the work executed on the remote cluster and failed
        # to resolve this cluster's Service names. Pin both to our chosen URL for
        # the duration of the submission.
        with _pinned_submission_address(dashboard_url):
            client = JobSubmissionClient(dashboard_url)

        # Upload working dir so the script is available on remote nodes
        # submission_id = client.submit_job(
        #     entrypoint="python sample_code.py",
        #     submission_id=job_id,
        #     runtime_env={
        #         **runtime_env,
        #         "working_dir": tmpdir,
        #     },
        # )
        # Get GCP SA JSON from local environment to pass to Ray workers
        # gcp_sa = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()

        # # Upload working dir so the script is available on remote nodes
        # # DO NOT pass pip — cluster already has gcsfs, fsspec, pandas installed
        # submission_id = client.submit_job(
        #     entrypoint="python sample_code.py",
        #     submission_id=job_id,
        #     runtime_env={
        #         "working_dir": tmpdir,
        #         "pip": ["gcsfs"], 
        #         "env_vars": {
        #             **runtime_env.get("env_vars", {}),
        #             **({"GCP_SERVICE_ACCOUNT_JSON": gcp_sa} if gcp_sa else {}),
        #         },
        #     },
        # )
        import json as _json

        # Resolve GCP SA JSON to its CONTENT: connection key → GCP_SERVICE_ACCOUNT_JSON_PATH
        # → GCP_SERVICE_ACCOUNT_JSON. `_coerce_sa_json` returns None on anything invalid.
        # GOOGLE_APPLICATION_CREDENTIALS is NOT in the default chain: in a deployment it is
        # the PLATFORM's own key, so falling back would run a tenant's transfer as us.
        # DTA_ALLOW_AMBIENT_GCP_CREDS=1 opts back in on a dev machine.
        _sa_env_chain = ["GCP_SERVICE_ACCOUNT_JSON_PATH", "GCP_SERVICE_ACCOUNT_JSON"]
        if os.getenv("DTA_ALLOW_AMBIENT_GCP_CREDS", "0").strip().lower() in ("1", "true", "yes"):
            _sa_env_chain.append("GOOGLE_APPLICATION_CREDENTIALS")

        gcp_sa = ""
        if cloud_creds:
            for _k in ("service_account_json", "gcp_service_account_json", "secret_key"):
                _raw = cloud_creds.get(_k)
                gcp_sa = _coerce_sa_json(_raw) or ""
                if gcp_sa:
                    logger.info("[ray_job_runner] Using SA JSON from cloud_creds.%s (%d bytes)", _k, len(gcp_sa))
                    break
                if _raw:
                    # A bad key must fail loudly, not fall through to the env chain.
                    logger.warning(
                        "[ray_job_runner] cloud_creds.%s is set but is not valid SA JSON "
                        "(bad path/JSON?) — ignoring it.", _k,
                    )

        if not gcp_sa:
            for _env in _sa_env_chain:
                gcp_sa = _coerce_sa_json(os.getenv(_env, "")) or ""
                if gcp_sa:
                    logger.info("[ray_job_runner] Resolved SA JSON from %s (%d bytes)", _env, len(gcp_sa))
                    break
            else:
                if os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip():
                    logger.warning(
                        "[ray_job_runner] No connection SA key resolved. GOOGLE_APPLICATION_"
                        "CREDENTIALS is set but is NOT used as a fallback (it is the platform's "
                        "identity, not the tenant's); the job will run as the cluster's default "
                        "identity. Set DTA_ALLOW_AMBIENT_GCP_CREDS=1 to opt in on a dev machine."
                    )

        # When we have a service-account key, upload it in the working dir AND
        # point GOOGLE_APPLICATION_CREDENTIALS at it so the job authenticates as
        # that SA on the remote node (working-dir files land in the worker's cwd,
        # so a relative path resolves). Without this the job runs as the cluster's
        # default identity, which may lack access to the user's buckets.
        gcp_env: Dict[str, str] = {}
        if gcp_sa:
            sa_path = os.path.join(tmpdir, "gcp_sa.json")
            with open(sa_path, "w", encoding="utf-8") as _f:
                _f.write(gcp_sa)
            logger.info("[ray_job_runner] Wrote gcp_sa.json (%d bytes)", len(gcp_sa))
            gcp_env["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_sa.json"

        # Destination-write SA (cloud->cloud): exposed as DEST_GCP_SA_JSON so the
        # upload helper writes as the destination's own identity rather than the
        # ambient (source) SA above. The source read still uses GOOGLE_APPLICATION_
        # CREDENTIALS; only the write helper reads this var.
        dest_gcp_sa = ""
        if cloud_creds:
            dest_gcp_sa = _coerce_sa_json(cloud_creds.get("dest_gcp_service_account_json")) or ""
        if dest_gcp_sa:
            gcp_env["DEST_GCP_SA_JSON"] = dest_gcp_sa
            logger.info("[ray_job_runner] Forwarding destination SA as DEST_GCP_SA_JSON (%d bytes)", len(dest_gcp_sa))

        pip_packages = runtime_env.get("pip", ["gcsfs"])
        logger.info("[ray_job_runner] DIRECT mode pip packages: %s", pip_packages)

        try:
            submission_id = client.submit_job(
                entrypoint="python sample_code.py",
                submission_id=job_id,
                runtime_env={
                    "working_dir": tmpdir,
                    "pip": pip_packages,
                    "env_vars": {
                        **runtime_env.get("env_vars", {}),
                        **gcp_env,
                    },
                },
            )
        except Exception as e:
            # Targeted retry only for submission_id collisions (common when two submits happen same-second)
            msg = str(e)
            if "already exists" in msg and "submission_id" in msg:
                retry_id = f"{job_id}-{str(uuid4())[:8]}"
                logger.warning("[ray_job_runner] submission_id collision for %s; retrying as %s", job_id, retry_id)
                submission_id = client.submit_job(
                    entrypoint="python sample_code.py",
                    submission_id=retry_id,
                    runtime_env={
                        "working_dir": tmpdir,
                        "pip": pip_packages,
                        "env_vars": {
                            **runtime_env.get("env_vars", {}),
                        },
                    },
                )
            else:
                raise
        logger.info("[ray_job_runner] Submitted job submission_id=%s", submission_id)

        # Poll until terminal
        start = time.time()
        last_status = None
        while True:
            status = client.get_job_status(submission_id)
            if status != last_status:
                logger.info("[ray_job_runner] Job %s status: %s", submission_id, status)
                last_status = status

            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED):
                break

            if (time.time() - start) > timeout_s:
                logs = ""
                try:
                    logs = client.get_job_logs(submission_id)
                except Exception:
                    pass
                return RayJobOutcome(
                    status="TIMEOUT",
                    rayjob_name=job_id,
                    namespace="direct",
                    job_status=str(status),
                    deployment_status=None,
                    runtime_s=None,
                    logs=logs,
                    details={"dashboard_url": dashboard_url},
                )
            time.sleep(poll_interval_s)

        runtime_s = round(time.time() - start, 2)

        # Fetch logs
        logs = ""
        try:
            logs = client.get_job_logs(submission_id)
        except Exception as e:
            logs = f"[log-fetch-failed] {e}"

        final_status = "SUCCEEDED" if status == JobStatus.SUCCEEDED else "FAILED"
        logger.info("[ray_job_runner] Job %s finished: %s in %.1fs", submission_id, final_status, runtime_s)

        return RayJobOutcome(
            status=final_status,
            rayjob_name=job_id,
            namespace="direct",
            job_status=final_status,
            deployment_status=None,
            runtime_s=runtime_s,
            logs=logs,
            details={"dashboard_url": dashboard_url, "submission_id": submission_id},
        )


# ============================================================
# KUBERAY MODE — kubectl apply
# ============================================================

def apply_yaml_text(yaml_text: str, namespace: str = "default") -> None:
    if not yaml_text.strip():
        raise ValueError("Empty YAML text")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_text)
        tmp_path = f.name
    try:
        rc, out, err = _kubectl("apply", "-n", namespace, "-f", tmp_path, timeout_s=120)
        if rc != 0:
            raise RuntimeError(f"kubectl apply failed:\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def get_rayjob_json(rayjob_name: str, namespace: str = "default") -> Dict[str, Any]:
    rc, out, err = _kubectl("get", "rayjob", rayjob_name, "-n", namespace, "-o", "json", timeout_s=60)
    if rc != 0:
        raise RuntimeError(f"kubectl get rayjob failed for {rayjob_name} in ns={namespace}: {err.strip()}")
    return json.loads(out)


def wait_for_rayjob(
    rayjob_name: str,
    namespace: str = "default",
    timeout_s: int = 1800,
    poll_interval_s: int = 5,
) -> Dict[str, Any]:
    start = time.time()
    last_status = None
    while True:
        obj = get_rayjob_json(rayjob_name, namespace=namespace)
        job_status, _, _st, _et = _rayjob_status_fields(obj)
        if job_status != last_status:
            logger.info("[ray_job_runner] RayJob %s status: %s", rayjob_name, job_status)
            last_status = job_status
        if _is_terminal(job_status):
            return obj
        if (time.time() - start) > timeout_s:
            raise TimeoutError(f"RayJob {rayjob_name} did not reach terminal state within {timeout_s}s")
        time.sleep(poll_interval_s)


def fetch_rayjob_logs(rayjob_name: str, namespace: str = "default", max_bytes: int = 200_000) -> str:
    def _truncate(logs: str) -> str:
        if len(logs) > max_bytes:
            return "[truncated]\n" + logs[-max_bytes:]
        return logs

    rc1, out1, err1 = _kubectl(
        "logs", "-n", namespace, "-l", f"job-name={rayjob_name}",
        "--all-containers=true", "--tail", "5000", timeout_s=180,
    )
    if rc1 == 0 and (out1 or "").strip():
        return _truncate(out1)

    rc2, pods_json, err2 = _kubectl("get", "pods", "-n", namespace, "-o", "json", timeout_s=60)
    if rc2 != 0 or not (pods_json or "").strip():
        return f"[log-fetch-failed] {(err2 or err1 or '').strip() or 'unable to list pods'}"

    try:
        obj = json.loads(pods_json)
        items = obj.get("items", [])
        candidates = []
        for it in items:
            meta = it.get("metadata", {}) or {}
            name = meta.get("name", "") or ""
            labels = meta.get("labels", {}) or {}
            if labels.get("ray-node-type") in {"head", "worker"}:
                continue
            label_job = (
                labels.get("ray.io/rayjob-name")
                or labels.get("rayjob-name")
                or labels.get("ray-job-name")
            )
            if label_job == rayjob_name:
                candidates.append(name)
                continue
            if name.startswith(rayjob_name + "-"):
                candidates.append(name)

        if not candidates:
            return f"[log-fetch-failed] {(err1 or '').strip() or 'no submitter pod found'}"

        candidates.sort()
        submitter_pod = candidates[-1]
        rc3, out3, err3 = _kubectl(
            "logs", "-n", namespace, submitter_pod,
            "--all-containers=true", "--tail", "5000", timeout_s=180,
        )
        if rc3 != 0:
            return f"[log-fetch-failed] {(err3 or out3 or '').strip()}"
        return _truncate(out3)
    except Exception as e:
        return f"[log-fetch-failed] {str(e)}"


def cleanup_rayjob_code_configmap(rayjob_name: str, namespace: str) -> None:
    _kubectl("delete", "configmap", f"{rayjob_name}-code", "-n", namespace,
             "--ignore-not-found=true", timeout_s=60)


# ============================================================
# MAIN ENTRY POINT — auto-selects mode
# ============================================================

def run_rayjob_from_yaml(
    yaml_text: str,
    namespace: str = "default",
    rayjob_name: Optional[str] = None,
    timeout_s: int = 1800,
    poll_interval_s: int = 5,
    cloud_creds: Optional[Dict[str, Any]] = None,
    dashboard_url: Optional[str] = None,
) -> RayJobOutcome:
    """
    End-to-end RayJob execution.

    If a dashboard URL is available → DIRECT mode (no kubectl/KubeRay needed).
    Otherwise → KUBERAY mode (kubectl apply).

    ``dashboard_url`` overrides RAY_DASHBOARD_URL for this job only. The caller
    uses it to send transfers that touch cluster-local databases to an in-cluster
    Ray cluster, while cloud-only transfers keep going to the remote/default one.
    """
    dashboard_url = (dashboard_url or os.getenv("RAY_DASHBOARD_URL", "")).strip()
    name = rayjob_name or _extract_rayjob_name_from_yaml(yaml_text)
    if not name:
        raise ValueError("rayjob_name not provided and could not be extracted from YAML")

    # ── DIRECT MODE ──────────────────────────────────────────────────
    if dashboard_url:
        logger.info("[ray_job_runner] RAY_DASHBOARD_URL=%s → using DIRECT submission mode", dashboard_url)
        try:
            script_text = _extract_script_from_yaml(yaml_text)
        except Exception as e:
            return RayJobOutcome(
                status="ERROR", rayjob_name=name, namespace=namespace,
                job_status=None, deployment_status=None, runtime_s=None,
                logs="", details={"error": f"Failed to extract script from YAML: {e}"},
            )

        runtime_env = _extract_runtime_env_from_yaml(yaml_text)

        try:
            return _run_ray_job_direct(
                script_text=script_text,
                runtime_env=runtime_env,
                job_id=name,
                dashboard_url=dashboard_url,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
                cloud_creds=cloud_creds,
            )
        except Exception as e:
            logger.exception("[ray_job_runner] DIRECT submission failed: %s", e)
            return RayJobOutcome(
                status="ERROR", rayjob_name=name, namespace=namespace,
                job_status=None, deployment_status=None, runtime_s=None,
                logs="", details={"error": str(e), "dashboard_url": dashboard_url},
            )

    # ── KUBERAY MODE ─────────────────────────────────────────────────
    logger.info("[ray_job_runner] No RAY_DASHBOARD_URL → using KubeRay/kubectl mode")
    ctx = _get_current_context()
    started = _now_utc()
    logs = ""

    try:
        apply_yaml_text(yaml_text, namespace=namespace)

        final_obj = wait_for_rayjob(
            rayjob_name=name, namespace=namespace,
            timeout_s=timeout_s, poll_interval_s=poll_interval_s,
        )

        job_status, deployment_status, st_str, et_str = _rayjob_status_fields(final_obj)
        st = _parse_rfc3339(st_str) or started
        et = _parse_rfc3339(et_str) or _now_utc()
        runtime_s = max(0.0, (et - st).total_seconds())

        logs = fetch_rayjob_logs(name, namespace=namespace)
        cleanup_rayjob_code_configmap(name, namespace)

        if _is_success(job_status):
            return RayJobOutcome(
                status="SUCCEEDED", rayjob_name=name, namespace=namespace,
                job_status=job_status, deployment_status=deployment_status,
                runtime_s=runtime_s, logs=logs, details={"kube_context": ctx},
            )
        return RayJobOutcome(
            status="FAILED", rayjob_name=name, namespace=namespace,
            job_status=job_status, deployment_status=deployment_status,
            runtime_s=runtime_s, logs=logs,
            details={"kube_context": ctx, "rayjob": final_obj},
        )

    except TimeoutError as e:
        logs = fetch_rayjob_logs(name, namespace=namespace)
        cleanup_rayjob_code_configmap(name, namespace)
        return RayJobOutcome(
            status="TIMEOUT", rayjob_name=name, namespace=namespace,
            job_status=None, deployment_status=None, runtime_s=None,
            logs=logs, details={"error": str(e), "kube_context": ctx},
        )
    except Exception as e:
        try:
            logs = fetch_rayjob_logs(name, namespace=namespace)
            cleanup_rayjob_code_configmap(name, namespace)
        except Exception:
            pass
        return RayJobOutcome(
            status="ERROR", rayjob_name=name, namespace=namespace,
            job_status=None, deployment_status=None, runtime_s=None,
            logs=logs, details={"error": str(e), "kube_context": ctx},
        )