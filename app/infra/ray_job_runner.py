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
import threading
import time
from uuid import uuid4
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ----------------------------
# Job tags (who a Ray job belongs to)
# ----------------------------
#
# Every DIRECT-mode submission carries Ray job metadata naming the user, the chat
# thread and the kind of work, and gets a row in the ray_jobs table. The Ray jobs
# API (app/api/ray_jobs.py) lists a user's jobs from those rows and checks
# ownership against them: Ray itself lists every user's jobs together, and forgets
# them all when the head pod restarts.
#
# The caller (the planner) knows the user; this module is called several layers
# below it. ray_job_context() carries the tags down without changing every
# signature in between; run_rayjob_from_yaml(job_tags=...) takes them explicitly.

JOB_TAG_KEYS = ("user_id", "thread_id", "analysis_id", "kind", "source", "destination")
_MAX_TAG_LEN = 512
_JOB_TAGS: ContextVar[Mapping[str, str]] = ContextVar("ray_job_tags", default={})

# user:password@ inside a URI (postgresql://u:p@host/db) must never reach Ray
# metadata or the jobs table: both are shown in the UI and the Ray dashboard.
_URI_CREDENTIALS = re.compile(r"(?<=://)[^/@\s]+@")


@contextmanager
def ray_job_context(**tags: Any):
    """Tag every Ray job submitted inside this block (nested blocks add to the tags).

        with ray_job_context(user_id=state["user_id"], thread_id=state["thread_id"],
                             kind="data_transfer", source=src, destination=dst):
            run_the_transfer(...)
    """
    merged = {**_JOB_TAGS.get(), **{k: str(v) for k, v in tags.items() if v not in (None, "")}}
    token = _JOB_TAGS.set(merged)
    try:
        yield merged
    finally:
        _JOB_TAGS.reset(token)


def current_job_tags() -> Dict[str, str]:
    return dict(_JOB_TAGS.get())


def _job_metadata(tags: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """Ray job metadata from tags: known keys only, strings, no URI credentials."""
    out: Dict[str, str] = {}
    for key in JOB_TAG_KEYS:
        value = (tags or {}).get(key)
        if value in (None, ""):
            continue
        text = _URI_CREDENTIALS.sub("", str(value))[:_MAX_TAG_LEN]
        out[key] = text
    if out:
        out.setdefault("kind", "data_transfer")
    return out


def _status_name(status: Any) -> str:
    """'SUCCEEDED' for JobStatus.SUCCEEDED, whatever str() of the enum prints."""
    return str(getattr(status, "value", status) or "").upper()


_ROWS_RE = re.compile(r"AVALOKA_RESULT\s+rows_inserted=(\d+)")


def _rows_from_logs(logs: str) -> Optional[int]:
    found = _ROWS_RE.findall(logs or "")
    return int(found[-1]) if found else None


def _store_call(fn_name: str, *args: Any, **kwargs: Any) -> None:
    """Best-effort write to the ray_jobs table: never fails the job itself."""
    try:
        from app.infra import ray_jobs_store
        getattr(ray_jobs_store, fn_name)(*args, **kwargs)
    except Exception as exc:  # table missing, Supabase down, import error in a Ray-only image
        logger.warning("[ray_job_runner] ray_jobs.%s failed (job continues): %s", fn_name, exc)


# JobSubmissionClient reads RAY_API_SERVER_ADDRESS / RAY_ADDRESS while it is
# constructed, and _pinned_submission_address() rewrites os.environ to stop that.
# The Ray jobs API builds clients from request threads, so construction is
# serialised: two threads pinning different URLs at once would hijack each other.
_CLIENT_LOCK = threading.Lock()


def make_job_client(dashboard_url: str):
    """A JobSubmissionClient for exactly ``dashboard_url`` (env overrides pinned out)."""
    try:
        from ray.job_submission import JobSubmissionClient
    except ImportError:
        raise RuntimeError('ray[default] not installed. Run: pip install "ray[default]"')
    with _CLIENT_LOCK, _pinned_submission_address(dashboard_url):
        return JobSubmissionClient(dashboard_url)


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


def _literal_block(yaml_text: str, key: str) -> Optional[str]:
    """Return the body of a YAML literal block ``<key>: |`` (dedented), or None.

    YAML rule: a literal block ends at the first non-blank line indented at or
    below the key's own indentation. The old regexes kept every line that merely
    *started* with whitespace, so the block ran on into the following sibling keys
    (``rayClusterSpec:`` etc.) and safe_load failed on the mixed indentation.
    """
    m = re.search(rf"(?m)^([ \t]*){re.escape(key)}:[ \t]*\|[-+]?[ \t]*\r?$", yaml_text or "")
    if not m:
        return None
    key_indent = len(m.group(1).expandtabs())
    body: list[str] = []
    for line in yaml_text[m.end():].splitlines()[1:]:
        if line.strip() and len(line.expandtabs()) - len(line.expandtabs().lstrip()) <= key_indent:
            break
        body.append(line)
    while body and not body[-1].strip():
        body.pop()
    import textwrap
    return textwrap.dedent("\n".join(body))


def _extract_script_from_yaml(yaml_text: str) -> str:
    """
    Extract the sample_code.py contents from the ConfigMap block in the YAML.
    The block looks like:
      data:
        sample_code.py: |
            import os
            ...
    """
    block = _literal_block(yaml_text, "sample_code.py")
    if block is None:
        raise ValueError("Could not extract sample_code.py from YAML ConfigMap block")

    raw = block
    # De-indent: find minimum indentation and strip it
    lines = raw.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        raise ValueError("sample_code.py block is empty")
    min_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
    dedented = "\n".join(l[min_indent:] if len(l) >= min_indent else l for l in lines)
    return dedented.strip()

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
    env_yaml = _literal_block(yaml_text, "runtimeEnvYAML")
    if env_yaml is None:
        logger.warning("[ray_job_runner] No runtimeEnvYAML block found in YAML")
        return runtime_env

    # Step 2: parse the isolated block as YAML.
    #
    # This used to scan the block line by line with regexes, stripping one
    # optional quote off each end of the value. That reads a *double-quoted YAML
    # scalar* as if it were plain text, so every escape survived into the value:
    # _render_cloud_env_vars writes the secret abc\def"ghi correctly as
    # "abc\\def\"ghi", and the regex handed Ray back the literal abc\\def\"ghi.
    # Any credential containing a backslash or a double quote -- a Windows path,
    # an ODBC connection string, roughly one AWS secret key in thirty -- reached
    # the job corrupted, and the failure surfaced as an authentication error
    # against the remote store rather than as a quoting bug here.
    #
    # yaml.safe_load does the unescaping correctly. The block scoping above is
    # kept: it is what stops K8s spec entries (name:, containerPort:) being
    # picked up, which is why the line scanner replaced an earlier safe_load.
    import yaml as _yaml
    import textwrap

    parsed = {}
    try:
        parsed = _yaml.safe_load(textwrap.dedent(env_yaml)) or {}
    except Exception as exc:
        logger.warning("[ray_job_runner] runtimeEnvYAML block did not parse: %s", exc)
    if not isinstance(parsed, dict):
        parsed = {}

    pkgs = parsed.get("pip")
    if isinstance(pkgs, list) and pkgs:
        runtime_env["pip"] = [str(pkg) for pkg in pkgs]

    env_vars = parsed.get("env_vars")
    if isinstance(env_vars, dict) and env_vars:
        runtime_env["env_vars"] = {str(k): str(v) for k, v in env_vars.items()}

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


def _preflight_dashboard(dashboard_url: str) -> None:
    """Raise a clear ConnectionError within a few seconds if the Ray dashboard is down.

    Standard library only (urllib), so this adds no dependency. Any HTTP answer,
    even an error status, means the dashboard is reachable.
    """
    import urllib.error
    import urllib.request

    timeout_s = float(os.getenv("RAY_DASHBOARD_CONNECT_TIMEOUT_S", "10"))
    url = f"{dashboard_url.rstrip('/')}/api/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s):
            pass
    except urllib.error.HTTPError:
        return
    except (urllib.error.URLError, OSError) as exc:
        raise ConnectionError(
            f"Ray cluster unreachable at {dashboard_url} (no answer in {timeout_s:.0f}s): {exc}"
        ) from exc


# Modules the generated DTA script imports that the Ray image may not have.
# import name -> pip distribution name.
_SCRIPT_PACKAGES = {"daft": "daft", "gcsfs": "gcsfs"}


def _ensure_script_packages(pip_packages, script_text: str) -> list:
    """Add any package the script imports but the runtime env doesn't install.

    Only when the job config does not already list it: a configured version
    always wins. The added package is pinned to the version installed in this
    backend, so the job runs the same daft that generated and validated the
    code. Set RAY_ADD_MISSING_SCRIPT_PACKAGES=0 to turn this off.
    """
    from importlib import metadata

    pkgs = [str(p) for p in (pip_packages or [])]
    if os.getenv("RAY_ADD_MISSING_SCRIPT_PACKAGES", "1").strip() == "0":
        return pkgs
    have = {re.split(r"[<>=!~\[ ;]", p, 1)[0].strip().lower() for p in pkgs}
    for module, dist in _SCRIPT_PACKAGES.items():
        if dist in have:
            continue
        if not re.search(rf"(?m)^\s*(?:import|from)\s+{module}\b", script_text or ""):
            continue
        try:
            pkgs.append(f"{dist}=={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            pkgs.append(dist)
    return pkgs


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
    job_tags: Optional[Mapping[str, Any]] = None,
) -> RayJobOutcome:
    """
    Submit a Python script to a running Ray cluster via the Job Submission API.
    No kubectl / KubeRay required.

    ``job_tags`` (user_id, thread_id, kind, ...) become Ray job metadata and a
    ray_jobs row; without a user_id the job runs untagged, as before.
    """
    try:
        from ray.job_submission import JobStatus
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

        # Fail fast when the dashboard is unreachable. JobSubmissionClient has no
        # connect timeout, so a dead cluster used to hang the chat turn for ~2 min.
        _preflight_dashboard(dashboard_url)

        # Ray's get_address_for_submission_client() OVERRIDES the address passed
        # here with RAY_API_SERVER_ADDRESS (then RAY_ADDRESS) when either is set —
        # see ray/dashboard/utils.py: "address is always overridden by the
        # RAY_ADDRESS environment variable". With those pointing at a different
        # cluster, every job silently ran there instead: the target dashboard
        # reported 0 jobs while the work executed on the remote cluster and failed
        # to resolve this cluster's Service names. Pin both to our chosen URL for
        # the duration of the submission.
        client = make_job_client(dashboard_url)

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

        pip_packages = _ensure_script_packages(
            runtime_env.get("pip", ["gcsfs"]), script_text,
        )
        logger.info("[ray_job_runner] DIRECT mode pip packages: %s", pip_packages)

        # One runtime env, used for the first submit AND the retry, so the retry
        # also gets the GCP key (gcp_env). The retry used to leave it out.
        job_runtime_env = {
            "working_dir": tmpdir,
            "pip": pip_packages,
            "env_vars": {
                **runtime_env.get("env_vars", {}),
                **gcp_env,
            },
        }

        # Who this job belongs to: Ray metadata (shown in the dashboard and read
        # back by the Ray jobs API) plus a ray_jobs row. Same for the retry.
        metadata = _job_metadata(job_tags)

        try:
            submission_id = client.submit_job(
                entrypoint="python sample_code.py",
                submission_id=job_id,
                runtime_env=job_runtime_env,
                metadata=metadata or None,
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
                    runtime_env=job_runtime_env,
                    metadata=metadata or None,
                )
            else:
                raise
        logger.info("[ray_job_runner] Submitted job submission_id=%s metadata=%s",
                    submission_id, sorted(metadata))
        tracked = bool(metadata.get("user_id"))
        if tracked:
            _store_call("record_submission", {
                "submission_id": submission_id,
                "dashboard_url": dashboard_url,
                "status": "PENDING",
                **metadata,
            })

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
                if tracked:
                    # The job keeps running on Ray; only this backend stopped waiting.
                    _store_call("record_update", submission_id, {
                        "status": _status_name(status),
                        "message": f"Backend stopped waiting after {timeout_s}s; the job may still be running.",
                    })
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

        if tracked:
            _store_call("record_update", submission_id, {
                "status": _status_name(status),   # STOPPED stays STOPPED here, unlike final_status
                "runtime_s": runtime_s,
                "rows_written": _rows_from_logs(logs),
                "ended_at": _now_utc().isoformat(),
            })

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
    job_tags: Optional[Mapping[str, Any]] = None,
) -> RayJobOutcome:
    """
    End-to-end RayJob execution.

    If a dashboard URL is available → DIRECT mode (no kubectl/KubeRay needed).
    Otherwise → KUBERAY mode (kubectl apply).

    ``dashboard_url`` overrides RAY_DASHBOARD_URL for this job only. The caller
    uses it to send transfers that touch cluster-local databases to an in-cluster
    Ray cluster, while cloud-only transfers keep going to the remote/default one.

    ``job_tags`` are merged over any ray_job_context() the caller is inside
    (explicit wins) and tag the job in DIRECT mode. KubeRay mode ignores them.
    """
    tags = {**current_job_tags(), **{k: v for k, v in (job_tags or {}).items() if v not in (None, "")}}
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
                job_tags=tags,
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