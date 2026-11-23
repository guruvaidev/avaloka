"""The ``/execute`` service: runs a script the k8s execution path sends, with nothing to steal.

    python -m app.executor.execute_service --host 0.0.0.0 --port 8080

``execution_agent.execute_code_on_k8s`` (the ``execute_on_k8s`` graph node and
Celery's ``deploy_on_k8s`` branch) POSTs model-written code here. The service
it replaces (``app/infra/python_app``, Flask) ran that code with ``exec()`` in
its own process, unauthenticated, behind a ``type: LoadBalancer`` Service, with
the namespace's default ServiceAccount token mounted: anyone who could reach
the load balancer could run code in the cluster with that token.

The request and response bodies are unchanged (see :func:`run_request`). What
changed is everything around them:

* **Auth.** Every POST needs ``Authorization: Bearer <EXECUTE_SERVICE_TOKEN>``,
  compared in constant time. A missing or wrong token gets 401 before a byte
  of the body is read. The service refuses to start without a token.
* **Process.** The code runs in a child (``_execute_runner.py``, ``python -I``)
  with ``app.executor.env.scrubbed_env`` (#662's allowlist), a private
  per-request workspace as ``cwd``/``HOME``/``TMPDIR`` that is deleted
  afterwards, a time limit and an address-space limit, in its own session so a
  timeout kills everything it started. This server marks itself non-dumpable
  on Linux, so the child (same uid, no capabilities) cannot read this
  process's ``/proc/<pid>/environ`` or memory.
* **Deployment** (``deploy/helm/avaloka/templates/execute-service.yaml``):
  ClusterIP only, a NetworkPolicy that admits the API, LangGraph and Celery
  worker pods and denies all egress, its own ServiceAccount with no RBAC and
  no token mounted, non-root, read-only root filesystem, every capability
  dropped, seccomp RuntimeDefault, and no server secret in its environment.

What the token does not protect: the child runs as the same uid as this
server, so it can read the mounted token file. That token authorises calling
this service and nothing else, which code already running here can do anyway.
That is why it is a dedicated token and not ``MCP_SERVICE_TOKEN``.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

from app.executor.env import scrubbed_env
from app.executor.sample import DEFAULT_MEMORY_MB

logger = logging.getLogger("avaloka.execute_service")

#: The shared secret between the callers and this service.
TOKEN_ENV = "EXECUTE_SERVICE_TOKEN"
#: Where the chart mounts it instead, so it is not in this pod's environment.
TOKEN_FILE_ENV = "EXECUTE_SERVICE_TOKEN_FILE"
#: Where the callers find this service (the chart's ConfigMap sets it).
URL_ENV = "AVALOKA_EXECUTE_SERVICE_URL"
TIMEOUT_ENV = "AVALOKA_EXECUTE_TIMEOUT"
MEMORY_ENV = "AVALOKA_EXECUTE_MEMORY_MB"
MAX_REQUEST_ENV = "AVALOKA_EXECUTE_MAX_REQUEST_BYTES"

EXECUTE_PATH = "/execute"
HEALTH_PATH = "/health"
#: Below the caller's 300 s request timeout, so the caller hears why.
DEFAULT_TIMEOUT = 240
DEFAULT_MAX_REQUEST = 256 * 1024 * 1024
#: A token shorter than this is a placeholder, not a secret.
MIN_TOKEN_LENGTH = 16

_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_execute_runner.py")
_RUN_LOCK = threading.Lock()


# --------------------------------------------------------------------------- callers' side

def configured_url() -> Optional[str]:
    """The service URL the deployment configured, or None."""
    url = (os.getenv(URL_ENV) or "").strip().rstrip("/")
    return url or None


def configured_host_port() -> Optional[Tuple[str, str]]:
    """(host, port) of :func:`configured_url`, or None when unset or unusable."""
    url = configured_url()
    if not url:
        return None
    parts = urlsplit(url if "://" in url else f"http://{url}")
    if not parts.hostname:
        return None
    return parts.hostname, str(parts.port or 80)


def auth_headers() -> Dict[str, str]:
    """The header a caller sends; empty when no token is configured (the service then answers 401)."""
    token = (os.getenv(TOKEN_ENV) or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


# --------------------------------------------------------------------------- service side

def load_token(environ=None) -> str:
    """The token from ``EXECUTE_SERVICE_TOKEN_FILE`` or ``EXECUTE_SERVICE_TOKEN``; raises if absent."""
    env = os.environ if environ is None else environ
    token = ""
    path = (env.get(TOKEN_FILE_ENV) or "").strip()
    if path:
        with open(path, encoding="utf-8") as f:
            token = f.read().strip()
    if not token:
        token = (env.get(TOKEN_ENV) or "").strip()
    if len(token) < MIN_TOKEN_LENGTH:
        raise SystemExit(
            f"the execute service refuses to start without {TOKEN_ENV} (or {TOKEN_FILE_ENV}) "
            f"of at least {MIN_TOKEN_LENGTH} characters; the Helm chart generates one"
        )
    return token


def execute_timeout() -> int:
    try:
        return max(1, int(os.getenv(TIMEOUT_ENV) or DEFAULT_TIMEOUT))
    except ValueError:
        return DEFAULT_TIMEOUT


def execute_memory_mb() -> int:
    try:
        return int(os.getenv(MEMORY_ENV) or DEFAULT_MEMORY_MB)
    except ValueError:
        return DEFAULT_MEMORY_MB


def _not_dumpable() -> None:
    """PR_SET_DUMPABLE=0: /proc/<this pid> becomes root-owned, so the child cannot read our environ."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
            logger.warning("prctl(PR_SET_DUMPABLE, 0) failed: errno %s", ctypes.get_errno())
    except OSError as exc:  # pragma: no cover - no libc
        logger.warning("could not mark the execute service non-dumpable: %s", exc)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _child_env(work: str) -> Dict[str, str]:
    return scrubbed_env(extra={
        "HOME": work,
        "TMPDIR": work,
        "MPLCONFIGDIR": os.path.join(work, ".mpl"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    })


def _b64(field: str, value: Any) -> str:
    try:
        return base64.b64decode(value).decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - reported to the caller as a 400
        raise ValueError(f"Invalid base64 for {field}: {exc}") from exc


def run_request(data: Any) -> Tuple[int, Dict[str, Any]]:
    """One ``/execute`` request body -> (HTTP status, response body).

    Request: ``{"code": base64 python, "data": optional base64 CSV}``. The code
    sees ``pd``, ``np``, ``os``, ``sys``, ``Path``, ``input_file``,
    ``output_file``, ``result`` and, with data, ``input_df``. Response on
    success: ``status``, ``stdout``, ``stderr``, and ``result``/``output``/
    ``output_file`` when the code set ``result`` or wrote ``output_file``.
    A failure in the code is a 500 with ``error``/``traceback``/``stdout``/``stderr``.
    """
    if not isinstance(data, dict):
        return 400, {"error": "Request must be a JSON object"}
    if not data.get("code"):
        return 400, {"error": "Missing 'code' field"}
    try:
        code = _b64("code", data["code"])
        csv_text = _b64("data", data["data"]) if data.get("data") else None
    except ValueError as exc:
        return 400, {"error": str(exc)}

    execution_id = uuid.uuid4().hex[:8]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = tempfile.mkdtemp(prefix=f"execute_{execution_id}_")
    try:
        with open(os.path.join(work, "code.py"), "w", encoding="utf-8") as f:
            f.write(code)
        if csv_text is not None:
            with open(os.path.join(work, "input.csv"), "w", encoding="utf-8") as f:
                f.write(csv_text)

        timeout = execute_timeout()
        proc = subprocess.Popen(
            [sys.executable, "-I", _RUNNER, work, str(execute_memory_mb())],
            cwd=work,
            env=_child_env(work),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            _, proc_stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return 500, {"error": f"Code execution failed: timed out after {timeout} seconds",
                         "stdout": "", "stderr": ""}
        finally:
            _kill_group(proc)  # whatever the code left running in its session

        try:
            with open(os.path.join(work, "result.json"), encoding="utf-8") as f:
                res = json.load(f)
            if not isinstance(res, dict):
                raise ValueError("result is not an object")
        except (OSError, ValueError):
            detail = (proc_stderr or "").strip()[-800:] or f"exit status {proc.returncode}"
            if proc.returncode is not None and proc.returncode < 0:
                detail = f"killed by signal {-proc.returncode}; {detail}"
            return 500, {"error": f"Code execution failed: {detail}", "stdout": "", "stderr": ""}

        if res.get("error"):
            return 500, {"error": f"Code execution failed: {res['error']}",
                         "traceback": res.get("traceback") or "",
                         "stdout": res.get("stdout") or "", "stderr": res.get("stderr") or ""}

        response: Dict[str, Any] = {"status": "success",
                                    "stdout": res.get("stdout") or "",
                                    "stderr": res.get("stderr") or ""}
        if res.get("result") is not None:
            response["result"] = res["result"]
        if res.get("output_csv") is not None:
            csv_out = res["output_csv"]
            output = "data:text/csv;base64," + base64.b64encode(csv_out.encode("utf-8")).decode()
            response["output"] = output
            response["output_file"] = {"filename": f"etl_results_{execution_id}_{stamp}.csv",
                                       "content": output, "size": len(csv_out)}
        return 200, response
    finally:
        shutil.rmtree(work, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    token: str = ""            # set by serve()
    max_request: int = DEFAULT_MAX_REQUEST
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet access log
        logger.debug("%s " + fmt, self.address_string(), *args)

    def _send_json(self, code: int, body: Dict[str, Any], close: bool = False) -> None:
        raw = json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization") or ""
        scheme, _, given = header.partition(" ")
        if scheme.lower() != "bearer" or not given:
            return False
        return hmac.compare_digest(given.strip().encode("utf-8"), self.token.encode("utf-8"))

    def do_GET(self):
        if self.path == HEALTH_PATH:
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        # Before the body: an unauthenticated caller gets nothing read, parsed or run.
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"}, close=True)
            return
        if self.path != EXECUTE_PATH:
            self._send_json(404, {"error": "not found"}, close=True)
            return
        try:
            length = int(self.headers.get("Content-Length") or "-1")
        except ValueError:
            length = -1
        if length < 0:
            self._send_json(411, {"error": "Content-Length required"}, close=True)
            return
        if length > self.max_request:
            self._send_json(413, {"error": "request larger than the execute service accepts"}, close=True)
            return
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._send_json(400, {"error": "Request must be JSON"})
            return
        started = time.monotonic()
        with _RUN_LOCK:
            try:
                status, body = run_request(data)
            except Exception as exc:  # pragma: no cover - logged, reported
                logger.exception("execute failed")
                status, body = 500, {"error": f"Unexpected error: {type(exc).__name__}"}
        logger.info("execute -> %s in %.1fs", status, time.monotonic() - started)
        self._send_json(status, body)


def serve(host: str, port: int) -> None:
    Handler.token = load_token()
    os.environ.pop(TOKEN_ENV, None)
    try:
        Handler.max_request = int(os.getenv(MAX_REQUEST_ENV) or DEFAULT_MAX_REQUEST)
    except ValueError:
        Handler.max_request = DEFAULT_MAX_REQUEST
    _not_dumpable()
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    logger.info("execute service on %s:%s", host, port)
    httpd.serve_forever()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args(argv)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
