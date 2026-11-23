"""The code executor: runs one model-written script at a time, with nothing to steal.

    python -m app.executor.server --host 0.0.0.0 --port 8020

It is deliberately the dullest possible server (stdlib ``http.server``, no
framework) because what makes it safe is not in this file but in how it is
deployed, and this file only has to not undo that:

* its network reaches the API and nothing else (compose: an ``internal``
  network shared only with ``api``; Helm: a default-deny-egress
  NetworkPolicy), so there is no database, Redis, MCP server, cloud metadata
  endpoint or internet to connect to;
* its environment carries no secret (compose and Helm give it an explicit,
  short ``environment:``, never the backend's shared env), and each script's
  environment is cut down again to ``app.executor.env``'s allowlist;
* it runs as a non-root user on a read-only root filesystem with every
  capability dropped and no-new-privileges, and has no /state volume: the only
  data it ever holds are the input files sent with the current run.

Per run, under one lock: stage the inputs into the tmpfs roots at their own
absolute paths, run the script in a fresh working directory with
``HOME``/``TMPDIR`` pointing into it, send back the declared outputs, then
kill anything the script left running and wipe every staging root, so the
next run (possibly another user's) starts from nothing.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from app.executor.env import scrubbed_env
from app.executor.protocol import (
    HEALTH_PATH, INPUT_PREFIX, OUTPUT_PREFIX, REQUEST_MEMBER, RESULT_MEMBER, RUN_PATH,
    SCRIPT_MEMBER, member_for, normalized_abs, path_for, under,
)

logger = logging.getLogger("avaloka.executor")

#: Set in the executor's own container only. Enables the steps that are right
#: for a container that exists to run untrusted code and wrong anywhere else:
#: refusing to run as root, killing every other process of this user after a
#: run, and wiping the staging roots.
SANDBOX_ENV = "AVALOKA_EXECUTOR_SANDBOX"
STAGE_ROOTS_ENV = "AVALOKA_EXECUTOR_STAGE_ROOTS"
MAX_REQUEST_ENV = "AVALOKA_EXECUTOR_MAX_REQUEST_BYTES"
MAX_OUTPUT_ENV = "AVALOKA_EXECUTOR_MAX_OUTPUT_BYTES"
MAX_TIMEOUT_ENV = "AVALOKA_EXECUTOR_MAX_TIMEOUT"
_STREAM_CAP = 8 * 1024 * 1024
_CHUNK = 1024 * 1024

_RUN_LOCK = threading.Lock()


class Config:
    def __init__(self, stage_roots: List[str], work_root: str, sandbox: bool,
                 max_request: int, max_output: int, max_timeout: int):
        self.stage_roots = stage_roots
        self.work_root = work_root
        self.sandbox = sandbox
        self.max_request = max_request
        self.max_output = max_output
        self.max_timeout = max_timeout

    @classmethod
    def from_env(cls) -> "Config":
        sandbox = os.getenv(SANDBOX_ENV) == "1"
        roots = [r for r in (os.getenv(STAGE_ROOTS_ENV) or "").split(":") if r.strip()]
        if not roots:
            roots = [tempfile.mkdtemp(prefix="avaloka-exec-stage-")]
        work_root = os.path.join(tempfile.gettempdir(), "avaloka-exec")
        return cls(
            stage_roots=[normalized_abs(r) for r in roots],
            work_root=work_root,
            sandbox=sandbox,
            max_request=int(os.getenv(MAX_REQUEST_ENV, str(8 * 1024 ** 3))),
            max_output=int(os.getenv(MAX_OUTPUT_ENV, str(2 * 1024 ** 3))),
            max_timeout=int(os.getenv(MAX_TIMEOUT_ENV, "3600")),
        )


class _Limited(io.RawIOBase):
    """Reads at most ``n`` bytes from ``raw``: the declared Content-Length."""

    def __init__(self, raw, n: int):
        self._raw, self._left = raw, n

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self._left <= 0:
            return 0
        data = self._raw.read(min(len(b), self._left))
        if not data:
            return 0
        n = len(data)
        b[:n] = data
        self._left -= n
        return n


def _cap(text: str) -> str:
    if len(text) <= _STREAM_CAP:
        return text
    return text[:_STREAM_CAP] + f"\n[... {len(text) - _STREAM_CAP} more characters cut by the executor]"


def _kill_strays() -> None:
    """Kill every process of ours except this server and init, then reap."""
    me, parent = os.getpid(), os.getppid()
    for _ in range(3):
        found = False
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            if pid in (me, parent, 1):
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                found = True
            except (ProcessLookupError, PermissionError):
                pass
        _reap()
        if not found:
            return
        time.sleep(0.05)


def _reap() -> None:
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _wipe(roots: List[str]) -> None:
    for root in roots:
        try:
            entries = os.listdir(root)
        except FileNotFoundError:
            continue
        for name in entries:
            p = os.path.join(root, name)
            try:
                if os.path.isdir(p) and not os.path.islink(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.unlink(p)
            except OSError:
                pass


def stage_request(fileobj, cfg: Config):
    """Unpack a request tar. Returns (request dict, script text, staged paths).

    Inputs land at their own absolute path, which must lie inside a staging
    root; links, devices and anything outside the roots are refused.
    """
    request, script, staged = None, None, []
    with tarfile.open(fileobj=fileobj, mode="r|") as tar:
        for member in tar:
            if member.name == REQUEST_MEMBER and member.isfile():
                request = json.loads(tar.extractfile(member).read().decode("utf-8"))
                continue
            if member.name == SCRIPT_MEMBER and member.isfile():
                script = tar.extractfile(member).read().decode("utf-8")
                continue
            target = path_for(INPUT_PREFIX, member.name)
            if target is None or not under(target, cfg.stage_roots):
                raise ValueError(f"input {member.name!r} is not inside a staging root {cfg.stage_roots}")
            if member.isdir():
                os.makedirs(target, exist_ok=True)
            elif member.isfile():
                os.makedirs(os.path.dirname(target), exist_ok=True)
                src = tar.extractfile(member)
                with open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, _CHUNK)
                os.chmod(target, 0o444)
                staged.append(target)
            else:
                raise ValueError(f"input {member.name!r} is not a regular file or directory")
    if request is None or script is None:
        raise ValueError("request is missing request.json or script.py")
    return request, script, staged


def run_staged(request: dict, script: str, cfg: Config) -> dict:
    """Run the script; returns the result dict plus the outputs to send back."""
    timeout = max(1, min(int(request.get("timeout") or 60), cfg.max_timeout))
    outputs = []
    for raw in request.get("outputs") or []:
        p = normalized_abs(str(raw))
        if not under(p, cfg.stage_roots):
            raise ValueError(f"output {p!r} is not inside a staging root {cfg.stage_roots}")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        outputs.append(p)

    os.makedirs(cfg.work_root, exist_ok=True)
    work = tempfile.mkdtemp(prefix="run-", dir=cfg.work_root)
    script_path = os.path.join(work, "avaloka_exec_script.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    env = scrubbed_env(extra={
        "HOME": work, "TMPDIR": work, "MPLCONFIGDIR": os.path.join(work, ".mpl"),
        "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
    })
    proc = subprocess.Popen(
        [sys.executable, script_path], cwd=work, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
    result = {
        "returncode": None if timed_out else proc.returncode,
        "stdout": _cap((out or b"").decode("utf-8", errors="replace")),
        "stderr": _cap((err or b"").decode("utf-8", errors="replace")),
        "timed_out": timed_out,
        "error": f"timed out after {timeout} seconds" if timed_out else None,
    }
    return {"result": result, "outputs": outputs, "work": work}


def write_reply(fileobj, result: dict, outputs: List[str], max_output: int) -> None:
    with tarfile.open(fileobj=fileobj, mode="w|") as tar:
        data = json.dumps(result).encode("utf-8")
        info = tarfile.TarInfo(RESULT_MEMBER)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        for p in outputs:
            try:
                fd = os.open(p, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            except OSError:
                continue                      # not written, or a link: not sent
            with os.fdopen(fd, "rb") as f:
                st = os.fstat(f.fileno())
                if not stat.S_ISREG(st.st_mode) or st.st_size > max_output:
                    continue
                info = tarfile.TarInfo(member_for(OUTPUT_PREFIX, p))
                info.size = st.st_size
                info.mode = 0o644
                tar.addfile(info, f)


def handle_run(body, length: int, cfg: Config, reply) -> None:
    """One run, start to finish, under the lock. ``reply`` receives the tar."""
    with _RUN_LOCK:
        work = None
        try:
            request, script, _ = stage_request(_Limited(body, length), cfg)
            ran = run_staged(request, script, cfg)
            work = ran["work"]
            write_reply(reply, ran["result"], ran["outputs"], cfg.max_output)
        finally:
            if cfg.sandbox:
                _kill_strays()
                _wipe(cfg.stage_roots + [cfg.work_root])
            elif work:
                shutil.rmtree(work, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    cfg: Config = None  # set by serve()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet access log; errors are logged below
        logger.debug("%s " + fmt, self.address_string(), *args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == HEALTH_PATH:
            self._send(200, b'{"status":"ok"}', "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != RUN_PATH:
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length") or "-1")
        except ValueError:
            length = -1
        if length < 0:
            self._send(411, b"Content-Length required", "text/plain")
            return
        if length > self.cfg.max_request:
            self._send(413, b"request larger than the executor accepts", "text/plain")
            return
        started = time.monotonic()
        with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024) as reply:
            try:
                handle_run(self.rfile, length, self.cfg, reply)
            except ValueError as exc:
                logger.warning("run refused: %s", exc)
                self.close_connection = True
                self._send(400, str(exc).encode("utf-8"), "text/plain")
                return
            except Exception as exc:  # pragma: no cover - logged, reported
                logger.exception("run failed")
                self.close_connection = True
                self._send(500, f"{type(exc).__name__}: {exc}".encode("utf-8"), "text/plain")
                return
            size = reply.tell()
            reply.seek(0)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-tar")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            shutil.copyfileobj(reply, self.wfile, _CHUNK)
        logger.info("run done in %.1fs", time.monotonic() - started)


def serve(host: str, port: int, cfg: Optional[Config] = None) -> None:
    cfg = cfg or Config.from_env()
    if cfg.sandbox and os.geteuid() == 0:
        raise SystemExit("the code executor refuses to run as root; set a non-root user")
    if cfg.sandbox:
        _wipe(cfg.stage_roots + [cfg.work_root])
    Handler.cfg = cfg
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    logger.info("code executor on %s:%s, staging roots %s, sandbox=%s",
                host, port, cfg.stage_roots, cfg.sandbox)
    httpd.serve_forever()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8020)
    args = ap.parse_args(argv)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
