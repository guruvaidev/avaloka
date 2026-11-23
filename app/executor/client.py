"""Run a model-written script: in the isolated executor when one is configured.

Callers (``execution_agent.execute_code_on_local``, the DTA sample preview,
the validator's sample check) build the script exactly as before and hand it
here with the files it reads and the file it writes. See ``app.executor``.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
from typing import Iterable, Mapping, Optional

from app.executor.env import scrubbed_env
from app.executor.protocol import (
    INPUT_PREFIX, OUTPUT_PREFIX, REQUEST_MEMBER, RESULT_MEMBER, RUN_PATH, SCRIPT_MEMBER,
    ScriptResult, member_for, normalized_abs, path_for,
)

logger = logging.getLogger(__name__)

EXECUTOR_URL_ENV = "AVALOKA_EXECUTOR_URL"
_SPOOL_BYTES = 64 * 1024 * 1024
_CHUNK = 1024 * 1024


def executor_url() -> Optional[str]:
    url = (os.getenv(EXECUTOR_URL_ENV) or "").strip()
    return url.rstrip("/") or None


def executor_configured() -> bool:
    return executor_url() is not None


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _local_inputs(inputs: Iterable[str]) -> list:
    """The inputs that are local files or directories, absolute and de-duplicated."""
    seen, out = set(), []
    for raw in inputs or ():
        if not raw:
            continue
        p = str(raw)
        if "://" in p:
            continue                      # a cloud URI is read by the script itself
        p = os.path.abspath(p)
        if p in seen or not os.path.exists(p):
            continue                      # a missing input fails in the script, as before
        seen.add(p)
        out.append(p)
    return out


def run_script(script: str, *, timeout: int, inputs: Iterable[str] = (),
               outputs: Iterable[str] = (), extra_env: Optional[Mapping[str, str]] = None,
               cwd: Optional[str] = None) -> ScriptResult:
    """Run ``script`` and return what it printed and how it exited.

    ``inputs`` are the files/directories the script reads and ``outputs`` the
    files it may write; with an executor configured only those cross the
    boundary. ``extra_env`` is filtered by the same allowlist as the rest.
    """
    url = executor_url()
    if url:
        return _run_remote(url, script, timeout=timeout, inputs=inputs, outputs=outputs)
    return _run_local(script, timeout=timeout, extra_env=extra_env, cwd=cwd)


def _run_local(script: str, *, timeout: int, extra_env=None, cwd=None) -> ScriptResult:
    fd, path = tempfile.mkstemp(suffix=".py", prefix="avaloka_exec_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(script)
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False, timeout=timeout, cwd=cwd,
            env=scrubbed_env(extra=extra_env),
        )
        return ScriptResult(returncode=proc.returncode, stdout=proc.stdout or "",
                            stderr=proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        return ScriptResult(returncode=None, stdout=_text(e.stdout), stderr=_text(e.stderr),
                            timed_out=True, error=f"timed out after {timeout} seconds")
    except FileNotFoundError:
        return ScriptResult(returncode=None, error=f"Python executable not found at {sys.executable}")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def build_request(script: str, *, timeout: int, inputs: Iterable[str], outputs: Iterable[str],
                  fileobj) -> list:
    """Write the request tar to ``fileobj``; returns the normalized output paths."""
    outs = [normalized_abs(os.path.abspath(str(o))) for o in outputs or () if o]
    with tarfile.open(fileobj=fileobj, mode="w|") as tar:
        def _add_bytes(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(data))

        _add_bytes(REQUEST_MEMBER, json.dumps({"timeout": int(timeout), "outputs": outs}).encode())
        _add_bytes(SCRIPT_MEMBER, script.encode("utf-8"))
        for p in _local_inputs(inputs):
            tar.add(p, arcname=member_for(INPUT_PREFIX, p), recursive=True)
    return outs


def _run_remote(url: str, script: str, *, timeout: int, inputs, outputs) -> ScriptResult:
    import httpx

    with tempfile.SpooledTemporaryFile(max_size=_SPOOL_BYTES) as body:
        outs = build_request(script, timeout=timeout, inputs=inputs, outputs=outputs, fileobj=body)
        size = body.tell()
        body.seek(0)

        def _chunks():
            while True:
                chunk = body.read(_CHUNK)
                if not chunk:
                    return
                yield chunk

        # The executor runs one script at a time, so a request can wait behind
        # another; the read timeout allows for that and for the run itself.
        http_timeout = httpx.Timeout(connect=10.0, read=float(timeout) * 3 + 60.0,
                                     write=120.0, pool=10.0)
        try:
            with httpx.Client(timeout=http_timeout, trust_env=False) as client:
                with client.stream("POST", url + RUN_PATH, content=_chunks(),
                                   headers={"Content-Type": "application/x-tar",
                                            "Content-Length": str(size)}) as resp:
                    if resp.status_code != 200:
                        detail = resp.read().decode("utf-8", errors="replace")[:500]
                        return ScriptResult(returncode=None,
                                            error=f"code executor refused the run (HTTP {resp.status_code}): {detail}")
                    with tempfile.SpooledTemporaryFile(max_size=_SPOOL_BYTES) as reply:
                        for chunk in resp.iter_bytes():
                            reply.write(chunk)
                        reply.seek(0)
                        return read_reply(reply, outs)
        except httpx.HTTPError as exc:
            # Never fall back to running here: a configured executor that is
            # down must not quietly put generated code back in the API process.
            return ScriptResult(returncode=None,
                                error=f"code executor unreachable at {url}: {type(exc).__name__}: {exc}")


def read_reply(fileobj, expected_outputs: Iterable[str]) -> ScriptResult:
    """Parse a reply tar; write each expected output file to its path.

    The reply comes from where untrusted code ran, so only regular-file
    members naming a declared output are written, each to exactly that path.
    """
    expected = set(expected_outputs)
    result: Optional[ScriptResult] = None
    written = []
    with tarfile.open(fileobj=fileobj, mode="r|") as tar:
        for member in tar:
            if member.name == RESULT_MEMBER and member.isfile():
                data = json.loads(tar.extractfile(member).read().decode("utf-8"))
                result = ScriptResult(
                    returncode=data.get("returncode"),
                    stdout=_text(data.get("stdout")),
                    stderr=_text(data.get("stderr")),
                    timed_out=bool(data.get("timed_out")),
                    error=data.get("error"),
                )
                continue
            target = path_for(OUTPUT_PREFIX, member.name)
            if not member.isfile() or target is None or target not in expected:
                continue
            os.makedirs(os.path.dirname(target) or "/", exist_ok=True)
            tmp = f"{target}.part"
            src = tar.extractfile(member)
            with open(tmp, "wb") as dst:
                while True:
                    chunk = src.read(_CHUNK)
                    if not chunk:
                        break
                    dst.write(chunk)
            os.replace(tmp, target)
            written.append(target)
    if result is None:
        return ScriptResult(returncode=None, error="code executor reply had no result")
    result.outputs = written
    return result
