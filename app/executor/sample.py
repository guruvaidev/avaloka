"""The validator's sample check, run in a child process instead of ``exec()`` in the server.

``validator.execute_code_node`` checks generated code by calling its
``main()`` on a few sample rows. It did that with ``exec(code, scope)`` inside
the server process: model-written code with the server's environment
(provider keys, ``DB_ENCRYPTION_KEY``, ``MCP_SERVICE_TOKEN``, ...), module
globals and memory. Ported from 2.0's ``app/executor/sample.py``, this runs
the same steps in ``_sample_runner.py`` as a child:

* with ``scrubbed_env`` (no secret), ``python -I``, and a private temp dir as
  ``cwd``, ``HOME`` and ``TMPDIR``;
* with a time limit (``AVALOKA_VALIDATION_TIMEOUT``, defaulting to the local
  executor's ``AVALOKA_LOCAL_EXEC_TIMEOUT``, 60 s); the child runs in its own
  session so a timeout kills everything it started;
* with a memory limit (``AVALOKA_VALIDATION_MEMORY_MB``, default 4096; 0
  turns it off): RLIMIT_AS, enforced on Linux, reported as unsupported on
  macOS.

The dict returned has the shape and messages ``execute_code_node`` returned.
The sample goes in as parquet (a pickle only when parquet cannot hold it:
written here, read by the child, the trusted direction). The result comes
back as parquet or JSON, never as a pickle.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

import pandas as pd

from app.executor.env import scrubbed_env

logger = logging.getLogger(__name__)

TIMEOUT_ENV = "AVALOKA_VALIDATION_TIMEOUT"
MEMORY_ENV = "AVALOKA_VALIDATION_MEMORY_MB"
DEFAULT_MEMORY_MB = 4096

_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sample_runner.py")

#: A 100-row sample needs one core; capping threads also keeps RLIMIT_AS from
#: being spent on per-thread stacks and malloc arenas. A host value wins.
_SAMPLE_THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "MALLOC_ARENA_MAX": "2",
}


def sample_timeout() -> int:
    return int(os.getenv(TIMEOUT_ENV) or os.getenv("AVALOKA_LOCAL_EXEC_TIMEOUT", "60"))


def sample_memory_mb() -> int:
    raw = os.getenv(MEMORY_ENV)
    try:
        return int(raw) if raw else DEFAULT_MEMORY_MB
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r", MEMORY_ENV, raw)
        return DEFAULT_MEMORY_MB


def _sample_env(work: str) -> Dict[str, str]:
    env = scrubbed_env(extra={
        "HOME": work,
        "TMPDIR": work,
        "MPLCONFIGDIR": os.path.join(work, "mpl"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    for name, value in _SAMPLE_THREAD_DEFAULTS.items():
        env.setdefault(name, value)
    return env


def _write_sample(sample_df: pd.DataFrame, work: str) -> None:
    try:
        sample_df.to_parquet(os.path.join(work, "sample.parquet"), index=False)
    except Exception:  # noqa: BLE001 - mixed-type preview columns parquet rejects
        try:
            os.remove(os.path.join(work, "sample.parquet"))
        except OSError:
            pass
        sample_df.to_pickle(os.path.join(work, "sample.pkl"))


def _load_frame(work: str, fmt: Optional[str]) -> Optional[pd.DataFrame]:
    if fmt == "parquet":
        frame = pd.read_parquet(os.path.join(work, "out.parquet"))
        with open(os.path.join(work, "out.columns.json"), encoding="utf-8") as f:
            names = json.load(f)
        if not isinstance(names, list) or len(names) != frame.shape[1]:
            raise ValueError("column names do not match the result frame")
        frame.columns = names
        return frame
    if fmt == "table":
        return pd.read_json(os.path.join(work, "out.json"), orient="table")
    if fmt == "records":
        with open(os.path.join(work, "out.json"), encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    return None


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def run_main_on_sample(code: str, sample_df: pd.DataFrame) -> Dict[str, Any]:
    """``execute_code_node``'s result for ``code`` run on ``sample_df`` in a child."""
    work = tempfile.mkdtemp(prefix="avaloka_sample_")
    try:
        with open(os.path.join(work, "code.py"), "w", encoding="utf-8") as f:
            f.write(code)
        _write_sample(sample_df, work)

        timeout = sample_timeout()
        memory_mb = sample_memory_mb()
        proc = subprocess.Popen(
            [sys.executable, "-I", _RUNNER, work, str(memory_mb)],
            cwd=work,
            env=_sample_env(work),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=(os.name == "posix"),
        )
        try:
            proc_stdout, proc_stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            message = f"Sample execution timed out after {timeout} seconds"
            logger.error(f"❌ Code execution failed: {message}")
            return {"execution_stdout": "", "execution_stderr": "",
                    "execution_error": message, "execution_output_data": None}
        finally:
            # Whatever the generated code left running in its session.
            _kill_group(proc)

        try:
            with open(os.path.join(work, "result.json"), encoding="utf-8") as f:
                res = json.load(f)
            if not isinstance(res, dict):
                raise ValueError("result is not an object")
        except (OSError, ValueError):
            detail = (proc_stderr or "").strip()[-800:] or f"exit status {proc.returncode}"
            if proc.returncode is not None and proc.returncode < 0:
                detail = f"killed by signal {-proc.returncode}; {detail}"
            message = f"Sample execution did not complete: {detail}"
            logger.error(f"❌ Code execution failed: {message}")
            return {"execution_stdout": "", "execution_stderr": proc_stderr or "",
                    "execution_error": message, "execution_output_data": None}

        if proc_stdout:
            logger.debug("Sample child stdout outside main():\n%s", proc_stdout)
        if res.get("memory_limit") not in (None, "on", "off"):
            logger.info("Sample memory limit not applied: %s", res.get("memory_limit"))

        stdout, stderr = _text(res.get("stdout")), _text(res.get("stderr"))
        if res.get("error") is not None:
            error = _text(res.get("error"))
            logger.error(f"❌ Code execution failed: {error}")
            return {"execution_stdout": stdout, "execution_stderr": stderr,
                    "execution_error": error, "execution_output_data": None}

        try:
            output_data = _load_frame(work, res.get("format"))
        except Exception as exc:  # noqa: BLE001 - the child wrote these files
            message = f"Sample execution did not complete: unreadable result ({exc})"
            logger.error(f"❌ Code execution failed: {message}")
            return {"execution_stdout": stdout, "execution_stderr": stderr,
                    "execution_error": message, "execution_output_data": None}
        if output_data is None:
            message = f"main() must return a pandas DataFrame (got {_text(res.get('type'))})."
            logger.error(message)
            return {"execution_stdout": stdout, "execution_stderr": stderr,
                    "execution_error": message, "execution_output_data": None}

        if output_data.empty:
            message = "Execution returned an empty DataFrame."
            logger.warning(message)
            # Graph state must stay serializable: a raw DataFrame here crashes
            # the checkpointer (msgpack) and 500s the whole request.
            return {"execution_stdout": stdout, "execution_stderr": stderr,
                    "execution_error": message, "execution_output_data": [],
                    "execution_output_preview": []}

        logger.info("✅ Code executed successfully.")
        logger.info(f"STDOUT:\n{stdout}")
        if stderr:
            logger.warning(f"STDERR:\n{stderr}")

        # Provide serializable artifacts for downstream consumers.
        preview = output_data.head(2).reset_index(drop=True)
        return {
            "execution_stdout": stdout,
            "execution_stderr": stderr,
            "execution_error": None,
            "execution_output_data": output_data.to_dict(orient="records"),
            "execution_output_preview": preview.to_dict(orient="records"),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
