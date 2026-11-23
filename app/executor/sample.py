"""The validator's sample check, run outside the API process instead of ``exec()`` in it.

``validator.execute_code_node`` checks generated code by calling its
``main()`` on a few sample rows. It did that with ``exec(code, scope)`` inside
the API process: model-written code with the API's memory, environment and
network, able to read every secret in ``os.environ`` and every module global.
The same steps now run in a script that does exactly what the in-process path
did (one exec scope with ``pd``, ``__name__`` not ``"__main__"`` so the
script's entry block stays inert, ``bind_main_frames`` for a join, stdout and
stderr captured around the ``main()`` call only), and the dict returned here
has the shape and messages ``execute_code_node`` returned.

Where the script runs:

* with an executor configured (``AVALOKA_EXECUTOR_URL``), in the executor,
  through ``app.executor.client.run_script`` -- the sample and the join's
  source files are the only inputs that cross, the result files the only
  outputs; a configured executor that is down is an error, never a fallback;
* otherwise as a child of this process: ``python -I`` (the repository is not
  on its ``sys.path``, so the generated code reaches no server module),
  ``scrubbed_env`` (no secret), a private temp dir as ``cwd``, ``HOME`` and
  ``TMPDIR``, a time limit (``AVALOKA_VALIDATION_TIMEOUT``, defaulting to the
  local executor's ``AVALOKA_LOCAL_EXEC_TIMEOUT``, 60 s) in its own session so
  a timeout kills everything it started, and a memory limit
  (``AVALOKA_VALIDATION_MEMORY_MB``, default 4096; 0 turns it off): RLIMIT_AS,
  enforced on Linux, reported as unsupported on macOS.

The sample goes in as parquet (a pickle only when parquet cannot hold it:
written here, read by the child, the trusted direction). The result comes
back as parquet with the column names alongside (a pivot's ints, duplicates),
or as JSON when parquet cannot carry it -- never as a pickle, which would
execute whatever the script chose when the API loaded it. The child-side
helpers (``_limit_memory``, ``_write_frame``) are ``_sample_runner.py``'s,
inlined into the script so the executor needs no path to this package.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from app.executor.client import executor_configured, run_script
from app.executor.env import scrubbed_env

logger = logging.getLogger(__name__)

TIMEOUT_ENV = "AVALOKA_VALIDATION_TIMEOUT"
#: 2.x's earlier name for the same limit; still honoured.
SAMPLE_TIMEOUT_ENV = "AVALOKA_SAMPLE_EXEC_TIMEOUT"
MEMORY_ENV = "AVALOKA_VALIDATION_MEMORY_MB"
#: Address-space limit (0 turns it off; Linux enforces it). The executor
#: service applies the same default per request (execute_service).
DEFAULT_MEMORY_MB = 4096

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

_RESULT_FILES = ("result.json", "out.parquet", "out.columns.json", "out.json")


def sample_timeout() -> int:
    for name in (TIMEOUT_ENV, SAMPLE_TIMEOUT_ENV, "AVALOKA_LOCAL_EXEC_TIMEOUT"):
        raw = os.getenv(name)
        if raw:
            return int(raw)
    return 60


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


def _write_sample(sample_df: pd.DataFrame, work: str) -> str:
    """The sample as parquet, or as a pickle when parquet rejects it; returns the path."""
    path = os.path.join(work, "sample.parquet")
    try:
        sample_df.to_parquet(path, index=False)
        return path
    except Exception:  # noqa: BLE001 - mixed-type preview columns parquet rejects
        try:
            os.remove(path)
        except OSError:
            pass
        path = os.path.join(work, "sample.pkl")
        sample_df.to_pickle(path)
        return path


def _sources_text() -> str:
    """The modules and helpers the script needs, as source: it imports nothing from ``app``."""
    from file_handler import null_policy
    from app.agents import main_frames
    from app.executor import _sample_runner

    helpers = (_sample_runner._has_nested_values, _sample_runner._plain_column_names,
               _sample_runner._write_frame)
    return "\n\n".join([inspect.getsource(null_policy), inspect.getsource(main_frames),
                        *(inspect.getsource(fn) for fn in helpers)])


# The memory limit goes on before pandas is imported (the inlined modules
# import it), then the sample check proper.
_HEAD = r'''
import io as _io, json as _json, os as _os, sys as _sys
import json, os   # the inlined _sample_runner helpers use these names
from contextlib import redirect_stdout as _rso, redirect_stderr as _rse

__LIMIT_MEMORY__

_MEMORY_LIMIT = _limit_memory(__MEMORY_MB__)
'''

_DRIVER = r'''

_res = {"error": None, "type": None, "stdout": "", "stderr": "", "format": None,
        "memory_limit": _MEMORY_LIMIT}
_so, _se = _io.StringIO(), _io.StringIO()
try:
    if __SAMPLE__.endswith(".parquet"):
        _sample = pd.read_parquet(__SAMPLE__)
    else:
        # Written by the server: the trusted direction for a pickle.
        _sample = pd.read_pickle(__SAMPLE__)
    # The names the full-run script is given besides its own imports
    # (app/agents/execution_agent.py's prelude): ``np``, and the
    # ``avaloka_result()`` helper the coder is told is pre-injected. On the
    # sample it records nothing; without it every scalar answer the coder
    # wrote as instructed failed validation with NameError.
    import numpy as _np
    def _avaloka_result_noop(value, kind=None, columns=None, **kw):
        return None
    _scope = {"pd": pd, "np": _np, "avaloka_result": _avaloka_result_noop,
              "__name__": "avaloka_sample_execution"}
    exec(compile(__CODE__, "<string>", "exec"), _scope)
    _main = _scope.get("main")
    if not callable(_main):
        raise NameError("main() function not found in the generated code.")
    _args = bind_main_frames(
        _main, _sample.copy(), __SOURCES__,
        lambda path: read_csv_best_effort(str(path), nrows=__NROWS__),
        primary_source=__PRIMARY__)
    with _rso(_so), _rse(_se):
        _out = _main(*_args)
    _res["type"] = type(_out).__name__
    if isinstance(_out, pd.DataFrame):
        _res["format"] = _write_frame(_out, __WORK__)
except MemoryError as _e:
    _res["error"] = str(_e) or (
        "MemoryError: sample execution exceeded the __MEMORY_MB__ MB memory limit")
except Exception as _e:
    _res["error"] = str(_e)
_res["stdout"], _res["stderr"] = _so.getvalue(), _se.getvalue()
with open(_os.path.join(__WORK__, "result.json"), "w", encoding="utf-8") as _f:
    _json.dump(_res, _f, default=str)
'''


def _script(code: str, work: str, sample_path: str, sources: Dict[str, str],
            primary_source: Optional[str], nrows: int, memory_mb: int) -> str:
    from app.executor import _sample_runner

    head = (_HEAD.replace("__LIMIT_MEMORY__", inspect.getsource(_sample_runner._limit_memory))
            .replace("__MEMORY_MB__", repr(int(memory_mb))))
    driver = (_DRIVER.replace("__SAMPLE__", repr(sample_path))
              .replace("__CODE__", repr(code))
              .replace("__SOURCES__", repr(sources))
              .replace("__NROWS__", repr(int(nrows)))
              .replace("__PRIMARY__", repr(str(primary_source) if primary_source else None))
              .replace("__WORK__", repr(work))
              .replace("__MEMORY_MB__", str(int(memory_mb))))
    return head + "\n\n" + _sources_text() + "\n\n" + driver


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


def _failed(message: str, stdout: str = "", stderr: str = "") -> Dict[str, Any]:
    logger.error(f"❌ Code execution failed: {message}")
    return {"execution_stdout": stdout, "execution_stderr": stderr,
            "execution_error": message, "execution_output_data": None}


def _run_in_child(script: str, work: str, timeout: int):
    """Run the script as a child of this process. Returns ``(returncode, stderr)``, or a result dict on failure."""
    runner = os.path.join(work, "runner.py")
    with open(runner, "w", encoding="utf-8") as f:
        f.write(script)
    proc = subprocess.Popen(
        [sys.executable, "-I", runner],
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
        return _failed(f"Sample execution timed out after {timeout} seconds")
    finally:
        # Whatever the generated code left running in its session.
        _kill_group(proc)
    if proc_stdout:
        logger.debug("Sample child stdout outside main():\n%s", proc_stdout)
    return proc.returncode, proc_stderr or ""


def run_main_on_sample(code: str, sample_df: pd.DataFrame, sources: Optional[Mapping[str, Any]] = None,
                       primary_source: Optional[str] = None, nrows: int = 5000) -> Dict[str, Any]:
    """``execute_code_node``'s result for ``code`` run on ``sample_df`` outside this process."""
    work = tempfile.mkdtemp(prefix="avaloka_sample_")
    try:
        sample_path = _write_sample(sample_df, work)
        clean_sources = {str(k): str(v) for k, v in (sources or {}).items() if v}
        timeout = sample_timeout()
        memory_mb = sample_memory_mb()
        script = _script(code, work, sample_path, clean_sources, primary_source, nrows, memory_mb)

        if executor_configured():
            ran = run_script(
                script, timeout=timeout,
                inputs=[sample_path, *clean_sources.values()],
                outputs=[os.path.join(work, name) for name in _RESULT_FILES],
            )
            returncode, proc_stderr = ran.returncode, ran.stderr or ""
            detail_hint = ran.error
        else:
            child = _run_in_child(script, work, timeout)
            if isinstance(child, dict):
                return child
            returncode, proc_stderr = child
            detail_hint = None

        try:
            with open(os.path.join(work, "result.json"), encoding="utf-8") as f:
                res = json.load(f)
            if not isinstance(res, dict):
                raise ValueError("result is not an object")
        except (OSError, ValueError):
            detail = detail_hint or proc_stderr.strip()[-800:] or f"exit status {returncode}"
            if returncode is not None and returncode < 0:
                detail = f"killed by signal {-returncode}; {detail}"
            return _failed(f"Sample execution did not complete: {detail}", stderr=proc_stderr)

        if res.get("memory_limit") not in (None, "on", "off"):
            logger.info("Sample memory limit not applied: %s", res.get("memory_limit"))

        stdout, stderr = _text(res.get("stdout")), _text(res.get("stderr"))
        if res.get("error") is not None:
            return _failed(_text(res.get("error")), stdout, stderr)

        try:
            output_data = _load_frame(work, res.get("format"))
        except Exception as exc:  # noqa: BLE001 - the child wrote these files
            return _failed(f"Sample execution did not complete: unreadable result ({exc})", stdout, stderr)
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
