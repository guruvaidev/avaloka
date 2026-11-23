"""Child side of ``app.executor.execute_service``: one ``/execute`` request's code.

Run as ``python -I _execute_runner.py <workdir> <memory_mb>`` with the
scrubbed environment, in ``<workdir>``. Like ``_sample_runner.py`` it imports
nothing from ``app``, so the code reaches no server module or object.

It gives the code the scope the old Flask service gave it (``pd``, ``np``,
``os``, ``sys``, ``Path``, ``input_file``, ``output_file``, ``result`` and,
when a CSV was sent, ``input_df``), captures stdout/stderr around the exec,
and writes what happened to ``<workdir>/result.json``: never a pickle.
"""

import io
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

# The address-space limit is _sample_runner's, not a copy of it. -I keeps this
# directory off sys.path, so add it for that one import and take it off again.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from _sample_runner import _limit_memory
finally:
    sys.path.pop(0)


def _jsonable(value):
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return json.loads(value.to_json(orient="records", date_format="iso", default_handler=str))
    except ImportError:  # pragma: no cover - pandas is in the image
        pass
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def run(work, memory_mb):
    res = {"error": None, "traceback": None, "stdout": "", "stderr": "", "result": None,
           "output_csv": None, "memory_limit": _limit_memory(memory_mb)}
    input_file = os.path.join(work, "input.csv")
    output_file = os.path.join(work, "output.csv")
    os.environ["INPUT_FILE"] = input_file
    os.environ["OUTPUT_FILE"] = output_file
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    try:
        from pathlib import Path

        import numpy as np
        import pandas as pd

        with open(os.path.join(work, "code.py"), encoding="utf-8") as f:
            code = f.read()
        scope = {"pd": pd, "np": np, "os": os, "sys": sys, "Path": Path,
                 "input_file": input_file, "output_file": output_file, "result": None,
                 "__name__": "avaloka_execute"}
        if os.path.exists(input_file):
            scope["input_df"] = pd.read_csv(input_file)
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compile(code, "<string>", "exec"), scope)
        if scope.get("result") is not None:
            res["result"] = _jsonable(scope["result"])
        if os.path.exists(output_file):
            out = pd.read_csv(output_file)
            if "timestamp" in out.columns:
                out = out.drop(columns=["timestamp"])
            res["output_csv"] = out.to_csv(index=False)
    except MemoryError as exc:
        res["error"] = str(exc) or f"MemoryError: exceeded the {memory_mb} MB memory limit"
        res["traceback"] = traceback.format_exc()
    except BaseException as exc:  # SystemExit from the code is a failure too
        res["error"] = str(exc) or type(exc).__name__
        res["traceback"] = traceback.format_exc()
    res["stdout"] = stdout_capture.getvalue()
    res["stderr"] = stderr_capture.getvalue()
    with open(os.path.join(work, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, default=str)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
