"""Child side of ``app.executor.sample``: the validator's sample check.

Run as ``python -I _sample_runner.py <workdir> <memory_mb>`` with the
scrubbed environment, in ``<workdir>``. It imports nothing from ``app`` (with
``-I`` the repository is not even on ``sys.path``), so the generated code
reaches no server module, global or object.

It repeats what ``validator.execute_code_node`` did in process: one exec scope
holding ``pd`` with ``__name__`` not ``"__main__"`` (so the script's entry
block stays inert), ``main`` looked up in it, and stdout/stderr captured
around the ``main()`` call only. What it found is written to
``<workdir>/result.json``; the frame ``main()`` returned goes to
``out.parquet`` (names in ``out.columns.json``) or, when parquet cannot
carry it faithfully, ``out.json``.
Never a pickle: the parent would execute whatever the script put in one.
"""

import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout


def _limit_memory(memory_mb):
    """Cap this process's address space before any generated code runs.

    Set here rather than in a ``preexec_fn``: the server is multithreaded and
    ``preexec_fn`` is not fork-safe there. The hard limit equals the soft one,
    so the generated code cannot raise it again. macOS does not enforce
    RLIMIT_AS (setrlimit raises or is ignored), so there it is reported, not
    applied.
    """
    if memory_mb <= 0:
        return "off"
    try:
        import resource

        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        return "on"
    except (ImportError, ValueError, OSError) as exc:
        return f"unsupported: {exc}"


def _has_nested_values(frame):
    """True if an object column holds lists/dicts, which parquet turns into arrays."""
    for position, dtype in enumerate(frame.dtypes):
        if dtype == object:
            for value in frame.iloc[:, position]:
                if isinstance(value, (list, tuple, dict, set)):
                    return True
    return False


def _plain_column_names(frame):
    """The column names as JSON scalars, or None if any is not one (a MultiIndex)."""
    names = []
    for name in frame.columns:
        if hasattr(name, "item") and not isinstance(name, str):
            name = name.item()
        if not (name is None or isinstance(name, (str, int, float, bool))):
            return None
        names.append(name)
    return names


def _write_frame(frame, work):
    nested = _has_nested_values(frame)
    names = _plain_column_names(frame)
    if not nested and names is not None:
        # Parquet keeps the dtypes; it wants unique string names, so the
        # columns go in by position and their real names (a pivot's ints,
        # duplicates) ride alongside.
        try:
            positional = frame.set_axis([f"c{i}" for i in range(frame.shape[1])], axis=1)
            positional.to_parquet(os.path.join(work, "out.parquet"), index=False)
            with open(os.path.join(work, "out.columns.json"), "w", encoding="utf-8") as f:
                json.dump(names, f)
            return "parquet"
        except Exception:
            pass
    if not nested:
        try:
            frame.to_json(os.path.join(work, "out.json"), orient="table",
                          date_format="iso", default_handler=str)
            return "table"
        except Exception:
            pass
    with open(os.path.join(work, "out.json"), "w", encoding="utf-8") as f:
        json.dump(frame.to_dict(orient="records"), f, default=str)
    return "records"


def run(work, memory_mb):
    res = {"error": None, "type": None, "stdout": "", "stderr": "", "format": None,
           "memory_limit": _limit_memory(memory_mb)}
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        import pandas as pd

        if os.path.exists(os.path.join(work, "sample.parquet")):
            sample_df = pd.read_parquet(os.path.join(work, "sample.parquet"))
        else:
            # Written by the server: the trusted direction for a pickle.
            sample_df = pd.read_pickle(os.path.join(work, "sample.pkl"))
        with open(os.path.join(work, "code.py"), encoding="utf-8") as f:
            code = f.read()

        exec_scope = {"pd": pd, "__name__": "avaloka_sample_execution"}
        exec(compile(code, "<string>", "exec"), exec_scope)

        main_func = exec_scope.get("main")
        if not callable(main_func):
            raise NameError("main() function not found in the generated code.")

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            output_data = main_func(sample_df.copy())

        res["type"] = type(output_data).__name__
        if isinstance(output_data, pd.DataFrame):
            res["format"] = _write_frame(output_data, work)
    except MemoryError as exc:
        res["error"] = str(exc) or (
            f"MemoryError: sample execution exceeded the {memory_mb} MB memory limit"
        )
    except Exception as exc:
        res["error"] = str(exc)
    res["stdout"] = stdout_capture.getvalue()
    res["stderr"] = stderr_capture.getvalue()
    with open(os.path.join(work, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, default=str)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
