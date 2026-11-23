"""A notebook run stops the kernel it started, whether its cells pass or fail.

``notebook_exec.execute`` hands nbclient its own KernelManager, and nbclient only
shuts down a kernel whose manager it created itself. So every parity run used to
leave an ipykernel alive until pytest exited: 36 of them, about 200 MB each, which
is what ran CI's 3 GB build container out of memory in the middle of tests/recipes.
"""

from __future__ import annotations

import time

import psutil
import pytest

from avaloka.recipes import notebook_exec

MISSING = notebook_exec.missing_dependency()
pytestmark = [pytest.mark.notebook, pytest.mark.skipif(MISSING is not None, reason=MISSING or "")]


def _notebook(evidence_source: str) -> dict:
    def cell(tag: str, source: str) -> dict:
        return {"cell_type": "code", "metadata": {"tags": [tag]}, "source": source,
                "outputs": [], "execution_count": None}
    return {"cells": [cell("parameters", "X = 1"), cell("evidence", evidence_source)],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _kernels() -> set[int]:
    """PIDs of live ipykernel processes descended from this test process."""
    found = set()
    for proc in psutil.Process().children(recursive=True):
        try:
            if proc.status() != psutil.STATUS_ZOMBIE and "ipykernel_launcher" in " ".join(proc.cmdline()):
                found.add(proc.pid)
        except psutil.Error:
            continue
    return found


def _left_after(before: set[int], grace_s: float = 5.0) -> set[int]:
    deadline = time.monotonic() + grace_s
    left = _kernels() - before
    while left and time.monotonic() < deadline:
        time.sleep(0.1)
        left = _kernels() - before
    return left


def test_a_run_leaves_no_kernel_running():
    before = _kernels()
    assert notebook_exec.execute(_notebook("import json\nprint(json.dumps({'x': X}))")) == {"x": 1}
    left = _left_after(before)
    assert not left, f"the notebook's kernel outlived its run: pids {sorted(left)}"


def test_a_failing_run_leaves_no_kernel_running():
    before = _kernels()
    with pytest.raises(notebook_exec.NotebookExecutionError):
        notebook_exec.execute(_notebook("raise ValueError('cell fails')"))
    left = _left_after(before)
    assert not left, f"a failing notebook's kernel outlived its run: pids {sorted(left)}"
