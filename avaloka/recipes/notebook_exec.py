"""Execute a recipe's generated notebook and read back its evidence.

The parity test (design §2.4) runs every recipe's notebook and checks that
the numbers it prints are the numbers the report shows. ``nbformat``,
``nbclient`` and ``ipykernel`` are the ``avaloka[recipes-test]`` extra, so
they are imported inside :func:`execute` and their absence is reported in
the same plain words the ``[recipes]`` loader uses, never as a traceback at
import. Generating a notebook needs none of them: :func:`apply_parameters`
is pure, and is also what ``run`` uses to write a sampled run's
``SAMPLE_ROWS`` / ``SAMPLE_SEED`` into the notebook it ships.

A run applies ``parameters`` the way papermill does (the ``parameters``
cell's assignments are rewritten for the names given, the rest kept), starts
a fresh kernel for ``sys.executable`` from a kernelspec written to a
temporary directory (so no user-level kernelspec is read), runs every cell in
a temporary working directory with network calls refused, and returns the
JSON the ``evidence``-tagged cell printed.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

INSTALL_HINT = ("Notebook execution needs nbclient, nbformat and ipykernel. "
                "Install the recipe test extra: pip install 'avaloka[recipes-test]'")
KERNEL_NAME = "avaloka-recipes"
PARAMETERS_TAG = "parameters"
EVIDENCE_TAG = "evidence"

#: Runs first, untagged: a notebook that reaches for the network fails its
#: cell instead of succeeding on something the data did not hold.
NO_NETWORK_SOURCE = (
    "import socket as _socket\n"
    "def _no_network(*args, **kwargs):\n"
    "    raise RuntimeError('network access is disabled while the notebook is checked against its report')\n"
    "_socket.create_connection = _no_network\n"
    "_socket.getaddrinfo = _no_network\n"
)

_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class NotebookExecutionError(RuntimeError):
    """A cell failed; ``cell_index`` is its position in the notebook given."""

    def __init__(self, cell_index: int, message: str):
        self.cell_index = cell_index
        super().__init__(f"notebook cell {cell_index} failed: {message}")


class NotebookDependencyError(ImportError):
    """nbclient, nbformat or ipykernel is absent; ``str(error)`` is the install message."""


def missing_dependency() -> str | None:
    """The install message when the test extra is absent, else None."""
    try:
        import ipykernel  # noqa: F401
        import nbclient  # noqa: F401
        import nbformat  # noqa: F401
    except ImportError:
        return INSTALL_HINT
    return None


def _deps() -> tuple[Any, Any]:
    try:
        import ipykernel  # noqa: F401
        import nbclient
        import nbformat
    except ImportError as exc:
        raise NotebookDependencyError(INSTALL_HINT) from exc
    return nbformat, nbclient


# ── parameters ─────────────────────────────────────────────────────────────

def _tagged(cell: Mapping[str, Any], tag: str) -> bool:
    return cell.get("cell_type") == "code" and tag in (cell.get("metadata") or {}).get("tags", [])


def _source(cell: Mapping[str, Any]) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def apply_parameters(notebook: dict[str, Any], parameters: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``notebook`` whose ``parameters`` cell assigns each name in
    ``parameters`` its new value (``repr``), keeping every other line; a name
    the cell does not assign is appended. Papermill's convention, without
    papermill."""
    nb = copy.deepcopy(notebook)
    cells = [c for c in nb["cells"] if _tagged(c, PARAMETERS_TAG)]
    if len(cells) != 1:
        raise ValueError(f"expected one cell tagged {PARAMETERS_TAG!r}, found {len(cells)}")
    pending = dict(parameters)
    lines = []
    for line in _source(cells[0]).splitlines():
        m = _ASSIGNMENT.match(line)
        if m and m.group(1) in pending:
            lines.append(f"{m.group(1)} = {pending.pop(m.group(1))!r}")
        else:
            lines.append(line)
    lines += [f"{name} = {value!r}" for name, value in pending.items()]
    cells[0]["source"] = [ln + "\n" for ln in lines[:-1]] + lines[-1:]
    return nb


# ── execution ──────────────────────────────────────────────────────────────

def _write_kernelspec(root: Path) -> Path:
    """A kernelspec for the running interpreter under ``root/kernels/<name>``."""
    folder = root / "kernels" / KERNEL_NAME
    folder.mkdir(parents=True)
    spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Avaloka recipes (parity run)", "language": "python"}
    (folder / "kernel.json").write_text(json.dumps(spec))
    return root / "kernels"


def _first_line(exc: Any) -> str:
    """``ename: evalue`` and the first line of the cell's traceback, without
    colour codes (nbclient's ``traceback`` is one string: the cell's source
    between rules, then the kernel's traceback)."""
    head = f"{getattr(exc, 'ename', type(exc).__name__)}: {getattr(exc, 'evalue', '')}".strip()
    text = getattr(exc, "traceback", None) or ""
    text = "".join(text) if isinstance(text, list) else str(text)
    lines = [_ANSI.sub("", ln).strip() for ln in text.rsplit("------------------", 1)[-1].splitlines()]
    first = next((ln for ln in lines if ln and set(ln) != {"-"}), "")
    return f"{head} | {first}" if first else head


def _evidence(cell: Mapping[str, Any]) -> dict[str, Any]:
    text = "".join("".join(o.get("text", "")) for o in cell.get("outputs", [])
                   if o.get("output_type") == "stream" and o.get("name") == "stdout")
    if not text.strip():
        raise NotebookExecutionError(-1, f"the {EVIDENCE_TAG!r} cell printed nothing")
    evidence = json.loads(text)
    if not isinstance(evidence, dict):
        raise NotebookExecutionError(-1, f"the {EVIDENCE_TAG!r} cell did not print a JSON object")
    return evidence


def _shutdown(km: Any) -> None:
    """Stop ``km``'s kernel now if it is still running (a failed start, or a cleanup
    nbclient skipped), so no kernel outlives the run that started it."""
    try:
        if km.has_kernel and km.is_alive():
            km.shutdown_kernel(now=True)
    except RuntimeError:
        pass  # "No kernel is running!": it is already gone


def execute(notebook: dict[str, Any], *, parameters: Mapping[str, Any] | None = None, timeout: int = 120,
            env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run ``notebook`` in a fresh kernel and return the evidence it printed.

    ``parameters`` overrides the ``parameters`` cell (:func:`apply_parameters`);
    ``timeout`` is per cell, in seconds; ``env`` adds to the kernel's
    environment (the caller's ``os.environ`` is the base). Raises
    :class:`NotebookExecutionError` naming the failing cell's index and the
    first line of its traceback, or when no ``evidence`` cell printed a JSON
    object.
    """
    nbformat, nbclient = _deps()
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client.manager import KernelManager

    nb = apply_parameters(notebook, parameters or {})
    if not any(_tagged(c, EVIDENCE_TAG) for c in nb["cells"]):
        raise NotebookExecutionError(-1, f"no cell tagged {EVIDENCE_TAG!r}")
    for i, cell in enumerate(nb["cells"]):
        cell.setdefault("id", f"cell-{i}")  # nbformat 4.5 requires one; avaloka.notebook writes them
    node = nbformat.reads(json.dumps(nb), as_version=4)  # validates, and joins each source's lines
    node.cells.insert(0, nbformat.v4.new_code_cell(NO_NETWORK_SOURCE))

    with tempfile.TemporaryDirectory(prefix="avaloka-notebook-") as tmp:
        root = Path(tmp)
        cwd = root / "run"
        cwd.mkdir()
        ksm = KernelSpecManager()
        ksm.kernel_dirs = [str(_write_kernelspec(root / "jupyter"))]
        km = KernelManager(kernel_name=KERNEL_NAME, kernel_spec_manager=ksm)
        client = nbclient.NotebookClient(node, km=km, timeout=int(timeout), allow_errors=False,
                                         record_timing=False, shutdown_kernel="immediate",
                                         resources={"metadata": {"path": str(cwd)}})
        kernel_env = {**os.environ, "MPLBACKEND": "Agg", **(env or {})}
        # nbclient shuts down only a kernel whose manager it created; this manager is ours,
        # so without cleanup_kc every run left its kernel alive until the process exited.
        try:
            with client.setup_kernel(cwd=str(cwd), env=kernel_env, cleanup_kc=True):
                for offset, cell in enumerate(node.cells):
                    index = offset - 1  # the network guard is not the caller's cell
                    try:
                        client.execute_cell(cell, offset)
                    except nbclient.exceptions.CellExecutionError as exc:
                        raise NotebookExecutionError(index, _first_line(exc)) from None
                    except nbclient.exceptions.CellTimeoutError:
                        raise NotebookExecutionError(index, f"timed out after {timeout}s") from None
        finally:
            _shutdown(km)
    evidence_cell = [c for c in node.cells if _tagged(c, EVIDENCE_TAG)][-1]
    return _evidence(evidence_cell)
