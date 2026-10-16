"""The CLI stays installable on its own, and stays separate from the app.

Two standing constraints, neither of which had a test:

* ``import avaloka.cli`` must not drag in ``app.*``. The CLI ships as a
  standalone wheel; the application graph is not in it, so an accidental
  import is not a layering opinion, it is an ``ImportError`` for whoever
  installed from PyPI.
* The CLI's runtime dependencies are **eight**, and must not grow. Every
  format reader the benchmark needs -- openpyxl, fastavro, lxml -- belongs in
  the ``benchmark`` extra, because ``pip install avaloka`` must not start
  pulling a spreadsheet library.

Both were documented and unenforced, which is the same thing as unprotected:
the benchmark gained a workbook builder and an Avro writer in the change that
added this file, and nothing would have noticed either leaking into the
runtime set.

Each check runs in a subprocess, because the only honest way to ask "what does
importing this pull in" is to import it in an interpreter that has not already
imported everything else.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The eight. Frozen here on purpose: this list is the contract, and a change
#: to it should be a visible edit to this file with a reason in the commit,
#: not a side effect of adding an import somewhere.
DECLARED_RUNTIME_DEPENDENCIES = {
    "typer", "rich", "pandas", "numpy", "scikit-learn", "pyarrow",
    "PyYAML", "jinja2",
}

#: Modules the benchmark's dataset builders need, which must stay optional.
BENCHMARK_ONLY_MODULES = ("openpyxl", "fastavro", "lxml")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)


def _requirement_name(spec: str) -> str:
    for separator in (">=", "<=", "==", "~=", ">", "<", "[", ";"):
        spec = spec.split(separator)[0]
    return spec.strip()


def _pyproject() -> dict:
    import tomllib
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


# --------------------------------------------------------------------------
def test_the_cli_runtime_dependency_set_has_not_grown():
    declared = {_requirement_name(d) for d in _pyproject()["project"]["dependencies"]}
    added = sorted(declared - DECLARED_RUNTIME_DEPENDENCIES)
    removed = sorted(DECLARED_RUNTIME_DEPENDENCIES - declared)
    assert not added, (
        f"the CLI gained runtime dependenc(ies) {added}. `pip install avaloka` "
        "must stay at the declared eight; if a reader is only needed by the "
        "benchmark it belongs in the [benchmark] extra. If this growth is "
        "intended, change DECLARED_RUNTIME_DEPENDENCIES here deliberately.")
    assert not removed, (
        f"the CLI lost runtime dependenc(ies) {removed}; update this contract "
        "deliberately if that was intended.")


def test_the_format_readers_are_an_extra_not_a_runtime_dependency():
    """Declared-format readers must be installable, and must not be in the base.

    Both halves matter. A reader that is not in any extra cannot be installed
    by the user the error message tells to install it, and a reader in the base
    breaks the eight-dependency contract. ``.xlsx`` spent this whole release in
    the first state: the CLI claimed Excel support while openpyxl was in no
    declared extra at all.
    """
    project = _pyproject()["project"]
    runtime = {_requirement_name(d) for d in project["dependencies"]}
    extras = project.get("optional-dependencies", {})
    assert "formats" in extras, (
        "there is no [formats] extra, so the readers for declared formats have "
        "nowhere to live except the runtime set")
    for name in ("formats", "benchmark"):
        assert name in extras, f"the [{name}] extra is missing"
        declared = {_requirement_name(d) for d in extras[name]}
        for module in BENCHMARK_ONLY_MODULES:
            assert module not in runtime, (
                f"{module} is a runtime dependency; it belongs in an extra")
            assert module in declared, (
                f"{module} is needed to read a declared format but is not in "
                f"the [{name}] extra ({sorted(declared)}), so "
                f"`pip install 'avaloka[{name}]'` would not install it")


def test_importing_the_cli_does_not_import_the_application():
    result = _run("""
        import sys
        import avaloka.cli  # noqa: F401
        leaked = sorted(m for m in sys.modules
                        if m == "app" or m.startswith("app.")
                        or m == "file_handler" or m.startswith("file_handler."))
        print("LEAKED:" + ",".join(leaked))
    """)
    assert result.returncode == 0, f"importing avaloka.cli failed:\n{result.stderr}"
    line = [l for l in result.stdout.splitlines() if l.startswith("LEAKED:")][0]
    leaked = [m for m in line[len("LEAKED:"):].split(",") if m]
    assert not leaked, (
        f"importing avaloka.cli pulled in {leaked}. The CLI ships without the "
        "application, so this is an ImportError for anyone who installed the "
        "wheel from PyPI.")


def test_importing_the_benchmark_does_not_require_the_optional_readers():
    """``avaloka.benchmark`` must import with no spreadsheet library present.

    ``avaloka/benchmark/__init__.py`` imports the suite, the suite imports the
    dataset modules, and those modules know how to write a workbook. If any of
    them imported openpyxl at module level, ``avaloka.benchmark`` -- and so
    ``avaloka benchmark --help`` -- would fail on a plain install.
    """
    blocked = ", ".join(repr(m) for m in BENCHMARK_ONLY_MODULES)
    result = _run(f"""
        import builtins, sys
        blocked = {{{blocked}}}
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            root = name.split(".")[0]
            if root in blocked:
                raise ImportError(f"{{root}} is blocked by this test")
            return real_import(name, *args, **kwargs)
        builtins.__import__ = guarded
        for m in list(sys.modules):
            if m.split(".")[0] in blocked:
                del sys.modules[m]
        import avaloka.benchmark            # noqa: F401
        from avaloka.benchmark.suite import all_tasks
        tasks = all_tasks()
        print(f"OK:{{len(tasks)}}")
    """)
    assert "OK:" in result.stdout, (
        "importing avaloka.benchmark (or building the task list) needs one of "
        f"{list(BENCHMARK_ONLY_MODULES)} at import time, so a plain "
        f"`pip install avaloka` could not run `avaloka benchmark --help`:\n"
        f"{result.stdout}\n{result.stderr}")


@pytest.mark.parametrize("module", BENCHMARK_ONLY_MODULES)
def test_the_optional_reader_guard_itself_can_fail(module):
    """Proof the blocking above actually blocks.

    Without this, a typo in the guard would make the previous test pass by
    importing the libraries normally -- a check whose passing state and
    not-running state are identical, which is the failure this repo keeps
    producing.
    """
    result = _run(f"""
        import builtins
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] == {module!r}:
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)
        builtins.__import__ = guarded
        try:
            import {module}
        except ImportError:
            print("BLOCKED")
        else:
            print("NOT_BLOCKED")
    """)
    assert "BLOCKED" in result.stdout, (
        f"the import guard does not actually block {module}: {result.stdout!r}")
