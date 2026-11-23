"""The CLI's copy of table_shape, pinned to the upload path's.

``file_handler/table_shape.py`` already reads the shapes the benchmark lists
as known defects -- totals and subtotal rows, a footnote block, a second
table, two-row headers -- but only on the upload path. The CLI reader, which
the benchmark grades, cannot import it (``avaloka`` must not import
``file_handler``), and ``file_handler`` must not import ``avaloka``
(``tests/test_upload_format_connectors.py``). So the CLI gets a second copy,
``avaloka/io/table_shape.py``, the way it got a second copy of the null-word
rule.

Two copies drift. This file is the only one that reads both, and it fails on
any difference that is not declared:

* every function, class and constant in the CLI copy must be the same source,
  character for character, as its namesake in the upload path's copy;
* the upload path's copy may hold exactly one thing the CLI's does not:
  ``keep_null_word_labels`` and the ``keeps_null_words`` import it needs,
  because that function asks the upload path's own null-word rule;
* the CLI copy imports only the standard library and pandas, and importing it
  loads neither ``file_handler`` nor ``app``;
* on every benchmark workbook, both copies find the same table.

``test_the_source_comparison_bites`` shows the comparison catches a one-token
change to a function body, so a green run here is not a check that compares
nothing.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI_COPY = ROOT / "avaloka" / "io" / "table_shape.py"
UPLOAD_COPY = ROOT / "file_handler" / "table_shape.py"

#: What the upload path's copy holds that the CLI's must not: the one function
#: that asks file_handler.null_policy, and the import it asks with.
UPLOAD_ONLY = {"keep_null_word_labels"}
UPLOAD_ONLY_IMPORT = "from .null_policy import keeps_null_words"


def _definitions(source: str) -> dict[str, str]:
    """``{name: source}`` for every top-level def, class and assignment."""
    tree = ast.parse(source)
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found[node.name] = ast.get_source_segment(source, node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    found[target.id] = ast.get_source_segment(source, node)
    return found


def _imports(source: str) -> list[str]:
    tree = ast.parse(source)
    return [ast.get_source_segment(source, node) for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))]


def _differences(cli_source: str, upload_source: str) -> list[str]:
    cli, upload = _definitions(cli_source), _definitions(upload_source)
    problems = [f"{name}: differs from file_handler/table_shape.py"
                for name in sorted(cli) if name in upload and cli[name] != upload[name]]
    problems += [f"{name}: only in avaloka/io/table_shape.py" for name in sorted(set(cli) - set(upload))]
    problems += [f"{name}: only in file_handler/table_shape.py, and not declared"
                 for name in sorted(set(upload) - set(cli) - UPLOAD_ONLY)]
    problems += [f"{name}: declared upload-only but missing there"
                 for name in sorted(UPLOAD_ONLY - set(upload))]
    extra = set(_imports(upload_source)) - set(_imports(cli_source))
    problems += [f"import only in file_handler/table_shape.py, not declared: {line}"
                 for line in sorted(extra - {UPLOAD_ONLY_IMPORT})]
    problems += [f"import only in avaloka/io/table_shape.py: {line}"
                 for line in sorted(set(_imports(cli_source)) - set(_imports(upload_source)))]
    return problems


def test_the_two_copies_are_the_same_source():
    problems = _differences(CLI_COPY.read_text(), UPLOAD_COPY.read_text())
    assert not problems, (
        "avaloka/io/table_shape.py and file_handler/table_shape.py have drifted. Make "
        "the same change in both:\n  " + "\n  ".join(problems))


def test_the_source_comparison_bites():
    cli = CLI_COPY.read_text()
    upload = UPLOAD_COPY.read_text()
    # One token in one function body: the check must name that function.
    assert "MAX_HEADER_SCAN" in cli
    drifted = cli.replace("def detect_header_rows(raw: pd.DataFrame, max_scan: int = MAX_HEADER_SCAN)",
                          "def detect_header_rows(raw: pd.DataFrame, max_scan: int = MAX_HEADER_SCAN + 1)")
    assert drifted != cli
    assert any(p.startswith("detect_header_rows:") for p in _differences(drifted, upload))
    # An undeclared function on the upload side is caught too.
    assert any("only in file_handler" in p
               for p in _differences(cli, upload + "\n\ndef sneaked_in():\n    return 1\n"))


def test_the_cli_copy_imports_only_the_standard_library_and_pandas():
    tree = ast.parse(CLI_COPY.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import in {CLI_COPY.name}"
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"__future__", "logging", "numbers", "re", "collections",
                        "typing", "pandas"}, imported


def test_importing_the_cli_copy_loads_neither_file_handler_nor_app():
    code = ("import sys, avaloka.io.table_shape; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in {'file_handler', 'app'}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]", out


@pytest.fixture(scope="module")
def raw_sheets(tmp_path_factory):
    pytest.importorskip("openpyxl", reason="the workbook builders need openpyxl")
    from avaloka.benchmark.datasets_spreadsheet import SPREADSHEET_DATASETS

    base = tmp_path_factory.mktemp("parity")
    sheets = {}
    for name, (builder, _truth) in sorted(SPREADSHEET_DATASETS.items()):
        path = builder(base / f"{name}.xlsx")
        for sheet, raw in pd.read_excel(path, sheet_name=None, header=None).items():
            sheets[f"{name}/{sheet}"] = raw
    assert len(sheets) >= 13
    return sheets


def test_both_copies_find_the_same_table_in_every_benchmark_workbook(raw_sheets):
    from avaloka.io import table_shape as cli
    from file_handler import table_shape as upload

    for label, raw in raw_sheets.items():
        cli_frame, cli_report, cli_rows, cli_cols = cli.reshape_with_positions(raw.copy())
        up_frame, up_report, up_rows, up_cols = upload.reshape_with_positions(raw.copy())
        pdt.assert_frame_equal(cli_frame, up_frame, obj=label)
        assert cli_report == up_report, label
        assert (cli_rows, cli_cols) == (up_rows, up_cols), label
