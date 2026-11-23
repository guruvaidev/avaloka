"""Every format the door accepts, the analysis reads.

``/api/upload`` declares what it accepts (``SUPPORTED_UPLOAD_EXTS`` in
app/api/server.py) and the CLI loader declares what it reads
(``avaloka.io.loader.SUPPORTED_SUFFIXES``). The execution agent read every
input as CSV, so Parquet and JSONL were accepted, profiled, planned and then
failed at the analysis. This test iterates both declared lists: a suffix the
execution reader (app/agents/input_reader.py) does not dispatch fails it,
and so does a tabular suffix that does not come back as the table it was
written from when read by the script the execution agent actually runs.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

import app.agents.execution_agent as exec_mod
from avaloka.io.complex import detect_complex_format
from avaloka.io.loader import SUPPORTED_SUFFIXES
from app.agents import input_reader as input_reader

ROOT = Path(__file__).resolve().parents[1]


def _declared_upload_suffixes() -> set[str]:
    """SUPPORTED_UPLOAD_EXTS, read from the source: importing the server is heavy."""
    tree = ast.parse((ROOT / "app" / "api" / "server.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) == "SUPPORTED_UPLOAD_EXTS" for t in targets):
                return {f".{ext}" for ext in ast.literal_eval(node.value)}
    raise AssertionError("SUPPORTED_UPLOAD_EXTS is no longer a literal in app/api/server.py")


UPLOAD_SUFFIXES = _declared_upload_suffixes()
DECLARED = sorted(UPLOAD_SUFFIXES | set(SUPPORTED_SUFFIXES))

FRAME = pd.DataFrame({"city": ["Pune", "Lyon", "Oslo"], "amount": [1.5, 2.25, 4.0]})


def _write(path: Path) -> bool:
    """Write FRAME at *path* in its suffix's format; False when this host cannot."""
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        FRAME.to_csv(path, index=False)
    elif suffix == ".tsv":
        FRAME.to_csv(path, sep="\t", index=False)
    elif suffix in (".parquet", ".pq"):
        FRAME.to_parquet(path)
    elif suffix == ".json":
        path.write_text(json.dumps(FRAME.to_dict(orient="records")))
    elif suffix in (".jsonl", ".ndjson"):
        FRAME.to_json(path, orient="records", lines=True)
    elif suffix == ".xml":
        FRAME.to_xml(path, index=False, parser="etree")
    elif suffix == ".orc":
        FRAME.to_orc(path)
    elif suffix == ".avro":
        fastavro = pytest.importorskip("fastavro")
        schema = fastavro.parse_schema({"type": "record", "name": "r", "fields": [
            {"name": "city", "type": "string"}, {"name": "amount", "type": "double"}]})
        with open(path, "wb") as out:
            fastavro.writer(out, schema, FRAME.to_dict(orient="records"))
    elif suffix in (".xlsx", ".xlsm"):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            FRAME.to_excel(writer, index=False)
    else:
        return False
    return True


def test_both_declared_lists_were_found():
    """A vacuous pass guard: the lists this test iterates are the real ones."""
    assert {".csv", ".parquet", ".jsonl", ".xlsx", ".pdf"} <= UPLOAD_SUFFIXES
    assert {".parquet", ".jsonl", ".orc", ".avro"} <= set(SUPPORTED_SUFFIXES)


@pytest.mark.parametrize("suffix", DECLARED)
def test_every_declared_suffix_has_a_reader(suffix):
    assert suffix in input_reader.HANDLED_SUFFIXES, (
        f"{suffix} is declared (upload or CLI loader) but the execution agent has no "
        "reader for it: add it to app/agents/input_reader.py")
    assert input_reader.input_kind(f"data{suffix}") != "unknown"


@pytest.mark.parametrize("suffix", DECLARED)
def test_every_declared_suffix_reaches_the_reader_the_upload_path_uses(suffix, monkeypatch, tmp_path):
    """CSV family -> the CSV reader, workbooks -> ExcelConnector, the rest -> load_dataset."""
    calls = []

    class _Handle:
        frame = FRAME

    def fake_load_dataset(path, **_):
        calls.append(("loader", path))
        return _Handle()

    class FakeExcel:
        def __init__(self, path, sheet_name=None):
            calls.append(("excel", path))

        def load_dataframe(self):
            return FRAME

    def fake_csv(path, **_):
        calls.append(("csv", path))
        return FRAME

    import avaloka.io.loader as loader
    import file_handler.excel_connector as excel
    monkeypatch.setattr(loader, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(excel, "ExcelConnector", FakeExcel)

    path = str(tmp_path / f"data{suffix}")
    input_reader.read_input_frame(path, fake_csv)

    expected = ("csv" if suffix in (".csv", ".tsv") else
                "excel" if suffix in (".xlsx", ".xlsm", ".xls") else "loader")
    assert calls == [(expected, path)]


#: Everything declared except the bounded complex formats (PDF, images, video),
#: which load_dataset turns into rows of extracted content, not FRAME.
TABULAR = [s for s in DECLARED if detect_complex_format(Path(f"data{s}")) is None]


@pytest.mark.parametrize("suffix", TABULAR)
def test_every_tabular_suffix_is_read_by_the_script_the_agent_runs(suffix, tmp_path):
    """The real harness in a real subprocess, one fixture per declared format."""
    path = tmp_path / f"data{suffix}"
    if not _write(path):
        pytest.skip(f"this host cannot write a {suffix} fixture (dispatch is checked above)")
    out = tmp_path / "out.csv"
    code = (
        "import pandas as pd\n"
        "def main(df):\n"
        "    return df.sort_values('city')[['city', 'amount']]\n"
    )

    result = exec_mod.execute_code_on_local(code, input_data_location=str(path), output_location=str(out))

    assert result["status"] == "success", result.get("execution_stderr")
    got = pd.read_csv(out)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True),
        FRAME.sort_values("city")[["city", "amount"]].reset_index(drop=True),
        check_dtype=False,
    )
