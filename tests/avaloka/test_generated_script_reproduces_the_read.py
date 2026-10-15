"""The script a user is handed must read the file the way the analysis did.

Every bundle carries ``analysis.py`` and ``analysis.ipynb``, and the product
says "reproduce with" them in three places. Nothing ever ran them. They read
the source with a bare ``pd.read_csv``, while the mission reads it through
``avaloka/io/loader.py``: sniffed delimiter, encoding fallback, null-like words
kept as labels in text columns. So the script loaded a different frame from the
one that was analysed, and for a spreadsheet it could not load anything.

These tests EXECUTE the generated code, in a fresh interpreter that has never
imported ``avaloka``, and compare the frame it loads with the frame the mission
analysed: columns, dtypes, row count, every cell, and which cells are missing.
Where pandas alone cannot repeat the read, the script must say so and stop.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pandas.testing as pdt
import pytest

from avaloka.benchmark import datasets as ds
from avaloka.io import materialize
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze

# Runs analysis.py's load() in a fresh interpreter and hands the frame back.
_RUN_SCRIPT = """
import runpy, sys
namespace = runpy.run_path(sys.argv[1], run_name="generated_script")
frame = namespace["load"]()
loaded = sorted(m for m in sys.modules if m == "avaloka" or m.startswith("avaloka."))
assert not loaded, f"the generated script imported avaloka: {loaded}"
frame.to_pickle(sys.argv[2])
"""

# Runs the notebook's import cell and its load cell, and hands `df` back.
_RUN_CELLS = """
import json, sys
cells = ["".join(c["source"]) for c in json.load(open(sys.argv[1]))["cells"]
         if c["cell_type"] == "code"]
load_cell = next(src for src in cells if "DATA_SOURCE" in src)
namespace = {}
exec(compile(cells[0], "<import cell>", "exec"), namespace)
exec(compile(load_cell, "<load cell>", "exec"), namespace)
loaded = sorted(m for m in sys.modules if m == "avaloka" or m.startswith("avaloka."))
assert not loaded, f"the notebook's load cell imported avaloka: {loaded}"
namespace["df"].to_pickle(sys.argv[2])
"""


def _mission(tmp_path, source, resolved=None):
    out = tmp_path / "bundle"
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal="Profile the dataset",
                         data_source=str(source), output_dir=out,
                         budget=Budget(limit_usd=None))
    if resolved is not None:
        ctx.blackboard["resolved_source"] = resolved.as_dict()
    run_analyze(ctx)
    return out, ctx.blackboard["frame"]


def _run(driver, target, tmp_path):
    """Run ``driver`` on ``target`` in a fresh interpreter, away from the repo."""
    pickle = tmp_path / "loaded.pkl"
    if pickle.exists():
        pickle.unlink()
    done = subprocess.run([sys.executable, "-c", driver, str(target), str(pickle)],
                          capture_output=True, text=True, cwd=tmp_path)
    frame = pd.read_pickle(pickle) if done.returncode == 0 else None
    return done, frame


def _load_cell(bundle):
    cells = ["".join(c["source"]) for c in json.loads((bundle / "analysis.ipynb").read_text())["cells"]
             if c["cell_type"] == "code"]
    return next(src for src in cells if "DATA_SOURCE" in src)


# --- the files ---------------------------------------------------------------
def _null_labelled(tmp_path):
    return ds.build_null_labelled_payments(tmp_path / "payments.csv")


def _semicolon(tmp_path):
    path = tmp_path / "semicolon.csv"
    path.write_text("city;visits;method\nleeds;3;card\nyork;4;None\nhull;5;NA\n")
    return path


def _tab_separated(tmp_path):
    path = tmp_path / "tabs.tsv"
    path.write_text("city\tvisits\tmethod\nleeds\t3\tcard\nyork\t4\tNone\nhull\t5\tcash\n")
    return path


def _latin_1(tmp_path):
    path = tmp_path / "latin1.csv"
    path.write_bytes("ville,méthode,montant\nOrléans,carte,1.5\nNîmes,None,NA\nSète,espèces,2.5\n"
                     .encode("latin-1"))
    return path


def _plain(tmp_path):
    return ds.build_churn(tmp_path / "churn.csv", n=120)


def _parquet(tmp_path):
    path = tmp_path / "table.parquet"
    pd.DataFrame({"method": ["card", "None", "cash"], "amount": [1.5, None, 2.5]}).to_parquet(path)
    return path


FILES = {
    "planted null-label file": _null_labelled,
    "semicolon-delimited": _semicolon,
    "tab-separated": _tab_separated,
    "latin-1": _latin_1,
    "plain comma UTF-8": _plain,
    "parquet": _parquet,
}


@pytest.mark.parametrize("name", list(FILES))
def test_analysis_py_loads_the_frame_the_mission_analysed(tmp_path, name):
    bundle, analysed = _mission(tmp_path, FILES[name](tmp_path))
    done, loaded = _run(_RUN_SCRIPT, bundle / "analysis.py", tmp_path)
    assert done.returncode == 0, f"analysis.py failed to load {name}:\n{done.stderr[-600:]}"
    pdt.assert_frame_equal(loaded, analysed)


@pytest.mark.parametrize("name", list(FILES))
def test_the_notebook_load_cell_loads_the_frame_the_mission_analysed(tmp_path, name):
    bundle, analysed = _mission(tmp_path, FILES[name](tmp_path))
    done, loaded = _run(_RUN_CELLS, bundle / "analysis.ipynb", tmp_path)
    assert done.returncode == 0, f"the notebook failed to load {name}:\n{done.stderr[-600:]}"
    pdt.assert_frame_equal(loaded, analysed)


def test_the_planted_file_is_a_real_test_of_the_read(tmp_path):
    """Guard against comparing two equally wrong frames: the analysed frame
    has the three null-word categories, and a bare read of the file does not."""
    source = _null_labelled(tmp_path)
    _, analysed = _mission(tmp_path, source)
    assert {"None", "NA", "N/A"} <= set(analysed["payment_method"])
    assert int(pd.read_csv(source)["payment_method"].isna().sum()) == 16


def test_analysis_py_runs_end_to_end_and_rewrites_the_prepared_data(tmp_path):
    bundle, _ = _mission(tmp_path, _null_labelled(tmp_path))
    shipped = pd.read_parquet(bundle / "transformed_dataset.parquet")
    work = tmp_path / "rerun"
    work.mkdir()
    done = subprocess.run([sys.executable, str(bundle / "analysis.py")],
                          capture_output=True, text=True, cwd=work)
    assert done.returncode == 0, done.stderr[-600:]
    again = pd.read_parquet(work / "transformed_dataset.parquet")
    assert {"None", "NA", "N/A"} <= set(again["payment_method"])
    assert again.shape == shipped.shape
    assert again["payment_method"].tolist() == shipped["payment_method"].tolist()


# --- sources pandas alone cannot re-read: say so and stop ---------------------
def _excel(tmp_path):
    path = tmp_path / "report.xlsx"
    rows = [["Quarterly report", None, None], [None, None, None],
            ["region", "units", "revenue"], ["north", 10, 100.0], ["south", 20, 200.0],
            ["east", 30, 300.0]]
    pd.DataFrame(rows).to_excel(path, index=False, header=False)
    return path, None, "Excel workbook"


def _sqlite(tmp_path):
    import sqlalchemy

    db = tmp_path / "app.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db}")
    pd.DataFrame({"region": ["north", "south", "east"], "units": [10, 20, 30]}).to_sql(
        "sales", engine, index=False)
    engine.dispose()
    resolved = materialize(f"sqlite:///{db}#sales", cache_dir=tmp_path / "cache")
    return resolved.local_path, resolved, "database"


def _object_store(tmp_path):
    import fsspec

    with fsspec.open("memory://reproduce-test/sales.csv", "wb") as handle:
        handle.write(b"region,units\nnorth,10\nsouth,20\neast,30\n")
    resolved = materialize("memory://reproduce-test/sales.csv", cache_dir=tmp_path / "cache")
    return resolved.local_path, resolved, "object stor"


REFUSED = {"excel": _excel, "sqlite": _sqlite, "object store": _object_store}


@pytest.mark.parametrize("name", list(REFUSED))
def test_a_source_the_script_cannot_reread_is_refused_not_misread(tmp_path, name):
    source, resolved, kind = REFUSED[name](tmp_path)
    bundle, _ = _mission(tmp_path, source, resolved)

    script = (bundle / "analysis.py").read_text()
    cell = _load_cell(bundle)
    for label, text in (("analysis.py", script), ("the notebook's load cell", cell)):
        assert "read_csv" not in text, f"{label} still emits a read_csv for a {name} source"
        assert "read_excel" not in text and "read_sql" not in text, label

    for driver, target in ((_RUN_SCRIPT, bundle / "analysis.py"),
                           (_RUN_CELLS, bundle / "analysis.ipynb")):
        done, loaded = _run(driver, target, tmp_path)
        assert done.returncode != 0 and loaded is None
        assert kind in done.stderr, done.stderr[-600:]
        assert "transformed_dataset.parquet" in done.stderr
        assert "Traceback" not in done.stderr, "a refusal is a message, not a crash"

    # and the place it points at is really there, with the analysed rows
    assert len(pd.read_parquet(bundle / "transformed_dataset.parquet")) == 3


def test_the_bundle_does_not_promise_reproduction_it_declines(tmp_path):
    source, resolved, _ = _excel(tmp_path)
    bundle, _ = _mission(tmp_path, source, resolved)
    readme = (bundle / "README.md").read_text()
    technical = (bundle / "technical_report.html").read_text()
    assert "reproducible pipeline" not in readme
    assert "Reproduce with <code>analysis.py</code>" not in technical
    assert "transformed_dataset.parquet" in technical

    bundle, _ = _mission(tmp_path / "csv", _null_labelled(tmp_path))
    assert "reproducible pipeline" in (bundle / "README.md").read_text()
    assert "Reproduce with <code>analysis.py</code>" in (bundle / "technical_report.html").read_text()


# --- through the CLI: the provenance has to reach the generator ---------------
def test_cli_tells_the_generator_where_the_data_came_from(tmp_path):
    from typer.testing import CliRunner

    from avaloka.cli import app

    source, resolved, _ = _sqlite(tmp_path)
    out = tmp_path / "from-db"
    result = CliRunner().invoke(app, ["analyze", resolved.uri, "--goal", "Profile sales",
                                      "-o", str(out), "--quiet"])
    assert result.exit_code == 0, result.output
    assert "read_csv" not in (out / "analysis.py").read_text()
    assert "reproduce this run" not in result.output
    assert "transformed_dataset.parquet" in result.output

    out = tmp_path / "from-file"
    result = CliRunner().invoke(app, ["analyze", str(_null_labelled(tmp_path)), "--goal",
                                      "Total revenue by payment_method", "-o", str(out), "--quiet"])
    assert result.exit_code == 0, result.output
    assert "reproduce this run" in result.output


# --- the record the generator reads -------------------------------------------
def test_the_loader_records_the_read_it_performed(tmp_path):
    from avaloka.io.loader import NULL_WORDS, load_dataset

    read = load_dataset(str(_null_labelled(tmp_path))).read
    assert read["kind"] == "csv" and read["sep"] == "," and read["encoding"] == "utf-8"
    assert read["encoding_errors"] is None
    assert read["keep_default_na"] is False
    assert read["na_values"]["payment_method"] == [""]
    assert set(read["na_values"]["amount"]) == NULL_WORDS | {""}

    numbers = tmp_path / "numbers.csv"
    numbers.write_text("a,b\n1,NA\n2,3.5\n")
    read = load_dataset(str(numbers)).read          # no text column: pandas' defaults
    assert read["keep_default_na"] is True and read["na_values"] is None

    assert load_dataset(str(_semicolon(tmp_path))).read["sep"] == ";"
    assert load_dataset(str(_tab_separated(tmp_path))).read["sep"] == "\t"
    assert load_dataset(str(_latin_1(tmp_path))).read["encoding"] == "latin-1"
    assert load_dataset(str(_parquet(tmp_path))).read == {"kind": "parquet"}
    assert load_dataset(str(_excel(tmp_path)[0])).read["kind"] == "excel"
