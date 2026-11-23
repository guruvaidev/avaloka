"""Only a gap token may become missing; every other string is a real value.

The diabetes re-run (2026-10-19, web): "diag_3 is 6% empty" counted 6,481
cells, and only 1,423 of them were ``?``. The other 5,058 were ICD-9 V and E
codes (``V45``, ``E888``), real diagnoses, read as missing by D3's second
clause (#530): a column whose real cells were at least 80% numbers was typed
numeric and every cell that did not parse was coerced to NaN.

The rule now: a null-like word in any case, ``-``, ``?`` or a blank cell may
become missing, and nothing else. A mostly-numeric column that holds real
strings is MIXED: it stays text, keeps every one of them, and the coercion
record says how many and which, so the profile can say why it is not numeric.

The fixture below has the diabetes shape: ~80% numeric codes, ~15% V/E codes,
~5% ``?``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from avaloka.io.loader import load_dataset
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze
from avaloka.stats import profile_column

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "null_words"
sys.path.insert(0, str(FIXTURES))
import make_fixtures as fx  # noqa: E402  (the fixture's own numbers are the truth)

_N = 200
_NUMERIC = ["250.01", "401", "428", "414", "276", "427", "250", "599", "786", "403"]
_VE = ["V45", "E888", "V58", "E849", "V15"]

_RUN_SCRIPT = """
import runpy, sys
namespace = runpy.run_path(sys.argv[1], run_name="generated_script")
frame = namespace["load"]()
loaded = sorted(m for m in sys.modules if m == "avaloka" or m.startswith("avaloka."))
assert not loaded, f"the generated script imported avaloka: {loaded}"
frame.to_pickle(sys.argv[2])
"""


def _icd_frame() -> pd.DataFrame:
    """160 numeric codes, 30 V/E codes, 10 ``?``; and a numeric column with
    stray ``none`` cells, so both kinds of record are made in one read."""
    diag, glucose = [], []
    for i in range(_N):
        if i % 20 == 7:
            diag.append("?")
        elif i % 20 in (3, 11, 15):
            diag.append(_VE[(i // 20) % len(_VE)])
        else:
            diag.append(_NUMERIC[i % len(_NUMERIC)])
        glucose.append("none" if i % 25 == 4 else str(80 + i % 60))
    outcome = ["Readmitted" if i % 3 == 0 else "Recovered" for i in range(_N)]
    return pd.DataFrame({"diag": diag, "glucose": glucose, "outcome": outcome})


_QUESTION = sum(1 for i in range(_N) if i % 20 == 7)
_CODES = sum(1 for i in range(_N) if i % 20 in (3, 11, 15))
_NONE = sum(1 for i in range(_N) if i % 25 == 4)


@pytest.fixture(scope="module")
def icd_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("icd") / "icd.csv"
    _icd_frame().to_csv(path, index=False)
    return path


def _assert_codes_kept(handle):
    source = _icd_frame()["diag"]
    diag = handle.frame["diag"]
    assert not pd.api.types.is_numeric_dtype(diag), f"diag read as {diag.dtype}"
    assert int(diag.isna().sum()) == _QUESTION, "only the ? cells are missing"
    assert diag.isna().equals(source == "?")
    assert (diag[source != "?"].astype(str) == source[source != "?"]).all(), \
        "every real code, numeric or V/E, comes back as written"
    assert int(diag.isin(_VE).sum()) == _CODES
    record = handle.read["coerced"]["diag"]
    assert record == {"kind": "mixed", "cells": _QUESTION, "words": ["?"],
                      "kept": _CODES, "examples": sorted(_VE)}


def test_icd_codes_are_kept_and_only_question_marks_are_missing(icd_csv):
    handle = load_dataset(str(icd_csv))
    _assert_codes_kept(handle)
    profile = profile_column(handle.frame["diag"], handle.n_rows)
    assert profile["role"] == "categorical" and profile["missing"] == _QUESTION
    # The numeric column beside it is still typed: only gap words stood in its way.
    glucose = handle.frame["glucose"]
    assert pd.api.types.is_float_dtype(glucose) and int(glucose.isna().sum()) == _NONE
    assert handle.read["coerced"]["glucose"] == {"kind": "numeric", "cells": _NONE, "words": ["none"]}


def test_the_scout_says_why_the_column_is_not_numeric(icd_csv):
    from avaloka.fireflies.data_scout import _loader_issues   # lazy: a new kind

    handle = load_dataset(str(icd_csv))
    issue = next(i for i in _loader_issues(handle) if i["column"] == "diag")
    assert issue["kind"] == "real_values_kept_as_text"
    assert f"{_CODES} cell(s)" in issue["detail"] and "'E888'" in issue["detail"]
    assert f"{_QUESTION} gap cell(s) ('?')" in issue["detail"]


def test_the_glucose_fixture_is_still_numeric_with_three_gaps():
    """#530's own case must survive: stray none/N/A/- are gaps, the column is float."""
    handle = load_dataset(str(FIXTURES / "edge_cases.csv"))
    glucose = handle.frame["Glucose"]
    assert pd.api.types.is_float_dtype(glucose), f"Glucose read as {glucose.dtype}"
    assert int(glucose.isna().sum()) == len(fx.GLUCOSE_WORDS) == 3
    assert handle.read["coerced"]["Glucose"]["kind"] == "numeric"


def _write_strings(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".avro":
        fastavro = pytest.importorskip("fastavro")
        schema = {"type": "record", "name": "row",
                  "fields": [{"name": c, "type": "string"} for c in frame.columns]}
        with path.open("wb") as fh:
            fastavro.writer(fh, fastavro.parse_schema(schema), frame.to_dict("records"))
    else:
        orc = pytest.importorskip("pyarrow.orc")
        import pyarrow as pa
        orc.write_table(pa.Table.from_pandas(frame, preserve_index=False), str(path))


@pytest.mark.parametrize("suffix", [".avro", ".orc"])
def test_avro_and_orc_follow_the_same_rule(suffix, tmp_path):
    path = tmp_path / f"icd{suffix}"
    _write_strings(_icd_frame(), path)
    handle = load_dataset(str(path))
    _assert_codes_kept(handle)
    assert pd.api.types.is_float_dtype(handle.frame["glucose"])
    assert int(handle.frame["glucose"].isna().sum()) == _NONE


def test_the_generated_script_reproduces_the_read_cell_for_cell(icd_csv, tmp_path):
    out = tmp_path / "bundle"
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal="What predicts readmission?",
                         data_source=str(icd_csv), output_dir=out,
                         budget=Budget(limit_usd=None))
    run_analyze(ctx)
    analysed = ctx.blackboard["frame"]
    assert int(analysed["diag"].isin(_VE).sum()) == _CODES, "the analysis kept the codes"
    script = (out / "analysis.py").read_text()
    assert "errors=\"coerce\"" not in script, "the script must not coerce real values away"
    pickle = tmp_path / "loaded.pkl"
    done = subprocess.run([sys.executable, "-c", _RUN_SCRIPT, str(out / "analysis.py"), str(pickle)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-600:]
    loaded = pd.read_pickle(pickle)
    pdt.assert_frame_equal(loaded, analysed)
    assert loaded["diag"].isna().equals(analysed["diag"].isna())
