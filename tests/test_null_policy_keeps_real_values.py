"""The platform half of "only a gap token may become missing".

``gap_cells`` is what the upload sampler (sampling_agent_daft) and the chart
profiler (visualization_agent) count missing cells with. In a mostly-numeric
column it used to call every string that did not parse a gap, so the ICD-9
codes ``V45`` and ``E888`` in a diagnosis column were counted "empty" and the
profilers charted the column as numbers. Now only a null-like word in any
case, ``-``, ``?`` or a blank cell is a gap there; a real string keeps the
column off the numeric path. The CLI's copy is pinned to this one by
tests/test_cli_null_policy_parity.py on the shared table.
"""
from __future__ import annotations

import pandas as pd

from file_handler import null_policy

_NUMERIC = ["250.01", "401", "428", "414", "276", "427", "250", "599", "786", "403"]
_VE = ["V45", "E888", "V58", "E849", "V15"]


def _diag_cells(n: int = 200) -> list:
    cells = []
    for i in range(n):
        if i % 20 == 7:
            cells.append("?")
        elif i % 20 == 13:
            cells.append(None)
        elif i % 20 in (3, 11, 15):
            cells.append(_VE[(i // 20) % len(_VE)])
        else:
            cells.append(_NUMERIC[i % len(_NUMERIC)])
    return cells


def test_gap_cells_counts_only_null_words_and_junk():
    cells = _diag_cells()
    gaps = null_policy.gap_cells(cells)
    assert sum(gaps) == cells.count("?") + cells.count(None) == 20
    assert not any(gap for cell, gap in zip(cells, gaps) if cell in _VE), \
        "a V/E code is a real value, never a gap"


def test_gap_cells_still_reads_stray_words_in_a_numeric_column_as_gaps():
    cells = ["98", "105", "none", "110", "-", "NONE", "92", " ", "101", "?"]
    assert null_policy.gap_cells(cells) == [False, False, True, False, True, True,
                                            False, True, False, True]


def test_the_platform_reader_keeps_the_codes(tmp_path):
    path = tmp_path / "icd.csv"
    pd.DataFrame({"diag": ["" if c is None else c for c in _diag_cells()]}).to_csv(path, index=False)
    frame = null_policy.read_csv_best_effort(str(path))
    diag = frame["diag"]
    assert not pd.api.types.is_numeric_dtype(diag)
    assert int(diag.isna().sum()) == 20
    assert int(diag.isin(_VE).sum()) == 30


def test_the_upload_profiler_does_not_call_the_column_numeric():
    from app.agents.sampling_agent_daft import _compute_stats_from_base

    cells = _diag_cells()
    stats = _compute_stats_from_base(cells)
    assert stats["type"] == "categorical", stats.get("type")
    assert stats["missing_count"] == 20
    # Every distinct code is a value, plus one <missing> bucket for the gaps.
    assert stats["n_unique"] == len({c for c in cells if c not in (None, "?")}) + 1
    assert len({v["value"] for v in stats["top_values"]} & set(_VE)) >= 4


def test_the_chart_profiler_does_not_chart_the_codes_as_numbers():
    from app.agents.visualization_agent import profile_columns

    rows = [{"diag": cell} for cell in _diag_cells()]
    profile = next(p for p in profile_columns(rows, ["diag"]) if p["name"] == "diag")
    assert profile["dtype"] == "categorical", profile["dtype"]
    assert profile["missing_ratio"] == 20 / 200
    top = {entry["value"] for entry in profile["stats"]["top_k"]}
    assert "?" not in top, "a gap is not a category"


def test_a_plain_numeric_column_is_still_numeric_in_both_profilers():
    from app.agents.sampling_agent_daft import _compute_stats_from_base
    from app.agents.visualization_agent import profile_columns

    cells = [str(80 + i % 60) for i in range(100)]
    cells[4] = cells[40] = "none"
    cells[9] = "?"
    assert _compute_stats_from_base(cells)["type"] == "numeric"
    assert _compute_stats_from_base(cells)["missing_count"] == 3
    rows = [{"g": cell} for cell in cells]
    assert profile_columns(rows, ["g"])[0]["dtype"] == "numeric"
