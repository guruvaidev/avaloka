"""Header-row detection at upload (app/api/server.py: _fix_header_row).

Leela's report: charts on test.avaloka.ai showed "Unnamed: 0, 1, ..." on the
axes. Cause: files whose column names are not on row 1 (a report title, an
"exported on" line, blank rows) were read with row 1 as the header. Also:
a blank header cell over real data (the index column DataFrame.to_csv
writes, a merged cell in an Excel export) became "Unnamed: N".

Each test writes a small CSV, runs the same two steps the upload runs
(_fix_header_row, then _strip_blank_header_columns), and reads the result the
way the sampler does. server.py is imported once; if it cannot be imported in
this environment the file is skipped with the reason.

    pytest tests/test_upload_header_detection.py -q
"""

from __future__ import annotations

import pandas as pd
import pytest

try:
    from app.api import server
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"app.api.server could not be imported here: {exc}", allow_module_level=True)

HEADER = ["patient_id", "admission_date", "department", "age", "length_of_stay_days"]
DATA = [
    [1000 + i, f"2026-0{1 + i % 9}-1{i % 9}", ("Cardiology", "Neurology", "Oncology")[i % 3],
     20 + i % 60, round(2.5 + (i % 7) * 0.8, 1)]
    for i in range(40)
]


def _write(tmp_path, text, name="upload.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _csv(rows, delim=","):
    return "\n".join(delim.join(str(c) for c in r) for r in rows) + "\n"


def _upload_steps(path):
    info = server._fix_header_row(path)
    dropped = server._strip_blank_header_columns(path)
    return info, dropped


def _columns(path, sep=","):
    return list(pd.read_csv(path, sep=sep).columns)


# ---------------------------------------------------------------- the bug


def test_a_title_and_blank_line_above_the_header_are_skipped(tmp_path):
    p = _write(tmp_path, "Hospital Admissions Report - Q3 2026\n\n" + _csv([HEADER] + DATA))

    info, _ = _upload_steps(p)

    assert info["header_row"] == 3
    assert info["skipped"] == ["Hospital Admissions Report - Q3 2026"]
    assert _columns(p) == HEADER
    assert len(pd.read_csv(p)) == len(DATA)          # no data row lost


def test_an_excel_style_padded_preamble_is_skipped(tmp_path):
    pad = lambda text: [text] + [""] * (len(HEADER) - 1)  # noqa: E731
    rows = [pad("City General Hospital"), pad("Exported on 2026-10-01"), [""] * len(HEADER), HEADER] + DATA
    p = _write(tmp_path, _csv(rows))

    info, _ = _upload_steps(p)

    assert info["header_row"] == 4
    assert _columns(p) == HEADER
    assert not [c for c in _columns(p) if c.startswith("Unnamed")]


def test_a_semicolon_file_with_a_title_row(tmp_path):
    p = _write(tmp_path, "Admissions export\n" + _csv([HEADER] + DATA, delim=";"))

    info, _ = _upload_steps(p)

    assert info["header_row"] == 2
    assert _columns(p, sep=";") == HEADER


def test_a_title_row_and_a_trailing_comma_are_both_fixed(tmp_path):
    body = "".join(",".join(map(str, r)) + ",\n" for r in [HEADER] + DATA)
    p = _write(tmp_path, "Admissions" + "," * len(HEADER) + "\n" + body)

    info, dropped = _upload_steps(p)

    assert info["header_row"] == 2
    assert dropped == [len(HEADER)]                  # the empty trailing column
    assert _columns(p) == HEADER


# ---------------------------------------------------------------- blank header cells


def test_the_index_column_written_by_to_csv_gets_a_name(tmp_path):
    rows = [[""] + HEADER] + [[i] + r for i, r in enumerate(DATA)]
    p = _write(tmp_path, _csv(rows))

    info, _ = _upload_steps(p)

    assert info == {"header_row": 1, "skipped": [], "named": [1]}
    assert _columns(p) == ["column_1"] + HEADER


def test_a_blank_header_over_real_data_is_named_by_position(tmp_path):
    header = HEADER[:2] + [""] + HEADER[3:]
    p = _write(tmp_path, _csv([header] + DATA))

    info, _ = _upload_steps(p)

    assert info["named"] == [3]
    assert _columns(p) == HEADER[:2] + ["column_3"] + HEADER[3:]


def test_a_blank_header_over_an_empty_column_is_still_dropped_not_named(tmp_path):
    header = HEADER + [""]
    rows = [r + [""] for r in DATA]
    p = _write(tmp_path, _csv([header] + rows))

    info, dropped = _upload_steps(p)

    assert info is None                              # nothing to name
    assert dropped == [len(HEADER)]
    assert _columns(p) == HEADER


# ---------------------------------------------------------------- left alone


def test_a_normal_file_is_not_touched(tmp_path):
    text = _csv([HEADER] + DATA)
    p = _write(tmp_path, text)

    info, dropped = _upload_steps(p)

    assert (info, dropped) == (None, [])
    assert p.read_text(encoding="utf-8") == text     # byte for byte


def test_a_file_with_no_header_at_all_is_not_touched(tmp_path):
    """All rows are numbers: there is no header to find, so nothing is guessed."""
    text = _csv([[r[0], r[3], r[4]] for r in DATA])
    p = _write(tmp_path, text)

    assert server._fix_header_row(p) is None
    assert p.read_text(encoding="utf-8") == text


def test_a_wide_data_row_above_the_header_means_no_guess(tmp_path):
    """Only sparse rows (titles, notes, blank lines) may sit above the header.
    A full row up there means the layout is not understood: leave it."""
    rows = [["1", "2", "3", "4", "5"], HEADER] + DATA
    text = _csv(rows)
    p = _write(tmp_path, text)

    assert server._fix_header_row(p) is None
    assert p.read_text(encoding="utf-8") == text


def test_a_header_further_down_than_the_limit_is_not_searched_for(tmp_path):
    preamble = "".join(f"note {i}\n" for i in range(server.HEADER_MAX_OFFSET + 2))
    text = preamble + _csv([HEADER] + DATA)
    p = _write(tmp_path, text)

    assert server._fix_header_row(p) is None


def test_a_file_that_is_not_utf8_is_left_alone(tmp_path):
    p = tmp_path / "latin1.csv"
    p.write_bytes(("Report\n" + _csv([HEADER] + DATA)).encode("utf-16"))

    assert server._fix_header_row(p) is None


def test_an_empty_file_is_left_alone(tmp_path):
    assert server._fix_header_row(_write(tmp_path, "")) is None


def test_a_utf8_bom_does_not_hide_the_first_column_name(tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes(("﻿Report title\n" + _csv([HEADER] + DATA)).encode("utf-8"))

    info = server._fix_header_row(p)

    assert info["header_row"] == 2
    assert _columns(p) == HEADER


# ---------------------------------------------------------------- Excel sheets
# A sheet with a merged title row (and a notes row) above the real header,
# exported with pandas' default header=0, arrives with the placeholders
# "Unnamed: 1".."Unnamed: N" already written into the header line.


def _excel_style_export(title, notes, header, data):
    first = [title] + [f"Unnamed: {i}" for i in range(1, len(header))]
    rows = [first, [notes] + [""] * (len(header) - 1), header] + data
    return _csv(rows)


def test_a_sheet_exported_with_unnamed_placeholders_finds_the_real_header(tmp_path):
    p = _write(tmp_path, _excel_style_export(
        "R1 Program - Itemized Plan", "Prepared by the team, updated 28 Aug", HEADER, DATA))

    info, _ = _upload_steps(p)

    assert info["header_row"] == 3
    assert _columns(p) == HEADER
    assert len(pd.read_csv(p)) == len(DATA)


def test_unnamed_placeholder_over_real_data_gets_a_position_name(tmp_path):
    header = HEADER[:2] + ["Unnamed: 2"] + HEADER[3:]
    p = _write(tmp_path, _csv([header] + DATA))

    info, _ = _upload_steps(p)

    assert info["named"] == [3]
    assert _columns(p) == HEADER[:2] + ["column_3"] + HEADER[3:]


def test_unnamed_placeholder_over_an_empty_column_is_dropped(tmp_path):
    header = HEADER + ["Unnamed: 5"]
    p = _write(tmp_path, _csv([header] + [r + [""] for r in DATA]))

    info, dropped = _upload_steps(p)

    assert info is not None and info["named"] == []
    assert dropped == [len(HEADER)]
    assert _columns(p) == HEADER


def test_a_real_column_that_mentions_unnamed_is_kept(tmp_path):
    header = HEADER[:4] + ["Unnamed accounts"]
    p = _write(tmp_path, _csv([header] + DATA))

    assert server._fix_header_row(p) is None
    assert _columns(p) == header