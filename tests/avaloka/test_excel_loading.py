"""Spreadsheets are how plans and trackers actually arrive.

Written against a real R1 programme plan whose header sits on row 4, under a
title and a subtitle. Reading row 1 as the header yields one named column and a
frame of mostly-NaN, which then gets profiled as if the data were that shape.
"""
from __future__ import annotations

import pandas as pd
import pytest

from avaloka.io.loader import load_dataset


@pytest.fixture()
def plan_workbook(tmp_path):
    """A sheet shaped like a hand-maintained plan: title, blank, then header."""
    path = tmp_path / "plan.xlsx"
    rows = [
        ["R1 Program - M3: Autonomy", None, None, None],
        ["Proposed by Leela - 12 Sep 2026", None, None, None],
        [None, None, None, None],
        ["ID", "Task", "Owner", "Effort"],
        ["M3-00", "Licence + stack decision", "Leela", "0.5 d"],
        ["M3-01", "Task definition", "Lei", "0.5 d"],
        ["M3-02", "Dataset contract", "Shyam", "1 d"],
    ]
    pd.DataFrame(rows).to_excel(path, index=False, header=False)
    return path


def test_excel_is_loadable_at_all(plan_workbook):
    """.xlsx used to fall through to a CSV attempt and fail.

    The error read "Unsupported or unreadable data source", which sounds like
    the file is broken rather than like the CLI cannot open spreadsheets.
    """
    result = load_dataset(str(plan_workbook))
    assert result is not None


def test_header_is_found_below_title_rows(plan_workbook):
    """The header is on row 4, not row 1."""
    frame = getattr(load_dataset(str(plan_workbook)), "frame", None)
    if frame is None:                      # loader returns the frame directly
        frame = load_dataset(str(plan_workbook))
    assert list(frame.columns)[:4] == ["ID", "Task", "Owner", "Effort"], (
        f"header not detected; got {list(frame.columns)[:4]}")


def test_rows_below_the_header_are_kept(plan_workbook):
    frame = load_dataset(str(plan_workbook))
    frame = getattr(frame, "frame", frame)
    assert len(frame) == 3, f"expected 3 task rows, got {len(frame)}"
    assert "M3-00" in set(frame["ID"]), "first data row was consumed as a header"


def test_title_rows_do_not_become_data(plan_workbook):
    frame = load_dataset(str(plan_workbook))
    frame = getattr(frame, "frame", frame)
    blob = " ".join(str(v) for v in frame.to_numpy().ravel())
    assert "Proposed by Leela" not in blob, (
        "a title row leaked into the data and will be profiled as a value")
