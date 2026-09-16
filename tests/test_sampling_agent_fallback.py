"""
The no-stratify fallback must be a representative sample, not head(n).

When no stratify column exists, sample_data_from_source used df.head(n) — the
first n physical rows — while presenting the result as a "representative" sample.
On data with any physical/temporal ordering this is biased. The fallback now
takes a seeded random sample.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("pandas")

from app.agents.sampling_agent import sample_data_from_source


def _write_sorted_csv(path, n):
    # Both columns unique and sorted -> no stratify column can be inferred,
    # so the no-stratify fallback is exercised, and ordering is a clean bias signal.
    path.write_text("id,val\n" + "\n".join(f"{i},{i}" for i in range(n)) + "\n")


def test_fallback_sample_is_representative_not_head(tmp_path):
    csv = tmp_path / "sorted.csv"
    _write_sorted_csv(csv, 1000)

    res = sample_data_from_source(str(csv), "csv", None, sample_size=100)

    assert not res.get("error")
    ids = [int(r["id"]) for r in res["rows"]]
    assert len(ids) == 100
    # head(100) would give max <= 99; a representative sample spans the range.
    assert max(ids) > 500


def test_fallback_sample_is_deterministic(tmp_path):
    csv = tmp_path / "sorted.csv"
    _write_sorted_csv(csv, 1000)

    res1 = sample_data_from_source(str(csv), "csv", None, sample_size=100)
    res2 = sample_data_from_source(str(csv), "csv", None, sample_size=100)

    assert [r["id"] for r in res1["rows"]] == [r["id"] for r in res2["rows"]]


def test_fallback_returns_all_rows_when_n_covers_dataset(tmp_path):
    csv = tmp_path / "small.csv"
    _write_sorted_csv(csv, 20)

    res = sample_data_from_source(str(csv), "csv", None, sample_size=100)

    assert not res.get("error")
    assert len(res["rows"]) == 20   # n >= total_rows -> all rows, no error
