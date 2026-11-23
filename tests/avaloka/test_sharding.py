"""Streamed shard files: every row lands once, in order, and nothing is held whole.

avaloka.io.sharding cuts the source into one file per clone without loading
it (CSV chunks, Parquet batches). These tests pin what the clones and the
converged profile rely on: the rows are conserved and contiguous, values come
through as the file wrote them, an empty shard is dropped with its directory,
and two shards that typed a column differently still converge as one frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avaloka.coordinator import MainAvaloka
from avaloka.io import sharding
from avaloka.io.sharding import _count_rows, _sizes, peek_columns, write_shards
from avaloka.mission.context import MissionKind
from avaloka.swarm_limits import UNLIMITED


def _frame(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "id": np.arange(n),
        "amount": rng.normal(100, 30, n).round(3),
        "region": rng.choice(["north", "south", "None"], n),
        "note": rng.choice(["a, quoted", 'say "hi"', ""], n),
    })


def _dirs(tmp_path, n):
    return [tmp_path / f"sub-avaloka-{i:02d}" for i in range(n)]


def test_sizes_match_numpy_array_split():
    for total, n in [(10, 3), (7, 7), (3, 5), (0, 2), (100_001, 4)]:
        assert _sizes(total, n) == [len(p) for p in np.array_split(np.arange(total), n)]


def test_csv_rows_are_conserved_contiguous_and_unchanged(tmp_path, monkeypatch):
    # A small chunk makes the stream cross shard boundaries mid-chunk.
    monkeypatch.setattr(sharding, "CSV_CHUNK_ROWS", 37)
    source = tmp_path / "src.csv"
    _frame(1_001).to_csv(source, index=False)
    plan = write_shards(source, _dirs(tmp_path, 4))
    assert plan.method == "csv_stream" and plan.total_rows == 1_001
    assert [s.n_rows for s in plan.shards] == [251, 250, 250, 250]
    assert plan.columns == ["id", "amount", "region", "note"]
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    back = pd.concat([pd.read_csv(s.path, dtype=str, keep_default_na=False) for s in plan.shards],
                     ignore_index=True)
    # Byte-for-byte the strings the source holds, in the source's order:
    # "None" stays a word and an empty cell stays empty for the clone's loader.
    pd.testing.assert_frame_equal(back, raw)


def test_a_fleet_wider_than_the_rows_is_clamped_and_empty_dirs_removed(tmp_path):
    source = tmp_path / "tiny.csv"
    _frame(3).to_csv(source, index=False)
    dirs = _dirs(tmp_path, 5)
    plan = write_shards(source, dirs)
    assert [s.n_rows for s in plan.shards] == [1, 1, 1]
    assert [d.exists() for d in dirs] == [True, True, True, False, False]


def test_a_quoted_newline_overcounts_but_loses_no_row(tmp_path):
    source = tmp_path / "multi.csv"
    source.write_text('id,text\n1,"line one\nline two"\n2,plain\n3,"a\nb\nc"\n4,end')
    assert _count_rows(source) == 7          # newlines, not rows: an upper bound
    plan = write_shards(source, _dirs(tmp_path, 2))
    assert plan.total_rows == 4 and sum(s.n_rows for s in plan.shards) == 4
    back = pd.concat([pd.read_csv(s.path) for s in plan.shards], ignore_index=True)
    assert back["text"].tolist() == ["line one\nline two", "plain", "a\nb\nc", "end"]


def test_a_latin1_csv_is_streamed_and_written_as_utf8(tmp_path, monkeypatch):
    monkeypatch.setattr(sharding, "CSV_CHUNK_ROWS", 50)
    source = tmp_path / "latin.csv"
    rows = ["city,n"] + [f"Zürich,{i}" if i == 400 else f"Bern,{i}" for i in range(500)]
    source.write_bytes(("\n".join(rows) + "\n").encode("latin-1"))
    plan = write_shards(source, _dirs(tmp_path, 2))
    back = pd.concat([pd.read_csv(s.path, encoding="utf-8") for s in plan.shards])
    assert plan.total_rows == 500 and "Zürich" in set(back["city"])


def test_parquet_streams_by_batch_and_keeps_its_types(tmp_path, monkeypatch):
    monkeypatch.setattr(sharding, "PARQUET_BATCH_ROWS", 64)
    source = tmp_path / "src.parquet"
    frame = _frame(500)
    frame.to_parquet(source, index=False, row_group_size=100)
    plan = write_shards(source, _dirs(tmp_path, 3))
    assert plan.method == "parquet_stream" and [s.n_rows for s in plan.shards] == [167, 167, 166]
    assert all(s.path.suffix == ".parquet" for s in plan.shards)
    back = pd.concat([pd.read_parquet(s.path) for s in plan.shards], ignore_index=True)
    pd.testing.assert_frame_equal(back, frame)


def test_a_format_that_cannot_stream_is_loaded_once_and_split(tmp_path):
    source = tmp_path / "rows.jsonl"
    _frame(40).to_json(source, orient="records", lines=True)
    plan = write_shards(source, _dirs(tmp_path, 3))
    assert plan.method == "loaded" and plan.total_rows == 40
    assert [s.n_rows for s in plan.shards] == [14, 13, 13] and plan.structure


def test_peek_columns_reads_only_the_header(tmp_path):
    source = tmp_path / "h.csv"
    _frame(5).to_csv(source, index=False)
    assert peek_columns(source) == ["id", "amount", "region", "note"]
    assert peek_columns(tmp_path / "nothing.xlsx") is None


def test_shards_that_type_a_column_differently_converge_as_one_frame(tmp_path):
    """The first shard holds only numbers in ``code``; the second holds text.
    Each clone types its own shard, so without alignment the first shard's
    numeric reading would win and the second's counts would be lost (or the
    reduce would raise on a missing sum)."""
    n = 400
    frame = pd.DataFrame({
        "code": [str(i % 7) for i in range(n // 2)] + ["X1"] * (n // 2),
        "value": np.arange(n, dtype=float),
        "empty_late": [1.0] * (n // 2) + [None] * (n // 2),
    })
    source = tmp_path / "drift.csv"
    frame.to_csv(source, index=False)
    cr = MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(source), goal="g", output_dir=tmp_path / "fleet", workers=2, kind=MissionKind.ANALYZE)
    cols = {c["name"]: c for c in cr.converged_profile["columns"]}
    assert cols["code"]["role"] == "categorical"
    top = {t["value"]: t["count"] for t in cols["code"]["top_values"]}
    assert top["X1"] == n // 2 and sum(top.values()) == n
    # A column all-empty in one shard is dropped by that shard's loader; it
    # still converges, with that shard's rows counted as missing.
    assert cols["empty_late"]["role"] == "numeric" and cols["empty_late"]["missing"] == n // 2
    assert cols["value"]["mean"] == pytest.approx(frame["value"].mean())


def test_a_training_swarm_refuses_an_unknown_target_before_writing_shards(tmp_path):
    source = tmp_path / "t.csv"
    _frame(20).to_csv(source, index=False)
    with pytest.raises(ValueError, match="available columns: id, amount, region, note"):
        MainAvaloka(swarm_limits=UNLIMITED).coordinate(
            str(source), goal="g", output_dir=tmp_path / "fleet", workers=2,
            kind=MissionKind.TRAIN, target="nope")
    assert not (tmp_path / "fleet").exists()
