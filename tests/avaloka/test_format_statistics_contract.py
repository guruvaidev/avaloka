"""Every declared format reaches the CLI's understanding and feature-engineering
fireflies with the statistics CSV gets.

``DataScout`` (the understanding step) profiles ``handle.frame`` column by
column with ``avaloka.stats.profile_column`` -- dtype, role, missing rate,
cardinality, numeric summary or top values -- and records ``handle.structure``
beside it. ``DataEngineer`` (feature engineering) reads that profile's quality
block. ``avaloka coordinate`` shards the same frame and converges exact
sufficient statistics per shard. So the contract for a format is: the loader
hands back the same frame CSV would, and the same manifest.

Two things were missing. ``load_dataset`` could not open a directory of part
files at all (a directory has no suffix, so it fell through to the CSV
sniffer and raised), and only CSV and Parquet recorded ``columns`` and
``text_columns`` in the manifest, so a Parquet and an ORC file of the same
table described themselves differently. Neither surfaced in a test because
the tests checked the frame and not what was said about it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from avaloka.io.loader import UnusableDataset, load_dataset
from avaloka.ray_batch import _combine, _partition_stats
from avaloka.stats import profile_column, quality_report
from avaloka import workload

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "formats"
TRUTH = json.loads((FIXTURES / "truth.json").read_text())

SINGLE_FILES = {
    "csv": "sales.csv", "tsv": "sales.tsv", "json": "sales.json", "jsonl": "sales.jsonl",
    "ndjson": "sales.ndjson", "xml": "sales.xml", "parquet": "sales.parquet",
    "orc": "sales.orc", "avro": "sales.avro", "xlsx": "sales.xlsx",
}
DIRECTORIES = {
    "parquet": ("sales_parquet_parts", "parquet_part_rows"),
    "csv": ("sales_csv_parts", "csv_part_rows"),
}
#: The role the understanding step must assign, per column kind in truth.json.
ROLE_FOR_KIND = {"category": "categorical", "numeric": "numeric", "date_text": "datetime", "text": "text"}


def _needs(fmt: str) -> None:
    if fmt == "avro":
        pytest.importorskip("fastavro")
    if fmt == "orc":
        pytest.importorskip("pyarrow.orc")
    if fmt == "xlsx":
        pytest.importorskip("openpyxl")


def _assert_profile_contract(frame: pd.DataFrame, *, label: str) -> None:
    n_rows = len(frame)
    assert n_rows == TRUTH["n_rows"], f"{label}: {n_rows} rows"
    assert list(frame.columns) == TRUTH["columns"], f"{label}: columns {list(frame.columns)}"
    columns = [profile_column(frame[c], n_rows) for c in frame.columns]
    by_name = {c["name"]: c for c in columns}
    for name, kind in TRUTH["kinds"].items():
        col = by_name[name]
        assert col["role"] == ROLE_FOR_KIND[kind], f"{label}: {name} role={col['role']}"
        assert col["missing"] == TRUTH["nulls"][name], f"{label}: {name} missing={col['missing']}"
        assert col["n_unique"] == TRUTH["n_unique"][name], f"{label}: {name} n_unique={col['n_unique']}"
        if kind == "numeric":
            truth = TRUTH["numeric"][name]
            assert col["stats"]["min"] == truth["min"] and col["stats"]["max"] == truth["max"]
            assert col["stats"]["mean"] == pytest.approx(truth["mean"], rel=1e-6)
        elif kind == "category":
            assert sum(v["count"] for v in col["top_values"]) == n_rows - TRUTH["nulls"][name]
    quality = quality_report(frame, columns)
    assert quality["n_rows"] == n_rows and quality["constant_columns"] == []
    # What the feature engineer acts on: the one column with a real null rate
    # is below the high-missing threshold, so nothing is dropped for it.
    assert quality["high_missing_columns"] == []


def _assert_manifest_contract(handle, *, label: str, n_files: int) -> None:
    structure = handle.structure
    assert structure["columns"] == {c: str(handle.frame[c].dtype) for c in handle.frame.columns}, label
    assert set(structure["text_columns"]) == {"region", "reported_on", "channel", "note"}, (
        f"{label}: text_columns={structure['text_columns']}")
    source = structure["source"]
    assert source["format"] == handle.fmt
    assert source["n_files"] == n_files
    assert source["total_bytes"] > 0
    assert sum(p["rows"] for p in source["partitions"]) == handle.n_rows
    assert sum(p["bytes"] for p in source["partitions"]) == source["total_bytes"]
    for part in source["partitions"]:
        assert set(part) == {"bytes", "rows"}, f"{label}: partition carries more than counts"


@pytest.mark.parametrize("fmt", sorted(SINGLE_FILES))
def test_every_format_profiles_like_csv(fmt):
    _needs(fmt)
    path = FIXTURES / SINGLE_FILES[fmt]
    handle = load_dataset(str(path))
    _assert_profile_contract(handle.frame, label=fmt)
    _assert_manifest_contract(handle, label=fmt, n_files=1)
    assert handle.structure["source"]["total_bytes"] == path.stat().st_size


@pytest.mark.parametrize("fmt", sorted(DIRECTORIES))
def test_a_directory_of_parts_loads_as_one_dataset_with_sizes_per_part(fmt):
    folder, key = DIRECTORIES[fmt]
    handle = load_dataset(str(FIXTURES / folder))
    assert handle.fmt == fmt
    _assert_profile_contract(handle.frame, label=f"{fmt} directory")
    _assert_manifest_contract(handle, label=f"{fmt} directory", n_files=len(TRUTH[key]))
    assert [p["rows"] for p in handle.structure["source"]["partitions"]] == TRUTH[key]
    # The generated script cannot repeat a stacked read with pandas alone and
    # must say so rather than read one part.
    assert handle.read["kind"] == "directory"
    assert handle.read["parts"] == len(TRUTH[key]) and handle.read["part_kind"] == fmt


def test_a_directory_mixing_formats_is_refused_by_name(tmp_path):
    import shutil
    folder = tmp_path / "mixed"
    folder.mkdir()
    shutil.copy(FIXTURES / "sales.csv", folder / "a.csv")
    shutil.copy(FIXTURES / "sales.parquet", folder / "b.parquet")
    with pytest.raises(UnusableDataset, match="csv, parquet"):
        load_dataset(str(folder))


def test_an_empty_directory_is_refused_not_sniffed(tmp_path):
    (tmp_path / "nothing").mkdir()
    (tmp_path / "nothing" / "_SUCCESS").write_text("")
    with pytest.raises(UnusableDataset, match="no data files"):
        load_dataset(str(tmp_path / "nothing"))


def test_workload_routes_a_directory_by_the_sum_of_its_parts():
    folder = FIXTURES / "sales_parquet_parts"
    n_rows, n_bytes, estimated = workload.peek(str(folder))
    assert n_rows == TRUTH["n_rows"] and estimated is False
    assert n_bytes == sum(f.stat().st_size for f in folder.glob("*.parquet"))


@pytest.mark.parametrize("fmt", ["orc", "avro"])
def test_workload_peeks_columnar_counts_without_loading(fmt):
    _needs(fmt)
    path = FIXTURES / SINGLE_FILES[fmt]
    n_rows, n_bytes, estimated = workload.peek(str(path))
    if fmt == "orc":
        assert (n_rows, estimated) == (TRUTH["n_rows"], False)
    assert n_bytes == path.stat().st_size


@pytest.mark.parametrize("fmt", ["parquet", "orc", "xml"])
def test_coordinate_statistics_converge_exactly_from_any_format(fmt):
    """What ``avaloka coordinate`` reports: per-shard sufficient statistics
    combined into one profile equal to a single-pass profile."""
    _needs(fmt)
    frame = load_dataset(str(FIXTURES / SINGLE_FILES[fmt])).frame
    shards = [frame.iloc[:40], frame.iloc[40:]]
    converged = _combine([_partition_stats(s) for s in shards])
    assert converged["n_rows"] == TRUTH["n_rows"] and converged["n_partitions"] == 2
    by_name = {c["name"]: c for c in converged["columns"]}
    for name, truth in TRUTH["numeric"].items():
        assert by_name[name]["role"] == "numeric"
        assert by_name[name]["mean"] == pytest.approx(truth["mean"], rel=1e-6)
        assert by_name[name]["min"] == truth["min"] and by_name[name]["max"] == truth["max"]
        assert by_name[name]["missing"] == TRUTH["nulls"][name]
    assert by_name["region"]["role"] == "categorical"
    assert sum(v["count"] for v in by_name["region"]["top_values"]) == TRUTH["n_rows"] - TRUTH["nulls"]["region"]
