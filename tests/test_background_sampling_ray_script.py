"""
Regression tests for the background Ray sampler script (build_sampling_ray_script).

Fixed 130 B/row byte offsets gapped/double-counted variable-width rows, the
header was parsed as a data row, and one numeric value flagged a whole column
numeric.

The script also ignored source_type and always parsed as CSV, so Parquet/JSON
datasets produced binary-garbage stats or crashed.

Each test runs the *emitted script* end-to-end in-process with a faked `ray`
module, over a dataset with variable-width rows, nulls, a numeric column, and a
mostly-categorical column containing one stray number, then checks exact ground
truth. No Ray cluster or cloud storage is required.
"""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pd = pytest.importorskip("pandas")

from app.agents.sampling_async import build_sampling_ray_script

N = 800
CATS = ["north", "south", "east", "west"]


def _dataset():
    """DataFrame + expectations: one stray number in `category`, nulls in `amount`."""
    rows, null_amount, amount_sum, cat = [], 0, 0.0, {}
    for i in range(N):
        category = "42" if i == 0 else CATS[i % 4]
        if i % 7 == 0:
            amount = None
            null_amount += 1
        else:
            amount = round(i * 1.5, 2)
            amount_sum += float(amount)
        rows.append({"id": i, "category": category, "amount": amount, "note": "x" * ((i % 50) + 1)})
        cat[category] = cat.get(category, 0) + 1
    exp = {"null_amount": null_amount, "amount_sum": amount_sum, "cat": cat}
    return pd.DataFrame(rows), exp


def _write(df, fmt, tmp_path):
    path = tmp_path / f"data.{fmt}"
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, row_group_size=100)
    elif fmt == "json":
        df.to_json(path, orient="records")
    elif fmt == "jsonl":
        df.to_json(path, orient="records", lines=True)
    return path


def _install_fake_ray():
    ray = types.ModuleType("ray")
    ray.init = lambda *a, **k: None
    ray.nodes = lambda: [{"Alive": True, "Resources": {"CPU": 4}, "NodeManagerAddress": "127.0.0.1"}]
    ray.cluster_resources = lambda: {"CPU": 4}

    def _remote(*da, **dk):
        def make(fn):
            return type("T", (), {"remote": staticmethod(lambda *a, **k: fn(*a, **k))})()
        return make(da[0]) if (len(da) == 1 and callable(da[0]) and not dk) else make

    ray.remote = _remote
    ray.get = lambda x: x
    ray.shutdown = lambda: None
    ray_util = types.ModuleType("ray.util")
    ray_util.get_node_ip_address = lambda: "127.0.0.1"
    ray.util = ray_util
    return ray, ray_util


def _run_emitted(fmt, data_path, art_path, monkeypatch):
    ray, ray_util = _install_fake_ray()
    saved = {k: sys.modules.get(k) for k in ("ray", "ray.util")}
    sys.modules["ray"] = ray
    sys.modules["ray.util"] = ray_util
    try:
        # Shrink partitions so the small file splits into many byte ranges /
        # one row group per partition, exercising the merge paths.
        src = build_sampling_ray_script(fmt)
        src = src.replace("ROWS_PER_PARTITION           = 500_000",
                          "ROWS_PER_PARTITION           = 50")
        src = src.replace("max(int(ROWS_PER_PARTITION * avg_row_bytes), 1_000_000)",
                          "max(int(ROWS_PER_PARTITION * avg_row_bytes), 400)")
        monkeypatch.setenv("DATA_SOURCE_URI", str(data_path))
        monkeypatch.setenv("OUTPUT_ARTIFACT_URI", str(art_path))
        monkeypatch.setenv("OUTPUT_METRICS_URI", "")
        exec(compile(src, f"<{fmt}>", "exec"), {"__name__": "__main__"})
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return json.loads(Path(art_path).read_text())


@pytest.mark.parametrize("fmt", ["csv", "parquet", "json", "jsonl"])
def test_ray_script_exact_stats_per_format(fmt, tmp_path, monkeypatch):
    df, exp = _dataset()
    data_path = _write(df, fmt, tmp_path)
    art = _run_emitted(fmt, data_path, tmp_path / "artifact.json", monkeypatch)
    cs = art["sample_statistics"]["column_statistics"]

    # Every row read exactly once -> exact count (not gapped/double-counted).
    assert art["total_rows"] == N

    # Numeric detection by aggregate ratio (a stray "42" must not flag category).
    # Types use the canonical profiling vocabulary ("numeric"/"categorical") so
    # profiling_agent.profile_full branches correctly (see MAJ-178).
    assert cs["id"]["type"] == "numeric"
    assert cs["amount"]["type"] == "numeric"
    assert cs["category"]["type"] == "categorical"
    assert cs["note"]["type"] == "categorical"

    # Null counting and sums.
    assert cs["amount"]["null_count"] == exp["null_amount"]
    amt_numeric = N - exp["null_amount"]
    assert abs(cs["amount"]["mean"] * amt_numeric - exp["amount_sum"]) < 1e-3

    # Header/keys not parsed as a data row; per-value counts exact.
    cat_tv = {t["value"]: t["count"] for t in cs["category"]["top_values"]}
    assert "category" not in cat_tv
    for k in CATS:
        assert cat_tv.get(k) == exp["cat"][k]
