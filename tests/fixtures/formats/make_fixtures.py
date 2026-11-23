"""Write the checked-in format fixtures and the truth they are graded against.

One small table, written once per declared upload format, so a difference
between formats is a difference in the reader and nothing else. The files are
committed; this script exists so a reader can see exactly how each was made
and regenerate them byte-for-byte (fixed seed, rounded values, sorted keys).

    python tests/fixtures/formats/make_fixtures.py

The table is deliberately not clean: two columns carry nulls, one column is
free text and one is a two-value category, so the statistics contract has
something to measure on every format -- a null rate of zero on every column
would let a reader that drops nulls pass. ``truth.json`` is computed from the
in-memory frame with pandas alone, never from a file read back, so a reader
bug cannot hide behind a matching writer bug.

Writers outside the CLI's runtime set (fastavro, openpyxl) are imported only
inside the builder that needs them; the ``formats`` extra has them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N_ROWS = 120
SEED = 2026

#: Rows per part of the partitioned fixtures. Uneven on purpose: the estimator
#: used to read the first part's row count and multiply by the part count,
#: which is only right when every part is the same size.
PARQUET_PART_ROWS = (40, 80)
CSV_PART_ROWS = (50, 70)

_NOTES = [
    "customer asked for a quote on the extended warranty and bulk pricing",
    "delivered late because the carrier rerouted the shipment via the depot",
    "repeat order from the spring campaign, same configuration as before",
    "returned one unit with a cracked housing; replacement sent same day",
]


def canonical_frame() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    region = rng.choice(["north", "south", "east", "west"], N_ROWS).astype(object)
    revenue = rng.normal(5_000.0, 900.0, N_ROWS).round(2)
    frame = pd.DataFrame({
        "region": region,
        "units_sold": rng.integers(10, 900, N_ROWS).astype("int64"),
        "revenue_usd": revenue,
        "reported_on": pd.date_range("2024-01-01", periods=N_ROWS, freq="D").strftime("%Y-%m-%d"),
        "channel": rng.choice(["online", "retail"], N_ROWS).astype(object),
        "note": [_NOTES[i % len(_NOTES)] for i in range(N_ROWS)],
    })
    # Nulls where the contract must see them: 6 regions, 3 revenues.
    frame.loc[[3, 17, 42, 58, 91, 110], "region"] = None
    frame.loc[[7, 64, 99], "revenue_usd"] = np.nan
    return frame


def truth(frame: pd.DataFrame) -> dict:
    """What every reader must agree with, computed by pandas on the frame."""
    numeric = {}
    for column in ("units_sold", "revenue_usd"):
        values = frame[column].dropna()
        numeric[column] = {
            "min": float(values.min()), "max": float(values.max()),
            "mean": round(float(values.mean()), 4),
        }
    return {
        "n_rows": int(len(frame)),
        "columns": list(frame.columns),
        "kinds": {  # what the understanding agent must decide per column
            "region": "category", "units_sold": "numeric", "revenue_usd": "numeric",
            "reported_on": "date_text", "channel": "category", "note": "text",
        },
        "nulls": {column: int(frame[column].isna().sum()) for column in frame.columns},
        "n_unique": {column: int(frame[column].dropna().nunique()) for column in frame.columns},
        "numeric": numeric,
        "parquet_part_rows": list(PARQUET_PART_ROWS),
        "csv_part_rows": list(CSV_PART_ROWS),
    }


def _records(frame: pd.DataFrame) -> list[dict]:
    out = []
    for record in frame.to_dict("records"):
        out.append({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in record.items()})
    return out


def write_all(root: Path = HERE) -> None:
    frame = canonical_frame()
    root.mkdir(parents=True, exist_ok=True)

    frame.to_csv(root / "sales.csv", index=False)
    frame.to_csv(root / "sales.tsv", index=False, sep="\t")
    records = _records(frame)
    (root / "sales.json").write_text(json.dumps(records, indent=1) + "\n")
    lines = "".join(json.dumps(r) + "\n" for r in records)
    (root / "sales.jsonl").write_text(lines)
    (root / "sales.ndjson").write_text(lines)
    frame.to_xml(root / "sales.xml", index=False, root_name="sales", row_name="sale")
    frame.to_parquet(root / "sales.parquet", index=False)

    import pyarrow as pa
    import pyarrow.orc as orc
    orc.write_table(pa.Table.from_pandas(frame, preserve_index=False), root / "sales.orc")

    import fastavro
    schema = fastavro.parse_schema({
        "type": "record", "name": "sale", "fields": [
            {"name": "region", "type": ["null", "string"]},
            {"name": "units_sold", "type": "long"},
            {"name": "revenue_usd", "type": ["null", "double"]},
            {"name": "reported_on", "type": "string"},
            {"name": "channel", "type": "string"},
            {"name": "note", "type": "string"},
        ],
    })
    with open(root / "sales.avro", "wb") as handle:
        fastavro.writer(handle, schema, records)

    frame.to_excel(root / "sales.xlsx", index=False, sheet_name="sales")

    parts = root / "sales_parquet_parts"
    parts.mkdir(exist_ok=True)
    start = 0
    for index, rows in enumerate(PARQUET_PART_ROWS):
        frame.iloc[start:start + rows].to_parquet(parts / f"part-{index}.parquet", index=False)
        start += rows
    (parts / "_SUCCESS").write_text("")  # a Spark marker, never data

    parts = root / "sales_csv_parts"
    parts.mkdir(exist_ok=True)
    start = 0
    for index, rows in enumerate(CSV_PART_ROWS):
        frame.iloc[start:start + rows].to_csv(parts / f"part-{index}.csv", index=False)
        start += rows

    (root / "truth.json").write_text(json.dumps(truth(frame), indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    write_all()
