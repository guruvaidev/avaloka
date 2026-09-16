"""Build small, deterministic MTA fixtures from downloaded Kaggle CSVs.

The source datasets are hundreds of megabytes, so they stay outside Git. This
script keeps real rows and only selects the columns needed by automated tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


FIXTURE_DIR = Path(__file__).resolve().parent


def _write_rows(destination: Path, columns: list[str], rows: Iterable[dict[str, str]]) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
            count += 1
    return count


def _read_head(source: Path, columns: list[str], limit: int) -> list[dict[str, str]]:
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(columns) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{source.name} is missing columns: {missing}")
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"{source.name} contains no data rows")
    return rows


def _read_balanced(
    source: Path,
    columns: list[str],
    target: str,
    labels: tuple[str, ...],
    rows_per_label: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(columns) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{source.name} is missing columns: {missing}")
        for row in reader:
            label = row.get(target, "")
            if label in labels and counts[label] < rows_per_label:
                selected.append(row)
                counts[label] += 1
            if all(counts[label] >= rows_per_label for label in labels):
                break
    if any(counts[label] < rows_per_label for label in labels):
        raise ValueError(
            f"{source.name} does not contain enough rows for {target}: {dict(counts)}"
        )
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source_dir: Path) -> dict[str, object]:
    instacart_columns = ["order_id", "product_id", "add_to_cart_order", "reordered"]
    fraud_columns = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card4",
        "card6",
        "C1",
        "D1",
    ]
    fraud_test_columns = [column for column in fraud_columns if column != "isFraud"]
    walmart_validation_columns = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        *[f"d_{day}" for day in range(1907, 1914)],
    ]
    walmart_evaluation_columns = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        *[f"d_{day}" for day in range(1935, 1942)],
    ]

    jobs = [
        (
            "instacart_prior_sample.csv",
            source_dir / "mba_order_products__prior.csv",
            instacart_columns,
            _read_head(source_dir / "mba_order_products__prior.csv", instacart_columns, 512),
        ),
        (
            "instacart_train_sample.csv",
            source_dir / "mba_order_products__train.csv",
            instacart_columns,
            _read_head(source_dir / "mba_order_products__train.csv", instacart_columns, 512),
        ),
        (
            "ieee_fraud_train_sample.csv",
            source_dir / "train_transaction.csv",
            fraud_columns,
            _read_balanced(
                source_dir / "train_transaction.csv",
                fraud_columns,
                target="isFraud",
                labels=("0", "1"),
                rows_per_label=192,
            ),
        ),
        (
            "ieee_fraud_test_sample.csv",
            source_dir / "test_transaction.csv",
            fraud_test_columns,
            _read_head(source_dir / "test_transaction.csv", fraud_test_columns, 256),
        ),
        (
            "walmart_validation_sample.csv",
            source_dir / "walmart_sales_train_validation.csv",
            walmart_validation_columns,
            _read_head(
                source_dir / "walmart_sales_train_validation.csv",
                walmart_validation_columns,
                512,
            ),
        ),
        (
            "walmart_evaluation_sample.csv",
            source_dir / "walmart_sales_train_evaluation.csv",
            walmart_evaluation_columns,
            _read_head(
                source_dir / "walmart_sales_train_evaluation.csv",
                walmart_evaluation_columns,
                512,
            ),
        ),
    ]

    manifest: dict[str, object] = {"fixtures": {}}
    fixture_manifest = manifest["fixtures"]
    assert isinstance(fixture_manifest, dict)
    for destination_name, source, columns, rows in jobs:
        destination = FIXTURE_DIR / destination_name
        row_count = _write_rows(destination, columns, rows)
        fixture_manifest[destination_name] = {
            "source": source.name,
            "source_bytes": source.stat().st_size,
            "rows": row_count,
            "columns": columns,
            "sha256": _sha256(destination),
        }

    manifest_path = FIXTURE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing the downloaded Kaggle CSV files",
    )
    args = parser.parse_args()
    manifest = build(args.source_dir.expanduser().resolve())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
