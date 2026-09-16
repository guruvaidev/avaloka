from __future__ import annotations

import os
from typing import Optional

import fsspec
import ray.data as rd


def detect_format_from_url(url: str, sniff_bytes: int = 4096) -> str:
    """
    Detect file format from URL extension or by sniffing the first few bytes.
    Does NOT read the full file — only reads up to sniff_bytes.
    """
    # Strip query strings for extension detection
    clean_url = url.split("?")[0]
    fs, fs_path = fsspec.core.url_to_fs(url)
    ext = os.path.splitext(fs_path)[1].lower()
    if ext in (".csv", ".json", ".parquet", ".jsonl", ".ndjson"):
        fmt = ext.lstrip(".")
        return "json" if fmt in ("jsonl", "ndjson") else fmt

    # Sniff only the first few bytes — never reads whole file
    with fs.open(fs_path, "rb") as f:
        content = f.read(sniff_bytes)

    if content[:4] == b"PAR1":
        return "parquet"
    stripped = content.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        return "json"
    if b"," in content and b"\n" in content:
        return "csv"

    raise ValueError(f"Unable to detect format for {url}")


def load_dataset_from_url(url: str, limit: Optional[int] = None) -> rd.Dataset:
    """
    Load a Ray Dataset from a URL (GCS, S3, HTTP, local).

    Streaming behaviour:
      - CSV / JSON / Parquet: Ray Data reads in blocks — never loads the full
        file into a single in-memory buffer.
      - `limit` is applied lazily via ds.limit() which short-circuits reading
        once enough rows have been produced.
    """
    fmt = detect_format_from_url(url)

    # All three Ray readers are block-streaming by default.
    # Ray splits large files into parallel blocks; each block is read on demand.
    if fmt == "csv":
        ds = rd.read_csv(url)
    elif fmt == "json":
        ds = rd.read_json(url)
    elif fmt == "parquet":
        # Parquet is the most streaming-friendly format:
        # Ray reads only the row-groups it needs, column projection is possible.
        ds = rd.read_parquet(url)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    # ds.limit() is lazy — it does NOT materialise the full dataset first.
    return ds.limit(limit) if limit else ds