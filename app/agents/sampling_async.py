"""
Async Background Sampling

Handles the two-speed sampling strategy:
- A quick preview is returned immediately using a row-limited read.
- A full portfolio profile runs in the background for large datasets.

Small datasets (< 100k rows) get the full profile on the first call with no background job needed.

For datasets >= 1 GB the background job runs as a Ray+Daft job on GKE, submitted
via the same infrastructure used by the execution agent.
"""

from __future__ import annotations

import json
import csv
import io
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Tier constants mirror calculate_adaptive_sample_sizes in sampling_agent_daft
TIER_TINY    = "TINY"     # < 1,000 rows
TIER_SMALL   = "SMALL"    # 1k – 100k rows
TIER_LARGE   = "LARGE"    # 100k – 1M rows
TIER_HUGE    = "HUGE"     # 1M – 10M rows
TIER_MASSIVE = "MASSIVE"  # > 10M rows or multi-file / S3 blob

# Rows to read for the quick preview on large datasets.
# Small enough to fit in ~20MB and read from cloud storage in < 3 seconds.
N_QUICK = 50_000

# NOTE: the old second trigger (BACKGROUND_TRIGGER_ROWS = 50_000) was dead
# code — defined but never consumed. Removed; the single background trigger is
# _should_run_background(tier), which under AVALOKA_BYTE_ROUTING is keyed on
# estimated in-memory bytes vs ANALYSIS_MAX_INMEMORY_BYTES (config-driven,
# per-environment — see app/core/analysis_limits.py) instead of row tiers.

# Byte-tier thresholds equivalent to the historical row tiers at ~130 B/row
# (the avg-row fallback used across the size estimators): 1k / 100k / 1M / 10M
# rows -> 128 KB / 13 MB / 130 MB / 1.3 GB. Narrow data routes as before;
# wide data is promoted by its true in-memory footprint.
TIER_BYTES_TINY  = 128 * 1024
TIER_BYTES_SMALL = 13 * 1024 * 1024
TIER_BYTES_LARGE = 130 * 1024 * 1024
TIER_BYTES_HUGE  = 1_300 * 1024 * 1024


def _determine_tier(estimated_rows: int, is_multi_file: bool = False) -> str:
    """Return the sizing tier for a dataset based on estimated row count."""
    if is_multi_file:
        return TIER_MASSIVE
    if estimated_rows < 1_000:
        return TIER_TINY
    elif estimated_rows < 100_000:
        return TIER_SMALL
    elif estimated_rows < 1_000_000:
        return TIER_LARGE
    elif estimated_rows < 10_000_000:
        return TIER_HUGE
    else:
        return TIER_MASSIVE


def _should_run_background(tier: str) -> bool:
    """Return True if the dataset is large enough to warrant a background full profile.

    Legacy row-tier trigger; under AVALOKA_BYTE_ROUTING the byte-based check in
    sample_quick replaces this (estimated bytes >= ANALYSIS_MAX_INMEMORY_BYTES).
    """
    return tier in (TIER_LARGE, TIER_HUGE, TIER_MASSIVE)


def _determine_tier_bytes(estimated_bytes: int, is_multi_file: bool = False) -> str:
    """Byte-based tier: same tiers, input is estimated in-memory bytes not rows."""
    if is_multi_file:
        return TIER_MASSIVE
    if estimated_bytes < TIER_BYTES_TINY:
        return TIER_TINY
    elif estimated_bytes < TIER_BYTES_SMALL:
        return TIER_SMALL
    elif estimated_bytes < TIER_BYTES_LARGE:
        return TIER_LARGE
    elif estimated_bytes < TIER_BYTES_HUGE:
        return TIER_HUGE
    else:
        return TIER_MASSIVE


def _probe_inmemory_bytes_per_row(path: str, source_type: str, **kwargs: Any) -> float:
    """Measure in-memory (Arrow) bytes/row on a 1000-row head slice.

    The same cost signal the Daft sampler's 5 MB clamp uses, taken *before* the
    sync-vs-background decision so wide rows are seen by routing, not just by
    the sampler. Returns 0.0 on failure (caller falls back to an on-disk
    expansion estimate).
    """
    temp_files: List[str] = []
    try:
        from app.agents.sampling_agent_daft import _load_file_with_daft
        df_lazy, temp_files = _load_file_with_daft(path, source_type, **kwargs)
        if df_lazy is None:
            return 0.0
        sample = df_lazy.limit(1000).to_arrow()
        return sample.nbytes / max(sample.num_rows, 1)
    except Exception as exc:
        logger.warning("[byte_routing] bytes/row probe failed for %r: %s", path, exc)
        return 0.0
    finally:
        import os as _os
        for fp in (temp_files or []):
            try:
                if fp and _os.path.exists(fp):
                    _os.unlink(fp)
            except Exception:
                pass


def _is_multi_file_source(path: str) -> bool:
    """
    Return True if the path points to a directory or a cloud prefix rather than a single file.
    Handles glob patterns, trailing slashes, and cloud URIs without a file extension.
    """
    import os
    is_cloud = path.startswith(("gs://", "s3://", "az://", "http://", "https://"))
    if "*" in path or path.endswith("/"):
        return True
    if is_cloud:
        base = path.rstrip("/").split("/")[-1]
        if "." not in base:
            return True
    else:
        if os.path.isdir(path):
            return True
    return False


def sample_quick(
    path: str,
    source_type: str,
    use_ray: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Return a quick profile immediately.

    For small datasets (< 100k rows) this runs the full portfolio algorithm synchronously
    and returns should_run_background=False.

    For large datasets it reads only the first N_QUICK rows via Daft .limit(), builds
    the same portfolio on that slice, and marks the result as a quick preview.
    Returns should_run_background=True so the caller can kick off the full background job.

    Return dict keys:
        schema, rows, ddl_schema, profiling_result, error,
        profile_status, accuracy, should_run_background, estimated_rows, tier
    """
    from app.agents.sampling_agent_daft import (
        sample_with_profiling,
        _estimate_dataset_size,
        _load_file_with_daft,
        create_base_sample,
        create_sample_portfolio,
        aggregate_statistics_from_portfolio,
        calculate_adaptive_sample_sizes,
    )

    try:
        estimated_rows, size_mb = _estimate_dataset_size(
            path, source_type, cloud_credentials=kwargs.get("cloud_credentials")
        )
        is_multi = _is_multi_file_source(path)

        from app.core.analysis_limits import analysis_max_inmemory_bytes, byte_routing_enabled
        if byte_routing_enabled():
            # Byte-aware routing: tier + the single background trigger keyed on
            # the estimated UNCOMPRESSED in-memory footprint (rows never predict
            # cost; bytes/row does). Probe failure falls back to on-disk size x
            # ANALYSIS_MEM_EXPANSION so a huge unprobeable file still defers.
            probe_bpr = 0.0 if is_multi else _probe_inmemory_bytes_per_row(path, source_type, **kwargs)
            if probe_bpr > 0:
                estimated_full_bytes_route = int(estimated_rows * probe_bpr)
            else:
                try:
                    _expansion = float(os.getenv("ANALYSIS_MEM_EXPANSION", "4.0"))
                except ValueError:
                    _expansion = 4.0
                estimated_full_bytes_route = int(size_mb * 1024 * 1024 * _expansion)
            tier = _determine_tier_bytes(estimated_full_bytes_route, is_multi)
            needs_background = is_multi or estimated_full_bytes_route >= analysis_max_inmemory_bytes()
            logger.info(
                f"[sample_quick] byte_routing path={path!r} tier={tier} "
                f"estimated_rows={estimated_rows:,} bytes_per_row={probe_bpr:.1f} "
                f"estimated_full_bytes={estimated_full_bytes_route:,} "
                f"cap={analysis_max_inmemory_bytes():,} is_multi={is_multi} "
                f"needs_background={needs_background}"
            )
        else:
            tier = _determine_tier(estimated_rows, is_multi)
            needs_background = _should_run_background(tier)

            logger.info(
                f"[sample_quick] path={path!r} tier={tier} "
                f"estimated_rows={estimated_rows:,} is_multi={is_multi} "
                f"needs_background={needs_background}"
            )

        # Cloud paths that return 0 rows are likely auth failures — default to background
        # so we don't silently return an empty result
        _is_cloud_path = any(path.startswith(p) for p in ("gs://", "s3://", "az://", "http://", "https://"))
        if _is_cloud_path and estimated_rows == 0:
            logger.warning(
                f"[sample_quick] Cloud path {path!r} returned estimated_rows=0 — "
                "metadata likely failed. Forcing background job as a safe default."
            )
            tier = TIER_MASSIVE
            needs_background = True

        # Small dataset: run the full algorithm now, no background needed
        if not needs_background:
            result = sample_with_profiling(
                path=path,
                source_type=source_type,
                use_ray=use_ray,
                **kwargs,
            )
            result["profile_status"] = "full_profile"
            result["accuracy"] = "full"
            result["should_run_background"] = False
            result["estimated_rows"] = estimated_rows
            result["tier"] = tier
            # New sampler-state keys — map from existing keys for backward compat
            result.setdefault("quick_sample_rows", result.get("rows", []))
            result.setdefault("full_sample_rows", result.get("rows", []))
            legacy_pr = result.get("profiling_result") or {}
            result.setdefault("sample_statistics", {
                "column_statistics": legacy_pr.get("column_statistics", {}),
                "data_quality": legacy_pr.get("data_quality", {}),
                "data_shape": legacy_pr.get("data_shape", {}),
            })
            result.setdefault("sample_status", "full_sample")
            return result

        # Large dataset: read only the first N_QUICK rows for the quick preview
        import daft

        df_lazy, temp_files = _load_file_with_daft(path, source_type, **kwargs)
        if df_lazy is None:
            raise RuntimeError(f"Could not load file: {path}")

        df_limited = df_lazy.limit(N_QUICK)
        full_df = df_limited.collect()

        actual_rows = full_df.count_rows()
        display_size, portfolio_size = calculate_adaptive_sample_sizes(estimated_rows)

        # Measured in-memory cost signal for byte-aware routing: bytes/row from
        # the already-collected quick slice (Arrow), projected to the whole frame.
        try:
            _arrow_tbl = full_df.to_arrow()
            bytes_per_row = _arrow_tbl.nbytes / max(actual_rows, 1)
        except Exception:
            bytes_per_row = 0.0
        estimated_full_bytes = int(estimated_rows * bytes_per_row)

        # create_sample_portfolio adds _weight per sample; don't add it here
        base_sample = full_df.sample(fraction=min(1.0, portfolio_size / max(actual_rows, 1)))

        portfolio = create_sample_portfolio(
            base_sample=base_sample,
            full_df=full_df,
            max_samples=5,
            sample_size_per=portfolio_size,
            total_rows=actual_rows,
        )

        stats_result = aggregate_statistics_from_portfolio(portfolio, base_sample)

        baseline_dict = portfolio.get("random_baseline")
        if baseline_dict is not None:
            bl = baseline_dict.limit(display_size).to_pydict()
            n = len(next(iter(bl.values()))) if bl else 0
            sample_rows = [
                {col: bl[col][i] for col in bl if col != "_weight"}
                for i in range(n)
            ]
        else:
            bl = base_sample.limit(display_size).to_pydict()
            n = len(next(iter(bl.values()))) if bl else 0
            sample_rows = [
                {col: bl[col][i] for col in bl if col != "_weight"}
                for i in range(n)
            ]

        schema_fields = full_df.schema()
        schema = {field.name: str(field.dtype) for field in schema_fields}
        cols = list(schema.keys())

        ddl_lines = ["CREATE TABLE dataset ("]
        for col_name in schema:
            ddl_lines.append(f"    {col_name} TEXT,")
        if ddl_lines[-1].endswith(","):
            ddl_lines[-1] = ddl_lines[-1][:-1]
        ddl_lines.append(");")

        sample_pct = (actual_rows / max(estimated_rows, 1)) * 100

        profiling_result = {
            "data_shape": {"rows": estimated_rows, "columns": len(cols)},
            "column_statistics": stats_result.get("column_statistics", {}),
            "data_quality": stats_result.get("data_quality", {}),
            "portfolio_metadata": {
                "num_samples": len(portfolio),
                "sample_names": list(portfolio.keys()),
                "portfolio_sample_size": portfolio_size,
                "display_sample_size": display_size,
                "adaptive_sizing_used": True,
                "sizing_tier": tier,
                "quick_mode": True,
                "rows_read": actual_rows,
                "sampled_percentage": f"{sample_pct:.2f}%",
            }
        }

        # Clean up any temp files created during CSV UTF-8 conversion
        # (uses the module-level os import — a function-local `import os` here
        # made `os` local to ALL of sample_quick, so the byte-routing block's
        # earlier os.getenv raised UnboundLocalError)
        for fp in (temp_files or []):
            if fp and os.path.exists(fp):
                try:
                    os.unlink(fp)
                except Exception:
                    pass

        # Build the sample_statistics dict that profiling_agent expects
        sample_statistics = {
            "column_statistics": stats_result.get("column_statistics", {}),
            "data_quality": stats_result.get("data_quality", {}),
            "data_shape": {"rows": estimated_rows, "columns": len(cols)},
            # bytes/row measured on the quick slice; estimated_full_bytes is the
            # whole frame's estimated uncompressed in-memory footprint
            "bytes_per_row": round(bytes_per_row, 2),
            "estimated_full_bytes": estimated_full_bytes,
        }

        return {
            "schema": schema,
            "rows": sample_rows,
            "ddl_schema": "\n".join(ddl_lines),
            "profiling_result": profiling_result,
            "error": None,
            "profile_status": "quick_profile",
            "accuracy": "quick",
            "should_run_background": True,
            "estimated_rows": estimated_rows,
            "tier": tier,
            "sample_pct": round(sample_pct, 4),
            # New sampler-state keys
            "quick_sample_rows": sample_rows,
            "sample_statistics": sample_statistics,
            "sample_status": "quick_sample",
        }

    except Exception as exc:
        logger.error(f"[sample_quick] failed for {path!r}: {exc}", exc_info=True)
        return {
            "schema": {},
            "rows": [],
            "ddl_schema": "",
            "profiling_result": {},
            "error": str(exc),
            "profile_status": "error",
            "accuracy": "none",
            "should_run_background": False,
            "estimated_rows": 0,
            "tier": TIER_TINY,
            "sample_pct": 0.0,
            # New sampler-state keys
            "quick_sample_rows": [],
            "sample_statistics": {},
            "sample_status": "error",
        }


# ---------------------------------------------------------------------------
# Background sampling via Ray+Daft on GKE
# ---------------------------------------------------------------------------

_RAY_THRESHOLD_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB

RAYJOB_YAML_PATH = str(
    Path(__file__).resolve().parent.parent / "infra" / "rayjob.yaml"
)


def build_sampling_ray_script(source_type: str = "csv") -> str:
    """
    Return a self-contained Python script that runs on a Ray cluster.

    Reads the entire dataset in a single pass across workers and builds the
    portfolio + statistics on the driver from merged partial results. The
    partitioning is format-aware: CSV/JSONL split into line-aligned byte
    ranges, Parquet splits by row groups, and JSON is read whole.
    """
    fmt = (source_type or "csv").lower()
    if fmt not in ("csv", "parquet", "json", "jsonl"):
        fmt = "csv"

    return f'SOURCE_TYPE = "{fmt}"\n' + '''\
import os, sys, json, time, math, io, random, traceback
import ray
import pandas as pd
import fsspec
from pathlib import Path
from collections import Counter
from ray.util import get_node_ip_address

# ── GCS auth (same pattern as execution agent) ────────────────────────
sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()
if sa_json and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    sa_json = sa_json.replace("\\\\n", "\\n")
    try:
        sa_json = json.dumps(json.loads(sa_json))
    except Exception:
        pass
    _sa_path = "/tmp/gcp_sa.json"
    Path(_sa_path).write_text(sa_json, encoding="utf-8")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _sa_path

_sa_path = None
if os.path.exists("gcp_sa.json"):
    _sa_path = os.path.abspath("gcp_sa.json")
elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    _sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if _sa_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _sa_path
    print(f"GCS auth set: {_sa_path}")
else:
    print("WARNING: No GCS credentials found")

ray.init(address="auto")

data_uri     = os.getenv("DATA_SOURCE_URI", "").strip()
artifact_uri = os.getenv("OUTPUT_ARTIFACT_URI", "").strip()
metrics_uri  = os.getenv("OUTPUT_METRICS_URI", "").strip()

print("DATA_SOURCE_URI =", repr(data_uri))
print("OUTPUT_ARTIFACT_URI =", repr(artifact_uri))

# ── Auth helpers (same as execution agent) ────────────────────────────
def _gcs_token():
    bundled = "gcp_sa.json"
    if os.path.exists(bundled):
        return os.path.abspath(bundled)
    s = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()
    if not s:
        return None
    try:
        s = json.dumps(json.loads(s.replace("\\\\n", "\\n")))
    except Exception:
        pass
    p = "/tmp/gcp_sa.json"
    Path(p).write_text(s)
    return p

def _storage_options(uri):
    if uri.startswith("gs://") or uri.startswith("gcs://"):
        t = _gcs_token()
        if t:
            return {"token": t}
    return {}

# ── Constants (aligned with sampling_agent_daft.py) ────────────────────
MAX_PORTFOLIO_SAMPLES        = 5
NUM_QUANTILES                = 4
K_CAP                        = 100_000
LOW_CARDINALITY_THRESHOLD    = 20
MEDIUM_CARDINALITY_THRESHOLD = 100
ROWS_PER_PARTITION           = 500_000

def calculate_adaptive_sample_sizes(n):
    if n < 1_000:       return n, n
    elif n < 100_000:   return 1_000, 10_000
    elif n < 1_000_000: return 2_000, 20_000
    elif n < 10_000_000:return 10_000, 50_000
    else:               return 20_000, 100_000

t_start = time.time()

# ── Wait for at least 1 worker ────────────────────────────────────────
deadline = time.time() + 240
while True:
    alive = [n for n in ray.nodes() if n.get("Alive")]
    workers = [n for n in alive if float((n.get("Resources") or {}).get("CPU", 0)) > 0]
    worker_ips = sorted({n.get("NodeManagerAddress") for n in workers if n.get("NodeManagerAddress")})
    print(f"worker_nodes = {worker_ips}  count = {len(workers)}")
    if len(workers) >= 1:
        break
    if time.time() > deadline:
        raise RuntimeError(f"No workers within 240s. workers={worker_ips}")
    time.sleep(3)

print(f"cluster_resources = {ray.cluster_resources()}")

# ── Detect file size + calculate partitions ───────────────────────────
def _get_file_size(uri):
    try:
        opts = _storage_options(uri)
        with fsspec.open(uri, "rb", **opts) as fh:
            fh.seek(0, 2)
            return fh.tell()
    except Exception:
        return 0

def _measure_header_and_row_bytes(uri):
    """Return (header_len_bytes, avg_data_row_bytes) from a small head read."""
    try:
        opts = _storage_options(uri)
        with fsspec.open(uri, "rb", **opts) as fh:
            header = fh.readline()
            sample = fh.read(1_000_000)
        lines = [l for l in sample.split(b"\\n") if l]
        if len(lines) > 1:
            lines = lines[:-1]  # drop the trailing partial line
        avg = (sum(len(l) + 1 for l in lines) / len(lines)) if lines else 130.0
        return len(header), max(avg, 1.0)
    except Exception:
        return 0, 130.0

total_bytes = _get_file_size(data_uri)
if total_bytes <= 0:
    total_bytes = int(os.getenv("FILE_SIZE_BYTES", "5000000000"))

TARGET_SAMPLED_ROWS = K_CAP * (MAX_PORTFOLIO_SAMPLES + NUM_QUANTILES)

chunk_bytes = 0          # csv / jsonl
num_row_groups = 0       # parquet
rgs_per_part = 0         # parquet

if SOURCE_TYPE == "parquet":
    import pyarrow.parquet as pq
    _opts = _storage_options(data_uri)
    with fsspec.open(data_uri, "rb", **_opts) as fh:
        md = pq.ParquetFile(fh).metadata
    num_row_groups = max(md.num_row_groups, 1)
    estimated_total_rows = max(md.num_rows, 1)
    rows_per_rg = max(estimated_total_rows / num_row_groups, 1)
    rgs_per_part = max(1, int(ROWS_PER_PARTITION / rows_per_rg))
    total_partitions = max(1, math.ceil(num_row_groups / rgs_per_part))
    print(f"Parquet: {num_row_groups} row groups, {rgs_per_part} groups/partition")
elif SOURCE_TYPE == "json":
    # a JSON array/records file is not line-splittable; read it in one pass
    header_len, avg_row_bytes = _measure_header_and_row_bytes(data_uri)
    estimated_total_rows = max(int(total_bytes / avg_row_bytes), 1)
    total_partitions = 1
    print("JSON: single-partition read")
else:
    # csv / jsonl: line-aligned byte ranges (every row read exactly once)
    header_len, avg_row_bytes = _measure_header_and_row_bytes(data_uri)
    data_bytes = max(total_bytes - (header_len if SOURCE_TYPE == "csv" else 0), 0)
    estimated_total_rows = max(int(data_bytes / avg_row_bytes), 1)
    partition_bytes = max(int(ROWS_PER_PARTITION * avg_row_bytes), 1_000_000)
    total_partitions = max(1, math.ceil(total_bytes / partition_bytes))
    chunk_bytes = math.ceil(total_bytes / total_partitions)
    print(f"{SOURCE_TYPE}: avg row {avg_row_bytes:.1f}B, chunk {chunk_bytes:,}B")

sample_frac = min(1.0, TARGET_SAMPLED_ROWS / max(estimated_total_rows, 1))

print(f"Total file size: {total_bytes:,} bytes")
print(f"Estimated total rows: {estimated_total_rows:,}")
print(f"Total partitions: {total_partitions}")
print(f"Sample fraction: {sample_frac:.6f} (target {TARGET_SAMPLED_ROWS:,} rows)")

# ── Partition task (byte-offset reads, same pattern as execution agent) ─
@ray.remote(num_cpus=1, max_retries=3)
def process_partition(uri, partition_id, total, fmt, start_byte, end_byte, rg_start, rg_end, samp_frac):
    import pandas as pd, fsspec, os, io, time, json, math
    from pathlib import Path
    from ray.util import get_node_ip_address

    opts = {}
    if uri.startswith("gs://") or uri.startswith("gcs://"):
        bundled = "gcp_sa.json"
        if os.path.exists(bundled):
            opts = {"token": os.path.abspath(bundled)}
        else:
            sa = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()
            if sa:
                try:
                    sa = json.dumps(json.loads(sa.replace("\\\\n", "\\n")))
                except Exception:
                    pass
                p = f"/tmp/gcp_sa_{partition_id}.json"
                Path(p).write_text(sa)
                opts = {"token": p}

    node_ip = get_node_ip_address()
    t0 = time.time()
    print(f"Partition {partition_id}/{total} ({fmt}) on node {node_ip}")

    if fmt == "parquet":
        import pyarrow.parquet as pq
        with fsspec.open(uri, "rb", **opts) as fh:
            pf = pq.ParquetFile(fh)
            groups = list(range(rg_start, min(rg_end, pf.num_row_groups)))
            chunk = pf.read_row_groups(groups).to_pandas() if groups else pd.DataFrame()
    elif fmt == "json":
        with fsspec.open(uri, "rb", **opts) as fh:
            data = fh.read()
        try:
            chunk = pd.read_json(io.BytesIO(data))
        except ValueError:
            chunk = pd.read_json(io.BytesIO(data), lines=True)
    else:
        # csv / jsonl: line-aligned byte range (every row read exactly once)
        has_header = (fmt == "csv")
        header = None
        with fsspec.open(uri, "rb", **opts) as fh:
            if has_header:
                header_bytes = fh.readline()
                header = header_bytes.decode("utf-8", errors="replace").rstrip("\\n\\r").split(",")
                header_end = fh.tell()
            else:
                header_end = 0
            if start_byte <= header_end:
                pos = header_end
            else:
                fh.seek(start_byte)
                fh.readline()
                pos = fh.tell()
            if pos >= end_byte:
                raw = b""
            else:
                fh.seek(pos)
                raw = fh.read(end_byte - pos)
                raw += fh.readline()

        if raw and fmt == "jsonl":
            chunk = pd.read_json(io.BytesIO(raw), lines=True)
        elif raw:
            buf = io.BytesIO(raw)
            try:
                chunk = pd.read_csv(buf, names=header, encoding="utf-8")
            except UnicodeDecodeError:
                buf.seek(0)
                chunk = pd.read_csv(buf, names=header, encoding="latin1")
        else:
            chunk = pd.DataFrame(columns=header) if header else pd.DataFrame()

    actual_rows = len(chunk)
    columns = list(chunk.columns)

    col_stats = {}
    for col in columns:
        series = chunk[col]
        total_c = len(series)
        nulls = int(series.isna().sum())
        non_null = series.dropna()
        stat = {"total_count": total_c, "null_count": nulls, "numeric_count": 0}

        try:
            nums = pd.to_numeric(non_null, errors="coerce")
            nums = nums[~nums.isin([float("inf"), float("-inf")])].dropna()
            if len(nums) > 0:
                stat["numeric_count"] = int(len(nums))
                stat["numeric_sum"] = float(nums.sum())
                stat["numeric_sum_sq"] = float((nums ** 2).sum())
                stat["numeric_min"] = float(nums.min())
                stat["numeric_max"] = float(nums.max())
        except Exception:
            pass

        if len(non_null) > 0:
            vc = non_null.astype(str).value_counts()
            stat["value_counts"] = vc.head(500).to_dict()
            stat["approx_unique"] = int(len(vc))
        else:
            stat["value_counts"] = {}
            stat["approx_unique"] = 0

        col_stats[col] = stat

    k = max(1, int(math.ceil(actual_rows * samp_frac)))
    k = min(k, actual_rows)
    sampled = chunk.sample(n=k, random_state=42) if actual_rows > 0 else chunk

    def _safe(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    sampled_records = []
    for _, row in sampled.iterrows():
        sampled_records.append({c: _safe(row[c]) for c in columns})

    elapsed = time.time() - t0
    print(f"Partition {partition_id} done: {actual_rows} rows, "
          f"{len(sampled_records)} sampled in {elapsed:.1f}s on {node_ip}")

    return {
        "partition_id": partition_id,
        "rows": actual_rows,
        "node_ip": node_ip,
        "elapsed_s": round(elapsed, 2),
        "columns": columns,
        "col_stats": col_stats,
        "sampled_rows": sampled_records,
    }

# ── Submit all partitions ─────────────────────────────────────────────
futures = []
for i in range(total_partitions):
    if SOURCE_TYPE == "parquet":
        start_byte = end_byte = 0
        rg_start = i * rgs_per_part
        rg_end = min((i + 1) * rgs_per_part, num_row_groups)
    else:
        rg_start = rg_end = 0
        start_byte = i * chunk_bytes
        end_byte = min((i + 1) * chunk_bytes, total_bytes)
    futures.append(process_partition.remote(
        data_uri, i + 1, total_partitions, SOURCE_TYPE,
        start_byte, end_byte, rg_start, rg_end, sample_frac,
    ))

print(f"Submitted {total_partitions} partitions across workers")
partition_results = ray.get(futures)

nodes_used = sorted({r["node_ip"] for r in partition_results})
print(f"NODES USED = {nodes_used}")
print(f"NUM NODES = {len(nodes_used)}")
for r in partition_results:
    print(f"  Partition {r['partition_id']}: {r['rows']} rows, "
          f"{len(r['sampled_rows'])} sampled on {r['node_ip']} in {r['elapsed_s']}s")

# ── Merge partial stats from all partitions ───────────────────────────
columns = partition_results[0]["columns"] if partition_results else []
schema_map = {c: "string" for c in columns}

all_col_stats = {}
all_sampled_rows = []

for r in partition_results:
    all_sampled_rows.extend(r["sampled_rows"])

    for col, stat in r["col_stats"].items():
        if col not in all_col_stats:
            all_col_stats[col] = {
                "total_count": 0,
                "null_count": 0,
                "numeric_count": 0,
                "numeric_sum": 0.0,
                "numeric_sum_sq": 0.0,
                "numeric_min": float("inf"),
                "numeric_max": float("-inf"),
                "value_counts": {},
            }
        m = all_col_stats[col]
        m["total_count"] += stat["total_count"]
        m["null_count"] += stat["null_count"]
        m["numeric_count"] += stat.get("numeric_count", 0)
        if stat.get("numeric_count", 0) > 0:
            m["numeric_sum"] += stat.get("numeric_sum", 0.0)
            m["numeric_sum_sq"] += stat.get("numeric_sum_sq", 0.0)
            m["numeric_min"] = min(m["numeric_min"], stat.get("numeric_min", float("inf")))
            m["numeric_max"] = max(m["numeric_max"], stat.get("numeric_max", float("-inf")))
        for val, cnt in stat.get("value_counts", {}).items():
            m["value_counts"][val] = m["value_counts"].get(val, 0) + cnt

# numeric only if most non-null values across the dataset parse as numbers
for col, m in all_col_stats.items():
    non_null = m["total_count"] - m["null_count"]
    m["is_numeric"] = non_null > 0 and m["numeric_count"] >= 0.9 * non_null
    if m["is_numeric"]:
        schema_map[col] = "double"

total_rows = 0
if all_col_stats:
    first_col = next(iter(all_col_stats))
    total_rows = all_col_stats[first_col]["total_count"]

print(f"Total rows (exact): {total_rows:,}")
print(f"Total sampled rows: {len(all_sampled_rows):,}")

# ── Adaptive sizing ───────────────────────────────────────────────────
display_sample_size, portfolio_sample_size = calculate_adaptive_sample_sizes(total_rows)
print(f"Adaptive sizes: display={display_sample_size:,}, portfolio={portfolio_sample_size:,}")

# ── Identify stratification candidates ────────────────────────────────
categorical_candidates = []
numeric_candidates = []

for col, stats in all_col_stats.items():
    if stats["is_numeric"]:
        n_count = stats["numeric_count"]
        if n_count <= LOW_CARDINALITY_THRESHOLD:
            continue
        n_sum = stats["numeric_sum"]
        n_sum_sq = stats["numeric_sum_sq"]
        mean = n_sum / n_count if n_count > 0 else 0
        variance = max(0, n_sum_sq / n_count - mean ** 2) if n_count > 0 else 0
        std = variance ** 0.5

        skew = 0.0
        if std > 0 and n_count > 2:
            col_vals = []
            for row in all_sampled_rows:
                try:
                    v = row.get(col)
                    if v is not None:
                        fv = float(v)
                        if not math.isnan(fv) and not math.isinf(fv):
                            col_vals.append(fv)
                except (ValueError, TypeError):
                    pass
            if len(col_vals) > 2:
                s_mean = sum(col_vals) / len(col_vals)
                s_var = sum((x - s_mean) ** 2 for x in col_vals) / len(col_vals)
                s_std = s_var ** 0.5
                if s_std > 0:
                    third = sum((x - s_mean) ** 3 for x in col_vals) / len(col_vals)
                    skew = abs(third / (s_std ** 3))

        numeric_candidates.append((col, skew, n_count))
    else:
        merged_vc = stats["value_counts"]
        n_unique = len(merged_vc)
        if n_unique < 2:
            continue
        if n_unique <= LOW_CARDINALITY_THRESHOLD:
            categorical_candidates.append((col, "categorical_low", n_unique))
        elif n_unique <= MEDIUM_CARDINALITY_THRESHOLD:
            categorical_candidates.append((col, "categorical_medium", n_unique))

priority_map = {"categorical_low": 0, "categorical_medium": 1}
categorical_candidates.sort(key=lambda x: (priority_map.get(x[1], 99), x[2]))
numeric_candidates.sort(key=lambda x: (-x[1], -x[2]))

strat_cols = []
for col, cat_type, _ in categorical_candidates:
    if len(strat_cols) >= MAX_PORTFOLIO_SAMPLES - 1:
        break
    strat_cols.append(col)

quantile_col = None
if numeric_candidates and len(strat_cols) < MAX_PORTFOLIO_SAMPLES - 1:
    quantile_col = numeric_candidates[0][0]

print(f"Stratification candidates: {strat_cols}")
print(f"Quantile candidate: {quantile_col}")

# ── Build portfolio from sampled rows ─────────────────────────────────
portfolio = {}

random.shuffle(all_sampled_rows)
portfolio["random_baseline"] = all_sampled_rows[:display_sample_size]
print(f"random_baseline: {len(portfolio['random_baseline'])} rows")

for strat_col in strat_cols:
    try:
        group_map = {}
        for row in all_sampled_rows:
            g = str(row.get(strat_col, "")) if row.get(strat_col) is not None else "__null__"
            group_map.setdefault(g, []).append(row)

        per_group = max(1, K_CAP // max(len(group_map), 1))
        sampled_rows = []
        for g, rows in group_map.items():
            sampled_rows.extend(rows[:per_group])

        name = f"strat_{strat_col}"
        portfolio[name] = sampled_rows[:display_sample_size]
        print(f"{name}: {len(portfolio[name])} rows")
    except Exception as exc:
        print(f"Skipping stratified sample for {strat_col}: {exc}")

if quantile_col:
    try:
        def _to_float(v):
            try:
                if v is None:
                    return None
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    return None
                return fv
            except (ValueError, TypeError):
                return None

        numeric_pairs = []
        for i, row in enumerate(all_sampled_rows):
            fv = _to_float(row.get(quantile_col))
            if fv is not None:
                numeric_pairs.append((fv, i))

        numeric_pairs.sort(key=lambda x: x[0])
        quantile_rows = []
        if numeric_pairs:
            rows_per_quantile = max(1, display_sample_size // NUM_QUANTILES)
            for q in range(NUM_QUANTILES):
                start = (len(numeric_pairs) * q) // NUM_QUANTILES
                end = (len(numeric_pairs) * (q + 1)) // NUM_QUANTILES
                bucket = numeric_pairs[start:end]
                if not bucket:
                    continue
                step = max(1, len(bucket) // rows_per_quantile)
                chosen = bucket[::step][:rows_per_quantile]
                for _, idx in chosen:
                    quantile_rows.append(all_sampled_rows[idx])

        if quantile_rows:
            name = f"quantile_{quantile_col}"
            portfolio[name] = quantile_rows[:display_sample_size]
            print(f"{name}: {len(portfolio[name])} rows")
    except Exception as exc:
        print(f"Skipping quantile sample for {quantile_col}: {exc}")

# ── Compute column statistics from merged stats + sampled rows ────────
def _safe_val(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v

bl_data = portfolio["random_baseline"]
column_statistics = {}

for col in columns:
    try:
        merged = all_col_stats.get(col, {})
        n_total = merged.get("total_count", 0)
        n_null = merged.get("null_count", 0)
        is_numeric = bool(merged.get("is_numeric"))
        vc = merged.get("value_counts", {})

        # Keys must match what profiling_agent.profile_full consumes:
        # canonical type ("numeric"/"categorical"), missing_ratio and n_unique —
        # not schema-map "double"/"string", null_pct or unique_count, which the
        # profiler ignores (reporting 0% missing / 0 unique for every column).
        stat = {
            "type": "numeric" if is_numeric else "categorical",
            "null_count": n_null,
            "missing_ratio": round(n_null / max(n_total, 1), 4),
            "total_count": n_total,
            # distinct count from merged value counts, not the numeric row count
            "n_unique": len(vc),
        }

        if is_numeric:
            nc = merged.get("numeric_count", 0)
            if nc > 0:
                mean = merged["numeric_sum"] / nc
                variance = max(0, merged["numeric_sum_sq"] / nc - mean ** 2)
                std = variance ** 0.5
                stat["mean"] = _safe_val(mean)
                stat["std"] = _safe_val(std)
                stat["min"] = _safe_val(merged["numeric_min"])
                stat["max"] = _safe_val(merged["numeric_max"])

                col_vals = []
                for row in bl_data:
                    try:
                        v = row.get(col)
                        if v is not None:
                            col_vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
                col_vals = [v for v in col_vals if not math.isnan(v) and not math.isinf(v)]
                if col_vals:
                    col_vals.sort()
                    stat["percentile_25"] = _safe_val(col_vals[len(col_vals) // 4])
                    stat["percentile_75"] = _safe_val(col_vals[(3 * len(col_vals)) // 4])
                    if std > 0 and len(col_vals) > 2:
                        s_mean = sum(col_vals) / len(col_vals)
                        third = sum((x - s_mean) ** 3 for x in col_vals) / len(col_vals)
                        s_var = sum((x - s_mean) ** 2 for x in col_vals) / len(col_vals)
                        s_std = s_var ** 0.5
                        stat["skewness"] = _safe_val(third / (s_std ** 3)) if s_std > 0 else 0.0
        else:
            top = sorted(vc.items(), key=lambda x: -x[1])[:5]
            stat["top_values"] = [{"value": t[0], "count": t[1]} for t in top]

        column_statistics[col] = stat
    except Exception as exc:
        column_statistics[col] = {"error": str(exc)}

# ── Build output artifact ─────────────────────────────────────────────
runtime_s = round(time.time() - t_start, 3)
ddl_lines = ["CREATE TABLE dataset ("]
for c in columns:
    ddl_lines.append(f"    {c} TEXT,")
if ddl_lines[-1].endswith(","):
    ddl_lines[-1] = ddl_lines[-1][:-1]
ddl_lines.append(");")

artifact = {
    "schema": schema_map,
    "ddl_schema": "\\n".join(ddl_lines),
    "total_rows": total_rows,
    "portfolio_samples": portfolio,
    "available_samples": list(portfolio.keys()),
    "sample_statistics": {
        "column_statistics": column_statistics,
        "data_quality": {},
        "data_shape": {"rows": total_rows, "columns": len(columns)},
    },
    "profiling_result": {
        "data_shape": {"rows": total_rows, "columns": len(columns)},
        "column_statistics": column_statistics,
        "data_quality": {},
        "portfolio_metadata": {
            "num_samples": len(portfolio),
            "sample_names": list(portfolio.keys()),
            "portfolio_sample_size": portfolio_sample_size,
            "display_sample_size": display_sample_size,
            "adaptive_sizing_used": True,
            "quick_mode": False,
            "rows_read": total_rows,
        },
    },
    "metrics": {
        "runtime_s": runtime_s,
        "total_rows": total_rows,
        "columns": len(columns),
        "portfolio_count": len(portfolio),
        "partitions": total_partitions,
        "nodes_used": len(nodes_used),
    },
}

# ── Write to GCS ──────────────────────────────────────────────────────
def _write_json(uri, obj):
    if not uri:
        return
    try:
        opts = _storage_options(uri)
        with fsspec.open(uri, "w", **opts) as f:
            json.dump(obj, f)
        print(f"Wrote {uri}")
    except Exception as e:
        print(f"Warning: could not write {uri}: {e}")
        traceback.print_exc()

_write_json(artifact_uri, artifact)
if metrics_uri:
    _write_json(metrics_uri, artifact["metrics"])

print(f"ARTIFACT_URI={artifact_uri}")
print(f"METRIC runtime_s={runtime_s}")
print(f"METRIC total_rows={total_rows}")
print(f"METRIC portfolio_count={len(portfolio)}")
print(f"METRIC partitions={total_partitions}")
print(f"METRIC nodes_used={len(nodes_used)}")

ray.shutdown()
print("Done.")
'''


def _run_coro_sync(coro):
    """Run an async coroutine synchronously (for use inside Celery workers)."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


async def _update_bg_sampling_session_status(
    session_id: str,
    portfolio_sample_uris: Dict[str, str],
    available_samples: Optional[List[str]] = None,
) -> bool:
    """Update Redis-backed session state inside a single event loop."""
    from app.services import session_service as _ss
    from app.services.session_service import get_session, save_session
    from app.core.cache import RedisCache
    from app.core.settings import Settings

    cache = RedisCache(Settings().redis_url)
    previous_cache = _ss.cache
    try:
        _ss.cache = cache
        sess = await get_session(session_id)
        if not sess:
            logger.warning("[bg_sampling] get_session(%s) returned None — session may have expired", session_id)
            return False

        sess["sample_status"] = "full_sample"
        sess["profiling_status"] = "full_profile"
        if not sess.get("analysis_fidelity") or sess.get("analysis_fidelity") == "quick_sample":
            sess["analysis_fidelity"] = "portfolio_samples"
        if portfolio_sample_uris:
            sess["portfolio_sample_uris"] = json.dumps(portfolio_sample_uris)
        if available_samples:
            sess["available_samples"] = json.dumps(available_samples)

        ok = await save_session(session_id, sess)
        if ok:
            logger.info("[bg_sampling] Updated session %s with full_sample status + %d available_samples", session_id, len(available_samples or []))
        return ok
    finally:
        try:
            await cache.close()
        except Exception:
            pass
        _ss.cache = previous_cache if previous_cache is not cache else None


def run_background_sampling(state: dict) -> dict:
    """
    Celery task handler for ``task_type == "sample_profile"``.

    Submits a Daft+Ray sampling job to GKE, waits for completion, then
    runs lightweight LLM profiling locally and persists everything to
    Supabase.  Updates the Redis session so ``/preview`` reflects the
    new ``sample_status = "full_sample"``.
    """
    dataset_id = state.get("dataset_id") or ""
    cloud_uri = (state.get("data_source_location_cloud") or "").strip()
    connection_id = (
        state.get("connection_id")
        or state.get("cloud_connection_id")
        or state.get("storage_connection_id")
        or ""
    )
    ray_ns = state.get("ray_namespace") or os.getenv("RAY_NAMESPACE", "default")
    ray_workers = int(state.get("ray_workers") or os.getenv("RAY_WORKERS", "3"))
    timeout_s = int(state.get("ray_timeout_s") or os.getenv("RAY_TIMEOUT_S", "3600"))

    logger.info(
        "[bg_sampling] Starting background sampling dataset_id=%s uri=%s",
        dataset_id, cloud_uri,
    )

    if not cloud_uri:
        logger.error("[bg_sampling] No cloud URI — cannot run Ray job")
        return {**state, "error": "No cloud URI for background sampling"}

    # ── Resolve cloud credentials ────────────────────────────────────
    from app.api.cloud_connections import get_cloud_connection
    conn: Dict[str, Any] = {}
    if connection_id:
        try:
            conn = _run_coro_sync(get_cloud_connection(connection_id))
        except Exception as exc:
            logger.error("[bg_sampling] get_cloud_connection failed: %s", exc)

    # ── Determine source type from URI extension ─────────────────────
    ext = cloud_uri.rstrip("/").split(".")[-1].lower()
    if ext not in ("csv", "parquet", "json", "jsonl"):
        ext = "csv"

    # ── Output URIs (write results next to the source) ───────────────
    bucket_and_prefix = cloud_uri.rsplit("/", 1)[0] if "/" in cloud_uri else cloud_uri
    ts = int(time.time())
    artifact_uri = f"{bucket_and_prefix}/_avaloka_sampling/{dataset_id}_{ts}_artifact.json"
    metrics_uri = f"{bucket_and_prefix}/_avaloka_sampling/{dataset_id}_{ts}_metrics.json"

    # ── Build the Ray job script + YAML ──────────────────────────────
    script_text = build_sampling_ray_script(source_type=ext)
    rayjob_name = f"sample-{dataset_id[:8]}-{ts}"

    from app.agents.execution_agent import render_rayjob_yaml
    from app.infra.ray_job_runner import run_rayjob_from_yaml
    from app.infra.k8s_secrets import create_cloud_secret, kubectl_delete_secret
    from app.agents.execution_agent import _normalize_secret_provider

    base_yaml = Path(RAYJOB_YAML_PATH).read_text(encoding="utf-8")

    # Create K8s secret for cloud credentials (KUBERAY mode)
    secret_name = None
    raw_provider = (conn.get("provider") or conn.get("backend") or "gcs").lower()
    provider = _normalize_secret_provider(raw_provider, cloud_uri)
    try:
        secret_name = create_cloud_secret(
            namespace=ray_ns,
            provider=provider,
            creds=conn,
            dataset_id=dataset_id,
        )
        logger.info("[bg_sampling] Created K8s secret %s", secret_name)
    except Exception as exc:
        logger.warning("[bg_sampling] create_cloud_secret failed (will try DIRECT mode): %s", exc)
        secret_name = os.getenv("CLOUD_SECRET_NAME", "cloud-creds")

    yaml_text = render_rayjob_yaml(
        base_yaml=base_yaml,
        name=rayjob_name,
        namespace=ray_ns,
        workers=ray_workers,
        data_uri=cloud_uri,
        script_text=script_text,
        cloud_secret_name=secret_name,
        output_artifact_uri=artifact_uri,
        output_metrics_uri=metrics_uri,
    )

    # ── Submit and wait ──────────────────────────────────────────────
    try:
        outcome = run_rayjob_from_yaml(
            yaml_text=yaml_text,
            namespace=ray_ns,
            rayjob_name=rayjob_name,
            timeout_s=timeout_s,
            poll_interval_s=10,
            cloud_creds=conn,
        )
        logger.info(
            "[bg_sampling] Ray job %s finished: status=%s runtime=%.1fs",
            rayjob_name, outcome.status, outcome.runtime_s or 0,
        )
    except Exception as exc:
        logger.error("[bg_sampling] Ray job submission failed: %s", exc, exc_info=True)
        return {**state, "error": f"Ray job failed: {exc}"}
    finally:
        if secret_name and secret_name != os.getenv("CLOUD_SECRET_NAME", "cloud-creds"):
            try:
                kubectl_delete_secret(secret_name, ray_ns)
            except Exception:
                pass

    if outcome.status != "SUCCEEDED":
        logger.error("[bg_sampling] Ray job did not succeed: %s\n%s", outcome.status, outcome.logs[-2000:])
        return {**state, "error": f"Ray job status: {outcome.status}"}

    # ── Read artifact from GCS ───────────────────────────────────────
    artifact: Dict[str, Any] = {}
    token_info: Optional[Dict[str, Any]] = None
    try:
        import fsspec
        storage_opts: Dict[str, Any] = {}
        if artifact_uri.startswith("gs://"):
            sa_json = (
                conn.get("service_account_json")
                or conn.get("gcp_service_account_json")
                or conn.get("secret_key")
                or ""
            ).strip()
            if sa_json:
                sa_json = sa_json.replace("\\n", "\n").replace("\\r", "").replace("\\t", " ")
                sa_json = re.sub(r"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]", "", sa_json)
                token_info = json.loads(sa_json, strict=False)
                storage_opts = {"token": token_info}

        with fsspec.open(artifact_uri, "r", **storage_opts) as f:
            artifact = json.load(f)
        logger.info("[bg_sampling] Read artifact (%d keys) from %s", len(artifact), artifact_uri)
    except Exception as exc:
        logger.error("[bg_sampling] Could not read artifact from %s: %s", artifact_uri, exc)
        return {**state, "error": f"Could not read sampling artifact: {exc}"}

    portfolio_samples = artifact.get("portfolio_samples") or {}
    available_samples = artifact.get("available_samples") or list(portfolio_samples.keys())
    sample_statistics = artifact.get("sample_statistics") or {}
    profiling_result = artifact.get("profiling_result") or {}
    schema = artifact.get("schema") or {}
    ddl_schema = artifact.get("ddl_schema") or ""
    total_rows = artifact.get("total_rows") or 0

    display_rows = (portfolio_samples.get("random_baseline") or [])[:5000]

    # ── Run lightweight semantic profiling locally ────────────────────
    full_profiling_result: Optional[Dict[str, Any]] = None
    try:
        from app.agents.profiling_agent import profile_full
        schema_list = list(schema.keys()) if isinstance(schema, dict) else list(schema)
        prof_res = profile_full(
            schema=schema_list,
            sample_rows=display_rows,
            sample_statistics=sample_statistics,
        )
        full_profiling_result = prof_res.get("full_profiling_result")
        logger.info("[bg_sampling] Semantic profiling completed")
    except Exception as exc:
        logger.warning("[bg_sampling] profile_full failed (non-fatal): %s", exc)

    # ── Eagerly materialize portfolio samples to cloud for Ray usage ─
    portfolio_sample_uris: Dict[str, str] = {}
    try:
        if cloud_uri and portfolio_samples and dataset_id:
            import fsspec
            scheme = cloud_uri.split("://", 1)[0]
            bucket_and_path = cloud_uri.split("://", 1)[1]
            bucket = bucket_and_path.split("/", 1)[0]
            base_prefix = f"{scheme}://{bucket}/_avaloka_sampling/{dataset_id}/portfolio"

            fs = None
            open_kwargs: Dict[str, Any] = {}
            if scheme == "gs" and token_info:
                fs = fsspec.filesystem("gcs", token=token_info)
                open_kwargs = {"token": token_info}
            else:
                # best-effort: rely on environment credentials for non-GCS
                fs = fsspec.filesystem(scheme)

            for name, rows in (portfolio_samples or {}).items():
                if not rows:
                    continue
                sample_name = str(name)
                uri = f"{base_prefix}/{sample_name}.csv"

                # skip if already exists
                try:
                    if fs.exists(uri): 
                        portfolio_sample_uris[sample_name] = uri
                        continue
                except Exception:
                    pass

                # Write CSV
                cols: List[str] = []
                for r in rows:
                    if isinstance(r, dict):
                        for k in r.keys():
                            if k not in cols:
                                cols.append(k)
                if not cols:
                    continue

                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=cols)
                writer.writeheader()
                for r in rows:
                    if isinstance(r, dict):
                        writer.writerow(r)

                with fsspec.open(uri, "w", **open_kwargs) as f:
                    f.write(buf.getvalue())
                portfolio_sample_uris[sample_name] = uri
    except Exception as exc:
        logger.warning("[bg_sampling] Failed to upload portfolio samples to cloud (non-fatal): %s", exc)

    # ── Persist to Supabase ──────────────────────────────────────────
    try:
        from app.agents.sampling_persistence import persist_full_profile, upsert_profile

        full_result = {
            "schema": schema,
            "rows": display_rows,
            "ddl_schema": ddl_schema,
            "profiling_result": profiling_result,
            "portfolio_samples": portfolio_samples,
            "sample_statistics": sample_statistics,
            "portfolio_sample_uris": portfolio_sample_uris,
        }
        persist_full_profile(
            dataset_id=dataset_id,
            full_result=full_result,
            source_path=cloud_uri,
            source_type=ext,
        )

        if full_profiling_result:
            upsert_profile(
                dataset_id=dataset_id,
                phase="semantic_profile",
                profiling_result=full_profiling_result,
            )

        logger.info("[bg_sampling] Persisted to Supabase for dataset %s", dataset_id)
    except Exception as exc:
        logger.warning("[bg_sampling] Supabase persistence failed (non-fatal): %s", exc)

    # ── Update Redis session (status + available_samples; full data is in Supabase) ──
    try:
        session_id = state.get("session_id") or ""
        if session_id:
            _run_coro_sync(_update_bg_sampling_session_status(
                session_id,
                portfolio_sample_uris,
                available_samples=available_samples,
            ))
    except Exception as exc:
        logger.warning("[bg_sampling] Session update failed (non-fatal): %s", exc)

    logger.info("[bg_sampling] Background sampling complete for %s", dataset_id)
    return {
        **state,
        "sample_status": "full_sample",
        "profiling_status": "full_profile",
        "portfolio_samples": portfolio_samples,
        "available_samples": available_samples,
        "sample_statistics": sample_statistics,
        "profiling_result": profiling_result,
        "full_profiling_result": full_profiling_result,
    }
