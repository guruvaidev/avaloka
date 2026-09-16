"""
loader/url.py — URL / File Dataset Loader
==========================================
Loads tabular data from a URI into either a Ray Dataset (when Ray is available)
or a pandas DataFrame (local fallback).

Supported URI schemes and formats
-----------------------------------
  Local filesystem
    /path/to/file.csv
    /path/to/file.parquet
    /path/to/file.json   (newline-delimited or array JSON)
    /path/to/file.jsonl
    /path/to/file.xlsx   (requires openpyxl)
    /path/to/dir/        (directory of same-format files, detected by extension)

  HTTP / HTTPS
    https://example.com/data.csv
    https://example.com/data.parquet

  Google Cloud Storage
    gs://bucket/path/to/file.csv
    gs://bucket/path/to/file.parquet
    gs://bucket/path/to/dir/

  Amazon S3
    s3://bucket/path/to/file.csv
    s3://bucket/path/to/file.parquet

Format detection
----------------
The file format is inferred from the URI extension. Override it by appending
a query parameter:  gs://bucket/myfile?format=parquet

Supported format tokens: csv, parquet, json, jsonl, ndjson, xlsx, xls

Ray path notes
--------------
  - CSV, Parquet, JSON, JSONL:  delegated to the native Ray read_* APIs which
    handle remote storage, block splitting, and parallelism transparently.
  - XLSX:  downloaded to /tmp, read with pandas, converted via from_pandas().
  - Directories:  Ray handles glob expansion natively for CSV and Parquet.

Local path notes
----------------
  - All formats read with pandas. GCS/S3 URIs use fsspec (gcsfs / s3fs) so no
    manual download is needed; install the relevant package for your cloud.

Environment variables
---------------------
  URL_READ_OPTIONS_CSV  — JSON string of kwargs forwarded to pandas.read_csv /
                          ray.data.read_csv (e.g. '{"sep": "\\t"}')
  GCP_SERVICE_ACCOUNT_JSON — if set, used as the GCS token for fsspec reads
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY — standard S3 credentials for fsspec
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests

# Ray is optional
try:
    import ray
    import ray.data as rd
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False


# =========================================================
# Helpers
# =========================================================


def _detect_format(uri: str) -> str:
    """
    Infer the file format from the URI extension or an explicit ?format= param.

    Returns a normalised lowercase token: csv | parquet | json | jsonl | xlsx.
    Defaults to 'csv' if the extension is unrecognised.
    """
    parsed = urlparse(uri)

    # Explicit override via query param takes priority
    qs = parse_qs(parsed.query)
    if "format" in qs:
        return qs["format"][0].lower().strip(".")

    # Strip query string before checking extension
    path = parsed.path.rstrip("/")
    ext  = Path(path).suffix.lower().lstrip(".")

    FORMAT_MAP = {
        "csv":     "csv",
        "tsv":     "csv",
        "parquet": "parquet",
        "pq":      "parquet",
        "json":    "json",
        "jsonl":   "jsonl",
        "ndjson":  "jsonl",
        "xlsx":    "xlsx",
        "xls":     "xlsx",
    }
    fmt = FORMAT_MAP.get(ext, "csv")
    if ext and ext not in FORMAT_MAP:
        print(f"[URL Loader] Unrecognised extension '.{ext}' — defaulting to CSV.")
    return fmt


def _strip_format_param(uri: str) -> str:
    """Remove the ?format= query param from the URI before passing to readers."""
    from urllib.parse import urlencode, urlparse, parse_qs, urlunparse
    parsed = urlparse(uri)
    qs     = {k: v for k, v in parse_qs(parsed.query).items() if k != "format"}
    flat   = {k: v[0] for k, v in qs.items()}
    return urlunparse(parsed._replace(query=urlencode(flat)))


def _csv_read_options() -> Dict[str, Any]:
    """Parse URL_READ_OPTIONS_CSV from env (JSON string → dict)."""
    raw = os.getenv("URL_READ_OPTIONS_CSV", "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[URL Loader] Warning: URL_READ_OPTIONS_CSV is not valid JSON: {exc}")
        return {}


def _gcs_storage_options() -> Dict[str, Any]:
    """Build fsspec storage_options for GCS (used in pandas path)."""
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()
    if sa_json:
        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
        Path(sa_path).write_text(sa_json, encoding="utf-8")
        return {"token": sa_path}
    # Fall back to ADC (Application Default Credentials)
    adc = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    return {"token": adc} if adc else {}


def _s3_endpoint_url() -> str:
    """Endpoint of an S3-compatible store (in-cluster MinIO, Ceph RGW), if any.

    Empty means real AWS S3. Ray workers read the same s3:// URIs the API writes,
    so they need this too — without it a worker silently tries to resolve the
    bucket on AWS and fails with NoSuchBucket rather than a connection error.
    """
    return (os.getenv("S3_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL", "")).strip()


def _s3_storage_options() -> Dict[str, Any]:
    """Build fsspec storage_options for S3."""
    opts: Dict[str, Any] = {}
    key    = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION", "")
    endpoint = _s3_endpoint_url()
    if key:
        opts["key"] = key
    if secret:
        opts["secret"] = secret
    client_kwargs: Dict[str, Any] = {}
    if region:
        client_kwargs["region_name"] = region
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    if client_kwargs:
        opts["client_kwargs"] = client_kwargs
    return opts


def _storage_options_for(uri: str) -> Dict[str, Any]:
    if uri.startswith("gs://"):
        return _gcs_storage_options()
    if uri.startswith("s3://"):
        return _s3_storage_options()
    return {}


def _is_http(uri: str) -> bool:
    return uri.startswith(("http://", "https://"))


def _is_local(uri: str) -> bool:
    return uri.startswith("/") or uri.startswith("./") or uri.startswith("file://")


def _download_to_tmp(uri: str, suffix: str) -> str:
    """Download a remote file to a temp path. Returns the local path."""
    print(f"[URL Loader] Downloading {uri} ...")
    if _is_http(uri):
        resp = requests.get(uri, timeout=120)
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(resp.content)
        tmp.flush()
        return tmp.name
    if uri.startswith("gs://"):
        import gcsfs
        opts = _gcs_storage_options()
        fs   = gcsfs.GCSFileSystem(**opts)
        tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        fs.get(uri[5:], tmp.name)
        return tmp.name
    if uri.startswith("s3://"):
        import s3fs
        opts = _s3_storage_options()
        fs   = s3fs.S3FileSystem(**opts)
        tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        fs.get(uri[5:], tmp.name)
        return tmp.name
    return uri  # local path — no download needed


# =========================================================
# Public API
# =========================================================


def load_dataset_from_url(uri: str):
    """
    Load a file-based dataset into a Ray Dataset (if Ray is available) or a
    pandas DataFrame (local fallback).

    Parameters
    ----------
    uri  : File URI. Supports local paths, HTTP(S), gs://, s3://.
           Append  ?format=parquet  (or csv/json/jsonl/xlsx) to override
           automatic format detection from the file extension.

    Returns
    -------
    ray.data.Dataset  — when Ray is available and initialised.
    pd.DataFrame      — otherwise (local_trainer.py path).
    """
    fmt      = _detect_format(uri)
    clean_uri = _strip_format_param(uri)

    print(f"[URL Loader] uri={clean_uri}  detected_format={fmt}")

    if _RAY_AVAILABLE and ray.is_initialized():
        return _load_ray(clean_uri, fmt)
    return _load_pandas(clean_uri, fmt)


# =========================================================
# Ray path
# =========================================================


def _load_ray(uri: str, fmt: str) -> "rd.Dataset":
    """
    Delegate to Ray's native read_* APIs where possible.
    Falls back to pandas + from_pandas() for XLSX and other unsupported formats.
    """
    storage_opts = _storage_options_for(uri)

    # ray.data read_* kwargs differ from pandas — only pass supported options
    if fmt == "csv":
        csv_opts = _csv_read_options()
        # Ray's read_csv passes extra kwargs to pyarrow.csv.ReadOptions /
        # ConvertOptions, not pandas — filter to safe common kwargs only
        ray_csv_kwargs: Dict[str, Any] = {}
        if "sep" in csv_opts or "delimiter" in csv_opts:
            delim = csv_opts.get("sep") or csv_opts.get("delimiter", ",")
            ray_csv_kwargs["parse_options"] = {"delimiter": delim}

        ds = rd.read_csv(
            uri,
            filesystem=_ray_filesystem(uri),
            **ray_csv_kwargs,
        )

    elif fmt == "parquet":
        ds = rd.read_parquet(uri, filesystem=_ray_filesystem(uri))

    elif fmt in ("json", "jsonl"):
        ds = rd.read_json(uri, filesystem=_ray_filesystem(uri))

    elif fmt == "xlsx":
        # Ray has no native Excel reader — download and convert via pandas
        local_path = _download_to_tmp(uri, suffix=".xlsx")
        df = pd.read_excel(local_path, engine="openpyxl")
        print(f"[URL Loader] XLSX loaded: {len(df)} rows  {len(df.columns)} cols")
        ds = rd.from_pandas(df)

    else:
        raise ValueError(f"[URL Loader] Unsupported format '{fmt}' for URI: {uri}")

    return ds


def _ray_filesystem(uri: str) -> Optional[Any]:
    """
    Return a PyArrow filesystem for the given URI scheme, or None for local/HTTP.
    Ray's read_* functions accept a pyarrow-compatible filesystem object.
    """
    if uri.startswith("gs://"):
        try:
            from pyarrow.fs import GcsFileSystem
            return GcsFileSystem()
        except ImportError:
            # Fall back to gcsfs-backed filesystem via fsspec
            import gcsfs
            from pyarrow.fs import PyFileSystem, FSSpecHandler
            opts = _gcs_storage_options()
            return PyFileSystem(FSSpecHandler(gcsfs.GCSFileSystem(**opts)))

    if uri.startswith("s3://"):
        try:
            from pyarrow.fs import S3FileSystem
            opts = _s3_storage_options()
            client_kwargs = opts.get("client_kwargs", {})
            kw: Dict[str, Any] = {
                "access_key": opts.get("key"),
                "secret_key": opts.get("secret"),
                "region": client_kwargs.get("region_name"),
            }
            # pyarrow does NOT read AWS_ENDPOINT_URL the way botocore does, so an
            # S3-compatible endpoint has to be passed explicitly or every Ray read
            # goes to AWS. MinIO in-cluster is plain HTTP, hence scheme.
            endpoint = client_kwargs.get("endpoint_url")
            if endpoint:
                kw["endpoint_override"] = endpoint
                kw["scheme"] = "https" if endpoint.startswith("https://") else "http"
            return S3FileSystem(**kw)
        except ImportError:
            import s3fs
            from pyarrow.fs import PyFileSystem, FSSpecHandler
            opts = _s3_storage_options()
            return PyFileSystem(FSSpecHandler(s3fs.S3FileSystem(**opts)))

    return None  # local path or HTTP — Ray handles these natively


# =========================================================
# Local / pandas path
# =========================================================


def _load_pandas(uri: str, fmt: str) -> pd.DataFrame:
    """
    Read the dataset into a pandas DataFrame.
    Remote URIs are handled by fsspec (gcsfs / s3fs) transparently.
    HTTP(S) URIs are downloaded to /tmp first.
    """
    storage_opts = _storage_options_for(uri)

    # pandas can't open HTTP directly for all formats — download to tmp
    if _is_http(uri) and fmt in ("parquet", "xlsx"):
        uri = _download_to_tmp(uri, suffix=f".{fmt}")
        storage_opts = {}

    if fmt == "csv":
        csv_opts = _csv_read_options()
        df = pd.read_csv(uri, storage_options=storage_opts or None, **csv_opts)

    elif fmt == "parquet":
        df = pd.read_parquet(uri, storage_options=storage_opts or None)

    elif fmt == "json":
        # Try records/orient=records first, fall back to lines=True
        try:
            df = pd.read_json(uri, storage_options=storage_opts or None)
        except ValueError:
            df = pd.read_json(uri, lines=True, storage_options=storage_opts or None)

    elif fmt == "jsonl":
        df = pd.read_json(uri, lines=True, storage_options=storage_opts or None)

    elif fmt == "xlsx":
        local_path = _download_to_tmp(uri, suffix=".xlsx") if not _is_local(uri) else uri
        df = pd.read_excel(local_path, engine="openpyxl")

    else:
        raise ValueError(f"[URL Loader] Unsupported format '{fmt}' for URI: {uri}")

    print(f"[URL Loader] Loaded DataFrame: {len(df)} rows  {len(df.columns)} cols")
    return df