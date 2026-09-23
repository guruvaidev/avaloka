"""
Sampling agent built on Daft.

Loads a dataset once into memory (or Ray object store), then builds a portfolio
of stratified and random samples. Statistics are derived from the portfolio so
no further source reads are needed.
"""

from __future__ import annotations
import os
import re
import json
import importlib.util
import logging

import statistics
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import Counter

# Configuration
DEFAULT_SAMPLE_MAX_ROWS = int(os.getenv("DEFAULT_SAMPLE_MAX_ROWS", "1000"))
PORTFOLIO_SAMPLE_SIZE = int(os.getenv("PORTFOLIO_SAMPLE_SIZE", "10000"))
MAX_PORTFOLIO_SAMPLES = int(os.getenv("MAX_PORTFOLIO_SAMPLES", "5"))


def calculate_adaptive_sample_sizes(total_rows: int) -> Tuple[int, int]:
    """
    Calculate variable sample sizes based on dataset size.
    Returns: (display_sample_size, portfolio_sample_size)
    """
    if total_rows < 1_000:       # TINY
        return total_rows, total_rows
    elif total_rows < 100_000:   # SMALL
        return 1_000, 10_000
    elif total_rows < 1_000_000: # LARGE
        return 2_000, 20_000
    elif total_rows < 10_000_000: # HUGE
        return 10_000, 50_000
    else:                        # MASSIVE (>10M)
        return 20_000, 100_000   # Cap display at 20k, portfolio at 100k

logger = logging.getLogger(__name__)

# Daft is loaded lazily, inside the functions that use it.
#
# This module is imported by app/api/server.py at startup, and every consumer
# of `daft` below is inside a function body -- so the only thing the old
# module-level `import daft` bought was the DAFT_AVAILABLE flag, at the cost of
# loading a large Rust extension on every import of the API server.
#
# It cost more than time. Daft's wheels are compiled for a modern x86-64
# baseline, and on a CPU without those instructions the import does not raise
# ImportError -- it dies with SIGILL, "Fatal Python error: Illegal
# instruction", taking the whole process with it. That is not catchable, so the
# only defence is not to execute the extension unless it is needed.
#
# find_spec answers "is the package installed?" without running any of its
# code, which is exactly the question DAFT_AVAILABLE was asking.
DAFT_AVAILABLE = importlib.util.find_spec("daft") is not None
if not DAFT_AVAILABLE:
    logger.warning("Daft not available. Install with: pip install getdaft")


class _LazyDaft:
    """Stands in for the ``daft`` module until something actually uses it.

    Every ``daft.read_csv(...)`` below goes through here and imports the real
    module on first attribute access, so the forty-odd call sites stay exactly
    as they were while the import itself moves to the moment of use.

    A module-level ``__getattr__`` (PEP 562) would not do: that is consulted
    for ``module.attr`` from outside, not for a global name looked up inside
    this module's own functions.
    """

    __slots__ = ()
    _module = None

    @classmethod
    def _real(cls):
        if cls._module is None:
            import daft as _daft          # may raise, or die on SIGILL -- but
            cls._module = _daft           # only for a caller that needs Daft
        return cls._module

    def __getattr__(self, name: str):
        return getattr(_LazyDaft._real(), name)

    # Writes and deletes forward to the real module rather than landing on the
    # proxy. Before this they could not land anywhere at all: __slots__ leaves
    # the instance with no __dict__, so `monkeypatch.setattr(sad.daft,
    # "read_csv", stub)` -- which every test that stubs the reader does -- died
    # with "'_LazyDaft' object has no attribute 'read_csv'". When `daft` was a
    # plain module that patch worked, so the proxy has to be transparent in
    # both directions, not just for reads. monkeypatch then restores the real
    # attribute on the module it actually changed.
    def __setattr__(self, name: str, value) -> None:
        setattr(_LazyDaft._real(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(_LazyDaft._real(), name)


daft = _LazyDaft()

# Import Ray for distributed execution
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not available. Install with: pip install ray")

__all__ = ["sample_with_profiling", "DEFAULT_SAMPLE_MAX_ROWS"]


# Ray connection settings. Set these environment variables before starting the server:
#   AVALOKA_RAY_ADDRESS   — remote Ray cluster address, e.g. ray://server:10001
#   AVALOKA_RAY_CPUS      — number of CPUs to use for a local cluster
#   AVALOKA_RAY_MEMORY_GB — memory limit in GB for a local cluster

RAY_CPUS = os.getenv("AVALOKA_RAY_CPUS")
RAY_MEMORY_GB = os.getenv("AVALOKA_RAY_MEMORY_GB")
RAY_ADDRESS = os.getenv("AVALOKA_RAY_ADDRESS")
#RAY_SHUTDOWN_AFTER = os.getenv("AVALOKA_RAY_SHUTDOWN", "1").strip().lower() not in {"0", "false", "no"}

# Persist the local Ray cluster for the process lifetime. Tearing it down and
# re-initing leaves Daft's process-global runner pointed at the dead cluster's
# scheduler actor (Daft can't rebind), causing "Can't find actor" on the next
# call. Opt back in with AVALOKA_RAY_SHUTDOWN=1 only if you know why.
RAY_SHUTDOWN_AFTER = os.getenv("AVALOKA_RAY_SHUTDOWN", "0").strip().lower() not in {"0", "false", "no"}

_RAY_INITIALIZED = False
_DAFT_RAY_CONFIGURED = False
_RAY_LAST_ERROR: Optional[str] = None
_RAY_LAST_INIT_MODE: Optional[str] = None


def _parse_service_account_json(sa: Any) -> Optional[Dict[str, Any]]:
    """Parse a GCP service-account credential (dict or JSON string) into a dict, or None."""
    if isinstance(sa, dict):
        return sa
    if not isinstance(sa, str) or not sa.strip():
        return None

    cleaned = sa.replace("\\n", "\n").replace("\\r", "").replace("\\t", "\t")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    for candidate in (sa, cleaned):
        for strict in (True, False):
            try:
                return json.loads(candidate, strict=strict)
            except Exception:
                continue
    return None


def _build_cloud_io_config(path: str, cloud_credentials: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Build a Daft IOConfig from existing connection credentials when reading cloud URIs directly."""
    if not cloud_credentials or not isinstance(cloud_credentials, dict):
        return None

    try:
        from daft.io import IOConfig, GCSConfig, S3Config, AzureConfig
    except Exception:
        logger.debug("Daft IOConfig classes unavailable; falling back to default cloud auth")
        return None

    scheme = path.split("://", 1)[0].lower()

    if scheme == "gs":
        sa_json = (
            cloud_credentials.get("secret_key")
            or cloud_credentials.get("service_account_json")
            or cloud_credentials.get("gcp_service_account_json")
            or ""
        )
        project = cloud_credentials.get("project") or cloud_credentials.get("project_id")
        if sa_json:
            # Re-dump to strict JSON: Daft's credential parser rejects the raw
            # newlines a naive unescape leaves inside private_key.
            creds_dict = _parse_service_account_json(sa_json)
            if creds_dict:
                if not project:
                    project = creds_dict.get("project_id")
                sa_json = json.dumps(creds_dict)
            return IOConfig(gcs=GCSConfig(project_id=project, credentials=sa_json))
        return None

    if scheme == "s3":
        access_key_id = cloud_credentials.get("access_key") or cloud_credentials.get("aws_access_key_id")
        secret_access_key = cloud_credentials.get("secret_key") or cloud_credentials.get("aws_secret_access_key")
        session_token = cloud_credentials.get("session_token") or cloud_credentials.get("aws_session_token")
        region = cloud_credentials.get("region") or cloud_credentials.get("region_name")
        endpoint_url = cloud_credentials.get("endpoint_url")
        return IOConfig(
            s3=S3Config(
                region_name=region,
                endpoint_url=endpoint_url,
                key_id=access_key_id,
                access_key=secret_access_key,
                session_token=session_token,
            )
        )

    if scheme == "az":
        account = (
            cloud_credentials.get("account")
            or cloud_credentials.get("account_name")
            or cloud_credentials.get("storage_account")
        )
        access_key = cloud_credentials.get("access_key") or cloud_credentials.get("secret_key")
        sas_token = cloud_credentials.get("sas_token")
        return IOConfig(
            azure=AzureConfig(
                storage_account=account,
                access_key=access_key,
                sas_token=sas_token,
            )
        )

    return None

def _iceberg_fileio_props(table_root: str, cloud_credentials: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """PyIceberg FileIO properties from the connection row.

    S3 keys are the documented PyIceberg names. For GCS we return {} and let
    PyIceberg fall back to ADC/gcsfs ambient auth (the SA-JSON blob in `conn`
    is NOT a property PyIceberg accepts directly) — confirm ADC is configured
    on the box. Azure Iceberg is not wired here.
    """
    props: Dict[str, Any] = {}
    if not isinstance(cloud_credentials, dict):
        return props

    scheme = table_root.split("://", 1)[0].lower()

    if scheme == "s3":
        ak = cloud_credentials.get("access_key") or cloud_credentials.get("aws_access_key_id")
        sk = cloud_credentials.get("secret_key") or cloud_credentials.get("aws_secret_access_key")
        st = cloud_credentials.get("session_token") or cloud_credentials.get("aws_session_token")
        region = cloud_credentials.get("region") or cloud_credentials.get("region_name")
        endpoint = cloud_credentials.get("endpoint_url")
        if ak:
            props["s3.access-key-id"] = ak
        if sk:
            props["s3.secret-access-key"] = sk
        if st:
            props["s3.session-token"] = st
        if region:
            props["s3.region"] = region
        if endpoint:
            props["s3.endpoint"] = endpoint

    # gs:// → rely on ADC / gcsfs ambient credentials (props stays empty).
    return props


def _read_iceberg_table(
    table_root: str,
    metadata_uri: Optional[str],
    cloud_credentials: Optional[Dict[str, Any]],
    io_config: Optional[Any] = None,
) -> "daft.DataFrame":
    """Read an Iceberg table from its metadata JSON via PyIceberg StaticTable."""
    if not metadata_uri:
        raise ValueError(
            "Iceberg read requires metadata_uri (.../metadata/vN.metadata.json)"
        )
    try:
        from pyiceberg.table import StaticTable
    except Exception as e:
        raise ImportError(f"pyiceberg is required to read Iceberg tables: {e}")

    props = _iceberg_fileio_props(table_root, cloud_credentials)
    tbl = StaticTable.from_metadata(metadata_uri, properties=props)

    # daft.read_iceberg accepts io_config on newer versions only.
    try:
        return daft.read_iceberg(tbl, io_config=io_config)
    except TypeError:
        return daft.read_iceberg(tbl)


def _gcs_client_from_creds(cloud_credentials: Optional[Dict[str, Any]]):
    """GCS client using the connection's service-account JSON, else ambient creds."""
    from google.cloud import storage as gcs_sdk  # type: ignore
    if isinstance(cloud_credentials, dict):
        sa_json = (
            cloud_credentials.get("secret_key")
            or cloud_credentials.get("service_account_json")
            or cloud_credentials.get("gcp_service_account_json")
            or ""
        )
        if sa_json:
            info = _parse_service_account_json(sa_json)
            if info:
                project = (
                    cloud_credentials.get("project")
                    or cloud_credentials.get("project_id")
                    or info.get("project_id")
                )
                return gcs_sdk.Client.from_service_account_info(info, project=project)
    return gcs_sdk.Client()


def _s3_client_from_creds(cloud_credentials: Optional[Dict[str, Any]]):
    """boto3 S3 client using the connection's keys, else ambient creds."""
    import boto3  # type: ignore
    if isinstance(cloud_credentials, dict):
        kw = {
            "aws_access_key_id": cloud_credentials.get("access_key") or cloud_credentials.get("aws_access_key_id"),
            "aws_secret_access_key": cloud_credentials.get("secret_key") or cloud_credentials.get("aws_secret_access_key"),
            "aws_session_token": cloud_credentials.get("session_token") or cloud_credentials.get("aws_session_token"),
            "region_name": cloud_credentials.get("region") or cloud_credentials.get("region_name"),
            "endpoint_url": cloud_credentials.get("endpoint_url"),
        }
        kw = {k: v for k, v in kw.items() if v}
        if kw:
            return boto3.client("s3", **kw)
    return boto3.client("s3")


def _azure_client_from_creds(cloud_credentials: Optional[Dict[str, Any]]):
    """Azure BlobServiceClient using the connection's account/key/SAS, falling back to env vars."""
    from azure.storage.blob import BlobServiceClient  # type: ignore
    import os
    account = access_key = sas_token = None
    if isinstance(cloud_credentials, dict):
        account = (
            cloud_credentials.get("account")
            or cloud_credentials.get("account_name")
            or cloud_credentials.get("storage_account")
        )
        access_key = cloud_credentials.get("access_key") or cloud_credentials.get("secret_key")
        sas_token = cloud_credentials.get("sas_token")
    account = account or os.getenv("AZURE_ACCOUNT", "")
    sas_token = sas_token or os.getenv("AZURE_SAS", "")
    url = f"https://{account}.blob.core.windows.net"
    return BlobServiceClient(account_url=url, credential=access_key or sas_token or None)


def init_ray_cluster(num_cpus: Optional[int] = None, memory_gb: Optional[float] = None):
    """
    Initialize Ray cluster.
    
    Behavior:
    - Remote: If AVALOKA_RAY_ADDRESS is set, connects to that cluster.
    - Local (Linux/Server): Auto-detects available resources if not specified.
    
    Args:
        num_cpus: Override CPU count.
        memory_gb: Override memory limit in GB.
    """
    global _RAY_INITIALIZED, _DAFT_RAY_CONFIGURED, _RAY_LAST_ERROR, _RAY_LAST_INIT_MODE
    
    if not RAY_AVAILABLE:
        logger.warning("Ray not installed. Using Daft native executor.")
        return False
    
    # if _RAY_INITIALIZED and _DAFT_RAY_CONFIGURED:
    #     logger.info("Ray+Daft already initialized")
    #     return True

     # If we think Ray is up but the cluster actually died, our cached Daft
    # runner is now stale. Reset so we re-init cleanly rather than submitting
    # to a dead actor.
    if _RAY_INITIALIZED and not (RAY_AVAILABLE and ray.is_initialized()):
        logger.warning("Ray marked initialized but cluster is down; resetting state.")
        _RAY_INITIALIZED = False

    if _RAY_INITIALIZED and _DAFT_RAY_CONFIGURED and ray.is_initialized():
        logger.info("Ray+Daft already initialized; reusing live cluster")
        return True
    
    try:
        _RAY_LAST_ERROR = None
        import platform
        is_mac = platform.system() == "Darwin"
        
        # CPU/memory: prefer explicit args, then env vars, then defaults
        cpus = num_cpus if num_cpus is not None else (int(RAY_CPUS) if RAY_CPUS else None)
        mem_gb = memory_gb if memory_gb is not None else (float(RAY_MEMORY_GB) if RAY_MEMORY_GB else None)
        
        if is_mac:
            if cpus is None: cpus = 4
        
        # Initialize Ray
        if not ray.is_initialized():
            if RAY_ADDRESS:
                _RAY_LAST_INIT_MODE = "remote"
                logger.info(f"🚀 Connecting to remote Ray cluster: {RAY_ADDRESS}")
                # Setup runtime environment with required packages
                runtime_env = {
                    "pip": ["getdaft", "ray"],
                    "env_vars": {"PIP_DISABLE_PIP_VERSION_CHECK": "1"},
                }
                logger.info("📦 Setting up runtime environment with getdaft...")
                ray.init(
                    address=RAY_ADDRESS, 
                    ignore_reinit_error=True,
                    runtime_env=runtime_env,
                )
            else:
                _RAY_LAST_INIT_MODE = "local"
                # Local Cluster - use all available resources on servers
                kwargs = {
                    "ignore_reinit_error": True, 
                    "include_dashboard": True,  # Enable dashboard for monitoring
                    "log_to_driver": True,
                    "dashboard_host": "0.0.0.0",  # Allow external access to dashboard
                }
                
                msg = "🚀 Starting local Ray cluster:"
                if cpus:
                    kwargs["num_cpus"] = cpus
                    msg += f" {cpus} CPUs,"
                else:
                    # Auto-detect all CPUs
                    import multiprocessing
                    available_cpus = multiprocessing.cpu_count()
                    msg += f" {available_cpus} CPUs (auto),"
                    
                if mem_gb:
                    memory_bytes = int(mem_gb * 1024 * 1024 * 1024)
                    kwargs["object_store_memory"] = memory_bytes
                    msg += f" {mem_gb}GB memory"
                else:
                    system_mem = _get_system_memory_gb()
                    msg += f" {system_mem:.1f}GB memory (auto)"
                
                logger.info(msg)
                try:
                    ray.init(**kwargs)
                    logger.info("📊 Ray Dashboard available at: http://127.0.0.1:8265")
                except Exception as e:
                    # NOTE: neither popping include_dashboard nor setting it to
                    # False avoids the ~60s stall here -- measured on ray 2.53,
                    # the retry still spends ~61s trying to start the dashboard
                    # process before falling back ("Failed to start the
                    # dashboard"). The stall is missing ray[default] extras
                    # (aiohttp_cors, opentelemetry.exporter.prometheus); install
                    # those to make this path fast, or avoid init_ray_cluster()
                    # entirely for small inputs.
                    logger.warning(f"Ray init with dashboard failed: {e}. Retrying without dashboard.")
                    kwargs.pop("include_dashboard", None)
                    kwargs.pop("dashboard_host", None)
                    ray.init(**kwargs)
                
            _RAY_INITIALIZED = True
            logger.info("✓ Ray cluster ready")
        else:
            logger.info("✓ Ray already running")
            _RAY_INITIALIZED = True
        
        # Configure Daft
        if not _DAFT_RAY_CONFIGURED:
            try:
                daft.set_runner_ray()
                _DAFT_RAY_CONFIGURED = True
                logger.info("✓ Daft configured to use Ray runner")
            except Exception as e:
                if "Cannot set runner more than once" in str(e):
                    _DAFT_RAY_CONFIGURED = True
                    logger.info("✓ Daft runner already set")
                else:
                    logger.warning(f"Could not configure Daft-Ray: {e}")
                    return False
        
        _log_ray_resources()
        return True
    
    except Exception as e:
        _RAY_LAST_ERROR = str(e)
        logger.error(f"Ray initialization failed: {e}")
        return False


def _log_ray_resources():
    """Log Ray cluster resources."""
    try:
        resources = ray.cluster_resources()
        cpus = resources.get("CPU", 0)
        mem_gb = resources.get("memory", 0) / (1024**3)
        obj_store_gb = resources.get("object_store_memory", 0) / (1024**3)
        logger.info(f"  Resources: {cpus:.0f} CPUs, {mem_gb:.1f}GB memory, {obj_store_gb:.1f}GB object store")
    except Exception as e:
        logger.warning(f"Could not get Ray resources: {e}")


def _get_system_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        try:
            return (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) / (1024**3)
        except:
            return 16.0  # Assume 16 GB if detection fails


def should_use_ray(
    total_rows: int = 0,
    file_size_mb: float = 0.0,
    threshold_rows: int = 50_000,
    file_path: str = "",
) -> bool:
    """
    Decide whether to use Ray for this dataset.

    Daft native is already parallel on a single machine. Ray mainly helps when:
      - AVALOKA_RAY_ADDRESS is set (remote/multi-node cluster)
      - Reading cloud files (gs://, s3://, az://,...) to parallelise network I/O
    """

    if not RAY_AVAILABLE:
        return False

    is_cloud_file = bool(file_path) and any(file_path.startswith(p) for p in ["gs://", "s3://", "az://", "http://", "https://"])

    if RAY_ADDRESS:
        return True

    # # Cloud files: Ray parallelises Daft's I/O across workers for faster reading
    # if is_cloud_file:
    #     return True

    # Cloud files benefit from Ray only when large enough to be worth the
    # cluster startup + parallel I/O. A tiny partitioned folder does not.
    if is_cloud_file and (file_size_mb >= 256 or (total_rows and total_rows >= threshold_rows)):
        return True


    if total_rows and total_rows < threshold_rows:
        return False

    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
    except:
        cpu_count = 1

    if cpu_count >= 16 and (file_size_mb >= 1024 or (total_rows and total_rows >= 5_000_000)):
        return True

    return False


def shutdown_ray():
    """Shutdown Ray cluster and reset state."""
    global _RAY_INITIALIZED
    
    if RAY_AVAILABLE and ray.is_initialized():
        logger.info("Shutting down Ray cluster...")
        try:
            ray.shutdown()
        except:
            pass
        _RAY_INITIALIZED = False
        logger.info("✓ Ray shutdown complete")


def _is_daft_on_ray_runner() -> bool:
    """
    Return True if Daft is actually using the RayRunner right now.

    We inspect the live Daft context instead of relying on _DAFT_RAY_CONFIGURED,
    because that flag can be stale — for example when daft.set_runner_ray() fails
    silently on the second call within the same process.
    """
    try:
        ctx = daft.context.get_context()
        return "Ray" in type(ctx.runner).__name__
    except Exception:
        # Fall back to the module flag if introspection fails
        return _DAFT_RAY_CONFIGURED


def _compute_partitions(available_cpus: int, file_size_mb: float) -> int:
    """
    Return the number of Daft partitions to use based on CPU count and file size.

    More partitions keep all CPUs busy, but too many add scheduling overhead.
    We scale the per-CPU multiplier with file size so small files don't get
    fragmented unnecessarily.

    File size bands:
        < 256 MB   → 1× cpus  (file is small; partitioning overhead not worth it)
        256 MB–2 GB → 2× cpus
        2 GB–20 GB  → 4× cpus
        > 20 GB     → 8× cpus (capped at 512 to limit scheduler overhead)
    """
    if file_size_mb < 256:
        multiplier = 1
    elif file_size_mb < 2_048:
        multiplier = 2
    elif file_size_mb < 20_480:
        multiplier = 4
    else:
        multiplier = 8

    return min(max(available_cpus * multiplier, 8), 512)


def get_ray_cluster_info() -> Dict[str, Any]:
    """
    Get Ray cluster information for monitoring and debugging.
    
    Returns:
        Dict with cluster status, resources, nodes
    """
    if not RAY_AVAILABLE or not ray.is_initialized():
        return {
            "initialized": False,
            "status": "Ray not initialized",
            "last_error": _RAY_LAST_ERROR,
            "init_mode": _RAY_LAST_INIT_MODE,
            "daft_runner_configured": _DAFT_RAY_CONFIGURED,
        }
    
    try:
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        nodes = ray.nodes()
        
        info = {
            "initialized": True,
            "status": "Ray cluster active",
            "total_cpus": cluster_resources.get("CPU", 0),
            "total_memory_gb": cluster_resources.get("memory", 0) / (1024**3),
            "available_cpus": available_resources.get("CPU", 0),
            "available_memory_gb": available_resources.get("memory", 0) / (1024**3),
            "num_nodes": len(nodes),
            "init_mode": _RAY_LAST_INIT_MODE,
            "daft_runner_configured": _DAFT_RAY_CONFIGURED,
            "nodes": [
                {
                    "node_id": node["NodeID"],
                    "alive": node["Alive"],
                    "resources": node.get("Resources", {})
                }
                for node in nodes
            ]
        }
        
        return info
    
    except Exception as e:
        return {
            "initialized": True,
            "status": f"Error getting cluster info: {e}",
            "last_error": _RAY_LAST_ERROR,
            "init_mode": _RAY_LAST_INIT_MODE,
            "daft_runner_configured": _DAFT_RAY_CONFIGURED,
        }


def log_ray_cluster_status():
    """Log Ray cluster status for debugging."""
    info = get_ray_cluster_info()
    
    if not info["initialized"]:
        logger.info("🔴 Ray Status: NOT INITIALIZED (local execution)")
        return
    
    logger.info("="*80)
    logger.info("🟢 RAY CLUSTER STATUS")
    logger.info("="*80)
    logger.info(f"Status: {info['status']}")
    logger.info(f"Total CPUs: {info.get('total_cpus', 'N/A')}")
    logger.info(f"Available CPUs: {info.get('available_cpus', 'N/A')}")
    logger.info(f"Total Memory: {info.get('total_memory_gb', 0):.2f} GB")
    logger.info(f"Available Memory: {info.get('available_memory_gb', 0):.2f} GB")
    logger.info(f"Nodes: {info.get('num_nodes', 0)}")
    
    for i, node in enumerate(info.get("nodes", [])):
        logger.info(f"  Node {i+1}: {'✓ Alive' if node['alive'] else '✗ Dead'} | ID: {node['node_id'][:8]}...")
    
    logger.info("="*80)


# Base Sample Creation

def _sniff_delimiter_daft(path: str, encoding: str = "utf-8", sample_bytes: int = 8192) -> str:
    """Detect the delimiter used in a CSV by reading a small sample."""
    try:
        with open(path, "rb") as f:
            sample = f.read(sample_bytes)
        try:
            text = sample.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            text = sample.decode(encoding, errors="ignore")
        
        import csv
        dialect = csv.Sniffer().sniff(text, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def _detect_csv_encoding(path: str) -> str:
    """Try a list of common encodings and return the first one that works."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]
    
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
            logger.info(f"Detected encoding: {enc} for file: {path}")
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    logger.warning(f"Could not detect encoding for {path}, using utf-8 with error handling")
    return "utf-8"


def _convert_to_utf8_if_needed(path: str, encoding: str, delimiter: str) -> str:
    """Convert a non-UTF-8 CSV to a UTF-8 temp file so Daft can read it."""
    import tempfile
    import shutil
    
    if encoding.lower() in ["utf-8", "utf8", "utf-8-sig"]:
        return path
    
    logger.info(f"Converting {encoding} file to UTF-8 for Daft processing")
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix=".csv", prefix="daft_utf8_")
        os.close(temp_fd)
        
        with open(path, "r", encoding=encoding, errors="replace") as f_in:
            with open(temp_path, "w", encoding="utf-8", newline="") as f_out:
                f_out.write(f_in.read())
        
        logger.info(f"Created UTF-8 temp file: {temp_path}")
        return temp_path
    except Exception as e:
        logger.warning(f"Failed to convert to UTF-8, using original file: {e}")
        return path




def _load_file_with_daft(path: str, source_type: str, **kwargs) -> Tuple[Optional[daft.DataFrame], List[str]]:
    """
    Load a file into a lazy Daft DataFrame.

    Works for local paths, GCS (gs://), S3 (s3://), Azure, and HTTP(S).
    Daft reads cloud files directly via native I/O — no download or temp file needed.
    For local non-UTF-8 CSVs, a temporary UTF-8 copy is created and tracked for cleanup.

    Returns: (DataFrame, temp_files_list)
    """
    if not DAFT_AVAILABLE:
        raise ImportError("Daft is not available. Install with: pip install getdaft")
    
    temp_files: List[str] = []
    
    try:
        is_cloud_path = any(path.startswith(prefix) for prefix in ["gs://", "s3://", "az://", "http://", "https://"])
        source_fmt = source_type.lower()

        if is_cloud_path:
            logger.info(f"☁️  Cloud file detected: {path}")

            safe_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ["delimiter", "encoding", "cloud_credentials",
                             "metadata_uri", "hive_partitioning", "sheet_name"]
            }
            cloud_io_config = safe_kwargs.get("io_config") or _build_cloud_io_config(path, kwargs.get("cloud_credentials"))
            if cloud_io_config is not None:
                safe_kwargs["io_config"] = cloud_io_config

            if source_fmt == "iceberg":
                df = _read_iceberg_table(
                    path,
                    kwargs.get("metadata_uri"),
                    kwargs.get("cloud_credentials"),
                    io_config=cloud_io_config,
                )
            elif source_fmt in ("csv", "tsv"):
                delimiter = kwargs.get("delimiter", ",")
                df = daft.read_csv(
                    path,
                    has_headers=True,
                    delimiter=delimiter,
                    **safe_kwargs
                )
            elif source_fmt == "parquet":
                pq_kwargs = dict(safe_kwargs)
                # Partition columns (col=value/) are recovered as data columns when
                # hive_partitioning is on. The flag is version-dependent, so pass it
                # only when the caller asked and degrade gracefully if unsupported.
                if kwargs.get("hive_partitioning") is not None:
                    pq_kwargs["hive_partitioning"] = kwargs["hive_partitioning"]
                try:
                    df = daft.read_parquet(path, **pq_kwargs)
                except TypeError as te:
                    if "hive_partitioning" in str(te):
                        pq_kwargs.pop("hive_partitioning", None)
                        df = daft.read_parquet(path, **pq_kwargs)
                    else:
                        raise
            elif source_fmt == "json":
                df = daft.read_json(path, **safe_kwargs)
            else:
                raise ValueError(f"Unsupported source type for cloud file: {source_fmt}")

            logger.info("✓ Daft lazy DataFrame created from cloud path")
            return df, temp_files
        
        abs_path = os.path.abspath(path)
        
        # TSV rides the CSV reader: _sniff_delimiter_daft detects the tab, so the
        # only thing that was missing was the branch. /api/upload advertises tsv
        # in SUPPORTED_UPLOAD_EXTS, and without this it answered HTTP 500
        # "File type not supported" for a format it says it supports.
        if source_fmt in ("csv", "tsv"):
            encoding = kwargs.get("encoding") or _detect_csv_encoding(abs_path)
            delimiter = kwargs.get("delimiter") or _sniff_delimiter_daft(abs_path, encoding)
            daft_path = _convert_to_utf8_if_needed(abs_path, encoding, delimiter)
            
            if daft_path != abs_path:
                temp_files.append(daft_path)
            
            df = daft.read_csv(
                daft_path,
                has_headers=True,
                delimiter=delimiter,
                **{k: v for k, v in kwargs.items() if k not in ["delimiter", "encoding"]}
            )
                    
        elif source_fmt == "parquet":
            df = daft.read_parquet(abs_path, **kwargs)
        elif source_fmt in ("xls", "xlsx"):
            from file_handler.excel_connector import ExcelConnector
            # Read the sheet as a typed DataFrame (dtypes preserved) and honor an
            # explicit sheet_name; otherwise ExcelConnector auto-picks the sheet
            # with the most data rows instead of blindly reading the first one.
            conn = ExcelConnector(path, sheet_name=kwargs.get("sheet_name"))
            df = daft.from_pandas(conn.load_dataframe())
        elif source_fmt == "avro":
            from file_handler.handler import FileHandler
            import pandas as pd
            handler = FileHandler(path, source_type)
            df = daft.from_pandas(pd.DataFrame(handler.load_data(), columns=handler.get_columns()))
        elif source_fmt == "json":
            df = daft.read_json(abs_path, **kwargs)
        elif source_fmt == "xml":
            # Daft has no XML reader and FileHandler has none either, so this
            # advertised format used to fall straight to the ValueError below.
            # pandas reads XML; hand the frame to Daft.
            import pandas as pd
            try:
                pdf = pd.read_xml(abs_path)
            except Exception:
                pdf = pd.read_xml(abs_path, xpath=".//*[*]")
            df = daft.from_pandas(pdf)
        else:
            raise ValueError(
                f"Unsupported source type: {source_fmt}. Supported: "
                f"csv, tsv, json, xml, parquet, avro, xls, xlsx.")
        
        return df, temp_files
    except Exception as e:
        logger.error(f"Failed to load file with Daft: {e}")
        for f in temp_files:
            try:
                if os.path.exists(f): os.unlink(f)
            except:
                pass
        raise


def create_base_sample(
    path: str,
    source_type: str,
    max_base_sample_rows: int = 100_000,
    use_ray: bool = False,
    file_size_mb: float = 0.0,
    **kwargs
) -> Tuple[daft.DataFrame, daft.DataFrame, int, List[str], float]:
    """
    Load the file once, materialise it into memory, and return a base sample.

    Partitions the lazy plan before collecting so Daft parses the file in parallel.
    All downstream operations work on the in-memory table — no further I/O.

    Args:
        file_size_mb: Estimated file size in MB, used to pick the optimal partition count.
                      Pass 0.0 if unknown (falls back to a conservative default).

    Returns: (base_sample, full_df_collected, total_rows, temp_files_list, bytes_per_row)
        bytes_per_row is the measured in-memory (Arrow) bytes per row from a
        1000-row head slice — the real cost signal for byte-aware routing.
    """
    logger.info(f"Loading {path} (type: {source_type})")

    df_lazy, temp_files = _load_file_with_daft(path, source_type, **kwargs)

    # Choose the right partitioning call for the current Daft runner.
    #   RayRunner  → repartition(N)    distributes across Ray workers (requires RayRunner)
    #   NativeRunner → into_partitions(N) re-slices cheaply without shuffle
    # We inspect the live Daft context (_is_daft_on_ray_runner) instead of the
    # _DAFT_RAY_CONFIGURED flag, which can be stale when daft.set_runner_ray()
    # was called but silently failed a second time in the same process.
    try:
        import os as _os
        if _is_daft_on_ray_runner() and ray.is_initialized():
            ray_resources = ray.available_resources()
            available_cpus = int(ray_resources.get("CPU", 1))
            num_partitions = _compute_partitions(available_cpus, file_size_mb)
            logger.info(f"🔄 [RayRunner] repartition({num_partitions}) across {available_cpus} CPUs ({file_size_mb:.0f} MB)")
            df_lazy = df_lazy.repartition(num_partitions)
        else:
            local_cpus = _os.cpu_count() or 4
            num_partitions = _compute_partitions(local_cpus, file_size_mb)
            logger.info(f"🔄 [NativeRunner] into_partitions({num_partitions}) across {local_cpus} CPUs ({file_size_mb:.0f} MB)")
            df_lazy = df_lazy.into_partitions(num_partitions)
    except Exception as e:
        logger.warning(f"Failed to set partitions: {e}. Continuing with Daft defaults.")

    logger.info("📖 Reading source file into memory...")
    full_df = df_lazy.collect()
    total_rows = len(full_df)
    logger.info(f"✅ Loaded {total_rows:,} rows")

    if total_rows == 0:
        logger.warning("Source has 0 rows; returning empty sample")
        return full_df, full_df, 0, temp_files, 0.0

    sample = df_lazy.limit(1000).to_arrow()
    bytes_per_row = sample.nbytes / max(sample.num_rows, 1)
    max_base_sample_rows = max(min(int(5_242_880 / bytes_per_row), max_base_sample_rows), 1)

    if total_rows <= max_base_sample_rows:
        base_sample = full_df
    else:
        sample_fraction = min(max_base_sample_rows / total_rows, 1.0)
        logger.info(f"Sampling {sample_fraction:.2%} ({max_base_sample_rows:,} rows) from in-memory data")
        base_sample = full_df.sample(fraction=sample_fraction, seed=42)

    return base_sample, full_df, total_rows, temp_files, bytes_per_row


# Portfolio Creation

def calculate_sparsity(df: daft.DataFrame, column: str, threshold: int = 1000) -> float:
    """
    Calculate sparsity score: fraction of groups with size < threshold.
    Returns 0.0 (dense) to 1.0 (very sparse).
    """
    try:
        stats = df.groupby(column).agg(daft.col(column).count().alias("count"))
        counts = stats.select("count").collect().to_pydict()["count"]
        if not counts:
            return 0.0
        n_small = sum(1 for c in counts if c < threshold)
        return n_small / len(counts)
    except:
        return 0.0


def calculate_skewness(df: daft.DataFrame, column: str) -> float:
    """
    Calculate absolute skewness of a numeric column.
    Uses a small head sample to approximate, avoiding full pass.
    """
    try:
        sample_data = df.select(column).limit(5000).collect().to_pydict()[column]
        clean_data = [x for x in sample_data if x is not None]
        if len(clean_data) < 10:
            return 0.0
            
        import statistics
        mean = statistics.mean(clean_data)
        std = statistics.stdev(clean_data) if len(clean_data) > 1 else 0
        if std == 0:
            return 0.0
            
        n = len(clean_data)
        cubed_diffs = sum((x - mean)**3 for x in clean_data)
        skew = (cubed_diffs / n) / (std ** 3)
        return abs(skew)
    except:
        return 0.0


def analyze_column_batch(df: daft.DataFrame, column: str, col_type_str: str, threshold: int = 1000) -> dict:
    """
    Compute all column statistics in ONE pass instead of multiple scans.
    
    Returns dict with: unique_count, sparsity (for categorical), skew (for numeric)
    """
    try:
        is_numeric = any(numeric_type in col_type_str for numeric_type in [
            'int8', 'int16', 'int32', 'int64', 
            'uint8', 'uint16', 'uint32', 'uint64',
            'float32', 'float64', 'decimal', 'double'
        ])
        
        unique_count = df.select(column).distinct().count_rows()
        
        result = {
            'unique_count': unique_count,
            'is_numeric': is_numeric,
            'sparsity': 0.0,
            'skew': 0.0
        }
        
        if not is_numeric:
            result['sparsity'] = calculate_sparsity(df, column, threshold)
        else:
            result['skew'] = calculate_skewness(df, column)
        
        return result
        
    except Exception as e:
        logger.warning(f"Batch analysis failed for {column}: {e}")
        return {
            'unique_count': 0,
            'is_numeric': False,
            'sparsity': 0.0,
            'skew': 0.0
        }


def identify_stratification_columns(
    df: daft.DataFrame,
    min_cardinality: int = 2,
    low_cardinality_threshold: int = 20,
    medium_cardinality_threshold: int = 100,
) -> List[Tuple[str, str, int]]:
    """
    Identify columns suitable for stratification, sorted by priority.

    Returns list of (column_name, column_type, unique_count):
      - categorical_low  (2–20 unique): highest priority
      - categorical_medium (21–100 unique): medium priority
      - numerical_skewed  (> 20 unique numeric): for quantile stratification
    """
    logger.info("Identifying stratification candidates")
    
    schema = df.schema()
    
    candidates = _identify_columns_sequential(df, schema, min_cardinality, low_cardinality_threshold, medium_cardinality_threshold)
    
    priority_map = {"categorical_low": 0, "categorical_medium": 1, "numerical_skewed": 2}
    
    candidates.sort(key=lambda x: (
        priority_map.get(x[1], 999), 
        -x[3]  # Sort descending by metric (Sparsity or Skew)
    ))
    
    final_candidates = [(c[0], c[1], c[2]) for c in candidates]
    
    logger.info(f"Found {len(final_candidates)} candidates: {[c[0] for c in final_candidates[:5]]}")
    return final_candidates


def _identify_columns_sequential(df, schema, min_cardinality, low_cardinality_threshold, medium_cardinality_threshold):
    """Sequential fallback for column analysis."""
    candidates = []
    
    for col_name in df.column_names:
        try:
            col_type = schema[col_name]
            
            unique_count = df.select(col_name).distinct().count_rows()
            
            if unique_count < min_cardinality:
                continue
            
            col_type_str = str(col_type).lower()
            is_numeric = any(numeric_type in col_type_str for numeric_type in [
                'int8', 'int16', 'int32', 'int64', 
                'uint8', 'uint16', 'uint32', 'uint64',
                'float32', 'float64', 'decimal', 'double'
            ])
            
            if not is_numeric:
                sparsity = calculate_sparsity(df, col_name)
                
                if unique_count <= low_cardinality_threshold:
                    candidates.append((col_name, "categorical_low", unique_count, sparsity))
                    logger.info(f"  → {col_name}: categorical_low ({unique_count} unique, sparsity {sparsity:.2f})")
                elif unique_count <= medium_cardinality_threshold:
                    candidates.append((col_name, "categorical_medium", unique_count, sparsity))
                    logger.info(f"  → {col_name}: categorical_medium ({unique_count} unique, sparsity {sparsity:.2f})")
            else:
                skew = calculate_skewness(df, col_name)
                
                if unique_count > low_cardinality_threshold:
                    candidates.append((col_name, "numerical_skewed", unique_count, skew))
                    logger.info(f"  → {col_name}: numerical_skewed ({unique_count} unique, skew {skew:.2f})")
        
        except Exception as e:
            logger.warning(f"Failed to analyze {col_name}: {e}")
            continue
    
    return candidates


def stratified_sample_daft(
    df: daft.DataFrame,
    column: str,
    k_cap: int = 100_000,
    seed: int = 42
) -> daft.DataFrame:
    """
    K-Capped Stratified Sampling with Bias Correction Weights.
    
    Logic:
    - For each group, take min(K, group_size) rows.
    - Add `_weight` column = true_group_size / sampled_group_size.
    - Returns LAZY DataFrame for parallel execution.
    """
    k_cap = max(int(k_cap), 1)
    try:
        # count a constant column so null groups report their true size
        group_stats_df = df.with_column("_row", daft.lit(1)).groupby(column).agg(
            daft.col("_row").alias("count").count()
        )
        group_stats = group_stats_df.collect().to_pydict()

        lazy_samples = []

        for i in range(len(group_stats[column])):
            group_value = group_stats[column][i]
            group_count = group_stats["count"][i]

            if group_count <= 0:
                continue

            group_sample_size = min(k_cap, group_count)
            sample_fraction = min(group_sample_size / group_count, 1.0)

            # weight = 1 when we kept every row, higher when we downsampled
            weight = group_count / group_sample_size if group_sample_size > 0 else 1.0

            group_filter = (
                daft.col(column).is_null() if group_value is None
                else daft.col(column) == group_value
            )
            group_sample = (
                df.where(group_filter)
                  .sample(fraction=sample_fraction, seed=seed + i)
                  .with_column("_weight", daft.lit(weight))
            )
            lazy_samples.append(group_sample)

        if lazy_samples:
            result = lazy_samples[0]
            for s in lazy_samples[1:]:
                result = result.concat(s)
            return result
        else:
            return df.limit(k_cap).with_column("_weight", daft.lit(1.0))

    except Exception as e:
        logger.warning(f"Stratified sampling failed for {column}: {e}")
        return df.limit(k_cap).with_column("_weight", daft.lit(1.0))


def quantile_stratified_sample(
    df: daft.DataFrame,
    column: str,
    sample_size: int,
    num_quantiles: int = 4,
    seed: int = 42,
    total_rows: Optional[int] = None,
) -> daft.DataFrame:
    """
    Quantile-based stratified sampling on numeric column.

    Accepts optional `total_rows` to avoid a redundant count scan when the
    caller already knows the row count (e.g. after create_base_sample).
    """
    sample_size = max(int(sample_size), 1)
    try:
        n_rows = total_rows if total_rows is not None else df.count_rows()
        if n_rows == 0:
            return df.with_column("_weight", daft.lit(1.0))

        col = daft.col(column).cast(daft.DataType.float64())

        pcts = [i / num_quantiles for i in range(1, num_quantiles)]
        agg = (
            df.where(~daft.col(column).is_null())
              .agg(col.approx_percentiles(pcts).alias("q"))
              .collect().to_pydict()
        )
        raw = (agg.get("q") or [None])[0] or []
        bounds = sorted({float(b) for b in raw if b is not None})

        edges = [float("-inf")] + bounds + [float("inf")]
        n_buckets = len(edges) - 1

        # quantile buckets are ~equal, so approximate per-bucket size without a count
        approx_bucket = max(n_rows / n_buckets, 1)
        per_bucket = max(sample_size // num_quantiles, 1)
        frac = min(per_bucket / approx_bucket, 1.0)
        weight = 1.0 / frac if frac > 0 else 1.0

        lazy_samples = []
        for i in range(n_buckets):
            lo, hi = edges[i], edges[i + 1]
            if i == 0:
                bucket = df.where(col < hi)
            elif i == n_buckets - 1:
                bucket = df.where(col >= lo)
            else:
                bucket = df.where((col >= lo) & (col < hi))
            lazy_samples.append(
                bucket.sample(fraction=frac, seed=seed + i).with_column("_weight", daft.lit(weight))
            )

        result = lazy_samples[0]
        for s in lazy_samples[1:]:
            result = result.concat(s)
        return result.limit(sample_size)

    except Exception as e:
        logger.warning(f"Quantile sampling failed for {column}: {e}")
        n_rows = total_rows if total_rows is not None else max(df.count_rows(), 1)
        frac = min(sample_size / max(n_rows, 1), 1.0)
        weight = 1.0 / frac if frac > 0 else 1.0
        return df.sample(fraction=frac, seed=seed).with_column("_weight", daft.lit(weight))


def create_sample_portfolio(
    base_sample: daft.DataFrame,
    full_df: Optional[daft.DataFrame] = None,
    max_samples: int = 5,
    sample_size_per: int = 10_000,
    total_rows: Optional[int] = None,
) -> Dict[str, daft.DataFrame]:
    """
    Create portfolio of specialized samples.
    
    Args:
        base_sample: The "Random Baseline" sample.
        full_df: The full source DataFrame for robust stratification.
    """
    logger.info(f"Creating portfolio (max {max_samples} samples)")
    
    source_df = full_df if full_df is not None else base_sample
    
    portfolio = {}
    
    candidates = identify_stratification_columns(base_sample)
    
    if not candidates:
        logger.warning("No stratification candidates found, using random only")
        portfolio["random_baseline"] = base_sample.with_column("_weight", daft.lit(1.0))
        return portfolio
    
    samples_created = 0

    sample = source_df.limit(1000).to_arrow()
    bytes_per_row = sample.nbytes / max(sample.num_rows, 1)
    max_rows = max(min(int(5_242_880 / bytes_per_row), 100_000), 1) # limit to 5MB of data to store
    K_CAP = max(int(max_rows / len(candidates)), 1)
    
    for col_name, col_type, unique_count in candidates:
        if samples_created >= max_samples - 1:
            break
        
        if col_type in ["categorical_low", "categorical_medium"]:
            logger.info(f"Creating stratified sample for {col_name}")
            sample_df = stratified_sample_daft(
                source_df,
                col_name,
                k_cap=K_CAP,
                seed=42 + samples_created
            )
            portfolio[f"strat_{col_name}"] = sample_df
            samples_created += 1
    
    for col_name, col_type, unique_count in candidates:
        if samples_created >= max_samples - 1:
            break
        
        if col_type == "numerical_skewed":
            logger.info(f"Creating quantile sample for {col_name}")
            sample_df = quantile_stratified_sample(
                source_df,
                col_name,
                sample_size=K_CAP * 4,
                num_quantiles=4,
                seed=42 + samples_created,
                total_rows=total_rows,  # avoids redundant count_rows() scan
            )
            portfolio[f"quantile_{col_name}"] = sample_df
            samples_created += 1
            break
    
    # Always include random baseline
    logger.info("Adding random baseline sample")
    portfolio["random_baseline"] = base_sample.with_column("_weight", daft.lit(1.0))
    
    logger.info(f"Portfolio created with {len(portfolio)} samples")
    return portfolio


# Statistics Aggregation

def _is_missing(value: Any) -> bool:
    """Check if value is missing/null."""
    if value is None:
        return True
    if isinstance(value, float):
        import math
        return math.isnan(value)
    if isinstance(value, str):
        v = value.strip().lower()
        return v in ("", "na", "nan", "null", "none", "n/a")
    return False


def _to_float(value: Any) -> Optional[float]:
    """Safe float conversion helper."""
    try:
        if value is None: return None
        return float(value)
    except:
        return None


def _compute_categorical_stats_hybrid(
    specialized_sample: daft.DataFrame,
    base_dict: Dict[str, List[Any]],
    col_name: str
) -> Dict[str, Any]:
    """
    Compute categorical stats using stratified sample for better representation.
    
    Use stratified sample for: unique count, top values, frequencies
    Use base sample for: missing ratio (more stable)
    """
    try:
        # From stratified sample: unique count and top values
        strat_dict = specialized_sample.collect().to_arrow()
        values_strat = strat_dict.column(col_name).to_pylist() if col_name in strat_dict.column_names else []
        weights = (
            strat_dict.column("_weight").to_pylist()
            if "_weight" in strat_dict.column_names else [1.0] * len(values_strat)
        )

        norm_vals = [str(v) if not _is_missing(v) else "<missing>" for v in values_strat]

        # weight-correct the K-cap oversampling of rare categories
        raw_counts = Counter(norm_vals)
        weighted: Dict[str, float] = {}
        for v, w in zip(norm_vals, weights):
            weighted[v] = weighted.get(v, 0.0) + (float(w) if isinstance(w, (int, float)) else 1.0)
        n_unique = len(weighted)
        total_weight = sum(weighted.values()) or 1.0
        top_values = sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)[:15]

        # From base sample: missing ratio
        values_base = base_dict.column(col_name).to_pylist() if col_name in base_dict.column_names else []
        missing_count = sum(1 for v in values_base if _is_missing(v))
        missing_ratio = missing_count / len(values_base) if values_base else 0.0

        return {
            "type": "categorical",
            "n_unique": n_unique,
            "top_values": [
                {
                    "value": val,
                    "count": int(round(wcount)),
                    "frequency": round(wcount / total_weight, 4),
                    "sample_count": raw_counts[val],
                }
                for val, wcount in top_values
            ],
            "missing_ratio": round(missing_ratio, 4),
            "missing_count": missing_count,
            "count": len(values_base) - missing_count,
            "source_sample": "stratified"
        }
    except Exception as e:
        logger.warning(f"Error computing hybrid categorical stats for {col_name}: {e}")
        return {"type": "categorical", "error": str(e)}


def _compute_numeric_stats_hybrid(
    quantile_sample: daft.DataFrame,
    base_dict: Dict[str, List[Any]],
    col_name: str
) -> Dict[str, Any]:
    """
    Compute numeric stats using hybrid approach.
    
    Use quantile sample for: percentiles, min, max (captures extremes)
    Use base sample for: mean, std (more stable)
    """
    try:
        quantile_dict = quantile_sample.collect().to_arrow()

        values_quantile = quantile_dict.column(col_name).to_pylist() if col_name in quantile_dict.column_names else []
        # Use pre-collected base_dict
        values_base = base_dict.column(col_name).to_pylist() if col_name in base_dict.column_names else []
        
        numeric_quantile = [_to_float(v) for v in values_quantile if _to_float(v) is not None]
        numeric_base = [_to_float(v) for v in values_base if _to_float(v) is not None]
        
        if not numeric_quantile or not numeric_base:
            return {"type": "numeric", "error": "No numeric values found"}
        
        # From quantile sample: percentiles, min, max
        sorted_quantile = sorted(numeric_quantile)
        percentile_25 = sorted_quantile[len(sorted_quantile) // 4]
        median = statistics.median(numeric_quantile)
        percentile_75 = sorted_quantile[3 * len(sorted_quantile) // 4]
        min_val = min(numeric_quantile)
        max_val = max(numeric_quantile)
        
        # From base sample: mean, std
        mean = statistics.fmean(numeric_base)
        std = statistics.pstdev(numeric_base) if len(numeric_base) > 1 else 0.0
        
        # Missing ratio from base
        missing_count = sum(1 for v in values_base if _is_missing(v))
        missing_ratio = missing_count / len(values_base) if values_base else 0.0
        
        # Skewness from quantile sample (better tail coverage)
        if std > 0 and len(numeric_quantile) > 2:
            third_moment = sum((x - mean) ** 3 for x in numeric_quantile) / len(numeric_quantile)
            skewness = third_moment / (std ** 3)
        else:
            skewness = 0.0
        
        return {
            "type": "numeric",
            "mean": round(mean, 4),
            "std": round(std, 4),
            "median": round(median, 4),
            "percentile_25": round(percentile_25, 4),
            "percentile_75": round(percentile_75, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "skewness": round(skewness, 4),
            "missing_ratio": round(missing_ratio, 4),
            "missing_count": missing_count,
            "count": len(numeric_base),
            "n_unique": len(set(numeric_base)),
            "source_sample": "quantile + base"
        }
    except Exception as e:
        logger.warning(f"Error computing hybrid numeric stats for {col_name}: {e}")
        return {"type": "numeric", "error": str(e)}


def _compute_stats_from_base(values: List[Any]) -> Dict[str, Any]:
    """Compute stats from base sample (fallback when no specialized sample exists)."""
    missing_count = sum(1 for v in values if _is_missing(v))
    missing_ratio = missing_count / len(values) if values else 0.0
    
    numeric_vals = [_to_float(v) for v in values if _to_float(v) is not None]
    
    # Determine if numeric (≥10% of values are numeric)
    if numeric_vals and len(numeric_vals) >= max(3, len(values) * 0.1):
        # Numeric column
        sorted_vals = sorted(numeric_vals)
        return {
            "type": "numeric",
            "count": len(numeric_vals),
            "missing_count": missing_count,
            "missing_ratio": round(missing_ratio, 4),
            "n_unique": len(set(numeric_vals)),
            "min": round(min(numeric_vals), 4),
            "max": round(max(numeric_vals), 4),
            "mean": round(statistics.fmean(numeric_vals), 4),
            "std": round(statistics.pstdev(numeric_vals), 4) if len(numeric_vals) > 1 else 0.0,
            "median": round(statistics.median(numeric_vals), 4),
            "percentile_25": round(sorted_vals[len(sorted_vals) // 4], 4),
            "percentile_75": round(sorted_vals[3 * len(sorted_vals) // 4], 4),
            "skewness": 0.0,
            "source_sample": "base"
        }
    else:
        # Categorical column
        norm_vals = [str(v) if not _is_missing(v) else "<missing>" for v in values]
        counts = Counter(norm_vals)
        top_values = counts.most_common(15)
        
        return {
            "type": "categorical",
            "n_unique": len(counts),
            "top_values": [
                {"value": val, "count": cnt, "frequency": round(cnt / len(values), 4)}
                for val, cnt in top_values
            ],
            "missing_ratio": round(missing_ratio, 4),
            "missing_count": missing_count,
            "count": len(values) - missing_count,
            "source_sample": "base"
        }


def aggregate_statistics_from_portfolio(
    portfolio: Dict[str, daft.DataFrame],
    base_sample: daft.DataFrame
) -> Dict[str, Any]:
    """
    Calculate statistics by combining insights from all samples.
    
    Logic:
    1. For each column, check if a specialized sample exists (stratified or quantile).
    2. If yes, use hybrid approach (combine specialized + base stats).
    3. If no, use base sample only.
    """
    logger.info("Aggregating statistics from portfolio")
    
    # Materialize base sample once for efficiency
    #Collect ONCE, use dict everywhere
    base_dict = base_sample.collect().to_arrow()
    column_names = [c for c in base_sample.column_names if c != "_weight"]
    total_rows = len(base_dict[column_names[0]]) if column_names else 0
    
    col_stats = {}
    
    for col_name in column_names:
        try:
            # Check for specialized samples
            strat_key = f"strat_{col_name}"
            quant_key = f"quantile_{col_name}"
            
            if strat_key in portfolio:
                # Use hybrid categorical stats
                # Pass collected base_dict instead of re-collecting dataframe
                col_stats[col_name] = _compute_categorical_stats_hybrid(
                    portfolio[strat_key], base_dict, col_name
                )
            elif quant_key in portfolio:
                # Use hybrid numeric stats
                # Pass collected base_dict instead of re-collecting dataframe
                col_stats[col_name] = _compute_numeric_stats_hybrid(
                    portfolio[quant_key], base_dict, col_name
                )
            else:
                # Fallback to base sample
                col_stats[col_name] = _compute_stats_from_base(base_dict.column(col_name).to_pylist())
                
        except Exception as e:
            logger.warning(f"Error computing stats for {col_name}: {e}")
            col_stats[col_name] = {"type": "unknown", "error": str(e)}

    # Data Quality Aggregation
    dq_issues = {
        "overall_completeness": 1.0 - (sum(c.get("missing_ratio", 0) for c in col_stats.values()) / len(col_stats) if col_stats else 0),
        "columns_with_high_missing": [c for c, stats in col_stats.items() if stats.get("missing_ratio", 0) > 0.3],
        "constant_columns": [c for c, stats in col_stats.items() if stats.get("n_unique", 0) <= 1]
    }

    return {
        "column_statistics": col_stats,
        "data_quality": dq_issues,
        "data_shape": {"rows": -1, "columns": len(column_names)}, # rows filled later
    }


# Main Sampling Entry Point

def _estimate_dataset_size(
    path: str,
    source_type: str = "csv",
    cloud_credentials: Optional[Dict[str, Any]] = None,
) -> Tuple[int, float]:
    """
    Fast dataset size estimation WITHOUT a full scan.

    Strategy:
    1. Get total bytes via filesystem metadata (file listing) — milliseconds.
    2. Read first ~64 KB of one file to estimate average row size.
    3. total_rows ≈ total_bytes / avg_row_bytes.

    Works for:
    - Local files and directories
    - Cloud prefixes / glob paths

    cloud_credentials authenticates the metadata probe against private buckets;
    without it the probe fails and callers fall back to a coarse row estimate.

    Returns: (estimated_rows, size_mb)
    """
    SAMPLE_BYTES = 64 * 1024  # 64 KB head read

    def _avg_row_bytes_from_head(data_bytes: bytes, source_fmt: str) -> float:
        """Estimate average row size from a small data chunk."""
        try:
            text = data_bytes.decode("utf-8", errors="replace")
            lines = [l for l in text.split("\n") if l.strip()]
            if len(lines) <= 1:
                return 200.0  # fallback
            # Skip header for CSV
            data_lines = lines[1:] if source_fmt == "csv" else lines
            if not data_lines:
                return 200.0
            avg = sum(len(l) + 1 for l in data_lines) / len(data_lines)  # +1 for newline
            return max(avg, 10.0)  # at least 10 bytes per row
        except Exception:
            return 200.0  # conservative default

    source_fmt = source_type.lower()
    is_cloud = any(path.startswith(p) for p in ["gs://", "s3://", "az://", "http://", "https://"])

    try:
        if is_cloud:

            def _gcs_metadata_and_head(gcs_path: str):
                """Return (total_bytes, head_bytes) for a gs:// path."""
                # parse  gs://bucket/object
                without_scheme = gcs_path[len("gs://"):]
                bucket_name, _, blob_path = without_scheme.partition("/")
                client = _gcs_client_from_creds(cloud_credentials)
                bucket = client.bucket(bucket_name)
                blobs: list = []

                if not blob_path or blob_path.endswith("/"):
                    # directory prefix — list all blobs
                    blobs = list(client.list_blobs(bucket_name, prefix=blob_path or ""))
                elif "*" in blob_path:
                    import fnmatch
                    prefix = blob_path[:blob_path.index("*")]
                    all_blobs = list(client.list_blobs(bucket_name, prefix=prefix))
                    blobs = [b for b in all_blobs if fnmatch.fnmatch(b.name, blob_path)]
                else:
                    b = bucket.get_blob(blob_path)
                    if b:
                        blobs = [b]

                total_bytes = sum(b.size for b in blobs if b.size)
                # read head of first blob
                head = b""
                if blobs:
                    first = blobs[0]
                    end   = min(SAMPLE_BYTES, first.size or SAMPLE_BYTES) - 1
                    head  = first.download_as_bytes(start=0, end=end)
                return total_bytes, head

            def _s3_metadata_and_head(s3_path: str):
                """Return (total_bytes, head_bytes) for an s3:// path."""
                without_scheme = s3_path[len("s3://"):]
                bucket_name, _, key = without_scheme.partition("/")
                s3 = _s3_client_from_creds(cloud_credentials)
                total_bytes = 0
                keys: list = []

                if not key or key.endswith("/") or "*" in key:
                    prefix = key.split("*")[0].rstrip("/")
                    paginator = s3.get_paginator("list_objects_v2")
                    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                        for obj in page.get("Contents", []):
                            if "*" not in key or obj["Key"].endswith(key.split("*")[-1]):
                                total_bytes += obj["Size"]
                                keys.append(obj["Key"])
                else:
                    try:
                        meta = s3.head_object(Bucket=bucket_name, Key=key)
                        total_bytes = meta["ContentLength"]
                        keys = [key]
                    except Exception:
                        pass

                head = b""
                if keys:
                    resp = s3.get_object(
                        Bucket=bucket_name, Key=keys[0],
                        Range=f"bytes=0-{SAMPLE_BYTES - 1}"
                    )
                    head = resp["Body"].read()
                return total_bytes, head

            def _azure_metadata_and_head(az_path: str):
                """Return (total_bytes, head_bytes) for an az:// or https://…blob.core path."""
                client = _azure_client_from_creds(cloud_credentials)

                # az://container/blob
                without_scheme = az_path.split("//", 1)[-1]
                container_name, _, blob_path = without_scheme.partition("/")
                container = client.get_container_client(container_name)
                total_bytes = 0
                blobs: list = []
                prefix = blob_path.split("*")[0] if "*" in blob_path else blob_path
                for b in container.list_blobs(name_starts_with=prefix or None):
                    total_bytes += b.size or 0
                    blobs.append(b.name)

                head = b""
                if blobs:
                    blob_client = container.get_blob_client(blobs[0])
                    head = blob_client.download_blob(offset=0, length=SAMPLE_BYTES).readall()
                return total_bytes, head

            total_bytes = 0
            head_bytes  = b""
            try:
                if path.startswith("gs://"):
                    total_bytes, head_bytes = _gcs_metadata_and_head(path)
                elif path.startswith("s3://"):
                    total_bytes, head_bytes = _s3_metadata_and_head(path)
                elif path.startswith(("az://", "https://")) and "blob.core" in path:
                    total_bytes, head_bytes = _azure_metadata_and_head(path)
                else:
                    # Unknown cloud scheme — fall through to conservative default
                    logger.warning(f"[size_estimate] Unknown cloud scheme for {path!r}, using default")
            except Exception as e:
                logger.warning(f"[size_estimate] Native SDK metadata failed for {path!r}: {e}")

            size_mb = total_bytes / (1024 * 1024)

            if source_fmt == "parquet" and total_bytes > 0:
                estimated_rows = int(total_bytes * 5 / 200)
                logger.info(f"📐 Size estimate (cloud Parquet bytes): ~{estimated_rows:,} rows, {size_mb:.1f} MB")
                return estimated_rows, size_mb

            if total_bytes == 0 and not head_bytes:
                logger.warning(
                    f"[size_estimate] Cloud path {path!r}: all size probes failed. "
                    "Defaulting to 500k rows to trigger background refinement."
                )
                return 500_000, 0.0

            avg_row = _avg_row_bytes_from_head(head_bytes, source_fmt) if head_bytes else 200.0
            estimated_rows = int(total_bytes / avg_row) if total_bytes > 0 else 500_000
            logger.info(f"📐 Size estimate (cloud CSV/JSON): ~{estimated_rows:,} rows, {size_mb:.1f} MB")
            return estimated_rows, size_mb


        else:
            # ── Local path ──
            abs_path = os.path.abspath(path)

            if os.path.isdir(abs_path):
                # Directory: sum all matching file sizes
                import glob as glob_mod
                ext = ".parquet" if source_fmt == "parquet" else (".csv" if source_fmt == "csv" else ".json")
                all_files = glob_mod.glob(os.path.join(abs_path, "**", f"*{ext}"), recursive=True)
                if not all_files:
                    all_files = glob_mod.glob(os.path.join(abs_path, "**", "*"), recursive=True)
                    all_files = [f for f in all_files if os.path.isfile(f)]
                total_bytes = sum(os.path.getsize(f) for f in all_files if os.path.isfile(f))
                sample_file = all_files[0] if all_files else None
            else:
                total_bytes = os.path.getsize(abs_path)
                sample_file = abs_path

            size_mb = total_bytes / (1024 * 1024)

            if source_fmt == "parquet" and sample_file:
                try:
                    import pyarrow.parquet as pq
                    meta = pq.read_metadata(sample_file)
                    rows_per_file = meta.num_rows
                    n_files = max(len(all_files) if os.path.isdir(abs_path) else 1, 1)
                    estimated_rows = rows_per_file * n_files
                    logger.info(f"📐 Size estimate (local Parquet metadata): {estimated_rows:,} rows, {size_mb:.1f} MB")
                    return estimated_rows, size_mb
                except Exception:
                    pass

            # Read first 64 KB
            head_bytes = b""
            if sample_file:
                try:
                    with open(sample_file, "rb") as f:
                        head_bytes = f.read(SAMPLE_BYTES)
                except Exception:
                    pass

            avg_row = _avg_row_bytes_from_head(head_bytes, source_fmt) if head_bytes else 100.0
            estimated_rows = int(total_bytes / avg_row)
            logger.info(f"📐 Size estimate (local head): ~{estimated_rows:,} rows, {size_mb:.1f} MB")
            return estimated_rows, size_mb

    except Exception as e:
        logger.warning(f"[size_estimate] Size estimation failed for {path!r}: {e}")
        # For cloud paths, default to a large estimate so background refinement is triggered
        if any(path.startswith(p) for p in ["gs://", "s3://", "az://", "http://", "https://"]):
            logger.warning(
                "[size_estimate] Cloud path with failed estimation — defaulting to 500k rows "
                "to ensure background refinement is triggered (conservative safe default)."
            )
            return 500_000, 0.0
        return 0, 0.0


def sample_with_profiling(
    path: str,
    source_type: str,
    stratify_by: Optional[str] = None,
    sample_size: Union[int, float] = 1000,
    use_ray: Optional[bool] = None,
    **extra_args: Any
) -> Dict[str, Any]:
    """
    Load the dataset, build a stratified sample portfolio, and return statistics.

    Returns:
        Dict with schema, rows (display sample), ddl_schema, profiling_result, error
    """
    temp_files: List[str] = []
    ray_was_initialized = False
    
    try:
        logger.info(f"=== Starting sampling for {path} ===")
        
        # Fast metadata-based size estimate (no full file scan)
        estimated_rows, file_size_mb = _estimate_dataset_size(
            path, source_type, cloud_credentials=extra_args.get("cloud_credentials")
        )
        logger.info(f"File: {file_size_mb:.1f}MB, ~{estimated_rows:,} rows (estimated)")
        
        if use_ray is True:
            logger.info("Ray forced ON by caller")
            ray_was_initialized = init_ray_cluster()
            if ray_was_initialized:
                log_ray_cluster_status()
            else:
                logger.info("Ray init failed, using Daft native executor")
        elif use_ray is False:
            logger.info("Ray forced OFF by caller")
        elif should_use_ray(estimated_rows, file_size_mb, file_path=path):
            ray_was_initialized = init_ray_cluster()
            if ray_was_initialized:
                log_ray_cluster_status()
            else:
                logger.info("Ray init failed, using Daft native executor")
        else:
            logger.info("Using Daft native executor")
        
        base_sample, full_df, total_rows, returned_temp_files, bytes_per_row = create_base_sample(
            path, source_type,
            use_ray=bool(ray_was_initialized),
            file_size_mb=file_size_mb,
            **extra_args
        )
        temp_files.extend(returned_temp_files)
        
        display_sample_size, portfolio_sample_size = calculate_adaptive_sample_sizes(total_rows)
        
        portfolio = create_sample_portfolio(
            base_sample,
            full_df=full_df,
            max_samples=MAX_PORTFOLIO_SAMPLES,
            sample_size_per=portfolio_sample_size,
            total_rows=total_rows,  # avoids redundant count_rows() inside portfolio
        )
        
        stats_result = aggregate_statistics_from_portfolio(portfolio, base_sample)
        column_statistics = stats_result.get("column_statistics", {})
        data_quality = stats_result.get("data_quality", {})
        
        if isinstance(sample_size, float):
            sample_size = int(total_rows * sample_size)
        else:
            sample_size = max(int(sample_size), 0)
        
        portfolio_output = {}
        limit = int(portfolio_sample_size * 1.1)

        try:
            _PORTFOLIO_TAG = "__portfolio_name__"

            # Tag each lazy plan with its name so we can split rows back after a single
            # .collect() — instead of calling collect() once per portfolio sample.
            tagged_parts: list = []
            for name, df in portfolio.items():
                tagged_parts.append(
                    df.limit(limit).with_column(_PORTFOLIO_TAG, daft.lit(name))
                )

            if tagged_parts:
                combined = tagged_parts[0]
                for part in tagged_parts[1:]:
                    combined = combined.concat(part)

                logger.info(f"⚡ Fused {len(tagged_parts)} portfolio plans → single Daft execution")
                combined_rows = combined.collect().to_pydict()

                # Split rows back by portfolio name
                n_total = len(combined_rows.get(_PORTFOLIO_TAG, []))
                name_col = combined_rows.pop(_PORTFOLIO_TAG, [])
                all_col_names = [c for c in combined_rows if c != "_weight"]

                for pname in portfolio:
                    portfolio_output[pname] = []

                for i in range(n_total):
                    pname = name_col[i]
                    row = {col: combined_rows[col][i] for col in all_col_names}
                    portfolio_output.setdefault(pname, []).append(row)

                logger.info(
                    "✅ Portfolio materialised: "
                    + ", ".join(f"{k}={len(v)} rows" for k, v in portfolio_output.items())
                )
        except Exception as e:
            logger.warning(f"Fused portfolio collection failed ({e}), falling back to per-sample collection")
            portfolio_output = {}
            for name, df in portfolio.items():
                try:
                    rows = df.limit(limit).to_pydict()
                    if rows:
                        n = len(next(iter(rows.values())))
                        portfolio_output[name] = [
                            {col: rows[col][i] for col in rows if col != "_weight"}
                            for i in range(n)
                        ]
                    else:
                        portfolio_output[name] = []
                except Exception as ex:
                    logger.warning(f"Failed to materialize portfolio sample {name}: {ex}")
                    portfolio_output[name] = []

        sample_rows = portfolio_output.get("random_baseline", [])[:sample_size]

        #schema = base_sample.column_names
        schema = {f.name: str(f.dtype) for f in base_sample.schema()}
        logger.info(f"=== SAMPLER SCHEMA OUTPUT === {schema}")
        ddl_lines = [f"CREATE TABLE {Path(path).stem} ("]
        for col, meta in column_statistics.items():
            dtype = "VARCHAR" if meta["type"] == "categorical" else "FLOAT"
            ddl_lines.append(f"  {col} {dtype},")
        ddl_lines[-1] = ddl_lines[-1][:-1]  # remove trailing comma
        ddl_lines.append(");")
        
        ray_info_snapshot = get_ray_cluster_info()
        profiling_result = {
            "data_shape": {"rows": total_rows, "columns": len(schema)},
            "column_statistics": column_statistics,
            "data_quality": data_quality,
            "portfolio_metadata": {
                "num_samples": len(portfolio),
                "sample_names": list(portfolio.keys()),
                "portfolio_sample_size": portfolio_sample_size,
                "display_sample_size": display_sample_size,
                "adaptive_sizing_used": True,
                "sizing_tier": "TINY" if total_rows < 1000 else 
                               "SMALL" if total_rows < 100_000 else
                               "LARGE" if total_rows < 1_000_000 else
                               "HUGE" if total_rows < 10_000_000 else "MASSIVE",
                "ray_used": ray_was_initialized,
                "ray_requested": use_ray if use_ray is not None else "auto",
                "ray_info": ray_info_snapshot,
                "ray_shutdown_after": RAY_SHUTDOWN_AFTER,
                "sampled_percentage": f"{(len(sample_rows)/total_rows*100):.2f}%" if total_rows > 0 else "0%"
            }
        }
        
        if ray_was_initialized and RAY_SHUTDOWN_AFTER:
            shutdown_ray()
            
        for fp in temp_files:
            if fp and os.path.exists(fp):
                try:
                    os.unlink(fp)
                except:
                    pass

        return {
            "schema": schema,
            "rows": sample_rows,
            "portfolio_samples": portfolio_output, # Return full portfolio
            "ddl_schema": "\n".join(ddl_lines),
            "profiling_result": profiling_result,
            "error": None,
            # New sampler-state keys
            "full_sample_rows": sample_rows,
            "sample_statistics": {
                "column_statistics": column_statistics,
                "data_quality": data_quality,
                "data_shape": profiling_result["data_shape"],
                # Measured in-memory cost signal for byte-aware routing:
                # bytes/row from the Arrow head slice; estimated_full_bytes is
                # the whole frame's estimated uncompressed in-memory footprint.
                "bytes_per_row": round(bytes_per_row, 2),
                "estimated_full_bytes": int(total_rows * bytes_per_row),
            },
            "sample_status": "full_sample",
            "total_rows_exact": total_rows,
            "sample_pct": round((len(sample_rows) / total_rows * 100) if total_rows > 0 else 0.0, 4),
        }

    except Exception as e:
        logger.error(f"Sampling failed: {e}", exc_info=True)
        if 'ray_was_initialized' in locals() and ray_was_initialized and RAY_SHUTDOWN_AFTER:
            shutdown_ray()
            
        for fp in temp_files:
            if fp and os.path.exists(fp):
                try:
                    os.unlink(fp)
                except:
                    pass
                
        return {"error": str(e), "rows": [], "portfolio_samples": {}, "schema": [], "profiling_result": {},
                "full_sample_rows": [], "sample_statistics": {}, "sample_status": "error"}
