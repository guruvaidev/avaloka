"""
Supabase Persistence for Sampling Results

Stores dataset profiles and display samples in Supabase so they survive
server restarts and can be served without re-running the sampler.

Tables required (run the SQL migration below in the Supabase dashboard first):
    dataset_profiles  — one row per profile per dataset
    dataset_samples   — one row per sample per dataset (rows_data as JSONB)

Environment variables:
    SUPABASE_URL              — e.g. https://xxxx.supabase.co
    SUPABASE_SERVICE_ROLE_KEY — service role key (server-side only)
    AVALOKA_PERSIST_WORKERS   — concurrent upserts in persist_full_profile
                                (default 4; 1 = sequential, the old behaviour)
    AVALOKA_PERSIST_FULL_SAMPLE — "auto" (default): skip the 'full' sample when
                                it duplicates random_baseline; "always": write
                                it anyway; "never": never write it
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

TABLE_PROFILES = os.getenv("SUPABASE_PROFILES_TABLE", "dataset_profiles")
TABLE_SAMPLES  = os.getenv("SUPABASE_SAMPLES_TABLE",  "dataset_samples")


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


# Each sample upsert is one HTTP round trip of several MB of JSON (2-10 s each
# on the ICU file). Running them concurrently turns ~30 s into roughly the
# slowest single upsert.
_PERSIST_WORKERS = _env_int("AVALOKA_PERSIST_WORKERS", 4)

# The display sample stored as phase 'full' is random_baseline[:sample_size];
# with the server's sample size it is the whole of random_baseline, so writing
# it again doubles the largest upload for no new data.
_PERSIST_FULL_SAMPLE = os.getenv("AVALOKA_PERSIST_FULL_SAMPLE", "auto").strip().lower()

# SQL migration — run once in the Supabase dashboard
MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS dataset_profiles (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id          TEXT        NOT NULL,
    user_id             TEXT,
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now(),
    phase               TEXT        NOT NULL,   -- quick_profile | full_profile | error
    tier                TEXT,                   -- TINY|SMALL|LARGE|HUGE|MASSIVE
    source_path         TEXT,
    source_type         TEXT,                   -- csv | parquet | json
    estimated_rows      BIGINT,
    exact_rows          BIGINT,
    file_size_mb        DOUBLE PRECISION,
    sample_pct          DOUBLE PRECISION,
    quick_elapsed_s     DOUBLE PRECISION,
    full_elapsed_s      DOUBLE PRECISION,
    column_count        INT,
    completeness_pct    DOUBLE PRECISION,
    profiling_result    JSONB,
    error               TEXT
);

CREATE INDEX IF NOT EXISTS idx_dataset_profiles_dataset_id ON dataset_profiles (dataset_id);
CREATE INDEX IF NOT EXISTS idx_dataset_profiles_updated_at ON dataset_profiles (updated_at DESC);

-- Required by upsert(on_conflict='dataset_id,phase'); without it PostgREST
-- rejects the ON CONFLICT and every persist becomes a silent no-op.
CREATE UNIQUE INDEX IF NOT EXISTS ux_dataset_profiles_dataset_id_phase
    ON dataset_profiles (dataset_id, phase);

CREATE TABLE IF NOT EXISTS dataset_samples (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id  TEXT        NOT NULL,
    phase       TEXT        NOT NULL,           -- quick | full | <portfolio_sample_name>
    created_at  TIMESTAMPTZ DEFAULT now(),
    rows_count  INT,
    columns     TEXT[],
    rows_data   JSONB                           -- List[Dict] — up to 20k rows
);

-- Supersedes the old non-unique index; required by upsert(on_conflict='dataset_id,phase').
DROP INDEX IF EXISTS idx_dataset_samples_dataset_id;
CREATE UNIQUE INDEX IF NOT EXISTS ux_dataset_samples_dataset_id_phase
    ON dataset_samples (dataset_id, phase);
"""


def _jwt_project_ref(token: str) -> Optional[str]:
    """The Supabase project a legacy JWT key belongs to, or None.

    Reads only the `ref` claim from the unverified payload. No signature check
    is wanted or possible here -- we are not authenticating anything, just
    asking which project the key was minted for.
    """
    parts = (token or "").split(".")
    if len(parts) != 3:
        return None
    try:
        import base64
        import json as _json
        pad = parts[1] + "=" * (-len(parts[1]) % 4)
        return _json.loads(base64.urlsafe_b64decode(pad)).get("ref")
    except Exception:
        return None


def _url_project_ref(url: str) -> Optional[str]:
    m = re.search(r"https?://([a-z0-9]+)\.supabase\.(?:co|in)", url or "")
    return m.group(1) if m else None


def check_supabase_credentials(*, url: str = None, key: str = None) -> Optional[str]:
    """Return a description of a credential mismatch, or None when consistent.

    Catches the failure that is otherwise invisible: a service_role key for one
    project paired with another project's URL. Supabase answers that with
    `401 Invalid API key`, which reads like a bad or rotated key and sends you
    to regenerate a key that was fine all along.

    It happens because credentials arrive from two places. `load_dotenv()` does
    NOT override variables already in the OS environment, so anything exported
    by a shell script silently outranks .env -- and nothing says so.
    """
    url = url if url is not None else (os.getenv("SUPABASE_URL") or SUPABASE_URL)
    key = key if key is not None else (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or SUPABASE_SERVICE_ROLE_KEY)
    if not (url and key):
        return None
    url_ref, key_ref = _url_project_ref(url), _jwt_project_ref(key)
    if url_ref and key_ref and url_ref != key_ref:
        return (
            f"SUPABASE_SERVICE_ROLE_KEY belongs to project {key_ref!r} but "
            f"SUPABASE_URL points at {url_ref!r}. Every request will fail with "
            f"401 'Invalid API key'. Note that a value exported into the shell "
            f"(e.g. by dev-env.bat) overrides .env, because load_dotenv() does "
            f"not override the OS environment."
        )
    return None


def warn_on_supabase_credential_mismatch() -> None:
    """Log the mismatch loudly at startup. Never raises: a bad key must not
    stop a server whose other features work fine."""
    problem = check_supabase_credentials()
    if problem:
        logger.error("[supabase] %s", problem)


_CLIENT_CACHE: Dict[Tuple[str, str], Any] = {}
_CLIENT_LOCK = threading.Lock()

def get_supabase_client():
    """Return a configured Supabase client, or raise RuntimeError if env vars are missing.

    Credentials are read HERE rather than from the module-level constants above.
    Those are bound at import time, so whether they hold anything depends on
    whether some other module happened to call load_dotenv() first -- an
    ordering nobody controls and which fails as "env vars not set" long after
    the real cause. Reading at call time also means a corrected value in the
    process environment takes effect without a code change.
    """
    url = os.getenv("SUPABASE_URL") or SUPABASE_URL
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or SUPABASE_SERVICE_ROLE_KEY
    if not (url and key):
        missing = ", ".join(n for n, v in (
            ("SUPABASE_URL", url), ("SUPABASE_SERVICE_ROLE_KEY", key)) if not v)
        raise RuntimeError(
            f"Supabase env vars not set ({missing}). "
            "Set them in your .env file or environment."
        )

    # Reuse one client per (url, key): creating a client per call meant a new
    # connection + TLS handshake on every lookup. Keyed on the credentials read
    # above, so a changed env value still yields a fresh client.
    cache_key = (url, key)
    client = _CLIENT_CACHE.get(cache_key)
    if client is not None:
        return client
    with _CLIENT_LOCK:
        client = _CLIENT_CACHE.get(cache_key)
        if client is None:
            try:
                from supabase import create_client  # type: ignore
            except ImportError:
                raise RuntimeError("supabase package not installed. Run: pip install supabase")
            client = create_client(url, key)
            _CLIENT_CACHE.clear()          # drop clients for stale credentials
            _CLIENT_CACHE[cache_key] = client
    return client


def _safe_json(obj: Any) -> Any:
    """Recursively make an object JSON-serialisable, dropping anything that can't be serialised."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        import math
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return None


def upsert_profile(
    dataset_id: str,
    phase: str,
    *,
    user_id: Optional[str] = None,
    source_path: Optional[str] = None,
    source_type: Optional[str] = None,
    tier: Optional[str] = None,
    estimated_rows: Optional[int] = None,
    exact_rows: Optional[int] = None,
    file_size_mb: Optional[float] = None,
    sample_pct: Optional[float] = None,
    phase_a_elapsed_s: Optional[float] = None,
    phase_b_elapsed_s: Optional[float] = None,
    column_count: Optional[int] = None,
    completeness_pct: Optional[float] = None,
    profiling_result: Optional[Dict] = None,
    error: Optional[str] = None,
) -> Optional[Dict]:
    """
    Insert or update a profile row for the given dataset and phase.

    If a row already exists for this (dataset_id, phase) pair it is updated in place.
    Returns the upserted row, or None if Supabase is not configured or the call fails.
    """
    try:
        client = get_supabase_client()
    except RuntimeError as e:
        logger.warning(f"[persistence] Skipping upsert_profile: {e}")
        return None

    now = datetime.now(timezone.utc).isoformat()
    row: Dict[str, Any] = {
        "dataset_id":  dataset_id,
        "phase":       phase,
        "updated_at":  now,
    }
    if user_id is not None:          row["user_id"]           = user_id
    if source_path is not None:      row["source_path"]       = source_path
    if source_type is not None:      row["source_type"]       = source_type
    if tier is not None:             row["tier"]              = tier
    if estimated_rows is not None:   row["estimated_rows"]    = estimated_rows
    if exact_rows is not None:       row["exact_rows"]        = exact_rows
    if file_size_mb is not None:     row["file_size_mb"]      = file_size_mb
    if sample_pct is not None:       row["sample_pct"]        = sample_pct
    if phase_a_elapsed_s is not None: row["quick_elapsed_s"]  = phase_a_elapsed_s
    if phase_b_elapsed_s is not None: row["full_elapsed_s"]   = phase_b_elapsed_s
    if column_count is not None:     row["column_count"]      = column_count
    if completeness_pct is not None: row["completeness_pct"]  = completeness_pct
    if profiling_result is not None: row["profiling_result"]  = _safe_json(profiling_result)
    if error is not None:            row["error"]             = error

    try:
        res = (
            client.table(TABLE_PROFILES)
            .upsert(row, on_conflict="dataset_id,phase")
            .execute()
        )
        data = (getattr(res, "data", None) or [])
        result = data[0] if data else None
        logger.info(
            f"[persistence] Upserted profile dataset_id={dataset_id!r} phase={phase!r} "
            f"tier={tier} rows={exact_rows or estimated_rows}"
        )
        return result
    except Exception as exc:
        logger.error(f"[persistence] upsert_profile failed for {dataset_id!r}: {exc}", exc_info=True)
        return None


def upsert_sample(
    dataset_id: str,
    phase: str,
    rows: List[Dict],
    columns: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    Insert or update a sample for the given dataset and phase.

    phase should be 'quick', 'full', or a portfolio sample name like 'strat_gender'.
    rows is the list of row dicts stored as JSONB.
    """
    try:
        client = get_supabase_client()
    except RuntimeError as e:
        logger.warning(f"[persistence] Skipping upsert_sample: {e}")
        return None

    cols = columns or (list(rows[0].keys()) if rows else [])
    row: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "phase":      phase,
        "rows_count": len(rows),
        "columns":    cols,
        "rows_data":  _safe_json(rows),
    }

    try:
        res = (
            client.table(TABLE_SAMPLES)
            .upsert(row, on_conflict="dataset_id,phase")
            .execute()
        )
        data = getattr(res, "data", None) or []
        result = data[0] if data else None
        logger.info(
            f"[persistence] Upserted sample dataset_id={dataset_id!r} "
            f"phase={phase!r} rows={len(rows)}"
        )
        return result
    except Exception as exc:
        logger.error(f"[persistence] upsert_sample failed for {dataset_id!r}: {exc}", exc_info=True)
        return None


def load_sample(dataset_id: str, phase: str = "full") -> Optional[List[Dict]]:
    """Load the row data for a given dataset and phase. Returns None if not found.

    phase 'full' falls back to 'random_baseline' when no 'full' row exists:
    persist_full_profile no longer stores 'full' when it would duplicate
    random_baseline, so readers of 'full' keep getting the same rows.
    """
    try:
        client = get_supabase_client()
    except RuntimeError as e:
        logger.warning(f"[persistence] Skipping load_sample: {e}")
        return None

    def _fetch(p: str) -> Optional[List[Dict]]:
        res = (
            client.table(TABLE_SAMPLES)
            .select("rows_data,rows_count,columns")
            .eq("dataset_id", dataset_id)
            .eq("phase", p)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        data = getattr(res, "data", None) or []
        return data[0].get("rows_data") if data else None

    try:
        rows = _fetch(phase)
        if rows is None and phase == "full":
            rows = _fetch("random_baseline")
        return rows
    except Exception as exc:
        logger.error(f"[persistence] load_sample failed for {dataset_id!r}/{phase!r}: {exc}", exc_info=True)
        return None


def load_portfolio(dataset_id: str) -> Optional[Dict[str, List[Dict]]]:
    """Load all portfolio samples for a dataset. Returns {sample_name: rows} or None."""
    try:
        client = get_supabase_client()
    except RuntimeError as e:
        logger.warning(f"[persistence] Skipping load_portfolio: {e}")
        return None

    try:
        res = (
            client.table(TABLE_SAMPLES)
            .select("phase,rows_data,rows_count")
            .eq("dataset_id", dataset_id)
            .execute()
        )
        data = getattr(res, "data", None) or []
        if not data:
            return None
        return {row["phase"]: row.get("rows_data", []) for row in data}
    except Exception as exc:
        logger.error(f"[persistence] load_portfolio failed: {exc}", exc_info=True)
        return None


def load_profile(dataset_id: str) -> Optional[Dict[str, Any]]:
    """
    Load persisted profiling data for a dataset.

    Returns a dict with keys:
      - profiling_result: statistical profile (column_statistics, data_quality, data_shape, ...)
      - full_profiling_result: semantic analysis (domain, column_explanations, insights, ...)
      - sample_statistics: {column_statistics, data_quality, data_shape} subset
      - portfolio_sample_uris: {sample_name: cloud_uri} for Ray reuse (if stored)
    Returns None if nothing is stored.
    """
    try:
        client = get_supabase_client()
    except RuntimeError as e:
        logger.warning(f"[persistence] Skipping load_profile: {e}")
        return None

    try:
        res = (
            client.table(TABLE_PROFILES)
            .select("phase,profiling_result")
            .eq("dataset_id", dataset_id)
            .in_("phase", ["full_profile", "semantic_profile"])
            .execute()
        )
        data = getattr(res, "data", None) or []
        if not data:
            return None

        out: Dict[str, Any] = {}
        for row in data:
            pr = row.get("profiling_result") or {}
            if row["phase"] == "full_profile":
                uris = pr.pop("portfolio_sample_uris", None)
                if uris:
                    out["portfolio_sample_uris"] = uris
                out["profiling_result"] = pr
                out["sample_statistics"] = {
                    "column_statistics": pr.get("column_statistics", {}),
                    "data_quality": pr.get("data_quality", {}),
                    "data_shape": pr.get("data_shape", {}),
                }
            elif row["phase"] == "semantic_profile":
                out["full_profiling_result"] = pr
        return out if out else None
    except Exception as exc:
        logger.error(f"[persistence] load_profile failed: {exc}", exc_info=True)
        return None


def persist_quick_profile(
    dataset_id: str,
    agent_res: Dict[str, Any],
    *,
    user_id: Optional[str] = None,
    source_path: Optional[str] = None,
    source_type: Optional[str] = None,
    phase_a_elapsed_s: Optional[float] = None,
) -> None:
    """
    Persist the quick profile and display sample after the initial fast scan completes.
    Designed to run in a thread pool since the Supabase calls are blocking HTTP.
    """
    pr = agent_res.get("profiling_result") or {}
    dq = pr.get("data_quality") or {}
    schema = agent_res.get("schema") or {}
    cols = list(schema.keys()) if isinstance(schema, dict) else list(schema)
    rows = agent_res.get("rows") or []

    upsert_profile(
        dataset_id=dataset_id,
        phase="quick_profile",
        user_id=user_id,
        source_path=source_path,
        source_type=source_type,
        tier=agent_res.get("tier"),
        estimated_rows=agent_res.get("estimated_rows"),
        file_size_mb=None,
        sample_pct=agent_res.get("sample_pct"),
        phase_a_elapsed_s=phase_a_elapsed_s,
        column_count=len(cols),
        completeness_pct=(dq.get("overall_completeness") or 0) * 100 or None,
        profiling_result=pr,
        error=agent_res.get("error"),
    )

    if rows:
        upsert_sample(dataset_id=dataset_id, phase="quick", rows=rows, columns=cols)


def _full_sample_is_duplicate(rows: List[Dict], portfolio_samples: Dict[str, List[Dict]]) -> bool:
    """True when `rows` is a prefix of random_baseline (the usual case)."""
    baseline = portfolio_samples.get("random_baseline") or []
    return bool(rows) and len(rows) <= len(baseline) and rows == baseline[:len(rows)]


def persist_full_profile(
    dataset_id: str,
    full_result: Dict[str, Any],
    *,
    user_id: Optional[str] = None,
    source_path: Optional[str] = None,
    source_type: Optional[str] = None,
    phase_a_elapsed_s: Optional[float] = None,
    phase_b_elapsed_s: Optional[float] = None,
) -> None:
    """
    Persist the full profile, display sample, and all portfolio samples after the background
    job completes. Designed to run in a thread pool since the Supabase calls are blocking HTTP.

    The profile and sample upserts are independent rows, so they run concurrently
    (AVALOKA_PERSIST_WORKERS). The 'full' display sample is skipped when it is just a
    copy of random_baseline (AVALOKA_PERSIST_FULL_SAMPLE=auto); load_sample('full')
    falls back to random_baseline in that case.
    """
    _t_persist = time.monotonic()
    pr = full_result.get("profiling_result") or {}
    dq = pr.get("data_quality") or {}
    pm = pr.get("portfolio_metadata") or {}
    ds = pr.get("data_shape") or {}
    
    schema = full_result.get("schema") or {}
    cols = list(schema.keys()) if isinstance(schema, dict) else list(schema)
    rows = full_result.get("rows") or []
    exact_rows = ds.get("rows") or dq.get("total_rows")

    sample_pct = None
    if exact_rows and pm.get("display_sample_size"):
        sample_pct = pm["display_sample_size"] / max(exact_rows, 1) * 100

    # persist portfolio URIs so they survive a session restart
    portfolio_sample_uris = full_result.get("portfolio_sample_uris") or {}
    pr_to_store = {**pr, "portfolio_sample_uris": portfolio_sample_uris} if portfolio_sample_uris else pr

    portfolio_samples: Dict[str, List[Dict]] = full_result.get("portfolio_samples") or {}

    # Build the list of independent writes.
    jobs: List[Tuple[str, Any, tuple, dict]] = [(
        "profile:full_profile",
        upsert_profile,
        (dataset_id, "full_profile"),
        dict(
            user_id=user_id,
            source_path=source_path,
            source_type=source_type,
            tier=pm.get("sizing_tier"),
            exact_rows=exact_rows,
            sample_pct=sample_pct,
            phase_a_elapsed_s=phase_a_elapsed_s,
            phase_b_elapsed_s=phase_b_elapsed_s,
            column_count=len(cols),
            completeness_pct=(dq.get("overall_completeness") or 0) * 100 or None,
            profiling_result=pr_to_store,
            error=full_result.get("error"),
        ),
    )]

    skipped_full = False
    if rows:
        if _PERSIST_FULL_SAMPLE == "never":
            skipped_full = True
        elif _PERSIST_FULL_SAMPLE == "always" or not _full_sample_is_duplicate(rows, portfolio_samples):
            jobs.append(("sample:full", upsert_sample, (dataset_id, "full", rows, cols), {}))
        else:
            skipped_full = True

    for sample_name, sample_rows in portfolio_samples.items():
        if sample_rows:
            jobs.append((f"sample:{sample_name}", upsert_sample,
                         (dataset_id, sample_name, sample_rows, cols), {}))

    workers = min(_PERSIST_WORKERS, len(jobs))
    if workers <= 1:
        for _label, fn, args, kwargs in jobs:
            fn(*args, **kwargs)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="persist-upsert"
        ) as pool:
            futures = {pool.submit(fn, *args, **kwargs): label for label, fn, args, kwargs in jobs}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()       # upsert_* already log and swallow their own errors
                except Exception as exc:
                    logger.error("[persistence] %s failed for %r: %s", futures[fut], dataset_id, exc)

    logger.info(
        "[persistence] persist_full_profile dataset_id=%r done in %.2fs "
        "(%d samples, %d workers%s)",
        dataset_id, time.monotonic() - _t_persist, len(portfolio_samples), workers,
        ", skipped duplicate 'full' sample" if skipped_full else "",
    )