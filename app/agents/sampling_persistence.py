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
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

TABLE_PROFILES = os.getenv("SUPABASE_PROFILES_TABLE", "dataset_profiles")
TABLE_SAMPLES  = os.getenv("SUPABASE_SAMPLES_TABLE",  "dataset_samples")

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
    try:
        from supabase import create_client  # type: ignore
    except ImportError:
        raise RuntimeError("supabase package not installed. Run: pip install supabase")
    return create_client(url, key)


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
    """Load the row data for a given dataset and phase. Returns None if not found."""
    try:
        client = get_supabase_client()
    except RuntimeError as e:
        logger.warning(f"[persistence] Skipping load_sample: {e}")
        return None

    try:
        res = (
            client.table(TABLE_SAMPLES)
            .select("rows_data,rows_count,columns")
            .eq("dataset_id", dataset_id)
            .eq("phase", phase)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        data = getattr(res, "data", None) or []
        if not data:
            return None
        return data[0].get("rows_data")
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
    pm = pr.get("portfolio_metadata") or {}
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
    """
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

    upsert_profile(
        dataset_id=dataset_id,
        phase="full_profile",
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
    )

    if rows:
        upsert_sample(dataset_id=dataset_id, phase="full", rows=rows, columns=cols)

    # Save individual portfolio samples
    portfolio_samples: Dict[str, List[Dict]] = full_result.get("portfolio_samples") or {}
    for sample_name, sample_rows in portfolio_samples.items():
        if sample_rows:
            upsert_sample(
                dataset_id=dataset_id,
                phase=sample_name,
                rows=sample_rows,
                columns=cols,
            )
