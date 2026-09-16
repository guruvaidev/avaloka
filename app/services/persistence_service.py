"""
  - No GCP hardcoding: uses existing IBlobStore / _store_from_connection_uri
    so AWS S3, GCS, and Azure all work identically
  - Per-prompt timestamped files: each code generation or execution creates
    its own file — multiple files per session are expected and handled
  - Visualization config also persisted so it is never re-computed
  - Exponential backoff on every cloud write (3 attempts: 0 → 2s → 4s)
  - On final failure, session is updated with persist_errors so the UI
    can surface the error to the user
  - Git persistence exposed as planner tool calls — planner decides when
    to write the job definition, not server.py auto-triggering it
  - Trigger condition: only fires when execution actually ran OR code was
    generated as part of a run/schedule — NOT on plain conversational turns

Public surface used by server.py / planner agent
-------------------------------------------------
  _persist_assets_background(...)   — asyncio.create_task() after send_message
  generate_signed_url(...)          — used by retrieval endpoints
  PERSISTENCE_TOOLS                 — LangChain tools for planner agent
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ── GitHub env vars ──────────────────────────────────────────────────────────
GITHUB_TOKEN    = os.getenv("GITHUB_SYSTEM_TOKEN", "")
GITHUB_JOB_REPO = os.getenv("GITHUB_JOB_REGISTRY_REPO", "avaloka/avaloka-job-registry")
WRITE_JOB_DEFINITION_JSON = os.getenv("AVALOKA_JOB_REGISTRY_WRITE_JSON", "true").lower() != "false"
# ── Signed URL expiry (for retrieval endpoints) ───────────────────────────────
SIGNED_URL_EXPIRY_M = int(os.getenv("AVALOKA_SIGNED_URL_EXPIRY_MINUTES", "15"))

# ── Retry config (Leela: "try once, wait 2s, wait 4s, then post error to UI") ─
_RETRY_DELAYS = (0, 2, 4)   # seconds before attempt 1, 2, 3


# ═══════════════════════════════════════════════════════════════════════════════
# Exponential-backoff retry helper
# ═══════════════════════════════════════════════════════════════════════════════

import re

def _slug(text: str, max_len: int = 40) -> str:
    """'Meridian_Molding_Production.csv' -> 'meridian-molding-production'"""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return s[:max_len].strip("-")

def _short(uid: str, n: int = 8) -> str:
    """Keep 8 chars of the GUID for uniqueness/traceability, drop the noise."""
    return (uid or "").replace("-", "")[:n]

async def _retry_async(coro_fn, label: str, *args, **kwargs):
    """
    Call an async function up to len(_RETRY_DELAYS) times.
    Raises the last exception so the caller can mark the session with an error.
    """
    last_exc: Optional[Exception] = None
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[persist] %s attempt %d/%d failed: %s",
                label, attempt, len(_RETRY_DELAYS), exc,
            )
    raise last_exc  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════════
# Cloud-agnostic store builder
# Uses existing IBlobStore abstraction (GCS / S3 / Azure) — no GCP hardcoding
# ( "abstract out GCP and use generic cloud abstraction for storage.
#          Dont hardcode GCP" / "any customer cloud storage (AWS / GCP / Azure)")
# ═══════════════════════════════════════════════════════════════════════════════

async def _get_store_for_connection(
    connection_id: Optional[str],
    storage_uri:   Optional[str] = None,
):
    """
    Return (store, base_prefix) using the existing IBlobStore abstraction.

    Priority:
      1. Customer cloud connection (connection_id + storage_uri) via
         _store_from_connection_uri — works for GCS, S3, Azure
      2. Avaloka system blob_store (already initialised in storage_service) —
         same store used for dataset uploads, guaranteed to have credentials
    """
    from app.services import storage_service          # type: ignore[import-not-found]
    from app.api.cloud_connections import (           # type: ignore[import-not-found]
        get_cloud_connection,
        _store_from_connection_uri,
    )

    if connection_id and storage_uri:
        try:
            conn  = await get_cloud_connection(connection_id)
            store, base_prefix = await _store_from_connection_uri(storage_uri, conn)
            return store, base_prefix
        except Exception as exc:
            logger.warning(
                "[persist] Could not build store from connection_id=%s — "
                "falling back to system store: %s",
                connection_id, exc,
            )

    # System store fallback
    return storage_service.blob_store, ""


def _object_key(
    prefix, user_id, session_id, dataset_id, prompt_ts, suffix,
    *, dataset_label: str = "", analysis_label: str = "", version: Optional[int] = None,
) -> str:
    ds  = f"{_slug(dataset_label) or 'dataset'}-{_short(dataset_id)}"
    an  = f"{_slug(analysis_label) or 'analysis'}-{_short(session_id)}"
    ver = f"v{version}__{prompt_ts}" if version else prompt_ts
    return f"{prefix}/{_short(user_id)}/{ds}/{an}/{ver}_{suffix}"


# ═══════════════════════════════════════════════════════════════════════════════
# Signed URL helper (cloud-agnostic)
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_signed_url(
    object_key:     str,
    connection_id:  Optional[str] = None,
    storage_uri:    Optional[str] = None,
    expiry_minutes: int = SIGNED_URL_EXPIRY_M,
) -> Optional[str]:
    """
    Generate a time-limited V4 signed URL for direct cloud download.
    GCSBlobStore has no signed_url method, so we sign off its underlying
    google-cloud-storage Bucket (_bucket), mirroring its _key() prefixing.
    Returns None on any error.
    """
    try:
        store, _ = await _get_store_for_connection(connection_id, storage_uri)
        if not store:
            return None

        def _sign() -> Optional[str]:
            from datetime import timedelta
            bucket = getattr(store, "_bucket", None)
            if bucket is None:
                logger.warning(
                    "[persist] store type=%s has no _bucket; cannot sign",
                    type(store).__name__,
                )
                return None
            keyer = getattr(store, "_key", None)
            key = keyer(object_key) if callable(keyer) else object_key
            return bucket.blob(key).generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=expiry_minutes),
                method="GET",
            )

        return await asyncio.to_thread(_sign)
    except Exception as exc:
        logger.warning("[persist] generate_signed_url failed for %s: %s", object_key, exc)
        return None
# ═══════════════════════════════════════════════════════════════════════════════
# Task A — Generated code → Cloud storage  (per-prompt timestamped file)
# ═══════════════════════════════════════════════════════════════════════════════
async def _do_upload_bytes(store, key: str, data: bytes, content_type: str) -> None:
    """
    Raw upload — wrapped by _retry_async.
    Writes bytes to a temp file then calls put_file() which is the only
    upload method available on GCSBlobStore, S3BlobStore, and AzureBlobStore.
    """
    import tempfile
    import os as _os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        await asyncio.to_thread(store.put_file, tmp_path, key)
    finally:
        try:
            _os.unlink(tmp_path)
        except Exception:
            pass


async def persist_generated_code_to_store(
    user_id:        str,
    session_id:     str,
    dataset_id:     str,
    generated_code: str,
    prompt_ts:      str,
    connection_id:  Optional[str] = None,
    storage_uri:    Optional[str] = None,
    *,
    dataset_label:  str = "",
    analysis_label: str = "",
) -> Dict[str, Any]:
    """
    Upload the generated Python script to cloud storage.

    Each call creates a NEW file (keyed by prompt_ts) so a session accumulates
    multiple versioned scripts the user can review and choose from later.
    """
    if not generated_code or not generated_code.strip():
        return {"status": "skipped", "reason": "no code to persist"}
    key = _object_key(
        "code-registry", user_id, session_id, dataset_id, prompt_ts, "transform.py",
        dataset_label=dataset_label, analysis_label=analysis_label,
    )
    store, _ = await _get_store_for_connection(connection_id, storage_uri)
    if not store:
        return {"status": "skipped", "reason": "no cloud store available"}

    await _retry_async(
        _do_upload_bytes, "code→store",
        store, key, generated_code.encode("utf-8"), "text/x-python",
    )
    logger.info("[persist] Code → store key: %s", key)
    return {"status": "success", "object_key": key, "prompt_ts": prompt_ts}


# ═══════════════════════════════════════════════════════════════════════════════
# Task B — Job definition → Git  (planner tool call — not auto-triggered)
# (Leela: "this should be a tool call from planner so user can choose")
# ═══════════════════════════════════════════════════════════════════════════════

async def _ensure_github_branch(repo: str, branch: str) -> None:
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(f"https://api.github.com/repos/{repo}", headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"GitHub repo lookup failed ({r.status_code}): {r.text}")
        default_branch = r.json().get("default_branch", "main")

        r = await client.get(
            f"https://api.github.com/repos/{repo}/git/ref/heads/{default_branch}",
            headers=headers,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Could not get SHA of {default_branch}: {r.text}")
        sha = r.json()["object"]["sha"]

        r = await client.post(
            f"https://api.github.com/repos/{repo}/git/refs",
            headers=headers,
            json={"ref": f"refs/heads/{branch}", "sha": sha},
            timeout=20.0,
        )
        if r.status_code not in (201, 422):  # 422 = already exists
            raise RuntimeError(f"Branch create failed ({r.status_code}): {r.text}")


async def _do_git_put(repo: str, token: str, branch: str, file_path: str,
                      content_b64: str, commit_msg: str) -> None:
    """Raw GitHub file write — wrapped by _retry_async."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    file_url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        get_r  = await client.get(file_url, headers=headers, params={"ref": branch})
        sha    = get_r.json().get("sha") if get_r.status_code == 200 else None
        payload: Dict[str, Any] = {
            "message": commit_msg, "content": content_b64, "branch": branch,
        }
        if sha:
            payload["sha"] = sha
        r = await client.put(file_url, headers=headers, json=payload, timeout=30.0)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"GitHub file write failed ({r.status_code}): {r.text}")


async def persist_job_definition_to_git(
    user_id:            str,
    session_id:         str,
    dataset_id:         str,
    planner_definition: Dict[str, Any],
    coder_definition:   Optional[Dict[str, Any]],
    execution_result:   Dict[str, Any],
    code_object_key:    str,
    prompt_ts:          str,
    *,
    dataset_label:      str = "",
    analysis_label:     str = "",
    version:            Optional[int] = None,
) -> Dict[str, Any]:
    """
    Persist a confirmed job to the Git registry.

    Writes the generated .py transform (so the CODE is visible in the repo —
    Leela's requirement) and, unless AVALOKA_JOB_REGISTRY_WRITE_JSON=false, the
    machine-readable job_definition.json next to it. The GitHub token + repo are
    resolved PER USER from the Settings → Integrations connection, falling back
    to the GITHUB_SYSTEM_TOKEN / GITHUB_JOB_REGISTRY_REPO env vars.

    ONLY writes when execution succeeded.
    """
    # Resolve the per-user connection (falls back to env vars inside the resolver).
    from app.api.integrations import resolve_github_config

    _exec_status = str((execution_result or {}).get("status", "")).lower()
    if _exec_status not in ("success", "succeeded", "completed", "done"):
        return {"status": "skipped", "reason": "execution not successful"}

    gh = await resolve_github_config(user_id)
    if not gh:
        return {"status": "skipped",
                "reason": "no GitHub connection for this user and no system token"}
    repo  = gh.repo
    token = gh.token

    branch = "main"                                   # push directly to main
    #base   = f"jobs/{user_id}/{session_id}/{dataset_id}_{prompt_ts}"
    ds   = f"{_slug(dataset_label) or 'dataset'}-{_short(dataset_id)}"
    an   = f"{_slug(analysis_label) or 'analysis'}-{_short(session_id)}"
    #base = f"jobs/{ds}/{an}/v{version}__{prompt_ts}"
    base = f"jobs/{ds}/{an}/v{version or 1}__{prompt_ts}"
    written: List[str] = []

    # 1) The .py transform — the human-readable code Leela wants in the repo.
    code = (coder_definition or {}).get("code")
    if isinstance(code, str) and code.strip():
        py_path = f"{base}_transform.py"
        py_b64  = base64.b64encode(code.encode("utf-8")).decode()
        await _retry_async(
            _do_git_put, "code.py→Git",
            repo, token, branch, py_path, py_b64,
            f"[Avaloka] Code persisted — {dataset_id} — {prompt_ts}",
        )
        written.append(py_path)
        logger.info("[persist] Code .py → Git %s@%s:%s", repo, branch, py_path)
    else:
        logger.info("[persist] No code in coder_definition; skipping .py commit "
                    "(ds=%s ts=%s)", dataset_id, prompt_ts)

    # 2) The job_definition.json — machine record (toggle-off with the env var).
    json_path: Optional[str] = None
    if WRITE_JOB_DEFINITION_JSON:
        job_def = {
            "job_id":             f"{session_id}_{dataset_id}_{prompt_ts}",
            "dataset_id":         dataset_id,
            "prompt_ts":          prompt_ts,
            "planner_definition": planner_definition,
            "coder_definition":   coder_definition or {},
            "execution_result":   execution_result,
            "code_object_key":    code_object_key,
            "code_git_path":      (written[0] if written else None),   # link to the .py
            "created_at":         datetime.utcnow().isoformat(),
        }
        json_path   = f"{base}_job_definition.json"
        content_b64 = base64.b64encode(
            json.dumps(job_def, indent=2, default=str).encode()
        ).decode()
        await _retry_async(
            _do_git_put, "job→Git",
            repo, token, branch, json_path, content_b64,
            f"[Avaloka] Job persisted — {dataset_id} — {prompt_ts}",
        )
        written.append(json_path)

    if not written:
        return {"status": "skipped", "reason": "nothing to write (no code, JSON disabled)"}

    return {
        "status":    "success",
        "branch":    branch,
        "repo":      repo,
        "source":    gh.source,          # "connection" | "env" — useful in logs
        "file_path": written[0],
        "written":   written,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Task C — Execution output → Cloud storage  (per-prompt timestamped file)
# (Leela: "stored by chat timestamp so we know which output is for each prompt")
# ═══════════════════════════════════════════════════════════════════════════════


async def persist_execution_output_to_store(
    user_id:          str,
    session_id:       str,
    dataset_id:       str,
    output_file_data: Optional[Dict[str, Any]],
    output_json:      Optional[List[Dict[str, Any]]],
    output_location:  Optional[str],
    prompt_ts:        str,
    connection_id:    Optional[str] = None,
    storage_uri:      Optional[str] = None,
    *,
    dataset_label:    str = "",
    analysis_label:   str = "",
) -> Dict[str, Any]:
    """
    Upload execution output CSV to cloud storage — one file per prompt run.
    Priority: data-uri CSV blob → output_json rows → local file path.
    """
    if not output_file_data and not output_json and not output_location:
        return {"status": "skipped", "reason": "no output to persist"}

    key = _object_key(
        "execution-outputs", user_id, session_id, dataset_id, prompt_ts, "output.csv",
        dataset_label=dataset_label, analysis_label=analysis_label,
    )

    # ── Resolve CSV bytes ─────────────────────────────────────────────────────
    csv_bytes: Optional[bytes] = None
    content_str = (output_file_data or {}).get("content", "")
    if content_str.startswith("data:text/csv;base64,"):
        csv_bytes = base64.b64decode(content_str.split("base64,", 1)[1])
    elif output_json:
        buf = io.StringIO()
        # Rows can be ragged; collect the union of keys (first-seen order) so a
        # later row with an extra key doesn't raise ValueError on writerows.
        fieldnames: List[str] = []
        seen: set = set()
        for row in output_json:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_json)
        csv_bytes = buf.getvalue().encode("utf-8")
    elif output_location:
        import os as _os
        if _os.path.exists(output_location):
            with open(output_location, "rb") as f:
                csv_bytes = f.read()

    if not csv_bytes:
        return {"status": "skipped", "reason": "no readable output source"}

    store, _ = await _get_store_for_connection(connection_id, storage_uri)
    if not store:
        return {"status": "skipped", "reason": "no cloud store available"}

    await _retry_async(
        _do_upload_bytes, "output→store", store, key, csv_bytes, "text/csv"
    )
    logger.info("[persist] Execution output → store key: %s", key)
    return {"status": "success", "object_key": key, "prompt_ts": prompt_ts}


# ═══════════════════════════════════════════════════════════════════════════════
# Task D — Visualization config → Cloud storage
# (Leela: "how do we store visualization data — we dont want to run
#          visualizations again")
# ═══════════════════════════════════════════════════════════════════════════════


async def persist_visualization_to_store(
    user_id:              str,
    session_id:           str,
    dataset_id:           str,
    visualization_config: Dict[str, Any],
    prompt_ts:            str,
    connection_id:        Optional[str] = None,
    storage_uri:          Optional[str] = None,
    *,
    dataset_label:        str = "",
    analysis_label:       str = "",
) -> Dict[str, Any]:
    """
    Persist the visualization config JSON so charts can be restored without
    re-running the visualization agent.
    """
    if not visualization_config:
        return {"status": "skipped", "reason": "no visualization config"}

    key = _object_key(
        "visualization-configs", user_id, session_id, dataset_id, prompt_ts, "viz_config.json",
        dataset_label=dataset_label, analysis_label=analysis_label,
    )
    viz_bytes = json.dumps(visualization_config, default=str).encode("utf-8")

    store, _ = await _get_store_for_connection(connection_id, storage_uri)
    if not store:
        return {"status": "skipped", "reason": "no cloud store available"}

    await _retry_async(
        _do_upload_bytes, "viz→store", store, key, viz_bytes, "application/json"
    )
    logger.info("[persist] Visualization config → store key: %s", key)
    return {"status": "success", "object_key": key, "prompt_ts": prompt_ts}


# ═══════════════════════════════════════════════════════════════════════════════
# Session asset-history helper
# ═══════════════════════════════════════════════════════════════════════════════

def _append_asset(
    sess: Dict[str, Any], list_key: str,
    object_key: Optional[str], prompt_ts: str,
) -> None:
    """
    Append an asset record to the session's per-type history list.
    Each entry: {"object_key": "...", "prompt_ts": "..."}
    This lets the UI show all code / output versions for the session.
    """
    if not object_key:
        return
    existing = sess.get(list_key) or []
    if isinstance(existing, str):
        try:
            existing = json.loads(existing)
        except Exception:
            existing = []
    existing.append({"object_key": object_key, "prompt_ts": prompt_ts})
    sess[list_key] = existing

# ═══════════════════════════════════════════════════════════════════════════════
# Background orchestrator — called via asyncio.create_task()
#
# TRIGGER CONDITION: only fires when execution ran or code was generated as part
# of a run.  NOT called on every conversational turn.
# (Leela: "this happens when we schedule or run, not during every interaction")
# ═══════════════════════════════════════════════════════════════════════════════

async def _persist_assets_background(
    user_id:              str,
    session_id:           str,
    dataset_id:           str,
    generated_code:       Optional[str],
    planner_definition:   Optional[Dict[str, Any]],
    coder_definition:     Optional[Dict[str, Any]],
    execution_result:     Dict[str, Any],
    exec_succeeded:       bool,
    output_file_data:     Optional[Dict[str, Any]],
    output_json:          Optional[List[Dict[str, Any]]],
    output_location:      Optional[str],
    visualization_config: Optional[Dict[str, Any]],
    connection_id:        Optional[str],
    storage_uri:          Optional[str] = None,
    *,
    dataset_label:        str = "",
    analysis_label:       str = "",
) -> None:
    """
    Runs AFTER ChatResponse is returned.  All task errors are caught and written
    to the Redis session as persist_errors so the UI can notify the user.
    (Leela: "retry with exponential backoff … post an error to UI on 3rd failure")
    """
    from app.services.session_service import (  # type: ignore[import-not-found]
        get_session, save_session, update_session,
    )

    # One timestamp anchors all assets from this execution together
    prompt_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    # Version number = how many code assets already exist + 1
    _sess0 = await get_session(session_id) or {}
    _hist  = _sess0.get("code_assets") or []
    if isinstance(_hist, str):
        _hist = json.loads(_hist or "[]")
    version = len(_hist) + 1

    code_key:   Optional[str] = None

    out_key:    Optional[str] = None
    viz_key:    Optional[str] = None
    git_branch: Optional[str] = None
    git_repo:   Optional[str] = None
    errors:     List[str]     = []

    # ── Task A: code → cloud store ────────────────────────────────────────────
    if generated_code:
        try:
            r = await persist_generated_code_to_store(
                user_id, session_id, dataset_id, generated_code,
                prompt_ts, connection_id, storage_uri,
                dataset_label=dataset_label, analysis_label=analysis_label,
            )
            code_key = r.get("object_key") if r.get("status") == "success" else None
        except Exception as exc:
            errors.append(f"Code persist failed: {exc}")
            logger.error("[persist] Task A failed: %s", exc, exc_info=True)

    # ── Task B: job definition → Git (safety-net path; planner tool is primary)
    if exec_succeeded and planner_definition:
        try:
            r = await persist_job_definition_to_git(
            user_id, session_id, dataset_id,
            planner_definition, coder_definition,
            execution_result, code_key or "not-persisted",
            prompt_ts,
            dataset_label=dataset_label,
            analysis_label=analysis_label,
            version=version,
            )
            # git_branch = r.get("branch") if r.get("status") == "success" else None
            if r.get("status") == "success":
                git_branch = r.get("branch")
                git_repo   = r.get("repo")
        except Exception as exc:
            errors.append(f"Git persist failed: {exc}")
            logger.error("[persist] Task B failed: %s", exc, exc_info=True)

   
    # ── Task C: output → cloud store ──────────────────────────────────────────
    if output_file_data or output_json or output_location:
        try:
            r = await persist_execution_output_to_store(
                user_id, session_id, dataset_id,
                output_file_data, output_json, output_location,
                prompt_ts, connection_id, storage_uri,
                dataset_label=dataset_label, analysis_label=analysis_label,
            )
            out_key = r.get("object_key") if r.get("status") == "success" else None
        except Exception as exc:
            errors.append(f"Output persist failed: {exc}")
            logger.error("[persist] Task C failed: %s", exc, exc_info=True)

    
    # ── Task D: visualization config → cloud store ────────────────────────────
    if visualization_config:
        try:
            r = await persist_visualization_to_store(
                user_id, session_id, dataset_id,
                visualization_config, prompt_ts,
                connection_id, storage_uri,
                dataset_label=dataset_label, analysis_label=analysis_label,
            )
            viz_key = r.get("object_key") if r.get("status") == "success" else None
        except Exception as exc:
            errors.append(f"Viz persist failed: {exc}")
            logger.error("[persist] Task D failed: %s", exc, exc_info=True)

    # ── Update Redis session ──────────────────────────────────────────────────
    if not any([code_key, out_key, viz_key, git_branch, errors]):
        return

    def _apply_asset_updates(sess: Dict[str, Any]) -> None:
        # Append to per-type history lists (user can pick any version)
        _append_asset(sess, "code_assets",  code_key, prompt_ts)
        _append_asset(sess, "output_assets", out_key, prompt_ts)
        _append_asset(sess, "viz_assets",   viz_key,  prompt_ts)
        
        vp = sess.get("version_prompts") or {}
        if isinstance(vp, str):
            vp = json.loads(vp or "{}")
        vp[prompt_ts] = analysis_label      # the user's actual question for this version
        sess["version_prompts"] = vp

        # Flat keys for quick access by retrieval API (most recent)
        if code_key:   sess["gcs_code_object_key"]   = code_key
        if out_key:    sess["gcs_output_object_key"]  = out_key
        if viz_key:    sess["gcs_viz_object_key"]     = viz_key
        if git_branch: sess["git_job_branch"]          = git_branch
        if git_repo:   sess["git_job_repo"]   = git_repo   

        # Surface errors to UI
        if errors:
            sess["persist_errors"] = errors
        else:
            sess.pop("persist_errors", None)

    try:
        # Atomic read-modify-write: re-reads the latest session under a lock and
        # applies only these asset fields, so a concurrent chat turn's save can't
        # be discarded (and vice-versa).
        await update_session(session_id, _apply_asset_updates)
        logger.info(
            "[persist] Session %s updated — code=%s out=%s viz=%s git=%s errors=%d",
            session_id, code_key, out_key, viz_key, git_branch, len(errors),
        )
    except Exception as exc:
        logger.error("[persist] Session update failed: %s", exc, exc_info=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Planner Tool definitions
# (Leela: "This should be an agent with a tool call so planner knows when to
#  write it — it should not be called directly")
# ═══════════════════════════════════════════════════════════════════════════════

def _make_persistence_tools():
    try:
        from langchain_core.tools import tool  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("[persist] langchain_core not available — planner tools not registered")
        return []

    @tool
    async def save_job_to_git(
        user_id: str,
        session_id: str,
        dataset_id: str,
        planner_definition: str,  # JSON string
        coder_definition: str,    # JSON string
        execution_result: str,    # JSON string
        code_object_key: str,
        prompt_ts: str,
    ) -> str:
        """
        Save the confirmed ETL job definition JSON to the Avaloka Git registry.
        Call this when the user approves a job and execution succeeds so the job
        can be versioned and later deployed to other cloud environments.
        """
        try:
            r = await persist_job_definition_to_git(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                planner_definition=json.loads(planner_definition),
                coder_definition=json.loads(coder_definition) if coder_definition else {},
                execution_result=json.loads(execution_result),
                code_object_key=code_object_key,
                prompt_ts=prompt_ts,
            )
            if r["status"] == "success":
                return (
                    f"Job definition saved to Git. Branch: {r['branch']}, "
                    f"Repo: {r['repo']}, File: {r['file_path']}"
                )
            return f"Job definition not saved: {r.get('reason', 'unknown')}"
        except Exception as exc:
            return f"Failed to save job definition: {exc}"

    @tool
    async def save_code_to_storage(
        user_id: str,
        session_id: str,
        dataset_id: str,
        generated_code: str,
        prompt_ts: str,
        connection_id: str = "",
        storage_uri: str = "",
    ) -> str:
        """
        Save the generated Python transformation script to cloud storage.
        Each call stores a new per-prompt timestamped file so the user can
        review all generated versions and choose which one to use.
        """
        try:
            r = await persist_generated_code_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                generated_code=generated_code,
                prompt_ts=prompt_ts,
                connection_id=connection_id or None,
                storage_uri=storage_uri or None,
            )
            if r["status"] == "success":
                return f"Code saved at: {r['object_key']}"
            return f"Code not saved: {r.get('reason', 'unknown')}"
        except Exception as exc:
            return f"Failed to save code: {exc}"

    return [save_job_to_git, save_code_to_storage]


# Module-level tool list — import this in the planner agent
PERSISTENCE_TOOLS = _make_persistence_tools()