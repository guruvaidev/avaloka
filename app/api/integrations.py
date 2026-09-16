"""app/api/integrations.py

UI-managed third-party integrations (GitHub first; Outlook/Slack/JIRA later).

This module holds the *reusable* pieces so both the API endpoints in server.py
and the asset-persistence layer can share them without a circular import:

  - Supabase CRUD for the integration_connections table
  - GitHub token/repo validation
  - resolve_github_config(): the per-user token+repo resolver with an env-var
    fallback, so existing deployments keep working while connections migrate
    from `export GITHUB_SYSTEM_TOKEN=...` to the Settings → Integrations UI.

Secret handling reuses the exact AES-GCM scheme used by mcp_connections /
cloud connections (encrypt_secret/decrypt_secret), so ciphertext stays
byte-compatible with everything else in the project.
"""
from __future__ import annotations

import os
import re
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, ConfigDict

# Same encryption + Supabase client the rest of the codebase already uses.
from app.api.cloud_connections import encrypt_secret, decrypt_secret  # noqa: F401
from app.agents.sampling_persistence import get_supabase_client

logger = logging.getLogger(__name__)

INTEGRATION_TABLE = os.getenv("SUPABASE_INTEGRATION_CONNECTIONS_TABLE", "integration_connections")
SUPPORTED_INTEGRATION_PROVIDERS = {"github", "outlook", "slack", "jira"}

GITHUB_API_BASE = "https://api.github.com"
# owner/repo — GitHub allows alphanumerics, dash, underscore and dot in both.
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


# --------------------------------------------------------------------------- #
# Request models (accept camelCase from the frontend too, like McpCredentialsIn)
# --------------------------------------------------------------------------- #
class GitHubConnectIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    token: str = Field(..., min_length=1)
    repo: str = Field(..., min_length=1)


class GitHubUpdateIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    enabled: Optional[bool] = None
    repo: Optional[str] = None


@dataclass
class GitHubConfig:
    """Resolved GitHub target for a user's job persistence."""
    token: str
    repo: str
    source: str  # "connection" (UI-configured) | "env" (system fallback)


# --------------------------------------------------------------------------- #
# Supabase CRUD (sync — call via asyncio.to_thread from async endpoints)
# --------------------------------------------------------------------------- #
def get_integration_connection(user_id: str, provider: str) -> Optional[Dict[str, Any]]:
    client = get_supabase_client()
    res = (
        client.table(INTEGRATION_TABLE)
        .select("id,user_id,provider,enabled,config,token_ciphertext,token_iv,created_at,updated_at")
        .eq("user_id", user_id)
        .eq("provider", provider)
        .limit(1)
        .execute()
    )
    data = getattr(res, "data", None) or []
    return data[0] if data else None


def list_integration_connections(user_id: str) -> List[Dict[str, Any]]:
    client = get_supabase_client()
    res = (
        client.table(INTEGRATION_TABLE)
        # token_ciphertext is selected only to compute a has_token boolean;
        # the raw value is never returned to the client.
        .select("id,provider,enabled,config,token_ciphertext,created_at,updated_at")
        .eq("user_id", user_id)
        .execute()
    )
    return getattr(res, "data", None) or []


def save_integration_connection(
    user_id: str,
    provider: str,
    *,
    enabled: Optional[bool] = None,
    config_updates: Optional[Dict[str, Any]] = None,
    token_ciphertext: Optional[str] = None,
    token_iv: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert or update the (user_id, provider) row.

    `config_updates` is merged into the existing config so a repo change does
    not wipe other keys, and a plain enable/disable toggle does not wipe the
    repo. Token columns are only touched when a new token is supplied, so a
    PATCH that just flips `enabled` keeps the stored token intact.
    """
    client = get_supabase_client()
    existing = get_integration_connection(user_id, provider)

    merged_config = dict(existing.get("config") or {}) if existing else {}
    if config_updates:
        merged_config.update(config_updates)

    payload: Dict[str, Any] = {
        "user_id": user_id,
        "provider": provider,
        "config": merged_config,
    }
    if enabled is not None:
        payload["enabled"] = enabled
    elif not existing:
        payload["enabled"] = False
    if token_ciphertext is not None:
        payload["token_ciphertext"] = token_ciphertext
        payload["token_iv"] = token_iv

    res = (
        client.table(INTEGRATION_TABLE)
        .upsert(payload, on_conflict="user_id,provider")
        .execute()
    )
    data = getattr(res, "data", None) or []
    return data[0] if data else payload


def delete_integration_connection(user_id: str, provider: str) -> bool:
    client = get_supabase_client()
    res = (
        client.table(INTEGRATION_TABLE)
        .delete()
        .eq("user_id", user_id)
        .eq("provider", provider)
        .execute()
    )
    return bool(getattr(res, "data", None))


# --------------------------------------------------------------------------- #
# GitHub validation
# --------------------------------------------------------------------------- #
async def validate_github_repo_access(token: str, repo: str) -> Dict[str, Any]:
    """Verify the token can reach `repo` and whether it has push (write) access.

    Job persistence commits to the repo, so read-only access is not enough —
    the caller should reject a connection whose `push` is False.
    Returns {"ok", "push", "error", "full_name"}.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{GITHUB_API_BASE}/repos/{repo}", headers=headers)
    except Exception as exc:
        return {"ok": False, "push": False, "error": f"Could not reach GitHub: {exc}", "full_name": None}

    if r.status_code == 200:
        data = r.json()
        push = bool((data.get("permissions") or {}).get("push"))
        return {"ok": True, "push": push, "error": None, "full_name": data.get("full_name")}
    if r.status_code == 401:
        return {"ok": False, "push": False, "error": "Invalid or expired token.", "full_name": None}
    if r.status_code == 403:
        return {"ok": False, "push": False,
                "error": "Token is valid but forbidden for this repo (SSO not authorized, or scope missing).",
                "full_name": None}
    if r.status_code == 404:
        return {"ok": False, "push": False,
                "error": "Repository not found, or the token cannot see it.", "full_name": None}
    return {"ok": False, "push": False, "error": f"GitHub returned HTTP {r.status_code}.", "full_name": None}


# --------------------------------------------------------------------------- #
# The resolver — the piece that actually makes it "connected through the UI"
# --------------------------------------------------------------------------- #
async def resolve_github_config(user_id: Optional[str]) -> Optional[GitHubConfig]:
    """Resolve which GitHub token+repo to use for this user's job persistence.

    Order:
      1) the user's enabled UI connection (token decrypted from Supabase)
      2) the process-wide GITHUB_SYSTEM_TOKEN / GITHUB_JOB_REGISTRY_REPO env
         vars (so nothing breaks before every user has migrated to the UI)
    Returns None when neither is usable (caller should then skip Git push).
    """
    if user_id:
        try:
            row = await asyncio.to_thread(get_integration_connection, user_id, "github")
        except Exception:
            logger.warning("[integrations] github connection lookup failed for %s", user_id, exc_info=True)
            row = None

        if row and row.get("enabled"):
            token = decrypt_secret({
                "ciphertext": row.get("token_ciphertext"),
                "iv": row.get("token_iv"),
            })
            config = row.get("config") or {}
            repo = (config.get("repo") if isinstance(config, dict) else None) or ""
            repo = repo.strip()
            if token and repo:
                return GitHubConfig(token=token, repo=repo, source="connection")
            logger.warning(
                "[integrations] github connection for %s is enabled but missing %s",
                user_id, "token" if not token else "repo",
            )

    env_token = os.getenv("GITHUB_SYSTEM_TOKEN", "").strip()
    env_repo = os.getenv("GITHUB_JOB_REGISTRY_REPO", "").strip()
    if env_token and env_repo:
        return GitHubConfig(token=env_token, repo=env_repo, source="env")
    return None


def resolve_github_config_sync(user_id: Optional[str]) -> Optional[GitHubConfig]:
    """Sync variant for non-async call sites (e.g. Celery workers / persistence
    running in a thread). Same precedence as resolve_github_config()."""
    if user_id:
        try:
            row = get_integration_connection(user_id, "github")
        except Exception:
            logger.warning("[integrations] github connection lookup failed for %s", user_id, exc_info=True)
            row = None
        if row and row.get("enabled"):
            token = decrypt_secret({
                "ciphertext": row.get("token_ciphertext"),
                "iv": row.get("token_iv"),
            })
            config = row.get("config") or {}
            repo = ((config.get("repo") if isinstance(config, dict) else None) or "").strip()
            if token and repo:
                return GitHubConfig(token=token, repo=repo, source="connection")

    env_token = os.getenv("GITHUB_SYSTEM_TOKEN", "").strip()
    env_repo = os.getenv("GITHUB_JOB_REGISTRY_REPO", "").strip()
    if env_token and env_repo:
        return GitHubConfig(token=env_token, repo=env_repo, source="env")
    return None
    