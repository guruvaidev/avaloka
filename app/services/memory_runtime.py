"""
Runtime guardrails for the context-memory stack.

Local and CI runs may use fallbacks, but production should fail fast when the
memory layers are not backed by real infrastructure.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Dict, List
from urllib.parse import urlparse


PRODUCTION_ENVS = {"prod", "production"}


class MemoryConfigError(RuntimeError):
    """Raised when context-memory infra is unsafe for the configured runtime."""


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def current_environment() -> str:
    """Return the normalized deployment environment name."""
    return (
        os.environ.get("APP_ENV")
        or os.environ.get("AVALOKA_ENV")
        or os.environ.get("ENVIRONMENT")
        or "development"
    ).strip().lower()


def strict_memory_infra_enabled() -> bool:
    """
    True when the app should reject dev-only fallbacks.

    Explicit MEMORY_STRICT_INFRA=true always wins. Production enables strict
    mode unless MEMORY_ALLOW_MEMORY_FALLBACKS=true is set intentionally.
    """
    if _env_flag("MEMORY_STRICT_INFRA", False):
        return True

    if current_environment() in PRODUCTION_ENVS:
        return not _env_flag("MEMORY_ALLOW_MEMORY_FALLBACKS", False)

    return False


def ensure_fallback_allowed(layer_name: str, reason: str) -> None:
    """Raise in strict mode when a memory layer tries to use a dev fallback."""
    if strict_memory_infra_enabled():
        raise MemoryConfigError(
            f"{layer_name} fallback is disabled in strict memory mode: {reason}"
        )


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _postgres_url_is_real(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"postgresql", "postgresql+psycopg2", "postgres"}


def validate_memory_runtime_config() -> Dict[str, str]:
    """
    Validate production memory configuration.

    Returns a small status dictionary in non-strict/dev mode. Raises
    MemoryConfigError in strict mode when required production config is missing.
    """
    strict = strict_memory_infra_enabled()
    issues: List[str] = []

    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not redis_url:
        issues.append("REDIS_URL is required for Layer 1 domain context.")

    if not _module_available("redis"):
        issues.append("redis package is required for Layer 1 domain context.")

    chroma_storage = os.environ.get("CHROMA_STORAGE_PATH", "").strip()
    if not chroma_storage:
        issues.append("CHROMA_STORAGE_PATH must be explicit for Layer 2 usage context.")

    if not _module_available("chromadb"):
        issues.append("chromadb package is required for Layer 2 usage context.")

    postgres_url = os.environ.get("POSTGRES_URL", "").strip()
    if not postgres_url:
        issues.append("POSTGRES_URL is required for Layer 3 artifact context.")
    elif not _postgres_url_is_real(postgres_url):
        issues.append("POSTGRES_URL must point to PostgreSQL, not SQLite/local fallback.")

    if not _module_available("sqlalchemy"):
        issues.append("sqlalchemy package is required for Layer 3 artifact context.")

    if not _module_available("psycopg2") and not _module_available("psycopg"):
        issues.append("psycopg2-binary or psycopg is required for PostgreSQL access.")

    milvus_host = os.environ.get("MILVUS_HOST", "").strip()
    milvus_port = os.environ.get("MILVUS_PORT", "").strip()
    milvus_collection = os.environ.get("MILVUS_COLLECTION", "").strip()
    if not milvus_host:
        issues.append("MILVUS_HOST is required for Layer 4 time-series memory.")
    if not milvus_port:
        issues.append("MILVUS_PORT is required for Layer 4 time-series memory.")
    if not milvus_collection:
        issues.append("MILVUS_COLLECTION is required for Layer 4 time-series memory.")

    if not _module_available("pymilvus"):
        issues.append("pymilvus package is required for Layer 4 time-series memory.")

    has_openai_embeddings = bool(os.environ.get("OPENAI_API_KEY"))
    has_local_embeddings = _module_available("sentence_transformers")
    if not has_openai_embeddings and not has_local_embeddings:
        issues.append(
            "A real embedding provider is required: set OPENAI_API_KEY or install sentence-transformers."
        )

    if strict and issues:
        joined = "\n- ".join(issues)
        raise MemoryConfigError(f"Invalid production context-memory config:\n- {joined}")

    return {
        "environment": current_environment(),
        "strict": str(strict),
        "status": "invalid" if issues else "ok",
        "issue_count": str(len(issues)),
    }
