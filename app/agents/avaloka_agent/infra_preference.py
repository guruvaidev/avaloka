"""Resolve the user's preferred infrastructure platform.

Priority:
1. Explicit signal in the most recent user message ("use GCP", "switch to AWS").
2. Last known infra preference from Redis thread context.
3. Cloud URI inference (s3 → aws, gs → gcp).
4. ``DEFAULT_INFRA_PLATFORM`` env (already used by the planner).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

from langchain_core.messages import BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

_GCP_PATTERNS = (
    r"\buse\s+(?:my\s+)?gke\b",
    r"\buse\s+gcp\b",
    r"\bswitch\s+to\s+gcp\b",
    r"\bon\s+google\s+cloud\b",
)
_AWS_PATTERNS = (
    r"\buse\s+(?:my\s+)?eks\b",
    r"\buse\s+aws\b",
    r"\bswitch\s+to\s+aws\b",
    r"\bon\s+amazon\s+web\s+services\b",
)
_AZURE_PATTERNS = (
    r"\buse\s+aks\b",
    r"\buse\s+azure\b",
    r"\bswitch\s+to\s+azure\b",
)
_LOCAL_PATTERNS = (
    r"\brun\s+(?:this\s+)?locally\b",
    r"\bno\s+cluster\b",
    r"\blocal\s+only\b",
)


def _last_human(messages):
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return (m.content if isinstance(m.content, str) else "") or ""
    return ""


def _explicit_platform_signal(text: str) -> Optional[str]:
    t = (text or "").lower()
    if not t:
        return None
    for pat in _LOCAL_PATTERNS:
        if re.search(pat, t):
            return "local"
    for pat in _GCP_PATTERNS:
        if re.search(pat, t):
            return "gcp"
    for pat in _AWS_PATTERNS:
        if re.search(pat, t):
            return "aws"
    for pat in _AZURE_PATTERNS:
        if re.search(pat, t):
            return "azure"
    return None


def _infer_from_uri(state: Dict[str, Any]) -> Optional[str]:
    uri = (
        state.get("data_source_location_cloud")
        or state.get("data_source_location")
        or ""
    ).lower()
    if uri.startswith("s3://"):
        return "aws"
    if uri.startswith(("gs://", "gcs://")):
        return "gcp"
    if uri.startswith(("az://", "azure://")):
        return "azure"
    return None


def resolve_infra_preference(
    state: Dict[str, Any],
    redis_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Returns updates to be merged into ``state``.

    Sets ``default_infra_platform`` when we can determine it. Does NOT touch
    ``infrastructure_request`` — the planner builds that when it decides
    provisioning is needed.
    """
    messages = state.get("messages") or []
    explicit = _explicit_platform_signal(_last_human(messages))
    if explicit:
        if explicit == "local":
            return {"execution_mode": "local", "default_infra_platform": "local"}
        return {"default_infra_platform": explicit}

    if redis_context:
        prior = redis_context.get("last_infra_platform")
        if prior in ("aws", "gcp", "azure"):
            return {"default_infra_platform": prior}

    inferred = _infer_from_uri(state)
    if inferred:
        return {"default_infra_platform": inferred}

    env_default = os.getenv("DEFAULT_INFRA_PLATFORM")
    if env_default:
        return {"default_infra_platform": env_default}

    return {}
