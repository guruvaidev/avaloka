"""Plan-aware upload limits (Free / Professional / Enterprise).

One table of limits per plan, used by the upload endpoint, the upload
pre-check middleware and GET /api/limits, so the backend and the frontend
enforce the same numbers.

Where the plan comes from (Supabase):

    auth.uid --profiles.user_id--> profiles.id
    subscriptions.owner_profile_id = profiles.id          (personal plan)
    subscriptions.organization_id  = an org the user owns  (organizations.owner_profile_id)
                                  or belongs to            (team_memberships -> organization_teams)
    subscriptions.plan_id -> plans.plan_type in {free, professional, enterprise}

A subscription counts only when its status is active/trialing AND its period
(or trial) has not ended: the payment webhooks leave `status` stale, so a
Stripe row still says 'active' after current_period_end, and a PayPal trial
still says 'trialing' weeks after trial_ends_at. A short grace period covers a
renewal webhook that arrives late. No valid subscription means Free.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

MB = 1024 * 1024

PLAN_FREE = "free"
PLAN_PROFESSIONAL = "professional"
PLAN_ENTERPRISE = "enterprise"
_PLAN_RANK = {PLAN_FREE: 0, PLAN_PROFESSIONAL: 1, PLAN_ENTERPRISE: 2}

_VALID_STATUSES = {"active", "trialing"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("[plan-limits] %s is not an integer; using %s", name, default)
        return default


@dataclass(frozen=True)
class PlanLimits:
    plan: str
    max_total_bytes: int      # all files of one upload together
    max_file_bytes: int       # any single file
    max_files: int
    batch_jobs: bool          # may work above the sync limit run as a batch job

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["max_total_mb"] = round(self.max_total_bytes / MB, 1)
        out["max_file_mb"] = round(self.max_file_bytes / MB, 1)
        return out


def limits_for(plan: Optional[str]) -> PlanLimits:
    """Limits for a plan name; anything unknown is treated as Free."""
    plan = (plan or PLAN_FREE).strip().lower()
    # Paid plans keep today's server-wide caps until the batch upload path
    # (Phase 2) lets them go beyond what one request can process.
    paid_total = _env_int("AVALOKA_MAX_UPLOAD_TOTAL_BYTES", 200 * MB)
    paid_file = _env_int("AVALOKA_MAX_UPLOAD_FILE_BYTES", 100 * MB)
    paid_files = _env_int("AVALOKA_MAX_UPLOAD_FILES", 10)

    if plan == PLAN_ENTERPRISE:
        return PlanLimits(PLAN_ENTERPRISE, paid_total, paid_file, paid_files, batch_jobs=True)
    if plan == PLAN_PROFESSIONAL:
        return PlanLimits(PLAN_PROFESSIONAL, paid_total, paid_file, paid_files, batch_jobs=True)

    # Free: 100 MB of data per analysis, all files together (confirmed 2026-10-04).
    free_total = _env_int("AVALOKA_FREE_MAX_TOTAL_BYTES", 100 * MB)
    return PlanLimits(
        PLAN_FREE,
        max_total_bytes=free_total,
        max_file_bytes=min(free_total, paid_file),
        max_files=paid_files,
        batch_jobs=False,
    )


def _human(num_bytes: int) -> str:
    if num_bytes >= MB:
        return f"{num_bytes / MB:.0f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes} bytes"


def limit_error(limits: PlanLimits, reason: str, actual: int) -> Dict[str, Any]:
    """Structured body for a 413, so the UI can show the right dialog.

    reason: 'total_size' | 'file_size' | 'file_count'
    """
    if reason == "file_count":
        message = (f"Too many files: {actual}. The {limits.plan.title()} plan allows "
                   f"{limits.max_files} files per upload.")
        limit_value = limits.max_files
    else:
        limit_value = limits.max_total_bytes if reason == "total_size" else limits.max_file_bytes
        scope = "in total" if reason == "total_size" else "per file"
        message = (f"This upload is {_human(actual)}. The {limits.plan.title()} plan allows "
                   f"{_human(limit_value)} {scope}.")
    upgrade = limits.plan == PLAN_FREE and reason != "file_count"
    if upgrade:
        message += " Reduce the file size or upgrade to Professional for larger datasets."
    return {
        "code": "plan_limit_exceeded",
        "reason": reason,
        "plan": limits.plan,
        "limit": limit_value,
        "actual": actual,
        "upgrade_available": upgrade,
        "upgrade_url": os.getenv("AVALOKA_UPGRADE_URL", "/pricing") if upgrade else None,
        "message": message,
    }


def check_sizes(limits: PlanLimits, sizes: Iterable[int]) -> Optional[Dict[str, Any]]:
    """None when the files fit the plan, else the 413 body for the first breach."""
    sizes = [int(s or 0) for s in sizes]
    if len(sizes) > limits.max_files:
        return limit_error(limits, "file_count", len(sizes))
    for s in sizes:
        if s > limits.max_file_bytes:
            return limit_error(limits, "file_size", s)
    total = sum(sizes)
    if total > limits.max_total_bytes:
        return limit_error(limits, "total_size", total)
    return None


# ---------------------------------------------------------------------------
# Plan resolution
# ---------------------------------------------------------------------------

def _get_client():
    # Imported lazily so this module (and its tests) load without Supabase.
    from app.agents.sampling_persistence import get_supabase_client
    return get_supabase_client()


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip().replace(" ", "T", 1)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # "+00" -> "+00:00" (Postgres short offset)
    if len(text) >= 3 and text[-3] in "+-" and text[-2:].isdigit():
        text += ":00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def subscription_is_current(sub: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    """Active/trialing AND not past its end date (plus a short grace period)."""
    status = str(sub.get("status") or "").lower()
    if status not in _VALID_STATUSES:
        return False
    now = now or datetime.now(timezone.utc)
    grace = timedelta(days=_env_int("AVALOKA_PLAN_GRACE_DAYS", 3))
    if status == "trialing":
        end = _parse_ts(sub.get("trial_ends_at")) or _parse_ts(sub.get("current_period_end"))
    else:
        end = _parse_ts(sub.get("current_period_end"))
    return end is None or now <= end + grace


def _rows(res: Any) -> List[Dict[str, Any]]:
    return list(getattr(res, "data", None) or [])


def _lookup_plan(user_id: str) -> str:
    client = _get_client()

    profile_rows = _rows(client.table("profiles").select("id").eq("user_id", user_id).limit(1).execute())
    profile_id = str(profile_rows[0]["id"]) if profile_rows else None
    if not profile_id:
        return PLAN_FREE

    fields = "status,plan_id,current_period_end,trial_ends_at,organization_id"
    subs = _rows(client.table("subscriptions").select(fields).eq("owner_profile_id", profile_id).execute())

    # Organisations the user owns or belongs to share the org's subscription.
    org_ids = {r["id"] for r in _rows(
        client.table("organizations").select("id").eq("owner_profile_id", profile_id).execute())}
    # team_memberships.user_id may hold the auth uid or the profile id; accept both.
    team_ids = {r["team_id"] for r in _rows(
        client.table("team_memberships").select("team_id").in_("user_id", [user_id, profile_id]).execute())}
    if team_ids:
        org_ids |= {r["organization_id"] for r in _rows(
            client.table("organization_teams").select("organization_id").in_("id", sorted(team_ids)).execute())
            if r.get("organization_id")}
    if org_ids:
        subs += _rows(client.table("subscriptions").select(fields).in_("organization_id", sorted(org_ids)).execute())

    current = [s for s in subs if subscription_is_current(s) and s.get("plan_id")]
    if not current:
        return PLAN_FREE

    plan_rows = _rows(client.table("plans").select("id,plan_type")
                      .in_("id", sorted({s["plan_id"] for s in current})).execute())
    types = [str(r.get("plan_type") or "").lower() for r in plan_rows]
    best = max(types, key=lambda t: _PLAN_RANK.get(t, -1), default=PLAN_FREE)
    return best if best in _PLAN_RANK else PLAN_FREE


_CACHE: Dict[str, Tuple[float, str]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_S = 300.0
_FAILURE_TTL_S = 30.0


def _billing_disabled() -> bool:
    """True on installs with no billing (self-hosted / OSS): there are no
    subscriptions to read, so nobody should be told to upgrade."""
    flag = os.getenv("AVALOKA_PLAN_LIMITS", "").strip().lower()
    if flag in ("0", "off", "false", "disabled"):
        return True
    return os.getenv("AVALOKA_EDITION", "").strip().lower() in ("oss", "community")


def _unbilled_plan() -> str:
    plan = os.getenv("AVALOKA_DEFAULT_PLAN", PLAN_ENTERPRISE).strip().lower()
    return plan if plan in _PLAN_RANK else PLAN_ENTERPRISE



def resolve_plan(user_id: Optional[str]) -> str:
    """The user's effective plan.

    - Billing disabled (AVALOKA_EDITION=oss/community or AVALOKA_PLAN_LIMITS=off):
      AVALOKA_DEFAULT_PLAN (default enterprise), no Supabase lookup.
    - No subscription row: Free.
    - Lookup failed: Free, cached only 30 s so a Supabase hiccup does not
      lock a paying user out for long.
    Successful lookups are cached 5 minutes."""
    if _billing_disabled():
        return _unbilled_plan()
    if not user_id:
        return PLAN_FREE
    now = time.monotonic()
    with _CACHE_LOCK:
        hit = _CACHE.get(user_id)
        if hit and hit[0] > now:
            return hit[1]
    try:
        plan, ttl = _lookup_plan(user_id), _CACHE_TTL_S
    except Exception as exc:
        logger.error("[plan-limits] plan lookup failed for %s (treating as free for %ss): %s",
                    user_id, int(_FAILURE_TTL_S), exc)
        plan, ttl = PLAN_FREE, _FAILURE_TTL_S
    with _CACHE_LOCK:
        _CACHE[user_id] = (now + ttl, plan)
    return plan


def clear_plan_cache(user_id: Optional[str] = None) -> None:
    """Drop cached plans, e.g. right after a checkout webhook."""
    with _CACHE_LOCK:
        if user_id:
            _CACHE.pop(user_id, None)
        else:
            _CACHE.clear()