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

Self-hosted installs (AVALOKA_EDITION=oss/community, AVALOKA_DEPLOYMENT=self_hosted,
or AVALOKA_PLAN_LIMITS=off) have no plan to enforce: a byte cap is a COST gate,
and per app/core/editions.py a COST gate never applies to a self-hosted
deployment. There the plan name is the edition this build actually is (``oss``
unless the commercial package is installed), and an upload is bounded only by
what the server can hold
-- free disk under the upload and storage roots, and memory for the profile
step -- and a refusal says which of those it hit and by how much.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.core.analysis_limits import container_memory_limit_bytes
from app.core.editions import commercial_build, deployment_state, normalize_edition

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
    self_hosted: bool = False # bounded by the server's resources, not a plan
    bound_by: str = "plan"    # plan | disk | memory | env
    bound_note: str = ""      # plain language: what the bound is and where it came from

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["max_total_mb"] = round(self.max_total_bytes / MB, 1)
        out["max_file_mb"] = round(self.max_file_bytes / MB, 1)
        return out


def limits_for(plan: Optional[str]) -> PlanLimits:
    """Limits for a plan name; anything unknown is treated as Free.

    On a self-hosted install the name is only a label (it follows the
    edition) and the limits come from the server's resources: see
    :func:`self_hosted_limits`.
    """
    plan = (plan or PLAN_FREE).strip().lower()
    if _billing_disabled():
        return self_hosted_limits(plan)
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

    A self-hosted refusal is ``resource_limit_exceeded``: it names the
    resource and the number, never a plan, and never offers an upgrade.
    """
    if reason == "file_count":
        who = "This server" if limits.self_hosted else f"The {limits.plan.title()} plan"
        message = f"Too many files: {actual}. {who} allows {limits.max_files} files per upload."
        limit_value = limits.max_files
    else:
        limit_value = limits.max_total_bytes if reason == "total_size" else limits.max_file_bytes
        scope = "in total" if reason == "total_size" else "per file"
        if limits.self_hosted:
            message = (f"This upload is {_human(actual)}. This server can take "
                       f"{_human(limit_value)} {scope}: {limits.bound_note}")
        else:
            message = (f"This upload is {_human(actual)}. The {limits.plan.title()} plan allows "
                       f"{_human(limit_value)} {scope}.")
    upgrade = limits.plan == PLAN_FREE and reason != "file_count" and not limits.self_hosted
    if upgrade:
        message += " Reduce the file size or upgrade to Professional for larger datasets."
    return {
        "code": "resource_limit_exceeded" if limits.self_hosted else "plan_limit_exceeded",
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


# user_id -> (expires_at, plan, looked_up). looked_up is False for a failed
# lookup, so callers that must not mistake "unknown" for Free can tell them apart.
_CACHE: Dict[str, Tuple[float, str, bool]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_S = 300.0
_FAILURE_TTL_S = 30.0


def _billing_disabled() -> bool:
    """True on installs with no billing (self-hosted / OSS): there are no
    subscriptions to read, so nobody should be told to upgrade.

    Uses the switches that already say "self-hosted": the edition, the
    deployment axis of app/core/editions.py, and the explicit kill switch.
    AVALOKA_PLAN_LIMITS=off always disables billing. A deployment declared
    hosted is billed whatever AVALOKA_EDITION says; one declared self_hosted
    is never billed. An undeclared deployment keeps the AVALOKA_EDITION rule
    (oss/community is unbilled), so the "unknown means self_hosted" default of
    normalize_deployment never lifts a hosted deployment's caps and the
    chart's self_hosted default changes no install's billing.
    """
    flag = os.getenv("AVALOKA_PLAN_LIMITS", "").strip().lower()
    if flag in ("0", "off", "false", "disabled"):
        return True
    state = deployment_state()
    if state == "hosted":
        return False
    if state == "self_hosted":
        return True
    return os.getenv("AVALOKA_EDITION", "").strip().lower() in ("oss", "community")


def _unbilled_plan() -> str:
    """The plan *name* of an install without billing: the edition this build
    actually is. ``oss`` unless AVALOKA_EDITION names a commercial edition AND
    the commercial package is installed -- the same rule as swarm_limits, so an
    open-source server never calls itself Enterprise. A label only; the limits
    of a self-hosted install never come from it (see self_hosted_limits)."""
    edition = normalize_edition(os.getenv("AVALOKA_EDITION"))
    if edition in (PLAN_PROFESSIONAL, PLAN_ENTERPRISE) and commercial_build():
        return edition
    return "oss"


# ---------------------------------------------------------------------------
# Self-hosted: bounded by the machine, not by a plan
# ---------------------------------------------------------------------------

#: Disk kept free under the upload roots whatever the upload size.
_DISK_RESERVE_DEFAULT = 512 * MB
#: An upload is written to the temp root and then to the local store, so it
#: needs this many times its own size in free disk.
_DISK_COPIES = 2
#: The profile step reads the file into pandas, which needs about this many
#: times the file size; the cap is that share of the container's memory.
_PROFILE_MEM_FRACTION_DEFAULT = 0.5


def _upload_roots() -> List[Path]:
    """Where an upload lands: the temp root, then the local store root.

    Same defaults as app/api/config.py and app/services/storage_service.py;
    re-derived here because this module must not import the API.
    """
    tmp = os.getenv("TMP_ROOT") or ("/var/tmp/avaloka" if os.name != "nt" else
                                    os.path.join(os.getenv("DATA_DIR", "."), "tmp"))
    store = (os.getenv("LOCAL_STORAGE_ROOT") or os.getenv("DATA_DIR")
             or str(Path.cwd() / "data" / "filestore"))
    roots: List[Path] = []
    for raw in (tmp, store):
        path = Path(raw).expanduser()
        while not path.exists() and path.parent != path:   # the nearest existing ancestor
            path = path.parent
        if path not in roots:
            roots.append(path)
    return roots


def _free_disk_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def self_hosted_limits(plan: str) -> PlanLimits:
    """What this server can hold, as upload limits.

    * AVALOKA_SELF_HOSTED_MAX_UPLOAD_BYTES, when set, is the operator's own
      number and wins.
    * Otherwise the smaller of the disk bound -- the least free space under
      the upload roots, minus AVALOKA_UPLOAD_DISK_RESERVE_BYTES, over the two
      copies an upload needs -- and the memory bound for the profile step,
      AVALOKA_UPLOAD_PROFILE_MEM_FRACTION of the container's memory.
    The note says which, in plain language, so a refusal can repeat it.
    """
    files = _env_int("AVALOKA_MAX_UPLOAD_FILES", 10)
    explicit = _env_int("AVALOKA_SELF_HOSTED_MAX_UPLOAD_BYTES", 0)
    if explicit > 0:
        return PlanLimits(plan, explicit, explicit, files, batch_jobs=True, self_hosted=True,
                          bound_by="env",
                          bound_note=f"AVALOKA_SELF_HOSTED_MAX_UPLOAD_BYTES is set to {_human(explicit)}.")

    reserve = _env_int("AVALOKA_UPLOAD_DISK_RESERVE_BYTES", _DISK_RESERVE_DEFAULT)
    roots = _upload_roots()
    free_by_root = {str(r): _free_disk_bytes(r) for r in roots}
    least_root, least_free = min(free_by_root.items(), key=lambda kv: kv[1])
    disk_bound = max(0, (least_free - reserve) // _DISK_COPIES)

    try:
        fraction = float(os.getenv("AVALOKA_UPLOAD_PROFILE_MEM_FRACTION", str(_PROFILE_MEM_FRACTION_DEFAULT)))
    except ValueError:
        fraction = _PROFILE_MEM_FRACTION_DEFAULT
    memory_total = container_memory_limit_bytes()
    memory_bound = int(memory_total * fraction)

    if disk_bound <= memory_bound:
        bound, by = disk_bound, "disk"
        note = (f"{_human(least_free)} of disk is free under {least_root}, an upload needs "
                f"{_DISK_COPIES}x its size while it is processed, and {_human(reserve)} is kept in reserve "
                f"(AVALOKA_UPLOAD_DISK_RESERVE_BYTES).")
    else:
        bound, by = memory_bound, "memory"
        note = (f"the server has {_human(memory_total)} of memory and the profile step needs about "
                f"twice the file size (AVALOKA_UPLOAD_PROFILE_MEM_FRACTION={fraction:g}).")
    return PlanLimits(plan, bound, bound, files, batch_jobs=True, self_hosted=True,
                      bound_by=by, bound_note=note)


def _cached_or_lookup(user_id: str) -> Tuple[str, bool]:
    """(plan, looked_up). A failed lookup gives (Free, False), cached only
    _FAILURE_TTL_S; a successful one is cached _CACHE_TTL_S."""
    now = time.monotonic()
    with _CACHE_LOCK:
        hit = _CACHE.get(user_id)
        if hit and hit[0] > now:
            # Entries written before looked_up existed (or seeded by tests as
            # (expires_at, plan)) count as successful lookups.
            return hit[1], (hit[2] if len(hit) > 2 else True)
    try:
        plan, ok, ttl = _lookup_plan(user_id), True, _CACHE_TTL_S
    except Exception as exc:
        logger.error("[plan-limits] plan lookup failed for %s (unknown for %ss): %s",
                     user_id, int(_FAILURE_TTL_S), exc)
        plan, ok, ttl = PLAN_FREE, False, _FAILURE_TTL_S
    with _CACHE_LOCK:
        _CACHE[user_id] = (now + ttl, plan, ok)
    return plan, ok


def resolve_plan(user_id: Optional[str]) -> str:
    """The user's effective plan, for upload limits.

    - Billing disabled (AVALOKA_EDITION=oss/community, AVALOKA_DEPLOYMENT=self_hosted
      or AVALOKA_PLAN_LIMITS=off): the edition's name (``oss`` for an
      open-source build); no Supabase lookup, and no plan limits
      (limits_for returns the server's resource bounds).
    - No subscription row: Free.
    - Lookup failed: Free, cached only 30 s so a Supabase hiccup does not
      lock a paying user out for long. Feature gates must not use this answer
      as a plan; they call try_resolve_plan, which reports the failure.
    Successful lookups are cached 5 minutes."""
    if _billing_disabled():
        return _unbilled_plan()
    if not user_id:
        return PLAN_FREE
    return _cached_or_lookup(user_id)[0]


def _unbilled_gate_plan() -> str:
    """The plan the feature gates see when there is no billing: every gated
    feature is allowed. AVALOKA_DEFAULT_PLAN (default enterprise) names it;
    this is a gate input, not a label -- resolve_plan still answers the edition
    (``oss``) so an open-source server never calls itself Enterprise."""
    plan = os.getenv("AVALOKA_DEFAULT_PLAN", PLAN_ENTERPRISE).strip().lower()
    return plan if plan in _PLAN_RANK else PLAN_ENTERPRISE


def try_resolve_plan(user_id: Optional[str]) -> Optional[str]:
    """Like resolve_plan, but None when the plan cannot be known right now
    (no user, or the lookup failed), instead of Free.

    For feature gates: an unknown plan must not be treated as Free, or a
    Supabase outage would tell paying users to upgrade. plan_gate refuses it as
    unavailable on a hosted deployment and allows it everywhere else. With
    billing off the answer is _unbilled_gate_plan, so no gate closes."""
    if _billing_disabled():
        return _unbilled_gate_plan()
    if not user_id:
        return None
    plan, looked_up = _cached_or_lookup(user_id)
    return plan if looked_up else None


def clear_plan_cache(user_id: Optional[str] = None) -> None:
    """Drop cached plans, e.g. right after a checkout webhook."""
    with _CACHE_LOCK:
        if user_id:
            _CACHE.pop(user_id, None)
        else:
            _CACHE.clear()