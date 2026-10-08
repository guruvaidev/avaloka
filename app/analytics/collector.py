"""The product-analytics collector: three routes on the existing API.

``GET  /analytics/config``   what the UI may send, and whether to send at all
``POST /analytics/events``   a batch of events
``PUT  /analytics/opt-out``  the analyst's own switch

The collector never raises into the product. A disabled deployment, a deployment
with no database configured for analytics, an opted-out analyst and a browser
sending Global Privacy Control all get a normal 200 with ``enabled: false`` and
nothing is stored -- the UI is told to stop rather than left retrying.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.analytics import config, export as export_mod, names as names_mod, redact, schema
from app.analytics.store import Store, default_store, to_row

logger = logging.getLogger(__name__)

UserResolver = Callable[[Request], Optional[str]]
#: (request, user_id) -> the column, table and dataset names registered for that
#: analyst, or None when they cannot be determined completely. May be async.
NamesProvider = Callable[[Request, str], Any]

#: Events one analyst may send per minute. A runaway render loop in the UI is
#: the realistic cause; the API shares one node with user workloads.
RATE_LIMIT_PER_MINUTE = 600
_PURGE_INTERVAL_S = 3600


def derive_actor_id(secret: bytes, user_id: str) -> str:
    """A keyed pseudonym for the analyst.

    Stable, so retention and "what do they retry" can be computed; keyed with a
    secret that lives only in this deployment's database, so the value cannot be
    reversed to a user id, or matched across deployments, by anyone who holds
    the events but not the key. It is a pseudonym, not anonymity: whoever holds
    both the key and the user table can recompute it.
    """
    return hmac.new(secret, b"actor:" + user_id.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def browser_declines(request: Request) -> bool:
    """Global Privacy Control or Do Not Track, sent by the analyst's browser."""
    return request.headers.get("Sec-GPC", "").strip() == "1" or request.headers.get("DNT", "").strip() == "1"


class _RateLimiter:
    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute = per_minute
        self._clock = clock
        self._lock = threading.Lock()
        self._windows: Dict[str, List[float]] = {}

    def allow(self, key: str, n: int) -> bool:
        minute = int(self._clock() // 60)
        with self._lock:
            if len(self._windows) > 10_000:
                self._windows.clear()
            window = self._windows.get(key)
            if window is None or window[0] != minute:
                window = self._windows[key] = [minute, 0]
            if window[1] + n > self.per_minute:
                return False
            window[1] += n
            return True


def process_event(raw: Any, *, keep_prompt: bool,
                  vocabulary: Optional[names_mod.Vocabulary] = None,
                  prepare_export: bool = False) -> Dict[str, Any]:
    """Validate one event and apply the capture-time controls. Raises ``schema.Rejected``.

    Produces both representations of the question at once: ``prompt`` (stored,
    sensitive, local only) and, when export is on and the vocabulary is known,
    ``prompt_export`` (names removed). With no vocabulary there is no
    ``prompt_export``, and so nothing for the export path to send.
    """
    event = schema.validate_event(raw)
    if "prompt" in event:
        original = event.pop("prompt")
        # The length bucket is kept even when the text is not: it is a number.
        event["prompt_chars_bucket"] = schema.prompt_chars_bucket(len(original))
        if keep_prompt:
            cleaned, counts = redact.scrub(original)
            if cleaned is not None:
                event["prompt"] = cleaned
                event["prompt_redactions"] = sum(counts.values())
                if prepare_export:
                    stripped = names_mod.strip_names(cleaned, vocabulary)
                    if stripped is not None:
                        event["prompt_export"], hits = stripped
                        for kind in names_mod.HIT_KINDS:
                            event[f"prompt_names_{kind}"] = hits[kind]
    return event


def build_router(resolve_user_id: UserResolver, store: Optional[Store] = None,
                 rate_limit_per_minute: int = RATE_LIMIT_PER_MINUTE,
                 names_provider: Optional[NamesProvider] = None,
                 clock: Callable[[], float] = time.monotonic) -> APIRouter:
    router = APIRouter(prefix="/analytics", tags=["analytics"])
    limiter = _RateLimiter(rate_limit_per_minute, clock=clock)
    state = {"last_purge": 0.0}

    def _store() -> Optional[Store]:
        """``None`` when no database is configured: analytics is then off."""
        return store or default_store()

    def _actor(request: Request, db: Store) -> Optional[str]:
        user_id = resolve_user_id(request)
        if not user_id:
            return None
        return derive_actor_id(db.identity_secret(), str(user_id))

    async def _vocabulary(request: Request) -> Optional[names_mod.Vocabulary]:
        """The analyst's registered names, or None. Any doubt is None."""
        if names_provider is None:
            return None
        try:
            user_id = resolve_user_id(request)
            found = names_provider(request, str(user_id))
            if inspect.isawaitable(found):
                found = await found
            if found is None:
                return None
            return names_mod.Vocabulary(found) or None
        except Exception as exc:
            logger.warning("[analytics] name vocabulary unavailable: %s", type(exc).__name__)
            return None

    def _maybe_purge(db: Store) -> None:
        now = time.monotonic()
        if now - state["last_purge"] < _PURGE_INTERVAL_S and state["last_purge"]:
            return
        state["last_purge"] = now
        try:
            db.purge_older_than(config.retention_days())
        except Exception as exc:  # retention must never fail an ingest
            logger.warning("[analytics] retention purge failed: %s", type(exc).__name__)

    def _stopped(reason: str) -> JSONResponse:
        return JSONResponse({"enabled": False, "reason": reason, "accepted": 0})

    @router.get("/config")
    def get_config(request: Request) -> Dict[str, Any]:
        base = {"schema_version": schema.SCHEMA_VERSION, "prompt_text": config.prompt_text_enabled()}
        if not config.capture_enabled():
            return {**base, "enabled": False, "reason": "disabled", "opted_out": False}
        if browser_declines(request):
            return {**base, "enabled": False, "reason": "browser_signal", "opted_out": False}
        db = _store()
        if db is None:
            return {**base, "enabled": False, "reason": "no_store", "opted_out": False}
        actor = _actor(request, db)
        opted_out = db.is_opted_out(actor)
        return {**base, "enabled": bool(actor) and not opted_out,
                "reason": "opted_out" if opted_out else ("ok" if actor else "unauthenticated"),
                "opted_out": opted_out}

    @router.put("/opt-out")
    async def put_opt_out(request: Request) -> JSONResponse:
        try:
            body = json.loads(await request.body() or b"{}")
            opted_out = body["opted_out"]
            if not isinstance(opted_out, bool):
                raise ValueError
        except Exception:
            return JSONResponse({"error": "rejected", "reason": "opted_out: boolean required"}, 400)
        # Deliberately works when capture is disabled: the choice is the
        # analyst's and should survive an operator turning capture back on.
        db = _store()
        if db is None:
            # Nowhere to record the choice, and nothing is being captured. Say
            # so rather than report an opt-out that was not kept.
            return JSONResponse({"error": "unavailable", "reason": "no_store"}, 503)
        actor = await run_in_threadpool(_actor, request, db)
        if not actor:
            return JSONResponse({"error": "unauthenticated"}, 401)
        deleted = await run_in_threadpool(db.set_opt_out, actor, opted_out)
        return JSONResponse({"opted_out": opted_out, "deleted_events": deleted})

    @router.post("/events")
    async def post_events(request: Request) -> JSONResponse:
        if not config.capture_enabled():
            return _stopped("disabled")
        if browser_declines(request):
            return _stopped("browser_signal")
        db = _store()
        if db is None:
            return _stopped("no_store")

        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > schema.MAX_BATCH_BYTES:
            return JSONResponse({"error": "rejected", "reason": "batch too large"}, 413)
        body = await request.body()
        if len(body) > schema.MAX_BATCH_BYTES:
            return JSONResponse({"error": "rejected", "reason": "batch too large"}, 413)

        actor = await run_in_threadpool(_actor, request, db)
        if not actor:
            return JSONResponse({"error": "unauthenticated"}, 401)
        if await run_in_threadpool(db.is_opted_out, actor):
            return _stopped("opted_out")

        try:
            batch = json.loads(body)
            version, session_id, ui_version, raw_events = schema.validate_envelope(batch)
        except schema.Rejected as exc:
            return JSONResponse({"error": "rejected", "reason": exc.reason}, 400)
        except ValueError:
            return JSONResponse({"error": "rejected", "reason": "batch: not JSON"}, 400)

        if not limiter.allow(actor, len(raw_events)):
            return JSONResponse({"error": "rate_limited"}, 429, headers={"Retry-After": "60"})

        keep_prompt = config.prompt_text_enabled()
        # The membership test and its vocabulary lookup run only when question
        # text could be exported. export.EXPORT_QUESTION_TEXT is False, so in
        # production this is always False and no lookup is ever made.
        prepare_export = export_mod.EXPORT_QUESTION_TEXT and keep_prompt and config.export_requested() and any(
            isinstance(e, dict) and "prompt" in e for e in raw_events)
        vocabulary = await _vocabulary(request) if prepare_export else None
        rows: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for index, raw in enumerate(raw_events):
            try:
                event = process_event(raw, keep_prompt=keep_prompt, vocabulary=vocabulary,
                                      prepare_export=prepare_export)
            except schema.Rejected as exc:
                rejected.append({"index": index, "reason": exc.reason})
                continue
            rows.append(to_row(event, schema_version=version, session_id=session_id,
                               ui_version=ui_version, actor_id=actor))

        try:
            stored, duplicates = await run_in_threadpool(db.insert_events, rows)
        except Exception as exc:
            # Class name only: a driver message can quote the row it choked on.
            logger.warning("[analytics] store unavailable: %s", type(exc).__name__)
            return JSONResponse({"error": "unavailable"}, 503, headers={"Retry-After": "60"})
        await run_in_threadpool(_maybe_purge, db)
        return JSONResponse(
            {"enabled": True, "accepted": stored, "duplicates": duplicates, "rejected": rejected},
            status_code=202,
        )

    return router
