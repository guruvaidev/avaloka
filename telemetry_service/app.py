"""The ingest endpoint.

One route: ``POST /v1/events``. It accepts a gzipped JSON Lines batch from an
Avaloka installation, validates every record against the allowlist, stores what
is new and ignores what it already has.

Deliberately unauthenticated. An anonymous sender has no credential to present,
and inventing one would defeat the purpose — a key shipped inside a downloadable
package identifies nobody and protects nothing. Abuse is handled by rate
limiting instead, which is the honest control for a public endpoint.

The sender's address is used to rate limit and is then discarded. It is never
passed to the store, and the store has no column to put it in.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from telemetry_service import __version__
from telemetry_service.ratelimit import RateLimiter
import asyncio

from telemetry_service.schema import MAX_RECORDS_PER_BATCH, Rejected, validate_batch
from telemetry_service.store import Store, open_store

logger = logging.getLogger(__name__)

#: Hard ceiling on a single request, checked before anything is decompressed.
#: The client caps batches at 5 MB uncompressed, which gzips to well under
#: this; the margin is for a client that has not been updated yet.
MAX_BODY_BYTES = int(os.environ.get("TELEMETRY_MAX_BODY_BYTES", str(8 * 1024 * 1024)))

#: A compressed payload that expands enormously is the classic decompression
#: bomb. Refusing above this bounds the memory one request can cost us.
# The client caps a batch at 5 MB uncompressed plus one install record, so
# 16 MB is already a threefold margin. The previous 64 MB ceiling let a 95 KB
# request expand into millions of parsed records before the batch cap was ever
# consulted.
MAX_DECOMPRESSED_BYTES = int(
    os.environ.get("TELEMETRY_MAX_DECOMPRESSED_BYTES", str(16 * 1024 * 1024))
)

router = APIRouter()


#: How many proxies sit in front of this service. Cloud Run and most managed
#: load balancers append exactly one entry, so the value the *infrastructure*
#: added is the last one. Set to 0 when nothing is in front.
TRUSTED_PROXY_HOPS = int(os.environ.get("TELEMETRY_TRUSTED_PROXY_HOPS", "1"))


_proxy_warning_given = False


def _warn_proxy_misconfigured() -> None:
    """Say once that TELEMETRY_TRUSTED_PROXY_HOPS does not match reality.

    Once, not per request: a misconfigured deployment receives every request
    this way, and a log line per request would bury the message it is trying
    to send.
    """
    global _proxy_warning_given
    if not _proxy_warning_given:
        _proxy_warning_given = True
        logger.warning(
            "TELEMETRY_TRUSTED_PROXY_HOPS=%d but requests arrive with no "
            "X-Forwarded-For, so nothing is proxying this service. Set it to 0, "
            "or rate limiting can be evaded by spoofing that header.",
            TRUSTED_PROXY_HOPS,
        )


def _client_address(request: Request) -> str | None:
    """The sender's address, for rate limiting only. Never stored or returned.

    ``X-Forwarded-For`` is a list the client can start and each proxy appends
    to, so the *first* entry is whatever the caller chose to put there. Reading
    that entry means an attacker rotates one header and gets an unlimited
    number of buckets — which was exactly the case here until it was tested:
    twenty requests against a limit of three, none blocked.

    So the entry counted from the *right* is used instead: that one was written
    by our own infrastructure and cannot be forged from outside. With one proxy
    in front, that is the last entry; with none, the header is ignored entirely
    and only the socket address counts.
    """
    if TRUSTED_PROXY_HOPS > 0:
        forwarded = request.headers.get("x-forwarded-for")
        if not forwarded:
            # Configured to sit behind a proxy, but nothing added the header —
            # so there is no proxy. That combination is the one case where a
            # single spoofed entry would be believed, so say so loudly. The
            # request itself is still handled, using the socket address.
            _warn_proxy_misconfigured()
        if forwarded:
            hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
            # Fewer entries than the proxies we expect means our own edge did
            # not write this header, so the whole thing is caller-supplied and
            # worth nothing. Fall through to the socket address, which cannot
            # be forged. Trusting a short header was the bypass: one spoofed
            # entry bought an attacker an unlimited number of rate-limit
            # buckets.
            if len(hops) >= TRUSTED_PROXY_HOPS:
                # Counted from the right: that entry was appended by our own
                # infrastructure. Anything to its left came from the caller.
                return hops[len(hops) - TRUSTED_PROXY_HOPS]
    return request.client.host if request.client else None


def _decompress(body: bytes, encoding: str | None) -> str:
    if (encoding or "").lower() != "gzip":
        return body.decode("utf-8")

    try:
        with gzip.GzipFile(fileobj=__import__("io").BytesIO(body)) as handle:
            data = handle.read(MAX_DECOMPRESSED_BYTES + 1)
    except Exception as exc:
        raise Rejected(f"body is not valid gzip: {exc}")

    if len(data) > MAX_DECOMPRESSED_BYTES:
        raise Rejected("decompressed body is too large")
    return data.decode("utf-8")


def _parse_lines(text: str) -> list[dict]:
    """Parse JSON Lines. One malformed line rejects the batch.

    The record cap is enforced on the raw line count BEFORE any JSON is
    parsed. Order matters: a small compressed request can decompress into
    millions of lines, and parsing each into a dict multiplies its size
    thirty-fold in the heap. Enforcing the cap after parsing — as this
    originally did — let one unauthenticated request OOM the container.
    """
    if text.count(chr(10)) > MAX_RECORDS_PER_BATCH:
        raise Rejected(f"batch exceeds {MAX_RECORDS_PER_BATCH} records")

    records = []
    for index, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise Rejected(f"line is not valid JSON: {exc.msg}", index)
    return records


@router.get("/health")
def health(request: Request) -> Response:
    """Alive AND able to store. A health check that never touches the store
    reports a green service in front of a dead database, and the orchestrator
    then has no reason to restart or reroute anything."""
    try:
        request.app.state.store.count()
    except Exception as exc:
        logger.warning("Health check cannot reach the store (%s).", exc)
        return JSONResponse({"status": "degraded", "version": __version__},
                            status_code=503)
    return JSONResponse({"status": "ok", "version": __version__})


@router.post("/v1/events")
async def ingest(request: Request) -> Response:
    """Accept one batch of telemetry from an unauthenticated sender.

    Status codes are part of the contract with the client, which treats them
    very differently:

    * ``202`` accepted — the client advances its cursor.
    * ``400`` rejected — permanent. The client steps over the batch rather
      than retrying forever, so this must only be returned when a retry
      genuinely cannot succeed.
    * ``429`` too many requests — transient. The client backs off and retries.
    * ``5xx`` — transient. Same.
    """
    now = time.time()

    limiter: RateLimiter = request.app.state.limiter
    if not limiter.allow(_client_address(request), now):
        # No Retry-After header naming a precise time: it would let a caller
        # probe the window boundary. The client's own backoff handles this.
        return JSONResponse({"error": "rate_limited"}, status_code=429)

    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse(
            {"error": "rejected", "reason": "body too large"}, status_code=413
        )

    try:
        # Off the event loop: decompressing, parsing and validating a full
        # batch is real CPU work, and a store write is real disk work. Done
        # inline — as they originally were — one slow request stalls every
        # concurrent request sharing the loop.
        encoding = request.headers.get("content-encoding")
        records = await asyncio.to_thread(
            lambda: validate_batch(_parse_lines(_decompress(body, encoding)))
        )
    except Rejected as rejected:
        # One line, truncated, newlines stripped: the reason can embed
        # attacker-chosen field names, and a log line is not a place an
        # attacker gets to write paragraphs into.
        logger.warning("Rejected a batch: %s",
                       rejected.reason[:200].replace(chr(10), " ").replace(chr(13), " "))
        return JSONResponse(rejected.as_dict(), status_code=400)
    except UnicodeDecodeError:
        return JSONResponse(
            {"error": "rejected", "reason": "body is not valid UTF-8"}, status_code=400
        )

    store: Store = request.app.state.store
    result = await asyncio.to_thread(store.store, records, now)

    # 202 rather than 200: the batch is accepted, and what happens to it
    # afterwards is not something the client waits on.
    return JSONResponse(
        {"accepted": result.stored, "duplicates": result.duplicates},
        status_code=202,
    )


def create_app(store: Store | None = None, limiter: RateLimiter | None = None) -> FastAPI:
    """Build the application.

    The store and limiter are injected so tests can supply their own and so a
    deployment can swap the database without touching this module.
    """
    app = FastAPI(
        title="Avaloka telemetry ingest",
        version=__version__,
        docs_url=None,      # nothing to explore on a public write-only endpoint
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store = store if store is not None else open_store()
    app.state.limiter = limiter or RateLimiter()
    app.include_router(router)
    return app


app = create_app
