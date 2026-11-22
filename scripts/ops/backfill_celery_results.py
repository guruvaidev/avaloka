#!/usr/bin/env python3
"""Copy Celery results that live only in Redis into Postgres, then let Redis expire them.

Before app/core/celery_results.py, every task result was kept in Redis with no
expiry, and nowhere else. This one-shot script moves them to the durable
store, one key at a time:

    1. read a ``celery-task-meta-*`` key that has no TTL;
    2. write its summary row to public.celery_task_results: the same summary the
       backend writes (celery_results.durable_row), so no messages, no task
       arguments, no inline output CSV and no traceback text are copied;
    3. read the row back and compare it;
    4. only then ``EXPIRE`` the key (7 days by default, plus jitter so the
       cache does not empty at once). A key whose row does not verify keeps no
       TTL and is counted.

It also drains ``celery-durable-pending``: results the backend could not
write to Postgres when they finished.

Safe to stop and re-run: rows are upserted by task_id and a key gets its TTL
only after its row verifies, so an interrupted run neither duplicates nor
loses anything, and a re-run skips keys that already have a TTL.

Memory stays bounded however large a result is: values are read in chunks
(GETRANGE) and parsed as a stream, and only the summary's fields are kept;
one key is processed at a time.

Dry run by default. Output is counts only, never key names or values. The
only pattern it reads is ``celery-task-meta-*``; broker (``_kombu*``,
``unacked*``), scheduler (``redbeat:*``) and session (``sess:*``) keys are never
read or changed. It refuses to run (exit 2) without Redis or Postgres
credentials, or before the celery_task_results migration is applied.

Inside the api or celery pod (both have the credentials in their environment):

    python scripts/ops/backfill_celery_results.py                    # dry run
    python scripts/ops/backfill_celery_results.py --apply --limit 50 # canary
    python scripts/ops/backfill_celery_results.py --apply

or as the chart's one-shot Job (``ops.backfillCeleryResults``, off by default).
``CELERY_REDIS_URL`` is used when set, else ``REDIS_URL``. No URL is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core import celery_results  # noqa: E402

READY = {"SUCCESS", "FAILURE", "REVOKED"}
CHUNK_BYTES = 1 << 20          # one GETRANGE read
MIGRATION = "20261117120000_celery_task_results"


class Refused(Exception):
    """The run cannot start safely; the message says why."""


class _RangeReader:
    """A Redis string value as a file object, read CHUNK_BYTES at a time."""

    def __init__(self, client, key: str, size: int, chunk: int = CHUNK_BYTES):
        self.client, self.key, self.size, self.chunk, self.pos = client, key, size, chunk, 0

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        want = self.chunk if n is None or n < 0 else min(n, self.chunk)
        end = min(self.pos + want, self.size) - 1
        data = self.client.getrange(self.key, self.pos, end) or b""
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.pos = end + 1
        return data


_TOP = {"status", "task_id", "date_done", "name"}
_RESULT_STR = {"result.output_location": "output_location", "result.sample_status": "sample_status",
               "result.profiling_status": "profiling_status", "result.exc_type": "exc_type",
               "result.exc_module": "exc_module"}
_STATS = {"result.task_run_statistics.start": "start", "result.task_run_statistics.end": "end",
          "result.task_run_statistics.time_spent": "time_spent"}


def extract_meta(stream) -> Optional[Dict[str, Any]]:
    """The fields of a stored result the summary needs, parsed as a stream.

    Large values (messages, arguments, an inline CSV) are passed over, never
    kept. Returns None when the value is not a JSON object.
    """
    import ijson

    meta: Dict[str, Any] = {}
    result: Dict[str, Any] = {}
    er: Dict[str, Any] = {}
    stats: Dict[str, float] = {}
    samples = []
    args_index = -1
    ids: Dict[str, Any] = {}
    seen_root = False
    try:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if prefix == "" and event == "start_map":
                seen_root = True
            elif prefix in _TOP and event in ("string", "number"):
                meta[prefix] = value
            elif prefix == "result" and event == "string":
                result["_scalar"] = True
            elif prefix in _RESULT_STR and event == "string":
                result[_RESULT_STR[prefix]] = value
            elif prefix == "result.execution_result.status" and event in ("string", "number"):
                er["status"] = value
            elif prefix == "result.execution_result.error" and event not in ("null",) and value not in ("", None):
                er["error"] = True
            elif prefix == "result.error" and event not in ("null",) and value not in ("", None, False):
                result["error"] = True
            elif prefix == "result.output_file_data" and event in ("start_map", "string"):
                result["output_file_data"] = True
            elif prefix in _STATS and event == "number":
                stats[_STATS[prefix]] = value
            elif prefix == "result.available_samples.item" and event == "string":
                if len(samples) < 200:
                    samples.append(value[:128])
            elif prefix == "args.item" and event == "start_map":
                args_index += 1
            elif prefix in ("args.item.user_id", "args.item.schedule_id") and args_index == 0 and event == "string":
                ids[prefix.rsplit(".", 1)[1]] = value
    except ijson.JSONError:
        return None
    if not seen_root:
        return None
    if er:
        result["execution_result"] = er
    if stats:
        result["task_run_statistics"] = stats
    if samples:
        result["available_samples"] = samples
    meta["result"] = result
    meta["_ids"] = ids
    return meta


def _same(row: Dict[str, Any], stored: Optional[Dict[str, Any]]) -> bool:
    if not stored:
        return False
    return (stored.get("task_id") == row["task_id"]
            and stored.get("status") == row["status"]
            and json.dumps(stored.get("result"), sort_keys=True) == json.dumps(row["result"], sort_keys=True))


def preflight(redis_client) -> None:
    """Refuse a run that cannot be safe: no Redis, no Postgres credentials, no table."""
    try:
        redis_client.ping()
    except Exception as exc:  # noqa: BLE001
        raise Refused(f"Redis is not reachable ({type(exc).__name__})") from None
    try:
        client = celery_results._client()
    except RuntimeError:
        raise Refused("Postgres credentials are missing: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY") from None
    try:
        client.table(celery_results.TABLE).select("task_id").limit(1).execute()
    except Exception as exc:  # noqa: BLE001
        if _missing_table(exc):
            raise Refused(
                f"the {celery_results.TABLE} table does not exist. "
                f"Apply migration {MIGRATION} first; nothing was changed."
            ) from None
        raise Refused(f"Postgres is not reachable ({type(exc).__name__}); nothing was changed.") from None


def _missing_table(exc: BaseException) -> bool:
    """PostgREST's answer for a table that is not there (or not in its schema cache)."""
    code = str(getattr(exc, "code", "") or "")
    text = f"{code} {exc}"
    return code in ("42P01", "PGRST205", "PGRST204") or "does not exist" in text or "PGRST205" in text \
        or "Could not find the table" in text


def _decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def run(redis_client, *, apply: bool, expire_seconds: int, jitter_seconds: int,
        limit: Optional[int] = None, batch: int = 200, rng: Optional[random.Random] = None) -> Counter:
    rng = rng or random.Random()
    counts: Counter = Counter()
    pending = {_decode(m) for m in redis_client.smembers(celery_results.PENDING_SET)}
    counts["pending_queued"] = len(pending)
    done_key = "backfilled" if apply else "would_backfill"
    keys = []

    def flush():
        if not keys:
            return
        with redis_client.pipeline() as pipe:
            for k in keys:
                pipe.ttl(k)
                pipe.strlen(k)
            info = pipe.execute()
        for i, key in enumerate(keys):
            _process(key, info[2 * i], info[2 * i + 1])
        counts["batches"] += 1
        keys.clear()

    def _process(key, ttl, size):
        if limit is not None and counts[done_key] >= limit:
            counts["left_for_next_run"] += 1
            return
        if ttl == -2:
            counts["gone"] += 1
            return
        if ttl >= 0:
            counts["already_ttl"] += 1
            return
        counts["no_ttl"] += 1
        meta = extract_meta(_RangeReader(redis_client, key, int(size or 0)))
        if meta is None:
            counts["undecodable"] += 1
            return
        if meta.get("status") not in READY:
            counts["not_finished"] += 1
            return
        task_id = str(meta.get("task_id") or key[len(celery_results.META_PREFIX):])
        row = celery_results.durable_row(task_id, meta, args=[meta.pop("_ids")], name=meta.get("name"))
        if not apply:
            counts["would_backfill"] += 1
            return
        try:
            celery_results.upsert_row(row)
            stored = celery_results.read_row(task_id)
        except Exception as exc:  # noqa: BLE001 - counted; the key keeps no TTL
            counts[f"write_failed:{type(exc).__name__}"] += 1
            return
        if not _same(row, stored):
            counts["verify_failed"] += 1
            return
        counts["backfilled"] += 1
        ttl_s = expire_seconds + (rng.randint(0, jitter_seconds) if jitter_seconds > 0 else 0)
        if redis_client.expire(key, ttl_s, nx=True):
            counts["expire_set"] += 1
        if task_id in pending:
            redis_client.srem(celery_results.PENDING_SET, task_id)
            counts["pending_drained"] += 1

    for key in celery_results.iter_meta_keys(redis_client, count=batch):
        counts["scanned"] += 1
        keys.append(key)
        if len(keys) >= batch:
            flush()
    flush()
    return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="apply", action="store_false",
                      help="count what would be done, change nothing (the default)")
    mode.add_argument("--apply", "--no-dry-run", dest="apply", action="store_true",
                      help="write rows and set TTLs")
    parser.set_defaults(apply=False)
    parser.add_argument("--batch", type=int, default=200, help="keys per SCAN batch (default 200)")
    parser.add_argument("--expire-seconds", type=int, default=celery_results.RESULT_EXPIRES_DEFAULT,
                        help="TTL given to a key once its row is verified (default 7 days)")
    parser.add_argument("--jitter-seconds", type=int, default=3600,
                        help="random extra TTL per key, so they do not all expire at once")
    parser.add_argument("--limit", type=int, default=None, help="back up at most N keys (canary run)")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.expire_seconds < 1:
        parser.error("--expire-seconds must be at least 1")
    if not 1 <= args.batch <= 10_000:
        parser.error("--batch must be between 1 and 10000")

    url = os.getenv("CELERY_REDIS_URL") or os.getenv("REDIS_URL")
    if not url:
        print("refused: set CELERY_REDIS_URL or REDIS_URL", file=sys.stderr)
        return 2
    import redis

    client = redis.Redis.from_url(url, socket_timeout=30, socket_connect_timeout=10)
    try:
        preflight(client)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    counts = run(client, apply=args.apply, expire_seconds=args.expire_seconds,
                 jitter_seconds=args.jitter_seconds, limit=args.limit, batch=args.batch)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", **dict(sorted(counts.items()))}, indent=1))
    failed = counts["verify_failed"] + sum(v for k, v in counts.items() if k.startswith("write_failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
