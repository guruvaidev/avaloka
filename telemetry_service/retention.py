"""The retention job.

A retention policy that is written in a design document but never runs is not a
policy. This is the thing that actually deletes.

Run it on a schedule — a nightly cron job, or a Cloud Scheduler trigger. It is
safe to run repeatedly: deleting from an empty window removes nothing.

    python -m telemetry_service.retention --days 400
"""

from __future__ import annotations

import argparse
import logging
import time

from telemetry_service.store import RETENTION_DAYS, Store, open_store

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400


def purge(store: Store, days: int | None = None, now: float | None = None) -> int:
    """Delete everything received more than ``days`` ago. Returns the count."""
    days = RETENTION_DAYS if days is None else days
    now = time.time() if now is None else now
    cutoff = now - (days * SECONDS_PER_DAY)
    removed = store.purge_older_than(cutoff)
    logger.info("Retention: removed %d events older than %d days.", removed, days)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete telemetry past its retention period.")
    parser.add_argument("--days", type=int, default=RETENTION_DAYS,
                        help=f"retention period in days (default {RETENTION_DAYS})")
    parser.add_argument("--database", default=None, help="database path or URL")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be deleted without deleting it")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    store = open_store(args.database)

    if args.dry_run:
        # Counting rather than deleting, so the number can be checked before
        # anybody points this at a production database.
        cutoff = time.time() - (args.days * SECONDS_PER_DAY)
        held = store.count()
        stale = store.count_older_than(cutoff)
        print(f"{held} events held, {stale} older than {args.days} days.")
        print("Dry run: nothing deleted.")
        return 0

    removed = purge(store, days=args.days)
    print(f"Removed {removed} events older than {args.days} days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
