"""Turning stored events into answers.

Phase 1 recorded, phase 2 delivered, phase 3 stored. None of that is worth
anything until somebody can ask a question and get a number back.

The questions here are the ones the design set out to answer: is it working for
the people who installed it, which capabilities get used, which fail, and on
what shape of hardware. Each report is a plain function returning plain data,
so it can be printed at a terminal, served as JSON, or charted later without
any of that being decided now.

Two rules hold throughout:

* **Count installations, not events.** "Ten thousand runs" means nothing until
  you know whether that was ten users or ten thousand. Almost every number here
  is reported both ways.
* **Never invent precision.** A failure rate over four installations is noise.
  Reports that could mislead carry the sample size alongside the number.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from telemetry_service.store import Store, open_store

SECONDS_PER_DAY = 86_400

#: Below this many installations, a rate is not reported as a percentage —
#: three failures out of four installs is not a "75% failure rate", it is four
#: data points.
MIN_INSTALLS_FOR_A_RATE = 5


@dataclass
class Report:
    """A named result with enough context to be read honestly."""

    title: str
    rows: list[dict] = field(default_factory=list)
    caveat: str | None = None

    def as_dict(self) -> dict:
        payload = {"title": self.title, "rows": self.rows}
        if self.caveat:
            payload["caveat"] = self.caveat
        return payload


def _rate(part: int, whole: int) -> float | None:
    return round(100.0 * part / whole, 1) if whole else None


#: What these reports are about: an installation and the agent runs on it.
#: The store also holds product-analytics events (the ``analytics.*`` types),
#: which describe something else -- an analyst's session in the web UI -- and
#: carry an install id minted for that export, not the installation's own.
#: Counting them here would report one deployment as two installations and a
#: page view as a run. No report in this module reads them; reports shaped for
#: sessions and questions do not exist yet.
PROCESS_EVENT_TYPES = frozenset({"install", "agent.started", "agent.completed", "dataset"})


def _process_records(store: Store) -> list[dict]:
    """Every stored process-telemetry record, and nothing from the other producer."""
    return [record for record in store.records(limit=10 ** 9)
            if record.get("event_type") in PROCESS_EVENT_TYPES]


# --------------------------------------------------------------------- reports

def adoption(store: Store) -> Report:
    """How many installations, and how much each of them actually used it.

    The distinction that matters most: a thousand events from one installation
    means one enthusiast, and from a thousand installations means a product
    people keep coming back to. Those call for opposite decisions.
    """
    per_install: dict[str, int] = {}
    for record in _process_records(store):
        install = record.get("install_id")
        if install:
            per_install[install] = per_install.get(install, 0) + 1

    if not per_install:
        return Report("Adoption", [], "No events received yet.")

    counts = sorted(per_install.values(), reverse=True)
    total = sum(counts)
    return Report("Adoption", [{
        "installations": len(per_install),
        "events": total,
        "events_per_install_median": counts[len(counts) // 2],
        "events_per_install_max": counts[0],
        "single_event_installs": sum(1 for c in counts if c == 1),
    }])


def capability_usage(store: Store) -> Report:
    """Which parts of Avaloka actually get used, and by how many installations.

    A capability used heavily by one installation and by nobody else is not a
    popular capability, so both numbers are reported.
    """
    runs: dict[str, int] = {}
    installs: dict[str, set] = {}

    for record in _process_records(store):
        if record.get("event_type") != "agent.completed":
            continue
        agent = record.get("agent")
        if not agent:
            continue
        runs[agent] = runs.get(agent, 0) + 1
        installs.setdefault(agent, set()).add(record.get("install_id"))

    rows = [
        {"agent": agent, "runs": count, "installations": len(installs[agent])}
        for agent, count in sorted(runs.items(), key=lambda kv: -kv[1])
    ]
    return Report("Capability usage", rows)


def reliability(store: Store) -> Report:
    """Which parts fail, how often, and with what kind of error.

    This is the report the whole feature exists for. A system that only
    recorded successes would be cheerful and useless.
    """
    stats: dict[str, dict] = {}

    for record in _process_records(store):
        if record.get("event_type") != "agent.completed":
            continue
        agent = record.get("agent")
        if not agent:
            continue
        entry = stats.setdefault(agent, {"runs": 0, "failures": 0, "errors": {}, "installs": set()})
        entry["runs"] += 1
        entry["installs"].add(record.get("install_id"))
        if record.get("outcome") in {"failed", "timeout"}:
            entry["failures"] += 1
            error = record.get("error_class") or "unknown"
            entry["errors"][error] = entry["errors"].get(error, 0) + 1

    rows = []
    for agent, entry in sorted(stats.items(), key=lambda kv: -kv[1]["failures"]):
        row = {
            "agent": agent,
            "runs": entry["runs"],
            "failures": entry["failures"],
            "installations": len(entry["installs"]),
            "top_error": max(entry["errors"], key=entry["errors"].get) if entry["errors"] else None,
        }
        # A percentage over a handful of installations invites a wrong decision.
        if len(entry["installs"]) >= MIN_INSTALLS_FOR_A_RATE:
            row["failure_rate_pct"] = _rate(entry["failures"], entry["runs"])
        rows.append(row)

    caveat = (
        f"Failure rates are omitted where fewer than {MIN_INSTALLS_FOR_A_RATE} "
        "installations contributed, because a rate over a handful of installs is noise."
    )
    return Report("Reliability", rows, caveat)


def performance(store: Store) -> Report:
    """How long each part takes. Reported as percentiles, not averages.

    An average duration hides the case that matters: one user in twenty waiting
    four minutes is a problem an average will never show.
    """
    durations: dict[str, list[int]] = {}
    for record in _process_records(store):
        value = record.get("duration_ms")
        agent = record.get("agent")
        if agent and isinstance(value, int):
            durations.setdefault(agent, []).append(value)

    def percentile(values: list[int], fraction: float) -> int:
        ordered = sorted(values)
        index = min(int(fraction * len(ordered)), len(ordered) - 1)
        return ordered[index]

    rows = []
    for agent, values in sorted(durations.items(), key=lambda kv: -percentile(kv[1], 0.5)):
        rows.append({
            "agent": agent,
            "samples": len(values),
            "p50_ms": percentile(values, 0.50),
            "p95_ms": percentile(values, 0.95),
            "max_ms": max(values),
        })
    return Report("Performance", rows)


def environments(store: Store) -> Report:
    """What sort of machines this runs on. Shapes what we optimise for."""
    buckets: dict[str, dict[str, int]] = {}
    seen_installs = set()

    for record in _process_records(store):
        if record.get("event_type") != "install":
            continue
        install = record.get("install_id")
        if install in seen_installs:
            continue          # one row per installation, not per upload
        seen_installs.add(install)
        for field_name in ("os", "arch", "deployment", "cloud", "edition",
                           "python_version", "avaloka_version", "cpu_count"):
            value = record.get(field_name)
            if value is not None:
                counts = buckets.setdefault(field_name, {})
                counts[str(value)] = counts.get(str(value), 0) + 1

    rows = [
        {"dimension": name, "distribution": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}
        for name, counts in buckets.items()
    ]
    return Report("Environments", rows)


def data_shapes(store: Store) -> Report:
    """What size of data people actually bring. Bucketed, never exact."""
    sizes: dict[str, int] = {}
    formats: dict[str, int] = {}
    sampled = {"true": 0, "false": 0}

    for record in _process_records(store):
        if record.get("event_type") != "dataset":
            continue
        bucket = record.get("row_count_bucket")
        if bucket:
            sizes[bucket] = sizes.get(bucket, 0) + 1
        fmt = record.get("file_format")
        if fmt:
            formats[fmt] = formats.get(fmt, 0) + 1
        if "sampled" in record:
            sampled["true" if record["sampled"] else "false"] += 1

    return Report("Data shapes", [{
        "row_count_buckets": dict(sorted(sizes.items())),
        "formats": dict(sorted(formats.items(), key=lambda kv: -kv[1])),
        "sampled": sampled,
    }])


def freshness(store: Store, now: float | None = None) -> Report:
    """How current the picture is.

    A dashboard that silently stops updating is worse than no dashboard,
    because it looks authoritative while being stale.

    The one report that counts everything held, product-analytics events
    included: it describes the store, not the installations.
    """
    now = time.time() if now is None else now
    total = store.count()
    if not total:
        return Report("Freshness", [], "No events received yet.")

    stale_cutoff = now - (7 * SECONDS_PER_DAY)
    older = store.count_older_than(stale_cutoff)
    return Report("Freshness", [{
        "events_held": total,
        "received_in_last_7_days": total - older,
        "older_than_7_days": older,
    }])


#: How long to wait before calling a start with no completion a crash.
#: A run writes both records milliseconds apart, but the client uploads on a
#: six-hour schedule, so a batch boundary or a file rotation can put them in
#: different uploads. Anything more recent than this is treated as still in
#: flight rather than killed -- reporting a healthy run as a crash is the same
#: class of mistake as missing a real one.
UNFINISHED_GRACE_SECONDS = SECONDS_PER_DAY          # 24 hours


def _by_session(store: Store, older_than: float | None = None) -> dict:
    """Group start and stop records by run, oldest position first.

    Every field is read with ``.get``. The server's own validator requires only
    ``event_id`` and ``event_type``, so a record without ``agent`` or ``seq`` is
    accepted by the public endpoint -- and indexing it would raise a KeyError
    that takes down not just this report but ``run_all`` with it.
    """
    grouped: dict[str, list[dict]] = {}
    for record in store.session_records(older_than=older_than):
        if not isinstance(record, dict):
            continue
        if record.get("event_type") not in {"agent.started", "agent.completed"}:
            continue
        session = record.get("session_id")
        if not isinstance(session, str) or not session:
            continue
        grouped.setdefault(session, []).append(record)

    for records in grouped.values():
        # By `seq`, never by arrival: batches upload out of order and a
        # redelivery can land months later.
        records.sort(key=lambda r: r.get("seq") if isinstance(r.get("seq"), int) else 0)
    return grouped


def _unfinished(records: list[dict], cutoff: float | None) -> list[str]:
    """Agents that started and never came back.

    ``cutoff`` excludes starts too recent for their completion to have been
    uploaded yet. Without it a run still in progress is reported as a crash,
    which is exactly the false signal this report exists to avoid.
    """
    finished = {r.get("started_seq") for r in records
                if r.get("event_type") == "agent.completed"}
    out = []
    for record in records:
        if record.get("event_type") != "agent.started":
            continue
        if record.get("seq") in finished:
            continue
        received = record.get("received_at")
        if cutoff is not None and isinstance(received, (int, float)) and received > cutoff:
            continue          # too recent to call; its completion may be in flight
        out.append(record.get("agent") or "unknown")
    return out


def _top_level_ms(records: list[dict]) -> int:
    """Total time for a run, counting each agent's work once.

    Agents nest — the coding sub-graph runs inside ``code_etl`` — so its
    duration is already contained in its parent's. Summing every completion
    reports the run as slower than it actually was.

    Walking the records in ``seq`` order, a start opens a frame and a completion
    closes one. A start seen at depth zero is a top-level agent, and only those
    are counted.
    """
    depth = 0
    top_level: set = set()
    for record in records:
        kind = record.get("event_type")
        if kind == "agent.started":
            if depth == 0:
                top_level.add(record.get("seq"))
            depth += 1
        elif kind == "agent.completed":
            depth = max(0, depth - 1)

    return sum(
        record.get("duration_ms") or 0
        for record in records
        if record.get("event_type") == "agent.completed"
        and record.get("started_seq") in top_level
    )


def sessions(store: Store, limit: int = 20, now: float | None = None) -> Report:
    """Rebuild individual runs from their start and stop records.

    The aggregate reports above answer "how often does the validator fail".
    This one answers "what actually happened in that run" -- the order agents
    ran in, which nested inside which, and how long each took. It is the report
    you reach for when a number in one of the others looks wrong.
    """
    now = time.time() if now is None else now
    cutoff = now - UNFINISHED_GRACE_SECONDS

    grouped = _by_session(store)
    if not grouped:
        return Report("Sessions", [], "No start or stop records received yet.")

    rows = []
    for session, records in grouped.items():
        starts = [r for r in records if r.get("event_type") == "agent.started"]
        completions = [r for r in records if r.get("event_type") == "agent.completed"]
        failed = [r.get("agent") or "unknown" for r in completions
                  if r.get("outcome") != "success"]

        rows.append({
            "session": session[:12],
            "agents_started": len(starts),
            "agents_completed": len(completions),
            "failed": failed,
            # A start with nothing pointing back at it, old enough that its
            # completion should have arrived. The agent did not fail -- it never
            # returned, which is what a killed process looks like from here.
            "never_finished": _unfinished(records, cutoff),
            # Outermost agents only. A nested agent's time is already inside its
            # parent's duration, so summing every completion counts the coding
            # sub-graph twice and reports a run as slower than it was.
            "total_ms": _top_level_ms(records),
            "order": [r.get("agent") or "unknown" for r in starts],
        })

    rows.sort(key=lambda r: (len(r["never_finished"]), len(r["failed"])), reverse=True)
    truncated = len(rows) > limit
    report = Report("Sessions", rows[:limit])
    if truncated:
        report.caveat = (f"Showing {limit} of {len(rows)} sessions, worst first. "
                         f"Sessions with unfinished agents are listed before failures.")
    return report


def interrupted(store: Store, now: float | None = None) -> Report:
    """Agents that started and never came back, counted per agent.

    A failure raises and is recorded. A process that is killed does neither, so
    before start records existed this was invisible: the installation simply
    went quiet and looked identical to a customer who had stopped using us.

    Kept separate from the reliability report on purpose. A ``KeyError`` is a
    bug in our code; a kill is the workload exceeding the machine it was given.
    They have different fixes, and one number covering both hides each of them.
    """
    now = time.time() if now is None else now
    cutoff = now - UNFINISHED_GRACE_SECONDS

    grouped = _by_session(store)
    started: dict[str, int] = {}
    stopped: dict[str, int] = {}

    for records in grouped.values():
        for record in records:
            if record.get("event_type") != "agent.started":
                continue
            received = record.get("received_at")
            if isinstance(received, (int, float)) and received > cutoff:
                continue          # still in flight; not yet countable either way
            agent = record.get("agent") or "unknown"
            started[agent] = started.get(agent, 0) + 1
        for agent in _unfinished(records, cutoff):
            stopped[agent] = stopped.get(agent, 0) + 1

    if not started:
        return Report("Interrupted agents", [],
                      "No start records old enough to judge yet.")

    rows = []
    for agent, count in started.items():
        kills = stopped.get(agent, 0)
        row = {"agent": agent, "started": count, "never_finished": kills}
        rate = _rate(kills, count)
        if rate is not None and count >= MIN_INSTALLS_FOR_A_RATE:
            row["interrupted_pct"] = rate
        rows.append(row)

    rows.sort(key=lambda r: r["never_finished"], reverse=True)
    caveat = (f"A start is only judged once {UNFINISHED_GRACE_SECONDS // 3600}h have "
              f"passed, so its completion has had time to upload. Rates are shown "
              f"only above {MIN_INSTALLS_FOR_A_RATE} starts.")
    return Report("Interrupted agents", rows, caveat)


ALL_REPORTS = {
    "adoption": adoption,
    "capabilities": capability_usage,
    "reliability": reliability,
    "interrupted": interrupted,
    "performance": performance,
    "environments": environments,
    "data-shapes": data_shapes,
    "sessions": sessions,
    "freshness": freshness,
}


def run_all(store: Store) -> dict:
    return {name: fn(store).as_dict() for name, fn in ALL_REPORTS.items()}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Report on collected telemetry.")
    parser.add_argument("report", nargs="?", default="all",
                        choices=["all", *ALL_REPORTS], help="which report to run")
    parser.add_argument("--database", default=None, help="database path or URL")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    store = open_store(args.database)
    result = run_all(store) if args.report == "all" else {
        args.report: ALL_REPORTS[args.report](store).as_dict()
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    for payload in result.values():
        print(f"\n{payload['title']}")
        print("-" * len(payload["title"]))
        if not payload["rows"]:
            print("  (nothing to report)")
        for row in payload["rows"]:
            for key, value in row.items():
                print(f"  {key:<28} {value}")
            print()
        if payload.get("caveat"):
            print(f"  note: {payload['caveat']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
