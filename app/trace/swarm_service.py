"""The server's view of a swarm mission: start one, serve its trace.

Kept out of ``app/api/server.py`` so the routes there stay a few lines each
and this is testable without the HTTP stack. The coordinator itself is
:class:`avaloka.coordinator.MainAvaloka`; this module only resolves what the
API lets a caller ask for, runs it on a daemon thread with a
:class:`~avaloka.swarm_trace.SwarmTraceRecorder`, and reads the trace back
through :mod:`app.trace.swarm_store`.

The run is a plain thread rather than an asyncio task on purpose: the server's
event loop must stay free while two clones train for a minute, and the
recorder is thread-safe, so the request that polls ``records?after=`` reads
the same list the clones append to.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.trace import swarm_store as store
from app.trace.swarm_store import Entry
from avaloka.swarm_trace import SwarmTraceRecorder, clone_document, episodes, is_finished
from avaloka.understanding_trace import RECORD_TYPES as UNDERSTANDING_TYPES

logger = logging.getLogger(__name__)

KINDS = ("analyze", "train")
MAX_RECORDS_PAGE = 5000
DEFAULT_RECORDS_PAGE = 500

#: The bundled samples a caller may name. Keys are what the API accepts;
#: values resolve lazily so a build without ``app/sample_data`` still imports.
SAMPLE_FILES: Dict[str, str] = {
    "churn": "avaloka/examples/churn.csv",
    "salaries": "app/sample_data/salaries.csv",
    "healthcare": "app/sample_data/healthcare_dataset.csv",
    "youtube": "app/sample_data/Global_YouTube_Statistics.csv",
    "football": "app/sample_data/2021-2022 Football Team Stats.csv",
    "taxi": "app/sample_data/yellow_tripdata_2015-01_dataset_50k.csv",
}

ThreadResolver = Callable[[str], Awaitable[Optional[str]]]


class SwarmRequestError(ValueError):
    """A request the API refuses; the route answers 400 with the message."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sample_path(name: str) -> Path:
    rel = SAMPLE_FILES.get((name or "").strip().lower())
    if not rel:
        raise SwarmRequestError(
            f"unknown sample {name!r}; one of: " + ", ".join(sorted(SAMPLE_FILES)))
    path = _repo_root() / rel
    if not path.is_file():
        raise SwarmRequestError(f"sample {name!r} is not bundled with this build")
    return path


def paths_allowed() -> bool:
    return (os.getenv("AVALOKA_SWARM_ALLOW_PATHS") or "").strip() == "1"


async def resolve_source(source: Any, *, thread_resolver: ThreadResolver) -> tuple[str, Dict[str, Any]]:
    """The local path a mission will read, and how the request named it.

    ``{"sample": name}`` is a bundled file; ``{"thread_id": id}`` is that
    thread's current dataset (the resolver enforces ownership and returns
    ``None`` for a thread the caller may not read); ``{"path": p}`` is refused
    unless ``AVALOKA_SWARM_ALLOW_PATHS=1``, because an API that reads any file
    the server can see is a different product from one that reads uploads.
    """
    if not isinstance(source, dict) or len(source) != 1:
        raise SwarmRequestError('source must be one of {"sample": ...}, {"thread_id": ...}, {"path": ...}')
    (key, value), = source.items()
    if not isinstance(value, str) or not value.strip():
        raise SwarmRequestError(f"source.{key} must be a non-empty string")
    if key == "sample":
        return str(sample_path(value)), {"kind": "sample", "sample": value.strip().lower()}
    if key == "thread_id":
        path = await thread_resolver(value)
        if not path:
            raise SwarmRequestError("that thread has no dataset this caller can read")
        return path, {"kind": "thread", "thread_id": value}
    if key == "path":
        if not paths_allowed():
            raise SwarmRequestError(
                "source.path is disabled; set AVALOKA_SWARM_ALLOW_PATHS=1 to allow server-side paths")
        p = Path(value).expanduser()
        if not p.is_file():
            raise SwarmRequestError("source.path is not a readable file")
        return str(p), {"kind": "path"}
    raise SwarmRequestError(f"unknown source kind {key!r}")


def _output_dir(mission_id: str) -> Path:
    base = Path(os.getenv("AVALOKA_SWARM_OUTPUT_DIR") or tempfile.gettempdir()) / "avaloka-swarm"
    out = base / mission_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _validate(kind: str, workers: Optional[int], max_parallel: Optional[int],
              target: Optional[str], limits: Any) -> tuple[str, Optional[int], Optional[int]]:
    kind = (kind or "analyze").strip().lower()
    if kind not in KINDS:
        raise SwarmRequestError("kind must be 'analyze' or 'train'")
    if kind == "train" and not target:
        raise SwarmRequestError("a training swarm requires a target column")
    # ``workers`` stays None when not asked for: as the CLI, the fleet is
    # then sized from the data (avaloka.swarm_sizing) once the source is
    # resolved, and the edition cap is applied after that.
    if workers is not None and workers < 1:
        raise SwarmRequestError("workers must be at least 1")
    if max_parallel is not None and max_parallel < 1:
        raise SwarmRequestError("max_parallel must be at least 1")
    return kind, (int(workers) if workers is not None else None), max_parallel


async def start_mission(
    user_id: str,
    *,
    source: Any,
    goal: str,
    kind: str = "analyze",
    workers: Optional[int] = None,
    max_parallel: Optional[int] = None,
    target: Optional[str] = None,
    metric: Optional[str] = None,
    edition: Optional[str] = None,
    thread_resolver: ThreadResolver,
) -> Dict[str, Any]:
    """Validate, register and dispatch. Returns the POST response body."""
    from app.core.editions import swarm_limits
    from avaloka.mission.context import MissionKind
    from avaloka.swarm_limits import grant

    if not isinstance(goal, str) or not goal.strip():
        raise SwarmRequestError("goal is required")
    limits = swarm_limits(edition)
    kind, workers, max_parallel = _validate(kind, workers, max_parallel, target, limits)
    path, named = await resolve_source(source, thread_resolver=thread_resolver)
    derived_from: Optional[str] = None
    if workers is None:
        # Sized here, on a local file, so the response can say what was
        # granted; the coordinator is handed the same number and reason.
        from avaloka.coordinator import derive_workers

        fleet, _ = derive_workers(path)
        workers, derived_from = fleet.clones, fleet.derived_from
    allowed = grant(workers, max_parallel, limits, derived_from=derived_from)

    mission_id = uuid.uuid4().hex[:12]
    recorder = SwarmTraceRecorder(mission_id=mission_id)
    entry = Entry(mission_id=mission_id, user_id=user_id, recorder=recorder,
                  output_dir=str(_output_dir(mission_id)))
    store.registry.add(entry)

    try:
        loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - always called from a route
        loop = None

    def run() -> None:
        from avaloka.coordinator import MainAvaloka

        try:
            MainAvaloka(swarm_limits=limits).coordinate(
                path, goal=goal, output_dir=entry.output_dir, workers=workers,
                kind=MissionKind.TRAIN if kind == "train" else MissionKind.ANALYZE,
                target=target, metric=metric, max_parallel=max_parallel, recorder=recorder,
                derived_from=derived_from,
            )
        except Exception as exc:  # noqa: BLE001 - the recorder already holds swarm.finished=failed
            entry.error_class = type(exc).__name__
            logger.info("[swarm] mission %s failed: %s", mission_id, type(exc).__name__)
        # Best effort from this thread; the first read after finishing
        # persists too, so a loop that is gone (tests) loses nothing.
        if loop is not None and not loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(store.persist(entry), loop)
            except Exception:  # noqa: BLE001
                pass

    entry.thread = threading.Thread(target=run, name=f"swarm-{mission_id}", daemon=True)
    entry.thread.start()
    return {
        "mission_id": mission_id,
        "status": "running",
        "grant": allowed.as_dict(),
        "limits": limits.as_dict(),
        "source": named,
        "persistence": "redis" if store.has_redis() else "memory",
    }


# ------------------------------------------------------------------- reading
async def _flush(entry: Entry) -> None:
    if entry.finished and not entry.persisted:
        await store.persist(entry)


def _with_persistence(doc: Dict[str, Any], entry: Optional[Entry]) -> Dict[str, Any]:
    if entry is not None:
        doc["persistence"] = "redis" if entry.persisted else "memory"
        if doc["persistence"] == "memory" and not store.has_redis():
            doc["persistence_note"] = "no Redis: this mission is lost when the server restarts"
    return doc


async def get_mission(user_id: str, mission_id: str, *, records: bool = False) -> Optional[Dict[str, Any]]:
    """The mission document, or ``None`` when it is not this caller's."""
    entry = store.registry.get(mission_id)
    if entry is not None:
        if entry.user_id != user_id:
            return None
        await _flush(entry)
        return _with_persistence(entry.recorder.mission(include_records=records), entry)
    doc = await store.load(mission_id, user_id)
    if doc is None:
        return None
    if not records:
        doc.pop("records", None)
    return doc


async def records_after(user_id: str, mission_id: str, *, after: int = 0,
                        limit: int = DEFAULT_RECORDS_PAGE) -> Optional[Dict[str, Any]]:
    """The live endpoint: records with ``seq > after``, and whether to keep polling."""
    limit = max(1, min(int(limit), MAX_RECORDS_PAGE))
    after = max(0, int(after))
    entry = store.registry.get(mission_id)
    if entry is not None:
        if entry.user_id != user_id:
            return None
        await _flush(entry)
        recs = entry.recorder.records_after(after, limit)
        status = entry.status
    else:
        doc = await store.load(mission_id, user_id)
        if doc is None:
            return None
        stored = doc.get("records", [])
        recs = [r for r in stored if r.get("seq", 0) > after][:limit]
        status = doc.get("status")
        last_seq = max((r.get("seq", 0) for r in stored), default=0)
    if entry is not None:
        last_seq = entry.recorder.last_seq
    cursor = recs[-1]["seq"] if recs else after
    # Done only once the caller has drained the tail: a finished mission with
    # records past the cursor still has something to show.
    done = is_finished(status) and cursor >= last_seq
    return {"mission_id": mission_id, "status": status, "records": recs, "cursor": cursor, "done": done}


async def understanding_after(user_id: str, mission_id: str, *, after: int = 0,
                              limit: int = DEFAULT_RECORDS_PAGE) -> Optional[Dict[str, Any]]:
    """The understanding records (``agent.understanding``, ``swarm.understanding``)
    with ``seq > after``, paged like :func:`records_after`. ``cursor`` is the
    last seq returned; ``done`` says the mission finished and no understanding
    record lies past the cursor."""
    limit = max(1, min(int(limit), MAX_RECORDS_PAGE))
    after = max(0, int(after))
    entry = store.registry.get(mission_id)
    if entry is not None:
        if entry.user_id != user_id:
            return None
        await _flush(entry)
        stored = entry.recorder.records_after(after)
        status = entry.status
    else:
        doc = await store.load(mission_id, user_id)
        if doc is None:
            return None
        stored = [r for r in doc.get("records", []) if r.get("seq", 0) > after]
        status = doc.get("status")
    matching = [r for r in stored if r.get("type") in UNDERSTANDING_TYPES]
    recs = matching[:limit]
    cursor = recs[-1]["seq"] if recs else after
    done = is_finished(status) and len(matching) == len(recs)
    return {"mission_id": mission_id, "status": status, "records": recs, "cursor": cursor, "done": done}


async def list_missions(user_id: str) -> List[Dict[str, Any]]:
    """The caller's missions, newest first, memory and Redis merged."""
    seen: Dict[str, Dict[str, Any]] = {}
    for entry in store.registry.for_user(user_id):
        await _flush(entry)
        seen[entry.mission_id] = entry.recorder.mission(include_records=False)
    for mid in await store.user_missions(user_id):
        if mid not in seen:
            doc = await store.load(mid, user_id)
            if doc:
                seen[mid] = doc
    rows = [{
        "mission_id": doc.get("mission_id"), "status": doc.get("status"), "kind": doc.get("kind"),
        "goal": doc.get("goal"), "started_at": doc.get("started_at"), "ended_at": doc.get("ended_at"),
        "clones": len(doc.get("clones") or []),
        "verdict": (doc.get("merge") or {}).get("verdict"),
    } for doc in seen.values()]
    rows.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return rows


async def get_clone(user_id: str, mission_id: str, clone_id: str) -> Optional[Dict[str, Any]]:
    doc = await get_mission(user_id, mission_id, records=True)
    if doc is None:
        return None
    return clone_document(doc, clone_id)


async def get_episodes(user_id: str, mission_id: str) -> Optional[Dict[str, Any]]:
    doc = await get_mission(user_id, mission_id, records=False)
    if doc is None:
        return None
    return {"mission_id": mission_id, "status": doc.get("status"), "episodes": episodes(doc)}


async def get_telemetry(user_id: str, mission_id: str) -> Optional[Dict[str, Any]]:
    """Exactly the records that would leave the machine, so a reviewer can read them."""
    from app.trace.telemetry import swarm_events

    doc = await get_mission(user_id, mission_id, records=False)
    if doc is None:
        return None
    return {"mission_id": mission_id, "status": doc.get("status"), "events": swarm_events(doc)}


async def get_understanding_telemetry(user_id: str, mission_id: str) -> Optional[Dict[str, Any]]:
    """Exactly the understanding events that would leave the machine; nothing is sent."""
    from app.trace.telemetry import understanding_events

    doc = await get_mission(user_id, mission_id, records=True)
    if doc is None:
        return None
    return {"mission_id": mission_id, "status": doc.get("status"), "events": understanding_events(doc)}


async def delete_mission(user_id: str, mission_id: str) -> bool:
    """Forget the mission. A run still in flight keeps running to completion
    on its thread, but nothing serves it any more."""
    entry = store.registry.get(mission_id)
    found = False
    if entry is not None:
        if entry.user_id != user_id:
            return False
        store.registry.remove(mission_id)
        found = True
    if await store.load(mission_id, user_id) is not None:
        found = True
    await store.delete(mission_id, user_id)
    return found
