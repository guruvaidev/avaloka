"""The swarm trace: one record per clone, and the reinforcement-learning view.

A coordinated mission (:mod:`avaloka.coordinator`) fans sub-Avalokas out over
shards and merges what they bring back. ``coordination.json`` keeps the merged
answer; this module keeps *how each clone got there*: what it was handed, every
firefly step it ran with that step's output and cost, what it returned, and
whether the merge selected it. The 1.7 learned layer trains on exactly that.

Records are appended under a lock with a gap-free, monotonic ``seq`` starting
at 1, so a reader can poll ``records after N`` and lose nothing. Each record
is JSON-plain by construction (:func:`_plain`), and an optional ``sink`` sees
every record as it is appended, which is how the server goes live and how the
CLI streams ``swarm_trace.jsonl``.

The mission document and the episodes are *derived* from the records; there
is one source of truth.

``episodes()`` leaves ``reward`` as ``null`` on purpose. Assigning reward is
the learned layer's decision, not the recorder's: whether a clone's work was
good depends on what the layer is optimising for, and a fixed formula here
would bake that choice into every trace ever written. The signals it will use
-- ``selected_by_merge``, ``verdict``, the score and the cost -- are all on the
episode's ``terminal``.

This module imports nothing outside the standard library (S1.08): the
standalone CLI records the same trace the server serves.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

SCHEMA_VERSION = "1.0"

#: Receives each record as it is appended. Must not raise; if it does, the
#: record is still kept and the run continues.
Sink = Callable[[dict[str, Any]], None]

_TERMINAL = ("done", "failed")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _plain(value: Any) -> Any:
    """JSON-plain copy of ``value``: numpy scalars, paths, enums and NaN included.

    Stdlib only, so it duck-types: anything with ``item()`` is treated as a
    numpy scalar, anything with ``tolist()`` as an array. A float that JSON
    cannot carry becomes ``None`` rather than breaking a strict encoder.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):  # str enums
        return value.value
    if hasattr(value, "item") and callable(value.item):
        try:
            return _plain(value.item())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _plain(value.tolist())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def clone_id_for(shard_id: int) -> str:
    return f"ava-{shard_id:02d}"


def clone_name_for(shard_id: int) -> str:
    return f"Ava-{shard_id:02d}"


class SwarmTraceRecorder:
    """Thread-safe, append-only recorder for one coordinated mission."""

    def __init__(self, mission_id: Optional[str] = None, sink: Optional[Sink] = None) -> None:
        self.mission_id = mission_id or uuid.uuid4().hex[:12]
        self._sink = sink
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._seq = 0

    # -- recording ---------------------------------------------------------
    def record(self, type: str, **fields: Any) -> dict[str, Any]:  # noqa: A002 - the record's key
        rec: dict[str, Any] = {"seq": 0, "ts": _now_iso(), "type": type, "mission_id": self.mission_id}
        rec.update(_plain(fields))
        with self._lock:
            self._seq += 1
            rec["seq"] = self._seq
            self._records.append(rec)
            if self._sink is not None:
                # Under the lock so the sink sees records in seq order.
                try:
                    self._sink(rec)
                except Exception:  # noqa: BLE001 - a sink must never fail the run
                    pass
        return rec

    @property
    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def records_after(self, seq: int, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Records with ``seq`` greater than ``seq``; the polling primitive."""
        with self._lock:
            # seq is dense and 1-based, so the slice is the index.
            start = max(0, int(seq))
            out = self._records[start:]
        return out[:limit] if limit is not None else out

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    # -- clone step bookkeeping ---------------------------------------------
    def clone_progress(self, clone_id: str, ledger: Any) -> "CloneStepProgress":
        """An ``on_progress`` callback for one clone's mission run.

        ``ledger`` is the clone's :class:`avaloka.mission.ledger.Ledger` (any
        object with ``units``); the latest unit is what ``"done"`` records.
        """
        return CloneStepProgress(self, clone_id, ledger)

    # -- derived views -----------------------------------------------------
    def mission(self, *, include_records: bool = True) -> dict[str, Any]:
        return mission_document(self.records, mission_id=self.mission_id,
                                include_records=include_records)


class CloneStepProgress:
    """Turns ``on_progress("start"|"done", ...)`` into step records for one clone."""

    def __init__(self, recorder: SwarmTraceRecorder, clone_id: str, ledger: Any) -> None:
        self._rec = recorder
        self.clone_id = clone_id
        self._ledger = ledger
        self._step_no = 0
        self._open: Optional[dict[str, Any]] = None  # the started, not yet completed, step
        self._started = 0.0

    def __call__(self, event: str, message: str) -> None:
        if event == "start":
            self._step_no += 1
            firefly = message
            self._open = {
                "step_id": f"{self.clone_id}/s{self._step_no:02d}",
                "step_no": self._step_no,
                "firefly": firefly,
                "human_role": None,
            }
            self._started = time.perf_counter()
            self._rec.record("clone.step.started", clone_id=self.clone_id, **self._open)
        elif event == "done" and self._open is not None:
            self.complete("ok")

    def complete(self, outcome: str, *, error_class: Optional[str] = None,
                 error: Optional[str] = None) -> None:
        """Close the open step. ``"ok"`` reads the unit the firefly just recorded;
        ``"failed"`` is for a firefly that raised and so recorded nothing."""
        step = self._open
        if step is None:
            return
        self._open = None
        units = list(getattr(self._ledger, "units", []) or [])
        unit = units[-1] if units and outcome == "ok" else None
        elapsed_ms = int(round((time.perf_counter() - self._started) * 1000))
        fields: dict[str, Any] = {
            **step,
            "human_role": getattr(unit, "human_role", None) if unit is not None else None,
            "output": getattr(unit, "output", None) if unit is not None else None,
            "duration_ms": (int(round(float(getattr(unit, "duration_seconds", 0.0)) * 1000))
                            if unit is not None else elapsed_ms),
            "manual_minutes": float(getattr(unit, "manual_minutes", 0.0)) if unit is not None else 0.0,
            "compute_cost_usd": float(getattr(unit, "compute_cost_usd", 0.0)) if unit is not None else 0.0,
            "model_cost_usd": float(getattr(unit, "model_cost_usd", 0.0)) if unit is not None else 0.0,
            "metadata": dict(getattr(unit, "metadata", {}) or {}) if unit is not None else {},
            "outcome": outcome,
        }
        if unit is not None and getattr(unit, "firefly", None):
            fields["firefly"] = unit.firefly
        if error_class:
            fields["error_class"] = error_class
        if error:
            fields["error"] = error
        self._rec.record("clone.step.completed", clone_id=self.clone_id, **fields)

    @property
    def open_step(self) -> Optional[dict[str, Any]]:
        return self._open


# --------------------------------------------------------------------------
# Derived views
# --------------------------------------------------------------------------
_ENVELOPE = ("seq", "ts", "type", "mission_id")


def _body(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rec.items() if k not in _ENVELOPE}


def mission_document(records: list[dict[str, Any]], *, mission_id: Optional[str] = None,
                     include_records: bool = True) -> dict[str, Any]:
    """The mission document, derived from its records.

    ``clones`` and ``merge`` are rebuilt from the records every time; nothing
    is kept beside them that could disagree.
    """
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mission_id": mission_id or (records[0]["mission_id"] if records else None),
        "kind": None, "goal": None, "source": None,
        "status": "running",
        "started_at": records[0]["ts"] if records else None,
        "ended_at": None,
        "limits": None, "grant": None,
        "clones": [],
        "merge": None,
        "understanding": None,
        "economics": None,
        "error_class": None,
    }
    clones: dict[str, dict[str, Any]] = {}
    steps: dict[str, dict[str, Any]] = {}
    for rec in records:
        kind = rec.get("type")
        body = _body(rec)
        if kind == "swarm.requested":
            doc.update(kind=body.get("kind"), goal=body.get("goal"), source=body.get("source"),
                       limits=body.get("limits"))
            doc["grant"] = {k: body.get(k) for k in (
                "requested_clones", "requested_parallel", "granted_clones", "granted_parallel",
                "limited", "message", "derived_from")}
        elif kind == "swarm.sharded":
            doc["sharding"] = body
        elif kind == "clone.dispatched":
            cid = body["clone_id"]
            received = body.get("received") or {}
            clones[cid] = {
                "clone_id": cid, "name": body.get("name"), "shard_id": body.get("shard_id"),
                "status": "running", "rows": received.get("rows"), "received": received,
                "dispatched_at": rec["ts"], "returned_at": None,
                "steps": [], "verdict": None, "max_safe_deployment_level": None,
                "model": None, "sampling": None, "cost_usd": None, "duration_ms": None,
                "error_class": None, "error": None,
            }
        elif kind == "clone.step.started":
            cid = body["clone_id"]
            step = {
                "step_id": body.get("step_id"), "step_no": body.get("step_no"),
                "firefly": body.get("firefly"), "human_role": body.get("human_role"),
                "started_at": rec["ts"], "ended_at": None, "status": "running",
                "output": None, "duration_ms": None, "manual_minutes": None,
                "compute_cost_usd": None, "model_cost_usd": None, "metadata": {},
                "error_class": None, "error": None,
            }
            steps[step["step_id"]] = step
            clones.setdefault(cid, _orphan_clone(cid))["steps"].append(step)
        elif kind == "clone.step.completed":
            cid = body["clone_id"]
            step = steps.get(body.get("step_id"))
            if step is None:
                step = {"step_id": body.get("step_id"), "started_at": None}
                steps[step["step_id"]] = step
                clones.setdefault(cid, _orphan_clone(cid))["steps"].append(step)
            step.update({
                "step_no": body.get("step_no"), "firefly": body.get("firefly"),
                "human_role": body.get("human_role"), "ended_at": rec["ts"],
                "status": body.get("outcome"), "output": body.get("output"),
                "duration_ms": body.get("duration_ms"), "manual_minutes": body.get("manual_minutes"),
                "compute_cost_usd": body.get("compute_cost_usd"),
                "model_cost_usd": body.get("model_cost_usd"), "metadata": body.get("metadata") or {},
                "error_class": body.get("error_class"), "error": body.get("error"),
            })
        elif kind == "clone.returned":
            cid = body["clone_id"]
            clone = clones.setdefault(cid, _orphan_clone(cid))
            clone.update({
                "status": body.get("status"), "returned_at": rec["ts"],
                "verdict": body.get("verdict"),
                "max_safe_deployment_level": body.get("max_safe_deployment_level"),
                "model": body.get("model"), "sampling": body.get("sampling"),
                "cost_usd": body.get("cost_usd"), "duration_ms": body.get("duration_ms"),
                "error_class": body.get("error_class"), "error": body.get("error"),
            })
        elif kind == "swarm.merged":
            doc["merge"] = body
        elif kind == "swarm.understanding":
            # What the fleet concluded (avaloka.understanding_trace); the
            # per-step agent.understanding records stay in ``records``.
            doc["understanding"] = body
        elif kind == "swarm.finished":
            doc["status"] = body.get("status")
            doc["ended_at"] = rec["ts"]
            doc["duration_ms"] = body.get("duration_ms")
            doc["economics"] = body.get("economics")
            doc["error_class"] = body.get("error_class")
    doc["clones"] = sorted(clones.values(), key=lambda c: (c.get("shard_id") is None, c.get("shard_id") or 0,
                                                            c["clone_id"]))
    if include_records:
        doc["records"] = list(records)
    return doc


def _orphan_clone(cid: str) -> dict[str, Any]:
    """A clone seen before (or without) its ``clone.dispatched`` record."""
    return {"clone_id": cid, "name": None, "shard_id": None, "status": "running", "rows": None,
            "received": {}, "dispatched_at": None, "returned_at": None, "steps": [],
            "verdict": None, "max_safe_deployment_level": None, "model": None, "sampling": None,
            "cost_usd": None, "duration_ms": None, "error_class": None, "error": None}


def clone_document(mission: dict[str, Any], clone_id: str) -> Optional[dict[str, Any]]:
    """One clone of a mission document, with the records that belong to it."""
    clone = next((c for c in mission.get("clones", []) if c.get("clone_id") == clone_id), None)
    if clone is None:
        return None
    out = dict(clone)
    out["mission_id"] = mission.get("mission_id")
    out["records"] = [r for r in mission.get("records", []) if r.get("clone_id") == clone_id]
    return out


def episodes(mission: dict[str, Any]) -> list[dict[str, Any]]:
    """The RL view: one episode per clone.

    ``reward`` is ``null`` on purpose (see the module docstring): the learned
    layer assigns it from ``terminal.selected_by_merge``, ``terminal.verdict``,
    ``terminal.score`` and the costs, with whatever objective it is given.
    """
    merge = mission.get("merge") or {}
    selected = {c.get("clone_id") for c in (merge.get("contributions") or []) if c.get("selected")}
    out = []
    for clone in mission.get("clones", []):
        received = clone.get("received") or {}
        model = clone.get("model") or {}
        steps = []
        for t, step in enumerate(clone.get("steps", [])):
            steps.append({
                "t": t,
                "step_id": step.get("step_id"),
                "action": {"firefly": step.get("firefly"), "human_role": step.get("human_role")},
                "observation": {"output": step.get("output"), "metadata": step.get("metadata") or {}},
                "cost": {
                    "duration_ms": step.get("duration_ms"),
                    "model_cost_usd": step.get("model_cost_usd"),
                    "compute_cost_usd": step.get("compute_cost_usd"),
                    "manual_minutes": step.get("manual_minutes"),
                },
                "outcome": step.get("status"),
            })
        out.append({
            "episode_id": clone["clone_id"],
            "mission_id": mission.get("mission_id"),
            "clone": {"clone_id": clone["clone_id"], "name": clone.get("name"),
                      "shard_id": clone.get("shard_id")},
            "context": {
                "rows": received.get("rows"),
                "columns": received.get("columns"),
                "goal": received.get("goal", mission.get("goal")),
                "kind": received.get("kind", mission.get("kind")),
                "target": received.get("target"),
                "metric": received.get("metric"),
            },
            "steps": steps,
            "terminal": {
                "status": clone.get("status"),
                "verdict": clone.get("verdict"),
                "max_safe_deployment_level": clone.get("max_safe_deployment_level"),
                "score": model.get("score") if isinstance(model, dict) else None,
                "selected_by_merge": clone["clone_id"] in selected,
                "cost_usd": clone.get("cost_usd"),
            },
            "reward": None,
        })
    return out


def is_finished(status: Optional[str]) -> bool:
    return status in _TERMINAL
