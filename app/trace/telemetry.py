"""The agent trace as process telemetry: agent-level facts and nothing else.

Projects recorded steps onto the ``agent.started`` / ``agent.completed`` pair
that ``telemetry_service/schema.py`` accepts (``AGENT_FIELDS``). The projection
is an allowlist, never a scrub: each field is built from a closed vocabulary,
a number or an exception class name, so there is no field a column name, a
file name, a value, a question or a code fragment could travel in.

* ``agent`` is the graph node name, from the ingest service's own vocabulary;
  a step whose node is not in it is dropped, not renamed.
* ``llm_model`` is a shipped default model or ``"other"``. A customer's model
  or deployment name can carry their name, so it is never sent.
* ``error_class`` is the class name only; the message is never read.
* ``session_id`` is a random id for this process's lifetime, never the thread,
  conversation or user id (TELEMETRY.md §5.2 refuses those).

Nothing here sends anything. The writer and uploader that would ship these
records are not built (TELEMETRY.md §4.3-4.4); :func:`agent_events` is what
they will consume, and ``tests/test_agent_trace_telemetry.py`` validates its
output against the ingest schema and against planted customer content.

:func:`swarm_events` is the same projection for a coordinated mission
(AGENT_TRACE.md §6): each clone step as an ``agent.*`` pair named after its
firefly, plus one ``swarm.completed`` carrying the fan-out and its outcome.
``tests/test_swarm_trace_telemetry.py`` plants a column name, a file name, a
goal and values through a real swarm and checks none of them travel.

:func:`understanding_events` projects the ``agent.understanding`` and
``swarm.understanding`` records (:mod:`avaloka.understanding_trace`): what each
agent concluded about the data, as buckets and histograms. The projection
itself lives beside the recorder so the CLI writes the same payload; this is
the server's entry point, numbering the events in this process's session.
``tests/test_understanding_trace_privacy.py`` plants a column name, a file
name, a cell value, a goal and a prompt fragment and checks none travel.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List

#: The graph nodes this projection may name. Every one must be in the ingest
#: service's AGENTS (telemetry_service/schema.py, which does not import the
#: product); the ingest also names CLI agents that never pass through here.
#: tests/test_agent_trace_telemetry.py holds both directions.
KNOWN_AGENTS = frozenset({
    "coder", "validator_syntax", "validator_static", "execute_code",
    "validator_contract", "validator_logical",
    "drivers_route", "memory_injection", "avaloka_agent", "plan_etl", "summarize_etl",
    "generate_planner_graph", "code_etl", "execute_actions", "provision_infra",
    "schedule_task", "execute_on_ray", "execute_locally", "execute_on_k8s",
    "visualize", "narrate_result", "verify_claims", "train_models", "profile_data",
    "replan",
})
KNOWN_PROVIDERS = frozenset({"groq", "openrouter", "local", "openai", "bedrock", "vertex", "azure"})
KNOWN_MODELS = frozenset({"openai/gpt-oss-120b", "openai/gpt-oss-20b"})

_SESSION_ID = str(uuid.uuid4())


def _model(name: Any) -> str:
    return name if isinstance(name, str) and name in KNOWN_MODELS else "other"


def _class(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    ok = name.replace("_", "a").replace(".", "a").isalnum() and name[:1].isalpha() and len(name) <= 64
    return name if ok else "Exception"


def agent_events(steps: Iterable[Dict[str, Any]], *, start_seq: int = 0) -> List[Dict[str, Any]]:
    """``agent.started`` + ``agent.completed`` per recorded agent step."""
    out: List[Dict[str, Any]] = []
    seq = start_seq
    for step in steps:
        if not isinstance(step, dict) or step.get("kind") != "agent":
            continue
        agent = step.get("node")
        if agent not in KNOWN_AGENTS:
            continue
        calls = step.get("inference") or []
        provider = next((c.get("provider") for c in calls if c.get("provider") in KNOWN_PROVIDERS), None)
        model = next((c.get("model") for c in calls if c.get("model")), None)
        tokens_in = sum(int(c.get("tokens_in") or 0) for c in calls)
        tokens_out = sum(int(c.get("tokens_out") or 0) for c in calls)
        failed_calls = sum(1 for c in calls if c.get("outcome") != "success")
        started_seq = seq
        out.append({"event_type": "agent.started", "event_id": str(uuid.uuid4()),
                    "agent": agent, "session_id": _SESSION_ID, "seq": seq})
        seq += 1
        completed: Dict[str, Any] = {
            "event_type": "agent.completed",
            "event_id": str(uuid.uuid4()),
            "agent": agent,
            "outcome": "success" if step.get("status") == "ok" else "failed",
            "duration_ms": max(0, int(step.get("latency_ms") or 0)),
            # Failed model calls retried within the step (a fallback answered).
            "retry_count": min(100, failed_calls),
            "started_seq": started_seq,
            "session_id": _SESSION_ID,
            "seq": seq,
        }
        if step.get("status") != "ok":
            cls = _class(step.get("error_class"))
            if cls:
                completed["error_class"] = cls
        if calls:
            if provider:
                completed["llm_provider"] = provider
            completed["llm_model"] = _model(model)
            completed["tokens_in"] = tokens_in
            completed["tokens_out"] = tokens_out
        out.append(completed)
        seq += 1
    return out


# ------------------------------------------------------------------ swarm
#: The CLI fireflies a clone step may be reported as. Every one must be in the
#: ingest's AGENTS; a firefly it does not know is dropped, never renamed.
KNOWN_FIREFLIES = frozenset({
    "analysis_planner", "data_engineer", "data_scout", "finops", "ml_engineer",
    "model_scientist", "reporter", "sampling_specialist", "validator",
    # The 1.7 line adds a feature engineer and a workload optimizer to every
    # mission; both are agent identifiers in the ingest vocabulary.
    "feature_engineer", "workload_optimizer",
})
_SWARM_ROW_BUCKETS = (("0-1e3", 10 ** 3), ("1e3-1e4", 10 ** 4), ("1e4-1e5", 10 ** 5),
                      ("1e5-1e6", 10 ** 6), ("1e6-1e7", 10 ** 7), ("1e7-1e8", 10 ** 8))
_EDITIONS = frozenset({"oss", "free", "professional", "enterprise"})
_VERDICTS = frozenset({"pass", "warn", "fail"})
_KINDS = frozenset({"analyze", "train"})


def _row_bucket(rows: Any) -> str:
    n = max(0, int(rows or 0))
    for label, hi in _SWARM_ROW_BUCKETS:
        if n < hi:
            return label
    return "1e8+"


def _bounded(value: Any, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return lo


def swarm_events(mission: Dict[str, Any], *, start_seq: int = 0) -> List[Dict[str, Any]]:
    """The metadata-only projection of a swarm mission document.

    One ``agent.started`` / ``agent.completed`` pair per completed clone step
    (the firefly is the agent), then one ``swarm.completed`` once the mission
    has finished. Built the same way as :func:`agent_events`: every field is a
    closed vocabulary, a bounded number or an exception class name. The goal,
    the source, the columns, the clone outputs and names, the mission id and
    the paths exist only in the mission document and are never read here.
    """
    out: List[Dict[str, Any]] = []
    seq = start_seq
    for clone in mission.get("clones") or []:
        for step in clone.get("steps") or []:
            if step.get("status") not in ("ok", "failed"):
                continue  # still running: a start with no completion is for a killed process
            agent = step.get("firefly")
            if agent not in KNOWN_FIREFLIES:
                continue
            meta = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
            started_seq = seq
            out.append({"event_type": "agent.started", "event_id": str(uuid.uuid4()),
                        "agent": agent, "session_id": _SESSION_ID, "seq": seq})
            seq += 1
            completed: Dict[str, Any] = {
                "event_type": "agent.completed",
                "event_id": str(uuid.uuid4()),
                "agent": agent,
                "outcome": "success" if step.get("status") == "ok" else "failed",
                "duration_ms": _bounded(step.get("duration_ms"), 0, 7 * 24 * 3600 * 1000),
                "retry_count": 0,
                "started_seq": started_seq,
                "session_id": _SESSION_ID,
                "seq": seq,
            }
            if step.get("status") != "ok":
                cls = _class(step.get("error_class"))
                if cls:
                    completed["error_class"] = cls
            provider = meta.get("llm_provider")
            if provider in KNOWN_PROVIDERS:
                completed["llm_provider"] = provider
            if "llm_model" in meta:
                completed["llm_model"] = _model(meta.get("llm_model"))
            for key in ("tokens_in", "tokens_out"):
                if isinstance(meta.get(key), int) and not isinstance(meta.get(key), bool):
                    completed[key] = _bounded(meta[key], 0, 1_000_000_000)
            out.append(completed)
            seq += 1
    if mission.get("status") in ("done", "failed"):
        grant = mission.get("grant") or {}
        limits = mission.get("limits") or {}
        merge = mission.get("merge") or {}
        rows = merge.get("n_rows")
        if rows is None:
            rows = sum(int(c.get("rows") or 0) for c in mission.get("clones") or [])
        record: Dict[str, Any] = {
            "event_type": "swarm.completed",
            "event_id": str(uuid.uuid4()),
            "clones_requested": _bounded(grant.get("requested_clones"), 1, 10_000),
            "clones_granted": _bounded(grant.get("granted_clones"), 1, 10_000),
            "parallel": _bounded(grant.get("granted_parallel"), 1, 10_000),
            "limited": bool(grant.get("limited")),
            "outcome": "success" if mission.get("status") == "done" else "failed",
            "duration_ms": _bounded(mission.get("duration_ms"), 0, 7 * 24 * 3600 * 1000),
            "row_count_bucket": _row_bucket(rows),
            "session_id": _SESSION_ID,
            "seq": seq,
        }
        if limits.get("edition") in _EDITIONS:
            record["edition"] = limits["edition"]
        if mission.get("kind") in _KINDS:
            record["kind"] = mission["kind"]
        if merge.get("verdict") in _VERDICTS:
            record["verdict"] = merge["verdict"]
        out.append(record)
    return out


# ---------------------------------------------------------- understanding
def understanding_events(mission: Dict[str, Any], *, start_seq: int = 0) -> List[Dict[str, Any]]:
    """The metadata-only projection of a mission's understanding records.

    Reads ``mission["records"]`` (the document must have been built with
    records). Each event is built by :func:`avaloka.understanding_trace.egress_event`
    from closed vocabularies, buckets and bounded counts; the column names,
    reasons, formulas, check details and ids in the records are never read.
    """
    from avaloka.understanding_trace import egress_events

    return egress_events(mission.get("records") or [], start_seq=start_seq, session_id=_SESSION_ID)
