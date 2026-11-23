"""Recording agent steps without the agents knowing.

Three hooks, all installed at construction time:

* :func:`traced_node` wraps a graph node. It times the node, diffs what it
  wrote, asks :mod:`app.trace.describe` what it decided and calculated, and
  returns the step on the ``agent_trace`` channel of the graph state.
* :func:`traced_router` wraps a conditional-edge function and records the route
  it chose and the state signals it chose on.
* :func:`instrument` attaches a LangChain callback to a chat model so every
  call's provider, model, tokens, cost and latency land on the step that made
  it. ``app.core.inference.build_chat_model`` calls it; it is the one place
  every agent's model is built.

A step being recorded is held in a context variable, so a model call made deep
inside a node, or a node of the coding sub-graph run from inside ``code_etl``,
attaches to the right parent without any argument being threaded through.

Recording must never change what the graph does. Every hook swallows its own
failures, and a node's exception is re-raised unchanged after the failed step
is noted.
"""

from __future__ import annotations

import contextvars
import functools
import itertools
import logging
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

TRACE_SCHEMA_VERSION = "1.0"

#: The step being recorded in this context, if any.
_FRAME: contextvars.ContextVar[Optional["StepFrame"]] = contextvars.ContextVar(
    "avaloka_trace_frame", default=None
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


# --------------------------------------------------------------------------- #
# Per-turn bookkeeping: sequence numbers, and entries with nowhere else to go
# --------------------------------------------------------------------------- #
#
# A router cannot write graph state, and a node that raised returns nothing, so
# their entries wait here until the next node of the same turn returns and
# carries them out on the state channel. Bounded: an abandoned turn is evicted.

_MAX_TURNS = 256
_TURNS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_TURNS_LOCK = threading.Lock()


def _turn(turn_id: str) -> Dict[str, Any]:
    with _TURNS_LOCK:
        entry = _TURNS.get(turn_id)
        if entry is None:
            entry = {"seq": itertools.count(1), "pending": [], "last_node": None}
            _TURNS[turn_id] = entry
            while len(_TURNS) > _MAX_TURNS:
                _TURNS.popitem(last=False)
        else:
            _TURNS.move_to_end(turn_id)
        return entry


def _next_seq(turn_id: str) -> int:
    entry = _turn(turn_id)
    with _TURNS_LOCK:
        return next(entry["seq"])


def _park(turn_id: str, item: Dict[str, Any]) -> None:
    entry = _turn(turn_id)
    with _TURNS_LOCK:
        entry["pending"].append(item)


def drain(turn_id: str) -> List[Dict[str, Any]]:
    """Entries recorded for ``turn_id`` that no node has carried out yet."""
    with _TURNS_LOCK:
        entry = _TURNS.get(turn_id)
        if not entry:
            return []
        out, entry["pending"] = entry["pending"], []
        return out


def _last_node(turn_id: str) -> Optional[str]:
    with _TURNS_LOCK:
        entry = _TURNS.get(turn_id)
        return entry["last_node"] if entry else None


def _set_last_node(turn_id: str, node: str) -> None:
    entry = _turn(turn_id)
    with _TURNS_LOCK:
        entry["last_node"] = node


# --------------------------------------------------------------------------- #
# The state channel
# --------------------------------------------------------------------------- #

def merge_agent_trace(existing: Optional[list], new: Optional[list]) -> list:
    """Reducer for ``ETLState.agent_trace``: one turn's steps, in order.

    The top-level graph is checkpointed per thread, so a plain list-append would
    carry every earlier turn forward. A batch from a new turn replaces the
    channel instead; the server persists each finished turn on its own.
    """
    existing = list(existing or [])
    if not new:
        return existing
    new = list(new)
    turn = new[0].get("turn_id") if isinstance(new[0], dict) else None
    if existing and isinstance(existing[-1], dict) and existing[-1].get("turn_id") == turn:
        return existing + new
    return new


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #

class StepFrame:
    """A step while it is running."""

    __slots__ = ("node", "turn_id", "step_id", "parent_id", "seq", "started_at",
                 "t0", "inference", "children", "lock")

    def __init__(self, node: str, turn_id: str, parent: Optional["StepFrame"]) -> None:
        self.node = node
        self.turn_id = turn_id
        self.step_id = _new_id()
        self.parent_id = parent.step_id if parent else None
        self.seq = _next_seq(turn_id)
        self.started_at = _now()
        self.t0 = time.perf_counter()
        self.inference: List[Dict[str, Any]] = []
        self.children: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def add_inference(self, record: Dict[str, Any]) -> None:
        with self.lock:
            self.inference.append(record)

    def add_child(self, entry: Dict[str, Any]) -> None:
        with self.lock:
            self.children.append(entry)


def current_frame() -> Optional[StepFrame]:
    return _FRAME.get()


def inference_mark() -> int:
    """How many model calls the current step has recorded so far.

    Taken before a call so :func:`fail_inference_since` can find the records
    that call produced. 0 outside a step.
    """
    frame = _FRAME.get()
    if frame is None:
        return 0
    with frame.lock:
        return len(frame.inference)


def fail_inference_since(mark: int, *, error_class: str, agent: str,
                         provider: str = "", model: str = "") -> None:
    """Count the model calls recorded since *mark* as failed.

    A provider that answers HTTP 200 with a reply the agent cannot use -- a
    tool-selection call that came back as prose -- is a successful call to
    the callback and a failed one to the agent. The callback cannot know; the
    agent can, and says so here, so ``inference[].outcome`` and the metrics
    built on it count what actually happened. A model that is not
    instrumented (a test double) leaves no record to amend, so one is added
    with the class and nothing else.
    """
    frame = _FRAME.get()
    if frame is None:
        return
    with frame.lock:
        since = frame.inference[mark:]
        if since:
            for record in since:
                record["outcome"] = "failed"
                record["error_class"] = error_class
            return
        frame.inference.append({
            "agent": agent, "provider": provider or None, "model": model or None,
            "latency_ms": None, "at": _now(), "tokens_in": None, "tokens_out": None,
            "cost_usd": None, "cost_source": None,
            "outcome": "failed", "error_class": error_class,
        })


def _safe_eq(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - DataFrames and friends refuse truthiness
        return False


def written_keys(before: Dict[str, Any], result: Any) -> List[str]:
    """State keys a node's return value actually changed."""
    if not isinstance(result, dict):
        return []
    out = []
    for key, value in result.items():
        if key in ("agent_trace", "trace_turn_id"):
            continue
        if key not in before:
            out.append(key)
            continue
        old = before[key]
        if value is old:
            continue
        if not _safe_eq(value, old):
            out.append(key)
    return sorted(out)


def recorded_failure(before: Dict[str, Any], result: Any) -> Optional[str]:
    """The class of a failure the node wrote to ``agent_errors``, if it did.

    A node that must answer the user cannot raise: an exception ends the graph
    turn and the reply with it. Such a node records its failure on the state
    instead (``app.agents.contract.failure_record``) and returns normally, and
    the step is still a failed step. Only entries new in this node's result
    count; what earlier nodes recorded is theirs.
    """
    if not isinstance(result, dict):
        return None
    after = result.get("agent_errors")
    if not isinstance(after, list) or not after:
        return None
    seen = before.get("agent_errors") if isinstance(before.get("agent_errors"), list) else []
    fresh = [e for e in after if isinstance(e, dict) and e not in seen]
    if not fresh:
        return None
    last = fresh[-1]
    cls = last.get("error_class")
    if not isinstance(cls, str) or not cls:
        head = str(last.get("error") or "").split(":", 1)[0].strip()
        cls = head or "Exception"
    return cls


def _close(frame: StepFrame, before: Dict[str, Any], result: Any,
           error: Optional[BaseException] = None) -> Dict[str, Any]:
    from app.trace.describe import describe_step  # deferred: keeps import light

    latency_ms = int(round((time.perf_counter() - frame.t0) * 1000))
    written = written_keys(before, result)
    updates = {k: result[k] for k in written} if isinstance(result, dict) else {}
    error_class = type(error).__name__ if error is not None else None
    if error_class is None:
        try:
            error_class = recorded_failure(before, result)
        except Exception:  # noqa: BLE001 - never lose the step over bookkeeping
            error_class = None
    step: Dict[str, Any] = {
        "kind": "agent",
        "step_id": frame.step_id,
        "parent_id": frame.parent_id,
        "turn_id": frame.turn_id,
        "seq": frame.seq,
        "node": frame.node,
        "started_at": frame.started_at,
        "ended_at": _now(),
        "latency_ms": latency_ms,
        "status": "error" if error_class is not None else "ok",
        "error_class": error_class,
        "inference": list(frame.inference),
        "written": written,
    }
    try:
        step.update(describe_step(frame.node, before, updates, error=error,
                                  error_class=error_class))
    except Exception:  # noqa: BLE001 - a describer bug must not lose the step
        logger.debug("[trace] describe failed for %s", frame.node, exc_info=True)
    _set_last_node(frame.turn_id, frame.node)
    return step


def _deposit(step: Dict[str, Any], parent: Optional[StepFrame], children: List[Dict[str, Any]]) -> None:
    """File a step that cannot travel on the state channel itself."""
    for item in [step, *children]:
        if parent is not None:
            parent.add_child(item)
        else:
            _park(item["turn_id"], item)


def _turn_for(state: Any, parent: Optional[StepFrame], entry: bool) -> str:
    if entry:
        # The entry node starts every turn. A checkpointed thread still holds
        # the previous turn's id, so a new one is minted here, always.
        return _new_id()
    if parent is not None:
        return parent.turn_id
    if isinstance(state, dict):
        return str(state.get("trace_turn_id") or "unscoped")
    return "unscoped"


def traced_node(node: str, fn: Callable, *, entry: bool = False, nested: bool = False) -> Callable:
    """Wrap a graph node so its step is recorded.

    ``entry``: this node starts a turn. ``nested``: this node runs in a
    sub-graph whose state has no ``agent_trace`` channel, so its step is handed
    to the enclosing step instead of returned.

    Test doubles are returned unwrapped: several tests detect a mocked node by
    its mock attributes, and recording a mock describes nothing.
    """
    if fn is None or getattr(fn, "__avaloka_traced__", False) or hasattr(fn, "_mock_return_value"):
        return fn

    @functools.wraps(fn, updated=())
    def wrapper(state, *args, **kwargs):
        parent = _FRAME.get()
        turn_id = _turn_for(state, parent, entry)
        try:
            frame = StepFrame(node, turn_id, parent)
            before = dict(state) if isinstance(state, dict) else {}
        except Exception:  # noqa: BLE001
            return fn(state, *args, **kwargs)
        token = _FRAME.set(frame)
        try:
            result = fn(state, *args, **kwargs)
        except BaseException as exc:
            _FRAME.reset(token)
            try:
                _deposit(_close(frame, before, None, error=exc), parent, frame.children)
            except Exception:  # noqa: BLE001
                logger.debug("[trace] could not record failed %s", node, exc_info=True)
            raise
        _FRAME.reset(token)
        try:
            step = _close(frame, before, result)
        except Exception:  # noqa: BLE001
            logger.debug("[trace] could not record %s", node, exc_info=True)
            return result

        if nested or parent is not None or not isinstance(result, dict):
            _deposit(step, parent, frame.children)
            return result
        out = dict(result)
        out["agent_trace"] = drain(turn_id) + [step] + list(frame.children)
        if entry:
            out["trace_turn_id"] = turn_id
        return out

    wrapper.__avaloka_traced__ = True  # type: ignore[attr-defined]
    return wrapper


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

def _signal(value: Any) -> Any:
    """A route signal, small enough to show and never a payload."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 80 else value[:77] + "..."
    if isinstance(value, dict):
        return {"present": True, "keys": len(value)}
    if isinstance(value, (list, tuple)):
        return {"present": True, "items": len(value)}
    return {"present": True}


def traced_router(name: str, fn: Callable, *, signals: tuple = ()) -> Callable:
    """Wrap a conditional-edge function so the route it chose is recorded."""
    if fn is None or getattr(fn, "__avaloka_traced__", False) or hasattr(fn, "_mock_return_value"):
        return fn

    @functools.wraps(fn, updated=())
    def wrapper(state, *args, **kwargs):
        chosen = fn(state, *args, **kwargs)
        try:
            from app.trace.describe import route_reason

            parent = _FRAME.get()
            turn_id = _turn_for(state, parent, False)
            values = state if isinstance(state, dict) else {}
            entry_ = {
                "kind": "decision",
                "step_id": _new_id(),
                "parent_id": parent.step_id if parent else None,
                "turn_id": turn_id,
                "seq": _next_seq(turn_id),
                "node": name,
                "after": _last_node(turn_id),
                "at": _now(),
                "decision": {
                    "type": "route",
                    "router": name,
                    "chosen": str(chosen),
                    "reason": route_reason(name, str(chosen), values),
                    "signals": {s: _signal(values.get(s)) for s in signals if s in values},
                },
            }
            if parent is not None:
                parent.add_child(entry_)
            else:
                _park(turn_id, entry_)
        except Exception:  # noqa: BLE001
            logger.debug("[trace] could not record route %s", name, exc_info=True)
        return chosen

    wrapper.__avaloka_traced__ = True  # type: ignore[attr-defined]
    return wrapper


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def _usage_from(response: Any) -> Dict[str, Any]:
    """Tokens, model and provider-reported cost from a LangChain LLMResult."""
    tokens_in = tokens_out = None
    model = None
    cost = None
    tool_calls = 0
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    model = llm_output.get("model_name") or llm_output.get("model")
    for generations in getattr(response, "generations", None) or []:
        for gen in generations or []:
            msg = getattr(gen, "message", None)
            meta = getattr(msg, "usage_metadata", None) or {}
            if meta:
                tokens_in = meta.get("input_tokens", tokens_in)
                tokens_out = meta.get("output_tokens", tokens_out)
            rmeta = getattr(msg, "response_metadata", None) or {}
            usage = usage or rmeta.get("token_usage") or rmeta.get("usage") or {}
            model = model or rmeta.get("model_name") or rmeta.get("model")
            tool_calls += len(getattr(msg, "tool_calls", None) or [])
    if not isinstance(usage, dict):
        usage = {}
    if tokens_in is None:
        tokens_in = usage.get("prompt_tokens") or usage.get("input_tokens")
    if tokens_out is None:
        tokens_out = usage.get("completion_tokens") or usage.get("output_tokens")
    if isinstance(usage.get("cost"), (int, float)):
        cost = float(usage["cost"])
    return {"tokens_in": tokens_in, "tokens_out": tokens_out, "model": model,
            "cost_usd": cost, "tool_calls": tool_calls}


try:  # langchain_core is a hard dependency of app/, but keep the import honest
    from langchain_core.callbacks import BaseCallbackHandler as _Base
except Exception:  # pragma: no cover
    _Base = object  # type: ignore[assignment,misc]


class UsageRecorder(_Base):  # type: ignore[misc,valid-type]
    """Attaches each model call's usage to the step that made it."""

    raise_error = False
    run_inline = True

    def __init__(self, *, agent: str, provider: str, model: str,
                 configured_model: Optional[str] = None, primary_provider: str = "openrouter") -> None:
        super().__init__()
        self.agent = agent
        self.provider = provider
        self.model = model
        #: Set on a fallback tier: the model it stands in for. Its successful call is
        #: then recorded with ``answered_by`` (owner ruling 2026-11-02: a model swap
        #: is never silent), built from app/core/model_tiers like the reply's record.
        self.configured_model = configured_model
        self.primary_provider = primary_provider
        self._starts: Dict[Any, tuple] = {}
        self._lock = threading.Lock()

    def _start(self, run_id: Any) -> None:
        with self._lock:
            self._starts[run_id] = (time.perf_counter(), _FRAME.get())
            if len(self._starts) > 1024:  # a run that never ended
                self._starts.pop(next(iter(self._starts)))

    def _finish(self, run_id: Any) -> tuple:
        with self._lock:
            return self._starts.pop(run_id, (None, _FRAME.get()))

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):  # noqa: D401
        self._start(run_id)

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._start(run_id)

    def _record(self, run_id: Any, extra: Dict[str, Any]) -> None:
        t0, frame = self._finish(run_id)
        if frame is None:
            return  # a call made outside any graph node: nothing to attach it to
        record = {
            "agent": self.agent,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": int(round((time.perf_counter() - t0) * 1000)) if t0 else None,
            "at": _now(),
        }
        record.update(extra)
        if record.get("cost_usd") is None:
            from app.trace.prices import cost_for

            priced = cost_for(record.get("model") or self.model,
                              record.get("tokens_in"), record.get("tokens_out"))
            record["cost_usd"] = priced
            record["cost_source"] = "price_table" if priced is not None else None
        else:
            record["cost_source"] = "provider"
        frame.add_inference(record)

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        try:
            usage = _usage_from(response)
            if not usage.get("model"):
                usage.pop("model", None)
            usage["outcome"] = "success"
            if self.configured_model:
                from app.core.model_tiers import answered_record

                usage["answered_by"] = answered_record(
                    configured_model=self.configured_model, answered_model=self.model,
                    provider=self.provider, primary_provider=self.primary_provider)
            self._record(run_id, usage)
        except Exception:  # noqa: BLE001
            logger.debug("[trace] usage capture failed", exc_info=True)

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        try:
            extra = {"outcome": "failed", "error_class": type(error).__name__,
                     "tokens_in": None, "tokens_out": None, "cost_usd": None}
            try:
                from app.core.model_tiers import classify

                failure = classify(error, model=self.model, provider=self.provider)
            except Exception:  # noqa: BLE001
                failure = None
            if failure is not None:
                # The status and the reason the next tier was asked (never the
                # provider's raw message, which can quote the request).
                extra["status"] = failure.status
                extra["reason"] = failure.reason
            self._record(run_id, extra)
        except Exception:  # noqa: BLE001
            logger.debug("[trace] error capture failed", exc_info=True)


def instrument(llm: Any, *, agent: str, provider: str, model: str,
               configured_model: Optional[str] = None, primary_provider: str = "openrouter") -> Any:
    """Attach a :class:`UsageRecorder` to ``llm`` and return it unchanged otherwise.

    ``configured_model`` marks ``llm`` as a fallback tier standing in for that model.
    """
    if llm is None or _Base is object:
        return llm
    try:
        existing = getattr(llm, "callbacks", None)
        if existing is not None and not isinstance(existing, list):
            return llm  # a callback manager someone configured; leave it alone
        existing = list(existing or [])
        if any(isinstance(h, UsageRecorder) for h in existing):
            return llm
        llm.callbacks = existing + [UsageRecorder(agent=agent, provider=provider, model=model,
                                                  configured_model=configured_model,
                                                  primary_provider=primary_provider)]
    except Exception:  # noqa: BLE001 - instrumentation is optional
        logger.debug("[trace] could not instrument %s", agent, exc_info=True)
    return llm
