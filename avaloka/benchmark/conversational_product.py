"""The conversational probes, driven through Avaloka's own agent graph.

``avaloka.benchmark.conversational`` grades the probe *scorers* (a fixed
dictionary of replies) and, with ``--live``, a raw model over HTTP. Neither
enters the product, so no number it produced was a measurement of Avaloka.
This module is the missing subject: every probe goes through the **compiled
graph** from ``app.api.workflow.build_graph()``, once per probe, and the reply
graded is the last assistant message the user would see.

**How a probe enters the graph.** The initial state is what a chat turn hands
the graph: ``user_prompt`` is the probe's last turn, ``messages`` holds its
turns, ``session_id`` is ``conv-product-<probe id>``, and the probe's evidence
rides in ``avaloka_recommendation`` -- the field ``_render_direct_reply``
already forwards into the reply prompt as ``<recommendation>``, and the channel
``benchmarks/adapters/conversational.py`` established. A probe the agent
delegates continues into the planner and beyond; whatever the product says
there is graded. It is not marked "not measured": that is the behaviour being
measured.

**Two subjects, kept apart by ``source_scheme``.**

``product:replay`` (hermetic, the default). The single model-routing chokepoint,
``app.core.inference.build_chat_model``, is replaced -- in a fresh interpreter,
before any agent module is imported, so module-level models are built through
it too -- by a factory returning :class:`ReplayChatModel`. It answers from a
cassette (``tests/conversational/cassettes/product.jsonl``) keyed by probe,
agent and the SHA-256 of the rendered request. **A request with no recording is
a failed check naming the agent and the hash, never a silent fallback**: agents
catch model exceptions and degrade to canned text, so the miss is written to a
ledger the moment it happens and the grader reads the ledger, not the reply.
Any prompt or graph change that alters a model input shows up as a miss, which
says the cassette must be re-recorded. That is the honest cost of determinism.
No socket leaves the graph's process in replay; an attempt is blocked and
reported. A probe that reaches ``execute_locally`` runs its recorded script in
that node's own child process, which the guard does not reach: the environment
there is the executor's allowlist, and a test holds the committed cassette to
code that imports no network module.

``product:live:<model>`` (``--live``). The same harness with the real router,
local model first, then Groq, never Anthropic -- the policy of
``conversational._resolve_live``. Spend is computed from provider-reported
usage at the operator's rate (``AVALOKA_CONV_INPUT_USD_PER_MTOK`` /
``AVALOKA_CONV_OUTPUT_USD_PER_MTOK``); a hosted run without both refuses to
start rather than spend an unpriced amount. ``--record`` is a live run that
also writes the cassette: against the local model, or a hosted provider named
with ``--provider`` (the committed cassette is OpenRouter's gpt-oss defaults),
capped at ``AVALOKA_CONV_MAX_USD``. Replay selects the provider the cassette
names, so provider-dependent request arguments match the recording.

**Why a child process.** Agent modules build their model at import time
(``planner.llm``, ``coder_llm``, ``narrator_llm`` ...). Patching the factory in a
process where they were already imported would leave those on whatever they
were built with -- a replay that silently was not one. A fresh interpreter is
the only place the patch provably reaches every model, and it keeps the graph's
environment (provider switches, memory settings) out of the caller's process.

**Guards.** ``recursion_limit`` of 25 and a wall-clock budget per probe.
Exceeding either is recorded with the node trace and graded as a failure, not
a skip.

This module imports nothing from ``app`` at module scope: the CLI wheel ships
without ``app``, and ``avaloka`` must stay importable without it. When ``app``
cannot be imported the run reports "unavailable", which the scorecard prints
as SKIP -- neither pass nor fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "CASSETTE_RELPATH",
    "CassetteMiss",
    "ProbeRecord",
    "ProductRun",
    "REPLAY",
    "RECORD",
    "LIVE",
    "request_key",
    "run_product",
]

REPLAY = "replay"
RECORD = "record"
LIVE = "live"
MODES = (REPLAY, RECORD, LIVE)

#: Where the cassette lives, relative to the repository root. Beside the probe
#: suite it records, and like it not shipped in the wheel.
CASSETTE_RELPATH = Path("tests") / "conversational" / "cassettes" / "product.jsonl"

#: LangGraph's step ceiling per probe. A direct reply takes four steps and a
#: delegated one about a dozen; 25 is headroom, not a target.
RECURSION_LIMIT = 25

#: Wall-clock seconds a probe may take, by mode. Replay answers from a file, so
#: a minute is a hang. A local 4B model on CPU takes tens of seconds per call
#: and a delegated probe makes several.
PROBE_BUDGET_S = {REPLAY: 60.0, RECORD: 1500.0, LIVE: 900.0}

#: Seconds the child may spend importing the application before the first
#: probe. ``app.api.workflow`` imports every agent.
IMPORT_BUDGET_S = 300.0

#: Agents given no model in every mode, so they take their documented
#: "provider not configured" path. The memory plane's model runs under a
#: wall-clock circuit breaker (MEMORY_STAGE_TIMEOUT, 3 s): a slow local model
#: times out while recording and the cassette answers instantly on replay, so
#: the graph would see different memory, and every later prompt would miss.
#: With no stores configured it has no history to extract from in any case.
NO_MODEL_AGENTS = frozenset({"MEMORY"})

#: What the evidence is called in the reply prompt -- the wording
#: benchmarks/adapters/conversational.py established for the same field.
EVIDENCE_SOURCE = "verified analysis results for the connected dataset"

#: Operator rates for a hosted live run, USD per million tokens.
INPUT_RATE_ENV = "AVALOKA_CONV_INPUT_USD_PER_MTOK"
OUTPUT_RATE_ENV = "AVALOKA_CONV_OUTPUT_USD_PER_MTOK"

#: The most a hosted run may spend, in USD at the operator rate. Checked before
#: every call: past it the call is refused, ledgered, and the run fails.
SPEND_CAP_ENV = "AVALOKA_CONV_MAX_USD"
DEFAULT_SPEND_CAP_USD = 1.0

#: Hosted providers a recording or live run may name explicitly, with the key
#: each needs. The default live policy is still local first, then Groq; a
#: provider is used only when asked for. Never Anthropic.
HOSTED_PROVIDERS = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}

#: Replay runs with the recorded provider selected, so the provider-dependent
#: request arguments (tool_choice, say) are the recorded ones. The agents check
#: that a key is present before they build a model; this placeholder satisfies
#: that check and is never sent anywhere: the router is replaced and the
#: replay has no network.
REPLAY_KEY_PLACEHOLDER = "replay-only-no-network"

#: The only variables a child inherits. Everything else -- provider keys,
#: Supabase, Redis, Chroma, Milvus, MCP, telemetry endpoints -- is absent, so
#: memory and telemetry take their no-env paths and a stored key in CI cannot
#: turn the replay into a paid run. Mode-specific settings are added on top.
_ENV_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP",
                    "SYSTEMROOT", "VIRTUAL_ENV", "CONDA_PREFIX", "SSL_CERT_FILE",
                    "REQUESTS_CA_BUNDLE", "HF_HOME", "TRANSFORMERS_CACHE",
                    "SENTENCE_TRANSFORMERS_HOME")

#: Settings every child runs with. Background memory learning starts threads
#: that call a model after the turn has been answered, at a time set by the
#: scheduler: harmless to the reply, fatal to a deterministic recording.
_CHILD_FIXED_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "MEMORY_BACKGROUND_LEARNING": "0",
    "AVALOKA_TELEMETRY": "off",
    "DO_NOT_TRACK": "1",
    "SCARF_NO_ANALYTICS": "true",
    "ORT_DISABLE_TELEMETRY": "1",
    "AWS_EC2_METADATA_DISABLED": "true",
    # The embedding and tokenizer caches are read, never fetched: a download
    # that succeeds while recording and is refused on replay would hand the
    # graph different memory, and every later prompt would miss.
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

#: Text that differs between two runs of the same turn. Masked before hashing
#: so the key is the prompt's meaning, not its incidental identifiers.
_VOLATILE = (
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"), "<time>"),
    (re.compile(r"\btmp[a-z0-9_]{6,}\b", re.I), "<tmp>"),
)


class CassetteMiss(RuntimeError):
    """A request the cassette holds no recording for."""


# --------------------------------------------------------------------------- #
# Request keys
# --------------------------------------------------------------------------- #

def _canonical_text(text: str) -> str:
    for pattern, placeholder in _VOLATILE:
        text = pattern.sub(placeholder, text)
    return text


def _render_message(message: Any) -> dict[str, Any]:
    """One message as the fields a provider would receive."""
    rendered: dict[str, Any] = {
        "type": getattr(message, "type", type(message).__name__),
        "content": getattr(message, "content", message),
    }
    for attr in ("tool_calls", "tool_call_id", "name"):
        value = getattr(message, attr, None)
        if value:
            if attr == "tool_calls":
                value = [{"name": c.get("name"), "args": c.get("args"), "id": c.get("id")}
                         for c in value]
            rendered[attr] = value
    return rendered


def _render_tools(tools: Any) -> Any:
    """Tool definitions in the provider schema, so equal tools hash equally."""
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool
    except Exception:                                 # noqa: BLE001 - hashing still works
        return repr(tools)
    out = []
    for tool in tools or []:
        try:
            out.append(convert_to_openai_tool(tool))
        except Exception:                             # noqa: BLE001 - an exotic tool object
            out.append(repr(tool))
    return out


def request_key(agent: str, messages: Sequence[Any], stop: Optional[Sequence[str]] = None,
                kwargs: Optional[Mapping[str, Any]] = None) -> tuple[str, str]:
    """``(sha256, excerpt)`` of a model request.

    The hash covers the agent, every message, the stop list and the call's
    keyword arguments (tools, tool_choice, response_format, reasoning effort),
    because any of them changes what the model is asked. The excerpt is the
    last message's opening, for a miss report a person can read.
    """
    extra = dict(kwargs or {})
    if "tools" in extra:
        extra["tools"] = _render_tools(extra["tools"])
    document = {
        "agent": agent,
        "messages": [_render_message(m) for m in messages],
        "stop": list(stop or []),
        "kwargs": extra,
    }
    text = _canonical_text(json.dumps(document, sort_keys=True, default=str))
    last = document["messages"][-1]["content"] if document["messages"] else ""
    excerpt = " ".join(str(last).split())[:160]
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), excerpt


# --------------------------------------------------------------------------- #
# The cassette
# --------------------------------------------------------------------------- #

def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _serialise_reply(message: Any) -> dict[str, Any]:
    meta = dict(getattr(message, "response_metadata", None) or {})
    return {
        "content": getattr(message, "content", ""),
        "tool_calls": [{"name": c.get("name"), "args": c.get("args"), "id": c.get("id")}
                       for c in (getattr(message, "tool_calls", None) or [])],
        "additional_kwargs": _jsonable(getattr(message, "additional_kwargs", None) or {}),
        "response_metadata": _jsonable({k: meta[k] for k in ("finish_reason", "model_name", "model")
                                        if k in meta}),
        "usage_metadata": _jsonable(getattr(message, "usage_metadata", None) or {}),
    }


def _generation_id(message: Any) -> Optional[str]:
    """The provider's id for the call that produced *message*, if it gave one."""
    meta = getattr(message, "response_metadata", None) or {}
    for candidate in (meta.get("id"), getattr(message, "id", None)):
        if isinstance(candidate, str) and candidate and not candidate.startswith("run-"):
            return candidate
    return None


def _deserialise_reply(data: Mapping[str, Any]) -> Any:
    from langchain_core.messages import AIMessage

    kwargs: dict[str, Any] = {
        "content": data.get("content", ""),
        "tool_calls": [{"name": c["name"], "args": c.get("args") or {}, "id": c.get("id"),
                        "type": "tool_call"} for c in data.get("tool_calls") or []],
        "additional_kwargs": dict(data.get("additional_kwargs") or {}),
        "response_metadata": dict(data.get("response_metadata") or {}),
    }
    usage = data.get("usage_metadata") or {}
    if {"input_tokens", "output_tokens", "total_tokens"} <= set(usage):
        kwargs["usage_metadata"] = usage
    return AIMessage(**kwargs)


class Cassette:
    """Recorded model replies, and the ledger of what was asked of them.

    Entries are keyed by ``(probe, agent, sha256, n)``: ``n`` counts earlier
    identical requests in the same probe, so a prompt asked twice is answered
    twice in the order it was recorded. ``begin(probe_id)`` scopes what follows
    to one probe.
    """

    def __init__(self, path: Optional[Path], *, recording: bool = False) -> None:
        self.path = Path(path) if path else None
        self.recording = recording
        self.header: dict[str, Any] = {}
        self._entries: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        self._seen: dict[tuple[str, str, str], int] = {}
        self.recorded: list[dict[str, Any]] = []
        self.probe = ""
        #: Per probe: requests answered from the recording (or recorded), and misses.
        self.calls: dict[str, int] = {}
        self.misses: dict[str, list[dict[str, Any]]] = {}
        self.usage: dict[str, dict[str, int]] = {}
        self.models: set[str] = set()
        self.loaded = False
        #: Operator rates and cap, set for a hosted run; spend is tracked
        #: across probes so the cap holds for the whole recording.
        self.rates: Optional[tuple[float, float]] = None
        self.cap_usd: Optional[float] = None
        self.spent_usd = 0.0
        if not recording and self.path is not None and self.path.is_file():
            self._load(self.path)

    def _load(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if "key" not in entry:
                self.header = entry
                continue
            self._entries[(entry["probe"], entry["agent"], entry["key"], int(entry["n"]))] = entry
        self.loaded = True

    def begin(self, probe_id: str) -> None:
        self.probe = probe_id
        self.calls.setdefault(probe_id, 0)
        self.misses.setdefault(probe_id, [])
        self.usage.setdefault(probe_id, {"input_tokens": 0, "output_tokens": 0})

    def _next_n(self, agent: str, key: str) -> int:
        slot = (self.probe, agent, key)
        n = self._seen.get(slot, 0)
        self._seen[slot] = n + 1
        return n

    def serve(self, agent: str, key: str, excerpt: str) -> Any:
        n = self._next_n(agent, key)
        entry = self._entries.get((self.probe, agent, key, n))
        if entry is None:
            # Written before raising: the agent that made this call will
            # probably catch the exception and answer in canned text, and the
            # grade must come from this ledger, not from that text.
            self.misses.setdefault(self.probe, []).append(
                {"agent": agent, "key": key, "n": n, "excerpt": excerpt})
            raise CassetteMiss(
                f"no recording for agent {agent} request {key[:16]} (#{n}) in probe "
                f"{self.probe}: re-record the cassette")
        self.calls[self.probe] = self.calls.get(self.probe, 0) + 1
        return _deserialise_reply(entry["response"])

    def check_budget(self) -> None:
        """Refuse the next hosted call once the cap is spent."""
        if self.cap_usd is not None and self.spent_usd >= self.cap_usd:
            raise RuntimeError(f"spend cap of ${self.cap_usd:.2f} reached "
                               f"(${self.spent_usd:.4f} at the operator rate)")

    def record(self, agent: str, key: str, excerpt: str, reply: Any) -> None:
        n = self._next_n(agent, key)
        data = _serialise_reply(reply)
        model = (data["response_metadata"].get("model_name")
                 or data["response_metadata"].get("model") or "")
        if model:
            self.models.add(str(model))
        usage = data.get("usage_metadata") or {}
        bucket = self.usage.setdefault(self.probe, {"input_tokens": 0, "output_tokens": 0})
        bucket["input_tokens"] += int(usage.get("input_tokens") or 0)
        bucket["output_tokens"] += int(usage.get("output_tokens") or 0)
        if self.rates is not None:
            self.spent_usd += (int(usage.get("input_tokens") or 0) * self.rates[0]
                               + int(usage.get("output_tokens") or 0) * self.rates[1]) / 1_000_000
        self.calls[self.probe] = self.calls.get(self.probe, 0) + 1
        entry = {"probe": self.probe, "agent": agent, "key": key, "n": n,
                 "excerpt": excerpt, "response": data}
        generation = _generation_id(reply)
        if generation:
            # The provider's own id for this call, so its billed cost can be
            # looked up afterwards. Not part of the key.
            entry["generation_id"] = generation
        self.recorded.append(entry)

    def failed(self, agent: str, key: str, excerpt: str, exc: BaseException) -> None:
        """A live call that raised: recorded as a miss, so the run cannot pass."""
        self.misses.setdefault(self.probe, []).append(
            {"agent": agent, "key": key, "n": None, "excerpt": excerpt,
             "error": f"{type(exc).__name__}: {exc}"[:300]})

    def save(self, header: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(header), sort_keys=True) + "\n")
            for entry in self.recorded:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        tmp.replace(self.path)


# --------------------------------------------------------------------------- #
# The chat models that stand in for the router's output
# --------------------------------------------------------------------------- #

def _model_classes() -> tuple[type, type]:
    """``(ReplayChatModel, TapChatModel)``, built on first use.

    Defined lazily because they subclass LangChain's ``BaseChatModel``, which
    the CLI wheel does not ship; this module must import without it.
    """
    global _MODEL_CLASSES
    if _MODEL_CLASSES is not None:
        return _MODEL_CLASSES

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import ConfigDict

    class ReplayChatModel(BaseChatModel):
        """Answers every request from the cassette, or records a miss and raises."""

        model_config = ConfigDict(arbitrary_types_allowed=True)
        agent: str
        cassette: Any

        @property
        def _llm_type(self) -> str:
            return "avaloka-replay"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            key, excerpt = request_key(self.agent, messages, stop, kwargs)
            reply = self.cassette.serve(self.agent, key, excerpt)
            return ChatResult(generations=[ChatGeneration(message=reply)])

        def bind_tools(self, tools, **kwargs):
            return self.bind(tools=tools, **kwargs)

    class TapChatModel(BaseChatModel):
        """Passes every request to the real model and keeps what came back.

        The same class wraps the live and the recording run, so a recorded
        request is built exactly as the replayed one is: same keyword
        arguments, same message objects, same key.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)
        agent: str
        cassette: Any
        inner: Any

        @property
        def _llm_type(self) -> str:
            return "avaloka-tap"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            key, excerpt = request_key(self.agent, messages, stop, kwargs)
            call_kwargs = dict(kwargs)
            if stop:
                call_kwargs["stop"] = stop
            try:
                self.cassette.check_budget()
                reply = self.inner.invoke(messages, **call_kwargs)
            except Exception as exc:
                # The agent will likely swallow this and answer in canned text;
                # the ledger is what says the recording has a hole in it.
                self.cassette.failed(self.agent, key, excerpt, exc)
                raise
            self.cassette.record(self.agent, key, excerpt, reply)
            return ChatResult(generations=[ChatGeneration(message=reply)])

        def bind_tools(self, tools, **kwargs):
            return self.bind(tools=tools, **kwargs)

    _MODEL_CLASSES = (ReplayChatModel, TapChatModel)
    return _MODEL_CLASSES


_MODEL_CLASSES: Optional[tuple[type, type]] = None


def _refuse_anthropic(model: str) -> None:
    lowered = (model or "").lower()
    if "claude" in lowered or "anthropic" in lowered:
        raise RuntimeError(f"refusing model {model!r}: an Anthropic model may not be used "
                           f"for any Avaloka benchmark or test bench")


def _routing_factory(mode: str, cassette: Cassette, real: Callable[..., Any]) -> Callable[..., Any]:
    """The function installed in place of ``build_chat_model``."""
    replay_cls, tap_cls = _model_classes()

    def build_chat_model(*, role: str, agent: str, groq_model: str = "", groq_api_key=None,
                         tier: str = "large", temperature: float = 0.2,
                         env_model_var: Optional[str] = None, **kwargs: Any) -> Any:
        name = (agent or "").upper()
        if name in NO_MODEL_AGENTS:
            return None
        if mode == REPLAY:
            return replay_cls(agent=name, cassette=cassette)
        from app.core import inference
        provider = inference.resolve_provider(role=role, agent=agent, groq_api_key=groq_api_key)
        if provider == inference.GROQ:
            intended = os.getenv(env_model_var or f"AVALOKA_{name}_MODEL", groq_model)
        else:
            intended = inference._resolve_model(provider, agent, tier)
        _refuse_anthropic(intended)
        inner = real(role=role, agent=agent, groq_model=groq_model, groq_api_key=groq_api_key,
                     tier=tier, temperature=temperature, env_model_var=env_model_var, **kwargs)
        if inner is None:
            return None
        return tap_cls(agent=name, cassette=cassette, inner=inner)

    return build_chat_model


# --------------------------------------------------------------------------- #
# The child: one fresh interpreter, every probe through the graph
# --------------------------------------------------------------------------- #

def _block_network(attempts: list[str]) -> None:
    """Refuse every non-local socket connection, and remember each attempt."""
    import errno
    import socket

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _caller() -> str:
        import traceback
        skip = ("socket.py", "ssl.py", "/http/", "urllib3", "requests", "httpx", "httpcore",
                "conversational_product.py", "connection.py", "selectors.py")
        frames = [f for f in traceback.extract_stack()[:-2]
                  if not any(token in f.filename for token in skip)]
        return " < ".join(f"{Path(f.filename).name}:{f.lineno}" for f in frames[-3:][::-1])

    def _note(address: Any) -> None:
        text = repr(address)
        attempts.append(text if _is_loopback(text) else f"{text} via {_caller()}")

    def _allowed(sock: Any) -> bool:
        return getattr(socket, "AF_UNIX", None) is not None and sock.family == socket.AF_UNIX

    def connect(self, address):                       # noqa: ANN001 - socket API
        if _allowed(self):
            return real_connect(self, address)
        _note(address)
        raise OSError(errno.ENETUNREACH, f"network blocked by the product replay: {address!r}")

    def connect_ex(self, address):                    # noqa: ANN001 - socket API
        if _allowed(self):
            return real_connect_ex(self, address)
        _note(address)
        return errno.ENETUNREACH

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex


def _is_loopback(address: str) -> bool:
    return any(token in address for token in ("'127.", "'localhost'", "'::1'", "'0.0.0.0'"))


def _reply_from_state(final: Mapping[str, Any]) -> str:
    """Every assistant message of this turn, in order, as the chat shows them.

    ``server._assistant_messages_from_state`` turns each AIMessage into its
    segments' text, and the chat response carries every assistant message,
    not only the last: a delegated turn says what it is about to do, then what
    it did, and the user reads both. Grading the last alone graded the result
    and never the narration before it. The turn's messages are the ones after
    the user's latest message.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from app.api import reply_segments

    messages = list(final.get("messages") or [])
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            start = i + 1
            break
    said = [reply_segments.text_of(reply_segments.of_message(m))
            for m in messages[start:] if isinstance(m, AIMessage)]
    return "\n\n".join(t for t in said if t.strip())


def _initial_state(probe: Any) -> dict[str, Any]:
    from langchain_core.messages import AIMessage, HumanMessage

    turns = list(probe.turns)
    return {
        "user_id": "bench",
        "session_id": f"conv-product-{probe.id}",
        "user_prompt": turns[-1].text if turns else "",
        "messages": [HumanMessage(content=t.text) if t.role == "user" else AIMessage(content=t.text)
                     for t in turns],
        "avaloka_recommendation": {"source": EVIDENCE_SOURCE, "results": dict(probe.evidence)},
        "execution_context": {"benchmark_evidence": dict(probe.evidence)},
    }


def _run_one(graph: Any, probe: Any, budget_s: float) -> dict[str, Any]:
    state = _initial_state(probe)
    config = {"recursion_limit": RECURSION_LIMIT,
              "configurable": {"thread_id": f"conv-product-{probe.id}"}}
    nodes: list[str] = []
    final: Mapping[str, Any] = state
    error: Optional[str] = None
    completed = False
    started = time.perf_counter()
    try:
        for mode, chunk in graph.stream(state, config, stream_mode=["updates", "values"]):
            if mode == "updates" and isinstance(chunk, Mapping):
                nodes.extend(str(k) for k in chunk.keys())
            elif mode == "values" and isinstance(chunk, Mapping):
                final = chunk
            if time.perf_counter() - started > budget_s:
                error = (f"wall-clock budget of {budget_s:.0f}s exceeded after "
                         f"{' -> '.join(nodes) or 'no node'}")
                break
        else:
            completed = True
    except Exception as exc:                          # noqa: BLE001 - graded, with the trace
        name = type(exc).__name__
        if name == "GraphRecursionError":
            error = f"recursion_limit {RECURSION_LIMIT} exceeded after {' -> '.join(nodes)}"
        else:
            error = f"{name}: {exc}"
    try:
        reply = _reply_from_state(final)
    except Exception as exc:                          # noqa: BLE001
        reply = ""
        error = error or f"reply could not be read from state: {type(exc).__name__}: {exc}"
    return {
        "probe_id": probe.id,
        "sent_question": state["user_prompt"],
        "reply": reply,
        "nodes": nodes,
        "completed": completed and error is None,
        "error": error,
        "latency_s": round(time.perf_counter() - started, 3),
        "intent": final.get("avaloka_intent"),
        "delegated": bool(final.get("delegate_to_planner")),
        "handoff": final.get("avaloka_handoff_message"),
    }


def _load_probes(path: str) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("_avaloka_product_probes", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _child(args: argparse.Namespace) -> int:
    out = Path(args.out)
    document: dict[str, Any] = {"mode": args.mode, "records": {}, "egress": [],
                                "unavailable": None, "error": None}

    def flush() -> None:
        out.write_text(json.dumps(document, indent=1, default=str), encoding="utf-8")

    attempts: list[str] = []
    if args.mode == REPLAY:
        _block_network(attempts)
    cassette = Cassette(Path(args.cassette) if args.cassette else None,
                        recording=args.mode == RECORD)
    if args.mode != REPLAY and _rate(INPUT_RATE_ENV) is not None and _rate(OUTPUT_RATE_ENV) is not None:
        cassette.rates = (_rate(INPUT_RATE_ENV), _rate(OUTPUT_RATE_ENV))
        cassette.cap_usd = _rate(SPEND_CAP_ENV) or DEFAULT_SPEND_CAP_USD
    if args.mode == REPLAY:
        document["cassette"] = {"path": args.cassette, "loaded": cassette.loaded,
                                "entries": len(cassette._entries), "header": cassette.header}

    probes = _load_probes(args.probes)
    wanted = [p for p in probes.PROBES if not args.probe_ids or p.id in args.probe_ids]

    started = time.perf_counter()
    try:
        from app.core import inference
        real = inference.build_chat_model
        factory = _routing_factory(args.mode, cassette, real)
    except ImportError as exc:
        document["unavailable"] = f"unavailable: app not installed ({type(exc).__name__}: {exc})"
        flush()
        return 0
    if "app.api.workflow" in sys.modules:             # pragma: no cover - a fresh process
        document["error"] = "app.api.workflow was imported before the router was replaced"
        flush()
        return 3
    inference.build_chat_model = factory
    try:
        from app.api.workflow import build_graph
        graph = build_graph(checkpointer=None).compile()
    except ImportError as exc:
        document["unavailable"] = f"unavailable: app not installed ({type(exc).__name__}: {exc})"
        flush()
        return 0
    except Exception as exc:                          # noqa: BLE001 - a broken graph is a failure
        document["error"] = f"build_graph failed: {type(exc).__name__}: {exc}"
        flush()
        return 3
    document["import_s"] = round(time.perf_counter() - started, 1)
    document["routing_replaced"] = inference.build_chat_model is not real
    flush()

    budget = float(args.budget or PROBE_BUDGET_S[args.mode])
    for probe in wanted:
        cassette.begin(probe.id)
        record = _run_one(graph, probe, budget)
        record["calls"] = cassette.calls.get(probe.id, 0)
        record["misses"] = cassette.misses.get(probe.id, [])
        record["usage"] = cassette.usage.get(probe.id, {})
        document["records"][probe.id] = record
        document["egress"] = list(attempts)
        flush()

    if args.mode == RECORD:
        header = {"cassette": "avaloka.benchmark.conversational_product", "version": 1,
                  "provider": os.environ.get("INFERENCE_PROVIDER", "local"),
                  "models": sorted(cassette.models), "probes": [p.id for p in wanted],
                  "calls": len(cassette.recorded),
                  "recorded_on": time.strftime("%Y-%m-%d", time.gmtime())}
        cassette.save(header)
        document["cassette"] = {"path": args.cassette, **header}
    document["models"] = sorted(cassette.models)
    document["spent_usd"] = round(cassette.spent_usd, 6)
    document["generation_ids"] = [e["generation_id"] for e in cassette.recorded
                                  if e.get("generation_id")]
    flush()
    return 0


# --------------------------------------------------------------------------- #
# The parent: spawn the child, read what it measured
# --------------------------------------------------------------------------- #

@dataclass
class ProbeRecord:
    """One probe as it went through the graph."""

    probe_id: str
    sent_question: str
    reply: str
    nodes: list[str] = field(default_factory=list)
    completed: bool = False
    error: Optional[str] = None
    latency_s: float = 0.0
    calls: int = 0
    misses: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    intent: Optional[str] = None
    delegated: bool = False
    handoff: Optional[str] = None

    @property
    def through_graph(self) -> bool:
        """The compiled graph ran this probe from its entry to its end."""
        return self.completed and "avaloka_agent" in self.nodes and "verify_claims" in self.nodes

    @property
    def path(self) -> str:
        return " -> ".join(self.nodes) or "<no node ran>"


@dataclass
class ProductRun:
    """Everything a product run measured, or why it measured nothing."""

    mode: str
    label: str
    records: dict[str, ProbeRecord] = field(default_factory=dict)
    unavailable: Optional[str] = None
    error: Optional[str] = None
    egress: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    cassette: dict[str, Any] = field(default_factory=dict)
    spend_usd: float = 0.0
    results_written: bool = False
    routing_replaced: bool = False
    #: The provider's id for each recorded call, for looking up billed cost.
    generation_ids: list[str] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return sum(r.calls for r in self.records.values())

    @property
    def misses(self) -> list[dict[str, Any]]:
        return [dict(m, probe=r.probe_id) for r in self.records.values() for m in r.misses]

    @property
    def external_egress(self) -> list[str]:
        return [a for a in self.egress if not _is_loopback(a)]


def repo_root_for(probes_file: str | Path) -> Path:
    """``tests/conversational/probes.py`` -> the checkout it sits in."""
    return Path(probes_file).resolve().parents[2]


def default_cassette(probes_file: str | Path) -> Path:
    return repo_root_for(probes_file) / CASSETTE_RELPATH


def _child_env(root: Path, extra: Mapping[str, str]) -> dict[str, str]:
    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    env.update(_CHILD_FIXED_ENV)
    existing = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) + (os.pathsep + existing if existing else "")
    env.update(extra)
    return env


def _rate(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _priced_env() -> dict[str, str]:
    if _rate(INPUT_RATE_ENV) is None or _rate(OUTPUT_RATE_ENV) is None:
        raise RuntimeError(f"a hosted run must be priced: set {INPUT_RATE_ENV} and "
                           f"{OUTPUT_RATE_ENV} (USD per million tokens)")
    env = {INPUT_RATE_ENV: os.environ[INPUT_RATE_ENV], OUTPUT_RATE_ENV: os.environ[OUTPUT_RATE_ENV]}
    if os.environ.get(SPEND_CAP_ENV, "").strip():
        env[SPEND_CAP_ENV] = os.environ[SPEND_CAP_ENV]
    return env


def live_settings(model: Optional[str],
                  provider: Optional[str] = None) -> tuple[dict[str, str], str, bool]:
    """``(child env, model label, hosted)`` under the policy: local, then Groq.

    ``provider`` names a hosted provider outright (``openrouter`` or ``groq``):
    its models are then the router's own defaults for it, so the run is of
    what the product ships. Raises ``RuntimeError`` when nothing is usable, or
    a hosted model would run with no operator rate to price it.
    """
    from avaloka.benchmark import conversational as conv

    if model:
        _refuse_anthropic(model)
    if provider:
        provider = provider.lower()
        if provider not in HOSTED_PROVIDERS:
            raise RuntimeError(f"provider must be one of {sorted(HOSTED_PROVIDERS)}, not {provider!r}")
        key_var = HOSTED_PROVIDERS[provider]
        key = os.environ.get(key_var, "").strip()
        if not key:
            raise RuntimeError(f"{key_var} is not set")
        env = {"INFERENCE_PROVIDER": provider, key_var: key, **_priced_env()}
        if model:
            for agent in _AGENTS:
                env[f"AVALOKA_{agent}_MODEL_{provider.upper()}"] = model
                env[f"AVALOKA_{agent}_MODEL"] = model
        return env, f"{provider}:{model or 'router-defaults'}", True
    local = conv._local_model(model)
    if local:
        return ({"INFERENCE_PROVIDER": "local",
                 "INFERENCE_LOCAL_BASE_URL": f"{conv._ollama_host()}/v1",
                 "INFERENCE_LOCAL_MODEL": local}, local, False)
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("no local model answered and GROQ_API_KEY is not set")
    priced = _priced_env()
    hosted = model or conv.GROQ_MODEL
    _refuse_anthropic(hosted)
    env = {"INFERENCE_PROVIDER": "groq", "GROQ_API_KEY": key, **priced}
    if model:
        # Every agent's Groq model, so the run is of the model named.
        for agent in _AGENTS:
            env[f"AVALOKA_{agent}_MODEL"] = model
    return env, hosted, True


_AGENTS = ("CONVERSATIONAL", "INTENT_CLASSIFIER", "PLANNER", "CODER", "VALIDATOR",
           "SUMMARIZER", "NARRATOR", "PROFILING", "VISUALIZATION")


def run_product(probes_file: str | Path, probe_ids: Iterable[str] = (), *, mode: str = REPLAY,
                model: Optional[str] = None, cassette: Optional[Path] = None,
                budget_s: Optional[float] = None, timeout_s: Optional[float] = None,
                python: Optional[str] = None, provider: Optional[str] = None) -> ProductRun:
    """Run the probes through the compiled graph in a child interpreter.

    Never raises for a product fault: an unavailable application, a crashed
    child or a timeout comes back on the :class:`ProductRun` for the grader
    to report.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    probes_file = Path(probes_file).resolve()
    root = repo_root_for(probes_file)
    ids = list(probe_ids)
    hosted = False
    extra: dict[str, str] = {}
    label = f"product:{REPLAY}"
    if mode == REPLAY:
        # The provider the recording was made under, so provider-dependent
        # request arguments (tool_choice, say) are the recorded ones. No
        # endpoint is set: the router is replaced and nothing may be reached.
        cassette = cassette or default_cassette(probes_file)
        recorded_under = str(Cassette(cassette).header.get("provider") or "local")
        extra["INFERENCE_PROVIDER"] = recorded_under
        if recorded_under in HOSTED_PROVIDERS:
            extra[HOSTED_PROVIDERS[recorded_under]] = REPLAY_KEY_PLACEHOLDER
    else:
        try:
            extra, resolved, hosted = live_settings(model, provider)
        except RuntimeError as exc:
            return ProductRun(mode=mode, label=f"product:{mode}",
                              unavailable=f"unavailable: live provider: {exc}")
        if mode == RECORD:
            if hosted and not provider:
                # A hosted recording is a deliberate, priced act: it must name
                # its provider rather than fall through to one.
                return ProductRun(mode=mode, label=f"product:{mode}",
                                  unavailable="unavailable: --record falls back to no hosted "
                                              "model; name one with --provider")
            cassette = cassette or default_cassette(probes_file)
        label = f"product:{'record' if mode == RECORD else 'live'}:{resolved}"

    n_probes = len(ids) or len(_load_probes(str(probes_file)).PROBES)
    budget = float(budget_s or PROBE_BUDGET_S[mode])
    timeout = timeout_s or (IMPORT_BUDGET_S + budget * n_probes + 60)
    with tempfile.TemporaryDirectory(prefix="avaloka-conv-product-") as tmp:
        out = Path(tmp) / "results.json"
        command = [python or sys.executable, "-m", "avaloka.benchmark.conversational_product",
                   "child", "--mode", mode, "--out", str(out), "--probes", str(probes_file),
                   "--budget", str(budget)]
        if cassette is not None:
            command += ["--cassette", str(cassette)]
        for probe_id in ids:
            command += ["--probe", probe_id]
        error: Optional[str] = None
        try:
            # cwd is the scratch directory, not the checkout: a node that writes
            # an artifact relative to cwd must not dirty the tree under test.
            done = subprocess.run(command, cwd=tmp, env=_child_env(root, extra),
                                  capture_output=True, text=True, timeout=timeout)
            if done.returncode != 0:
                error = (f"the graph run exited {done.returncode}: "
                         + " ".join((done.stderr or done.stdout or "").split())[-600:])
        except subprocess.TimeoutExpired:
            error = f"the graph run did not finish within {timeout:.0f}s"
        if not out.is_file():
            return ProductRun(mode=mode, label=label,
                              error=error or "the graph run wrote no results")
        document = json.loads(out.read_text(encoding="utf-8"))

    run = ProductRun(mode=mode, label=label, unavailable=document.get("unavailable"),
                     error=error or document.get("error"), egress=list(document.get("egress") or []),
                     models=list(document.get("models") or []),
                     cassette=dict(document.get("cassette") or {}), results_written=True,
                     routing_replaced=bool(document.get("routing_replaced")))
    for probe_id, raw in (document.get("records") or {}).items():
        run.records[probe_id] = ProbeRecord(**{k: raw.get(k) for k in ProbeRecord.__dataclass_fields__
                                               if k in raw})
    if hosted:
        rate_in, rate_out = _rate(INPUT_RATE_ENV) or 0.0, _rate(OUTPUT_RATE_ENV) or 0.0
        tokens_in = sum(int(r.usage.get("input_tokens") or 0) for r in run.records.values())
        tokens_out = sum(int(r.usage.get("output_tokens") or 0) for r in run.records.values())
        run.spend_usd = (tokens_in * rate_in + tokens_out * rate_out) / 1_000_000
    run.generation_ids = list(document.get("generation_ids") or [])
    return run


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m avaloka.benchmark.conversational_product")
    sub = parser.add_subparsers(dest="cmd", required=True)

    child = sub.add_parser("child", help=argparse.SUPPRESS)
    child.add_argument("--mode", choices=MODES, required=True)
    child.add_argument("--out", required=True)
    child.add_argument("--probes", required=True)
    child.add_argument("--cassette", default=None)
    child.add_argument("--budget", type=float, default=None)
    child.add_argument("--probe", dest="probe_ids", action="append", default=[])

    for name, text in ((RECORD, "Re-record the cassette (local model, or --provider)."),
                       (LIVE, "Run the probes through the graph on a live model."),
                       (REPLAY, "Replay the cassette through the graph.")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--model", default=None)
        p.add_argument("--probe", dest="probe_ids", action="append", default=[])
        p.add_argument("--json", default=None, help="Write the run here.")
        p.add_argument("--provider", default=None, choices=sorted(HOSTED_PROVIDERS),
                       help="Name a hosted provider; its router defaults are the models.")
        p.add_argument("--cassette", default=None,
                       help=f"Cassette path (default: <repo>/{CASSETTE_RELPATH}).")
    args = parser.parse_args(argv)

    if args.cmd == "child":
        return _child(args)

    from avaloka.benchmark.conversational import load_probe_suite
    suite = load_probe_suite()
    run = run_product(suite.module.__file__, args.probe_ids, mode=args.cmd, model=args.model,
                      cassette=Path(args.cassette) if args.cassette else None,
                      provider=getattr(args, "provider", None))
    summary = {"label": run.label, "unavailable": run.unavailable, "error": run.error,
               "calls": run.calls, "misses": run.misses, "egress": run.egress,
               "models": run.models, "spend_usd": round(run.spend_usd, 6),
               "generation_ids": run.generation_ids,
               "cassette": run.cassette,
               "records": {k: vars(v) for k, v in run.records.items()}}
    text = json.dumps(summary, indent=1, default=str)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    for record in run.records.values():
        print(f"{record.probe_id:28s} calls={record.calls:2d} misses={len(record.misses)} "
              f"{record.latency_s:7.1f}s  {record.path}")
        print(f"    reply: {' '.join(record.reply.split())[:200]!r}")
        if record.error:
            print(f"    error: {record.error}")
    print(f"{run.label}: {run.calls} model calls, {len(run.misses)} misses, "
          f"spend ${run.spend_usd:.4f}, models {run.models}")
    if run.unavailable or run.error:
        print(run.unavailable or run.error)
        return 1
    return 0 if not run.misses else 1


if __name__ == "__main__":
    raise SystemExit(_main())
