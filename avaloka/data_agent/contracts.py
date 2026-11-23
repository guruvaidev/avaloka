"""Provider-, tool- and benchmark-independent data reasoning contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class MissionLimits:
    max_cost_usd: Optional[float] = None
    max_latency_s: Optional[float] = None
    max_tokens: Optional[int] = None
    max_data_scanned_ratio: Optional[float] = None


@dataclass(frozen=True)
class MissionPolicy:
    max_iterations: int = 25
    tool_timeout_s: int = 120
    max_output_tokens: int = 8_192
    max_prompt_tool_result_bytes: int = 100_000
    # Bound on all tool results in the prompt together. The per-result cap
    # alone did not hold: on DataAgentBench rc7, gpt-oss-120b hit Groq's
    # 131k-token context_length_exceeded twice (DEPS_DEV_V1:2, PATENTS:3)
    # with every oversized result already cut to 100 KB -- three of them is
    # ~110k tokens. Past this total the oldest results are compacted to
    # ``compacted_tool_result_bytes`` (head/tail plus a note of what was cut).
    # 0 or less disables the bound.
    max_prompt_tool_results_total_bytes: int = 240_000
    compacted_tool_result_bytes: int = 2_000
    repeated_call_limit: int = 3
    # When one call is repeated ``repeated_call_limit`` times, the mission
    # first blocks it and asks the model to re-plan with a summary of the
    # recent tool errors (a strategy switch); only a repeat after that many
    # switches ends the mission as repeated_tool_call. 0 restores the old
    # stop-at-the-third-repeat behaviour.
    max_strategy_switches: int = 1
    max_tool_errors: int = 8
    # How many times one mission may re-prompt after the provider itself
    # rejected the model's tool call (ToolCallRejected) before the mission
    # ends as provider_error. Bounded: a model that cannot name a real tool
    # twice in a row is not going to on the fifth try.
    max_tool_call_rejections: int = 2
    require_data_evidence: bool = True
    enable_specialists: bool = True
    max_review_repairs: int = 1


@dataclass(frozen=True)
class DataMissionRequest:
    task_id: str
    question: Any
    database_description: str
    model: str
    seed: int = 0
    source_bytes: Optional[int] = None
    limits: MissionLimits = field(default_factory=MissionLimits)
    policy: MissionPolicy = field(default_factory=MissionPolicy)


@dataclass(frozen=True)
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Opaque provider protocol state replayed with the assistant tool call.
    # It is intentionally excluded from public evidence trajectories.
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    response_id: Optional[str]
    content: Optional[str]
    tool_calls: list[ModelToolCall]
    usage: ModelUsage = field(default_factory=ModelUsage)


@dataclass(frozen=True)
class ToolObservation:
    success: bool
    content: str
    result_key: Optional[str] = None
    rows_read: Optional[int] = None
    bytes_read: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceRecord:
    tool_call_id: str
    tool: str
    result_key: Optional[str]
    rows_read: Optional[int]
    bytes_read: Optional[int]
    content_sha256: str
    content_truncated: bool


@dataclass
class DataMissionResult:
    answer: Optional[str]
    latency_s: float
    cost_usd: Optional[float]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_calls: int
    successful_tool_calls: int
    retries: int
    iterations: int
    rows_read: int
    bytes_read: int
    data_scanned_ratio: Optional[float]
    evidence: list[EvidenceRecord]
    trajectory: list[dict[str, Any]]
    termination_reason: str
    error: Optional[str] = None
    error_kind: Optional[str] = None
    question_plan: Optional[dict[str, Any]] = None
    source_map: Optional[dict[str, Any]] = None
    specialists: tuple[str, ...] = ()
    review_repairs: int = 0
    review_status: Optional[str] = None


class ProviderError(RuntimeError):
    """A hosted or local model provider failed after bounded retries."""


class ToolCallRejected(ProviderError):
    """The provider could not turn the model's reply into a tool call.

    Groq answers HTTP 400 ``code=tool_use_failed`` when its parser fails on
    the generation -- a tool that does not exist (a phantom "json" tool),
    malformed arguments -- and ``failed_generation`` carries what the model
    wrote. The request was well formed, so sending it again unchanged repeats
    the failure; the mission re-prompts with an instruction instead
    (bounded by ``MissionPolicy.max_tool_call_rejections``), and only then
    ends as ``provider_error``. ``failed_generation`` is for logs: it is the
    model's text and may quote the data, so it never enters a trajectory.
    """

    def __init__(self, message: str, *, failed_generation: str = "") -> None:
        super().__init__(message)
        self.failed_generation = failed_generation


def tool_use_failed_text(exc: BaseException) -> Optional[str]:
    """The ``failed_generation`` of a provider's tool_use_failed error, else None.

    Reads the body the OpenAI-compatible SDKs attach (``exc.body``, either the
    bare error dict or ``{"error": {...}}``). Pure: no SDK import, so the CLI
    data agent keeps its dependency footprint.
    """
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else body
    if isinstance(err, dict) and err.get("code") == "tool_use_failed":
        return str(err.get("failed_generation") or "")
    if getattr(exc, "code", None) == "tool_use_failed":
        return ""
    return None


class ModelProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout_s: float,
    ) -> ModelResponse:
        ...


class ToolRegistry(Protocol):
    def specs(self) -> list[dict[str, Any]]:
        ...

    def execute(
        self, name: str, arguments: dict[str, Any], *, call_id: str
    ) -> ToolObservation:
        ...

    def close(self) -> None:
        ...
