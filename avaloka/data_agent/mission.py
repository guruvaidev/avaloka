"""A bounded plan/act/inspect loop owned by Avaloka.

The mission is deliberately independent of any benchmark harness. It accepts a
model provider and read-only tool registry, maintains safe evidence references,
and stops deterministically when progress, cost, latency, token or scan budgets
are exhausted.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections import Counter
from dataclasses import replace
from typing import Any

from .contracts import (DataMissionRequest, DataMissionResult, EvidenceRecord,
                        ToolCallRejected,
                        ModelProvider, ModelResponse, ModelToolCall, ModelUsage,
                        ProviderError, ToolObservation, ToolRegistry)
from .pricing import PricingPolicy
from .specialists import (build_question_plan, build_source_map,
                          review_answer, route_specialists,
                          specialist_context)

PROMPT_VERSION = "avaloka-data-reasoning-v4-specialists"

RETURN_ANSWER_SPEC = {
    "type": "function",
    "function": {
        "name": "return_answer",
        "description": (
            "Return the final evidence-supported answer and stop. Evidence refs "
            "are successful data tool-call ids used to support the answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "evidence_refs": {
                    "type": "array", "items": {"type": "string"}
                },
            },
            "required": ["answer"],
        },
    },
}

#: Tool names accepted as a misnamed return_answer -- only when the arguments
#: are answer-shaped (a non-empty string "answer", optionally "evidence_refs",
#: nothing else). "final_answer", "submit", "submit_answer" and "answer" are
#: the finish tools models are trained on elsewhere. "commentary" and "json"
#: are gpt-oss's Harmony channel / constraint names leaking into the tool name:
#: on DataAgentBench rc7 they were 10 of Groq's 11 tool_use_failed rejections,
#: every one carrying {"answer": ..., "evidence_refs": [...]}, and two of those
#: answers were the ground truth. Any other name (the eleventh was
#: "repo_browser.search"), or any of these with other arguments, is not an
#: answer and keeps the existing handling.
ANSWER_TOOL_ALIASES = frozenset({
    "final_answer", "submit", "submit_answer", "answer", "commentary", "json",
})
_ANSWER_ARGUMENT_KEYS = frozenset({"answer", "evidence_refs"})


def answer_alias_arguments(name: Any, arguments: Any) -> dict[str, Any] | None:
    """The return_answer arguments of a misnamed answer call, else None."""
    if not isinstance(name, str) or name.strip().casefold() not in ANSWER_TOOL_ALIASES:
        return None
    if not isinstance(arguments, dict) or not arguments:
        return None
    if not set(arguments) <= _ANSWER_ARGUMENT_KEYS:
        return None
    answer = arguments.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    refs = arguments.get("evidence_refs")
    if refs is not None and not (
        isinstance(refs, list) and all(isinstance(ref, str) for ref in refs)
    ):
        return None
    return dict(arguments)


def answer_alias_from_failed_generation(text: str) -> tuple[str, dict[str, Any]] | None:
    """(tool name, arguments) when a rejected generation was a misnamed answer.

    Groq's ``failed_generation`` is the model's call as JSON:
    ``{"name": "commentary", "arguments": {"answer": ...}}``.
    """
    try:
        payload = json.loads((text or "").strip())
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            return None
    accepted = answer_alias_arguments(payload.get("name"), arguments)
    if accepted is None:
        return None
    return str(payload["name"]), accepted


SYSTEM_PROMPT = """
You are Avaloka's data reasoning mission. Answer the user's question only from
the supplied sources and available read-only tools.

Work evidence-first:
- Decompose every requested fact, population, filter, time period, geography,
  unit and output format.
- Inspect schemas before assuming table, collection, field or join names. For
  MongoDB/JSON, inspect several records, use exact dotted paths, distinguish
  missing from null, and unwind arrays only when the requested grain requires it.
- For multi-source work, normalize join-key types and verify cardinality and
  unmatched counts before aggregation. Never silently multiply rows.
- Push filters and aggregation to the data system when safe, retrieving only
  the evidence needed.
- When a requested attribute is not stored but can be derived from text, treat
  it as a semantic classification task. Define the criterion from the question
  and source description, retrieve bounded batches, keep an auditable running
  tally, and review ambiguous cases. Do not repeatedly search for a nonexistent
  label, install packages, or use the internet when the supplied text is enough.
- Assign each record by its primary subject using mutually exclusive categories;
  a keyword mention or secondary subject is not enough. Use batches of roughly
  10-15 typical text records, make sure every record is fully visible, and never
  extrapolate an exact count from a preview or truncated batch.
- If a result is file-backed or truncated, use the available read-only code
  tool to inspect or slice it into bounded JSON batches. Follow that tool's
  machine-readable output contract exactly; serialize rather than printing a
  language-specific representation. Apply deterministic filters, joins, sorts,
  and candidate reduction before semantic review whenever they preserve recall.
- Verify denominators, distinct-vs-row counts, null handling, units, date
  boundaries, grouping level and rounding.
- When SQL or Mongo syntax fails, use the error plus observed schema to make one
  targeted repair. Do not repeat an equivalent call.
- Never invent a column, result, citation or successful tool execution.
- Stop exploring once the numerator, denominator, and requested calculation are
  supported. Prefer a timely, qualified answer over redundant environment or
  package probes.
- Before finishing, check that every part of the question is supported by a
  successful tool observation. Use return_answer exactly once and cite its tool
  call ids in evidence_refs when possible.
- Do not expose private chain-of-thought. Return a concise result and only the
  qualifications needed to interpret it.
""".strip()


class DataQueryMission:
    """Run one data mission with hard progress and resource stopping rules."""

    def __init__(
        self,
        provider: ModelProvider,
        tools: ToolRegistry,
        pricing: PricingPolicy | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.pricing = pricing or PricingPolicy()

    def run(self, request: DataMissionRequest) -> DataMissionResult:
        started = time.perf_counter()
        question_plan = build_question_plan(request.question, request.database_description)
        source_map = build_source_map(request.database_description)
        specialists = (
            route_specialists(question_plan, source_map)
            if request.policy.enable_specialists else ()
        )
        specialist_payload = specialist_context(question_plan, source_map, specialists)
        deadline = (
            started + request.limits.max_latency_s
            if request.limits.max_latency_s is not None else None
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._user_message(request, specialist_payload)},
        ]
        trajectory: list[dict[str, Any]] = [{
            "kind": "mission_start",
            "task_id": request.task_id,
            "prompt_version": PROMPT_VERSION,
            "model": request.model,
            "limits": request.limits.__dict__,
            "policy": request.policy.__dict__,
        }]
        trajectory.extend([
            {"kind": "question_plan_created", "plan": question_plan.as_dict()},
            {"kind": "source_map_created", "source_map": source_map.as_dict()},
            {"kind": "specialist_routed", "specialists": list(specialists)},
        ])
        evidence: list[EvidenceRecord] = []
        prompt_tokens = completion_tokens = total_tokens = 0
        total_cost: float | None = 0.0 if self.pricing.configured else None
        tool_calls = successful_tool_calls = retries = tool_errors = 0
        tool_call_rejections = 0
        rows_read = bytes_read = 0
        signatures: Counter[str] = Counter()
        strategy_switches = 0
        recent_failures: list[str] = []
        compacted_tool_ids: set[str] = set()
        tool_result_keys: dict[str, str] = {}
        guidance_sent: set[str] = set()
        review_repairs = 0
        review_status: str | None = None

        try:
            if (
                request.limits.max_cost_usd is not None
                and not self.pricing.configured
            ):
                return self._result(
                    request, started, total_cost, prompt_tokens,
                    completion_tokens, total_tokens, tool_calls,
                    successful_tool_calls, retries, 0, rows_read, bytes_read,
                    evidence, trajectory, "pricing_unconfigured",
                    error=("pricing_unconfigured: max_cost_usd requires an "
                           "explicit PricingPolicy"),
                    error_kind="pricing_unconfigured",
                )

            for iteration in range(1, request.policy.max_iterations + 1):
                violation = self._budget_violation(
                    request, started, deadline, total_tokens, total_cost, bytes_read
                )
                if violation:
                    return self._result(
                        request, started, total_cost, prompt_tokens,
                        completion_tokens, total_tokens, tool_calls,
                        successful_tool_calls, retries, iteration - 1, rows_read,
                        bytes_read, evidence, trajectory, "budget_exceeded",
                        error=f"budget_exceeded: {violation}",
                        error_kind="budget_exceeded",
                    )

                guidance = self._progress_guidance(
                    request,
                    iteration=iteration,
                    total_cost=total_cost,
                    tool_errors=tool_errors,
                    successful_tool_calls=successful_tool_calls,
                    sent=guidance_sent,
                )
                if guidance:
                    messages.append({"role": "user", "content": guidance})
                    trajectory.append({
                        "kind": "progress_guidance",
                        "iteration": iteration,
                        "reasons": sorted(guidance_sent),
                    })

                compaction = self._compact_tool_history(
                    messages, request.policy, compacted_tool_ids, tool_result_keys
                )
                if compaction:
                    trajectory.append({
                        "kind": "tool_history_compacted",
                        "iteration": iteration,
                        **compaction,
                    })

                timeout = self._remaining_timeout(
                    deadline, request.policy.tool_timeout_s
                )
                tool_specs = self.tools.specs()
                try:
                    response = self.provider.complete(
                        messages,
                        [*tool_specs, RETURN_ANSWER_SPEC],
                        timeout_s=timeout,
                    )
                except ToolCallRejected as exc:
                    recovered = answer_alias_from_failed_generation(
                        exc.failed_generation
                    )
                    if recovered is None:
                        # The provider's parser failed on the model's tool call (Groq
                        # 400 tool_use_failed: a phantom tool, malformed arguments).
                        # DataAgentBench on integration/2.0 lost 17 of 54 queries to
                        # this with no retry. The request was fine, so retrying it
                        # unchanged repeats the failure; tell the model what went
                        # wrong and let it try again, a bounded number of times.
                        tool_call_rejections += 1
                        trajectory.append({
                            "kind": "tool_call_rejected",
                            "iteration": iteration,
                            "count": tool_call_rejections,
                        })
                        if tool_call_rejections > request.policy.max_tool_call_rejections:
                            raise
                        retries += 1  # a re-prompt that happens, not one refused
                        messages.append({
                            "role": "user",
                            "content": (
                                "Your previous reply could not be executed: the "
                                "provider could not parse it as a tool call. Call "
                                "exactly one of the provided tools by its exact name "
                                "with JSON arguments that match its schema, or call "
                                "return_answer with your final answer. Do not invent "
                                "tools and do not reply in prose."
                            ),
                        })
                        continue
                    # The provider refused a call to a tool that does not exist,
                    # but the call was the answer (see ANSWER_TOOL_ALIASES).
                    # Process it as the return_answer it meant: the evidence and
                    # review checks below still apply. The provider reported no
                    # usage for the refused generation.
                    alias, arguments = recovered
                    trajectory.append({
                        "kind": "answer_tool_alias",
                        "iteration": iteration,
                        "tool": alias,
                        "path": "provider_rejection",
                    })
                    response = ModelResponse(
                        response_id=None, content=None,
                        tool_calls=[ModelToolCall(
                            f"avaloka-recovered-{iteration}",
                            "return_answer", arguments,
                        )],
                        usage=ModelUsage(),
                    )
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens
                total_tokens += response.usage.total_tokens
                call_cost = self.pricing.cost(response.usage)
                if total_cost is not None:
                    total_cost += call_cost or 0.0
                trajectory.append(self._model_event(iteration, response, call_cost))
                response = self._normalize_answer_aliases(
                    response, tool_specs, trajectory, iteration
                )

                if (
                    request.policy.max_output_tokens >= 0
                    and response.usage.completion_tokens
                    > request.policy.max_output_tokens
                ):
                    return self._result(
                        request, started, total_cost, prompt_tokens,
                        completion_tokens, total_tokens, tool_calls,
                        successful_tool_calls, retries, iteration, rows_read,
                        bytes_read, evidence, trajectory, "budget_exceeded",
                        error=("budget_exceeded: completion tokens "
                               f"{response.usage.completion_tokens} exceeded "
                               f"{request.policy.max_output_tokens}"),
                        error_kind="budget_exceeded",
                    )

                violation = self._budget_violation(
                    request, started, deadline, total_tokens, total_cost, bytes_read,
                    strict=True,
                )
                if violation:
                    return self._result(
                        request, started, total_cost, prompt_tokens,
                        completion_tokens, total_tokens, tool_calls,
                        successful_tool_calls, retries, iteration, rows_read,
                        bytes_read, evidence, trajectory, "budget_exceeded",
                        error=f"budget_exceeded: {violation}",
                        error_kind="budget_exceeded",
                    )

                if not response.tool_calls:
                    answer = (response.content or "").strip()
                    if answer and (
                        successful_tool_calls > 0
                        or not request.policy.require_data_evidence
                    ):
                        trajectory.append({
                            "kind": "termination", "reason": "content_answer"
                        })
                        return self._result(
                            request, started, total_cost, prompt_tokens,
                            completion_tokens, total_tokens, tool_calls,
                            successful_tool_calls, retries, iteration, rows_read,
                            bytes_read, evidence, trajectory, "content_answer",
                            answer=answer,
                        )
                    retries += 1
                    messages.append({
                        "role": "user",
                        "content": (
                            "A final answer requires successful data-tool evidence. "
                            "Inspect/query the source, or make one valid tool call."
                        ),
                    })
                    continue

                messages.append(self._assistant_tool_message(response))
                strategy_note: str | None = None
                for call in response.tool_calls:
                    tool_calls += 1
                    signature = self._signature(call)
                    signatures[signature] += 1
                    if signatures[signature] >= request.policy.repeated_call_limit:
                        trajectory.append({
                            "kind": "progress_guard",
                            "tool_call_id": call.id,
                            "tool": call.name,
                            "signature": signature,
                            "count": signatures[signature],
                        })
                        if strategy_switches < request.policy.max_strategy_switches:
                            # Second strategy instead of a give-up: do not run the
                            # call again (it cannot return anything new), and once
                            # every call in this reply has its tool message, ask
                            # for a re-plan that names the recent errors and the
                            # other routes. A repeat after the switch budget is
                            # spent still ends the mission below.
                            strategy_switches += 1
                            retries += 1
                            trajectory.append({
                                "kind": "strategy_switch",
                                "iteration": iteration,
                                "tool_call_id": call.id,
                                "tool": call.name,
                                "signature": signature,
                                "switch_index": strategy_switches,
                            })
                            messages.append(self._tool_message(
                                call, False,
                                f"not run: this exact {call.name} call was already "
                                f"made {signatures[signature] - 1} times and its "
                                "results are above; repeating it cannot change them",
                            ))
                            strategy_note = self._strategy_switch_message(
                                call, signatures[signature], recent_failures,
                                [spec.get("function", {}).get("name")
                                 for spec in tool_specs],
                            )
                            continue
                        return self._result(
                            request, started, total_cost, prompt_tokens,
                            completion_tokens, total_tokens, tool_calls,
                            successful_tool_calls, retries, iteration, rows_read,
                            bytes_read, evidence, trajectory,
                            "repeated_tool_call",
                            error="tool_error: repeated equivalent call made no progress",
                            error_kind="tool_error",
                        )

                    if call.name == "return_answer":
                        answer = call.arguments.get("answer")
                        if not isinstance(answer, str) or not answer.strip():
                            retries += 1
                            messages.append(self._tool_message(
                                call, False,
                                "return_answer requires a non-empty string answer",
                            ))
                            continue
                        if (
                            request.policy.require_data_evidence
                            and successful_tool_calls == 0
                        ):
                            retries += 1
                            messages.append(self._tool_message(
                                call, False,
                                "No successful data-tool evidence exists yet",
                            ))
                            continue
                        valid_ids = {item.tool_call_id for item in evidence}
                        # A model shown "result_key=var_fc_..." cites that key;
                        # it names the same successful observation as its call
                        # id. Every rc7 phantom-tool answer cited keys this way.
                        by_key = {item.result_key: item.tool_call_id
                                  for item in evidence if item.result_key}
                        refs = [by_key.get(ref, ref) if isinstance(ref, str) else str(ref)
                                for ref in (call.arguments.get("evidence_refs") or [])]
                        unknown = [ref for ref in refs if ref not in valid_ids]
                        if unknown:
                            retries += 1
                            messages.append(self._tool_message(
                                call, False,
                                f"Unknown evidence_refs: {unknown}; use successful ids only",
                            ))
                            continue
                        effective_refs = refs or sorted(valid_ids)
                        verdict = review_answer(
                            question_plan, answer.strip(),
                            evidence_refs=effective_refs,
                            successful_evidence=successful_tool_calls,
                        )
                        review_status = verdict.status
                        trajectory.append({
                            "kind": "review_verdict",
                            "status": verdict.status,
                            "findings": list(verdict.findings),
                            "covered_outputs": list(verdict.covered_outputs),
                        })
                        if verdict.status == "repair":
                            if review_repairs < request.policy.max_review_repairs:
                                review_repairs += 1
                                retries += 1
                                trajectory.append({
                                    "kind": "review_repair",
                                    "repair_index": review_repairs,
                                    "findings": list(verdict.findings),
                                })
                                messages.append(self._tool_message(
                                    call, False,
                                    "Answer review requires repair: "
                                    + "; ".join(verdict.findings),
                                ))
                                continue
                            return self._result(
                                request, started, total_cost, prompt_tokens,
                                completion_tokens, total_tokens, tool_calls,
                                successful_tool_calls, retries, iteration, rows_read,
                                bytes_read, evidence, trajectory, "review_failed",
                                error="answer_error: semantic review failed after repair",
                                error_kind="answer_error",
                            )
                        trajectory.append({
                            "kind": "termination",
                            "reason": "return_answer",
                            "tool_call_id": call.id,
                            "evidence_refs": effective_refs,
                        })
                        return self._result(
                            request, started, total_cost, prompt_tokens,
                            completion_tokens, total_tokens, tool_calls,
                            successful_tool_calls, retries, iteration, rows_read,
                            bytes_read, evidence, trajectory, "return_answer",
                            answer=answer.strip(),
                        )

                    tool_timeout = self._remaining_timeout(
                        deadline, request.policy.tool_timeout_s
                    )
                    observation = self._execute_tool(
                        self.tools, call, timeout_s=tool_timeout
                    )
                    if observation.success:
                        successful_tool_calls += 1
                        rows_read += max(0, observation.rows_read or 0)
                        bytes_read += max(0, observation.bytes_read or 0)
                    else:
                        retries += 1
                        tool_errors += 1

                    bounded, truncated, digest = self._bound_observation(
                        observation,
                        request.policy.max_prompt_tool_result_bytes,
                    )
                    if observation.success:
                        evidence.append(EvidenceRecord(
                            tool_call_id=call.id,
                            tool=call.name,
                            result_key=observation.result_key,
                            rows_read=observation.rows_read,
                            bytes_read=observation.bytes_read,
                            content_sha256=digest,
                            content_truncated=truncated,
                        ))
                    trajectory.append({
                        "kind": "tool",
                        "iteration": iteration,
                        "tool_call_id": call.id,
                        "tool": call.name,
                        "success": observation.success,
                        "result_key": observation.result_key,
                        "rows_read": observation.rows_read,
                        "bytes_read": observation.bytes_read,
                        "content_sha256": digest,
                        "content_truncated": truncated,
                        "metadata": observation.metadata,
                    })
                    messages.append(self._tool_message(
                        call, observation.success, bounded
                    ))
                    if observation.result_key:
                        tool_result_keys[call.id] = observation.result_key
                    if not observation.success:
                        recent_failures.append(
                            f"{call.name}: {' '.join(bounded.split())[:300]}"
                        )
                        del recent_failures[:-3]

                    if tool_errors >= request.policy.max_tool_errors:
                        return self._result(
                            request, started, total_cost, prompt_tokens,
                            completion_tokens, total_tokens, tool_calls,
                            successful_tool_calls, retries, iteration, rows_read,
                            bytes_read, evidence, trajectory, "tool_error_limit",
                            error="tool_error: maximum failed tool calls reached",
                            error_kind="tool_error",
                        )

                    violation = self._budget_violation(
                        request, started, deadline, total_tokens, total_cost,
                        bytes_read, strict=True,
                    )
                    if violation:
                        return self._result(
                            request, started, total_cost, prompt_tokens,
                            completion_tokens, total_tokens, tool_calls,
                            successful_tool_calls, retries, iteration, rows_read,
                            bytes_read, evidence, trajectory, "budget_exceeded",
                            error=f"budget_exceeded: {violation}",
                            error_kind="budget_exceeded",
                        )

                if strategy_note:
                    # After the tool messages: the protocol requires every
                    # tool_call id to be answered before the next user turn.
                    messages.append({"role": "user", "content": strategy_note})

            return self._result(
                request, started, total_cost, prompt_tokens, completion_tokens,
                total_tokens, tool_calls, successful_tool_calls, retries,
                request.policy.max_iterations, rows_read, bytes_read, evidence,
                trajectory, "max_iterations",
                error="budget_exceeded: max iterations reached",
                error_kind="budget_exceeded",
            )
        except ProviderError as exc:
            return self._exception_result(
                request, started, total_cost, prompt_tokens, completion_tokens,
                total_tokens, tool_calls, successful_tool_calls, retries,
                rows_read, bytes_read, evidence, trajectory, "provider_error", exc,
            )
        except TimeoutError as exc:
            return self._exception_result(
                request, started, total_cost, prompt_tokens, completion_tokens,
                total_tokens, tool_calls, successful_tool_calls, retries,
                rows_read, bytes_read, evidence, trajectory, "timeout", exc,
            )
        except Exception as exc:  # noqa: BLE001 - classified at boundary
            return self._exception_result(
                request, started, total_cost, prompt_tokens, completion_tokens,
                total_tokens, tool_calls, successful_tool_calls, retries,
                rows_read, bytes_read, evidence, trajectory, "unknown_error", exc,
            )
        finally:
            try:
                self.tools.close()
            except Exception as exc:  # noqa: BLE001 - cleanup is non-fatal
                trajectory.append({
                    "kind": "cleanup_error",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    @staticmethod
    def _user_message(request: DataMissionRequest, specialist_payload: str = "") -> str:
        question = request.question
        if not isinstance(question, str):
            question = json.dumps(question, sort_keys=True, default=str)
        return (
            f"QUESTION:\n{question.strip()}\n\n"
            f"SOURCE DESCRIPTION:\n{request.database_description.strip()}\n\n"
            f"SPECIALIST EVIDENCE CONTRACT:\n{specialist_payload}"
        )

    @staticmethod
    def _signature(call: ModelToolCall) -> str:
        raw = json.dumps(
            {"name": call.name, "arguments": call.arguments},
            sort_keys=True, default=str, separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _normalize_answer_aliases(
        response: ModelResponse,
        tool_specs: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        iteration: int,
    ) -> ModelResponse:
        """Rename a misnamed answer call to return_answer (ANSWER_TOOL_ALIASES).

        A provider that does not validate tool names (OpenRouter, Ollama)
        hands the call through; without this the registry answers "unknown
        tool", it counts as a tool error and the answer is lost. A real tool
        that happens to share an alias name is never rewritten.
        """
        real = {spec.get("function", {}).get("name") for spec in tool_specs}
        calls: list[ModelToolCall] = []
        changed = False
        for call in response.tool_calls:
            arguments = (
                None if call.name in real or call.name == "return_answer"
                else answer_alias_arguments(call.name, call.arguments)
            )
            if arguments is None:
                calls.append(call)
                continue
            changed = True
            trajectory.append({
                "kind": "answer_tool_alias",
                "iteration": iteration,
                "tool": call.name,
                "tool_call_id": call.id,
                "path": "tool_call",
            })
            calls.append(replace(call, name="return_answer", arguments=arguments))
        return replace(response, tool_calls=calls) if changed else response

    @staticmethod
    def _strategy_switch_message(
        call: ModelToolCall,
        count: int,
        recent_failures: list[str],
        tool_names: list[Any],
    ) -> str:
        """The re-plan prompt sent instead of ending at the repeat limit."""
        others = [
            name for name in tool_names
            if isinstance(name, str) and name != call.name
        ]
        errors = (
            "Recent tool errors:\n" + "\n".join(f"- {item}" for item in recent_failures)
            + "\n"
            if recent_failures else "No recent tool errors; the call simply "
            "returned the same result each time.\n"
        )
        return (
            f"STRATEGY SWITCH: you requested the same {call.name} call {count} "
            "times. It was not run again.\n" + errors
            + "Re-plan before your next call: state in one line what is still "
            "missing, then take a materially different route -- "
            + (f"a different tool ({', '.join(others)}), " if others else "")
            + "a corrected call that addresses the error above, work on earlier "
            "results you already have, or return_answer with the best "
            "evidence-supported answer. Repeating any call that many times "
            "again ends the mission."
        )

    @classmethod
    def _compact_tool_history(
        cls,
        messages: list[dict[str, Any]],
        policy: Any,
        compacted: set[str],
        result_keys: dict[str, str],
    ) -> dict[str, int] | None:
        """Keep all tool results in the prompt under the total byte bound.

        Oldest first, each over-budget result is cut to its head and tail with
        a note of how much was removed; the newest result is reached last.
        Mutates ``messages`` in place.
        """
        budget = policy.max_prompt_tool_results_total_bytes
        if budget is None or budget <= 0:
            return None
        tool_messages = [m for m in messages if m.get("role") == "tool"]

        def size(message: dict[str, Any]) -> int:
            return len(str(message.get("content") or "").encode("utf-8"))

        total = sum(size(m) for m in tool_messages)
        if total <= budget:
            return None
        before, count = total, 0
        keep = max(0, int(policy.compacted_tool_result_bytes))
        candidates = [
            m for m in tool_messages if m.get("tool_call_id") not in compacted
        ]
        for message in candidates:
            if total <= budget:
                break
            content = str(message.get("content") or "")
            old = size(message)
            if old <= keep:
                continue
            call_id = str(message.get("tool_call_id"))
            key = result_keys.get(call_id)
            prefix = (
                "[earlier result compacted to keep the prompt bounded"
                + (f"; the full result is still available as {key}" if key else "")
                + "] "
            )
            bounded, _, _ = cls._bound_observation(
                ToolObservation(True, content), max(0, keep - len(prefix.encode()))
            )
            message["content"] = prefix + bounded
            compacted.add(call_id)
            total += size(message) - old
            count += 1
        if not count:
            return None
        return {
            "compacted_results": count,
            "bytes_before": before,
            "bytes_after": total,
            "budget_bytes": budget,
        }

    @staticmethod
    def _progress_guidance(
        request: DataMissionRequest,
        *,
        iteration: int,
        total_cost: float | None,
        tool_errors: int,
        successful_tool_calls: int,
        sent: set[str],
    ) -> str | None:
        """Give the model one-shot convergence cues before resources expire."""
        new_reasons: list[str] = []
        cost_limit = request.limits.max_cost_usd
        if (
            "cost" not in sent
            and cost_limit is not None
            and total_cost is not None
            and total_cost >= cost_limit * 0.60
        ):
            sent.add("cost")
            new_reasons.append("the cost budget is more than 60% consumed")
        late_iteration = max(2, int(request.policy.max_iterations * 0.70))
        if "iterations" not in sent and iteration >= late_iteration:
            sent.add("iterations")
            new_reasons.append("fewer than 30% of planned iterations remain")
        if "tool_errors" not in sent and tool_errors >= 3:
            sent.add("tool_errors")
            new_reasons.append("multiple tool attempts have failed")
        if not new_reasons:
            return None

        evidence_note = (
            f"You already have {successful_tool_calls} successful data observations. "
            if successful_tool_calls else ""
        )
        return (
            "PROGRESS CHECK: " + "; ".join(new_reasons) + ". " + evidence_note
            + "Stop environment, package, internet, and alternate-schema probes. "
            "Use the successful evidence, perform at most one targeted missing "
            "calculation, then call return_answer with the best evidence-supported "
            "answer. If exactness is not supportable, state the narrow limitation."
        )

    @staticmethod
    def _assistant_tool_message(response: ModelResponse) -> dict[str, Any]:
        tool_calls: list[dict[str, Any]] = []
        for call in response.tool_calls:
            payload: dict[str, Any] = {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, default=str),
                },
            }
            if "extra_content" in call.provider_metadata:
                payload["extra_content"] = call.provider_metadata["extra_content"]
            tool_calls.append(payload)
        return {
            "role": "assistant",
            "content": response.content,
            "tool_calls": tool_calls,
        }

    @staticmethod
    def _tool_message(
        call: ModelToolCall, success: bool, content: str
    ) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": f"{'SUCCESS' if success else 'FAILED'}: {content}",
        }

    @staticmethod
    def _model_event(
        iteration: int, response: ModelResponse, call_cost: float | None
    ) -> dict[str, Any]:
        return {
            "kind": "model",
            "iteration": iteration,
            "response_id": response.response_id,
            "tool_call_ids": [call.id for call in response.tool_calls],
            "tool_names": [call.name for call in response.tool_calls],
            "usage": response.usage.__dict__,
            "cost_usd": call_cost,
        }

    @staticmethod
    def _execute_tool(
        tools: ToolRegistry, call: ModelToolCall, *, timeout_s: float
    ) -> ToolObservation:
        """Run a synchronous connector behind a hard daemon-thread deadline.

        Connector implementations should still configure their own network and
        database timeouts. The daemon boundary prevents one non-converging call
        from blocking the mission response indefinitely.
        """
        outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                value = tools.execute(call.name, call.arguments, call_id=call.id)
                outcome.put((True, value), block=False)
            except Exception as exc:  # noqa: BLE001 - re-raised in caller
                outcome.put((False, exc), block=False)

        worker = threading.Thread(
            target=invoke, name=f"avaloka-tool-{call.id}", daemon=True
        )
        worker.start()
        try:
            ok, value = outcome.get(timeout=max(0.001, timeout_s))
        except queue.Empty as exc:
            raise TimeoutError(
                f"tool {call.name!r} exceeded {timeout_s:.3f}s"
            ) from exc
        if not ok:
            raise value
        if not isinstance(value, ToolObservation):
            raise TypeError(
                f"tool {call.name!r} returned {type(value).__name__}, "
                "expected ToolObservation"
            )
        return value

    @staticmethod
    def _bound_observation(
        observation: ToolObservation, maximum: int
    ) -> tuple[str, bool, str]:
        raw = observation.content.encode("utf-8", errors="replace")
        digest = hashlib.sha256(raw).hexdigest()
        if maximum <= 0:
            return "", bool(raw), digest
        if len(raw) <= maximum:
            return observation.content, False, digest
        marker = (
            f"\n...[truncated {len(raw) - maximum} bytes; sha256={digest}]...\n"
        ).encode("utf-8")
        if len(marker) >= maximum:
            bounded = marker[:maximum]
        else:
            payload_budget = maximum - len(marker)
            head = payload_budget // 2
            tail = payload_budget - head
            bounded = raw[:head] + marker + (raw[-tail:] if tail else b"")
        # Ignore a partial trailing code point so re-encoding can never exceed
        # the advertised byte limit.
        return bounded[:maximum].decode("utf-8", errors="ignore"), True, digest

    @staticmethod
    def _remaining_timeout(deadline: float | None, tool_timeout: int) -> float:
        if deadline is None:
            return float(tool_timeout)
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("mission deadline reached")
        return max(0.001, min(float(tool_timeout), remaining))

    @staticmethod
    def _scan_ratio(request: DataMissionRequest, bytes_read: int) -> float | None:
        if not request.source_bytes or request.source_bytes <= 0:
            return None
        return bytes_read / request.source_bytes

    @classmethod
    def _budget_violation(
        cls,
        request: DataMissionRequest,
        started: float,
        deadline: float | None,
        total_tokens: int,
        total_cost: float | None,
        bytes_read: int,
        *,
        strict: bool = False,
    ) -> str | None:
        if deadline is not None and time.perf_counter() >= deadline:
            return "latency limit reached"
        token_limit = request.limits.max_tokens
        if token_limit is not None and (
            total_tokens > token_limit if strict else total_tokens >= token_limit
        ):
            return f"token limit reached ({total_tokens}/{token_limit})"
        cost_limit = request.limits.max_cost_usd
        if cost_limit is not None and total_cost is not None and (
            total_cost > cost_limit if strict else total_cost >= cost_limit
        ):
            return f"cost limit reached (${total_cost:.6f}/${cost_limit:.6f})"
        ratio = cls._scan_ratio(request, bytes_read)
        ratio_limit = request.limits.max_data_scanned_ratio
        if ratio is not None and ratio_limit is not None and ratio > ratio_limit:
            return f"data scan ratio {ratio:.6f} exceeded {ratio_limit:.6f}"
        return None

    @classmethod
    def _result(
        cls,
        request: DataMissionRequest,
        started: float,
        cost_usd: float | None,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        tool_calls: int,
        successful_tool_calls: int,
        retries: int,
        iterations: int,
        rows_read: int,
        bytes_read: int,
        evidence: list[EvidenceRecord],
        trajectory: list[dict[str, Any]],
        termination_reason: str,
        *,
        answer: str | None = None,
        error: str | None = None,
        error_kind: str | None = None,
    ) -> DataMissionResult:
        question_event = next((event for event in trajectory if event.get("kind") == "question_plan_created"), {})
        source_event = next((event for event in trajectory if event.get("kind") == "source_map_created"), {})
        route_event = next((event for event in trajectory if event.get("kind") == "specialist_routed"), {})
        review_events = [event for event in trajectory if event.get("kind") == "review_verdict"]
        repair_events = [event for event in trajectory if event.get("kind") == "review_repair"]
        return DataMissionResult(
            answer=answer,
            latency_s=time.perf_counter() - started,
            cost_usd=cost_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
            successful_tool_calls=successful_tool_calls,
            retries=retries,
            iterations=iterations,
            rows_read=rows_read,
            bytes_read=bytes_read,
            data_scanned_ratio=cls._scan_ratio(request, bytes_read),
            evidence=list(evidence),
            trajectory=trajectory,
            termination_reason=termination_reason,
            error=error,
            error_kind=error_kind,
            question_plan=question_event.get("plan"),
            source_map=source_event.get("source_map"),
            specialists=tuple(route_event.get("specialists") or ()),
            review_repairs=len(repair_events),
            review_status=(review_events[-1].get("status") if review_events else None),
        )

    @classmethod
    def _exception_result(
        cls,
        request: DataMissionRequest,
        started: float,
        total_cost: float | None,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        tool_calls: int,
        successful_tool_calls: int,
        retries: int,
        rows_read: int,
        bytes_read: int,
        evidence: list[EvidenceRecord],
        trajectory: list[dict[str, Any]],
        kind: str,
        exc: Exception,
    ) -> DataMissionResult:
        iterations = len([
            event for event in trajectory if event.get("kind") == "model"
        ])
        return cls._result(
            request, started, total_cost, prompt_tokens, completion_tokens,
            total_tokens, tool_calls, successful_tool_calls, retries, iterations,
            rows_read, bytes_read, evidence, trajectory, kind,
            error=f"{kind}: {type(exc).__name__}: {exc}", error_kind=kind,
        )
