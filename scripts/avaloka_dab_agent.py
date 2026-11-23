#!/usr/bin/env python3
"""Avaloka-native executable boundary for official DataAgentBench.

The benchmark supplies database fixtures and low-level transports. Avaloka owns the
reasoning prompt, model loop, progress policy, budgets, evidence trace, and answer.
This module deliberately never imports ``common_scaffold.DataAgent`` or validators.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avaloka.data_agent import (DataMissionRequest, DataQueryMission, MissionLimits,
                                MissionPolicy, ModelResponse, ModelToolCall,
                                ModelUsage, PricingPolicy, ProviderError,
                                ToolObservation)
from avaloka.data_agent.contracts import (ToolCallRejected, ToolRegistry,
                                          tool_use_failed_text)
from avaloka.data_agent.mission import PROMPT_VERSION


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def result_key_for(call_id: str) -> str:
    """The variable name a tool result is stored under for execute_python.

    Must be a Python identifier: providers' call ids carry hyphens (Groq's
    "fc_1b15-..."), and a stored "var_fc_1b15-..." can never be read back --
    the model writes the name it was shown and gets a NameError.
    """
    return "var_" + re.sub(r"\W", "_", call_id)


class OpenAICompatibleProvider:
    """Provider adapter for DAB-supported OpenAI-compatible model endpoints."""

    def __init__(self, model: str, log_path: Path, *, max_output_tokens: int) -> None:
        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError as exc:
            raise ProviderError("the openai package is required for the DAB provider") from exc

        lowered = model.lower()
        # Explicit provider prefixes win over family matching. Without them the
        # policy's benchmark models could not run: "openai/gpt-oss-120b" matched
        # "gpt" and was sent to Azure, and there was no route to a local model.
        if lowered.startswith("groq/"):
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise ProviderError("GROQ_API_KEY is required")
            self.client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
            model = model[len("groq/"):]
        elif lowered.startswith("openrouter/"):
            # Hosted route for vendors this harness must not call directly:
            # the benchmark policy forbids an internal Anthropic key, so Claude
            # runs as "openrouter/anthropic/claude-haiku-5.5" and is billed to
            # the OpenRouter credit the spend ceiling can see.
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise ProviderError("OPENROUTER_API_KEY is required")
            self.client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
            model = model[len("openrouter/"):]
        elif lowered.startswith("ollama/"):
            host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
            if "://" not in host:              # Ollama's own form is host:port
                host = "http://" + host
            self.client = OpenAI(api_key="ollama", base_url=f"{host}/v1")
            model = model[len("ollama/"):]
        elif "gpt" in lowered:
            key = os.environ.get("AZURE_API_KEY")
            if not key:
                raise ProviderError("AZURE_API_KEY is required")
            self.client = AzureOpenAI(
                api_key=key,
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure_endpoint=os.environ.get("AZURE_API_BASE"),
            )
        elif "gemini" in lowered:
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ProviderError("GEMINI_API_KEY is required")
            self.client = OpenAI(
                api_key=key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        elif "kimi" in lowered:
            key = os.environ.get("TOGETHER_API_KEY")
            if not key:
                raise ProviderError("TOGETHER_API_KEY is required")
            self.client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
        elif "claude" in lowered:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ProviderError("ANTHROPIC_API_KEY is required")
            self.client = OpenAI(api_key=key, base_url="https://api.anthropic.com/v1")
        else:
            raise ProviderError(f"unsupported model family: {model}")
        self.model = model
        self.log_path = log_path
        self.max_output_tokens = max_output_tokens
        self.retries = 0

    def complete(self, messages: List[Dict[str, Any]],
                 tools: List[Dict[str, Any]], *, timeout_s: float) -> ModelResponse:
        started = time.perf_counter()
        response = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    timeout=timeout_s,
                    max_tokens=self.max_output_tokens,
                )
                break
            except Exception as exc:
                failed_generation = tool_use_failed_text(exc)
                if failed_generation is not None:
                    # A 400 from the provider's tool-call parser is deterministic
                    # on the same payload: two more identical attempts bought
                    # nothing on the DataAgentBench column (17/54 lost). The
                    # mission re-prompts the model instead.
                    raise ToolCallRejected(
                        f"provider rejected the model's tool call: "
                        f"{type(exc).__name__}: {exc}",
                        failed_generation=failed_generation,
                    ) from exc
                last_error = exc
                self.retries += 1
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 4))
        if response is None:
            raise ProviderError(
                f"model request failed after 3 attempts: {type(last_error).__name__}: "
                f"{last_error}"
            )

        usage_obj = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage_obj, "total_tokens", 0) or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        details = getattr(usage_obj, "prompt_tokens_details", None)
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        message = response.choices[0].message
        calls: List[ModelToolCall] = []
        for index, call in enumerate(getattr(message, "tool_calls", None) or []):
            raw_arguments = call.function.arguments
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) \
                    else dict(raw_arguments)
            except (TypeError, ValueError, json.JSONDecodeError):
                arguments = {"_invalid_arguments": str(raw_arguments)}
            call_id = getattr(call, "id", None)
            if not call_id:
                # Providers should return an id. The deterministic fallback keeps the
                # protocol valid and is explicitly visible in the trace.
                call_id = f"avaloka-{getattr(response, 'id', 'response')}-{index}"
            raw_call = call.model_dump(exclude_none=True) \
                if hasattr(call, "model_dump") else {}
            provider_metadata = {}
            if isinstance(raw_call, dict) and "extra_content" in raw_call:
                provider_metadata["extra_content"] = raw_call["extra_content"]
            calls.append(ModelToolCall(
                id=str(call_id),
                name=str(call.function.name),
                arguments=arguments,
                provider_metadata=provider_metadata,
            ))

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": time.perf_counter() - started,
            "model": self.model,
            "response_id": getattr(response, "id", None),
            "tool_call_ids": [call.id for call in calls],
            "tool_names": [call.name for call in calls],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cached_prompt_tokens": cached_tokens,
            },
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=_json_default) + "\n")
        return ModelResponse(
            response_id=getattr(response, "id", None),
            content=getattr(message, "content", None),
            tool_calls=calls,
            usage=ModelUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_prompt_tokens=cached_tokens,
            ),
        )


def _ensure_databases(config_path: Path) -> None:
    """Provision Mongo/Postgres fixtures once and leave them available for later trials."""
    from common_scaffold.tools.db_utils import mongo_utils, postgres_utils
    from common_scaffold.tools.db_utils.db_config import load_db_clients

    lock_id = hashlib.sha256(str(config_path).encode("utf-8")).hexdigest()[:16]
    lock_path = Path("/private/tmp") / f"avaloka-dab-db-{lock_id}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for client in load_db_clients(str(config_path)).values():
            db_type = client["db_type"]
            if db_type == "mongo" and not mongo_utils.check_db_exists(client["db_name"]):
                mongo_utils.load_db(client["dump_folder"], client["db_name"])
            elif db_type == "postgres" and not postgres_utils.check_db_exists(client["db_name"]):
                postgres_utils.load_db(client["sql_file"], client["db_name"])


class DABToolRegistry(ToolRegistry):
    """Wrap sanctioned DAB database transports behind Avaloka's tool contract."""

    def __init__(self, config_path: Path, run_dir: Path, *, python_timeout_s: int,
                 max_prompt_bytes: int) -> None:
        from common_scaffold.tools.ExecTool import ExecTool
        from common_scaffold.tools.ListDBTool import ListDBTool
        from common_scaffold.tools.QueryDBTool import QueryDBTool

        _ensure_databases(config_path)
        self.tool_log_path = run_dir / "tool_calls.jsonl"
        self.exec_dir = run_dir / "exec_tool_work_dir"
        self.exec_dir.mkdir(parents=True, exist_ok=True)
        self.file_storage = self.exec_dir / "file_storage"
        self.file_storage.mkdir(parents=True, exist_ok=True)
        self.max_prompt_bytes = max_prompt_bytes
        self.storage: Dict[str, Any] = {}
        self.tools = {
            "query_db": QueryDBTool(
                log_path=self.tool_log_path,
                name="query_db",
                db_config_path=str(config_path),
                check_load=False,
            ),
            "list_db": ListDBTool(
                log_path=self.tool_log_path,
                name="list_db",
                db_config_path=str(config_path),
                check_load=False,
            ),
            "execute_python": ExecTool(
                log_path=self.tool_log_path,
                name="execute_python",
                work_dir=self.exec_dir,
                timeout=python_timeout_s,
            ),
        }

    def specs(self) -> List[Dict[str, Any]]:
        return [tool.get_spec() for tool in self.tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any], *,
                call_id: str) -> ToolObservation:
        if name not in self.tools:
            return ToolObservation(False, f"unknown tool: {name}")
        args = dict(arguments)
        if name == "execute_python":
            args["env"] = dict(self.storage)
        result = self.tools[name].exec(args)
        success = bool(result.get("success"))
        value = result.get("result")
        serialized = json.dumps(value, default=_json_default, ensure_ascii=False)
        rows = len(value) if isinstance(value, list) else None
        result_key = result_key_for(call_id)
        if success:
            if len(serialized.encode("utf-8")) > self.max_prompt_bytes:
                file_name = hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:20] + ".json"
                file_path = self.file_storage / file_name
                file_path.write_text(serialized, encoding="utf-8")
                self.storage[result_key] = str(file_path.relative_to(self.exec_dir))
                # The mission cuts this to its prompt cap; say here how large
                # the full result is, which that cut alone cannot know.
                content = (
                    f"result_key={result_key}; full result "
                    f"({len(serialized.encode('utf-8'))} bytes) is in "
                    f"{self.storage[result_key]}; "
                    f"preview={serialized[:self.max_prompt_bytes]}"
                )
            else:
                self.storage[result_key] = value
                content = f"result_key={result_key}; result={serialized}"
        else:
            content = serialized[:self.max_prompt_bytes]
        return ToolObservation(
            success=success,
            content=content,
            result_key=result_key if success else None,
            rows_read=rows,
            bytes_read=len(serialized.encode("utf-8")),
            metadata={"tool": name},
        )

    def close(self) -> None:
        # Query/List cleanup drops shared Mongo/Postgres fixtures. The benchmark owns
        # fixture lifecycle, so only the per-run Python executor is stopped here.
        execute_python = self.tools.get("execute_python")
        if execute_python is not None:
            execute_python.clean_up()


#: The sanctioned tools' interface, which DAB states only in its reference
#: agent's system prompt. execute_python's own spec says "print the final result
#: following the required PRINT FORMAT" without defining it, so an agent not
#: told the format cannot succeed with the tool at all. This is the
#: environment's interface, given to every DAB agent -- not task knowledge.
DAB_TOOL_CONTRACT = """
TOOL CONTRACT (the benchmark environment's interface):
- query_db on SQL databases takes one SQL statement for that database only; a
  query cannot reference tables in another database. Combine sources with
  execute_python. In PostgreSQL, double-quote mixed-case identifiers and use
  single quotes for string literals.
- query_db on MongoDB takes a JSON string with "collection" and optional
  "filter", "projection" and "limit" fields.
- list_db takes a db_name exactly as named in the database description.
- execute_python runs your code with exec() in Python 3.12 with only pandas and
  pyarrow. Earlier tool results are available as variables named by their
  result_key. Do not use triple double-quotes in the code.
- execute_python returns only what follows a marker line. End the code with:
      print("__RESULT__:")
      print(<a string that json.loads() can parse>)
  Use json.dumps(...) on plain values; convert DataFrames with
  .to_json(orient="records").
""".strip()


def _optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else None


def _policy_int(env_name: str, raw_policy: Dict[str, Any], key: str,
                default: int) -> int:
    """An operator env override, else the request's policy value, else default."""
    value = os.environ.get(env_name)
    if value not in (None, ""):
        return int(value)
    return int(raw_policy.get(key, default))


def _mission_policy(raw_policy: Dict[str, Any]) -> MissionPolicy:
    """The mission policy from the request, with operator env overrides for
    the prompt-size and strategy-switch bounds."""
    defaults = MissionPolicy()
    return MissionPolicy(
        max_iterations=int(raw_policy.get("max_iterations", 25)),
        tool_timeout_s=int(raw_policy.get("tool_timeout_s", 120)),
        max_output_tokens=int(raw_policy.get("max_output_tokens", 8_192)),
        max_prompt_tool_result_bytes=_policy_int(
            "AVALOKA_DAB_MAX_TOOL_RESULT_BYTES", raw_policy,
            "max_prompt_tool_result_bytes", 100_000,
        ),
        max_prompt_tool_results_total_bytes=_policy_int(
            "AVALOKA_DAB_MAX_TOOL_RESULTS_TOTAL_BYTES", raw_policy,
            "max_prompt_tool_results_total_bytes",
            defaults.max_prompt_tool_results_total_bytes,
        ),
        compacted_tool_result_bytes=_policy_int(
            "AVALOKA_DAB_COMPACTED_TOOL_RESULT_BYTES", raw_policy,
            "compacted_tool_result_bytes", defaults.compacted_tool_result_bytes,
        ),
        max_strategy_switches=_policy_int(
            "AVALOKA_DAB_MAX_STRATEGY_SWITCHES", raw_policy,
            "max_strategy_switches", defaults.max_strategy_switches,
        ),
    )


def _pricing() -> PricingPolicy:
    threshold = os.environ.get("AVALOKA_DAB_PRICING_THRESHOLD_TOKENS")
    return PricingPolicy(
        input_usd_per_mtok=_optional_float("AVALOKA_DAB_INPUT_USD_PER_MTOK"),
        output_usd_per_mtok=_optional_float("AVALOKA_DAB_OUTPUT_USD_PER_MTOK"),
        cached_input_usd_per_mtok=_optional_float(
            "AVALOKA_DAB_CACHED_INPUT_USD_PER_MTOK"
        ),
        threshold_tokens=int(threshold) if threshold else None,
        input_usd_per_mtok_over_threshold=_optional_float(
            "AVALOKA_DAB_INPUT_USD_PER_MTOK_OVER_THRESHOLD"
        ),
        output_usd_per_mtok_over_threshold=_optional_float(
            "AVALOKA_DAB_OUTPUT_USD_PER_MTOK_OVER_THRESHOLD"
        ),
        version=os.environ.get("AVALOKA_DAB_PRICING_VERSION", "environment"),
    )


def _with_catalog_tier(pricing: PricingPolicy, model: str,
                       catalog_path: str | None = None) -> PricingPolicy:
    """Add the OpenRouter long-prompt tier to operator rates that left it out.

    The operator rates are flat, but OpenRouter bills some models more per
    token once a call's prompt crosses a threshold: Claude Haiku 5.5 costs
    $0.10/$0.50 per MTok below 100k prompt tokens and $0.50/$2.50 from 100k.
    The rc7 DAB Haiku cell priced all 452 calls flat and reported $1.98; the
    43 calls over 100k made the real bill $4.48, above the $3.00 ceiling the
    harness was enforcing against the flat figure. The per-trial cap and the
    run ceiling both read this policy, so the tier has to be in it.

    Only ``openrouter/`` models: the vendored catalog is that provider's price
    sheet and no other's. An operator who sets
    AVALOKA_DAB_PRICING_THRESHOLD_TOKENS keeps exactly the rates they set.
    """
    if (not pricing.configured or pricing.threshold_tokens is not None
            or not model.lower().startswith("openrouter/")):
        return pricing
    from benchmarks.core.pricing import SNAPSHOT_PATH, load_snapshot
    model_id = model[len("openrouter/"):]
    entry = next((m for m in load_snapshot(catalog_path or SNAPSHOT_PATH).get("data") or []
                  if m.get("id") == model_id), None)
    tiers = ((entry or {}).get("pricing") or {}).get("overrides") or []
    if not tiers:
        return pricing
    if len(tiers) > 1:
        # PricingPolicy holds one tier; pricing a second as the first would
        # under-count, which is the failure this function exists to prevent.
        raise RuntimeError(
            f"{model} has {len(tiers)} long-prompt price tiers in the catalog; set "
            "AVALOKA_DAB_PRICING_THRESHOLD_TOKENS and the over-threshold rates explicitly"
        )
    tier = tiers[0]
    # A cached rate for the tier only when the operator priced cached tokens at
    # all; otherwise cached tokens bill as input on both sides of the threshold.
    tier_cached = (float(tier["input_cache_read"]) * 1_000_000.0
                   if pricing.cached_input_usd_per_mtok is not None
                   and tier.get("input_cache_read") is not None else None)
    return replace(
        pricing,
        # The catalog tier starts AT min_prompt_tokens; the policy's starts
        # ABOVE threshold_tokens.
        threshold_tokens=int(tier["min_prompt_tokens"]) - 1,
        input_usd_per_mtok_over_threshold=float(tier["prompt"]) * 1_000_000.0,
        output_usd_per_mtok_over_threshold=float(tier["completion"]) * 1_000_000.0,
        cached_input_usd_per_mtok_over_threshold=tier_cached,
        version=f"{pricing.version}+openrouter-catalog-tier",
    )


def main() -> int:
    request = json.load(sys.stdin)
    if request.get("protocol_version") != "dab-command-v2":
        raise ValueError("unsupported DAB command protocol; expected dab-command-v2")

    dab_root = Path(os.environ.get("DAB_ROOT", "")).expanduser().resolve()
    if not (dab_root / "common_scaffold" / "tools").is_dir():
        raise RuntimeError("DAB_ROOT does not contain the sanctioned DAB tools")
    # The low-level tool modules use common_scaffold imports. The agent and validator
    # modules are intentionally never imported.
    sys.path.insert(0, str(dab_root))
    bridge_bin = Path(__file__).resolve().parent / "dab-bin"
    os.environ["PATH"] = str(bridge_bin) + os.pathsep + os.environ.get("PATH", "")

    run = request["run"]
    mission = request["mission"]
    source = request["sources"]
    model_info = request["model"]
    raw_budget = request.get("budget") or {}
    raw_policy = request.get("policy") or {}
    model = str(model_info["provider_model"])
    policy = _mission_policy(raw_policy)
    limits = MissionLimits(
        max_cost_usd=raw_budget.get("max_cost_usd"),
        max_latency_s=raw_budget.get("max_latency_s"),
        max_tokens=raw_budget.get("max_tokens"),
        max_data_scanned_ratio=raw_budget.get("max_data_scanned_ratio"),
    )
    pricing = _with_catalog_tier(_pricing(), model)
    if limits.max_cost_usd is not None and not pricing.configured:
        raise RuntimeError(
            "a max_cost_usd budget requires configured AVALOKA_DAB input/output pricing"
        )

    working_directory = Path(source["working_directory"]).resolve()
    run_name = (
        f"avaloka_native_{run['dataset']}_q{run['query']}_r{run['trial']}_"
        f"{time.time_ns()}_{os.getpid()}"
    )
    run_dir = working_directory / "logs" / "avaloka" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    provider_log = run_dir / "model_calls.jsonl"
    provider = OpenAICompatibleProvider(
        model=model,
        log_path=provider_log,
        max_output_tokens=policy.max_output_tokens,
    )
    tools = DABToolRegistry(
        Path(source["database_config_path"]).resolve(),
        run_dir,
        python_timeout_s=int(raw_policy.get("python_timeout_s", policy.tool_timeout_s)),
        max_prompt_bytes=policy.max_prompt_tool_result_bytes,
    )
    result = DataQueryMission(provider, tools, pricing).run(DataMissionRequest(
        task_id=str(run["task_id"]),
        question=mission["question"],
        database_description=str(mission["database_description"]) + "\n\n" + DAB_TOOL_CONTRACT,
        model=model,
        seed=int(run.get("seed", 0)),
        limits=limits,
        policy=policy,
    ))
    response = {
        "answer": result.answer,
        "latency_s": result.latency_s,
        "cost_usd": result.cost_usd,
        "tokens": result.total_tokens,
        "tool_calls": result.tool_calls,
        "retries": result.retries + provider.retries,
        "trajectory": result.trajectory,
        "error": result.error,
        "error_kind": result.error_kind,
        "artifacts": {
            "sut_name": "avaloka.data_agent.DataQueryMission",
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "run_dir": str(run_dir),
            "model_log": str(provider_log),
            "tool_log": str(tools.tool_log_path),
            "termination_reason": result.termination_reason,
            "iterations": result.iterations,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "pricing_version": pricing.version,
            "pricing_configured": pricing.configured,
            "pricing_threshold_tokens": pricing.threshold_tokens,
            "effective_policy": asdict(policy),
            "effective_limits": asdict(limits),
        },
    }
    json.dump(response, sys.stdout, default=_json_default)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"avaloka_dab_agent: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
