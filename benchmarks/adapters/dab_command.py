"""Isolated JSON command boundary for the Avaloka-native DAB system under test."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from . import register_adapter
from ..core.config import RunConfig
from ..core.task import FailureKind, BenchmarkTask, MissionOutcome, classify_failure


def _cost_provenance(cost_usd, artifacts, model):
    """Label where a DAB trial's cost came from, so the spend ceiling can tell.

    Without a label every DAB trial was cost_source="unknown", which the ledger
    counts as a call it could not see -- a run reported "10 unpriced/usage-
    missing" beside an observed $0.32 that the ceiling had in fact charged.
    """
    from benchmarks.core.pricing import (SOURCE_LOCAL, SOURCE_OPERATOR_RATE,
                                         SOURCE_UNPRICED)
    if str(model or "").lower().startswith("ollama/"):
        return 0.0, SOURCE_LOCAL                      # local model: no API cost
    if cost_usd is not None and artifacts.get("pricing_configured"):
        return cost_usd, SOURCE_OPERATOR_RATE
    return cost_usd, SOURCE_UNPRICED                   # genuinely blind


@register_adapter("dab_command")  # compatibility key for pre-v2 local configs
@register_adapter("avaloka_dab")
class AvalokaDABCommandAdapter:
    """Invoke Avaloka once per DAB task using the contamination-safe v2 contract."""

    key = "avaloka_dab"
    requires_cloud = True

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        command = os.environ.get("AVALOKA_DAB_AGENT_CMD")
        if not command:
            return MissionOutcome(
                answer=None,
                error=("sut_start_error: AVALOKA_DAB_AGENT_CMD is not set; configure "
                       "Avaloka's dab-command-v2 executable."),
                error_kind=FailureKind.SUT_START,
            )

        request = {
            "protocol_version": "dab-command-v2",
            "run": {
                "task_id": task.id,
                "dataset": task.params["dataset"],
                "query": task.params["query"],
                "trial": int(config.extra.get("_trial", 0)),
                "seed": config.seed,
            },
            "mission": {
                "question": task.mission,
                "database_description": task.params["db_description"],
            },
            "sources": {
                "database_config_path": task.params["db_config_path"],
                "working_directory": task.params["dataset_dir"],
                "database_types": task.params.get("database_types", ()),
            },
            "model": {
                "provider_model": config.llm,
                "routing_policy": config.model_routing,
            },
            "budget": asdict(config.budget),
            "policy": {
                "max_iterations": int(config.extra.get("max_iterations", 25)),
                "tool_timeout_s": int(config.extra.get("tool_timeout_s", 120)),
                "python_timeout_s": int(config.extra.get("python_timeout_s", 120)),
                "max_output_tokens": int(config.extra.get("max_output_tokens", 8_192)),
                "max_prompt_tool_result_bytes": int(
                    config.extra.get("max_prompt_tool_result_bytes", 100_000)
                ),
            },
        }
        mission_timeout = config.budget.max_latency_s or 600
        # Leave a small grace period for child-side cleanup and structured reporting.
        timeout = mission_timeout + 30
        started = time.perf_counter()
        try:
            process = subprocess.run(
                shlex.split(command),
                input=json.dumps(request),
                text=True,
                capture_output=True,
                cwd=Path(__file__).resolve().parents[2],
                timeout=timeout,
                check=False,
                env={**os.environ, "DAB_NETWORK_DISABLED": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            return MissionOutcome(
                answer=None,
                latency_s=time.perf_counter() - started,
                error=f"timeout: Avaloka DAB command exceeded {timeout}s: {exc}",
                error_kind=FailureKind.TIMEOUT,
            )
        except OSError as exc:
            return MissionOutcome(
                answer=None,
                latency_s=time.perf_counter() - started,
                error=f"sut_start_error: {exc}",
                error_kind=FailureKind.SUT_START,
            )

        latency = time.perf_counter() - started
        if process.returncode:
            error = f"sut_start_error: command exited {process.returncode}: {process.stderr[-4000:]}"
            return MissionOutcome(
                answer=None,
                latency_s=latency,
                error=error,
                error_kind=classify_failure(error),
                artifacts={"stderr": process.stderr[-4000:]},
            )
        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            return MissionOutcome(
                answer=None,
                latency_s=latency,
                error=f"sut_start_error: invalid command JSON: {exc}: {process.stdout[-1000:]}",
                error_kind=FailureKind.SUT_START,
                artifacts={"stderr": process.stderr[-4000:]},
            )

        trajectory = payload.get("trajectory", [])
        artifacts = dict(payload.get("artifacts", {}))
        artifacts["stderr"] = process.stderr[-4000:]
        artifacts["protocol_version"] = request["protocol_version"]
        artifacts["mission_latency_s"] = payload.get("latency_s")
        raw_kind = payload.get("error_kind")
        try:
            error_kind = FailureKind(raw_kind) if raw_kind else None
        except ValueError:
            error_kind = classify_failure(payload.get("error"))
        raw_cost = payload.get("cost_usd")
        cost_usd = float(raw_cost) if raw_cost is not None else None
        cost_usd, cost_source = _cost_provenance(cost_usd, artifacts, config.llm)
        return MissionOutcome(
            answer=payload.get("answer"),
            cost_usd=cost_usd,
            cost_source=cost_source,
            latency_s=latency,
            data_scanned_ratio=payload.get("data_scanned_ratio"),
            tool_calls=int(payload.get("tool_calls", 0)),
            tokens=int(payload.get("tokens", 0)),
            retries=int(payload.get("retries", 0)),
            trajectory=trajectory if isinstance(trajectory, list) else [{"trace": trajectory}],
            artifacts=artifacts,
            error=payload.get("error"),
            error_kind=error_kind,
        )
