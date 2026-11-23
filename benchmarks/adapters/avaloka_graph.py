"""Avaloka system-under-test adapters.

Two ways to drive Avaloka as a benchmarked system, both grounded in the real code:

* ``avaloka`` (:class:`AvalokaGraphAdapter`) — end-to-end mission through the compiled
  LangGraph. Built by ``app/api/workflow.py:build_graph`` and exposed as
  ``app/api/langgraph_app.py:graph``. State is ``ETLState`` (``app/graph/etl_state.py``).
  The mechanical execution knob is ``analysis_fidelity``
  (``quick_sample`` | ``portfolio_samples`` | ``entire_dataset``).

* ``avaloka_sampler`` (:class:`AvalokaSamplerAdapter`) — Track-1 / Track-4 unit
  entrypoint that calls the Sampling agent directly
  (``app/agents/sampling_agent_daft.py:sample_with_profiling``) so adaptive sampling
  can be measured in isolation without standing up the full graph.

Both import the Avaloka app **lazily**, inside ``run_mission``. Importing this module
never imports the app, so the rest of the harness (and its unit tests) run without
Groq keys, cloud creds, Ray, or a server.

NOTE (accuracy): the shipped sampler is *tiered + multi-strategy portfolio* — it has
**no statistical stopping / confidence-convergence loop** today (see benchmarks.md §4
Track 1 "Implementation status"). Until that lands, this adapter reports the sizes and
strategies the real agent chose; it does not invent confidence intervals the agent did
not produce.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from . import register_adapter
from ..core.config import RunConfig
from ..core.task import BenchmarkTask, MissionOutcome


_FIDELITY_BY_SAMPLING_MODE = {
    "quick_sample": "quick_sample",
    "portfolio_samples": "portfolio_samples",
    "entire_dataset": "entire_dataset",
    # Track-1 strategy names → nearest fidelity knob.
    "adaptive": "portfolio_samples",
    "full": "entire_dataset",
    "random": "quick_sample",
    "stratified": "portfolio_samples",
}


@register_adapter("avaloka")
class AvalokaGraphAdapter:
    """Runs a full mission through the compiled Avaloka LangGraph.

    This is the black-box, end-to-end SUT used by external benchmarks (DAB, DSBench,
    DAComp, ...) and by Tracks 2/5/6. It requires the Avaloka runtime (Groq key at
    minimum); the harness surfaces a clear error if that runtime is unavailable rather
    than silently degrading.
    """

    key = "avaloka"
    # Drives the real graph (Groq + possibly Ray/cloud); gated behind --allow-cloud.
    requires_cloud = True

    def _build_graph(self):
        # Imported lazily; see module docstring.
        from app.api.workflow import build_graph  # type: ignore
        return build_graph(checkpointer=None).compile()

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        try:
            graph = self._build_graph()
        except Exception as exc:  # missing deps / keys → honest, actionable error
            return MissionOutcome(
                answer=None,
                error=f"avaloka_runtime_unavailable: {type(exc).__name__}: {exc}. "
                      f"The Avaloka graph could not be built — ensure the app deps and "
                      f"GROQ_API_KEY* are configured to run the `avaloka` SUT.",
            )

        fidelity = _FIDELITY_BY_SAMPLING_MODE.get(config.sampling_mode, "portfolio_samples")
        state: Dict[str, Any] = {
            "user_prompt": task.mission,
            "messages": [{"role": "user", "content": task.mission}],
            "analysis_fidelity": fidelity,
            **task.params.get("etl_state", {}),
        }
        thread_id = f"bench-{task.id}-{config.id}-{config.extra.get('_trial', 0)}"

        start = time.perf_counter()
        try:
            final_state = graph.invoke(
                state, config={"configurable": {"thread_id": thread_id}}
            )
        except Exception as exc:
            return MissionOutcome(answer=None, latency_s=time.perf_counter() - start,
                                  error=f"avaloka_invoke_error: {type(exc).__name__}: {exc}")

        latency = time.perf_counter() - start
        return MissionOutcome(
            answer=_extract_answer(final_state, task),
            latency_s=latency,
            # Real cost/token/data-scanned accounting requires instrumenting the graph
            # (benchmark_metrics/ray_execution_profile in ETLState). Left at defaults
            # until that telemetry is threaded through — never fabricated.
            data_scanned_ratio=_data_scanned_ratio(final_state),
            tokens=int(final_state.get("token_usage", 0) or 0)
            if isinstance(final_state, dict) else 0,
            artifacts={"final_state_keys": sorted(final_state.keys())
                       if isinstance(final_state, dict) else []},
            trajectory=[{"analysis_fidelity": fidelity}],
        )


@register_adapter("avaloka_sampler")
class AvalokaSamplerAdapter:
    """Direct call to the Sampling agent for Track-1 isolation (Track-4 ablation).

    Wraps ``app/agents/sampling_agent_daft.py:sample_with_profiling``. ``task.params``
    must carry the ``path``/``source_type`` the sampler expects.
    """

    key = "avaloka_sampler"
    requires_cloud = True

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        try:
            from app.agents.sampling_agent_daft import sample_with_profiling  # type: ignore
        except Exception as exc:
            return MissionOutcome(
                answer=None,
                error=f"avaloka_sampler_unavailable: {type(exc).__name__}: {exc}",
            )
        p = task.params
        start = time.perf_counter()
        try:
            result = sample_with_profiling(
                path=p["path"],
                source_type=p.get("source_type", "csv"),
                stratify_by=p.get("stratify_column"),
                sample_size=int(config.extra.get("sample_size", 1000)),
            )
        except Exception as exc:
            return MissionOutcome(answer=None, latency_s=time.perf_counter() - start,
                                  error=f"avaloka_sampler_error: {type(exc).__name__}: {exc}")
        return MissionOutcome(
            answer=result,
            latency_s=time.perf_counter() - start,
            artifacts={"keys": sorted(result.keys()) if isinstance(result, dict) else []},
        )


def _extract_answer(final_state: Any, task: BenchmarkTask) -> Any:
    """Pull the gradable answer out of the final ETLState.

    Evaluators for graph-driven tasks decide what "answer" means; we hand them the
    most useful slices (the assistant's last message, output json, artifacts) rather
    than guessing a single field.
    """
    if not isinstance(final_state, dict):
        return final_state
    msgs = final_state.get("messages") or []
    last = msgs[-1] if msgs else None
    return {
        "last_message": getattr(last, "content", None) if last is not None
        and not isinstance(last, dict) else (last or {}).get("content") if isinstance(last, dict) else None,
        "output_json": final_state.get("output_json"),
        "output_file_data": final_state.get("output_file_data"),
        "plan": final_state.get("plan"),
        "training_result": final_state.get("training_result"),
    }


def _data_scanned_ratio(final_state: Any) -> float | None:
    if not isinstance(final_state, dict):
        return None
    prof = final_state.get("ray_execution_profile") or {}
    if isinstance(prof, dict) and "rows_scanned" in prof and "rows_total" in prof:
        total = prof.get("rows_total") or 0
        return (prof["rows_scanned"] / total) if total else None
    return None
