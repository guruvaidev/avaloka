"""The Avaloka 1.0 reply surface, for the 1.0-vs-2.0 comparison.

Same probes, same deterministic scoring and the same production model as
``avaloka_reply_surface`` (the 2.0 surface in ``conversational.py``); the only
difference is the ``app`` package on ``sys.path``, which must be the 1.0 tree
(github/main f772b0fb). 1.0's ``_render_direct_reply`` predates the
conversation-policy directive, so it takes five arguments, not six — this
adapter exists for that signature and refuses to run against a tree that has
the newer one, so a 2.0 tree can never be reported as 1.0 by accident.
"""

from __future__ import annotations

import inspect
import os
import time
from typing import Any, Dict

from benchmarks.adapters import register_adapter
from benchmarks.adapters.base import SystemUnderTest
from benchmarks.core.config import RunConfig
from benchmarks.core.task import BenchmarkTask, MissionOutcome


@register_adapter("avaloka_reply_surface_v1")
class AvalokaReplySurfaceV1(SystemUnderTest):
    key = "avaloka_reply_surface_v1"
    requires_cloud = True

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        if not (os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_PLANNING_AGENT")):
            return MissionOutcome(answer=None, error=(
                "sut_start_error: avaloka_reply_surface_v1 needs GROQ_API_KEY "
                "(or GROQ_API_KEY_PLANNING_AGENT)"))
        from app.agents.avaloka_agent.agent import _render_direct_reply
        from app.agents.avaloka_agent.intent_classifier import IntentResult
        from langchain_core.messages import AIMessage, HumanMessage

        params = list(inspect.signature(_render_direct_reply).parameters)
        if "conversation_directive" in params:
            return MissionOutcome(answer=None, error=(
                "sut_start_error: the app on sys.path has the 2.0 reply surface "
                "(conversation_directive); point PYTHONPATH at the 1.0 tree"))

        turns = task.turns if hasattr(task, "turns") else []
        msgs = [HumanMessage(content=t.text) if t.role == "user" else AIMessage(content=t.text)
                for t in turns]
        evidence = task.params.get("evidence", {}) if getattr(task, "params", None) else {}
        state: Dict[str, Any] = {"user_id": "bench", "session_id": f"conv-reply-v1-{task.id}",
                                 "messages": msgs}
        if evidence:
            state["avaloka_recommendation"] = {
                "source": "verified analysis results for the connected dataset",
                "results": evidence,
            }
        intent = IntentResult(intent="exploration", confidence=1.0,
                              reasoning="benchmark: reply surface pinned")
        start = time.perf_counter()
        reply = _render_direct_reply(state, intent, {}, None, "exploring")
        return MissionOutcome(answer=reply or "", latency_s=time.perf_counter() - start,
                              data_scanned_ratio=0.0)
