# app/api/graph_runtime.py
from pathlib import Path
import sys
from typing import Any, Dict

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from app.api.config import logger

try:
    here = Path(__file__).resolve()
    for c in [here.parent, here.parent.parent, here.parent.parent.parent]:
        p = str(c)
        if p not in sys.path:
            sys.path.append(p)

    from app.agents import planner as _planner_mod

    _orig_plan = _planner_mod.plan_etl_job
    # ... your _fallback_decide and _plan_etl_job_safe here ...

    from app.api.workflow import build_graph
    from langgraph.checkpoint.memory import MemorySaver

    GRAPH = build_graph().compile(checkpointer=MemorySaver())
    GRAPH_READY = True
    logger.info("LangGraph build_graph loaded successfully with patched planner.")
except Exception as e:
    logger.warning("[warn] build_graph failed (%s); using echo fallback.", e)
    GRAPH_READY = False

    def _echo_handler(state: Dict[str, Any]) -> Dict[str, Any]:
        msgs = state.get("messages") or []
        text = ""
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                text = m.content
                break
        out = (state.get("messages") or []) + [AIMessage(content=f"(fallback graph) you said: {text}")]
        return {"messages": out, "ready_to_summarize": False, "ready_to_code": False}

    GRAPH = RunnableLambda(_echo_handler)
