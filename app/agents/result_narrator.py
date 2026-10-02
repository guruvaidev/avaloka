"""Turns the computed result table into a short, grounded chat answer.

Before this node existed nothing after execution answered the user's question:
the chat showed a canned "Analysis complete" line while every requested number
sat in the Data table. This node reads that table and answers each part of the
request using only values the table contains, so the claim verifier that runs
next has real evidence to check the answer against.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.inference import build_chat_model
from app.core.log_utils import describe_response
from app.core.model_config import resolve as resolve_model

logger = logging.getLogger(__name__)

_api_key = os.environ.get("GROQ_API_KEY_CODING_AGENT") or os.environ.get("GROQ_API_KEY")
narrator_llm = build_chat_model(
    role="coding",
    agent="NARRATOR",
    tier="small",
    temperature=0,
    groq_model=resolve_model("summarizer"),
    groq_api_key=_api_key,
)

MAX_ROWS = int(os.getenv("AVALOKA_NARRATOR_MAX_ROWS", "60"))
MAX_CHARS = int(os.getenv("AVALOKA_NARRATOR_MAX_CHARS", "9000"))

_SYSTEM = """You write the chat answer for a data-analysis result. You receive the user's
request and the table the analysis computed.

Rules:
- Answer every part of the request in the order it was asked, as short markdown bullets
  (one bullet per part). Use sub-bullets only for a list of data-quality issues.
- Use ONLY values that appear in the table. Never compute new statistics, percentages,
  ratios or totals yourself, never estimate, never add outside knowledge.
- Round decimals to at most 2 places and write numbers with thousands separators.
- When a fact is the same for every column (e.g. zero missing values everywhere), say it
  in one line instead of listing the columns.
- Do not repeat the whole table - the user sees it below your answer. Name at most 5
  individual columns in any one bullet.
- If the table has an `issues` column, report every non-empty issue as stated, with its
  count. If all are empty, say the checks that ran found no issues.
- If something the user asked for is not in the table, say it was not computed. Never guess.
- No code, no headings, no preamble. Under 200 words."""

_CONTEXT_SPLIT = re.compile(r"\n\s*\[(?:analysis context|context)\]", re.IGNORECASE)


def _user_request(state: Dict[str, Any]) -> str:
    text = ""
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            text = m.content or ""
            break
        if isinstance(m, dict) and m.get("role") in ("user", "human"):
            text = m.get("content") or ""
            break
    text = text or state.get("user_prompt") or ""
    return _CONTEXT_SPLIT.split(str(text), maxsplit=1)[0].strip()


def _result_rows(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    ofd = state.get("output_file_data")
    if isinstance(ofd, dict) and ofd.get("content"):
        try:
            from app.api.helpers import _datauri_csv_to_records
            rows = _datauri_csv_to_records(ofd["content"])
            if rows:
                return rows
        except Exception:
            logger.debug("narrator: could not decode output_file_data", exc_info=True)
    rows = state.get("output_json")
    return rows if isinstance(rows, list) else []


def _fallback(rows: List[Dict[str, Any]]) -> str:
    cols = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
    shown = ", ".join(cols[:8]) + (" …" if len(cols) > 8 else "")
    return (f"The analysis produced {len(rows):,} row(s) with columns: {shown}. "
            "The full result is in the Data table.")


def result_narrator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    passthrough = {"messages": list(state.get("messages") or [])}
    execution_result = state.get("execution_result") or {}
    if execution_result.get("status") in ("error", "failed") or state.get("execution_error"):
        return passthrough

    rows = _result_rows(state)
    if not rows or not isinstance(rows[0], dict):
        return passthrough

    question = _user_request(state)
    text = None
    if narrator_llm is not None and question:
        shown = rows[:MAX_ROWS]
        note = f", first {MAX_ROWS} shown" if len(rows) > MAX_ROWS else ""
        human = (
            f"User request:\n{question}\n\n"
            f"Result table ({len(rows):,} rows{note}; columns: {list(rows[0].keys())}):\n"
            f"{json.dumps(shown, default=str)[:MAX_CHARS]}"
        )
        try:
            resp = narrator_llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=human)])
            logger.info("Narrator LLM answered: %s", describe_response(resp))
            text = str(getattr(resp, "content", "") or "").strip()
        except Exception as exc:
            logger.warning("Result narration failed; using fallback text: %s", exc)

    return {"messages": passthrough["messages"] + [AIMessage(content=text or _fallback(rows))]}