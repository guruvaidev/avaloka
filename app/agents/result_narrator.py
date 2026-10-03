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
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.inference import build_chat_model
from app.core.log_utils import describe_response
from app.core.model_config import resolve as resolve_model

logger = logging.getLogger(__name__)


def _resolve_narrator_model() -> str:
    # A dedicated "narrator" entry lets the chat answer use a stronger model than
    # the summarizer. Fall back to the summarizer's model until model_config.py
    # has that entry, so this file can ship on its own.
    try:
        return resolve_model("narrator")
    except KeyError:
        return resolve_model("summarizer")


_NARRATOR_MODEL = _resolve_narrator_model()
_api_key = os.environ.get("GROQ_API_KEY_CODING_AGENT") or os.environ.get("GROQ_API_KEY")
narrator_llm = build_chat_model(
    role="coding",
    agent="NARRATOR",
    tier="small",
    temperature=0,
    groq_model=_NARRATOR_MODEL,
    groq_api_key=_api_key,
)

MAX_ROWS = int(os.getenv("AVALOKA_NARRATOR_MAX_ROWS", "60"))
MAX_CHARS = int(os.getenv("AVALOKA_NARRATOR_MAX_CHARS", "9000"))

# gpt-oss is a reasoning model: its thinking counts against max_tokens.
#
# At "medium" the 20b model reasoned through the whole 4,096-token budget to
# write a ~1,900-character reply, returned nothing (finish=length), and the
# retry with a doubled budget spent another 7,164 tokens: 15.4s for one short
# answer. Describing a table that is already computed needs little reasoning,
# so the default is "low" (typically 1-2s). The values themselves are checked
# afterwards by the claim verifier, not by the model's own reasoning.
_NARRATOR_MAX_TOKENS = int(os.getenv("AVALOKA_NARRATOR_MAX_TOKENS", "4096"))
_NARRATOR_EFFORT = os.getenv("AVALOKA_NARRATOR_REASONING_EFFORT", "low")
_IS_REASONING = "gpt-oss" in (_NARRATOR_MODEL or "").lower()
# When a reply comes back empty because reasoning used the whole budget, the
# retry thinks LESS (faster, almost always answers) instead of being given an
# even bigger budget to think through.
_EFFORT_STEP_DOWN = {"high": "medium", "medium": "low"}

_SYSTEM = """You write the chat reply shown next to a result table. The user can already
see every row and column of that table, so your job is to say what it MEANS, not to
read it back.

How to answer:
- Start with one sentence that directly answers the question (who leads, whether the
  thing asked about is true, the overall picture).
- Then 2 to 4 short markdown bullets with the most useful findings: the leader and the
  laggard, the biggest gaps, anything surprising or contradictory (for example the group
  with the most items is not the one with the highest average).
- Quote at most 2 numbers per bullet and at most 8 numbers in the whole reply.
- Never list a value for every row or every category, and never write one bullet per
  column the user asked for. The table already shows all of that.
- If the table has 3 rows or fewer, simply state the answer with its values.

Accuracy:
- Use ONLY values that appear in the table. Picking the largest or smallest value, or
  comparing two values, is fine. Never compute new statistics, percentages, ratios or
  totals yourself, never estimate, never add outside knowledge.
- If only part of the table is provided, describe it as the rows shown, not the whole.
- Round decimals to at most 2 places and write numbers with thousands separators.
- A cell left out of a row in the JSON was empty in the table.
- If something the user asked for is not in the table, say it was not computed. Never guess.
- No code, no headings, no preamble, no "the table below shows". Under 120 words."""

# Data quality is only discussed when the user asked about it, or when the result
# itself carries a non-empty `issues` column. Otherwise a reply like "No data-quality
# issues were reported" is unrequested filler about checks that never ran.
_DQ_RULES = """

Data quality (the user asked about it, or the result reports issues):
- If the table has an `issues` column, summarise the non-empty issues (how many, the most
  important ones). If all are empty, say the checks that ran found no issues.
- If there is no `issues` column, point out only what the table itself shows, and say
  that no dedicated data-quality checks were run."""
_NO_DQ_RULE = """

- Do not mention data quality, missing values, issues or checks. The user did not ask
  about them and no such checks were part of this analysis."""

_DQ_REQUEST_RE = re.compile(
    r"\b(?:data[\s\-\u2010\u2011]*quality|quality|missing|null|nan|blank|empty\s+values?|duplicat\w*|"
    r"outliers?|anomal\w*|issues?|problems?|errors?|invalid|inconsisten\w*|clean\w*|"
    r"validat\w*|integrity|sanity)\b",
    re.IGNORECASE,
)
_DQ_LINE_RE = re.compile(
    r"data[\s\-\u2010\u2011]*quality|\bno\s+(?:issues|problems|anomalies)\b|\bissues?\s+were\b|"
    r"\bmissing\s+values?\b|\bchecks?\s+(?:ran|run|found)\b",
    re.IGNORECASE,
)

_CONTEXT_SPLIT = re.compile(r"\n\s*\[(?:analysis context|context)\]", re.IGNORECASE)
_EMPTY_CELLS = {"", "nan", "none", "null", "<na>", "nat"}


def _wants_data_quality(question: str, rows: List[Dict[str, Any]]) -> bool:
    if _DQ_REQUEST_RE.search(question or ""):
        return True
    return any(
        str(r.get("issues") or "").strip().lower() not in _EMPTY_CELLS
        for r in rows if isinstance(r, dict) and "issues" in r
    )


def _strip_unrequested_quality(text: str) -> str:
    """Drop bullet lines about data quality when the user never asked for it."""
    kept = [
        line for line in text.splitlines()
        if not (line.lstrip().startswith(("-", "*", "•")) and _DQ_LINE_RE.search(line))
    ]
    return "\n".join(kept).strip()


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


def _compact_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop empty cells. A profiling table is mostly NaN for text columns; sending
    every blank wastes the prompt budget and pushes real rows past MAX_CHARS."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append({
            k: v for k, v in row.items()
            if v is not None and str(v).strip().lower() not in _EMPTY_CELLS
        })
    return out


def _finish_reason(resp: Any) -> str:
    meta = getattr(resp, "response_metadata", None) or {}
    return str(meta.get("finish_reason") or "").lower()


def _invoke(messages: List[Any], max_tokens: int, effort: str = _NARRATOR_EFFORT):
    kwargs: Dict[str, Any] = {"max_tokens": max_tokens}
    if _IS_REASONING:
        kwargs["reasoning_effort"] = effort
    try:
        return narrator_llm.invoke(messages, **kwargs)
    except Exception as exc:
        # A provider that rejects these kwargs must not cost us the answer.
        logger.warning("Narrator LLM rejected tuning kwargs (%s); retrying plain.", exc)
        return narrator_llm.invoke(messages)


def _narrate(question: str, rows: List[Dict[str, Any]]) -> Optional[str]:
    shown = _compact_rows(rows[:MAX_ROWS])
    note = f", first {MAX_ROWS} shown" if len(rows) > MAX_ROWS else ""
    human = (
        f"User request:\n{question}\n\n"
        f"Result table ({len(rows):,} rows{note}; columns: {list(rows[0].keys())}):\n"
        f"{json.dumps(shown, default=str)[:MAX_CHARS]}"
    )
    wants_dq = _wants_data_quality(question, rows)
    system = _SYSTEM + (_DQ_RULES if wants_dq else _NO_DQ_RULE)
    messages = [SystemMessage(content=system), HumanMessage(content=human)]

    resp = _invoke(messages, _NARRATOR_MAX_TOKENS)
    logger.info("Narrator LLM answered: %s", describe_response(resp))
    text = str(getattr(resp, "content", "") or "").strip()

    if not text and _finish_reason(resp) == "length":
        lower = _EFFORT_STEP_DOWN.get(_NARRATOR_EFFORT)
        if lower:
            # Think less, same budget: faster and almost always answers.
            logger.warning(
                "Narrator used its whole token budget at effort=%s with no answer; "
                "retrying once at effort=%s.", _NARRATOR_EFFORT, lower,
            )
            resp = _invoke(messages, _NARRATOR_MAX_TOKENS, effort=lower)
        else:
            # Already at the lowest effort: give it more room once.
            logger.warning(
                "Narrator used its whole token budget at effort=%s with no answer; "
                "retrying once with max_tokens=%d.", _NARRATOR_EFFORT, _NARRATOR_MAX_TOKENS * 2,
            )
            resp = _invoke(messages, _NARRATOR_MAX_TOKENS * 2)
        logger.info("Narrator LLM (retry) answered: %s", describe_response(resp))
        text = str(getattr(resp, "content", "") or "").strip()

    if text and _too_detailed(text, len(rows)):
        logger.info(
            "Narrator reply listed table values (%d numbers); asking for a summary instead.",
            len(_NUMBER_RE.findall(text)),
        )
        retry_messages = messages + [AIMessage(content=text), HumanMessage(content=_REWRITE_NOTE)]
        # Shortening an answer it already wrote needs no deep reasoning.
        resp = _invoke(retry_messages, _NARRATOR_MAX_TOKENS, effort="low")
        summary = str(getattr(resp, "content", "") or "").strip()
        if summary:
            text = summary

    if text and not wants_dq:
        cleaned = _strip_unrequested_quality(text)
        if cleaned != text:
            logger.info("Narrator added unrequested data-quality text; removed it.")
        text = cleaned or text

    if not text:
        logger.warning(
            "Narrator returned empty content (finish_reason=%r); using fallback text.",
            _finish_reason(resp),
        )
    return text or None


# A summary that reads the table back is the failure this guards against: the
# prompt alone is not reliable on long multi-part requests ("for each X compute
# A, B and C"), where models tend to emit one bullet per part listing every row.
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_MAX_REPLY_NUMBERS = int(os.getenv("AVALOKA_NARRATOR_MAX_NUMBERS", "12"))
_MAX_REPLY_BULLETS = int(os.getenv("AVALOKA_NARRATOR_MAX_BULLETS", "6"))
_REWRITE_NOTE = (
    "Your answer repeats values the user can already see in the table. Rewrite it as "
    "a summary: one sentence that answers the question, then at most 4 bullets with "
    "the key findings, at most 8 numbers in total, and no per-row or per-category lists."
)


def _too_detailed(text: str, n_rows: int) -> bool:
    """True when a reply recites the table instead of summarising it."""
    if n_rows <= 3:
        return False  # a tiny result IS the answer; stating its values is correct
    numbers = len(_NUMBER_RE.findall(text))
    bullets = sum(1 for line in text.splitlines() if line.lstrip().startswith(("-", "*", "•")))
    return numbers > _MAX_REPLY_NUMBERS or bullets > _MAX_REPLY_BULLETS


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
    text: Optional[str] = None
    if narrator_llm is not None and question:
        try:
            text = _narrate(question, rows)
        except Exception as exc:
            logger.warning("Result narration failed; using fallback text: %s", exc)
    elif narrator_llm is None:
        logger.warning("Narrator LLM not configured; using fallback text.")

    return {"messages": passthrough["messages"] + [AIMessage(content=text or _fallback(rows))]}