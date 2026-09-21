"""Asking what a table *is*, once we know where it is.

table_shape finds the grid. That is enough to stop the "Unnamed: 13" failure,
and not enough to be useful: a sheet can be structurally perfect and still be
unreadable to the planner because nothing says whether it is a balance sheet, a
tax form, a project plan or a hardware cost estimate. Those want different
questions asked of them, and the difference is not recoverable from dtypes.

So the structure is derived and the *meaning* is asked for. One cheap call per
table, with the answer constrained to a schema, and the whole thing optional --
every caller must work when there is no model, because there often is not.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

#: Rows shown to the model. Enough to see the shape of the values, few enough
#: that a 70-column sheet still fits comfortably in a cheap call.
SAMPLE_ROWS = 8
MAX_CELL_CHARS = 60

#: Kinds worth distinguishing because they change what should be asked of the
#: data. Open-ended -- "other" is a real answer and better than a wrong label.
DOCUMENT_KINDS = [
    "balance_sheet", "income_statement", "cash_flow", "tax_form", "invoice",
    "project_plan", "cost_estimate", "rate_card", "budget", "forecast",
    "inventory", "timesheet", "survey_results", "transaction_log",
    "key_value_parameters", "reference_table", "other",
]

_SCHEMA_HINT = """Return JSON only, with exactly this shape:
{
  "kind": one of %s,
  "title": short human title for this table, 2-6 words,
  "description": one sentence on what the table records,
  "orientation": "rows_are_records" | "key_value_pairs" | "matrix",
  "columns": [
    {"name": the column's name as given,
     "meaning": short phrase for what it holds,
     "semantic_type": "identifier"|"category"|"quantity"|"currency"|"percentage"|"date"|"duration"|"text"|"boolean"|"empty",
     "unit": unit or currency if any, else null,
     "suggested_name": a better snake_case name if the given one is a
                       placeholder like column_3, else null}
  ]
}""" % json.dumps(DOCUMENT_KINDS)


def _cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:MAX_CELL_CHARS]


def render_for_model(df: pd.DataFrame, rows: int = SAMPLE_ROWS) -> str:
    """A compact, honest rendering of the table for the prompt."""
    columns = [str(c) for c in df.columns]
    lines = ["columns: " + " | ".join(columns), "rows:"]
    for _, row in df.head(rows).iterrows():
        lines.append("  " + " | ".join(_cell(v) for v in row.tolist()))
    if len(df) > rows:
        lines.append(f"  ... {len(df) - rows} more rows")
    return "\n".join(lines)


def _heuristic(df: pd.DataFrame) -> Dict[str, Any]:
    """What can be said without a model. Always the fallback, never a lie.

    Deliberately does not guess a ``kind``: claiming a sheet is a balance sheet
    on the strength of a column called "Total" is worse than saying unknown,
    because everything downstream would treat the guess as established.
    """
    columns = []
    for name in df.columns:
        series = df[name]
        if series.dropna().empty:
            semantic = "empty"
        elif pd.api.types.is_numeric_dtype(series):
            semantic = "quantity"
        elif pd.api.types.is_datetime64_any_dtype(series):
            semantic = "date"
        elif series.nunique(dropna=True) <= max(2, len(series) * 0.2):
            semantic = "category"
        else:
            semantic = "text"
        columns.append({
            "name": str(name), "meaning": None, "semantic_type": semantic,
            "unit": None, "suggested_name": None,
        })
    return {"kind": "other", "title": None, "description": None,
            "orientation": "rows_are_records", "columns": columns,
            "source": "heuristic"}


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object in interpreter response")
    return json.loads(match.group(0))


def _llm():
    """The interpreter model, or None. Never raises."""
    if os.environ.get("AVALOKA_DISABLE_TABLE_INTERPRETER") == "1":
        return None
    try:
        from app.core.agent_llm import build_agent_llm
        return build_agent_llm(
            agent="TABLE_INTERPRETER",
            api_key=(os.environ.get("GROQ_API_KEY_PLANNING_AGENT")
                     or os.environ.get("GROQ_API_KEY")),
            default_model="openai/gpt-oss-120b",
            temperature=0,
            default_effort="low",
        )
    except Exception as exc:                                   # pragma: no cover
        logger.warning("Table interpreter unavailable: %s", exc)
        return None


def interpret(df: pd.DataFrame, *, sheet_name: Optional[str] = None,
              file_name: Optional[str] = None) -> Dict[str, Any]:
    """Describe what this table is. Falls back to structure alone.

    The sheet and file names are supplied because they are often the only place
    the domain is written down -- a sheet called "Rate Card" says more than its
    columns do.
    """
    base = _heuristic(df)
    if df.empty:
        return base

    model = _llm()
    if model is None:
        return base

    context = []
    if file_name:
        context.append(f"file: {file_name}")
    if sheet_name:
        context.append(f"sheet: {sheet_name}")

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = model.invoke([
            SystemMessage(content=(
                "You identify what a spreadsheet table records, so an analysis "
                "agent can ask sensible questions of it.\n\n" + _SCHEMA_HINT +
                "\n\nName every column in the same order they are given. Do not "
                "invent columns. If you cannot tell what the table is, say "
                '"other" rather than guessing.'
            )),
            HumanMessage(content="\n".join(context + [render_for_model(df)])),
        ])
        parsed = _extract_json(getattr(response, "content", "") or "")
    except Exception as exc:
        logger.warning("Table interpretation failed; using structure only: %s", exc)
        return base

    # Trust the model for meaning, never for structure: the column list is the
    # one the frame actually has, in the order it actually has them.
    by_name = {str(c.get("name", "")).strip(): c for c in (parsed.get("columns") or [])
               if isinstance(c, dict)}
    columns = []
    for position, name in enumerate(df.columns):
        described = by_name.get(str(name)) or {}
        fallback = base["columns"][position]
        columns.append({
            "name": str(name),
            "meaning": described.get("meaning") or None,
            "semantic_type": described.get("semantic_type") or fallback["semantic_type"],
            "unit": described.get("unit") or None,
            "suggested_name": described.get("suggested_name") or None,
        })

    kind = parsed.get("kind")
    return {
        "kind": kind if kind in DOCUMENT_KINDS else "other",
        "title": parsed.get("title") or None,
        "description": parsed.get("description") or None,
        "orientation": parsed.get("orientation") or "rows_are_records",
        "columns": columns,
        "source": "llm",
    }


def apply_suggested_names(df: pd.DataFrame, interpretation: Dict[str, Any]) -> pd.DataFrame:
    """Rename only the placeholders.

    A name the sheet actually gave is left alone even when the model proposes a
    tidier one: the user's word for their own column is the one they will use
    when they ask about it.
    """
    mapping = {}
    for column in interpretation.get("columns") or []:
        name, suggested = column.get("name"), column.get("suggested_name")
        if suggested and re.match(r"^column_\d+$", str(name)):
            mapping[name] = re.sub(r"\W+", "_", str(suggested)).strip("_").lower()
    return df.rename(columns=mapping) if mapping else df


# ---------------------------------------------------------------------------
# Saying it back
# ---------------------------------------------------------------------------
#: How each detected kind reads in a sentence. The label is what a person would
#: call the document, not the enum.
_KIND_PHRASES = {
    "balance_sheet": "balance sheet",
    "income_statement": "income statement",
    "cash_flow": "cash-flow statement",
    "tax_form": "tax form",
    "invoice": "invoice",
    "project_plan": "project plan",
    "cost_estimate": "cost estimate",
    "rate_card": "rate card",
    "budget": "budget",
    "forecast": "forecast",
    "inventory": "inventory",
    "timesheet": "timesheet",
    "survey_results": "survey results",
    "transaction_log": "transaction log",
    "key_value_parameters": "parameter sheet",
    "reference_table": "reference table",
}


def confirm_understanding(interpretation: Dict[str, Any], *, user: Optional[str] = None,
                          goal: Optional[str] = None,
                          dataset: Optional[str] = None) -> str:
    """Say back what we think this is, and ask whether that is right.

    Checking the reading before acting on it is what a competent analyst does
    with someone else's spreadsheet, and it is cheap here: the domain was just
    inferred by a model, so it can be wrong, and a wrong domain quietly steers
    every question that follows. Asking costs one line and makes the correction
    happen now rather than three answers later.
    """
    who = f"{user.strip()}, " if user and user.strip() else ""
    kind = _KIND_PHRASES.get(interpretation.get("kind") or "", "")
    subject = interpretation.get("title") or dataset or "this data"

    if kind:
        described = f"this {kind}"
        if dataset:
            described += f" ({dataset})"
    else:
        described = f"this dataset ({dataset})" if dataset else "this dataset"

    if goal and goal.strip():
        ask = f"and work out {goal.strip()}"
    else:
        ask = "and understand what it holds"

    opener = f"So {who}you want me to read {described} {ask} — have I got that right?"

    detail = interpretation.get("description")
    if detail:
        opener += f"\n\nWhat I can see: {detail}"
    return opener
