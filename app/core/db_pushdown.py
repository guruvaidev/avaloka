"""Exact group aggregates computed by the database, over every row.

A question like "what is the stroke rate for patients with hypertension versus
without" has one honest answer and the database can compute it directly:
``AVG`` of a 0/1 column grouped by another column. Answering it from the
analysis sample gives a sample estimate at best, and gave a head-of-table
figure before the sample was made random. This module recognises that narrow
family of questions -- a rate, mean or count, optionally by one group column --
and runs it as SQL over the MCP ``query`` tool.

It is deliberately conservative. The question has to name exactly one measure
and at most one grouping column, both by their schema names; any other schema
column, a number, or a filter word means the question is more than a plain
group aggregate, and the caller falls through to the normal sampled analysis
(which discloses its sample). A rate needs a column whose values are 0/1 or
boolean, checked against the data before the aggregate runs; a mean needs a
numeric column. Dialects: PostgreSQL, MySQL/MariaDB and SQLite.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from app.core.db_sampling import normalize_dialect
from avaloka.goal_analysis import canonical_column_words

PUSHDOWN_DIALECTS = ("postgresql", "mysql", "sqlite")
MAX_GROUPS = 50

RunQuery = Callable[[str], Awaitable[Dict[str, Any]]]

# Words that mean the question is not a plain whole-table group aggregate.
_NOT_PLAIN = re.compile(
    r"\b(where|filter\w*|only|exclud\w*|except|top|bottom|rank\w*|trend|over time|per (day|week|month|year)"
    r"|median|percentile|quantile|std|deviation|variance|correlat\w*|regress\w*|predict\w*|model|train\w*"
    r"|transfer|copy|move|export|join|merge|plot|chart|graph|visuali[sz]e|between|above|below|over|under"
    r"|older|younger|greater|less|more than|at least|at most|after|before|since|during|last|first)\b"
    r"|[<>=]|\d",
    re.IGNORECASE,
)
_RATE_WORDS = r"(?:rate|proportion|share|percentage|percent|fraction|prevalence|incidence)"
_MEAN_WORDS = r"(?:average|mean|avg)"
_COUNT_WORDS = r"(?:how many|count|number)"


@dataclass
class AggregateIntent:
    operation: str                  # "rate" | "mean" | "count"
    measure: Optional[str]          # column, None for count
    group_by: Optional[str]         # column, None for the whole table


@dataclass
class PushdownResult:
    intent: AggregateIntent
    groups: List[Tuple[Any, Optional[float], int]]   # (group value, figure, rows in group)
    overall: Optional[float]
    total_rows: int
    dialect: str
    notes: List[str] = field(default_factory=list)


def _col_core(col: str) -> str:
    # "blood_pressure" matches "blood_pressure" and "blood pressure".
    # A flag asked about by its verb ("the return rate" for ``returned``) is
    # rewritten to its name by canonical_column_words before any pattern here
    # runs, so the stem rule lives in one place (avaloka.goal_analysis).
    return r"[_\s]+".join(re.escape(p) for p in re.split(r"[_\s]+", col) if p)


def _col_pattern(col: str) -> str:
    return r"\b" + _col_core(col) + r"\b"


def _mentions(text: str, columns: List[str]) -> List[Tuple[int, str]]:
    found: List[Tuple[int, str]] = []
    for col in columns:
        if not col or not re.search(r"[A-Za-z]", col):
            continue
        m = re.search(_col_pattern(col), text, re.IGNORECASE)
        if m:
            found.append((m.start(), col))
    # A column whose name is part of a longer matched column is not a mention.
    keep = [(pos, col) for pos, col in found
            if not any(other != col and col.lower() in other.lower()
                       and re.search(_col_pattern(other), text, re.IGNORECASE)
                       for _p, other in found)]
    return sorted(keep)


def parse_aggregate_intent(prompt: str, columns: List[str]) -> Optional[AggregateIntent]:
    """The plain group aggregate ``prompt`` asks for, or None when it asks for more."""
    text = (prompt or "").split("\n[Analysis context]", 1)[0].strip()
    if not text or len(text) > 200 or not columns:
        return None
    cols = [str(c) for c in columns]
    # "return rate by channels" asks about ``returned`` by ``channel``: an
    # inflected column word is rewritten to the column's name first, whole
    # words only, so the rules below see the names they match on.
    text = canonical_column_words(text, cols)
    mentioned = _mentions(text, cols)
    if not mentioned or len(mentioned) > 2:
        return None
    # Filter words are judged with the column names removed, so a column called
    # "first_visit" or "age_over" cannot trip them and a phrase cannot hide in one.
    stripped = text
    for _pos, col in mentioned:
        stripped = re.sub(_col_pattern(col), " ", stripped, flags=re.IGNORECASE)
    if _NOT_PLAIN.search(stripped):
        return None
    names = {c.lower(): c for _p, c in mentioned}
    alt = "|".join(sorted((_col_pattern(c) for c in names.values()), key=len, reverse=True))

    def grab(pattern: str) -> Optional[str]:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            return None
        hit = m.group("col")
        return next((c for c in names.values() if re.fullmatch(_col_core(c), hit, re.IGNORECASE)), None)

    group = (grab(rf"\b(?:by|per|across|for each|for every|grouped by)\s+(?:the\s+)?(?P<col>{alt})")
             or grab(rf"\bwith(?:\s+an?|\s+the)?\s+(?P<col>{alt})\s+(?:versus|vs\.?|compared (?:to|with)|and)\s+without\b")
             or grab(rf"\b(?P<col>{alt})\s+(?:versus|vs\.?)\s+(?:no|non|not)\b"))

    rate = (grab(rf"(?P<col>{alt})\s+{_RATE_WORDS}\b")
            or grab(rf"\b{_RATE_WORDS}\s+of\s+(?P<col>{alt})"))
    mean = (grab(rf"\b{_MEAN_WORDS}\s+(?:of\s+)?(?:the\s+)?(?P<col>{alt})")
            or grab(rf"(?P<col>{alt})\s+{_MEAN_WORDS}\b"))
    wants_count = bool(re.search(rf"\b{_COUNT_WORDS}\b", text, re.IGNORECASE))

    measures = [m for m in (rate, mean) if m]
    if len(measures) > 1:
        return None
    others = [c for _p, c in mentioned if c not in (group, rate, mean)]
    if others:
        return None
    if rate:
        if rate == group:
            return None
        return AggregateIntent("rate", rate, group)
    if mean:
        if mean == group:
            return None
        return AggregateIntent("mean", mean, group)
    if wants_count and group:
        return AggregateIntent("count", None, group)
    return None


def quote_ident(name: str, dialect: str) -> str:
    d = normalize_dialect(dialect)
    if d == "mysql":
        return "`" + str(name).replace("`", "``") + "`"
    return '"' + str(name).replace('"', '""') + '"'


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, Decimal)):
        return repr(int(value)) if float(value).is_integer() else repr(float(value))
    return "'" + str(value).replace("'", "''") + "'"


def _binary_levels(values: List[Any]) -> Optional[Tuple[Any, Any]]:
    """``(one, zero)`` literals for a 0/1 or boolean column, else None."""
    if not values or len(values) > 2:
        return None
    if all(isinstance(v, bool) for v in values):
        return True, False
    if all(isinstance(v, (int, float, Decimal)) and not isinstance(v, bool) for v in values):
        if {float(v) for v in values} <= {0.0, 1.0}:
            return 1, 0
        return None
    if all(isinstance(v, str) for v in values):
        lowered = {v.strip().lower() for v in values}
        for one, zero in (("1", "0"), ("true", "false"), ("t", "f"), ("yes", "no"), ("y", "n")):
            if lowered <= {one, zero}:
                ones = [v for v in values if v.strip().lower() == one]
                zeros = [v for v in values if v.strip().lower() == zero]
                return (ones[0] if ones else one), (zeros[0] if zeros else zero)
    return None


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


async def run_pushdown(run_query: RunQuery, table: str, intent: AggregateIntent,
                       dialect: str) -> Optional[PushdownResult]:
    """Run ``intent`` on the whole of ``table``; None when it cannot be done exactly."""
    d = normalize_dialect(dialect)
    if d not in PUSHDOWN_DIALECTS:
        return None
    q = lambda c: quote_ident(c, d)  # noqa: E731

    if intent.operation == "rate":
        probe = await run_query(
            f"SELECT DISTINCT {q(intent.measure)} AS v FROM {table} WHERE {q(intent.measure)} IS NOT NULL LIMIT 3")
        if not isinstance(probe, dict) or probe.get("error"):
            return None
        levels = _binary_levels([r[0] for r in (probe.get("rows") or []) if r])
        if levels is None:
            return None
        one, zero = levels
        expr = (f"AVG(CASE WHEN {q(intent.measure)} = {_literal(one)} THEN 1.0 "
                f"WHEN {q(intent.measure)} = {_literal(zero)} THEN 0.0 END)")
        counted = f"COUNT({q(intent.measure)})"
    elif intent.operation == "mean":
        probe = await run_query(
            f"SELECT {q(intent.measure)} AS v FROM {table} WHERE {q(intent.measure)} IS NOT NULL LIMIT 20")
        if not isinstance(probe, dict) or probe.get("error"):
            return None
        vals = [r[0] for r in (probe.get("rows") or []) if r]
        # A text column would be coerced silently by MySQL; refuse anything non-numeric.
        if not vals or not all(isinstance(v, (int, float, Decimal)) and not isinstance(v, bool) for v in vals):
            return None
        expr = f"AVG({q(intent.measure)})"
        counted = f"COUNT({q(intent.measure)})"
    elif intent.operation == "count":
        expr = "COUNT(*)"
        counted = "COUNT(*)"
    else:
        return None

    overall_res = await run_query(f"SELECT {expr} AS figure, COUNT(*) AS row_count FROM {table}")
    if not isinstance(overall_res, dict) or overall_res.get("error") or not overall_res.get("rows"):
        return None
    overall_row = overall_res["rows"][0]
    total_rows = int(_num(overall_row[1]) or 0)
    overall = _num(overall_row[0])

    groups: List[Tuple[Any, Optional[float], int]] = []
    if intent.group_by:
        g = q(intent.group_by)
        grouped = await run_query(
            f"SELECT {g} AS grp, {expr} AS figure, {counted} AS row_count FROM {table} "
            f"GROUP BY {g} ORDER BY {g} LIMIT {MAX_GROUPS + 1}")
        if not isinstance(grouped, dict) or grouped.get("error"):
            return None
        rows = grouped.get("rows") or []
        if len(rows) > MAX_GROUPS:
            return None
        groups = [(r[0], _num(r[1]), int(_num(r[2]) or 0)) for r in rows]
    return PushdownResult(intent=intent, groups=groups, overall=overall,
                          total_rows=total_rows, dialect=d)


def _fmt(value: Optional[float], operation: str) -> str:
    if value is None:
        return "n/a"
    if operation == "count":
        return f"{int(value):,}"
    return f"{value:.4f}"


def render_pushdown_answer(result: PushdownResult, table: str) -> str:
    """The chat answer: a short lead, the table of figures, and how they were computed."""
    it = result.intent
    label = {"rate": f"{it.measure} rate", "mean": f"mean {it.measure}", "count": "row count"}[it.operation]
    rows_word = "rows with a value" if it.operation != "count" else "rows"
    lines: List[str] = []
    if it.group_by:
        lead = f"**{label.capitalize()} by {it.group_by}**"
        if it.operation != "count" and result.overall is not None:
            lead += f" (overall {_fmt(result.overall, it.operation)})"
        lines.append(lead + ":")
        lines.append("")
        lines.append(f"| {it.group_by} | {label} | {rows_word} |")
        lines.append("|---|---|---|")
        for grp, fig, n in result.groups:
            shown = "(missing)" if grp is None else grp
            lines.append(f"| {shown} | {_fmt(fig, it.operation)} | {n:,} |")
    else:
        lines.append(f"**{label.capitalize()}:** {_fmt(result.overall, it.operation)}")
    lines.append("")
    lines.append(
        f"Computed by the database over all {result.total_rows:,} rows of `{table}`, "
        "so these figures are exact, not a sample estimate."
    )
    return "\n".join(lines)


def pushdown_output_table(result: PushdownResult) -> Dict[str, Any]:
    """The same figures as a ChatResponse output table."""
    it = result.intent
    name = {"rate": f"{it.measure}_rate", "mean": f"mean_{it.measure}", "count": "row_count"}[it.operation]
    if it.group_by:
        rows = [{it.group_by: grp, name: fig, "rows": n} for grp, fig, n in result.groups]
    else:
        rows = [{name: result.overall, "rows": result.total_rows}]
    label = {"rate": f"{it.measure} rate", "mean": f"mean {it.measure}", "count": "row count"}[it.operation]
    return {
        "title": label.capitalize() + (f" by {it.group_by}" if it.group_by else ""),
        "rows": rows, "action_ids": ["db_pushdown"],
        "group_by": [it.group_by] if it.group_by else [], "metric_columns": [name],
    }
