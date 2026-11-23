"""What the recipes share: binding helpers, plain-JSON chart builders, the
notebook cells every recipe writes, and the date-convention detector.

Nothing here reads a model. Chart options are built by hand so they are plain
JSON (no functions), and every label passes :func:`safe_label` so a category
named ``<script>`` draws as text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from avaloka import notebook as _nb
from avaloka.domains import normalise
from avaloka.outcomes.semantics import ColumnSemantics
from avaloka.outcomes.templates import PALETTE, build_waterfall, safe_label

from ..base import Ask, Bound, GateReport, Intent, Refusal
from ..budget import DEFAULT as DEFAULT_BUDGET
from ..engine import RENAME_SQL, SAMPLE_SQL, ident, is_parquet

TOP_GROUPS = 12          # describe: groups shown by count; the rest are "other"
OTHER = "other"
DEFAULT_SOURCE = "data.csv"


# ── binding ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RecipeBound(Bound):
    """A :class:`Bound` that also knows where the data came from, so the
    notebook can write ``DATA_SOURCE`` without a second argument."""
    source: str = DEFAULT_SOURCE


def columns_of_role(sem: Mapping[str, ColumnSemantics], *roles: str) -> list[str]:
    return [n for n, c in sem.items() if c.role in roles]


def bind_column(intent: Intent, sem: Mapping[str, ColumnSemantics], name: str, roles: Sequence[str],
                *, required: bool, missing_reason: str,
                question: str) -> str | None | Ask | Refusal:
    """Resolve one column parameter: the intent's value when it names a column
    of the right role; else the single candidate; else an Ask (several) or a
    Refusal (none, when required)."""
    value = intent.params.get(name)
    if isinstance(value, str):
        if value in sem and sem[value].role in roles:
            return value
        return Refusal(missing_reason, f"The {name} column named in the question is not a {roles[0]} column.")
    if not required:
        return None
    candidates = columns_of_role(sem, *roles)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return Refusal(missing_reason, f"This data has no {roles[0]} column.")
    return Ask(name, question, tuple(candidates[:8]))


def source_of(profile: Mapping[str, Any]) -> str:
    return str(profile.get("source") or DEFAULT_SOURCE)


def rows_of(profile: Mapping[str, Any]) -> int:
    return int(profile.get("rows") or profile.get("n_rows") or 0)


# ── evidence leaves ────────────────────────────────────────────────────────

def leaf(value: Any, unit: str) -> dict[str, Any]:
    return {"value": float(value), "unit": unit}


def figure_unit(measure_unit: str | None) -> str:
    """The figure unit a measure's statistics are shown in."""
    return {"currency": "currency", "percent": "percent", "fraction": "percent"}.get(measure_unit or "", "unitless")


def evidence_unit(measure_unit: str | None) -> str:
    """The evidence unit a measure's statistics are stored in (a fraction stays a fraction)."""
    return {"currency": "currency", "percent": "percent", "fraction": "fraction"}.get(measure_unit or "", "unitless")


def label(name: str) -> str:
    return normalise(name) or str(name)


# ── charts (plain JSON) ────────────────────────────────────────────────────

def _option(**extra: Any) -> dict[str, Any]:
    opt = {"aria": {"enabled": True, "decal": {"show": True}}, "color": PALETTE,
           "tooltip": {"trigger": "item", "confine": True}, "animation": True}
    opt.update(extra)
    return opt


def _cell(v: Any) -> Any:
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "item"):  # numpy scalar
        return _cell(v.item())
    return str(v)


def table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> dict[str, Any]:
    return {"columns": [str(c) for c in columns][:12],
            "rows": [[_cell(v) for v in list(r)[:12]] for r in list(rows)[:500]]}


def alt(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return (text if len(text) >= 20 else text + " Data table below.")[:400]


def bar_chart(cid: str, title: str, categories: Sequence[str], values: Sequence[float], *,
              x_name: str, y_name: str, alt_text: str, question_type: str = "ranking") -> dict[str, Any]:
    option = _option(xAxis={"type": "category", "data": [safe_label(c) for c in categories], "name": x_name},
                     yAxis={"type": "value", "name": y_name},
                     series=[{"type": "bar", "data": [_cell(v) for v in values]}])
    return {"id": cid, "title": title[:80], "chart_kind": "bar", "question_type": question_type,
            "option": option, "alt_text": alt(alt_text),
            "data_table": table([x_name, y_name], zip(categories, values))}


def histogram_chart(cid: str, title: str, bins: Sequence[str], counts: Sequence[int], *,
                    x_name: str, alt_text: str) -> dict[str, Any]:
    option = _option(xAxis={"type": "category", "data": [safe_label(b) for b in bins], "name": x_name},
                     yAxis={"type": "value", "name": "rows"},
                     series=[{"type": "bar", "barCategoryGap": "2%", "data": [int(c) for c in counts]}])
    return {"id": cid, "title": title[:80], "chart_kind": "histogram", "question_type": "distribution",
            "option": option, "alt_text": alt(alt_text),
            "data_table": table(["bin", "rows"], zip(bins, [int(c) for c in counts]))}


def boxplot_chart(cid: str, title: str, groups: Sequence[str], stats: Sequence[Sequence[float]], *,
                  group_name: str, y_name: str, alt_text: str) -> dict[str, Any]:
    """``stats`` rows are ``[min, q1, median, q3, max]`` per group."""
    option = _option(xAxis={"type": "category", "data": [safe_label(g) for g in groups], "name": group_name},
                     yAxis={"type": "value", "name": y_name},
                     series=[{"type": "boxplot", "data": [[_cell(v) for v in s] for s in stats]}])
    return {"id": cid, "title": title[:80], "chart_kind": "boxplot", "question_type": "distribution",
            "option": option, "alt_text": alt(alt_text),
            "data_table": table([group_name, "min", "q1", "median", "q3", "max"],
                                [[g, *s] for g, s in zip(groups, stats)])}


def line_chart(cid: str, title: str, periods: Sequence[str], series: Mapping[str, Sequence[Any]], *,
               x_name: str, y_name: str, alt_text: str, series_name: str | None = None) -> dict[str, Any]:
    """One line per entry of ``series`` (name -> values aligned with ``periods``)."""
    option = _option(xAxis={"type": "category", "data": [safe_label(p) for p in periods], "name": x_name},
                     yAxis={"type": "value", "name": y_name},
                     legend={"show": len(series) > 1}, tooltip={"trigger": "axis", "confine": True},
                     series=[{"type": "line", "name": safe_label(name), "data": [_cell(v) for v in values]}
                             for name, values in series.items()])
    rows: list[list[Any]] = []
    for name, values in series.items():
        for p, v in zip(periods, values):
            rows.append([p] + ([name] if series_name else []) + [v])
    return {"id": cid, "title": title[:80], "chart_kind": "line", "question_type": "trend",
            "option": option, "alt_text": alt(alt_text),
            "data_table": table([x_name] + ([series_name] if series_name else []) + [y_name], rows)}


def waterfall_chart(cid: str, title: str, start: tuple[str, float], steps: Sequence[tuple[str, float]], *,
                    measure_label: str) -> dict[str, Any]:
    """A bridge from ``start`` through signed ``steps`` to the end total,
    drawn by :func:`avaloka.outcomes.templates.build_waterfall`; the keys the
    report schema does not know (``facts``, the table's ``_cut``) are dropped."""
    chart = build_waterfall(list(steps), start=start, title=title[:80], measure_label=measure_label)
    chart.pop("facts", None)
    chart["data_table"].pop("_cut", None)
    chart["data_table"]["rows"] = [[_cell(v) for v in r] for r in chart["data_table"]["rows"]]
    return {"id": cid, **chart}


def scatter_chart(cid: str, title: str, points: Sequence[Sequence[float]], *, x_name: str, y_name: str,
                  alt_text: str) -> dict[str, Any]:
    pts = [[_cell(a), _cell(b)] for a, b in points]
    option = _option(xAxis={"type": "value", "name": x_name, "scale": True},
                     yAxis={"type": "value", "name": y_name, "scale": True},
                     series=[{"type": "scatter", "symbolSize": 5, "data": pts}])
    return {"id": cid, "title": title[:80], "chart_kind": "scatter", "question_type": "relationship",
            "option": option, "alt_text": alt(alt_text), "data_table": table([x_name, y_name], pts)}


def heatmap_chart(cid: str, title: str, labels: Sequence[str], matrix: Sequence[Sequence[float]], *,
                  value_name: str, alt_text: str) -> dict[str, Any]:
    """A square matrix (``matrix[i][j]`` for ``labels[i]`` against ``labels[j]``)."""
    cats = [safe_label(c) for c in labels]
    data = [[i, j, _cell(matrix[i][j])] for i in range(len(labels)) for j in range(len(labels))]
    option = _option(xAxis={"type": "category", "data": cats}, yAxis={"type": "category", "data": cats},
                     visualMap={"min": -1, "max": 1, "calculable": True, "orient": "horizontal", "left": "center"},
                     series=[{"type": "heatmap", "data": data, "label": {"show": True}}])
    rows = [[labels[i], labels[j], matrix[i][j]] for i in range(len(labels)) for j in range(len(labels))]
    return {"id": cid, "title": title[:80], "chart_kind": "heatmap", "question_type": "relationship",
            "option": option, "alt_text": alt(alt_text), "data_table": table(["row", "column", value_name], rows)}


def cohort_heatmap_chart(cid: str, title: str, cohorts: Sequence[str], periods: Sequence[str],
                         matrix: Sequence[Sequence[float | None]], *, value_name: str, alt_text: str) -> dict[str, Any]:
    """A cohort × period-number matrix (``matrix[i][j]`` for ``cohorts[i]`` at
    ``periods[j]``; None where the period is not yet observable)."""
    rows_l, cols_l = [safe_label(c) for c in cohorts], [safe_label(p) for p in periods]
    data = [[j, i, _cell(matrix[i][j])] for i in range(len(cohorts)) for j in range(len(periods))
            if matrix[i][j] is not None]
    option = _option(xAxis={"type": "category", "data": cols_l, "name": "period"},
                     yAxis={"type": "category", "data": rows_l, "name": "cohort"},
                     visualMap={"min": 0, "max": 1, "calculable": True, "orient": "horizontal", "left": "center"},
                     series=[{"type": "heatmap", "data": data, "label": {"show": True}}])
    rows = [[cohorts[i], periods[j], matrix[i][j]] for i in range(len(cohorts)) for j in range(len(periods))
            if matrix[i][j] is not None]
    return {"id": cid, "title": title[:80], "chart_kind": "cohort_heatmap", "question_type": "cohort",
            "option": option, "alt_text": alt(alt_text), "data_table": table(["cohort", "period", value_name], rows)}


def hbar_interval_chart(cid: str, title: str, categories: Sequence[str], values: Sequence[float],
                        lows: Sequence[float], highs: Sequence[float], *, x_name: str, y_name: str,
                        alt_text: str) -> dict[str, Any]:
    """A horizontal bar per category with its interval drawn as two markers."""
    cats = [safe_label(c) for c in categories]
    option = _option(xAxis={"type": "value", "name": x_name}, yAxis={"type": "category", "data": cats, "name": y_name},
                     legend={"show": True},
                     series=[{"type": "bar", "name": x_name, "data": [_cell(v) for v in values]},
                             {"type": "scatter", "name": "interval low", "symbol": "rect", "symbolSize": [2, 14],
                              "data": [[_cell(v), i] for i, v in enumerate(lows)]},
                             {"type": "scatter", "name": "interval high", "symbol": "rect", "symbolSize": [2, 14],
                              "data": [[_cell(v), i] for i, v in enumerate(highs)]}])
    return {"id": cid, "title": title[:80], "chart_kind": "bar", "question_type": "ranking",
            "option": option, "alt_text": alt(alt_text),
            "data_table": table([y_name, x_name, "low", "high"], zip(categories, values, lows, highs))}


def flagged_line_chart(cid: str, title: str, labels: Sequence[str], values: Sequence[float],
                       flagged: Sequence[bool], *, x_name: str, y_name: str, alt_text: str,
                       kind: str = "line") -> dict[str, Any]:
    """The series with its flagged points as a second series (``kind`` is
    ``line`` over periods or ``scatter`` over a row index)."""
    cats = [safe_label(p) for p in labels]
    base = {"type": kind, "name": y_name, "data": [_cell(v) for v in values]}
    if kind == "line":
        base.update(showSymbol=False)
    marks = {"type": "scatter", "name": "flagged", "symbolSize": 9,
             "data": [[i, _cell(v)] for i, (v, f) in enumerate(zip(values, flagged)) if f]}
    option = _option(xAxis={"type": "category", "data": cats, "name": x_name},
                     yAxis={"type": "value", "name": y_name, "scale": True},
                     legend={"show": True}, tooltip={"trigger": "axis", "confine": True}, series=[base, marks])
    rows = [[p, v, int(bool(f))] for p, v, f in zip(labels, values, flagged)]
    return {"id": cid, "title": title[:80], "chart_kind": kind, "question_type": "trend" if kind == "line" else "distribution",
            "option": option, "alt_text": alt(alt_text), "data_table": table([x_name, y_name, "flagged"], rows)}


# ── column names ───────────────────────────────────────────────────────────

#: Name tokens that mark a numeric column as a label, not a quantity (the rule
#: of ``avaloka.goal_analysis.is_categorical_code``, which this package may not
#: import because that module reaches into the CLI).
ID_TOKENS = frozenset({"id", "ids", "code", "codes", "key", "number", "no", "num", "uuid"})


def name_tokens(name: Any) -> set[str]:
    """``RatecodeID`` -> {ratecode, id}; ``order_number`` -> {order, number}."""
    return {t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(name))}


def is_id_like(name: str, sem: Mapping[str, ColumnSemantics]) -> bool:
    """An id by role, or a measure whose name says it is an identifier."""
    col = sem.get(name)
    return (col is not None and col.role == "id") or bool(name_tokens(name) & ID_TOKENS)


# ── notebook cells ─────────────────────────────────────────────────────────

def header_cell(recipe_id: str, version: str, title: str) -> dict[str, Any]:
    return _nb.md(f"# {title}\n\nRecipe `{recipe_id}`, version {version}. Generated by Avaloka; "
                  "every number below is recomputed from `DATA_SOURCE` when the notebook runs.")


def parameters_cell(bound: RecipeBound, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The papermill ``parameters`` cell: the source, every bound parameter,
    the recipe id and version, and any derived parameter the recipe needs."""
    lines = [f"DATA_SOURCE = {bound.source!r}",
             # None: the whole source. A sampled run writes the sample's size here
             # (avaloka.recipes.run), and the load cell takes the same sample.
             "SAMPLE_ROWS = None", f"SAMPLE_SEED = {DEFAULT_BUDGET.seed!r}"]
    lines += [f"{name} = {value!r}" for name, value in bound.params.items()]
    lines += [f"{name} = {value!r}" for name, value in (extra or {}).items()]
    lines += [f"RECIPE = {bound.recipe_id!r}", f"RECIPE_VERSION = {bound.version!r}"]
    return _nb.code("\n".join(lines), tags=["parameters"])


def load_cell(source: str) -> dict[str, Any]:
    """``t`` is the source, or — when ``SAMPLE_ROWS`` is set — the same sample
    of it the recipe analysed: the statement is ``engine.SAMPLE_SQL`` with the
    notebook's parameters in the template's slots, so the two cannot drift."""
    reader = "read_parquet" if is_parquet(source) else "read_csv"
    sample_sql = SAMPLE_SQL.replace("{rows}", "{SAMPLE_ROWS}").replace("{seed}", "{SAMPLE_SEED}")
    return _nb.code("import duckdb, pandas as pd\n"
                    "con = duckdb.connect()\n"
                    f'con.{reader}(DATA_SOURCE).create_view("t")\n'
                    "if SAMPLE_ROWS is not None:  # the report was computed on this sample of the rows\n"
                    f'    con.execute(f"{sample_sql}")\n'
                    '    con.execute("DROP VIEW t")\n'
                    f'    con.execute("{RENAME_SQL}")\n\n'
                    "def ident(name):\n"
                    "    return '\"' + str(name).replace('\"', '\"\"') + '\"'")


def sql_cell(name: str, sql: str, call: str) -> dict[str, Any]:
    """``NAME = '''<the same SQL string the recipe ran>'''`` followed by the call that runs it."""
    return _nb.code(f"{name} = {sql!r}\n{call}")


def limitations_cell(gates: GateReport | None, words: Mapping[str, str]) -> dict[str, Any]:
    lines = ["## Limitations", ""]
    if gates is None or not gates.gates:
        lines.append("- No quality gate was recorded for this run.")
    for g in (gates.gates if gates else ()):
        if g.status == "passed":
            lines.append(f"- `{g.name}`: passed.")
        elif g.status == "qualified":
            lines.append(f"- `{g.name}`: qualified ({g.caveat_kind}). {words.get(g.name, '')}")
        else:
            lines.append(f"- `{g.name}`: failed ({g.reason}).")
    return _nb.md("\n".join(lines))


#: What every evidence cell starts with: the leaf shape the report uses,
#: the ``_num`` rule, the evidence unit of a measure, and the two leaf groups
#: nearly every recipe reports (row counts, unparsed dates).
EVIDENCE_PREAMBLE = """\
import json

def leaf(value, unit):
    return {"value": float(value), "unit": unit}

def num(v):
    return 0.0 if v is None or pd.isna(v) else float(v)

def evidence_unit(measure_unit):
    return {"currency": "currency", "percent": "percent", "fraction": "fraction"}.get(measure_unit or "", "unitless")

def rows_leaves(counts):
    c = counts.iloc[0]
    return {"raw": leaf(c["rows_raw"], "count"), "distinct": leaf(c["rows_distinct"], "count"),
            "duplicates": leaf(int(c["rows_raw"]) - int(c["rows_distinct"]), "count")}

def unparsed_leaves(parse):
    non_null, parsed = int(parse.iloc[0]["non_null"]), int(parse.iloc[0]["parsed"])
    return {"unparsed_share": leaf((non_null - parsed) / non_null if non_null else 1.0, "fraction")}
"""


def evidence_cell(body: str) -> dict[str, Any]:
    """The last code cell, tagged ``evidence``: ``body`` builds ``evidence``,
    a dict with the same JSON-pointer paths and ``{"value", "unit"}`` leaves
    as the report's evidence tree for every leaf a figure cites, from what
    the cells above computed; the cell prints it as one JSON line so the
    parity test (and any runner) can read it back."""
    return _nb.code(EVIDENCE_PREAMBLE + "\n" + body.rstrip() + "\n\nprint(json.dumps(evidence, default=float))",
                    tags=["evidence"])


# ── date conventions ───────────────────────────────────────────────────────

_SLASH_DATE = re.compile(r"^\s*(\d{1,2})([/\-.])(\d{1,2})\2(\d{2}|\d{4})\s*$")
_ISO_DATE = re.compile(r"^\s*\d{4}-\d{1,2}-\d{1,2}")

CONVENTIONS = ("iso", "day_first", "month_first", "ambiguous", "unknown")


def date_convention(values: Iterable[Any]) -> tuple[str, str]:
    """``(convention, separator)`` of a text date column.

    Looks only at values written as two short fields and a year: a value whose
    first field exceeds twelve can only be day-first; one whose second field
    exceeds twelve can only be month-first. Both kinds present → ``ambiguous``
    (the recipe asks). Neither kind present with slash dates → ``ambiguous``
    too: nothing in the data decides. ISO-only text is ``iso``.
    """
    day_first = month_first = slashed = iso = 0
    seps: dict[str, int] = {}
    for v in values:
        if v is None or (isinstance(v, float) and v != v):
            continue
        s = str(v)
        if _ISO_DATE.match(s):
            iso += 1
            continue
        m = _SLASH_DATE.match(s)
        if not m:
            continue
        slashed += 1
        seps[m.group(2)] = seps.get(m.group(2), 0) + 1
        first, second = int(m.group(1)), int(m.group(3))
        if first > 12:
            day_first += 1
        if second > 12:
            month_first += 1
    sep = max(seps, key=seps.get) if seps else "/"
    if not slashed:
        return ("iso" if iso else "unknown"), sep
    if day_first and month_first:
        return "ambiguous", sep
    if day_first:
        return "day_first", sep
    if month_first:
        return "month_first", sep
    return "ambiguous", sep


def strptime_format(convention: str, sep: str, values: Iterable[Any]) -> str | None:
    """The DuckDB strptime format for an unambiguous slash convention, else None."""
    if convention not in ("day_first", "month_first"):
        return None
    years = {len(m.group(4)) for m in (_SLASH_DATE.match(str(v)) for v in values if v is not None) if m}
    year = "%y" if years == {2} else "%Y"
    order = ("%d", "%m") if convention == "day_first" else ("%m", "%d")
    return sep.join(order) + sep + year


def profile_date_convention(s: pd.Series) -> dict[str, Any] | None:
    """What :func:`avaloka.recipes.run.build_profile` records for a text column
    that holds dates: the convention and the parse format, so ``bind`` can
    ask about an ambiguous column without reading rows itself."""
    if s.dtype != object and not pd.api.types.is_string_dtype(s):
        return None
    values = s.dropna().astype(str).head(5000).tolist()
    convention, sep = date_convention(values)
    if convention == "unknown":
        return None
    return {"convention": convention, "format": strptime_format(convention, sep, values)}
