# output_contract.py
"""Does a Ray or scheduled result answer the grouped question it was asked?

Why this module exists (MAJ-030, MAJ-031)
-----------------------------------------
Interactive turns pass through the coding subgraph, where
``validator.contract_validator_node`` checks the executed result against the
blueprint's own contract. Two paths skip that subgraph entirely:

* the batch Ray path (``execution_agent_node_ray``), which runs the coder's code
  or, when there is none, ``_autogenerate_code``'s fallback; and
* scheduled runs (``celery_app._run_agent_task``), which re-run stored code on
  Ray or on the executor pod.

Whatever those paths returned was presented as the answer. "List the lowest
price per category" came back as a per-category row count under a price-like
name (MAJ-031), or without any price column, and nothing noticed.

What it checks
--------------
The question is read deterministically (no LLM) into a small spec: the group
column(s), the measure column, and the aggregation(s) asked for -- min, max,
mean, sum or count. When any part cannot be read with confidence the spec is
incomplete and nothing is asserted: an unstated postcondition is better than a
guessed one (the same rule as ``blueprint_contract``).

Against the result frame, using ``blueprint_contract`` for the structure:

* the group column(s) are present and hold one row per group
  (``BlueprintContract`` with ``row_relation: one_per_group``, 1.6's
  ``check_contract``);
* a column for each requested aggregation of the measure is present and
  numeric (``ColumnSpec`` dtype ``numeric``);
* when a reference is available, each group's figure equals the requested
  aggregation recomputed by pandas on the rows the code was given. Group keys
  are compared with ``_key_text`` and a result holding keys the reference does
  not is not compared at all (``_foreign_group_keys``); code that reads its own
  data is not compared either (``code_reads_own_data``). Those three are
  ported from integration/2.0 @ 360ba64f.

A result that fails is refused in plain, code-written words, and its rows are
not presented as the answer.
"""

from __future__ import annotations

import base64
import inspect
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.agents.blueprint_contract import (
    BlueprintContract,
    ColumnSpec,
    RowRelation,
    _foreign_group_keys,
    _key_text,
    _norm,
    check_contract,
    code_reads_own_data,
)

logger = logging.getLogger(__name__)

AGGS = ("min", "max", "mean", "sum", "count")

#: Words in a question that ask for each aggregation, longest first.
_QUESTION_TERMS = {
    "min": ("least expensive", "minimum", "cheapest", "smallest", "lowest", "min"),
    "max": ("most expensive", "maximum", "priciest", "largest", "highest", "biggest", "max"),
    "mean": ("average", "mean", "avg"),
    "sum": ("summed", "total", "sum"),
    "count": ("how many", "number of", "counts", "count"),
}

#: Fragments that tag a result column with an aggregation ("lowest_price",
#: "price_min", "('price', 'min')", "avgPrice").
_COLUMN_TAGS = {
    "min": ("minimum", "lowest", "smallest", "cheapest", "min"),
    "max": ("maximum", "highest", "largest", "biggest", "max"),
    "mean": ("average", "mean", "avg"),
    "sum": ("total", "sum"),
    "count": ("count", "cnt", "number", "num", "size", "rows", "records", "frequency"),
}

_LABELS = {"min": "minimum", "max": "maximum", "mean": "mean", "sum": "total", "count": "number of rows"}

_GROUP_MARKER = re.compile(
    r"(?<![\w])(?:grouped by|group by|broken down by|for each|for every|in each|per|each|by)\s+(?:the\s+)?",
    re.IGNORECASE,
)
_NOT_A_GROUP_BEFORE = re.compile(r"(?:sort(?:ed)?|order(?:ed)?|rank(?:ed)?)\s*$", re.IGNORECASE)
_TOP_N = re.compile(r"\b(?:top|bottom|first|last)\s+(\d{1,4})\b", re.IGNORECASE)
_NUMERIC_DTYPE = re.compile(r"int|float|double|decimal|numeric|number", re.IGNORECASE)

#: The reference is printed into the RayJob logs; keep it bounded.
REFERENCE_MAX_GROUPS = 2000
REFERENCE_MARKER = "OUTPUT_CONTRACT_REFERENCE="
#: A local input larger than this is not re-read to recompute the reference.
LOCAL_REFERENCE_MAX_BYTES = 512 * 1024 * 1024


# --- the question -----------------------------------------------------------

@dataclass(frozen=True)
class OutputSpec:
    """What a grouped question asks for, read from its words and the schema."""

    group_by: Tuple[str, ...]
    aggs: Tuple[str, ...]
    measure: Optional[str] = None
    top_n: Optional[int] = None

    @property
    def complete(self) -> bool:
        """True when every part needed to check a result was read."""
        if not self.group_by or not self.aggs:
            return False
        return self.measure is not None or set(self.aggs) == {"count"}

    def describe(self) -> str:
        groups = " and ".join(repr(g) for g in self.group_by)
        parts = [_LABELS[a] for a in self.aggs if a != "count"]
        text = []
        if parts:
            names = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
            text.append(f"the {names} of {self.measure!r}")
        if "count" in self.aggs:
            text.append("the number of rows")
        return " and ".join(text) + f" per {groups}"

    def to_dict(self) -> Dict[str, Any]:
        return {"group_by": list(self.group_by), "aggs": list(self.aggs),
                "measure": self.measure, "top_n": self.top_n}


def _dtype_of(schema: Any, column: str) -> str:
    raw = schema.get(column) if isinstance(schema, Mapping) else None
    if isinstance(raw, Mapping):
        raw = raw.get("dtype") or raw.get("type") or ""
    return str(raw or "")


def _phrases(column: str) -> List[str]:
    """How a question may write a column name: as is, spaced, plural, singular."""
    base = str(column).strip().lower()
    spaced = re.sub(r"[_\-]+", " ", base).strip()
    out = set()
    for form in {base, spaced}:
        if not form:
            continue
        out.add(form)
        if form.endswith("y") and not form.endswith(("ay", "ey", "oy", "uy")):
            out.add(form[:-1] + "ies")
        elif form.endswith("s"):
            out.add(form[:-1])
        else:
            out.add(form + "s")
            out.add(form + "es")
    return sorted(out, key=len, reverse=True)


def _mentions(text: str, columns: Sequence[str]) -> List[Tuple[int, int, str]]:
    """(start, end, column) for each column the text names, longest match first."""
    hits = []
    for col in columns:
        for phrase in _phrases(col):
            for m in re.finditer(rf"(?<![\w]){re.escape(phrase)}(?![\w])", text):
                hits.append((m.start(), m.end(), col))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept: List[Tuple[int, int, str]] = []
    for h in hits:
        if not any(k[0] <= h[0] < k[1] for k in kept):
            kept.append(h)
    return kept


def _column_at(text: str, pos: int, columns: Sequence[str]) -> Optional[Tuple[str, int]]:
    best = None
    for col in columns:
        for phrase in _phrases(col):
            if text.startswith(phrase, pos) and not re.match(r"\w", text[pos + len(phrase):pos + len(phrase) + 1]):
                if best is None or len(phrase) > best[1] - pos:
                    best = (col, pos + len(phrase))
    return best


def _question_aggs(text: str) -> List[Tuple[int, int, str]]:
    found = []
    for agg, terms in _QUESTION_TERMS.items():
        for term in terms:
            for m in re.finditer(rf"(?<![\w]){re.escape(term)}(?![\w])", text):
                found.append((m.start(), m.end(), agg))
    found.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept: List[Tuple[int, int, str]] = []
    for h in found:
        if not any(k[0] <= h[0] < k[1] for k in kept):
            kept.append(h)
    return kept


def parse_question(question: str, schema: Any) -> Optional[OutputSpec]:
    """Read the group column(s), measure and aggregation(s) a question asks for.

    Returns None when the question names no aggregation or no group column of
    the schema. The spec's ``measure`` is None when no single numeric column
    can be told apart as the one aggregated; ``complete`` is then False.
    """
    text = " ".join(str(question or "").split("\n\n[Analysis context]", 1)[0].lower().split())
    columns = [str(c) for c in (schema.keys() if isinstance(schema, Mapping) else schema or [])]
    if not text or not columns:
        return None

    hits = _question_aggs(text)
    # "highest average price", "lowest total revenue": the superlative ranks
    # the mean or total, it is not a second aggregation.
    ranked = set()
    for i, (s, e, agg) in enumerate(hits[:-1]):
        nxt = hits[i + 1]
        if agg in ("min", "max") and nxt[2] in ("mean", "sum") and re.fullmatch(r"\s*", text[e:nxt[0]]):
            ranked.add(i)
    # "total number of orders": a count, not a sum.
    for i, (s, e, agg) in enumerate(hits[:-1]):
        nxt = hits[i + 1]
        if agg == "sum" and nxt[2] == "count" and re.fullmatch(r"\s*", text[e:nxt[0]]):
            ranked.add(i)
    aggs: List[str] = []
    for i, (_s, _e, agg) in enumerate(hits):
        if i not in ranked and agg not in aggs:
            aggs.append(agg)
    if not aggs:
        return None

    group_by: List[str] = []
    for m in _GROUP_MARKER.finditer(text):
        if m.group(0).lower().startswith("by") and _NOT_A_GROUP_BEFORE.search(text[:m.start()]):
            continue
        pos = m.end()
        while True:
            hit = _column_at(text, pos, columns)
            if hit is None:
                break
            if hit[0] not in group_by:
                group_by.append(hit[0])
            more = re.match(r"\s*(?:,|and)\s*(?:(?:by|per)\s+)?(?:the\s+)?", text[hit[1]:])
            if not more:
                break
            pos = hit[1] + more.end()
        if group_by:
            break
    if not group_by:
        return None

    measure = None
    if any(a != "count" for a in aggs):
        numeric = [c for c in columns if c not in group_by
                   and (not _dtype_of(schema, c) or _NUMERIC_DTYPE.search(_dtype_of(schema, c)))]
        named = _mentions(text, numeric)
        candidates = list(dict.fromkeys(c for _s, _e, c in named))
        if len(candidates) == 1:
            measure = candidates[0]
        elif len(candidates) > 1:
            # Several numeric columns named: the one right after an aggregation word.
            bound = []
            for s, e, agg in hits:
                if agg == "count":
                    continue
                for ns, _ne, col in named:
                    if ns >= e and re.fullmatch(r"\s*(?:of\s+)?(?:the\s+)?(?:\w+\s+)?", text[e:ns]):
                        bound.append(col)
                        break
            bound = list(dict.fromkeys(bound))
            if len(bound) == 1:
                measure = bound[0]

    top = _TOP_N.search(text)
    return OutputSpec(group_by=tuple(group_by), aggs=tuple(aggs), measure=measure,
                      top_n=int(top.group(1)) if top else None)


def question_from_state(state: Mapping[str, Any]) -> str:
    """The last user message, whether the messages are objects or dicts."""
    for m in reversed(list(state.get("messages") or [])):
        if isinstance(m, Mapping):
            role, content = m.get("role") or m.get("type"), m.get("content")
        else:
            role, content = getattr(m, "type", ""), getattr(m, "content", "")
        if role in ("human", "user"):
            return str(content or "")
    return ""


def spec_from_state(state: Mapping[str, Any]) -> Optional[OutputSpec]:
    schema = state.get("schema") or {}
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except ValueError:
            schema = {}
    spec = parse_question(question_from_state(state), schema)
    return spec if spec is not None and spec.complete else None


# --- the reference ------------------------------------------------------------
# reference_aggregates is also inlined into the Ray script (reference_script),
# so it must stand alone: no module globals, imports inside.

def reference_aggregates(df, spec, max_groups=2000):
    """Per-group size, min, max, mean and sum of the measure, computed by pandas
    on ``df``: the yardstick a result is checked against."""
    import math
    import pandas as pd

    group_by = list(spec.get("group_by") or [])
    measure = spec.get("measure")
    needed = group_by + ([measure] if measure else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return {"error": "missing_input_columns", "columns": missing}
    work = df[needed].copy()
    stats = {}
    if measure:
        values = pd.to_numeric(work[measure], errors="coerce")
        if int(values.isna().sum()) > int(work[measure].isna().sum()):
            # Non-numeric text in the measure: the code must have cleaned it,
            # and the reference cannot know how.
            return {"error": "measure_not_numeric", "columns": [measure]}
        work[measure] = values
    grouped = work.groupby(group_by, dropna=True, sort=True)
    stats["size"] = grouped.size()
    if measure:
        for agg in ("min", "max", "mean", "sum"):
            stats[agg] = grouped[measure].agg(agg)

    def plain(v):
        if v is None:
            return None
        if hasattr(v, "item"):
            try:
                v = v.item()
            except (ValueError, AttributeError):
                pass
        if isinstance(v, float) and math.isnan(v):
            return None
        return v if isinstance(v, (bool, int, float, str)) else str(v)

    index = list(stats["size"].index)
    out = {"rows": int(len(df)), "group_by": group_by, "measure": measure,
           "truncated": len(index) > max_groups, "keys": [], "values": {k: [] for k in stats}}
    for key in index[:max_groups]:
        key_t = key if isinstance(key, tuple) else (key,)
        out["keys"].append([plain(k) for k in key_t])
        for name, series in stats.items():
            out["values"][name].append(plain(series.loc[key]))
    return out


def reference_script(spec: OutputSpec) -> str:
    """Script lines that compute the reference on the Ray script's ``df`` (the
    rows main(df) is about to be given) and print it into the job's logs."""
    return "\n".join([
        "# ---- output contract reference: the question's aggregation, recomputed by pandas ----",
        "# on the same rows main(df) is given (see app/agents/output_contract.py).",
        inspect.getsource(reference_aggregates).rstrip(),
        "",
        "try:",
        "    import base64 as _oc_b64, json as _oc_json",
        f"    _oc_ref = reference_aggregates(df, _oc_json.loads({json.dumps(json.dumps(spec.to_dict()))}),"
        f" max_groups={REFERENCE_MAX_GROUPS})",
        f"    print({REFERENCE_MARKER!r} + _oc_b64.b64encode(_oc_json.dumps(_oc_ref, default=str).encode()).decode())",
        "except Exception as _oc_err:",
        "    print('OUTPUT_CONTRACT_REFERENCE_ERROR=' + type(_oc_err).__name__)",
        "",
    ]) + "\n"


def parse_reference(logs: str) -> Optional[Dict[str, Any]]:
    m = re.search(re.escape(REFERENCE_MARKER) + r"([A-Za-z0-9+/=]+)", logs or "")
    if not m:
        return None
    try:
        return json.loads(base64.b64decode(m.group(1)).decode())
    except Exception:  # noqa: BLE001 - an unreadable reference checks nothing
        logger.warning("[output-contract] could not decode the reference from the job logs")
        return None


def local_reference(spec: OutputSpec, path: Any) -> Optional[Dict[str, Any]]:
    """The reference from a local input file, when it is small enough to re-read."""
    try:
        p = str(path or "")
        if not p or not os.path.isfile(p) or os.path.getsize(p) > LOCAL_REFERENCE_MAX_BYTES:
            return None
        import pandas as pd
        cols = list(spec.group_by) + ([spec.measure] if spec.measure else [])
        if p.lower().endswith(".parquet"):
            frame = pd.read_parquet(p, columns=cols)
        elif p.lower().endswith((".csv", ".tsv", ".txt")):
            try:
                frame = pd.read_csv(p, usecols=lambda c: c in cols, sep="\t" if p.lower().endswith(".tsv") else ",")
            except UnicodeDecodeError:
                frame = pd.read_csv(p, usecols=lambda c: c in cols, encoding="latin-1",
                                    sep="\t" if p.lower().endswith(".tsv") else ",")
        else:
            return None
        return reference_aggregates(frame, spec.to_dict(), max_groups=REFERENCE_MAX_GROUPS)
    except Exception:  # noqa: BLE001
        logger.debug("[output-contract] local reference failed for %s", path, exc_info=True)
        return None


# --- the check ----------------------------------------------------------------

def _tagged(column: str, measure: Optional[str], agg: str) -> bool:
    rest = _norm(column)
    if measure:
        rest = rest.replace(_norm(measure), " ")
    return any(tag in rest for tag in _COLUMN_TAGS[agg])


def resolve_measure_columns(columns: Sequence[str], spec: OutputSpec) -> Dict[str, Optional[str]]:
    """Map each requested aggregation to the result column that holds it."""
    others = [str(c) for c in columns if _norm(c) not in {_norm(g) for g in spec.group_by}]
    out: Dict[str, Optional[str]] = {}
    value_aggs = [a for a in spec.aggs if a != "count"]
    mnorm = _norm(spec.measure) if spec.measure else ""
    measure_cols = [c for c in others if mnorm and mnorm in _norm(c)]
    for agg in value_aggs:
        tagged = [c for c in measure_cols if _tagged(c, spec.measure, agg)]
        if len(tagged) == 1:
            out[agg] = tagged[0]
            continue
        if len(value_aggs) == 1:
            untagged = [c for c in measure_cols
                        if not any(_tagged(c, spec.measure, a) for a in AGGS if a != agg)]
            if len(untagged) == 1:
                out[agg] = untagged[0]
                continue
        out[agg] = None
    if "count" in spec.aggs:
        taken = {c for c in out.values() if c}
        rest = [c for c in others if c not in taken and not (mnorm and mnorm in _norm(c))]
        tagged = [c for c in rest if _tagged(c, None, "count") or _norm(c) == "n"]
        out["count"] = tagged[0] if len(tagged) == 1 else (rest[0] if len(rest) == 1 and len(spec.aggs) == 1 else None)
    return out


def _decimals(value: float) -> int:
    text = repr(float(value))
    if "e" in text or "." not in text:
        return 0
    return min(6, len(text.split(".", 1)[1].rstrip("0")))


def _same_figure(got: Any, want: Any) -> bool:
    try:
        g, w = float(got), float(want)
    except (TypeError, ValueError):
        return got == want
    if math.isnan(g) or math.isnan(w):
        return math.isnan(g) and math.isnan(w)
    if math.isclose(g, w, rel_tol=1e-6, abs_tol=1e-9):
        return True
    # A figure the code rounded: 17.09 for 17.09375.
    d = _decimals(g)
    return d < 6 and abs(g - w) <= 0.5 * 10 ** (-d) + 1e-12


def check_output(spec: OutputSpec, result: Any, reference: Optional[Mapping[str, Any]] = None,
                 code: Optional[str] = None) -> Dict[str, Any]:
    """Check a result frame against what the question asked for.

    Returns ``{"status": "passed"|"refused", "violations": [...],
    "advisories": [...], "values_checked": bool, "spec": {...}}``.
    """
    import pandas as pd

    report: Dict[str, Any] = {"status": "passed", "spec": spec.to_dict(), "violations": [],
                              "advisories": [], "values_checked": False, "columns": {}}
    violations: List[Dict[str, str]] = report["violations"]
    if isinstance(result, pd.Series):
        result = result.to_frame().reset_index()
    if not isinstance(result, pd.DataFrame):
        violations.append({"rule": "result", "detail": "the code did not return a table"})
        report["status"] = "refused"
        return report

    columns = resolve_measure_columns(list(result.columns), spec)
    report["columns"] = dict(columns)
    for agg, col in columns.items():
        if col is None:
            if agg == "count":
                detail = (f"the question asks for the number of rows per "
                          f"{' and '.join(repr(g) for g in spec.group_by)}, but the result has no count "
                          f"column (its columns are: {', '.join(map(str, result.columns))})")
            else:
                detail = (f"the question asks for the {_LABELS[agg]} of {spec.measure!r}, but the result "
                          f"has no {spec.measure!r} column for it (its columns are: "
                          f"{', '.join(map(str, result.columns))})")
            violations.append({"rule": "missing_measure", "detail": detail})

    # Structure, through the blueprint contract: group keys present, one row
    # per group, measure columns numeric.
    contract = BlueprintContract(
        result_kind="aggregate",
        result_columns=tuple(ColumnSpec(name=c, role="metric", dtype="numeric")
                             for c in dict.fromkeys(c for c in columns.values() if c)),
        row_relation=RowRelation(type="one_per_group", group_by=tuple(spec.group_by)),
    )
    for v in check_contract(contract, result, None):
        detail = v.detail
        if v.rule == "missing_group_key":
            detail = (f"the question asks for one row per {' and '.join(repr(g) for g in spec.group_by)}, "
                      f"but the result has no such column (its columns are: {', '.join(map(str, result.columns))})")
        elif v.rule == "duplicate_group_rows":
            detail = detail.replace("the plan promised", "the question asks for")
        elif v.rule == "wrong_dtype":
            detail = detail.replace("was declared numeric", "should hold numbers")
        violations.append({"rule": v.rule, "detail": detail})

    # Figures, against the reference recomputed on the rows the code was given.
    group_cols = [c for c in (next((rc for rc in result.columns if _norm(rc) == _norm(g)), None)
                              for g in spec.group_by)]
    usable = (reference and not reference.get("error") and reference.get("keys") is not None
              and None not in group_cols and not violations)
    if reference and reference.get("error"):
        report["advisories"].append({"rule": "not_recomputed", "detail": (
            f"the figures were not recomputed ({reference['error']}: {', '.join(reference.get('columns') or [])})")})
    elif usable and code and code_reads_own_data(code):
        usable = False
        report["advisories"].append({"rule": "input_not_compared", "detail": (
            "the code reads data itself, so its figures were not compared with a recomputation "
            "on the rows it was given")})
    if usable:
        ref_frame = pd.DataFrame(reference["keys"], columns=list(spec.group_by))
        foreign = _foreign_group_keys(result, group_cols, ref_frame, list(spec.group_by))
        if foreign and not reference.get("truncated"):
            report["advisories"].append({"rule": "input_not_compared", "detail": (
                f"the result holds {foreign} group key(s) the input rows do not, so its figures "
                "were not compared with a recomputation")})
        else:
            index = {tuple(_key_text(k) for k in key): i for i, key in enumerate(reference["keys"])}
            values = reference.get("values") or {}
            stat_of = {"count": "size"}
            for agg, col in columns.items():
                want_all = values.get(stat_of.get(agg, agg))
                if not col or want_all is None:
                    continue
                for row in result[group_cols + [col]].itertuples(index=False, name=None):
                    key = tuple(_key_text(v) for v in row[:-1])
                    if key not in index:
                        continue
                    want = want_all[index[key]]
                    got = row[-1]
                    if _same_figure(got, want):
                        continue
                    same_as = [name for name, vals in values.items()
                               if name != stat_of.get(agg, agg) and _same_figure(got, vals[index[key]])]
                    label = {"size": "row count", "min": "minimum", "max": "maximum",
                             "mean": "mean", "sum": "total"}
                    hint = (f"; its figures are the per-group {label[same_as[0]]}"
                            + (f" of {spec.measure!r}" if same_as[0] != "size" else "")
                            + ", not the " + _LABELS[agg]) if same_as else ""
                    what = (f"the {_LABELS[agg]} of {spec.measure!r}" if agg != "count"
                            else "the number of rows")
                    shown = ", ".join(str(k) for k in row[:-1])
                    violations.append({"rule": "wrong_aggregation", "detail": (
                        f"column {col!r} is not {what}: for {shown} it holds {got}, while "
                        f"{what} recomputed on the same rows is {want}{hint}")})
                    break
            report["values_checked"] = True

    if violations:
        report["status"] = "refused"
    return report


def refusal_text(spec: OutputSpec, report: Mapping[str, Any]) -> str:
    """The plain reply a refused result gets instead of its table."""
    lines = [
        "This result was not shown as an answer, because it does not match the question.",
        f"The question asks for {spec.describe()}.",
    ]
    lines += [f"- {v['detail']}" for v in report.get("violations") or []]
    lines.append("Ask again naming the column to aggregate, or review the analysis code; "
                 "no figure from this run is given as the answer.")
    return "\n".join(lines)


def frame_from_output_file(ofd: Any):
    """The table an ``output_file_data`` CSV holds, or None."""
    if not (isinstance(ofd, Mapping) and str(ofd.get("content", "")).startswith("data:text/csv;base64,")):
        return None
    import io
    import pandas as pd
    try:
        raw = base64.b64decode(str(ofd["content"]).split("base64,", 1)[1]).decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(raw))
    except Exception:  # noqa: BLE001
        return None
