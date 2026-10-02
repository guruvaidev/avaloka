"""
Type-directed rendering of execution results.

The execution backends (local / SSH / k8s / Ray) all hand us the same raw
signals: a status, a captured ``execution_stdout`` (which mixes the real answer
with helper prints, warnings and cluster markers), an ``execution_stderr``
traceback on failure, and — for transformations — an output dataframe/CSV.

This module turns those signals into ONE structured *result artifact* and then
renders it *by type*:

    kind == "scalar"  -> a specific value was asked for  -> conclusion (template)
    kind == "table"   -> a data transformation           -> table + download, NO conclusion
    kind == "empty"   -> ran but produced no rows         -> a short note
    kind == "error"   -> failure                          -> a clean, actionable cause
    kind == "none"    -> ran, produced nothing            -> a neutral note

The conclusion is generated *only* for ``scalar`` results, and it is a
deterministic template by default (zero tokens, works with the LLM disabled).
An LLM phrasing can be layered on later behind a flag — the gate here never
calls one.

The module deliberately depends only on the standard library + pandas (via
duck-typing) so it can be unit-tested without importing the heavy agent stack.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# Sentinel a generated script *may* use to emit an exact, structured result.
# When present we trust it over anything scraped from stdout.
RESULT_START = "<<<AVALOKA_RESULT>>>"
RESULT_END = "<<<END_AVALOKA_RESULT>>>"

# Lines emitted by the injected helpers / Ray wrappers that are pure noise and
# must never appear in a user-facing answer. Matched against each stripped line.
_NOISE_LINE_PATTERNS = [
    re.compile(p)
    for p in (
        r"^Grouping by:",
        r"^Aggregating columns?:",
        r"^Warning: No valid grouping",
        r"^INTERACTIVE_SAMPLE_ANALYSIS_(START|DONE)$",
        r"^USER_CODE_(START|END)$",
        r"^DATA_SOURCE_URI\s*=",
        r"^INPUT_DF_SHAPE\s*=",
        r"^INPUT_COLUMNS\s*=",
        r"^OUTPUT_DF_SHAPE\s*=",
        r"^OUTPUT_ROW_COUNT\s*=",
        r"^SHAPE \(sample\):",
        r"^METRIC\s+\w+\s*=",
        r"^ARTIFACT_URI\s*=",
        r"^METRICS_URI\s*=",
        r"^worker_nodes\s*=",
        r"^cluster_resources\s*=",
        r"^NODES USED",
        r"^NUM NODES",
        r"^TOTAL ROWS PROCESSED",
        r"^Submitted \d+ partition",
        r"^\s*Partition \d+",
        r"^Total file size:",
        r"^Estimated total rows:",
        r"^Total partitions",
        r"^Wrote (gs://|s3://|az://|/)",
    )
]

# Transformation-intent verbs: if ANY of these appear in the user question the
# answer is modified data, not a scalar — even if a result block is present.
# This protects against the coder mistakenly calling avaloka_result() on an
# intermediate stat (e.g. max/min inside a Min-Max normalisation formula).
_TRANSFORM_VERBS = (
    "normalize", "normalise", "normalization", "normalisation",
    "scale", "scaling", "min-max", "minmax", "z-score", "zscore",
    "standardize", "standardise", "log transform", "log-transform",
    "encode", "encoding", "one-hot", "onehot", "label encode",
    "transform", "apply a", "add column", "add a column", "create column",
    "compute column", "new column", "fill null", "fill missing", "impute",
    "drop column", "drop columns", "rename", "merge", "join", "pivot",
    "filter", "clean", "deduplicate", "dedup", "sort", "bin ", "bucket",
    "clip ", "clamp ", "convert ", "replace ",
)

# Math/aggregation intent keywords (longest/most-specific first wins).
_METRIC_KEYWORDS = [
    # First, so "relationship" wins over the "count" that a column like
    # "rating count" matches by accident.
    ("correlation", (
        "correlation", "correlated", "correlate", "corr",
        "relationship", "related", "relate", "relates",
        "association", "associated", "vary with", "varies with",
    )),
    ("mean", ("mean", "average", "avg")),
    ("median", ("median",)),
    ("sum", ("sum", "total")),
    ("count", ("count", "how many", "number of")),
    ("variance", ("variance", "variances")),
    ("std", ("standard deviation", "std")),
    ("max", ("maximum", "max", "highest", "largest")),
    ("min", ("minimum", "min", "lowest", "smallest")),
]

# Wider than this, a 1-row frame is a summary and renders as a table.
_MAX_SCALAR_SUMMARY_COLS = 3

_METRIC_PHRASE = {
    "mean": "The mean{col} is **{val}**.",
    "median": "The median{col} is **{val}**.",
    "sum": "The sum{col} is **{val}**.",
    "count": "The count{col} is **{val}**.",
    "variance": "The variance{col} is **{val}**.",
    "std": "The standard deviation{col} is **{val}**.",
    "max": "The maximum{col} is **{val}**.",
    "min": "The minimum{col} is **{val}**.",
}


# ---------------------------------------------------------------------------
# stdout cleaning + structured-block extraction
# ---------------------------------------------------------------------------

def clean_stdout(stdout: Optional[str]) -> str:
    """Drop known helper/cluster noise lines and the structured block, keeping
    only what looks like the script's real printed output."""
    if not stdout:
        return ""
    # Remove an explicit result block entirely (handled separately).
    text = re.sub(
        re.escape(RESULT_START) + r".*?" + re.escape(RESULT_END),
        "",
        stdout,
        flags=re.DOTALL,
    )
    kept: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if any(p.search(s) for p in _NOISE_LINE_PATTERNS):
            continue
        kept.append(s)
    return "\n".join(kept).strip()


def extract_result_block(stdout: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the JSON object from a ``<<<AVALOKA_RESULT>>>…<<<END>>>`` block,
    or None if absent/unparseable. The last block wins."""
    if not stdout or RESULT_START not in stdout:
        return None
    matches = re.findall(
        re.escape(RESULT_START) + r"(.*?)" + re.escape(RESULT_END),
        stdout,
        flags=re.DOTALL,
    )
    for raw in reversed(matches):
        try:
            obj = json.loads(raw.strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# small parsing/formatting helpers
# ---------------------------------------------------------------------------

def _num(value: Any) -> Optional[Union[int, float]]:
    """Coerce a value to int/float if possible, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if not s:
        return None
    try:
        i = int(s)
        return i
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        return None


def _parse_scalar(text: Optional[str]) -> Optional[Union[int, float]]:
    """If the (cleaned) text is a single numeric token, return it as a number."""
    if not text:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) != 1:
        return None
    return _num(lines[0])


def _fmt(value: Any) -> str:
    """Human-friendly number formatting; pass through non-numbers."""
    n = _num(value)
    if n is None:
        return str(value)
    if isinstance(n, int) or (isinstance(n, float) and n.is_integer() and abs(n) < 1e15):
        return f"{int(n):,}"
    return f"{float(n):,.4f}"


def _detect_metric(question: Optional[str]) -> Optional[str]:
    """Which aggregation, if any, the question is asking for.

    Matches on whole words. Plain substring matching produced false positives
    that are easy to miss: "sum" inside "consumer", "min" inside "terminal",
    "max" inside "maximal_price". Multi-word keywords ("how many") still work --
    \\b applies at each end of the phrase.
    """
    q = (question or "").lower()
    for metric, keywords in _METRIC_KEYWORDS:
        for k in keywords:
            if re.search(rf"\b{re.escape(k.strip())}\b", q):
                return metric
    return None


def _extract_last_number(text: str) -> Optional[Union[int, float]]:
    """Find the last numeric value at the end of a line in (possibly noisy) stdout.
    Used when avaloka_result() was not called but the scalar was printed.
    Matches patterns like "Mean of X: 123.45", "result = 0.99", or bare "42".
    """
    if not text:
        return None
    _pat = re.compile(
        r'(?:^|[:\s=])([-+]?\d{1,20}(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$'
    )
    for line in reversed(text.splitlines()):
        s = line.strip()
        if not s:
            continue
        m = _pat.search(s)
        if m:
            raw = m.group(1).replace(",", "")
            n = _num(raw)
            if n is not None:
                return n
    return None


def _schema_columns(schema: Any) -> List[str]:
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except Exception:
            return []
    if isinstance(schema, dict):
        return [str(c) for c in schema.keys()]
    if isinstance(schema, list):
        return [str(c) for c in schema]
    return []


def _detect_columns(question: Optional[str], schema: Any, limit: int = 2) -> List[str]:
    """Find dataset column names mentioned in the question (longest match first)."""
    q = (question or "").lower()
    if not q:
        return []
    found: List[str] = []
    for col in sorted(_schema_columns(schema), key=len, reverse=True):
        if not col:
            continue
        if re.search(rf"\b{re.escape(col.lower())}\b", q) and col not in found:
            found.append(col)
            if len(found) >= limit:
                break
    return found


# ---------------------------------------------------------------------------
# error cleaning
# ---------------------------------------------------------------------------

# Python raises the same handful of exception TYPES for data problems, and the
# type says what KIND of thing went wrong. Matching on type is a closed set;
# matching on message text (which is what this used to do) is an open one that
# grows a new branch after every incident and still lets the next unseen message
# through as a raw traceback line.
_EXC_EXPLANATIONS: Dict[str, str] = {
    "KeyError": "The analysis referred to a column that isn't in this dataset.",
    "AttributeError": "The analysis used an operation that doesn't apply to that "
                      "kind of column.",
    "ValueError": "The analysis hit values it couldn't work with — usually a "
                  "column holding a different kind of data than the step expected.",
    "TypeError": "The analysis combined two incompatible kinds of value, such as "
                 "text and a number.",
    "ZeroDivisionError": "The analysis divided by zero — a group or filter it "
                         "relied on matched no rows.",
    "IndexError": "The analysis looked for a position that doesn't exist in the data.",
    "MemoryError": "The dataset was too large for this operation to run in memory.",
    "FileNotFoundError": "An input or output file could not be found.",
    "PermissionError": "A file could not be read or written because of permissions.",
    "OverflowError": "A computed number grew too large to represent.",
    "EmptyDataError": "The input file had no data to read.",
    "ParserError": "The input file could not be parsed as the expected format.",
    "MergeError": "A join between two tables could not be performed as specified.",
}

# Narrower, high-confidence readings that let us name the actual cause. These
# refine an explanation; they are never the only thing standing between the user
# and a raw traceback.
_EXC_DETAILS: List[Tuple[str, str]] = [
    ("could not convert string to float",
     "A column used in a numeric calculation contains non-numeric text."),
    ("invalid literal for int",
     "A column used as a whole number contains non-numeric text."),
    ("cannot reindex", "Two tables being combined had mismatched labels."),
    ("length of values", "A new column was built with a different number of "
                         "values than the table has rows."),
    ("unhashable type", "A grouping key was built from something that can't be "
                        "grouped on, such as a list."),
    ("no numeric data to plot", "The columns selected for the chart hold no numbers."),
    ("out-of-bounds", "A date in the data falls outside the supported range."),
    ("unknown string format", "A column expected to hold dates contains values "
                              "that aren't dates."),
    ("not in index", "The analysis referred to a column that isn't in this dataset."),
]

_EXC_LINE_RE = re.compile(r"^([A-Za-z_][\w.]*)(Error|Exception|Warning|Interrupt)\b:?\s*(.*)$")


def _last_exception_line(stderr: str) -> str:
    for ln in reversed(stderr.splitlines()):
        s = ln.strip()
        if not s:
            continue
        if _EXC_LINE_RE.match(s):
            return s
    return stderr.splitlines()[-1].strip() if stderr else ""


def _clean_error(execution_result: Dict[str, Any]) -> str:
    """Turn a raw traceback / error payload into a short, actionable cause.

    Contract of this function: it must NEVER return a bare traceback line. A
    ``ValueError: Length of values (3) does not match length of index (55500)``
    reaching a chat user as-is is the defect this exists to prevent, so the
    fallback path still produces a sentence and keeps the technical detail in
    parentheses for anyone who wants it.
    """
    er = execution_result or {}
    stderr = (er.get("execution_stderr") or er.get("stderr") or "").strip()
    exec_error = (er.get("execution_error") or er.get("message") or "").strip()

    exc_line = _last_exception_line(stderr) or exec_error

    exc_type, message = "", exc_line
    m = _EXC_LINE_RE.match(exc_line)
    if m:
        exc_type = f"{m.group(1)}{m.group(2)}".split(".")[-1]
        message = (m.group(3) or "").strip()

    lowered = exc_line.lower()

    # Timeouts are about budget, not data, and carry their own advice.
    if "timed out" in exec_error.lower() or "TimeoutExpired" in exc_line:
        base = exec_error or "The computation timed out."
        return f"{base} Try a sample or a narrower query."

    parts: List[str] = []
    explanation = _EXC_EXPLANATIONS.get(exc_type)
    if explanation:
        parts.append(explanation)

    for needle, detail in _EXC_DETAILS:
        if needle in lowered:
            parts.append(detail)
            break

    if exc_type == "KeyError" and message:
        parts = [f"The analysis referred to a column {message} that isn't in this dataset."]

    if parts:
        seen: List[str] = []
        for p in parts:
            if p not in seen:
                seen.append(p)
        out = " ".join(seen)
        if message and message not in out:
            out += f" (technical detail: {exc_type}: {message})" if exc_type else f" ({message})"
        return out

    # Nothing recognised. Still a sentence, never a bare traceback line.
    if exc_type:
        return (
            f"The analysis stopped with an unexpected {exc_type}. "
            f"(technical detail: {exc_type}: {message})" if message
            else f"The analysis stopped with an unexpected {exc_type}."
        )
    if exc_line:
        return f"The script did not complete successfully. (technical detail: {exc_line})"
    return "The script did not complete successfully."


# ---------------------------------------------------------------------------
# artifact construction (the type gate)
# ---------------------------------------------------------------------------

def build_artifact(
    execution_result: Optional[Dict[str, Any]],
    output_df: Any = None,
    user_question: Optional[str] = None,
    schema: Any = None,
) -> Dict[str, Any]:
    """Reduce the raw execution signals to one structured artifact whose
    ``kind`` drives rendering. This is the deterministic, zero-token gate that
    decides whether a conclusion is even warranted.

    Priority order (highest wins):
      1. failure      — status==error or non-zero returncode
      2. result block — <<<AVALOKA_RESULT>>> in stdout (coder explicitly signalled
                        a scalar answer even when an output file also exists)
      3. output df    — transformation result (table / empty)
      4. cleaned stdout — bare printed scalar when there is no output file
    """
    er = execution_result or {}
    status = er.get("status")
    rc = er.get("returncode")
    stdout = er.get("execution_stdout") or er.get("stdout") or ""
    has_df = output_df is not None and hasattr(output_df, "empty")

    # 1) Failure takes priority over everything.
    if status == "error" or (rc is not None and rc != 0):
        return {"kind": "error", "error": _clean_error(er), "stdout": stdout}

    # 2) An explicit structured block is always the exact, trusted answer —
    #    UNLESS the question is clearly a transformation (normalize, scale,
    #    encode, filter, …).  In that case the coder misfired avaloka_result()
    #    on an intermediate stat; we skip the block and fall through to the df.
    q_lower = (user_question or "").lower()
    _question_is_transform = any(v in q_lower for v in _TRANSFORM_VERBS)

    block = extract_result_block(stdout)
    if block is not None and not _question_is_transform:
        bkind = block.get("kind")
        if bkind in ("table", "none", "error", "empty"):
            return {"kind": bkind, "result": block, "stdout": stdout}
        return {
            "kind": "scalar",
            "result": block,
            "stdout": stdout,
            # Keep df available so render_response can show it as supplementary.
            "supplementary_df": True if has_df else False,
        }

    # 3) A dataframe output means a transformation (or a tiny scalar result).
    if has_df and not bool(output_df.empty):
        try:
            nrows, ncols = output_df.shape
        except Exception:
            nrows, ncols = None, None

        # 3a) 1-row df → the coder returned a summary — pick the first numeric value.
        #     Handles both 1×1 and 1×N summary DataFrames (e.g. {"mean": 123.45}).
        #     Guard: skip for transformation queries — a 1-row intermediate result
        #     (e.g. just the col_max from a Min-Max formula) must not surface as a scalar.
        if nrows == 1 and not _question_is_transform:
            _ql = (user_question or "").lower()
            # A "which/what/where <category> has the highest/lowest ..." lookup wants
            # the category value named (e.g. "California"), not a number. Only fires
            # when the question names a column whose single value is text; every
            # numeric case falls through to the first-numeric logic below untouched.
            try:
                if re.search(r"\b(which|what|where|who|name)\b", _ql) and re.search(
                    r"\b(highest|lowest|max(?:imum)?|min(?:imum)?|most|least|top|"
                    r"bottom|largest|smallest|greatest|biggest)\b",
                    _ql,
                ):
                    # Match a word inside the column name too: "product" finds
                    # "extreme_product_name", which never appears verbatim.
                    candidates = []
                    for col in output_df.columns:
                        name = str(col).lower().strip()
                        if not name or _num(output_df[col].iloc[0]) is not None:
                            continue
                        label = output_df[col].iloc[0]
                        if not (isinstance(label, str) and label.strip()):
                            continue
                        tokens = [name] + [
                            t for t in re.split(r"[^a-z0-9]+", name) if len(t) >= 4
                        ]
                        if any(re.search(rf"\b{re.escape(t)}\b", _ql) for t in tokens):
                            candidates.append((name, label.strip()))
                    if candidates:
                        # "which product" wants the name, not the id.
                        candidates.sort(key=lambda kv: ("name" not in kv[0], "id" in kv[0]))
                        return {
                            "kind": "scalar",
                            "result": {"raw": f"**{candidates[0][1]}**"},
                            "stdout": stdout,
                        }
            except Exception:
                pass
            try:
                # Whole-word match: a column literally named "a" must not match
                # every question containing the letter a.
                names_in_question = [
                    c for c in output_df.columns
                    if str(c).strip() and re.search(
                        rf"\b{re.escape(str(c).lower().strip())}\b", _ql)
                ]
                # Only collapse to one number when we know which one. Otherwise
                # the first numeric wins arbitrarily and hides the real answer.
                # Wide summary rows carry several related answers (for example
                # group means + correlation + p-value). Do not collapse those
                # to whichever named number happens to sort first.
                if ncols is not None and ncols <= _MAX_SCALAR_SUMMARY_COLS:
                    # Prefer a numeric column the question names.
                    for col in sorted(output_df.columns, key=lambda c: c not in names_in_question):
                        v = _num(output_df[col].iloc[0])
                        if v is not None:
                            return {
                                "kind": "scalar",
                                "result": {"value": v, "columns": [str(col)]},
                                "stdout": stdout,
                            }
            except Exception:
                pass

        # 3b) Multi-row df for an analytical question (no sentinel found): if the
        #     output df looks like the original data echoed back unchanged (its
        #     column set is a superset of the schema), the coder forgot to call
        #     avaloka_result() and just returned df.  Try stdout as the scalar.
        if block is None and not _question_is_transform and _detect_metric(user_question) is not None:
            schema_cols = set(_schema_columns(schema)) if schema else set()
            df_cols = set(str(c) for c in output_df.columns)
            is_passthrough = bool(schema_cols) and df_cols.issuperset(schema_cols)
            if is_passthrough:
                cleaned = clean_stdout(stdout)
                n = _extract_last_number(cleaned)
                if n is not None:
                    return {"kind": "scalar", "result": {"value": n}, "stdout": stdout}
                # No extractable number → fall through to table.
                # Do NOT return raw noisy stdout (dtype prints etc.) as a scalar.

        try:
            return {
                "kind": "table",
                "row_count": int(len(output_df)),
                "columns": [str(c) for c in output_df.columns],
                # A 1-row analytical summary still deserves a sentence, so carry
                # the values through for conclude() to narrate.
                **({"row": {str(c): output_df[c].iloc[0] for c in output_df.columns}}
                   if nrows == 1 else {}),
                "is_transform": _question_is_transform,
                "stdout": stdout,
            }
        except Exception:
            return {"kind": "table", "row_count": None, "columns": [], "stdout": stdout}
    if has_df and bool(output_df.empty):
        return {"kind": "empty", "stdout": stdout}

    # 4) Bare printed scalar (no output file, no structured block).
    cleaned = clean_stdout(stdout)
    if cleaned:
        return {"kind": "scalar", "result": {"raw": cleaned}, "stdout": stdout}

    return {"kind": "none", "stdout": stdout}


# ---------------------------------------------------------------------------
# conclusion (scalar only) + final rendering
# ---------------------------------------------------------------------------


def _fmt_compact(value: Any) -> str:
    """Like _fmt but without trailing zeros: 61.8, not 61.8000."""
    text = _fmt(value)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _conclude_summary_row(row: Dict[str, Any]) -> Optional[str]:
    """Narrate a 1-row analytical summary.

    Answers the comparison outright when the frame carries a pair of comparable
    group figures, then qualifies it with correlation strength and significance
    where those columns exist.
    """
    if not row or len(row) > 12:
        return None

    nums = {str(k): _num(v) for k, v in row.items()}
    nums = {k: v for k, v in nums.items() if v is not None}
    if not nums:
        return None

    parts: List[str] = []

    # A comparable pair: two columns sharing a prefix, e.g.
    # avg_discount_expensive / avg_discount_cheaper.
    pair = None
    keys = list(nums)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ta, tb = a.lower().split("_"), b.lower().split("_")
            if len(ta) > 1 and len(tb) > 1 and ta[:-1] == tb[:-1]:
                pair = (a, b)
                break
        if pair:
            break

    if pair:
        a, b = pair
        va, vb = nums[a], nums[b]
        la, lb = a.split("_")[-1], b.split("_")[-1]
        if va == vb:
            parts.append(f"**{la}** and **{lb}** are equal at {_fmt_compact(va)}")
        else:
            hi, lo = (la, lb) if va > vb else (lb, la)
            hv, lv = (va, vb) if va > vb else (vb, va)
            parts.append(f"**{hi}** is higher: {_fmt_compact(hv)} vs {_fmt_compact(lv)} for {lo}")

    for key, val in nums.items():
        if "correl" in key.lower() and -1.0 <= val <= 1.0:
            a_ = abs(val)
            strength = "no" if a_ < 0.1 else "weak" if a_ < 0.4 else "moderate" if a_ < 0.7 else "strong"
            direction = "positive" if val >= 0 else "negative"
            label = key.replace("_", " ")
            parts.append(f"{label} shows a **{strength} {direction}** relationship (r = {val:.3f})"
                         if strength != "no" else f"{label} shows **no relationship** (r = {val:.3f})")
            break

    for key, val in nums.items():
        if key.lower() in {"p_value", "pvalue", "p"}:
            verdict = "statistically significant" if val < 0.05 else "not statistically significant"
            parts.append(f"the difference is **{verdict}** (p = {val:.4g})")
            break

    if not parts:
        # No recognisable shape: still better to name the figures than show a bare row.
        shown = ", ".join(f"{k.replace('_', ' ')} **{_fmt_compact(v)}**" for k, v in list(nums.items())[:4])
        return f"{shown}." if shown else None

    return "; ".join(parts).capitalize() + "."


def conclude(
    artifact: Dict[str, Any],
    user_question: Optional[str] = None,
    schema: Any = None,
) -> Optional[str]:
    """Build a one-line natural-language conclusion for a *scalar* result.
    Returns None for every other kind — that is the gate that stops
    transformations from ever producing a conclusion."""
    kind = (artifact or {}).get("kind")
    if kind == "table":
        # A 1-row analytical summary (group means, correlation, p-value) is an
        # answer, not a dataset. Without this the user gets an unlabelled row.
        if artifact.get("is_transform"):
            return None
        return _conclude_summary_row(artifact.get("row") or {})
    if kind != "scalar":
        return None

    result = artifact.get("result") or {}
    metric = result.get("kind") or _detect_metric(user_question)
    columns = [str(c) for c in (result.get("columns") or [])]
    if not columns:
        columns = _detect_columns(user_question, schema)

    value = result.get("value")
    if value is None:
        value = _parse_scalar(result.get("raw"))

    # Correlation gets an interpreted strength/direction, still token-free.
    if metric == "correlation":
        n = _num(value)
        if n is not None and -1.0 <= float(n) <= 1.0:
            a = abs(float(n))
            strength = "no" if a < 0.1 else "weak" if a < 0.4 else "moderate" if a < 0.7 else "strong"
            direction = "positive" if float(n) >= 0 else "negative"
            pair = f" between `{columns[0]}` and `{columns[1]}`" if len(columns) >= 2 else ""
            if strength == "no":
                return f"There is **no significant correlation**{pair} (r = {float(n):.3f})."
            return f"There is a **{strength} {direction}** correlation{pair} (r = {float(n):.3f})."

    if value is not None:
        # If the output column was named after the metric itself (e.g. the coder
        # named its result column "variance_profit"), drop that redundant token so
        # it reads "The variance of Profit", not "of variance_profit". Only the
        # metric's own name is stripped — ordinary columns like "Profit" or
        # "Total Sales" are never touched.
        if len(columns) == 1:
            _kept = [t for t in re.split(r"[\s_\-]+", columns[0])
                     if t and t.lower() != (metric or "").lower()]
            col_txt = f" of `{' '.join(_kept)}`" if _kept else ""
        else:
            col_txt = ""
        template = _METRIC_PHRASE.get(metric or "")
        if template:
            return template.format(col=col_txt, val=_fmt(value))
        return f"The result{col_txt} is **{_fmt(value)}**."

    # Non-numeric printed answer (e.g. value_counts): present it denoised, as-is.
    raw = result.get("raw")
    return raw or None


def render_response(
    execution_result: Optional[Dict[str, Any]],
    output_df: Any = None,
    user_question: Optional[str] = None,
    schema: Any = None,
    max_table_rows: int = 100,
) -> str:
    """Produce the user-facing message body for an execution outcome.

    Never raises — on any internal error it falls back to the (cleaned) stdout
    so the calling node always has something to show.
    """
    try:
        artifact = build_artifact(execution_result, output_df, user_question, schema)
        kind = artifact.get("kind")

        if kind == "error":
            return f"❌ **Execution failed:** {artifact.get('error')}"

        if kind == "empty":
            return "The query ran successfully but returned no rows."

        if kind == "table":
            conclusion = conclude(artifact, user_question, schema)
            if conclusion:
                return conclusion
            try:
                total = len(output_df)
                preview = output_df.head(max_table_rows)
                table_md = preview.to_markdown(index=False)
            except Exception:
                total = artifact.get("row_count")
                table_md = ""
            cols = artifact.get("columns") or []
            count_txt = f"{total:,}" if isinstance(total, int) else "?"
            note = f"Transformed data — {count_txt} rows × {len(cols)} columns."
            if isinstance(total, int) and total > max_table_rows:
                note += f" Showing the first {max_table_rows}; full result in the download."
            return f"{note}\n\n{table_md}".strip()

        if kind == "scalar":
            conclusion = conclude(artifact, user_question, schema)
            if conclusion:
                return conclusion
            # Fall through to a denoised stdout if we couldn't phrase anything.
            return clean_stdout(artifact.get("stdout")) or "Code executed successfully."

        # kind == "none"
        return clean_stdout(artifact.get("stdout")) or "Code executed successfully (no output produced)."

    except Exception:
        # Last-resort safety net — never break the execution node.
        cleaned = clean_stdout((execution_result or {}).get("execution_stdout"))
        return cleaned or "Code executed."
