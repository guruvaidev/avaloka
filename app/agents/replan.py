"""Recover from an analysis that failed while it ran: diagnose, repair, run again.

Before this module, a failure of the generated code on the full data ended the
turn: ``route_after_execution`` sent any failed run to the verifier and the
user read "I couldn't complete this step". The coder/validator loop never sees
the full run -- it executes on the sample -- and most data problems (a price
column of ``"$1,234.50"`` strings, twelve blank revenues in row 4,000) only
show there.

The ``replan`` node runs after a failed local (or executor) run and does one
of three things:

* **data step** -- the error is the data's: code picks a repair from a closed
  list (:data:`TRANSFORMS`), applies it to a per-turn copy of the input, and
  the same code runs again on that copy. The source dataset is never written;
* **code fix** -- the error is the code's: the coder gets the error as
  feedback and the prior code, and its fix passes the validators again;
* **stop** -- the failure cannot be repaired here (memory, time limit, an
  unreadable file), a repair would exceed a limit and needs the user's word,
  or the attempts are spent. The reply says what was tried, in plain words.

Three rules hold throughout.

**Code decides; the model may only be asked which kind of error it is.** The
diagnosis is deterministic first: the exception class and message, checked
against a profile of the frames the run read. Only when no rule matches is a
model asked, and its answer is a closed-vocabulary verdict that code checks
against the profile. Fill values, methods, rows and counts are always code's.

**Nothing is applied silently.** Every repair is a record with counts measured
on the frame before and after it, and the reply carries one code-written
sentence per repair (:mod:`app.agents.replan_disclosure`), as a plain segment.

**Nothing about the data leaves the process.** The attempt records carry
column names (the user's own run record); :func:`recovery_telemetry` is the
only projection meant for telemetry and carries closed vocabularies and
counts of attempts, never a column, a value or an error message.

Scope (v1): the ``execute_locally`` node, which also covers the sandboxed
executor when one is configured. Ray, Kubernetes and scheduled runs are not
routed here yet.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Limits. Each is configurable; each is a judgement, so each is named.
# --------------------------------------------------------------------------- #

#: Re-executions per turn. 0 turns the loop off.
DEFAULT_MAX_ATTEMPTS = 3
#: No configuration raises the bound above this.
HARD_MAX_ATTEMPTS = 5
#: A column more than this share missing is not filled without asking.
DEFAULT_MAX_IMPUTE_SHARE = 0.10
#: Rows left out, across every repair of the turn, before asking.
DEFAULT_MAX_EXCLUDE_SHARE = 0.05
#: Columns repaired per turn before asking.
DEFAULT_MAX_COLUMNS = 5
#: Extra wall time the loop may add to a turn.
DEFAULT_BUDGET_S = 120.0
#: Model calls the loop may add to a turn (diagnosis + coder + reviewer).
DEFAULT_MAX_LLM_CALLS = 6
#: A code fix is estimated as one coder call and one reviewer call.
CODE_FIX_LLM_CALLS = 2
#: A text column is numeric "in a costume" at or above this parse rate
#: (app.agents.preparation_agent.NUMERIC_COERCION_CONFIDENCE).
COERCION_CONFIDENCE = 0.90

#: What to do with missing values in the column the calculation measures.
#: The owner's ruling: ASK (app/agents/replan_choice.py). "exclude" and
#: "impute" are test hooks only (AVALOKA_REPLAN_MEASURED_POLICY), not settings.
MEASURED_ASK = "ask"
MEASURED_EXCLUDE = "exclude"
MEASURED_IMPUTE = "impute"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Limits:
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    max_impute_share: float = DEFAULT_MAX_IMPUTE_SHARE
    max_exclude_share: float = DEFAULT_MAX_EXCLUDE_SHARE
    max_columns: int = DEFAULT_MAX_COLUMNS
    budget_s: float = DEFAULT_BUDGET_S
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    measured_policy: str = MEASURED_ASK

    @staticmethod
    def from_env() -> "Limits":
        policy = os.getenv("AVALOKA_REPLAN_MEASURED_POLICY", MEASURED_ASK).strip().lower()
        return Limits(
            max_attempts=max(0, min(_env_int("AVALOKA_REPLAN_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS),
                                    HARD_MAX_ATTEMPTS)),
            max_impute_share=_env_float("AVALOKA_REPLAN_MAX_IMPUTE_SHARE", DEFAULT_MAX_IMPUTE_SHARE),
            max_exclude_share=_env_float("AVALOKA_REPLAN_MAX_EXCLUDE_SHARE", DEFAULT_MAX_EXCLUDE_SHARE),
            max_columns=_env_int("AVALOKA_REPLAN_MAX_COLUMNS", DEFAULT_MAX_COLUMNS),
            budget_s=_env_float("AVALOKA_REPLAN_BUDGET_S", DEFAULT_BUDGET_S),
            max_llm_calls=_env_int("AVALOKA_REPLAN_MAX_LLM_CALLS", DEFAULT_MAX_LLM_CALLS),
            measured_policy=(policy if policy in (MEASURED_ASK, MEASURED_EXCLUDE, MEASURED_IMPUTE)
                             else MEASURED_ASK),
        )


# --------------------------------------------------------------------------- #
# Closed vocabularies (run truth and telemetry read only these)
# --------------------------------------------------------------------------- #

DATA, CODE, INFRA, UNKNOWN = "data", "code", "infra", "unknown"
DTYPE_MISMATCH = "dtype_mismatch"
MISSING_VALUES = "missing_values"
NON_FINITE = "non_finite"
BAD_DATES = "bad_dates"
COLUMN_REFERENCE = "column_reference"
CODE_BUG = "code_bug"
NOT_RECOVERABLE = "not_recoverable"
DIAGNOSES = (DTYPE_MISMATCH, MISSING_VALUES, NON_FINITE, BAD_DATES, COLUMN_REFERENCE, CODE_BUG,
             NOT_RECOVERABLE)

DATA_STEP, CODE_FIX = "data_step", "code_fix"
PENDING, RECOVERED, FAILED_AGAIN = "pending", "recovered", "failed_again"

#: How the loop ended (``recovery.final``).
FINAL_RECOVERED = "recovered"
FINAL_EXHAUSTED = "exhausted"
FINAL_NEEDS_USER = "needs_user"
FINAL_NOT_RECOVERABLE = "not_recoverable"
FINAL_NOT_ATTEMPTED = "not_attempted"
FINALS = (FINAL_RECOVERED, FINAL_EXHAUSTED, FINAL_NEEDS_USER, FINAL_NOT_RECOVERABLE, FINAL_NOT_ATTEMPTED)

#: Why the loop stopped before the bound (``recovery.stopped.reason``).
STOP_REASONS = ("attempts", "oscillating", "code_unchanged", "budget", "needs_user", "not_recoverable",
                "no_repair")

COERCE_NUMERIC = "coerce_numeric"
PARSE_DATETIME = "parse_datetime"
IMPUTE = "impute"
EXCLUDE_ROWS = "exclude_rows"
TRANSFORMS = (COERCE_NUMERIC, PARSE_DATETIME, IMPUTE, EXCLUDE_ROWS)

#: State keys this loop owns; the server resets them every turn.
STATE_KEYS = ("replan_attempts", "replan_transforms", "replan_input_overrides", "replan_feedback",
              "replan_final", "replan_stopped", "replan_question", "replan_started_at",
              "replan_llm_calls", "replan_measured_choice", "replan_resume")


def fresh_state() -> Dict[str, Any]:
    """The per-turn reset of every key this loop owns."""
    return {"replan_attempts": [], "replan_transforms": [], "replan_input_overrides": {},
            "replan_feedback": None, "replan_final": None, "replan_stopped": None,
            "replan_question": None, "replan_started_at": None, "replan_llm_calls": 0,
            "replan_measured_choice": None, "replan_resume": False}


# --------------------------------------------------------------------------- #
# The error, as code reads it
# --------------------------------------------------------------------------- #

_EXC_LINE = re.compile(r"^\s*([A-Za-z_][\w.]*?(?:Error|Exception|Warning|Interrupt|Exit))\s*(?::\s*(.*))?$")
_CLASS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def parse_exception(state: Mapping[str, Any]) -> Tuple[str, str]:
    """(exception class, its message) from the failed run's stderr.

    A subprocess run reports only "Script failed with exit status 1" in
    ``execution_error``; the traceback is on stderr, and its last exception
    line is the error. The message stays in this process: it is diagnosis
    input and coder feedback, never user text and never telemetry.
    """
    result = state.get("execution_result") if isinstance(state.get("execution_result"), Mapping) else {}
    texts = [state.get("execution_stderr"), result.get("message"), state.get("execution_error"),
             result.get("error")]
    for text in texts:
        if not text:
            continue
        for line in reversed([ln for ln in str(text).splitlines() if ln.strip()]):
            m = _EXC_LINE.match(line)
            if m:
                cls = m.group(1).rsplit(".", 1)[-1]
                return cls, (m.group(2) or "").strip()
    joined = " ".join(str(t) for t in texts if t)
    if re.search(r"timed out", joined, re.I):
        return "TimeoutError", ""
    return "UnknownError", joined.strip()[:300]


def safe_class(name: Any) -> str:
    """An exception class name fit for a record, or ``"other"``."""
    text = str(name or "")
    return text if _CLASS_NAME.match(text) else "other"


# --------------------------------------------------------------------------- #
# The frames the run read
# --------------------------------------------------------------------------- #

@dataclass
class Source:
    """One input of the failed run: where the code read it, and the frame."""

    key: str            # the path the executor resolved (overrides map from it)
    path: str           # the path actually read this run (a prepared copy after a data step)
    frame: Any


def _read_frame(path: str):
    """The frame the analysis code saw: CSV through the same reader it uses."""
    import pandas as pd

    suffix = Path(path).suffix.lower()
    if suffix in (".csv", ".txt", ".tsv"):
        from file_handler.null_policy import read_csv_best_effort

        kwargs = {"sep": "\t"} if suffix == ".tsv" else {}
        return read_csv_best_effort(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return None


def run_inputs(state: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """(resolved path, path read this run) for each file input of the run.

    Resolved the way ``execution_agent_node_local`` resolves them, then mapped
    through the overrides earlier data steps wrote.
    """
    from app.agents import execution_agent as ex

    overrides = state.get("replan_input_overrides") or {}
    out: List[Tuple[str, str]] = []
    primary = ex._resolve_local_execution_input(dict(state), state.get("data_source_location"),
                                                state.get("output_location"))
    for path in [primary, *(ex._resolve_read_paths(state) or {}).values()]:
        if not path:
            continue
        key = str(path)
        if key in (k for k, _ in out):
            continue
        out.append((key, str(overrides.get(key, key))))
    return out


def load_sources(state: Mapping[str, Any]) -> List[Source]:
    sources = []
    for key, path in run_inputs(state):
        try:
            frame = _read_frame(path)
        except Exception:  # noqa: BLE001 - an unreadable input means no data step
            logger.info("[replan] input could not be read for profiling", exc_info=True)
            frame = None
        if frame is not None:
            sources.append(Source(key=key, path=path, frame=frame))
    return sources


# --------------------------------------------------------------------------- #
# What the code does with which column (AST, never executed)
# --------------------------------------------------------------------------- #

_AGGREGATIONS = {"mean", "sum", "median", "max", "min", "std", "var", "quantile", "agg", "aggregate",
                 "cumsum", "describe", "nlargest", "nsmallest", "prod", "idxmax", "idxmin"}


@dataclass
class CodeUse:
    used: List[str]
    measured: List[str]
    keys: List[str]


def code_use(code: str, columns: Sequence[str]) -> CodeUse:
    """Columns the code names, the ones it aggregates, the ones it groups or joins on."""
    cols = {str(c) for c in columns}
    used: List[str] = []
    measured: List[str] = []
    keys: List[str] = []
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return CodeUse(used, measured, keys)

    def strings(node) -> List[str]:
        return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant)
                and isinstance(n.value, str) and n.value in cols]

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in cols:
            if node.value not in used:
                used.append(node.value)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr
        if name in ("groupby", "merge", "join", "pivot_table", "set_index"):
            for arg in list(node.args) + [k.value for k in node.keywords
                                          if k.arg in ("by", "on", "left_on", "right_on", "index", "keys")]:
                for c in strings(arg):
                    if c not in keys:
                        keys.append(c)
        if name in _AGGREGATIONS:
            # df["revenue"].mean(), df["revenue"].astype(int).sum(),
            # df.groupby(..)["revenue"].sum(), .agg(x=("revenue", "mean"))
            subject = node.func.value
            while isinstance(subject, ast.Call) and isinstance(subject.func, ast.Attribute):
                subject = subject.func.value
            found = strings(subject.slice) if isinstance(subject, ast.Subscript) else []
            for arg in list(node.args) + [k.value for k in node.keywords]:
                if isinstance(arg, (ast.Tuple, ast.Dict)):
                    found += strings(arg)
            for c in found:
                if c not in measured:
                    measured.append(c)
    # A key the code groups by is a key, whatever else it does with it.
    measured = [c for c in measured if c not in keys]
    return CodeUse(used, measured, keys)


def _id_like(name: str, series, *, by_values: bool = True) -> bool:
    """An identifier: by its name, or (``by_values``) a text column of all-distinct values.

    The value test is off for a type repair: a column of prices stored as
    text is all-distinct too, and it is exactly what the repair is for.
    """
    lowered = str(name).lower()
    if lowered == "id" or lowered.endswith(("_id", " id", "_key", "uuid")) or lowered in ("code", "key"):
        return True
    if not by_values:
        return False
    try:
        non_null = series.dropna()
        return len(non_null) > 20 and non_null.is_unique and not _is_numeric(series)
    except Exception:  # noqa: BLE001
        return False


def _is_numeric(series) -> bool:
    import pandas as pd

    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _is_text(series) -> bool:
    import pandas as pd

    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #

_DTYPE_PATTERNS = re.compile(
    r"could not convert string to float|invalid literal for int\(\)|Unable to parse string"
    r"|unsupported operand type\(s\)|can only concatenate str|can't multiply sequence"
    r"|not supported between instances of|could not convert .* to numeric"
    r"|Could not convert .* to numeric|agg function failed|cannot perform \w+ with type"
    r"|must be real number, not str|ufunc '\w+' did not contain a loop", re.I)
_NAN_PATTERNS = re.compile(
    r"Cannot convert non-finite values \(NA or inf\) to integer|cannot convert float NaN to integer"
    r"|Input (?:X |y )?contains NaN|contains NaN|Integer column has NA values"
    r"|cannot convert NA to integer|NAType|boolean value of NA is ambiguous", re.I)
_INF_PATTERNS = re.compile(r"Input (?:X |y )?contains infinity|value too large for dtype", re.I)
_DATE_PATTERNS = re.compile(
    r"OutOfBoundsDatetime|Unknown (?:datetime )?string format|does not match format"
    r"|doesn't match format|time data .* does not match|Can only use \.dt accessor", re.I)
_NOT_RECOVERABLE_CLASSES = {
    "MemoryError", "TimeoutError", "TimeoutExpired", "ReadMapError", "ParserError", "UnicodeDecodeError",
    "EmptyDataError", "ArrowInvalid", "UnusableDataset", "PermissionError", "FileNotFoundError",
    "KeyboardInterrupt", "SystemExit", "OSError", "DatabaseWriteRefused",
}
_CODE_CLASSES = {
    "NameError", "SyntaxError", "IndentationError", "ImportError", "ModuleNotFoundError",
    "UnboundLocalError", "RecursionError", "AssertionError", "NotImplementedError",
}


@dataclass
class Diagnosis:
    kind: str                    # data | code | infra | unknown
    diagnosis: str               # one of DIAGNOSES
    columns: List[str]
    diagnosed_by: str            # rule | model


def _costume_count(series) -> int:
    """Values a plain float() rejects but the numeric parser reads."""
    from app.agents.preparation_agent import parse_numeric_token

    n = 0
    for value in series.dropna():
        text = str(value).strip()
        try:
            float(text)
        except ValueError:
            if parse_numeric_token(value) is not None:
                n += 1
    return n


def _coercible(series) -> bool:
    from app.agents.preparation_agent import detect_numeric_coercion

    return _is_text(series) and detect_numeric_coercion(series, confidence=COERCION_CONFIDENCE) is not None


def _dateable(series) -> bool:
    import pandas as pd

    if not _is_text(series):
        return False
    non_null = series.dropna().astype(str)
    if non_null.empty or not non_null.str.contains(r"[-/:]").any():
        return False
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= COERCION_CONFIDENCE


def _quoted(message: str) -> List[str]:
    """Values the error message quotes ('$1,234.50', 'revnue')."""
    return [a or b for a, b in re.findall(r"'([^']{1,80})'|\"([^\"]{1,80})\"", message or "")]


def _frame_of(sources: Sequence[Source], column: str):
    for src in sources:
        if column in src.frame.columns:
            return src
    return None


def diagnose(state: Mapping[str, Any], sources: Sequence[Source]) -> Diagnosis:
    """Data or code? Deterministic rules first; a model only when none matches."""
    cls, message = parse_exception(state)
    code = (state.get("coder_definition") or {}).get("code") or ""
    columns = [c for s in sources for c in s.frame.columns]
    use = code_use(code, columns)
    used = use.used or [str(c) for c in columns]

    if cls in _NOT_RECOVERABLE_CLASSES or cls == "UnknownError":
        # UnknownError: the run failed without the code raising (no output
        # path, an executor that was not there, nothing written). Neither a
        # data step nor a rewrite of the code can fix that.
        return Diagnosis(INFRA, NOT_RECOVERABLE, [], "rule")
    if cls in _CODE_CLASSES:
        return Diagnosis(CODE, CODE_BUG, [], "rule")
    if cls in ("KeyError", "AttributeError", "IndexError") or "not in index" in message:
        missing = [q for q in _quoted(message) if q not in columns]
        if missing or cls == "KeyError":
            return Diagnosis(CODE, COLUMN_REFERENCE, [], "rule")
        return Diagnosis(CODE, CODE_BUG, [], "rule")

    frames = {c: _frame_of(sources, c) for c in used}
    present = [c for c in used if frames.get(c) is not None]

    if _DATE_PATTERNS.search(message) or _DATE_PATTERNS.search(cls):
        cols = [c for c in present if _dateable(frames[c].frame[c])]
        if cols:
            return Diagnosis(DATA, BAD_DATES, cols, "rule")
        return Diagnosis(CODE, CODE_BUG, [], "rule")

    if _DTYPE_PATTERNS.search(message):
        cols = [c for c in present if _coercible(frames[c].frame[c])]
        quoted = _quoted(message)
        if quoted:
            hit = [c for c in cols if frames[c].frame[c].astype(str).str.strip().isin(
                [q.strip() for q in quoted]).any()]
            cols = hit or cols
        if cols:
            return Diagnosis(DATA, DTYPE_MISMATCH, cols, "rule")
        return Diagnosis(CODE, CODE_BUG, [], "rule")

    if _INF_PATTERNS.search(message):
        import numpy as np

        cols = [c for c in present if _is_numeric(frames[c].frame[c])
                and np.isinf(frames[c].frame[c].astype("float64")).any()]
        if cols:
            return Diagnosis(DATA, NON_FINITE, cols, "rule")
        return Diagnosis(CODE, CODE_BUG, [], "rule")

    if _NAN_PATTERNS.search(message) or cls == "IntCastingNaNError":
        cols = [c for c in present if frames[c].frame[c].isna().any()]
        if cols:
            return Diagnosis(DATA, MISSING_VALUES, cols, "rule")
        return Diagnosis(CODE, CODE_BUG, [], "rule")

    return _model_diagnosis(state, cls, message, code, sources, present)


#: The model the diagnosis falls back to. ``None`` means "use the reviewer
#: model the validator already built through app/core/inference.py"; tests
#: replace it with a stub.
diagnosis_llm: Any = None


def _profile(sources: Sequence[Source], columns: Sequence[str], limit: int = 30) -> List[Dict[str, Any]]:
    out = []
    for c in list(columns)[:limit]:
        src = _frame_of(sources, c)
        if src is None:
            continue
        s = src.frame[c]
        out.append({"column": c, "dtype": str(s.dtype), "rows": int(len(s)), "missing": int(s.isna().sum()),
                    "numeric_text": bool(_coercible(s)), "date_text": bool(_dateable(s))})
    return out


def _model_diagnosis(state, cls, message, code, sources, present) -> Diagnosis:
    """One closed-vocabulary verdict from a model, checked by code."""
    llm = diagnosis_llm
    if llm is None:
        try:
            from app.agents import validator as _validator

            llm = getattr(_validator, "validator_llm", None)
        except Exception:  # noqa: BLE001
            llm = None
    if llm is None:
        return Diagnosis(CODE, CODE_BUG, [], "rule")
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = (
        "An analysis script failed. Decide whether the DATA or the CODE is at fault.\n"
        f"Exception: {cls}: {message[:300]}\n"
        f"Column profile (code-measured): {json.dumps(_profile(sources, present))}\n"
        f"Code:\n```python\n{code[:6000]}\n```\n"
        'Answer JSON only: {"kind": "data"|"code"|"stop", "diagnosis": one of '
        f"{list(DIAGNOSES)}, \"columns\": [column names from the profile]}}"
    )
    try:
        reply = llm.invoke([SystemMessage(content="You classify runtime errors."), HumanMessage(content=prompt)])
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        match = re.search(r"\{.*\}", text, re.S)
        verdict = json.loads(match.group(0)) if match else {}
    except Exception:  # noqa: BLE001 - no verdict is a code verdict
        logger.info("[replan] diagnosis model gave no verdict", exc_info=True)
        return Diagnosis(CODE, CODE_BUG, [], "model")
    kind = str(verdict.get("kind") or "").lower()
    diag = str(verdict.get("diagnosis") or "")
    cols = [str(c) for c in verdict.get("columns") or [] if str(c) in present]
    if kind == "stop":
        return Diagnosis(INFRA, NOT_RECOVERABLE, [], "model")
    if kind == "data" and diag in (DTYPE_MISMATCH, MISSING_VALUES, NON_FINITE, BAD_DATES) and cols:
        # The model names the kind; the profile must show it, or it is code's.
        check = {
            DTYPE_MISMATCH: lambda s: _coercible(s),
            BAD_DATES: lambda s: _dateable(s),
            MISSING_VALUES: lambda s: bool(s.isna().any()),
            NON_FINITE: lambda s: _is_numeric(s) and bool((s.astype("float64").abs() == math.inf).any()),
        }[diag]
        cols = [c for c in cols if check(_frame_of(sources, c).frame[c])]
        if cols:
            return Diagnosis(DATA, diag, cols, "model")
    return Diagnosis(CODE, COLUMN_REFERENCE if diag == COLUMN_REFERENCE else CODE_BUG, [], "model")


# --------------------------------------------------------------------------- #
# The repairs. Pure functions over a frame; every count measured.
# --------------------------------------------------------------------------- #

def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return str(value)


def coerce_numeric(frame, column: str) -> Tuple[Any, Dict[str, Any]]:
    """Read a text column as numbers with the shared parser."""
    import pandas as pd

    from app.agents.preparation_agent import _NULL_TOKENS, parse_numeric_token

    out = frame.copy()
    before = out[column]
    present = before.notna()
    tokens = present & before.astype(str).str.strip().str.lower().isin(_NULL_TOKENS)
    parsed = before.map(parse_numeric_token)
    converted = present & ~tokens & parsed.notna()
    plain = converted & pd.to_numeric(before.where(converted), errors="coerce").notna()
    costume = converted & ~plain
    unreadable = present & ~tokens & parsed.isna()
    example = next((str(v) for v in before[costume]), None)
    out[column] = parsed.astype("float64")
    return out, {
        "op": COERCE_NUMERIC, "column": column, "rows_total": int(len(out)),
        "values_converted": int(costume.sum()), "null_tokens": int(tokens.sum()),
        "unreadable": int(unreadable.sum()), "example": example,
        "null_token_examples": sorted({str(v).strip() for v in before[tokens]})[:3],
    }


def parse_datetime(frame, column: str) -> Tuple[Any, Dict[str, Any]]:
    import pandas as pd

    out = frame.copy()
    before = out[column]
    parsed = pd.to_datetime(before, errors="coerce", format="mixed")
    present = before.notna()
    out[column] = parsed
    return out, {"op": PARSE_DATETIME, "column": column, "rows_total": int(len(out)),
                 "values_converted": int((present & parsed.notna()).sum()),
                 "unreadable": int((present & parsed.isna()).sum())}


def impute(frame, column: str) -> Tuple[Any, Dict[str, Any]]:
    """Fill the missing values of a numeric column with its median.

    Always the median, never the mean: one fill rule the user can be told in
    a word, robust to the tail that makes a mean misleading, and the owner's
    stated example. (preparation_agent.choose_imputation picks mean or median
    by skew for model features; a repair made to let an answer run is not a
    modelling choice.) The policy sends only numeric columns here.
    """
    out = frame.copy()
    series = out[column]
    missing = series.isna()
    strategy, value = "median", float(series.median())
    out[column] = series.fillna(value)
    return out, {"op": IMPUTE, "column": column, "rows_total": int(len(out)), "rows_affected": int(missing.sum()),
                 "method": strategy, "value": _jsonable(value)}


def exclude_rows(frame, column: str) -> Tuple[Any, Dict[str, Any]]:
    """Leave out the rows whose value in *column* is missing or not finite."""
    import numpy as np

    series = frame[column]
    bad = series.isna()
    if _is_numeric(series):
        bad = bad | np.isinf(series.astype("float64"))
    out = frame.loc[~bad].reset_index(drop=True)
    return out, {"op": EXCLUDE_ROWS, "column": column, "rows_total": int(len(frame)),
                 "rows_affected": int(bad.sum()), "non_finite": int((bad & series.notna()).sum())}


# --------------------------------------------------------------------------- #
# Policy: which repair, and when to stop and ask
# --------------------------------------------------------------------------- #

@dataclass
class Decision:
    """A repair to apply (``op``), or a question for the user (``ask``)."""

    column: str
    op: Optional[str] = None
    ask: Optional[Dict[str, Any]] = None
    #: "user" (answered the question) or "env" (AVALOKA_REPLAN_MEASURED_CHOICE).
    chosen_by: Optional[str] = None
    measured: bool = False


def missing_value_policy(series, *, column: str, measured: bool, protected: bool, limits: Limits,
                         already_excluded: int, rows_total: int, n_bad: Optional[int] = None,
                         choice: Optional[Mapping[str, Any]] = None) -> Decision:
    """What to do with the missing values of one column.

    The measured column -- the one the calculation averages or sums -- is
    asked about, always (the owner's ruling): filling it moves the answer
    toward the fill value, and leaving rows out answers over fewer rows; only
    the user can say which is right. A *choice* (the user's answer, or the
    run's up-front AVALOKA_REPLAN_MEASURED_CHOICE) is applied as given, limits
    included: it is the answer to a question that stated the counts. Keys and
    ids are never filled. Above a limit nothing is applied and the user is
    asked.
    """
    n_missing = int(series.isna().sum()) if n_bad is None else int(n_bad)
    share = n_missing / max(rows_total, 1)
    excluded_share = (already_excluded + n_missing) / max(rows_total, 1)

    if choice and choice.get("choice") in ("median", "leave_out"):
        op = IMPUTE if choice["choice"] == "median" and _is_numeric(series) and not protected else EXCLUDE_ROWS
        return Decision(column, op=op, chosen_by=choice.get("chosen_by"), measured=measured)

    def ask(reason: str, limit: Optional[float] = None) -> Decision:
        return Decision(column, ask={"reason": reason, "column": column, "rows_affected": n_missing,
                                     "rows_total": rows_total, "limit": limit, "measured": measured})

    def exclude_or_ask() -> Decision:
        if excluded_share <= limits.max_exclude_share:
            return Decision(column, op=EXCLUDE_ROWS, measured=measured)
        return ask("exclude_limit", limits.max_exclude_share)

    if protected:
        return exclude_or_ask()
    if measured:
        if limits.measured_policy == MEASURED_EXCLUDE:      # test hook only
            return exclude_or_ask()
        if limits.measured_policy != MEASURED_IMPUTE:       # the shipped default: ask
            return ask("measured_choice")
    if share <= limits.max_impute_share and _is_numeric(series):
        return Decision(column, op=IMPUTE, measured=measured)
    if share <= limits.max_impute_share and not _is_numeric(series):
        # A missing label is not a number to fill; leave the rows out instead.
        return exclude_or_ask()
    return ask("impute_limit", limits.max_impute_share)


def choice_for(column: str, n_missing: int, state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The answer that applies to this column's missing values, if any.

    The user's answer (``replan_measured_choice``, set by the planner's resume
    step) applies only to the column it was asked about and only while the
    count still matches the question; a different count is a different
    question, and it is asked again. Otherwise the run's up-front answer
    (AVALOKA_REPLAN_MEASURED_CHOICE).
    """
    from app.agents import replan_choice

    given = state.get("replan_measured_choice")
    if isinstance(given, Mapping) and given.get("column") == column \
            and given.get("choice") in replan_choice.CHOICES:
        expected = given.get("rows_affected")
        if expected is None or int(expected) == int(n_missing):
            return {"choice": given["choice"], "chosen_by": given.get("chosen_by") or "user",
                    "question": given.get("question")}
        return None
    env = replan_choice.env_choice()
    if env:
        return {"choice": env, "chosen_by": "env", "question": None}
    return None


def plan_repairs(diag: Diagnosis, sources: Sequence[Source], state: Mapping[str, Any],
                 limits: Limits) -> List[Decision]:
    code = (state.get("coder_definition") or {}).get("code") or ""
    columns = [c for s in sources for c in s.frame.columns]
    use = code_use(code, columns)
    target = (((state.get("training_plan") or {}).get("data_config") or {}).get("target_column")
              if isinstance(state.get("training_plan"), Mapping) else None)
    excluded = sum(int(t.get("rows_affected") or 0) for t in state.get("replan_transforms") or []
                   if t.get("op") == EXCLUDE_ROWS)
    decisions: List[Decision] = []
    for column in diag.columns:
        src = _frame_of(sources, column)
        if src is None:
            continue
        series = src.frame[column]
        if diag.diagnosis == DTYPE_MISMATCH:
            if _id_like(column, series, by_values=False):
                continue
            decisions.append(Decision(column, op=COERCE_NUMERIC))
        elif diag.diagnosis == BAD_DATES:
            decisions.append(Decision(column, op=PARSE_DATETIME))
        elif diag.diagnosis == NON_FINITE:
            # An infinity is never filled: there is no honest finite stand-in.
            import numpy as np

            n_bad = int((series.isna() | np.isinf(series.astype("float64"))).sum())
            d = missing_value_policy(series, column=column, measured=True, protected=True, limits=limits,
                                     already_excluded=excluded, rows_total=len(series), n_bad=n_bad)
            if d.op == EXCLUDE_ROWS:
                excluded += n_bad
            decisions.append(d)
        elif diag.diagnosis == MISSING_VALUES:
            protected = column in use.keys or column == target or _id_like(column, series)
            d = missing_value_policy(series, column=column, measured=column in use.measured,
                                     protected=protected, limits=limits, already_excluded=excluded,
                                     rows_total=len(series),
                                     choice=choice_for(column, int(series.isna().sum()), state))
            if d.op == EXCLUDE_ROWS:
                excluded += int(series.isna().sum())
            decisions.append(d)
    return decisions


_APPLY = {COERCE_NUMERIC: coerce_numeric, PARSE_DATETIME: parse_datetime, IMPUTE: impute,
          EXCLUDE_ROWS: exclude_rows}


#: A prepared copy older than this is swept when the next turn writes one. A
#: copy is only ever read within its own turn (the overrides are reset every
#: turn), and a resumed question runs on the source file again; the age is a
#: margin for a turn still running on another thread of the same dataset.
PREPARED_MAX_AGE_S = 3600.0


def _sweep_prepared(base: Path, now: Optional[float] = None) -> int:
    """Remove this dataset's stale prepared copies; returns how many went."""
    now = now or time.time()
    gone = 0
    try:
        entries = list(base.glob("*_prepared_*.csv"))
    except OSError:
        return 0
    for entry in entries:
        try:
            if now - entry.stat().st_mtime > PREPARED_MAX_AGE_S:
                entry.unlink()
                gone += 1
        except OSError:  # noqa: PERF203 - a copy another process holds stays
            continue
    return gone


def _prepared_path(state: Mapping[str, Any], key: str) -> str:
    base = Path(str(state.get("output_location") or tempfile.gettempdir())).parent / ".replan"
    base.mkdir(parents=True, exist_ok=True)
    if not state.get("replan_input_overrides"):
        # The turn's first copy: the copies of earlier turns are not needed.
        _sweep_prepared(base)
    stem = Path(key).stem
    fd, path = tempfile.mkstemp(prefix=f"{stem}_prepared_", suffix=".csv", dir=str(base))
    os.close(fd)
    return path


def apply_repairs(decisions: Sequence[Decision], sources: Sequence[Source],
                  state: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Apply each repair to its frame and write the prepared copies.

    Returns the transform records and the new overrides (resolved path ->
    prepared copy). The source files are never written.
    """
    records: List[Dict[str, Any]] = []
    touched: Dict[str, Source] = {}
    for d in decisions:
        src = _frame_of(sources, d.column)
        if src is None or d.op not in _APPLY:
            continue
        src.frame, record = _APPLY[d.op](src.frame, d.column)
        if d.chosen_by:
            record["chosen_by"] = d.chosen_by
        if d.measured:
            record["measured"] = True
        records.append(record)
        touched[src.key] = src
    overrides = dict(state.get("replan_input_overrides") or {})
    for key, src in touched.items():
        path = _prepared_path(state, key)
        src.frame.to_csv(path, index=False)
        overrides[key] = path
    return records, overrides


# --------------------------------------------------------------------------- #
# The node
# --------------------------------------------------------------------------- #

def _code_sha(state: Mapping[str, Any]) -> str:
    """A hash of what the code does, not how it is laid out.

    The static validator reformats code on its way through (its fixers
    unparse the tree), so a rewrite that changed nothing but comments or
    spacing hashes the same as the code it replaced.
    """
    code = (state.get("coder_definition") or {}).get("code") or ""
    try:
        code = ast.dump(ast.parse(code))
    except SyntaxError:
        pass
    return hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()[:12]


def _failed(state: Mapping[str, Any]) -> bool:
    from app.agents.analysis_outcome import FAILED, analysis_outcome

    return analysis_outcome(state) == FAILED


def replan_enabled(limits: Optional[Limits] = None) -> bool:
    return (limits or Limits.from_env()).max_attempts > 0


def _signature(cls: str, diag: Diagnosis) -> str:
    return f"{cls}|{diag.diagnosis}|{','.join(sorted(diag.columns))}"


def replan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node, after a failed ``execute_locally``. See the module docstring."""
    from app.agents import replan_disclosure as words
    from app.api import reply_segments as rs

    limits = Limits.from_env()
    now = time.time()
    started = state.get("replan_started_at") or now
    attempts = [dict(a) for a in state.get("replan_attempts") or []]
    if attempts and attempts[-1].get("outcome") == PENDING:
        attempts[-1]["outcome"] = FAILED_AGAIN
        attempts[-1]["duration_ms"] = int((now - float(attempts[-1].get("_t0") or now)) * 1000)

    cls, _message = parse_exception(state)
    sources = load_sources(state)
    diag = diagnose(state, sources)
    sha = _code_sha(state)
    sig = _signature(cls, diag)
    llm_calls = int(state.get("replan_llm_calls") or 0) + (1 if diag.diagnosed_by == "model" else 0)

    def stop(reason: str, final: str, question: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        stopped = {"reason": reason, "error_class": safe_class(cls), "diagnosis": diag.diagnosis,
                   "diagnosed_by": diag.diagnosed_by, "error_kind": diag.kind}
        out: Dict[str, Any] = {
            "replan_attempts": [_public(a) for a in attempts], "replan_final": final,
            "replan_stopped": stopped, "replan_started_at": started, "replan_llm_calls": llm_calls,
            "replan_question": question, "replan_feedback": None,
        }
        text = None
        if final == FINAL_NEEDS_USER and question:
            from app.agents import replan_choice

            can_answer = replan_choice.interactive(state)
            if question.get("column"):
                stopped["question"] = replan_choice.question_text(question)
                stopped["choice"] = replan_choice.NO_ANSWER
                stopped["interactive"] = can_answer
                if can_answer:
                    # Paused, not ended: the next turn resumes on the answer
                    # (replan_choice.resolve, from plan_etl_job).
                    data_key = (run_inputs(state) or [(None, None)])[0][0]
                    out["pending_clarification"] = replan_choice.pending(question, state, data_key=data_key,
                                                                         code_sha=sha)
            text = words.needs_user_reply(question, state.get("replan_transforms") or [],
                                          interactive=can_answer)
        elif attempts:
            text = words.exhausted_reply(attempts, stopped)
        if text:
            out["messages"] = list(state.get("messages") or []) + [rs.plain_message(content=text)]
        logger.info("[replan] stop: %s (%s, %s, %d attempt(s))", reason, safe_class(cls), diag.diagnosis,
                    len(attempts))
        return out

    if diag.diagnosis == NOT_RECOVERABLE:
        return stop("not_recoverable", FINAL_NOT_RECOVERABLE)
    if len(attempts) >= limits.max_attempts:
        return stop("attempts", FINAL_EXHAUSTED)
    if now - float(started) > limits.budget_s:
        return stop("budget", FINAL_EXHAUSTED)
    if attempts:
        last = attempts[-1]
        if last.get("action") == CODE_FIX and last.get("code_sha") == sha:
            return stop("code_unchanged", FINAL_EXHAUSTED)
        earlier = [a.get("_signature") for a in attempts[:-1]]
        if sig != last.get("_signature") and sig in earlier:
            return stop("oscillating", FINAL_EXHAUSTED)

    record: Dict[str, Any] = {
        "n": len(attempts) + 1, "error_class": safe_class(cls), "error_kind": diag.kind,
        "diagnosis": diag.diagnosis, "diagnosed_by": diag.diagnosed_by, "code_sha": sha,
        "outcome": PENDING, "_signature": sig, "_t0": now, "duration_ms": None,
    }

    if diag.kind == DATA:
        decisions = plan_repairs(diag, sources, state, limits)
        asks = [d.ask for d in decisions if d.ask]
        if asks:
            return stop("needs_user", FINAL_NEEDS_USER, asks[0])
        done_cols = {t.get("column") for t in state.get("replan_transforms") or []}
        repeat = [d for d in decisions if any(t.get("column") == d.column and t.get("op") == d.op
                                               for t in state.get("replan_transforms") or [])]
        decisions = [d for d in decisions if d.op and d not in repeat]
        if len(done_cols | {d.column for d in decisions}) > limits.max_columns:
            question = {"reason": "column_limit", "columns": len(done_cols | {d.column for d in decisions}),
                        "limit": limits.max_columns}
            return stop("needs_user", FINAL_NEEDS_USER, question)
        if decisions:
            transforms, overrides = apply_repairs(decisions, sources, state)
            record.update(action=DATA_STEP, transforms=[_run_truth_transform(t) for t in transforms],
                          code_changed=False)
            chosen = [d for d in decisions if d.chosen_by]
            if chosen:
                given = state.get("replan_measured_choice") if isinstance(
                    state.get("replan_measured_choice"), Mapping) else {}
                record["decision"] = {
                    "question": given.get("question"), "column": chosen[0].column,
                    "choice": "median" if chosen[0].op == IMPUTE else "leave_out",
                    "chosen_by": chosen[0].chosen_by,
                }
            attempts.append(record)
            return {
                "replan_attempts": [_public(a, keep_private=True) for a in attempts],
                "replan_transforms": list(state.get("replan_transforms") or []) + transforms,
                "replan_input_overrides": overrides, "replan_started_at": started,
                "replan_llm_calls": llm_calls, "replan_final": None, "replan_feedback": None,
            }
        # Nothing left to repair in the data: the code has to change.
        diag = Diagnosis(CODE, CODE_BUG, [], diag.diagnosed_by)
        record.update(error_kind=CODE, diagnosis=CODE_BUG)

    if state.get("replan_code_fix") is False:
        # A scheduled run: code changes nobody reviewed are not made here.
        return stop("no_repair", FINAL_EXHAUSTED if attempts else FINAL_NOT_RECOVERABLE)
    if llm_calls + CODE_FIX_LLM_CALLS > limits.max_llm_calls:
        return stop("budget", FINAL_EXHAUSTED)
    record.update(action=CODE_FIX, transforms=[], code_changed=True)
    attempts.append(record)
    return {
        "replan_attempts": [_public(a, keep_private=True) for a in attempts],
        "replan_feedback": coder_feedback(state, cls, _message, diag, sources),
        "replan_started_at": started, "replan_llm_calls": llm_calls + CODE_FIX_LLM_CALLS,
        "replan_final": None,
    }


def _public(attempt: Mapping[str, Any], keep_private: bool = False) -> Dict[str, Any]:
    """An attempt as stored. The signature and start time are bookkeeping."""
    if keep_private:
        return dict(attempt)
    return {k: v for k, v in attempt.items() if not k.startswith("_")}


def _run_truth_transform(t: Mapping[str, Any]) -> Dict[str, Any]:
    """A transform as run truth keeps it: counts, method and fill statistic; no cell values."""
    keep = ("op", "column", "rows_total", "rows_affected", "values_converted", "null_tokens", "unreadable",
            "method", "value", "non_finite", "chosen_by", "measured")
    return {k: t[k] for k in keep if k in t}


def coder_feedback(state: Mapping[str, Any], cls: str, message: str, diag: Diagnosis,
                   sources: Sequence[Source]) -> str:
    """What the coder is told: the error, and the columns that do exist."""
    columns = [str(c) for s in sources for c in s.frame.columns]
    parts = [f"The code failed when run on the full dataset with {cls}: {message[:500]}."]
    if diag.diagnosis == COLUMN_REFERENCE and columns:
        parts.append("Use only these columns: " + ", ".join(columns[:60]) + ".")
    if state.get("replan_transforms"):
        parts.append("The data was already repaired before this run; do not repeat the repair.")
    parts.append("Fix the code so it runs on this data. Keep main(df) returning a DataFrame.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Routing and the record
# --------------------------------------------------------------------------- #

def route_after_replan(state: Mapping[str, Any]) -> str:
    """``execute_locally`` after a data step, ``code_etl`` for a code fix, else ``end``."""
    if state.get("replan_final"):
        return "end"
    attempts = state.get("replan_attempts") or []
    if not attempts or attempts[-1].get("outcome") != PENDING:
        return "end"
    return "code_etl" if attempts[-1].get("action") == CODE_FIX else "execute_locally"


def should_replan(state: Mapping[str, Any]) -> bool:
    """Whether a run that just ended goes to the replan node.

    A failed run does, while the loop is on; so does a run that failed after
    an attempt, so the node can close the record and word the ending.
    """
    if not replan_enabled():
        return False
    if state.get("replan_final"):
        return False
    return _failed(state)


def recovery_block(final: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The ``recovery`` record for run truth, from the turn's final state. Code-written."""
    attempts = [dict(a) for a in final.get("replan_attempts") or []]
    stopped = final.get("replan_stopped")
    if not attempts and not stopped:
        return None
    failed = _failed(final)
    for a in attempts:
        if a.get("outcome") == PENDING:
            a["outcome"] = FAILED_AGAIN if failed else RECOVERED
    final_word = final.get("replan_final")
    if not final_word:
        final_word = FINAL_EXHAUSTED if failed else FINAL_RECOVERED
    return {
        "written_by": "harness",
        "max_attempts": Limits.from_env().max_attempts,
        "final": final_word,
        "attempts": [_public(a) for a in attempts],
        "stopped": dict(stopped) if isinstance(stopped, Mapping) else None,
    }


def recovery_telemetry(recovery: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The only projection of the record meant for telemetry.

    Closed vocabularies, a class name and counts. No column, value, path or
    message travels: the transform list is reduced to its operation names.
    """
    if not isinstance(recovery, Mapping):
        return {}

    def pick(value: Any, vocab: Sequence[str]) -> str:
        return value if value in vocab else "other"

    attempts = []
    for a in recovery.get("attempts") or []:
        attempts.append({
            "error_class": safe_class(a.get("error_class")),
            "error_kind": pick(a.get("error_kind"), (DATA, CODE, INFRA, UNKNOWN)),
            "diagnosis": pick(a.get("diagnosis"), DIAGNOSES),
            "diagnosed_by": pick(a.get("diagnosed_by"), ("rule", "model")),
            "action": pick(a.get("action"), (DATA_STEP, CODE_FIX)),
            "outcome": pick(a.get("outcome"), (RECOVERED, FAILED_AGAIN, PENDING)),
            "ops": [pick(t.get("op"), TRANSFORMS) for t in a.get("transforms") or []],
            "choice": pick((a.get("decision") or {}).get("choice"), ("median", "leave_out"))
            if a.get("decision") else None,
        })
    stopped = recovery.get("stopped") or {}
    return {
        "final": pick(recovery.get("final"), FINALS),
        "attempts_used": len(attempts),
        "attempts": attempts,
        "stop_reason": pick(stopped.get("reason"), STOP_REASONS) if stopped else None,
        "stop_choice": pick(stopped.get("choice"), ("no_answer",)) if stopped.get("choice") else None,
    }


def run_scheduled_local(state: Dict[str, Any], execute) -> Dict[str, Any]:
    """A scheduled local run with the loop's data steps, and no one to ask.

    Celery's scheduled analysis calls the local executor directly, not the
    graph. Here the loop does what it can without a person: the data steps
    within the limits, and the run's up-front answer for the measured column
    (AVALOKA_REPLAN_MEASURED_CHOICE). It never rewrites code (nobody reviews
    it), and where a decision is needed it ends with the question and how to
    answer it ahead of time -- nothing filled, nothing left out.
    """
    state = {**state, **fresh_state(), "replan_interactive": False, "replan_code_fix": False}
    state = execute(state)
    for _ in range(HARD_MAX_ATTEMPTS + 1):
        if not should_replan(state):
            break
        state = {**state, **replan_node(state)}
        if route_after_replan(state) != "execute_locally":
            break
        state = execute(state)
    if state.get("replan_transforms") and not _failed(state):
        # No verifier runs here to attach the repair lines: say them in the
        # run's own reply, as plain text.
        from app.agents.replan_disclosure import disclosure_lines
        from app.api import reply_segments as rs

        state["messages"] = list(state.get("messages") or []) + [
            rs.plain_message(content="\n".join(disclosure_lines(state["replan_transforms"])))]
    return state
