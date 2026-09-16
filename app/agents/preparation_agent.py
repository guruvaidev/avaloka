"""Data cleansing and feature engineering, fitted on training rows only.

Two jobs sit between a raw dataset and a usable one, and Avaloka had neither as
a first-class agent: **cleansing** (a numeric column that arrived as
``"$1,234.50"`` is not numeric, and a missing value is not a zero) and **feature
engineering** (a timestamp is far more useful as day-of-week and hour than as an
epoch integer).

Three properties shape this module.

**Everything is fit/apply, never fit_transform.** Imputation values, category
vocabularies and bin edges are learned from *training rows only* and then
applied to validation and test. Fitting on the full frame before splitting is
target leakage: the model sees test-set statistics and reports a score it cannot
reproduce in production. That failure leaves no trace in the data — the rows are
byte-identical either way; what leaks is a statistic computed at the wrong
moment — so no downstream check can catch it, and the discipline has to live
here. :func:`fit_preparation` takes the training frame; :func:`apply_preparation`
takes any frame and a fitted :class:`PreparationPlan`.

**The objective changes what is built.** A plan prepared for modelling wants
missingness indicators, cyclical encodings and rare-category grouping. A plan
prepared for analysis wants readable columns a human will put in a table.
Building ML encodings for someone who asked "what is the average revenue by
region" produces a frame nobody can read.

**Every decision records why.** A number that changed, or a column that
appeared, is traceable to the rule that produced it — the same standard the
integrity and evaluation agents hold. An imputation that silently replaces 40%
of a column is a modelling decision disguised as a cleanup.

Pure functions over a DataFrame: no LLM, no network, no cluster, so the whole
module is testable in milliseconds.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.agents.contract import AgentSpec, Stage, agent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Thresholds — each one is a judgement call, so each one is named and explained
# --------------------------------------------------------------------------- #

#: Above this fraction of missing values, imputing invents more than it repairs;
#: the column is flagged for a decision instead of quietly filled.
IMPUTE_REFUSAL_THRESHOLD = 0.60

#: Missingness above this is worth its own indicator column: whether a value was
#: absent is often more predictive than whatever we fill in.
MISSINGNESS_INDICATOR_THRESHOLD = 0.05

#: |skew| above this makes the mean a poor centre; use the median.
SKEW_MEDIAN_THRESHOLD = 1.0

#: A category appearing in fewer than this fraction of rows is grouped into
#: ``__other__`` rather than becoming its own sparse level.
RARE_CATEGORY_THRESHOLD = 0.01

#: A string column parsing at or above this rate is treated as numeric-with-noise
#: rather than genuinely categorical.
NUMERIC_COERCION_CONFIDENCE = 0.90

#: Currency and unit marks stripped before numeric parsing.
_CURRENCY_MARKS = "$£€¥₹₽₩¢"

_NULL_TOKENS = {
    "", "na", "n/a", "n.a.", "nan", "null", "none", "nil", "-", "--", "?",
    "unknown", "missing", "not available", "#n/a", "#null!", "\\n",
}


class Objective(str, Enum):
    """What the prepared frame is for. The planner supplies this."""

    MODELLING = "modelling"   # features for a model: encodings, indicators
    ANALYSIS = "analysis"     # columns for a human: readable, no encodings
    BOTH = "both"


class Action(str, Enum):
    COERCE_NUMERIC = "coerce_numeric"
    PARSE_DATETIME = "parse_datetime"
    COERCE_BOOLEAN = "coerce_boolean"
    NORMALISE_TEXT = "normalise_text"
    IMPUTE = "impute"
    FLAG_UNIMPUTABLE = "flag_unimputable"
    GROUP_RARE = "group_rare"
    ADD_INDICATOR = "add_indicator"
    ADD_DATE_PARTS = "add_date_parts"
    ADD_CYCLICAL = "add_cyclical"
    ADD_RATIO = "add_ratio"


@dataclass
class Decision:
    """One thing done to one column, and the reason it was done."""

    action: Action
    column: str
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    produced: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "column": self.column,
            "detail": self.detail,
            "evidence": self.evidence,
            "produced": list(self.produced),
        }


@dataclass
class PreparationPlan:
    """What was learned from the training rows, and what to do with any frame.

    This object is the whole point of the fit/apply split: it carries fitted
    *statistics*, so applying it to validation data cannot consult that data.
    """

    objective: Objective = Objective.BOTH
    target: Optional[str] = None
    numeric_coercions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    datetime_columns: List[str] = field(default_factory=list)
    boolean_coercions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    text_normalisations: List[str] = field(default_factory=list)
    imputations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unimputable: Dict[str, float] = field(default_factory=dict)
    rare_categories: Dict[str, List[str]] = field(default_factory=dict)
    indicator_columns: List[str] = field(default_factory=list)
    date_part_columns: Dict[str, List[str]] = field(default_factory=dict)
    cyclical_columns: Dict[str, List[str]] = field(default_factory=dict)
    ratio_features: List[Tuple[str, str, str]] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    fitted_rows: int = 0

    def record(self, decision: Decision) -> None:
        self.decisions.append(decision)

    @property
    def engineered_columns(self) -> List[str]:
        out: List[str] = list(self.indicator_columns)
        for cols in self.date_part_columns.values():
            out.extend(cols)
        for cols in self.cyclical_columns.values():
            out.extend(cols)
        out.extend(name for _, _, name in self.ratio_features)
        return out

    def summary(self) -> str:
        if not self.decisions:
            return "no preparation needed"
        counts: Dict[str, int] = {}
        for d in self.decisions:
            counts[d.action.value] = counts.get(d.action.value, 0) + 1
        return ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective.value,
            "target": self.target,
            "fitted_rows": self.fitted_rows,
            "summary": self.summary(),
            "engineered_columns": self.engineered_columns,
            "unimputable": self.unimputable,
            "decisions": [d.as_dict() for d in self.decisions],
        }


# --------------------------------------------------------------------------- #
# Cleansing — value-level repair
# --------------------------------------------------------------------------- #

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_PARENS_NEGATIVE = re.compile(r"^\((.*)\)$")


def parse_numeric_token(raw: Any) -> Optional[float]:
    """Parse one value that *means* a number but does not look like one.

    Handles currency, percentages, accounting negatives, and unambiguous US,
    Indian, or European grouping. A lone decimal comma remains deliberately
    unsupported because ``1,234`` can mean either 1234 or 1.234.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return None if (isinstance(raw, float) and math.isnan(raw)) else float(raw)

    text = str(raw).strip()
    if text.lower() in _NULL_TOKENS:
        return None

    # Currency marks come off first: "$(500)" is an accounting negative wearing a
    # currency symbol, and testing for the parentheses before stripping the "$"
    # never matches.
    text = "".join(ch for ch in text if ch not in _CURRENCY_MARKS).strip()

    negative = False
    m = _PARENS_NEGATIVE.match(text)         # accounting negatives: (1,234)
    if m:
        negative, text = True, m.group(1).strip()

    percent = text.endswith("%")
    text = text.rstrip("%").strip()
    text = text.replace("\u00a0", "").replace(" ", "")
    # When both separators exist their order identifies the decimal mark:
    # 1,234.56 is US-style, while 1.234,56 is European-style. A single comma is
    # accepted only when it forms an unambiguous thousands/Indian grouping.
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+,\d+", text):
        text = text.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(?:,\d{3})+\.\d+", text):
        text = text.replace(",", "")
    elif "," in text and re.fullmatch(r"-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?", text):
        text = text.replace(",", "")
    else:
        text = _THOUSANDS.sub("", text)
    if not text or not re.fullmatch(r"[+-]?\d*\.?\d+(?:[eE][-+]?\d+)?", text):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if percent:
        value /= 100.0
    return -value if negative else value


# Compatibility for existing callers and tests. New code should use the public
# name so planner, preparation, and generated analysis share one parser.
_clean_numeric_token = parse_numeric_token


def detect_numeric_coercion(series, *, confidence: float = NUMERIC_COERCION_CONFIDENCE
                            ) -> Optional[Dict[str, Any]]:
    """Decide whether a text column is really numeric wearing a costume."""
    import pandas as pd

    if not _is_object_like(series):
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    parsed = [_clean_numeric_token(v) for v in non_null]
    ok = sum(1 for p in parsed if p is not None)
    rate = ok / len(parsed)
    if rate < confidence:
        return None
    sample = next((str(v) for v in non_null if any(c in str(v) for c in _CURRENCY_MARKS)), None)
    return {
        "parse_rate": round(rate, 4),
        "parsed": ok,
        "failed": len(parsed) - ok,
        "example": sample or str(non_null.iloc[0]),
        "had_currency_marks": sample is not None,
    }


def _is_object_like(series) -> bool:
    import pandas as pd
    return series.dtype == object or str(series.dtype).startswith("string")


def _skew(series) -> float:
    try:
        value = float(series.skew())
        return 0.0 if math.isnan(value) else value
    except Exception:  # noqa: BLE001
        return 0.0


def choose_imputation(series, *, missing_rate: float) -> Tuple[str, Any, str]:
    """Pick a fill value and say why. Returns (strategy, value, reason)."""
    import pandas as pd

    if pd.api.types.is_numeric_dtype(series):
        skew = _skew(series.dropna())
        if abs(skew) >= SKEW_MEDIAN_THRESHOLD:
            return ("median", float(series.median()),
                    f"numeric with skew {skew:.2f}; the mean would be pulled by the tail")
        return ("mean", float(series.mean()),
                f"numeric and near-symmetric (skew {skew:.2f}); the mean is the natural centre")

    if pd.api.types.is_datetime64_any_dtype(series):
        return ("none", None,
                "datetime gaps are not filled: an invented timestamp implies an event that "
                "did not happen")

    non_null = series.dropna()
    if non_null.empty:
        return ("none", None, "column is entirely missing; nothing to learn a fill from")
    mode = non_null.mode()
    if mode.empty:
        return ("constant", "__missing__", "no modal value; a sentinel keeps the gap visible")
    share = float((non_null == mode.iloc[0]).mean())
    if share >= 0.5:
        return ("mode", mode.iloc[0],
                f"categorical with a dominant level ({share:.0%} of non-null rows)")
    return ("constant", "__missing__",
            "categorical with no dominant level; a sentinel is honest where a mode would "
            "invent a majority")


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #

def fit_preparation(train_df, *, objective: Objective = Objective.BOTH,
                    target: Optional[str] = None,
                    columns: Optional[Sequence[str]] = None) -> PreparationPlan:
    """Learn every cleansing and feature decision from **training rows only**."""
    import pandas as pd

    plan = PreparationPlan(objective=Objective(objective), target=target,
                           fitted_rows=int(len(train_df)))
    cols = [c for c in (columns or train_df.columns) if c != target]
    n = max(len(train_df), 1)
    wants_ml = plan.objective in (Objective.MODELLING, Objective.BOTH)

    for col in cols:
        if col not in train_df.columns:
            continue
        series = train_df[col]

        # --- 1. type repair, before anything is measured -------------------
        coercion = detect_numeric_coercion(series)
        if coercion:
            plan.numeric_coercions[col] = coercion
            plan.record(Decision(
                Action.COERCE_NUMERIC, col,
                f"parsed as numeric ({coercion['parse_rate']:.0%} of values)"
                + (" after stripping currency marks" if coercion["had_currency_marks"] else ""),
                coercion))
            series = series.map(_clean_numeric_token)
        elif _is_object_like(series):
            parsed_dt = _try_datetime(series)
            if parsed_dt is not None:
                plan.datetime_columns.append(col)
                plan.record(Decision(Action.PARSE_DATETIME, col,
                                     "parsed as datetime", {"parse_rate": parsed_dt}))
                series = pd.to_datetime(series, errors="coerce")
            else:
                mapping = _boolean_mapping(series)
                if mapping:
                    plan.boolean_coercions[col] = mapping
                    plan.record(Decision(Action.COERCE_BOOLEAN, col,
                                         "two-valued text read as boolean", {"mapping": mapping}))
                else:
                    plan.text_normalisations.append(col)

        # --- 2. missingness ------------------------------------------------
        missing_rate = float(series.isna().mean())
        if missing_rate == 0:
            # Clean training data does not mean clean production data. Learn the
            # fill anyway, from train statistics, so a gap appearing only at apply
            # time is handled rather than propagating as NaN. No decision is
            # recorded: nothing was changed here.
            strategy, value, _ = choose_imputation(series, missing_rate=0.0)
            if strategy != "none" and value is not None:
                plan.imputations[col] = {"strategy": strategy, "value": value,
                                         "missing_rate": 0.0, "precautionary": True}
        else:
            if missing_rate >= IMPUTE_REFUSAL_THRESHOLD:
                plan.unimputable[col] = round(missing_rate, 4)
                plan.record(Decision(
                    Action.FLAG_UNIMPUTABLE, col,
                    f"{missing_rate:.0%} missing — filling would invent more than it repairs; "
                    f"decide whether to drop the column or the rows",
                    {"missing_rate": round(missing_rate, 4)}))
            else:
                strategy, value, reason = choose_imputation(series, missing_rate=missing_rate)
                if strategy != "none":
                    plan.imputations[col] = {"strategy": strategy, "value": value,
                                             "missing_rate": round(missing_rate, 4)}
                    plan.record(Decision(
                        Action.IMPUTE, col,
                        f"{missing_rate:.1%} missing → {strategy}: {reason}",
                        {"strategy": strategy, "value": _jsonable(value),
                         "missing_rate": round(missing_rate, 4)}))
                else:
                    plan.record(Decision(Action.IMPUTE, col, reason,
                                         {"strategy": "none",
                                          "missing_rate": round(missing_rate, 4)}))

            if wants_ml and missing_rate >= MISSINGNESS_INDICATOR_THRESHOLD:
                name = f"{col}__was_missing"
                plan.indicator_columns.append(name)
                plan.record(Decision(
                    Action.ADD_INDICATOR, col,
                    "whether the value was absent is often more predictive than the fill",
                    {"missing_rate": round(missing_rate, 4)}, produced=(name,)))

        # --- 3. categorical tidying ---------------------------------------
        if _is_object_like(series) and col not in plan.numeric_coercions:
            counts = series.dropna().astype(str).str.strip().value_counts()
            rare = [v for v, c in counts.items() if c / n < RARE_CATEGORY_THRESHOLD]
            if rare and len(rare) < len(counts):
                plan.rare_categories[col] = rare
                plan.record(Decision(
                    Action.GROUP_RARE, col,
                    f"{len(rare)} level(s) under {RARE_CATEGORY_THRESHOLD:.0%} of rows "
                    f"grouped into __other__",
                    {"levels": rare[:20], "n_levels": len(counts)}))

    # --- 4. features from datetimes ---------------------------------------
    for col in plan.datetime_columns + _native_datetime_columns(train_df, cols):
        parts = ["year", "month", "day", "dayofweek", "hour"]
        names = [f"{col}__{p}" for p in parts]
        plan.date_part_columns[col] = names
        plan.record(Decision(Action.ADD_DATE_PARTS, col,
                             "a timestamp is more useful decomposed than as an instant",
                             {"parts": parts}, produced=tuple(names)))
        if wants_ml:
            cyc = [f"{col}__month_sin", f"{col}__month_cos"]
            plan.cyclical_columns[col] = cyc
            plan.record(Decision(
                Action.ADD_CYCLICAL, col,
                "December and January are adjacent; a raw month number says they are 11 apart",
                {}, produced=tuple(cyc)))

    return plan


def suggest_ratio_features(df, plan: PreparationPlan,
                           pairs: Sequence[Tuple[str, str, str]]) -> PreparationPlan:
    """Add caller-specified ratios, e.g. rooms per household.

    Ratios are not guessed. Dividing two arbitrary numeric columns produces
    noise that looks like a feature, so the planner (which knows the user's
    objective) names the pairs and this records them.
    """
    for numerator, denominator, name in pairs:
        if numerator in df.columns and denominator in df.columns:
            plan.ratio_features.append((numerator, denominator, name))
            plan.record(Decision(Action.ADD_RATIO, numerator,
                                 f"{numerator} per {denominator}",
                                 {"denominator": denominator}, produced=(name,)))
    return plan


# --------------------------------------------------------------------------- #
# Apply
# --------------------------------------------------------------------------- #

def apply_preparation(df, plan: PreparationPlan):
    """Apply a fitted plan to any frame. Consults **no statistic** of *df*."""
    import numpy as np
    import pandas as pd

    out = df.copy()

    for col, _ in plan.numeric_coercions.items():
        if col in out.columns:
            out[col] = out[col].map(_clean_numeric_token).astype("float64")

    for col in plan.datetime_columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    for col, mapping in plan.boolean_coercions.items():
        if col in out.columns:
            lowered = out[col].astype(str).str.strip().str.lower()
            out[col] = lowered.map(mapping).astype("object")

    for col in plan.text_normalisations:
        if col in out.columns and _is_object_like(out[col]):
            out[col] = out[col].astype(str).str.strip()
            out.loc[out[col].str.lower().isin(_NULL_TOKENS), col] = np.nan

    # Indicators are computed BEFORE imputation — after it, there is nothing left
    # to indicate.
    for name in plan.indicator_columns:
        source = name[: -len("__was_missing")]
        if source in out.columns:
            out[name] = out[source].isna().astype("int8")

    for col, spec in plan.imputations.items():
        if col in out.columns and spec.get("value") is not None:
            out[col] = out[col].fillna(spec["value"])

    for col, rare in plan.rare_categories.items():
        if col in out.columns:
            as_text = out[col].astype("object")
            mask = as_text.isin(rare)
            out.loc[mask, col] = "__other__"

    for col, names in plan.date_part_columns.items():
        if col in out.columns:
            stamps = pd.to_datetime(out[col], errors="coerce")
            for name in names:
                part = name.rsplit("__", 1)[1]
                out[name] = getattr(stamps.dt, part)

    for col, names in plan.cyclical_columns.items():
        if col in out.columns:
            month = pd.to_datetime(out[col], errors="coerce").dt.month
            out[names[0]] = np.sin(2 * np.pi * month / 12.0)
            out[names[1]] = np.cos(2 * np.pi * month / 12.0)

    for numerator, denominator, name in plan.ratio_features:
        if numerator in out.columns and denominator in out.columns:
            denom = pd.to_numeric(out[denominator], errors="coerce")
            num = pd.to_numeric(out[numerator], errors="coerce")
            out[name] = np.where(denom.isna() | (denom == 0), np.nan, num / denom)

    return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _try_datetime(series) -> Optional[float]:
    import pandas as pd
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None
    # A bare integer year or id parses as a date under some settings; require a
    # separator so "20240101" alone does not become a timestamp by accident.
    if not non_null.str.contains(r"[-/:]").any():
        return None
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    rate = float(parsed.notna().mean())
    return rate if rate >= NUMERIC_COERCION_CONFIDENCE else None


_BOOL_PAIRS = (
    ({"yes", "no"}, {"yes": True, "no": False}),
    ({"true", "false"}, {"true": True, "false": False}),
    ({"y", "n"}, {"y": True, "n": False}),
    ({"t", "f"}, {"t": True, "f": False}),
)


def _boolean_mapping(series) -> Optional[Dict[str, bool]]:
    values = set(series.dropna().astype(str).str.strip().str.lower().unique())
    if not values or len(values) > 2:
        return None
    for expected, mapping in _BOOL_PAIRS:
        if values <= expected:
            return mapping
    return None


def _native_datetime_columns(df, cols) -> List[str]:
    import pandas as pd
    return [c for c in cols
            if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c])]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- #
# Graph node
# --------------------------------------------------------------------------- #

PREPARATION_SPEC = AgentSpec(
    name="preparation",
    stage=Stage.PREPARE,
    reads=("dataframe",),
    writes=("preparation_plan", "prepared_dataframe", "preparation_summary"),
    description=("Cleanses values (currency marks, null tokens, types), imputes "
                 "missing data with a stated reason, and engineers features "
                 "appropriate to the objective. Fitted on training rows only."),
)


@agent(PREPARATION_SPEC)
def preparation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node.

    Reads ``objective`` and ``target_column`` from the planner's state, so the
    same dataset prepared for a model and prepared for a report produces
    different — and appropriate — frames.

    When ``train_index`` is present the plan is fitted on those rows alone and
    applied to the whole frame. That is the leakage-safe path and the planner
    should always supply it for a modelling objective.
    """
    df = state.get("dataframe")
    if df is None:
        return {}

    objective = state.get("objective") or Objective.BOTH
    try:
        objective = Objective(objective)
    except ValueError:
        logger.warning("[preparation] unknown objective %r; preparing for both", objective)
        objective = Objective.BOTH

    train_index = state.get("train_index")
    fit_frame = df.loc[train_index] if train_index is not None else df
    if train_index is None and objective is not Objective.ANALYSIS:
        logger.warning(
            "[preparation] no train_index supplied for a modelling objective; fitting on "
            "the full frame. Imputation values and category vocabularies will have seen "
            "the evaluation rows.")

    plan = fit_preparation(fit_frame, objective=objective,
                           target=state.get("target_column"),
                           columns=state.get("feature_columns"))
    pairs = state.get("ratio_features") or ()
    if pairs:
        plan = suggest_ratio_features(fit_frame, plan, pairs)

    prepared = apply_preparation(df, plan)
    logger.info("[preparation] %s (fitted on %d rows)", plan.summary(), plan.fitted_rows)
    return {
        "preparation_plan": plan.as_dict(),
        "prepared_dataframe": prepared,
        "preparation_summary": plan.summary(),
    }
