from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Union

from app.core.inference import build_chat_model

from app.graph.etl_state import ETLState
from app.core.model_config import resolve as resolve_model
from app.core.model_fallback import attach_fallback
from app.core.log_utils import describe_response

logger = logging.getLogger(__name__)

Numeric = Union[int, float]

# -------------------------------------------------------------------
# LLM setup (Groq) – used as the "feature-aware chart picker"
# -------------------------------------------------------------------

_VIZ_API_KEY = os.environ.get("GROQ_API_KEY")
viz_llm: Optional[Any] = build_chat_model(
    role="viz",
    agent="VIZ",
    tier="large",
    temperature=0.2,
    groq_model=resolve_model("visualization"),
    groq_api_key=_VIZ_API_KEY,
)
if viz_llm is not None:
    logger.info("Visualization LLM enabled for chart selection")
else:
    logger.warning(
        "Visualization LLM disabled; set GROQ_API_KEY to enable intelligent chart suggestions."
    )

# -------------------------------------------------------------------
# Low-level helpers
# -------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """Define 'missing' for profiling."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        v = value.strip().lower()
        return v in ("", "na", "nan", "null", "none")
    return False


def _to_float(value: Any) -> Optional[float]:
    """Best-effort conversion to float, respecting _is_missing."""
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            # handle "1,234,567"
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


# -------------------------------------------------------------------
# Column profiling  (acts as a basic "feature agent" input)
# -------------------------------------------------------------------


def profile_columns(
    sample_rows: List[Dict[str, Any]],
    schema: Optional[Union[List[str], Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build per-column profiles:
    - name, dtype ("numeric" / "categorical")
    - missing_ratio
    - n_unique
    - stats: numeric -> {min, max, mean, std, skewness}
             categorical -> {top_k: [{value, count}, ...]}
    """
    if not sample_rows:
        return []

    # Decide which columns to consider
    if isinstance(schema, dict):
        cols = list(schema.keys())
    elif isinstance(schema, list) and schema:
        cols = list(schema)
    else:
        cols = list(sample_rows[0].keys())

    profiles: List[Dict[str, Any]] = []
    n_rows = len(sample_rows)

    for col in cols:
        values = [row.get(col) for row in sample_rows]
        missing_count = sum(1 for v in values if _is_missing(v))
        missing_ratio = missing_count / n_rows if n_rows else 0.0

        # Try to interpret as numeric
        numeric_vals: List[float] = []
        for v in values:
            fv = _to_float(v)
            if fv is not None:
                numeric_vals.append(fv)

        # heuristic: enough numeric values => numeric column
        if numeric_vals and len(numeric_vals) >= max(3, n_rows * 0.1):
            dtype = "numeric"
            n_unique = len(set(numeric_vals))

            stats: Dict[str, Any] = {
                "min": min(numeric_vals),
                "max": max(numeric_vals),
                "mean": statistics.fmean(numeric_vals),
            }
            stats["std"] = (
                statistics.pstdev(numeric_vals) if len(numeric_vals) > 1 else 0.0
            )

            # Fisher–Pearson skewness
            if len(numeric_vals) > 2 and stats["std"] > 0:
                m = stats["mean"]
                s = stats["std"]
                n = len(numeric_vals)
                third_moment = sum((x - m) ** 3 for x in numeric_vals) / n
                stats["skewness"] = third_moment / (s**3)
            else:
                stats["skewness"] = 0.0
        else:
            dtype = "categorical"

            # normalize values for counting
            norm_vals: List[str] = []
            for v in values:
                if _is_missing(v):
                    norm_vals.append("nan")
                else:
                    norm_vals.append(str(v))

            counts = Counter(norm_vals)
            n_unique = len(counts)
            top_k = [
                {"value": val, "count": cnt} for val, cnt in counts.most_common(15)
            ]
            stats = {"top_k": top_k}

        profiles.append(
            {
                "name": col,
                "dtype": dtype,
                "missing_ratio": missing_ratio,
                "n_unique": n_unique,
                "stats": stats,
            }
        )

    return profiles


# -------------------------------------------------------------------
# Unsupervised feature ranking (simple "feature agent" logic)
# -------------------------------------------------------------------


def compute_unsupervised_feature_ranking(
    columns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ranking heuristic:
    - Numeric: use variance proxy (std^2)
    - Categorical: use cardinality * log(cardinality)
    Scores are normalized to [0, 1].
    """
    features: List[Dict[str, Any]] = []

    for col in columns:
        name = col.get("name")
        if not name:
            continue

        dtype = (col.get("dtype") or "").lower()
        n_unique = col.get("n_unique", 0) or 0
        stats = col.get("stats") or {}

        score: float = 0.0

        if dtype in {"numeric", "number", "float", "int", "integer"}:
            std = stats.get("std", 0.0) or 0.0
            score = std * std  # variance proxy
        else:
            if n_unique > 1:
                score = float(n_unique) * math.log(n_unique)

        features.append({"name": name, "score": float(score)})

    max_score = max((f["score"] for f in features), default=1.0)
    if max_score <= 0:
        max_score = 1.0

    for f in features:
        f["score"] /= max_score

    return {
        "strategy": "variance+cardinality",
        "features": sorted(features, key=lambda f: f["score"], reverse=True),
    }


# -------------------------------------------------------------------
# Heuristic chart selection policy (fallback / top-up)
# -------------------------------------------------------------------

MAX_TOTAL_CHARTS = 5
MAX_DIST_CHARTS = 2
MAX_RELATIONSHIP_CHARTS = 2
MAX_CATEGORY_CHARTS = 1
LOW_CARDINALITY_THRESHOLD = 25


def select_charts(viz_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Given a viz_profile with dataset / columns / feature_ranking,
    choose 3-5 charts:
      - 1–2 histograms on important numeric features
      - 1–2 scatterplots (e.g., rank/index vs other numeric)
      - 0–1 bar chart on a low-cardinality categorical feature
    """
    dataset = viz_profile.get("dataset", {}) or {}
    rows_sampled = dataset.get("rows_sampled")
    columns = viz_profile.get("columns", []) or []
    feature_ranking = viz_profile.get("feature_ranking", {}) or {}
    ranked_features = feature_ranking.get("features", []) or []

    # Group columns by type
    numeric_cols: List[Dict[str, Any]] = []
    categorical_cols: List[Dict[str, Any]] = []

    for col in columns:
        name = col.get("name")
        if not name:
            continue

        dtype = (col.get("dtype") or "").lower()
        n_unique = col.get("n_unique", 0) or 0

        if dtype in {"numeric", "number", "float", "int", "integer"}:
            numeric_cols.append(col)
        elif dtype in {"categorical", "category", "string"}:
            categorical_cols.append(col)
        else:
            # fallback based on cardinality
            if n_unique > LOW_CARDINALITY_THRESHOLD * 2:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    scores = {f["name"]: f.get("score", 0.0) for f in ranked_features}

    # Important numeric features first
    important_numeric = sorted(
        numeric_cols,
        key=lambda c: scores.get(c["name"], 0.0),
        reverse=True,
    )

    charts: List[Dict[str, Any]] = []

    # 1) Distributions of important numeric features
    dist_count = 0
    for col in important_numeric:
        if dist_count >= MAX_DIST_CHARTS or len(charts) >= MAX_TOTAL_CHARTS:
            break

        field = col["name"]
        charts.append(
            {
                "id": f"dist_{field}",
                "title": f"Distribution of {field}",
                "type": "histogram",
                "intent": "distribution",
                "rank": len(charts) + 1,
                "data_source": "sample",
                "encodings": {
                    "x": {"field": field, "type": "quantitative", "bin": True},
                    "y": {"aggregate": "count", "type": "quantitative"},
                },
                "config": {"nbins": 30},
                "derived_data": {},
            }
        )
        dist_count += 1

    # 2) Relationship charts (scatter)
    if len(charts) < MAX_TOTAL_CHARTS and len(important_numeric) >= 2:

        def looks_like_rank(name: str) -> bool:
            lname = name.lower()
            return any(
                token in lname for token in ("rank", "index", "id", "time", "date", "year")
            )

        # Prefer a rank/index/time-like column for X axis
        x_col: Optional[Dict[str, Any]] = None
        for col in important_numeric:
            if looks_like_rank(col["name"]):
                x_col = col
                break

        if x_col is None:
            x_col = important_numeric[0]

        x_name = x_col["name"]
        rel_count = 0
        for y_col in important_numeric:
            if y_col["name"] == x_name:
                continue

            if rel_count >= MAX_RELATIONSHIP_CHARTS or len(charts) >= MAX_TOTAL_CHARTS:
                break

            y_name = y_col["name"]
            charts.append(
                {
                    "id": f"scatter_{x_name}_vs_{y_name}",
                    "title": f"{x_name} vs {y_name}",
                    "type": "scatter",
                    "intent": "relationship",
                    "rank": len(charts) + 1,
                    "data_source": "sample",
                    "encodings": {
                        "x": {"field": x_name, "type": "quantitative"},
                        "y": {"field": y_name, "type": "quantitative"},
                    },
                    "config": {"sample_rows": rows_sampled},
                    "derived_data": {},
                }
            )
            rel_count += 1

    # 3) One categorical bar chart on a low-cardinality important categorical feature
    if len(charts) < MAX_TOTAL_CHARTS:

        def cat_sort_key(c: Dict[str, Any]):
            return (scores.get(c["name"], 0.0), -c.get("n_unique", 0))

        sorted_cats = sorted(categorical_cols, key=cat_sort_key, reverse=True)
        cat_count = 0

        for col in sorted_cats:
            if cat_count >= MAX_CATEGORY_CHARTS or len(charts) >= MAX_TOTAL_CHARTS:
                break

            n_unique = col.get("n_unique", 0) or 0
            # A counts bar chart needs a column whose values REPEAT, not one
            # under an absolute cardinality cap. The chart below already renders
            # only the top 15 categories, so a 49-country column was never going
            # to draw 49 bars -- yet the flat `<= 25` gate skipped it, and a
            # result of (Youtuber, Country) produced no charts at all. Before the
            # agent was fixed to profile the result table that went unnoticed,
            # because it silently charted the uploaded file's numeric columns
            # instead.
            #
            # So: keep the old cap as-is (never removes a chart that used to be
            # drawn), and additionally allow a column that genuinely groups --
            # average group size of 2 or more. Country (49 across 939 rows)
            # qualifies; Youtuber (939 across 939) is an identifier, where every
            # bar would be 1 and the chart says nothing.
            groups_well = (
                isinstance(rows_sampled, int)
                and rows_sampled > 0
                and 1 < n_unique <= rows_sampled / 2
            )
            if not (0 < n_unique <= LOW_CARDINALITY_THRESHOLD or groups_well):
                continue

            field = col["name"]
            charts.append(
                {
                    "id": f"bar_{field}",
                    "title": f"Counts by {field}",
                    "type": "bar",
                    "intent": "distribution",
                    "rank": len(charts) + 1,
                    "data_source": "sample",
                    "encodings": {
                        "x": {"field": field, "type": "nominal"},
                        "y": {"aggregate": "count", "type": "quantitative"},
                    },
                    "config": {"top_k": min(15, n_unique)},
                    "derived_data": {},
                }
            )
            cat_count += 1

    # Normalize ranks
    for i, ch in enumerate(charts, start=1):
        ch["rank"] = i

    return charts


# -------------------------------------------------------------------
# Bias / data-quality diagnostics
# -------------------------------------------------------------------


def detect_bias_and_issues(columns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple checks:
      - constant columns
      - high missing ratio
      - extreme imbalance when n_unique <= 3
    """
    warnings: List[str] = []
    issues: List[Dict[str, Any]] = []

    for col in columns:
        name = col.get("name")
        if not name:
            continue

        n_unique = col.get("n_unique", 0) or 0
        missing_ratio = col.get("missing_ratio", 0.0) or 0.0
        stats = col.get("stats") or {}
        top_k = []
        if isinstance(stats, dict):
            top_k = stats.get("top_k") or []

        if n_unique <= 1:
            msg = (
                f"Column '{name}' has only a single unique value; "
                f"it may not be useful for modeling."
            )
            warnings.append(msg)
            issues.append({"column": name, "type": "constant", "message": msg})

        if missing_ratio > 0.4:
            msg = f"Column '{name}' has a high missing ratio ({missing_ratio:.0%})."
            warnings.append(msg)
            issues.append({"column": name, "type": "missing", "message": msg})

        if top_k and n_unique <= 3:
            total = sum(t.get("count", 0) for t in top_k)
            if total > 0:
                p0 = (top_k[0].get("count", 0) or 0) / total
                if p0 > 0.95:
                    dominant = top_k[0].get("value")
                    msg = (
                        f"Column '{name}' is highly imbalanced: "
                        f"{dominant!r} makes up ~{p0:.0%} of values."
                    )
                    warnings.append(msg)
                    issues.append(
                        {"column": name, "type": "imbalance", "message": msg}
                    )

    return {"enabled": True, "issues": issues, "warnings": warnings}


# -------------------------------------------------------------------
# LLM-based chart selection (primary path)
# -------------------------------------------------------------------


def _llm_suggest_chart_specs(viz_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ask Groq LLM to act as a "feature-aware" viz agent and propose chart specs.

    Returns a list of suggestions like:
    {
        "type": "bar" | "line" | "pie" | "scatter" | "histogram",
        "title": "...",
        "intent": "distribution | comparison | relationship | composition",
        "x_field": "...",
        "y_field": "...",
        "config": {...optional extras...}
    }
    """
    if viz_llm is None:
        return []

    payload = {
        "columns": viz_profile.get("columns", []),
        "feature_ranking": viz_profile.get("feature_ranking", {}),
        "task": viz_profile.get("task", {}),
        "max_charts": MAX_TOTAL_CHARTS,
    }

    system_prompt = (
        "You are a senior data-visualization expert acting as a feature-aware "
        "visualization agent for a BI tool.\n"
        "Given a dataset profile, you MUST choose BETWEEN 3 AND 5 of the most useful charts.\n"
        "You are given `feature_ranking.features`, where higher scores mean more important columns. "
        "Prioritize those high-score columns when picking fields for the charts.\n\n"
        "Allowed chart types: 'bar', 'line', 'pie', 'scatter', 'histogram'.\n"
        "- Use 'histogram' for numeric distributions.\n"
        "- Use 'scatter' for relationships between two numeric fields.\n"
        "- Use 'bar' for comparisons across categories.\n"
        "- Use 'line' for trends over an ordered or time-like field.\n"
        "- Use 'pie' for compositions / proportions over a categorical field.\n\n"
        "Respond with ONLY a JSON array (no markdown, no explanation) of objects.\n"
        "Each object MUST have keys:\n"
        "  - type: one of ['bar','line','pie','scatter','histogram']\n"
        "  - title: short human-readable title\n"
        "  - intent: short phrase like 'distribution', 'comparison', 'relationship', 'composition'\n"
        "  - x_field: column name to use on the x-axis (or categories / slices)\n"
        "  - y_field: column name for numeric values (may be null for histogram or when not needed)\n"
        "  - config: optional object (e.g. {\"aggregate\": \"sum\", \"top_k\": 10})\n"
        "Return between 3 and 5 chart objects in the array."
    )

    user_prompt = (
        "Here is the dataset profile JSON:\n"
        f"{json.dumps(payload, default=str, indent=2)}\n\n"
        "Now return ONLY the JSON array of chart suggestions as described above."
    )

    try:
        resp = viz_llm.invoke(system_prompt + "\n\n" + user_prompt)
        logger.info("Visualization LLM answered: %s", describe_response(resp))
        content = (resp.content or "").strip()

        # Try to extract a JSON array
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1 or end <= start:
            logger.warning("LLM viz response has no JSON array; content=%r", content[:200])
            return []

        json_str = content[start : end + 1]
        suggestions = json.loads(json_str)
        if not isinstance(suggestions, list):
            logger.warning("LLM viz JSON is not a list")
            return []
        return [s for s in suggestions if isinstance(s, dict)]
    except Exception as e:
        logger.warning("LLM chart suggestion failed: %s", e)
        return []


def _charts_from_llm_suggestions(
    suggestions: List[Dict[str, Any]],
    viz_profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Convert LLM suggestions into the same chart schema our frontend expects.
    """
    if not suggestions:
        return []

    dataset = viz_profile.get("dataset", {}) or {}
    rows_sampled = dataset.get("rows_sampled")

    # Column names the profile actually contains. A chart encoding pointing at
    # anything else renders blank in the UI beside an otherwise-correct answer,
    # and every stage still reports success -- so nothing upstream catches it.
    # Feeding the model the right menu (see visualization_agent_node) removes the
    # usual cause; this drops the rest, because a suggestion naming a column we
    # do not have is not a chart we can draw.
    known_fields = {
        str(c.get("name")) for c in (viz_profile.get("columns") or [])
        if isinstance(c, dict) and c.get("name") is not None
    }

    charts: List[Dict[str, Any]] = []
    for idx, s in enumerate(suggestions):
        if len(charts) >= MAX_TOTAL_CHARTS:
            break

        ctype = (s.get("type") or "").lower()
        title = s.get("title") or f"Chart {idx + 1}"
        intent = s.get("intent") or "distribution"
        x_field = s.get("x_field")
        y_field = s.get("y_field")
        extra_cfg = s.get("config") or {}

        if not x_field:
            # we need at least an x_field to do anything sensible
            continue

        # y_field is legitimately absent for a histogram, or when the chart does
        # not need one -- only a field that IS named has to be real.
        named = [f for f in (x_field, y_field) if f]
        unknown = [f for f in named if known_fields and str(f) not in known_fields]
        if unknown:
            logger.warning(
                "Visualization agent: dropping suggested %s chart %r -- it plots "
                "%s, which is not in the table being charted (%s)",
                ctype or "unknown", title, unknown, sorted(known_fields),
            )
            continue

        base = {
            "id": f"llm_{idx}",
            "title": title,
            "type": ctype,
            "intent": intent,
            "rank": idx + 1,
            "data_source": "sample",
            "encodings": {},
            "config": {},
            "derived_data": {},
        }

        if ctype == "histogram":
            base["type"] = "histogram"
            base["encodings"] = {
                "x": {"field": x_field, "type": "quantitative", "bin": True},
                "y": {"aggregate": "count", "type": "quantitative"},
            }
            base["config"] = {"nbins": extra_cfg.get("nbins", 30)}
        elif ctype in ("bar", "line"):
            base["type"] = ctype
            base["encodings"] = {
                "x": {"field": x_field, "type": "nominal"},
                "y": {
                    "field": y_field or x_field,
                    "type": "quantitative",
                    "aggregate": extra_cfg.get("aggregate", "count"),
                },
            }
            base["config"] = {"top_k": extra_cfg.get("top_k", 15)}
        elif ctype == "pie":
            base["type"] = "pie"
            # If y_field is given, use as numeric measure; otherwise count categories
            if y_field:
                agg = extra_cfg.get("aggregate") or "sum"
                base["encodings"] = {
                    "theta": {
                        "field": y_field,
                        "type": "quantitative",
                        "aggregate": agg,
                    },
                    "color": {"field": x_field, "type": "nominal"},
                }
            else:
                agg = extra_cfg.get("aggregate") or "count"
                base["encodings"] = {
                    "theta": {
                        "field": x_field,
                        "type": "quantitative",
                        "aggregate": agg,
                    },
                    "color": {"field": x_field, "type": "nominal"},
                }
            base["config"] = {"top_k": extra_cfg.get("top_k", 10)}
        elif ctype == "scatter":
            if not y_field:
                continue
            base["type"] = "scatter"
            base["encodings"] = {
                "x": {"field": x_field, "type": "quantitative"},
                "y": {"field": y_field, "type": "quantitative"},
            }
            base["config"] = {"sample_rows": rows_sampled}
        else:
            # Unknown type; skip
            continue

        charts.append(base)

    # normalize ranks
    for i, ch in enumerate(charts, start=1):
        ch["rank"] = i

    return charts


def _merge_charts(
    primary: List[Dict[str, Any]],
    extras: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge heuristic charts into LLM charts, avoiding obvious duplicates
    and respecting MAX_TOTAL_CHARTS.
    """
    charts = list(primary)

    def sig(ch: Dict[str, Any]):
        enc = ch.get("encodings") or {}
        x = enc.get("x") or {}
        y = enc.get("y") or {}
        theta = enc.get("theta") or {}
        return (
            ch.get("type"),
            x.get("field") if isinstance(x, dict) else None,
            y.get("field") if isinstance(y, dict) else None,
            theta.get("field") if isinstance(theta, dict) else None,
        )

    seen = {sig(c) for c in charts}

    for extra in extras:
        if len(charts) >= MAX_TOTAL_CHARTS:
            break
        s = sig(extra)
        if s in seen:
            continue
        charts.append(extra)
        seen.add(s)

    for i, ch in enumerate(charts, start=1):
        ch["rank"] = i

    return charts


# -------------------------------------------------------------------
# NEW: attach per-chart "reason" based on type, fields & feature importance
# -------------------------------------------------------------------


def _attach_chart_reasons(
    charts: List[Dict[str, Any]],
    feature_ranking: Dict[str, Any],
    target_column: Optional[str],
) -> None:
    """
    Fill chart['reason'] with a human-friendly explanation if missing.

    Uses:
      - chart type (scatter/bar/line/pie/histogram)
      - encoded fields (x, y, theta, color)
      - feature_ranking importance scores
      - whether the target_column is involved (if provided)
    """
    scores = {
        f["name"]: f.get("score", 0.0)
        for f in feature_ranking.get("features", []) or []
        if f.get("name")
    }

    for ch in charts:
        if ch.get("reason"):
            continue  # don't overwrite if already present

        ctype = (ch.get("type") or "").lower()
        intent = ch.get("intent") or "distribution"
        enc = ch.get("encodings") or {}

        x_field = None
        y_field = None

        # Try to infer x/y/theta/color fields
        if isinstance(enc.get("x"), dict):
            x_field = enc["x"].get("field")
        if isinstance(enc.get("y"), dict):
            y_field = enc["y"].get("field")
        if not y_field and isinstance(enc.get("theta"), dict):
            y_field = enc["theta"].get("field")
        if not x_field and isinstance(enc.get("color"), dict):
            x_field = enc["color"].get("field")

        important_feature = y_field or x_field
        importance = scores.get(important_feature or "", 0.0)

        if importance >= 0.8:
            importance_phrase = "a key driver in this dataset"
        elif importance >= 0.5:
            importance_phrase = "one of the more important fields"
        elif important_feature:
            importance_phrase = "a potentially informative field"
        else:
            importance_phrase = "the overall structure of the data"

        pieces: List[str] = []

        # Base explanation by chart type
        if ctype == "histogram":
            if x_field:
                pieces.append(
                    f"Shows the distribution of **{x_field}**, highlighting spread and outliers."
                )
            else:
                pieces.append("Shows the distribution of a key numeric field.")
        elif ctype == "scatter":
            if x_field and y_field:
                pieces.append(
                    f"Plots **{x_field}** against **{y_field}** to reveal correlations or clusters."
                )
            else:
                pieces.append(
                    "Plots two numeric fields to reveal correlations or clusters."
                )
        elif ctype == "bar":
            if x_field and y_field:
                pieces.append(
                    f"Compares **{y_field}** across categories of **{x_field}**."
                )
            elif x_field:
                pieces.append(
                    f"Shows counts by categories of **{x_field}** for a quick comparison."
                )
            else:
                pieces.append(
                    "Compares aggregated values across different categories."
                )
        elif ctype == "line":
            if x_field and y_field:
                pieces.append(
                    f"Tracks how **{y_field}** changes over **{x_field}**, which often behaves like time or rank."
                )
            else:
                pieces.append(
                    "Shows how a numeric value changes over an ordered field."
                )
        elif ctype == "pie":
            if x_field and y_field and x_field != y_field:
                pieces.append(
                    f"Shows how **{y_field}** is distributed across categories of **{x_field}**."
                )
            elif x_field:
                pieces.append(
                    f"Shows proportional contributions of categories in **{x_field}**."
                )
            else:
                pieces.append("Shows proportional contributions of key categories.")
        else:
            pieces.append("Highlights a pattern that is often useful as a first pass.")

        # Tie in feature importance
        if important_feature:
            pieces.append(
                f"'{important_feature}' is {importance_phrase}, so it’s a good starting point."
            )

        # If this chart touches the target column (when there is one), mention it
        if target_column and (
            x_field == target_column or y_field == target_column
        ):
            pieces.append(
                f"This chart directly relates to the target column **{target_column}**, "
                "helping you see how it behaves."
            )

        ch["reason"] = " ".join(pieces)


# -------------------------------------------------------------------
# Top-level API: build_visualization_config_from_sample
# -------------------------------------------------------------------


def build_visualization_config_from_sample(
    dataset_id: str,
    sample_rows: List[Dict[str, Any]],
    schema: Optional[Union[List[str], Dict[str, str]]] = None,
    task_type: str = "unsupervised",
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry-point for the Viz Agent.

    Inputs:
      - dataset_id: Avaloka dataset id
      - sample_rows: list of row dicts (your uploaded_csv_preview / sample)
      - schema: optional schema (list of column names or {col: dtype})
      - task_type: "unsupervised" by default; can be extended later
      - target_column: name of label/target if present

    Output matches your visualization_config structure (plus visualization_status).
    """
    # 1) Profile columns (basic "feature agent" behavior)
    column_profiles = profile_columns(sample_rows, schema=schema)

    # 2) Unsupervised feature ranking
    feature_ranking = compute_unsupervised_feature_ranking(column_profiles)

    rows_sampled = len(sample_rows)

    viz_profile: Dict[str, Any] = {
        "version": "1.0",
        "dataset": {
            "dataset_id": dataset_id,
            "rows_sampled": rows_sampled,
            "columns": [c["name"] for c in column_profiles],
        },
        "task": {
            "type": task_type,
            "target_column": target_column,
            "reason": (
                "No target column provided; using structure-only views."
                if not target_column
                else "Supervised: target column provided."
            ),
        },
        "columns": column_profiles,
        "feature_ranking": feature_ranking,
        "feature_agent": {
            "strategy": feature_ranking.get("strategy"),
            "status": "computed",
        },
    }

    # 3) Primary path: let LLM choose the chart types / fields
    charts: List[Dict[str, Any]] = []
    llm_used = False
    topped_up = False

    if viz_llm is not None:
        suggestions = _llm_suggest_chart_specs(viz_profile)
        llm_charts = _charts_from_llm_suggestions(suggestions, viz_profile)

        if llm_charts:
            charts = llm_charts
            llm_used = True

            # Ensure we have at least 3 charts; if not, top up with heuristics
            if len(charts) < 3:
                heuristic_charts = select_charts(viz_profile)
                before_len = len(charts)
                charts = _merge_charts(charts, heuristic_charts)
                if len(charts) > before_len:
                    topped_up = True

    # 4) Fallback if LLM is disabled or produced nothing useful
    if not charts:
        charts = select_charts(viz_profile)
        llm_reason = (
            "fallback heuristic charts (LLM unavailable or returned no valid suggestions)"
        )
    elif llm_used and topped_up:
        llm_reason = (
            "charts primarily selected by Groq LLM based on dataset profile, "
            "topped up with heuristic defaults to reach at least 3 charts"
        )
    else:
        llm_reason = "charts selected by Groq LLM based on dataset profile"

    # attach per-chart reasons here
    _attach_chart_reasons(charts, feature_ranking, target_column)

    # 5) Bias/data-quality diagnostics
    bias_diag = detect_bias_and_issues(column_profiles)

    # 6) Attach diagnostics & status
    viz_profile["charts"] = charts
    viz_profile["model_diagnostics"] = {
        "enabled": False,
        "reason": "Baseline model diagnostics not yet implemented in auto-viz.",
    }
    viz_profile["bias_diagnostics"] = bias_diag
    viz_profile["warnings"] = bias_diag.get("warnings", [])
    viz_profile["visualization_status"] = "ready"
    viz_profile["llm_selection_reason"] = llm_reason

    return viz_profile


# -------------------------------------------------------------------
# Helper: normalize sample rows for the node
# -------------------------------------------------------------------


def _normalize_sample_rows(
    raw_rows: Union[str, List[Any]],
    columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Ensure we always end up with a list[dict] rows, which
    build_visualization_config_from_sample expects.
    """
    if not raw_rows:
        return []

    # state["sample_data"] is a CSV string (see ETLState); parse it into rows
    # rather than iterating the string character by character (MAJ-181).
    if isinstance(raw_rows, str):
        try:
            reader = csv.DictReader(io.StringIO(raw_rows))
            return [dict(row) for row in reader]
        except Exception:
            logger.warning("Visualization agent: failed parsing sample_data CSV", exc_info=True)
            return []

    # Already dicts
    if isinstance(raw_rows[0], dict):
        return raw_rows  # type: ignore[return-value]

    # Rows as lists + column names from state
    if isinstance(raw_rows[0], list) and columns:
        out: List[Dict[str, Any]] = []
        for row in raw_rows:
            row_dict = {
                columns[i]: row[i]
                for i in range(min(len(columns), len(row)))
            }
            out.append(row_dict)
        return out

    # Fallback: best-effort dict-ification with synthetic column names
    if isinstance(raw_rows[0], list):
        n_cols = len(raw_rows[0])
        col_names = columns or [f"col_{i}" for i in range(n_cols)]
        out: List[Dict[str, Any]] = []
        for row in raw_rows:
            row_dict = {
                col_names[i]: row[i]
                for i in range(min(len(col_names), len(row)))
            }
            out.append(row_dict)
        return out

    # If it's something weird, just wrap in a single-column dict
    return [{"value": r} for r in raw_rows]


# -------------------------------------------------------------------
# LangGraph node wrapper: visualization_agent_node
# -------------------------------------------------------------------


def visualization_agent_node(state: ETLState) -> ETLState:
    """
    LangGraph node function for the auto-viz agent.

    It:
      - Chooses a data sample (execution result if available, otherwise upload preview)
      - Calls build_visualization_config_from_sample(...)
      - Writes visualization_config and visualization_status back into state
    """
    try:
        logger.info("--- Entering Visualization Agent (LLM + auto-viz) ---")

        # Prefer execution result rows if present
        execution_result = state.get("execution_result") or {}
        sample_rows_raw: List[Any] = []

        # Chart the computed result — but only when the user actually produced a
        # transformation worth charting: a multi-row result whose columns aren't
        # just the uploaded file echoed back. Scalars, empty results and raw
        # echoes fall through to the uploaded-data overview below. (The old
        # execution_result["output_json"]/["output_data"] keys are never populated
        # in state, so this replaces that dead fallback path — MAJ-180.)
        uploaded_cols = state.get("uploaded_csv_columns") or []

        # Which columns to profile. The uploaded schema is correct only while we
        # are charting the uploaded file. The moment we chart a COMPUTED result
        # it is wrong, and profile_columns trusts the schema over the rows it was
        # handed -- it iterates schema.keys() and row.get()s each name, so every
        # input column absent from the result profiles as 100% missing with a
        # single "nan" value. The LLM is then offered a menu of phantom columns
        # and picks chart fields from them, which is how "how many countries are
        # in this data?" returned a correct one-column answer next to charts of
        # `video views`. Track it alongside sample_rows_raw so the two can never
        # describe different tables.
        profile_schema: Any = state.get("schema")

        try:
            from app.api.helpers import _datauri_csv_to_records
            ofd = state.get("output_file_data")
            result_rows = (
                _datauri_csv_to_records(ofd["content"])
                if isinstance(ofd, dict) and ofd.get("content") else None
            )
            # Any non-empty result, not >= 2 rows. The old bound sent every
            # single-row answer -- a count, an average, any scalar the user asked
            # for -- to the uploaded-file overview instead, so the chart beside
            # the answer described a different table entirely. One row is still
            # this run's result; charting the input instead is never the more
            # useful thing to show.
            if result_rows:
                is_echo = bool(uploaded_cols) and set(result_rows[0].keys()).issuperset(uploaded_cols)
                if not is_echo:
                    sample_rows_raw = result_rows
                    profile_schema = list(result_rows[0].keys())
        except Exception:
            logger.warning("Visualization agent: failed reading computed result; using upload", exc_info=True)

        # Fallback to uploaded preview / sample_data from state
        if not sample_rows_raw:
            sample_rows_raw = (
                state.get("uploaded_csv_preview")
                or state.get("sample_data")
                or []
            )
        sample_rows = _normalize_sample_rows(sample_rows_raw, uploaded_cols)

        if not sample_rows:
            logger.warning("Visualization agent: no data available to profile")
            new_state = state.copy()
            new_state["visualization_config"] = {}
            new_state["visualization_status"] = "skipped: no data available"
            return new_state

        dataset_id = state.get("dataset_id", "unknown")

        viz_config = build_visualization_config_from_sample(
            dataset_id=dataset_id,
            sample_rows=sample_rows,
            schema=profile_schema,
            task_type="unsupervised",
            target_column=None,
        )

        logger.info(
            "Visualization agent: built config with %d charts",
            len(viz_config.get("charts", []) or []),
        )

        new_state = state.copy()
        new_state["visualization_config"] = viz_config
        new_state["visualization_status"] = viz_config.get(
            "visualization_status", "ready"
        )
        return new_state

    except Exception as e:
        logger.error("Visualization agent failed: %s", e, exc_info=True)
        new_state = state.copy()
        new_state["visualization_config"] = {}
        new_state["visualization_status"] = f"error: {e}"
        return new_state





