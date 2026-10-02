"""Measured, goal-specific results for the local ``avaloka analyze`` mission.

This is deliberately deterministic. A goal selects a supported calculation;
the report never asks an LLM to manufacture an answer from a generic profile.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _mentioned_columns(goal: str, columns: list[str]) -> list[str]:
    return [
        col for col in columns
        if re.search(rf"(?<!\w){re.escape(col)}(?!\w)", goal, re.IGNORECASE)
    ]


def _number(value: Any) -> float:
    return round(float(value), 6)


def _display_number(value: Any) -> str:
    """Format a measured value without hiding it behind scientific notation."""
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.6f}".rstrip("0").rstrip(".")


def _requested_numeric_metrics(goal: str) -> list[str]:
    """Return numeric statistics explicitly requested in a natural-language goal.

    Keep this vocabulary column-agnostic: the same intent detection applies to
    salary, revenue, temperature, or any other numeric column.  When no
    statistic is requested, callers retain the broader mean/median summary.
    """
    patterns = (
        ("max", r"\b(highest|maximum|max|largest|greatest)\b"),
        ("min", r"\b(lowest|minimum|min|smallest)\b"),
        ("sum", r"\b(total|sum)\b"),
        ("mean", r"\b(average|mean)\b"),
        ("median", r"\bmedian\b"),
        ("count", r"\b(count|how many|number of|non[- ]?null)\b"),
        ("range", r"\b(range|spread)\b"),
    )
    requested = [name for name, pattern in patterns if re.search(pattern, goal, re.IGNORECASE)]
    if "range" in requested:
        for bound in ("min", "max"):
            if bound not in requested:
                requested.append(bound)
    return requested


def analyze_goal(
    frame: pd.DataFrame, goal: str, profile: dict[str, Any], *, sampled: bool = False,
) -> dict[str, Any]:
    """Answer supported goals from actual working rows, or disclose the gap.

    Results are descriptive, never causal. The scope is explicit because the
    working frame can be a sample for large input datasets.
    """
    lower = goal.lower()
    columns = [str(col) for col in frame.columns]
    mentioned = _mentioned_columns(goal, columns)
    scope = f"{'Sample of' if sampled else 'All'} {len(frame):,} working rows"
    result: dict[str, Any] = {
        "goal": goal, "kind": "dataset_overview", "status": "limited",
        "scope": scope, "columns": mentioned, "metrics": {}, "findings": [],
    }

    if re.search(r"\b(missing|missingness|null|nan|imput|incomplete)\w*\b", lower):
        selected = mentioned or columns
        counts = {col: int(frame[col].isna().sum()) for col in selected}
        ordered = sorted(counts, key=lambda col: (-counts[col], col))[:10]
        result.update(kind="missingness", status="answered", metrics={
            col: {
                "missing": counts[col],
                "missing_pct": _number(counts[col] / len(frame) * 100) if len(frame) else 0.0,
            }
            for col in ordered
        })
        result["findings"] = [
            f"{col}: {counts[col]:,} missing of {len(frame):,} rows "
            f"({result['metrics'][col]['missing_pct']:.1f}%)."
            for col in ordered
        ] or ["No columns are available to check for missing values."]
        if "imput" in lower:
            for col in ordered:
                if not counts[col]:
                    continue
                if pd.api.types.is_numeric_dtype(frame[col]):
                    strategy = "median imputation plus a missingness indicator"
                else:
                    strategy = "an explicit 'Unknown' category plus a missingness indicator"
                result["findings"].append(f"For {col}, consider {strategy}; validate it before use.")
            result["findings"].append(
                "No imputation was applied. For modelling, fit imputation values on training rows only."
            )
        return result

    if re.search(r"\b(correlat\w*|relationship|associated)\b", lower):
        numeric = frame.select_dtypes(include="number")
        selected = [col for col in mentioned if col in numeric.columns]
        pairs: list[dict[str, Any]] = []
        if numeric.shape[1] >= 2:
            corr = numeric.corr()
            for i, left in enumerate(corr.columns):
                for right in corr.columns[i + 1:]:
                    value = corr.loc[left, right]
                    if len(selected) >= 2:
                        include = left in selected and right in selected
                    elif selected:
                        include = left in selected or right in selected
                    else:
                        include = True
                    if pd.notna(value) and include:
                        pairs.append({"left": str(left), "right": str(right), "r": _number(value)})
        pairs.sort(key=lambda pair: (-abs(pair["r"]), pair["left"], pair["right"]))
        result.update(
            kind="correlation", status="answered" if pairs else "limited",
            metrics={"pairs": pairs[:5]},
        )
        result["findings"] = [
            f"{pair['left']} and {pair['right']}: Pearson r={pair['r']:.3f}."
            for pair in pairs[:5]
        ] or ["No comparable numeric column pair has a defined correlation."]
        result["findings"].append("Correlation alone does not establish causation.")
        return result

    if re.search(r"\b(trend|over time|by month|monthly|quarter|q[1-4]|rose|fell|drop)\b", lower):
        date_cols = [
            col["name"] for col in profile.get("columns", [])
            if col.get("role") == "datetime" and col["name"] in frame
        ]
        date_col = next((col for col in mentioned if col in date_cols), None)
        date_col = date_col or (date_cols[0] if date_cols else None)
        if not date_col:
            result.update(kind="trend", findings=[
                "A time trend cannot be calculated: no date/time column was identified."
            ])
            return result
        dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
        numeric = [
            col for col in mentioned
            if col != date_col and pd.api.types.is_numeric_dtype(frame[col])
        ]
        metric = numeric[0] if numeric else None
        series = (
            pd.to_numeric(frame[metric], errors="coerce")
            if metric
            else pd.Series(1, index=frame.index, dtype="float64")
        )
        table = pd.DataFrame({
            "period": dates.dt.tz_localize(None).dt.to_period("M"),
            "value": series,
        }).dropna()
        aggregation = (
            "sum" if not metric or re.search(r"\b(total|sum|revenue|sales)\b", lower)
            else "mean"
        )
        monthly = getattr(table.groupby("period")["value"], aggregation)()
        values = {str(period): _number(value) for period, value in monthly.items()}
        causal_question = bool(re.search(r"\b(why|cause|caused|driver|drivers)\b", lower))
        result.update(
            kind="trend",
            status="answered" if len(values) >= 2 and not causal_question else "limited",
            metrics={
                "time_column": date_col,
                "value_column": metric,
                "aggregation": aggregation,
                "monthly_values": values,
            },
        )
        if len(values) >= 2:
            first, last = next(iter(values.items())), next(reversed(values.items()))
            measure = f"{aggregation} {metric}" if metric else "row count"
            result["findings"] = [
                f"Monthly {measure} changed from {first[1]:.3g} in {first[0]} "
                f"to {last[1]:.3g} in {last[0]} (change {last[1] - first[1]:+.3g}).",
                "This is a descriptive change, not an explanation of its cause.",
            ]
        else:
            result["findings"] = ["At least two months with valid values are needed for a time trend."]
        return result

    by_match = re.search(r"\bby\s+([\w ]+)", goal, re.IGNORECASE)
    group_col = None
    if by_match:
        group_col = next((
            col for col in mentioned
            if re.match(rf"{re.escape(col)}(?:\b|$)", by_match.group(1), re.IGNORECASE)
        ), None)
    if group_col or re.search(r"\b(segment|compare|breakdown|grouped)\b", lower):
        group_col = group_col or next((
            col for col in mentioned if not pd.api.types.is_numeric_dtype(frame[col])
        ), None)
        if not group_col:
            result.update(kind="group_comparison", findings=[
                "A grouped comparison needs a group column named in the goal."
            ])
            return result
        metric = next((
            col for col in mentioned
            if col != group_col and pd.api.types.is_numeric_dtype(frame[col])
        ), None)
        grouped = frame.groupby(group_col, dropna=False)
        if metric:
            aggregation = "sum" if re.search(r"\b(total|sum|revenue|sales)\b", lower) else "mean"
            values = getattr(grouped[metric], aggregation)().sort_values(ascending=False).head(10)
        else:
            aggregation = "count"
            values = grouped.size().sort_values(ascending=False).head(10)
        rows = [
            {"group": str(key), "value": _number(value)}
            for key, value in values.items() if pd.notna(value)
        ]
        result.update(
            kind="group_comparison", status="answered" if rows else "limited",
            metrics={
                "group_column": group_col,
                "value_column": metric,
                "aggregation": aggregation,
                "groups": rows,
            },
        )
        result["findings"] = [
            f"{group_col}={row['group']}: {aggregation} {metric or 'rows'}={row['value']:.3g}."
            for row in rows[:5]
        ] or ["No groups with valid values were available to compare."]
        return result

    if mentioned:
        summaries: dict[str, Any] = {}
        findings: list[str] = []
        requested_metrics = _requested_numeric_metrics(goal)
        for col in mentioned[:10]:
            series = frame[col]
            if pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce").dropna()
                metrics = requested_metrics or ["mean", "median"]
                summary: dict[str, Any] = {"non_null": int(len(values))}
                if len(values):
                    calculations = {
                        "max": values.max,
                        "min": values.min,
                        "sum": values.sum,
                        "mean": values.mean,
                        "median": values.median,
                        "range": lambda: values.max() - values.min(),
                    }
                    for metric in metrics:
                        if metric != "count":
                            summary[metric] = _number(calculations[metric]())
                else:
                    for metric in metrics:
                        if metric != "count":
                            summary[metric] = None
                summaries[col] = summary
                if not len(values):
                    findings.append(f"{col}: no numeric values to summarise.")
                    continue
                labels = {
                    "max": "highest",
                    "min": "lowest",
                    "sum": "total",
                    "mean": "mean",
                    "median": "median",
                    "count": "non-null count",
                    "range": "range",
                }
                parts = [
                    f"{labels[metric]}=" + _display_number(
                        summary["non_null"] if metric == "count" else summary[metric]
                    )
                    for metric in metrics
                ]
                findings.append(
                    f"{col}: {', '.join(parts)} (from {len(values):,} non-null rows)."
                )
            else:
                counts = series.astype("string").value_counts().head(3)
                summaries[col] = {
                    "non_null": int(series.notna().sum()),
                    "top_values": {str(key): int(value) for key, value in counts.items()},
                }
                findings.append(
                    f"{col}: {int(series.notna().sum()):,} non-null rows; "
                    + (
                        "most common: " + ", ".join(
                            f"{key} ({value})" for key, value in counts.items()
                        ) if len(counts) else "no values to summarise."
                    )
                )
        result.update(
            kind="column_summary", status="answered", metrics=summaries,
            findings=findings,
        )
        return result

    result["findings"] = [
        "This goal does not map to a supported goal-specific calculation. "
        "The remaining report is a general dataset profile, not an answer to the question."
    ]
    return result
