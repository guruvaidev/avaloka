"""Measured, goal-specific results for the local ``avaloka analyze`` mission.

This is deliberately deterministic. A goal selects a supported calculation;
the report never asks an LLM to manufacture an answer from a generic profile.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

#: The largest in-memory dataset a goal's figures are computed over in full.
#: Below it a grouped mean or a total over every row costs seconds, so a
#: figure is never estimated from the sample when the exact one is at hand.
FULL_PASS_MAX_ROWS = 10_000_000

#: A driver target needs this many rows on each side of the outcome before an
#: association with it means anything.
MIN_OUTCOME_ROWS = 30

#: Name words that mark a numeric column as a label, not a quantity: the
#: mean of ``payment_type`` or ``RatecodeID`` is the mean of category codes.
_CODE_TOKENS = frozenset({
    "type", "code", "codes", "id", "ids", "key", "zip", "zipcode", "postal", "postcode",
    "flag", "status", "category", "cat", "class", "kind",
})


def _name_tokens(name: Any) -> set[str]:
    """``RatecodeID`` -> {ratecode, id}; ``payment_type`` -> {payment, type}."""
    return {t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(name))}


def is_categorical_code(name: Any, series: pd.Series) -> bool:
    """True for a numeric column that holds category codes, not quantities.

    The rule is the one a reader applies: whole numbers in a column whose name
    says it is a type, code, identifier, status or class. Averaging such a
    column ("mean of payment_type: 1.16") is arithmetic on labels.
    """
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    if not _name_tokens(name) & _CODE_TOKENS:
        return False
    values = pd.to_numeric(series, errors="coerce").dropna()
    if not len(values):
        return False
    return bool(np.all(np.isclose(values.to_numpy(dtype="float64") % 1, 0)))


def _is_quantity(frame: pd.DataFrame, col: str) -> bool:
    """A numeric column whose mean or total means something."""
    return (pd.api.types.is_numeric_dtype(frame[col]) and not pd.api.types.is_bool_dtype(frame[col])
            and not is_categorical_code(col, frame[col]))

from avaloka.io.loader import NULL_WORDS


def _mentioned_columns(goal: str, columns: list[str]) -> list[str]:
    return [
        col for col in columns
        if re.search(rf"(?<!\w){re.escape(col)}(?!\w)", goal, re.IGNORECASE)
    ]


#: Words in a goal that never name a measure, whatever the columns are called.
_NOT_A_MEASURE = frozenset({
    "the", "and", "for", "per", "each", "every", "all", "what", "which", "show",
    "total", "sum", "average", "mean", "median", "count", "number", "highest",
    "lowest", "maximum", "minimum", "max", "min", "largest", "smallest", "over",
    "time", "month", "monthly", "trend", "compare", "breakdown", "segment",
    "grouped", "group", "how", "many", "much", "between", "across", "value",
})


def _unresolved_measure(goal: str, frame: pd.DataFrame, exclude: list[str]) -> dict[str, Any] | None:
    """The measure a goal asks for when no column it names can supply it.

    "Total revenue by region" over ``gross_revenue`` and ``net_revenue`` names
    no column: ``_mentioned_columns`` matches whole names, and "revenue" is
    neither. Falling back to counting rows answers a different question and
    labelled it answered. A goal is unresolved when it asks for a statistic
    other than a count, or uses a word that is part of a numeric column's
    name (identifier columns aside); a goal that asks only for a count is not; the numeric columns sharing a word with the goal are returned as the
    candidates, so the report can say which ones the user could mean rather
    than pick one of them.
    """
    requested = _requested_numeric_metrics(goal)
    statistics = [m for m in requested if m != "count"]
    if "count" in requested and not statistics:
        return None                       # "how many ... by region": a row count is the answer
    words = {w for w in re.findall(r"[a-z0-9]+", goal.lower())
             if len(w) >= 3 and w not in _NOT_A_MEASURE}
    candidates = [
        str(col) for col in frame.columns
        if str(col) not in exclude and pd.api.types.is_numeric_dtype(frame[col])
        and "id" not in (tokens := set(re.findall(r"[a-z0-9]+", str(col).lower())))
        and words & tokens
    ]
    if not statistics and not candidates:
        return None
    asked = " ".join(sorted(words & {t for c in candidates for t in re.findall(r"[a-z0-9]+", c.lower())})) or None
    return {"statistics": statistics, "measure": asked, "candidates": candidates}


def _unresolved_findings(unresolved: dict[str, Any]) -> list[str]:
    asked = unresolved["measure"] or " ".join(unresolved["statistics"]) or "a measure"
    if unresolved["candidates"]:
        named = ", ".join(unresolved["candidates"])
        return [f"The goal asks for {asked}, but no column is named that; it could mean "
                f"{named}. Name the column in the goal to get a figure."]
    return [f"The goal asks for {asked}, but names no numeric column to calculate it from. "
            "Name the column in the goal to get a figure."]


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


def _aggregation(goal: str, metric: str | None) -> str:
    """The statistic a grouped or monthly figure is: the one the goal asked for.

    "Average revenue by region" asks for a mean. The word "revenue" used to
    force a SUM whatever statistic the goal named, which printed a total of
    a 50,000-row sample as the answer to a question about the average. A
    total is only the default when no statistic is named and the measure is
    revenue or sales.
    """
    if not metric:
        return "sum"
    requested = _requested_numeric_metrics(goal)
    for name in ("mean", "median", "sum", "max", "min"):
        if name in requested:
            return name
    return "sum" if re.search(r"\b(revenue|sales)\b", goal, re.IGNORECASE) else "mean"


def analyze_goal(
    frame: pd.DataFrame, goal: str, profile: dict[str, Any], *, sampled: bool = False,
    filters: list[dict[str, Any]] | None = None, population: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Answer supported goals from actual rows, or disclose the gap.

    ``frame`` is the working rows, a sample when ``sampled``. ``population``
    is every source row when the caller holds them in memory: a goal's
    figures are then computed over all of them (up to
    :data:`FULL_PASS_MAX_ROWS`), because a grouped mean of a 50,000-row
    sample can differ in sign from the exact one. Otherwise each figure is
    labelled as estimated from the sample. ``result["basis"]`` records
    which, ``result["figures"]`` lists every reported number with how to
    recompute it, for the Validator to reconcile.
    """
    source_rows = int((profile.get("dataset") or {}).get("n_rows")
                      or len(population if population is not None else frame))
    full_pass = population is not None and sampled and len(population) <= FULL_PASS_MAX_ROWS
    rows = population if full_pass else frame
    basis_sampled = sampled and not full_pass
    result = _analyze_rows(rows, goal, profile, sampled=basis_sampled, filters=filters)
    result.setdefault("figures", [])
    result["basis"] = {"sampled": basis_sampled, "rows_used": int(len(rows)), "source_rows": source_rows,
                       "full_pass": full_pass}
    sheet, others = _sheet_basis(profile)
    if sheet is not None:
        result["basis"].update(sheet=sheet, other_sheets=others)
    of_sheet = f" of sheet '{sheet}'" if sheet is not None else ""
    not_read = (f"; {len(others)} other sheet{'s' if len(others) != 1 else ''} not analysed"
                + (f" ({_sheet_list(others)})" if others else "")) if sheet is not None and others else ""
    if result["figures"] and full_pass:
        result["findings"].append(f"Computed over all {len(rows):,} rows{of_sheet}, not the "
                                  f"{len(frame):,}-row working sample{not_read}.")
    elif result["figures"] and basis_sampled:
        result["findings"].append(f"Estimated from a {len(rows):,}-row sample of {source_rows:,} rows"
                                  f"{of_sheet}{not_read}.")
    elif result["figures"] and not_read:
        result["findings"].append(f"Computed over all {len(rows):,} rows{of_sheet}{not_read}.")
    _with_units(result, profile)
    return result


#: Statistics that carry a column's unit and scale. A count does not.
_SCALED_METRICS = ("sum", "mean", "median", "min", "max", "range")


def _profile_units(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{column: unit}`` for the columns the file states a unit for, as the
    Data Scout put them on the profile (avaloka/units.py)."""
    return {str(c["name"]): c["unit"] for c in profile.get("columns") or []
            if isinstance(c, dict) and isinstance(c.get("unit"), dict)}


def _with_units(result: dict[str, Any], profile: dict[str, Any]) -> None:
    """Give each figure over a column with a stated unit that unit, and, when
    the file states a scale, the figure at full scale beside it.

    The figure as written is unchanged; the full-scale one is that figure
    times the stated factor, labelled as such, with its own check so the
    Validator recomputes it. A column the file states nothing for is left
    exactly as it was: no unit is added that the file did not write.
    """
    from avaloka.units import unit_words

    units = _profile_units(profile)
    if not units:
        return
    metrics = result.get("metrics") or {}
    notes: list[str] = []
    scaled_figures: list[dict[str, Any]] = []

    def full(value: Any, scale: int) -> Any:
        return _number(value * scale) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def note(column: str, unit: dict[str, Any], said: str) -> str:
        source = unit.get("source") or {}
        where = (f"stated in the title on row {source['sheet_row']}" if source.get("sheet_row")
                 else "stated in its header")
        return f"{column} is in {unit_words(unit)} ({where}){said}."

    if result.get("kind") == "column_summary":
        for column, summary in metrics.items():
            unit = units.get(str(column))
            if not unit or not isinstance(summary, dict):
                continue
            summary["unit"] = unit_words(unit)
            scale = int(unit.get("scale") or 1)
            said = ""
            if scale > 1:
                full_unit = unit.get("currency")
                for metric in _SCALED_METRICS:
                    value = full(summary.get(metric), scale)
                    if value is None:
                        continue
                    summary[f"{metric}_at_full_scale"] = value
                    scaled_figures.append({
                        "label": f"{metric} {column} at full scale", "value": value,
                        "check": {"op": "agg", "agg": metric, "value": {"column": column},
                                  "where": [], "scale": scale}})
                if full_unit:
                    summary["unit_at_full_scale"] = full_unit
                words = {"sum": "total", "min": "lowest", "max": "highest"}
                shown = [f"{words.get(metric, metric)}="
                         f"{_display_number(summary[f'{metric}_at_full_scale'])}"
                         for metric in _SCALED_METRICS if f"{metric}_at_full_scale" in summary]
                if shown:
                    said = (f"; at full scale {', '.join(shown)}"
                            + (f" {full_unit}" if full_unit else ""))
            notes.append(note(str(column), unit, said))
    elif result.get("kind") == "group_comparison":
        column = metrics.get("value_column")
        unit = units.get(str(column)) if column else None
        if unit and metrics.get("aggregation") in _SCALED_METRICS:
            metrics["unit"] = unit_words(unit)
            scale = int(unit.get("scale") or 1)
            if scale > 1:
                for row in metrics.get("groups") or []:
                    value = full(row.get("value"), scale)
                    if value is not None:
                        row["value_at_full_scale"] = value
                if unit.get("currency"):
                    metrics["unit_at_full_scale"] = unit["currency"]
            notes.append(note(str(column), unit, "" if scale == 1 else
                              f"; each group's figure times {scale:,} is its value at full scale"))
    if notes:
        result.setdefault("findings", []).extend(notes)
    if scaled_figures:
        result.setdefault("figures", []).extend(scaled_figures)


def _sheet_basis(profile: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    """The workbook sheet the rows came from and the sheets left unread
    (avaloka.io.loader's ``other_sheets``), or ``(None, [])`` for a source
    that is not a workbook: "all N rows" of one sheet is not all the rows."""
    sheets = ((profile.get("dataset") or {}).get("structure") or {}).get("sheets") or {}
    if sheets.get("analysed") is None:
        return None, []
    return str(sheets["analysed"]), [dict(o) for o in sheets.get("others") or []]


def _sheet_list(others: list[dict[str, Any]]) -> str:
    return ", ".join(f"'{o.get('name')}' with {int(o.get('rows') or 0):,} rows" for o in others[:5]) + (
        f" and {len(others) - 5} more" if len(others) > 5 else "")


def _analyze_rows(
    frame: pd.DataFrame, goal: str, profile: dict[str, Any], *, sampled: bool = False,
    filters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The goal's answer over ``frame`` (see :func:`analyze_goal`).

    Results are descriptive, never causal. The scope is explicit because the
    working frame can be a sample for large input datasets, and because
    ``filters`` (the predicates the goal names, see :func:`goal_filters`)
    restrict the rows the answer is calculated over.
    """
    lower = goal.lower()
    columns = [str(col) for col in frame.columns]
    mentioned = _mentioned_columns(goal, columns)
    scope = (f"Sample of {len(frame):,} of {int((profile.get('dataset') or {}).get('n_rows') or len(frame)):,} rows"
             if sampled else f"All {len(frame):,} working rows")
    if filters:
        frame = apply_filters(frame, filters)
        scope += f"; {len(frame):,} where {describe_filters(filters)}"
    result: dict[str, Any] = {
        "goal": goal, "kind": "dataset_overview", "status": "limited",
        "scope": scope, "columns": mentioned, "metrics": {}, "findings": [],
    }
    if filters:
        result["filters"] = list(filters)

    # "What predicts default?" names an outcome, not a subset of the rows: it
    # is answered by what moves with the outcome, before any rule below can
    # read "default" as a value to keep or a column to summarise.
    targeted = _target_answer(frame, goal, profile, result, sampled=sampled)
    if targeted is not None:
        return targeted

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
        result["figures"] = [
            {"label": f"{col} missing", "value": counts[col], "check": {"op": "missing", "column": col}}
            for col in ordered
        ]
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
        # Every pair carries both coefficients, so the pairs reported are chosen
        # by the stronger of the two: a relationship one extreme row hides from
        # Pearson is still picked up by its rank correlation.
        candidates: list[dict[str, Any]] = []
        if numeric.shape[1] >= 2:
            corr = numeric.corr()
            rank_corr = numeric.corr(method="spearman")
            for i, left in enumerate(corr.columns):
                for right in corr.columns[i + 1:]:
                    value = corr.loc[left, right]
                    rho = rank_corr.loc[left, right]
                    if len(selected) >= 2:
                        include = left in selected and right in selected
                    elif selected:
                        include = left in selected or right in selected
                    else:
                        include = True
                    if not include:
                        continue
                    if pd.notna(value):
                        pairs.append({"left": str(left), "right": str(right), "r": _number(value)})
                    if pd.notna(value) or pd.notna(rho):
                        candidates.append({
                            "left": str(left), "right": str(right),
                            "r": _number(value) if pd.notna(value) else None,
                            "rho": _number(rho) if pd.notna(rho) else None,
                        })
        pairs.sort(key=lambda pair: (-abs(pair["r"]), pair["left"], pair["right"]))
        candidates.sort(key=lambda pair: (
            -max(abs(pair["r"] or 0.0), abs(pair["rho"] or 0.0)), pair["left"], pair["right"],
        ))
        rank_pairs = candidates[:5]
        result["figures"] = [
            {"label": f"r({pair['left']}, {pair['right']})", "value": pair["r"],
             "check": {"op": "pearson", "x": {"column": pair["left"]}, "y": {"column": pair["right"]}}}
            for pair in pairs[:5]
        ]
        result.update(
            kind="correlation", status="answered" if candidates else "limited",
            metrics={"pairs": pairs[:5], "rank_pairs": rank_pairs},
        )
        result["findings"] = [
            f"{pair['left']} and {pair['right']}: "
            + (f"Pearson r={pair['r']:.3f}" if pair["r"] is not None else "Pearson r undefined")
            + (f", Spearman rho={pair['rho']:.3f}." if pair["rho"] is not None else ".")
            for pair in rank_pairs
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
        numeric = [col for col in mentioned if col != date_col and _is_quantity(frame, col)]
        metric = numeric[0] if numeric else None
        if not metric:
            unresolved = _unresolved_measure(goal, frame, [date_col])
            if unresolved:
                result.update(kind="trend", status="limited", metrics={
                    "time_column": date_col, "value_column": None,
                    "aggregation": None, "unresolved_measure": unresolved,
                }, findings=_unresolved_findings(unresolved))
                return result
        series = (
            pd.to_numeric(frame[metric], errors="coerce")
            if metric
            else pd.Series(1, index=frame.index, dtype="float64")
        )
        table = pd.DataFrame({
            "period": dates.dt.tz_localize(None).dt.to_period("M"),
            "value": series,
        }).dropna()
        aggregation = _aggregation(goal, metric)
        monthly = getattr(table.groupby("period")["value"], aggregation)()
        values = {str(period): _number(value) for period, value in monthly.items()}
        result["figures"] = [
            {"label": f"{period} {aggregation} {metric or 'rows'}", "value": value,
             "check": {"op": "agg", "agg": aggregation if metric else "count",
                       "value": {"column": metric} if metric else None,
                       "where": [{"month": period, "column": date_col, "utc": True}]}}
            for period, value in values.items()
        ]
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
            measure = (f"{'sample ' if sampled and aggregation == 'sum' else ''}{aggregation} {metric}"
                       if metric else ("sample row count" if sampled else "row count"))
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
        metric = next((col for col in mentioned if col != group_col and _is_quantity(frame, col)), None)
        code = next((col for col in mentioned if col != group_col and is_categorical_code(col, frame[col])),
                    None)
        if not metric and code and _requested_numeric_metrics(goal) not in ([], ["count"]):
            result.update(kind="group_comparison", status="limited", metrics={
                "group_column": group_col, "value_column": None, "aggregation": None,
                "categorical_code": code,
            }, findings=[f"{code} holds category codes, not quantities, so its "
                         f"{' or '.join(_requested_numeric_metrics(goal))} would be arithmetic on "
                         "labels. Name a numeric measure, or ask how many rows fall in each code."])
            return result
        if not metric:
            unresolved = _unresolved_measure(goal, frame, [group_col])
            if unresolved:
                result.update(kind="group_comparison", status="limited", metrics={
                    "group_column": group_col, "value_column": None,
                    "aggregation": None, "unresolved_measure": unresolved,
                }, findings=_unresolved_findings(unresolved))
                return result
        grouped = frame.groupby(group_col, dropna=False)
        sizes = grouped.size()

        # A true missing value is its own bucket; say so instead of "nan". If a
        # real category already reads "(missing)", the bucket gets a longer
        # label so the two are never merged.
        real_labels = {str(key) for key in sizes.index if not pd.isna(key)}
        missing_label = "(missing)" if "(missing)" not in real_labels else "(missing: no value)"

        def _label(key: Any) -> str:
            return missing_label if pd.isna(key) else str(key)

        spread: dict[str, tuple[float, float, int]] = {}
        if metric:
            aggregation = _aggregation(goal, metric)
            numbers = pd.to_numeric(frame[metric], errors="coerce")
            by_group = numbers.groupby(frame[group_col], dropna=False)
            values = getattr(by_group, aggregation)().sort_values(ascending=False).head(10)
            if sampled and aggregation == "mean":
                # A mean of a sample carries its interval: a rare group's mean
                # can change sign from one sample to the next.
                stats = by_group.agg(["std", "count"])
                for key, (std, n) in stats.loc[stats.index.isin(values.index)].iterrows():
                    n = int(n)
                    if n > 1 and pd.notna(std):
                        half = 1.96 * float(std) / math.sqrt(n)
                        spread[_label(key)] = (float(values[key]) - half, float(values[key]) + half, n)
        else:
            aggregation = "count"
            values = grouped.size().sort_values(ascending=False).head(10)
        rows = [
            {"group": _label(key), "value": _number(value)}
            for key, value in values.items() if pd.notna(value)
        ]
        for row in rows:
            if row["group"] in spread:
                low, high, n = spread[row["group"]]
                row.update(n=n, ci95=[_number(low), _number(high)])
        group_sizes = {_label(key): int(sizes[key]) for key, value in values.items() if pd.notna(value)}
        result.update(
            kind="group_comparison", status="answered" if rows else "limited",
            metrics={
                "group_column": group_col,
                "value_column": metric,
                "aggregation": aggregation,
                "groups": rows,
                "group_sizes": group_sizes,
            },
        )
        result["figures"] = [
            {"label": f"{group_col}={row['group']} {aggregation} {metric or 'rows'}", "value": row["value"],
             "check": {"op": "agg", "agg": aggregation, "value": {"column": metric} if metric else None,
                       "where": [{"column": group_col, "equals_text": row["group"]}]}}
            for row in rows
        ]
        label = f"{'sample ' if sampled and aggregation in {'sum', 'count'} else ''}{aggregation}"
        findings = []
        for row in rows[:5]:
            line = (f"{group_col}={row['group']}: {label} {metric or 'rows'}="
                    f"{_display_number(round(row['value'], 4))} (n={group_sizes[row['group']]})")
            if "ci95" in row:
                low, high = row["ci95"]
                line += f" (95% CI {low:.4g} to {high:.4g}"
                line += "; the sign is not established)" if low < 0 < high else ")"
            findings.append(line + ".")
        if sampled and aggregation in {"sum", "count"} and rows:
            findings.append(f"These are {aggregation}s over the sample only, not dataset totals.")
        result["findings"] = findings or ["No groups with valid values were available to compare."]
        if missing_label in group_sizes and missing_label not in real_labels:
            result["findings"].append(
                f"{group_sizes[missing_label]} rows have no {group_col} value; they are shown "
                f"as '{missing_label}' and are not a real category."
            )
        # The loader keeps null-like words in text columns as labels (so "NA" can
        # be Namibia). When one ranks here, say so: it may be missing data.
        for label in group_sizes:
            if label in NULL_WORDS:
                result["findings"].append(
                    f"'{label}' ({group_sizes[label]} rows) is a null-like word that is counted "
                    f"as its own group in {group_col}. If it means missing data, exclude it."
                )
        return result

    if mentioned:
        summaries: dict[str, Any] = {}
        findings: list[str] = []
        requested_metrics = _requested_numeric_metrics(goal)
        figures: list[dict[str, Any]] = []
        for col in mentioned[:10]:
            series = frame[col]
            if _is_quantity(frame, col):
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
                            if metric != "range":
                                figures.append({"label": f"{metric} {col}", "value": summary[metric],
                                                "check": {"op": "agg", "agg": metric,
                                                          "value": {"column": col}, "where": []}})
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
                    "sum": "sample total" if sampled else "total",
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
                if sampled and "sum" in metrics:
                    findings.append(f"{col}: the total is over the sample only, not the dataset total.")
            else:
                counts = series.astype("string").value_counts().head(3)
                summaries[col] = {
                    "non_null": int(series.notna().sum()),
                    "top_values": {str(key): int(value) for key, value in counts.items()},
                }
                if is_categorical_code(col, series):
                    summaries[col]["categorical_code"] = True
                findings.append(
                    f"{col}: {int(series.notna().sum()):,} non-null rows"
                    + (" (category codes, so not averaged)" if is_categorical_code(col, series) else "")
                    + "; "
                    + (
                        "most common: " + ", ".join(
                            f"{key} ({value})" for key, value in counts.items()
                        ) if len(counts) else "no values to summarise."
                    )
                )
        result.update(
            kind="column_summary", status="answered", metrics=summaries,
            findings=findings, figures=figures,
        )
        return result

    result["findings"] = [
        "This goal does not map to a supported goal-specific calculation. "
        "The remaining report is a general dataset profile, not an answer to the question."
    ]
    return result


# --- the predicates a goal names -------------------------------------------
# A goal that names a subset of the rows ("west sales", "tenure above 12",
# "customers where contract is m2m") is a scan with a predicate, and the
# workload optimizer plans that scan. The grammar is deliberately small and
# column-grounded, like the rest of this module: a predicate is only derived
# when the goal names a column the profile knows, or a value the profile saw
# in exactly one low-cardinality column. Nothing is guessed from a bare word.
_COMPARATORS: tuple[tuple[str, str], ...] = (
    ("gte", r">=|at least|no less than|not less than"),
    ("lte", r"<=|at most|no more than|not more than"),
    ("gt", r">|greater than|more than|above|over|higher than|exceeding|exceeds|later than|after"),
    ("lt", r"<|less than|fewer than|below|under|lower than|earlier than|before"),
    ("eq", r"==|=|equal to|equals|is|of"),
)
_COMPARATOR_RE = "|".join(f"(?P<{name}>{pattern})" for name, pattern in _COMPARATORS)
_VALUE_RE = r"\"(?P<dq>[^\"]+)\"|'(?P<sq>[^']+)'|(?P<bare>[\w.+-]+)"
_NULL_WORDS = r"(?:is\s+)?(?:missing|null|empty|blank|unknown)"
#: Values a profile can hold that are English, not data, when they appear bare.
_NOT_A_VALUE = _NOT_A_MEASURE | frozenset({
    "yes", "no", "true", "false", "none", "null", "nan", "other", "unknown",
    "with", "without", "from", "into", "than", "that", "this", "these", "those",
    "where", "when", "who", "why", "are", "was", "were", "has", "have", "had",
    "not", "but", "its", "our", "your", "their", "does", "did", "can", "will",
    "rows", "row", "data", "dataset", "column", "columns", "table", "goal",
})
#: Bare values are only matched in columns this small; a word that is a value
#: of a free-text or identifier column is not a filter anyone meant.
_BARE_VALUE_MAX_UNIQUE = 50


def _column_catalogue(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(col.get("name")): col for col in profile.get("columns", []) if col.get("name")}


def _typed(value: str, col: dict[str, Any]) -> Any:
    """The value as the column holds it: a number for a numeric column, the
    profile's spelling for a value the profile saw, the text otherwise."""
    if col.get("role") == "numeric":
        try:
            number = float(value)
        except ValueError:
            return None
        return int(number) if number.is_integer() else number
    lowered = value.lower()
    for top in col.get("top_values", []):
        if str(top.get("value")).lower() == lowered:
            return top["value"]
    return value


def _comparator(match: "re.Match[str]") -> str | None:
    return next((name for name, _ in _COMPARATORS if match.group(name)), None)


def _matched_value(match: "re.Match[str]") -> str:
    return next(v for v in (match.group("dq"), match.group("sq"), match.group("bare")) if v)


def goal_filters(goal: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
    """The row predicates a goal names, as ``{column, operator, value}``.

    Three forms are read, all grounded in the profile:

    * ``<column> <comparator> <value>``: ``tenure above 12``,
      ``contract is m2m``, ``monthly >= 100``, ``region = "West"``. A
      comparison other than equality needs a numeric column and a number.
    * ``<column> between <a> and <b>``: two predicates, ``gte`` and ``lte``.
    * ``<column> is missing`` (null, empty, blank): ``is_null``.
    * A bare value the profile saw in exactly one low-cardinality column:
      ``west sales`` over a ``region`` column whose values include ``west``.
      Several bare values of one column become one ``in`` predicate
      ("compare east and west" keeps both, it does not filter to none).

    The result is deterministic and never names a column the profile lacks.
    An empty list means the goal asks about every row.
    """
    catalogue = _column_catalogue(profile)
    if not catalogue:
        return []
    names = sorted(catalogue, key=len, reverse=True)
    column_re = "|".join(re.escape(name) for name in names)
    text = " ".join(goal.split())
    lower = text.lower()
    found: list[dict[str, Any]] = []
    bound: set[str] = set()                   # columns an explicit predicate already names
    consumed: list[tuple[int, int]] = []

    def take(span: tuple[int, int]) -> None:
        consumed.append(span)

    for match in re.finditer(
            rf"(?<!\w)(?P<col>{column_re})(?!\w)\s+between\s+(?P<a>[\w.+-]+)\s+and\s+(?P<b>[\w.+-]+)",
            text, re.IGNORECASE):
        col = catalogue[_canonical(match.group("col"), catalogue)]
        low, high = _typed(match.group("a"), col), _typed(match.group("b"), col)
        if col.get("role") == "numeric" and low is not None and high is not None:
            found += [{"column": col["name"], "operator": "gte", "value": min(low, high)},
                      {"column": col["name"], "operator": "lte", "value": max(low, high)}]
            bound.add(col["name"]); take(match.span())

    for match in re.finditer(rf"(?<!\w)(?P<col>{column_re})(?!\w)\s+{_NULL_WORDS}(?!\w)",
                             text, re.IGNORECASE):
        if any(s <= match.start() < e for s, e in consumed):
            continue
        col = catalogue[_canonical(match.group("col"), catalogue)]
        found.append({"column": col["name"], "operator": "is_null", "value": None})
        bound.add(col["name"]); take(match.span())

    for match in re.finditer(
            rf"(?<!\w)(?P<col>{column_re})(?!\w)\s+(?:{_COMPARATOR_RE})\s+(?:{_VALUE_RE})(?!\w)",
            text, re.IGNORECASE):
        if any(s <= match.start() < e for s, e in consumed):
            continue
        col = catalogue[_canonical(match.group("col"), catalogue)]
        operator = _comparator(match)
        raw = _matched_value(match)
        if col["name"] in bound or operator is None:
            continue
        if raw.lower() in _NOT_A_VALUE and operator == "eq":
            continue                                   # "tenure is the ..." names nothing
        value = _typed(raw, col)
        if value is None:
            continue                                   # a comparison with a word, not a number
        if operator != "eq" and col.get("role") != "numeric":
            continue
        found.append({"column": col["name"], "operator": operator, "value": value})
        bound.add(col["name"]); take(match.span())

    # Bare values: a word of the goal that is a value of exactly one small
    # categorical column, and not a word already read as part of a predicate.
    by_value: dict[str, list[tuple[str, Any]]] = {}
    for name, col in catalogue.items():
        if name in bound or col.get("role") not in {"categorical", "boolean"}:
            continue
        if int(col.get("n_unique") or 0) > _BARE_VALUE_MAX_UNIQUE:
            continue
        for top in col.get("top_values", []):
            key = str(top.get("value")).lower()
            if len(key) >= 3 and key.isalpha() and key not in _NOT_A_VALUE and key not in catalogue \
                    and key not in {n.lower() for n in catalogue}:
                by_value.setdefault(key, []).append((name, top["value"]))
    hits: dict[str, list[Any]] = {}
    for match in re.finditer(r"[a-z0-9]+", lower):
        if any(s <= match.start() < e for s, e in consumed):
            continue
        owners = by_value.get(match.group(0))
        if owners and len({owner for owner, _ in owners}) == 1:
            owner, value = owners[0]
            if value not in hits.setdefault(owner, []):
                hits[owner].append(value)
    for owner, values in hits.items():
        if len(values) == 1:
            found.append({"column": owner, "operator": "eq", "value": values[0]})
        else:
            found.append({"column": owner, "operator": "in", "value": values})
    # The outcome a predicts-goal names is what the analysis measures, never a
    # subset to keep: "what predicts default?" filtered to loan_status=Default
    # and left nothing to compare it with.
    target = driver_target(goal, profile)
    if target and target["column"]:
        found = [p for p in found if p["column"] != target["column"]]
    return found


def _canonical(name: str, catalogue: dict[str, dict[str, Any]]) -> str:
    return next(n for n in catalogue if n.lower() == name.lower())


def _predicate_mask(frame: pd.DataFrame, predicate: dict[str, Any]) -> pd.Series | None:
    """The rows ``predicate`` keeps, or ``None`` when it names no column of ``frame``."""
    column, operator = predicate.get("column"), str(predicate.get("operator", "eq")).lower()
    if column not in frame.columns:
        return None
    series, value = frame[column], predicate.get("value")
    if operator in {"is_null", "null"}:
        return series.isna()
    if operator in {"lt", "lte", "gt", "gte", "<", "<=", ">", ">="}:
        numbers = pd.to_numeric(series, errors="coerce")
        op = {"<": "lt", "lt": "lt", "<=": "le", "lte": "le", ">": "gt", "gt": "gt",
              ">=": "ge", "gte": "ge"}[operator]
        return getattr(numbers, op)(value).fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        wanted = value if operator == "in" else [value]
        try:
            numbers = [float(v) for v in wanted]
        except (TypeError, ValueError):
            return pd.Series(False, index=frame.index)
        return pd.to_numeric(series, errors="coerce").isin(numbers).fillna(False)
    text = series.astype("string").str.lower()
    wanted = value if operator == "in" else [value]
    return text.isin([str(v).lower() for v in wanted]).fillna(False)


def apply_filters(frame: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    """``frame`` restricted to the rows every predicate in ``filters`` keeps.

    Equality on text is case-insensitive, as :func:`goal_filters` reads values
    from a goal; a predicate on a column the frame lacks keeps every row.
    """
    if not filters:
        return frame
    mask = pd.Series(True, index=frame.index)
    for predicate in filters:
        keep = _predicate_mask(frame, predicate)
        if keep is not None:
            mask &= keep.astype(bool)
    return frame[mask]


def describe_filters(filters: list[dict[str, Any]]) -> str:
    """``region = west and tenure > 12``: the predicates as one readable clause."""
    symbols = {"eq": "=", "in": "in", "is_null": "is missing", "gt": ">", "gte": ">=",
               "lt": "<", "lte": "<="}
    parts = []
    for predicate in filters:
        operator = str(predicate.get("operator", "eq")).lower()
        symbol = symbols.get(operator, operator)
        if operator == "is_null":
            parts.append(f"{predicate['column']} is missing")
        elif operator == "in":
            parts.append(f"{predicate['column']} in ({', '.join(map(str, predicate['value']))})")
        else:
            parts.append(f"{predicate['column']} {symbol} {predicate['value']}")
    return " and ".join(parts)


# --- goals that name an outcome ---------------------------------------------
# "What predicts default?", "what drives the fare?", "which patients are
# readmitted?" name an outcome. The outcome is the target of a driver
# analysis: every other column is measured against it. It is never a filter
# -- keeping only the rows where loan_status is Default leaves nothing that
# differs from it, which is how a predicts-goal once analysed two rows.
_DRIVER_VERBS = (r"predicts?|predicting|drives?|driving|influences?|determines?|explains?|"
                 r"affects?|impacts?|distinguish(?:es)?|separates?|causes?|leads?\s+to")
_DRIVER_PATTERNS = (
    re.compile(rf"\b(?:what|which|who)\b(?P<subject>(?:\s+[\w-]+){{0,4}}?)\s+(?:{_DRIVER_VERBS})\s+"
               r"(?P<phrase>[^?.!;]+)", re.IGNORECASE),
    re.compile(r"\b(?:drivers?|predictors?|determinants?|causes?|factors?)\s+(?:of|for|behind|affecting)\s+"
               r"(?P<phrase>[^?.!;]+)", re.IGNORECASE),
    re.compile(r"\bwhich\s+(?P<subject>[\w-]+)\s+(?:are|were|get|got|become|became)\s+"
               r"(?P<phrase>[^?.!;]+)", re.IGNORECASE),
)
#: The outcome phrase ends where a qualifier starts: "default for west loans".
_PHRASE_END = re.compile(r"\b(?:for|in|among|where|when|by|across|within|during|from|at|with|"
                         r"between|over|per|most|more|best)\b", re.IGNORECASE)
#: Words naming the kind of thing a goal asks about, not a column.
_GENERIC_SUBJECTS = frozenset({
    "a", "an", "the", "kinds", "kind", "types", "type", "factors", "factor", "features", "feature",
    "variables", "variable", "columns", "column", "things", "attributes", "fields", "signals",
    "of", "and", "or", "most", "best", "top", "main", "key", "patients", "customers", "users",
    "people", "rows", "records", "items", "it", "is", "do", "does",
})
_ITEM_WORDS = frozenset({"product", "products", "item", "items", "article", "articles", "sku", "skus"})
_ITEM_TOKENS = ("description", "product", "item", "sku", "article", "title", "stockcode")
_QUANTITY_TOKENS = frozenset({"quantity", "qty", "units"})
_PRICE_TOKENS = frozenset({"price"})
_MONEY_WORDS = frozenset({"revenue", "sales", "turnover", "income"})
_SELL_RE = re.compile(r"\b(sells?|sold|selling|best[- ]?sell\w*)\b", re.IGNORECASE)
_PEAK_RE = re.compile(r"\bwhen\b.*\b(highest|peaks?|peaking|most|busiest|maximum|max|lowest|least)\b",
                      re.IGNORECASE)
#: Associations at or above this strength usually mean the column restates
#: the outcome (a total that includes it, a status recorded after it).
_LEAKAGE_STRENGTH = 0.95


def _singular(word: str) -> str:
    word = word.lower()
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def word_stems(word: str) -> frozenset[str]:
    """The forms a goal word may share with a column word: the word itself and
    its regular plural or past tense taken off ("returns", "returned" and
    "return" share ``return``; "channels" shares ``channel``). This is the one
    stem rule: the CLI's driver_target, the app's step narration and the DB
    pushdown parser (#559's "``<verb>ed`` is asked about by its verb, the verb
    four letters or more") all match through it. A stem is compared whole:
    "turn" shares nothing with "returned", and "used" is not "us"."""
    w = word.lower()
    forms = {w}
    if w.endswith("ies") and len(w) > 4:
        forms.add(w[:-3] + "y")
    elif w.endswith(("sses", "xes", "ches", "shes", "zes")) and len(w) > 4:
        forms.add(w[:-2])
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        forms.add(w[:-1])
    if w.endswith("ed") and len(w) - 2 >= 4:
        forms.add(w[:-2])                # "returned" -> return, "churned" -> churn
    return frozenset(f for f in forms if len(f) >= 3)


def column_mentions(text: str, columns: Any) -> list[tuple[int, int, str]]:
    """``(start, end, column)`` for each column ``text`` names, in text order.

    A column is named by its whole name (any case, ``_`` or space between its
    words), or by its words in order, each matched whole through
    :func:`word_stems` ("return rate" names ``returned``, "unit prices" names
    ``unit_price``). The whole name wins: where some column is named exactly,
    no inflected match is taken over the same words, and a longer inflected
    match beats a shorter one inside it. A span two columns could both be
    read from is left unnamed, never guessed.
    """
    text = text or ""
    cols = [str(c) for c in columns or [] if str(c).strip() and re.search(r"[A-Za-z]", str(c))]
    exact: list[tuple[int, int, str]] = []
    for col in cols:
        core = r"[_\s]+".join(re.escape(p) for p in re.split(r"[_\s]+", col) if p)
        for m in re.finditer(rf"(?<![\w]){core}(?![\w])", text, re.IGNORECASE):
            exact.append((m.start(), m.end(), col))
    tokens = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[A-Za-z]+|\d+", text)]
    inflected: dict[tuple[int, int], set[str]] = {}
    for col in cols:
        words = _ordered_tokens(col)   # unit_price / UnitPrice -> [unit, price]
        if not words:
            continue
        for i in range(len(tokens) - len(words) + 1):
            window = tokens[i:i + len(words)]
            if all(word_stems(tok) & word_stems(want) for (_s, _e, tok), want in zip(window, words)):
                inflected.setdefault((window[0][0], window[-1][1]), set()).add(col)
    taken = [(s, e) for s, e, _c in exact]
    found = list(exact)
    # Longest first, so "unit prices" is ``unit_price``, not ``units``.
    for (start, end), owners in sorted(inflected.items(), key=lambda kv: kv[0][0] - kv[0][1]):
        if len(owners) != 1 or any(s < end and start < e for s, e in taken):
            continue
        taken.append((start, end))
        found.append((start, end, next(iter(owners))))
    return sorted(set(found))


def canonical_column_words(text: str, columns: Any) -> str:
    """``text`` with each inflected column mention replaced by the column's
    name: "return rate by channels" -> "returned rate by channel"."""
    out, last = [], 0
    for start, end, col in column_mentions(text, columns):
        if start < last:
            continue
        out.append(text[last:start])
        out.append(col if text[start:end].lower() != col.lower() else text[start:end])
        last = end
    out.append(text[last:])
    return "".join(out)


def _catalogue_for(profile: dict[str, Any], frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """The profile's columns, completed from ``frame`` for any it lacks."""
    catalogue = {name: col for name, col in _column_catalogue(profile).items() if name in frame.columns}
    for name in map(str, frame.columns):
        if name in catalogue:
            continue
        series = frame[name]
        n_unique = int(series.nunique(dropna=True))
        if pd.api.types.is_bool_dtype(series):
            role = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            role = "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            role = "numeric"
        else:
            role = "categorical"
        col = {"name": name, "role": role, "n_unique": n_unique}
        if role in {"categorical", "boolean"}:
            counts = series.dropna().astype(str).value_counts().head(10)
            col["top_values"] = [{"value": k, "count": int(v)} for k, v in counts.items()]
        catalogue[name] = col
    return catalogue


def _resolve_column_word(word: str, catalogue: dict[str, dict[str, Any]],
                         allowed: set[str] | None = None) -> str | None:
    """The one column a goal word names: whole name, then a name word, then a
    shared stem of six letters ("readmission" names ``readmitted``)."""
    word = word.lower()
    names = [n for n in catalogue if allowed is None or n in allowed]
    stem = _singular(word)
    tiers = (
        [n for n in names if n.lower() == word or _singular(n) == stem],
        [n for n in names if stem in {_singular(t) for t in _name_tokens(n)}],
        # "returns" / "return" name ``returned``: a shared inflection stem, whole.
        [n for n in names if any(word_stems(word) & word_stems(t) for t in _name_tokens(n) | {n.lower()})],
        [n for n in names if len(word) >= 6 and any(
            len(t) >= 6 and t[:6] == word[:6] for t in _name_tokens(n) | {n.lower()})],
    )
    for tier in tiers:
        if len(tier) == 1:
            return tier[0]
        if tier:
            return None                      # ambiguous: name it rather than guess
    return None


def _resolve_value_word(word: str, catalogue: dict[str, dict[str, Any]]) -> tuple[str, Any] | None:
    word, owners = word.lower(), {}
    for name, col in catalogue.items():
        if col.get("role") not in {"categorical", "boolean"} or int(col.get("n_unique") or 0) > _BARE_VALUE_MAX_UNIQUE:
            continue
        for top in col.get("top_values", []):
            if str(top.get("value")).lower() in {word, _singular(word)}:
                owners.setdefault(name, top["value"])
                break
    if len(owners) == 1:
        return next(iter(owners.items()))
    return None


def _column_values(name: str, col: dict[str, Any], frame: pd.DataFrame | None) -> list[str]:
    """A small column's distinct values: the profile's, and the frame's when held."""
    values = [str(t.get("value")) for t in col.get("top_values", []) if t.get("value") is not None]
    if frame is not None and name in frame.columns:
        values += [str(v) for v in frame[name].dropna().astype(str).unique()[:_BARE_VALUE_MAX_UNIQUE + 1]]
    return list(dict.fromkeys(v.strip() for v in values if v.strip()))


def _longest_value(text: str, catalogue: dict[str, dict[str, Any]], frame: pd.DataFrame | None, *,
                   columns: set[str] | None = None, anchored: bool = False, multiword: bool = True,
                   limit: int | None = None) -> tuple[str, Any] | None:
    """(column, value) of the longest distinct value in ``text``, whole words,
    case-insensitive: at its start when ``anchored``, starting at or before
    ``limit`` otherwise; with ``multiword`` only values of several words.
    ``None`` when none appears, or when two columns hold the longest one."""
    hits: list[tuple[int, str, str]] = []
    for name, col in catalogue.items():
        if columns is not None and name not in columns:
            continue
        if col.get("role") not in {"categorical", "boolean"} or int(col.get("n_unique") or 0) > _BARE_VALUE_MAX_UNIQUE:
            continue
        for value in _column_values(name, col, frame):
            if multiword and not re.search(r"\s", value):
                continue
            pattern = rf"(?<!\w){re.escape(value)}(?!\w)"
            found = re.match(pattern, text, re.IGNORECASE) if anchored else re.search(pattern, text, re.IGNORECASE)
            if found and (limit is None or found.start() <= limit):
                hits.append((len(value), name, value))
    if not hits:
        return None
    longest = max(h[0] for h in hits)
    owners = {name: value for size, name, value in hits if size == longest}
    return next(iter(owners.items())) if len(owners) == 1 else None


def driver_target(goal: str, profile: dict[str, Any], frame: pd.DataFrame | None = None) -> dict[str, Any] | None:
    """The outcome a "what predicts / drives / distinguishes X" goal names.

    Returns ``{"phrase", "subject", "column", "value"}``: ``column`` (and
    ``value`` when X is a value of a column, as "default" is of
    ``loan_status``) is resolved against the profile and is ``None`` when X
    names nothing in the data. ``None`` when the goal is not such a question.
    """
    text = " ".join(goal.split())
    found = next(((i, m) for i, p in enumerate(_DRIVER_PATTERNS) if (m := p.search(text))), None)
    if found is None:
        return None
    index, match = found
    raw = " ".join(match.group("phrase").split()).strip(" ,")
    phrase = _PHRASE_END.split(match.group("phrase"))[0].strip(" ,")
    subject = (match.groupdict().get("subject") or "").strip()
    catalogue = _catalogue_for(profile, frame) if frame is not None else _column_catalogue(profile)
    # "Which patients are readmitted" is only an outcome question when the
    # outcome resolves; "which columns are missing" is not one at all.
    target: dict[str, Any] = {"phrase": phrase, "subject": subject, "column": None, "value": None,
                              "explicit_verb": index < 2}
    if not catalogue or not raw:
        return target
    column_re = "|".join(re.escape(n) for n in sorted(catalogue, key=len, reverse=True))
    explicit = re.search(rf"(?<!\w)(?P<col>{column_re})(?!\w)\s*(?:==|=|\bis\b|\bequals\b)\s*(?:{_VALUE_RE})",
                         raw, re.IGNORECASE)
    # Only inside the outcome phrase: "default for grade = B" filters on grade.
    if explicit and explicit.start() >= len(" ".join(phrase.split())):
        explicit = None
    if explicit:
        col = catalogue[_canonical(explicit.group("col"), catalogue)]
        value = _typed(_matched_value(explicit), col)
        if explicit.group("bare"):
            # "loan_status = Charged Off": a bare value runs to the longest
            # value of that column the text starts with, not its first word.
            longer = _longest_value(raw[explicit.start("bare"):], catalogue, frame,
                                    columns={col["name"]}, anchored=True, multiword=False)
            if longer is not None:
                value = longer[1]
        if value is not None:
            target.update(column=col["name"], value=value)
            return target
    # "What predicts Charged Off loans": a value of several words is matched
    # whole, the longest one any column holds, before the goal is read word by
    # word (where "Charged" alone is no value). It may begin where the phrase
    # was cut ("In Grace Period" starts with "in").
    owned = _longest_value(raw, catalogue, frame, limit=len(phrase))
    if owned is not None:
        target.update(column=owned[0], value=owned[1])
        return target
    for word in re.findall(r"[\w-]+", phrase):
        if len(word) < 3 or word.lower() in _NOT_A_VALUE or word.lower() in _GENERIC_SUBJECTS:
            continue
        column = _resolve_column_word(word, catalogue)
        if column is not None:
            col = catalogue[column]
            free_text = col.get("role") in {"text", "id"} or (
                col.get("role") == "categorical" and int(col.get("n_unique") or 0) > _BARE_VALUE_MAX_UNIQUE)
            if not free_text:
                target["column"] = column
                return target
        owned = _resolve_value_word(word, catalogue)
        if owned is not None:
            target.update(column=owned[0], value=owned[1])
            return target
    return target


def _values(frame: pd.DataFrame, expr: dict[str, Any]) -> pd.Series:
    """A measure as numbers: a column, an outcome indicator, or a product."""
    if "column" in expr:
        return pd.to_numeric(frame[expr["column"]], errors="coerce")
    if "indicator" in expr:
        spec = expr["indicator"]
        mask = _predicate_mask(frame, {"column": spec["column"], "operator": "eq", "value": spec["equals"]})
        return mask.astype("float64").where(frame[spec["column"]].notna())
    left, right = expr["product"]
    return pd.to_numeric(frame[left], errors="coerce") * pd.to_numeric(frame[right], errors="coerce")


def _labels(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<missing>")


def _eta(values: pd.Series, groups: pd.Series) -> float | None:
    """The correlation ratio: how much of ``values``' variance the groups explain."""
    keep = values.notna()
    values, groups = values[keep], groups[keep]
    if len(values) < MIN_OUTCOME_ROWS:
        return None
    total = float(((values - values.mean()) ** 2).sum())
    if total <= 0:
        return None
    stats = values.groupby(groups).agg(["mean", "count"])
    between = float((stats["count"] * (stats["mean"] - values.mean()) ** 2).sum())
    return math.sqrt(max(0.0, min(1.0, between / total)))


def _cramers_v(left: pd.Series, right: pd.Series) -> float | None:
    table = pd.crosstab(left, right)
    n = float(table.to_numpy().sum())
    k = min(table.shape) - 1
    if n < MIN_OUTCOME_ROWS or k < 1:
        return None
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float(((table.to_numpy() - expected) ** 2 / np.where(expected > 0, expected, 1)).sum())
    return math.sqrt(chi2 / (n * k))


_ID_TOKENS = frozenset({"id", "ids", "key", "nbr", "uuid", "guid", "index", "idx", "no", "number", "row",
                        "rowid", "rownum", "unnamed"})

#: THE IDENTIFIER RULE. A column is a row identifier, never a predictor, when
#: (a) a token of its name says so (``id``, ``patient_id``, ``patientId``,
#:     ``uuid``, ``key``, ``row``/``index``/``idx``, pandas' ``Unnamed: 0``;
#:     see _ID_TOKENS) and over half its values are distinct; or
#: (b) it is non-numeric and near-unique (98% of 50 or more values distinct)
#:     and does not hold dates, or the profile's role for it is ``id``; or
#: (c) it is a whole-number key: 98% of 50 or more values distinct, and either
#:     strictly monotonic in row order (1, 2, 3, ...; an auto-increment) or
#:     dense, its span (max - min + 1) at most IDENTIFIER_DENSE_SPAN times
#:     its distinct count.
#: A near-unique whole-number measure is not a key: an income rounded to the
#: dollar or a country's population is spread far wider than its row count
#: and in no particular row order, and is what a driver analysis is for.
IDENTIFIER_NEAR_UNIQUE = 0.98
IDENTIFIER_MIN_ROWS = 50
IDENTIFIER_DENSE_SPAN = 2.0
IDENTIFIER_RULE = ("a row identifier is named id, *_id, *Id, uuid, key, row or index with over half its values "
                   "distinct, or is a near-unique text column that does not hold dates, or a near-unique "
                   "whole-number column that is monotonic in row order or packed into a range under twice "
                   "its row count")


def _whole_numbers(values: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(values) or not pd.api.types.is_numeric_dtype(values):
        return False
    if pd.api.types.is_integer_dtype(values):
        return True
    v = values.to_numpy(dtype="float64")
    return bool(len(v)) and bool(np.isfinite(v).all()) and bool((v == np.floor(v)).all())


def _integer_key(n: int, distinct: int, span: float | None, monotonic: bool) -> bool:
    """Rule (c): a near-unique whole-number column that is monotonic or dense."""
    if n < IDENTIFIER_MIN_ROWS or distinct < IDENTIFIER_NEAR_UNIQUE * n:
        return False
    return monotonic or (span is not None and span <= IDENTIFIER_DENSE_SPAN * distinct)


def _key_facts(non_null: pd.Series) -> dict[str, Any] | None:
    """What rule (c) reads from one frame's values: None for a column that
    is not whole numbers."""
    if not _whole_numbers(non_null):
        return None
    v = non_null.to_numpy(dtype="float64")
    diffs = np.diff(v)
    return {"integer": True, "monotonic": bool(len(v) >= 2 and ((diffs > 0).all() or (diffs < 0).all()))}


def is_identifier_column(name: str, series: pd.Series, role: str | None = None) -> bool:
    """:data:`IDENTIFIER_RULE` over one frame's column."""
    non_null = series.dropna()
    n_unique = non_null.nunique()
    if _name_tokens(name) & _ID_TOKENS and n_unique > 0.5 * max(1, len(non_null)):
        return True
    if pd.api.types.is_numeric_dtype(series):
        facts = _key_facts(non_null)
        if facts is None or not len(non_null):
            return False
        span = float(non_null.max()) - float(non_null.min()) + 1
        return _integer_key(len(non_null), n_unique, span, facts["monotonic"])
    if role == "id":
        return True
    if len(non_null) < IDENTIFIER_MIN_ROWS or n_unique < IDENTIFIER_NEAR_UNIQUE * len(non_null):
        return False
    # A near-unique date is when each row happened, not a label for it: a
    # visit_date with 58 distinct days in 60 rows was called "a row
    # identifier ... its values label rows" (rc17 nullxlsx).
    return not _is_dates(non_null)


def _is_dates(non_null: pd.Series) -> bool:
    """Datetime values, or text that reads as dates (avaloka.stats' test)."""
    if pd.api.types.is_datetime64_any_dtype(non_null):
        return True
    from avaloka.stats import _looks_like_dates

    return _looks_like_dates(non_null)


def _is_identifier(name: str, series: pd.Series, role: str | None) -> bool:
    """A row label (:data:`IDENTIFIER_RULE`)."""
    return is_identifier_column(name, series, role)


def _describe_target(target: dict[str, Any]) -> str:
    if target.get("value") is not None:
        return f"{target['column']} = {target['value']}"
    return str(target.get("column") or target.get("label"))


def _target_answer(frame: pd.DataFrame, goal: str, profile: dict[str, Any], result: dict[str, Any],
                   *, sampled: bool) -> dict[str, Any] | None:
    """Answer a goal that names an outcome, or ``None`` when it names none."""
    catalogue = _catalogue_for(profile, frame)
    if _SELL_RE.search(goal) and re.search(r"\b(most|best|top|highest)\b", goal, re.IGNORECASE):
        item = _item_column(frame, catalogue)
        if item is None:
            result.update(kind="contribution", findings=[
                "The goal asks what sells most, but no column names the product or item sold."])
            return result
        quantity = _token_column(frame, _QUANTITY_TOKENS)
        measure = ({"column": quantity}, f"units sold ({quantity})") if quantity else (None, "rows")
        return _contribution(frame, result, [item], *measure, sampled=sampled)
    if _PEAK_RE.search(goal):
        return _peak(frame, goal, catalogue, result, sampled=sampled)
    target = driver_target(goal, profile, frame)
    if target is None or (target["column"] is None and not target["explicit_verb"]):
        return None
    dims = _subject_columns(target["subject"], frame, catalogue, exclude=target["column"])
    words = {w.lower() for w in re.findall(r"[a-z]+", target["phrase"], re.IGNORECASE)}
    if target["column"] is None:
        quantity = _token_column(frame, _QUANTITY_TOKENS)
        price = _token_column(frame, _PRICE_TOKENS)
        if words & _MONEY_WORDS and quantity and price:
            # No revenue column, but quantity and price: revenue is their product,
            # and the finding says so.
            label = f"revenue ({quantity} x {price})"
            expr = {"product": [quantity, price]}
            if dims:
                return _contribution(frame, result, dims, expr, label, sampled=sampled)
            target = {**target, "label": label, "expr": expr}
            return _drivers(frame, target, catalogue, result, sampled=sampled)
        two_valued = [n for n, c in catalogue.items()
                      if int(c.get("n_unique") or 0) == 2 and n in frame.columns][:5]
        hint = (f" Two-valued columns that could be the outcome: {', '.join(two_valued)}." if two_valued else "")
        result.update(kind="drivers", status="limited", metrics={"target": None, "phrase": target["phrase"]},
                      findings=[f"The goal asks about '{target['phrase']}', but no column or value in the data "
                                f"is named that, so there is no outcome to measure the other columns "
                                f"against.{hint} Name the outcome column in the goal."])
        return result
    if dims and target["value"] is None and _is_quantity(frame, target["column"]):
        return _contribution(frame, result, dims, {"column": target["column"]}, target["column"], sampled=sampled)
    return _drivers(frame, target, catalogue, result, sampled=sampled)


def _token_column(frame: pd.DataFrame, tokens: frozenset[str]) -> str | None:
    found = [str(c) for c in frame.columns if _name_tokens(c) & tokens and _is_quantity(frame, str(c))]
    return found[0] if len(found) == 1 else None


def _item_column(frame: pd.DataFrame, catalogue: dict[str, dict[str, Any]]) -> str | None:
    """The column naming what is sold: a description or product name first."""
    for token in _ITEM_TOKENS:
        for name in map(str, frame.columns):
            if token in _name_tokens(name) | {name.lower()} and frame[name].notna().any():
                if catalogue.get(name, {}).get("role") != "numeric" or token == "stockcode":
                    return name
    return None


def _subject_columns(subject: str, frame: pd.DataFrame, catalogue: dict[str, dict[str, Any]],
                     exclude: str | None) -> list[str]:
    """The dimensions a goal names before its verb: "which products and countries"."""
    out: list[str] = []
    for word in re.findall(r"[\w-]+", subject or ""):
        lowered = word.lower()
        if lowered in _ITEM_WORDS:
            column = _item_column(frame, catalogue)
        elif lowered in _GENERIC_SUBJECTS or len(lowered) < 3:
            continue
        else:
            column = _resolve_column_word(lowered, catalogue)
        if column and column != exclude and column not in out and not _is_quantity(frame, column):
            out.append(column)
    return out


def _contribution(frame: pd.DataFrame, result: dict[str, Any], dims: list[str],
                  expr: dict[str, Any] | None, label: str, *, sampled: bool) -> dict[str, Any]:
    """Which values of each dimension carry the most of a measure, with their share."""
    values = _values(frame, expr) if expr else pd.Series(1.0, index=frame.index)
    total = float(values.sum(skipna=True))
    result.update(kind="contribution", metrics={"measure": label, "measure_expr": expr, "total": _number(total),
                                                "dimensions": {}})
    findings, figures = [], []
    if expr:
        figures.append({"label": f"total {label}", "value": _number(total),
                        "check": {"op": "agg", "agg": "sum", "value": expr, "where": []}})
    scope = "sample " if sampled else ""
    for dim in dims:
        keys = frame[dim].map(str).where(frame[dim].notna(), "<missing>")
        sums = values.groupby(keys).sum().sort_values(ascending=False).head(10)
        rows = [{"group": str(k), "value": _number(v), "share": _number(v / total) if total else None}
                for k, v in sums.items()]
        result["metrics"]["dimensions"][dim] = rows
        shown = ", ".join(f"{r['group']} ({_display_number(round(r['value'], 2))}"
                          + (f", {r['share']:.1%}" if r["share"] is not None else "") + ")" for r in rows[:5])
        findings.append(f"{dim}: the largest {scope}{label} comes from {shown}.")
        figures += [{"label": f"{dim}={r['group']} {label}", "value": r["value"],
                     "check": {"op": "agg", "agg": "sum" if expr else "count", "value": expr,
                               "where": [{"column": dim, "equals_text": r["group"], "missing_as": "<missing>"}]}}
                    for r in rows[:5]]
    negative = int((values < 0).sum())
    if negative:
        findings.append(f"{negative:,} rows carry a negative {label} (returns or cancellations); "
                        "the figures are net of them.")
    if sampled:
        findings.append("These are totals over the sample only, not dataset totals.")
    result.update(status="answered" if figures else "limited", findings=findings, figures=figures)
    return result


def _drivers(frame: pd.DataFrame, target: dict[str, Any], catalogue: dict[str, dict[str, Any]],
             result: dict[str, Any], *, sampled: bool) -> dict[str, Any]:
    """Every other column measured against the outcome, strongest first.

    The measure is chosen by the two columns' kinds, each on a 0..1 scale:
    Pearson r between two quantities (signed; point-biserial when the
    outcome is 0/1), the correlation ratio (eta) between a quantity and a
    category, Cramer's V between two categories. A category code is a
    category. Descriptive association, never a cause.

    The ranking is computed from :func:`driver_partials`, the same
    sufficient statistics a swarm clone hands back, so one frame and N
    merged shards of it give the same drivers (:func:`merge_driver_partials`).
    """
    column = target.get("column")
    y_expr = target["expr"] if "expr" in target else outcome_expr(frame, column, target.get("value"))
    described = _describe_target(target)
    # The profile's roles that exclude a column, so a clone's partials
    # exclude the same columns this answer does.
    roles = {n: c.get("role") for n, c in catalogue.items() if c.get("role") in _EXCLUDED_ROLES}
    metrics: dict[str, Any] = {"target": column, "target_value": target.get("value"), "target_expr": y_expr,
                               "roles": roles}
    result.update(kind="drivers", metrics=metrics)
    # One shard's statistics through the same merge a fleet's go through, so
    # one pass and N merged shards decide every column by the same rule.
    stats, drivers, skipped = driver_ranking(frame, column, target.get("value"), y_expr=y_expr, roles=roles)
    outcome = stats["outcome"]
    if outcome["binary"]:
        positives, negatives = outcome["positives"], outcome["negatives"]
        metrics.update(positives=positives, negatives=negatives)
        if min(positives, negatives) < MIN_OUTCOME_ROWS:
            others = ", ".join(f"{t['value']} ({int(t['count']):,})"
                               for t in catalogue.get(column, {}).get("top_values", [])[:6]
                               if str(t["value"]).lower() != str(target.get("value")).lower())
            result.update(status="limited", findings=[
                f"{described} holds {positives:,} of {positives + negatives:,} rows: too few on one side to "
                f"measure what predicts it." + (f" {column} also holds: {others}; name the outcome value "
                                                f"you mean, e.g. 'what predicts {column} = <value>'."
                                                if others else "")])
            return result
    metrics.update(drivers=[{k: v for k, v in d.items() if k != "check"} for d in drivers[:10]],
                   skipped=skipped)
    findings, shown = driver_findings(described, drivers,
                                      keep=goal_named_columns(result.get("goal", ""), stats["columns"]))
    # The columns the findings name, so a web answer names the same ones.
    metrics["shown"] = [d["column"] for d in shown]
    if not shown:
        result.update(status="limited", findings=findings)
        return result
    result.update(status="answered", findings=findings, figures=[
        {"label": f"{_SYMBOL[d['measure']]}({d['column']}, {described})", "value": d["strength"],
         "check": d["check"]} for d in shown])
    return result


# --- drivers from sufficient statistics ---------------------------------------
#: Profile roles that keep a column out of the driver ranking.
_EXCLUDED_ROLES = frozenset({"datetime", "text", "id"})
#: Distinct values of an identifier-named numeric column a partial carries, so
#: the merge counts them exactly; past it the merge bounds the count.
_MAX_DISTINCT_CARRIED = 50_000
_SYMBOL = {"pearson_r": "r", "eta": "eta", "cramers_v": "Cramer's V"}

#: Fields recorded after the outcome they would "predict" (a loan's
#: settlement, its last payment, what was received on it), matched as whole
#: name tokens, never substrings: one token that is itself post-outcome...
#: (Not "chargeoff" alone: LendingClub's chargeoff_within_12_mths counts the
#: borrower's charge-offs before the application; charge_off is caught below.)
_LEAKAGE_TOKENS = frozenset({"settlement", "settlements", "settled", "recoveries", "hardship", "payoff",
                             "writeoff", "deferral"})
#: ...or a run of tokens that is (LendingClub's last_pymnt_d, total_rec_prncp).
_LEAKAGE_SEQUENCES = (("last", "pymnt"), ("next", "pymnt"), ("total", "rec"), ("total", "pymnt"),
                      ("out", "prncp"), ("charge", "off"), ("charged", "off"), ("write", "off"),
                      ("collection", "recovery"), ("last", "fico"), ("last", "credit", "pull"),
                      ("orig", "projected"), ("payment", "plan"))
#: A timing word counts only beside a payment or outcome noun: last_name,
#: days_since_last_visit, after_hours_calls and post_code are predictors.
_LEAKAGE_TIMING = frozenset({"last", "next", "final", "after", "post"})
_LEAKAGE_NOUNS = frozenset({"pymnt", "payment", "payments", "paid", "pay", "repayment", "recovery",
                            "collection", "settlement", "default", "chargeoff"})
LEAKAGE_RULE = ("a name heuristic, not a measured timeline: settlement, recoveries, hardship, payoff, "
                "charge-off, last or next payment, total received and the like")


def _ordered_tokens(name: Any) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(name))]


def likely_leakage(name: str) -> str | None:
    """Why ``name`` looks like a field known only after the outcome, or
    ``None``: ``"name"`` for a post-outcome token or run of tokens,
    ``"timing"`` for last/next/final/after/post beside a payment noun."""
    tokens = _ordered_tokens(name)
    if set(tokens) & _LEAKAGE_TOKENS:
        return "name"
    for seq in _LEAKAGE_SEQUENCES:
        if any(tuple(tokens[i:i + len(seq)]) == seq for i in range(len(tokens) - len(seq) + 1)):
            return "name"
    if set(tokens) & _LEAKAGE_TIMING and set(tokens) & _LEAKAGE_NOUNS:
        return "timing"
    return None


def _moments(values: np.ndarray) -> list[Any]:
    """[n, mean, M2, min, max] of ``values`` (no gaps): Chan's mergeable form."""
    n = int(len(values))
    if not n:
        return [0, 0.0, 0.0, None, None]
    mean = float(values.mean())
    return [n, mean, float(((values - mean) ** 2).sum()), float(values.min()), float(values.max())]


def _add_moments(a: list[Any], b: list[Any]) -> list[Any]:
    if not a[0]:
        return list(b)
    if not b[0]:
        return list(a)
    n, d = a[0] + b[0], b[1] - a[1]
    return [n, a[1] + d * b[0] / n, a[2] + b[2] + d * d * a[0] * b[0] / n, min(a[3], b[3]), max(a[4], b[4])]


def _co_moments(x: np.ndarray, y: np.ndarray) -> list[Any]:
    """[n, mean x, mean y, M2 x, M2 y, co-moment, min x, max x, min y, max y]."""
    n = int(len(x))
    if not n:
        return [0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, None]
    mx, my = float(x.mean()), float(y.mean())
    dx, dy = x - mx, y - my
    return [n, mx, my, float((dx * dx).sum()), float((dy * dy).sum()), float((dx * dy).sum()),
            float(x.min()), float(x.max()), float(y.min()), float(y.max())]


def _add_co(a: list[Any], b: list[Any]) -> list[Any]:
    if not a[0]:
        return list(b)
    if not b[0]:
        return list(a)
    n = a[0] + b[0]
    dx, dy, w = b[1] - a[1], b[2] - a[2], a[0] * b[0] / n
    return [n, a[1] + dx * b[0] / n, a[2] + dy * b[0] / n, a[3] + b[3] + dx * dx * w, a[4] + b[4] + dy * dy * w,
            a[5] + b[5] + dx * dy * w, min(a[6], b[6]), max(a[7], b[7]), min(a[8], b[8]), max(a[9], b[9])]


def _group_moments(values: pd.Series, groups: pd.Series) -> dict[str, list[Any]]:
    """Per label, the moments of ``values`` where it has one."""
    keep = values.notna()
    v, g = values[keep].astype("float64"), groups[keep].astype(str)
    if not len(v):
        return {}
    by = v.groupby(g)
    agg = by.agg(["count", "mean", "min", "max"])
    m2 = ((v - by.transform("mean")) ** 2).groupby(g).sum()
    return {str(k): [int(r["count"]), float(r["mean"]), float(m2[k]), float(r["min"]), float(r["max"])]
            for k, r in agg.iterrows()}


def driver_partials(frame: pd.DataFrame, column: str, value: Any, y_expr: dict[str, Any] | None,
                    roles: dict[str, Any] | None = None) -> dict[str, Any]:
    """The sufficient statistics of a driver ranking over ``frame``.

    Per candidate column: co-moments with a numeric outcome (Pearson r),
    per-label moments (eta), or a contingency table with a categorical
    outcome (Cramer's V), plus what decides whether the column is ranked at
    all (distinct values, non-null count). Every part adds across shards, so
    the merged ranking is the one a single pass over all the rows gives.
    """
    roles = roles or {}
    y = _values(frame, y_expr) if y_expr else None
    target_labels = _labels(frame[column])
    outcome: dict[str, Any] = {"numeric": y is not None, "binary": False, "positives": 0, "negatives": 0}
    if y is not None:
        outcome["binary"] = bool(set(pd.unique(y.dropna())) <= {0.0, 1.0})
        outcome["positives"], outcome["negatives"] = int((y == 1).sum()), int((y == 0).sum())
        outcome["moments"] = _moments(y.dropna().to_numpy(dtype="float64"))
    else:
        outcome["counts"] = {str(k): int(v) for k, v in
                             target_labels[frame[column].notna()].value_counts().items()}
    columns: dict[str, Any] = {}
    product = (y_expr or {}).get("product") or []
    for name in map(str, frame.columns):
        if name == column or name in product:
            continue
        series = frame[name]
        role = roles.get(name)
        numeric = bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))
        # A whole-number measure the profile calls ``id`` is still a measure
        # (see _is_identifier); only a non-numeric one is a row label.
        if role in {"datetime", "text"} or (role == "id" and not pd.api.types.is_numeric_dtype(series)) \
                or pd.api.types.is_datetime64_any_dtype(series):
            columns[name] = {"excluded": role or "datetime"}
            continue
        non_null = series.dropna()
        entry: dict[str, Any] = {"quantity": _is_quantity(frame, name), "numeric": numeric,
                                 "non_null": int(len(non_null))}
        if entry["quantity"]:
            x = pd.to_numeric(series, errors="coerce")
            # Rule (c) of IDENTIFIER_RULE: whether the values are whole and
            # monotonic, and (when near-unique) the values themselves, so the
            # merge counts what is distinct across shards.
            facts = _key_facts(non_null)
            near_unique = facts is not None and non_null.nunique() >= IDENTIFIER_NEAR_UNIQUE * len(non_null)
            if facts is not None:
                entry.update(facts)
            if _name_tokens(name) & _ID_TOKENS or near_unique:
                distinct = pd.unique(non_null)
                entry["distinct"] = int(len(distinct))
                if len(distinct) <= _MAX_DISTINCT_CARRIED:
                    entry["values"] = sorted(map(str, distinct))
            entry["range"] = ([float(x.min()), float(x.max())] if x.notna().any() else None)
            if y is not None:
                both = x.notna() & y.notna()
                xb, yb = x[both].to_numpy(dtype="float64"), y[both].to_numpy(dtype="float64")
                entry["co"] = _co_moments(xb, yb)
                if outcome["binary"]:
                    hit = yb == 1
                    entry["classes"] = [int(hit.sum()), float(xb[hit].sum()), int((~hit).sum()),
                                        float(xb[~hit].sum())]
            else:
                entry["groups"] = _group_moments(x, target_labels)
        else:
            labels = non_null.astype("string").unique()
            if len(labels) > _BARE_VALUE_MAX_UNIQUE:
                entry.update(wide=True, distinct=int(len(labels)))
            else:
                entry["values"] = sorted(map(str, labels))
                groups = _labels(series)
                if y is not None:
                    entry["groups"] = _group_moments(y, groups)
                else:
                    keep = frame[column].notna()
                    table = pd.crosstab(groups[keep].astype(str), target_labels[keep].astype(str))
                    entry["table"] = {str(r): {str(c): int(v) for c, v in row.items() if v}
                                      for r, row in table.iterrows()}
        columns[name] = entry
    return {"kind": "drivers", "target": column, "target_value": value, "target_expr": y_expr,
            "rows": int(len(frame)), "outcome": outcome, "columns": columns}


def _missing_column(outcome: dict[str, Any]) -> dict[str, Any]:
    """A column a clone's loader dropped (empty throughout her shard): every
    one of her rows is ``<missing>`` in it, which is how a single pass reads it."""
    entry: dict[str, Any] = {"quantity": None, "numeric": None, "non_null": 0, "values": [], "range": None}
    if outcome["numeric"]:
        entry["groups"] = {"<missing>": outcome["moments"]} if outcome["moments"][0] else {}
    else:
        entry["table"] = {"<missing>": dict(outcome["counts"])} if outcome["counts"] else {}
    return entry


def merge_driver_partials(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """One set of driver statistics from several shards' (counts, sums and
    co-moments add; tables add cell by cell). A column two shards typed
    differently is marked ``disagree`` and not ranked."""
    first = parts[0]
    outcome: dict[str, Any] = {"numeric": first["outcome"]["numeric"],
                               "binary": all(p["outcome"]["binary"] for p in parts),
                               "positives": sum(p["outcome"]["positives"] for p in parts),
                               "negatives": sum(p["outcome"]["negatives"] for p in parts)}
    if outcome["numeric"]:
        acc = [0, 0.0, 0.0, None, None]
        for p in parts:
            acc = _add_moments(acc, p["outcome"]["moments"])
        outcome["moments"] = acc
    else:
        counts: dict[str, int] = {}
        for p in parts:
            for k, v in p["outcome"]["counts"].items():
                counts[k] = counts.get(k, 0) + v
        outcome["counts"] = counts
    names = list(dict.fromkeys(n for p in parts for n in p["columns"]))
    columns: dict[str, Any] = {}
    for name in names:
        entries = [p["columns"].get(name) or _missing_column(p["outcome"]) for p in parts]
        excluded = next((e["excluded"] for e in entries if "excluded" in e), None)
        if excluded:
            columns[name] = {"excluded": excluded}
            continue
        kinds = {(e["quantity"], e["numeric"]) for e in entries if e["quantity"] is not None}
        if len(kinds) > 1:
            columns[name] = {"disagree": True}
            continue
        quantity, numeric = next(iter(kinds), (False, False))
        merged: dict[str, Any] = {"quantity": quantity, "numeric": numeric,
                                  "non_null": sum(e["non_null"] for e in entries)}
        held = [e for e in entries if e["non_null"]]
        if held and all(e.get("integer") for e in held):
            # Monotonic only when every shard is, each in its own row order.
            merged.update(integer=True, monotonic=all(e.get("monotonic") for e in held))
        if any(e.get("wide") for e in entries):
            merged.update(wide=True, distinct=max(int(e.get("distinct") or 0) for e in entries))
            columns[name] = merged
            continue
        if any("values" in e for e in entries) or any("distinct" in e for e in entries):
            if all("values" in e for e in entries):
                merged["values"] = sorted(set().union(*(e["values"] for e in entries)))
            else:
                # Some clone held too many to carry: the count is bounded, not known.
                merged["distinct_bounds"] = [max(int(e.get("distinct") or len(e.get("values") or []))
                                                 for e in entries),
                                             sum(int(e.get("distinct") or len(e.get("values") or []))
                                                 for e in entries)]
        if not quantity and merged.get("values") is not None and len(merged["values"]) > _BARE_VALUE_MAX_UNIQUE:
            merged.update(wide=True, distinct=len(merged["values"]))
            columns[name] = merged
            continue
        ranges = [e["range"] for e in entries if e.get("range")]
        merged["range"] = [min(r[0] for r in ranges), max(r[1] for r in ranges)] if ranges else None
        if any("co" in e for e in entries):
            acc = [0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, None]
            for e in entries:
                if "co" in e:
                    acc = _add_co(acc, e["co"])
            merged["co"] = acc
        if any("classes" in e for e in entries):
            merged["classes"] = [sum(e["classes"][i] for e in entries if "classes" in e) for i in range(4)]
        if any("groups" in e for e in entries):
            groups: dict[str, list[Any]] = {}
            for e in entries:
                for k, m in (e.get("groups") or {}).items():
                    groups[k] = _add_moments(groups.get(k, [0, 0.0, 0.0, None, None]), m)
            merged["groups"] = groups
        if any("table" in e for e in entries):
            table: dict[str, dict[str, int]] = {}
            for e in entries:
                for r, row in (e.get("table") or {}).items():
                    cell = table.setdefault(r, {})
                    for c, v in row.items():
                        cell[c] = cell.get(c, 0) + v
            merged["table"] = table
        columns[name] = merged
    return {"kind": "drivers", "target": first["target"], "target_value": first["target_value"],
            "target_expr": first["target_expr"], "rows": sum(p["rows"] for p in parts),
            "outcome": outcome, "columns": columns}


def _eta_of(groups: dict[str, list[Any]]) -> float | None:
    """The correlation ratio from per-group moments (see :func:`_eta`)."""
    total = [0, 0.0, 0.0, None, None]
    for m in groups.values():
        total = _add_moments(total, m)
    if total[0] < MIN_OUTCOME_ROWS or total[2] <= 0 or total[3] == total[4]:
        return None
    between = sum(m[0] * (m[1] - total[1]) ** 2 for m in groups.values() if m[0])
    return math.sqrt(max(0.0, min(1.0, between / total[2])))


def _cramers_v_of(table: dict[str, dict[str, int]]) -> float | None:
    """Cramer's V from a contingency table (see :func:`_cramers_v`)."""
    rows = sorted(table)
    cols = sorted({c for row in table.values() for c, v in row.items() if v})
    rows = [r for r in rows if any(table[r].get(c) for c in cols)]
    if not rows or not cols:
        return None
    counts = np.array([[table[r].get(c, 0) for c in cols] for r in rows], dtype="float64")
    n = float(counts.sum())
    k = min(counts.shape) - 1
    if n < MIN_OUTCOME_ROWS or k < 1:
        return None
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / n
    chi2 = float(((counts - expected) ** 2 / np.where(expected > 0, expected, 1)).sum())
    return math.sqrt(chi2 / (n * k))


def _identifier_from_stats(name: str, entry: dict[str, Any], role: str | None) -> bool:
    """:func:`_is_identifier` over merged statistics."""
    non_null = entry["non_null"]
    if entry.get("values") is not None:
        low = high = len(entry["values"])
    elif entry.get("distinct_bounds"):
        low, high = entry["distinct_bounds"]
    else:
        low = high = int(entry.get("distinct") or 0)
    if _name_tokens(name) & _ID_TOKENS:
        if low > 0.5 * max(1, non_null):
            return True
        if high > 0.5 * max(1, non_null):
            # Each shard's values were too many to carry, and between them
            # they could be half the rows: an identifier-named column that
            # may be one is not ranked.
            return True
    if entry["numeric"]:
        if not entry.get("integer") or not (entry.get("values") is not None or entry.get("distinct_bounds")):
            return False
        lo_hi = entry.get("range")
        span = lo_hi[1] - lo_hi[0] + 1 if lo_hi else None
        # Merged across shards the distinct count may only be bounded: a key
        # by its smallest possible count is one.
        return _integer_key(non_null, low, span, bool(entry.get("monotonic")))
    return role == "id" or (non_null >= IDENTIFIER_MIN_ROWS and low >= IDENTIFIER_NEAR_UNIQUE * non_null)


def rank_drivers(stats: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """The driver ranking from (merged) :func:`driver_partials`: entries
    ``{column, measure, strength, check, detail?, leakage?}`` strongest
    first, and the columns that were not measured."""
    column, y_expr, outcome = stats["target"], stats["target_expr"], stats["outcome"]
    binary = outcome["binary"]
    y_mean = outcome["moments"][1] if outcome["numeric"] and outcome["moments"][0] else None
    drivers: list[dict[str, Any]] = []
    skipped: list[str] = []
    for name, e in stats["columns"].items():
        if "excluded" in e or e.get("disagree") or e.get("wide"):
            skipped.append(name)
            continue
        if e["quantity"]:
            lo_hi = e.get("range")
            if lo_hi is None or lo_hi[0] == lo_hi[1] or _identifier_from_stats(name, e, None):
                skipped.append(name)
                continue
            if y_expr is not None:
                co = e.get("co") or [0]
                if co[0] < MIN_OUTCOME_ROWS or co[6] == co[7] or co[8] == co[9]:
                    continue
                den = math.sqrt(co[3] * co[4])
                strength = co[5] / den if den > 0 else None
                measure, check = "pearson_r", {"op": "pearson", "x": {"column": name}, "y": y_expr}
            else:
                strength, measure = _eta_of(e.get("groups") or {}), "eta"
                check = {"op": "eta", "y": {"column": name}, "groups": column}
        else:
            values = e.get("values") or []
            if len(values) <= 1 or _identifier_from_stats(name, e, None):
                skipped.append(name)
                continue
            if y_expr is not None:
                strength, measure = _eta_of(e.get("groups") or {}), "eta"
                check = {"op": "eta", "y": y_expr, "groups": name}
            else:
                strength, measure = _cramers_v_of(e.get("table") or {}), "cramers_v"
                check = {"op": "cramers_v", "a": name, "b": column}
        if strength is None or not math.isfinite(strength):
            continue
        entry: dict[str, Any] = {"column": name, "measure": measure, "strength": _number(strength),
                                 "check": check}
        if binary:
            entry["detail"] = _binary_detail_of(name, e, y_mean)
        leak = likely_leakage(name)
        if leak:
            entry["leakage"] = leak
        drivers.append(entry)
    drivers.sort(key=lambda d: (-abs(d["strength"]), d["column"]))
    return drivers, skipped


def outcome_expr(frame: pd.DataFrame, column: str, value: Any = None) -> dict[str, Any] | None:
    """How a driver ranking reads the outcome: an indicator of ``column ==
    value``, the column itself when it is a quantity, or None for a
    categorical outcome (ranked by Cramer's V and eta)."""
    if value is not None:
        return {"indicator": {"column": column, "equals": value}}
    if _is_quantity(frame, column):
        return {"column": column}
    return None


def driver_ranking(frame: pd.DataFrame, column: str, value: Any = None, *,
                   y_expr: dict[str, Any] | None | bool = True,
                   roles: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """THE ranking of what predicts an outcome: ``(stats, drivers, skipped)``.

    Every place that says which columns matter most -- the CLI's drivers
    answer, the swarm's merge, the claim verifier -- ranks with this (or,
    across shards, with :func:`merge_driver_partials` and
    :func:`rank_drivers`, which this is one shard of). The measures are
    scale-free, each on 0..1: Pearson r (point-biserial for a 0/1 outcome),
    the correlation ratio eta for a category, Cramer's V for two categories.
    A model coefficient is not one of them: on unscaled features it is per
    unit of each column (a year of age against a 0/1 dummy), so its size
    says nothing about which column matters more (stroke: ever_married_Yes
    1.27, age 0.07, while age has r=0.245 and marriage eta=0.108).

    What it measures is univariate association, one column at a time; it
    does not control for the other columns. Every consumer that shows it --
    the outcome report when it lands included -- labels it as association
    ("age has the strongest association with stroke (r = 0.245)",
    :func:`association_label`), never as importance.

    ``y_expr`` True reads the outcome by :func:`outcome_expr`."""
    if y_expr is True:
        y_expr = outcome_expr(frame, column, value)
    stats = merge_driver_partials([driver_partials(frame, column, value, y_expr or None, roles)])
    drivers, skipped = rank_drivers(stats)
    return stats, drivers, skipped


def _binary_detail_of(name: str, e: dict[str, Any], y_mean: float | None) -> str:
    """The association in the reader's units: a mean either side, or the group most often positive."""
    if e["quantity"]:
        n1, s1, n0, s0 = e.get("classes") or (0, 0.0, 0, 0.0)
        hit = s1 / n1 if n1 else float("nan")
        rest = s0 / n0 if n0 else float("nan")
        return (f"mean {_display_number(round(hit, 3))} where the outcome holds vs "
                f"{_display_number(round(rest, 3))} otherwise")
    groups = e.get("groups") or {}
    rates = {k: m for k, m in groups.items() if m[0] >= MIN_OUTCOME_ROWS}
    if not rates or y_mean is None:
        return ""

    def said(pool: dict[str, Any]) -> str:
        best = max(m[1] for m in pool.values())
        tied = sorted(k for k, m in pool.items() if m[1] == best)
        at = (f"{name}={tied[0]}" if len(tied) == 1
              else " and ".join(f"{name}={k}" for k in tied[:3]) + (f" (and {len(tied) - 3} more)" if len(tied) > 3 else "")
              + " (tied)")
        return f"highest rate at {at} ({best:.1%} vs {y_mean:.1%} overall)"

    text = said(rates)
    if len(rates) < len(groups) and text != said(groups):
        # A smaller group's rate is higher (or ties): say whose "highest" this is.
        text += f" among {name} values with at least {MIN_OUTCOME_ROWS} rows"
    return text


def goal_named_columns(goal: str, columns: Any) -> set[str]:
    """The columns ``goal`` names by their whole name: a field the user asks
    for by name is ranked even when it looks like leakage."""
    text = (goal or "").lower()
    return {str(c) for c in columns if re.search(rf"(?<![\w]){re.escape(str(c).lower())}(?![\w])", text)}


def driver_findings(described: str, drivers: list[dict[str, Any]], *,
                    keep: set[str] | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    """The findings of a driver ranking, and the drivers they show.

    A field that looks known only after the outcome (:func:`likely_leakage`)
    is not ranked but is never dropped silently: it is listed, with its
    strength, as possible leakage, and naming it in the goal (``keep``)
    ranks it. One that is almost the outcome itself (association >= 0.95)
    is ranked and flagged."""
    keep = keep or set()
    ranked = [d for d in drivers if not d.get("leakage") or d["column"] in keep]
    leaked = [d for d in drivers if d.get("leakage") and d["column"] not in keep]
    shown = ranked[:5]
    if not shown:
        findings = [f"No column could be measured against {described}: the others are identifiers, dates or "
                    "free text" + (", or possible leakage." if leaked else ".")]
    else:
        findings = [f"What moves with {described}, strongest first (association, not cause):"]
        for d in shown:
            findings.append(f"{d['column']}: {_SYMBOL[d['measure']]}={_signed(d)}"
                            + (f"; {d['detail']}" if d.get("detail") else "") + ".")
        findings.append(association_label(shown[0], described))
        near = [d["column"] for d in shown if abs(d["strength"]) >= _LEAKAGE_STRENGTH]
        if near:
            findings.append(f"{', '.join(near)} {'is' if len(near) == 1 else 'are'} almost the outcome itself "
                            "(a total that contains it, or a field recorded after it); check before treating "
                            "it as a predictor.")
    if leaked:
        findings.append("Set aside as possible leakage, fields that may be known only after the outcome ("
                        + LEAKAGE_RULE + "): "
                        + ", ".join(f"{d['column']} ({_SYMBOL[d['measure']]}={_signed(d)})" for d in leaked)
                        + ". Name one in the goal to rank it.")
    return findings, shown


def association_label(driver: dict[str, Any], described: str) -> str:
    """How the top of a :func:`driver_ranking` is said to a user: as an
    association, never as an importance that controls for the other columns
    ("age has the strongest association with stroke (r = 0.245)")."""
    return (f"{driver['column']} has the strongest association with {described} "
            f"({_SYMBOL[driver['measure']]} = {abs(driver['strength']):.3f}); each figure is one column's "
            "association with the outcome on its own and does not control for the other columns.")


def _signed(d: dict[str, Any]) -> str:
    return f"{d['strength']:+.3f}" if d["measure"] == "pearson_r" else f"{d['strength']:.3f}"


def _peak(frame: pd.DataFrame, goal: str, catalogue: dict[str, dict[str, Any]], result: dict[str, Any],
          *, sampled: bool) -> dict[str, Any]:
    """"When is X highest?": the hour, weekday and month where X's mean peaks."""
    quantities = {n for n in map(str, frame.columns) if _is_quantity(frame, n)
                  and not _is_identifier(n, frame[n], catalogue.get(n, {}).get("role"))}
    lowest = bool(re.search(r"\b(lowest|least)\b", goal, re.IGNORECASE))
    words = [w for w in re.findall(r"[\w-]+", goal.lower())
             if len(w) >= 3 and w not in _NOT_A_VALUE and w not in {"peak", "peaks", "busiest", "least"}]
    measure = next((c for w in words if (c := _resolve_column_word(w, catalogue, quantities))), None)
    measure = measure or next((c for c in _mentioned_columns(goal, sorted(quantities))), None)
    result["kind"] = "peak_timing"
    if measure is None:
        asked = " ".join(w for w in words if w not in {"when", "what", "time", "day", "hour", "month"}) or "the measure"
        named = ", ".join(sorted(quantities)[:8])
        result.update(status="limited", metrics={"value_column": None, "candidates": sorted(quantities)},
                      findings=[f"The goal asks when {asked} peaks, but no column is named that"
                                + (f"; it could mean {named}" if named else "")
                                + ". Name the column in the goal to get a figure."])
        return result
    date_col = next((n for n, c in catalogue.items() if c.get("role") == "datetime" and n in frame.columns), None)
    if date_col is None:
        result.update(status="limited", metrics={"value_column": measure},
                      findings=["A peak time cannot be found: no date/time column was identified."])
        return result
    dates = pd.to_datetime(frame[date_col], errors="coerce", dayfirst=_day_first(frame[date_col]))
    values = pd.to_numeric(frame[measure], errors="coerce")
    findings, figures, peaks = [], [], {}
    parts = [("month", dates.dt.to_period("M").astype("string")), ("weekday", dates.dt.day_name())]
    if dates.dt.hour.fillna(0).ne(0).any():
        parts.insert(0, ("hour", dates.dt.hour))
    for part, keys in parts:
        means = values.groupby(keys).mean().dropna()
        if len(means) < 2:
            continue
        key = means.idxmin() if lowest else means.idxmax()
        peaks[part] = {"at": str(key), "mean": _number(means[key])}
        findings.append(f"By {part}: {'lowest' if lowest else 'highest'} mean {measure} at {key} "
                        f"({_display_number(round(means[key], 4))}).")
        figures.append({"label": f"mean {measure} at {part} {key}", "value": _number(means[key]),
                        "check": {"op": "agg", "agg": "mean", "value": {"column": measure},
                                  "where": [{part: str(key), "column": date_col, "dayfirst": _day_first(frame[date_col])}]}})
    result.update(status="answered" if figures else "limited",
                  metrics={"value_column": measure, "time_column": date_col, "peaks": peaks},
                  findings=findings or ["At least two periods with valid values are needed to find a peak."],
                  figures=figures)
    return result


def _day_first(series: pd.Series) -> bool:
    """``16/12/2006`` is day-first; a text date column whose first field exceeds 12 says so."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    first = series.dropna().astype(str).head(200).str.extract(r"^(\d{1,2})[/.-]")[0].dropna()
    return bool(len(first)) and bool((first.astype(int) > 12).any())
