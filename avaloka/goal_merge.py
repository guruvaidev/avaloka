"""One answer to the goal from the clones' partial answers.

Each sub-Avaloka answers the goal over her own shard
(:func:`avaloka.goal_analysis.analyze_goal` writes it to her
``goal_analysis.json``). Those answers are not the answer: the mean of two
shard means is not the dataset's mean, and a shard's top ten groups are not
the dataset's. So, beside her answer, each clone hands back the *sufficient
statistics* of it (:func:`goal_partials`), computed over the same rows and
filters her answer used, and the main Avaloka merges those
(:func:`merge_goal_answers`):

* sums add, counts add, and a mean is the merged sum over the merged count
  (weighted by n, never a mean of means); min and max are the min and max;
  Pearson r is recomputed from merged co-moment sums;
* a median or a distinct count cannot be merged from partials, so it is
  given as an approximation and labelled one, never passed off as exact.

The merged answer cites what each clone contributed, by the row range of the
source she was given, so a reader can see how one answer was made of N.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from avaloka.goal_analysis import (_SYMBOL, MIN_OUTCOME_ROWS, _describe_target, _display_number, _number, apply_filters,
                                   describe_filters, driver_findings, driver_partials, goal_named_columns,
                                   merge_driver_partials, _signed, rank_drivers)

#: Distinct groups (or values) a clone hands back. Beyond this the merged
#: ranking may miss a group another clone ranked low, so it says approximate.
MAX_KEYS = 50_000
#: Numeric columns a correlation partial covers (the co-moment sums are
#: quadratic in the column count).
MAX_CORRELATION_COLUMNS = 40

_CAUSAL = re.compile(r"\b(why|cause|caused|driver|drivers)\b", re.IGNORECASE)


# --- what a clone hands back -------------------------------------------------
def goal_partials(frame: pd.DataFrame | None, analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    """The mergeable statistics behind one clone's ``analysis``, or ``None``
    when it carries no figure to merge (an overview, an unresolved measure).

    ``frame`` is the working frame ``analysis`` was calculated from; the
    analysis's own filters are applied again so both see the same rows.
    """
    if frame is None or not analysis:
        return None
    frame = apply_filters(frame, analysis.get("filters") or [])
    kind, metrics = analysis.get("kind"), analysis.get("metrics") or {}
    if kind == "group_comparison" and metrics.get("group_column") in frame:
        return _group_partials(frame, metrics)
    if kind == "trend" and metrics.get("time_column") in frame and metrics.get("aggregation"):
        return _trend_partials(frame, metrics)
    if kind == "column_summary" and metrics:
        return _summary_partials(frame, metrics)
    if kind == "missingness":
        columns = [c for c in (analysis.get("columns") or [str(c) for c in frame.columns]) if c in frame]
        return {"kind": kind, "rows": int(len(frame)),
                "missing": {c: int(frame[c].isna().sum()) for c in columns}}
    if kind == "correlation":
        return _correlation_partials(frame)
    if kind == "drivers" and metrics.get("target") in frame:
        # Computed whatever this clone concluded on her own rows: a shard of
        # a sorted file can hold no positive at all ("stroke holds 0 of
        # 2,555 rows"), and her rows still count towards the dataset's.
        return driver_partials(frame, metrics["target"], metrics.get("target_value"),
                               metrics.get("target_expr"), metrics.get("roles"))
    return None


def _group_partials(frame: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any] | None:
    group, value, aggregation = metrics["group_column"], metrics.get("value_column"), metrics.get("aggregation")
    if aggregation is None or (value is not None and value not in frame):
        return None
    grouped = frame.groupby(group, dropna=False)
    sizes = grouped.size()
    if value is not None:
        numbers = pd.to_numeric(frame[value], errors="coerce")
        n = numbers.groupby(frame[group], dropna=False).count()
        sums = numbers.groupby(frame[group], dropna=False).sum()
    else:
        n, sums = sizes, sizes.astype(float)
    keys = sizes.sort_values(ascending=False).index[:MAX_KEYS]
    return {
        "kind": "group_comparison", "group_column": group, "value_column": value,
        "aggregation": aggregation, "rows": int(len(frame)), "truncated": len(sizes) > MAX_KEYS,
        "groups": {str(k): {"rows": int(sizes[k]), "n": int(n.get(k, 0)), "sum": float(sums.get(k, 0.0))}
                   for k in keys},
    }


def _trend_partials(frame: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    date_col, value = metrics["time_column"], metrics.get("value_column")
    dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    series = (pd.to_numeric(frame[value], errors="coerce") if value
              else pd.Series(1, index=frame.index, dtype="float64"))
    table = pd.DataFrame({"period": dates.dt.tz_localize(None).dt.to_period("M"), "value": series}).dropna()
    by = table.groupby("period")["value"]
    counts, sums = by.count(), by.sum()
    return {"kind": "trend", "time_column": date_col, "value_column": value,
            "aggregation": metrics["aggregation"], "rows": int(len(frame)),
            "periods": {str(p): {"n": int(counts[p]), "sum": float(sums[p])} for p in counts.index}}


def _summary_partials(frame: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for col, summary in metrics.items():
        if col not in frame:
            continue
        series = frame[col]
        if "top_values" not in summary and pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna()
            asked = [k for k in summary if k != "non_null"]
            columns[col] = {
                "numeric": True, "asked": asked, "n": int(len(values)),
                "sum": float(values.sum()) if len(values) else 0.0,
                "min": float(values.min()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
                "median": float(values.median()) if len(values) else None,
            }
        else:
            counts = series.astype("string").value_counts()
            columns[col] = {
                "numeric": False, "non_null": int(series.notna().sum()), "distinct": int(len(counts)),
                "truncated": len(counts) > MAX_KEYS,
                "counts": {str(k): int(v) for k, v in counts.head(MAX_KEYS).items()},
            }
    return {"kind": "column_summary", "rows": int(len(frame)), "columns": columns}


def _correlation_partials(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame.select_dtypes(include="number")
    names = [str(c) for c in numeric.columns[:MAX_CORRELATION_COLUMNS]]
    values = numeric.iloc[:, :MAX_CORRELATION_COLUMNS].to_numpy(dtype="float64")
    present = ~np.isnan(values)
    x = np.where(present, values, 0.0)
    m = present.astype("float64")
    # Pairwise-complete sums, as pandas' corr counts them: entry [i, j] is
    # over the rows where both column i and column j are present.
    return {"kind": "correlation", "rows": int(len(frame)), "columns": names,
            "capped": numeric.shape[1] > MAX_CORRELATION_COLUMNS,
            "n": (m.T @ m).tolist(), "sx": (x.T @ m).tolist(), "sxx": ((x * x).T @ m).tolist(),
            "sxy": (x.T @ x).tolist()}


# --- the main Avaloka's merge ------------------------------------------------
def merge_goal_answers(goal: str, clones: list[dict[str, Any]]) -> dict[str, Any]:
    """One answer to ``goal`` from the clones' answers and partials.

    ``clones`` holds, per clone in shard order: ``name`` (Ava-00),
    ``rows`` ([first, last] row of the source), ``analysis`` (her
    goal_analysis) and ``partials`` (:func:`goal_partials`).

    The result has the shape of a ``goal_analysis.json`` (goal, kind,
    status, scope, metrics, findings) plus ``headline`` (the answer in one
    sentence), ``contributions`` (each clone's part, cited by row range),
    ``exact`` and ``approximate`` (the statistics that are not exact, and why).
    """
    answers = [c for c in clones if c.get("analysis")]
    mergeable = [c for c in answers if c.get("partials")]
    specs = {_spec(c["partials"]) for c in mergeable}
    base: dict[str, Any] = {
        "goal": goal, "kind": "dataset_overview", "status": "limited", "scope": "", "columns": [],
        "metrics": {}, "findings": [], "exact": True, "approximate": [], "contributions": [],
        "merge": "none", "clones": len(clones),
    }
    if not answers:
        base["headline"] = f"No clone returned an answer to “{goal}”."
        return base
    first = answers[0]["analysis"]
    base.update(kind=first.get("kind", "dataset_overview"), columns=list(first.get("columns") or []))
    if not mergeable or len(mergeable) < len(answers) or len(specs) != 1:
        return _unmerged(base, goal, answers, disagree=len(specs) > 1)
    kind = mergeable[0]["partials"]["kind"]
    merged = {"group_comparison": _merge_groups, "trend": _merge_trend, "column_summary": _merge_summary,
              "missingness": _merge_missingness, "correlation": _merge_correlation,
              "drivers": _merge_drivers}[kind](goal, mergeable)
    base.update(merged)
    rows = sum(c["partials"]["rows"] for c in mergeable)
    # The basis is the clones' own: a shard answered from a sample makes the
    # merged answer a sample's too (the same basis the reporter states), and
    # the headline says so: an estimate, never the exact answer.
    sampled = any(_sampled(c["analysis"]) for c in mergeable)
    base["sampled"] = sampled
    base["scope"] = f"{'Sample of' if sampled else 'All'} {rows:,} working rows of {len(mergeable)} clones, merged" + (
        f" where {describe_filters(first['filters'])}" if first.get("filters") else "")
    base["merge"] = "sufficient statistics"
    if sampled:
        base["basis"] = {"sampled": True, "per_clone": [_basis_rows(c["analysis"]) for c in mergeable]}
        estimate = f"estimated from samples of {_samples_phrase(base['basis']['per_clone'])} per clone"
        base["headline"] = base["headline"].rstrip(".") + f" ({estimate})."
        base["approximate"].append({"statistic": "every merged figure",
                                    "reason": f"{estimate}, not computed over every row"})
    else:
        base["basis"] = {"sampled": False, "rows_used": rows}
        if base["status"] == "answered":
            base["headline"] = (base["headline"].rstrip(".")
                                + f" (computed over every row of the {len(mergeable)} shards).")
    base["exact"] = not base["approximate"]
    return base


def _basis_rows(analysis: dict[str, Any]) -> tuple[int | None, int | None]:
    """(rows used, rows held) of one clone's answer."""
    basis = analysis.get("basis") or {}
    return basis.get("rows_used"), basis.get("source_rows")


def _samples_phrase(per_clone: list[tuple[int | None, int | None]]) -> str:
    """"50,000 of 2,000,000 rows", or each clone's pair when they differ."""
    pairs = sorted({(used, held) for used, held in per_clone}, key=str)

    def one(used: int | None, held: int | None) -> str:
        return f"{used:,} of {held:,} rows" if used is not None and held is not None else "an unrecorded share of rows"

    return one(*pairs[0]) if len(pairs) == 1 else ", ".join(one(u, h) for u, h in per_clone)


def _spec(p: dict[str, Any]) -> tuple:
    """What a partial measures: two clones that measured different things
    (one read a column as numbers, the other as text) are not merged."""
    if p["kind"] == "column_summary":
        return (p["kind"], tuple(sorted((c, v["numeric"]) for c, v in p["columns"].items())))
    if p["kind"] == "correlation":
        return (p["kind"], tuple(p["columns"]))
    if p["kind"] == "missingness":
        return (p["kind"], tuple(sorted(p["missing"])))
    if p["kind"] == "drivers":
        return (p["kind"], p["target"], str(p["target_value"]), json.dumps(p["target_expr"], sort_keys=True),
                p["outcome"]["numeric"])
    return (p["kind"], p.get("group_column") or p.get("time_column"), p.get("value_column"), p.get("aggregation"))


def _sampled(analysis: dict[str, Any]) -> bool:
    """Whether a clone's answer rests on a sample: its ``basis`` when it
    records one, otherwise the scope analyze_goal wrote."""
    basis = analysis.get("basis")
    if isinstance(basis, dict) and "sampled" in basis:
        # analyze_goal's own flag. Searching the dict's text for "sample"
        # matched the key itself, so every answer read as sampled.
        return bool(basis["sampled"])
    if basis is not None:
        return "sample" in str(basis.get("kind", basis) if isinstance(basis, dict) else basis).lower()
    return str(analysis.get("scope") or "").startswith("Sample of")


def _cite(c: dict[str, Any]) -> str:
    first, last = c["rows"]
    return f"{c['name']} (rows {first:,}–{last:,})"


def _unmerged(base: dict[str, Any], goal: str, answers: list[dict[str, Any]], *, disagree: bool) -> dict[str, Any]:
    """No figure to merge: say so, and give each clone's own finding."""
    first = answers[0]["analysis"]
    finding = next(iter(first.get("findings") or []), "no goal-specific calculation applied")
    if disagree:
        reason = "the clones measured it differently on their shards, so their figures were not merged"
    else:
        reason = "no clone computed a figure that merges"
    base.update(status="limited", exact=False, merge="none", findings=list(first.get("findings") or []),
                scope=f"{len(answers)} clones; not merged")
    base["headline"] = f"The fleet could not give one measured answer to “{goal}”: {reason}. {finding}"
    base["contributions"] = [{"clone": c["name"], "rows": c["rows"],
                              "finding": f"{_cite(c)}: " + next(iter(c["analysis"].get("findings") or []),
                                                                c["analysis"].get("status", "no answer"))}
                             for c in answers]
    return base


def _contribution(c: dict[str, Any], text: str) -> dict[str, Any]:
    return {"clone": c["name"], "rows": c["rows"], "finding": f"{_cite(c)}: {text}"}


def _value(entry: dict[str, Any], aggregation: str) -> float | None:
    if aggregation == "mean":
        return entry["sum"] / entry["n"] if entry["n"] else None
    if aggregation == "count":
        return float(entry["rows"]) if "rows" in entry else float(entry["n"])
    return entry["sum"]


def _merge_groups(goal: str, clones: list[dict[str, Any]]) -> dict[str, Any]:
    p0 = clones[0]["partials"]
    group, value, aggregation = p0["group_column"], p0["value_column"], p0["aggregation"]
    totals: dict[str, dict[str, float]] = {}
    for c in clones:
        for key, e in c["partials"]["groups"].items():
            t = totals.setdefault(key, {"rows": 0, "n": 0, "sum": 0.0})
            t["rows"] += e["rows"]; t["n"] += e["n"]; t["sum"] += e["sum"]
    values = {k: _value(t, aggregation) for k, t in totals.items()}
    ranked = sorted(((k, v) for k, v in values.items() if v is not None and not math.isnan(v)),
                    key=lambda kv: (-kv[1], kv[0]))[:10]
    rows = [{"group": k, "value": _number(v), "rows": int(totals[k]["rows"]), "n": int(totals[k]["n"])}
            for k, v in ranked]
    measure = f"{aggregation} {value or 'rows'}"
    approximate = []
    if any(c["partials"]["truncated"] for c in clones):
        approximate.append({"statistic": f"ranking of {group}",
                            "reason": f"a clone had more than {MAX_KEYS:,} groups; the ranking covers "
                                      f"the {MAX_KEYS:,} largest of each"})
    lead = rows[0]["group"] if rows else None
    contributions = []
    for c in clones:
        own = c["partials"]["groups"].get(lead) if lead is not None else None
        v = _value(own, aggregation) if own else None
        contributions.append(_contribution(
            c, f"{group}={lead}: {measure}={_display_number(round(v, 6))} over {own['n']:,} rows"
            if v is not None else f"no rows in {group}={lead}"))
    top = ", ".join(f"{r['group']} {_display_number(r['value'])}" for r in rows[:3])
    weighting = " (each mean weighted by its rows)" if aggregation == "mean" else ""
    headline = (f"Answer to “{goal}”: {measure} by {group}, merged from {len(clones)} clones{weighting} — "
                f"{top}" + (f"; highest {lead}." if lead is not None else ".")) if rows else \
        f"No group of {group} had a valid {measure} for “{goal}”."
    return {
        "status": "answered" if rows else "limited", "headline": headline,
        "metrics": {"group_column": group, "value_column": value, "aggregation": aggregation,
                    "groups": rows, "n_groups": len(totals)},
        "findings": [f"{group}={r['group']}: {measure}={_display_number(r['value'])} "
                     f"(n={r['n']:,})." for r in rows[:5]],
        "contributions": contributions, "approximate": approximate,
    }


def _merge_trend(goal: str, clones: list[dict[str, Any]]) -> dict[str, Any]:
    p0 = clones[0]["partials"]
    date_col, value, aggregation = p0["time_column"], p0["value_column"], p0["aggregation"]
    totals: dict[str, dict[str, float]] = {}
    for c in clones:
        for period, e in c["partials"]["periods"].items():
            t = totals.setdefault(period, {"n": 0, "sum": 0.0})
            t["n"] += e["n"]; t["sum"] += e["sum"]
    monthly = {p: _number(_value(totals[p], aggregation)) for p in sorted(totals)
               if _value(totals[p], aggregation) is not None}
    measure = f"{aggregation} {value}" if value else "row count"
    contributions = [_contribution(c, f"{len(c['partials']['periods'])} months, "
                                      f"{sum(e['n'] for e in c['partials']['periods'].values()):,} dated rows")
                     for c in clones]
    findings: list[str] = []
    if len(monthly) >= 2:
        (p1, v1), (p2, v2) = next(iter(monthly.items())), next(reversed(monthly.items()))
        findings = [f"Monthly {measure} changed from {v1:.3g} in {p1} to {v2:.3g} in {p2} "
                    f"(change {v2 - v1:+.3g}).", "This is a descriptive change, not an explanation of its cause."]
        headline = f"Answer to “{goal}”: monthly {measure} by {date_col}, merged from {len(clones)} clones — " \
                   + findings[0]
    else:
        findings = ["At least two months with valid values are needed for a time trend."]
        headline = f"The fleet found fewer than two months of {measure} for “{goal}”."
    answered = len(monthly) >= 2 and not _CAUSAL.search(goal)
    return {"status": "answered" if answered else "limited", "headline": headline,
            "metrics": {"time_column": date_col, "value_column": value, "aggregation": aggregation,
                        "monthly_values": monthly},
            "findings": findings, "contributions": contributions, "approximate": []}


_LABELS = {"max": "highest", "min": "lowest", "sum": "total", "mean": "mean", "median": "median",
           "count": "non-null count", "range": "range"}


def _merge_summary(goal: str, clones: list[dict[str, Any]]) -> dict[str, Any]:
    columns = list(clones[0]["partials"]["columns"])
    metrics: dict[str, Any] = {}
    findings: list[str] = []
    approximate: list[dict[str, str]] = []
    for col in columns:
        parts = [c["partials"]["columns"][col] for c in clones]
        if parts[0]["numeric"]:
            n = sum(p["n"] for p in parts)
            total = sum(p["sum"] for p in parts)
            lows = [p["min"] for p in parts if p["min"] is not None]
            highs = [p["max"] for p in parts if p["max"] is not None]
            exact = {"sum": total, "mean": total / n if n else None, "min": min(lows) if lows else None,
                     "max": max(highs) if highs else None, "count": n}
            exact["range"] = exact["max"] - exact["min"] if lows and highs else None
            summary: dict[str, Any] = {"non_null": n}
            asked = parts[0]["asked"] or ["mean", "median"]
            for m in asked:
                if m == "median":
                    medians = [(p["median"], p["n"]) for p in parts if p["median"] is not None]
                    w = sum(k for _, k in medians)
                    summary["median"] = _number(sum(v * k for v, k in medians) / w) if w else None
                    approximate.append({"statistic": f"median of {col}",
                                        "reason": "a median cannot be merged from the clones' medians; this "
                                                  "is their row-weighted average — run `avaloka analyze` "
                                                  "for the exact median"})
                elif m != "count":
                    summary[m] = _number(exact[m]) if exact.get(m) is not None else None
            metrics[col] = summary
            if not n:
                findings.append(f"{col}: no numeric values to summarise.")
                continue
            shown = []
            for m in asked:
                figure = n if m == "count" else summary.get(m)
                if figure is None:
                    continue
                shown.append(f"{_LABELS[m]}=" + _display_number(figure) + (" (approximate)" if m == "median" else ""))
            findings.append(f"{col}: {', '.join(shown)} (from {n:,} non-null rows).")
        else:
            counts: dict[str, int] = {}
            for p in parts:
                for k, v in p["counts"].items():
                    counts[k] = counts.get(k, 0) + v
            top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            non_null = sum(p["non_null"] for p in parts)
            distinct_exact = not any(p["truncated"] for p in parts)
            metrics[col] = {"non_null": non_null, "top_values": dict(top),
                            "distinct": len(counts) if distinct_exact else None,
                            "distinct_approximate": None if distinct_exact else max(p["distinct"] for p in parts)}
            if not distinct_exact:
                approximate.append({"statistic": f"distinct values and ranking of {col}",
                                    "reason": f"a clone had more than {MAX_KEYS:,} distinct values"})
            findings.append(f"{col}: {non_null:,} non-null rows; " + (
                "most common: " + ", ".join(f"{k} ({v})" for k, v in top) if top else "no values to summarise."))
    lead = columns[0] if columns else None
    contributions = []
    for c in clones:
        p = c["partials"]["columns"].get(lead) if lead else None
        if p is None:
            text = "no rows"
        elif p["numeric"]:
            text = (f"{lead}: {p['n']:,} non-null rows, total={_display_number(round(p['sum'], 6))}"
                    + (f", mean={_display_number(round(p['sum'] / p['n'], 6))}" if p["n"] else ""))
        else:
            text = f"{lead}: {p['non_null']:,} non-null rows, {p['distinct']:,} distinct values"
        contributions.append(_contribution(c, text))
    headline = f"Answer to “{goal}”, merged from {len(clones)} clones — " + " ".join(findings)
    return {"status": "answered", "headline": headline, "metrics": metrics, "findings": findings,
            "contributions": contributions, "approximate": approximate}


def _merge_missingness(goal: str, clones: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sum(c["partials"]["rows"] for c in clones)
    missing: dict[str, int] = {}
    for c in clones:
        for col, m in c["partials"]["missing"].items():
            missing[col] = missing.get(col, 0) + m
    ordered = sorted(missing, key=lambda col: (-missing[col], col))[:10]
    metrics = {col: {"missing": missing[col], "missing_pct": _number(missing[col] / rows * 100) if rows else 0.0}
               for col in ordered}
    findings = [f"{col}: {missing[col]:,} missing of {rows:,} rows ({metrics[col]['missing_pct']:.1f}%)."
                for col in ordered] or ["No columns are available to check for missing values."]
    lead = ordered[0] if ordered else None
    contributions = [_contribution(c, f"{lead}: {c['partials']['missing'].get(lead, 0):,} missing of "
                                      f"{c['partials']['rows']:,} rows" if lead else "no columns")
                     for c in clones]
    headline = f"Answer to “{goal}”, merged from {len(clones)} clones — " + " ".join(findings[:3])
    return {"status": "answered", "headline": headline, "metrics": metrics, "findings": findings,
            "contributions": contributions, "approximate": []}


def _merge_correlation(goal: str, clones: list[dict[str, Any]]) -> dict[str, Any]:
    names = clones[0]["partials"]["columns"]
    keys = ("n", "sx", "sxx", "sxy")
    acc = {k: sum(np.asarray(c["partials"][k], dtype="float64") for c in clones) for k in keys}
    selected = [c for c in (clones[0]["analysis"].get("columns") or []) if c in names]

    def r_of(s: dict[str, np.ndarray], i: int, j: int) -> float | None:
        n, sx, sy = s["n"][i, j], s["sx"][i, j], s["sx"][j, i]
        sxx, syy, sxy = s["sxx"][i, j], s["sxx"][j, i], s["sxy"][i, j]
        den = math.sqrt(max(n * sxx - sx * sx, 0.0) * max(n * syy - sy * sy, 0.0))
        return (n * sxy - sx * sy) / den if n >= 2 and den > 0 else None

    pairs = []
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            right = names[j]
            if len(selected) >= 2:
                include = left in selected and right in selected
            elif selected:
                include = left in selected or right in selected
            else:
                include = True
            r = r_of(acc, i, j) if include else None
            if r is not None:
                pairs.append({"left": left, "right": right, "r": _number(max(-1.0, min(1.0, r)))})
    pairs.sort(key=lambda p: (-abs(p["r"]), p["left"], p["right"]))
    pairs = pairs[:5]
    findings = [f"{p['left']} and {p['right']}: Pearson r={p['r']:.3f}." for p in pairs] \
        or ["No comparable numeric column pair has a defined correlation."]
    findings.append("Correlation alone does not establish causation.")
    contributions = []
    for c in clones:
        if pairs:
            s = {k: np.asarray(c["partials"][k], dtype="float64") for k in keys}
            i, j = names.index(pairs[0]["left"]), names.index(pairs[0]["right"])
            r = r_of(s, i, j)
            text = (f"{pairs[0]['left']} and {pairs[0]['right']}: r={r:.3f} over {int(s['n'][i, j]):,} rows"
                    if r is not None else "no defined correlation on her rows")
        else:
            text = "no defined correlation on her rows"
        contributions.append(_contribution(c, text))
    approximate = [{"statistic": "correlation pairs",
                    "reason": f"only the first {MAX_CORRELATION_COLUMNS} numeric columns were paired"}] \
        if any(c["partials"]["capped"] for c in clones) else []
    headline = f"Answer to “{goal}”, merged from {len(clones)} clones — " + " ".join(findings[:3])
    return {"status": "answered" if pairs else "limited", "headline": headline, "metrics": {"pairs": pairs},
            "findings": findings, "contributions": contributions, "approximate": approximate}


def _merge_drivers(goal: str, clones: list[dict[str, Any]]) -> dict[str, Any]:
    """What moves with the outcome, from every clone's rows at once.

    Each clone's ranking is of her shard only; a shard of a sorted file may
    hold none of the outcome. The merged statistics (co-moments, per-group
    moments, contingency counts) give the ranking one pass over all the
    rows gives, whatever the shards held."""
    stats = merge_driver_partials([c["partials"] for c in clones])
    described = _describe_target({"column": stats["target"], "value": stats["target_value"]})
    outcome = stats["outcome"]
    metrics: dict[str, Any] = {"target": stats["target"], "target_value": stats["target_value"],
                               "target_expr": stats["target_expr"]}

    def held(c: dict[str, Any]) -> str:
        o = c["partials"]["outcome"]
        if outcome["binary"]:
            return f"{described} in {o['positives']:,} of {o['positives'] + o['negatives']:,} rows"
        return f"{c['partials']['rows']:,} rows"

    contributions = [_contribution(c, held(c)) for c in clones]
    if outcome["binary"]:
        metrics.update(positives=outcome["positives"], negatives=outcome["negatives"])
        if min(outcome["positives"], outcome["negatives"]) < MIN_OUTCOME_ROWS:
            finding = (f"{described} holds {outcome['positives']:,} of "
                       f"{outcome['positives'] + outcome['negatives']:,} rows across the clones: too few on "
                       "one side to measure what predicts it."
                       + (f" Name the outcome value you mean, e.g. 'what predicts {stats['target']} = <value>'."
                          if stats["target_value"] is not None else ""))
            return {"status": "limited", "headline": f"The fleet could not rank what moves with {described} "
                                                     f"for “{goal}”: {finding}",
                    "metrics": metrics, "findings": [finding], "contributions": contributions,
                    "approximate": []}
    drivers, skipped = rank_drivers(stats)
    metrics.update(drivers=[{k: v for k, v in d.items() if k != "check"} for d in drivers[:10]],
                   skipped=skipped)
    findings, shown = driver_findings(described, drivers, keep=goal_named_columns(goal, stats["columns"]))
    # The columns the findings name, as analyze_goal records them, so the
    # merged answer's table (avaloka.drivers_answer) is the one-frame table.
    metrics["shown"] = [d["column"] for d in shown]
    approximate = []
    disagree = [n for n, e in stats["columns"].items() if e.get("disagree")]
    if disagree:
        approximate.append({"statistic": "driver ranking",
                            "reason": f"{len(disagree)} column{'s' if len(disagree) != 1 else ''} typed "
                                      "differently by the clones (" + ", ".join(disagree[:5])
                                      + ") left out"})
    if shown:
        top = ", ".join(f"{d['column']} ({_SYMBOL[d['measure']]}={_signed(d)})" for d in shown)
        headline = (f"Answer to “{goal}”: what moves with {described}, merged from {len(clones)} clones "
                    f"(association, not cause) — {top}.")
    else:
        headline = f"The fleet found no column to measure against {described} for “{goal}”. " + findings[0]
    return {"status": "answered" if shown else "limited", "headline": headline, "metrics": metrics,
            "findings": findings, "contributions": contributions, "approximate": approximate}


def answer_markdown(answer: dict[str, Any]) -> str:
    """The merged answer as the opening section of a report."""
    lines = ["## Answer to the goal", "", answer["headline"], ""]
    if answer.get("table"):
        from avaloka.drivers_answer import table_markdown

        lines += table_markdown(answer["table"]) + [""]
    if answer.get("findings"):
        lines += [f"- {f}" for f in answer["findings"]] + [""]
    if answer.get("contributions"):
        lines += ["### What each clone contributed", ""]
        lines += [f"- {c['finding']}" for c in answer["contributions"]] + [""]
    how = ("Merged from the clones' sufficient statistics: sums and counts added, means weighted by "
           "their rows." if answer.get("merge") == "sufficient statistics" else
           "Not merged: each clone's own finding is listed above.")
    if answer.get("sampled"):
        how += " At least one clone answered from a sample of her shard, so these are a sample's figures."
    lines += [f"_{how}_" + (" Exact." if answer.get("exact") and not answer.get("sampled") else ""), ""]
    for a in answer.get("approximate") or []:
        lines.append(f"- **Approximate — {a['statistic']}:** {a['reason']}.")
    if answer.get("approximate"):
        lines.append("")
    return "\n".join(lines)
