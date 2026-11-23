"""Domain templates and the chart builders behind them (§4.3).

A template is a list of panels; each panel names the roles it needs. A panel
whose roles are missing is dropped and becomes **one**
``missing_template_data`` caveat naming the role ("needs an origination
date"), never a column. 2.0 ships ``retail_performance``,
``fintech_credit`` and ``generic`` (the recommender alone);
``adtech_performance`` and ``marketing_performance`` are the second cut and
fall back to ``generic`` here.

Chart builders turn a recommender :class:`~avaloka.outcomes.recommend.Candidate`
into the §4.1 chart object: an ECharts ``option`` (JSON only; no string holds
``<``, ``=>`` or ``function``), the ``data_table`` twin, and ``alt_text``
written from the data. No model call anywhere in this module.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from avaloka.domains import normalise
from avaloka.outcomes import geo as _geo
from avaloka.outcomes.recommend import Candidate, is_rate
from avaloka.outcomes.semantics import ColumnSemantics, by_role

TABLE_MAX_ROWS = 500
TABLE_MAX_COLS = 12
LIFT_MIN_PAIRS = 30
LIFT_TOP = 15

# Colour-blind-safe categorical palette (Okabe-Ito), AA against white.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]


# ── option hygiene ────────────────────────────────────────────────────────

_FORBIDDEN = re.compile(r"function|=>")


def safe_label(v: Any) -> str:
    """A data label as it may appear inside an ECharts option.

    The spec forbids ``<``, ``=>`` and ``function`` in option strings (§4.1);
    the label is shown with a look-alike instead, so a category named
    ``<script>`` draws as text. The data table keeps the original, escaped at
    render.
    """
    s = str(v).replace("<", "‹").replace(">", "›")
    return _FORBIDDEN.sub(lambda m: m.group(0)[0] + "​" + m.group(0)[1:], s)


def _num(v: Any, nd: int = 6) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, nd)


def _base(**extra: Any) -> dict[str, Any]:
    opt = {"aria": {"enabled": True, "decal": {"show": True}}, "color": PALETTE,
           "tooltip": {"trigger": "item", "confine": True}, "animation": True}
    opt.update(extra)
    return opt


def _table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> dict[str, Any]:
    cols = [str(c) for c in columns][:TABLE_MAX_COLS]
    rows = [list(r)[:TABLE_MAX_COLS] for r in rows]
    # Cut rows are reported by the assembler as a caveat ("first 500 rows"):
    # the spec's data_table is closed to anything but columns and rows.
    return {"columns": cols, "rows": [[_cell(v) for v in r] for r in rows[:TABLE_MAX_ROWS]],
            "_cut": max(0, len(rows) - TABLE_MAX_ROWS)}


def _cell(v: Any) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return _num(v)
    if v is None:
        return None
    return str(v)


def _label(name: str) -> str:
    return normalise(name) or str(name)


def _agg(m: str, sem: Mapping[str, ColumnSemantics]) -> str:
    return "mean" if is_rate(m, sem[m]) else "sum"


def _period(s: pd.Series, grain: str | None) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s) or s.dtype == object:
        ts = pd.to_datetime(s, errors="coerce")
        if ts.notna().mean() > 0.9:
            fmt = {"year": "%Y", "quarter": None, "month": "%Y-%m"}.get(grain or "", "%Y-%m-%d")
            if grain == "quarter":
                return ts.dt.to_period("Q").astype(str)
            return ts.dt.strftime(fmt)
    return s.astype(str)


def _alt(kind: str, what: str, extent: str, fact: str) -> str:
    text = f"{kind} of {what}{extent}. {fact}".strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) < 20:
        text = (text + " Data table below.").strip()
    return text[:400]


# ── builders: one per chart_kind ──────────────────────────────────────────


def _xy_series(df, x, y, series, agg, grain=None):
    keys = [x] + ([series] if series else [])
    frame = df[keys + [y]].dropna(subset=[x, y]).copy()
    frame["_x"] = _period(frame[x], grain) if grain is not None else frame[x].astype(str)
    g = frame.groupby(["_x"] + ([series] if series else []))[y].agg(agg).reset_index()
    xs = sorted(g["_x"].unique())
    return g, xs


def build_trend(c: Candidate, df, sem, *, title: str) -> dict:
    x, y, series = c.encoding["x"], c.encoding["y"], c.encoding.get("series")
    agg = _agg(y, sem)
    g, xs = _xy_series(df, x, y, series, agg, sem[x].grain or "")
    kind = c.chart_kind
    etype = "bar" if kind == "bar" else "line"
    groups = sorted(g[series].unique(), key=str) if series else [None]
    out_series, rows = [], []
    for grp in groups:
        sub = g if grp is None else g[g[series] == grp]
        lookup = dict(zip(sub["_x"], sub[y]))
        s = {"type": etype, "name": safe_label(grp if grp is not None else _label(y)),
             "data": [_num(lookup.get(v)) for v in xs]}
        if kind == "area":
            s.update({"stack": "total", "areaStyle": {}})
        out_series.append(s)
        rows += [[v] + ([grp] if series else []) + [_num(lookup.get(v))] for v in xs]
    option = _base(xAxis={"type": "category", "data": [safe_label(v) for v in xs], "name": _label(x)},
                   yAxis={"type": "value", "name": _label(y)}, series=out_series,
                   legend={"show": bool(series)}, tooltip={"trigger": "axis", "confine": True})
    total = g.groupby("_x")[y].agg("sum" if agg == "sum" else "mean")
    first, last = total.iloc[0], total.iloc[-1]
    fact = f"It moves from {first:.4g} in {xs[0]} to {last:.4g} in {xs[-1]}."
    return {"chart_kind": kind, "question_type": "trend", "title": title, "option": option,
            "alt_text": _alt(f"{kind.capitalize()} chart", _label(y),
                             f" by {_label(x)} over {len(xs)} periods" + (f", one series per {_label(series)}" if series else ""),
                             fact),
            "data_table": _table([x] + ([series] if series else []) + [y], rows),
            "facts": {"first": float(first), "last": float(last), "first_label": xs[0], "last_label": xs[-1],
                      "measure": y}}


def build_bar(c: Candidate, df, sem, *, title: str, question_type: str) -> dict:
    x, y = c.encoding["x"], c.encoding["y"]
    agg = _agg(y, sem)
    g = df[[x, y]].dropna().groupby(x)[y].agg(agg).sort_values(ascending=False)
    top = c.encoding.get("top")
    labels, values = list(g.index), list(g.values)
    if top and len(labels) > top:
        rest = values[top:]
        labels, values = labels[:top], values[:top]
        if c.encoding.get("other") and agg == "sum":
            labels.append("other")
            values.append(float(np.sum(rest)))
    option = _base(xAxis={"type": "category", "data": [safe_label(v) for v in labels], "name": _label(x),
                          "axisLabel": {"interval": 0, "rotate": 30 if len(labels) > 6 else 0}},
                   yAxis={"type": "value", "name": _label(y)},
                   series=[{"type": "bar", "name": safe_label(_label(y)), "data": [_num(v) for v in values]}])
    lead, lead_v = labels[0], values[0]
    return {"chart_kind": "bar", "question_type": question_type, "title": title, "option": option,
            "alt_text": _alt("Bar chart", _label(y), f" for {len(labels)} {_label(x)} values, largest first",
                             f"{lead} leads at {lead_v:.4g}."),
            "data_table": _table([x, y], [[a, _num(b)] for a, b in zip(labels, values)]),
            "facts": {"leader": str(lead), "leader_value": float(lead_v), "measure": y}}


def build_stacked_bar(c: Candidate, df, sem, *, title: str) -> dict:
    x, y, series = c.encoding["x"], c.encoding["y"], c.encoding["series"]
    g = df[[x, series, y]].dropna().groupby([x, series])[y].sum().reset_index()
    xs = sorted(g[x].astype(str).unique())
    groups = sorted(g[series].astype(str).unique())
    g[x], g[series] = g[x].astype(str), g[series].astype(str)
    ser, rows = [], []
    for grp in groups:
        sub = dict(zip(g[g[series] == grp][x], g[g[series] == grp][y]))
        ser.append({"type": "bar", "stack": "total", "name": safe_label(grp), "data": [_num(sub.get(v)) for v in xs]})
        rows += [[v, grp, _num(sub.get(v))] for v in xs]
    option = _base(xAxis={"type": "category", "data": [safe_label(v) for v in xs]}, yAxis={"type": "value"},
                   series=ser, legend={"show": True})
    return {"chart_kind": "stacked_bar", "question_type": "composition", "title": title, "option": option,
            "alt_text": _alt("Stacked bar chart", _label(y), f" by {_label(x)}, split by {_label(series)}",
                             f"{len(groups)} parts per bar."),
            "data_table": _table([x, series, y], rows), "facts": {"measure": y}}


def build_pie(c: Candidate, df, sem, *, title: str) -> dict:
    d, m = c.encoding["category"], c.encoding["value"]
    g = df[[d, m]].dropna().groupby(d)[m].sum().sort_values(ascending=False)
    total = float(g.sum())
    option = _base(series=[{"type": "pie", "radius": ["35%", "70%"],
                            "data": [{"name": safe_label(k), "value": _num(v)} for k, v in g.items()]}],
                   legend={"show": True})
    share = float(g.iloc[0]) / total if total else 0.0
    return {"chart_kind": "pie", "question_type": "composition", "title": title, "option": option,
            "alt_text": _alt("Pie chart", _label(m), f" split across {len(g)} {_label(d)} values",
                             f"{g.index[0]} is the largest part."),
            "data_table": _table([d, m], [[k, _num(v)] for k, v in g.items()]),
            "facts": {"leader": str(g.index[0]), "leader_share": share, "measure": m}}


def build_histogram(c: Candidate, df, sem, *, title: str) -> dict:
    m = c.encoding["x"]
    vals = df[m].dropna().astype(float).to_numpy()
    counts, edges = np.histogram(vals, bins=min(30, max(5, int(np.sqrt(len(vals))))))
    labels = [f"{edges[i]:.4g} to {edges[i + 1]:.4g}" for i in range(len(counts))]
    option = _base(xAxis={"type": "category", "data": labels, "name": _label(m)}, yAxis={"type": "value", "name": "rows"},
                   series=[{"type": "bar", "barCategoryGap": "2%", "data": [int(v) for v in counts]}])
    return {"chart_kind": "histogram", "question_type": "distribution", "title": title, "option": option,
            "alt_text": _alt("Histogram", _label(m), f" over {len(vals)} rows in {len(counts)} bins",
                             f"Values run from {vals.min():.4g} to {vals.max():.4g}; median {np.median(vals):.4g}."),
            "data_table": _table(["bin", "rows"], [[a, int(b)] for a, b in zip(labels, counts)]),
            "facts": {"median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max()), "measure": m}}


def build_boxplot(c: Candidate, df, sem, *, title: str) -> dict:
    m, d = c.encoding["y"], c.encoding["by"]
    groups = sorted(df[d].dropna().astype(str).unique())
    stats = []
    for grp in groups:
        v = df.loc[df[d].astype(str) == grp, m].dropna().astype(float)
        q = np.percentile(v, [0, 25, 50, 75, 100]) if len(v) else [0] * 5
        stats.append([_num(x) for x in q])
    option = _base(xAxis={"type": "category", "data": [safe_label(g) for g in groups]}, yAxis={"type": "value", "name": _label(m)},
                   series=[{"type": "boxplot", "data": stats}])
    return {"chart_kind": "boxplot", "question_type": "distribution", "title": title, "option": option,
            "alt_text": _alt("Box plot", _label(m), f" for each of {len(groups)} {_label(d)} values",
                             "Each box spans the middle half of the values."),
            "data_table": _table([d, "min", "q1", "median", "q3", "max"], [[g] + s for g, s in zip(groups, stats)]),
            "facts": {"measure": m}}


def build_scatter(c: Candidate, df, sem, *, title: str) -> dict:
    x, y, n = c.encoding["x"], c.encoding["y"], c.encoding["sample"]
    sub = df[[x, y]].dropna()
    if len(sub) > n:
        sub = sub.sample(n=n, random_state=0)
    pts = [[_num(a), _num(b)] for a, b in sub.itertuples(index=False)]
    r = float(sub[x].corr(sub[y])) if len(sub) > 2 else float("nan")
    option = _base(xAxis={"type": "value", "name": _label(x), "scale": True},
                   yAxis={"type": "value", "name": _label(y), "scale": True},
                   series=[{"type": "scatter", "symbolSize": 5, "data": pts}])
    return {"chart_kind": "scatter", "question_type": "relationship", "title": title, "option": option,
            "alt_text": _alt("Scatter plot", f"{_label(y)} against {_label(x)}", f", {len(pts)} points",
                             f"Correlation {r:.2f}." if not math.isnan(r) else ""),
            "data_table": _table([x, y], pts), "facts": {"r": r}}


def build_heatmap(c: Candidate, df, sem, *, title: str, question_type: str) -> dict:
    if c.encoding.get("matrix"):
        a, b, m = c.encoding["source"], c.encoding["target"], c.encoding["value"]
        g = df[[a, b, m]].dropna().groupby([a, b])[m].sum()
        xs = sorted({str(k[0]) for k in g.index})
        ys = sorted({str(k[1]) for k in g.index})
        data = [[xs.index(str(i)), ys.index(str(j)), _num(v)] for (i, j), v in g.items()]
        rows = [[i, j, _num(v)] for (i, j), v in g.items()]
        cols, what = [a, b, m], f"{_label(m)} from {_label(a)} to {_label(b)}"
    else:
        a, b = c.encoding["x"], c.encoding["y"]
        sub = df[[a, b]].dropna().astype(float)
        h, xe, ye = np.histogram2d(sub[a], sub[b], bins=12)
        xs = [f"{v:.3g}" for v in xe[:-1]]
        ys = [f"{v:.3g}" for v in ye[:-1]]
        data = [[i, j, int(h[i, j])] for i in range(h.shape[0]) for j in range(h.shape[1])]
        rows = [[xs[i], ys[j], v] for i, j, v in data]
        cols, what = [f"{a} bin", f"{b} bin", "rows"], f"rows binned by {_label(a)} and {_label(b)}"
    vmax = max([d[2] or 0 for d in data] or [1])
    option = _base(xAxis={"type": "category", "data": [safe_label(v) for v in xs]},
                   yAxis={"type": "category", "data": [safe_label(v) for v in ys]},
                   visualMap={"min": 0, "max": vmax, "calculable": True, "orient": "horizontal", "left": "center", "bottom": 0},
                   series=[{"type": "heatmap", "data": data}])
    return {"chart_kind": "heatmap", "question_type": question_type, "title": title, "option": option,
            "alt_text": _alt("Heatmap", what, f", {len(xs)} by {len(ys)} cells", "Darker cells hold more."),
            "data_table": _table(cols, rows), "facts": {}}


def build_funnel(c: Candidate, df, sem, *, title: str) -> dict:
    if "stages" in c.encoding:
        stages = c.encoding["stages"]
        vals = [float(df[s].sum()) for s in stages]
        names = [_label(s) for s in stages]
    else:
        d, m = c.encoding["stage"], c.encoding["value"]
        g = df.groupby(d)[m].sum().sort_values(ascending=False) if m else df[d].value_counts()
        names, vals = [str(k) for k in g.index], [float(v) for v in g.values]
    conv = [None] + [b / a if a else None for a, b in zip(vals, vals[1:])]
    if c.chart_kind == "funnel":
        option = _base(series=[{"type": "funnel", "sort": "none", "left": "10%", "width": "80%",
                                "data": [{"name": safe_label(n), "value": _num(v)} for n, v in zip(names, vals)]}])
    else:
        option = _base(xAxis={"type": "category", "data": [safe_label(n) for n in names]}, yAxis={"type": "value"},
                       series=[{"type": "bar", "data": [_num(v) for v in vals]}])
    end = vals[-1] / vals[0] if vals[0] else 0.0
    return {"chart_kind": c.chart_kind, "question_type": "funnel", "title": title, "option": option,
            "alt_text": _alt("Funnel", " to ".join(names), "", f"The last stage keeps {end:.1%} of the first."),
            "data_table": _table(["stage", "value", "step_conversion"],
                                 [[n, _num(v), _num(cv, 4)] for n, v, cv in zip(names, vals, conv)]),
            "facts": {"end_to_end": end}}


def build_cohort(c: Candidate, df, sem, *, title: str) -> dict:
    t, age, m = c.encoding["cohort"], c.encoding["age"], c.encoding["value"]
    frame = df[[t, age, m]].dropna().copy()
    frame["_c"] = _period(frame[t], sem[t].grain or "")
    agg = _agg(m, sem)
    g = frame.groupby(["_c", age])[m].agg(agg).reset_index()
    cohorts = sorted(g["_c"].unique())
    ages = sorted(g[age].unique())
    rows = [[r["_c"], _num(r[age]), _num(r[m])] for _, r in g.iterrows()]
    if c.chart_kind == "vintage_lines":
        ser = []
        for co in cohorts:
            sub = g[g["_c"] == co]
            ser.append({"type": "line", "name": safe_label(co), "data": [[_num(a), _num(v)] for a, v in zip(sub[age], sub[m])]})
        option = _base(xAxis={"type": "value", "name": _label(age)}, yAxis={"type": "value", "name": _label(m)},
                       series=ser, legend={"show": True}, tooltip={"trigger": "axis", "confine": True})
        kind_txt = "Vintage lines"
    else:
        data = [[ages.index(r[age]), cohorts.index(r["_c"]), _num(r[m])] for _, r in g.iterrows()]
        option = _base(xAxis={"type": "category", "data": [safe_label(a) for a in ages], "name": _label(age)},
                       yAxis={"type": "category", "data": [safe_label(co) for co in cohorts]},
                       visualMap={"min": 0, "max": max([d[2] or 0 for d in data] or [1]), "calculable": True,
                                  "orient": "horizontal", "left": "center", "bottom": 0},
                       series=[{"type": "heatmap", "data": data}])
        kind_txt = "Cohort heatmap"
    return {"chart_kind": c.chart_kind, "question_type": "cohort", "title": title, "option": option,
            "alt_text": _alt(kind_txt, _label(m), f" for {len(cohorts)} cohorts by {_label(age)}",
                             f"Cohorts run from {cohorts[0]} to {cohorts[-1]}."),
            "data_table": _table(["cohort", age, m], rows), "facts": {"cohorts": len(cohorts)}}


def build_choropleth(c: Candidate, df, sem, *, title: str, k_floor: int = _geo.K_FLOOR) -> dict:
    g, m, kind = c.encoding["region"], c.encoding.get("value"), c.encoding["geo_kind"]
    t = _geo.region_table(df, g, kind, m, agg=_agg(m, sem) if m else "sum", k=k_floor)
    asset = t.asset
    name_prop = "iso_a3" if kind == "country" else "usps"
    vals = [r["value"] for r in t.rows]
    option = _base(
        visualMap={"min": _num(min(vals)), "max": _num(max(vals)), "calculable": True, "orient": "horizontal",
                   "left": "center", "bottom": 0, "inRange": {"color": ["#deebf7", "#08519c"]}},
        series=[{"type": "map", "map": asset, "nameProperty": name_prop, "roam": True,
                 "itemStyle": {"areaColor": "#eeeeee", "borderColor": "#999999"},
                 "data": [{"name": r["key"], "value": _num(r["value"])} for r in t.rows]}])
    top = max(t.rows, key=lambda r: r["value"])
    sup = f" {t.suppressed} regions with fewer than {k_floor} rows are not shown." if t.suppressed else ""
    chart = {"chart_kind": "choropleth", "question_type": "geo", "title": title, "option": option,
             "alt_text": _alt("Map", _label(m) if m else "rows", f" by {'country' if kind == 'country' else 'US state'}",
                              f"{top['name']} is highest.{sup}"),
             "data_table": _table(["region", m or "rows"], [[r["name"], _num(r["value"])] for r in
                                                            sorted(t.rows, key=lambda r: -r["value"])]),
             "geo_kind": kind, "asset": asset, "suppressed_regions": t.suppressed, "k_floor": k_floor,
             "facts": {"leader": top["name"], "leader_value": top["value"], "unresolved": t.unresolved}}
    return chart


def build_point_map(c: Candidate, df, sem, *, title: str, k_floor: int = _geo.K_FLOOR) -> dict:
    lat, lon, m, binned = c.encoding["lat"], c.encoding["lon"], c.encoding.get("value"), c.encoding.get("binned")
    pts = _geo.point_table(df, lat, lon, m, pii=bool(binned), k=k_floor)
    option = _base(geo={"map": "world_110m", "roam": True, "itemStyle": {"areaColor": "#eeeeee", "borderColor": "#999999"}},
                   series=[{"type": "scatter", "coordinateSystem": "geo", "symbolSize": 6,
                            "data": [[p["lon"], p["lat"], _num(p["value"])] for p in pts.points]}])
    sup = f" {pts.suppressed} cells with fewer than {k_floor} rows are not shown." if pts.suppressed else ""
    return {"chart_kind": "point_map", "question_type": "geo", "title": title, "option": option,
            "alt_text": _alt("Point map", _label(m) if m else "rows",
                             f", {len(pts.points)} {'cells of about 5 km' if pts.binned else 'points'}", sup or "Ranked table below."),
            "data_table": _table(["lat", "lon", m or "rows"], [[p["lat"], p["lon"], _num(p["value"])] for p in pts.points]),
            "geo_kind": "latlon", "asset": "world_110m", "suppressed_regions": pts.suppressed, "k_floor": k_floor,
            "facts": {}}


def build_sankey(c: Candidate, df, sem, *, title: str) -> dict:
    a, b, m = c.encoding["source"], c.encoding["target"], c.encoding["value"]
    g = df[[a, b, m]].dropna().groupby([a, b])[m].sum()
    nodes = sorted({str(x) for pair in g.index for x in pair})
    option = _base(series=[{"type": "sankey", "emphasis": {"focus": "adjacency"},
                            "data": [{"name": safe_label(n)} for n in nodes],
                            "links": [{"source": safe_label(i), "target": safe_label(j), "value": _num(v)}
                                      for (i, j), v in g.items()]}])
    return {"chart_kind": "sankey", "question_type": "flow", "title": title, "option": option,
            "alt_text": _alt("Sankey diagram", f"{_label(m)} from {_label(a)} to {_label(b)}",
                             f", {len(nodes)} nodes", "Band width is proportional to the value."),
            "data_table": _table([a, b, m], [[i, j, _num(v)] for (i, j), v in g.items()]), "facts": {}}


def build_waterfall(steps: Sequence[tuple[str, float]], *, start: tuple[str, float], title: str,
                    measure_label: str, question_type: str = "bridge") -> dict:
    """A bridge from ``start`` through signed ``steps`` to the end total."""
    labels = [start[0]] + [s[0] for s in steps] + ["end"]
    base, up, down = [0.0], [start[1]], [0.0]
    running = start[1]
    for _, v in steps:
        lo = running + min(v, 0)
        base.append(lo)
        up.append(max(v, 0))
        down.append(-min(v, 0))
        running += v
    base.append(0.0)
    up.append(running)
    down.append(0.0)
    option = _base(xAxis={"type": "category", "data": [safe_label(x) for x in labels]}, yAxis={"type": "value"},
                   tooltip={"trigger": "axis", "confine": True},
                   series=[{"type": "bar", "stack": "w", "itemStyle": {"color": "transparent"}, "silent": True,
                            "data": [_num(v) for v in base]},
                           {"type": "bar", "stack": "w", "name": "increase", "data": [_num(v) for v in up]},
                           {"type": "bar", "stack": "w", "name": "decrease", "data": [_num(v) for v in down]}])
    big = max(steps, key=lambda s: abs(s[1])) if steps else ("none", 0.0)
    return {"chart_kind": "waterfall", "question_type": question_type, "title": title, "option": option,
            "alt_text": _alt("Waterfall", measure_label, f" from {start[0]} to end through {len(steps)} steps",
                             f"{big[0]} moves it most."),
            "data_table": _table(["component", "change"], [[start[0], _num(start[1])]] + [[s, _num(v)] for s, v in steps]
                                 + [["end", _num(running)]]),
            "facts": {"start": start[1], "end": running, "largest": big[0]}}


def build_chart(c: Candidate, question_type: str, df: pd.DataFrame, sem: Mapping[str, ColumnSemantics], *,
                title: str, k_floor: int = _geo.K_FLOOR) -> dict:
    k = c.chart_kind
    if question_type == "trend":
        return build_trend(c, df, sem, title=title)
    if k == "bar" and "x" in c.encoding:
        return build_bar(c, df, sem, title=title, question_type=question_type)
    if k == "bar":
        return build_funnel(c, df, sem, title=title)
    table = {"stacked_bar": build_stacked_bar, "pie": build_pie, "histogram": build_histogram,
             "boxplot": build_boxplot, "scatter": build_scatter, "funnel": build_funnel,
             "cohort_heatmap": build_cohort, "vintage_lines": build_cohort, "sankey": build_sankey}
    if k in table:
        return table[k](c, df, sem, title=title)
    if k == "heatmap":
        return build_heatmap(c, df, sem, title=title, question_type=question_type)
    if k == "choropleth":
        return build_choropleth(c, df, sem, title=title, k_floor=k_floor)
    if k == "point_map":
        return build_point_map(c, df, sem, title=title, k_floor=k_floor)
    if k == "waterfall":
        e = c.encoding
        periods = sorted(df[e["period"]].dropna().unique())[-2:]
        g = df[df[e["period"]].isin(periods)].groupby([e["period"], e["component"]])[e["value"]].sum().unstack(0).fillna(0)
        start = float(g[periods[0]].sum())
        steps = [(str(i), float(r[periods[1]] - r[periods[0]])) for i, r in g.iterrows()]
        return build_waterfall(steps, start=(str(periods[0]), start), title=title, measure_label=_label(e["value"]))
    raise ValueError(f"no builder for {k}")


# ── price / volume / mix (§4.3) ───────────────────────────────────────────


def price_volume_mix(p0: Sequence[float], q0: Sequence[float], p1: Sequence[float],
                     q1: Sequence[float]) -> dict[str, float]:
    """ΔR = price + volume, and volume = mix + pure volume (§4.3).

    price  = Σ (p1 - p0) q1
    volume = Σ p0 (q1 - q0)
    mix    = volume - p̄0 (Q1 - Q0), with p̄0 the base-period average price
    and ``volume_pure`` = volume - mix. price + volume_pure + mix == ΔR exactly
    (up to float rounding), which the test holds on random data.
    """
    p0, q0, p1, q1 = (np.asarray(a, dtype=float) for a in (p0, q0, p1, q1))
    r0, r1 = float((p0 * q0).sum()), float((p1 * q1).sum())
    price = float(((p1 - p0) * q1).sum())
    volume = float((p0 * (q1 - q0)).sum())
    Q0, Q1 = float(q0.sum()), float(q1.sum())
    pbar0 = r0 / Q0 if Q0 else 0.0
    volume_pure = pbar0 * (Q1 - Q0)
    mix = volume - volume_pure
    return {"r0": r0, "r1": r1, "delta": r1 - r0, "price": price, "volume": volume,
            "volume_pure": volume_pure, "mix": mix}


def basket_lift(orders: pd.DataFrame, order_col: str, product_col: str, *,
                min_pairs: int = LIFT_MIN_PAIRS, top: int = LIFT_TOP) -> list[dict[str, Any]]:
    """lift(A,B) = P(A∧B) / (P(A)·P(B)) over orders, pairs with >= ``min_pairs`` co-orders."""
    lines = orders[[order_col, product_col]].dropna().drop_duplicates()
    lines = pd.DataFrame({"o": lines[order_col].to_numpy(), "p": lines[product_col].astype(str).to_numpy()})
    n = int(lines["o"].nunique())
    if n == 0:
        return []
    single = lines["p"].value_counts()
    # Only products that could reach the floor can be in a pair that does.
    keep = set(single[single >= min_pairs].index)
    lines = lines[lines["p"].isin(keep)]
    pairs = lines.merge(lines, on="o")
    pairs = pairs[pairs["p_x"] < pairs["p_y"]]
    counts = pairs.groupby(["p_x", "p_y"]).size()
    out = [{"a": a, "b": b, "orders": int(c), "lift": (c / n) / ((single[a] / n) * (single[b] / n))}
           for (a, b), c in counts.items() if c >= min_pairs]
    out.sort(key=lambda r: (-r["lift"], r["a"], r["b"]))
    return out[:top]


# ═════════════════════════════════════════════════════════════════════════
# Report assembly: data -> a valid Outcome Report (spec v1.0)
# ═════════════════════════════════════════════════════════════════════════

import hashlib  # noqa: E402

from avaloka.outcomes import semantics as _sem  # noqa: E402
from avaloka.outcomes import spec as _spec  # noqa: E402
from avaloka.outcomes.recommend import recommend as _recommend  # noqa: E402

TEMPLATES = ("retail_performance", "fintech_credit", "generic")
_ANALYST_RE = re.compile(r"\b(model|regression|feature|auc|diagnostics?|distribution|code|notebook)s?\b", re.I)


def choose_template(domain_id: str | None) -> str:
    """§4.3: finance -> fintech_credit, retail -> retail_performance; marketing's
    two templates are the second cut, so it and everything else -> generic."""
    return {"finance": "fintech_credit", "retail": "retail_performance"}.get(domain_id or "", "generic")


def infer_audience(goal: str) -> str:
    """§4D: ``analyst`` when the goal names methods, else ``executive``."""
    return "analyst" if _ANALYST_RE.search(goal or "") else "executive"


class Figures:
    """Figure registry: every number a text slot or KPI shows, with its evidence.

    Each evidence leaf is a ``{value, unit[, currency]}`` object in the
    figure's own unit, so the verifier's match needs no shift.
    """

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.evidence: dict[str, Any] = {}

    def add(self, path: str, value: float, unit: str, decimals: int, currency: str | None = None) -> str:
        fid = f"f{len(self.items) + 1}"
        node = self.evidence
        parts = [p for p in path.split("/") if p]
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        v = float(value)
        leaf: dict[str, Any] = {"value": v, "unit": unit}
        if currency and unit == "currency":
            leaf["currency"] = currency
        node[parts[-1]] = leaf
        fig = {"id": fid, "value": v, "unit": unit, "decimals": decimals, "evidence_ref": "/" + "/".join(parts)}
        if currency and unit == "currency":
            fig["currency"] = currency
        self.items.append(fig)
        return fid


def _slug(s: str) -> str:
    out = re.sub(r"[^a-z0-9_]+", "_", normalise(s)).strip("_") or "x"
    out = out if out[0].isalpha() else "c_" + out
    return out[:32]


def _find(sem: Mapping[str, ColumnSemantics], pattern: str, roles: Sequence[str]) -> str | None:
    rx = re.compile(pattern)
    for n, c in sem.items():
        if c.role in roles and rx.search(normalise(n)):
            return n
    return None


def _label_ok(label: Any) -> str | None:
    """A data label may appear in a caption only when it carries no digits:
    a slot holds numbers only as figures (§3.2 rule 1)."""
    s = str(label)
    return s if not _spec.free_numbers(s) and len(s) <= 40 else None


class _Ctx:
    def __init__(self, df, sem, goal, k_floor, pii_columns, currency):
        self.df, self.sem, self.goal = df, sem, goal
        self.k_floor, self.pii, self.currency = k_floor, pii_columns, currency
        self.figs = Figures()
        self.kpis: list[dict[str, Any]] = []
        self.charts: list[dict[str, Any]] = []
        self.maps: list[dict[str, Any]] = []
        self.captions: dict[str, str] = {}
        self.missing: list[str] = []         # roles, for the one missing_template_data caveat
        self.caveats: list[dict[str, str]] = []
        self.facts: dict[str, Any] = {}
        self.method: list[str] = []
        self.steps: list[dict[str, Any]] = []

    def chart(self, chart: dict, cid: str, caption: str | None = None) -> None:
        chart = dict(chart)
        chart.pop("facts", None)
        cut = chart["data_table"].pop("_cut", 0)
        if cut:
            self.caveats.append({"kind": "other", "text": f"The table under “{chart['title'][:60]}” shows the "
                                                          f"first {TABLE_MAX_ROWS} rows."})
        chart["id"] = cid
        if caption:
            self.captions[cid] = caption
        (self.maps if chart["chart_kind"] in ("choropleth", "point_map") else self.charts).append(chart)

    def need(self, role: str, step_title: str) -> None:
        self.missing.append(role)
        self.steps.append({"id": _slug("need " + role), "title": step_title[:80], "status": "needs_data",
                           "needs": role[:120]})


def _periods(df: pd.DataFrame, t: str, grain: str | None) -> tuple[pd.Series, str]:
    ts = pd.to_datetime(df[t], errors="coerce") if not pd.api.types.is_integer_dtype(df[t]) else None
    if ts is None:
        return df[t].astype(int).astype(str), "year"
    years = ts.dt.year.dropna().unique()
    if len(years) >= 2:
        return ts.dt.year.astype("Int64").astype(str), "year"
    g = grain if grain in ("month", "quarter", "week", "day") else "month"
    freq = {"month": "M", "quarter": "Q", "week": "W", "day": "D"}[g]
    return ts.dt.to_period(freq).astype(str), g


# ── retail_performance ────────────────────────────────────────────────────

def _retail(ctx: _Ctx) -> None:
    df, sem, figs, cur = ctx.df, ctx.sem, ctx.figs, ctx.currency
    meas = ("measure", "target")
    price = _find(sem, r"\b(unit )?price\b", meas)
    qty = _find(sem, r"\b(qty|quantity|units?|units sold)\b", meas)
    received = _find(sem, r"\b(received|stock in|units received|inventory in)\b", meas)
    if received == qty:
        qty = _find({k: v for k, v in sem.items() if k != received}, r"\b(qty|quantity|units?|units sold)\b", meas)
    sales = _find(sem, r"\b(net sales|sales|revenue|amount|gmv)\b", meas)
    product = _find(sem, r"\b(sku|product|item)\b", ("id", "dimension", "high_card_dimension"))
    order = _find(sem, r"\border\b", ("id",))
    category = _find(sem, r"\b(category|department|dept|class)\b", ("dimension",))
    times = by_role(sem, "time")
    t = times[0] if times else None

    if sales is not None:
        revenue = df[sales].astype(float)
    elif price is not None and qty is not None:
        revenue = df[price].astype(float) * df[qty].astype(float)
    else:
        revenue = None
        ctx.need("a sales amount, or a price and a quantity", "Add sales amounts or prices and quantities")

    if revenue is not None:
        f_sales = figs.add("/kpi/net_sales/value", revenue.sum(), "currency", 1, cur)
        kpi = {"id": "net_sales", "label": "Net sales", "value": f_sales, "direction_good": "up"}
        ctx.facts["net_sales"] = f_sales
        if t is not None:
            per, grain = _periods(df, t, sem[t].grain)
            by = revenue.groupby(per).sum().sort_index()
            if len(by) >= 2 and by.iloc[-2] != 0:
                pct = (by.iloc[-1] / by.iloc[-2] - 1) * 100
                f_d = figs.add("/kpi/net_sales/delta_pct", pct, "percent", 1)
                kpi.update({"delta": f_d, "delta_kind": "pct",
                            "baseline_label": "prior year" if grain == "year" else f"prior {grain}"})
                ctx.facts["net_sales_delta"] = f_d
        else:
            ctx.need("an order date", "Add order dates to compare periods")
        ctx.kpis.append(kpi)
        ctx.method.append("net sales summed over order lines")
    if qty is not None:
        ctx.kpis.append({"id": "units", "label": "Units", "direction_good": "up",
                         "value": figs.add("/kpi/units/value", df[qty].sum(), "count", 0)})
    if revenue is not None and order is not None and df[order].nunique():
        ctx.kpis.append({"id": "avg_basket", "label": "Average basket", "direction_good": "up",
                         "value": figs.add("/kpi/avg_basket/value", revenue.sum() / df[order].nunique(),
                                           "currency", 2, cur)})
    if qty is not None and received is not None and df[received].sum():
        st = df[qty].sum() / df[received].sum() * 100
        ctx.kpis.append({"id": "sell_through", "label": "Sell-through", "direction_good": "up",
                         "value": figs.add("/kpi/sell_through/value", st, "percent", 0)})
    elif received is None:
        ctx.need("units received", "Add units received to measure sell-through")

    # Map: geo + sales.
    geo_cols = by_role(sem, "geo")
    if geo_cols and revenue is not None:
        frame = df.copy()
        frame["_net_sales"] = revenue
        fsem = dict(sem)
        fsem["_net_sales"] = ColumnSemantics(name="_net_sales", role="measure", unit="currency")
        fsem = {k: v for k, v in fsem.items() if v.role != "measure" or k == "_net_sales"}
        rec = _recommend("map", frame, fsem, question="geo", k_floor=ctx.k_floor, pii_columns=ctx.pii)
        if rec.candidates:
            ch = build_chart(rec.candidates[0], "geo", frame, fsem, title="Net sales by region", k_floor=ctx.k_floor)
            ch["data_table"]["columns"] = [c if c != "_net_sales" else "net sales" for c in ch["data_table"]["columns"]]
            lead = _label_ok(ch["facts"].get("leader")) if ch["facts"] else None
            cap = None
            if lead and "leader_value" in ch["facts"]:
                f = figs.add("/map/leader", ch["facts"]["leader_value"], "currency", 1, cur)
                cap = f"{lead} has the highest net sales, at {{{f}}}."
            ctx.chart(ch, "region_map", cap)
            ctx.method.append("net sales by region")
            if ch["suppressed_regions"]:
                ctx.caveats.append({"kind": "suppressed_geo",
                                    "text": f"Regions with fewer than {ctx.k_floor} rows are not shown."})
        else:
            ctx.caveats.append({"kind": "suppressed_geo", "text": "No region had enough rows to map."})
    elif not geo_cols:
        ctx.need("a store or region location", "Add store or region locations to map sales")

    # Bridge: two periods, price and quantity per product.
    if t is not None and price is not None and qty is not None and product is not None:
        per, grain = _periods(df, t, sem[t].grain)
        ps = sorted(per.dropna().unique())
        if len(ps) >= 2:
            p0, p1 = ps[-2], ps[-1]
            lines = df.assign(_p=per, _r=df[price].astype(float) * df[qty].astype(float))
            agg = lines.groupby(["_p", product]).agg(q=(qty, "sum"), r=("_r", "sum"))
            a0 = agg.xs(p0, level=0) if p0 in agg.index.get_level_values(0) else agg.iloc[:0]
            a1 = agg.xs(p1, level=0) if p1 in agg.index.get_level_values(0) else agg.iloc[:0]
            idx = a0.index.union(a1.index)
            a0, a1 = a0.reindex(idx, fill_value=0), a1.reindex(idx, fill_value=0)
            pr0 = (a0["r"] / a0["q"].where(a0["q"] != 0)).fillna(a1["r"] / a1["q"].where(a1["q"] != 0)).fillna(0)
            pr1 = (a1["r"] / a1["q"].where(a1["q"] != 0)).fillna(pr0)
            b = price_volume_mix(pr0, a0["q"], pr1, a1["q"])
            steps = [("price", b["price"]), ("volume", b["volume_pure"]), ("mix", b["mix"])]
            ch = build_waterfall(steps, start=(str(p0), b["r0"]), title="Where the change came from",
                                 measure_label="net sales")
            f_p = figs.add("/bridge/price", b["price"], "currency", 1, cur)
            f_t = figs.add("/bridge/total", b["delta"], "currency", 1, cur)
            ctx.facts.update({"bridge_price": f_p, "bridge_total": f_t})
            ctx.chart(ch, "yoy_bridge", f"Price changes account for {{{f_p}}} of the {{{f_t}}} change.")
            ctx.method.append("price-volume-mix bridge between the last two periods")
            ctx.steps.insert(0, {"id": "bridge", "title": "Explain the change between periods", "status": "done"})
    else:
        ctx.need("two periods with a price and a quantity per product",
                 "Add prices and quantities per product to split the change")

    # Sell-through by category.
    if qty is not None and received is not None and category is not None:
        g = df.groupby(category)[[qty, received]].sum()
        g = g[g[received] > 0]
        st = (g[qty] / g[received] * 100).sort_values(ascending=False)
        option = _base(xAxis={"type": "category", "data": [safe_label(x) for x in st.index]},
                       yAxis={"type": "value", "name": "sell-through %"},
                       series=[{"type": "bar", "data": [_num(v, 2) for v in st.values]}])
        ctx.chart({"chart_kind": "bar", "question_type": "ranking", "title": "Sell-through by category",
                   "option": option,
                   "alt_text": _alt("Bar chart", "sell-through", f" for {len(st)} categories, highest first",
                                    f"{st.index[0]} sells through the most."),
                   "data_table": _table([category, "sell_through_pct"], [[k, _num(v, 2)] for k, v in st.items()])},
                  "sell_through")
    elif received is not None and category is None:
        ctx.need("a product category", "Add product categories to compare sell-through")

    # Basket affinity.
    if order is not None and product is not None:
        pairs = basket_lift(df, order, product)
        if pairs:
            labels = [f"{p['a']} + {p['b']}" for p in pairs]
            option = _base(yAxis={"type": "category", "data": [safe_label(x) for x in labels][::-1]},
                           xAxis={"type": "value", "name": "lift"},
                           series=[{"type": "bar", "data": [_num(p["lift"], 3) for p in pairs][::-1]}])
            ctx.chart({"chart_kind": "bar", "question_type": "relationship", "title": "Products bought together",
                       "option": option,
                       "alt_text": _alt("Bar chart", "basket lift", f" for the top {len(pairs)} product pairs",
                                        "Lift above one means bought together more than chance."),
                       "data_table": _table(["product_a", "product_b", "orders", "lift"],
                                            [[p["a"], p["b"], p["orders"], _num(p["lift"], 3)] for p in pairs])},
                      "affinity")
            ctx.method.append(f"basket lift for pairs in at least {LIFT_MIN_PAIRS} orders")
    else:
        ctx.need("an order id and a product id", "Add order and product ids to find products bought together")

    ctx.steps.append({"id": "act", "title": "Choose where to act first", "status": "needs_decision"})


# ── fintech_credit ────────────────────────────────────────────────────────

def _bucket(dpd: float) -> str:
    return "current" if dpd < 1 else "1-29" if dpd < 30 else "30-59" if dpd < 60 else "60-89" if dpd < 90 else "90+"


_BUCKETS = ["current", "1-29", "30-59", "60-89", "90+"]


def _fintech(ctx: _Ctx) -> None:
    df, sem, figs, cur = ctx.df, ctx.sem, ctx.figs, ctx.currency
    dpd = _find(sem, r"\b(dpd|days past due|days delinquent|delinquen\w*)\b", ("measure", "target"))
    times = by_role(sem, "time")
    orig = next((c for c in times if re.search(r"\b(orig\w*|booked|vintage|issue\w*|open\w*)\b", normalise(c))), None)
    obs = next((c for c in times if c != orig), None)
    account = _find(sem, r"\b(account|loan|contract)\b", ("id",))
    exposure = _find(sem, r"\b(balance|exposure|outstanding|principal)\b", ("measure",))
    cparty = _find(sem, r"\b(merchant|counterparty|borrower|dealer|partner|customer|obligor)\b",
                   ("dimension", "high_card_dimension", "id"))
    chargeoff = _find(sem, r"\b(charge ?offs?|write ?offs?|net loss\w*)\b", ("measure",))

    if dpd is None:
        ctx.need("days past due", "Add days past due to measure delinquency")
    else:
        frame = df
        prior = None
        if obs is not None:
            per = pd.to_datetime(df[obs], errors="coerce")
            last = per.max()
            frame = df[per == last]
            earlier = sorted(per.dropna().unique())
            if len(earlier) >= 2:
                prior = df[per == earlier[-2]]
        for n, kid, label in ((30, "dq30", "30+ DPD rate"), (60, "dq60", "60+ DPD rate"), (90, "dq90", "90+ DPD rate")):
            rate = (frame[dpd] >= n).mean() * 100
            kpi = {"id": kid, "label": label, "direction_good": "down",
                   "value": figs.add(f"/kpi/{kid}/value", rate, "percent", 1)}
            if prior is not None and len(prior):
                kpi.update({"delta": figs.add(f"/kpi/{kid}/delta_pp", rate - (prior[dpd] >= n).mean() * 100, "pp", 1),
                            "delta_kind": "pp", "baseline_label": "prior period"})
            ctx.kpis.append(kpi)
            ctx.facts.setdefault("dq30", kpi["value"])
            if kid == "dq30" and "delta" in kpi:
                ctx.facts["dq30_delta"] = kpi["delta"]
        ctx.method.append("delinquency rates at the latest observation")
    if chargeoff is not None and exposure is not None and df[exposure].sum():
        ctx.kpis.append({"id": "nco_rate", "label": "Net charge-off rate", "direction_good": "down",
                         "value": figs.add("/kpi/nco_rate/value", df[chargeoff].sum() / df[exposure].sum() * 100,
                                           "percent", 2)})
    if exposure is not None:
        ctx.kpis.append({"id": "exposure", "label": "Exposure", "direction_good": "neutral",
                         "value": figs.add("/kpi/exposure/value", df[exposure].sum(), "currency", 1, cur)})
    else:
        ctx.need("an exposure or balance amount", "Add balances to measure exposure")

    # Vintage curves.
    if dpd is not None and orig is not None and obs is not None:
        o = pd.to_datetime(df[orig], errors="coerce")
        b = pd.to_datetime(df[obs], errors="coerce")
        mob = ((b.dt.year - o.dt.year) * 12 + (b.dt.month - o.dt.month)).rename("mob")
        vint = o.dt.to_period("Q").astype(str)
        frame = pd.DataFrame({"vintage": vint, "mob": mob, "bad": (df[dpd] >= 30).astype(float)}).dropna()
        frame = frame[frame["mob"] >= 0]
        g = frame.groupby(["vintage", "mob"])["bad"].agg(["mean", "size"]).reset_index()
        g = g[g["size"] >= ctx.k_floor]
        vintages = sorted(g["vintage"].unique())
        if len(vintages) >= 3:
            series = [{"type": "line", "name": safe_label(v),
                       "data": [[int(m), _num(r * 100, 2)] for m, r in zip(g[g.vintage == v]["mob"], g[g.vintage == v]["mean"])]}
                      for v in vintages]
            option = _base(xAxis={"type": "value", "name": "months on book"}, yAxis={"type": "value", "name": "30+ DPD %"},
                           series=series, legend={"show": True}, tooltip={"trigger": "axis", "confine": True})
            ctx.chart({"chart_kind": "vintage_lines", "question_type": "cohort",
                       "title": "Delinquency by months on book, per vintage", "option": option,
                       "alt_text": _alt("Vintage lines", "the 30+ day delinquency rate",
                                        f", one line per quarterly vintage ({len(vintages)} vintages)",
                                        "Steeper lines are vintages that go bad faster."),
                       "data_table": _table(["vintage", "mob", "dq30_pct"],
                                            [[v, int(m), _num(r * 100, 2)] for v, m, r in zip(g.vintage, g.mob, g["mean"])])},
                      "vintage")
            ctx.method.append("vintage curves from origination quarter and months on book")
            ctx.steps.insert(0, {"id": "vintage", "title": "Compare vintages", "status": "done"})
        else:
            ctx.caveats.append({"kind": "small_n", "text": "Fewer than three vintages had enough loans to draw curves."})
    elif dpd is not None:
        ctx.need("an origination date and an observation date", "Add origination and observation dates for vintages")

    # Roll rates.
    if dpd is not None and account is not None and obs is not None:
        s = df[[account, obs, dpd]].dropna().copy()
        s[obs] = pd.to_datetime(s[obs], errors="coerce")
        s = s.sort_values([account, obs])
        s["b0"] = s[dpd].map(_bucket)
        s["b1"] = s.groupby(account)["b0"].shift(-1)
        t = s.dropna(subset=["b1"])
        if len(t):
            m = pd.crosstab(t["b0"], t["b1"])
            counts = m.sum(axis=1)
            rates = m.div(counts, axis=0) * 100
            rows_, data = [], []
            fb = [b for b in _BUCKETS if b in rates.index]
            tb = [b for b in _BUCKETS if b in rates.columns]
            for i, b0 in enumerate(fb):
                if counts[b0] < ctx.k_floor:
                    continue
                for j, b1 in enumerate(tb):
                    v = _num(rates.loc[b0, b1], 2)
                    data.append([j, i, v])
                    rows_.append([b0, b1, v])
            option = _base(xAxis={"type": "category", "data": tb, "name": "to"},
                           yAxis={"type": "category", "data": fb, "name": "from"},
                           visualMap={"min": 0, "max": 100, "calculable": True, "orient": "horizontal",
                                      "left": "center", "bottom": 0},
                           series=[{"type": "heatmap", "data": data}])
            ctx.chart({"chart_kind": "heatmap", "question_type": "flow", "title": "Roll rates between delinquency buckets",
                       "option": option,
                       "alt_text": _alt("Heatmap", "one-period roll rates between delinquency buckets",
                                        f", {len(fb)} by {len(tb)} buckets", "Off-diagonal cells are accounts that moved."),
                       "data_table": _table(["from", "to", "rate_pct"], rows_)}, "roll")
            ctx.method.append("one-period roll-rate matrix")
    elif dpd is not None:
        ctx.need("an account id and an observation date", "Add account ids and observation dates for roll rates")

    # P&L waterfall.
    pnl = [n for n, c in sem.items() if c.role == "measure" and c.unit == "currency"
           and re.search(r"\b(interest|fees?|income|charge ?offs?|losses|loss|provisions?|funding|costs?|expenses?|recover\w*)\b",
                         normalise(n))]
    if len(pnl) >= 2:
        steps = [(_label(n), float(df[n].sum())) for n in pnl]
        ch = build_waterfall(steps, start=("start", 0.0), title="P&L by component", measure_label="P&L")
        ctx.chart(ch, "pnl")
        net = sum(v for _, v in steps)
        ctx.facts["pnl_net"] = figs.add("/pnl/net", net, "currency", 1, cur)
        ctx.method.append("P&L summed by component")

    # Concentration.
    if cparty is not None and exposure is not None and df[exposure].sum() > 0:
        e = df.groupby(cparty)[exposure].sum().sort_values(ascending=False)
        share = e / e.sum()
        hhi = float((share ** 2).sum())
        ctx.kpis.append({"id": "hhi", "label": "HHI of exposure", "direction_good": "down",
                         "value": figs.add("/kpi/hhi/value", hhi, "ratio", 3)})
        top = share.head(10)
        f_top = figs.add("/concentration/top10", top.sum() * 100, "percent", 0)
        ctx.facts["top10"] = f_top
        lorenz = np.cumsum(share.values) * 100
        option = _base(xAxis={"type": "category", "data": [safe_label(x) for x in top.index]},
                       yAxis=[{"type": "value", "name": "exposure"}, {"type": "value", "name": "cumulative %", "max": 100}],
                       series=[{"type": "bar", "name": "exposure", "data": [_num(v) for v in e.head(10).values]},
                               {"type": "line", "name": "cumulative share", "yAxisIndex": 1,
                                "data": [_num(v, 2) for v in lorenz[:10]]}],
                       legend={"show": True}, tooltip={"trigger": "axis", "confine": True})
        ctx.chart({"chart_kind": "bar", "question_type": "ranking", "title": "Exposure concentration",
                   "option": option,
                   "alt_text": _alt("Bar chart", "exposure", f" for the {len(top)} largest counterparties, with a cumulative line",
                                    "The line shows how much of all exposure the largest hold."),
                   "data_table": _table([cparty, exposure, "cumulative_pct"],
                                        [[k, _num(v), _num(c, 2)] for (k, v), c in zip(e.items(), lorenz)])},
                  "concentration", f"The ten largest counterparties hold {{{f_top}}} of exposure.")
        ctx.method.append("Herfindahl index and top-N exposure")
    elif exposure is not None:
        ctx.need("a counterparty", "Add a counterparty column to measure concentration")
    ctx.steps.append({"id": "review", "title": "Decide whether limits need to change", "status": "needs_decision"})


# ── generic: the recommender alone ────────────────────────────────────────

def _generic(ctx: _Ctx) -> None:
    df, sem, figs, cur = ctx.df, ctx.sem, ctx.figs, ctx.currency
    ctx.kpis.append({"id": "rows", "label": "Rows", "direction_good": "neutral",
                     "value": figs.add("/kpi/rows/value", len(df), "count", 0)})
    measures = [m for m in by_role(sem, "measure", "target") if not is_rate(m, sem[m])]
    if measures:
        m = measures[0]
        unit = "currency" if sem[m].unit == "currency" else "count" if sem[m].unit == "count" else "unitless"
        f = figs.add(f"/kpi/{_slug(m)}/total", df[m].sum(), unit, 0 if unit == "count" else 1, cur)
        ctx.kpis.append({"id": _slug(m)[:32], "label": _label(m)[:40], "direction_good": "neutral", "value": f})
        ctx.facts["total"] = f
        ctx.facts["total_label"] = _label(m)
    seen: set[str] = set()
    for goal in (ctx.goal, "trend", "top", "map", "distribution"):
        rec = _recommend(goal, df, sem, k_floor=ctx.k_floor, pii_columns=ctx.pii)
        if not rec.candidates or rec.question_type in seen:
            continue
        seen.add(rec.question_type)
        ch = build_chart(rec.candidates[0], rec.question_type, df, sem,
                         title=f"{rec.question_type.capitalize()}: {_label(next(iter(rec.candidates[0].encoding.values())))}"[:80],
                         k_floor=ctx.k_floor)
        facts = ch.get("facts") or {}
        cap = None
        if "leader" in facts and _label_ok(facts["leader"]) and "leader_value" in facts:
            mu = sem.get(facts.get("measure", ""))
            unit = "currency" if mu is not None and mu.unit == "currency" else "unitless"
            f = figs.add(f"/charts/{len(seen)}/leader", facts["leader_value"], unit, 1, cur)
            cap = f"{facts['leader']} leads at {{{f}}}."
        elif "first" in facts:
            fa = figs.add(f"/charts/{len(seen)}/first", facts["first"], "unitless", 1)
            fb = figs.add(f"/charts/{len(seen)}/last", facts["last"], "unitless", 1)
            cap = f"{_label(facts['measure']).capitalize()} moves from {{{fa}}} in the first period to {{{fb}}} in the last."
        if ch["chart_kind"] in ("choropleth", "point_map") and ch.get("suppressed_regions"):
            ctx.caveats.append({"kind": "suppressed_geo",
                                "text": f"Regions with fewer than {ctx.k_floor} rows are not shown."})
        ctx.chart(ch, _slug(rec.question_type), cap)
        ctx.method.append(f"{rec.question_type} chart chosen by the rule table")
        if len(seen) >= 3:
            break
    ctx.steps.append({"id": "next", "title": "Pick the question to answer next", "status": "next"})


_BUILDERS = {"retail_performance": _retail, "fintech_credit": _fintech, "generic": _generic}


def _headline(template: str, ctx: _Ctx) -> tuple[str, list[dict[str, str]]]:
    f = ctx.facts
    if template == "retail_performance" and "net_sales" in f:
        text = f"Net sales were {{{f['net_sales']}}}" + (f", a change of {{{f['net_sales_delta']}}} on the prior period." if "net_sales_delta" in f else ".")
        bullets = []
        if "bridge_price" in f:
            bullets.append({"so_what": f"Price changes account for {{{f['bridge_price']}}} of the {{{f['bridge_total']}}} change.",
                            "now_what": "Check whether the price gain holds in the next period."})
        if ctx.maps:
            bullets.append({"so_what": "Sales are not spread evenly across regions.",
                            "now_what": "Look at the strongest and weakest regions first."})
        return text, bullets
    if template == "fintech_credit" and "dq30" in f:
        # "30+ day" would be a raw number in a slot (§3.2 rule 1); the threshold is a figure.
        days = ctx.figs.add("/kpi/dq30/threshold_days", 30, "days", 0)
        text = (f"Accounts {{{days}}} or more past due are {{{f['dq30']}}} of the book"
                + (f", a change of {{{f['dq30_delta']}}} on the prior period." if "dq30_delta" in f else "."))
        bullets = []
        if "top10" in f:
            bullets.append({"so_what": f"The ten largest counterparties hold {{{f['top10']}}} of exposure.",
                            "now_what": "Review concentration limits before adding exposure."})
        if any(c["id"] == "vintage" for c in ctx.charts):
            bullets.append({"so_what": "Vintages differ in how fast they go delinquent.",
                            "now_what": "Compare the newest vintages with the underwriting rules they were booked under."})
        return text, bullets
    if "total" in f:
        return f"{str(f['total_label']).capitalize()[:60]} totals {{{f['total']}}} in this data.", []
    return f"This data has {{{ctx.kpis[0]['value']}}} rows.", []


def report_id_for(analysis_id: str, version: int) -> str:
    return "rpt_" + hashlib.sha256(f"{analysis_id}:{version}".encode()).hexdigest()[:16]


def build_report(df: pd.DataFrame, goal: str, *, analysis: Mapping[str, Any], created_at: str,
                 domain: Mapping[str, Any] | None = None, template: str | None = None,
                 target: str | None = None, slots: Mapping[str, Any] | None = None,
                 audience: str | None = None, k_floor: int = _geo.K_FLOOR,
                 pii_columns: Sequence[str] = (), total_rows: int | None = None,
                 generator_version: str | None = None) -> dict[str, Any]:
    """Build one Outcome Report from a table, deterministically, and validate it.

    ``domain`` is #532's ``classify`` result (computed here when omitted).
    ``slots`` replaces model-written text: ``headline`` (str), ``bullets``
    (list of ``{so_what, now_what}``) and ``captions`` (chart id -> str); each
    may use ``{fN}`` placeholders. No model is called: when ``slots`` is absent
    the deterministic sentences are used.
    """
    from avaloka.domains import classify
    from avaloka.version import __version__

    domain = domain or classify(list(map(str, df.columns)), df.head(20).to_dict("records"))
    template = template or choose_template(domain.get("domain"))
    sem = _sem.infer(df, target=target, domain=domain)
    currency = _sem.currency_code(df, sem)
    ctx = _Ctx(df, sem, goal, k_floor, tuple(pii_columns), currency)
    _BUILDERS[template](ctx)

    if ctx.missing:
        ctx.caveats.insert(0, {"kind": "missing_template_data",
                               "text": ("Some panels are not shown: this data needs "
                                        + "; ".join(dict.fromkeys(ctx.missing)) + ".")[:300]})
    if currency is None and any(f["unit"] == "currency" for f in ctx.figs.items):
        ctx.caveats.append({"kind": "currency_unstated",
                            "text": "Amounts are shown as plain numbers: the data does not state a currency."})
    slots = dict(slots or {})
    text, bullets = _headline(template, ctx)
    captions = {**ctx.captions, **(slots.get("captions") or {})}
    for section in (ctx.charts, ctx.maps):
        for ch in section:
            if captions.get(ch["id"]):
                ch["caption"] = str(captions[ch["id"]])[:200]
    steps = []
    for i, st in enumerate(ctx.steps[:12], 1):
        steps.append({**st, "order": i})
    rows = len(df)
    total = total_rows if total_rows is not None else rows
    report: dict[str, Any] = {
        "schema_version": _spec.SCHEMA_VERSION,
        "report_id": report_id_for(str(analysis["analysis_id"]), int(analysis.get("version", 1))),
        "kind": "analysis",
        "domain": domain.get("domain", "generic"),
        "template": template,
        "audience": audience or infer_audience(goal),
        "created_at": created_at,
        "analysis": {"analysis_id": str(analysis["analysis_id"]), "thread_id": str(analysis.get("thread_id", "")),
                     "turn_id": str(analysis.get("turn_id", "")), "version": int(analysis.get("version", 1)),
                     "parent_version": analysis.get("parent_version"),
                     "dataset_ids": list(analysis.get("dataset_ids", []))[:16]},
        "headline": {"text": str(slots.get("headline") or text)[:160],
                     "bullets": [{"so_what": str(b["so_what"])[:140], "now_what": str(b["now_what"])[:140]}
                                 for b in (slots.get("bullets") or bullets)][:3]},
        "figures": ctx.figs.items,
        "kpis": ctx.kpis[:8],
        "charts": ctx.charts[:12],
        "maps": ctx.maps[:4],
        "method": {"summary": ("; ".join(ctx.method) or "Summary statistics of the table.")[:400].capitalize(),
                   "sample": {"rows_analyzed": rows, "total_rows": total, "is_sampled": rows < total}},
        "caveats": ctx.caveats[:12],
        "steps": steps or [{"id": "review", "order": 1, "title": "Review the result", "status": "next"}],
        "completion": {"state": "partial" if ctx.missing else "answered",
                       **({"missing": [f"Needs {m}."[:140] for m in dict.fromkeys(ctx.missing)][:6]} if ctx.missing else {})},
        "evidence": ctx.figs.evidence,
        "provenance": {"generator": "avaloka", "generator_version": generator_version or __version__,
                       "llm_used": False, "llm_model": None,
                       # Nothing in this branch verifies; the verify branch sets this.
                       "verifier": {"passed": False, "rejections": []}},
    }
    if rows < total:
        report["caveats"].insert(0, {"kind": "sampling", "text": "Figures are computed on a sample of the rows."})
        report["caveats"] = report["caveats"][:12]
    _spec.assert_valid(report)
    return report
