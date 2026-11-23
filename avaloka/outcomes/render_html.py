"""The self-contained HTML export of an Outcome Report (§4.4, §4.7, §4D).

A pure function of the spec: no network, no model, no clock. One file holds
the ECharts runtime, the geo assets the report's maps use, the spec itself as
``<script type="application/json" id="avaloka-report">`` (every ``<``, ``>``
and ``&`` written as a JSON escape, so a caption holding ``</script>`` cannot
end the block) and one init script. A meta CSP allows exactly those two
executable scripts by sha256 and nothing else to load (``connect-src
'none'``, ``default-src 'none'``), so the file is inert under
``<iframe sandbox="allow-scripts">`` and offline.

Every string the template did not write (headline, bullets, captions, labels,
column and category names, caveats) goes through Jinja2 autoescape. Figures
appear only as ``<span data-fig="fN">``, formatted by
:func:`avaloka.outcomes.format.format_figure`; regions that hold model or
template slots carry ``data-slot`` so a test can hold that no number appears
there outside a figure span.

A 1.1 document's ``data_checks`` render as the Evidence section ("Source-data
checks", one row per gate, the deciding number as a figure span) and its
``provenance.recipe`` and ``provenance.engine`` in the footer; a 1.0 document
has none of them and renders exactly as it did.

``executive`` and ``analyst`` render from the same spec (§4D); the executive
page shows a subset of the analyst page's figures, never a different
rounding of them.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Iterable, Mapping

from jinja2 import Environment
from markupsafe import Markup, escape

from avaloka.outcomes import format as _fmt
from avaloka.outcomes import scenario as _scn
from avaloka.outcomes import spec as _spec
from avaloka.outcomes.geo import asset_text

MAX_BYTES = 1_500_000      # §4.7 hard cap
TARGET_BYTES = 1_200_000   # §4.7 target

AUDIENCES = ("executive", "analyst")

# Closed wording per evidence kind (§2.5, §3.2 rule 5): associational evidence
# never reads as an effect.
EVIDENCE_PHRASE = {
    "associational": "a pattern in this data, not a tested effect",
    "experimental": "measured in a controlled test",
    "causal_estimated": "an estimated causal effect",
}
STATUS_TEXT = {"done": "Done", "needs_data": "Needs data", "needs_decision": "Needs your decision", "next": "Next"}

# Closed, human labels for the recipe gates (``avaloka.recipes.base.GATES``)
# a 1.1 document's ``data_checks`` name. No digits: a check's number is its
# figure. A gate the map does not know is written out, never as snake_case.
GATE_LABEL = {
    "duplicates": "Duplicate rows",
    "missing_measure": "Missing values",
    "date_parse": "Dates read",
    "constant": "Measure varies",
    "fanout_join": "Join fan-out",
    "small_groups": "Group sizes",
    "high_cardinality": "Number of groups",
    "sample_ratio": "Sample ratio",
    "leakage": "Leakage",
    "period_gaps": "Period gaps",
    "min_rows": "Enough rows",
    "incomplete_period": "Latest period complete",
    "simpson": "Within-group reversal",
    "collinearity": "Collinearity",
    "heteroscedasticity": "Residual spread",
    "flag_share": "Share flagged",
    "min_cohorts": "Enough cohorts",
    "power": "Statistical power",
}
CHECK_STATUS = {"passed": "Passed", "qualified": "Qualified", "failed": "Failed"}
# What a caveat kind means in two or three words, for a check whose report
# carries no caveat sentence of that kind (the sentence is used when it does).
CAVEAT_LABEL = {
    "sampling": "Sampled rows", "associational": "Association only", "additive_model": "Model assumption",
    "independence": "Drivers move together", "extrapolation": "Beyond the data", "clamped": "Held at the edge",
    "small_n": "Small sample", "suppressed_geo": "Regions withheld", "missing_template_data": "Data not present",
    "currency_unstated": "Currency unstated", "data_quality": "Data quality", "other": "See caveats",
}


class ExportTooLarge(ValueError):
    def __init__(self, size: int):
        self.size = size
        super().__init__(f"the export is {size:,} bytes, over the {MAX_BYTES:,}-byte cap")


# ── shared view selection (Markdown reuses it) ────────────────────────────


def figures_by_id(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["id"]: f for f in report.get("figures") or []}


def evidence_kinds(report: Mapping[str, Any]) -> list[str]:
    kinds: list[str] = []
    for section in ("whatif", "strategies"):
        for item in report.get(section) or []:
            k = item.get("evidence_kind")
            if k and k not in kinds:
                kinds.append(k)
    sc = report.get("scenario") or {}
    if sc.get("evidence_kind") and sc["evidence_kind"] not in kinds:
        kinds.append(sc["evidence_kind"])
    return kinds


def pictorial_chart(report: Mapping[str, Any]) -> dict[str, Any] | None:
    """§4D: the executive page's one chart: a map, else a waterfall, else the first."""
    maps = report.get("maps") or []
    charts = report.get("charts") or []
    if maps:
        return maps[0]
    for c in charts:
        if c.get("chart_kind") == "waterfall":
            return c
    return charts[0] if charts else None


def humanize(token: str) -> str:
    """``fanout_join`` → ``Fanout join``: a closed-vocabulary token the label maps do not know."""
    return str(token).replace("_", " ").strip().capitalize()


def check_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The Evidence section's rows: label, status word, figure id and note per data check.

    The note is the report's own caveat sentence for the check's caveat kind
    when it carries one, else the kind's short label; a failed check's reason
    is written out. Every string here is digit-free; the number is ``figure``.
    """
    by_kind: dict[str, str] = {}
    for c in report.get("caveats") or []:
        by_kind.setdefault(c.get("kind", ""), c.get("text", ""))
    rows = []
    for check in report.get("data_checks") or []:
        kind, reason = check.get("caveat_kind"), check.get("reason")
        note = by_kind.get(kind) or CAVEAT_LABEL.get(kind) or (humanize(kind) if kind else "")
        if reason:
            note = humanize(reason) if not note else f"{note}; {humanize(reason).lower()}"
        rows.append({"label": GATE_LABEL.get(check["gate"]) or humanize(check["gate"]),
                     "status": CHECK_STATUS.get(check["status"]) or humanize(check["status"]),
                     "figure": check.get("figure"), "note": note})
    return rows


def select(report: Mapping[str, Any], audience: str) -> dict[str, Any]:
    """What each audience shows, from one spec."""
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}")
    strategies = sorted(report.get("strategies") or [], key=lambda s: s.get("rank", 99))
    steps = sorted(report.get("steps") or [], key=lambda s: s.get("order", 99))
    if audience == "executive":
        chart = pictorial_chart(report)
        ask = next((s for s in steps if s.get("status") == "needs_decision"), None)
        return {"kpis": (report.get("kpis") or [])[:1], "charts": [chart] if chart else [],
                "strategies": strategies[:1], "risks": (report.get("caveats") or [])[:3],
                "ask": ask, "steps": [], "whatif": [], "method": None, "follow_ups": [], "side_notes": [],
                "completion": None, "checks": check_rows(report)}
    return {"kpis": report.get("kpis") or [], "charts": (report.get("charts") or []) + (report.get("maps") or []),
            "strategies": strategies, "risks": [], "ask": None, "steps": steps,
            "whatif": report.get("whatif") or [], "method": report.get("method"),
            "follow_ups": report.get("follow_ups") or [], "side_notes": report.get("side_notes") or [],
            "completion": report.get("completion"), "checks": check_rows(report)}


def provenance_extras(report: Mapping[str, Any]) -> tuple[str, str]:
    """``("describe 1.0", "duckdb 1.2.0, scipy 1.14.1")`` from a 1.1 document's
    ``provenance.recipe`` and ``provenance.engine``; empty strings for a 1.0 one."""
    prov = report.get("provenance") or {}
    recipe = prov.get("recipe") or {}
    recipe_text = f"{recipe['id']} {recipe['version']}" if recipe else ""
    engines = prov.get("engine") or {}
    return recipe_text, ", ".join(f"{name} {version}" for name, version in engines.items())


# ── HTML pieces ───────────────────────────────────────────────────────────


def _fig(figs: Mapping[str, Any], fid: str | None, *, signed: bool = False) -> Markup:
    if not fid or fid not in figs:
        return Markup("")
    return Markup('<span class="fig" data-fig="{}">{}</span>').format(fid, _fmt.format_figure(figs[fid], signed=signed))


def _slot(text: str | None, figs: Mapping[str, Any]) -> Markup:
    """Escape a text slot, substituting each ``{fN}`` with its figure span."""
    if not text:
        return Markup("")
    out, pos = [], 0
    for m in _spec.PLACEHOLDER.finditer(text):
        out.append(escape(text[pos:m.start()]))
        fid = m.group(1)
        out.append(_fig(figs, fid) if fid in figs else escape(m.group(0)))
        pos = m.end()
    out.append(escape(text[pos:]))
    return Markup("").join(out)


def json_for_script(obj: Any) -> str:
    """JSON safe inside ``<script type="application/json">``."""
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for a, b in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), (" ", "\\u2028"), (" ", "\\u2029")):
        text = text.replace(a, b)
    return text


def csp_hash(script: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii") + "'"


INIT_JS = """(function () {
  "use strict";
  var spec = JSON.parse(document.getElementById("avaloka-report").textContent);
  var reduce = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  var geo = document.querySelectorAll("script[data-geo]");
  for (var g = 0; g < geo.length; g++) {
    echarts.registerMap(geo[g].getAttribute("data-geo"), JSON.parse(geo[g].textContent));
  }
  var all = (spec.charts || []).concat(spec.maps || []);
  var byId = {};
  for (var i = 0; i < all.length; i++) { byId[all[i].id] = all[i]; }
  var shown = [];
  var els = document.querySelectorAll("[data-chart]");
  for (var j = 0; j < els.length; j++) {
    var c = byId[els[j].getAttribute("data-chart")];
    if (!c) { continue; }
    var opt = c.option;
    if (reduce) { opt.animation = false; }
    try {
      var inst = echarts.init(els[j], null, { renderer: "canvas" });
      inst.setOption(opt);
      shown.push(inst);
    } catch (err) {
      // One bad option must not blank the rest of the report; the data table stays.
      console.error("avaloka: chart " + c.id + " could not be drawn: " + err);
    }
  }
  window.addEventListener("resize", function () {
    for (var k = 0; k < shown.length; k++) { shown[k].resize(); }
  });
  var R = window.AvalokaOutcome;
  var panels = document.querySelectorAll("[data-whatif]");
  for (var p = 0; R && p < panels.length; p++) {
    (function (panel) {
      var w = null;
      for (var q = 0; q < (spec.whatif || []).length; q++) {
        if (spec.whatif[q].id === panel.getAttribute("data-whatif")) { w = spec.whatif[q]; }
      }
      if (!w) { return; }
      var fmt = JSON.parse(panel.getAttribute("data-kpi-format") || "{}");
      var inputs = panel.querySelectorAll("input[data-driver]");
      var update = function () {
        var values = {};
        for (var i = 0; i < inputs.length; i++) { values[inputs[i].getAttribute("data-driver")] = parseFloat(inputs[i].value); }
        try {
          var out = R.evaluate(w, values);
          var show = function (v) { return R.formatNumber(v, fmt.unit || "unitless", fmt.decimals || 0, fmt.currency || null); };
          panel.querySelector('[data-out="y"]').textContent = show(out.result.y);
          panel.querySelector('[data-out="lo"]').textContent = show(out.result.lo);
          panel.querySelector('[data-out="hi"]').textContent = show(out.result.hi);
          panel.querySelector('[data-out="sure"]').textContent = R.SURE_TEXT[out.result.sure];
          for (var k = 0; k < inputs.length; k++) {
            var id = inputs[k].getAttribute("data-driver");
            var d = null;
            for (var m = 0; m < w.drivers.length; m++) { if (w.drivers[m].id === id) { d = w.drivers[m]; } }
            var held = false;
            for (var c = 0; c < out.clamped.length; c++) { if (out.clamped[c].driver === id) { held = true; } }
            inputs[k].setAttribute("aria-valuetext", R.formatNumber(out.applied[id], d.unit, 1, null) + (held ? ", held at the edge of the data" : ""));
          }
        } catch (err) {
          console.error("avaloka: what-if " + w.id + " could not be evaluated: " + err);
        }
      };
      for (var i = 0; i < inputs.length; i++) { inputs[i].addEventListener("input", update); }
    })(panels[p]);
  }
  if (window.parent && window.parent !== window) {
    var send = function () {
      window.parent.postMessage({ type: "avaloka-report-height", h: document.documentElement.scrollHeight }, "*");
    };
    window.addEventListener("load", send);
    window.addEventListener("resize", send);
  }
})();
"""

_CSS = """
:root{--fg:#1b1f24;--muted:#57606a;--bg:#f6f8fa;--card:#ffffff;--line:#d0d7de;--accent:#0969da}
@media (prefers-color-scheme: dark){:root{--fg:#e6edf3;--muted:#9da7b3;--bg:#0d1117;--card:#161b22;--line:#30363d;--accent:#4493f8}}
*{box-sizing:border-box} body{margin:0;font:15px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg)}
.wrap{max-width:980px;margin:0 auto;padding:24px 16px}
header .brand{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
h1{font-size:24px;line-height:1.3;margin:4px 0 12px} h2{font-size:18px;margin:28px 0 8px} h3{font-size:16px;margin:0 0 6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:12px 0}
.kpis{display:flex;flex-wrap:wrap;gap:12px} .kpi{min-width:160px;flex:1}
.kpi .v{font-size:22px;font-weight:600} .kpi .l{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.kpi .d{font-size:13px;color:var(--muted)}
ul.bullets{padding-left:20px} .now{color:var(--muted)}
.chart{width:100%;height:340px} .caption{color:var(--muted);font-size:14px}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
details{margin-top:8px} summary{cursor:pointer;color:var(--accent)}
.tablewrap{overflow-x:auto}
footer{margin-top:32px;font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:12px}
footer dt{font-weight:600;display:inline} footer dd{display:inline;margin:0 12px 0 4px}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
"""

_PAGE = """{% macro evidence_section(rows) %}<h2>Evidence</h2><section class="card" aria-label="Source-data checks"><h3>Source-data checks</h3>
<div class="tablewrap data-table"><table><tr><th scope="col">Check</th><th scope="col">Result</th><th scope="col">Figure</th><th scope="col">Note</th></tr>
{% for c in rows %}<tr data-slot="check"><td>{{ c.label }}</td><td>{{ c.status }}</td><td>{{ c.figure }}</td><td>{{ c.note }}</td></tr>
{% endfor %}</table></div></section>
{% endmacro %}<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{{ csp }}">
<meta name="referrer" content="no-referrer">
<title>Avaloka analysis</title>
<style>{{ css }}</style></head>
<body><div class="wrap">
<header><div class="brand">Avaloka analysis &middot; {{ audience_label }}</div>
<h1 data-slot="headline">{{ headline }}</h1></header>
{% if bullets %}<section class="card" aria-label="Key points"><ul class="bullets">
{% for b in bullets %}<li data-slot="bullet"><span>{{ b.so_what }}</span> <span class="now">{{ b.now_what }}</span></li>
{% endfor %}</ul></section>{% endif %}
{% if kpis %}<section class="card kpis" aria-label="Key figures">
{% for k in kpis %}<div class="kpi"><div class="l">{{ k.label }}</div><div class="v" data-slot="kpi">{{ k.value }}</div>
{% if k.delta %}<div class="d" data-slot="kpi">{{ k.delta }}{% if k.baseline_label %} vs {{ k.baseline_label }}{% endif %}</div>{% endif %}</div>
{% endfor %}</section>{% endif %}
{% if strategies %}<h2>{{ "The move" if executive else "Strategies" }}</h2>
{% for s in strategies %}<section class="card"><h3 data-slot="strategy">{{ s.action }}</h3><p data-slot="strategy">{{ s.text }}</p>
{% if s.impact %}<p data-slot="strategy">Impact on {{ s.impact_kpi }}: {{ s.impact }}</p>{% endif %}
<p class="caption">Evidence: {{ s.phrase }}; {{ s.confidence }} confidence.</p>
{% if s.design %}<div class="tablewrap data-table"><table><tr><th>Design</th><th>Metric</th><th>Reason</th></tr>
<tr><td>{{ s.design.type }}</td><td>{{ s.design.metric }}</td><td>{{ s.design.reason }}</td></tr></table></div>{% endif %}
</section>{% endfor %}{% endif %}
{% for c in charts %}<section class="card"><h2>{{ c.title }}</h2>
<div class="chart" data-chart="{{ c.id }}" role="img" aria-label="{{ c.alt_text }}"></div>
{% if c.caption %}<p class="caption" data-slot="caption">{{ c.caption }}</p>{% endif %}
<details><summary>View data</summary><div class="tablewrap data-table"><table>
<tr>{% for col in c.columns %}<th scope="col">{{ col }}</th>{% endfor %}</tr>
{% for row in c.rows %}<tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>{% endfor %}
</table></div></details></section>
{% endfor %}
{% if whatif %}<h2>What if</h2>{% for w in whatif %}<section class="card"{% if w.live %} data-whatif="{{ w.id }}" data-kpi-format="{{ w.kpi_format }}"{% endif %}><p>Drivers of {{ w.kpi }} ({{ w.phrase }}).</p>
{% if w.live %}{% for d in w.drivers %}<p><label>{{ d.label }} ({{ d.unit }})
<input type="range" data-driver="{{ d.id }}" min="{{ d.lo }}" max="{{ d.hi }}" step="{{ d.step or "any" }}" value="{{ d.baseline }}" aria-valuetext="{{ d.valuetext }}"></label></p>{% endfor %}
<p class="result" aria-live="polite">{{ w.kpi }}: <output data-out="y">{{ w.y }}</output> (range <output data-out="lo">{{ w.lo }}</output> to <output data-out="hi">{{ w.hi }}</output>). <span data-out="sure">{{ w.sure }}</span></p>
{% else %}<p class="caption">Interactive what-if is not available for this driver form in this version.</p>{% endif %}
<div class="tablewrap data-table"><table><tr><th>Driver</th><th>Baseline</th><th>Range</th><th>Form</th></tr>
{% for d in w.drivers %}<tr><td>{{ d.label }}</td><td>{{ d.baseline }}</td><td>{{ d.lo }} to {{ d.hi }}</td><td>{{ d.form }}</td></tr>{% endfor %}
</table></div></section>{% endfor %}{% endif %}
{% if checks and executive %}{{ evidence_section(checks) }}{% endif %}
{% if risks %}<h2>Risks</h2><section class="card"><ul>{% for r in risks %}<li>{{ r.text }}</li>{% endfor %}</ul></section>{% endif %}
{% if ask %}<h2>The ask</h2><section class="card"><p>{{ ask.title }}</p></section>{% endif %}
{% if steps %}<h2>Steps</h2><section class="card"><ol>{% for s in steps %}<li><strong>{{ s.status }}</strong>: {{ s.title }}{% if s.needs %} (needs {{ s.needs }}){% endif %}</li>{% endfor %}</ol>
{% if completion %}<p class="caption">Status: {{ completion.state }}{% for m in completion.missing %}; {{ m }}{% endfor %}</p>{% endif %}</section>{% endif %}
{% if follow_ups %}<h2>Follow-ups</h2><section class="card"><ul>{% for f in follow_ups %}<li>{{ f.label }}: {{ f.prompt }}</li>{% endfor %}</ul></section>{% endif %}
{% if side_notes %}<h2>Would sharpen this</h2><section class="card"><ul>{% for n in side_notes %}<li>{{ n.what }}: {{ n.why }}</li>{% endfor %}</ul></section>{% endif %}
{% if method %}<h2>Method</h2><section class="card"><p>{{ method.summary }}</p>
{% for s in method.steps %}<p>{{ s }}</p>{% endfor %}
<p class="caption">Rows analysed: {{ method.rows }}{% if method.sampled %} (a sample){% endif %}. Code: <code>GET /analysis/{{ aid }}/code</code></p></section>{% endif %}
{% if checks and not executive %}{{ evidence_section(checks) }}{% endif %}
<footer aria-label="Provenance"><dl>
<dt>Analysis</dt><dd>{{ aid }}</dd><dt>Version</dt><dd>{{ version }}</dd>
<dt>Generated</dt><dd>{{ created_at }}</dd><dt>Generator</dt><dd>{{ generator }}</dd>
<dt>Evidence</dt><dd>{{ evidence }}</dd><dt>Verified</dt><dd>{{ verified }}</dd>{% if recipe %}<dt>Recipe</dt><dd>{{ recipe }}</dd>{% endif %}{% if engines %}<dt>Engines</dt><dd>{{ engines }}</dd>{% endif %}</dl>
{% if caveats %}<p>Caveats:</p><ul>{% for c in caveats %}<li>{{ c.text }}</li>{% endfor %}</ul>{% endif %}
</footer>
</div>
<script type="application/json" id="avaloka-report">{{ spec_json }}</script>
{% for name, text in geo %}<script type="application/json" data-geo="{{ name }}">{{ text }}</script>
{% endfor %}<script>{{ echarts_js }}</script>
{% if runtime_js %}<script>{{ runtime_js }}</script>
{% endif %}
<script>{{ init_js }}</script>
</body></html>
"""

_env = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)
_template = _env.from_string(_PAGE)


def _kpi_rows(kpis: Iterable[Mapping[str, Any]], figs) -> list[dict[str, Any]]:
    out = []
    for k in kpis:
        out.append({"label": k.get("label", ""), "value": _fig(figs, k.get("value")),
                    "delta": _fig(figs, k.get("delta"), signed=True) if k.get("delta") else None,
                    "baseline_label": k.get("baseline_label")})
    return out


def _strategy_rows(strategies, figs) -> list[dict[str, Any]]:
    out = []
    for s in strategies:
        imp = s.get("impact") or {}
        impact = None
        if imp.get("mid"):
            impact = Markup("{} (range {} to {})").format(_fig(figs, imp["mid"]), _fig(figs, imp.get("low")),
                                                          _fig(figs, imp.get("high")))
        out.append({"action": _slot(s.get("action"), figs), "text": _slot(s.get("text"), figs),
                    "impact": impact, "impact_kpi": imp.get("kpi", ""),
                    "phrase": EVIDENCE_PHRASE.get(s.get("evidence_kind"), "unstated"),
                    "confidence": s.get("confidence", ""), "design": s.get("design")})
    return out


def _kpi_format(report, kpi_id) -> dict[str, Any]:
    figs = figures_by_id(report)
    for k in report.get("kpis") or []:
        if k.get("id") == kpi_id and k.get("value") in figs:
            f = figs[k["value"]]
            return {"unit": f["unit"], "decimals": f["decimals"], "currency": f.get("currency"), "label": k.get("label")}
    return {"unit": "unitless", "decimals": 1, "currency": None, "label": kpi_id}


def _whatif_row(w, report, figs) -> dict[str, Any]:
    """A what-if panel. Live (sliders over outcome-runtime.js) only when every
    driver's form is one the 2.0 scenario model computes."""
    kf = _kpi_format(report, w.get("kpi"))
    row: dict[str, Any] = {"id": w.get("id"), "kpi": kf["label"],
                           "phrase": EVIDENCE_PHRASE.get(w.get("evidence_kind"), "unstated"),
                           "drivers": [], "live": False}
    try:
        out = _scn.evaluate(w)
    except _scn.ScenarioError:
        out = None
    for d in w.get("drivers") or []:
        row["drivers"].append({**d, "valuetext": _fmt.format_number(d["baseline"], d["unit"], 1)})
    if out is not None:
        show = lambda v: _fmt.format_number(v, kf["unit"], kf["decimals"], kf["currency"])  # noqa: E731
        row.update({"live": True, "y": show(out["result"]["y"]), "lo": show(out["result"]["lo"]),
                    "hi": show(out["result"]["hi"]), "sure": _scn.SURE_TEXT[out["result"]["sure"]],
                    "kpi_format": json.dumps({k: kf[k] for k in ("unit", "decimals", "currency")})})
    return row


def _geo_assets(report) -> list[tuple[str, str]]:
    names = []
    for m in report.get("maps") or []:
        a = m.get("asset")
        if a and a != "none" and a not in names:
            names.append(a)
        geo_opt = (m.get("option") or {}).get("geo") or {}
        if geo_opt.get("map") in ("world_110m", "us_states_20m") and geo_opt["map"] not in names:
            names.append(geo_opt["map"])
    return [(n, json_for_script(json.loads(asset_text(f"{n}.json")))) for n in sorted(names)]


def render_html(report: Mapping[str, Any], *, audience: str | None = None, enforce_cap: bool = True) -> str:
    """The export as one HTML string. Raises :class:`ExportTooLarge` over the cap."""
    report = _spec.accept(report)
    audience = audience or report.get("audience") or "executive"
    view = select(report, audience)
    figs = figures_by_id(report)
    headline = report["headline"]
    bullets = [{"so_what": _slot(b["so_what"], figs), "now_what": _slot(b["now_what"], figs)}
               for b in headline.get("bullets") or []]
    if audience == "executive":
        bullets = bullets[:3]
    charts = [{"id": c["id"], "title": c.get("title", ""), "alt_text": c.get("alt_text", ""),
               "caption": _slot(c.get("caption"), figs), "columns": c["data_table"]["columns"],
               "rows": c["data_table"]["rows"]} for c in view["charts"]]
    whatif = [_whatif_row(w, report, figs) for w in view["whatif"]]
    steps = [{"status": STATUS_TEXT.get(s.get("status"), s.get("status")), "title": s.get("title"),
              "needs": s.get("needs")} for s in view["steps"]]
    method = None
    if view["method"]:
        sample = view["method"].get("sample") or {}
        method = {"summary": view["method"].get("summary", ""), "steps": view["method"].get("steps") or [],
                  "rows": sample.get("rows_analyzed"), "sampled": sample.get("is_sampled")}
    checks = [{**c, "figure": _fig(figs, c["figure"])} for c in view["checks"]]
    analysis = report["analysis"]
    prov = report["provenance"]
    kinds = evidence_kinds(report)
    recipe, engines = provenance_extras(report)
    echarts_js = asset_text("echarts.min.js")
    if "</script" in echarts_js.lower():
        raise ValueError("echarts.min.js contains a script terminator")
    runtime_js = asset_text("outcome-runtime.js") if any(w["live"] for w in whatif) else ""
    if "</script" in runtime_js.lower():
        raise ValueError("outcome-runtime.js contains a script terminator")
    hashes = [csp_hash(echarts_js)] + ([csp_hash(runtime_js)] if runtime_js else []) + [csp_hash(INIT_JS)]
    csp = ("default-src 'none'; script-src " + " ".join(hashes)
           + "; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; "
             "form-action 'none'; base-uri 'none'")
    html = _template.render(
        csp=csp, css=Markup(_CSS), audience_label="Executive summary" if audience == "executive" else "Analyst report",
        executive=audience == "executive",
        headline=_slot(headline["text"], figs), bullets=bullets,
        kpis=_kpi_rows(view["kpis"], figs), strategies=_strategy_rows(view["strategies"], figs),
        charts=charts, whatif=whatif, risks=view["risks"], ask=view["ask"], steps=steps,
        completion=view["completion"], follow_ups=view["follow_ups"], side_notes=view["side_notes"],
        method=method, checks=checks, aid=analysis["analysis_id"], version=analysis["version"],
        created_at=report["created_at"], generator=f"{prov['generator']} {prov['generator_version']}",
        recipe=recipe, engines=engines,
        evidence=", ".join(kinds) if kinds else "descriptive figures only",
        verified="yes" if prov["verifier"]["passed"] else "not verified",
        caveats=report.get("caveats") or [],
        spec_json=Markup(json_for_script(report)), geo=[(n, Markup(t)) for n, t in _geo_assets(report)],
        echarts_js=Markup(echarts_js), runtime_js=Markup(runtime_js), init_js=Markup(INIT_JS))
    size = len(html.encode("utf-8"))
    if enforce_cap and size > MAX_BYTES:
        raise ExportTooLarge(size)
    return html


def main(argv: list[str] | None = None) -> int:
    """``python -m avaloka.outcomes.render_html REPORT.json [--audience A]`` writes the export to stdout.

    The Playwright fixture is regenerated by ``python tests/avaloka/outcomes/conftest.py``.
    """
    import argparse
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(prog="python -m avaloka.outcomes.render_html")
    ap.add_argument("report", type=Path)
    ap.add_argument("--audience", choices=AUDIENCES)
    args = ap.parse_args(argv)
    sys.stdout.write(render_html(json.loads(args.report.read_text(encoding="utf-8")), audience=args.audience))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
