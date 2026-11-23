"""Markdown export of an Outcome Report (§4D).

Charts become their ``data_table`` plus alt text. Every string the renderer
did not write is escaped (:func:`md_escape`): the CommonMark punctuation
``\\ ` * _ { } [ ] ( ) # + - . ! | ~`` gets a backslash, ``<`` and ``>``
become ``&lt;`` / ``&gt;`` so no raw HTML survives, and control characters are
stripped. No link is ever built from text; the page holds no link at all.
Same audience selection as the HTML export.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from avaloka.outcomes import format as _fmt
from avaloka.outcomes import spec as _spec
from avaloka.outcomes.render_html import (EVIDENCE_PHRASE, STATUS_TEXT, evidence_kinds, figures_by_id,
                                          provenance_extras, select)

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f  ]")
_PUNCT = re.compile(r"([\\`*_{}\[\]()#+\-.!|~])")


def md_escape(text: Any) -> str:
    s = _CONTROL.sub("", str(text)).replace("\r", "").replace("\n", " ")
    s = s.replace("&", "&amp;")
    s = _PUNCT.sub(r"\\\1", s)
    return s.replace("<", "&lt;").replace(">", "&gt;")


def _fig(figs, fid, *, signed=False) -> str:
    return md_escape(_fmt.format_figure(figs[fid], signed=signed)) if fid in figs else ""


def _slot(text: str | None, figs) -> str:
    if not text:
        return ""
    out, pos = [], 0
    for m in _spec.PLACEHOLDER.finditer(text):
        out.append(md_escape(text[pos:m.start()]))
        out.append(_fig(figs, m.group(1)) if m.group(1) in figs else md_escape(m.group(0)))
        pos = m.end()
    out.append(md_escape(text[pos:]))
    return "".join(out)


def _table(columns, rows) -> list[str]:
    if not columns:
        return []
    lines = ["| " + " | ".join(md_escape(c) for c in columns) + " |",
             "|" + "|".join(" --- " for _ in columns) + "|"]
    for r in rows:
        cells = [md_escape("" if v is None else v) for v in r] + [""] * (len(columns) - len(r))
        lines.append("| " + " | ".join(cells[:len(columns)]) + " |")
    return lines


def _checks(rows, figs) -> list[str]:
    """The Evidence section: one table row per data check, the figure formatted like any other."""
    lines = ["## Evidence", "", "Source-data checks:", "", "| Check | Result | Figure | Note |", "| --- | --- | --- | --- |"]
    for c in rows:
        lines.append(f"| {md_escape(c['label'])} | {md_escape(c['status'])} | {_fig(figs, c['figure'])} | {md_escape(c['note'])} |")
    return lines + [""]


def render_md(report: Mapping[str, Any], *, audience: str | None = None) -> str:
    report = _spec.accept(report)
    audience = audience or report.get("audience") or "executive"
    view = select(report, audience)
    figs = figures_by_id(report)
    out: list[str] = ["*Avaloka analysis* · " + ("Executive summary" if audience == "executive" else "Analyst report"),
                      "", "# " + _slot(report["headline"]["text"], figs), ""]
    for b in report["headline"].get("bullets") or []:
        out.append(f"- {_slot(b['so_what'], figs)} {_slot(b['now_what'], figs)}")
    if report["headline"].get("bullets"):
        out.append("")
    if view["kpis"]:
        out += ["## Key figures", ""]
        for k in view["kpis"]:
            line = f"- **{md_escape(k.get('label', ''))}**: {_fig(figs, k.get('value'))}"
            if k.get("delta"):
                line += f" ({_fig(figs, k['delta'], signed=True)}"
                line += f" vs {md_escape(k['baseline_label'])})" if k.get("baseline_label") else ")"
            out.append(line)
        out.append("")
    if view["strategies"]:
        out += ["## " + ("The move" if audience == "executive" else "Strategies"), ""]
        for s in view["strategies"]:
            out.append(f"### {_slot(s.get('action'), figs)}")
            out += ["", _slot(s.get("text"), figs), ""]
            imp = s.get("impact") or {}
            if imp.get("mid"):
                out += [f"Impact on {md_escape(imp.get('kpi', ''))}: {_fig(figs, imp['mid'])} "
                        f"(range {_fig(figs, imp.get('low'))} to {_fig(figs, imp.get('high'))})", ""]
            out += [f"Evidence: {EVIDENCE_PHRASE.get(s.get('evidence_kind'), 'unstated')}; "
                    f"{md_escape(s.get('confidence', ''))} confidence.", ""]
    for c in view["charts"]:
        out += [f"## {md_escape(c.get('title', ''))}", ""]
        if c.get("caption"):
            out += [_slot(c["caption"], figs), ""]
        out += [f"*{md_escape(c.get('alt_text', ''))}*", ""]
        out += _table(c["data_table"]["columns"], c["data_table"]["rows"]) + [""]
    for w in view["whatif"]:
        out += [f"## What if: {md_escape(w.get('kpi', ''))}", "",
                f"Evidence: {EVIDENCE_PHRASE.get(w.get('evidence_kind'), 'unstated')}.", ""]
        out += _table(["driver", "baseline", "low", "high", "form"],
                      [[d.get("label"), d.get("baseline"), d.get("lo"), d.get("hi"), d.get("form")]
                       for d in w.get("drivers") or []]) + [""]
    if view["checks"] and audience == "executive":
        out += _checks(view["checks"], figs)
    if view["risks"]:
        out += ["## Risks", ""] + [f"- {md_escape(r['text'])}" for r in view["risks"]] + [""]
    if view["ask"]:
        out += ["## The ask", "", md_escape(view["ask"]["title"]), ""]
    if view["steps"]:
        out += ["## Steps", ""]
        for i, s in enumerate(view["steps"], 1):
            needs = f" (needs {md_escape(s['needs'])})" if s.get("needs") else ""
            out.append(f"{i}. **{STATUS_TEXT.get(s['status'], s['status'])}**: {md_escape(s['title'])}{needs}")
        out.append("")
    if view["follow_ups"]:
        out += ["## Follow-ups", ""] + [f"- {md_escape(f['label'])}: {md_escape(f['prompt'])}" for f in view["follow_ups"]] + [""]
    if view["side_notes"]:
        out += ["## Would sharpen this", ""] + [f"- {md_escape(n['what'])}: {md_escape(n['why'])}" for n in view["side_notes"]] + [""]
    if view["method"]:
        sample = view["method"].get("sample") or {}
        out += ["## Method", "", md_escape(view["method"].get("summary", "")), ""]
        out += [md_escape(s) for s in view["method"].get("steps") or []]
        out += [f"Rows analysed: {md_escape(sample.get('rows_analyzed'))}"
                + (" \\(a sample\\)" if sample.get("is_sampled") else "") + ".", ""]
    if view["checks"] and audience != "executive":
        out += _checks(view["checks"], figs)
    a, p = report["analysis"], report["provenance"]
    kinds = evidence_kinds(report)
    recipe, engines = provenance_extras(report)
    out += ["---", "", "**Provenance**", "",
            f"- Analysis: {md_escape(a['analysis_id'])}, version {md_escape(a['version'])}",
            f"- Generated: {md_escape(report['created_at'])}",
            f"- Generator: {md_escape(p['generator'])} {md_escape(p['generator_version'])}",
            f"- Evidence: {md_escape(', '.join(kinds) if kinds else 'descriptive figures only')}",
            f"- Verified: {'yes' if p['verifier']['passed'] else 'not verified'}"]
    if recipe:
        out.append(f"- Recipe: {md_escape(recipe)}")
    if engines:
        out.append(f"- Engines: {md_escape(engines)}")
    if report.get("caveats"):
        out += ["", "Caveats:", ""] + [f"- {md_escape(c['text'])}" for c in report["caveats"]]
    return "\n".join(out).rstrip() + "\n"
