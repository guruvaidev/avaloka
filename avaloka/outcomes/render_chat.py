"""The chat summary of an Outcome Report (§4.6): plain text, at most 700 chars.

::

    {headline}
    • {so_what 1} → {now_what 1}
    ...
    Top move: {strategy 1 action} ({evidence-kind phrase}, {confidence} confidence)
    Full report: /analysis?aid=…&report=v  |  avaloka report {aid}

Deterministic, no model call. Figures go through the same formatter as the
exports, so the summary prints the strings the report prints. When the full
text would pass the limit, bullets are dropped from the last one up, then the
``now_what`` halves; the headline and the report pointer always stay.
"""

from __future__ import annotations

import re
from typing import Any, Mapping
from urllib.parse import quote

from avaloka.outcomes import format as _fmt
from avaloka.outcomes import spec as _spec
from avaloka.outcomes.render_html import EVIDENCE_PHRASE, figures_by_id

MAX_CHARS = 700
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", _CONTROL.sub(" ", s)).strip()


def _text(slot: str | None, figs) -> str:
    return _clean(_fmt.render_text(slot or "", figs)) if slot else ""


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "…"


def render_chat(report: Mapping[str, Any]) -> str:
    report = _spec.accept(report)
    figs = figures_by_id(report)
    a = report["analysis"]
    pointer = (f"Full report: /analysis?aid={quote(str(a['analysis_id']), safe='')}&report={a['version']}"
               f"  |  avaloka report {_clean(str(a['analysis_id']))}")
    head = _text(report["headline"]["text"], figs)
    bullets = [(_text(b["so_what"], figs), _text(b["now_what"], figs)) for b in report["headline"].get("bullets") or []]
    top = sorted(report.get("strategies") or [], key=lambda s: s.get("rank", 99))
    move = ""
    if top:
        s = top[0]
        move = (f"Top move: {_text(s.get('action'), figs)} "
                f"({EVIDENCE_PHRASE.get(s.get('evidence_kind'), 'unstated')}, {s.get('confidence', '')} confidence)")

    def compose(bs, with_now=True):
        lines = [head]
        lines += [f"• {sw} → {nw}" if with_now and nw else f"• {sw}" for sw, nw in bs]
        if move:
            lines.append(move)
        lines.append(pointer)
        return "\n".join(lines)

    for with_now in (True, False):
        for n in range(len(bullets), -1, -1):
            text = compose(bullets[:n], with_now)
            if len(text) <= MAX_CHARS:
                return text
    budget = MAX_CHARS - len(pointer) - 1
    return _clip(head, budget) + "\n" + pointer
