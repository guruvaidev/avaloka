"""Render a benchmark scorecard as one self-contained HTML page.

The benchmark had three result surfaces and none of them was readable by a
human who was not already holding the code: a terminal print
(:func:`avaloka.benchmark.__main__._print_scorecard`), a ``--json`` dump, and a
Rich table. So the evidence a run produces -- the per-check ``detail`` strings,
which are the only place the *reason* for a verdict is written down -- was
visible either as a wall of scrolling text or not at all, and there was no way
to look at a score next to the commit it came from. The owner asked twice to see
the plots in a browser.

This module writes that page. Two properties it has to keep:

*Self-contained.* All CSS and every chart is inline. No CDN, no webfont, no
``<script src>``. The page is an artifact you can attach to a PR, drop in a
bucket or open from a USB stick in a meeting room with no network, and it looks
the same in all three.

*Hand-rolled charts.* ``avaloka``'s runtime dependencies are exactly eight and a
contract test holds them there, so there is no plotting library to reach for.
The geometry is computed here in Python and emitted as ``<svg>`` markup, which
adds nothing to the dependency set. Colours are the data-viz status palette
(pass/fail/skip is a *state*, not a series) over a single sequential hue for the
family scores, and both are declared twice -- once per ``prefers-color-scheme``
-- because a dark page is chosen, not an inverted light one.

It renders the ``dict`` from :meth:`Scorecard.as_dict`, never the live objects,
so it works just as well on a ``scorecard.json`` written by a run that finished
last week. :mod:`avaloka.benchmark.serve` depends on that.

Two rules about text. Every interpolated string goes through :func:`_e`: a check
detail legitimately contains ``<``, ``>`` and ``&``, and a column name arrives
from whatever file the benchmark was pointed at, so it is untrusted. And a
detail is never truncated -- it is the evidence, and a report that elides the
evidence is back to being a number nobody can check.
"""

from __future__ import annotations

import html
import math
import re
from pathlib import Path
from typing import Any, Optional

__all__ = ["render_html", "write_html"]

#: Checks whose ``detail`` carries resolved-vs-ground-truth and so earns a
#: comparison block instead of a one-line string. These grade whether the frame
#: Avaloka worked on is the table that was in the file, which is the whole point
#: of the comprehension family -- rendering them as prose would hide exactly the
#: thing they were added to show.
#: Comprehension checks earn a comparison block instead of a one-line string.
#: These are the ones that grade whether the frame Avaloka worked on is the
#: table that was actually in the file -- the thing every other scorer assumes.
#: Rendering "missing 3 of 4 ground-truth column(s): [...]; resolved instead:
#: [...]" as prose hides exactly what the check was added to show.
#:
#: The names and the detail formats below are taken from the scorers in
#: :mod:`avaloka.benchmark.comprehension`, not from a convention. Two of them
#: are per-column and arrive as ``dtype[Net revenue]``, so prefix matching is
#: needed as well as exact names.
_COMPREHENSION_NAMES = frozenset({
    "no_placeholder_column_names",
    "header_row_identified",
    "expected_columns_resolved",
    "row_count",
    "narration_names_only_real_columns",
})
_COMPREHENSION_PREFIXES = ("dtype[",)


def _is_comprehension(name: str) -> bool:
    return name in _COMPREHENSION_NAMES or name.startswith(_COMPREHENSION_PREFIXES)

# ---------------------------------------------------------------------------
# Text and number handling
# ---------------------------------------------------------------------------
def _e(value: Any) -> str:
    """Escape anything for inclusion in HTML text or an attribute.

    Everything interpolated into the page goes through here. ``quote=True`` so
    the same function is safe inside an attribute value.
    """
    return html.escape("" if value is None else str(value), quote=True)


def _num(value: Any, default: float = 0.0) -> float:
    """A finite float, or ``default``.

    A score can arrive as ``None`` from a truncated file or as NaN from a scorer
    that divided by an empty weight sum. Neither should reach the page: "nan" in
    a score column reads as a result.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _pct(value: float) -> str:
    return f"{_num(value) * 100:.0f}%"


def _score(value: Any) -> str:
    return f"{_num(value):.2f}"


# ---------------------------------------------------------------------------
# Scorecard reading
# ---------------------------------------------------------------------------
def _status(result: dict) -> str:
    """``PASS`` | ``FAIL`` | ``SKIP`` for one run.

    A run whose error mentions "unavailable" is a skip, not a failure -- the
    dataset or connector was not there to exercise. Same precedence as
    ``_print_scorecard``: passed wins, then the unavailable test, then fail.
    """
    if result.get("passed"):
        return "PASS"
    error = result.get("error")
    if error and "unavailable" in str(error):
        return "SKIP"
    return "FAIL"


#: Failures first, then skips, then passes; worst score first inside a group.
_STATUS_ORDER = {"FAIL": 0, "SKIP": 1, "PASS": 2}


def _ordered(results: list[dict]) -> list[dict]:
    return sorted(
        results,
        key=lambda r: (_STATUS_ORDER.get(_status(r), 3), _num(r.get("score")),
                       str(r.get("task", "")), str(r.get("source_scheme", ""))),
    )


def _results(card: dict) -> list[dict]:
    raw = card.get("results")
    return [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []


def _by_family(card: dict, results: list[dict]) -> dict[str, float]:
    """Per-family scores, from the card when it has them, else recomputed.

    Recomputing matters for a hand-built or partial card; it is the same mean
    :meth:`Scorecard.by_family` takes.
    """
    declared = card.get("by_family")
    if isinstance(declared, dict) and declared:
        return {str(k): _num(v) for k, v in declared.items()}
    buckets: dict[str, list[float]] = {}
    for r in results:
        buckets.setdefault(str(r.get("family") or "unknown"), []).append(_num(r.get("score")))
    return {f: sum(v) / len(v) for f, v in buckets.items() if v}


# ---------------------------------------------------------------------------
# Comprehension details: resolved beside ground truth
#
# Each parser below matches one scorer in avaloka.benchmark.comprehension and
# returns a small structure the renderer turns into a table; any parser may
# return None, and the caller then prints the detail verbatim. Printing it
# verbatim loses the layout and no information, so an unrecognised detail is a
# cosmetic miss rather than a lost result -- which is the right failure mode
# when the scorers' wording changes.
# ---------------------------------------------------------------------------
def _items(inner: str) -> list[str]:
    """Split the inside of a repr'd list into its elements.

    Quote-aware, so a column name containing a comma survives: ``'A, B', 'C'``
    is two names, not three. Never raises.
    """
    out: list[str] = []
    buf: list[str] = []
    quote: Optional[str] = None
    for char in inner:
        if quote:
            if char == quote:
                quote = None
            else:
                buf.append(char)
        elif char in "'\"":
            quote = char
        elif char == ",":
            text = "".join(buf).strip()
            if text:
                out.append(text)
            buf = []
        else:
            buf.append(char)
    text = "".join(buf).strip()
    if text:
        out.append(text)
    return out


_L = r"\[([^\]]*)\]"

_RE_PLACEHOLDER_FAIL = re.compile(
    r"placeholder name\(s\) resolved:\s*" + _L + r"\s*\(of\s+\d+\s+columns:\s*" + _L)
_RE_PLACEHOLDER_OK = re.compile(r"all\s+\d+\s+resolved names are real:\s*" + _L)
_RE_COLUMNS_FAIL = re.compile(
    r"missing\s+\d+\s+of\s+(\d+)\s+ground-truth column\(s\):\s*" + _L
    + r";\s*resolved instead:\s*" + _L)
_RE_COLUMNS_OK = re.compile(
    r"all\s+(\d+)\s+ground-truth columns resolved;\s*resolved:\s*" + _L)
_RE_HEADER = re.compile(
    r"reader reports header on row\s+(\S+?)\s*\(1-based\),\s*expected\s+(\S+)")
_RE_HEADER_NONE = re.compile(r"reported no header row at all.*?expected\s+(\d+)", re.S)
_RE_ROWCOUNT = re.compile(
    r"frame has\s+(\d+)\s+rows,\s*expected\s+(\d+)\s*\(difference\s*([+-]?\d+)\)")
_RE_ROWCOUNT_REPORTED = re.compile(r"data_quality\.json reports n_rows=(\S+?)\s*$")
_RE_DTYPE = re.compile(r"dtype is\s+(.+?),\s*expected kind\s*'([^']*)'")
_RE_DTYPE_ABSENT = re.compile(
    r"column is ABSENT from the frame.*?against\s*'([^']*)';\s*frame has\s*" + _L, re.S)
_RE_NARRATION_FAIL = re.compile(
    r"name\(s\) in generated text are not columns of the frame:\s*" + _L
    + r";\s*frame columns:\s*" + _L)


def _pair(expected: str, resolved: str, *, expected_label: str = "ground truth",
          resolved_label: str = "resolved", agree: Optional[bool] = None) -> dict:
    return {"kind": "pair", "expected": expected, "resolved": resolved,
            "expected_label": expected_label, "resolved_label": resolved_label,
            "agree": (expected.strip() == resolved.strip()) if agree is None else agree}


def _table(headers: tuple[str, ...], rows: list[tuple], caption: str = "") -> dict:
    return {"kind": "table", "headers": headers, "rows": rows, "caption": caption}


def _parse_comprehension(name: str, detail: str,
                         passed: bool = False) -> Optional[dict]:
    """Structure one comprehension detail, or None to fall back to plain text."""
    if not detail:
        return None

    if name == "expected_columns_resolved":
        m = _RE_COLUMNS_FAIL.search(detail)
        if m:
            total, missing, resolved = m.group(1), _items(m.group(2)), _items(m.group(3))
            rows = [(n, "&mdash;", "in ground truth, not resolved", "no")
                    for n in missing]
            # Neutral, not green. The detail says which ground-truth names are
            # missing; it does not say which of the resolved names are the
            # ground-truth ones, so marking them "matched" would be a claim the
            # scorer did not make -- and would paint 'Unnamed: 13' as a pass.
            rows += [("&mdash;", n, "resolved by the reader", "extra")
                     for n in resolved]
            return _table(("ground truth", "resolved", ""), rows,
                          f"{len(missing)} of {total} ground-truth columns were not "
                          f"resolved; the reader returned {len(resolved)} columns.")
        m = _RE_COLUMNS_OK.search(detail)
        if m:
            resolved = _items(m.group(2))
            return _table(("ground truth", "resolved", ""),
                          [(n, n, "matched", "ok") for n in resolved],
                          f"all {m.group(1)} ground-truth columns resolved.")
        return None

    if name == "no_placeholder_column_names":
        m = _RE_PLACEHOLDER_FAIL.search(detail)
        if m:
            bad, every = set(_items(m.group(1))), _items(m.group(2))
            return _table(("resolved column", "", ""),
                          [(n, "", "invented by the reader" if n in bad
                            else "real name", "no" if n in bad else "ok")
                           for n in every],
                          f"{len(bad)} of {len(every)} resolved names are "
                          "placeholders, so the header row was misread.")
        m = _RE_PLACEHOLDER_OK.search(detail)
        if m:
            every = _items(m.group(1))
            return _table(("resolved column", "", ""),
                          [(n, "", "real name", "ok") for n in every],
                          f"all {len(every)} resolved names are real.")
        return None

    if name == "header_row_identified":
        m = _RE_HEADER.search(detail)
        if m:
            return _pair(m.group(2), m.group(1), expected_label="expected row",
                         resolved_label="reader reports row")
        m = _RE_HEADER_NONE.search(detail)
        if m:
            return _pair(m.group(1), "none reported", expected_label="expected row",
                         resolved_label="reader reports row", agree=False)
        return None

    if name == "row_count":
        m = _RE_ROWCOUNT.search(detail)
        if m:
            got, want, diff = m.group(1), m.group(2), m.group(3)
            rows = [("rows in the file", want, got,
                     "matched" if diff in ("+0", "0", "-0") else f"off by {diff}",
                     "ok" if diff in ("+0", "0", "-0") else "no")]
            reported = _RE_ROWCOUNT_REPORTED.search(detail)
            if reported:
                rows.append(("n_rows in data_quality.json", want,
                             reported.group(1),
                             "matched" if reported.group(1) == want else "differs",
                             "ok" if reported.group(1) == want else "no"))
            return _table(("measure", "ground truth", "resolved", ""), rows)
        return None

    if name.startswith("dtype["):
        column = name[len("dtype["):-1] if name.endswith("]") else name
        m = _RE_DTYPE.search(detail)
        if m:
            return _table(("column", "ground truth", "resolved", ""),
                          [(column, f"kind '{m.group(2)}'", m.group(1),
                            "matches the expected kind" if passed
                            else "not the expected kind",
                            "ok" if passed else "no")])
        m = _RE_DTYPE_ABSENT.search(detail)
        if m:
            return _table(("column", "ground truth", "resolved", ""),
                          [(column, f"kind '{m.group(1)}'", "&mdash;",
                            "absent from the frame", "no")],
                          f"the frame has: {', '.join(_items(m.group(2))) or 'nothing'}")
        return None

    if name == "narration_names_only_real_columns":
        m = _RE_NARRATION_FAIL.search(detail)
        if m:
            violations, columns = _items(m.group(1)), _items(m.group(2))
            rows = [(v, "", "named in generated text, not a column", "no")
                    for v in violations]
            return _table(("name used in the prose", "", ""), rows,
                          "the frame's real columns are: "
                          + (", ".join(columns) or "none"))
        return None

    return None


def _comparison_html(block: dict) -> str:
    """Render a parsed comprehension block. Escapes every cell."""
    if block["kind"] == "pair":
        agree = block["agree"]
        return (
            '<div class="cmp"><div class="cmp-pair">'
            f'<div class="cmp-cell"><span class="cmp-k">{_e(block["expected_label"])}'
            f'</span><span class="cmp-v">{_e(block["expected"]) or "&mdash;"}</span></div>'
            f'<div class="cmp-cell {"ok" if agree else "no"}">'
            f'<span class="cmp-k">{_e(block["resolved_label"])}</span>'
            f'<span class="cmp-v">{_e(block["resolved"]) or "&mdash;"}</span></div>'
            "</div></div>"
        )

    rows = block["rows"]
    if not rows:
        return ""
    width = len(block["headers"])
    body: list[str] = []
    for row in rows:
        css = row[-1] or ""
        verdict = row[-2]
        cells = row[:-2]
        # "&mdash;" is the one value these parsers pass through as markup; every
        # other cell is text from the scorer and is escaped.
        tds = "".join(f'<td class="mono">{c if c == "&mdash;" else _e(c)}</td>'
                      for c in cells[:width - 1])
        tds += f'<td class="vd">{_e(verdict)}</td>'
        body.append(f'<tr class="{css}">{tds}</tr>')
    headers = "".join(f"<th>{_e(h)}</th>" for h in block["headers"])
    caption = (f'<div class="cmp-cap">{_e(block["caption"])}</div>'
               if block.get("caption") else "")
    return ('<div class="cmp"><table class="cmp-t"><thead><tr>'
            f"{headers}</tr></thead><tbody>{''.join(body)}</tbody></table>"
            f"{caption}</div>")


def _check_html(check: dict) -> str:
    """One check: name, verdict, weight, and its detail in full.

    The comparison block is attempted inside a try/except so a detail in an
    unexpected shape degrades to the plain line rather than taking the page
    down. A renderer that raises on a hostile column name is worse than one
    that renders it plainly.
    """
    name = str(check.get("name", ""))
    passed = bool(check.get("passed"))
    detail = "" if check.get("detail") is None else str(check.get("detail"))
    weight = _num(check.get("weight"), 1.0)

    body = ""
    if _is_comprehension(name):
        try:
            block = _parse_comprehension(name, detail, passed)
            if block:
                body = _comparison_html(block)
        except Exception:  # noqa: BLE001 - never fail a render over one detail
            body = ""
    if not body and detail:
        body = f'<div class="detail">{_e(detail)}</div>'

    weight_html = f'<span class="wt">&times;{weight:g}</span>' if weight != 1.0 else ""
    return (
        f'<li class="chk {"ok" if passed else "no"}">'
        f'<div class="chk-h"><span class="mark">{"PASS" if passed else "FAIL"}</span>'
        f'<span class="mono cname">{_e(name)}</span>{weight_html}</div>'
        f"{body}</li>"
    )


# ---------------------------------------------------------------------------
# Charts. Geometry in Python, markup out; no plotting library.
# ---------------------------------------------------------------------------
def _rounded_bar(x: float, y: float, width: float, height: float,
                 radius: float = 4.0) -> str:
    """A horizontal bar path: square at the baseline, rounded at the data end.

    Clamps the radius so a near-zero bar degrades to a sliver instead of an
    inverted arc, and emits nothing at all for a zero-width bar.
    """
    if width <= 0.75 or height <= 0:
        return ""
    r = max(0.0, min(radius, width, height / 2.0))
    return (f'<path d="M{x:.2f},{y:.2f} H{x + width - r:.2f} '
            f'A{r:.2f},{r:.2f} 0 0 1 {x + width:.2f},{y + r:.2f} '
            f'V{y + height - r:.2f} '
            f'A{r:.2f},{r:.2f} 0 0 1 {x + width - r:.2f},{y + height:.2f} '
            f'H{x:.2f} Z"/>')


def _family_chart(by_family: dict[str, float]) -> str:
    """Per-family mean score, 0..1, as horizontal bars.

    One series, so magnitude: a single sequential hue and no legend. Each bar is
    labelled with its own value, which is what makes the chart readable without
    gridline-counting.
    """
    if not by_family:
        return ('<p class="empty">No family scores &mdash; the scorecard has no '
                "runs to group.</p>")

    rows = sorted(by_family.items(), key=lambda kv: (kv[1], kv[0]))
    gutter, right, bar_h, row_h, top = 176.0, 56.0, 16.0, 30.0, 10.0
    plot_w = 700.0 - gutter - right
    height = top + len(rows) * row_h + 28.0

    ticks: list[str] = []
    for frac in (0.0, 0.5, 1.0):
        x = gutter + plot_w * frac
        ticks.append(f'<line class="grid" x1="{x:.1f}" y1="{top:.1f}" '
                     f'x2="{x:.1f}" y2="{top + len(rows) * row_h:.1f}"/>')
        ticks.append(f'<text class="tick" x="{x:.1f}" '
                     f'y="{top + len(rows) * row_h + 16:.1f}" '
                     f'text-anchor="middle">{frac:.1f}</text>')

    bars: list[str] = []
    for index, (family, raw) in enumerate(rows):
        score = min(max(_num(raw), 0.0), 1.0)
        y = top + index * row_h + (row_h - bar_h) / 2.0
        bars.append(f'<text class="cat" x="{gutter - 10:.1f}" y="{y + bar_h - 3:.1f}" '
                    f'text-anchor="end">{_e(family)}</text>')
        bars.append(f'<g class="bar">{_rounded_bar(gutter, y, plot_w * score, bar_h)}</g>')
        bars.append(f'<text class="val" x="{gutter + plot_w * score + 8:.1f}" '
                    f'y="{y + bar_h - 3:.1f}">{_score(score)}</text>')

    return (
        f'<svg class="chart" viewBox="0 0 700 {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" '
        f'aria-label="Mean score by task family, 0 to 1">'
        f"{''.join(ticks)}{''.join(bars)}"
        f'<line class="axis" x1="{gutter:.1f}" y1="{top:.1f}" x2="{gutter:.1f}" '
        f'y2="{top + len(rows) * row_h:.1f}"/></svg>'
    )


def _breakdown_chart(passed: int, failed: int, skipped: int) -> str:
    """Pass / fail / skip as one stacked bar, with a counted legend.

    Status encoding, so the reserved status palette rather than series colours,
    and skip takes the muted grey because "not exercised" is not a verdict. The
    segments are clipped to a rounded track, which keeps the outer ends rounded
    and the interior joins square without per-segment corner arithmetic; a 2px
    surface gap separates them. Every segment is named in the legend, so colour
    never carries the meaning alone.
    """
    total = passed + failed + skipped
    segments = (("PASS", passed, "s-pass"), ("FAIL", failed, "s-fail"),
                ("SKIP", skipped, "s-skip"))
    width, height = 700.0, 30.0

    if total <= 0:
        track = f'<rect class="track" x="0" y="0" width="{width:.0f}" rx="4" height="{height:.0f}"/>'
        label = (f'<text class="val" x="{width / 2:.0f}" y="{height / 2 + 4:.0f}" '
                 'text-anchor="middle">no runs</text>')
        return (f'<svg class="chart" viewBox="0 0 700 {height:.0f}" width="100%" '
                f'height="{height:.0f}" role="img" aria-label="No runs to break down">'
                f"{track}{label}</svg>")

    pieces: list[str] = []
    x = 0.0
    for label, count, css in segments:
        if count <= 0:
            continue
        seg_w = width * (count / total)
        pieces.append(f'<rect class="{css}" x="{x:.2f}" y="0" width="{seg_w:.2f}" '
                      f'height="{height:.0f}"/>')
        if seg_w >= 64.0:
            pieces.append(f'<text class="seg" x="{x + seg_w / 2:.2f}" '
                          f'y="{height / 2 + 4:.0f}" text-anchor="middle">'
                          f"{label} {count}</text>")
        x += seg_w

    legend = "".join(
        f'<span class="lg"><span class="sw {css}"></span>{label}'
        f'<span class="lgn">{count}</span></span>'
        for label, count, css in segments
    )
    return (
        f'<svg class="chart" viewBox="0 0 700 {height:.0f}" width="100%" '
        f'height="{height:.0f}" role="img" aria-label="'
        f'{passed} passed, {failed} failed, {skipped} skipped of {total} runs">'
        '<defs><clipPath id="av-track">'
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" rx="4"/>'
        "</clipPath></defs>"
        f'<g clip-path="url(#av-track)">{"".join(pieces)}</g></svg>'
        f'<div class="legend">{legend}</div>'
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
_CSS = """
:root{
  color-scheme:light;
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --hair:rgba(11,11,11,.10);
  --seq:#2a78d6; --pass:#0ca30c; --fail:#d03b3b; --skip:#898781;
  --fail-wash:rgba(208,59,59,.08); --pass-wash:rgba(12,163,12,.07);
  --skip-wash:rgba(137,135,129,.10); --code:#eff1f3;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --hair:rgba(255,255,255,.10);
    --seq:#3987e5; --pass:#0ca30c; --fail:#d03b3b; --skip:#898781;
    --fail-wash:rgba(208,59,59,.14); --pass-wash:rgba(12,163,12,.12);
    --skip-wash:rgba(137,135,129,.14); --code:#262625;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --hair:rgba(255,255,255,.10);
  --seq:#3987e5; --pass:#0ca30c; --fail:#d03b3b; --skip:#898781;
  --fail-wash:rgba(208,59,59,.14); --pass-wash:rgba(12,163,12,.12);
  --skip-wash:rgba(137,135,129,.14); --code:#262625;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px;line-height:1.5}
.wrap{max-width:1000px;margin:0 auto;padding:32px 16px 72px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:17px;margin:34px 0 10px;padding-bottom:7px;
  border-bottom:1px solid var(--hair);font-weight:600}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.card{background:var(--surface);border:1px solid var(--hair);border-radius:10px;
  padding:18px 20px;margin:14px 0}

/* provenance: the commit is the point, so it is not a footnote */
.prov{display:flex;flex-wrap:wrap;gap:8px 10px;margin:10px 0 0;
  font-size:12.5px;color:var(--ink2)}
.prov .p{background:var(--code);border-radius:6px;padding:3px 9px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.prov .k{color:var(--muted);text-transform:uppercase;letter-spacing:.05em;
  font-size:10.5px;margin-right:6px;font-family:system-ui,sans-serif}
.prov .none{color:var(--fail)}
.warnbar{margin:12px 0 0;padding:9px 12px;border-radius:8px;
  background:var(--fail-wash);border:1px solid var(--fail);
  font-size:13px;color:var(--ink)}

.hero{display:flex;flex-wrap:wrap;align-items:flex-end;gap:12px 40px}
.hero .big{font-size:52px;font-weight:650;line-height:1;letter-spacing:-.02em}
.hero .big .of{font-size:20px;font-weight:400;color:var(--muted)}
.tiles{display:flex;flex-wrap:wrap;gap:10px 34px}
.tile .v{font-size:23px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .l{font-size:10.5px;color:var(--muted);text-transform:uppercase;
  letter-spacing:.06em}
.tile.f .v{color:var(--fail)} .tile.p .v{color:var(--pass)}
.tile.s .v{color:var(--skip)}

.chart{display:block;max-width:100%;overflow:visible}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .axis{stroke:var(--axis);stroke-width:1}
.chart .bar path{fill:var(--seq)}
.chart .track{fill:var(--skip-wash)}
.chart text{font-family:system-ui,-apple-system,sans-serif}
.chart .cat{font-size:12.5px;fill:var(--ink2)}
.chart .tick{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}
.chart .val{font-size:12.5px;fill:var(--ink2);font-variant-numeric:tabular-nums}
.chart .seg{font-size:12px;fill:#fff;font-weight:600}
.chart .s-pass{fill:var(--pass);stroke:var(--surface);stroke-width:2}
.chart .s-fail{fill:var(--fail);stroke:var(--surface);stroke-width:2}
.chart .s-skip{fill:var(--skip);stroke:var(--surface);stroke-width:2}
.legend{display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:12px;font-size:12.5px;
  color:var(--ink2)}
.lg{display:inline-flex;align-items:center;gap:7px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.sw.s-pass{background:var(--pass)} .sw.s-fail{background:var(--fail)}
.sw.s-skip{background:var(--skip)}
.lgn{font-variant-numeric:tabular-nums;color:var(--ink);font-weight:600}

table.runs{border-collapse:collapse;width:100%;font-size:13.5px}
table.runs th,table.runs td{text-align:left;padding:7px 10px;
  border-bottom:1px solid var(--hair);vertical-align:top}
table.runs th{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);font-weight:600}
table.runs td.n{text-align:right;font-variant-numeric:tabular-nums}

.st{font-size:11px;font-weight:700;letter-spacing:.04em;padding:2px 7px;
  border-radius:5px;display:inline-block}
.st.PASS{background:var(--pass-wash);color:var(--pass)}
.st.FAIL{background:var(--fail-wash);color:var(--fail)}
.st.SKIP{background:var(--skip-wash);color:var(--ink2)}

details.run{background:var(--surface);border:1px solid var(--hair);
  border-radius:10px;margin:10px 0;overflow:hidden}
details.run[data-status="FAIL"]{border-left:3px solid var(--fail)}
details.run[data-status="SKIP"]{border-left:3px solid var(--skip)}
details.run[data-status="PASS"]{border-left:3px solid var(--pass)}
details.run>summary{cursor:pointer;padding:12px 18px;display:flex;
  flex-wrap:wrap;gap:6px 14px;align-items:center;list-style:none}
details.run>summary::-webkit-details-marker{display:none}
details.run>summary:hover{background:var(--code)}
summary .tname{font-weight:600}
summary .scheme{font-size:12px;color:var(--ink2);background:var(--code);
  border-radius:5px;padding:1px 7px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
summary .sc{margin-left:auto;font-variant-numeric:tabular-nums;color:var(--ink2);
  font-size:13px}
.run-body{padding:4px 18px 16px;border-top:1px solid var(--hair)}
.err{background:var(--fail-wash);border-radius:8px;padding:10px 12px;margin:12px 0;
  font-size:12.5px;white-space:pre-wrap;overflow-wrap:anywhere;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}

ul.checks{list-style:none;margin:10px 0 0;padding:0}
li.chk{padding:8px 0;border-top:1px solid var(--hair)}
li.chk:first-child{border-top:none}
.chk-h{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline}
.chk .mark{font-size:10px;font-weight:700;letter-spacing:.04em;padding:1px 6px;
  border-radius:4px}
.chk.ok .mark{background:var(--pass-wash);color:var(--pass)}
.chk.no .mark{background:var(--fail-wash);color:var(--fail)}
.cname{font-size:13px;font-weight:600}
.wt{font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
.detail{margin:5px 0 0 54px;font-size:12.5px;color:var(--ink2);
  white-space:pre-wrap;overflow-wrap:anywhere;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}

.cmp{margin:8px 0 2px 54px}
table.cmp-t{border-collapse:collapse;font-size:12.5px}
table.cmp-t th{font-size:10px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);text-align:left;padding:3px 14px 3px 0;font-weight:600}
table.cmp-t td{padding:3px 14px 3px 0;overflow-wrap:anywhere}
table.cmp-t td.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
table.cmp-t tr.no td{color:var(--fail)}
table.cmp-t tr.extra td{color:var(--ink2)}
td.vd{font-size:11.5px;color:var(--muted);white-space:nowrap}
.cmp-cap{font-size:11.5px;color:var(--muted);margin-top:4px;overflow-wrap:anywhere}
.cmp-pair{display:flex;flex-wrap:wrap;gap:8px}
.cmp-cell{background:var(--code);border-radius:7px;padding:6px 11px;min-width:128px}
.cmp-cell.no{background:var(--fail-wash)}
.cmp-cell.ok{background:var(--pass-wash)}
.cmp-k{display:block;font-size:10px;text-transform:uppercase;
  letter-spacing:.06em;color:var(--muted)}
.cmp-v{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:13px;overflow-wrap:anywhere}

.bar-tools{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 0}
button{font:inherit;font-size:12.5px;padding:5px 12px;border-radius:7px;
  border:1px solid var(--hair);background:var(--surface);color:var(--ink);
  cursor:pointer}
button:hover{background:var(--code)}
button[aria-pressed="true"]{background:var(--ink);color:var(--plane);
  border-color:var(--ink)}
.empty{color:var(--muted);font-size:13.5px;margin:4px 0}
footer{margin-top:40px;color:var(--muted);font-size:12px}
body.only-fail details.run:not([data-status="FAIL"]){display:none}
body.only-fail tr.run-row:not([data-status="FAIL"]){display:none}
"""

#: Collapse/filter only. The page is complete without it: failing runs are open
#: in the markup and passing ones are ordinary closed <details>, so a reader with
#: no JS loses the buttons and nothing else.
_JS = """
(function(){
  var body=document.body;
  var only=document.getElementById('f-fail');
  var all=document.getElementById('f-all');
  var ex=document.getElementById('f-expand');
  var co=document.getElementById('f-collapse');
  function setFilter(on){
    body.classList.toggle('only-fail',on);
    if(only)only.setAttribute('aria-pressed',on?'true':'false');
    if(all)all.setAttribute('aria-pressed',on?'false':'true');
  }
  if(only)only.addEventListener('click',function(){setFilter(true);});
  if(all)all.addEventListener('click',function(){setFilter(false);});
  function setOpen(open){
    var runs=document.querySelectorAll('details.run');
    for(var i=0;i<runs.length;i++){runs[i].open=open;}
  }
  if(ex)ex.addEventListener('click',function(){setOpen(true);});
  if(co)co.addEventListener('click',function(){setOpen(false);});
})();
"""


def _run_row(result: dict) -> str:
    status = _status(result)
    return (
        f'<tr class="run-row" data-status="{status}">'
        f'<td><span class="st {status}">{status}</span></td>'
        f'<td class="mono">{_e(result.get("task"))}</td>'
        f'<td class="mono">{_e(result.get("source_scheme"))}</td>'
        f'<td>{_e(result.get("family"))}</td>'
        f'<td>{_e(result.get("kind"))}</td>'
        f'<td class="n">{_score(result.get("score"))}</td>'
        f'<td class="n">{_num(result.get("duration_s")):.1f}s</td>'
        "</tr>"
    )


def _run_card(result: dict) -> str:
    status = _status(result)
    checks = result.get("checks")
    checks = [c for c in checks if isinstance(c, dict)] if isinstance(checks, list) else []
    # Failed checks first: inside a broken run, the broken assertions are what
    # the reader came for.
    checks = sorted(checks, key=lambda c: (bool(c.get("passed")), str(c.get("name", ""))))
    n_failed = sum(1 for c in checks if not c.get("passed"))

    error = result.get("error")
    error_html = ""
    if error:
        kind = "dataset or connector unavailable" if status == "SKIP" else "run error"
        error_html = (f'<div class="err"><strong>{_e(kind)}:</strong> '
                      f"{_e(str(error).strip())}</div>")

    if checks:
        checks_html = f'<ul class="checks">{"".join(_check_html(c) for c in checks)}</ul>'
    else:
        checks_html = ('<p class="empty">No checks were recorded for this run.</p>')

    artifacts = result.get("artifacts_dir")
    artifacts_html = (f'<p class="empty">artifacts: <span class="mono">'
                      f"{_e(artifacts)}</span></p>" if artifacts else "")

    summary_counts = (f"{n_failed} of {len(checks)} checks failed" if n_failed
                      else f"{len(checks)} checks passed")
    return (
        f'<details class="run" data-status="{status}"{" open" if status != "PASS" else ""}>'
        f'<summary><span class="st {status}">{status}</span>'
        f'<span class="tname mono">{_e(result.get("task"))}</span>'
        f'<span class="scheme">{_e(result.get("source_scheme"))}</span>'
        f'<span class="sc">{_e(summary_counts)} &nbsp;&middot;&nbsp; score '
        f'{_score(result.get("score"))}</span></summary>'
        f'<div class="run-body">{error_html}{checks_html}{artifacts_html}</div>'
        "</details>"
    )


def render_html(card: dict, *, title: str = "Avaloka benchmark scorecard") -> str:
    """Render a scorecard dict as one self-contained HTML document.

    ``card`` is the mapping :meth:`Scorecard.as_dict` returns, or anything with
    the same shape -- a ``scorecard.json`` loaded back off disk, say. Missing
    keys are tolerated: an empty card, or one where every run failed, renders as
    a page that says so rather than raising.
    """
    if not isinstance(card, dict):
        card = {}
    results = _results(card)
    ordered = _ordered(results)

    n_runs = len(results)
    n_pass = sum(1 for r in results if _status(r) == "PASS")
    n_skip = sum(1 for r in results if _status(r) == "SKIP")
    n_fail = n_runs - n_pass - n_skip

    overall = card.get("overall_score")
    if overall is None and results:
        overall = sum(_num(r.get("score")) for r in results) / len(results)
    overall = _num(overall)

    commit = card.get("commit")
    commit_html = (f'<span class="p">{_e(commit)}</span>' if commit
                   else '<span class="p none">not recorded</span>')
    schema = card.get("schema_version")
    generated = card.get("generated_at")

    # A declared run count that disagrees with the rows present means the file
    # was truncated or hand-edited. Say so rather than quietly reporting the
    # count of whatever survived.
    declared = card.get("n_runs")
    mismatch = ""
    if isinstance(declared, int) and declared != n_runs:
        mismatch = (f'<div class="warnbar">This scorecard declares '
                    f"<strong>{_e(declared)}</strong> runs but carries "
                    f"<strong>{n_runs}</strong>. Everything below counts the rows "
                    "that are present.</div>")
    if not commit:
        mismatch += ('<div class="warnbar">No commit is recorded on this '
                     "scorecard, so these numbers cannot be tied to a revision of "
                     "the code. Treat them as indicative, not as a result.</div>")

    by_family = _by_family(card, results)

    if ordered:
        rows = "".join(_run_row(r) for r in ordered)
        table = (
            '<table class="runs"><thead><tr><th></th><th>task</th><th>source</th>'
            "<th>family</th><th>kind</th><th>score</th><th>time</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
        cards = "".join(_run_card(r) for r in ordered)
        tools = (
            '<div class="bar-tools">'
            '<button id="f-all" aria-pressed="true">All runs</button>'
            f'<button id="f-fail" aria-pressed="false">Failures only ({n_fail})</button>'
            '<button id="f-expand">Expand all</button>'
            '<button id="f-collapse">Collapse all</button></div>'
        )
    else:
        table = ('<p class="empty">This scorecard contains no runs. Nothing was '
                 "exercised, so there is nothing to report.</p>")
        cards = ""
        tools = ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">

<h1>{_e(title)}</h1>
<div class="prov">
<span><span class="k">commit</span>{commit_html}</span>
<span><span class="k">schema</span><span class="p">{_e(schema) if schema is not None
    else "unknown"}</span></span>
<span><span class="k">generated</span><span class="p">{_e(generated) if generated
    else "unknown"}</span></span>
</div>
{mismatch}

<div class="card">
  <div class="hero">
    <div><div class="big">{_score(overall)}<span class="of"> / 1.00</span></div>
      <div class="tile"><div class="l">overall score</div></div></div>
    <div class="tiles">
      <div class="tile"><div class="v">{n_runs}</div><div class="l">runs</div></div>
      <div class="tile p"><div class="v">{n_pass}</div><div class="l">passed</div></div>
      <div class="tile f"><div class="v">{n_fail}</div><div class="l">failed</div></div>
      <div class="tile s"><div class="v">{n_skip}</div><div class="l">skipped</div></div>
    </div>
  </div>
</div>

<h2>Outcome of every run</h2>
<div class="card">{_breakdown_chart(n_pass, n_fail, n_skip)}
<p class="empty">A run is a skip, not a failure, when its error says the dataset
or connector was unavailable &mdash; it was never exercised.</p></div>

<h2>Score by family</h2>
<div class="card">{_family_chart(by_family)}</div>

<h2>Runs</h2>
<div class="card">{table}</div>

<h2>Checks, failures first</h2>
{tools}
{cards}

<footer>Rendered by <span class="mono">avaloka.benchmark.report</span>. Self-contained:
no external styles, scripts or fonts.</footer>
</div>
<script>{_JS}</script>
</body></html>
"""


def write_html(card: dict, path: Path,
               *, title: str = "Avaloka benchmark scorecard") -> Path:
    """Render ``card`` and write it to ``path``, returning the path written.

    Creates the parent directory, so ``--html out/run-7/scorecard.html`` works
    against a directory the run has not made yet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(card, title=title), encoding="utf-8")
    return path
