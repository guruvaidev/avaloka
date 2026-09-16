"""Render + open avaloka visualizations from non-browser surfaces (CLI, MCP).

The backend represents a visualization as a JSON *chart spec* (``visualization_config``
with a ``charts`` list; each chart's encodings are Vega-Lite-flavored) — not an image.
Terminals can't render that, so this module turns a chart spec into a self-contained
HTML page (Vega-Lite via CDN) that a browser can open, and provides open/link helpers
plus a fetcher for the one real image the backend serves (the planner-graph PNG).

Pure and dependency-light so it is easy to unit-test:
  * visualization_to_html(config, rows) -> str      (no I/O)
  * write_artifact(text|bytes, suffix) -> Path
  * to_file_url(path) -> "file://..."
  * open_artifact(path_or_url, mode) -> str          (mode: browser|link|none)
  * resolve_open_mode(flag) -> mode                  (honours a TTY / env)
  * fetch_planner_graph(endpoint, thread_id, token) -> Path
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import webbrowser
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

# Custom chart "type" -> Vega-Lite mark. The encodings themselves are already
# Vega-Lite-compatible (field/type/bin/aggregate), so they pass through as-is.
_MARK = {
    "histogram": "bar",
    "bar": "bar",
    "line": "line",
    "scatter": "point",
    "pie": "arc",
    "area": "area",
}

_VEGA_CDN = (
    "https://cdn.jsdelivr.net/npm/vega@5",
    "https://cdn.jsdelivr.net/npm/vega-lite@5",
    "https://cdn.jsdelivr.net/npm/vega-embed@6",
)


def _json_for_inline_script(value: Any) -> str:
    """Serialize JSON without allowing data to terminate an HTML script tag.

    ``json.dumps`` is valid JavaScript but does not escape ``</script>``. CSV
    cells and MCP arguments are untrusted inputs, so encode HTML-significant
    characters before placing the object literal in an executable script.
    """
    return (
        json.dumps(value)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _chart_to_vegalite(chart: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map one avaloka chart spec + the sample rows to a Vega-Lite spec."""
    spec: Dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": chart.get("title", chart.get("id", "chart")),
        "data": {"values": rows},
        "mark": {"type": _MARK.get((chart.get("type") or "").lower(), "bar"), "tooltip": True},
        "width": "container",
        "height": 260,
    }
    enc = chart.get("encodings")
    if isinstance(enc, dict) and enc:
        spec["encoding"] = enc
    return spec


def visualization_to_html(
    config: Dict[str, Any],
    rows: Optional[List[Dict[str, Any]]] = None,
    *,
    title: str = "Avaloka visualization",
) -> str:
    """Return a self-contained HTML page rendering every chart in ``config``.

    ``rows`` is the sample data the charts encode (the backend's ``visualization_config``
    references ``data_source: "sample"`` and usually leaves ``derived_data`` empty). If
    a chart carries non-empty ``derived_data`` with a ``values`` list, that wins.
    """
    rows = rows or []
    charts = config.get("charts") or []
    specs = []
    for c in charts:
        dd = c.get("derived_data") or {}
        chart_rows = dd.get("values") if isinstance(dd, dict) and dd.get("values") else rows
        specs.append(_chart_to_vegalite(c, chart_rows))

    scripts = "\n".join(f'<script src="{u}"></script>' for u in _VEGA_CDN)
    cards = "\n".join(
        f'<section class="card"><div id="viz{i}" class="viz"></div></section>'
        for i in range(len(specs))
    )
    embed = "\n".join(
        f"vegaEmbed('#viz{i}', {_json_for_inline_script(s)}, {{actions:false}});"
        for i, s in enumerate(specs)
    )
    empty = "" if specs else '<p class="muted">No charts in this visualization.</p>'
    safe_title = escape(str(title), quote=True)
    safe_status = escape(str(config.get("visualization_status", "")), quote=True)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{safe_title}</title>
{scripts}
<style>
  body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0b1020;color:#e7ecf5}}
  header{{padding:16px 24px;border-bottom:1px solid #223}}
  h1{{font-size:18px;margin:0}} .muted{{color:#8b95a7}}
  main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px;padding:24px}}
  .card{{background:#121a30;border:1px solid #223;border-radius:12px;padding:12px}}
  .viz{{width:100%}}
</style></head>
<body>
  <header><h1>{safe_title}</h1><div class="muted">{safe_status}</div></header>
  <main>{cards}{empty}</main>
  <script>{embed}</script>
</body></html>
"""


def to_file_url(path: os.PathLike) -> str:
    return Path(path).resolve().as_uri()


def write_artifact(content, *, suffix: str = ".html", out_dir: Optional[os.PathLike] = None) -> Path:
    """Write text or bytes to a file and return its path."""
    d = Path(out_dir) if out_dir else Path(tempfile.gettempdir()) / "avaloka-artifacts"
    d.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
    fd, name = tempfile.mkstemp(suffix=suffix, dir=str(d))
    os.close(fd)
    p = Path(name)
    if isinstance(content, (bytes, bytearray)):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")
    return p


def resolve_open_mode(flag: Optional[str] = None) -> str:
    """Decide how to surface an artifact.

    Explicit ``flag`` (browser|link|none) wins; else AVALOKA_OPEN env; else open a
    browser when attached to a TTY, otherwise just print a link (headless/CI/piped).
    """
    val = (flag or os.getenv("AVALOKA_OPEN") or "").strip().lower()
    if val in ("browser", "link", "none"):
        return val
    return "browser" if sys.stdout.isatty() else "link"


def open_artifact(path_or_url: str, *, mode: str = "browser") -> str:
    """Open the artifact per ``mode`` and return the URL/link used.

    browser -> webbrowser.open (falls back to link if no browser); link -> return the
    file://-style URL without opening; none -> just return it.
    """
    url = path_or_url
    if "://" not in str(path_or_url):
        url = to_file_url(path_or_url)
    if mode == "browser":
        try:
            if webbrowser.open(url):
                return url
        except Exception:
            pass  # headless: fall through to returning the link
    return url


def fetch_planner_graph(
    endpoint: str,
    thread_id: str,
    *,
    token: Optional[str] = None,
    out_dir: Optional[os.PathLike] = None,
) -> Path:
    """GET the planner-graph PNG for a thread from a deployed API and save it."""
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = endpoint.rstrip("/") + f"/threads/{thread_id}/planner-graph"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return write_artifact(resp.content, suffix=".png", out_dir=out_dir)
