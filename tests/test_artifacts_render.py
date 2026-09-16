"""Tests for the shared CLI/MCP artifact render+open helpers (app/interfaces/artifacts)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.interfaces import artifacts

SAMPLE_ROWS = [
    {"country": "USA", "price": 10, "qty": 1},
    {"country": "UK", "price": 20, "qty": 2},
]

CONFIG = {
    "visualization_status": "ready",
    "charts": [
        {"id": "c1", "title": "Price distribution", "type": "histogram",
         "encodings": {"x": {"field": "price", "type": "quantitative", "bin": True},
                       "y": {"aggregate": "count", "type": "quantitative"}}, "derived_data": {}},
        {"id": "c2", "title": "Price by qty", "type": "scatter",
         "encodings": {"x": {"field": "qty", "type": "quantitative"},
                       "y": {"field": "price", "type": "quantitative"}}, "derived_data": {}},
    ],
}


# --------------------------------------------------------------- visualization_to_html
def test_html_embeds_vega_and_one_chart_per_spec():
    html = artifacts.visualization_to_html(CONFIG, SAMPLE_ROWS, title="My Viz")
    assert "My Viz" in html
    assert "vega-embed" in html and html.count("vegaEmbed(") == 2
    assert html.count('class="viz"') == 2
    # sample rows are inlined for Vega-Lite to plot
    assert "USA" in html and "price" in html


def test_html_empty_when_no_charts():
    html = artifacts.visualization_to_html({"charts": []}, [])
    assert "No charts" in html
    assert "vegaEmbed(" not in html


@pytest.mark.parametrize("ctype,mark", [
    ("histogram", "bar"), ("bar", "bar"), ("line", "line"),
    ("scatter", "point"), ("pie", "arc"), ("unknown", "bar"),
])
def test_chart_type_maps_to_vegalite_mark(ctype, mark):
    spec = artifacts._chart_to_vegalite({"type": ctype, "title": "t", "encodings": {}}, [])
    assert spec["mark"]["type"] == mark


def test_derived_data_values_override_sample_rows():
    cfg = {"charts": [{"type": "bar", "title": "t", "encodings": {},
                       "derived_data": {"values": [{"x": "only-derived"}]}}]}
    html = artifacts.visualization_to_html(cfg, SAMPLE_ROWS)
    assert "only-derived" in html


def test_html_escapes_script_breakout_from_rows_and_metadata():
    payload = '</script><script>globalThis.pwned = true</script>'
    cfg = {
        "visualization_status": '<img src=x onerror="globalThis.statusPwned=true">',
        "charts": [{"type": "bar", "encodings": {"x": {"field": "x"}}}],
    }
    html = artifacts.visualization_to_html(
        cfg,
        [{"x": payload}],
        title='<svg onload="globalThis.titlePwned=true">',
    )

    assert payload not in html
    assert "</script><script>" not in html
    assert "\\u003c/script\\u003e" in html
    assert "<svg onload=" not in html
    assert "<img src=x" not in html


# --------------------------------------------------------------- write / url
def test_write_artifact_text_and_bytes(tmp_path):
    p1 = artifacts.write_artifact("<html></html>", suffix=".html", out_dir=tmp_path)
    assert p1.suffix == ".html" and p1.read_text() == "<html></html>"
    p2 = artifacts.write_artifact(b"\x89PNG", suffix=".png", out_dir=tmp_path)
    assert p2.read_bytes() == b"\x89PNG"


def test_to_file_url(tmp_path):
    p = tmp_path / "x.html"
    p.write_text("x")
    assert artifacts.to_file_url(p).startswith("file://")
    assert artifacts.to_file_url(p).endswith("x.html")


# --------------------------------------------------------------- open mode resolution
def test_resolve_open_mode_explicit_wins(monkeypatch):
    monkeypatch.delenv("AVALOKA_OPEN", raising=False)
    assert artifacts.resolve_open_mode("link") == "link"
    assert artifacts.resolve_open_mode("none") == "none"


def test_resolve_open_mode_env_then_tty(monkeypatch):
    monkeypatch.setenv("AVALOKA_OPEN", "browser")
    assert artifacts.resolve_open_mode(None) == "browser"
    monkeypatch.delenv("AVALOKA_OPEN", raising=False)
    with patch("app.interfaces.artifacts.sys.stdout.isatty", return_value=False):
        assert artifacts.resolve_open_mode(None) == "link"   # headless -> link
    with patch("app.interfaces.artifacts.sys.stdout.isatty", return_value=True):
        assert artifacts.resolve_open_mode(None) == "browser"


# --------------------------------------------------------------- open_artifact
def test_open_artifact_browser_calls_webbrowser(tmp_path):
    p = tmp_path / "a.html"
    p.write_text("x")
    with patch("app.interfaces.artifacts.webbrowser.open", return_value=True) as wb:
        url = artifacts.open_artifact(str(p), mode="browser")
    wb.assert_called_once()
    assert url.startswith("file://")


def test_open_artifact_link_does_not_open(tmp_path):
    p = tmp_path / "a.html"
    p.write_text("x")
    with patch("app.interfaces.artifacts.webbrowser.open") as wb:
        url = artifacts.open_artifact(str(p), mode="link")
    wb.assert_not_called()
    assert url.startswith("file://")


def test_open_artifact_browser_falls_back_to_link_when_headless(tmp_path):
    p = tmp_path / "a.html"
    p.write_text("x")
    with patch("app.interfaces.artifacts.webbrowser.open", return_value=False):
        url = artifacts.open_artifact(str(p), mode="browser")
    assert url.startswith("file://")   # no crash; returns the link


# --------------------------------------------------------------- planner-graph fetch
def test_fetch_planner_graph_saves_png(tmp_path):
    class FakeResp:
        content = b"\x89PNG\r\n"
        def raise_for_status(self): pass

    with patch("requests.get", return_value=FakeResp()) as get:
        png = artifacts.fetch_planner_graph("http://api:9000", "th1", token="jwt", out_dir=tmp_path)
    assert Path(png).exists() and Path(png).read_bytes().startswith(b"\x89PNG")
    called_url = get.call_args.args[0]
    assert called_url == "http://api:9000/threads/th1/planner-graph"
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer jwt"
