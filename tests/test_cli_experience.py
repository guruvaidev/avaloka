"""CLI experience tests — every subcommand, the viz render/open flow, and clean errors.

Runs `python -m app.interfaces.cli.main ...` as a subprocess so exit codes and
stdout/stderr separation are exercised exactly as a user sees them.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# sales_data.csv was never committed; salaries.csv is the shipped fixture with
# the numeric + categorical columns these visualisation assertions need.
SAMPLE = "app/sample_data/salaries.csv"


def cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "app.interfaces.cli.main", *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )


# ----------------------------------------------------------------- plan / analyze
def test_plan_json_has_mission_plan_estimate():
    r = cli("plan", SAMPLE, "--goal", "Explain churn", "--rows", "84000000", "--json")
    assert r.returncode == 0, r.stderr
    assert set(json.loads(r.stdout)) == {"mission", "plan", "estimate"}


def test_analyze_prints_a_human_plan():
    r = cli("analyze", SAMPLE, "--goal", "Explain churn")
    assert r.returncode == 0, r.stderr
    assert "Mission plan" in r.stdout


def test_no_subcommand_is_a_clean_argparse_error():
    r = cli()
    assert r.returncode == 2
    assert "usage:" in (r.stderr + r.stdout).lower()


# ----------------------------------------------------------------- viz
def test_viz_json_emits_a_visualization_config():
    r = cli("viz", SAMPLE, "--json")
    assert r.returncode == 0, r.stderr
    cfg = json.loads(r.stdout)  # stdout stays clean; the "LLM disabled" note is on stderr
    assert "charts" in cfg
    assert cfg.get("visualization_status") == "ready"
    assert len(cfg["charts"]) >= 1


def test_viz_link_renders_html_and_prints_a_file_url(tmp_path):
    r = cli("viz", SAMPLE, "--open", "link", "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "rendered" in r.stdout and "file://" in r.stdout
    htmls = list(tmp_path.glob("*.html"))
    assert htmls, "no HTML artifact written"
    body = htmls[0].read_text()
    assert "vega-embed" in body and "vegaEmbed(" in body


def test_viz_none_mode_does_not_open_but_still_reports(tmp_path):
    r = cli("viz", SAMPLE, "--open", "none", "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "rendered" in r.stdout


def test_viz_missing_source_is_a_clean_error():
    r = cli("viz")
    assert r.returncode != 0
    assert "error:" in r.stderr.lower()
    assert "Traceback" not in r.stderr


def test_viz_planner_graph_requires_endpoint():
    r = cli("viz", "--planner-graph", "th1")
    assert r.returncode != 0
    assert "endpoint" in r.stderr.lower()
    assert "Traceback" not in r.stderr


# ----------------------------------------------------------------- in-process viz (mocked open)
def test_cmd_viz_opens_browser_via_resolve(monkeypatch, tmp_path):
    # Exercise the code path directly with a fake TTY so it chooses "browser",
    # and assert we don't actually spawn one in CI.
    from app.interfaces.cli import main as cli_main
    from app.interfaces import artifacts

    opened = {}
    monkeypatch.setattr(artifacts.webbrowser, "open", lambda url: opened.setdefault("url", url) or True)
    monkeypatch.setattr(artifacts.sys.stdout, "isatty", lambda: True)

    args = cli_main.build_parser().parse_args(["viz", SAMPLE, "--out", str(tmp_path)])
    rc = cli_main.cmd_viz(args)
    assert rc == 0
    assert opened.get("url", "").startswith("file://")
