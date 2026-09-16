"""T4d — CLI (in-process + remote parity).

Runs the CLI as a subprocess (real exit codes). Remote-parity tests (T4d.5/.6)
skip cleanly unless AVALOKA_API_URL points at a running API; the in-process
parity invariant is proven directly (T4d.7) since the CLI, MCP tool and the
/api/missions/plan route all return the same planned_to_dict().
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = "app/sample_data/sales_data.csv"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "app.interfaces.cli.main", *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )


def _plan_json(*extra: str) -> dict:
    r = run_cli("plan", SAMPLE, "--goal", "Explain churn drivers", *extra, "--json")
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


# --------------------------------------------------------------------------- T4d.1
def test_t4d_1_plan_json_shape():
    r = run_cli("plan", SAMPLE, "--goal", "Explain churn drivers",
                "--rows", "84000000", "--max-cost", "20", "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert set(d) == {"mission", "plan", "estimate"}


# --------------------------------------------------------------------------- T4d.2
def test_t4d_2_rows_push_mode():
    big = _plan_json("--rows", "84000000")
    small = _plan_json("--rows", "1000")
    # Large rows -> sampled/distributed; small -> local, interactive.
    assert big["estimate"]["recommended_mode"] in {"sampled", "hybrid", "batch"}
    assert small["plan"]["target"] == "local"
    assert small["estimate"]["recommended_mode"] in {"interactive", "auto"}


# --------------------------------------------------------------------------- T4d.3
def test_t4d_3_execution_and_context_reflected():
    d = _plan_json("--execution", "ray", "--cluster-context", "kind-avaloka",
                   "--rows", "84000000", "--mode", "batch")
    # full-data engine surfaces as the plan target...
    assert d["plan"]["target"] == "ray"
    # ...and the requested cluster context is captured in the compiled mission.
    assert "kind-avaloka" in json.dumps(d["mission"])


# --------------------------------------------------------------------------- T4d.4
def test_t4d_4_over_budget_requires_approval():
    # The approval gate fires for a full-data pass (batch/hybrid) over budget.
    d = _plan_json("--rows", "84000000", "--mode", "batch", "--max-cost", "0.01")
    assert d["plan"]["requires_approval"] is True
    assert d["plan"]["approval_reason"]


# --------------------------------------------------------------------------- T4d.7
def test_t4d_7_cli_equals_mcp(monkeypatch):
    from app.interfaces.cli.main import build_intent, build_parser
    from app.interfaces.service import plan_mission, planned_to_dict
    from app.interfaces.mcp.server import tool_plan_mission

    # This asserts the shared LOCAL computation is identical across surfaces; make
    # sure an ambient AVALOKA_API_URL doesn't send the MCP tool to a remote backend.
    monkeypatch.delenv("AVALOKA_API_URL", raising=False)

    args = build_parser().parse_args(
        ["plan", SAMPLE, "--goal", "Explain churn drivers", "--rows", "84000000", "--max-cost", "20"])
    intent = build_intent(args)
    cli = json.dumps(planned_to_dict(plan_mission(intent)), sort_keys=True)
    mcp = json.dumps(tool_plan_mission(intent), sort_keys=True)
    assert cli == mcp


# --------------------------------------------------------------------------- T4d.8
def test_t4d_8_unreachable_endpoint_clean_error():
    r = run_cli("plan", SAMPLE, "--goal", "g",
                "--endpoint", "http://127.0.0.1:59999", "--json")
    assert r.returncode != 0
    assert "error:" in r.stderr.lower()
    assert "Traceback" not in r.stderr  # a clean message, not a stack trace


# ----------------------------------------------------------------------- T4d.5/.6
@pytest.mark.skipif(not os.getenv("AVALOKA_API_URL"),
                    reason="set AVALOKA_API_URL + AVALOKA_API_TOKEN to a running API for live parity")
def test_t4d_5_remote_parity_live():
    endpoint = os.getenv("AVALOKA_API_URL")
    token = os.getenv("AVALOKA_API_TOKEN")
    local = run_cli("plan", SAMPLE, "--goal", "Explain churn drivers",
                    "--rows", "84000000", "--max-cost", "20", "--json")
    remote = run_cli("plan", SAMPLE, "--goal", "Explain churn drivers",
                     "--rows", "84000000", "--max-cost", "20", "--json",
                     "--endpoint", endpoint, "--token", token or "")
    assert local.returncode == 0 and remote.returncode == 0, remote.stderr
    assert local.stdout == remote.stdout  # byte-identical
