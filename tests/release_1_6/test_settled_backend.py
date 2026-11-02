"""1.6 release suite: rows whose expected behaviour was settled from the code and
docs after the owner's answers to the plan's open questions.

Each test names the case it certifies; the evidence for the settled reading is in
its docstring and in docs/test-reports/1.6-traceability.csv (``note``). Hermetic:
no network, no keys, no cluster.
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from tests.server_harness import client  # noqa: F401  (fixture for UI-11)

# ---------------------------------------------------------------------------
# K8-12: an OOM is diagnosed, not just failed
# ---------------------------------------------------------------------------


def test_k812_oom_failure_is_classified_with_a_remediation():
    """K8-12: a training run killed for memory is classified OOM_KILLED with a
    memory remediation, not reported as a bare failure.

    Settled in scope for 1.6: app/agents/mta_v2/failure_diagnostics.py ships on
    develop-1.6 with production callers (ray_trainer.py, mta_v2/agent.py,
    celery_app.py, server.py). The existing test pins pod/exit-code fields; this
    one pins the remediation the case asks for.
    """
    from app.agents.mta_v2.failure_diagnostics import build_training_failure

    failure = build_training_failure("Ray worker container exceeded its memory limit and was OOMKilled")

    assert failure["code"] == "OOM_KILLED"
    assert failure["retryable"] is True
    assert any("memory" in action.lower() for action in failure["actions"])
    assert "Increase the training worker memory limit" in failure["user_message"]


# ---------------------------------------------------------------------------
# DQ-02: integrity checks are wired into the MTA training path
# ---------------------------------------------------------------------------


def _leaking_frame() -> pd.DataFrame:
    # 12 distinct rows (every Pclass x Deck x Survived combination), so neither the
    # duplicate-row nor the identifier check fires first; Survived_leak copies the target.
    rows = [(p, d, s) for p in (1, 2, 3) for d in ("A", "B") for s in (0, 1)]
    frame = pd.DataFrame(rows, columns=["Pclass", "Deck", "Survived"])
    frame["Survived_leak"] = frame["Survived"]
    return frame


def test_dq02_local_training_refuses_a_leaking_dataset(tmp_path, monkeypatch):
    """DQ-02: the MTA training path gates on the integrity checks: a feature that
    copies the target blocks training with a target-leakage error, before any
    MLflow run starts.

    Settled reading of "wired into the MTA path": app/agents/integrity_agent.py has
    no callers, but LocalTrainer.train -> training_main runs the shared detectors
    (app/agents/mta_v2/local_trainer.py, find_target_leakage before preprocessing).
    That behaviour, not the module name, is what the case certifies.
    """
    import mlflow

    import app.agents.mta_v2.local_trainer as lt

    csv = tmp_path / "titanic_leak.csv"
    _leaking_frame().to_csv(csv, index=False)
    started: List[Any] = []
    monkeypatch.setattr(lt, "setup_mlflow", lambda: None)

    def _no_run(*a, **k):
        started.append((a, k))
        raise RuntimeError("an MLflow run was started")

    monkeypatch.setattr(mlflow, "start_run", _no_run)

    plan = {
        "model_type": "classification",
        "data_config": {
            "data_source_location": str(csv),
            "target_column": "Survived",
            "feature_columns": ["Pclass", "Deck", "Survived_leak"],
        },
        "hyperparameter_config": {"epochs": 1, "batch_size": 4},
    }
    result = lt.LocalTrainer(user_id="user-1", session_id="s-1").train(plan)

    assert result["status"] == "error"
    assert "target leakage" in result["error"].lower()
    assert "Survived_leak" in result["error"]
    assert started == []


# ---------------------------------------------------------------------------
# DQ-09: the small-integer exemption covers prose counts only
# ---------------------------------------------------------------------------


def test_dq09_prose_count_stays_exempt():
    """DQ-09: a bare small count in prose ("3 segments") is not treated as an
    invented statistic (the exemption's stated purpose, claim_verifier.py)."""
    from app.agents.claim_verifier import check_invented_numbers

    assert check_invented_numbers("The data splits into 3 segments.", {"mean_spend": 12.4}) is None


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "DQ-09: check_invented_numbers exempts any non-percentage value below 10 whose float "
    "is an integer, so a figure written as '8.0' is never checked against the evidence"))
def test_dq09_decimal_written_statistic_is_verified():
    """DQ-09: a figure written with a decimal part ("8.0") is a statistic, not a
    prose count, and is flagged when the evidence does not contain it.

    Settled from app/agents/claim_verifier.py (comment on the exemption: it exists for
    "ordinary prose counts ('3 segments') and calendar years") and the plan's case text
    (float('8.0').is_integer() must not exempt a stated statistic).
    """
    from app.agents.claim_verifier import check_invented_numbers

    finding = check_invented_numbers("Average order value was 8.0 this quarter.", {"mean_order_value": 12.4})

    assert finding is not None
    assert finding.check == "invented_number"
    assert "8.0" in finding.detail


# ---------------------------------------------------------------------------
# DQ-16: an all-null column is not dropped silently
# ---------------------------------------------------------------------------


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "DQ-16: avaloka/io/loader.py load_dataset runs frame.dropna(axis=1, how='all') and "
    "records nothing, so an all-empty column disappears without a trace"))
def test_dq16_all_null_column_is_not_dropped_silently(tmp_path):
    """DQ-16: a 100%-null column is either kept or its removal is recorded on the
    loaded dataset; it never vanishes silently.

    Settled reading: the loader comment calls the drop a deliberate normalisation of
    dirty exports, and the plan case asks that the column not be dropped *silently*;
    both hold if the drop is recorded. The app sampler already keeps and flags it
    (tests/test_sampler_stats_regression.py::test_all_null_column_is_preserved_and_flagged).
    """
    from avaloka.io.loader import load_dataset

    csv = tmp_path / "with_empty.csv"
    csv.write_text("amount,region,notes\n1,north,\n2,south,\n3,east,\n4,west,\n")

    handle = load_dataset(str(csv))

    assert list(handle.frame.columns[:2]) == ["amount", "region"], "precondition: the file was read"
    # ``read`` carries the per-column na_values map, which names every column read;
    # only a record outside it counts as reporting the drop.
    recorded = {k: v for k, v in (handle.read or {}).items() if k != "na_values"}
    assert "notes" in handle.frame.columns or "notes" in repr(recorded)


# ---------------------------------------------------------------------------
# UI-03: an oversize upload is refused with a message naming the limit
# ---------------------------------------------------------------------------


def test_ui03_oversize_message_names_the_per_file_limit():
    """UI-03: the 413 body for a file over the limit names the plan and its per-file
    limit in MB; on Free it also offers the upgrade path.

    Settled from docs/USER_GUIDE.md §3 ("up to 100 MB per file") and
    app/core/plan_limits.py: the message names the limit; it does not suggest a
    connection, because connections are a paid capability (USER_GUIDE §5) and the
    Free path offers "reduce or upgrade" instead.
    """
    from app.core.plan_limits import MB, limit_error, limits_for

    pro = limits_for("professional")
    assert pro.max_file_bytes == 100 * MB
    body = limit_error(pro, "file_size", 150 * MB)
    assert body["reason"] == "file_size"
    assert body["limit"] == 100 * MB
    assert "150 MB" in body["message"] and "100 MB per file" in body["message"]
    assert body["upgrade_available"] is False

    free = limit_error(limits_for("free"), "file_size", 150 * MB)
    assert "100 MB" in free["message"]
    assert free["upgrade_available"] is True
    assert "upgrade" in free["message"].lower()


# ---------------------------------------------------------------------------
# UI-04: the Quick Insights briefing carries every documented section
# ---------------------------------------------------------------------------

_PROFILE = {
    "status": "success",
    "domain": {"category": "Real Estate", "reasoning": "prices and rooms"},
    "quick_insights": {
        "data_readiness_score": 82.4,
        "readiness_reasoning": "Few missing values; one skewed column.",
        "key_relationships": [
            {"columns": ["median_income", "median_house_value"], "relationship": "move together"},
        ],
        "red_flags": ["total_bedrooms has 207 missing values"],
        "quick_wins": ["Impute total_bedrooms with the median"],
    },
    "data_quality_insights": {"strengths": ["s"], "concerns": ["c"]},
}


def test_ui04_overview_carries_score_reasoning_flags_and_wins():
    """UI-04: the profile overview shown in the UI carries the readiness score with
    its reasoning, the red flags and the quick wins (docs/USER_GUIDE.md §3)."""
    from app.agents.visualization_agent import build_profile_overview

    overview = build_profile_overview(_PROFILE)

    assert overview["readiness_score"] == 82
    assert overview["readiness_reasoning"] == "Few missing values; one skewed column."
    assert overview["red_flags"] == ["total_bedrooms has 207 missing values"]
    assert overview["quick_wins"] == ["Impute total_bedrooms with the median"]


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "UI-04: build_profile_overview (app/agents/visualization_agent.py) drops the profiler's "
    "quick_insights.key_relationships, so the documented Key Relationships section never reaches the UI"))
def test_ui04_overview_carries_key_relationships():
    """UI-04: the overview carries Key Relationships, which docs/USER_GUIDE.md §3 and
    docs/TECHNICAL_USER_GUIDE.md §3.2 list as part of the briefing and the profiler
    produces (app/agents/profiling_agent.py prompt schema)."""
    from app.agents.visualization_agent import build_profile_overview

    overview = build_profile_overview(_PROFILE)
    relationships = next((v for k, v in overview.items() if "relationship" in k.lower()), None)

    assert relationships, "no relationships field in the overview"
    assert "median_income" in repr(relationships) and "median_house_value" in repr(relationships)


# ---------------------------------------------------------------------------
# UI-11: inference returns the prediction for the requested run
# ---------------------------------------------------------------------------


@pytest.fixture()
def inference_client(monkeypatch):
    import app.api.server as server

    async def _allowed(user_id, feature):
        return None

    monkeypatch.setattr(server, "_require_premium_plan", _allowed)
    calls: List[Any] = []
    models: Dict[str, Dict[str, Any]] = {
        "run-owned": {"user_id": "ui11-user", "inference_service_details": {"endpoint": "svc"}},
        "run-foreign": {"user_id": "someone-else", "inference_service_details": {"endpoint": "svc"}},
        "run-unconfigured": {"user_id": "ui11-user", "inference_service_details": None},
    }

    class FakeMLflowManager:
        def get_model_details(self, run_id):
            return models.get(run_id)

    class FakeInferenceServiceManager:
        def inference(self, run_id, input_data):
            calls.append((run_id, input_data))
            return {"prediction": [0.73], "model_run": run_id}

    monkeypatch.setattr("app.agents.mta_v2.mlflow_manager.MLflowManager", FakeMLflowManager)
    monkeypatch.setattr(
        "app.agents.mta_v2.inference_service_manager.InferenceServiceManager", FakeInferenceServiceManager
    )
    return calls


def test_ui11_inference_returns_the_prediction_for_the_requested_run(client, inference_client):
    """UI-11: POST /api/models/{run_id}/inference returns the prediction produced by
    that run's service (the run is bound by the URL; the service is called with it),
    refuses another user's model (403) and an unconfigured one (409).

    Settled: docs/api.md lists the endpoint without a response echo of run_id, and the
    run_id is the path parameter the client already holds, so "with the run_id it came
    from" is satisfied by the URL binding asserted here. The live path needs a trained
    model on a cluster (manual).
    """
    from tests.server_harness import make_auth_headers

    headers = make_auth_headers("ui11-user")
    resp = client.post("/api/models/run-owned/inference", json={"x": 1.5}, headers=headers)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"result": {"prediction": [0.73], "model_run": "run-owned"}}
    assert inference_client == [("run-owned", {"x": 1.5})]

    assert client.post("/api/models/run-foreign/inference", json={"x": 1}, headers=headers).status_code == 403
    assert client.post("/api/models/run-unconfigured/inference", json={"x": 1}, headers=headers).status_code == 409
    assert len(inference_client) == 1, "a refused request must not reach the inference service"


# ---------------------------------------------------------------------------
# CM-03: small talk and consent words never launch a swarm mission
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_csv(tmp_path):
    rows = ["tenure,monthly,tickets,churned"]
    for i in range(40):
        rows.append(f"{i % 24},{20 + (i * 7) % 60},{i % 5},{1 if i % 3 == 0 else 0}")
    path = tmp_path / "churn_small.csv"
    path.write_text("\n".join(rows) + "\n")
    return path


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "CM-03: avaloka/chat.py treats any unrecognised input as an analysis goal "
    "('Otherwise: treat it as the goal and analyse') and launches the swarm mission"))
@pytest.mark.parametrize("utterance", ["yes", "ok", "thanks", "hello", "sure"])
def test_cm03_small_talk_does_not_launch_a_mission(small_csv, tmp_path, monkeypatch, utterance):
    """CM-03: a consent word or small talk in `avaloka chat` gets a reply, not a swarm
    mission (no ./avaloka-analysis output is written).

    Settled from the plan's case text ("must NOT launch a swarm mission"). The
    researcher's original 16 inputs are not in the tree; these five are representative
    consent/small-talk utterances.
    """
    from typer.testing import CliRunner

    from avaloka.cli import app

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["chat", str(small_csv)], input=f"{utterance}\nquit\n")

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "avaloka-analysis").exists()


# ---------------------------------------------------------------------------
# CM-05: the documented MCP server exposes its tools
# ---------------------------------------------------------------------------

_INTENT = {"source_uri": "customers.parquet", "goal_description": "Explain churn",
           "rows": 84_000_000, "max_cost_usd": 20}


def test_cm05_built_mcp_server_lists_and_calls_its_tools():
    """CM-05: the MCP server documented in docs/cli.md ("MCP server") builds, lists its
    five tools, and a listed tool is callable through the server.

    Settled: the plan cites docs/cli.md, which documents app/interfaces/mcp/server.py;
    app/mcp_server/ is the customer-database registry (docs/api.md "two unrelated sets"),
    whose unmounted SSE endpoint is inventory BLO-096, tracked separately.
    """
    from app.interfaces.mcp import server as mcp

    srv = mcp.build_server()
    names = sorted(t.name for t in asyncio.run(srv.list_tools()))

    assert len(names) == 5
    assert {"estimate_cost", "describe_environment", "inspect_intent", "visualize"} <= set(names)
    out = asyncio.run(srv.call_tool("estimate_cost", {"intent": _INTENT}))
    assert "recommended_mode" in repr(out) and "workload_score" in repr(out)


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "CM-05: TOOL_SPECS declares 'plan_mission' but build_server registers the function "
    "name 'plan_mission_tool' (docs/cli.md MCP server note), so clients see a different name"))
def test_cm05_listed_tool_names_match_the_specs():
    """CM-05: the names a client lists are the names TOOL_SPECS advertises
    (docs/cli.md: "if you are fixing the code, pass an explicit name= so the two agree")."""
    from app.interfaces.mcp import server as mcp

    listed = {t.name for t in asyncio.run(mcp.build_server().list_tools())}

    assert listed == {s["name"] for s in mcp.TOOL_SPECS}


# ---------------------------------------------------------------------------
# CM-07: "what is in this datasource" is answered natively
# ---------------------------------------------------------------------------


def test_cm07_dataset_discovery_describes_the_bound_dataset():
    """CM-07: with a dataset bound to the thread, discovery describes it (id, type and
    columns) without any external service.

    Settled: the cited openclaw skill path is not in the 1.6 tree; openclaw is an
    optional sidecar and app/agents/avaloka_agent/discovery_tools.py is the shipped
    native discovery the conversational agent calls (avaloka_agent/agent.py).
    """
    from app.agents.avaloka_agent.discovery_tools import discover

    ctx = {"dataset_id": "ds-1", "input_data_type": "csv",
           "schema_columns": ["region", "revenue", "units"], "data_source_location": "local://ds-1.csv"}
    out = discover(ctx, {})

    assert out["dataset_id"] == "ds-1"
    assert out["file_type"] == "csv"
    assert out["columns"] == ["region", "revenue", "units"]
    assert "error" not in out


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "CM-07: discovery_tools.discover_database imports app.mcp_server.client.list_tables, "
    "a module that does not exist, so database discovery always answers 'client not available'"))
def test_cm07_database_discovery_client_exists():
    """CM-07: native database discovery has the MCP client it delegates to
    (discovery_tools.py docstring: "DB introspection is provided by app.mcp_server")."""
    spec = importlib.util.find_spec("app.mcp_server.client")

    assert spec is not None
    module = importlib.import_module("app.mcp_server.client")
    assert callable(getattr(module, "list_tables", None))
