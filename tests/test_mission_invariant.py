"""Tests for the Avaloka mission spine.

Covers the load-bearing contracts of the new execution brain:

  * the cross-interface invariant ``cli_mission == mcp_mission``;
  * DataMission schema round-trips through its canonical form;
  * the estimator + router make sane, budget-aware decisions;
  * cloud-environment detection is deterministic and override-able (no network);
  * the MCP tool functions work without the ``mcp`` SDK installed.

All hermetic — nothing here provisions a cluster or touches the network.
"""

import argparse

import pytest

from app.execution.environment import (
    CloudProvider,
    detect_environment,
    provider_for_uri,
    resolve_cloud_target,
)
from app.execution.estimator import DatasetStats, estimate_workload
from app.execution.router import route
from app.interfaces.cli.main import build_intent, build_parser
from app.interfaces.mcp.server import (
    tool_describe_environment,
    tool_estimate_cost,
    tool_inspect_intent,
    tool_plan_mission,
)
from app.interfaces.service import plan_mission
from app.missions.compiler import compile_mission
from app.missions.schema import DataMission, ExecutionMode, GoalType


# --- cross-interface invariant ---------------------------------------------

def _cli_intent(argv):
    from app.interfaces.cli import main as cli_main  # noqa: F401
    args = build_parser().parse_args(argv)
    return build_intent(args)


def test_cli_and_mcp_compile_identical_missions():
    """Equivalent intent through CLI and MCP -> byte-identical DataMission."""
    cli_intent = _cli_intent([
        "plan", "s3://bucket/customers.parquet",
        "--goal", "Explain churn drivers",
        "--goal-type", "predictive_model",
        "--target", "churned",
        "--rows", "84000000",
        "--max-cost", "20",
    ])
    # The MCP surface builds the same flat intent dict directly.
    mcp_intent = {
        "source_uri": "s3://bucket/customers.parquet",
        "goal": "Explain churn drivers",
        "goal_type": "predictive_model",
        "target": "churned",
        "rows": 84000000,
        "max_cost_usd": 20,
    }
    # Normalise the CLI key name for goal description.
    mcp_intent["goal_description"] = mcp_intent.pop("goal")

    cli_mission = compile_mission(cli_intent).canonical()
    mcp_mission = compile_mission(mcp_intent).canonical()
    assert cli_mission == mcp_mission


def test_plan_mission_matches_mcp_tool():
    intent = {"source_uri": "gs://b/data.parquet", "goal_description": "analyze"}
    from app.interfaces.service import planned_to_dict
    assert planned_to_dict(plan_mission(intent)) == tool_plan_mission(intent)


# --- schema round-trip ------------------------------------------------------

def test_datamission_canonical_roundtrip():
    intent = {
        "source_uri": "gs://b/data.parquet",
        "goal_type": "analysis",
        "goal_description": "segment customers",
        "max_cost_usd": 5.0,
    }
    mission = compile_mission(intent)
    canon = mission.canonical()
    # Re-hydrate from canonical (alias-keyed) form and re-serialize: must match.
    rebuilt = DataMission.model_validate(canon)
    assert rebuilt.canonical() == canon
    assert canon["apiVersion"] == "avaloka.ai/v1"
    assert canon["kind"] == "DataMission"


def test_compile_requires_source():
    with pytest.raises(ValueError):
        compile_mission({"goal_description": "no source"})


# --- estimator / router sanity ---------------------------------------------

def test_small_dataset_runs_interactive_no_sampling():
    mission = compile_mission({"source_uri": "file://x.parquet", "rows": 10_000})
    plan = route(mission, stats=DatasetStats(rows=10_000))
    assert plan.mode is ExecutionMode.INTERACTIVE
    assert plan.estimate.full_pass_worthwhile is False


def test_large_analysis_prefers_sample():
    mission = compile_mission({"source_uri": "file://x.parquet", "rows": 84_000_000})
    plan = route(mission, stats=DatasetStats(rows=84_000_000))
    assert plan.mode is ExecutionMode.SAMPLED
    # Sampling is dramatically cheaper than the full pass.
    assert plan.estimate.sampled_cost_usd[1] < plan.estimate.full_cost_usd[1]


def test_rare_class_forces_full_pass():
    mission = compile_mission({
        "source_uri": "file://x.parquet",
        "goal_type": "predictive_model",
        "target": "fraud",
    })
    stats = DatasetStats(rows=84_000_000, rare_class_fraction=0.01)
    plan = route(mission, stats=stats)
    assert plan.estimate.full_pass_worthwhile is True
    assert plan.mode in (ExecutionMode.HYBRID, ExecutionMode.BATCH)
    assert plan.requires_approval is True


def test_full_pass_estimate_in_vision_ballpark():
    """Vision worked example: ~84M rows full pass ≈ $14, ≈17 min."""
    mission = compile_mission({"source_uri": "file://x.parquet", "rows": 84_000_000})
    est = estimate_workload(mission, DatasetStats(rows=84_000_000))
    lo, hi = est.full_cost_usd
    assert lo <= 14.28 <= hi
    rt_lo, rt_hi = est.full_runtime_minutes
    assert rt_lo <= 16.8 <= rt_hi


def test_explicit_mode_is_honoured():
    mission = compile_mission({
        "source_uri": "file://x.parquet", "rows": 10_000, "mode": "batch",
    })
    plan = route(mission, stats=DatasetStats(rows=10_000))
    assert plan.mode is ExecutionMode.BATCH


def test_ray_target_flagged_not_implemented():
    mission = compile_mission({
        "source_uri": "file://x.parquet",
        "rows": 84_000_000,
        "goal_type": "predictive_model",
        "target": "fraud",
        "full_data_engine": "ray",
        "mode": "batch",
    })
    plan = route(mission, stats=DatasetStats(rows=84_000_000, rare_class_fraction=0.01))
    assert plan.target == "ray"
    assert plan.implemented is False


# --- cloud-environment awareness -------------------------------------------

def test_detect_defaults_to_local_when_hermetic(monkeypatch):
    for k in ("AVALOKA_CLOUD", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT_ID",
              "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_EXECUTION_ENV",
              "AZURE_SUBSCRIPTION_ID", "K_SERVICE", "WEBSITE_SITE_NAME"):
        monkeypatch.delenv(k, raising=False)
    env = detect_environment(allow_network=False)
    assert env.provider is CloudProvider.LOCAL
    assert env.source == "default"


@pytest.mark.parametrize("value,expected", [
    ("gcp", CloudProvider.GCP),
    ("aws", CloudProvider.AWS),
    ("azure", CloudProvider.AZURE),
    ("local", CloudProvider.LOCAL),
])
def test_override_env_var(monkeypatch, value, expected):
    monkeypatch.setenv("AVALOKA_CLOUD", value)
    env = detect_environment()
    assert env.provider is expected
    assert env.source == "override"


def test_env_fingerprint_detects_gcp(monkeypatch):
    monkeypatch.delenv("AVALOKA_CLOUD", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-proj")
    env = detect_environment()
    assert env.provider is CloudProvider.GCP
    assert env.source == "env"
    assert env.service_catalog["container_registry"] == "Artifact Registry"


def test_uri_scheme_pins_cloud():
    assert provider_for_uri("s3://bucket/x") is CloudProvider.AWS
    assert provider_for_uri("gs://bucket/x") is CloudProvider.GCP
    assert provider_for_uri("abfss://c@a.dfs.core.windows.net/x") is CloudProvider.AZURE
    assert provider_for_uri("/local/path.parquet") is None


def test_router_is_environment_aware_via_source(monkeypatch):
    monkeypatch.setenv("AVALOKA_CLOUD", "local")  # runtime is local...
    mission = compile_mission({"source_uri": "s3://b/x.parquet", "rows": 1000})
    plan = route(mission)  # ...but the data lives on S3 -> AWS-flavoured plan.
    assert plan.cloud_target == "aws"
    assert plan.cloud_source == "source_uri"
    assert "Amazon S3" in plan.agent_context


def test_explicit_cloud_target_wins(monkeypatch):
    monkeypatch.setenv("AVALOKA_CLOUD", "aws")
    mission = compile_mission({
        "source_uri": "file://x.parquet", "rows": 1000, "cloud_target": "azure",
    })
    plan = route(mission)
    assert plan.cloud_target == "azure"
    assert plan.cloud_source == "mission"
    assert "Azure OpenAI" in plan.agent_context


def test_resolve_cloud_target_priority(monkeypatch):
    monkeypatch.setenv("AVALOKA_CLOUD", "gcp")
    # pinned beats uri beats env
    assert resolve_cloud_target("azure", "s3://b/x").provider is CloudProvider.AZURE
    assert resolve_cloud_target(None, "s3://b/x").provider is CloudProvider.AWS
    assert resolve_cloud_target(None, "/local").provider is CloudProvider.GCP


# --- MCP tools work without the SDK ----------------------------------------

def test_mcp_tools_without_sdk(monkeypatch):
    monkeypatch.setenv("AVALOKA_CLOUD", "gcp")
    intent = {"source_uri": "gs://b/x.parquet", "rows": 84_000_000}

    plan = tool_plan_mission(intent)
    assert plan["plan"]["cloud_target"] == "gcp"
    assert plan["estimate"]["recommended_mode"] in {m.value for m in ExecutionMode}

    est = tool_estimate_cost(intent)
    assert est["full_cost_usd"][0] > 0

    env = tool_describe_environment()
    assert env["provider"] == "gcp"
    assert env["service_catalog"]["warehouse"] == "BigQuery"

    ins = tool_inspect_intent(intent)
    assert ins["mission"]["kind"] == "DataMission"


def test_mcp_build_server_raises_clear_error_without_sdk(monkeypatch):
    import builtins
    import app.interfaces.mcp.server as srv

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("mcp"):
            raise ImportError("no mcp")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="requires the 'mcp' package"):
        srv.build_server()
