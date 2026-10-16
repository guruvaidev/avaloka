"""CLI test plan, suites S1–S4 (docs/TEST_PLAN-CLI.md).

S1 — packaging and documentation contracts (L0): nothing runs a mission; the
     assertions are over pyproject, VERSION, README_CLI.md and the Typer app.
S2 — workload sizing and sampling (L1): the lane thresholds and the Sampling
     Specialist's obedience to them.
S3 — budget, ledger and economics (L1): the arithmetic the economics table is
     built from, asserted exactly.
S4 — deployment gates and serving plan (L1): synthetic package directories, so
     every gate branch is reachable without training a model.

Each test names its plan case ID in its docstring so the plan and the suite can
be reconciled mechanically.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from avaloka import __version__, deploy, serving, workload
from avaloka.cli import VALID_METRICS, app
from avaloka.fireflies.sampling_specialist import SamplingSpecialist
from avaloka.mission.budget import Budget, BudgetExceeded
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.mission.economics import compute_economics
from avaloka.mission.ledger import Ledger, WorkUnit
from avaloka.workload import Lane

REPO = Path(__file__).resolve().parents[2]
README_CLI = (REPO / "README_CLI.md").read_text()
runner = CliRunner()


def registered_commands() -> set[str]:
    # The built Click group, not app.registered_commands: that list holds only
    # plain commands, so a sub-app such as `benchmark` (app.add_typer) is
    # missing from it although `avaloka benchmark` runs.
    import typer.main
    return set(typer.main.get_command(app).commands)


# ═══════════════════════════════════════════════════════════════════════════
# S1 — packaging and documentation contracts (L0)
# ═══════════════════════════════════════════════════════════════════════════

def test_s1_01_console_script_points_at_an_importable_callable():
    """S1.01 — pyproject's entry point resolves to avaloka.cli:main."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert project["scripts"]["avaloka"] == "avaloka.cli:main"
    from avaloka.cli import main
    assert callable(main)


def test_s1_02_version_has_one_source():
    """S1.02 — VERSION, avaloka.__version__, `version` and `--version` agree."""
    on_disk = (REPO / "VERSION").read_text().strip()
    assert __version__ == on_disk

    flag = runner.invoke(app, ["--version"])
    cmd = runner.invoke(app, ["version"])
    assert flag.exit_code == 0 and cmd.exit_code == 0
    assert on_disk in flag.output and on_disk in cmd.output


def test_s1_03_every_documented_command_exists():
    """S1.03 — README_CLI.md never names a command the binary does not have."""
    documented = set(re.findall(r"\bavaloka ([a-z]+)\b", README_CLI))
    unknown = documented - registered_commands()
    assert not unknown, f"README_CLI.md documents commands that do not exist: {sorted(unknown)}"


def test_s1_04_help_lists_every_command_and_leads_with_an_example():
    """S1.04 — `avaloka --help` is the complete, runnable table of contents."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in registered_commands():
        assert name in result.output, f"--help omits the {name!r} command"
    assert "avaloka analyze" in result.output


def test_s1_05_train_help_lists_exactly_the_metrics_the_code_accepts():
    """S1.05 — the --metric help string and VALID_METRICS cannot drift apart."""
    import typer.main

    train = typer.main.get_command(app).commands["train"]
    metric = next(p for p in train.params if p.name == "metric")
    listed = re.search(r"[a-z0-9_]+(?:\|[a-z0-9_]+)+", metric.help or "")
    assert listed, "train --metric help must list the accepted metrics as a | separated set"
    assert set(listed.group(0).split("|")) == set(VALID_METRICS)


def test_s1_06_deployment_levels_match_the_documentation():
    """S1.06 — the four gate names in code are the four in README_CLI.md."""
    assert set(deploy.LEVELS) == {1, 2, 3, 4}
    for name in deploy.LEVELS.values():
        assert name in README_CLI, f"level {name!r} is not documented"


def test_s1_07_bundled_sample_is_packaged_and_usable():
    """S1.07 — avaloka/examples/churn.csv ships with the wheel and loads."""
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]["avaloka"]
    assert "examples/*.csv" in package_data

    sample = REPO / "avaloka" / "examples" / "churn.csv"
    assert sample.exists()
    frame = pd.read_csv(sample)
    assert "churned" in frame.columns and len(frame) > 100


def test_s1_08_cli_imports_without_the_platform_package():
    """S1.08 — the CLI is standalone: importing it never imports app.*.

    Checked in a subprocess because other tests in this session may already
    have imported the platform.
    """
    probe = ("import sys, avaloka.cli; "
             "print(any(m.split('.')[0] == 'app' for m in sys.modules))")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            cwd=str(REPO))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_s1_09_runtime_dependency_footprint_is_the_declared_eight():
    """S1.09 — a laptop install must stay small; anthropic is an extra, not a dep."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    names = {re.split(r"[<>=!~\[ ]", d, 1)[0].lower() for d in project["dependencies"]}
    assert names == {"typer", "rich", "pandas", "numpy", "scikit-learn", "pyarrow", "pyyaml", "jinja2"}
    assert "anthropic" not in names
    llm_extra = project["optional-dependencies"]["llm"]
    assert any(d.lower().startswith("anthropic") for d in llm_extra)


# ═══════════════════════════════════════════════════════════════════════════
# S2 — workload sizing and sampling (L1)
# ═══════════════════════════════════════════════════════════════════════════

MB = 1024 * 1024
GB = 1024 ** 3


@pytest.mark.parametrize("rows,n_bytes,lane", [
    pytest.param(100_000, 50 * MB, Lane.ONLINE, id="online-upper-edge"),
    pytest.param(100_001, 50 * MB, Lane.SAMPLED_ONLINE, id="one-row-over"),
    pytest.param(1_000, 50 * MB + 1, Lane.SAMPLED_ONLINE, id="bytes-alone-promote"),
    pytest.param(5_000_000, GB, Lane.SAMPLED_ONLINE, id="sampled-upper-edge"),
    pytest.param(5_000_001, GB, Lane.BATCH, id="one-row-into-batch"),
    pytest.param(10, GB + 1, Lane.BATCH, id="bytes-alone-batch"),
])
def test_s2_01_lane_boundaries_are_inclusive_and_either_axis_promotes(rows, n_bytes, lane):
    """S2.01 — thresholds are inclusive; rows OR bytes can push a lane up."""
    assert workload.classify(rows, n_bytes).lane is lane


def test_s2_02_batch_is_recommended_exactly_when_the_lane_is_not_online():
    """S2.02 — recommend_batch is a function of the lane, nothing else."""
    assert workload.classify(10, 10).recommend_batch is False
    assert workload.classify(200_000, MB).recommend_batch is True
    assert workload.classify(10_000_000, MB).recommend_batch is True


def test_s2_03_workload_plan_serialises_for_the_bundle():
    """S2.03 — as_dict() is what lands in workload.json; it must be JSON-able."""
    plan = workload.classify(5, 5, estimated=True)
    d = plan.as_dict()
    assert d["lane"] == "online" and d["estimated"] is True
    assert {"n_rows", "n_bytes", "rationale", "spoken", "recommend_batch"} <= set(d)
    json.dumps(d)
    assert "~5" in plan.spoken, "an estimated count is spoken as approximate"


def test_s2_04_route_agrees_with_peek_plus_classify(churn_csv):
    """S2.04 — route() is nothing more than classify(peek())."""
    n_rows, n_bytes, estimated = workload.peek(str(churn_csv))
    assert workload.route(str(churn_csv)) == workload.classify(n_rows, n_bytes, estimated=estimated)


def _ctx_with_frame(tmp_path, frame, *, lane: str | None, target: str | None = None):
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal="t", data_source="x.csv",
                         output_dir=tmp_path / "out", budget=Budget(), target=target)
    ctx.blackboard["frame"] = frame
    if lane:
        ctx.blackboard["workload"] = {"lane": lane}
    return ctx


def test_s2_05_sampling_specialist_obeys_the_lane_not_the_row_count(tmp_path):
    """S2.05 — the lane decides whether to sample; row count is only the fallback."""
    frame = pd.DataFrame({"x": range(400), "y": [0, 1] * 200})

    online = SamplingSpecialist(_ctx_with_frame(tmp_path, frame, lane="online"))
    online.perform()
    assert online.ctx.blackboard["sampling"]["sampled"] is False
    assert len(online.ctx.blackboard["working_frame"]) == 400

    sampled = SamplingSpecialist(_ctx_with_frame(tmp_path, frame, lane="sampled_online"))
    sampled.perform()
    decision = sampled.ctx.blackboard["sampling"]
    assert decision["sampled"] is True and decision["strategy"] == "random"
    assert "seed=42" in decision["reason"], "the sample must be reproducible and say so"


def test_s2_06_sampling_stratifies_on_a_low_cardinality_target(tmp_path):
    """S2.06 — with a target of ≤ 50 classes the sample is stratified on it."""
    frame = pd.DataFrame({"x": range(1000), "y": [0] * 900 + [1] * 100})
    ff = SamplingSpecialist(_ctx_with_frame(tmp_path, frame, lane="sampled_online", target="y"))
    ff.perform()
    decision = ff.ctx.blackboard["sampling"]
    assert decision["strategy"] == "stratified" and decision["stratify_by"] == "y"
    working = ff.ctx.blackboard["working_frame"]
    # Every class survives sampling; stratification is the whole point.
    assert set(working["y"].unique()) == {0, 1}


# ═══════════════════════════════════════════════════════════════════════════
# S3 — budget, ledger and economics (L1)
# ═══════════════════════════════════════════════════════════════════════════

def test_s3_01_a_zero_budget_is_a_ceiling_not_unlimited():
    """S3.01 — None means unconstrained; 0 means nothing may be spent."""
    zero = Budget(limit_usd=0.0)
    assert zero.unlimited is False
    assert zero.spend(0.0, "free") == 0.0
    with pytest.raises(BudgetExceeded) as excinfo:
        zero.spend(0.0001, "data_scout")
    message = str(excinfo.value)
    assert "data_scout" in message and "$0.00" in message, "name the action and the amounts"


def test_s3_02_budget_arithmetic_at_the_boundary():
    """S3.02 — remaining and fraction_used track spend; the boundary is inclusive."""
    b = Budget(limit_usd=1.0)
    assert b.can_afford(1.0) is True
    assert b.can_afford(1.0 + 1e-6) is False
    b.spend(0.25)
    assert b.remaining() == pytest.approx(0.75)
    assert b.fraction_used() == pytest.approx(0.25)
    assert Budget().remaining() == float("inf")
    assert Budget().fraction_used() == 0.0


def test_s3_03_multiplier_denominator_is_total_customer_cost():
    """S3.03 — value / (Avaloka invoice + human review), review never free."""
    ledger = Ledger()
    ledger.record(WorkUnit(firefly="f", human_role="r", output="o", manual_minutes=60.0,
                           compute_cost_usd=0.0, model_cost_usd=0.0, duration_seconds=1.0))
    econ = compute_economics(ledger, tier="community", loaded_hourly_rate=80.0)

    assert econ.estimated_manual_effort_hours == 1.0
    assert econ.estimated_labor_value_usd == 80.0
    # 9% of one hour is below the 0.25 h floor, so the floor applies.
    assert econ.human_review_hours == 0.25
    assert econ.human_review_cost_usd == 20.0
    assert econ.total_customer_cost_usd == 20.0
    assert econ.economic_multiplier == 4.0


def test_s3_04_empty_ledger_still_yields_finite_honest_economics():
    """S3.04 — a mission that did no work reports 0 value and a 0x multiplier, not NaN."""
    econ = compute_economics(Ledger(), loaded_hourly_rate=80.0)
    assert econ.estimated_manual_effort_hours == 0.0
    assert econ.human_review_hours == 0.25, "review is floored even when nothing was replaced"
    assert econ.economic_multiplier == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# S4 — deployment gates and serving plan (L1, synthetic packages)
# ═══════════════════════════════════════════════════════════════════════════

L1_FILES = ["model/model.pkl", "model_card.md"]
L2_FILES = ["service.py", "Dockerfile", "inference_schema.json", "requirements.lock",
            "tests/test_service.py", "feature_contract.yaml"]
L3_FILES = ["deployment/kubernetes.yaml", "monitoring_config.yaml", "mlflow/MLmodel"]


def make_package(root: Path, files, *, cap: int | None = None) -> Path:
    """A package directory holding exactly `files`, optionally with a validator cap."""
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    if cap is not None:
        (root / "validation_report.json").write_text(json.dumps({"max_safe_deployment_level": cap}))
    return root


def test_s4_01_missing_package_and_bad_target_fail_loudly(tmp_path):
    """S4.01 — a missing directory and an unknown target are named, not guessed."""
    with pytest.raises(FileNotFoundError):
        deploy.plan_deployment(str(tmp_path / "nope"))
    pkg = make_package(tmp_path / "pkg", L1_FILES)
    with pytest.raises(ValueError) as excinfo:
        deploy.plan_deployment(str(pkg), "mainframe")
    assert "kubernetes" in str(excinfo.value), "list the valid targets"


def test_s4_02_artifact_gates_are_cumulative(tmp_path):
    """S4.02 — level N needs every lower level; missing files are itemised."""
    pkg = make_package(tmp_path / "pkg", L1_FILES, cap=4)
    plan = deploy.plan_deployment(str(pkg), "kubernetes", requested_level=2)
    assert plan.achieved_level == 1
    gate2 = next(g for g in plan.gates if g["level"] == 2)
    assert gate2["passed"] is False and set(gate2["missing_artifacts"]) == set(L2_FILES)
    assert "missing artifacts" in plan.blocked_reason


def test_s4_03_no_validation_report_caps_at_deployment_package(tmp_path):
    """S4.03 — without validation evidence a package can never claim staging."""
    pkg = make_package(tmp_path / "pkg", L1_FILES + L2_FILES + L3_FILES)
    plan = deploy.plan_deployment(str(pkg), "kubernetes", requested_level=3)
    assert plan.validator_cap == 2 and plan.achieved_level == 2
    assert "caps this package at Level 2" in plan.blocked_reason


def test_s4_04_validator_cap_bounds_the_achieved_level(tmp_path):
    """S4.04 — full artifacts plus a cap of 2 means level 2, whatever was asked."""
    pkg = make_package(tmp_path / "pkg", L1_FILES + L2_FILES + L3_FILES, cap=2)
    plan = deploy.plan_deployment(str(pkg), "kubernetes", requested_level=3)
    assert plan.achieved_level == 2
    assert "validation_report.json" in plan.blocked_reason


def test_s4_05_production_needs_approval_and_a_clean_cap(tmp_path):
    """S4.05 — level 4 is a governance gate: artifacts + cap 4 + --approve."""
    pkg = make_package(tmp_path / "pkg", L1_FILES + L2_FILES + L3_FILES, cap=4)

    unapproved = deploy.plan_deployment(str(pkg), "kubernetes", requested_level=4, approved=False)
    assert unapproved.achieved_level == 3
    assert "--approve" in unapproved.blocked_reason
    gate4 = next(g for g in unapproved.gates if g["level"] == 4)
    assert gate4["missing_artifacts"] == ["human approval (--approve)"]

    approved = deploy.plan_deployment(str(pkg), "kubernetes", requested_level=4, approved=True)
    assert approved.achieved_level == 4 and approved.blocked_reason is None

    capped = make_package(tmp_path / "capped", L1_FILES + L2_FILES + L3_FILES, cap=3)
    plan = deploy.plan_deployment(str(capped), "kubernetes", requested_level=4, approved=True)
    assert plan.achieved_level == 3
    gate4 = next(g for g in plan.gates if g["level"] == 4)
    assert gate4["missing_artifacts"] == ["clean validation verdict for production"]


@pytest.mark.parametrize("target,manifest", [
    ("local", "deployment/local-compose.yaml"),
    ("docker", "deployment/local-compose.yaml"),
    ("kubernetes", "deployment/kubernetes.yaml"),
    ("ray", "deployment/rayservice.yaml"),
])
def test_s4_06_manifest_follows_the_target_and_is_none_when_absent(tmp_path, target, manifest):
    """S4.06 — each target has one manifest; an absent manifest is reported as None."""
    pkg = make_package(tmp_path / "pkg", L1_FILES)
    assert deploy.plan_deployment(str(pkg), target).manifest is None
    make_package(pkg, [manifest])
    assert deploy.plan_deployment(str(pkg), target).manifest == manifest


def test_s4_07_commands_reflect_target_and_cost_options(tmp_path):
    """S4.07 — commands are target-specific; cost guard and scale-to-zero are visible."""
    pkg = make_package(tmp_path / "pkg", L1_FILES + ["deployment/rayservice.yaml"])
    plan = deploy.plan_deployment(str(pkg), "ray", scale_to_zero=True, max_monthly_cost=200)
    assert plan.commands[0].startswith("# Cost guard") and "$200" in plan.commands[0]
    assert any("scale-to-zero" in c for c in plan.commands)
    assert any("kubectl apply -f deployment/rayservice.yaml" in c for c in plan.commands)

    local = deploy.plan_deployment(str(pkg), "local")
    assert any(c.startswith("docker run") for c in local.commands)
    assert not any("kubectl" in c for c in local.commands)


def _deployable(tmp_path: Path, *, with_mlflow: bool = True) -> Path:
    pkg = tmp_path / "pkg"
    make_package(pkg, ["model/model.pkl", "service.py", "Dockerfile"])
    manifest = pkg / "deployment" / "kubernetes.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(yaml.safe_dump_all([
        {"kind": "Deployment", "metadata": {"name": "churn-api"},
         "spec": {"template": {"spec": {"containers": [{"image": "avaloka-inference:abc123"}]}}}},
        {"kind": "Service", "metadata": {"name": "churn-svc"}},
    ]))
    if with_mlflow:
        (pkg / "mlflow").mkdir()
        (pkg / "mlflow" / "register.py").write_text("print('ok')")
    return pkg


def test_s4_08_serving_prepare_reads_names_from_the_manifest_and_orders_the_steps(tmp_path):
    """S4.08 — image, Deployment and Service names come from the manifest; stages are fixed."""
    pkg = _deployable(tmp_path)
    plan = serving.prepare(str(pkg))
    assert plan.image == "avaloka-inference:abc123"
    assert plan.deployment_name == "churn-api" and plan.service_name == "churn-svc"
    assert [s["stage"] for s in plan.steps] == ["mlflow", "docker", "kubernetes", "endpoint"]
    assert plan.commands == [
        f"docker build -t avaloka-inference:abc123 {pkg}",
        f"kubectl apply -f {pkg / 'deployment' / 'kubernetes.yaml'}",
        "kubectl rollout status deploy/churn-api",
    ]


def test_s4_09_registry_and_mlflow_options_change_the_plan(tmp_path):
    """S4.09 — a registry rewrites the image; an MLflow URI adds registration first."""
    pkg = _deployable(tmp_path)
    plan = serving.prepare(str(pkg), registry="ghcr.io/acme/", mlflow_uri="http://mlflow:5000")
    assert plan.image == "ghcr.io/acme/avaloka-inference:abc123"
    assert plan.commands[0].endswith("python mlflow/register.py")
    assert "MLFLOW_TRACKING_URI=http://mlflow:5000" in plan.commands[0]
    assert "docker push ghcr.io/acme/avaloka-inference:abc123" in plan.commands
    assert plan.steps[0]["action"] == "register"


def test_s4_10_mlflow_uri_requires_the_register_script(tmp_path):
    """S4.10 — asking for MLflow registration without the script is refused up front."""
    pkg = _deployable(tmp_path, with_mlflow=False)
    with pytest.raises(ValueError) as excinfo:
        serving.prepare(str(pkg), mlflow_uri="http://mlflow:5000")
    assert "mlflow/register.py" in str(excinfo.value)


def test_s4_11_curl_example_uses_the_packaged_request_when_present(tmp_path):
    """S4.11 — the printed curl is runnable against the resolved endpoint."""
    pkg = _deployable(tmp_path)
    (pkg / "tests").mkdir()
    (pkg / "tests" / "example_request.json").write_text('{"instances": [{"tenure": 3}]}\n')
    curl = serving.curl_example("http://10.0.0.1:80", str(pkg))
    assert curl.startswith("curl -s http://10.0.0.1:80/predict")
    assert '{"instances": [{"tenure": 3}]}' in curl
    assert "\n" not in curl, "a copy-pasteable one-liner"
