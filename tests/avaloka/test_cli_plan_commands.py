"""CLI test plan, suites S5–S8 (docs/TEST_PLAN-CLI.md).

S5 — every command end to end, in process, through Typer's CliRunner (L2).
S6 — the chat session: verbs, numbered picks, refusals (L2).
S7 — the process boundary: `python -m avaloka`, the console script, and the
     crash contract in `main()` (L3).
S8 — the external substrate: the hermetic halves run here with the tool
     probes patched; the live halves are gated behind the `cluster` marker and
     AVALOKA_TEST_ALLOW_APPLY=1 and skip otherwise (L4).

Missions run for real on the 400-row fixtures; nothing here needs a key, a
cluster, Docker or the network.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from avaloka import cli, serving
from avaloka.cli import app
from avaloka.coordinator import load_coordination
from avaloka.io.loader import UnusableDataset

REPO = Path(__file__).resolve().parents[2]
runner = CliRunner()


def flat(output: str) -> str:
    """Rich wraps at 80 columns under CliRunner; undo it before asserting on prose."""
    return " ".join(output.split())


# ═══════════════════════════════════════════════════════════════════════════
# S5 — commands end to end (L2)
# ═══════════════════════════════════════════════════════════════════════════

def test_s5_01_analyze_defaults_its_output_directory(churn_csv, tmp_path, monkeypatch):
    """S5.01 — without -o the bundle lands in a unique subfolder under ./avaloka-analysis."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["analyze", str(churn_csv), "--goal", "x", "--quiet"])
    assert result.exit_code == 0, result.output
    subdirs = [p for p in (tmp_path / "avaloka-analysis").iterdir() if p.is_dir()]
    assert len(subdirs) == 1, f"expected exactly one run directory, found {subdirs}"
    assert (subdirs[0] / "mission.json").exists()


def test_s5_02_analyze_bundle_records_lane_kind_and_verdict(churn_csv, tmp_path):
    """S5.02 — mission.json and workload.json carry the facts the console showed."""
    out = tmp_path / "out"
    result = runner.invoke(app, ["analyze", str(churn_csv), "--goal", "why churn?",
                                 "-o", str(out), "--quiet"])
    assert result.exit_code == 0, result.output
    mission = json.loads((out / "mission.json").read_text())
    assert mission["kind"] == "analyze" and mission["goal"] == "why churn?"
    assert mission["verdict"] in {"pass", "warn"}
    assert mission["max_safe_deployment_level"] == 1, "analysis is a research artifact"
    assert json.loads((out / "workload.json").read_text())["lane"] == "online"
    assert mission["budget"] == {"limit_usd": None, "spent_usd": pytest.approx(0, abs=0.01)}


def test_s5_03_quiet_silences_the_persona_but_not_the_facts(churn_csv, tmp_path):
    """S5.03 — --quiet drops narration; verdict, swarm table and deliverables stay."""
    quiet = runner.invoke(app, ["analyze", str(churn_csv), "--goal", "x",
                                "-o", str(tmp_path / "q"), "--quiet"])
    loud = runner.invoke(app, ["analyze", str(churn_csv), "--goal", "x",
                               "-o", str(tmp_path / "l")])
    assert quiet.exit_code == 0 and loud.exit_code == 0

    assert "Cloning myself" not in quiet.output and "Hi, I'm Avaloka" not in quiet.output
    assert "Cloning myself" in loud.output and "Hi, I'm Avaloka" in loud.output
    for must in ("Validation verdict", "Mission economics", "Deliverables", "data_scout"):
        assert must in quiet.output, f"--quiet must still print {must!r}"


def test_s5_04_train_packages_only_when_asked_and_points_at_infer(churn_csv, tmp_path):
    """S5.04 — --deployable adds the serving package and the next command."""
    plain = tmp_path / "plain"
    result = runner.invoke(app, ["train", str(churn_csv), "--target", "churned",
                                 "-o", str(plain), "--quiet"])
    assert result.exit_code == 0, result.output
    assert (plain / "model" / "model.pkl").exists()
    assert not (plain / "Dockerfile").exists()
    assert "Ship it" not in result.output

    packaged = tmp_path / "packaged"
    result = runner.invoke(app, ["train", str(churn_csv), "--target", "churned",
                                 "--deployable", "-o", str(packaged), "--quiet"])
    assert result.exit_code == 0, result.output
    for rel in ("Dockerfile", "service.py", "deployment/kubernetes.yaml", "mlflow/MLmodel"):
        assert (packaged / rel).exists(), rel
    assert "Ship it" in result.output
    # The panel wraps a long path across bordered lines, so assert on the pieces.
    text = flat(result.output.split("Ship it")[1])
    assert "avaloka infer" in text and "--target kubernetes" in text and "--apply" in text
    assert packaged.name in text


def test_s5_05_train_summary_carries_the_baseline(churn_csv, tmp_path):
    """S5.05 — mission.json never reports a score without the baseline it beat."""
    out = tmp_path / "out"
    result = runner.invoke(app, ["train", str(churn_csv), "--target", "churned",
                                 "--metric", "roc_auc", "-o", str(out), "--quiet"])
    assert result.exit_code == 0, result.output
    model = json.loads((out / "mission.json").read_text())["model"]
    assert model["metric"] == "roc_auc"
    assert isinstance(model["score"], float) and isinstance(model["baseline_score"], float)
    assert model["baseline_strategy"] and model["beats_baseline"] in (True, False)
    assert "Best model" in result.output and "holdout" in result.output


def test_s5_06_budget_exceeded_exits_with_code_2(churn_csv, tmp_path):
    """S5.06 — a declared ceiling is a hard stop with its own exit status."""
    result = runner.invoke(app, ["analyze", str(churn_csv), "--goal", "x", "--budget", "0",
                                 "-o", str(tmp_path / "out"), "--quiet"])
    assert result.exit_code == 2, result.output
    assert "Budget exceeded" in result.output
    assert "Traceback" not in result.output


def test_s5_07_deploy_reports_gates_and_writes_the_plan(trained_package):
    """S5.07 — the gate table, the cap, and deploy_plan.json; level 4 is refused."""
    result = runner.invoke(app, ["deploy", str(trained_package), "--target", "kubernetes",
                                 "--level", "4"])
    assert result.exit_code == 0, result.output
    assert "Deployment-level gates" in result.output
    assert "Validator cap" in result.output
    assert "--approve" in flat(result.output), "say what unblocks production"
    plan = json.loads((trained_package / "deployment" / "deploy_plan.json").read_text())
    assert plan["requested_level"] == 4 and plan["achieved_level"] < 4
    assert plan["blocked_reason"]


def test_s5_08_deploy_errors_are_sentences(tmp_path, trained_package):
    """S5.08 — missing package and bad target exit 1 with the reason, no traceback."""
    missing = runner.invoke(app, ["deploy", str(tmp_path / "nope")])
    assert missing.exit_code == 1 and "not found" in flat(missing.output)
    bad = runner.invoke(app, ["deploy", str(trained_package), "--target", "mainframe"])
    assert bad.exit_code == 1 and "Unsupported target" in flat(bad.output)
    assert "Traceback" not in missing.output + bad.output


def test_s5_09_infer_without_apply_only_plans(trained_package, monkeypatch):
    """S5.09 — the plan is printed and saved; no docker or kubectl is ever run."""
    def forbidden(*a, **k):  # pragma: no cover - the assertion is that this is never hit
        raise AssertionError("infer without --apply must not shell out")
    monkeypatch.setattr(serving, "_run", forbidden)

    result = runner.invoke(app, ["infer", str(trained_package), "--target", "kubernetes",
                                 "--quiet"])
    assert result.exit_code == 0, result.output
    assert "Inference pipeline" in result.output
    assert "docker build" in result.output and "kubectl apply" in result.output
    assert "This was a plan" in flat(result.output)
    saved = json.loads((trained_package / "deployment" / "inference_plan.json").read_text())
    assert saved["applied"] is False and saved["endpoint"] is None


def test_s5_10_infer_refuses_a_non_deployable_bundle(churn_csv, tmp_path):
    """S5.10 — an analysis bundle is not a model package; say which files are missing."""
    out = tmp_path / "ana"
    assert runner.invoke(app, ["analyze", str(churn_csv), "--goal", "x", "-o", str(out),
                               "--quiet"]).exit_code == 0
    result = runner.invoke(app, ["infer", str(out), "--quiet"])
    assert result.exit_code == 1
    text = flat(result.output)
    assert "not deployable" in text and "--deployable" in text
    assert "Traceback" not in result.output


def test_s5_11_coordinate_converges_a_fleet(churn_csv, tmp_path):
    """S5.11 — two sub-Avalokas, one converged profile, one coordination.json."""
    out = tmp_path / "fleet"
    result = runner.invoke(app, ["coordinate", str(churn_csv), "--goal", "x",
                                 "--workers", "2", "-o", str(out), "--quiet"])
    assert result.exit_code == 0, result.output
    assert "Converged full-dataset profile" in result.output
    assert "2 sub-Avalokas" in result.output and "400 rows" in result.output
    report = load_coordination(out)
    assert report["workers"] == 2 and report["n_rows"] == 400


@pytest.mark.slow
def test_s5_12_batch_is_exact_and_writes_the_rayjob(churn_csv, tmp_path):
    """S5.12 — the local Ray swarm counts every row and leaves a submittable manifest."""
    out = tmp_path / "batch"
    result = runner.invoke(app, ["batch", str(churn_csv), "--partitions", "2",
                                 "-o", str(out), "--quiet"])
    assert result.exit_code == 0, result.output
    assert "Exact rows analysed" in result.output and "400" in result.output
    profile = json.loads((out / "batch_profile.json").read_text())
    assert profile["n_rows"] == 400
    assert any(p.suffix in {".yaml", ".yml"} for p in out.iterdir()), "no RayJob manifest written"


@pytest.mark.slow
def test_s5_13_batch_kubernetes_prints_commands_and_never_applies_without_kubectl(
        churn_csv, tmp_path, monkeypatch):
    """S5.13 — the hermetic half of S8: plan without --apply; with --apply and no kubectl,
    say so and do not shell out."""
    calls = []
    monkeypatch.setattr(serving, "_have", lambda tool: False)
    monkeypatch.setattr(serving, "_run", lambda *a, **k: calls.append(a) or (True, ""))

    plan_only = runner.invoke(app, ["batch", str(churn_csv), "--target", "kubernetes",
                                    "--partitions", "2", "-o", str(tmp_path / "a"), "--quiet"])
    assert plan_only.exit_code == 0, plan_only.output
    assert "kubectl apply -f" in plan_only.output and calls == []

    applied = runner.invoke(app, ["batch", str(churn_csv), "--target", "kubernetes", "--apply",
                                  "--partitions", "2", "-o", str(tmp_path / "b"), "--quiet"])
    assert applied.exit_code == 0, applied.output
    assert "kubectl not found" in flat(applied.output)
    assert calls == [], "no kubectl on PATH means no kubectl invocation"


@pytest.mark.slow
def test_s5_14_benchmark_passes_and_writes_a_scorecard(tmp_path):
    """S5.14 — the shipped regression harness is green and exports JSON."""
    card = tmp_path / "card.json"
    result = runner.invoke(app, ["benchmark", "--family", "data_engineering",
                                 "-o", str(tmp_path / "bench"), "--json", str(card)])
    assert result.exit_code == 0, result.output
    assert "OVERALL" in result.output
    scorecard = json.loads(card.read_text())
    assert scorecard["n_runs"] >= 1 and scorecard["n_passed"] == scorecard["n_runs"]


def test_s5_15_llm_flag_without_a_key_degrades_to_an_identical_bundle(churn_csv, tmp_path):
    """S5.15 — --llm is narration only: no key, same files, same verdict, exit 0."""
    plain = tmp_path / "plain"
    llm = tmp_path / "llm"
    a = runner.invoke(app, ["analyze", str(churn_csv), "--goal", "x", "-o", str(plain)])
    b = runner.invoke(app, ["analyze", str(churn_csv), "--goal", "x", "-o", str(llm), "--llm"])
    assert a.exit_code == 0 and b.exit_code == 0, a.output + b.output

    files = lambda d: sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    assert files(plain) == files(llm)
    read = lambda d: json.loads((d / "mission.json").read_text())
    assert read(plain)["verdict"] == read(llm)["verdict"]
    assert read(llm)["economics"]["model_cost_usd"] == 0.0


def test_s5_16_a_sqlite_table_is_a_dataset(tmp_path):
    """S5.16 — the sqlite:///path#table URI form works end to end through analyze."""
    import sqlite3
    import pandas as pd

    db = tmp_path / "warehouse.db"
    pd.DataFrame({"a": range(300), "b": [0, 1] * 150}).to_sql(
        "customers", sqlite3.connect(db), index=False)
    result = runner.invoke(app, ["analyze", f"sqlite:///{db}#customers", "--goal", "x",
                                 "-o", str(tmp_path / "out"), "--quiet"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "mission.json").exists()


def test_s5_17_an_excel_workbook_is_a_dataset(churn_csv, tmp_path):
    """S5.17 — .xlsx goes through the same loader and the same mission."""
    pytest.importorskip("openpyxl")
    import pandas as pd

    xlsx = tmp_path / "churn.xlsx"
    pd.read_csv(churn_csv).to_excel(xlsx, index=False)
    result = runner.invoke(app, ["analyze", str(xlsx), "--goal", "x",
                                 "-o", str(tmp_path / "out"), "--quiet"])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "out" / "data_quality.json").read_text())["dataset"]["n_rows"] == 400


# ═══════════════════════════════════════════════════════════════════════════
# S6 — the chat session (L2)
# ═══════════════════════════════════════════════════════════════════════════

def chat(*args, lines: list[str]):
    return runner.invoke(app, ["chat", *args], input="".join(f"{l}\n" for l in lines))


def test_s6_01_columns_and_quit(churn_csv):
    """S6.01 — the schema she read, column by column, then a clean goodbye."""
    result = chat(str(churn_csv), lines=["columns", "quit"])
    assert result.exit_code == 0, result.output
    for col in ("tenure", "monthly", "tickets", "contract", "churned"):
        assert col in result.output
    assert "Anytime" in result.output


def test_s6_02_suggestions_name_real_columns_and_help_lists_the_verbs(churn_csv):
    """S6.02 — every numbered idea mentions a column this dataset has."""
    result = chat(str(churn_csv), lines=["suggest", "help", "quit"])
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "1. train churned" in text, "a two-valued numeric column is the obvious target"
    ideas = [l.strip() for l in result.output.splitlines() if l.strip()[:2] in {"1.", "2.", "3.", "4.", "5."}]
    assert ideas
    columns = {"tenure", "monthly", "tickets", "contract", "churned", "id"}
    for idea in ideas:
        assert any(c in idea for c in columns), idea
    for verb in ("analyze <goal>", "train <column>", "columns", "suggest", "status", "quit"):
        assert verb in result.output


def test_s6_03_a_number_runs_the_numbered_suggestion(churn_csv, tmp_path, monkeypatch):
    """S6.03 — 'Say the number and I will start' is a promise the loop keeps."""
    monkeypatch.chdir(tmp_path)
    result = chat(str(churn_csv), lines=["suggest", "2", "quit"])
    assert result.exit_code == 0, result.output
    assert "Taking 2" in flat(result.output)
    assert "Deliverables" in result.output
    assert (tmp_path / "avaloka-analysis" / "mission.json").exists()


def test_s6_04_an_out_of_range_pick_is_refused_without_running(churn_csv, tmp_path, monkeypatch):
    """S6.04 — picking 9 from a list of 4 gets a correction, not a mission."""
    monkeypatch.chdir(tmp_path)
    result = chat(str(churn_csv), lines=["suggest", "9", "quit"])
    assert result.exit_code == 0, result.output
    assert "I only offered" in flat(result.output)
    assert not (tmp_path / "avaloka-analysis").exists()


def test_s6_05_train_typos_and_missing_column_are_caught_in_the_session(churn_csv, tmp_path,
                                                                        monkeypatch):
    """S6.05 — did-you-mean inside chat, and no training on a bad column."""
    monkeypatch.chdir(tmp_path)
    result = chat(str(churn_csv), lines=["train churnd", "train", "quit"])
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "Did you mean" in text and "churned" in text
    assert "Which column should I predict" in text
    assert not (tmp_path / "avaloka-model").exists()


def test_s6_06_nothing_loaded_means_ask_for_a_dataset():
    """S6.06 — a goal before a dataset is answered with what to do first."""
    result = chat(lines=["analyze why churn", "quit"])
    assert result.exit_code == 0, result.output
    assert "point me at a dataset" in flat(result.output)


def test_s6_07_batch_and_deploy_are_redirected_to_the_shell(churn_csv, tmp_path, monkeypatch):
    """S6.07 — the session says which shell command does it, and runs nothing."""
    monkeypatch.chdir(tmp_path)
    result = chat(str(churn_csv), lines=["batch", "deploy", "quit"])
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    assert "avaloka batch" in text and "avaloka infer" in text
    assert not (tmp_path / "avaloka-analysis").exists(), "no analysis bundle may be written"
    assert not (tmp_path / "avaloka-model").exists(), "no model bundle may be written"
    assert not (tmp_path / "avaloka-batch").exists(), "no batch bundle may be written"


def test_s6_08_end_of_input_ends_the_session_cleanly(churn_csv):
    """S6.08 — a closed stdin is a quiet exit, not a traceback."""
    result = runner.invoke(app, ["chat", str(churn_csv)], input="")
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output


def test_s6_09_suggest_next_steps_is_grounded_deduplicated_and_bounded():
    """S6.09 — the suggestion builder on a synthetic profile."""
    from avaloka.chat import suggest_next_steps

    profile = {
        "columns": [
            {"name": "churned", "role": "numeric", "n_unique": 2},
            {"name": "region", "role": "categorical", "n_unique": 3},
            {"name": "spend", "role": "numeric", "n_unique": 90},
            {"name": "day", "role": "datetime", "n_unique": 30},
        ],
        "correlations": [{"a": "spend", "b": "churned", "corr": 0.4}],
        "quality": {"issues": [{"column": "region", "detail": "x"}]},
    }
    ideas = suggest_next_steps(profile, limit=3)
    assert ideas[0] == "train churned"
    assert len(ideas) == 3 and len(set(i.lower() for i in ideas)) == 3
    names = {"churned", "region", "spend", "day"}
    assert all(any(n in i for n in names) for i in ideas)


# ═══════════════════════════════════════════════════════════════════════════
# S7 — the process boundary (L3)
# ═══════════════════════════════════════════════════════════════════════════

def test_s7_01_python_m_avaloka_is_the_same_program():
    """S7.01 — `python -m avaloka --version` works from a fresh interpreter."""
    result = subprocess.run([sys.executable, "-m", "avaloka", "--version"],
                            capture_output=True, text=True, cwd=str(REPO))
    assert result.returncode == 0, result.stderr
    assert (REPO / "VERSION").read_text().strip() in result.stdout


def test_s7_02_console_script_runs_when_installed():
    """S7.02 — the `avaloka` script next to this interpreter answers --help."""
    script = Path(sys.executable).with_name("avaloka")
    if not script.exists() and shutil.which("avaloka") is None:
        pytest.skip("avaloka console script not installed in this environment")
    exe = str(script) if script.exists() else shutil.which("avaloka")
    result = subprocess.run([exe, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "analyze" in result.stdout


def _run_main_with(monkeypatch, exc: BaseException, *, traceback_env: bool = False):
    def boom():
        raise exc
    monkeypatch.setattr(cli, "app", boom)
    if traceback_env:
        monkeypatch.setenv("AVALOKA_TRACEBACK", "1")
    else:
        monkeypatch.delenv("AVALOKA_TRACEBACK", raising=False)


def test_s7_03_main_turns_a_crash_into_a_sentence(monkeypatch, capsys):
    """S7.03 — an unexpected exception exits 1 with the cause named, no stack frames."""
    _run_main_with(monkeypatch, RuntimeError("the parquet footer is corrupt"))
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
    out = flat(capsys.readouterr().out)
    assert "I hit a problem I couldn't recover from" in out
    assert "RuntimeError: the parquet footer is corrupt" in out
    assert "AVALOKA_TRACEBACK=1" in out, "tell the user how to get the trace"
    assert "Traceback" not in out


def test_s7_04_main_reraises_when_a_traceback_is_requested(monkeypatch):
    """S7.04 — AVALOKA_TRACEBACK=1 hands the real exception to the interpreter."""
    _run_main_with(monkeypatch, RuntimeError("boom"), traceback_env=True)
    with pytest.raises(RuntimeError, match="boom"):
        cli.main()


def test_s7_05_main_treats_an_unusable_dataset_as_a_refusal(monkeypatch, capsys):
    """S7.05 — UnusableDataset is the user's data, not Avaloka's bug; say so."""
    _run_main_with(monkeypatch, UnusableDataset("header with no rows"))
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
    out = flat(capsys.readouterr().out)
    assert "isn't something I can analyse" in out and "header with no rows" in out


def test_s7_06_main_exits_130_on_interrupt(monkeypatch, capsys):
    """S7.06 — Ctrl-C is the conventional 130, with a note that nothing is half-written."""
    _run_main_with(monkeypatch, KeyboardInterrupt())
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 130
    assert "Stopped" in capsys.readouterr().out


def test_s7_07_main_passes_through_normal_exits(monkeypatch):
    """S7.07 — typer's own exit statuses are not re-wrapped."""
    _run_main_with(monkeypatch, SystemExit(2))
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 2


# ═══════════════════════════════════════════════════════════════════════════
# S8 — external substrate (L4): hermetic halves here, live halves gated
# ═══════════════════════════════════════════════════════════════════════════

def test_s8_01_apply_sequence_with_tools_present_is_build_apply_rollout_resolve(
        synthetic_package, monkeypatch):
    """S8.01 — the exact argv order serving.apply issues, with every tool faked present.

    Runs on the synthetic package: the generated one cannot get past manifest
    rendering today (F8), so using it here would test the defect, not the
    sequence.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        return True, "ok"

    monkeypatch.setattr(serving, "_have", lambda tool: True)
    monkeypatch.setattr(serving, "_run", fake_run)
    monkeypatch.setattr(serving, "_resolve_endpoint", lambda name, **k: "http://10.1.2.3:80")

    plan = serving.prepare(str(synthetic_package), registry="ghcr.io/acme")
    plan = serving.apply(plan, push=True)

    assert [c[:2] for c in calls] == [
        ["docker", "build"], ["docker", "push"], ["kubectl", "apply"], ["kubectl", "rollout"],
    ]
    assert calls[0][3] == plan.image and plan.image.startswith("ghcr.io/acme/")
    assert calls[2] == ["kubectl", "apply", "-f", "-"], "the manifest is piped, never a temp file"
    assert calls[3] == ["kubectl", "rollout", "status", f"deployment/{plan.deployment_name}"]
    assert plan.applied is True and plan.endpoint == "http://10.1.2.3:80"


def test_s8_02_apply_stops_at_the_first_failure_and_records_why(trained_package, monkeypatch):
    """S8.02 — a failed docker build means no kubectl, and a note saying so."""
    calls: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        return (False, "no space left on device") if cmd[:2] == ["docker", "build"] else (True, "")

    monkeypatch.setattr(serving, "_have", lambda tool: True)
    monkeypatch.setattr(serving, "_run", fake_run)

    plan = serving.apply(serving.prepare(str(trained_package)))
    assert [c[:2] for c in calls] == [["docker", "build"]]
    assert plan.applied is False and plan.endpoint is None
    assert any("docker build: failed" in n and "no space left" in n for n in plan.notes)


def test_s8_03_apply_without_tools_is_a_plan_with_notes_not_a_crash(trained_package, monkeypatch):
    """S8.03 — no docker, no kubectl: both are named in the notes; nothing is run."""
    monkeypatch.setattr(serving, "_have", lambda tool: False)
    monkeypatch.setattr(serving, "_run", lambda *a, **k: pytest.fail("must not shell out"))
    plan = serving.apply(serving.prepare(str(trained_package)))
    assert plan.applied is False
    assert any("docker not found" in n for n in plan.notes)
    assert any("kubectl not found" in n for n in plan.notes)


def test_s8_04_rendered_manifest_rewrites_every_deployment_image(synthetic_package):
    """S8.04 — the applied manifest carries the planned image, not the packaged default."""
    import yaml
    plan = serving.prepare(str(synthetic_package), image="registry.example/churn:9")
    docs = list(yaml.safe_load_all(serving._manifest_for_image(plan)))
    deployments = [d for d in docs if isinstance(d, dict) and d.get("kind") == "Deployment"]
    assert deployments
    for d in deployments:
        assert d["spec"]["template"]["spec"]["containers"][0]["image"] == "registry.example/churn:9"


LIVE = pytest.mark.skipif(
    os.environ.get("AVALOKA_TEST_ALLOW_APPLY") != "1"
    or shutil.which("docker") is None or shutil.which("kubectl") is None,
    reason="live apply needs docker + kubectl on PATH and AVALOKA_TEST_ALLOW_APPLY=1",
)


@LIVE
@pytest.mark.cluster
@pytest.mark.slow
def test_s8_05_live_infer_apply_resolves_an_endpoint(trained_package):
    """S8.05 — the live half: build, apply to the current kube context, get a URL.

    Gated twice (marker and env flag) because it builds an image and mutates a
    cluster. Run it on kind, never on a shared context.
    """
    context = subprocess.run(["kubectl", "config", "current-context"],
                             capture_output=True, text=True).stdout.strip()
    assert context.startswith(("kind-", "minikube")), f"refusing to apply to {context!r}"
    result = runner.invoke(app, ["infer", str(trained_package), "--target", "kubernetes",
                                 "--apply", "--quiet"])
    assert result.exit_code == 0, result.output
    assert "REST API live" in result.output
