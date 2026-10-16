"""CLI test plan §6 — findings F1–F7, each pinned by a test.

Every test here asserts the *intended* behaviour and is marked
``xfail(strict=True)`` under the ``defect`` marker (pytest.ini). While the
defect stands the test fails as expected and the suite stays green. The moment
a fix lands the test passes, strict xfail turns that into a failure, and the
person fixing it removes the marker in the same change. Nothing is skipped and
nothing is deleted; the suite always says exactly what is still broken.

Each finding was reproduced against the CLI before being written down.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from avaloka.cli import app
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_train

REPO = Path(__file__).resolve().parents[2]
runner = CliRunner()
pytestmark = pytest.mark.defect


def flat(output: str) -> str:
    return " ".join(output.split())


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="F1: string-dtype nulls (pd.NA) reach sklearn's imputer; "
                          "modeling.build_preprocessor must see object/NaN")
def test_f1_training_survives_nulls_in_text_columns(titanic_like_csv, tmp_path):
    """F1 — a dataset with missing values in text columns trains.

    The Data Engineer casts text to pandas' ``string`` dtype, whose missing
    value is ``pd.NA``. scikit-learn's imputer checks for NaN with ``X != X``,
    and ``pd.NA`` refuses to be a boolean, so the Model Scientist dies with
    "boolean value of NA is ambiguous". Titanic has nulls in Cabin and
    Embarked; so does almost every real export.
    """
    out = tmp_path / "model"
    ctx = MissionContext(kind=MissionKind.TRAIN, goal="Predict survival",
                         data_source=str(titanic_like_csv), output_dir=out,
                         budget=Budget(), target="survived", metric="roc_auc")
    result = run_train(ctx)
    assert (out / "model" / "model.pkl").exists()
    assert result.summary["model"]["selected"]


def _analyze_with_relative_path(churn_csv, tmp_path, monkeypatch) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "churn.csv").write_bytes(churn_csv.read_bytes())
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "bundle"
    result = runner.invoke(app, ["analyze", "data/churn.csv", "--goal", "x",
                                 "-o", str(bundle), "--quiet"])
    assert result.exit_code == 0, result.output
    return bundle


@pytest.mark.xfail(strict=True,
                   reason="F2: analysis.py embeds the data path exactly as typed, so a "
                          "relative path only resolves from the original working directory")
def test_f2_generated_script_reproduces_from_inside_the_bundle(churn_csv, tmp_path, monkeypatch):
    """F2 — `python analysis.py`, run where the README tells you to run it, works.

    The bundle README says ``python analysis.py``. The script's DATA_SOURCE is
    the path the user typed, so from inside the bundle a relative path gives
    FileNotFoundError. The Data Engineer should resolve the source to an
    absolute path (or copy the data into the bundle) at generation time.
    """
    bundle = _analyze_with_relative_path(churn_csv, tmp_path, monkeypatch)
    result = subprocess.run([sys.executable, "analysis.py"], cwd=str(bundle),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.xfail(strict=True,
                   reason="F3: analysis.py writes transformed_dataset.parquet into the "
                          "current directory instead of the bundle")
def test_f3_generated_script_writes_its_output_into_the_bundle(churn_csv, tmp_path, monkeypatch):
    """F3 — the reproduced parquet lands next to the script, not wherever you stood."""
    bundle = _analyze_with_relative_path(churn_csv, tmp_path, monkeypatch)
    result = subprocess.run([sys.executable, str(bundle / "analysis.py")], cwd=str(tmp_path),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "transformed_dataset.parquet").exists(), \
        "the script wrote into the working directory"


@pytest.mark.xfail(strict=True,
                   reason="F4: analyze/train --llm with no ANTHROPIC_API_KEY degrades "
                          "silently; only chat prints a notice")
def test_f4_llm_flag_without_a_key_says_so(churn_csv, tmp_path):
    """F4 — asking for Claude and not getting it must be visible.

    ``avaloka chat --llm`` prints "(No ANTHROPIC_API_KEY found ...)". The
    mission commands accept ``--llm`` and say nothing, so a user with a typo in
    their key sees templated prose and assumes that is what the LLM produced.
    The platform fixed the same class of silence in September 2026
    (architecture.md §11, "Honest failure"); the CLI should match.
    """
    result = runner.invoke(app, ["analyze", str(churn_csv), "--goal", "x",
                                 "-o", str(tmp_path / "out"), "--llm"])
    assert result.exit_code == 0, result.output
    assert "ANTHROPIC_API_KEY" in result.output


@pytest.mark.xfail(strict=True,
                   reason="F5: with no |corr| >= 0.5 and no quality issue the key findings "
                          "fall back to a generic sentence with no measured statistic")
def test_f5_key_findings_always_carry_a_measured_statistic(uncorrelated_csv, tmp_path):
    """F5 — "every observation is pinned to a measured statistic" (README_CLI.md).

    On a dataset with no strong correlations the reporter's only finding is
    the planner's hypothesis, "Surface the dominant segments and the variables
    that separate them", which names no column and no number. The strongest
    measured association, however weak, is still a measurement; silence
    dressed as a finding is not.
    """
    out = tmp_path / "out"
    result = runner.invoke(app, ["analyze", str(uncorrelated_csv), "--goal", "what matters?",
                                 "-o", str(out), "--quiet"])
    assert result.exit_code == 0, result.output
    findings = re.findall(r"^\s*•\s*(.+)$", result.output.split("What to be careful about")[0],
                          flags=re.MULTILINE)
    assert findings, "the console printed no findings at all"
    columns = {"alpha", "beta", "gamma", "group"}
    grounded = [f for f in findings if any(c in f for c in columns) or re.search(r"\d", f)]
    assert grounded, f"no finding names a column or a number: {findings}"


@pytest.mark.xfail(strict=True,
                   reason="F6: in --quiet mode the console prints the model's holdout score "
                          "without its baseline; the baseline only appears in persona prose")
def test_f6_quiet_mode_never_prints_a_score_without_its_baseline(churn_csv, tmp_path):
    """F6 — "a model score never appears without the baseline it has to beat".

    The baseline sentence lives in the persona's closing line. ``--quiet``
    removes the persona, and the verdict block prints only WARN and FAIL
    checks, so a passing beats_baseline check is invisible and the console
    shows "Best model 'x' scored 0.81 roc_auc on holdout" with nothing to
    compare it to. Scripted runs are exactly where this matters.
    """
    result = runner.invoke(app, ["train", str(churn_csv), "--target", "churned",
                                 "--metric", "roc_auc", "-o", str(tmp_path / "out"), "--quiet"])
    assert result.exit_code == 0, result.output
    text = flat(result.output)
    if "beats_baseline:" in text:
        pytest.skip("this run did not beat its baseline, so the FAIL check printed it; "
                    "the finding is about the passing case")
    assert "scored" in text
    assert "baseline" in text.lower()


@pytest.mark.xfail(strict=True,
                   reason="F8: ml_engineer writes deployment/kubernetes.yaml as one YAML list, "
                          "not two documents; serving cannot find the Deployment in it")
def test_f8_generated_kubernetes_manifest_is_multi_document(trained_package):
    """F8 — the package's own manifest must be what `kubectl apply -f` and
    `serving` both expect: a Deployment document and a Service document.

    ``write_yaml(path, [deployment, service])`` dumps a single document whose
    top level is a sequence. ``serving._read_manifest`` sees no dict, so the
    plan falls back to the default names instead of the real ``avaloka-<id>``;
    ``serving._manifest_for_image`` raises "No Deployment found"; and kubectl
    would reject a bare list. ``avaloka infer --apply`` therefore cannot pass
    the manifest step on any bundle Avaloka produced.
    """
    import yaml

    docs = list(yaml.safe_load_all((trained_package / "deployment" / "kubernetes.yaml").read_text()))
    assert len(docs) == 2 and all(isinstance(d, dict) for d in docs), \
        f"expected two documents, got {len(docs)} of type {[type(d).__name__ for d in docs]}"
    assert {d["kind"] for d in docs} == {"Deployment", "Service"}


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="F8: serving.prepare cannot read the generated manifest, so the plan "
                          "carries default names and rendering raises")
def test_f8_serving_reads_the_generated_manifest(trained_package):
    """F8 — prepare() on Avaloka's own package knows the real Deployment name,
    and the manifest renders for apply."""
    from avaloka import serving

    plan = serving.prepare(str(trained_package))
    rendered = serving._manifest_for_image(plan)          # raises ValueError today
    assert "kind: Deployment" in rendered
    assert plan.deployment_name.startswith("avaloka-") and plan.deployment_name != "avaloka-inference"


@pytest.mark.xfail(strict=True,
                   reason="F7: README_CLI.md does not document `avaloka version`")
def test_f7_every_command_is_documented():
    """F7 — the CLI guide names every command the binary has."""
    readme = (REPO / "README_CLI.md").read_text()
    import typer.main
    commands = set(typer.main.get_command(app).commands)   # includes sub-apps
    undocumented = sorted(c for c in commands if not re.search(rf"\bavaloka {c}\b", readme))
    assert not undocumented, f"undocumented commands: {undocumented}"
