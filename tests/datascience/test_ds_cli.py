"""The same data-science problems, driven the way a user drives them.

tests/datascience/test_ds_lifecycle.py calls the missions directly. This asks
whether the *terminal* is a usable instrument for the same work: does the
command surface reach these capabilities, and does it say enough on the way out
that a data scientist can act on the result without opening the bundle?
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from avaloka.cli import app

runner = CliRunner()


def _run(args):
    result = runner.invoke(app, args)
    assert "Traceback" not in result.output, result.output
    return result


# ── The everyday commands land ──────────────────────────────────────────────

def test_analyze_answers_an_exploratory_question(churn_like, write_csv, tmp_path):
    csv = write_csv(churn_like)
    out = tmp_path / "out"

    result = _run(["analyze", str(csv), "--goal", "Who is at risk of churning?",
                   "-o", str(out), "--quiet"])

    assert result.exit_code == 0, result.output
    assert "What I found" in result.output
    assert (out / "executive_report.html").exists()


def test_train_builds_a_model_from_the_terminal(churn_like, write_csv, tmp_path):
    csv = write_csv(churn_like)
    out = tmp_path / "out"

    result = _run(["train", str(csv), "--target", "churned", "--metric", "roc_auc",
                   "-o", str(out), "--quiet"])

    assert result.exit_code == 0, result.output
    assert (out / "model" / "model.pkl").exists()


def test_the_terminal_reports_a_leak_rather_than_a_score(leaky, write_csv, tmp_path):
    """The dangerous output is a 1.000 with nothing said about why."""
    csv = write_csv(leaky)
    out = tmp_path / "out"

    result = _run(["train", str(csv), "--target", "defaulted", "--metric", "roc_auc",
                   "-o", str(out), "--quiet"])

    report = json.loads((out / "validation_report.json").read_text())
    assert report["verdict"] == "fail"
    assert "FAIL" in result.output or "fail" in result.output.lower()
    assert "target_leakage" in result.output, (
        "the console must name the failing check, not just the verdict"
    )


def test_a_warned_run_explains_which_check_warned(write_csv, tmp_path):
    frame = pd.DataFrame({"x": range(12), "y": [0, 1] * 6})
    csv = write_csv(frame)
    out = tmp_path / "out"

    result = _run(["analyze", str(csv), "--goal", "anything", "-o", str(out), "--quiet"])

    assert "Validation verdict" in result.output
    assert "data_sufficiency" in result.output


# ── The formats a data scientist actually has ───────────────────────────────

@pytest.mark.parametrize("suffix,writer", [
    ("csv", lambda df, p: df.to_csv(p, index=False)),
    ("tsv", lambda df, p: df.to_csv(p, sep="\t", index=False)),
    ("parquet", lambda df, p: df.to_parquet(p)),
])
def test_the_common_tabular_formats_all_analyse(churn_like, tmp_path, suffix, writer):
    path = tmp_path / f"data.{suffix}"
    writer(churn_like, path)
    out = tmp_path / f"out_{suffix}"

    result = _run(["analyze", str(path), "--goal", "profile this",
                   "-o", str(out), "--quiet"])

    assert result.exit_code == 0, result.output
    assert (out / "data_quality.json").exists()


def test_excel_with_a_header_below_a_title_block(churn_like, tmp_path):
    """Spreadsheets are how plans and trackers arrive, titles and all."""
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "plan.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"A": ["Quarterly plan", "", "owner: analytics"]}).to_excel(
            writer, index=False, header=False, sheet_name="Sheet1")
        churn_like.to_excel(writer, index=False, startrow=4, sheet_name="Sheet1")
    out = tmp_path / "out"

    result = _run(["analyze", str(path), "--goal", "profile this",
                   "-o", str(out), "--quiet"])

    assert result.exit_code == 0, result.output
    profile = json.loads((out / "data_quality.json").read_text())
    names = {c["name"] for c in profile["columns"]}
    assert "churned" in names, f"header row not found; got {names}"


# ── Getting it wrong at the command line ────────────────────────────────────

def test_a_target_typo_is_caught_before_the_work(churn_like, write_csv, tmp_path):
    csv = write_csv(churn_like)
    result = _run(["train", str(csv), "--target", "churn",
                   "-o", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "churned" in result.output, "suggest the column they meant"


def test_an_unknown_metric_names_the_valid_ones(churn_like, write_csv, tmp_path):
    csv = write_csv(churn_like)
    result = _run(["train", str(csv), "--target", "churned", "--metric", "auc",
                   "-o", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "roc_auc" in result.output


def _run_as_user(monkeypatch, capsys, args):
    """Drive the real entry point, boundary and all.

    CliRunner invokes ``app`` directly, which skips ``main()`` -- and ``main()``
    is where the exception boundary lives. The dataset is loaded lazily inside
    the mission, so the failures worth testing here surface past every argument
    check and are only rendered by that boundary. Testing through ``app`` would
    assert on a path no user takes.
    """
    from avaloka.cli import main

    monkeypatch.setattr("sys.argv", ["avaloka", *args])
    with pytest.raises(SystemExit) as exit_info:
        main()
    return exit_info.value.code, capsys.readouterr().out


def test_an_empty_export_is_refused_not_analysed(tmp_path, monkeypatch, capsys):
    """A header with no rows once produced a full report and a multiplier."""
    path = tmp_path / "empty.csv"
    path.write_text("customer_id,churned\n")

    code, output = _run_as_user(
        monkeypatch, capsys,
        ["analyze", str(path), "--goal", "x", "-o", str(tmp_path / "out"), "--quiet"],
    )

    assert code == 1
    assert "no rows" in output
    assert "Traceback" not in output


def test_a_ragged_export_fails_with_a_sentence(tmp_path, monkeypatch, capsys):
    """Mismatched field counts are a pandas ParserError deep in the mission."""
    path = tmp_path / "ragged.csv"
    path.write_text("a,b\n1,2\n3\n4,5,6\n")

    code, output = _run_as_user(
        monkeypatch, capsys,
        ["analyze", str(path), "--goal", "x", "-o", str(tmp_path / "out"), "--quiet"],
    )

    assert code == 1
    assert "Traceback" not in output
    assert "AVALOKA_TRACEBACK" in output, "tell the developer how to get the trace"
