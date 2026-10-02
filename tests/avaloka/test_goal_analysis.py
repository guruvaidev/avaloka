"""Different CLI goals must produce different measured answers, not new titles."""

import json

import pandas as pd
from typer.testing import CliRunner

from avaloka.cli import app
from avaloka.goal_analysis import analyze_goal
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze


def _dataset(tmp_path):
    source = tmp_path / "sales.csv"
    pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"],
        "region": ["East", "East", "West", "West"],
        "revenue": [10.0, 20.0, 30.0, 40.0],
        "cost": [5.0, 10.0, 15.0, 20.0],
        "Age": [25.0, None, 35.0, None],
    }).to_csv(source, index=False)
    return source


def _run(tmp_path, source, goal, name):
    out = tmp_path / name
    ctx = MissionContext(
        kind=MissionKind.ANALYZE,
        goal=goal,
        data_source=str(source),
        output_dir=out,
        budget=Budget(),
    )
    run_analyze(ctx)
    return out, json.loads((out / "goal_analysis.json").read_text())


def test_two_goals_on_same_csv_change_the_report_findings(tmp_path):
    source = _dataset(tmp_path)
    missing_out, missing = _run(tmp_path, source, "Analyze missing values in Age", "missing")
    group_out, group = _run(tmp_path, source, "Compare total revenue by region", "groups")

    assert missing["kind"] == "missingness"
    assert missing["metrics"]["Age"]["missing"] == 2
    assert group["kind"] == "group_comparison"
    assert group["metrics"]["groups"] == [
        {"group": "West", "value": 70.0},
        {"group": "East", "value": 30.0},
    ]
    assert missing["findings"] != group["findings"]
    missing_html = (missing_out / "executive_report.html").read_text()
    group_html = (group_out / "executive_report.html").read_text()
    assert "Age: 2 missing" in missing_html
    assert "region=West: sum revenue=70" in group_html
    assert "region=West: sum revenue=70" not in missing_html
    assert "Age: 2 missing" not in group_html
    assert "Goal-specific analysis" in (group_out / "technical_report.html").read_text()
    assert "avaloka analyze" in (group_out / "README.md").read_text()


def test_correlation_and_time_trend_are_measured(tmp_path):
    source = _dataset(tmp_path)
    _, corr = _run(tmp_path, source, "Correlation between revenue and cost", "corr")
    _, trend = _run(tmp_path, source, "Monthly trend in revenue", "trend")
    assert corr["kind"] == "correlation"
    assert {"left": "revenue", "right": "cost", "r": 1.0} in corr["metrics"]["pairs"]
    assert trend["kind"] == "trend"
    assert trend["metrics"]["monthly_values"] == {"2026-01": 30.0, "2026-02": 70.0}


def test_causal_why_question_reports_measured_trend_but_not_a_causal_answer(tmp_path):
    source = _dataset(tmp_path)
    _, result = _run(tmp_path, source, "Why did revenue drop over time?", "why")
    assert result["kind"] == "trend"
    assert result["status"] == "limited"
    assert "not an explanation of its cause" in result["findings"][-1]


def test_unsupported_goal_is_disclosed_instead_of_pretending_to_answer():
    frame = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    result = analyze_goal(frame, "What caused the CEO to resign?", {"columns": []})
    assert result["status"] == "limited"
    assert "not an answer" in result["findings"][0]


def test_highest_numeric_value_answers_the_requested_statistic_only():
    frame = pd.DataFrame({"salary": [50_000, 70_000, 95_000, 95_000, 120_000, 8_000]})
    result = analyze_goal(frame, "what is the highest salary?", {"columns": []})

    assert result["kind"] == "column_summary"
    assert result["metrics"]["salary"] == {"non_null": 6, "max": 120_000.0}
    assert result["findings"] == ["salary: highest=120,000 (from 6 non-null rows)."]
    assert "mean" not in result["metrics"]["salary"]
    assert "median" not in result["metrics"]["salary"]


def test_numeric_statistic_intents_are_general_across_columns():
    frame = pd.DataFrame({
        "temperature": [10.0, 15.0, 20.0, None],
        "revenue": [100.0, 200.0, 300.0, 400.0],
        "score": [2.0, 6.0, 10.0, 14.0],
    })
    cases = [
        ("What is the lowest temperature?", "temperature", {"min": 10.0}),
        ("What is the total revenue?", "revenue", {"sum": 1000.0}),
        ("What is the average score?", "score", {"mean": 8.0}),
        ("What is the range of score?", "score", {"range": 12.0, "min": 2.0, "max": 14.0}),
        ("How many temperature values are non-null?", "temperature", {}),
    ]

    for goal, column, expected in cases:
        metrics = analyze_goal(frame, goal, {"columns": []})["metrics"][column]
        assert {key: metrics[key] for key in expected} == expected
        if "mean" not in expected:
            assert "mean" not in metrics
        if "median" not in expected:
            assert "median" not in metrics


def test_cli_goal_reaches_the_measured_report(tmp_path):
    source = _dataset(tmp_path)
    output = tmp_path / "cli"
    invocation = CliRunner().invoke(app, [
        "analyze", str(source), "--goal", "Analyze missing values in Age",
        "--output", str(output), "--quiet",
    ])
    assert invocation.exit_code == 0, invocation.output
    report = json.loads((output / "goal_analysis.json").read_text())
    assert report["metrics"]["Age"]["missing"] == 2
    assert "Age: 2 missing" in invocation.output


def test_cli_default_output_preserves_two_runs(tmp_path, monkeypatch):
    source = _dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    for goal in ("Analyze missing values in Age", "Compare total revenue by region"):
        invocation = runner.invoke(app, ["analyze", str(source), "--goal", goal, "--quiet"])
        assert invocation.exit_code == 0, invocation.output
    outputs = sorted((tmp_path / "avaloka-analysis").iterdir())
    assert len(outputs) == 2
    kinds = {
        json.loads((out / "goal_analysis.json").read_text())["kind"]
        for out in outputs
    }
    assert kinds == {"missingness", "group_comparison"}
