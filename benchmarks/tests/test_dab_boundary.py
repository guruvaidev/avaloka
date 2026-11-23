"""DAB integration must exercise Avaloka reasoning without importing its reference agent."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

from benchmarks.adapters import requires_cloud
from benchmarks.adapters.dab_command import AvalokaDABCommandAdapter
from benchmarks.core.config import Budget, RunConfig
from benchmarks.core.task import BenchmarkTask, DatasetSpec
from benchmarks.run import build_parser


def test_dab_bridge_does_not_import_reference_data_agent():
    script = Path(__file__).parents[2] / "scripts" / "avaloka_dab_agent.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert "common_scaffold.DataAgent" not in imports


def test_avaloka_dab_is_a_paid_hosted_model_boundary():
    assert requires_cloud("avaloka_dab")


def test_runner_accepts_database_type_and_resume_controls():
    args = build_parser().parse_args([
        "run", "dab", "--database-type", "mongo", "--no-resume",
    ])
    assert args.database_type == ["mongo"]
    assert args.no_resume


def test_v2_request_excludes_validator_and_expected_answer(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(json.loads(kwargs["input"]))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"answer": "7", "latency_s": 0.01,
                               "tokens": 12, "tool_calls": 1}),
            stderr="",
        )

    monkeypatch.setenv("AVALOKA_DAB_AGENT_CMD", "fake-avaloka")
    monkeypatch.setattr("benchmarks.adapters.dab_command.subprocess.run", fake_run)
    task = BenchmarkTask(
        id="dab:yelp:1",
        track="dab",
        mission="What is the value?",
        dataset=DatasetSpec(id="yelp", fingerprint="fixture"),
        evaluator=lambda _task, _answer: {"pass": 1.0},
        params={
            "dataset": "yelp",
            "query": "1",
            "db_description": "two sanctioned sources",
            "db_config_path": "/fixtures/db_config.yaml",
            "dataset_dir": "/fixtures/yelp",
            "database_types": ("duckdb", "mongo"),
        },
    )
    config = RunConfig(
        id="dab", sut="avaloka_dab", llm="gemini-test", trials=1,
        budget=Budget(max_cost_usd=2.0, max_latency_s=10, max_tokens=100),
    )
    outcome = AvalokaDABCommandAdapter().run_mission(task, config)
    assert outcome.answer == "7"
    assert outcome.cost_usd is None
    assert captured["protocol_version"] == "dab-command-v2"
    encoded = json.dumps(captured)
    assert "validate.py" not in encoded
    assert "validator" not in encoded
    assert "expected_answer" not in encoded
    assert captured["budget"]["max_cost_usd"] == 2.0
