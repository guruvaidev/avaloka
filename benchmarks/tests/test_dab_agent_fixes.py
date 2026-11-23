"""DAB pilot fixes, found by running the agent against the official checkout.

Each test fails on the code before its fix.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# --- provider routing ------------------------------------------------------
# Bench models are local Gemma first, then Groq gpt-oss-120b, never Anthropic.
# Routed by family name alone, "openai/gpt-oss-120b" matched "gpt" and was sent
# to Azure, and no local route existed, so neither policy model could run.

def _provider(model, tmp_path):
    from scripts.avaloka_dab_agent import OpenAICompatibleProvider
    return OpenAICompatibleProvider(model, tmp_path / "calls.jsonl", max_output_tokens=64)


def test_groq_prefix_builds_a_groq_client_not_azure(tmp_path, monkeypatch):
    from openai import AzureOpenAI
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    provider = _provider("groq/openai/gpt-oss-120b", tmp_path)
    assert not isinstance(provider.client, AzureOpenAI)
    assert str(provider.client.base_url).startswith("https://api.groq.com/openai/v1")
    assert provider.model == "openai/gpt-oss-120b"     # prefix stripped for the API


def test_groq_prefix_without_a_key_is_a_provider_error(tmp_path, monkeypatch):
    from avaloka.data_agent.contracts import ProviderError
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        _provider("groq/openai/gpt-oss-120b", tmp_path)


@pytest.mark.parametrize("host,expected", [
    (None, "http://127.0.0.1:11434/v1"),
    ("http://gpu-box:11434/", "http://gpu-box:11434/v1"),
    ("127.0.0.1:11434", "http://127.0.0.1:11434/v1"),   # Ollama's own host:port form
])
def test_ollama_prefix_routes_to_the_local_server(tmp_path, monkeypatch, host, expected):
    if host is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", host)
    provider = _provider("ollama/gemma4:e4b", tmp_path)
    assert str(provider.client.base_url).rstrip("/") == expected.rstrip("/")
    assert provider.model == "gemma4:e4b"


# --- hinted descriptions -----------------------------------------------------
# Upstream run_agent.py appends db_description_withhint.txt (which holds ONLY
# the hints) to db_description.txt. Using it in place of the description
# dropped every schema and db_name: hinted runs scored 0 on "Unknown db_name".

def _fake_dab_root(root: Path) -> Path:
    from benchmarks.external.dab.adapter import CANONICAL_DATASETS
    (root / "SUBMISSION_RUBRIC.md").write_text("rubric")
    # The loader insists on the canonical 54 queries; spread them 5/4.
    for index, dataset in enumerate(CANONICAL_DATASETS):
        d = root / f"query_{dataset}"
        for q in range(1, (5 if index < 6 else 4) + 1):
            (d / f"query{q}").mkdir(parents=True)
            (d / f"query{q}" / "query.json").write_text('"How many reviews?"')
            (d / f"query{q}" / "validate.py").write_text(
                "def validate(a):\n    return True, 'ok'\n")
        (d / "db_description.txt").write_text(
            f"1. {dataset}_database\n   - table reviews(business_ref, stars)\n")
        (d / "db_description_withhint.txt").write_text(
            "HINTS:\n- business_id maps to business_ref by prefix\n")
        (d / "db_config.yaml").write_text("type: postgres\n")
    return root


def test_hinted_description_keeps_the_schema_and_adds_the_hints(tmp_path, monkeypatch):
    from benchmarks.external.dab.adapter import load_dab_tasks
    monkeypatch.setenv("DAB_ROOT", str(_fake_dab_root(tmp_path)))
    task = next(t for t in load_dab_tasks(use_hints=True) if t.dataset.id == "yelp")
    description = task.dataset.description
    assert "yelp_database" in description and "business_ref, stars" in description
    assert "HINTS:" in description and "maps to business_ref" in description
    assert description.index("yelp_database") < description.index("HINTS:")


def test_unhinted_description_has_no_hints(tmp_path, monkeypatch):
    from benchmarks.external.dab.adapter import load_dab_tasks
    monkeypatch.setenv("DAB_ROOT", str(_fake_dab_root(tmp_path)))
    task = next(t for t in load_dab_tasks(use_hints=False) if t.dataset.id == "yelp")
    assert "yelp_database" in task.dataset.description
    assert "HINTS:" not in task.dataset.description


# --- tool-result keys ---------------------------------------------------------
# Groq call ids carry hyphens. "var_fc_1b15-..." is not an identifier, so the
# model's execute_python code that names it raised NameError every time.

@pytest.mark.parametrize("call_id", ["fc_1b15-ab2c-4d", "call.7", "x y", "abc123"])
def test_a_result_key_is_a_python_identifier_that_code_can_read(call_id):
    from scripts.avaloka_dab_agent import result_key_for
    key = result_key_for(call_id)
    assert key.isidentifier(), key
    namespace = {key: [1, 2, 3]}
    exec(f"value = {key}", namespace)          # what the agent's code does
    assert namespace["value"] == [1, 2, 3]


def test_the_tool_contract_states_the_result_marker():
    """execute_python's spec defers to a PRINT FORMAT DAB defines only in its
    reference agent's prompt; without it every result was '__RESULT__ not found'."""
    from scripts.avaloka_dab_agent import DAB_TOOL_CONTRACT
    assert 'print("__RESULT__:")' in DAB_TOOL_CONTRACT
    assert "collection" in DAB_TOOL_CONTRACT and "filter" in DAB_TOOL_CONTRACT


# --- spend ceiling visibility ---------------------------------------------------
# Every DAB trial came back cost_source="unknown", which the ledger counts as a
# call it could not see: a run reported "10 unpriced/usage-missing -- the
# ceiling could not see these" beside an observed $0.32 it had in fact charged.

def _run_dab_command(tmp_path, monkeypatch, *, model, cost, pricing_configured):
    import json
    import sys
    from benchmarks.adapters.dab_command import AvalokaDABCommandAdapter
    from benchmarks.core.config import RunConfig
    from benchmarks.core.task import BenchmarkTask, DatasetSpec

    payload = {"answer": "42", "cost_usd": cost, "tokens": 10,
               "artifacts": {"pricing_configured": pricing_configured}}
    stub = tmp_path / "stub_agent.py"
    stub.write_text("import sys, json\nsys.stdin.read()\n"
                    f"print(json.dumps({payload!r}))\n")
    monkeypatch.setenv("AVALOKA_DAB_AGENT_CMD", f"{sys.executable} {stub}")
    task = BenchmarkTask(
        id="dab:yelp:1", track="dab", mission="How many?",
        dataset=DatasetSpec(id="yelp", fingerprint="f"), evaluator=lambda t, a: {"pass": 1.0},
        params={"dataset": "yelp", "query": "1", "db_description": "d",
                "db_config_path": str(tmp_path / "c.yaml"), "dataset_dir": str(tmp_path)},
    )
    config = RunConfig(id="dab", sut="avaloka_dab", llm=model, trials=1)
    return AvalokaDABCommandAdapter().run_mission(task, config)


@pytest.mark.parametrize("model,cost,configured,source,blind", [
    ("groq/openai/gpt-oss-120b", 0.032, True, "operator_rate", 0),
    ("groq/openai/gpt-oss-120b", None, False, "unpriced", 1),
    ("ollama/gemma4:e4b", None, False, "local_zero", 0),
])
def test_dab_trials_are_charged_with_their_provenance(tmp_path, monkeypatch,
                                                      model, cost, configured, source, blind):
    from benchmarks.core.spend import SpendLedger
    out = _run_dab_command(tmp_path, monkeypatch, model=model, cost=cost,
                           pricing_configured=configured)
    assert out.error is None, out.error
    assert out.cost_source == source
    ledger = SpendLedger(ceiling_usd=None)
    ledger.charge(out.cost_usd, cost_source=out.cost_source)
    assert ledger.blind_calls == blind
    if cost is not None:
        assert abs(ledger.spent_usd - cost) < 1e-12
