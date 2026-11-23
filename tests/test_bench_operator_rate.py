"""The DAB cell rate check compares a model against the provider that bills it.

``run_cell.sh`` refuses to start a cell unless the operator rate it charges
trials at equals the provider's rate. It used to strip ``groq/`` and look the
id up in the OpenRouter catalog, which prices the same model id for
OpenRouter's routing, not Groq's bill. A ``groq/`` model is now checked
against the Groq table in ``benchmarks/catalog/provider_rates.json``, and a
mismatch or a missing entry still refuses.

Run as a subprocess, the way run_cell.sh runs it (``python3 -I``).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench" / "check_operator_rate.py"
CATALOG = ROOT / "benchmarks" / "catalog"


def _check(*args, catalog=None):
    cmd = [sys.executable, "-I", str(SCRIPT), *args]
    if catalog is not None:
        cmd += ["--catalog-dir", str(catalog)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


@pytest.fixture
def diverging_catalog(tmp_path):
    # The same id priced differently by OpenRouter and by Groq.
    (tmp_path / "openrouter_models.json").write_text(json.dumps({"data": [
        {"id": "openai/gpt-oss-120b",
         "pricing": {"prompt": "0.0000001", "completion": "0.0000005"}},
        {"id": "anthropic/claude-haiku-5.5",
         "pricing": {"prompt": "0.0000001", "completion": "0.0000005"}},
    ]}))
    (tmp_path / "provider_rates.json").write_text(json.dumps({"groq": {
        "openai/gpt-oss-120b": {"input_usd_per_mtok": 0.15, "output_usd_per_mtok": 0.60,
                                "source_url": "https://example.invalid", "checked_at": "x"},
    }}))
    return tmp_path


def test_a_groq_model_is_checked_against_the_groq_rate(diverging_catalog):
    ok = _check("groq/openai/gpt-oss-120b", "0.15", "0.60", catalog=diverging_catalog)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "groq table" in ok.stdout and "OK" in ok.stdout


def test_a_groq_model_at_the_openrouter_price_is_refused(diverging_catalog):
    wrong = _check("groq/openai/gpt-oss-120b", "0.10", "0.50", catalog=diverging_catalog)
    assert wrong.returncode == 1, wrong.stdout + wrong.stderr
    assert "MISMATCH" in wrong.stdout


def test_a_groq_model_without_a_groq_entry_is_refused(diverging_catalog):
    # Present in the OpenRouter catalog, absent from the Groq table: no fallback.
    missing = _check("groq/anthropic/claude-haiku-5.5", "0.10", "0.50",
                     catalog=diverging_catalog)
    assert missing.returncode == 1, missing.stdout + missing.stderr
    assert "REFUSED" in missing.stdout


def test_an_openrouter_model_is_still_checked_against_the_catalog(diverging_catalog):
    ok = _check("openrouter/anthropic/claude-haiku-5.5", "0.10", "0.50",
                catalog=diverging_catalog)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    bad = _check("openrouter/anthropic/claude-haiku-5.5", "0.15", "0.50",
                 catalog=diverging_catalog)
    assert bad.returncode == 1


def test_the_committed_groq_entry_states_its_source_and_date():
    table = json.loads((CATALOG / "provider_rates.json").read_text())
    entry = table["groq"]["openai/gpt-oss-120b"]
    assert entry["source_url"].startswith("https://console.groq.com/")
    assert entry["checked_at"]
    ok = _check("groq/openai/gpt-oss-120b", str(entry["input_usd_per_mtok"]),
                str(entry["output_usd_per_mtok"]))
    assert ok.returncode == 0, ok.stdout + ok.stderr
