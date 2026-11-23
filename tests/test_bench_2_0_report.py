"""docs/benchmarks/2.0-vs-1.0.md must stay rendered from the committed records.

The report's tables are the output of scripts/bench/summarize_2_0_vs_1_0.py over
benchmarks/results/2.0/. If a record changes and the report is not regenerated,
or a number in the tables is edited by hand, this fails.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "benchmarks" / "2.0-vs-1.0.md"
RESULTS = ROOT / "benchmarks" / "results" / "2.0"
SCRIPT = ROOT / "scripts" / "bench" / "summarize_2_0_vs_1_0.py"


def _rendered() -> str:
    out = subprocess.run([sys.executable, str(SCRIPT), str(RESULTS)], check=True,
                         capture_output=True, text=True).stdout
    return re.sub(r"(?m)^### ", "#### ", out).strip()


def _report_tables() -> str:
    text = REPORT.read_text()
    start = text.index("### 5.4 Tables (rendered)")
    end = text.index("### 5.5 Provenance")
    return text[start:end].split("\n", 1)[1].strip()


def test_report_tables_are_the_script_output():
    assert _report_tables() == _rendered()


def test_dab_headlines_print_their_baseline_and_error_counts():
    rendered = _rendered()
    haiku = next(l for l in rendered.splitlines() if l.startswith("| 2.0 harness | `openrouter/anthropic/claude-haiku-5.5`"))
    groq = next(l for l in rendered.splitlines() if l.startswith("| 2.0 harness | `groq/openai/gpt-oss-120b`"))
    # system | model | n | pass@1 | CI | macro | passed | all-errored | infra | provider codes | ...
    assert [c.strip() for c in haiku.split("|")[3:11]] == ["54", "0.54", "0.41–0.66", "0.52", "29", "7", "0", "0"]
    assert [c.strip() for c in groq.split("|")[3:10]] == ["54", "0.22", "0.13–0.35", "0.23", "12", "19", "7"]
    assert "`tool_use_failed` 6" in groq and "`context_length_exceeded` 1" in groq
    headline = REPORT.read_text().split("### 5.1 DAB headline", 1)[1].split("### 5.2", 1)[0]
    for needle in ("f772b0fb", "29/54 = 0.54", "14/54 = 0.26", "12/54 = 0.22", "6/54 = 0.11", "$1.88", "$2.05"):
        assert needle in headline, needle


def test_every_dab_record_has_a_log_and_a_commit():
    import json
    for record in sorted((RESULTS / "dab").glob("*.json")):
        meta = json.loads(record.read_text())["meta"]
        assert meta.get("git_commit"), record.name
        assert (RESULTS / "dab" / "logs" / f"{record.stem}.log").is_file(), record.name
