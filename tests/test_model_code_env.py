"""Model-written code in the bench harnesses must not see the parent's credentials.

Both scripts ran the program the model wrote with ``{**os.environ}``, so a
provider key the harness itself needs was readable by that program. Each test
plants fake keys in the parent, runs a program that reports the NAMES of the
credential-like variables it can see (never a value), and expects none.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for entry in (str(REPO / "scripts" / "bench"), str(REPO / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

PLANTED = ("OPENROUTER_API_KEY", "GROQ_API_KEY", "BITBUCKET_ACCESS_TOKEN", "DB_ENCRYPTION_KEY")

#: Prints the names (only) of every credential-like variable the program can see.
REPORT_NAMES = (
    "import os\n"
    "names = sorted(k for k in os.environ\n"
    "               if any(p in k.upper() for p in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')))\n"
)


@pytest.fixture
def planted_keys(monkeypatch):
    for name in PLANTED:
        monkeypatch.setenv(name, "planted-not-a-real-key")
    return PLANTED


def _visible(line_prefix: str, text: str) -> list[str]:
    line = next(ln for ln in text.splitlines() if ln.startswith(line_prefix))
    rest = line[len(line_prefix):].strip()
    return [n for n in rest.split(",") if n]


# --- scripts/bench/ds_model_alone.py ------------------------------------------

def test_ds_model_alone_program_sees_no_credential(planted_keys):
    import pandas as pd
    import ds_model_alone as ds
    program = (REPORT_NAMES +
               "import sys, pandas as pd\n"
               "print('VISIBLE:' + ','.join(names), file=sys.stderr)\n"
               "t = pd.read_csv('test.csv')\n"
               "pd.DataFrame({'prediction': [0.5] * len(t)}).to_csv('predictions.csv', index=False)\n")
    train = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    values, stderr = ds._run_program(program, train, train[["x"]], timeout=60)
    assert values is not None, stderr
    assert _visible("VISIBLE:", stderr) == []


# --- scripts/bench/oneshot_dab_agent.py ---------------------------------------

class _StubProvider:
    """Answers with a program that prints the visible credential names as its result."""

    def __init__(self, *args, **kwargs):
        self.retries = 0

    def complete(self, messages, tools, *, timeout_s):
        from avaloka.data_agent import ModelUsage
        from avaloka.data_agent.contracts import ModelResponse
        program = REPORT_NAMES + "print('__RESULT__:VISIBLE:' + ','.join(names))\n"
        return ModelResponse(response_id=None, content=f"```python\n{program}```",
                             tool_calls=[], usage=ModelUsage(1, 1, 2, 0))


def test_oneshot_program_sees_no_credential(planted_keys, tmp_path, monkeypatch, capsys):
    import oneshot_dab_agent as oneshot
    dab_root = tmp_path / "dab"
    (dab_root / "common_scaffold" / "tools").mkdir(parents=True)
    config = tmp_path / "db_config.yaml"
    config.write_text("db_clients: {}\n", encoding="utf-8")
    monkeypatch.setenv("DAB_ROOT", str(dab_root))
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))    # main() prepends dab-bin
    monkeypatch.setattr(oneshot, "_ensure_databases", lambda path: None)
    monkeypatch.setattr(oneshot, "OpenAICompatibleProvider", _StubProvider)
    request = {
        "protocol_version": "dab-command-v2",
        "run": {"dataset": "probe", "query": 1, "trial": 0},
        "mission": {"question": "q", "database_description": "d"},
        "sources": {"database_config_path": str(config), "working_directory": str(tmp_path)},
        "model": {"provider_model": "stub/model"},
        "budget": {}, "policy": {},
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    oneshot.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] is None, payload["error"]
    assert _visible("VISIBLE:", payload["answer"]) == []


# --- the shared allowlist ------------------------------------------------------

@pytest.mark.parametrize("module", ["ds_model_alone", "oneshot_dab_agent"])
def test_both_harnesses_use_the_executor_allowlist(module):
    from app.executor.env import scrubbed_env
    mod = __import__(module)
    assert mod.scrubbed_env is scrubbed_env
