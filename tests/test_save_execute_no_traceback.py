"""save-and-execute never hands the user a traceback.

When the edited code fails, the executor leaves the script's stderr on
``execution_result.message``. That text used to go back verbatim as
``execution_message``. The user must get plain words instead, and the detail
must stay in the server log.
"""
import json
import logging
import re

import pytest
from fastapi.testclient import TestClient

import app.api.server as server

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import SESSIONS, client, make_auth_headers  # noqa: F401
except ImportError:
    from server_harness import SESSIONS, client, make_auth_headers  # noqa: F401


# What a user-facing string must never carry: a traceback header, a stack
# frame, or a raw exception (its repr, or "Type: message").
RAW_ERROR = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r'File "[^"]*", line \d+'),
    re.compile(r"\b[A-Z]\w*(?:Error|Exception)\((?:'|\")"),
    re.compile(r"\b[A-Z]\w*(?:Error|Exception)\s*:\s*\S"),
]

STDERR = (
    "Traceback (most recent call last):\n"
    '  File "/tmp/avaloka/run_1234/script.py", line 41, in <module>\n'
    "    result = main(df)\n"
    '  File "/tmp/avaloka/run_1234/script.py", line 12, in main\n'
    "    return df.groupby('region')['revenue_2026'].sum()\n"
    "KeyError: 'revenue_2026'\n"
)


def _raw_errors(text: str) -> list:
    return [rx.pattern for rx in RAW_ERROR if rx.search(text)]


def _failed_run(state):
    return {
        **state,
        "execution_error": "Script failed with exit status 1",
        "execution_stderr": STDERR,
        "execution_result": {
            "status": "failed",
            "message": f"Script failed with exit status 1\n\n{STDERR.strip()}",
            "returncode": 1,
            "output_file": None,
        },
        "fresh_output_produced": False,
    }


@pytest.fixture
def failing_save_and_execute(client: TestClient, monkeypatch):
    async def _aid(analysis_id, user_id):
        return {"session_id": "s-noleak"}

    monkeypatch.setattr(server, "_resolve_session_from_aid", _aid, raising=False)
    SESSIONS["s-noleak"] = {"user_id": "noleak-user", "schema": {"region": "object"}}
    monkeypatch.setattr(server, "review_edited_code",
                        lambda state, code, user_intent="": {"code": code, "status": "approved"},
                        raising=False)
    monkeypatch.setattr(server, "execution_agent_node_local", _failed_run, raising=False)
    code = "import pandas as pd\n\ndef main(df):\n    return df.groupby('region')['revenue_2026'].sum()\n"
    return lambda: client.post("/analysis/aid-noleak/save-and-execute",
                               json={"code": code, "run_validation": False},
                               headers=make_auth_headers("noleak-user"))


def test_failed_save_and_execute_returns_plain_words(failing_save_and_execute, caplog):
    with caplog.at_level(logging.ERROR, logger=server.logger.name):
        resp = failing_save_and_execute()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "execution_failed"
    assert body["available"] is False

    message = body["execution_message"]
    assert _raw_errors(message) == [], f"execution_message leaks a raw error: {message!r}"
    assert "revenue_2026" not in message and "script.py" not in message, message
    assert message.startswith("Your code didn't finish"), message
    assert "column the data doesn't have" in message, message

    # Nothing anywhere in the body carries the traceback either.
    assert _raw_errors(json.dumps(body)) == [], body

    # The detail is not lost: it is in the server log.
    assert any("KeyError: 'revenue_2026'" in r.getMessage() for r in caplog.records), \
        "the stderr detail must still reach the server log"


def test_failure_without_detail_still_says_nothing_was_saved():
    message = server._save_and_execute_failure_message({"execution_result": {"status": "failed"}})
    assert "nothing was saved" in message
    assert _raw_errors(message) == []
