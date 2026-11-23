"""The sampler's local SQLite reader is off unless a deployment turns it on.

It opens arbitrary local .sqlite files, so ENABLE_SQLITE defaults to 0 and SQL
sources go through the Postgres MCP. The benchmark's sqlite-backed sources opt
in per test (tests/benchmark/conftest.py::sqlite_reader_for); nothing may turn
it on for the whole CI step or session.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REFUSAL = "Local SQLite reading is disabled (ENABLE_SQLITE=0)"

_PROBE = """
import json, sys
import app.agents.sampling_agent as sa
out = sa.sample_data_from_source(sys.argv[1], "sql", stratify_by=None, sample_size=2)
print(json.dumps({"enabled": sa.ENABLE_SQLITE, "error": out.get("error"), "rows": len(out.get("rows") or [])}))
"""


def test_the_default_refuses_a_local_sqlite_file(tmp_path):
    db = tmp_path / "data.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "x"), (2, "y"), (3, "z")])
    env = {k: v for k, v in os.environ.items() if k != "ENABLE_SQLITE"}

    proc = subprocess.run([sys.executable, "-c", _PROBE, str(db)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)

    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["enabled"] is False, "ENABLE_SQLITE must default to off"
    assert result["rows"] == 0, "rows were read from a local .sqlite file with the reader off"
    assert REFUSAL in (result["error"] or ""), result["error"]


def test_the_test_session_has_not_turned_the_reader_on():
    """Guards against ENABLE_SQLITE=1 on the CI step: the benchmark opts in per test."""
    import app.agents.sampling_agent as sampling_agent

    assert os.environ.get("ENABLE_SQLITE", "0") == "0", "ENABLE_SQLITE is set for the whole session"
    assert sampling_agent.ENABLE_SQLITE is False
