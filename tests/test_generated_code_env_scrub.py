"""Model-written code never sees the server's secrets, memory or globals.

On develop-1.6 the full run (``execution_agent.execute_code_on_local``)
started the generated script with ``subprocess.run`` and no ``env=``, so it
inherited every provider key, database password, ``DB_ENCRYPTION_KEY``,
``MCP_SERVICE_TOKEN`` and ``AVALOKA_PII_KEY`` the server holds; the DTA sample
preview (``daft_execution.execute_code_on_local``) passed ``{**os.environ}``;
and the validator's trial run (``validator.execute_code_node``) was
``exec(code, scope)`` inside the server process.

Every probe value below is synthetic. The positive controls turn the scrub
off and show the same probe does reach the generated code, so a pass here
means the scrub stopped it, not that the probe was broken.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd
import pytest

import app.agents.validator as validator_mod
from app.executor import sample as sample_mod
from app.executor.env import ALLOWED_ENV_NAMES, FORBIDDEN_NAME_PARTS, scrubbed_env

PROBES = {
    "DB_ENCRYPTION_KEY": "PROBE-dbenc-6e1f0c",
    "MCP_SERVICE_TOKEN": "PROBE-mcptok-93ab27",
    "GROQ_API_KEY": "PROBE-groq-41d8c2",
    "POSTGRES_PASSWORD": "PROBE-pgpw-b7730e",
    "PGPASSWORD": "PROBE-pgpass-0c55aa",
    "AVALOKA_PII_KEY": "PROBE-pii-5f2e91",
    "SUPABASE_SERVICE_ROLE_KEY": "PROBE-supa-d40b16",
}

SAMPLE_CSV = "a,b\n1,x\n2,y\n3,z\n"

# Reads the environment every way a script can: the whole of os.environ,
# each probe by name, at import time and inside main(), and puts it in the
# stdout, the returned frame and (for the error variant) the exception.
PROBE_CODE = """
import os
import pandas as pd

AT_IMPORT = dict(os.environ)
print("IMPORT_ENV", AT_IMPORT, flush=True)

def main(df):
    names = %(names)r
    print("ENV", dict(os.environ), flush=True)
    for n in names:
        print("PROBE", n, os.environ.get(n), os.getenv(n), flush=True)
    rows = sorted(os.environ.items()) + sorted(AT_IMPORT.items())
    return pd.DataFrame({"k": [k for k, _ in rows], "v": [v for _, v in rows]})
""" % {"names": sorted(PROBES)}

PROBE_RAISING_CODE = """
import os

def main(df):
    raise RuntimeError("ENV=" + repr(dict(os.environ)))
"""


@pytest.fixture
def probes(monkeypatch):
    for name, value in PROBES.items():
        monkeypatch.setenv(name, value)
    return PROBES


def _leaked(obj) -> list[str]:
    text = json.dumps(obj, default=str)
    return sorted(name for name, value in PROBES.items() if value in text)


def _passthrough_env(source=None, extra=None):
    """The scrub turned off: what the child got before this fix."""
    return {**os.environ, **(extra or {})}


# -------------------------------------------------------------------------
# The allowlist itself (ported from 2.0's tests/test_executor_env.py)
# -------------------------------------------------------------------------

def test_no_secret_survives_the_allowlist():
    secrets = {**PROBES, "OPENROUTER_API_KEY": "o", "SUPABASE_URL": "http://s",
               "DATABASE_URL": "postgresql://u:p@h/d", "REDIS_URL": "redis://r",
               "AWS_SECRET_ACCESS_KEY": "a", "AWS_ACCESS_KEY_ID": "a",
               "LANGGRAPH_API_URL": "http://l", "PYTHONPATH": "/app", "DATA_DIR": "/d"}
    env = scrubbed_env({**secrets, "PATH": "/usr/bin", "HOME": "/root", "LANG": "C.UTF-8"})
    assert not set(secrets) & set(env), sorted(set(secrets) & set(env))
    assert env["PATH"] == "/usr/bin" and env["HOME"] == "/root" and env["LANG"] == "C.UTF-8"


def test_allowlist_names_no_secret_shaped_variable():
    bad = [n for n in ALLOWED_ENV_NAMES if any(p in n.upper() for p in FORBIDDEN_NAME_PARTS)]
    assert not bad, f"the allowlist names a secret-shaped variable: {bad}"


def test_extra_cannot_smuggle_a_secret():
    env = scrubbed_env({}, extra={"DB_ENCRYPTION_KEY": "k", "TMPDIR": "/w"})
    assert "DB_ENCRYPTION_KEY" not in env and env["TMPDIR"] == "/w"


# -------------------------------------------------------------------------
# The full run: execution_agent.execute_code_on_local
# -------------------------------------------------------------------------

def _full_run(tmp_path, code):
    from app.agents.execution_agent import execute_code_on_local

    src = tmp_path / "in.csv"
    src.write_text(SAMPLE_CSV)
    out_csv = tmp_path / "out.csv"
    result = execute_code_on_local(code, str(src), str(out_csv))
    written = out_csv.read_text() if out_csv.exists() else ""
    return result, written


def test_full_run_child_sees_no_probe(probes, tmp_path):
    result, written = _full_run(tmp_path, PROBE_CODE)
    assert result["status"] == "success", result
    assert "PROBE DB_ENCRYPTION_KEY None None" in result["execution_stdout"]
    assert _leaked(result) == [], _leaked(result)
    assert _leaked(written) == [], _leaked(written)


def test_full_run_error_carries_no_probe(probes, tmp_path):
    result, _ = _full_run(tmp_path, PROBE_RAISING_CODE)
    assert result["status"] == "error"
    assert "ENV=" in result["execution_stderr"]
    assert _leaked(result) == [], _leaked(result)


def test_full_run_positive_control_probe_reaches_unscrubbed_child(probes, tmp_path, monkeypatch):
    import app.agents.execution_agent as ea

    monkeypatch.setattr(ea, "scrubbed_env", _passthrough_env, raising=False)
    result, _ = _full_run(tmp_path, PROBE_CODE)
    assert result["status"] == "success", result
    assert _leaked(result) == sorted(PROBES)


# -------------------------------------------------------------------------
# The DTA sample preview: daft_execution.execute_code_on_local
# -------------------------------------------------------------------------

DAFT_PROBE_CODE = """
import os
print("ENV", dict(os.environ), flush=True)

def transform_data(df):
    for n in %(names)r:
        print("PROBE", n, os.environ.get(n), flush=True)
    return df
""" % {"names": sorted(PROBES)}


def _daft_run(tmp_path, monkeypatch):
    pytest.importorskip("daft")
    from app.agents.data_transfer_agent import daft_execution

    monkeypatch.chdir(tmp_path)
    return daft_execution.execute_code_on_local(DAFT_PROBE_CODE, pd.DataFrame({"a": [1, 2]}))


def test_dta_sample_child_sees_no_probe(probes, tmp_path, monkeypatch):
    result = _daft_run(tmp_path, monkeypatch)
    assert "PROBE DB_ENCRYPTION_KEY None" in result.get("execution_stdout", ""), result
    assert _leaked(result) == [], _leaked(result)


def test_dta_sample_positive_control(probes, tmp_path, monkeypatch):
    pytest.importorskip("daft")
    from app.agents.data_transfer_agent import daft_execution

    monkeypatch.setattr(daft_execution, "scrubbed_env", _passthrough_env, raising=False)
    result = _daft_run(tmp_path, monkeypatch)
    assert _leaked(result) == sorted(PROBES)


# -------------------------------------------------------------------------
# The trial run: validator.execute_code_node
# -------------------------------------------------------------------------

def _trial(code):
    return validator_mod.execute_code_node({
        "generated_code": code,
        "sample_data": SAMPLE_CSV,
        "syntax_error": False,
        "static_semantic_error": False,
    })


def test_trial_run_sees_no_probe(probes):
    result = _trial(PROBE_CODE)
    assert result["execution_error"] is None, result
    assert "PROBE DB_ENCRYPTION_KEY None None" in result["execution_stdout"]
    assert result["execution_output_data"], result
    assert _leaked(result) == [], _leaked(result)


def test_trial_run_error_carries_no_probe(probes):
    result = _trial(PROBE_RAISING_CODE)
    assert result["execution_error"].startswith("ENV="), result
    assert _leaked(result) == [], _leaked(result)


def test_trial_run_positive_control(probes, monkeypatch):
    monkeypatch.setattr(sample_mod, "scrubbed_env", _passthrough_env, raising=False)
    result = _trial(PROBE_CODE)
    assert result["execution_error"] is None, result
    assert _leaked(result) == sorted(PROBES)


class _ParentSentinel:
    def __init__(self, token):
        self.token = token


SENTINEL_TOKEN = "PARENT-SENTINEL-3c9d"

SENTINEL_CODE = """
import gc, os, sys
import pandas as pd

def main(df):
    mod = sys.modules.get("app.agents.validator")
    seen = getattr(getattr(mod, "_AVALOKA_PARENT_SENTINEL", None), "token", None)
    if mod is not None:
        mod._AVALOKA_PARENT_SENTINEL = "CHILD-OVERWROTE"
        mod.execute_code_node = None
    found = [o.token for o in gc.get_objects() if type(o).__name__ == "_ParentSentinel"]
    pd._avaloka_child_mark = True
    os.environ["AVALOKA_CHILD_WROTE"] = "1"
    return pd.DataFrame({"seen": [str(seen)], "found": [repr(found)],
                         "validator_loaded": [mod is not None]})
"""


def test_trial_run_cannot_see_or_touch_the_parent(monkeypatch):
    sentinel = _ParentSentinel(SENTINEL_TOKEN)
    monkeypatch.setattr(validator_mod, "_AVALOKA_PARENT_SENTINEL", sentinel, raising=False)
    monkeypatch.delenv("AVALOKA_CHILD_WROTE", raising=False)
    original_node = validator_mod.execute_code_node
    try:
        result = _trial(SENTINEL_CODE)
        assert result["execution_error"] is None, result
        # The parent is untouched ...
        assert validator_mod._AVALOKA_PARENT_SENTINEL is sentinel
        assert sentinel.token == SENTINEL_TOKEN
        assert validator_mod.execute_code_node is original_node
        assert not hasattr(pd, "_avaloka_child_mark")
        assert "AVALOKA_CHILD_WROTE" not in os.environ
        # ... and was invisible: no server module, no parent object.
        row = result["execution_output_data"][0]
        assert row["validator_loaded"] in (False, 0), row
        assert row["seen"] == "None" and row["found"] == "[]", row
        assert SENTINEL_TOKEN not in json.dumps(result, default=str)
    finally:
        # Undo what an in-process exec (the bug) would have done to this process.
        validator_mod.execute_code_node = original_node
        if hasattr(pd, "_avaloka_child_mark"):
            del pd._avaloka_child_mark


# -------------------------------------------------------------------------
# Limits
# -------------------------------------------------------------------------

# Bounded, so the in-process bug (no limit at all) returns after 20 s instead
# of hanging the suite.
BUSY_CODE = """
import time

def main(df):
    end = time.time() + 20
    while time.time() < end:
        pass
    return df
"""


def test_trial_run_busy_loop_is_killed_by_the_time_limit(monkeypatch):
    monkeypatch.setenv(sample_mod.TIMEOUT_ENV, "2")
    started = time.monotonic()
    result = _trial(BUSY_CODE)
    elapsed = time.monotonic() - started
    assert result["execution_error"] == "Sample execution timed out after 2 seconds", result
    assert result["execution_output_data"] is None
    assert elapsed < 15, elapsed


def test_trial_run_timeout_defaults_to_the_local_executor_timeout(monkeypatch):
    monkeypatch.delenv(sample_mod.TIMEOUT_ENV, raising=False)
    monkeypatch.setenv("AVALOKA_LOCAL_EXEC_TIMEOUT", "37")
    assert sample_mod.sample_timeout() == 37
    monkeypatch.delenv("AVALOKA_LOCAL_EXEC_TIMEOUT", raising=False)
    assert sample_mod.sample_timeout() == 60


# 1.5 GiB of address space, never touched: harmless without a limit (the
# positive control), refused under a 1024 MB RLIMIT_AS.
BOMB_CODE = """
import numpy as np

def main(df):
    block = np.empty(1536 * 1024 * 1024, dtype=np.uint8)
    df["n"] = len(block)
    return df
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="RLIMIT_AS is enforced only on Linux; macOS ignores or rejects it")
def test_trial_run_allocation_bomb_is_stopped_by_the_memory_limit(monkeypatch):
    monkeypatch.setenv(sample_mod.MEMORY_ENV, "1024")
    result = _trial(BOMB_CODE)
    assert result["execution_error"], result
    assert "allocate" in result["execution_error"] or "MemoryError" in result["execution_error"], result
    assert result["execution_output_data"] is None

    monkeypatch.setenv(sample_mod.MEMORY_ENV, "0")
    control = _trial(BOMB_CODE)
    assert control["execution_error"] is None, control


# -------------------------------------------------------------------------
# The result has the in-process shape and types
# -------------------------------------------------------------------------

FIDELITY_CODE = """
import pandas as pd

def main(df):
    out = pd.DataFrame({
        "a": df["a"].astype("int64"),
        "f": df["a"] * 1.5,
        "s": df["b"],
        "t": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "ok": df["a"] > 1,
    })
    out[2024] = df["a"]
    print("in main")
    return out
"""


def test_trial_run_result_matches_what_in_process_produced():
    result = _trial(FIDELITY_CODE)
    assert result["execution_error"] is None, result
    scope = {"pd": pd, "__name__": "avaloka_sample_execution"}
    exec(FIDELITY_CODE, scope)  # the trusted reference: this test's own code
    expected = scope["main"](pd.read_csv(__import__("io").StringIO(SAMPLE_CSV)))
    assert result["execution_output_data"] == expected.to_dict(orient="records")
    assert result["execution_output_preview"] == expected.head(2).to_dict(orient="records")
    assert isinstance(result["execution_output_data"][0]["t"], pd.Timestamp)
    assert 2024 in result["execution_output_data"][0]
    assert result["execution_stdout"] == "in main\n"


# -------------------------------------------------------------------------
# Self-provisioned secrets (app/core/provisioning.py, #649 era)
# -------------------------------------------------------------------------

@pytest.fixture
def provisioned(monkeypatch, tmp_path):
    """DB_ENCRYPTION_KEY, MCP_SERVICE_TOKEN, AVALOKA_PII_KEY as provisioning leaves them.

    Emptied first (so provision() generates them, and monkeypatch restores
    the originals afterwards), then generated into os.environ and saved to an
    env file, the way a service start does it.
    """
    from app.core.provisioning import GENERATED_SECRETS, provision

    for name in GENERATED_SECRETS:
        monkeypatch.setenv(name, "")
    env_file = tmp_path / "provisioned.env"
    done = provision(env_file, GENERATED_SECRETS)
    assert set(done.values()) <= {"generated", "replaced-published"}, done
    values = {name: os.environ[name] for name in GENERATED_SECRETS}
    assert all(len(v) == 64 for v in values.values())
    return values, env_file


def _leaked_values(obj, values) -> list[str]:
    text = json.dumps(obj, default=str)
    return sorted(name for name, value in values.items() if value in text)


def test_provisioned_secrets_reach_no_path(provisioned, tmp_path, monkeypatch):
    values, _ = provisioned
    work = tmp_path / "full"
    work.mkdir()
    full, written = _full_run(work, PROBE_CODE)
    assert full["status"] == "success", full
    trial = _trial(PROBE_CODE)
    assert trial["execution_error"] is None, trial
    assert _leaked_values([full, written, trial], values) == []


def test_provisioned_secrets_reach_no_dta_path(provisioned, tmp_path, monkeypatch):
    values, _ = provisioned
    result = _daft_run(tmp_path, monkeypatch)
    assert _leaked_values(result, values) == []


def test_provisioned_secrets_positive_control(provisioned, monkeypatch):
    values, _ = provisioned
    monkeypatch.setattr(sample_mod, "scrubbed_env", _passthrough_env, raising=False)
    result = _trial(PROBE_CODE)
    assert _leaked_values(result, values) == sorted(values)


READ_DOTENV_CODE = """
def main(df):
    try:
        print(open(".env").read())
    except OSError as exc:
        print("NO .env:", exc)
    return df
"""


@pytest.mark.xfail(strict=True, reason=(
    "Open, out of this hotfix's scope: the full run keeps the server's working "
    "directory, where provisioning saves the generated secrets to ./.env, and "
    "nothing isolates the filesystem. Follow-up: run generated code in its own "
    "cwd/mount namespace (2.0's executor service) or keep .env outside the "
    "server's cwd."))
def test_full_run_cannot_read_the_provisioned_dotenv(monkeypatch, tmp_path):
    from app.core.provisioning import GENERATED_SECRETS, provision

    server_cwd = tmp_path / "server"
    server_cwd.mkdir()
    monkeypatch.chdir(server_cwd)
    for name in GENERATED_SECRETS:
        monkeypatch.setenv(name, "")
    provision(server_cwd / ".env", GENERATED_SECRETS)
    values = {name: os.environ[name] for name in GENERATED_SECRETS}
    result, _ = _full_run(tmp_path, READ_DOTENV_CODE)
    assert result["status"] == "success", result
    assert _leaked_values(result, values) == []
