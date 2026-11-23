"""Model-written code never inherits a secret (app/executor/env.py).

The live probe this answers: generated code run through execute_code_on_local
listed DB_ENCRYPTION_KEY, AVALOKA_LOCAL_AUTH_SECRET, AVALOKA_LOCAL_AUTH_PASSWORD,
GROQ_API_KEY, OPENROUTER_API_KEY and REDIS_URL in its os.environ.
"""

import os
import sys

import pytest

from app.executor.env import ALLOWED_ENV_NAMES, FORBIDDEN_NAME_PARTS, scrubbed_env
from app.core.thirdparty_optout import THIRDPARTY_OPTOUT_ENV

# Every secret the sandboxnet probe found, plus the rest of the deployment's.
SECRETS = {
    "DB_ENCRYPTION_KEY": "k", "AVALOKA_LOCAL_AUTH_SECRET": "s", "AVALOKA_LOCAL_AUTH_PASSWORD": "p",
    "GROQ_API_KEY": "g", "OPENROUTER_API_KEY": "o", "OPENAI_API_KEY": "o", "ANTHROPIC_API_KEY": "a",
    "REDIS_URL": "redis://redis:6379/0", "CELERY_REDIS_URL": "redis://redis:6379/0",
    "SUPABASE_JWT_SECRET": "j", "AWS_SECRET_ACCESS_KEY": "a", "AWS_ACCESS_KEY_ID": "a",
    "S3_ENDPOINT_URL": "http://seaweedfs:8333", "DATABASE_URL": "postgresql://u:p@h/d",
    "QS_POSTGRES_PASSWORD": "p", "MCP_SERVER_URL": "http://mcp:8010", "MCP_ADMIN_TOKEN": "t",
    "LANGGRAPH_API_URL": "http://langgraph:2024", "DTA_SOURCE_CONN_STR": "postgresql://u:p@h/d",
    "DTA_DEST_CONN_STR": "postgresql://u:p@h/d", "GOOGLE_APPLICATION_CREDENTIALS": "/x.json",
    "HF_TOKEN": "h", "DATA_DIR": "/state/data",
}


def test_no_secret_survives():
    env = scrubbed_env({**SECRETS, "AVALOKA_EXECUTOR_URL": "http://executor:8020",
                        "PATH": "/usr/bin", "HOME": "/root"})
    assert not set(SECRETS) & set(env), sorted(set(SECRETS) & set(env))
    assert "AVALOKA_EXECUTOR_URL" not in env
    assert env["PATH"] == "/usr/bin" and env["HOME"] == "/root"


def test_allowlist_itself_names_no_secret():
    bad = [n for n in ALLOWED_ENV_NAMES if any(p in n.upper() for p in FORBIDDEN_NAME_PARTS)]
    assert not bad, f"the allowlist names a secret-shaped variable: {bad}"


def test_extra_cannot_smuggle_a_secret():
    env = scrubbed_env({}, extra={"DB_ENCRYPTION_KEY": "k", "TMPDIR": "/w"})
    assert "DB_ENCRYPTION_KEY" not in env and env["TMPDIR"] == "/w"


def test_optouts_default_on_and_keep_an_explicit_value():
    env = scrubbed_env({"DO_NOT_TRACK": "false"})
    for k, v in THIRDPARTY_OPTOUT_ENV.items():
        assert k in env
    assert env["DO_NOT_TRACK"] == "false"


def test_local_child_sees_only_the_allowlist(monkeypatch):
    from app.executor.client import run_script
    monkeypatch.delenv("AVALOKA_EXECUTOR_URL", raising=False)
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    ran = run_script("import os, json; print(json.dumps(sorted(os.environ)))", timeout=60)
    assert ran.returncode == 0, ran.stderr
    import json
    seen = set(json.loads(ran.stdout))
    assert not seen & set(SECRETS), sorted(seen & set(SECRETS))
    assert seen <= ALLOWED_ENV_NAMES | {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}, sorted(seen - ALLOWED_ENV_NAMES)


def test_execute_code_on_local_child_has_no_secret(monkeypatch, tmp_path):
    """The real entry point the coder's code goes through (execution_agent.py)."""
    from app.agents.execution_agent import execute_code_on_local
    monkeypatch.delenv("AVALOKA_EXECUTOR_URL", raising=False)
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    src = tmp_path / "in.csv"
    src.write_text("id\n1\n2\n")
    code = (
        "import os\n"
        "def main(df):\n"
        "    print('LEAK', sorted(k for k in os.environ if k in %r), flush=True)\n"
        "    return df\n" % (sorted(SECRETS),)
    )
    out = execute_code_on_local(code=code, input_data_location=str(src),
                                output_location=str(tmp_path / "out.csv"),
                                read_path_map={"t": str(src)})
    assert out["status"] == "success", out
    assert "LEAK []" in out["execution_stdout"]
    assert (tmp_path / "out.csv").exists()
