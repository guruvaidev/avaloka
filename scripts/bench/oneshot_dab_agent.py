#!/usr/bin/env python3
"""Model-alone baseline for official DataAgentBench, speaking dab-command-v2.

Same request and response contract as ``scripts/avaloka_dab_agent.py``, so the
same harness, the same official per-query validators, the same spend ledger and
the same resumable journal grade it. What is missing is everything the Avaloka
harness adds: there is no plan/act/inspect loop, no tool calls, no schema
inspection, no specialist routing, no answer review and no retry on a wrong
result. The model sees the question and the database description once, writes
ONE Python program, the program runs once with the same database connections
the Avaloka tools use, and whatever it prints after the marker is the answer.

Routing, pricing and database provisioning are imported from the Avaloka DAB
command so the two systems under test differ in nothing but the harness.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
for entry in (str(REPO_ROOT), str(HERE.parents[1])):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from avaloka.data_agent import MissionLimits, MissionPolicy, ModelUsage  # noqa: E402
from avaloka_dab_agent import (OpenAICompatibleProvider, _ensure_databases,  # noqa: E402
                               _json_default, _pricing)
from app.executor.env import scrubbed_env  # noqa: E402

SUT_NAME = "oneshot.single_program"
PROMPT_VERSION = "oneshot-dab-v1"
RESULT_MARKER = "__RESULT__:"

SYSTEM_PROMPT = """You are a data analyst answering one question over the databases described below.
You get exactly one attempt: write ONE complete Python 3 program that connects to the
databases, computes the answer, and prints it. The program is executed once, as written,
with no feedback and no second attempt, so inspect schemas defensively inside the program
if you are unsure of names (e.g. query information_schema / sqlite_master / list collections)
and still print a final answer.

Available libraries: pandas, pyarrow, numpy, sqlite3, duckdb, psycopg2, sqlalchemy, pymongo.
No internet access. Do not install packages.

Output contract (mandatory): the program must end by printing the marker line and then the
answer on the next line, exactly:
    print("__RESULT__:")
    print(<a string that json.loads() can parse>)
Use json.dumps(...) on plain values; convert DataFrames with .to_json(orient="records").
The text after the marker is graded as your answer. Reply with the program in a single
```python fenced block and nothing else."""


def _connection_facts(config_path: Path) -> str:
    import yaml  # present wherever DAB's own tools run
    clients = yaml.safe_load(config_path.read_text(encoding="utf-8"))["db_clients"]
    base = config_path.parent
    pg = {
        "host": os.environ.get("PG_HOST", "127.0.0.1"),
        "port": int(os.environ.get("PG_PORT", 5432)),
        "user": os.environ.get("PG_USER", "postgres"),
        "password": os.environ.get("PG_PASSWORD", ""),
    }
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    lines: List[str] = ["CONNECTION FACTS (the benchmark environment's interface):"]
    for name, client in clients.items():
        kind = str(client.get("db_type"))
        if kind in ("sqlite", "duckdb"):
            path = (base / client["db_path"]).resolve()
            lines.append(f"- {name}: {kind} file at {path}")
        elif kind == "postgres":
            lines.append(
                f"- {name}: PostgreSQL database '{client['db_name']}' at host={pg['host']} "
                f"port={pg['port']} user={pg['user']} password={pg['password']!r} "
                f"(psycopg2.connect(dbname=..., host=..., port=..., user=..., password=...))"
            )
        elif kind == "mongo":
            lines.append(f"- {name}: MongoDB database '{client['db_name']}' at {mongo_uri}")
        else:
            lines.append(f"- {name}: {kind} (see description)")
    lines.append("Each SQL database is separate; combine sources in pandas. In PostgreSQL, "
                 "double-quote mixed-case identifiers and use single quotes for string literals.")
    return "\n".join(lines)


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def _extract_program(content: str) -> str:
    blocks = _FENCE.findall(content or "")
    if blocks:
        return max(blocks, key=len)
    return content or ""


def _answer_from_stdout(stdout: str) -> str | None:
    if RESULT_MARKER in stdout:
        tail = stdout.rsplit(RESULT_MARKER, 1)[1].strip()
        return tail if tail else None
    text = stdout.strip()
    return text[-4000:] if text else None


def program_env(run_dir: Path) -> Dict[str, str]:
    """The model-written program's environment: the executor's allowlist, HOME = run dir.

    It ran with ``{**os.environ}``, provider keys included. The program needs
    no variable for the databases: the connection facts are in its prompt, and
    PATH (kept) carries the dab-bin bridge.
    """
    return scrubbed_env(extra={"HOME": str(run_dir), "PYTHONDONTWRITEBYTECODE": "1"})


def main() -> int:
    request = json.load(sys.stdin)
    if request.get("protocol_version") != "dab-command-v2":
        raise ValueError("unsupported DAB command protocol; expected dab-command-v2")

    dab_root = Path(os.environ.get("DAB_ROOT", "")).expanduser().resolve()
    if not (dab_root / "common_scaffold" / "tools").is_dir():
        raise RuntimeError("DAB_ROOT does not contain the sanctioned DAB tools")
    sys.path.insert(0, str(dab_root))
    bridge_bin = HERE.parents[1] / "dab-bin"
    os.environ["PATH"] = str(bridge_bin) + os.pathsep + os.environ.get("PATH", "")

    run = request["run"]
    mission = request["mission"]
    source = request["sources"]
    model = str(request["model"]["provider_model"])
    raw_budget = request.get("budget") or {}
    raw_policy = request.get("policy") or {}
    policy = MissionPolicy(
        max_iterations=1,
        tool_timeout_s=int(raw_policy.get("tool_timeout_s", 120)),
        max_output_tokens=int(raw_policy.get("max_output_tokens", 8_192)),
        max_prompt_tool_result_bytes=int(raw_policy.get("max_prompt_tool_result_bytes", 100_000)),
    )
    limits = MissionLimits(
        max_cost_usd=raw_budget.get("max_cost_usd"),
        max_latency_s=raw_budget.get("max_latency_s"),
        max_tokens=raw_budget.get("max_tokens"),
        max_data_scanned_ratio=raw_budget.get("max_data_scanned_ratio"),
    )
    pricing = _pricing()
    if limits.max_cost_usd is not None and not pricing.configured:
        raise RuntimeError("a max_cost_usd budget requires configured AVALOKA_DAB input/output pricing")

    config_path = Path(source["database_config_path"]).resolve()
    working_directory = Path(source["working_directory"]).resolve()
    run_name = (f"oneshot_{run['dataset']}_q{run['query']}_r{run['trial']}_"
                f"{time.time_ns()}_{os.getpid()}")
    run_dir = working_directory / "logs" / "oneshot" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    _ensure_databases(config_path)

    provider = OpenAICompatibleProvider(model=model, log_path=run_dir / "model_calls.jsonl",
                                        max_output_tokens=policy.max_output_tokens)
    question = mission["question"]
    question_text = question if isinstance(question, str) else json.dumps(question, default=str)
    user_message = (
        f"QUESTION:\n{question_text}\n\nDATABASE DESCRIPTION:\n{mission['database_description']}"
        f"\n\n{_connection_facts(config_path)}"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}]

    started = time.perf_counter()
    trajectory: List[Dict[str, Any]] = []
    error = None
    error_kind = None
    answer = None
    usage = ModelUsage(0, 0, 0, 0)
    program = ""
    stdout = stderr = ""
    try:
        # One call. Transport retries (3, as in the Avaloka command) are the
        # provider adapter's, not the model's: a dropped connection is not a
        # second attempt at the question.
        response = provider.complete(messages, tools=[], timeout_s=float(policy.tool_timeout_s) * 3)
        usage = response.usage
        program = _extract_program(response.content or "")
        trajectory.append({"kind": "model_call", "prompt_tokens": usage.prompt_tokens,
                           "completion_tokens": usage.completion_tokens,
                           "program_chars": len(program)})
        if not program.strip():
            error, error_kind = "answer_error: the model returned no program", "answer"
    except Exception as exc:  # noqa: BLE001
        error, error_kind = f"provider_error: {type(exc).__name__}: {exc}", "provider"

    if program.strip() and error is None:
        program_path = run_dir / "program.py"
        program_path.write_text(program, encoding="utf-8")
        env = program_env(run_dir)
        exec_started = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-I", str(program_path)], cwd=run_dir,
                                  capture_output=True, text=True,
                                  timeout=int(raw_policy.get("python_timeout_s", policy.tool_timeout_s)),
                                  check=False, env=env)
            stdout, stderr = proc.stdout, proc.stderr
            answer = _answer_from_stdout(stdout)
            trajectory.append({"kind": "program_run", "returncode": proc.returncode,
                               "duration_s": time.perf_counter() - exec_started,
                               "stdout_chars": len(stdout), "marker_found": RESULT_MARKER in stdout})
            if answer is None:
                error = ("execution_error: the program printed no answer"
                         + (f"; exit {proc.returncode}: {stderr[-800:]}" if proc.returncode else ""))
                error_kind = "execution"
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            error, error_kind = f"timeout: program exceeded {exc.timeout}s", "timeout"
            trajectory.append({"kind": "program_run", "timeout": True})

    cost = pricing.cost(usage)
    response_payload = {
        "answer": answer,
        "latency_s": time.perf_counter() - started,
        "cost_usd": cost,
        "tokens": usage.total_tokens,
        "tool_calls": 0,
        "retries": provider.retries,
        "trajectory": trajectory,
        "error": error,
        "error_kind": error_kind,
        "artifacts": {
            "sut_name": SUT_NAME,
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "run_dir": str(run_dir),
            "program": str(run_dir / "program.py") if program.strip() else None,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
            "termination_reason": error_kind or "answered",
            "iterations": 1,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "pricing_version": pricing.version,
            "pricing_configured": pricing.configured,
            "effective_policy": asdict(policy),
            "effective_limits": asdict(limits),
        },
    }
    json.dump(response_payload, sys.stdout, default=_json_default)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"oneshot_dab_agent: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
