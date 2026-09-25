"""Measurement harness for the chat-surface coder/validator pipeline.

Purpose: produce a NUMBER. Success rate and validator-iteration count over a
corpus of complex data-manipulation prompts, so a reliability claim can be
checked instead of asserted.

What it does per task:
  1. builds the same CodingAgentState that app/api/workflow.py:coding_subgraph_node
     builds for the chat surface,
  2. runs a mirror of build_coding_graph() whose five nodes are wrapped in
     recorders (production code is not modified),
  3. executes the resulting code against the FULL dataset -- not the two-row
     validation sample -- and
  4. scores the output against the independent ground truth in tasks.py.

Run it:
    python -m tests.blueprint_contract.harness --out before.json
    python -m tests.blueprint_contract.harness --out before.json --tasks hc_los_by_condition,taxi_hourly_tip_rate
    python -m tests.blueprint_contract.harness --out before.json --repeat 3

Requires a live coder/validator LLM (GROQ_API_KEY_CODING_AGENT or GROQ_API_KEY).
Without one the coder falls back to deterministic stub code and the numbers
measure the stub, not the pipeline -- the harness refuses to run in that state
unless --allow-stub is passed.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.blueprint_contract.tasks import TASKS, TASKS_BY_ID, ComplexTask  # noqa: E402

LOGGER = logging.getLogger("blueprint_harness")

# Rows of the dataset handed to the pipeline as the uploaded preview. The chat
# surface sends a preview, not the whole file; 100 rows matches app/api.
PREVIEW_ROWS = 100


# ---------------------------------------------------------------------------
# per-node recording
# ---------------------------------------------------------------------------

@dataclass
class NodeCall:
    node: str
    seq: int
    duration_s: float
    # Only the decision-relevant keys; full frames are never recorded.
    out: Dict[str, Any] = field(default_factory=dict)


_INTERESTING = (
    "syntax_error",
    "static_semantic_error",
    "logical_semantic_error",
    "contract_error",
    "execution_error",
    "retry_count",
    "coder_blocked_reason",
    "stub_fallback_used",
    "code_validation_feedback",
)


def _summarise_node_output(out: Any) -> Dict[str, Any]:
    if not isinstance(out, dict):
        return {}
    d: Dict[str, Any] = {}
    for k in _INTERESTING:
        if k in out and out[k] not in (None, False):
            v = out[k]
            d[k] = v if not isinstance(v, str) else v[:400]
    if out.get("coder_pseudocode"):
        d["blueprint_chars"] = len(out["coder_pseudocode"])
    if out.get("generated_code"):
        d["code_chars"] = len(out["generated_code"])
    return d


def build_instrumented_coding_graph(record: List[NodeCall]):
    """Mirror of app/api/workflow.py::build_coding_graph with recording wrappers.

    Kept structurally identical to the production graph -- same nodes, same
    edges, same router -- so the numbers describe production behaviour.
    """
    from langgraph.graph import END, StateGraph

    from app.agents.coder import coder_node
    from app.agents.state import CodingAgentState
    from app.agents.validator import (
        execute_code_node,
        logical_semantic_validator_node,
        static_semantic_validator_node,
        syntactic_validator_node,
    )
    try:
        from app.agents.validator import contract_validator_node
    except ImportError:  # pre-contract revisions, so before/after stay comparable
        contract_validator_node = None
    from app.api.workflow import check_validation_status

    def wrap(name: str, fn: Callable):
        def inner(state):
            t0 = time.time()
            try:
                out = fn(state)
            except Exception:
                record.append(NodeCall(name, len(record), time.time() - t0,
                                       {"raised": traceback.format_exc()[-600:]}))
                raise
            record.append(NodeCall(name, len(record), time.time() - t0,
                                   _summarise_node_output(out)))
            return out
        return inner

    wf = StateGraph(CodingAgentState)
    wf.add_node("coder", wrap("coder", coder_node))
    wf.add_node("validator_syntax", wrap("validator_syntax", syntactic_validator_node))
    wf.add_node("validator_static", wrap("validator_static", static_semantic_validator_node))
    wf.add_node("execute_code", wrap("execute_code", execute_code_node))
    if contract_validator_node is not None:
        wf.add_node("validator_contract", wrap("validator_contract", contract_validator_node))
    wf.add_node("validator_logical", wrap("validator_logical", logical_semantic_validator_node))

    wf.set_entry_point("coder")
    wf.add_edge("coder", "validator_syntax")
    wf.add_edge("validator_syntax", "validator_static")
    wf.add_edge("validator_static", "execute_code")
    if contract_validator_node is not None:
        wf.add_edge("execute_code", "validator_contract")
        wf.add_edge("validator_contract", "validator_logical")
    else:
        wf.add_edge("execute_code", "validator_logical")
    wf.add_conditional_edges(
        "validator_logical", check_validation_status,
        {"refine": "coder", "end": END},
    )
    return wf.compile()


# ---------------------------------------------------------------------------
# state construction (mirrors coding_subgraph_node)
# ---------------------------------------------------------------------------

def _schema_of(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c): str(t) for c, t in df.dtypes.items()}


def _preview_records(df: pd.DataFrame, n: int = PREVIEW_ROWS) -> List[Dict[str, Any]]:
    head = df.head(n)
    return json.loads(head.to_json(orient="records", date_format="iso"))


def _default_plan(task: ComplexTask, data_path: Path, out_path: Path) -> str:
    """The plan the planner would hand over. Using the planner's own fallback
    shape (app/api/workflow.py:576-582) keeps the coder the variable under test
    rather than mixing in planner variance."""
    return (
        f"1. Load the data from {data_path} into a pandas DataFrame.\n"
        f'2. Fulfill this request: "{task.prompt}" using pandas operations.\n'
        f"3. Save the final DataFrame to {out_path}."
    )


def build_state(task: ComplexTask, df: pd.DataFrame, out_path: Path,
                plan: Optional[str] = None) -> Dict[str, Any]:
    datasets_context = [{
        "dataset_id": "primary",
        "alias": task.dataset.stem,
        "filename": task.dataset.name,
        "columns": list(df.columns),
        "data_source_location": str(task.dataset),
        "full_data_location": str(task.dataset),
        "input_data_type": "csv",
    }]
    for alias, p in task.extra_datasets.items():
        extra = pd.read_csv(p, nrows=5)
        datasets_context.append({
            "dataset_id": alias,
            "alias": alias,
            "filename": p.name,
            "columns": list(extra.columns),
            "data_source_location": str(p),
            "full_data_location": str(p),
            "input_data_type": "csv",
        })

    return {
        "user_prompt": task.prompt,
        "plan": plan or _default_plan(task, task.dataset, out_path),
        "data_source_location": str(task.dataset),
        "output_location": str(out_path),
        "schema": _schema_of(df),
        "input_data_type": "csv",
        "sample_data": df.head(PREVIEW_ROWS).to_csv(index=False),
        "requirements": None,
        "messages": [],
        "retry_count": 0,
        "coder_pseudocode": None,
        "uploaded_csv_preview": _preview_records(df),
        "uploaded_csv_columns": list(df.columns),
        "analysis_fidelity": "quick_sample",
        "selected_sample_name": "random_baseline",
        "multi_dataset_state": datasets_context if task.extra_datasets else [],
        "datasets_context": datasets_context,
        "memory_hints": None,
        "session_logic_signature": None,
    }


# ---------------------------------------------------------------------------
# running generated code for real
# ---------------------------------------------------------------------------

def run_generated_code(code: str, df: pd.DataFrame, timeout_note: str = "") -> Dict[str, Any]:
    """Execute main(df) against the FULL dataset.

    The validator only ever runs generated code against two rows; the whole
    point of the measurement is to find out what the code does on real data.
    """
    if not code or not code.strip():
        return {"ok": False, "error": "no code generated", "result": None}

    scope: Dict[str, Any] = {"__name__": "avaloka_harness_execution"}
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, scope)  # noqa: S102 -- this is the thing under measurement
            main = scope.get("main")
            if not callable(main):
                return {"ok": False, "error": "no main() in generated code", "result": None}
            result = main(df.copy())
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
            "result": None,
            "stdout": stdout.getvalue()[-500:],
        }

    if isinstance(result, pd.Series):
        result = result.to_frame()
    if not isinstance(result, pd.DataFrame):
        return {
            "ok": False,
            "error": f"main() returned {type(result).__name__}, not a DataFrame",
            "result": None,
        }
    return {"ok": True, "error": None, "result": result, "stdout": stdout.getvalue()[-500:]}


# ---------------------------------------------------------------------------
# one task
# ---------------------------------------------------------------------------

@dataclass
class TaskOutcome:
    task_id: str
    family: str
    # pipeline behaviour
    coder_invocations: int = 0
    final_retry_count: int = 0
    stub_fallback_used: bool = False
    blocked_reason: Optional[str] = None
    validation_refused: bool = False
    failure_stage: Optional[str] = None
    # what came out
    blueprint: Optional[str] = None
    blueprint_chars: int = 0
    had_contract: bool = False
    contract: Optional[Dict[str, Any]] = None
    contract_violations: List[Any] = field(default_factory=list)
    code: Optional[str] = None
    # ground truth on the full dataset
    ran_on_full_data: bool = False
    full_data_error: Optional[str] = None
    checks_passed: int = 0
    checks_total: int = 0
    failed_checks: List[str] = field(default_factory=list)
    success: bool = False
    duration_s: float = 0.0
    node_trace: List[Dict[str, Any]] = field(default_factory=list)
    graph_error: Optional[str] = None


def run_task(task: ComplexTask, out_dir: Path, keep_code: bool = True) -> TaskOutcome:
    t0 = time.time()
    outcome = TaskOutcome(task_id=task.id, family=task.family)
    record: List[NodeCall] = []

    df = task.load()
    out_path = out_dir / f"{task.id}.csv"
    state = build_state(task, df, out_path)

    graph = build_instrumented_coding_graph(record)
    try:
        final = graph.invoke(state, config={"recursion_limit": 50})
    except Exception as exc:
        outcome.graph_error = f"{type(exc).__name__}: {exc}"
        final = {}

    outcome.node_trace = [asdict(c) for c in record]
    outcome.coder_invocations = sum(1 for c in record if c.node == "coder")
    outcome.final_retry_count = int(final.get("retry_count") or 0)
    outcome.stub_fallback_used = bool(final.get("stub_fallback_used"))
    outcome.blocked_reason = final.get("coder_blocked_reason")
    outcome.blueprint = final.get("coder_pseudocode") or None
    outcome.blueprint_chars = len(outcome.blueprint or "")

    code = final.get("generated_code") or ""
    outcome.code = code if keep_code else None

    # Mirror workflow.py:629 -- the chat surface refuses to execute when any
    # validation flag survived the retries.
    outcome.contract_violations = final.get("contract_violations") or []
    outcome.had_contract = bool(final.get("coder_contract"))
    outcome.contract = final.get("coder_contract")
    outcome.validation_refused = bool(
        final.get("syntax_error")
        or final.get("static_semantic_error")
        or final.get("logical_semantic_error")
        or final.get("contract_error")
    )
    if final.get("syntax_error"):
        outcome.failure_stage = "syntax"
    elif final.get("static_semantic_error"):
        outcome.failure_stage = "static_semantic"
    elif final.get("logical_semantic_error"):
        outcome.failure_stage = "logical"
    elif final.get("contract_error"):
        outcome.failure_stage = "contract"

    if outcome.graph_error:
        outcome.failure_stage = outcome.failure_stage or "graph_raised"
    elif outcome.blocked_reason and not outcome.failure_stage:
        # A coder guard refused the request outright (coder.py:_detect_math_on_
        # string_column_coder and friends). That is a distinct failure mode from
        # a validation rejection and must not be reported as "no stage".
        outcome.failure_stage = "coder_guard_blocked"

    if outcome.validation_refused or outcome.blocked_reason or outcome.graph_error:
        outcome.duration_s = time.time() - t0
        return outcome

    run = run_generated_code(code, df)
    outcome.ran_on_full_data = run["ok"]
    outcome.full_data_error = run["error"]
    if not run["ok"]:
        outcome.failure_stage = outcome.failure_stage or "full_data_execution"
        outcome.duration_s = time.time() - t0
        return outcome

    try:
        checks = task.check(run["result"], df)
    except Exception as exc:
        outcome.failure_stage = "acceptance_check_raised"
        outcome.full_data_error = f"check raised: {type(exc).__name__}: {exc}"
        outcome.duration_s = time.time() - t0
        return outcome

    outcome.checks_total = len(checks)
    outcome.checks_passed = sum(1 for _, ok, _ in checks if ok)
    outcome.failed_checks = [f"{name}: {detail}" for name, ok, detail in checks if not ok]
    outcome.success = outcome.checks_total > 0 and not outcome.failed_checks
    if not outcome.success:
        outcome.failure_stage = outcome.failure_stage or "wrong_result"
    outcome.duration_s = time.time() - t0
    return outcome


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def summarise(outcomes: List[TaskOutcome]) -> Dict[str, Any]:
    n = len(outcomes)
    if not n:
        return {"tasks": 0}
    succ = [o for o in outcomes if o.success]
    stages: Dict[str, int] = {}
    for o in outcomes:
        if not o.success:
            stages[o.failure_stage or "unknown"] = stages.get(o.failure_stage or "unknown", 0) + 1
    fam: Dict[str, Dict[str, int]] = {}
    for o in outcomes:
        d = fam.setdefault(o.family, {"n": 0, "ok": 0})
        d["n"] += 1
        d["ok"] += int(o.success)
    return {
        "tasks": n,
        "success": len(succ),
        "success_rate": round(len(succ) / n, 4),
        "mean_coder_invocations": round(sum(o.coder_invocations for o in outcomes) / n, 3),
        "mean_retry_count": round(sum(o.final_retry_count for o in outcomes) / n, 3),
        "tasks_needing_retry": sum(1 for o in outcomes if o.coder_invocations > 1),
        "validation_refused": sum(1 for o in outcomes if o.validation_refused),
        "coder_guard_blocked": sum(1 for o in outcomes if o.blocked_reason),
        "tasks_with_contract": sum(1 for o in outcomes if o.had_contract),
        "contract_violations_raised": sum(1 for o in outcomes if o.contract_violations),
        "stub_fallback_used": sum(1 for o in outcomes if o.stub_fallback_used),
        "failed_on_full_data": sum(
            1 for o in outcomes if not o.validation_refused and not o.ran_on_full_data
        ),
        "ran_but_wrong": sum(
            1 for o in outcomes if o.ran_on_full_data and not o.success
        ),
        "mean_check_pass_rate": round(
            sum((o.checks_passed / o.checks_total) if o.checks_total else 0.0 for o in outcomes) / n, 4
        ),
        "failure_stages": stages,
        "by_family": fam,
        "mean_duration_s": round(sum(o.duration_s for o in outcomes) / n, 2),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("blueprint_harness_result.json"))
    ap.add_argument("--tasks", type=str, default="", help="comma-separated task ids")
    ap.add_argument("--family", type=str, default="", help="only this family")
    ap.add_argument("--repeat", type=int, default=1, help="runs per task (LLMs are stochastic)")
    ap.add_argument("--allow-stub", action="store_true",
                    help="run even when no coder LLM is configured")
    ap.add_argument("--label", type=str, default="", help="label recorded in the result file")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )

    from app.agents import coder as coder_mod
    if coder_mod.coder_llm is None and not args.allow_stub:
        print(
            "REFUSING TO RUN: no coder LLM is configured, so every task would "
            "measure the deterministic stub rather than the pipeline.\n"
            "Set GROQ_API_KEY_CODING_AGENT (or GROQ_API_KEY), or pass --allow-stub "
            "if measuring the stub path is what you intend.",
            file=sys.stderr,
        )
        return 2

    selected = TASKS
    if args.tasks:
        ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        missing = [i for i in ids if i not in TASKS_BY_ID]
        if missing:
            print(f"unknown task ids: {missing}", file=sys.stderr)
            return 2
        selected = [TASKS_BY_ID[i] for i in ids]
    if args.family:
        selected = [t for t in selected if t.family == args.family]

    out_dir = args.out.parent / (args.out.stem + "_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    outcomes: List[TaskOutcome] = []
    for rep in range(args.repeat):
        for task in selected:
            print(f"[{rep + 1}/{args.repeat}] {task.id} ({task.family}) ...",
                  file=sys.stderr, flush=True)
            o = run_task(task, out_dir)
            outcomes.append(o)
            flag = "OK " if o.success else "FAIL"
            print(
                f"    {flag} coder_calls={o.coder_invocations} "
                f"checks={o.checks_passed}/{o.checks_total} "
                f"stage={o.failure_stage or '-'} {o.duration_s:.1f}s",
                file=sys.stderr, flush=True,
            )

    payload = {
        "label": args.label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repeat": args.repeat,
        "summary": summarise(outcomes),
        "outcomes": [asdict(o) for o in outcomes],
    }
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload["summary"], indent=2))
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
