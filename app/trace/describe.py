"""What a step decided and calculated, read from what it wrote.

Each describer sees the state a node received and the keys it changed, and
returns plain, JSON-ready facts: the decision and its stated reason, the code
or operation, the data scope, and result values when they are small. Nothing
here re-runs anything or calls a model; it only reads.

Decisions are described by their outputs and the reasons the agents
themselves recorded (an intent classifier's ``reasoning``, a validator's
feedback, a claim verifier's findings). Prompts are never included.

This is the user's own trace, so it may show their column names, formulas and
values. None of it is telemetry; see :mod:`app.trace.telemetry` for that.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Dict, List, Optional, Tuple

#: node -> (agent, role, swarm clone that narrates it)
AGENT_CATALOG: Dict[str, Tuple[str, str, Optional[str]]] = {
    "drivers_route": ("Drivers", "Answers \"what predicts X\" from the shared driver ranking", None),
    "memory_injection": ("Memory", "Recalls what this thread already established", None),
    "avaloka_agent": ("Avaloka", "Lead: reads the intent and decides who takes it", None),
    "plan_etl": ("Planner", "Plans the analysis and chooses the route", "scout"),
    "summarize_etl": ("Summarizer", "Turns the plan into a structured definition", None),
    "generate_planner_graph": ("Plan drafter", "Draws the plan as a graph", None),
    "code_etl": ("Coding team", "Writes the code, validates it and dry-runs it", "coder"),
    "coder": ("Coder", "Writes the analysis code", "coder"),
    "validator_syntax": ("Syntax check", "Confirms the code parses", "skeptic"),
    "validator_static": ("Static check", "AST checks for code that runs but answers wrongly", "skeptic"),
    "execute_code": ("Dry run", "Runs the code on the sample before it is trusted", "runner"),
    "validator_contract": ("Contract check", "Checks the result has the shape the question asked for", "skeptic"),
    "validator_logical": ("Logical review", "Judges whether the code answers the question", "skeptic"),
    "execute_actions": ("Action runner", "Runs structured actions the planner emitted", "runner"),
    "provision_infra": ("Infrastructure", "Stands up the compute the job needs", "builder"),
    "schedule_task": ("Scheduler", "Puts the job on a schedule", "scheduler"),
    "execute_on_ray": ("Executor (Ray)", "Runs the validated code on the Ray cluster", "runner"),
    "execute_locally": ("Executor (local)", "Runs the validated code locally", "runner"),
    "execute_on_k8s": ("Executor (Kubernetes)", "Runs the validated code as a Kubernetes job", "runner"),
    "visualize": ("Visualizer", "Chooses and configures the chart", "voice"),
    "narrate_result": ("Narrator", "Writes the answer from the computed result", "voice"),
    "verify_claims": ("Claim verifier", "Checks every number in the answer against the evidence", "skeptic"),
    "train_models": ("Model trainer", "Trains candidates and compares them to a baseline", "scientist"),
    "profile_data": ("Profiler", "Profiles the dataset", "scout"),
    "replan": ("Recovery", "Diagnoses a failed run and repairs the data or the code", "skeptic"),
}

_VALIDATORS = {
    "validator_syntax": ("syntax_error", "Syntax"),
    "validator_static": ("static_semantic_error", "Static semantics"),
    "validator_contract": ("contract_error", "Result contract"),
    "validator_logical": ("logical_semantic_error", "Logical review"),
}

_MAX_CODE = 20_000
_MAX_TEXT = 600
_MAX_FORMULA = 300


def _clip(text: Any, limit: int = _MAX_TEXT) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _by_model(entry: Optional[Dict[str, Any]], key: str = "reason") -> Optional[Dict[str, Any]]:
    """*entry* with ``<key>_segments``: its text a model wrote, as commentary
    (c-strict). The trace is the user's own view of their analysis
    (GET /threads/{id}/trace, every edition), so it marks what the chat marks."""
    if isinstance(entry, dict) and isinstance(entry.get(key), str) and entry[key].strip():
        from app.api import reply_segments as rs

        entry[key + "_segments"] = rs.commentary(entry[key])
    return entry


# --------------------------------------------------------------------------- #
# Small values
# --------------------------------------------------------------------------- #

def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value if not isinstance(value, str) else _clip(value, 120)
    if isinstance(value, float):
        return value if value == value else None  # NaN is not JSON
    try:
        import numpy as np  # noqa: WPS433

        if isinstance(value, np.generic):
            return _scalar(value.item())
    except Exception:  # noqa: BLE001
        pass
    return _clip(value, 120)


def small_table(records: Any, *, max_rows: int = 20, max_cols: int = 12) -> Optional[Dict[str, Any]]:
    """A result small enough to show on a graph node, or ``None``."""
    if not isinstance(records, list) or not records or not isinstance(records[0], dict):
        return None
    columns = [str(c) for c in list(records[0].keys())[:max_cols]]
    rows = [[_scalar(r.get(c)) for c in columns] for r in records[:max_rows] if isinstance(r, dict)]
    return {
        "columns": columns,
        "rows": rows,
        "total_rows": len(records),
        "truncated": len(records) > max_rows or len(records[0]) > max_cols,
    }


def result_records(state: Dict[str, Any]) -> Optional[list]:
    """The executed result as records, from wherever this graph left it."""
    out = state.get("output_json")
    if isinstance(out, list) and out:
        return out
    ofd = state.get("output_file_data")
    if isinstance(ofd, dict) and ofd.get("content"):
        try:
            from app.api.helpers import _datauri_csv_to_records

            return _datauri_csv_to_records(ofd["content"])
        except Exception:  # noqa: BLE001
            return None
    return None


# --------------------------------------------------------------------------- #
# The code, as operations
# --------------------------------------------------------------------------- #

_READERS = {"read_csv", "read_parquet", "read_excel", "read_json", "read_table", "read_feather",
            "read_orc", "read_avro", "read_delta", "read_iceberg", "read_sql", "read_sql_query",
            "read_sql_table", "scan_csv", "scan_parquet", "read_xml"}
_WRITERS = {"to_csv", "to_parquet", "to_json", "to_excel", "write_csv", "write_parquet", "to_sql"}
_AGGS = {"mean", "sum", "count", "median", "min", "max", "std", "var", "nunique", "size",
         "agg", "aggregate", "describe", "quantile", "value_counts", "mode", "prod", "cumsum",
         "corr", "cov", "first", "last"}
_JOINS = {"merge", "join"}
_CLEAN = {"dropna", "fillna", "drop_duplicates", "astype", "rename", "drop", "replace", "clip"}
_SORTS = {"sort_values", "sort_index", "nlargest", "nsmallest"}
_CASTS = {"to_numeric", "to_datetime", "to_timedelta", "astype"}


def _src(node: ast.AST) -> str:
    try:
        return _clip(ast.unparse(node), _MAX_FORMULA) or ""
    except Exception:  # noqa: BLE001
        return ""


def _consts(node: ast.AST) -> List[str]:
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and len(sub.value) <= 80:
            out.append(sub.value)
    return list(dict.fromkeys(out))[:12]


_PATH_SUFFIXES = (".csv", ".parquet", ".json", ".xlsx", ".xls", ".tsv", ".orc", ".avro", ".feather", ".txt")


def _looks_like_path(text: str) -> bool:
    return "/" in text or "\\" in text or text.lower().endswith(_PATH_SUFFIXES)


def _query_columns(call: ast.Call) -> List[str]:
    """Column names a ``df.query("...")`` string refers to."""
    if not call.args or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
        return []
    try:
        expr = ast.parse(call.args[0].value, mode="eval")
    except SyntaxError:
        return []
    names = [n.id for n in ast.walk(expr) if isinstance(n, ast.Name)]
    return list(dict.fromkeys(names))[:12]


def _column_names(candidates: List[str], known: Optional[List[str]]) -> List[str]:
    """The names in ``candidates`` that are columns: the known ones when the
    schema is known, otherwise every string that is not a path or an
    aggregation's name (``('salary', 'mean')`` names a column and a function)."""
    names = [c for c in candidates if c and not _looks_like_path(c) and len(c) <= 64 and c not in _AGGS]
    if known:
        matched = [c for c in names if c in known]
        if matched:
            return matched
    return names


def known_columns(state: Dict[str, Any]) -> List[str]:
    """Column names of the dataset this turn works on, from wherever the state has them."""
    out: List[str] = []
    cols = state.get("uploaded_csv_columns")
    if isinstance(cols, list):
        out.extend(str(c) for c in cols)
    schema = state.get("schema")
    if isinstance(schema, dict):
        out.extend(str(c) for c in schema.keys())
    elif isinstance(schema, list):
        out.extend(str(c) for c in schema if isinstance(c, str))
    stats = state.get("sample_statistics")
    if isinstance(stats, dict):
        cols = stats.get("columns")
        if isinstance(cols, dict):
            out.extend(str(c) for c in cols.keys())
        elif isinstance(cols, list):
            out.extend(str(c) for c in cols if isinstance(c, str))
    preview = state.get("sample_data")
    if isinstance(preview, list) and preview and isinstance(preview[0], dict):
        out.extend(str(c) for c in preview[0].keys())
    return list(dict.fromkeys(out))[:80]


def _chain_top(node: ast.AST, parents: Dict[ast.AST, ast.AST]) -> ast.AST:
    """The outermost expression of the method chain ``node`` sits in."""
    top = node
    while True:
        parent = parents.get(top)
        if isinstance(parent, ast.Attribute) and parent.value is top:
            top = parent
        elif isinstance(parent, ast.Subscript) and parent.value is top:
            top = parent
        elif isinstance(parent, ast.Call) and parent.func is top:
            top = parent
        else:
            return top


def _chain_methods(top: ast.AST) -> List[Tuple[str, ast.Call]]:
    """Method calls along a chain, innermost first."""
    calls: List[Tuple[str, ast.Call]] = []
    node = top
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.append((node.func.attr, node))
            node = node.func.value
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            break
    return list(reversed(calls))


def extract_operations(code: Optional[str], *, limit: int = 40,
                       known: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """The data operations in ``code``, in source order, with their formulas.

    Recognises pandas-style loads, filters, derived columns, group-bys,
    aggregations, joins, sorts, samples and writes. Every operation carries
    the ``columns`` it touches: the string constants in it that are column
    names, checked against ``known`` (the dataset's columns) when given.
    Unparseable code yields an empty list rather than an error: the code
    itself is still in the trace.
    """
    if not code or not isinstance(code, str):
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    parents: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    ops: List[Dict[str, Any]] = []
    claimed: set = set()

    def add(op: str, node: ast.AST, columns: Optional[List[str]] = None, **extra: Any) -> None:
        ops.append({"op": op, "line": getattr(node, "lineno", None),
                    "col": getattr(node, "col_offset", 0), "formula": _src(node),
                    "columns": _column_names(columns if columns is not None else _consts(node), known),
                    **extra})

    # Group-bys first: ast.walk is breadth-first, so the outer ``.mean()`` of
    # ``df.groupby(k)[c].mean()`` would otherwise be claimed as a bare aggregate.
    nodes = list(ast.walk(tree))
    nodes.sort(key=lambda n: 0 if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                   and n.func.attr == "groupby") else 1)
    for node in nodes:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if id(node) in claimed:
                continue
            if name in _READERS:
                add("load", node, columns=[], source=_consts(node)[:1] or None)
                claimed.add(id(node))
            elif name == "groupby":
                top = _chain_top(node, parents)
                methods = _chain_methods(top)
                for _, call in methods:
                    claimed.add(id(call))
                aggs = [_src(c)[_src(c).rfind(m):] if m in ("agg", "aggregate") else m
                        for m, c in methods if m in _AGGS]
                group_by = _consts(node.args[0]) if node.args else []
                add("group_aggregate", top, columns=group_by + [c for c in _consts(top) if c not in group_by],
                    group_by=group_by, aggregations=aggs)
            elif name == "pivot_table":
                add("pivot", node, columns=_consts(node))
                claimed.add(id(node))
            elif name in _JOINS:
                # ``", ".join(names)`` and ``os.path.join(a, b)`` are string
                # operations that share the method name; only a frame's
                # ``.join``/``.merge`` (or ``pd.merge``) is a data join.
                recv = node.func.value if isinstance(node.func, ast.Attribute) else None
                is_str_join = isinstance(recv, ast.Constant) and isinstance(recv.value, str)
                is_path_join = isinstance(recv, ast.Attribute) and recv.attr == "path"
                if not (is_str_join or is_path_join):
                    add("join", node, on=_consts(node))
                claimed.add(id(node))
            elif name == "concat":
                # Appending frames is not a join on keys; say what it is.
                add("concat", node, columns=[])
                claimed.add(id(node))
            elif name in _SORTS:
                add("sort", node)
                claimed.add(id(node))
            elif name == "sample":
                add("sample", node)
                claimed.add(id(node))
            elif name == "query":
                add("filter", node, columns=_query_columns(node))
                claimed.add(id(node))
            elif name in _WRITERS:
                add("write", node, columns=[], target=next((c for c in _consts(node) if _looks_like_path(c)), None))
                claimed.add(id(node))
            elif name in _CLEAN:
                add("clean", node, method=name)
                claimed.add(id(node))
            elif name in _AGGS:
                top = _chain_top(node, parents)
                for _, call in _chain_methods(top):
                    claimed.add(id(call))
                add("aggregate", top)
        elif isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, (ast.Compare, ast.BoolOp)) or (
                isinstance(sl, ast.BinOp) and isinstance(sl.op, (ast.BitAnd, ast.BitOr))
            ):
                add("filter", node, columns=_consts(sl))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) \
                    and isinstance(target.slice.value, str):
                value = node.value
                cast = (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                        and value.func.attr in _CASTS)
                ops.append({"op": "cast" if cast else "derive", "line": node.lineno,
                            "col": node.col_offset, "column": target.slice.value,
                            "formula": _clip(f"{_src(target)} = {_src(value)}", _MAX_FORMULA),
                            # The written column is new to the schema; keep it first.
                            "columns": [target.slice.value] + [
                                c for c in _column_names(_consts(value), known) if c != target.slice.value]})
                if cast:
                    claimed.add(id(value))

    # Source order, except that data flows from its loads and into its writes:
    # a script whose entry point sits at the bottom (``if __name__ ...``) loads
    # last in the text but first in the run.
    rank = {"load": 0, "write": 2}
    ops.sort(key=lambda o: (rank.get(o["op"], 1), o.get("line") or 0, o.get("col") or 0))
    for op in ops:
        op.pop("col", None)
    return ops[:limit]


# --------------------------------------------------------------------------- #
# Inputs and scope
# --------------------------------------------------------------------------- #

def _question(state: Dict[str, Any]) -> Optional[str]:
    """The question as the user asked it, not the run context appended to it."""
    from app.agents.user_text import last_user_text

    return _clip(last_user_text(state) or None, 400)


def data_scope(state: Dict[str, Any]) -> Dict[str, Any]:
    """Which data a step saw: the sample, its method and its size."""
    stats = state.get("sample_statistics")
    if isinstance(stats, str):
        try:
            import json

            stats = json.loads(stats)
        except Exception:  # noqa: BLE001
            stats = None
    total_rows = None
    if isinstance(stats, dict):
        total_rows = (stats.get("data_shape") or {}).get("rows") if isinstance(stats.get("data_shape"), dict) else None
    preview = state.get("sample_data")
    sample_rows = len(preview) if isinstance(preview, list) else None
    location = state.get("data_source_location")
    scope = {
        "dataset": os.path.basename(str(location)) if location else None,
        "fidelity": state.get("analysis_fidelity"),
        "sample": state.get("selected_sample_name"),
        "sample_rows_in_prompt": sample_rows,
        "total_rows": total_rows,
        "execution_mode": state.get("execution_mode"),
        "columns": known_columns(state) or None,
    }
    return {k: v for k, v in scope.items() if v is not None}


# --------------------------------------------------------------------------- #
# Per-node describers
# --------------------------------------------------------------------------- #

def _check(node: str, merged: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    flag, label = _VALIDATORS[node]
    failed = bool(merged.get(flag))
    detail = None
    if failed:
        detail = merged.get("code_validation_feedback")
        if node == "validator_contract" and merged.get("contract_violations"):
            detail = "; ".join(f"{v.get('rule')}: {v.get('detail')}" for v in merged["contract_violations"][:5]
                               if isinstance(v, dict)) or detail
    elif node == "validator_logical" and merged.get("logical_review_feedback"):
        detail = merged.get("logical_review_feedback")
    elif node == "validator_contract" and merged.get("contract_advisories"):
        detail = "; ".join(f"{v.get('rule')}: {v.get('detail')}" for v in merged["contract_advisories"][:5]
                           if isinstance(v, dict)) or None
    out = {"type": "check", "check": label, "value": "fail" if failed else "pass",
           "reason": _clip(detail)}
    if node == "validator_contract" and merged.get("contract_row_count"):
        # Where an exact row count came from: data, question, plan_columns or model.
        out["row_count"] = merged.get("contract_row_count")
    if node == "validator_contract" and merged.get("contract_bounds"):
        # Where each min/max bound came from: data, domain, question or model.
        out["bounds"] = merged.get("contract_bounds")
    if node == "validator_logical":
        out["decided_by"] = merged.get("logical_review_decided_by")
        out["overruled"] = merged.get("logical_review_overruled")
    return out


def _code_calc(merged: Dict[str, Any]) -> Dict[str, Any]:
    code = merged.get("generated_code")
    calc = {
        "operation": "generate code",
        "code": _clip(code, _MAX_CODE),
        "operations": extract_operations(code, known=known_columns(merged)),
        "pseudocode": _clip(merged.get("coder_pseudocode"), 2000),
    }
    contract = merged.get("coder_contract")
    if isinstance(contract, dict) and contract:
        calc["contract"] = {k: contract[k] for k in list(contract)[:8]}
    return calc


def _executed_code(state: Dict[str, Any]) -> Optional[str]:
    """The code an executor runs: ``generated_code``, else the coder definition."""
    code = state.get("generated_code")
    if not code:
        definition = state.get("coder_definition")
        code = definition.get("code") if isinstance(definition, dict) else None
    return code if isinstance(code, str) and code.strip() else None


def _numeric(values: Any, limit: int = 16) -> Dict[str, Any]:
    if not isinstance(values, dict):
        return {}
    out = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) and value != value:
            continue
        out[str(key)] = value
        if len(out) >= limit:
            break
    return out


def trained_models(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each model this turn trained: name, type, target, features and its metrics.

    One training turn trains one model in this build; ``training_result`` may
    one day carry ``candidates``, which are listed the same way.
    """
    result = state.get("training_result")
    plan = state.get("training_plan") if isinstance(state.get("training_plan"), dict) else {}
    data_config = plan.get("data_config") if isinstance(plan.get("data_config"), dict) else {}
    if not isinstance(result, dict):
        return []
    candidates = result.get("candidates") if isinstance(result.get("candidates"), list) else [result]
    out = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        model = cand.get("model") if isinstance(cand.get("model"), dict) else {}
        target = (model.get("target_column") or data_config.get("target_column")
                  or cand.get("target_column"))
        features = (model.get("feature_names") or data_config.get("feature_columns")
                    or cand.get("feature_columns") or [])
        out.append({
            "name": _clip(cand.get("model_name") or plan.get("model_name") or "model", 120),
            "type": _clip(cand.get("model_type") or plan.get("model_type"), 60),
            "target": _clip(target, 120) if target else None,
            "features": [str(f) for f in features][:40] if isinstance(features, list) else [],
            "metrics": _numeric(cand.get("metrics")),
            "runtime_s": cand.get("runtime_s") if isinstance(cand.get("runtime_s"), (int, float)) else None,
            "status": "failed" if (cand.get("status") == "error" or cand.get("error")
                                   or cand.get("execution_error")) else "trained",
        })
    return out


def evaluation_facts(report: Dict[str, Any]) -> Dict[str, Any]:
    """The evaluation's metric names and values, the baseline, and the verdict."""
    metrics = {}
    for name, value in (report.get("metrics") or {}).items() if isinstance(report.get("metrics"), dict) else []:
        if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
            metrics[str(name)] = value["mean"]
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[str(name)] = value
    return {
        "task": report.get("task"),
        "primary_metric": report.get("primary_metric"),
        "metrics": metrics,
        "baseline": _numeric(report.get("baseline")),
        "beats_baseline": report.get("beats_baseline"),
        "lift_over_baseline": report.get("lift_over_baseline"),
        "n_folds": report.get("n_folds"),
        "n_rows": report.get("n_rows"),
        "splitter": report.get("splitter"),
    }


def describe_step(node: str, before: Dict[str, Any], updates: Dict[str, Any],
                  *, error: Optional[BaseException] = None,
                  error_class: Optional[str] = None) -> Dict[str, Any]:
    # ``error`` is the exception a node raised; ``error_class`` is the class
    # name of a failure a node recorded on the state instead of raising
    # (app.trace.recorder.recorded_failure). Either one makes this a failed step.
    merged = {**before, **updates}
    if error is not None:
        error_class = type(error).__name__
    agent, role, clone = AGENT_CATALOG.get(node, (node, "", None))
    out: Dict[str, Any] = {
        "agent": agent,
        "role": role,
        "clone": clone,
        "inputs": {"question": _question(before), "data_scope": data_scope(before)},
        "decision": None,
        "calculation": None,
        "gates": [],
    }
    if error_class:
        # The class only: an exception message can carry a path or a value.
        out["decision"] = {"type": "error", "value": "failed", "reason": error_class}
        return out

    if node == "avaloka_agent":
        delegate = bool(merged.get("delegate_to_planner"))
        out["decision"] = {
            "type": "intent",
            "value": merged.get("avaloka_intent"),
            "reason": _clip(merged.get("avaloka_intent_reasoning")),
            "confidence": merged.get("avaloka_intent_confidence"),
            "delegate": delegate,
            "mode": merged.get("avaloka_mode"),
        }
        # The intent classifier's reasoning records no provenance: the model's.
        _by_model(out["decision"])
        rec = updates.get("avaloka_recommendation")
        if isinstance(rec, dict):
            out["calculation"] = {
                "operation": "choose execution profile",
                "values": {k: rec.get(k) for k in ("size_human", "size_class", "fidelity",
                                                  "execution_mode", "eta_text")},
                "reason": rec.get("cost_note"),
            }
            if rec.get("confirm_before_full_run"):
                out["gates"].append({"gate": "confirm_full_run", "status": "required",
                                     "reason": "Full-fidelity run on a large input needs the user's confirmation."})
        if merged.get("avaloka_handoff_message") and delegate:
            out["handoff_note"] = merged.get("avaloka_handoff_message")
    elif node == "plan_etl":
        pending = updates.get("pending_clarification")
        if isinstance(pending, dict) and pending:
            # The planner's question, model-written or not recorded: commentary.
            out["decision"] = _by_model({"type": "clarify", "value": pending.get("type"),
                                         "reason": _clip(pending.get("question") or pending.get("message"))})
            out["gates"].append(_by_model({"gate": "clarification", "status": "waiting_on_user",
                                           "reason": _clip(pending.get("question") or pending.get("message"))}))
        else:
            ready = "code" if merged.get("ready_to_code") else (
                "summarize" if merged.get("ready_to_summarize") else "answer")
            # value: the route the flags chose (code); reason: the planner's prose.
            out["decision"] = _by_model({"type": "plan", "value": ready, "reason": _clip(merged.get("plan"), 1500)})
        action = merged.get("training_plan_reply_action")
        if action and action != "none":
            out["gates"].append({"gate": "training_plan_approval", "status": action})
    elif node == "summarize_etl":
        definition = merged.get("planner_definition")
        steps = []
        if isinstance(definition, dict):
            steps = definition.get("transformations") or definition.get("steps") or []
        out["calculation"] = {"operation": "structure the plan", "values": {"steps": len(steps)}}
    elif node == "generate_planner_graph":
        out["decision"] = {"type": "artifact", "value": merged.get("planner_graph_status")}
    elif node in ("code_etl", "coder"):
        out["calculation"] = _code_calc(merged)
        if node == "code_etl":
            violations = merged.get("contract_violations") or []
            out["decision"] = {
                "type": "verdict",
                "value": "refused" if merged.get("coder_blocked_reason") else (
                    "fail" if merged.get("contract_error") else "pass"),
                "reason": _clip(merged.get("coder_blocked_reason") or (
                    "; ".join(str(v.get("detail")) for v in violations[:3] if isinstance(v, dict)) or None)),
            }
        else:
            out["calculation"]["attempt"] = int(merged.get("retry_count") or 0) + 1
    elif node in _VALIDATORS:
        out["decision"] = _check(node, merged, updates)
        if node == "validator_logical":
            # The logical review is a model's judgement.
            _by_model(out["decision"])
    elif node == "execute_code":
        preview = merged.get("execution_output_preview") or merged.get("execution_output_data")
        failed = bool(updates.get("execution_error"))
        out["decision"] = {"type": "check", "check": "Dry run", "value": "fail" if failed else "pass",
                           "reason": type_only(updates.get("execution_error"))}
        code = before.get("generated_code")
        out["calculation"] = {"operation": "dry run on the sample", "result": small_table(preview),
                              "operations": extract_operations(code, known=known_columns(merged))}
    elif node in ("execute_locally", "execute_on_ray", "execute_on_k8s", "execute_actions"):
        result = merged.get("execution_result") or {}
        status = result.get("status") if isinstance(result, dict) else None
        records = result_records(merged)
        out["decision"] = {"type": "verdict", "value": status or ("fail" if merged.get("execution_error") else None),
                           "reason": _clip(result.get("message")) if isinstance(result, dict) else None}
        out["calculation"] = {
            "operation": {"execute_locally": "run locally", "execute_on_ray": "run on Ray",
                          "execute_on_k8s": "run as a Kubernetes job",
                          "execute_actions": "run structured actions"}[node],
            "data_scope": data_scope(merged),
            "values": {"output_rows": result.get("output_row_count") if isinstance(result, dict) else None},
            "result": small_table(records),
        }
        if node != "execute_actions":
            # The code this step ran, as the executors read it: the validated
            # code on the state, or the coder's definition it falls back to.
            code = _executed_code(merged)
            out["calculation"]["code"] = _clip(code, _MAX_CODE)
            out["calculation"]["operations"] = extract_operations(code, known=known_columns(merged))
    elif node == "visualize":
        cfg = merged.get("visualization_config") or {}
        if isinstance(cfg, dict):
            charts = [c for c in (cfg.get("charts") or []) if isinstance(c, dict)]
            out["calculation"] = {
                "operation": "configure charts",
                "values": {"charts": len(charts),
                           "rows_charted": (cfg.get("dataset") or {}).get("rows_charted")
                           if isinstance(cfg.get("dataset"), dict) else None},
                "charts": [{"type": c.get("type"), "title": _clip(c.get("title"), 120),
                            "intent": c.get("intent"), "reason": _clip(c.get("reason"), 300)}
                           for c in charts[:6]],
            }
        out["decision"] = {"type": "artifact", "value": merged.get("visualization_status")}
    elif node == "narrate_result":
        text = None
        answer_msg = None
        for msg in reversed(updates.get("messages") or []):
            if getattr(msg, "type", None) == "ai":
                text = msg.content
                answer_msg = msg
                break
        out["calculation"] = {"operation": "write the answer", "answer": _clip(text, 1200)}
        if answer_msg is not None and out["calculation"]["answer"]:
            # The answer's own provenance (an untagged message is commentary).
            from app.api import reply_segments as rs

            segs = rs.of_message(answer_msg)
            if rs.text_of(segs) == out["calculation"]["answer"]:
                out["calculation"]["answer_segments"] = segs
            else:   # clipped: one kind for the lot, never plain unless all of it was
                kind = rs.PLAIN if segs and all(s["kind"] == rs.PLAIN for s in segs) else rs.COMMENTARY
                out["calculation"]["answer_segments"] = rs.normalize(
                    [rs.segment(out["calculation"]["answer"], kind)])
    elif node == "verify_claims":
        report = merged.get("verification_report") or {}
        if isinstance(report, dict):
            # A failed or empty analysis was not verified; "unsafe" would
            # read as claims checked and found wanting.
            not_verified = report.get("verified") is False and report.get("analysis") in ("failed", "no_result")
            checked = report.get("claims_checked", 0)
            seen = report.get("claims_seen")
            # D13.2: what a check resolved, out of what the reply says, when fewer.
            partly = isinstance(seen, int) and isinstance(checked, int) and seen > checked
            out["decision"] = {
                "type": "evidence",
                "value": "not_verified" if not_verified else (
                    "safe" if report.get("safe_to_present", True) else "unsafe"),
                "reason": report.get("skipped") or (
                    (f"{report.get('n_problems', 0)} of {seen} statement(s) need attention; {checked} checked "
                     "against the numbers" if partly else
                     f"{report.get('n_problems', 0)} of {checked} claim(s) need attention")
                    if report.get("n_problems") else
                    # The count, no verdict (owner's ruling, as in the chat's
                    # step card): checking found no problem; it did not prove
                    # every claim true.
                    (f"Checked {checked} of {seen} statement(s) against the numbers" if partly else
                     f"Checked {checked} claim(s) against the numbers")),
            }
            out["calculation"] = {
                "operation": "verify claims against the evidence",
                "values": {"claims_checked": report.get("claims_checked"),
                           **({"claims_seen": seen} if isinstance(seen, int) else {}),
                           "problems": report.get("n_problems")},
                "findings": [
                    {k: (_clip(f.get(k), 300) if isinstance(f.get(k), str) else f.get(k))
                     for k in ("claim", "verdict", "check", "detail", "evidence_ref")}
                    for f in (report.get("findings") or [])[:12] if isinstance(f, dict)
                ],
            }
    elif node == "train_models":
        ev = merged.get("evaluation_report") or {}
        out["decision"] = {"type": "evidence", "value": (
            "beats_baseline" if merged.get("evaluation_beats_baseline") else
            ("does_not_beat_baseline" if merged.get("evaluation_beats_baseline") is False else None)),
            "reason": _clip(ev.get("summary")) if isinstance(ev, dict) else None}
        out["calculation"] = {"operation": "train and compare to baseline",
                              "values": {k: v for k, v in (merged.get("training_metrics") or {}).items()
                                         if isinstance(v, (int, float))} if isinstance(
                                  merged.get("training_metrics"), dict) else None,
                              "models": trained_models(merged),
                              "evaluation": evaluation_facts(ev) if isinstance(ev, dict) else None}
        if merged.get("integrity_safe_to_train") is False:
            out["gates"].append({"gate": "integrity", "status": "blocked",
                                 "reason": "The integrity check found the training data unsafe to use."})
    elif node == "memory_injection":
        hints = merged.get("memory_hints")
        out["calculation"] = {"operation": "recall thread memory",
                              "values": {"hints": len(hints) if isinstance(hints, list) else 0}}
    return out


def type_only(error: Any) -> Optional[str]:
    """An error's first line, without the value-bearing tail."""
    if not error:
        return None
    first = str(error).strip().splitlines()[0] if str(error).strip() else ""
    return _clip(first, 200)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

def route_reason(router: str, chosen: str, state: Dict[str, Any]) -> str:
    """One sentence saying why a router took the branch it took."""
    intent = state.get("avaloka_intent")
    mode = state.get("execution_mode") or "local"
    if router == "route_avaloka":
        return (f"The '{intent}' intent needs the data, so Avaloka hands it to the planner."
                if chosen == "delegate_to_planner" else
                f"The '{intent}' intent is conversational, so Avaloka answers directly.")
    if router == "route_planner_output":
        return {
            "prepare_code": "The plan is ready, so it goes to the coding team.",
            "summarize": "The plan needs a structured definition before code is written.",
            "execute_locally": f"Code already exists; run it ({mode}).",
            "execute_on_ray": "Code already exists; run it on the Ray cluster.",
            "execute_on_k8s": "Code already exists; run it as a Kubernetes job.",
            "schedule_task": "The request is a scheduled or long-running job.",
            "task_operation": "The request manages an existing task.",
            "train_models": "The request is model training or inference.",
            "provision_infra": "Requested infrastructure is not provisioned yet.",
            "continue_planning": ("The planner needs more from the user before any code runs."
                                  if state.get("pending_clarification") else
                                  "The planner answered without running code."),
            "end": "The planner's answer is complete.",
        }.get(chosen, f"Route '{chosen}'.")
    if router == "route_after_code":
        if chosen == "end":
            return (f"No runnable code: {_clip(state.get('coder_blocked_reason'), 200)}"
                    if state.get("coder_blocked_reason") else "No runnable code was produced.")
        return f"Validated code goes to {chosen.replace('_', ' ')} (mode {mode})."
    if router == "route_after_execution":
        if chosen == "visualize":
            return "Execution produced a result, so it goes to the visualizer."
        status = (state.get("execution_result") or {}).get("status") if isinstance(
            state.get("execution_result"), dict) else None
        return f"Execution ended with status '{status or 'unknown'}'; nothing to chart."
    if router == "check_validation_status":
        failed = [label for node, (flag, label) in _VALIDATORS.items() if state.get(flag)]
        attempts = int(state.get("retry_count") or 0)
        if chosen == "end":
            return ("All validators passed." if not failed else
                    f"Retry limit reached after {attempts} attempt(s); still failing: {', '.join(failed)}.")
        return f"{', '.join(failed) or 'A validator'} failed, so the code goes back to the coder."
    if router == "route_after_summary":
        return "The structured plan is drawn as a graph."
    if router == "route_after_graph":
        return "The drawn plan goes to the coding team."
    if router == "route_after_training":
        return "Training finished." if chosen == "end" else f"Training continues with {chosen}."
    return f"Route '{chosen}'."
