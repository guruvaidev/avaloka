"""Assemble recorded steps into the trace API and the granular planner graph.

Input is the flat list the recorder leaves on ``ETLState.agent_trace`` for one
turn: agent steps and route decisions, each with a ``seq``, and children of a
step (the coding sub-graph's nodes) pointing at their parent by ``parent_id``.

Output shapes are documented in ``docs/design/AGENT_TRACE.md`` and pinned by
``tests/test_agent_trace.py``.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional

from app.trace.describe import AGENT_CATALOG
from app.trace.recorder import TRACE_SCHEMA_VERSION

_TOP = None  # parent_id of a top-level step


def _sorted(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted((e for e in entries if isinstance(e, dict)), key=lambda e: e.get("seq") or 0)


def _inference_totals(steps: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    calls = failed = tokens_in = tokens_out = 0
    cost = 0.0
    unpriced = 0
    latency = 0
    for step in steps:
        for call in step.get("inference") or []:
            calls += 1
            if call.get("outcome") != "success":
                failed += 1
            tokens_in += int(call.get("tokens_in") or 0)
            tokens_out += int(call.get("tokens_out") or 0)
            latency += int(call.get("latency_ms") or 0)
            if call.get("cost_usd") is None:
                unpriced += 1
            else:
                cost += float(call["cost_usd"])
    return {
        "llm_calls": calls,
        "llm_failures": failed,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "llm_latency_ms": latency,
        # Known cost only. ``unpriced_calls`` says how much of the total is
        # missing, rather than presenting a partial sum as the whole.
        "cost_usd": round(cost, 6) if calls and unpriced < calls else None,
        "unpriced_calls": unpriced,
    }


def _handoffs(ordered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Who passed what to whom, per level of the graph."""
    out = []
    by_parent: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for entry in ordered:
        by_parent.setdefault(entry.get("parent_id"), []).append(entry)
    for parent_id, entries in by_parent.items():
        prev: Optional[Dict[str, Any]] = None
        route: Optional[Dict[str, Any]] = None
        for entry in entries:
            if entry.get("kind") == "decision":
                route = entry
                continue
            if entry.get("kind") != "agent":
                continue
            if prev is not None:
                out.append({
                    "from": prev["node"], "from_step": prev["step_id"],
                    "to": entry["node"], "to_step": entry["step_id"],
                    "parent_id": parent_id,
                    # State keys, never their values: what was handed over.
                    "passed": list(prev.get("written") or [])[:24],
                    "route": (route or {}).get("decision", {}).get("chosen") if route else None,
                    "reason": (route or {}).get("decision", {}).get("reason") if route else None,
                })
            prev, route = entry, None
    return out


def _clones(swarm_plan: Optional[list], steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The narrated swarm as steps, with the real agent steps that did each clone's work."""
    out = []
    for clone in swarm_plan or []:
        if not isinstance(clone, dict):
            continue
        key = clone.get("clone")
        mine = [s for s in steps if s.get("kind") == "agent" and s.get("clone") == key]
        if not mine:
            status = "not_run"
        elif any(s.get("status") == "error" for s in mine):
            status = "failed"
        else:
            status = "done"
        out.append({
            "kind": "swarm_clone",
            "clone": key,
            "name": clone.get("name"),
            "role": clone.get("role"),
            "focus": clone.get("focus"),
            "status": status,
            "agent_steps": [s["step_id"] for s in mine],
            "latency_ms": sum(int(s.get("latency_ms") or 0) for s in mine) if mine else None,
        })
    return out


def build_turn(
    entries: Iterable[Dict[str, Any]],
    *,
    turn_id: Optional[str] = None,
    question: Optional[str] = None,
    data_scope: Optional[Dict[str, Any]] = None,
    swarm_plan: Optional[list] = None,
    swarm_message: Optional[str] = None,
) -> Dict[str, Any]:
    ordered = _sorted(entries)
    if turn_id:
        ordered = [e for e in ordered if e.get("turn_id") == turn_id]
    agents = [e for e in ordered if e.get("kind") == "agent"]
    top = [e for e in agents if e.get("parent_id") is _TOP]
    started = min((e.get("started_at") for e in agents if e.get("started_at")), default=None)
    ended = max((e.get("ended_at") for e in agents if e.get("ended_at")), default=None)
    turn = {
        "turn_id": turn_id or (ordered[0].get("turn_id") if ordered else None),
        "question": question or next(
            ((e.get("inputs") or {}).get("question") for e in top if (e.get("inputs") or {}).get("question")), None),
        "started_at": started,
        "ended_at": ended,
        "latency_ms": sum(int(e.get("latency_ms") or 0) for e in top),
        "data_scope": data_scope or {},
        "steps": ordered,
        "handoffs": _handoffs(ordered),
        "swarm": {"clones": _clones(swarm_plan, agents), "message": swarm_message},
        "totals": {
            "agent_steps": len(agents),
            "decisions": sum(1 for e in ordered if e.get("kind") == "decision"),
            "errors": sum(1 for e in agents if e.get("status") == "error"),
            **_inference_totals(agents),
        },
    }
    return turn


def agent_metrics(turns: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-agent aggregates across turns: calls, latency, tokens, cost, outcomes, retries."""
    rows: Dict[str, Dict[str, Any]] = {}
    for turn in turns:
        coder_attempts: Dict[str, int] = {}
        for step in turn.get("steps") or []:
            if step.get("kind") != "agent":
                continue
            node = step.get("node")
            row = rows.setdefault(node, {
                "node": node, "agent": step.get("agent") or AGENT_CATALOG.get(node, (node,))[0],
                "role": step.get("role"), "activations": 0, "succeeded": 0, "failed": 0,
                "retries": 0, "_lat": [], "llm_calls": 0, "llm_failures": 0, "tokens_in": 0,
                "tokens_out": 0, "cost_usd": 0.0, "unpriced_calls": 0, "models": set(),
            })
            row["activations"] += 1
            row["succeeded" if step.get("status") == "ok" else "failed"] += 1
            row["_lat"].append(int(step.get("latency_ms") or 0))
            if node == "coder":
                parent = step.get("parent_id") or ""
                coder_attempts[parent] = coder_attempts.get(parent, 0) + 1
            totals = _inference_totals([step])
            for key in ("llm_calls", "llm_failures", "tokens_in", "tokens_out", "unpriced_calls"):
                row[key] += totals[key]
            row["cost_usd"] += totals["cost_usd"] or 0.0
            for call in step.get("inference") or []:
                if call.get("model"):
                    row["models"].add(f"{call.get('provider')}/{call.get('model')}")
        for parent, attempts in coder_attempts.items():
            if "coder" in rows and attempts > 1:
                rows["coder"]["retries"] += attempts - 1
    out = []
    for row in rows.values():
        lat = row.pop("_lat")
        row["latency_ms"] = {
            "total": sum(lat),
            "p50": int(statistics.median(lat)) if lat else None,
            "max": max(lat) if lat else None,
        }
        row["models"] = sorted(row["models"])
        row["cost_usd"] = round(row["cost_usd"], 6) if row["llm_calls"] > row["unpriced_calls"] else None
        out.append(row)
    order = list(AGENT_CATALOG)
    out.sort(key=lambda r: order.index(r["node"]) if r["node"] in order else len(order))
    return out


def trace_response(thread_id: str, turns: List[Dict[str, Any]], *,
                   swarm_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "thread_id": thread_id,
        "turns": turns,
        "agents": agent_metrics(turns),
        "totals": _inference_totals(
            s for t in turns for s in (t.get("steps") or []) if s.get("kind") == "agent"),
        "swarm_limits": swarm_limits,
    }


# --------------------------------------------------------------------------- #
# The granular planner graph
# --------------------------------------------------------------------------- #

NODE_TYPES = ("question", "agent", "decision", "gate", "data", "operation", "check",
              "result", "evidence", "train", "evaluate")
EDGE_TYPES = ("flow", "route", "handoff", "data", "check", "retry", "evidence", "gate")


class _Graph:
    def __init__(self) -> None:
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self._ids: set = set()

    def node(self, node_id: str, type_: str, label: str, *, lane: str, **extra: Any) -> str:
        if node_id not in self._ids:
            self._ids.add(node_id)
            self.nodes.append({"id": node_id, "type": type_, "label": label, "lane": lane,
                               **{k: v for k, v in extra.items() if v is not None}})
        return node_id

    def edge(self, src: str, dst: str, type_: str, label: Optional[str] = None, **extra: Any) -> None:
        if src in self._ids and dst in self._ids and src != dst:
            self.edges.append({"id": f"e{len(self.edges)}", "source": src, "target": dst,
                               "type": type_, **({"label": label} if label else {}),
                               **{k: v for k, v in extra.items() if v is not None}})


def _metrics(step: Dict[str, Any]) -> Dict[str, Any]:
    totals = _inference_totals([step])
    out = {"latency_ms": step.get("latency_ms")}
    if totals["llm_calls"]:
        out.update({"llm_calls": totals["llm_calls"],
                    "tokens": totals["tokens_in"] + totals["tokens_out"],
                    "cost_usd": totals["cost_usd"]})
        models = sorted({c.get("model") for c in step.get("inference") or [] if c.get("model")})
        if models:
            out["model"] = models[0] if len(models) == 1 else f"{models[0]} +{len(models) - 1}"
    return out


def _subtitle(step: Dict[str, Any]) -> Optional[str]:
    decision = step.get("decision") or {}
    if decision.get("type") == "intent":
        return f"intent: {decision.get('value')}"
    if decision.get("type") == "evidence" and decision.get("value"):
        values = (step.get("calculation") or {}).get("values") or {}
        seen, checked = values.get("claims_seen"), values.get("claims_checked")
        # D13.2: the statements of the reply when a check resolved fewer of them.
        partly = isinstance(seen, int) and isinstance(checked, int) and seen > checked
        if values.get("problems"):
            # "safe" alone would hide that claims were flagged.
            if partly:
                return f"{decision['value']} · {values['problems']} of {seen} statements flagged"
            return f"{decision['value']} · {values['problems']} of {checked} claims flagged"
        if decision["value"] == "safe" and checked is not None:
            # The count, not a verdict: "safe" read as every claim verified.
            return f"{checked} of {seen} statement(s) checked" if partly else f"{checked} claim(s) checked"
        return str(decision["value"])
    if decision.get("type") in ("check", "verdict") and decision.get("value"):
        return str(decision.get("value"))
    if decision.get("type") in ("plan", "clarify", "artifact") and decision.get("value"):
        return f"{decision['type']}: {decision['value']}"
    calc = step.get("calculation") or {}
    if calc.get("operation"):
        return str(calc["operation"])
    return None


def _op_label(op: Dict[str, Any]) -> str:
    kind = op.get("op")
    cols = ", ".join(op.get("columns") or [])
    if kind == "group_aggregate":
        by = ", ".join(op.get("group_by") or []) or "keys"
        aggs = ", ".join(op.get("aggregations") or []) or "aggregate"
        return f"Group by {by} → {aggs}"
    if kind == "derive":
        return f"Derive {op.get('column')}"
    if kind == "cast":
        return f"Cast {op.get('column')}"
    if kind == "load":
        return "Load"
    if kind == "filter" and cols:
        return f"Filter on {cols}"
    if kind == "sort" and cols:
        return f"Sort by {cols}"
    if kind == "join" and cols:
        return f"Join on {cols}"
    if kind == "write" and op.get("target"):
        return f"Write {op['target']}"
    return {"filter": "Filter", "join": "Join", "concat": "Append rows", "sort": "Sort", "sample": "Sample",
            "write": "Write", "clean": f"Clean ({op.get('method')})", "aggregate": "Aggregate",
            "pivot": "Pivot"}.get(kind, str(kind))


def _layout(graph: _Graph) -> None:
    """Layer by longest path from the root.

    A retry edge joins a failing check to the *next* coder attempt, a node of
    its own, so it points forward like every other edge; the ``seen`` guard
    below still stops any cycle a future edge type might introduce.
    """
    incoming: Dict[str, List[str]] = {n["id"]: [] for n in graph.nodes}
    for e in graph.edges:
        incoming[e["target"]].append(e["source"])
    layer: Dict[str, int] = {}
    order = [n["id"] for n in graph.nodes]

    def depth(node_id: str, seen: frozenset = frozenset()) -> int:
        if node_id in layer:
            return layer[node_id]
        if node_id in seen:
            return 0
        preds = incoming.get(node_id) or []
        value = 0 if not preds else 1 + max(depth(p, seen | {node_id}) for p in preds)
        layer[node_id] = value
        return value

    for node_id in order:
        depth(node_id)
    slots: Dict[tuple, int] = {}
    for n in graph.nodes:
        key = (layer[n["id"]], n["lane"])
        n["layer"] = layer[n["id"]]
        n["order"] = slots.get(key, 0)
        slots[key] = n["order"] + 1


def planner_graph(turn: Dict[str, Any], *, plan_steps: Optional[list] = None) -> Dict[str, Any]:
    """A layered, typed graph of one turn: agents, decisions, data, checks, evidence.

    Every node that came from a recorded step carries its ``step_id`` so the UI
    can open the same step in the trace.
    """
    g = _Graph()
    steps = turn.get("steps") or []
    q = g.node("q", "question", turn.get("question") or "Question", lane="agents")
    prev_top = q
    pending_route: Optional[Dict[str, Any]] = None
    final_ops_from: Optional[Dict[str, Any]] = None
    executor: Optional[Dict[str, Any]] = None
    trainer: Optional[Dict[str, Any]] = None
    verifier: Optional[Dict[str, Any]] = None

    group_tail: Optional[str] = None
    by_id = steps_by_id(steps)
    children: Dict[str, List[Dict[str, Any]]] = {}
    for s in steps:
        if s.get("parent_id"):
            children.setdefault(s["parent_id"], []).append(s)

    for s in steps:
        if s.get("parent_id"):
            continue
        if s.get("kind") == "decision":
            d = s.get("decision") or {}
            nid = g.node(s["step_id"], "decision", d.get("router", "route").replace("_", " "),
                         lane="agents", step_id=s["step_id"], value=d.get("chosen"),
                         reason=d.get("reason"), signals=d.get("signals"))
            g.edge(prev_top, nid, "flow")
            if group_tail:
                g.edge(group_tail, nid, "handoff", "validated code")
                group_tail = None
            prev_top, pending_route = nid, s
            continue
        if s.get("kind") != "agent":
            continue
        type_ = "evidence" if s.get("node") == "verify_claims" else "agent"
        nid = g.node(s["step_id"], type_, s.get("agent") or s.get("node"), lane="agents",
                     step_id=s["step_id"], node=s.get("node"), status=s.get("status"),
                     subtitle=_subtitle(s), metrics=_metrics(s), clone=s.get("clone"))
        if pending_route is not None:
            g.edge(prev_top, nid, "route", (pending_route.get("decision") or {}).get("chosen"))
        else:
            passed = list(by_id.get(prev_top, {}).get("written") or [])[:4]
            g.edge(prev_top, nid, "handoff" if passed else "flow", ", ".join(passed) or None)
        if group_tail:
            g.edge(group_tail, nid, "handoff", "validated code")
            group_tail = None
        prev_top, pending_route = nid, None

        for gate in s.get("gates") or []:
            gid = g.node(f"{nid}:gate:{gate.get('gate')}", "gate", gate.get("gate", "gate").replace("_", " "),
                         lane="agents", value=gate.get("status"), reason=gate.get("reason"),
                         step_id=s["step_id"])
            g.edge(nid, gid, "gate")

        if s.get("node") == "code_etl":
            final_ops_from = s
            group_tail = _coding_group(g, nid, children.get(s["step_id"], []))
        elif s.get("node", "").startswith("execute_"):
            executor = s
        elif s.get("node") == "train_models":
            trainer = s
        elif s.get("node") == "verify_claims":
            verifier = s

    _data_lane(g, turn, q, final_ops_from,
               children.get(final_ops_from["step_id"], []) if final_ops_from else [],
               executor, trainer, verifier)
    _layout(g)
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "turn_id": turn.get("turn_id"),
        "node_types": list(NODE_TYPES),
        "edge_types": list(EDGE_TYPES),
        "nodes": g.nodes,
        "edges": g.edges,
        "plan_steps": plan_steps or None,
    }


def steps_by_id(steps: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {s.get("step_id"): s for s in steps if isinstance(s, dict)}


def _coding_group(g: _Graph, parent_nid: str, kids: List[Dict[str, Any]]) -> Optional[str]:
    """The coding sub-graph: each attempt and each check, with retries looping back."""
    prev = parent_nid
    last_failed_check: Optional[str] = None
    attempt = 0
    for k in _sorted(kids):
        if k.get("kind") == "decision":
            d = k.get("decision") or {}
            if d.get("chosen") == "refine" and last_failed_check:
                # The retry edge carries the reason; a separate node would only repeat it.
                g.edges.append({"id": f"e{len(g.edges)}", "source": last_failed_check, "target": "__next_coder__",
                                "type": "retry", "label": d.get("reason")})
            continue
        if k.get("kind") != "agent":
            continue
        node = k.get("node")
        if node == "coder":
            attempt += 1
            nid = g.node(k["step_id"], "agent", f"Coder · attempt {attempt}", lane="code",
                         step_id=k["step_id"], node=node, status=k.get("status"),
                         subtitle=f"{len((k.get('calculation') or {}).get('operations') or [])} operations",
                         metrics=_metrics(k), clone=k.get("clone"), group=parent_nid)
            retried = False
            for e in g.edges:
                if e.get("target") == "__next_coder__":
                    e["target"] = nid
                    retried = True
            if not retried:
                # A retry edge already joins the failing check to this attempt.
                g.edge(prev, nid, "flow")
            prev, last_failed_check = nid, None
            continue
        d = k.get("decision") or {}
        nid = g.node(k["step_id"], "check", d.get("check") or k.get("agent") or node, lane="code",
                     step_id=k["step_id"], node=node, status=k.get("status"), value=d.get("value"),
                     reason=d.get("reason"), metrics=_metrics(k), group=parent_nid)
        g.edge(prev, nid, "check")
        prev = nid
        if d.get("value") == "fail":
            last_failed_check = nid
    # A retry that never got a next attempt points nowhere; drop it.
    g.edges[:] = [e for e in g.edges if e.get("target") != "__next_coder__"]
    return prev if prev != parent_nid else None


def validated_attempt(kids: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The last coder attempt none of whose checks failed, from the coding sub-graph's steps."""
    chosen: Optional[Dict[str, Any]] = None
    attempt: Optional[Dict[str, Any]] = None
    failed = False
    for k in _sorted(kids):
        if k.get("kind") != "agent":
            continue
        if k.get("node") == "coder":
            if attempt is not None and not failed:
                chosen = attempt
            attempt, failed = k, False
        elif (k.get("decision") or {}).get("value") == "fail":
            failed = True
    if attempt is not None and not failed:
        chosen = attempt
    return chosen


def operations_source(code_step: Optional[Dict[str, Any]], kids: List[Dict[str, Any]],
                      executor: Optional[Dict[str, Any]]) -> tuple:
    """Where the data operations come from, in order of trust: the code the
    executor actually ran, else the last validated coder attempt, else the
    coding team's final code. Never the plan text."""
    def ops_of(step: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        calc = (step or {}).get("calculation") or {}
        ops = list(calc.get("operations") or [])
        if ops and calc.get("code") and any("columns" not in o for o in ops):
            # Recorded before operations carried their columns: read them again
            # from the code the step kept, so an older trace draws the same picture.
            from app.trace.describe import extract_operations

            scope = ((step or {}).get("inputs") or {}).get("data_scope") or {}
            fresh = extract_operations(calc["code"], known=scope.get("columns"))
            if fresh:
                return fresh
        return ops

    if executor and ops_of(executor):
        return "executed", executor, ops_of(executor)
    attempt = validated_attempt(kids)
    if attempt and ops_of(attempt):
        return "validated", attempt, ops_of(attempt)
    if code_step and ops_of(code_step):
        return "generated", code_step, ops_of(code_step)
    return None, None, []


def _data_lane(g: _Graph, turn: Dict[str, Any], question: str, code_step: Optional[Dict[str, Any]],
               kids: List[Dict[str, Any]], executor: Optional[Dict[str, Any]],
               trainer: Optional[Dict[str, Any]], verifier: Optional[Dict[str, Any]]) -> None:
    """The spine: question → load → sample → operations (columns, formulas)
    → result, or → train → evaluate; → evidence.

    Operations are read from the code that actually ran (see
    :func:`operations_source`), each with the columns it touches.
    """
    scope = dict(turn.get("data_scope") or {})
    for step in (code_step, executor, trainer):
        if step:
            scope = {**((step.get("inputs") or {}).get("data_scope") or {}), **scope}
    if not (code_step or executor or trainer):
        return
    source, ops_step, ops = operations_source(code_step, kids, executor)
    columns = scope.get("columns") if isinstance(scope.get("columns"), list) else None
    load_op = next((o for o in ops if o.get("op") == "load"), None)
    load = g.node("data:load", "data", f"Load {scope.get('dataset') or 'dataset'}", lane="data",
                  values={k: v for k, v in {"rows": scope.get("total_rows"),
                                            "columns": len(columns) if columns else None}.items()
                          if v is not None} or None,
                  columns=columns, formula=(load_op or {}).get("formula"), step_id=(ops_step or {}).get("step_id"))
    g.edge(question, load, "data", "acts on")
    sample_label = scope.get("sample") or scope.get("fidelity") or "full data"
    sample = g.node("data:sample", "data", f"Sample · {sample_label}", lane="data",
                    values={k: scope.get(k) for k in ("fidelity", "sample", "rows_analyzed",
                                                      "total_rows", "is_sampled")
                            if scope.get(k) is not None} or None)
    g.edge(load, sample, "data")
    prev = sample
    if code_step:
        g.edge(code_step["step_id"], load, "data", "reads")
    elif executor:
        g.edge(executor["step_id"], load, "data", "reads")
    elif trainer:
        g.edge(trainer["step_id"], load, "data", "reads")
    # Writing the output file is plumbing, not an answer to the prompt: when the
    # executor produced a result, the write folds into the result node.
    writes = [o for o in ops if o.get("op") == "write"]
    for i, op in enumerate(ops):
        if op.get("op") == "load" or (op.get("op") == "write" and executor):
            continue
        nid = g.node(f"op:{i}", "operation", _op_label(op), lane="data", formula=op.get("formula"),
                     op=op.get("op"), line=op.get("line"), columns=op.get("columns") or None,
                     source=source, step_id=ops_step["step_id"] if ops_step else None)
        g.edge(prev, nid, "data")
        prev = nid
    if executor:
        calc = executor.get("calculation") or {}
        table = calc.get("result")
        values = dict(calc.get("values") or {})
        if writes and writes[-1].get("target"):
            values["written_to"] = writes[-1]["target"]
        rid = g.node("result", "result", "Result", lane="data", step_id=executor["step_id"],
                     values=values or None, table=table)
        g.edge(prev, rid, "data")
        g.edge(executor["step_id"], rid, "data", "computes")
        prev = rid
    if trainer:
        calc = trainer.get("calculation") or {}
        trained = [m for m in (calc.get("models") or []) if isinstance(m, dict)]
        heads = []
        for i, model in enumerate(trained):
            cols = [c for c in [model.get("target"), *(model.get("features") or [])] if c]
            tid = g.node(f"train:{i}", "train", f"Train {model.get('name') or 'model'}", lane="data",
                         subtitle=model.get("type"), step_id=trainer["step_id"],
                         status="error" if model.get("status") == "failed" else "ok",
                         values={k: v for k, v in {"target": model.get("target"),
                                                   "features": len(model.get("features") or []) or None,
                                                   **(model.get("metrics") or {})}.items() if v is not None} or None,
                         columns=cols or None)
            g.edge(prev, tid, "data")
            g.edge(trainer["step_id"], tid, "data", "trains")
            heads.append(tid)
        ev = calc.get("evaluation") if isinstance(calc.get("evaluation"), dict) else None
        if ev or trained:
            primary = (ev or {}).get("primary_metric")
            metrics = dict((ev or {}).get("metrics") or {})
            baseline = dict((ev or {}).get("baseline") or {})
            verdict = (trainer.get("decision") or {}).get("value")
            values = {**{k: v for k, v in metrics.items()},
                      **({f"baseline {primary}": baseline[primary]} if primary in baseline else {})}
            eid = g.node("evaluate", "evaluate",
                         f"Evaluate · {primary}" if primary else "Evaluate", lane="data",
                         step_id=trainer["step_id"], value=verdict,
                         subtitle=(f"{primary} {metrics[primary]:.4g}" if primary in metrics else None),
                         values=values or None, metrics=None, evaluation=ev)
            for tid in heads or [prev]:
                g.edge(tid, eid, "data")
            prev = eid
    if verifier and prev not in (sample, load):
        g.edge(prev, verifier["step_id"], "evidence", "checked against")
