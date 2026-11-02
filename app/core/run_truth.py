"""Run truth: how a run ended, why, and what it covered, written by code.

Item 8a of docs/design/1.6-CAPABILITY-GAPS.md (decisions D8.1, D8.2). Every run
(interactive turn, deferred turn, scheduled run, Ray job, executor call) ends
in exactly one of ``succeeded``, ``failed`` or ``partial``, and its record
carries a ``reason_code`` from a closed list, a ``reason`` sentence from a
template, and a ``scope`` (rows processed of total, sample or full, engine).

Who writes it: the harness, from fields the executor and the graph's code set
(``execution_result.status``, ``execution_error``, the Ray scope fields). The
model never does. ``classify_*`` read no ``run_truth`` that arrives inside a
state, and every writer replaces whatever ``run_truth`` was already there.

Where it lives on 1.6 (D8.2, no migration): ``scheduled_runs.result.run_truth``,
the pending-turn record (``pending_turn.outcome`` plus ``result.run_truth``)
and ``ChatResponse.run_truth``. The legacy ``scheduled_runs.status`` column
keeps its ``success``/``failure`` words for the UI (see ``legacy_status``).

Scope fields from the Ray-honesty hotfix are read when present:
``execution_result.data_scope`` ({rows_used, rows_total, sample}),
``execution_result.ray_fallback`` ({reason, rows_used}) and
``execution_result.refused_reason``. Without them the scope says ``unknown``
rather than guessing.

Out of scope here: heartbeats and the stall watchdog (8b), per-job pods (8c).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

SUCCEEDED = "succeeded"
FAILED = "failed"
PARTIAL = "partial"
OUTCOMES = (SUCCEEDED, FAILED, PARTIAL)

# The closed reason list. ``partial`` means the run produced a result but
# covered less than it was asked to: a sample where the whole file was meant
# (``sampled``), a different engine than the one routed to (``fallback``), or
# only some of the requested metrics (``partial_metrics``).
REASON_TEMPLATES: Dict[str, str] = {
    "ok": "The run finished and its result was recorded.",
    "reply_only": "This turn replied without running an analysis.",
    "exception": "The run failed: {detail}",
    "timeout": "The run was stopped at its time limit: {detail}",
    "worker_lost": "The worker running this job died before it finished: {detail}",
    "nothing_ran": "Nothing was executed: {detail}",
    "refused": "The run was refused: {detail}",
    # MAJ-030: the run produced a table, but it did not hold what the question
    # asked for (app/agents/output_contract.py), so it was not shown.
    "output_contract": "The result did not answer the question, so it was not shown: {detail}",
    "partial_metrics": "Some requested results could not be computed: {detail}",
    "sampled": "The run covered a sample, not the whole dataset: {detail}",
    "fallback": "The run did not use the engine it was routed to: {detail}",
}
REASON_CODES = frozenset(REASON_TEMPLATES)
OUTCOME_OF_REASON = {
    "ok": SUCCEEDED, "reply_only": SUCCEEDED,
    "exception": FAILED, "timeout": FAILED, "worker_lost": FAILED,
    "nothing_ran": FAILED, "refused": FAILED, "output_contract": FAILED,
    "partial_metrics": PARTIAL, "sampled": PARTIAL, "fallback": PARTIAL,
}
ENGINES = ("local", "ray", "executor", "warehouse", "starrocks")
ROWS_TOTAL_KINDS = ("exact", "estimate", "unknown")
RUN_KINDS = ("interactive", "deferred", "scheduled")

WRITTEN_BY = "harness"
_SUCCESS_WORDS = ("success", "succeeded", "completed", "done")


@dataclass(frozen=True)
class RunOutcome:
    outcome: str
    reason_code: str
    reason: str
    kind: str
    scope: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reason_code not in REASON_CODES:
            raise ValueError(f"unknown reason_code {self.reason_code!r}")
        if self.outcome != OUTCOME_OF_REASON[self.reason_code]:
            raise ValueError(f"{self.reason_code!r} cannot end as {self.outcome!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "kind": self.kind,
            "scope": dict(self.scope),
            "written_by": WRITTEN_BY,
        }


def build_scope(
    *,
    rows_processed: Optional[int] = None,
    rows_total: Optional[int] = None,
    rows_total_kind: Optional[str] = None,
    sample: Optional[bool] = None,
    engine: Optional[str] = None,
    fallback: Optional[str] = None,
) -> Dict[str, Any]:
    """The scope block. ``None`` means not known, never zero."""
    if rows_total_kind is None:
        rows_total_kind = "exact" if rows_total is not None else "unknown"
    if rows_total_kind not in ROWS_TOTAL_KINDS:
        raise ValueError(f"unknown rows_total_kind {rows_total_kind!r}")
    if engine is not None and engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}")
    return {
        "rows_processed": _int_or_none(rows_processed),
        "rows_total": _int_or_none(rows_total),
        "rows_total_kind": rows_total_kind,
        "sample": sample,
        "engine": engine,
        "fallback": fallback,
    }


def make(reason_code: str, *, kind: str, scope: Optional[Mapping[str, Any]] = None,
         detail: Optional[str] = None) -> RunOutcome:
    detail = " ".join(str(detail or "").split())[:500] or "no detail was reported"
    return RunOutcome(
        outcome=OUTCOME_OF_REASON[reason_code],
        reason_code=reason_code,
        reason=REASON_TEMPLATES[reason_code].format(detail=detail),
        kind=kind,
        scope=dict(scope) if scope else build_scope(),
    )


def failed(reason_code: str, *, kind: str, detail: Optional[str] = None,
           scope: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """A terminal ``failed`` record for a run that never produced a final state."""
    return make(reason_code, kind=kind, scope=scope, detail=detail).to_dict()


def legacy_status(outcome: str) -> str:
    """The word for ``scheduled_runs.status``, which the UI already reads.

    A partial run produced its output, as it did before this contract, so it
    keeps ``success`` there; ``run_truth`` says it was partial and why.
    """
    return "failure" if outcome == FAILED else "success"


# ---------------------------------------------------------------------------
# Classifiers. They read only code-written fields and ignore any run_truth
# already present in the state.
# ---------------------------------------------------------------------------

def _int_or_none(v: Any) -> Optional[int]:
    try:
        return None if v is None or isinstance(v, bool) else int(v)
    except (TypeError, ValueError):
        return None


def _engine_of(final: Mapping[str, Any], er: Mapping[str, Any]) -> Optional[str]:
    if er.get("ray_fallback"):
        return "local"
    mode = str(er.get("mode") or final.get("execution_mode") or "").lower()
    if mode == "k8s-ray" or "ray" in mode:
        return "ray"
    if mode in ("k8s", "scheduled_pod", "executor") or final.get("deploy_on_k8s"):
        return "executor"
    if er:
        return "local"
    return None


def _execution_scope(final: Mapping[str, Any], er: Mapping[str, Any],
                     defaults: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Scope from the executor's own fields, else from ``defaults`` (server-computed)."""
    d = dict(defaults or {})
    engine = d.pop("engine", None) or _engine_of(final, er)
    fb = er.get("ray_fallback") if isinstance(er.get("ray_fallback"), Mapping) else None
    ds = er.get("data_scope") if isinstance(er.get("data_scope"), Mapping) else None
    if ds:
        rows_total = _int_or_none(ds.get("rows_total"))
        return build_scope(
            rows_processed=ds.get("rows_used"), rows_total=rows_total,
            rows_total_kind="exact" if rows_total is not None else "unknown",
            sample=bool(ds.get("sample")), engine=engine,
            fallback=(fb or {}).get("reason"),
        )
    if fb:
        rows = _int_or_none(fb.get("rows_used"))
        return build_scope(rows_processed=rows, rows_total=rows,
                           sample=False if rows is not None else None,
                           engine="local", fallback=fb.get("reason"))
    return build_scope(engine=engine, **{k: d.get(k) for k in (
        "rows_processed", "rows_total", "rows_total_kind", "sample") if k in d})


def _error_signal(final: Mapping[str, Any], er: Mapping[str, Any]) -> Optional[tuple]:
    """A failure the executor or a node recorded, as (reason_code, detail)."""
    st = str(er.get("status") or "").lower()
    oc = er.get("output_contract") if isinstance(er.get("output_contract"), Mapping) else None
    if oc and oc.get("status") == "refused":
        # Checked first: the executor also marks the refusal as a non-success
        # status, which would otherwise read as an exception.
        details = [str((v or {}).get("detail") or (v or {}).get("rule") or "")
                   for v in oc.get("violations") or [] if isinstance(v, Mapping)]
        return ("output_contract", "; ".join(d for d in details if d) or "the output contract refused it")
    if er.get("refused_reason"):
        return ("refused", str(er.get("message") or er.get("refused_reason")))
    if st == "timeout":
        return ("timeout", str(er.get("message") or er.get("error") or "the execution engine timed out"))
    if st and st not in _SUCCESS_WORDS and st != "dryrun":
        return ("exception", str(er.get("error") or er.get("message") or f"execution ended with status {st}"))
    for k in ("execution_error", "configure_inference_error", "stop_inference_error",
              "inference_error", "training_error", "error"):
        if final.get(k):
            return ("exception", str(final.get(k)))
    return None


def classify_scheduled(final: Any, *, task_type: str = "execute",
                       training_failure: Optional[str] = None) -> Dict[str, Any]:
    """Run truth for a scheduled run's final state (``scheduled_runs.result.run_truth``).

    ``training_failure`` is the message the caller already derived for a failed
    scheduled training (``_run_status_from_final``), so the two cannot disagree.
    """
    kind = "scheduled"
    if not isinstance(final, Mapping):
        return failed("nothing_ran", kind=kind, detail="the task returned no final state")
    er = final.get("execution_result") if isinstance(final.get("execution_result"), Mapping) else {}
    defaults = None
    if task_type == "execute" and _engine_of(final, er) not in ("ray", None) and not er.get("ray_fallback"):
        # A scheduled local analysis runs in the shared executor pod
        # (execution_agent_node_scheduled_pod) or a k8s job, not in the worker.
        defaults = {"engine": "executor"}
    scope = _execution_scope(final, er, defaults)
    if training_failure:
        return failed("exception", kind=kind, detail=training_failure, scope=scope)
    sig = _error_signal(final, er)
    if sig:
        return failed(sig[0], kind=kind, detail=sig[1], scope=scope)
    if task_type == "execute":
        st = str(er.get("status") or "").lower()
        if st not in _SUCCESS_WORDS:
            return failed("nothing_ran", kind=kind, scope=scope,
                          detail="the scheduled analysis produced no execution result")
    return _success_or_partial(er, kind=kind, scope=scope).to_dict()


def _success_or_partial(er: Mapping[str, Any], *, kind: str, scope: Mapping[str, Any],
                        missing: Optional[list] = None, entire_on_head: bool = False) -> RunOutcome:
    fb = er.get("ray_fallback") if isinstance(er.get("ray_fallback"), Mapping) else None
    ds = er.get("data_scope") if isinstance(er.get("data_scope"), Mapping) else None
    if missing:
        labels = "; ".join(str((m or {}).get("label") or (m or {}).get("id") or "metric") for m in missing)
        return make("partial_metrics", kind=kind, scope=scope, detail=labels)
    if fb:
        return make("fallback", kind=kind, scope=scope,
                    detail=f"it ran locally, not on Ray, because {fb.get('reason') or 'Ray was unavailable'}")
    if ds and ds.get("sample"):
        return make("sampled", kind=kind, scope=scope, detail=_rows_phrase(scope))
    if entire_on_head:
        return make("sampled", kind=kind, scope=scope,
                    detail="the entire dataset was asked for, but only the head of the file was read")
    return make("ok", kind=kind, scope=scope)


def _rows_phrase(scope: Mapping[str, Any]) -> str:
    n, t = scope.get("rows_processed"), scope.get("rows_total")
    if n is not None and t is not None:
        return f"{n:,} of {t:,} rows were processed"
    if n is not None:
        return f"{n:,} rows were processed; the total was not reported"
    return "the number of rows processed was not reported"


def classify_turn(final: Any, *, deferred: bool = False,
                  scope_defaults: Optional[Mapping[str, Any]] = None,
                  entire_on_head: bool = False) -> Dict[str, Any]:
    """Run truth for a chat turn's final graph state.

    A turn that ran nothing and reported no error is ``succeeded/reply_only``
    with an empty scope (a question answered, a plan shown, a job scheduled):
    unlike a scheduled analysis, a turn is not obliged to execute anything.
    """
    kind = "deferred" if deferred else "interactive"
    if not isinstance(final, Mapping):
        return failed("nothing_ran", kind=kind, detail="the graph returned no final state")
    er = final.get("execution_result") if isinstance(final.get("execution_result"), Mapping) else {}
    tr = final.get("training_result") if isinstance(final.get("training_result"), Mapping) else {}
    scope = _execution_scope(final, er, scope_defaults) if er else build_scope()
    sig = _error_signal(final, er)
    if not sig and (tr.get("failure") or tr.get("error") or tr.get("execution_error")
                    or str(tr.get("status") or "").lower() == "error"):
        sig = ("exception", str(tr.get("error") or tr.get("execution_error") or "model training failed"))
    if sig:
        return failed(sig[0], kind=kind, detail=sig[1], scope=scope)
    st = str(er.get("status") or "").lower()
    if st in _SUCCESS_WORDS:
        missing = final.get("multi_action_missing") if final.get("structured_action_plan") else None
        return _success_or_partial(er, kind=kind, scope=scope, missing=missing,
                                   entire_on_head=entire_on_head).to_dict()
    if final.get("training_completed") and (tr.get("mlflow_run_id") or final.get("mlflow_run_id")):
        return make("ok", kind=kind, scope=build_scope(
            engine="ray" if tr.get("ray_job_id") else None)).to_dict()
    return make("reply_only", kind=kind, scope=build_scope()).to_dict()
