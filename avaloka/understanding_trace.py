"""What each agent understood about the data and decided to do, in two layers.

The swarm trace (:mod:`avaloka.swarm_trace`) says *which* firefly ran and
what it cost. This module says what the firefly concluded while it ran: the
shape of the data it looked at, whether it sampled and why, which operations
it chose, which checks it ran and how they came out, and what the model calls
behind it cost in tokens. One record per completed agent step
(:data:`AGENT_RECORD`), one per mission (:data:`SWARM_RECORD`), appended to
the same recorder stream as the swarm trace so they share its ``seq``.

There is one schema and two layers of it:

* The **local log** -- ``<output_dir>/understanding.jsonl`` and the swarm
  trace itself -- is the customer's own record of their own fleet at work.
  It may name columns, files, formulas and the reasons an agent gave, because
  that is what the CLI's ``--debug`` narration and the UI read back to them.
* The **egress projection** (:func:`egress_event`) is what telemetry would
  send. It is an allowlist, never a scrub: every field is a closed
  vocabulary, a bucket or a bounded count, so there is no field a column
  name, a file name, a value, a goal or a prompt could travel in. The key set
  is closed and :mod:`telemetry_service.schema` rejects anything outside it;
  ``tests/test_understanding_trace_privacy.py`` plants all five and checks.

Buckets, not sizes: ``row_count_bucket`` rather than ``rows``, a null-rate
histogram rather than per-column missingness, a cardinality histogram rather
than ``n_unique`` per column. A histogram keyed by a vocabulary cannot carry
a column name; a per-column list can.

This module imports nothing outside the standard library (S1.08), like the
swarm trace it extends: the standalone CLI writes the same records the
server serves.
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA_VERSION = "1.0"
AGENT_RECORD = "agent.understanding"
SWARM_RECORD = "swarm.understanding"
RECORD_TYPES = frozenset({AGENT_RECORD, SWARM_RECORD})
LOG_FILE = "understanding.jsonl"
EGRESS_FILE = "understanding_egress.json"

# --------------------------------------------------------------- vocabularies
#: What an agent is *for*. A firefly or a graph node maps onto exactly one.
ROLES = ("understanding", "sampling", "feature_engineering", "planning", "coding",
         "modeling", "execution", "verification", "finops", "deployment", "reporting", "other")

#: agent name -> role, for the CLI fireflies and the server graph's nodes.
ROLE_OF: dict[str, str] = {
    # CLI fireflies (avaloka.fireflies)
    "data_scout": "understanding",
    "sampling_specialist": "sampling",
    "workload_optimizer": "planning",
    "data_engineer": "feature_engineering",
    "feature_engineer": "feature_engineering",
    "analysis_planner": "planning",
    "model_scientist": "modeling",
    "statistician": "modeling",
    "experimenter": "modeling",
    "conformal_calibrator": "verification",
    "validator": "verification",
    "finops": "finops",
    "ml_engineer": "deployment",
    "reporter": "reporting",
    # the server graph (app/api/workflow.py) and its coding sub-graph
    "drivers_route": "planning",
    "memory_injection": "understanding",
    "avaloka_agent": "understanding",
    "profile_data": "understanding",
    "plan_etl": "planning",
    "summarize_etl": "planning",
    "generate_planner_graph": "planning",
    "code_etl": "coding",
    "coder": "coding",
    "validator_syntax": "verification",
    "validator_static": "verification",
    "validator_contract": "verification",
    "validator_logical": "verification",
    "verify_claims": "verification",
    "execute_code": "execution",
    "execute_actions": "execution",
    "execute_locally": "execution",
    "execute_on_ray": "execution",
    "execute_on_k8s": "execution",
    "replan": "execution",
    "train_models": "modeling",
    "visualize": "reporting",
    "narrate_result": "reporting",
    "provision_infra": "deployment",
    "schedule_task": "deployment",
}
#: Every agent the projection may name. Must equal the ingest service's
#: AGENTS (telemetry_service/schema.py); tests/test_understanding_trace.py
#: holds the two together.
KNOWN_AGENTS = frozenset(ROLE_OF)

COLUMN_KINDS = ("numeric", "text", "boolean", "datetime", "categorical", "id", "constant", "other")
NULL_RATE_BUCKETS = ("0", "0-1%", "1-10%", "10-50%", "50%+")
CARDINALITY_BUCKETS = ("1", "2", "3-10", "11-100", "101-1000", "1000+")
ROW_BUCKETS = ("0-1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6", "1e6-1e7", "1e7-1e8", "1e8+")
COLUMN_BUCKETS = ("1-5", "5-10", "10-50", "50-200", "200+")
SIZE_BUCKETS = ("<1MB", "1-10MB", "10-100MB", "100MB-1GB", "1GB+")
FRACTION_BUCKETS = ("full", "50-100%", "10-50%", "1-10%", "<1%")
FILE_FORMATS = ("csv", "tsv", "xlsx", "parquet", "json", "other")
SAMPLING_STRATEGIES = ("none", "random", "stratified", "other")
#: Kinds of data operation: the Data Engineer's transformations and the
#: operations :mod:`app.trace.describe` extracts from executed code. Anything
#: else is counted under "other"; the name itself never travels.
OP_KINDS = ("drop_constant_columns", "drop_duplicate_rows", "strip_whitespace",
            "flag_high_missingness", "load", "filter", "derive", "cast", "aggregate", "join",
            "concat", "sort", "sample", "write", "clean", "other")
VERDICTS = ("pass", "warn", "fail")
OUTCOMES = ("ok", "failed")
AGREEMENTS = ("unanimous", "split")
LLM_PROVIDERS = ("groq", "openrouter", "local", "openai", "bedrock", "vertex", "azure")
#: The models the product ships as defaults; anything else is "other" on the
#: wire, because a customer's deployment name can carry their name.
LLM_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")

MAX_COUNT = 1_000_000
MAX_DURATION_MS = 7 * 24 * 3600 * 1000

_SESSION_ID = uuid.uuid4().hex


# ------------------------------------------------------------------ buckets
def row_bucket(rows: Any) -> str:
    n = _int(rows)
    for label, hi in zip(ROW_BUCKETS, (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8)):
        if n < hi:
            return label
    return ROW_BUCKETS[-1]


def column_bucket(columns: Any) -> str:
    n = _int(columns)
    for label, hi in zip(COLUMN_BUCKETS, (5, 10, 50, 200)):
        if n < hi:
            return label
    return COLUMN_BUCKETS[-1]


def null_rate_bucket(fraction: Any) -> str:
    f = _float(fraction)
    if f <= 0.0:
        return "0"
    if f < 0.01:
        return "0-1%"
    if f < 0.10:
        return "1-10%"
    if f < 0.50:
        return "10-50%"
    return "50%+"


def cardinality_bucket(n_unique: Any) -> str:
    n = _int(n_unique)
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 10:
        return "3-10"
    if n <= 100:
        return "11-100"
    if n <= 1000:
        return "101-1000"
    return "1000+"


def size_bucket(size_bytes: Any) -> Optional[str]:
    if size_bytes is None:
        return None
    n = _int(size_bytes)
    mb = 1024 * 1024
    for label, hi in zip(SIZE_BUCKETS, (mb, 10 * mb, 100 * mb, 1024 * mb)):
        if n < hi:
            return label
    return SIZE_BUCKETS[-1]


def fraction_bucket(fraction: Any) -> str:
    f = _float(fraction)
    if f >= 1.0:
        return "full"
    if f >= 0.5:
        return "50-100%"
    if f >= 0.1:
        return "10-50%"
    if f >= 0.01:
        return "1-10%"
    return "<1%"


def _int(value: Any, default: int = 0) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n


def _float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _vocab(value: Any, vocabulary: tuple, other: Optional[str] = "other") -> Optional[str]:
    if isinstance(value, str) and value in vocabulary:
        return value
    return other


def _counts(items: Any, vocabulary: tuple, key: str = "op") -> dict[str, int]:
    """Histogram of ``items`` by ``key`` over ``vocabulary``; unknowns are "other"."""
    out: dict[str, int] = {}
    for item in items or []:
        value = item.get(key) if isinstance(item, dict) else item
        label = _vocab(value, vocabulary) or "other"
        out[label] = out.get(label, 0) + 1
    return out


def _file_format(fmt: Any) -> Optional[str]:
    if not isinstance(fmt, str):
        return None
    fmt = fmt.lower().lstrip(".")
    return fmt if fmt in FILE_FORMATS else "other"


def _shape_of(frame: Any) -> tuple[Optional[int], Optional[int]]:
    """``(rows, columns)`` of anything with a ``shape``; stdlib, so duck-typed."""
    shape = getattr(frame, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 1:
        rows = _int(shape[0], -1)
        cols = _int(shape[1], -1) if len(shape) > 1 else None
        return (rows if rows >= 0 else None), (cols if cols is not None and cols >= 0 else None)
    return None, None


# ---------------------------------------------------------------- components
@dataclass
class DataShape:
    """The shape of the data an agent looked at, as counts and histograms.

    ``column_names``, ``issues`` and the exact ``rows``/``columns`` stay in the
    local layer; the wire carries the buckets and the histograms only.
    """

    rows: Optional[int] = None
    columns: Optional[int] = None
    column_type_counts: dict[str, int] = field(default_factory=dict)
    null_rate_counts: dict[str, int] = field(default_factory=dict)
    cardinality_counts: dict[str, int] = field(default_factory=dict)
    file_format: Optional[str] = None
    size_bytes: Optional[int] = None
    quality_score: Optional[int] = None
    column_names: list[str] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: Any, *, size_bytes: Optional[int] = None) -> "DataShape":
        """From the Data Scout's profile (``avaloka.stats.profile_column`` per column)."""
        if not isinstance(profile, dict):
            return cls(size_bytes=size_bytes)
        dataset = profile.get("dataset") if isinstance(profile.get("dataset"), dict) else {}
        columns = [c for c in (profile.get("columns") or []) if isinstance(c, dict)]
        quality = profile.get("quality") if isinstance(profile.get("quality"), dict) else {}
        score = quality.get("score")
        if size_bytes is None:
            # The statistics contract's ``dataset.structure.source`` carries
            # byte counts for every format (counts only, never names).
            structure = dataset.get("structure") if isinstance(dataset.get("structure"), dict) else {}
            source = structure.get("source") if isinstance(structure.get("source"), dict) else {}
            if source.get("total_bytes") is not None:
                size_bytes = _int(source.get("total_bytes"))
        return cls(
            rows=_int(dataset.get("n_rows"), 0) if dataset.get("n_rows") is not None else None,
            columns=len(columns) or (_int(dataset.get("n_cols")) if dataset.get("n_cols") else None),
            column_type_counts=_counts(columns, COLUMN_KINDS, key="role"),
            null_rate_counts=_counts([null_rate_bucket(c.get("missing_pct")) for c in columns],
                                     NULL_RATE_BUCKETS),
            cardinality_counts=_counts([cardinality_bucket(c.get("n_unique")) for c in columns],
                                       CARDINALITY_BUCKETS),
            file_format=_file_format(dataset.get("format")),
            size_bytes=size_bytes,
            quality_score=max(0, min(100, _int(score))) if score is not None else None,
            column_names=[str(c.get("name")) for c in columns],
            issues=[dict(i) for i in (quality.get("issues") or []) if isinstance(i, dict)],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "row_count_bucket": row_bucket(self.rows) if self.rows is not None else None,
            "column_count_bucket": column_bucket(self.columns) if self.columns is not None else None,
            "column_type_counts": dict(self.column_type_counts),
            "null_rate_counts": dict(self.null_rate_counts),
            "cardinality_counts": dict(self.cardinality_counts),
            "file_format": self.file_format,
            "size_bytes": self.size_bytes,
            "size_bucket": size_bucket(self.size_bytes),
            "quality_score": self.quality_score,
            "column_names": list(self.column_names),
            "issues": list(self.issues),
        }


@dataclass
class SamplingDecision:
    sampled: bool = False
    strategy: str = "none"
    input_rows: Optional[int] = None
    working_rows: Optional[int] = None
    reason: str = ""

    @classmethod
    def from_blackboard(cls, sampling: Any) -> Optional["SamplingDecision"]:
        if not isinstance(sampling, dict):
            return None
        return cls(
            sampled=bool(sampling.get("sampled")),
            strategy=str(sampling.get("strategy") or "none"),
            input_rows=_int(sampling["n_input_rows"]) if sampling.get("n_input_rows") is not None else None,
            working_rows=(_int(sampling["n_working_rows"])
                          if sampling.get("n_working_rows") is not None else None),
            reason=str(sampling.get("reason") or ""),
        )

    @property
    def fraction(self) -> Optional[float]:
        if not self.input_rows or self.working_rows is None:
            return None if self.sampled else 1.0
        return round(min(1.0, self.working_rows / self.input_rows), 6)

    def as_dict(self) -> dict[str, Any]:
        frac = self.fraction
        return {
            "sampled": self.sampled,
            "strategy": self.strategy,
            "fraction": frac,
            "fraction_bucket": fraction_bucket(frac) if frac is not None else None,
            "input_rows": self.input_rows,
            "working_rows": self.working_rows,
            "reason": self.reason,
        }


@dataclass
class OperationSummary:
    """The operations an agent chose. ``detail`` (columns, formulas) is local only."""

    detail: list[dict[str, Any]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.detail)

    @property
    def op_counts(self) -> dict[str, int]:
        return _counts(self.detail, OP_KINDS, key="op")

    def as_dict(self) -> dict[str, Any]:
        return {"count": self.count, "op_counts": self.op_counts, "detail": list(self.detail)}


@dataclass
class VerificationSummary:
    """Checks run and how they came out. ``checks`` (with their detail text) is local only."""

    checks: list[dict[str, Any]] = field(default_factory=list)
    retries: int = 0
    verdict: Optional[str] = None

    def _n(self, status: str) -> int:
        return sum(1 for c in self.checks if isinstance(c, dict) and c.get("status") == status)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks_run": len(self.checks),
            "passed": self._n("pass"),
            "warned": self._n("warn"),
            "failed": self._n("fail"),
            "retries": self.retries,
            "verdict": self.verdict,
            "checks": list(self.checks),
        }


@dataclass
class ModelUsage:
    provider: Optional[str] = None
    model_id: Optional[str] = None
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @classmethod
    def from_metadata(cls, meta: Any) -> Optional["ModelUsage"]:
        """From a firefly's unit metadata (``llm_provider``, ``llm_model``, ``tokens_*``)."""
        if not isinstance(meta, dict) or not any(k in meta for k in ("llm_provider", "llm_model",
                                                                      "tokens_in", "tokens_out")):
            return None
        return cls(provider=meta.get("llm_provider"), model_id=meta.get("llm_model"),
                   calls=_int(meta.get("llm_calls"), 1 if meta.get("llm_model") else 0),
                   tokens_in=_int(meta.get("tokens_in")), tokens_out=_int(meta.get("tokens_out")))

    @classmethod
    def from_inference(cls, calls: Any) -> Optional["ModelUsage"]:
        """From an app-trace step's ``inference`` list (one entry per model call)."""
        calls = [c for c in (calls or []) if isinstance(c, dict)]
        if not calls:
            return None
        return cls(
            provider=next((c.get("provider") for c in calls if c.get("provider")), None),
            model_id=next((c.get("model") for c in calls if c.get("model")), None),
            calls=len(calls),
            tokens_in=sum(_int(c.get("tokens_in")) for c in calls),
            tokens_out=sum(_int(c.get("tokens_out")) for c in calls),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "model_id": self.model_id, "calls": self.calls,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}


# ------------------------------------------------------------------ records
@dataclass
class AgentUnderstanding:
    """What one agent step concluded and decided. The body of an ``agent.understanding`` record."""

    agent: str
    role: str = "other"
    clone_id: Optional[str] = None
    clone_name: Optional[str] = None
    clone_index: Optional[int] = None
    step_id: Optional[str] = None
    step_no: Optional[int] = None
    human_role: Optional[str] = None
    outcome: str = "ok"
    error_class: Optional[str] = None
    concluded: Optional[str] = None
    decided: dict[str, Any] = field(default_factory=dict)
    data: Optional[DataShape] = None
    sampling: Optional[SamplingDecision] = None
    operations: Optional[OperationSummary] = None
    verification: Optional[VerificationSummary] = None
    model: Optional[ModelUsage] = None
    timing: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "clone_id": self.clone_id, "clone_name": self.clone_name, "clone_index": self.clone_index,
            "step_id": self.step_id, "step_no": self.step_no,
            "agent": self.agent, "role": self.role, "human_role": self.human_role,
            "outcome": self.outcome, "error_class": self.error_class,
            "concluded": self.concluded,
            "decided": dict(self.decided),
            "data": self.data.as_dict() if self.data else None,
            "sampling": self.sampling.as_dict() if self.sampling else None,
            "operations": self.operations.as_dict() if self.operations else None,
            "verification": self.verification.as_dict() if self.verification else None,
            "model": self.model.as_dict() if self.model else None,
            "timing": dict(self.timing),
        }

    # -- builders ----------------------------------------------------------
    @classmethod
    def from_firefly(cls, *, step: dict[str, Any], unit: Any, blackboard: dict[str, Any],
                     clone_id: Optional[str] = None, clone_name: Optional[str] = None,
                     clone_index: Optional[int] = None, data_source: Optional[str] = None,
                     outcome: str = "ok", error_class: Optional[str] = None) -> "AgentUnderstanding":
        """After a firefly step: ``step`` is the trace's step dict (``step_id``,
        ``step_no``, ``firefly``), ``unit`` the WorkUnit it recorded (``None``
        when it raised), ``blackboard`` the clone's mission blackboard."""
        agent = str(step.get("firefly") or getattr(unit, "firefly", None) or "unknown")
        meta = dict(getattr(unit, "metadata", {}) or {}) if unit is not None else {}
        size = None
        if data_source:
            try:
                size = os.path.getsize(data_source)
            except OSError:
                size = None
        data = DataShape.from_profile(blackboard.get("profile"), size_bytes=size)
        if blackboard.get("profile") is None and data_source is None:
            data = None
        rec = cls(
            agent=agent, role=ROLE_OF.get(agent, "other"),
            clone_id=clone_id, clone_name=clone_name, clone_index=clone_index,
            step_id=step.get("step_id"), step_no=step.get("step_no"),
            human_role=getattr(unit, "human_role", None) if unit is not None else step.get("human_role"),
            outcome=outcome, error_class=error_class,
            concluded=getattr(unit, "output", None) if unit is not None else None,
            data=data,
            model=ModelUsage.from_metadata(meta),
            timing={
                "duration_ms": int(round(_float(getattr(unit, "duration_seconds", 0.0)) * 1000))
                if unit is not None else None,
                "manual_minutes": _float(getattr(unit, "manual_minutes", 0.0)) if unit is not None else None,
                "compute_cost_usd": _float(getattr(unit, "compute_cost_usd", 0.0)) if unit is not None else None,
                "model_cost_usd": _float(getattr(unit, "model_cost_usd", 0.0)) if unit is not None else None,
            },
        )
        rec._read_blackboard(blackboard, meta)
        return rec

    def _read_blackboard(self, bb: dict[str, Any], meta: dict[str, Any]) -> None:
        """Fill ``decided``/``sampling``/``operations``/``verification`` from
        what this firefly left on the blackboard. Each reads only what its own
        firefly writes, so a later step does not re-report an earlier one's."""
        role = self.role
        if self.agent == "feature_engineer":
            # Shares the Data Engineer's role but writes its own keys: the
            # recipe it applied (steps, inputs, outputs) and the frame it left.
            result = bb.get("feature_result")
            recipe = getattr(result, "recipe", None)
            if recipe is None:
                recipe = bb.get("feature_recipe")
            steps = getattr(recipe, "steps", None)
            if steps is None and isinstance(recipe, dict):
                steps = recipe.get("steps")
            steps = list(steps or [])
            ops = []
            for st in steps:
                op = getattr(st, "operation", None) or (st.get("operation") if isinstance(st, dict) else None)
                inputs = getattr(st, "inputs", None) or (st.get("inputs") if isinstance(st, dict) else ())
                output = getattr(st, "output", None) or (st.get("output") if isinstance(st, dict) else None)
                ops.append({"op": str(op), "columns": [str(c) for c in (inputs or ())] + ([str(output)] if output else [])})
            self.operations = OperationSummary(detail=ops)
            self.decided = {"n_steps": len(ops), "operations": [o["op"] for o in ops],
                            "recipe_hash": getattr(result, "recipe_hash", None) or meta.get("recipe_hash"),
                            "diagnostics": [str(d) for d in (getattr(result, "diagnostics", None) or [])]}
            # ``is None``, never ``or``: a DataFrame's truth value raises.
            frame = getattr(result, "frame", None)
            if frame is None:
                frame = bb.get("transformed_frame")
            rows, cols = _shape_of(frame)
            if self.data is not None and rows is not None:
                self.data.rows, self.data.columns = rows, cols
            return
        if self.agent == "workload_optimizer":
            plan = bb.get("workload_optimization") if isinstance(bb.get("workload_optimization"), dict) else {}
            if plan:
                self.decided = {
                    "strategy": plan.get("strategy"),
                    "n_filter_factors": len(plan.get("filter_factors") or []),
                    "n_pushdown_filters": len(plan.get("pushdown_filters") or []),
                    "n_projection_columns": len(plan.get("projection_columns") or []),
                    "n_indexed_columns": len(plan.get("indexed_columns") or []),
                    "n_partition_columns": len(plan.get("partition_columns") or []),
                    "estimated_rows_after_filters": plan.get("estimated_rows_after_filters"),
                    "predicate_order": list(plan.get("predicate_order") or []),
                }
            return
        if role == "understanding":
            profile = bb.get("profile") if isinstance(bb.get("profile"), dict) else {}
            quality = profile.get("quality")
            if isinstance(quality, dict):
                self.decided = {
                    "quality_score": quality.get("score"),
                    "n_issues": len(quality.get("issues") or []),
                    "issue_kinds": _counts(quality.get("issues") or [], tuple(
                        sorted({str(i.get("kind")) for i in quality.get("issues") or [] if isinstance(i, dict)})
                    ), key="kind"),
                }
            domain = profile.get("domain")
            if isinstance(domain, dict) and domain.get("domain"):
                # Read from the column names (avaloka.domains), so it is schema:
                # local layer only. egress_event never reads ``decided``, and
                # tests/test_domain_vocabulary_telemetry_privacy.py checks it.
                self.decided["domain"] = domain["domain"]
                self.decided["domain_confidence"] = domain.get("confidence")
        elif role == "sampling":
            self.sampling = SamplingDecision.from_blackboard(bb.get("sampling"))
            if self.sampling is not None:
                self.decided = {"sampled": self.sampling.sampled, "strategy": self.sampling.strategy,
                                "reason": self.sampling.reason}
                if self.data is not None and self.sampling.working_rows is not None:
                    self.data.rows = self.sampling.working_rows
        elif role == "feature_engineering":
            ops = [dict(t) for t in (bb.get("transformations") or []) if isinstance(t, dict)]
            self.operations = OperationSummary(detail=ops)
            self.decided = {"n_transformations": len(ops),
                            "applied": [o["op"] for o in ops if o.get("applied", True)],
                            "deferred": [o["op"] for o in ops if not o.get("applied", True)]}
            rows, cols = _shape_of(bb.get("transformed_frame"))
            if self.data is not None and rows is not None:
                self.data.rows, self.data.columns = rows, cols
        elif role == "planning":
            plan = bb.get("plan") if isinstance(bb.get("plan"), dict) else {}
            split = plan.get("split") if isinstance(plan.get("split"), dict) else {}
            self.decided = {
                "task": plan.get("task"), "task_reason": plan.get("task_reason"),
                "split": split.get("strategy"), "split_reason": split.get("reason"),
                "candidate_algorithms": list(plan.get("candidate_algorithms") or []),
                "human_approval_gates": list(plan.get("human_approval_gates") or []),
                "data_sufficiency": plan.get("data_sufficiency"),
            }
            goal = bb.get("goal_analysis")
            if isinstance(goal, dict):
                self.decided["goal_analysis"] = {k: goal.get(k) for k in ("kind", "status", "scope", "findings")}
        elif role == "modeling":
            ts = bb.get("training_summary") if isinstance(bb.get("training_summary"), dict) else {}
            if ts:
                metric = ts.get("metric")
                self.decided = {
                    "selected": ts.get("selected"), "metric": metric,
                    "score": (ts.get("selected_metrics") or {}).get(metric) if metric else None,
                    "baseline_score": ts.get("baseline_score"),
                    "baseline_strategy": ts.get("baseline_strategy"),
                    "beats_baseline": ts.get("beats_baseline"),
                    "n_candidates": len(ts.get("candidates") or []),
                    "n_train": ts.get("n_train"), "n_test": ts.get("n_test"),
                    "n_features": len(ts.get("feature_columns") or []),
                }
        elif role == "verification":
            report = bb.get("validation") if isinstance(bb.get("validation"), dict) else {}
            if report:
                self.verification = VerificationSummary(
                    checks=[dict(c) for c in (report.get("checks") or []) if isinstance(c, dict)],
                    verdict=_vocab(report.get("verdict"), VERDICTS, None))
                self.decided = {"verdict": report.get("verdict"),
                                "max_safe_deployment_level": report.get("max_safe_deployment_level"),
                                "blocking_issues": list(report.get("blocking_issues") or []),
                                "leakage_findings": len(report.get("leakage") or [])}
        elif role == "finops":
            fin = bb.get("finops") if isinstance(bb.get("finops"), dict) else {}
            if fin:
                self.decided = {"choice": fin.get("choice"), "rationale": fin.get("rationale")}
        if not self.decided and meta:
            self.decided = {k: v for k, v in meta.items()
                            if k not in ("llm_provider", "llm_model", "tokens_in", "tokens_out", "llm_calls")}

    @classmethod
    def from_app_step(cls, step: dict[str, Any]) -> Optional["AgentUnderstanding"]:
        """From one step of the server's agent trace (``app.trace.recorder``):
        the coder, the validators, the dry run, the planner. ``None`` for a
        step that is not an agent step."""
        if not isinstance(step, dict) or step.get("kind") != "agent":
            return None
        node = str(step.get("node") or "unknown")
        decision = step.get("decision") if isinstance(step.get("decision"), dict) else {}
        calc = step.get("calculation") if isinstance(step.get("calculation"), dict) else {}
        inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
        scope = inputs.get("data_scope") if isinstance(inputs.get("data_scope"), dict) else {}
        rec = cls(
            agent=node, role=ROLE_OF.get(node, "other"),
            step_id=step.get("step_id"), step_no=step.get("seq"),
            human_role=step.get("role") or None,
            outcome="ok" if step.get("status") == "ok" else "failed",
            error_class=step.get("error_class") if step.get("status") != "ok" else None,
            concluded=decision.get("reason") if isinstance(decision.get("reason"), str) else None,
            decided={k: v for k, v in decision.items() if k != "reason"},
            model=ModelUsage.from_inference(step.get("inference")),
            timing={"duration_ms": _int(step.get("latency_ms"))},
        )
        cols = scope.get("columns")
        if scope.get("rows") is not None or cols:
            rec.data = DataShape(rows=_int(scope["rows"]) if scope.get("rows") is not None else None,
                                 columns=len(cols) if isinstance(cols, list) else None,
                                 column_names=[str(c) for c in cols] if isinstance(cols, list) else [])
        ops = calc.get("operations")
        if isinstance(ops, list):
            rec.operations = OperationSummary(detail=[dict(o) for o in ops if isinstance(o, dict)])
        if rec.role == "verification" and decision.get("type") == "check":
            rec.verification = VerificationSummary(
                checks=[{"name": decision.get("check"), "status": decision.get("value")}],
                retries=max(0, _int(calc.get("attempt"), 1) - 1),
                verdict=_vocab(decision.get("value"), VERDICTS, None))
        return rec


@dataclass
class SwarmUnderstanding:
    """The mission's understanding records folded into one: the fan-out, what
    each clone did, what the fleet concluded. The body of ``swarm.understanding``."""

    clones: int = 0
    parallel: Optional[int] = None
    steps_total: int = 0
    steps_per_clone: dict[str, int] = field(default_factory=dict)
    roles: dict[str, int] = field(default_factory=dict)
    data: Optional[DataShape] = None
    sampling: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    merge: dict[str, Any] = field(default_factory=dict)
    model: Optional[ModelUsage] = None
    timing: dict[str, Any] = field(default_factory=dict)
    outcome: str = "done"

    @classmethod
    def from_records(cls, records: list[dict[str, Any]], *, merge: Optional[dict[str, Any]] = None,
                     duration_ms: Optional[int] = None, outcome: str = "done") -> "SwarmUnderstanding":
        """Fold a mission's records (swarm trace and ``agent.understanding``
        together) into the mission-level view. Reads records, never frames.
        ``merge`` is the ``swarm.merged`` body when it is not yet recorded (the
        coordinator folds before it announces, so the trace still ends
        ``merged, finished``)."""
        agents = [r for r in records if r.get("type") == AGENT_RECORD]
        requested = next((r for r in records if r.get("type") == "swarm.requested"), {})
        sharded = next((r for r in records if r.get("type") == "swarm.sharded"), {})
        merged = merge if isinstance(merge, dict) else next(
            (r for r in records if r.get("type") == "swarm.merged"), {})
        returned = [r for r in records if r.get("type") == "clone.returned"]

        out = cls(outcome=outcome, timing={"duration_ms": duration_ms})
        out.clones = _int(sharded.get("clones"), len({r.get("clone_id") for r in returned}))
        out.parallel = requested.get("granted_parallel")
        out.steps_total = len(agents)
        for r in agents:
            cid = str(r.get("clone_id"))
            out.steps_per_clone[cid] = out.steps_per_clone.get(cid, 0) + 1
            role = _vocab(r.get("role"), ROLES) or "other"
            out.roles[role] = out.roles.get(role, 0) + 1

        # The fleet's data shape: the merged row count, the shared columns, and
        # the histograms of the clone that saw the most rows (every clone sees
        # the same columns, so one clone's dtype histogram is the fleet's).
        scouted = [r for r in agents if r.get("role") == "understanding" and isinstance(r.get("data"), dict)]
        if scouted:
            rep = max(scouted, key=lambda r: _int((r.get("data") or {}).get("rows")))["data"]
            out.data = DataShape(
                rows=_int(merged.get("n_rows")) if merged.get("n_rows") is not None else sum(
                    _int((r.get("data") or {}).get("rows")) for r in scouted),
                columns=len(sharded.get("columns") or []) or rep.get("columns"),
                column_type_counts=dict(rep.get("column_type_counts") or {}),
                null_rate_counts=dict(rep.get("null_rate_counts") or {}),
                cardinality_counts=dict(rep.get("cardinality_counts") or {}),
                file_format=rep.get("file_format"),
                size_bytes=sum(_int((r.get("data") or {}).get("size_bytes")) for r in scouted) or None,
                quality_score=min((_int(r["data"].get("quality_score"), 100) for r in scouted
                                   if r["data"].get("quality_score") is not None), default=None),
                column_names=[str(c) for c in (sharded.get("columns") or rep.get("column_names") or [])],
            )

        sampled = [r for r in agents if r.get("role") == "sampling" and isinstance(r.get("sampling"), dict)]
        out.sampling = {
            "sampled_clones": sum(1 for r in sampled if r["sampling"].get("sampled")),
            "strategy_counts": _counts([r["sampling"].get("strategy") for r in sampled], SAMPLING_STRATEGIES),
            "strategies": sorted({str(r["sampling"].get("strategy")) for r in sampled}),
        }

        checks = [r["verification"] for r in agents
                  if r.get("role") == "verification" and isinstance(r.get("verification"), dict)]
        verdict = _vocab(merged.get("verdict"), VERDICTS, None)
        out.verification = {
            "checks_run": sum(_int(v.get("checks_run")) for v in checks),
            "passed": sum(_int(v.get("passed")) for v in checks),
            "warned": sum(_int(v.get("warned")) for v in checks),
            "failed": sum(_int(v.get("failed")) for v in checks),
            "retries": sum(_int(v.get("retries")) for v in checks),
            "verdict": verdict,
        }

        verdicts = {r.get("verdict") for r in returned}
        contributions = [c for c in (merged.get("contributions") or []) if isinstance(c, dict)]
        selected = [c.get("clone_id") for c in contributions if c.get("selected")]
        out.merge = {
            "verdict": verdict,
            "max_safe_deployment_level": merged.get("max_safe_deployment_level"),
            "rules": dict(merged.get("rules") or {}),
            "agreement": "unanimous" if len(verdicts) <= 1 else "split",
            "clone_verdicts": {str(r.get("clone_id")): r.get("verdict") for r in returned},
            "selected_clone": selected[0] if selected else None,
            "model_selected": bool(selected),
            "candidates": sum(1 for r in returned if r.get("model")),
        }

        usage = [r["model"] for r in agents if isinstance(r.get("model"), dict)]
        if usage:
            out.model = ModelUsage(
                provider=next((u.get("provider") for u in usage if u.get("provider")), None),
                model_id=next((u.get("model_id") for u in usage if u.get("model_id")), None),
                calls=sum(_int(u.get("calls")) for u in usage),
                tokens_in=sum(_int(u.get("tokens_in")) for u in usage),
                tokens_out=sum(_int(u.get("tokens_out")) for u in usage))
        return out

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "clones": self.clones, "parallel": self.parallel,
            "steps_total": self.steps_total, "steps_per_clone": dict(self.steps_per_clone),
            "roles": dict(self.roles),
            "data": self.data.as_dict() if self.data else None,
            "sampling": dict(self.sampling),
            "verification": dict(self.verification),
            "merge": dict(self.merge),
            "model": self.model.as_dict() if self.model else None,
            "timing": dict(self.timing),
            "outcome": self.outcome,
        }


# -------------------------------------------------------------- local layer
class UnderstandingLog:
    """``understanding.jsonl``: one record per line, flushed as it is written,
    so ``avaloka --debug`` and a tailing reader see each step as it lands.
    Thread-safe (clones run in parallel). Never raises into the mission."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._fh = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        except OSError:
            self._fh = None

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._write(record)

    def record(self, recorder: Any, type: str, **fields: Any) -> dict[str, Any]:  # noqa: A002 - the record's key
        """Append a record to ``recorder`` and to this log as one step.

        The recorder hands out ``seq`` under its own lock. Appending to the
        log afterwards, under this one, let two clones take seq 9 and 10 and
        write 10 first, and a reader tailing with ``read_log(after=10)`` then
        never saw 9. Taking ``seq`` while holding this lock keeps the file in
        seq order. Whatever ``recorder.record`` raises propagates, and then
        nothing is written."""
        with self._lock:
            rec = recorder.record(type, **fields)
            self._write(rec)
        return rec

    def _write(self, record: dict[str, Any]) -> None:
        # The caller holds self._lock.
        if record.get("type") not in RECORD_TYPES or self._fh is None:
            return
        try:
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()
        except (OSError, TypeError, ValueError):
            pass

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None

    def __enter__(self) -> "UnderstandingLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_log(path: str | Path, *, after: int = 0) -> Iterator[dict[str, Any]]:
    """Records in ``understanding.jsonl`` with ``seq > after``, in file order.
    The incremental reader: keep the last ``seq`` you saw and pass it back.
    A partially written last line (the writer is mid-flush) is skipped, not
    an error; it is complete on the next read."""
    p = Path(path)
    if not p.is_file():
        return
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and _int(rec.get("seq")) > after:
                yield rec


def understanding_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("type") in RECORD_TYPES]


# ------------------------------------------------------------ egress layer
def _bounded(value: Any, lo: int, hi: int) -> int:
    return max(lo, min(hi, _int(value, lo)))


def _hist(value: Any, vocabulary: tuple) -> dict[str, int]:
    """A histogram re-keyed through the vocabulary: an unknown key is dropped,
    never passed through, so a histogram cannot smuggle a name."""
    if not isinstance(value, dict):
        return {}
    return {k: _bounded(v, 0, MAX_COUNT) for k, v in value.items() if k in vocabulary}


def _class(name: Any) -> Optional[str]:
    if not isinstance(name, str):
        return None
    ok = name.replace("_", "a").replace(".", "a").isalnum() and name[:1].isalpha() and len(name) <= 64
    return name if ok else "Exception"


def egress_event(record: dict[str, Any], *, seq: int = 0,
                 session_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    """The metadata-only projection of one understanding record, or ``None``
    for a record that is not one, names an agent telemetry does not know, or
    is still running.

    Built as an allowlist from closed vocabularies, buckets and bounded
    counts. The goal, the source, the column names, the reasons, the check
    details, the formulas, the clone ids and the mission id exist only in the
    record and are never read here. ``seq`` and ``session_id`` are the
    telemetry batch's, not the trace's: a trace seq is positional within a
    mission and would re-identify it.
    """
    kind = record.get("type")
    sid = session_id or _SESSION_ID
    if kind == AGENT_RECORD:
        agent = record.get("agent")
        if agent not in KNOWN_AGENTS:
            return None
        out: dict[str, Any] = {
            "event_type": AGENT_RECORD,
            "event_id": uuid.uuid4().hex,
            "agent": agent,
            "role": _vocab(record.get("role"), ROLES) or "other",
            "outcome": "success" if record.get("outcome") == "ok" else "failed",
            "session_id": sid,
            "seq": seq,
        }
        if record.get("clone_index") is not None:
            out["clone_index"] = _bounded(record.get("clone_index"), 0, 10_000)
        if record.get("step_no") is not None:
            out["step_no"] = _bounded(record.get("step_no"), 0, 10_000)
        if out["outcome"] != "success":
            cls = _class(record.get("error_class"))
            if cls:
                out["error_class"] = cls
        timing = record.get("timing") if isinstance(record.get("timing"), dict) else {}
        if timing.get("duration_ms") is not None:
            out["duration_ms"] = _bounded(timing.get("duration_ms"), 0, MAX_DURATION_MS)
        _project_data(out, record.get("data"))
        sampling = record.get("sampling")
        if isinstance(sampling, dict):
            out["sampled"] = bool(sampling.get("sampled"))
            out["sampling_strategy"] = _vocab(sampling.get("strategy"), SAMPLING_STRATEGIES) or "other"
            if sampling.get("fraction_bucket") in FRACTION_BUCKETS:
                out["sample_fraction_bucket"] = sampling["fraction_bucket"]
        ops = record.get("operations")
        if isinstance(ops, dict):
            out["operation_count"] = _bounded(ops.get("count"), 0, MAX_COUNT)
            out["op_counts"] = _hist(ops.get("op_counts"), OP_KINDS)
        _project_verification(out, record.get("verification"))
        _project_model(out, record.get("model"))
        return out
    if kind == SWARM_RECORD:
        out = {
            "event_type": SWARM_RECORD,
            "event_id": uuid.uuid4().hex,
            "clones": _bounded(record.get("clones"), 1, 10_000),
            "steps_total": _bounded(record.get("steps_total"), 0, MAX_COUNT),
            "outcome": "success" if record.get("outcome") == "done" else "failed",
            "session_id": sid,
            "seq": seq,
        }
        if record.get("parallel") is not None:
            out["parallel"] = _bounded(record.get("parallel"), 1, 10_000)
        per_clone = record.get("steps_per_clone") if isinstance(record.get("steps_per_clone"), dict) else {}
        if per_clone:
            counts = [_int(v) for v in per_clone.values()]
            out["steps_per_clone_min"] = _bounded(min(counts), 0, MAX_COUNT)
            out["steps_per_clone_max"] = _bounded(max(counts), 0, MAX_COUNT)
        out["role_counts"] = _hist(record.get("roles"), ROLES)
        timing = record.get("timing") if isinstance(record.get("timing"), dict) else {}
        if timing.get("duration_ms") is not None:
            out["duration_ms"] = _bounded(timing.get("duration_ms"), 0, MAX_DURATION_MS)
        _project_data(out, record.get("data"))
        sampling = record.get("sampling") if isinstance(record.get("sampling"), dict) else {}
        out["sampled_clones"] = _bounded(sampling.get("sampled_clones"), 0, 10_000)
        out["sampling_strategy_counts"] = _hist(sampling.get("strategy_counts"), SAMPLING_STRATEGIES)
        _project_verification(out, record.get("verification"))
        merge = record.get("merge") if isinstance(record.get("merge"), dict) else {}
        if merge.get("agreement") in AGREEMENTS:
            out["merge_agreement"] = merge["agreement"]
        out["model_selected"] = bool(merge.get("model_selected"))
        out["candidates"] = _bounded(merge.get("candidates"), 0, 10_000)
        _project_model(out, record.get("model"))
        return out
    return None


def _project_data(out: dict[str, Any], data: Any) -> None:
    if not isinstance(data, dict):
        return
    if data.get("rows") is not None:
        out["row_count_bucket"] = row_bucket(data.get("rows"))
    if data.get("columns") is not None:
        out["column_count_bucket"] = column_bucket(data.get("columns"))
    out["column_type_counts"] = _hist(data.get("column_type_counts"), COLUMN_KINDS)
    out["null_rate_counts"] = _hist(data.get("null_rate_counts"), NULL_RATE_BUCKETS)
    out["cardinality_counts"] = _hist(data.get("cardinality_counts"), CARDINALITY_BUCKETS)
    fmt = _file_format(data.get("file_format"))
    if fmt:
        out["file_format"] = fmt
    bucket = size_bucket(data.get("size_bytes"))
    if bucket:
        out["size_bucket"] = bucket
    if data.get("quality_score") is not None:
        out["quality_score"] = _bounded(data.get("quality_score"), 0, 100)


def _project_verification(out: dict[str, Any], ver: Any) -> None:
    if not isinstance(ver, dict):
        return
    out["checks_run"] = _bounded(ver.get("checks_run"), 0, MAX_COUNT)
    out["checks_passed"] = _bounded(ver.get("passed"), 0, MAX_COUNT)
    out["checks_warned"] = _bounded(ver.get("warned"), 0, MAX_COUNT)
    out["checks_failed"] = _bounded(ver.get("failed"), 0, MAX_COUNT)
    out["retry_count"] = _bounded(ver.get("retries"), 0, 100)
    if ver.get("verdict") in VERDICTS:
        out["verdict"] = ver["verdict"]


def _project_model(out: dict[str, Any], model: Any) -> None:
    if not isinstance(model, dict):
        return
    if model.get("provider") in LLM_PROVIDERS:
        out["llm_provider"] = model["provider"]
    if model.get("model_id") is not None:
        out["llm_model"] = model["model_id"] if model["model_id"] in LLM_MODELS else "other"
    out["model_calls"] = _bounded(model.get("calls"), 0, MAX_COUNT)
    out["tokens_in"] = _bounded(model.get("tokens_in"), 0, 1_000_000_000)
    out["tokens_out"] = _bounded(model.get("tokens_out"), 0, 1_000_000_000)


def egress_events(records: list[dict[str, Any]], *, start_seq: int = 0,
                  session_id: Optional[str] = None) -> list[dict[str, Any]]:
    """The projection of every understanding record in ``records``, in order,
    numbered from ``start_seq``: exactly the batch an uploader would send."""
    out: list[dict[str, Any]] = []
    seq = start_seq
    for rec in records:
        event = egress_event(rec, seq=seq, session_id=session_id)
        if event is not None:
            out.append(event)
            seq += 1
    return out


#: Every key an egress event may carry, by type. Closed: a test holds these
#: equal to the ingest service's field lists, and the projection above can
#: produce no other.
EGRESS_KEYS = {
    AGENT_RECORD: frozenset({
        "event_type", "event_id", "session_id", "seq", "agent", "role", "outcome", "error_class",
        "clone_index", "step_no", "duration_ms",
        "row_count_bucket", "column_count_bucket", "column_type_counts", "null_rate_counts",
        "cardinality_counts", "file_format", "size_bucket", "quality_score",
        "sampled", "sampling_strategy", "sample_fraction_bucket",
        "operation_count", "op_counts",
        "checks_run", "checks_passed", "checks_warned", "checks_failed", "retry_count", "verdict",
        "llm_provider", "llm_model", "model_calls", "tokens_in", "tokens_out",
    }),
    SWARM_RECORD: frozenset({
        "event_type", "event_id", "session_id", "seq", "clones", "parallel", "steps_total",
        "steps_per_clone_min", "steps_per_clone_max", "role_counts", "outcome", "duration_ms",
        "row_count_bucket", "column_count_bucket", "column_type_counts", "null_rate_counts",
        "cardinality_counts", "file_format", "size_bucket", "quality_score",
        "sampled_clones", "sampling_strategy_counts",
        "checks_run", "checks_passed", "checks_warned", "checks_failed", "retry_count", "verdict",
        "merge_agreement", "model_selected", "candidates",
        "llm_provider", "llm_model", "model_calls", "tokens_in", "tokens_out",
    }),
}
