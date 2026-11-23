"""Diagnose out-of-memory failures and propose a bounded, auditable recovery.

MTA training and Ray jobs fail on memory, and today that failure is terminal: a
worker is OOM-killed, the job reports FAILED, and someone reads logs at an
inconvenient hour to discover the dataset needed more room than the manifest
gives it. The information required to fix it — how big the data is, what the
worker was given, what it actually used — is all present at the moment of
failure and nothing looks at it.

This module turns that into a decision.

**Classification is by structured signal first, log text second.** A container
OOM-kill is exit code 137 and a pod reason of ``OOMKilled``; those are facts the
orchestrator reports. Log scraping is the fallback, not the primary, because
a message mentioning "memory" is not evidence that memory was the cause — a
stack trace quoting a variable named ``memory_limit`` would match, and
remediating a non-memory failure by doubling the cluster wastes money without
fixing anything.

**A retry that changes nothing is not a recovery.** :class:`RecoveryPolicy`
refuses to re-run a workload under conditions identical to the ones that just
failed. Every attempt must strictly increase the resource that ran out, attempts
are capped, and the ceiling is a real limit rather than advice — an unbounded
"try bigger" loop against a cloud autoscaler is a way to spend a lot of money on
a job that was never going to fit.

**Recovery proposes; a human disposes above a threshold.** Doubling a worker
from 8 GiB to 16 GiB is routine. Going to 64 GiB across ten workers is a
spending decision, and the module marks it as one rather than making it.

Pure functions over dataclasses: no cluster, no kubectl, no network, so the
whole decision path is testable in milliseconds.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Linux OOM-killer: 128 + SIGKILL(9). The kernel reports this, we do not infer it.
OOM_EXIT_CODE = 137

#: Kubernetes container termination reasons that mean "the kernel took it".
OOM_POD_REASONS = frozenset({"OOMKilled", "OOMKilling"})

#: pandas holds far more than the bytes on disk: intermediate copies during a
#: merge or groupby, plus per-object overhead. Measured against this codebase's
#: own training path, 3x the estimated frame is a working figure rather than a
#: guess, and it is deliberately conservative — under-asking causes exactly the
#: failure this module exists to recover from.
FRAME_OVERHEAD_FACTOR = 3.0

#: Bytes a column costs per row, by dtype family. Object columns dominate every
#: real OOM we have seen: a 20-character string is ~69 bytes in CPython, not 20.
_BYTES_PER_ROW = {
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "float32": 4, "float64": 8, "bool": 1,
    "datetime64[ns]": 8, "timedelta64[ns]": 8,
    "category": 4,
    "object": 69,
}
_DEFAULT_BYTES_PER_ROW = 8

#: Never propose beyond this per-worker figure without explicit human approval.
DEFAULT_MEMORY_CEILING_GIB = 64.0

#: Above this, a proposal is a spending decision rather than a routine bump.
HUMAN_APPROVAL_GIB = 32.0

#: More than this many attempts and the shape of the problem is wrong, not the size.
DEFAULT_MAX_ATTEMPTS = 3


class FailureKind(str, Enum):
    """What actually went wrong. Only the memory kinds are remediable here."""

    OOM_KILLED = "oom_killed"                 # kernel killed the container
    HEAP_EXHAUSTED = "heap_exhausted"         # Python MemoryError inside the process
    OBJECT_STORE_FULL = "object_store_full"   # Ray plasma store / spilling
    WORKER_LOST = "worker_lost"               # worker died; often but not always memory
    NOT_MEMORY = "not_memory"                 # do not remediate by resizing
    UNKNOWN = "unknown"                       # insufficient evidence; escalate


#: Which kinds a resource change can plausibly fix.
REMEDIABLE = frozenset({
    FailureKind.OOM_KILLED, FailureKind.HEAP_EXHAUSTED, FailureKind.OBJECT_STORE_FULL,
})


class Strategy(str, Enum):
    RAISE_MEMORY = "raise_memory"           # same shape, bigger worker
    ADD_WORKERS = "add_workers"             # same worker, smaller shard each
    RAISE_OBJECT_STORE = "raise_object_store"
    REDUCE_BATCH = "reduce_batch"           # process less at a time
    ESCALATE = "escalate"                   # a person needs to decide


@dataclass(frozen=True)
class WorkloadResources:
    """What a workload was given. Memory is GiB per worker."""

    memory_gib: float
    cpus: float = 1.0
    workers: int = 1
    object_store_gib: float = 0.5
    batch_rows: Optional[int] = None

    def with_memory(self, gib: float) -> "WorkloadResources":
        return WorkloadResources(gib, self.cpus, self.workers,
                                 self.object_store_gib, self.batch_rows)

    def with_workers(self, n: int) -> "WorkloadResources":
        return WorkloadResources(self.memory_gib, self.cpus, n,
                                 self.object_store_gib, self.batch_rows)

    @property
    def total_memory_gib(self) -> float:
        return self.memory_gib * max(self.workers, 1)


@dataclass(frozen=True)
class FailureSignal:
    """Everything known at the moment of failure. All fields optional.

    Structured fields carry the weight; ``logs`` is corroboration.
    """

    exit_code: Optional[int] = None
    pod_reason: Optional[str] = None
    logs: str = ""
    job_status: Optional[str] = None
    restart_count: int = 0


@dataclass
class Diagnosis:
    kind: FailureKind
    confidence: float
    evidence: List[str] = field(default_factory=list)

    @property
    def remediable(self) -> bool:
        return self.kind in REMEDIABLE

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "confidence": round(self.confidence, 3),
                "evidence": list(self.evidence), "remediable": self.remediable}


@dataclass
class Remediation:
    strategy: Strategy
    resources: Optional[WorkloadResources]
    reason: str
    needs_human_approval: bool = False
    estimated_need_gib: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        r = self.resources
        return {
            "strategy": self.strategy.value,
            "reason": self.reason,
            "needs_human_approval": self.needs_human_approval,
            "estimated_need_gib": self.estimated_need_gib,
            "resources": None if r is None else {
                "memory_gib": r.memory_gib, "cpus": r.cpus, "workers": r.workers,
                "object_store_gib": r.object_store_gib, "batch_rows": r.batch_rows,
            },
        }


# --------------------------------------------------------------------------- #
# 1. Classify
# --------------------------------------------------------------------------- #

_LOG_PATTERNS = (
    (FailureKind.HEAP_EXHAUSTED, re.compile(r"\bMemoryError\b"), 0.9,
     "Python MemoryError raised in the worker process"),
    (FailureKind.HEAP_EXHAUSTED,
     re.compile(r"Unable to allocate [\d.]+ [KMG]iB", re.I), 0.9,
     "numpy/pandas allocation failure"),
    (FailureKind.OBJECT_STORE_FULL,
     re.compile(r"object store.{0,40}(full|out of memory)|ObjectStoreFullError", re.I), 0.9,
     "Ray object store exhausted"),
    (FailureKind.OBJECT_STORE_FULL,
     re.compile(r"[Ss]pilled .{0,30} to (local )?disk"), 0.55,
     "Ray spilled objects to disk, indicating object-store pressure"),
    (FailureKind.OOM_KILLED,
     re.compile(r"Out of memory: Killed process|oom-kill|OutOfMemoryError", re.I), 0.85,
     "kernel OOM message in logs"),
    (FailureKind.WORKER_LOST,
     re.compile(r"(RayActorError|The actor died unexpectedly|Worker .{0,20}died)", re.I), 0.5,
     "worker died; memory is a likely but unproven cause"),
)

#: Failures that are definitely NOT about memory. Checked FIRST — a traceback
#: mentioning a variable called `memory` must not read as an OOM.
_NON_MEMORY_PATTERNS = (
    (re.compile(r"\b(FileNotFoundError|NoSuchKey|403 Forbidden|401 Unauthorized)\b"),
     "missing input or credentials"),
    (re.compile(r"\b(SyntaxError|ImportError|ModuleNotFoundError|NameError)\b"),
     "code or environment defect"),
    (re.compile(r"\b(ConnectionRefused|Name or service not known|TLS handshake)\b"),
     "network or endpoint failure"),
)


def classify_failure(signal: FailureSignal) -> Diagnosis:
    """Decide what kind of failure this was, preferring structured evidence."""
    evidence: List[str] = []

    # Structured signals are facts reported by the orchestrator.
    if signal.pod_reason and signal.pod_reason in OOM_POD_REASONS:
        return Diagnosis(FailureKind.OOM_KILLED, 1.0,
                         [f"pod terminated with reason {signal.pod_reason}"])
    if signal.exit_code == OOM_EXIT_CODE:
        return Diagnosis(FailureKind.OOM_KILLED, 0.95,
                         [f"exit code {OOM_EXIT_CODE} (128 + SIGKILL): the kernel OOM-killer"])

    text = signal.logs or ""

    # Definite non-memory causes win over any keyword match below.
    for pattern, why in _NON_MEMORY_PATTERNS:
        if pattern.search(text):
            return Diagnosis(FailureKind.NOT_MEMORY, 0.9,
                             [f"{why}; resizing would not fix this"])

    best: Optional[Diagnosis] = None
    for kind, pattern, confidence, why in _LOG_PATTERNS:
        if pattern.search(text):
            if best is None or confidence > best.confidence:
                best = Diagnosis(kind, confidence, [why])
    if best:
        if signal.restart_count > 1:
            best.evidence.append(
                f"{signal.restart_count} restarts already, each ending the same way")
        return best

    if signal.exit_code not in (None, 0):
        evidence.append(f"exit code {signal.exit_code} with no memory signature in logs")
    return Diagnosis(FailureKind.UNKNOWN, 0.2,
                     evidence or ["no usable failure signal was captured"])


# --------------------------------------------------------------------------- #
# 2. Estimate what the workload actually needs
# --------------------------------------------------------------------------- #

def estimate_working_set_gib(*, rows: int, dtypes: Sequence[str],
                             overhead_factor: float = FRAME_OVERHEAD_FACTOR) -> float:
    """Estimate peak memory for holding and transforming a frame, in GiB.

    Deliberately conservative. Under-estimating reproduces the failure; the cost
    of over-estimating is some idle RAM for the length of one job.
    """
    if rows <= 0 or not dtypes:
        return 0.0
    per_row = sum(_BYTES_PER_ROW.get(str(d), _DEFAULT_BYTES_PER_ROW) for d in dtypes)
    raw = rows * per_row
    return (raw * max(overhead_factor, 1.0)) / (1024 ** 3)


def estimate_from_profile(profile: Dict[str, Any]) -> float:
    """Estimate from the profile the Profiling agent already produces."""
    dataset = profile.get("dataset") or {}
    rows = int(dataset.get("n_rows") or 0)
    columns = profile.get("columns") or []
    dtypes = [str(c.get("dtype") or c.get("type") or "object") for c in columns]
    return estimate_working_set_gib(rows=rows, dtypes=dtypes)


# --------------------------------------------------------------------------- #
# 3. Propose a remediation
# --------------------------------------------------------------------------- #

def _next_memory_step(current: float, needed: Optional[float]) -> float:
    """At least double, and clear the estimate when one exists."""
    target = current * 2.0
    if needed:
        target = max(target, needed * 1.15)   # a little headroom over the estimate
    return float(math.ceil(target * 2) / 2)    # round up to the next 0.5 GiB


def propose_remediation(diagnosis: Diagnosis, resources: WorkloadResources, *,
                        estimated_need_gib: Optional[float] = None,
                        memory_ceiling_gib: float = DEFAULT_MEMORY_CEILING_GIB,
                        max_workers: int = 10) -> Remediation:
    """Decide what to change. Never returns a no-op disguised as a retry."""
    if not diagnosis.remediable:
        return Remediation(
            Strategy.ESCALATE, None,
            f"{diagnosis.kind.value}: not a memory failure, so more memory would spend "
            f"money without fixing it",
            needs_human_approval=True, estimated_need_gib=estimated_need_gib)

    if diagnosis.kind is FailureKind.OBJECT_STORE_FULL:
        bumped = WorkloadResources(
            resources.memory_gib, resources.cpus, resources.workers,
            round(min(resources.object_store_gib * 2, resources.memory_gib * 0.4), 3),
            resources.batch_rows)
        if bumped.object_store_gib > resources.object_store_gib:
            return Remediation(
                Strategy.RAISE_OBJECT_STORE, bumped,
                "the Ray object store filled before worker heap did; raising the store is "
                "cheaper than raising every worker",
                estimated_need_gib=estimated_need_gib)

    target = _next_memory_step(resources.memory_gib, estimated_need_gib)

    if target > memory_ceiling_gib:
        # Spreading the work is the alternative to a bigger machine, and it is
        # usually the right one: a frame that does not fit in 64 GiB should be
        # sharded, not chased with a larger instance type.
        if resources.workers < max_workers:
            more = min(max_workers, max(resources.workers * 2, 2))
            return Remediation(
                Strategy.ADD_WORKERS, resources.with_workers(more),
                f"a single worker would need {target:.1f} GiB, above the "
                f"{memory_ceiling_gib:.0f} GiB ceiling; sharding across {more} workers "
                f"keeps each one within it",
                estimated_need_gib=estimated_need_gib)
        if resources.batch_rows:
            return Remediation(
                Strategy.REDUCE_BATCH,
                WorkloadResources(resources.memory_gib, resources.cpus, resources.workers,
                                  resources.object_store_gib, max(resources.batch_rows // 2, 1)),
                "already at the worker ceiling and the maximum worker count; halving the "
                "batch is the remaining lever",
                estimated_need_gib=estimated_need_gib)
        return Remediation(
            Strategy.ESCALATE, None,
            f"needs ~{target:.1f} GiB per worker, above the {memory_ceiling_gib:.0f} GiB "
            f"ceiling, and neither sharding nor batching is available",
            needs_human_approval=True, estimated_need_gib=estimated_need_gib)

    return Remediation(
        Strategy.RAISE_MEMORY, resources.with_memory(target),
        f"raising worker memory {resources.memory_gib:.1f} → {target:.1f} GiB"
        + (f" against an estimated need of {estimated_need_gib:.1f} GiB"
           if estimated_need_gib else ""),
        needs_human_approval=target >= HUMAN_APPROVAL_GIB,
        estimated_need_gib=estimated_need_gib)


# --------------------------------------------------------------------------- #
# 4. Bounded, auditable recovery
# --------------------------------------------------------------------------- #

@dataclass
class RecoveryAttempt:
    attempt: int
    diagnosis: Diagnosis
    remediation: Remediation

    def as_dict(self) -> Dict[str, Any]:
        return {"attempt": self.attempt, "diagnosis": self.diagnosis.as_dict(),
                "remediation": self.remediation.as_dict()}


class RecoveryExhausted(RuntimeError):
    """No further automatic recovery is available. The trail explains why."""


@dataclass
class RecoveryPolicy:
    """Bounded retry. Refuses to repeat a failure under identical conditions."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    memory_ceiling_gib: float = DEFAULT_MEMORY_CEILING_GIB
    max_workers: int = 10
    auto_approve_below_gib: float = HUMAN_APPROVAL_GIB
    history: List[RecoveryAttempt] = field(default_factory=list)

    def next_step(self, signal: FailureSignal, resources: WorkloadResources, *,
                  estimated_need_gib: Optional[float] = None) -> Remediation:
        """Diagnose, propose, and record. Raises when recovery is exhausted."""
        attempt = len(self.history) + 1
        diagnosis = classify_failure(signal)

        if attempt > self.max_attempts:
            raise RecoveryExhausted(
                f"{self.max_attempts} attempts already made; the last diagnosis was "
                f"{diagnosis.kind.value}. Repeated OOM at increasing size means the shape "
                f"of the job is wrong, not its size.")

        remediation = propose_remediation(
            diagnosis, resources, estimated_need_gib=estimated_need_gib,
            memory_ceiling_gib=self.memory_ceiling_gib, max_workers=self.max_workers)

        # A retry that changes nothing reproduces the failure and bills for it.
        if remediation.resources is not None and remediation.resources == resources:
            raise RecoveryExhausted(
                "the proposed resources are identical to the ones that just failed; "
                "retrying would reproduce the failure at full cost")

        self.history.append(RecoveryAttempt(attempt, diagnosis, remediation))
        logger.info("[recovery] attempt %d: %s → %s (%s)", attempt,
                    diagnosis.kind.value, remediation.strategy.value, remediation.reason)
        return remediation

    def may_apply_automatically(self, remediation: Remediation) -> bool:
        """Whether this change can be made without asking a person."""
        if remediation.needs_human_approval or remediation.strategy is Strategy.ESCALATE:
            return False
        r = remediation.resources
        return r is not None and r.memory_gib < self.auto_approve_below_gib

    def audit_trail(self) -> List[Dict[str, Any]]:
        return [a.as_dict() for a in self.history]
