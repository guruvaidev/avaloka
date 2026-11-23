"""
Swarm coordination harness — "multiple Avalokas under one main Avaloka".

This models the spec:

    Avaloka is an open-claw based agent (planner, coder, ... sub-agents) that
    supports swarm intelligence of multiple data scientists working on the same
    dataset to sample, analyze features, build features, run transformations,
    train ML, and provision inference-as-a-service over k8s. Multiple Avalokas
    coordinate under one main Avaloka.

Design
------
``WorkerAvaloka``  : one Avaloka instance (wraps an ``AvalokaClient`` = its
                     planner/coder/etc. sub-agents). Executes assigned subtasks.
``MainAvaloka``    : the coordinator. Decomposes a dataset goal into subtasks,
                     dispatches them across workers (the *swarm*), then performs
                     consensus/aggregation and conflict resolution.
``SwarmPlan``      : the decomposition + assignment (inspected by tests).
``SwarmResult``    : aggregated outputs + provenance + consensus decisions.

The coordinator is deliberately deterministic and side-effect-light so swarm
*coordination properties* (coverage, disjoint partitioning, consensus,
fault-tolerance, deterministic aggregation) can be asserted as unit tests,
independent of LLM nondeterminism. Workers run concurrently via a thread pool to
exercise real parallelism and shared-dataset contention.
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .adapters import AvalokaClient, ReferenceClient


# --------------------------------------------------------------------------- #
# Subtasks
# --------------------------------------------------------------------------- #


@dataclass
class SubTask:
    sub_id: str
    kind: str  # sample | analyze_features | build_features | transform | train | serve
    params: Dict[str, Any] = field(default_factory=dict)
    # Optional partition descriptor for data-parallel work (e.g. column shard).
    partition: Optional[Dict[str, Any]] = None
    depends_on: List[str] = field(default_factory=list)


@dataclass
class SubResult:
    sub_id: str
    worker: str
    kind: str
    ok: bool
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    attempts: int = 1


@dataclass
class SwarmPlan:
    goal: str
    subtasks: List[SubTask]
    assignment: Dict[str, str]  # sub_id -> worker name

    def coverage(self) -> List[str]:
        return [s.kind for s in self.subtasks]


@dataclass
class SwarmResult:
    goal: str
    sub_results: List[SubResult]
    consensus: Dict[str, Any] = field(default_factory=dict)
    aggregated: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)  # sub_id -> worker

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.sub_results)

    def successes(self) -> List[SubResult]:
        return [r for r in self.sub_results if r.ok]


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #


class WorkerAvaloka:
    """One Avaloka agent in the swarm. Wraps a client and tracks its own log."""

    def __init__(self, name: str, client: Optional[AvalokaClient] = None,
                 fail_kinds: Optional[set] = None, max_retries: int = 1):
        self.name = name
        self.client = client or ReferenceClient()
        self.fail_kinds = fail_kinds or set()  # for fault-injection tests
        self.max_retries = max_retries
        self.log: List[str] = []
        self._lock = threading.Lock()

    def execute(self, st: SubTask, uri: str) -> SubResult:
        with self._lock:
            self.log.append(st.sub_id)
        if st.kind in self.fail_kinds:
            return SubResult(st.sub_id, self.name, st.kind, ok=False,
                             error=f"injected failure in {st.kind}")
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                out = self._dispatch(st, uri)
                if out.get("error"):
                    last_err = out["error"]
                    continue
                return SubResult(st.sub_id, self.name, st.kind, ok=True,
                                 output=out, attempts=attempt)
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
        return SubResult(st.sub_id, self.name, st.kind, ok=False,
                         error=last_err, attempts=self.max_retries)

    def _dispatch(self, st: SubTask, uri: str) -> Dict[str, Any]:
        c, p = self.client, dict(st.params)
        if st.kind == "sample":
            return c.sample(uri, **p)
        if st.kind == "analyze_features":
            return c.analyze_features(uri, **p)
        if st.kind == "build_features":
            return c.build_features(uri, **p)
        if st.kind == "transform":
            return c.transform(uri, **p)
        if st.kind == "train":
            return c.train(uri, **p)
        if st.kind == "serve":
            return c.serve(p.pop("model_ref"), **p)
        raise ValueError(f"unknown subtask kind {st.kind!r}")


# --------------------------------------------------------------------------- #
# Main coordinator
# --------------------------------------------------------------------------- #


class MainAvaloka:
    """The coordinating Avaloka that orchestrates a swarm of WorkerAvalokas."""

    def __init__(self, workers: List[WorkerAvaloka], max_parallel: int = 8):
        if not workers:
            raise ValueError("swarm needs at least one worker")
        self.workers = {w.name: w for w in workers}
        self.max_parallel = max_parallel

    # -- planning ----------------------------------------------------------- #
    def plan(self, goal: str, *, target: Optional[str], task_kind: str,
             feature_columns: List[str], n_model_candidates: int = 3,
             heavy: bool = False) -> SwarmPlan:
        """Decompose a 'build an inference service for X' goal into subtasks.

        Lifecycle order mirrors the spec. Feature analysis is *data-parallel*:
        the feature columns are sharded across workers so multiple data
        scientists genuinely work on the same dataset concurrently. Model
        training is *redundant/competitive*: several workers train candidates
        and the coordinator picks a consensus winner.
        """
        subtasks: List[SubTask] = []
        subtasks.append(SubTask("s_sample", "sample",
                                 {"stratify_by": target, "sample_size": 0.1}))

        shards = _shard(feature_columns, len(self.workers))
        analyze_ids = []
        for i, shard in enumerate(shards):
            sid = f"s_analyze_{i}"
            analyze_ids.append(sid)
            subtasks.append(SubTask(sid, "analyze_features", {"target": target},
                                    partition={"columns": shard},
                                    depends_on=["s_sample"]))

        subtasks.append(SubTask("s_build", "build_features", {"target": target},
                                depends_on=analyze_ids))
        subtasks.append(SubTask("s_transform", "transform",
                                {"operations": [{"op": "fillna", "value": 0}]},
                                depends_on=["s_build"]))

        train_ids = []
        for k in range(n_model_candidates):
            sid = f"s_train_{k}"
            train_ids.append(sid)
            subtasks.append(SubTask(sid, "train",
                                    {"target": target, "task_kind": task_kind,
                                     "heavy": heavy, "time_budget_s": 30},
                                    depends_on=["s_transform"]))

        subtasks.append(SubTask("s_serve", "serve", {"platform": "k8s", "replicas": 2},
                                depends_on=train_ids))

        assignment = self._assign(subtasks)
        return SwarmPlan(goal=goal, subtasks=subtasks, assignment=assignment)

    def _assign(self, subtasks: List[SubTask]) -> Dict[str, str]:
        """Round-robin assignment; keeps load balanced + reproducible."""
        names = sorted(self.workers)
        return {st.sub_id: names[i % len(names)] for i, st in enumerate(subtasks)}

    # -- execution ---------------------------------------------------------- #
    def run(self, plan: SwarmPlan, uri: str) -> SwarmResult:
        """Execute the plan respecting dependencies; parallel within a wave."""
        done: Dict[str, SubResult] = {}
        by_id = {s.sub_id: s for s in plan.subtasks}
        remaining = set(by_id)

        while remaining:
            wave = [sid for sid in remaining
                    if all(d in done for d in by_id[sid].depends_on)]
            if not wave:
                # Unsatisfiable deps (e.g. all parents failed) -> mark failed.
                for sid in remaining:
                    done[sid] = SubResult(sid, "-", by_id[sid].kind, ok=False,
                                          error="dependency unsatisfied")
                break
            # Redundant/competitive parents: a dependent runs as long as *at
            # least one* of its dependencies succeeded (e.g. serve proceeds once
            # any trainer in the race produced a model). It is only blocked when
            # it has dependencies and all of them failed.
            runnable, blocked = [], []
            for sid in wave:
                deps = by_id[sid].depends_on
                deps_ok = (not deps) or any(done[d].ok for d in deps)
                (runnable if deps_ok else blocked).append(sid)
            for sid in blocked:
                done[sid] = SubResult(sid, "-", by_id[sid].kind, ok=False,
                                      error="upstream failure")
                remaining.discard(sid)

            with cf.ThreadPoolExecutor(max_workers=self.max_parallel) as ex:
                futs = {}
                for sid in runnable:
                    st = self._materialize(by_id[sid], done)
                    worker = self.workers[plan.assignment[sid]]
                    futs[ex.submit(worker.execute, st, uri)] = sid
                for fut in cf.as_completed(futs):
                    res = fut.result()
                    done[res.sub_id] = res
                    remaining.discard(res.sub_id)

        sub_results = [done[s.sub_id] for s in plan.subtasks]
        result = SwarmResult(goal=plan.goal, sub_results=sub_results,
                             provenance={r.sub_id: r.worker for r in sub_results})
        self._aggregate(result)
        return result

    def _materialize(self, st: SubTask, done: Dict[str, SubResult]) -> SubTask:
        """Inject upstream outputs into dependent subtasks (e.g. model_ref)."""
        if st.kind == "serve":
            best = self._best_model(done)
            st = SubTask(st.sub_id, st.kind, {**st.params, "model_ref": best},
                         st.partition, st.depends_on)
        return st

    # -- consensus + aggregation ------------------------------------------- #
    def _best_model(self, done: Dict[str, SubResult]) -> str:
        trains = [r for r in done.values() if r.kind == "train" and r.ok]
        if not trains:
            return "__no_model__"
        # Prefer higher accuracy, else lower rmse.
        def key(r):
            m = r.output.get("metrics", {})
            return (m.get("accuracy", -1), -m.get("rmse", 1e18))
        return max(trains, key=key).output.get("model_path", "__no_model__")

    def _aggregate(self, result: SwarmResult) -> None:
        # Merge feature analyses from all shards into one feature catalog.
        feats: Dict[str, dict] = {}
        for r in result.successes():
            if r.kind == "analyze_features":
                for f in r.output.get("features", []):
                    feats.setdefault(f["name"], f)  # de-dupe by column name
        result.aggregated["feature_catalog"] = feats

        # Consensus model: majority vote on model_kind among successful trainers,
        # and report the selected best model's metrics.
        trains = [r for r in result.successes() if r.kind == "train"]
        if trains:
            kinds = Counter(r.output.get("model_kind") for r in trains)
            winner_kind, votes = kinds.most_common(1)[0]
            best = max(trains, key=lambda r: (
                r.output.get("metrics", {}).get("accuracy", -1),
                -r.output.get("metrics", {}).get("rmse", 1e18),
            ))
            result.consensus = {
                "winner_kind": winner_kind,
                "votes": votes,
                "n_candidates": len(trains),
                "selected_metrics": best.output.get("metrics", {}),
                "selected_model_path": best.output.get("model_path"),
            }

        serve = [r for r in result.successes() if r.kind == "serve"]
        if serve:
            result.aggregated["endpoint"] = serve[0].output.get("endpoint")
            result.aggregated["ready"] = serve[0].output.get("ready", False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _shard(items: List[str], n: int) -> List[List[str]]:
    """Split ``items`` into ``n`` disjoint, balanced shards (order-stable)."""
    n = max(1, min(n, len(items) or 1))
    shards: List[List[str]] = [[] for _ in range(n)]
    for i, it in enumerate(items):
        shards[i % n].append(it)
    return [s for s in shards if s]


def build_swarm(n_workers: int = 3, client_factory: Optional[Callable[[], AvalokaClient]] = None,
                **worker_kwargs) -> MainAvaloka:
    """Convenience: build a homogeneous swarm of ``n_workers`` Avalokas."""
    factory = client_factory or (lambda: ReferenceClient())
    workers = [WorkerAvaloka(f"avaloka-{i}", client=factory(), **worker_kwargs)
               for i in range(n_workers)]
    return MainAvaloka(workers)
