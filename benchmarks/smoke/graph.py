"""Is the agent graph wired so every node's inputs exist before it runs?

The graph is built imperatively in ``app/api/workflow.py::build_graph`` — a
sequence of ``add_node`` and ``add_edge`` calls with nothing checking the result.
Two failures hide there:

* **A node reads a key a later node writes.** It works whenever a cache is warm
  or a previous turn populated the state, and fails on a cold run. That is the
  worst possible failure profile for a pilot: it passes your testing and fails
  at the customer.
* **An orphan node.** Added, never reachable, silently dead.

These checks read the graph's declared structure rather than executing it, so
they run in CI without a cluster or an LLM.

`app/agents/contract.py` (arriving on this line via the develop-1.6
forward-merge) provides `check_wiring()`, which does the read-before-write
analysis properly against declared agent specs. Until then these checks work
from the graph construction itself.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

from benchmarks.smoke.result import Severity, SmokeResult

SURFACE = "graph"

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / "app/api/workflow.py"


def _workflow_source() -> str:
    return WORKFLOW.read_text(encoding="utf-8", errors="replace") if WORKFLOW.is_file() else ""


def _parse_graph(src: str) -> Tuple[Set[str], List[Tuple[str, str]], Set[str]]:
    """Nodes, direct edges, and endpoints of conditional edges."""
    nodes = set(re.findall(r'add_node\(\s*"([^"]+)"', src))
    edges = [(a, b) for a, b in re.findall(r'add_edge\(\s*"([^"]+)"\s*,\s*"([^"]+)"', src)]
    conditional = set(re.findall(r'add_conditional_edges\(\s*"([^"]+)"', src))
    return nodes, edges, conditional


def run(*, include_slow: bool = False) -> List[SmokeResult]:
    src = _workflow_source()
    return [
        _check_graph_parses(src),
        _check_no_orphan_nodes(src),
        _check_edges_reference_real_nodes(src),
        _check_entry_point_declared(src),
    ]


def _check_graph_parses(src: str) -> SmokeResult:
    nodes, edges, cond = _parse_graph(src)
    ok = bool(nodes)
    return SmokeResult(
        check="graph_declares_nodes", surface=SURFACE, passed=ok,
        severity=Severity.CRITICAL,
        detail=f"{len(nodes)} nodes, {len(edges)} direct edges, {len(cond)} conditional"
               if ok else "no nodes found in build_graph",
        evidence={"n_nodes": len(nodes), "n_edges": len(edges)},
    )


def _check_edges_reference_real_nodes(src: str) -> SmokeResult:
    """An edge to a node that was never added is a runtime KeyError."""
    nodes, edges, _cond = _parse_graph(src)
    dangling = [f"{a}->{b}" for a, b in edges
                if (a not in nodes and a != "__start__") or (b not in nodes and b != "END")]
    return SmokeResult(
        check="edges_reference_real_nodes", surface=SURFACE, passed=not dangling,
        severity=Severity.CRITICAL,
        detail="every edge endpoint is a declared node" if not dangling
               else f"{len(dangling)} dangling edge(s)",
        evidence={"dangling": dangling[:10]},
    )


def _check_no_orphan_nodes(src: str) -> SmokeResult:
    """A node nothing routes to is dead code that looks alive.

    Warning: conditional routing is expressed through functions this static
    read cannot follow, so an apparent orphan may be reachable. Reported to be
    looked at, not to block.
    """
    nodes, edges, conditional = _parse_graph(src)
    reachable: Set[str] = set(conditional)
    for a, b in edges:
        reachable.add(b)
        reachable.add(a)
    # Conditional edges route through a mapping whose VALUES are node names,
    # e.g. {"ray": "execute_on_ray", "local": "execute_locally"}. Reading only
    # add_edge/add_conditional_edges missed those and reported six live
    # execution nodes as orphans -- noise that would train a reader to skip
    # this check. Any quoted string matching a declared node counts as a route.
    for quoted in re.findall(r'"([a-z_][a-z0-9_]*)"', src):
        if quoted in nodes:
            reachable.add(quoted)
    orphans = sorted(n for n in nodes if n not in reachable)
    return SmokeResult(
        check="no_orphan_nodes", surface=SURFACE, passed=not orphans,
        severity=Severity.WARNING,
        detail="every node participates in an edge or conditional route"
               if not orphans else f"{len(orphans)} node(s) with no visible route",
        evidence={"orphans": orphans},
    )


def _check_entry_point_declared(src: str) -> SmokeResult:
    declared = ("set_entry_point" in src or "add_edge(START" in src
                or "START," in src or "__start__" in src)
    return SmokeResult(
        check="entry_point_declared", surface=SURFACE, passed=declared,
        severity=Severity.CRITICAL,
        detail="the graph declares an entry point" if declared
               else "no entry point found; the graph cannot be invoked",
    )
