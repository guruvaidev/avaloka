"""NEW-RS-01: the RayJob renderer applies the requested worker count to the SHIPPED template.

Found while writing K8-10/K8-11 (1.6 release suite, 2026-10-31); not in any
source sheet, so it carries a release-suite ID. ``render_rayjob_yaml``
(app/agents/execution_agent.py) rewrites ``replicas:`` / ``maxReplicas:`` with
patterns anchored at end of line (``\\s*$``). The shipped template
app/infra/rayjob.yaml ends those lines with comments (``replicas: 2  # start
with 2 workers``), so the rewrite never matches: whatever worker count the
planner asks for, the job starts with 2 workers and a ceiling of 10. The
renderer tests in tests/test_ray_unit.py strip the comments first, which is why
nothing caught it. Fixed by the RayJob worker-count hotfix
(hotfix/1.6-rayjob-worker-count): the rewrites anchor on the key and allow a
trailing comment. The strict xfail below is now a plain test that keeps it
fixed.
"""
from __future__ import annotations

import re
from pathlib import Path


from app.agents.execution_agent import render_rayjob_yaml

TEMPLATE = Path(__file__).resolve().parents[2] / "app" / "infra" / "rayjob.yaml"


def _render(workers: int) -> str:
    return render_rayjob_yaml(
        TEMPLATE.read_text(),
        name="newrs01",
        namespace="ns-newrs01",
        workers=workers,
        data_uri="gs://example-bucket/data.csv",
        script_text="print('x')\n",
    )


def _worker_group(text: str) -> str:
    block = text.split("workerGroupSpecs:", 1)
    assert len(block) == 2, "the shipped template has no workerGroupSpecs block"
    return block[1]


def test_newrs01_shipped_template_still_renders_a_worker_group():
    """NEW-RS-01 (control): the shipped template renders, and the namespace rewrite applies."""
    out = _render(3)
    assert "namespace: ns-newrs01" in out
    assert re.search(r"(?m)^\s*maxReplicas:\s*\d+", _worker_group(out))


# NEW-RS-01: converted from a strict xfail after the worker-count hotfix.
def test_newrs01_requested_worker_count_reaches_the_shipped_template():
    """NEW-RS-01: rendering with workers=3 sets the worker group's replicas and maxReplicas to 3."""
    group = _worker_group(_render(3))
    assert re.search(r"(?m)^\s*replicas:\s*3\b", group)
    assert re.search(r"(?m)^\s*maxReplicas:\s*3\b", group)
