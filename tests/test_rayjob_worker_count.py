"""NEW-RS-01: render_rayjob_yaml applies the requested worker count.

The shipped app/infra/rayjob.yaml ends its count lines in comments
(``replicas: 2  # start with 2 workers``). The renderer's rewrites were
anchored at end of line, so they never matched that template and every job
started with its 2 workers and ceiling of 10, whatever ``workers`` asked for.
The tests in tests/test_ray_unit.py render a comment-free template, which is
why nothing caught it; these render the shipped file and parse the result.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
import yaml

from app.agents.execution_agent import render_rayjob_yaml

SHIPPED = Path(__file__).resolve().parents[1] / "app" / "infra" / "rayjob.yaml"


def _render(template: str, workers: int) -> str:
    return render_rayjob_yaml(
        template,
        name="worker-count",
        namespace="ns-worker-count",
        workers=workers,
        data_uri="gs://example-bucket/data.csv",
        script_text="print('x')\n",
    )


def _worker_groups(text: str) -> list:
    docs = [d for d in yaml.safe_load_all(text) if d]
    rayjob = next(d for d in docs if d.get("kind") == "RayJob")
    return rayjob["spec"]["rayClusterSpec"]["workerGroupSpecs"]


@pytest.mark.parametrize("workers", [1, 3, 7])
def test_shipped_template_takes_the_requested_worker_count(workers):
    template = SHIPPED.read_text()
    # The shape this test exists for: the shipped count lines carry comments.
    assert re.search(r"(?m)^\s*replicas:\s*\d+\s*#", template)
    assert re.search(r"(?m)^\s*maxReplicas:\s*\d+\s*#", template)

    out = _render(template, workers)
    groups = _worker_groups(out)
    assert groups, "the shipped template has no worker group"
    for group in groups:
        assert group["replicas"] == workers, group["groupName"]
        assert group["maxReplicas"] == workers, group["groupName"]
        # minReplicas has its own rule; it must still leave a valid range.
        assert int(group["minReplicas"]) <= group["replicas"], group["groupName"]


def test_shipped_template_render_leaves_no_placeholder_or_stale_count_comment():
    out = _render(SHIPPED.read_text(), 3)
    assert re.findall(r"__[A-Z0-9_]+__", out) == []
    assert "start with 2 workers" not in out
    assert "scale UP to 10" not in out


COMMENTED = textwrap.dedent("""\
    apiVersion: ray.io/v1
    kind: RayJob
    metadata:
      name: rayjob-sample-3w
      namespace: ray-jobs
    spec:
      rayClusterSpec:
        workerGroupSpecs:
          - groupName: commented
            replicas: 2           # start with 2 workers
            minReplicas: 1        # scale DOWN to 1
            maxReplicas: 10       # scale UP to 10
          - groupName: tight
            replicas:4#no space before the comment
            maxReplicas:	8	# tab-separated

          # replicas: 9   (a commented-out line is not a count line)
          - groupName: plain
            replicas: 1
            maxReplicas: 2
        headGroupSpec:
          serveConfig:
            num_replicas: 1  # a different key
    ---
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: rayjob-sample-3w-code
    data:
      sample_code.py: |
        print("placeholder")
    """)


def test_count_lines_with_trailing_comments_are_rewritten():
    out = _render(COMMENTED, 5)
    counts = re.findall(r"(?m)^[ \t]*(replicas|maxReplicas):(.*)$", out)
    assert counts == [("replicas", " 5"), ("maxReplicas", " 5")] * 3, counts
    groups = {g["groupName"]: g for g in _worker_groups(out)}
    for name in ("commented", "tight", "plain"):
        assert groups[name]["replicas"] == 5, name
        assert groups[name]["maxReplicas"] == 5, name

    lines = out.splitlines()
    assert "        replicas: 5" in lines  # indentation kept, comment dropped
    assert "      # replicas: 9   (a commented-out line is not a count line)" in lines
    assert "        num_replicas: 1  # a different key" in lines
    # Blank lines survive: nothing outside the count lines was rewritten.
    assert out.count("\n\n") == COMMENTED.count("\n\n")


def test_interactive_profile_shrinks_count_lines_that_carry_comments():
    """The interactive profile gives one worker that idles to zero, even when
    the count lines it is handed still carry comments. It used to work only
    because render_rayjob_yaml had already stripped them."""
    from app.agents.execution_agent import _apply_interactive_profile_to_yaml

    out = _apply_interactive_profile_to_yaml(COMMENTED)
    # The profile rewrites the first count line of each kind: the
    # "commented" group's, every one of which ends in a comment.
    group = out.split("- groupName: commented", 1)[1].split("- groupName:", 1)[0]
    counts = re.findall(r"(?m)^[ \t]*(replicas|minReplicas|maxReplicas):(.*)$", group)
    assert counts == [("replicas", " 1"), ("minReplicas", " 0"), ("maxReplicas", " 1")], counts
