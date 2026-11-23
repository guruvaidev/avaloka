"""Pre-flight smoke checks: is the system sound enough to trust a run?

Distinct from the benchmark tracks. A track measures *how good* Avaloka is; a
smoke check answers *can we believe the measurement at all*. Run these first —
a red smoke suite invalidates every number produced after it.

The distinction is not academic. A runner/adapter type mismatch once made every
baseline trial raise inside the adapter; the runner recorded each as a failed
trial, and the harness reported a legitimate-looking 0.0 quality score. Anyone
reading that would have investigated the agent. The bug was in the harness.

Three surfaces, because these are the three places Avaloka can be quietly wrong:

* :mod:`benchmarks.smoke.harness` — is the measuring instrument itself sound?
* :mod:`benchmarks.smoke.prompts` — do the prompts still contain the
  instructions that downstream parsing depends on?
* :mod:`benchmarks.smoke.loops`   — do the retry loops terminate, and do they
  make progress rather than oscillate?
* :mod:`benchmarks.smoke.graph`   — is the agent graph wired so every node's
  inputs exist before it runs?
* :mod:`benchmarks.smoke.models`  — can the agents reach a model at all, and is
  their traffic observable in one place?
"""

from __future__ import annotations

from benchmarks.smoke.result import Severity, SmokeReport, SmokeResult

__all__ = ["Severity", "SmokeReport", "SmokeResult", "run_all"]


def run_all(*, include_slow: bool = False) -> SmokeReport:
    """Run every smoke surface and collect the results."""
    from benchmarks.smoke import graph, harness, loops, models, prompts

    report = SmokeReport()
    for module in (harness, prompts, loops, graph, models):
        report.extend(module.run(include_slow=include_slow))
    return report
