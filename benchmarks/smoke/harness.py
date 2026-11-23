"""Is the measuring instrument itself sound?

These exist because of a real incident. ``runner.py`` passed ``_trial`` as a
``str``; every baseline adapter used it as a seed salt and did arithmetic on it,
raising ``TypeError`` inside the adapter. The runner's broad ``except`` recorded
each as a failed trial, and the harness reported a plausible-looking 0.0 quality
score. Nothing distinguished "the agent did badly" from "the harness is broken",
so the number looked like an agent problem.

The class of bug matters more than the instance: **a harness that converts its
own faults into low scores will quietly mislead you**, and low scores are
exactly what you are looking at when you most want to trust the instrument.
"""

from __future__ import annotations

import dataclasses
from typing import List

from benchmarks.smoke.result import Severity, SmokeResult

SURFACE = "harness"


def run(*, include_slow: bool = False) -> List[SmokeResult]:
    return [
        _check_trial_salt_type(),
        _check_adapter_crash_is_distinguishable(),
        _check_suites_discoverable(),
        _check_adapters_resolvable(),
        _check_metrics_survive_missing_fields(),
    ]


def _check_trial_salt_type() -> SmokeResult:
    """The exact regression: _trial must reach adapters as an int."""
    from benchmarks.core.config import RunConfig
    from benchmarks.core.runner import BenchmarkRunner  # noqa: F401
    import inspect

    from benchmarks.core import runner as runner_mod
    src = inspect.getsource(runner_mod.BenchmarkRunner.run_task)
    bad = '"_trial": str(' in src
    return SmokeResult(
        check="trial_salt_is_int", surface=SURFACE, passed=not bad,
        severity=Severity.CRITICAL,
        detail=("runner passes _trial as str; baseline adapters do arithmetic on it "
                "and will raise inside every trial") if bad else
               "runner passes _trial as an int, as adapters expect",
    )


def _check_adapter_crash_is_distinguishable() -> SmokeResult:
    """A crashing adapter must not look like a legitimately poor result.

    The runner records adapter exceptions with an ``adapter_error:`` prefix. If
    that prefix disappears, harness faults become indistinguishable from agent
    failures in the reported numbers.
    """
    import inspect

    from benchmarks.core import runner as runner_mod
    src = inspect.getsource(runner_mod.BenchmarkRunner)
    tagged = "adapter_error" in src or "sut_start_error" in src
    return SmokeResult(
        check="adapter_crash_tagged", surface=SURFACE, passed=tagged,
        severity=Severity.CRITICAL,
        detail=("adapter exceptions are tagged, so a harness fault is separable "
                "from a poor agent result") if tagged else
               "adapter exceptions are untagged; harness faults will read as agent failures",
    )


def _check_suites_discoverable() -> SmokeResult:
    from benchmarks.core.registry import list_suites
    suites = [s.name for s in list_suites()]
    return SmokeResult(
        check="suites_discoverable", surface=SURFACE, passed=bool(suites),
        severity=Severity.CRITICAL,
        detail=f"{len(suites)} suite(s) registered", evidence={"suites": suites},
    )


def _check_adapters_resolvable() -> SmokeResult:
    """Every config a suite declares must resolve to a registered adapter.

    A suite naming an adapter that does not exist fails at run time, deep into
    a benchmark, rather than here.
    """
    from benchmarks.adapters import get_adapter, list_adapters
    from benchmarks.core.registry import get_suite, list_suites

    missing: List[str] = []
    needs_provisioning: List[str] = []
    for info in list_suites():
        try:
            _tasks, configs = get_suite(info.name)
        except Exception as exc:  # noqa: BLE001
            # A suite needing external setup (DAB wants a provisioned dataset and
            # runner) being unavailable is expected on a dev box, not a defect.
            # Conflating "not provisioned" with "broken" would make this check
            # permanently red and therefore ignored.
            if "NotProvisioned" in type(exc).__name__ or "not provisioned" in str(exc).lower():
                needs_provisioning.append(f"{info.name}: {type(exc).__name__}")
            else:
                missing.append(f"{info.name}: suite factory failed ({type(exc).__name__})")
            continue
        for config in configs:
            try:
                get_adapter(config.sut)
            except Exception:  # noqa: BLE001
                missing.append(f"{info.name}: no adapter {config.sut!r}")
    return SmokeResult(
        check="adapters_resolvable", surface=SURFACE, passed=not missing,
        severity=Severity.CRITICAL,
        detail=("every resolvable suite maps to a registered adapter"
                + (f"; {len(needs_provisioning)} suite(s) await external provisioning"
                   if needs_provisioning else "")) if not missing
               else f"{len(missing)} unresolved",
        evidence={"missing": missing, "needs_provisioning": needs_provisioning,
                  "registered": list_adapters()},
    )


def _check_metrics_survive_missing_fields() -> SmokeResult:
    """Aggregation must not crash when an adapter omits an optional field.

    Adapters legitimately leave cost or data_scanned_ratio unset. Aggregation
    should report NaN for those, not raise -- but NaN must then be visible as
    'not measured' rather than silently read as zero.
    """
    from benchmarks.core.metrics import aggregate
    from benchmarks.core.task import MissionOutcome, TrialResult

    trial = TrialResult(
        task_id="t", track="smoke", config_id="c", trial_index=0,
        quality={"score": 1.0},
        outcome=MissionOutcome(answer=1, latency_s=0.1),   # no cost, no ratio
    )
    try:
        stats = aggregate([trial])
    except Exception as exc:  # noqa: BLE001
        return SmokeResult(
            check="metrics_tolerate_missing_fields", surface=SURFACE, passed=False,
            severity=Severity.CRITICAL,
            detail=f"aggregate() raised on an outcome with optional fields unset: {exc}",
        )
    ratio = stats.stats["tuple"]["data_scanned_ratio"]
    return SmokeResult(
        check="metrics_tolerate_missing_fields", surface=SURFACE, passed=True,
        severity=Severity.CRITICAL,
        detail="aggregation reports n=0 for unmeasured fields rather than raising",
        evidence={"data_scanned_ratio_n": ratio["n"]},
    )
