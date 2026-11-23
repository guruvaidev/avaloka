"""The smoke suite itself must be trustworthy.

A pre-flight check that cries wolf gets ignored, and an ignored gate is worse
than no gate. These tests pin the two properties that matter: it catches the
regression it was built for, and it does not fire on correct code.
"""

from __future__ import annotations

import pytest

from benchmarks.smoke import run_all
from benchmarks.smoke.result import Severity, SmokeReport, SmokeResult


def test_the_suite_covers_every_surface():
    report = run_all()
    assert set(report.by_surface()) == {"harness", "prompts", "loops", "graph", "models"}
    assert len(report.results) >= 18


def test_blocking_failures_always_name_their_evidence():
    """Whatever is red, it must be actionable.

    Not asserting safe_to_run here: on develop-1.7 the models surface is
    legitimately red (agents still reference models Groq removed), and a test
    that demanded green would push someone to weaken the check rather than fix
    the branch. What must hold is that every blocking failure says what to fix.
    """
    report = run_all()
    for r in report.blocking:
        assert r.detail, f"{r.check} failed without a detail message"


# --------------------------------------------------------------------------- #
# It catches the regression it exists for
# --------------------------------------------------------------------------- #

def test_the_trial_salt_regression_would_be_caught(monkeypatch):
    """The original bug: runner passed _trial as str, every adapter raised."""
    from benchmarks.smoke import harness

    class FakeRunner:
        def run_task(self):  # pragma: no cover - source is what is inspected
            trial_config = 'replace(config, extra={"_trial": str(i)})'

    import benchmarks.core.runner as runner_mod
    monkeypatch.setattr(runner_mod, "BenchmarkRunner", FakeRunner)
    result = harness._check_trial_salt_type()
    assert result.passed is False
    assert result.severity is Severity.CRITICAL


def test_trial_salt_check_passes_on_the_fixed_code():
    from benchmarks.smoke import harness
    assert harness._check_trial_salt_type().passed


def test_adapter_resolution_separates_unprovisioned_from_broken():
    """DAB needs external setup; that is not a defect and must not gate."""
    from benchmarks.smoke import harness
    r = harness._check_adapters_resolvable()
    assert r.passed, r.evidence
    assert r.evidence.get("needs_provisioning")   # DAB is reported, not counted as broken


# --------------------------------------------------------------------------- #
# It does not fire on correct code — the property it failed twice
# --------------------------------------------------------------------------- #

def test_anti_fabrication_check_does_not_flag_prohibitions():
    """Avaloka's prompts say 'do not invent'. That is the desired behaviour.

    An earlier keyword hunt flagged every prohibition and every identifier in
    the anti-fabrication machinery — 25 false positives on correct code.
    """
    from benchmarks.smoke import prompts
    r = prompts._check_anti_fabrication_guardrails()
    assert r.passed, r.evidence
    assert len(r.evidence["present"]) >= 3


def test_orphan_detection_follows_conditional_routing():
    """Execution nodes are routed via a mapping, not a direct edge."""
    from benchmarks.smoke import graph
    r = graph._check_no_orphan_nodes(graph._workflow_source())
    assert r.passed, r.evidence.get("orphans")


def test_loop_checks_pass_on_the_real_coder():
    from benchmarks.smoke import loops
    critical = [r for r in loops.run()
                if r.severity is Severity.CRITICAL and not r.passed]
    assert not critical, [r.check for r in critical]


# --------------------------------------------------------------------------- #
# Report semantics
# --------------------------------------------------------------------------- #

def test_a_warning_does_not_block():
    report = SmokeReport(results=[
        SmokeResult("w", "loops", passed=False, severity=Severity.WARNING),
    ])
    assert report.safe_to_run is True
    assert report.failures and not report.blocking


def test_a_critical_failure_blocks():
    report = SmokeReport(results=[
        SmokeResult("c", "harness", passed=False, severity=Severity.CRITICAL),
    ])
    assert report.safe_to_run is False


def test_report_serialises():
    import json
    json.dumps(run_all().as_dict())


# --------------------------------------------------------------------------- #
# CLI contract — this is what CI calls
# --------------------------------------------------------------------------- #

def test_cli_exit_code_reflects_blocking_failures():
    """1 when something CRITICAL failed, 0 otherwise. On develop-1.7 the dead
    model references make this 1, which is the correct signal to a pilot."""
    from benchmarks.smoke import run_all as _run
    from benchmarks.smoke.__main__ import main
    expected = 1 if _run().blocking else 0
    assert main([]) == expected


def test_cli_exit_two_is_reserved_for_a_broken_checker(monkeypatch):
    """'The suite could not run' must stay distinct from 'the system is broken'."""
    import benchmarks.smoke as smoke
    monkeypatch.setattr(smoke, "run_all",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("checker down")))
    from benchmarks.smoke.__main__ import main
    assert main([]) == 2


def _report_with(*results):
    from benchmarks.smoke.result import SmokeReport
    report = SmokeReport()
    report.extend(results)
    return report


def test_cli_strict_fails_on_warnings(monkeypatch):
    """--strict turns a WARNING into a failing exit; without it, it passes.

    This used to call main(["--strict"]) against the live tree and expect 1,
    which only held while develop-1.7 lacked app/core/inference.py (the one
    standing warning). Once the develop-1.6 merge supplied it, every check
    passed and the test failed for the right code. A warning is injected so
    the test exercises the flag, not the current state of the repository.
    """
    import benchmarks.smoke as smoke
    from benchmarks.smoke.__main__ import main
    from benchmarks.smoke.result import Severity, SmokeResult

    warn = SmokeResult(check="injected_warning", surface="harness", passed=False,
                       severity=Severity.WARNING, detail="test")
    monkeypatch.setattr(smoke, "run_all", lambda **_: _report_with(warn))
    assert main(["--strict"]) == 1
    assert main([]) == 0

    crit = SmokeResult(check="injected_critical", surface="harness", passed=False,
                       severity=Severity.CRITICAL, detail="test")
    monkeypatch.setattr(smoke, "run_all", lambda **_: _report_with(crit))
    assert main([]) == 1

    ok = SmokeResult(check="injected_pass", surface="harness", passed=True)
    monkeypatch.setattr(smoke, "run_all", lambda **_: _report_with(ok))
    assert main(["--strict"]) == 0


def test_cli_json_is_parseable(capsys):
    import json

    from benchmarks.smoke.__main__ import main
    main(["--json"])
    json.loads(capsys.readouterr().out)


def test_cli_can_limit_to_one_surface(capsys):
    from benchmarks.smoke.__main__ import main
    main(["--surface", "graph"])
    out = capsys.readouterr().out
    assert "graph" in out and "retry_ceiling_exists" not in out


# --------------------------------------------------------------------------- #
# models surface — the pre-condition for any benchmark run
# --------------------------------------------------------------------------- #

def test_models_surface_is_included():
    report = run_all()
    assert "models" in report.by_surface()


def test_dead_model_references_are_blocking():
    """A benchmark run against agents that 404 measures nothing.

    develop-1.7 currently references llama-3.3-70b-versatile and
    llama-3.1-8b-instant, both removed from the Groq account. This check is
    what stops a DAB run producing numbers that look like agent failures.
    """
    from benchmarks.smoke import models
    r = models._check_no_dead_models()
    assert r.severity is Severity.CRITICAL
    if not r.passed:
        assert r.evidence["hits"], "a failure must name the offending files"


def test_dead_model_list_is_not_empty():
    """The check is only meaningful if it knows what is dead."""
    from benchmarks.smoke.models import KNOWN_DEAD, KNOWN_LIVE
    assert KNOWN_DEAD and KNOWN_LIVE
    assert not (KNOWN_DEAD & KNOWN_LIVE), "a model cannot be both live and dead"


def test_chokepoint_check_is_a_warning_not_a_gate():
    """Direct client construction still works; it only caps observability."""
    from benchmarks.smoke import models
    assert models._check_single_inference_chokepoint().severity is Severity.WARNING
