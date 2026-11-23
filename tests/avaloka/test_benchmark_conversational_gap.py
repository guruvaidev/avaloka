"""The conversational measurement check passes on a measurement, not on a name.

``conv_avaloka_behaviour_is_unmeasured`` used to pass the moment
``avaloka.benchmark.conversational_product`` existed with a
``product_responder`` factory that returned -- a lazy factory passes in an
environment where the product cannot run at all. These pin the tightened rule:
it passes only when every probe was driven through the compiled graph, sent its
own question, reached a model, missed no recording, and the results came back.

Nothing here spawns the graph: the product run is replaced with a stub, which
is exactly the object the old check could not tell from the real thing.
"""

import sys
import types

import pytest

from avaloka.benchmark import conversational as conv
from avaloka.benchmark.conversational_product import ProbeRecord, ProductRun
from avaloka.benchmark.known_defects import KNOWN_DEFECTS, gate

GRAPH_PATH = ["drivers_route", "memory_injection", "avaloka_agent", "verify_claims"]


def _probes():
    return conv.load_probe_suite().module.PROBES


def _measured_records(**overrides):
    records = {}
    for probe in _probes():
        fields = dict(probe_id=probe.id, sent_question=probe.turns[-1].text,
                      reply="There are 20,640 rows.", nodes=list(GRAPH_PATH),
                      completed=True, calls=1, latency_s=0.01)
        fields.update(overrides)
        records[probe.id] = ProbeRecord(**fields)
    return records


def _install(monkeypatch, module):
    monkeypatch.setitem(sys.modules, conv.PRODUCT_MODULE, module)


def _stub_runner(monkeypatch, run):
    module = types.ModuleType(conv.PRODUCT_MODULE)
    module.REPLAY, module.LIVE = "replay", "live"
    module.run_product = lambda *a, **k: run
    _install(monkeypatch, module)


def _gap(results):
    (gap,) = [r for r in results if r.task == conv.GAP_TASK]
    return gap


def test_the_entry_is_gone_from_the_ledger():
    assert conv.GAP_TASK not in KNOWN_DEFECTS


def test_a_factory_alone_no_longer_clears_the_check(monkeypatch):
    """The base passed this: a module with a responder factory and no measurement."""
    stub = types.ModuleType(conv.PRODUCT_MODULE)
    stub.product_responder = lambda: (lambda probe, question: "canned")
    _install(monkeypatch, stub)
    gap = _gap(conv.conversational_results(live=False))
    assert gap.error is None, "a factory with no measurement is a failure, not a skip"
    assert not gap.passed
    assert "nothing drives the probes" in gap.checks[0].detail
    assert gate([gap]).new_failures == [f"{conv.GAP_TASK} (—)"]


def test_a_run_that_measured_nothing_fails(monkeypatch):
    _stub_runner(monkeypatch, ProductRun(mode="replay", label="product:replay",
                                         results_written=True, routing_replaced=True))
    gap = _gap(conv.conversational_results(live=False))
    assert not gap.passed
    assert "11/11 probes have no record" in gap.checks[0].detail


def test_replies_that_never_went_through_the_graph_fail(monkeypatch):
    """A responder answering every probe directly has replies and no graph."""
    _stub_runner(monkeypatch, ProductRun(
        mode="replay", label="product:replay", results_written=True, routing_replaced=True,
        records=_measured_records(nodes=[], completed=True)))
    gap = _gap(conv.conversational_results(live=False))
    assert not gap.passed
    assert "did not run through the graph" in gap.checks[0].detail


def test_a_run_with_no_model_behind_it_fails(monkeypatch):
    _stub_runner(monkeypatch, ProductRun(
        mode="replay", label="product:replay", results_written=True, routing_replaced=True,
        records=_measured_records(calls=0)))
    gap = _gap(conv.conversational_results(live=False))
    assert not gap.passed
    assert "no model call was made" in gap.checks[0].detail


@pytest.mark.parametrize("breakage,expected", [
    (dict(error="the graph run exited 1: Traceback ..."), "the run failed"),
    (dict(records_override=dict(misses=[{"agent": "CONVERSATIONAL", "key": "ab" * 32,
                                         "excerpt": "How many rows"}])),
     "model requests had no recording"),
    (dict(records_override=dict(completed=False, error="wall-clock budget of 60s exceeded")),
     "did not run through the graph"),
    (dict(records_override=dict(sent_question="")), "was not the one sent"),
    (dict(egress=["('203.0.113.7', 443)"]), "tried to reach the network"),
    (dict(routing_replaced=False), "router was not replaced"),
])
def test_a_broken_run_fails(monkeypatch, breakage, expected):
    records = _measured_records(**breakage.pop("records_override", {}))
    run = ProductRun(mode="replay", label="product:replay", results_written=True,
                     routing_replaced=True, records=records)
    for name, value in breakage.items():
        setattr(run, name, value)
    _stub_runner(monkeypatch, run)
    results = conv.conversational_results(live=False)
    gap = _gap(results)
    assert not gap.passed and expected in gap.checks[0].detail


def test_a_cassette_miss_fails_its_probe_task(monkeypatch):
    records = _measured_records()
    records["int_unambiguous_rowcount"].misses = [
        {"agent": "CONVERSATIONAL", "key": "cd" * 32, "excerpt": "How many rows"}]
    _stub_runner(monkeypatch, ProductRun(mode="replay", label="product:replay",
                                         results_written=True, routing_replaced=True,
                                         records=records))
    results = conv.conversational_results(live=False)
    (task,) = [r for r in results if r.task == "conv_product_interactivity"]
    (check,) = [c for c in task.checks if c.name == "int_unambiguous_rowcount::recorded"]
    assert not check.passed and "CONVERSATIONAL" in check.detail
    assert not task.passed


def test_no_application_is_a_skip_not_a_failure(monkeypatch):
    _stub_runner(monkeypatch, ProductRun(
        mode="replay", label="product:replay",
        unavailable="unavailable: app not installed (ModuleNotFoundError: langgraph)"))
    results = conv.conversational_results(live=False)
    product = [r for r in results if r.family == conv.PRODUCT_FAMILY or r.task == conv.GAP_TASK]
    assert len(product) == 5
    assert all(r.error and "unavailable" in r.error for r in product)
    verdict = gate(product)
    assert verdict.ok and not verdict.known_failing


def test_a_measured_run_passes_and_is_graded_on_its_replies(monkeypatch):
    _stub_runner(monkeypatch, ProductRun(mode="replay", label="product:replay",
                                         results_written=True, routing_replaced=True,
                                         records=_measured_records()))
    results = conv.conversational_results(live=False)
    gap = _gap(results)
    assert gap.passed, gap.checks[0].detail
    assert gap.source_scheme == "product:replay"
    by_task = {r.task: r for r in results}
    # The reply was graded, not the run: one canned line passes the row-count
    # probe and fails the rest.
    checks = {c.name: c.passed for c in by_task["conv_product_interactivity"].checks}
    assert checks["int_unambiguous_rowcount"] is True
    assert checks["int_ambiguous_revenue"] is False
    assert all(by_task[f"conv_product_{d}"].family == conv.PRODUCT_FAMILY
               for d in ("interactivity", "responsiveness", "competence", "integrity"))


def test_being_measured_does_not_raise_a_score(monkeypatch):
    """Every probe asked properly and every reply wrong is 0.00, not 0.75. The
    plumbing checks can still fail a task; they cannot earn it anything."""
    _stub_runner(monkeypatch, ProductRun(mode="replay", label="product:replay",
                                         results_written=True, routing_replaced=True,
                                         records=_measured_records(reply="nothing useful")))
    results = conv.conversational_results(live=False)
    product = [r for r in results if r.family == conv.PRODUCT_FAMILY]
    assert [r.score for r in product] == [0.0, 0.0, 0.0, 0.0]
    assert _gap(results).passed, "measured, and measured to be bad"
