"""Smoke tests: the harness runs end-to-end on Track 1 with zero external deps.

These are pure-stdlib and must pass in CI (`pytest tests/` picks up the thin
forwarder at repo-root `tests/test_benchmarks_smoke.py`). They assert the *machinery*
works and that the Track-1 metric story holds directionally: full-data is exact,
sampling scans less, adaptive keeps quality high while scanning a fraction.
"""

from __future__ import annotations

from benchmarks.core.metrics import (
    aggregate,
    population_stability_index,
    quality_from_error,
    rare_event_recall,
    relative_error,
)
from benchmarks.core.registry import get_suite, list_suites
from benchmarks.core.runner import BenchmarkRunner


def test_suites_are_discoverable():
    names = {s.name for s in list_suites()}
    assert "track1_adaptive_sampling" in names
    assert "dab" in names  # registered even though not yet loadable


def test_metric_primitives():
    assert relative_error(100, 100) == 0.0
    assert quality_from_error(0.0) == 1.0
    assert quality_from_error(1.0) == 0.0
    assert rare_event_recall([1, 2], [1, 2, 3, 4]) == 0.5
    # Identical distributions → ~0 divergence.
    d = {"a": 0.5, "b": 0.5}
    assert population_stability_index(d, d) < 1e-9


def test_track1_runs_and_aggregates():
    tasks, configs = get_suite("track1_adaptive_sampling")
    assert tasks and configs
    runner = BenchmarkRunner()
    # Keep it fast: 2 trials.
    configs = [__import__("dataclasses").replace(c, trials=2) for c in configs]
    results = runner.run_suite(tasks, configs)
    assert len(results) == len(tasks) * len(configs)
    for r in results:
        assert r.stats["n_trials"] == 2
        assert 0.0 <= r.stats["pass_at_1"] <= 1.0
        assert "tuple" in r.stats


def test_full_data_reference_is_exact_and_scans_everything():
    tasks, configs = get_suite("track1_adaptive_sampling")
    full = [c for c in configs if c.sut == "full_data_reference"]
    mean_task = [t for t in tasks if t.id == "t1_mean_amount"][0]
    runner = BenchmarkRunner()
    res = runner.run_task(mean_task, __import__("dataclasses").replace(full[0], trials=1))
    trial = res.trials[0]
    assert trial.outcome.data_scanned_ratio == 1.0
    assert trial.quality["answer_rel_error"] < 1e-9  # exact on full data


def test_adaptive_scans_less_than_full_but_stays_accurate():
    tasks, configs = get_suite("track1_adaptive_sampling")
    adaptive = [c for c in configs if c.sut == "adaptive_sample"][0]
    mean_task = [t for t in tasks if t.id == "t1_mean_amount"][0]
    runner = BenchmarkRunner()
    res = runner.run_task(mean_task, __import__("dataclasses").replace(adaptive, trials=3))
    scanned = res.stats["tuple"]["data_scanned_ratio"]["mean"]
    assert scanned < 1.0  # the whole point: less than full
    # Quality stays high (within a loose bound so the test isn't flaky).
    assert res.stats["quality"]["score"]["mean"] > 0.8


def test_domain_sample_beats_random_on_rare_recall():
    tasks, configs = get_suite("track1_adaptive_sampling")
    fraud_task = [t for t in tasks if t.id == "t1_fraud_recall"][0]
    dataclasses = __import__("dataclasses")
    runner = BenchmarkRunner()
    domain = [c for c in configs if c.sut == "domain_sample"][0]
    rnd = [c for c in configs if c.sut == "random_sample"][0]
    d_res = runner.run_task(fraud_task, dataclasses.replace(domain, trials=3))
    r_res = runner.run_task(fraud_task, dataclasses.replace(rnd, trials=3))
    d_recall = d_res.stats["quality"]["rare_event_recall"]["mean"]
    r_recall = r_res.stats["quality"]["rare_event_recall"]["mean"]
    assert d_recall >= r_recall  # domain oversampling retains the rare segment
