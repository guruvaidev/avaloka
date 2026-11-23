"""
Avaloka Data Engineering & Data Science Benchmark
=================================================

A capability benchmark + swarm-intelligence test harness for the Avaloka CLI.

The benchmark drives the CLI through a stable *Python entrypoint* contract
(`tests.benchmark.adapters.AvalokaClient`) over real datasets pulled from Kaggle
(with deterministic synthetic fallback so it runs in plain CI), exercising the
full data lifecycle the CLI is meant to automate:

    sample -> analyze features -> build features -> transform -> train -> serve

It also contains *swarm* tests that model multiple Avaloka agents (open-claw based,
each with planner/coder/etc. sub-agents) coordinating under one main Avaloka agent
on the same dataset.

Layout
------
    registry.py        Kaggle dataset registry (refs + synthetic schemas)
    datasets.py        Downloader (kagglehub) + deterministic synthetic fallback
    sources.py         Multi-source connectors: local / db / gcs / s3 / azure
    adapters.py        AvalokaClient contract + resolver + reference client
    tasks.py           DE/DS task spec dataclasses
    scoring.py         Metrics, thresholds, scorecard
    ml_train.py        Heavy real DL training path + baseline oracle
    swarm.py           Main-avaloka coordinator + worker avalokas
    catalog/           Concrete DE/DS tasks + swarm scenarios
    test_*.py          pytest suites
    run_benchmark.py   Standalone runner / report generator
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
