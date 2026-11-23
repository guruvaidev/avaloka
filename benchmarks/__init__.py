"""Avaloka benchmark test suite.

See ``benchmarks.md`` (repo root) for the strategy and ``benchmarks/README.md`` for
the harness architecture. The public surface is intentionally small:

    from benchmarks.core.runner import BenchmarkRunner
    from benchmarks.core.config import RunConfig
    from benchmarks.core.registry import get_suite, list_suites

Nothing in this package imports the Avaloka application at module-import time, so the
harness can be exercised (and unit-tested) without provisioning cloud credentials or a
running LangGraph server. The Avaloka adapter loads the app lazily, only when a run
actually targets it.
"""

__version__ = "0.2.0"
