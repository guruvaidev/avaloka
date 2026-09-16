"""
Quarantine guard for tests written against APIs that no longer exist.

Several suites still import symbols that were removed or restructured. Because
those imports fail at *collection* time, one stale module aborts the entire run
(`Interrupted: N errors during collection`) — which is why CI has historically
needed `--continue-on-collection-errors` and `|| true` to appear green.

`requires_api` turns that hard collection error into a skip carrying a specific
reason, so the rest of the suite runs and `pytest -rs` lists exactly what is
quarantined and what it should be repaired against. This is a holding position,
not a resting place: each call site is a test that currently covers nothing.
"""

from __future__ import annotations

import importlib

import pytest


def requires_api(module: str, *names: str, replacement: str) -> None:
    """Skip the calling module unless `module` exists and exposes every name in `names`.

    `replacement` states what the old API became, so whoever picks the file up
    knows whether to repair it or delete it.
    """
    try:
        imported = importlib.import_module(module)
    except ImportError as exc:
        pytest.skip(
            f"QUARANTINED: cannot import {module} ({exc}). {replacement}",
            allow_module_level=True,
        )
        return

    missing = [name for name in names if not hasattr(imported, name)]
    if missing:
        pytest.skip(
            f"QUARANTINED: {module} no longer provides {', '.join(missing)}. {replacement}",
            allow_module_level=True,
        )
