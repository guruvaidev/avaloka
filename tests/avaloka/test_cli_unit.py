"""Unit tests for the Avaloka CLI components.

Covers pure functions and isolated component logic:
- Workload profiling and tier routing arithmetic (Online, Sampled, Batch)
- Budget limits, cost accumulation, and ceiling assertions
- CLI option preparation, execution mode mapping, and output path handling
- Persona initialization, formatting, and greeting logic
- Local data source resolution and dataset validation
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avaloka.cli import _mode, _prepare_output, _avaloka
from avaloka.mission.budget import Budget
from avaloka.version import __version__
from avaloka.workload import Lane, WorkloadPlan, classify, ONLINE_MAX_ROWS, ONLINE_MAX_BYTES


# ---------------------------------------------------------------------------
# 1. Workload Profiling Unit Tests
# ---------------------------------------------------------------------------

def test_workload_classification_boundaries():
    """Verify lane boundary arithmetic for Online, Sampled, and Batch."""
    # Online: <= 100,000 rows and <= 50 MB
    p_online = classify(ONLINE_MAX_ROWS, ONLINE_MAX_BYTES)
    assert p_online.lane == Lane.ONLINE
    assert not p_online.recommend_batch

    # Sampled: > 100k rows or > 50MB, but within 5M rows / 1GB
    p_sampled = classify(ONLINE_MAX_ROWS + 1, 1024)
    assert p_sampled.lane == Lane.SAMPLED_ONLINE
    assert p_sampled.recommend_batch

    # Batch: > 5,000,000 rows
    p_batch = classify(5_000_001, 1024)
    assert p_batch.lane == Lane.BATCH
    assert p_batch.recommend_batch


def test_workload_profile_serialization():
    """WorkloadPlan serializes cleanly to dict and JSON."""
    plan = classify(500, 1024 * 1024)
    d = plan.as_dict()
    assert d["lane"] == "online"
    assert d["n_rows"] == 500
    assert d["n_bytes"] == 1024 * 1024
    assert not d["recommend_batch"]

    # JSON round-trip
    dumped = json.dumps(d)
    assert "online" in dumped


# ---------------------------------------------------------------------------
# 2. Budget and Economics Unit Tests
# ---------------------------------------------------------------------------

def test_budget_initialization_and_defaults():
    """Budget defaults to None (unconstrained), with proper spent tracking."""
    b = Budget()
    assert b.limit_usd is None
    assert b.spent_usd == 0.0
    assert b.unlimited
    assert b.remaining() == float("inf")
    assert b.can_afford(100.0)

    # Zero ceiling is a real limit, not unconstrained
    b_zero = Budget(limit_usd=0.0)
    assert b_zero.limit_usd == 0.0
    assert not b_zero.unlimited
    assert b_zero.remaining() == 0.0
    assert not b_zero.can_afford(0.01)


def test_budget_cost_accumulation():
    """Recording costs correctly accumulates total spent and respects ceilings."""
    from avaloka.mission.budget import BudgetExceeded

    b = Budget(limit_usd=10.0)
    b.spend(2.50)
    b.spend(3.75)
    assert pytest.approx(b.spent_usd) == 6.25
    assert pytest.approx(b.remaining()) == 3.75
    assert b.can_afford(3.75)
    assert not b.can_afford(4.00)

    with pytest.raises(BudgetExceeded):
        b.spend(4.00)


# ---------------------------------------------------------------------------
# 3. CLI Helper and Mode Resolution Unit Tests
# ---------------------------------------------------------------------------

def test_execution_mode_mapping():
    """_mode resolves correctly across local, byoc, and managed flags."""
    from avaloka.mission.context import ExecutionMode

    assert _mode(True, None) == ExecutionMode.LOCAL
    assert _mode(False, "managed") == ExecutionMode.MANAGED
    assert _mode(False, "byoc") == ExecutionMode.BYOC
    assert _mode(True, "managed") == ExecutionMode.LOCAL  # local overrides
    assert _mode(False, "invalid_mode") == ExecutionMode.LOCAL  # fallback


def test_prepare_output_directory(tmp_path):
    """_prepare_output creates directories and expands paths."""
    target = tmp_path / "deep" / "nested" / "dir"
    prepared = _prepare_output(target)
    assert prepared == target
    assert target.exists()
    assert target.is_dir()


def test_persona_initialization():
    """_avaloka returns Ava persona instance or None in quiet mode."""
    # Quiet mode silences the persona
    assert _avaloka(quiet=True, llm=False) is None
    assert _avaloka(quiet=True, llm=True) is None

    # Normal mode initializes Ava
    ava = _avaloka(quiet=False, llm=False)
    assert ava is not None
    greeting = ava.greet()
    assert isinstance(greeting, str)
    assert len(greeting) > 0


def test_version_format():
    """CLI version string follows semantic versioning."""
    import re
    assert re.match(r"^\d+\.\d+\.\d+", __version__), f"Invalid version: {__version__}"
