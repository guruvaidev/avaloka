"""Model discovery: sizing arithmetic, family detection, offline behaviour."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "discover_models", Path(__file__).resolve().parent.parent / "scripts/ops/discover_models.py")
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)


# ---- sizing: the recommendation must show its arithmetic -------------------

@pytest.mark.parametrize("model_id,expected", [
    ("google/gemma-4-31b-it", 31.0),
    ("gemma3:4b", 4.0),
    ("qwen/qwen3.8-27b", 27.0),
    ("google/gemma-3n-e4b-it", 4.0),          # e4b = 4B ACTIVE params
    ("google/gemma-4-26b-a4b-it", 4.0),       # a4b MoE: active, not total
    ("groq/compound", None),                  # no size in the id
])
def test_param_parsing(model_id, expected):
    assert dm.params_b(model_id) == expected


def test_moe_models_sized_by_active_params_not_total():
    """A 26B MoE with 4B active fits a laptop; sizing by 26 would forbid it."""
    assert dm.fits("google/gemma-4-26b-a4b-it", budget_gb=10) is True


def test_31b_does_not_fit_a_small_budget_but_fits_half_of_64gb():
    assert dm.fits("gemma4:31b", budget_gb=8) is False
    assert dm.fits("gemma4:31b", budget_gb=32) is True


def test_unknown_size_is_none_not_a_guess():
    assert dm.fits("groq/compound", budget_gb=32) is None


def test_memory_env_override_wins(monkeypatch):
    monkeypatch.setenv("AVALOKA_MEM_GB", "16")
    assert dm.machine_memory_gb() == 16.0


# ---- discovery: offline must degrade, not crash ----------------------------

def test_offline_discovery_reports_empty_not_raises(monkeypatch):
    monkeypatch.setattr(dm, "_get_json", lambda *a, **k: None)
    report = dm.discover(("gemma",), budget_gb=8)
    assert report["families"]["gemma"]["catalogue_count"] == 0
    assert report["local"] == []


def test_new_family_version_appears_without_code_change(monkeypatch):
    """The gemma5 scenario: a new id in the catalogue is discovered as-is."""
    monkeypatch.setattr(dm, "_get_json", lambda url, **k: (
        {"data": [{"id": "google/gemma-5-12b-it"}]} if "openrouter" in url
        else {"models": []}))
    report = dm.discover(("gemma",), budget_gb=32)
    assert "gemma-5" in report["families"]["gemma"]["known_versions"]
    assert report["families"]["gemma"]["largest_that_fits"] == "google/gemma-5-12b-it"


def test_report_is_json_serialisable():
    import json
    json.dumps(dm.discover(("gemma",), budget_gb=1))  # tiny budget, no net needed? uses net; tolerate
