"""Suite discovery: a broken plugin must not take the harness down."""

from __future__ import annotations

import sys
import types

import pytest

from benchmarks.core import plugins
from benchmarks.core.plugins import (BUILTIN_MODULES, ENTRY_POINT_GROUP,
                                     LoadReport, discover_local_tracks,
                                     load_all)


def test_builtin_modules_load():
    report = load_all(include_entry_points=False)
    assert report.ok, f"built-in suites failed: {report.failed}"
    for module in BUILTIN_MODULES:
        assert module in report.loaded


def test_local_tracks_are_found_by_walking_the_directory(tmp_path):
    """Adding a track becomes creating a folder, not editing the registry."""
    tracks = tmp_path / "tracks"
    (tracks / "my_new_track").mkdir(parents=True)
    (tracks / "my_new_track" / "track.py").write_text("# suite")
    (tracks / "not_a_track").mkdir()          # no track.py
    (tracks / "__pycache__").mkdir()

    found = discover_local_tracks(tracks)
    assert found == ["benchmarks.tracks.my_new_track.track"]


def test_discovery_finds_the_tracks_shipped_here():
    found = discover_local_tracks()
    assert any("adaptive_sampling" in m for m in found)
    assert any("reliability" in m for m in found)


def test_missing_tracks_directory_is_not_an_error(tmp_path):
    assert discover_local_tracks(tmp_path / "absent") == []


# --------------------------------------------------------------------------- #
# One broken plugin must not sink the harness
# --------------------------------------------------------------------------- #

def test_a_plugin_that_cannot_import_is_reported_not_raised():
    report = load_all(include_entry_points=False,
                      extra_modules=("benchmarks.tracks.does_not_exist.track",))
    assert not report.ok
    assert any("does_not_exist" in k for k in report.failed)
    # ...and everything else still loaded
    assert len(report.loaded) >= len(BUILTIN_MODULES)


def test_a_plugin_raising_at_import_time_is_contained(monkeypatch):
    broken = types.ModuleType("broken_suite")
    def explode():
        raise RuntimeError("bad suite")
    monkeypatch.setitem(sys.modules, "broken_suite", broken)

    real_import = plugins.importlib.import_module
    def fake(name, *a, **k):
        if name == "broken_suite":
            raise RuntimeError("registration blew up")
        return real_import(name, *a, **k)
    monkeypatch.setattr(plugins.importlib, "import_module", fake)

    report = load_all(include_entry_points=False, extra_modules=("broken_suite",))
    assert "broken_suite" in report.failed
    assert "registration blew up" in report.failed["broken_suite"]


def test_entry_point_scan_failure_is_survivable(monkeypatch):
    monkeypatch.setattr(plugins, "discover_entry_point_suites",
                        lambda: (_ for _ in ()).throw(RuntimeError("metadata broken")))
    with pytest.raises(RuntimeError):
        plugins.discover_entry_point_suites()   # the helper itself propagates
    # but load_all guards its own call site
    report = load_all(include_entry_points=False)
    assert report.ok


def test_a_module_is_never_loaded_twice():
    report = load_all(include_entry_points=False,
                      extra_modules=(BUILTIN_MODULES[0],))
    assert report.loaded.count(BUILTIN_MODULES[0]) == 1


# --------------------------------------------------------------------------- #
# Report shape
# --------------------------------------------------------------------------- #

def test_report_serialises():
    import json
    json.dumps(load_all(include_entry_points=False).as_dict())


def test_summary_names_failures():
    report = load_all(include_entry_points=False, extra_modules=("nope.nope",))
    assert "failed" in report.summary()


def test_entry_point_group_is_namespaced():
    """A generic group name would collide with other tools' plugins."""
    assert ENTRY_POINT_GROUP.startswith("avaloka.")


def test_loaded_suites_reach_the_registry():
    """Discovery is pointless if registration does not follow."""
    from benchmarks.core.registry import list_suites
    load_all(include_entry_points=False)
    assert len(list_suites()) >= 3
