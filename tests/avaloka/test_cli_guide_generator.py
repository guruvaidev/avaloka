"""The CLI-guide generator reads parameter types the same way on every
typer/click pairing the project allows (typer 0.12-0.29, click 8.1-8.5).

Typer 0.12-0.19 hands the generator click parameters with ``.type``; typer
0.20+ keeps the annotation and resolves the click type at call time, so
``.type`` is absent and the generator used to crash
(``'TyperOption' object has no attribute 'type'``). Both shapes are modelled
here with plain objects, so the test runs on whichever typer is installed and
fails if either path stops yielding click's vocabulary. The live check -- the
generated page equals the committed one -- is tests/avaloka/test_cli_guide.py.
"""

from __future__ import annotations

import enum
import importlib.util
import pathlib
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_cli_guide_under_test", REPO / "scripts" / "gen_cli_guide.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _OldStyle:
    """A click parameter: ``.type`` with click's own ``name``."""

    def __init__(self, name: str):
        self.type = SimpleNamespace(name=name)


class _NewStyle:
    """A typer 0.20+ parameter: no ``.type``; the annotation behind get_runtime_param()."""

    def __init__(self, annotation, bare: str = "thing"):
        self._annotation = annotation
        self._bare = bare

    def get_runtime_param(self):
        return SimpleNamespace(annotation=self._annotation, parameter_info=None)

    def bare_type(self):
        return self._bare


class Colour(enum.Enum):
    RED = "red"


@pytest.mark.parametrize("click_name", ["text", "integer", "float", "boolean", "path", "choice"])
def test_a_click_parameter_keeps_clicks_name(gen, click_name):
    assert gen._type_name(_OldStyle(click_name)) == click_name


@pytest.mark.parametrize("annotation, expected", [
    (str, "text"), (Optional[str], "text"), (int, "integer"), (Optional[int], "integer"),
    (float, "float"), (bool, "boolean"), (pathlib.Path, "path"), (Optional[pathlib.Path], "path"),
    (Colour, "choice"),
])
def test_an_annotation_only_parameter_maps_to_clicks_vocabulary(gen, annotation, expected):
    """Bite: return bare_type() ('str', 'int') instead of click's names, and the
    page's Type column changes with the typer version."""
    assert gen._type_name(_NewStyle(annotation)) == expected


def test_an_unknown_annotation_falls_back_to_typers_bare_type_not_a_crash(gen):
    assert gen._type_name(_NewStyle(dict, bare="dict")) == "dict"
    assert gen._type_name(SimpleNamespace()) == "text"   # neither shape: the most common type, not an exception


def test_the_live_command_tree_uses_only_clicks_vocabulary(gen):
    """Whatever typer is installed here, the page's types come out of one vocabulary."""
    types = {p["type"] for cmd in gen.command_tree() for p in cmd["arguments"] + cmd["options"]}
    assert types <= {"text", "integer", "float", "boolean", "path", "choice"}, types
    assert {"text", "float", "boolean", "path"} <= types
