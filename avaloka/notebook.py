"""Minimal Jupyter notebook (nbformat v4) construction without a hard dep."""

from __future__ import annotations

from typing import Any


def md(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str, tags: list[str] | None = None) -> dict[str, Any]:
    """A code cell. ``tags`` is written to the cell metadata when given: a
    cell tagged ``parameters`` is what papermill overrides on a re-run."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": list(tags)} if tags else {},
        "outputs": [],
        "source": _lines(source),
    }


def _lines(source: str) -> list[str]:
    lines = source.splitlines(keepends=True)
    return lines or [source]


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """nbformat 4.5 gives every cell an ``id``; a stable positional one is
    enough and keeps two generations of the same notebook identical."""
    return {
        "cells": [{**cell, "id": f"cell-{i}"} for i, cell in enumerate(cells)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "avaloka": {"generated_by": "avaloka", "reproducible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
