"""Minimal Jupyter notebook (nbformat v4) construction without a hard dep."""

from __future__ import annotations

from typing import Any


def md(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def _lines(source: str) -> list[str]:
    lines = source.splitlines(keepends=True)
    return lines or [source]


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "avaloka": {"generated_by": "avaloka", "reproducible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
