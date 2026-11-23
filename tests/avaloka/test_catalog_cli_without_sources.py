"""The CLI without the ``sources`` extra: SQLAlchemy is absent, so there is no
local catalog -- and nothing else changes.

``pip install avaloka`` gives the eight runtime dependencies and no SQLAlchemy.
Then ``avaloka analyze`` must still analyse (registration is skipped, silently)
and ``avaloka catalog ...`` must say which extra to install rather than trace.
Here SQLAlchemy is made unimportable for the test; the same file also runs in a
requirements-only virtualenv where it is genuinely absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from avaloka import cli as avaloka_cli
from avaloka.catalog import local as catalog_local


@pytest.fixture
def no_sqlalchemy(monkeypatch, tmp_path):
    for name in [m for m in list(sys.modules) if m == "sqlalchemy" or m.startswith("sqlalchemy.")
                 or m == "avaloka.catalog.store"]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)   # import sqlalchemy -> ImportError
    monkeypatch.setenv(catalog_local.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setenv(catalog_local.ENV_USER, "cli-user")
    monkeypatch.delenv("AVALOKA_API_URL", raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    yield tmp_path / "home"
    monkeypatch.delitem(sys.modules, "avaloka.catalog.store", raising=False)


def test_the_store_says_which_extra_to_install(no_sqlalchemy):
    with pytest.raises(RuntimeError, match=r"avaloka\[sources\]"):
        catalog_local.store()


def test_analyze_still_loads_the_file_and_registers_nothing(no_sqlalchemy, tmp_path):
    path = tmp_path / "orders.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    local_path, wl, resolved = avaloka_cli._load_or_die(str(path))
    assert Path(local_path) == path and resolved.kind == "file"
    assert not (no_sqlalchemy / "catalog.db").exists()


def test_catalog_commands_refuse_in_one_sentence_not_a_traceback(no_sqlalchemy):
    result = CliRunner().invoke(avaloka_cli.app, ["catalog", "list"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "avaloka[sources]" in result.stdout and "Traceback" not in result.stdout
    result = CliRunner().invoke(avaloka_cli.app, ["catalog", "use", "my.orders"], catch_exceptions=False)
    assert result.exit_code == 1 and "avaloka[sources]" in result.stdout


def test_a_reference_without_a_catalog_dies_with_a_sentence(no_sqlalchemy):
    import typer
    with pytest.raises(typer.Exit) as exc:
        avaloka_cli._load_or_die("my.orders")
    assert exc.value.exit_code == 1
