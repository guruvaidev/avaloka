"""The release suite's service URLs: skip on a laptop, FAIL when CI is set.

Hermetic. The service-backed tests (test_supabase_rls.py, its settled-RLS
companion test_settled_rls.py, and test_db_connectors_live.py) take their
server URLs through service_urls.require_url, so a CI run that loses a URL goes
red instead of reporting a green line that tested nothing.
"""

from __future__ import annotations

import pytest

from tests.release_1_6.service_urls import require_url

VAR = "AVALOKA_TEST_SERVICE_URL_PROBE"


def _outcome(monkeypatch, *, ci: str | None, value: str | None) -> tuple[str, str]:
    """What require_url does: ("value", url), ("fail", msg) or ("skip", msg).

    Catches BaseException so that a skip raised where a failure is expected is
    reported here as a wrong outcome (an assertion), not as a skipped test."""
    if ci is None:
        monkeypatch.delenv("CI", raising=False)
    else:
        monkeypatch.setenv("CI", ci)
    if value is None:
        monkeypatch.delenv(VAR, raising=False)
    else:
        monkeypatch.setenv(VAR, value)
    try:
        return ("value", require_url(VAR, "a probe server"))
    except BaseException as exc:  # noqa: BLE001 -- pytest's outcomes are BaseException
        if isinstance(exc, pytest.fail.Exception):
            return ("fail", str(exc))
        if isinstance(exc, pytest.skip.Exception):
            return ("skip", str(exc))
        raise


@pytest.mark.parametrize("ci", ["true", "1", "True"])
def test_missing_url_fails_when_ci_is_set(monkeypatch, ci):
    kind, msg = _outcome(monkeypatch, ci=ci, value=None)
    assert kind == "fail", (kind, msg)
    assert VAR in msg and "CI is set" in msg


@pytest.mark.parametrize("ci", [None, "", "0", "false"])
def test_missing_url_skips_on_a_laptop(monkeypatch, ci):
    kind, msg = _outcome(monkeypatch, ci=ci, value=None)
    assert kind == "skip", (kind, msg)
    assert VAR in msg


def test_a_blank_url_counts_as_missing(monkeypatch):
    assert _outcome(monkeypatch, ci="true", value="   ")[0] == "fail"


@pytest.mark.parametrize("ci", [None, "true"])
def test_a_set_url_is_returned(monkeypatch, ci):
    assert _outcome(monkeypatch, ci=ci, value="postgresql://u@h/db") == ("value", "postgresql://u@h/db")


def test_every_service_backed_test_takes_its_url_through_require_url():
    """No release-suite test reads a server URL and skips by itself any more."""
    import pathlib
    here = pathlib.Path(__file__).parent
    offenders = []
    for path in sorted(here.glob("test_*.py")):
        text = path.read_text()
        for var in ("AVALOKA_TEST_POSTGRES_URL", "AVALOKA_TEST_MYSQL_URL", "AVALOKA_TEST_PG_ADMIN_URL"):
            if f'environ.get("{var}"' in text or f"environ['{var}'" in text or f'environ["{var}"' in text:
                offenders.append(f"{path.name}: {var}")
    assert offenders == []
