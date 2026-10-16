"""Container-backed integration tests: a real Postgres and a real object store.

Run with ``scripts/integration.sh``, which starts ``compose.yaml`` with
credentials it generates for that run and hands their location to pytest in the
``AVALOKA_IT_*`` variables. These tests carry two markers: ``integration`` (so
the hermetic gate, ``-m "not ... integration"``, never selects them) and
``containers`` (so they can be selected on their own, without the dozens of
other ``integration`` files that need a cluster, a cloud account or a model).

**Nothing here may skip.** A skip reads as a pass in a summary line, and this
project has shipped that mistake more than once: a suite that quietly skips
when its service is missing reports green on exactly the run that exercised
nothing. So inside this directory:

* a missing variable or an unreachable container is an error raised by the
  fixture, never ``pytest.skip``;
* any skip at all, from any cause, is rewritten into a failure by the hook
  below, so a later ``skipif`` cannot reintroduce the hole.

No credential is written in this directory. The fixtures read what the runner
generated; a test that needs a bad credential derives one from the good one.
No test talks to anything but the containers the runner started: no deployed
endpoint, no cloud API, no model provider.
"""

from __future__ import annotations

import os
import pathlib
import time
import uuid

import pytest

_HERE = pathlib.Path(__file__).resolve().parent

RUNNER_HINT = "Start the suite with scripts/integration.sh; it starts the containers and sets this."


def _in_this_suite(item) -> bool:
    try:
        return _HERE in pathlib.Path(str(item.fspath)).resolve().parents
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config, items):
    for item in items:
        if _in_this_suite(item):
            item.add_marker(pytest.mark.containers)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and not hasattr(report, "wasxfail") and _in_this_suite(item):
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = "failed"
        report.longrepr = (
            f"{item.nodeid} skipped ({reason}). Nothing in tests/containers/ may skip: a "
            "skipped integration test is indistinguishable from one that never ran.")


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.fail(f"{name} is not set. {RUNNER_HINT}", pytrace=False)
    return value


def _redacted(url: str) -> str:
    from sqlalchemy.engine import make_url
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001
        return "<unparseable url>"


def _wait(probe, seconds: float):
    """Call ``probe`` until it stops raising; return the last error if it never does."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            probe()
            return None
        except Exception as exc:  # noqa: BLE001
            if time.monotonic() >= deadline:
                return exc
            time.sleep(0.5)


@pytest.fixture(scope="session")
def postgres_url() -> str:
    """The runner's Postgres, proven reachable with these credentials.

    Checked with a plain driver connection rather than through the code under
    test: ``LineageStore`` swallows every database error by design, so a suite
    that only asked the store would see an unreachable database as an empty,
    well-behaved one.
    """
    from sqlalchemy import create_engine, text

    url = required_env("AVALOKA_IT_POSTGRES_URL")

    def probe():
        engine = create_engine(url, connect_args={"connect_timeout": 3})
        try:
            with engine.connect() as conn:
                assert conn.execute(text("SELECT 1")).scalar_one() == 1
        finally:
            engine.dispose()

    error = _wait(probe, float(os.environ.get("AVALOKA_IT_WAIT_S", "20")))
    if error is not None:
        detail = str(error).splitlines()[0] if str(error) else ""
        pytest.fail(
            f"Postgres at {_redacted(url)} is not usable: {type(error).__name__}: {detail}. "
            "This suite requires the container; it does not skip without it.", pytrace=False)
    return url


@pytest.fixture(scope="session")
def s3_settings() -> dict:
    """Endpoint and key pair for the runner's SeaweedFS, proven to authenticate."""
    import boto3
    from botocore.config import Config

    settings = {
        "endpoint_url": required_env("AVALOKA_IT_S3_ENDPOINT"),
        "access_key": required_env("AVALOKA_IT_S3_ACCESS_KEY"),
        "secret_key": required_env("AVALOKA_IT_S3_SECRET_KEY"),
        "region": os.environ.get("AVALOKA_IT_S3_REGION", "us-east-1"),
    }
    client = boto3.client(
        "s3", endpoint_url=settings["endpoint_url"], region_name=settings["region"],
        aws_access_key_id=settings["access_key"], aws_secret_access_key=settings["secret_key"],
        config=Config(connect_timeout=3, read_timeout=10, retries={"max_attempts": 1}))

    error = _wait(client.list_buckets, float(os.environ.get("AVALOKA_IT_WAIT_S", "20")))
    if error is not None:
        detail = str(error).splitlines()[0] if str(error) else ""
        pytest.fail(
            f"The object store at {settings['endpoint_url']} is not usable: "
            f"{type(error).__name__}: {detail}. This suite requires the container; it does not "
            "skip without it.", pytrace=False)
    settings["client"] = client
    return settings


@pytest.fixture
def bucket(s3_settings) -> str:
    """A bucket of this test's own, so nothing depends on another test's leftovers."""
    name = "it-" + uuid.uuid4().hex[:20]
    s3_settings["client"].create_bucket(Bucket=name)
    return name
