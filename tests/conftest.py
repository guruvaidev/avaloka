# conftest.py
import os

# ---------------------------------------------------------------------------
# Windows hard-crash guard (0xC0000005) -- MUST run before any app import.
#
# retrieve_memory abandons its worker thread after a 3s circuit breaker. The
# orphan then calls into ChromaDB's Rust bindings while the graph has already
# moved on, and on Windows that takes the whole interpreter down with an access
# violation partway through a run. The symptom is brutal to diagnose: pytest
# prints a dot or two and returns to the prompt with exit code -1073741819 and
# no summary, so it reads as "the tests won't run" rather than as a crash.
# Neither `-p no:faulthandler` nor `2> nul` helps -- those hide the reporting,
# not the death.
#
# Pointing Layer 2 at a dead port forces it onto its supported in-process
# fallback, which never reaches the Rust bindings. setdefault so an explicit
# CHROMA_HOST/CHROMA_PORT from the environment still wins.
#
# conftest.py is imported before any test module, which is what makes this the
# right place: setting it in the shell works too, but has to be redone in every
# new terminal and is undiscoverable when forgotten.
# Remove once the orphaned-thread teardown is fixed.
# ---------------------------------------------------------------------------
os.environ.setdefault("CHROMA_HOST", "127.0.0.1")
os.environ.setdefault("CHROMA_PORT", "59999")

# ---------------------------------------------------------------------------
# No memory warm-up thread -- MUST run before any app import.
#
# Importing app.services.memory_plane starts a daemon thread that loads the
# embedding model, which imports packages in the background while the tests
# run. pytest's warnings plugin iterates sys.modules around every test, so a
# test failed now and then with "RuntimeError: dictionary changed size during
# iteration" raised from contextlib, in whichever test happened to be running
# (1 in 30 runs of the H3 selection, a different test each time). Tests that
# need the embedding model load it on first use, as a request does.
# ---------------------------------------------------------------------------
os.environ.setdefault("MEMORY_WARMUP", "0")

# ---------------------------------------------------------------------------
# Lineage goes to a throwaway file -- MUST run before any app import.
#
# Every test that runs an execution node, a training node or the preparation
# pipeline reaches app/agents/lineage_hooks.py, and the store resolves its
# location from AVALOKA_LINEAGE_DB_URL, then POSTGRES_URL. A developer with
# POSTGRES_URL exported, or a CI job that provides one, would have the suite
# write test datasets into a real shared database.
#
# Assigned, not setdefault: whatever the environment says, the suite does not
# write there. Tests that want a real Postgres name it themselves through
# AVALOKA_TEST_POSTGRES_URL (tests/test_lineage.py).
# ---------------------------------------------------------------------------
import tempfile

_LINEAGE_TEST_DIR = tempfile.mkdtemp(prefix="avaloka-lineage-tests-")
os.environ["AVALOKA_LINEAGE_DB_URL"] = "sqlite:///" + os.path.join(
    _LINEAGE_TEST_DIR, "lineage.db").replace(os.sep, "/")
os.environ.pop("AVALOKA_LINEAGE", None)

# ---------------------------------------------------------------------------
# A deployment key for sealed credentials -- MUST run before any app import.
#
# Customer database credentials are sealed at rest and in transit under
# DB_ENCRYPTION_KEY (app/mcp_server/credentials.py); a registration with no
# key is refused. A real deployment always has one; so does the suite: a fresh
# random value per run, never a published one. setdefault so a key from the
# environment still wins. Tests about a missing or published key set their own.
# ---------------------------------------------------------------------------
import secrets as _secrets

os.environ.setdefault("DB_ENCRYPTION_KEY", "pytest-" + _secrets.token_hex(32))
# Services provision generated secrets into the .env file at startup
# (app/core/provisioning.py). The suite never writes the repository's .env.
os.environ["AVALOKA_ENV_FILE"] = os.path.join(_LINEAGE_TEST_DIR, "avaloka-test.env")

import re

# ---------------------------------------------------------------------------
# Third-party usage beacons off for the whole session -- before any app import.
#
# `import daft` pings daft.gateway.scarf.sh and osstelemetry.io from a
# background thread, and mlflow, onnxruntime, Ray and others have beacons of
# their own (app/core/thirdparty_optout.py lists each one). A gate run on the
# base made about 154 such lookups. The deployment images switch them off
# with ENV; the suite has no image, so it switches them off here.
#
# Assigned, not setdefault: a developer's shell does not opt the suite back in.
# Tests that need a beacon (tests/test_thirdparty_optout.py's controls) strip
# these from a child process's environment themselves.
# ---------------------------------------------------------------------------
from app.core.thirdparty_optout import THIRDPARTY_TELEMETRY_OPTOUT_ENV as _TELEMETRY_OPTOUT

os.environ.update(_TELEMETRY_OPTOUT)

# ---------------------------------------------------------------------------
# A developer's .env never reaches the suite -- MUST run before any app import.
#
# A dozen app modules call load_dotenv() when imported, most with an explicit
# <checkout>/.env path. On a developer machine that file holds real
# credentials, so a test that does not go through the harness stubs -- one that
# reaches _schedule_small_file_persistence, resolve_plan or an MCP connection
# lookup -- would read and write a real Supabase project. AVALOKA_ENV_FILE
# above only moves where provisioning WRITES; it does not stop those reads.
#
# So, for this session, python-dotenv loads nothing from inside the checkout:
# an explicit path under it is refused, and so is a bare load_dotenv(), which
# searches upward from its caller -- that is, the checkout. A test that writes
# its own env file under tmp_path and loads it is unaffected.
#
# And the session starts with no SUPABASE_* variables at all, whatever the
# shell exported. A test that genuinely needs the shell's values asks for the
# `real_supabase_env` fixture below; a whole session (the e2e pack in deployed
# mode reads SUPABASE_JWT_SECRET at import) opts in with
# AVALOKA_TEST_REAL_SUPABASE=1 in the environment that starts pytest.
# ---------------------------------------------------------------------------
import pathlib as _pathlib

_CHECKOUT = _pathlib.Path(__file__).resolve().parent.parent

try:
    import dotenv as _dotenv
    import dotenv.main as _dotenv_main
except ImportError:  # python-dotenv not installed: nothing can load a .env
    _dotenv = None

if _dotenv is not None:
    _real_load_dotenv = _dotenv_main.load_dotenv

    def _inside_checkout(path) -> bool:
        try:
            _pathlib.Path(path).resolve().relative_to(_CHECKOUT)
            return True
        except (TypeError, ValueError, OSError):
            return False

    def _load_dotenv_outside_checkout(dotenv_path=None, stream=None, *args, **kwargs):
        if stream is None and (not dotenv_path or _inside_checkout(dotenv_path)):
            return False
        return _real_load_dotenv(dotenv_path, stream, *args, **kwargs)

    _dotenv.load_dotenv = _dotenv_main.load_dotenv = _load_dotenv_outside_checkout

_SHELL_SUPABASE_ENV = {k: v for k, v in os.environ.items() if "SUPABASE" in k.upper()}
if os.environ.get("AVALOKA_TEST_REAL_SUPABASE") != "1":
    for _name in _SHELL_SUPABASE_ENV:
        del os.environ[_name]

import pytest


@pytest.fixture
def real_supabase_env(monkeypatch):
    """The SUPABASE_* values the shell exported, back for this one test.

    Opt-in only: the session starts without them (see above). Values come from
    the shell that started pytest, never from a .env file."""
    for name, value in _SHELL_SUPABASE_ENV.items():
        monkeypatch.setenv(name, value)
    return dict(_SHELL_SUPABASE_ENV)


def pytest_configure(config):
    config.addinivalue_line("markers", "cloud: mark test as requiring real cloud credentials")


# ---------------------------------------------------------------------------
# Collection-time marking
# ---------------------------------------------------------------------------
# pytest.ini has declared `integration` ("exercises real agents/LLMs") since
# before the open-source release, and almost nothing ever carried it. So the
# hermetic CI stage -- which selects `not integration` precisely to stay
# runnable without Redis, a server or cloud credentials -- pulled in the whole
# end-to-end suite anyway and failed 378 tests on a clean checkout.
#
# The tests are not wrong and the marker expression is not wrong; the marker was
# simply never applied. Applying it by location is deliberate: a per-file
# decision drifts the moment someone adds a file, whereas "everything under
# tests/e2e/ is end-to-end" is a rule that keeps being true. Anything needing a
# service also needs the marker, and a file can still opt in by hand.
#
# Nothing is deleted and nothing is skipped by default. `pytest` with no -m
# still runs all of it; only the hermetic selection filters these out.

# ---------------------------------------------------------------------------
# Files named test_*.py that are not tests
# ---------------------------------------------------------------------------
# These two are operator diagnostics: an argparse __main__, a required CSV path
# supplied by the person running them, a real LLM call, and a printed report to
# read. They contain no assertions and cannot pass as tests -- pytest collects
# them only because of the filename, then errors with "fixture 'csv_path' not
# found" on every run.
#
# Ignored rather than renamed: their own usage lines, and the docs, say
# `python tests/test_new_sampler_standalone.py --csv-path ...`, and that still
# works. Ignoring them at collection also skips their module-level daft import,
# which is not free.
collect_ignore = [
    "test_new_sampler_standalone.py",
    "test_profiling_agent_standalone.py",
]


# Both rules are anchored at tests/. They used to be a bare substring and a bare
# suffix -- "tests/e2e/" anywhere in the path, "_integration.py" at the end of
# any path -- which is broader than "these suites" and would have silently
# deselected a test collected from anywhere else that happened to match, e.g.
# a vendored package's own tests/e2e/, or a module beside
# app/agents/mta/mlflow_integration.py. Nothing matched that way when this was
# tightened (the deselected count did not move), so this closes a hole rather
# than fixing a miscount; tests/test_integration_marker_scope.py pins it.
#
# A name ending _integration.py is a claim that the file needs something outside
# the process. Twelve files made that claim while mocking every boundary, and
# the gate believed the name and never ran them. They are now *_unit.py. The
# converse also happened: tests that looked mocked still shelled out to scp and
# kubectl or read a gs:// object. Before naming a file either way, run it with
# the network off and read what it touches.
#
# tests/e2e/ is NOT in the list. Its own harness runs every leg in-process over
# fakes by default (tests/e2e/conftest.py, "safe as a PR gate"); the legs that
# need a deployment skip via requires_deployed(), and the files that need cloud
# data or a model provider say so with a module marker (test_e3_real_dataset_gcs
# is `cloud`, test_e5_golden_missions is `integration`). Marking the whole
# directory took 500+ hermetic tests out of every gate from 2026-09-24, and the
# pack rotted unseen. Verified with the container network off before removal.
_TESTS_ROOT = "tests/"
_INTEGRATION_DIRS = ("tests/k8s/", "tests/infra/", "tests/containers/")
_INTEGRATION_SUFFIXES = ("_integration.py",)


@pytest.fixture(autouse=True)
def _no_live_ray_env(monkeypatch):
    """Tests must not inherit the developer's Ray cluster address. With
    RAY_DASHBOARD_URL set, ray_job_runner switches to DIRECT mode and builds a
    real JobSubmissionClient instead of the mocked kubectl path. A test that
    needs DIRECT mode sets the variable itself."""
    for var in ("RAY_DASHBOARD_URL", "RAY_API_SERVER_ADDRESS", "RAY_ADDRESS"):
        monkeypatch.delenv(var, raising=False)


def _wants_integration_marker(relpath: str) -> bool:
    normalised = relpath.replace("\\", "/")
    if not normalised.startswith(_TESTS_ROOT):
        return False
    if normalised.startswith(_INTEGRATION_DIRS):
        return True
    return normalised.endswith(_INTEGRATION_SUFFIXES)


# Tests that start a real `langgraph dev`. The CLI is not in requirements.txt (its
# langgraph-api caps cryptography <45; see requirements-langgraph-server.txt), so
# the main test environment cannot run them, and they must not quietly skip
# there either. They are marked here, from the module's own source, so a test
# module needs no marker of its own to be routed correctly:
#   main env:   -m "... and not langgraph_server"  (scripts/ci.sh, the pipeline)
#   server venv: scripts/langgraph_server_tests.sh runs -m langgraph_server with
#                the CLI installed, where they fail (never skip) if it is missing.
_LANGGRAPH_CLI_IMPORT = re.compile(r"^\s*(from|import)\s+langgraph_cli\b|['\"]from langgraph_cli[ .]", re.M)
_langgraph_cli_modules: dict = {}


def _starts_langgraph_cli(source: str) -> bool:
    return bool(_LANGGRAPH_CLI_IMPORT.search(source))


def _module_starts_langgraph_cli(path: str) -> bool:
    if path not in _langgraph_cli_modules:
        try:
            with open(path, encoding="utf-8") as fh:
                _langgraph_cli_modules[path] = _starts_langgraph_cli(fh.read())
        except OSError:
            _langgraph_cli_modules[path] = False
    return _langgraph_cli_modules[path]


def pytest_collection_modifyitems(config, items):
    """Mark end-to-end and integration suites so `-m 'not integration'` means it."""
    import pathlib
    root = pathlib.Path(str(config.rootpath))
    for item in items:
        if str(item.fspath).endswith(".py") and _module_starts_langgraph_cli(str(item.fspath)) \
                and not any(m.name == "langgraph_server" for m in item.iter_markers()):
            item.add_marker(pytest.mark.langgraph_server)
        try:
            rel = pathlib.Path(str(item.fspath)).relative_to(root).as_posix()
        except ValueError:
            continue
        if _wants_integration_marker(rel) and not any(
            m.name == "integration" for m in item.iter_markers()
        ):
            item.add_marker(pytest.mark.integration)
