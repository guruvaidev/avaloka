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

import pytest


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

_INTEGRATION_DIRS = ("tests/e2e/", "tests/k8s/", "tests/infra/")
_INTEGRATION_SUFFIXES = ("_integration.py",)


def _wants_integration_marker(relpath: str) -> bool:
    normalised = relpath.replace("\\", "/")
    if any(seg in normalised for seg in _INTEGRATION_DIRS):
        return True
    return normalised.endswith(_INTEGRATION_SUFFIXES)


def pytest_collection_modifyitems(config, items):
    """Mark end-to-end and integration suites so `-m 'not integration'` means it."""
    import pathlib
    root = pathlib.Path(str(config.rootpath))
    for item in items:
        try:
            rel = pathlib.Path(str(item.fspath)).relative_to(root).as_posix()
        except ValueError:
            continue
        if _wants_integration_marker(rel) and not any(
            m.name == "integration" for m in item.iter_markers()
        ):
            item.add_marker(pytest.mark.integration)
