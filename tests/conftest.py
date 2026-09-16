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
