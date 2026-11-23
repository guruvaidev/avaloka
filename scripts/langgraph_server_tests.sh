#!/usr/bin/env bash
# Run the tests that start a real `langgraph dev` (marker: langgraph_server), in a
# venv of their own that has the LangGraph CLI.
#
# Why a separate venv: langgraph-cli[inmem] brings langgraph-api 0.5.x, which caps
# cryptography <45. The main test environment installs requirements.txt only, so
# it runs the API's tests on the cryptography the API image ships (50.x). This
# venv layers requirements-langgraph-server.txt on top of it (--system-site-packages):
# only the CLI stack and the cryptography it needs land in the venv, and the
# main environment is not touched.
#
# Used by scripts/ci.sh (stage "langgraph server") and bitbucket-pipelines.yml
# (&test). tests/contract/test_c1_langgraph_server_venv.py keeps that wiring.
#
# Env: PYTHON (main interpreter, default python3), LANGGRAPH_SERVER_VENV
# (default .venv-langgraph-server). Extra arguments go to pytest.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
VENV="${LANGGRAPH_SERVER_VENV:-.venv-langgraph-server}"
MARKERS="langgraph_server and not cluster and not cloud and not integration"

[ -x "$VENV/bin/python" ] || "$PY" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install -q --disable-pip-version-check --prefer-binary \
    -r requirements-langgraph-server.txt
ver() { "$1" -c 'import importlib.metadata as m; print(m.version("cryptography"))'; }
echo "[langgraph-server] main env cryptography $(ver "$PY"); server venv cryptography $(ver "$VENV/bin/python")"
# CI=true is what makes these tests fail rather than skip if the CLI is missing.
exec "$VENV/bin/python" -m pytest -q -p no:cacheprovider -m "$MARKERS" "$@" tests/
