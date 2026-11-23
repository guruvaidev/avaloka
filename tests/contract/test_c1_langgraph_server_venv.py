"""The LangGraph server's tests run in a venv of their own; the main CI environment never gets the CLI.

requirements-langgraph-server.txt brings langgraph-api 0.5.x, which caps
cryptography below 45. Installed into the environment the API's tests run in,
it would put cryptography 44 back under every one of them, while the API image
ships 50. So:

  * no step that installs the main test environment names that file;
  * the main pytest selection (scripts/ci.sh HERMETIC_MARKERS, which the
    pipeline must match) excludes the ``langgraph_server`` marker;
  * tests that start `langgraph dev` get that marker automatically
    (tests/conftest.py), so a new one cannot land in the main run and fail
    there, or be missed;
  * scripts/langgraph_server_tests.sh builds the separate venv
    (--system-site-packages, the server file installed into the venv only)
    and runs ``-m langgraph_server`` with the venv's interpreter;
  * both CI entry points run that script: scripts/ci.sh (GitHub) and the
    pipeline's tiers that gate (bitbucket-pipelines.yml).
"""
from __future__ import annotations

import importlib.util
import pathlib
import re

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
SERVER_REQS = "requirements-langgraph-server.txt"
SCRIPT = "scripts/langgraph_server_tests.sh"


def _pipeline():
    return yaml.safe_load((REPO / "bitbucket-pipelines.yml").read_text())


def _lines(step) -> list[str]:
    out = []
    for entry in (step or {}).get("script", []):
        out += [l.strip() for l in str(entry).splitlines() if l.strip()]
    return out


def _defined_steps() -> list[dict]:
    return [item["step"] for item in _pipeline()["definitions"]["steps"] if "step" in item]


def _step(name: str) -> dict:
    found = [s for s in _defined_steps() if s.get("name") == name]
    assert len(found) == 1, f"bitbucket-pipelines.yml has {len(found)} steps named {name!r}"
    return found[0]


def test_no_main_environment_install_names_the_server_requirements():
    offenders = []
    for path in ("scripts/install.sh", ".github/workflows/ci.yml", "scripts/ci.sh"):
        text = (REPO / path).read_text()
        if SERVER_REQS in text:
            offenders.append(path)
    for step in _defined_steps():
        if step.get("name") == "Build LangGraph App":
            continue  # a step that exists to start the server; its env is the server's
        for line in _lines(step):
            if SERVER_REQS in line:
                offenders.append(f"pipeline step {step.get('name')!r}: {line}")
    assert not offenders, (
        f"{SERVER_REQS} installed into a main test environment (cryptography 44 under the "
        f"API's tests): {offenders}. Only {SCRIPT} installs it, into its own venv."
    )


def test_the_main_selection_excludes_the_server_tests():
    ci_sh = (REPO / "scripts/ci.sh").read_text()
    m = re.search(r'HERMETIC_MARKERS="([^"]+)"', ci_sh)
    assert m and "not langgraph_server" in m.group(1), m and m.group(1)
    hermetic = [l for l in _lines(_step("Run Tests"))
                if "-m \"not cluster and not cloud and not integration" in l]
    assert hermetic and all("not langgraph_server" in l for l in hermetic), hermetic


def test_the_separate_venv_script_layers_the_server_file_on_the_main_env():
    text = (REPO / SCRIPT).read_text()
    assert re.search(r'-m venv --system-site-packages "\$VENV"', text), "the venv must layer on the main env"
    assert re.search(r'"\$VENV/bin/python" -m pip install .*\n?.*-r ' + re.escape(SERVER_REQS), text), \
        "the server file must be installed with the venv's pip, never the main one"
    assert re.search(r'exec "\$VENV/bin/python" -m pytest .*-m "\$MARKERS"', text)
    assert re.search(r'MARKERS="langgraph_server and ', text)
    # The main interpreter only creates the venv and reports a version; it installs nothing.
    assert not re.search(r'"\$PY" -m pip install', text)


def test_both_ci_entry_points_run_the_server_tests():
    ci_sh = (REPO / "scripts/ci.sh").read_text()
    assert re.search(r'^stage "langgraph server" .*' + re.escape(SCRIPT), ci_sh, re.M)
    gh = (REPO / ".github/workflows/ci.yml").read_text()
    assert "scripts/ci.sh" in gh, "GitHub CI no longer runs scripts/ci.sh"
    pipelines = _pipeline()["pipelines"]
    tiers = [pipelines["branches"][b] for b in ("develop-1.6", "develop-1.7") if b in pipelines["branches"]]
    assert tiers, "no mainline branch tier in bitbucket-pipelines.yml"
    for tier in tiers + [pipelines["pull-requests"]["**"]]:
        lines = [l for item in tier for l in _flatten(item)]
        assert any(SCRIPT in l for l in lines), "a gating tier does not run the server tests"


def _flatten(item) -> list[str]:
    out = []
    if isinstance(item, dict):
        if "step" in item:
            out += _lines(item["step"])
        if "parallel" in item:
            par = item["parallel"]
            for sub in (par.get("steps", []) if isinstance(par, dict) else par):
                out += _flatten(sub)
    elif isinstance(item, list):
        for sub in item:
            out += _flatten(sub)
    return out


def test_the_marker_is_registered_and_applied_from_the_source():
    ini = (REPO / "pytest.ini").read_text()
    assert re.search(r"^\s+langgraph_server:", ini, re.M)
    spec = importlib.util.spec_from_file_location("_tests_conftest", REPO / "tests/conftest.py")
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)
    starts = conftest._starts_langgraph_cli
    cli = "langgraph" + "_cli"  # spelt in pieces, so this module is not marked itself
    assert starts(f'proc = subprocess.Popen([sys.executable, "-c", "from {cli}.cli import cli; cli()"])')
    assert starts(f"import {cli}\n")
    assert starts(f"from {cli}.cli import cli\n")
    assert not starts(f'find_spec("{cli}")\n')
    assert not starts("from langgraph.graph import StateGraph\n")
    assert conftest._module_starts_langgraph_cli(str(REPO / "tests/test_langgraph_server_env.py"))
