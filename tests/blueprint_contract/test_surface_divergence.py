"""Pin the CLI/chat surface boundary.

The question "what do the two surfaces share?" had never been answered, and the
answer turned out to be: **nothing**. The `avaloka/` CLI never generates or
executes LLM-authored code -- its deliverables are f-string templates over
already-computed decisions -- so none of the coder/validator hardening in
`app/agents/` applies to it, and none of its model-risk gates apply to chat.

That is a load-bearing architectural fact. These tests record it so that a
future change which couples the surfaces (or quietly gives the CLI a codegen
path) is a deliberate decision rather than a surprise.

Deterministic: import-level and source-level checks only. No LLM, no network.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PKG = REPO_ROOT / "avaloka"
APP_PKG = REPO_ROOT / "app"


def _python_files(root: Path):
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _imported_modules(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


# The one sanctioned crossing: deploying an inference service reuses the k8s
# invoker. It is not part of the analysis pipeline.
ALLOWED_APP_IMPORTS = {
    "app.infra",              # `from app.infra import k8s_invoker`
    "app.infra.k8s_invoker",
}


def test_cli_does_not_import_the_chat_agent_pipeline():
    offenders: list[str] = []
    for path in _python_files(CLI_PKG):
        for mod in _imported_modules(path):
            if not mod.startswith("app."):
                continue
            if mod in ALLOWED_APP_IMPORTS:
                continue
            offenders.append(f"{path.relative_to(REPO_ROOT)} imports {mod}")
    assert not offenders, (
        "The CLI has started importing from app/. The two surfaces have always "
        "been independent; if this is intended, update this test and say so in "
        "the change.\n" + "\n".join(offenders)
    )


def test_cli_never_imports_the_coder_or_validator():
    for path in _python_files(CLI_PKG):
        mods = _imported_modules(path)
        assert "app.agents.coder" not in mods, path
        assert "app.agents.validator" not in mods, path


def test_cli_does_not_execute_generated_code():
    """The CLI writes analysis.py as a deliverable; it must not run model output.

    If this ever changes, the CLI needs a validation story of its own -- today
    it has none, because it has never needed one.
    """
    offenders: list[str] = []
    for path in _python_files(CLI_PKG):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"exec", "eval", "compile"}:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {node.func.id}()")
    assert not offenders, "\n".join(offenders)


def test_chat_surface_owns_the_codegen_pipeline():
    """The mirror assertion: the pieces under test really do live in app/."""
    from app.agents import coder, validator

    assert hasattr(coder, "coder_node")
    assert hasattr(coder, "_generate_pseudocode")
    assert hasattr(validator, "syntactic_validator_node")
    assert hasattr(validator, "static_semantic_validator_node")
    assert hasattr(validator, "execute_code_node")
    assert hasattr(validator, "logical_semantic_validator_node")
    assert hasattr(validator, "contract_validator_node")


def test_contract_gate_is_in_the_production_coding_graph():
    from app.api.workflow import build_coding_graph

    graph = build_coding_graph()
    nodes = set(graph.get_graph().nodes)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}

    assert "validator_contract" in nodes
    assert ("execute_code", "validator_contract") in edges
    assert ("validator_contract", "validator_logical") in edges


def test_contract_violation_routes_back_to_the_coder():
    from app.api.workflow import check_validation_status

    assert check_validation_status({"contract_error": True, "retry_count": 0}) == "refine"
    assert check_validation_status({"contract_error": True, "retry_count": 1}) == "refine"
    # and respects the same budget as every other failure
    assert check_validation_status({"contract_error": True, "retry_count": 3}) == "end"


def test_dta_keeps_its_own_parallel_pipeline():
    """The real duplication is inside app/, not between app/ and avaloka/: the
    Data Transfer Agent has its own pseudocode node, coder and logical reviewer,
    with a different retry budget. Recorded so it is not mistaken for shared
    code."""
    pytest.importorskip("daft", reason="DTA pulls daft, which is optional")
    from app.agents.data_transfer_agent import daft_coder

    assert hasattr(daft_coder, "daft_pseudocode_node")
    assert hasattr(daft_coder, "daft_coder_node")
