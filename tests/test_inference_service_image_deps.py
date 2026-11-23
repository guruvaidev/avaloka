"""The MTA inference service image installs no LangChain or LangGraph, and needs none.

app/agents/mta_v2/inference_service_image/ is built from its own directory
(Dockerfile: ``COPY requirements.txt`` then ``COPY . .``), so it can only run the
modules in that directory. None of them imports langchain*, langgraph* or the
LangGraph CLI, yet its requirements.txt had been copied from the root one and
installed that whole stack: 13 langchain/langgraph advisories, plus langgraph-api
0.5.x, whose ``cryptography<45`` cap held the image at cryptography 44.0.3.

Two halves, so neither can drift on its own:
  * the manifest names no langchain/langgraph distribution, and cryptography's
    floor is at the release that fixes its advisories;
  * no module in the image imports one -- if a future change does need LangChain
    here, this fails and the dependency is added deliberately, not by copy.
"""
from __future__ import annotations

import ast
import pathlib
import re

IMAGE_DIR = (pathlib.Path(__file__).resolve().parents[1]
             / "app" / "agents" / "mta_v2" / "inference_service_image")
FORBIDDEN = re.compile(r"^(langchain|langgraph)([-_].*)?$", re.I)


def _requirements() -> dict[str, str]:
    """Distribution name (normalised) -> the rest of the line."""
    out: dict[str, str] = {}
    for raw in (IMAGE_DIR / "requirements.txt").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)", line)
        assert m, f"unparsed requirement line: {raw!r}"
        out[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(3)
    return out


def test_manifest_installs_no_langchain_or_langgraph():
    reqs = _requirements()
    assert len(reqs) > 20, f"parsed only {len(reqs)} requirements from {IMAGE_DIR}"
    found = sorted(name for name in reqs if FORBIDDEN.match(name))
    assert not found, (
        f"{found} in {IMAGE_DIR.name}/requirements.txt: the inference service imports "
        "no LangChain/LangGraph, and langgraph-cli[inmem] pins cryptography below its fixes"
    )


def test_no_module_in_the_image_imports_langchain_or_langgraph():
    modules = sorted(IMAGE_DIR.rglob("*.py"))
    assert len(modules) >= 10, f"found only {len(modules)} modules under {IMAGE_DIR}"
    hits = []
    for path in modules:
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            hits += [f"{path.relative_to(IMAGE_DIR)}:{node.lineno} {n}"
                     for n in names if FORBIDDEN.match(n.split(".")[0])]
    assert not hits, f"the inference service image imports LangChain/LangGraph: {hits}"


def test_cryptography_floor_is_the_fixed_release():
    spec = _requirements().get("cryptography", "")
    m = re.search(r">=\s*(\d+)", spec)
    assert m and int(m.group(1)) >= 50, (
        f"cryptography spec {spec!r}: 50.0.0 is the first release without "
        "GHSA-g6cj-pr64-35w5 (and 49/48.0.1/46.0.5 the fixes for the other four)"
    )
