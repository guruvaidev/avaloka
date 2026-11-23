"""The LangChain/LangGraph code paths with open advisories stay unused.

Until the LangGraph 1.x migration lands, the API runs langchain-core 0.3,
langchain 0.3, langgraph 0.6, langgraph-checkpoint 3 and langgraph-sdk 0.2, and
the LangGraph server runs langgraph-api 0.5. Each advisory below is reachable
only through an entry point that we do not use. The dependency inventory checked
that by grep, once; this makes the check part of the gate, so the first change
that does reach one fails here instead of quietly re-opening the advisory.

| Advisory | Package | Entry point that must stay unused |
|---|---|---|
| GHSA-qh6h-p6c9-ff54 | langchain-core | legacy prompt loading: ``load_prompt``, ``prompts.loading`` |
| GHSA-2g6r-c272-w58r, GHSA-r7w7-9xr2-qq2r | langchain-core, langchain-openai | image token counting, which fetches ``image_url``: ``get_num_tokens_from_messages`` |
| GHSA-gr75-jv2w-4656 | langchain, langchain-anthropic | file-search / file-management tools |
| GHSA-fv5p-p927-qmxr | langchain-text-splitters | ``split_text_from_url`` |
| GHSA-fvww-7h3r-vfhp, GHSA-w39p-vh2g-g8g5 | langgraph-sdk | any ``langgraph_sdk`` client or auth decorator |
| GHSA-mhr3-j7m5-c7c9, GHSA-fjqc-hq36-qh5p, GHSA-g48c-2wqr-h844 | langgraph-checkpoint, langgraph | a durable checkpointer or a node cache: state is only ever written and read by our own process, in memory |
| GHSA-4hm6-w6qq-w73v, GHSA-2c9q-c2q9-qgqv, GHSA-jfj5-wrj9-63x4 | langgraph-api | JS/TS custom auth, webhooks and run creation on the LangGraph server: langgraph.json declares no webhooks and only Python ``auth``/``http.app`` modules, and the API calls only the thread registry |

Each rule is proven to bite on a planted source string, so an empty scan cannot
pass by scanning nothing.
"""
from __future__ import annotations

import ast
import json
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("app", "avaloka", "file_handler", "scripts", "benchmarks")

# (advisories, kind, pattern). kind: "module" matches an imported module (or a
# prefix of it); "name" matches an imported name, an attribute or a bare name.
RULES: list[tuple[str, str, re.Pattern[str]]] = [
    ("GHSA-qh6h-p6c9-ff54", "module", re.compile(r"^langchain(_core)?\.prompts\.loading$")),
    ("GHSA-qh6h-p6c9-ff54", "name", re.compile(r"^load_prompt(_from_config)?$")),
    ("GHSA-2g6r-c272-w58r/GHSA-r7w7-9xr2-qq2r", "name", re.compile(r"^get_num_tokens_from_messages$")),
    ("GHSA-gr75-jv2w-4656", "module", re.compile(r"^langchain(_community)?\.tools\.file_management")),
    ("GHSA-gr75-jv2w-4656", "name", re.compile(r"FileSearch")),
    ("GHSA-fv5p-p927-qmxr", "name", re.compile(r"^split_text_from_url$")),
    ("GHSA-fvww-7h3r-vfhp/GHSA-w39p-vh2g-g8g5", "module", re.compile(r"^langgraph_sdk(\.|$)")),
    ("GHSA-mhr3/fjqc/g48c", "module", re.compile(r"^langgraph\.(checkpoint\.(postgres|sqlite|redis|mongodb)|cache)(\.|$)")),
    ("GHSA-mhr3/fjqc/g48c", "name", re.compile(r"^(Async)?(Postgres|Sqlite|Redis|MongoDB)Saver$")),
]


def _violations(source: str, filename: str = "<planted>") -> list[str]:
    out = []
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        modules, names = [], []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules = [node.module]
            names = [a.name for a in node.names]
            modules += [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Call):
            # compile(cache=...) puts a node cache (pickle-capable) on the graph.
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "compile"
                    and any(k.arg == "cache" for k in node.keywords)):
                out.append(f"{filename}:{node.lineno} GHSA-mhr3-j7m5-c7c9 compile(cache=...)")
        for adv, kind, rx in RULES:
            for value in modules if kind == "module" else names:
                if rx.search(value):
                    out.append(f"{filename}:{node.lineno} {adv} {value}")
    return sorted(set(out))


def _sources() -> list[pathlib.Path]:
    files = []
    for root in SCAN_ROOTS:
        base = REPO / root
        if base.is_dir():
            files += [p for p in base.rglob("*.py")
                      if "node_modules" not in p.parts and "__pycache__" not in p.parts]
    return sorted(files)


def test_no_source_reaches_a_vulnerable_langchain_entry_point():
    files = _sources()
    assert len(files) > 300, f"scanned only {len(files)} files"
    hits = []
    for path in files:
        try:
            src = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            hits += _violations(src, str(path.relative_to(REPO)))
        except SyntaxError:
            continue  # templates with placeholders; not importable code
    assert not hits, (
        "code now reaches an entry point with an open advisory on the 0.x LangChain/LangGraph "
        "line; move that package to its 1.x release first:\n"
        + "\n".join(hits)
    )


@pytest.mark.parametrize("planted", [
    "from langchain_core.prompts.loading import load_prompt",
    "from langchain_core.prompts import load_prompt",
    "import langchain.prompts.loading",
    "n = llm.get_num_tokens_from_messages(msgs)",
    "from langchain_community.tools.file_management import ReadFileTool",
    "from langchain_community.tools import FileSearchTool",
    "chunks = splitter.split_text_from_url(u)",
    "from langgraph_sdk import get_client",
    "import langgraph_sdk.auth",
    "from langgraph.checkpoint.postgres import PostgresSaver",
    "from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver",
    "from langgraph.cache.memory import InMemoryCache",
    "g = builder.compile(checkpointer=c, cache=InMemoryCache())",
])
def test_each_rule_bites_on_a_planted_use(planted):
    assert _violations(planted), f"the scan missed: {planted}"


def test_the_allowed_surface_is_not_flagged():
    allowed = (
        "from langchain_core.messages import AIMessage, HumanMessage\n"
        "from langgraph.checkpoint.memory import MemorySaver\n"
        "from langgraph.graph import StateGraph, END\n"
        "g = StateGraph(dict).compile(checkpointer=MemorySaver())\n"
    )
    assert _violations(allowed) == []


def _langgraph_config_violations(config: dict) -> list[str]:
    """What in langgraph.json reaches an open langgraph-api 0.5.x advisory.

    * webhooks: GHSA-2c9q-c2q9-qgqv (relative/loopback webhooks bypass auth);
    * a JavaScript/TypeScript custom auth handler or app: GHSA-4hm6-w6qq-w73v
      (cross-user Store access) is in JS/TS custom auth, and Python handlers
      are not affected. A Python ``auth.path`` or ``http.app`` (the service-token
      middleware, for one) is allowed.
    """
    out = []
    if "webhooks" in config:
        out.append("webhooks (GHSA-2c9q-c2q9-qgqv)")
    refs = [("auth.path", (config.get("auth") or {}).get("path"))]
    for key in ("http", "http_server"):
        refs.append((f"{key}.app", (config.get(key) or {}).get("app")))
    for name, ref in refs:
        if ref and not str(ref).split(":", 1)[0].endswith(".py"):
            out.append(f"{name}={ref!r} is not a Python module (GHSA-4hm6-w6qq-w73v covers JS/TS handlers)")
    return out


def test_langgraph_server_config_reaches_no_open_advisory():
    config = json.loads((REPO / "langgraph.json").read_text())
    assert config.get("graphs"), "langgraph.json lost its graph; the scan is reading the wrong file"
    found = _langgraph_config_violations(config)
    assert not found, f"langgraph.json reaches langgraph-api 0.5.x advisories; upgrade it first: {found}"


@pytest.mark.parametrize("config,bites", [
    ({"webhooks": {"url": "http://x"}}, True),
    ({"auth": {"path": "./src/auth.ts:auth"}}, True),
    ({"http": {"app": "./app.mjs:app"}}, True),
    ({"auth": {"path": "./app/api/auth.py:auth"}}, False),
    ({"http": {"app": "./app/api/langgraph_service_auth.py:app"}}, False),
    ({"http_server": {"cors": {"allow_origins": ["*"]}}}, False),
])
def test_langgraph_config_rule_bites(config, bites):
    assert bool(_langgraph_config_violations(config)) is bites, config


def test_api_uses_the_langgraph_server_only_as_a_thread_registry():
    """Run creation (GHSA-jfj5) and run webhooks (GHSA-2c9q) are never requested."""
    paths = set()
    for path in (REPO / "app" / "api").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        paths |= {m.group(2) for m in re.finditer(
            r"lg_(json|request|stream|get|post)\(\s*\"[A-Z]+\",\s*f?\"([^\"]+)\"", text)}
    assert paths, "found no calls to the LangGraph server; the pattern is stale"
    allowed = re.compile(r"^/(ok|threads|threads/search|threads/\{[a-z_]+\})$")
    unexpected = sorted(p for p in paths if not allowed.match(p))
    assert not unexpected, f"the API now calls LangGraph server routes beyond the thread registry: {unexpected}"
