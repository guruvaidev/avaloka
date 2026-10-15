"""The model table must say what runs.

``app/core/model_config.DEFAULT_MODELS`` names a model per agent, but only the
Groq path reads it (as ``groq_model``). On every other provider -- including
OpenRouter, the default -- ``build_chat_model`` ignores ``groq_model`` and picks
the model from the ``tier`` the call site passes. So a call site can pass
``tier="small"`` while its table entry says 120b, and nothing complains: the
validator, the gate between generated code and its execution, shipped that way
and ran on the 20b model while the table said it "gets the full 120b".

What is pinned here is the property, not the literal:

* every agent that builds an LLM has a table entry, and on the default provider
  with no model env vars set it resolves to that entry;
* a gate resolves to the large model, because "table and call site agree on
  small" is agreement, and still not acceptable for a gate.

Call sites are found by parsing the source, so a new agent is covered without
being added here, and nothing is imported that would open a connection. No
network: clients are constructed with a dummy key and never invoked.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple

import pytest

from app.core import inference
from app.core import model_config as mc
from app.core.agent_llm import build_agent_llm
from app.core.inference import build_chat_model

REPO = Path(__file__).resolve().parents[1]

_FACTORIES = {"build_chat_model": build_chat_model, "build_agent_llm": build_agent_llm}
#: The factories themselves: agent_llm.py forwards ``agent=agent``, which is a
#: pass-through and not an agent.
_FACTORY_FILES = {"app/core/inference.py", "app/core/agent_llm.py"}
#: avaloka/ is the CLI package, outside app/. Both ship.
_SCAN_ROOTS = ("app", "avaloka")

#: ``agent=`` id -> table key, where it is not simply the id lower-cased.
_TABLE_KEY = {"VIZ": "visualization"}

#: Agents whose verdict decides whether something runs or is written.
GATES = ("VALIDATOR", "DTA_VALIDATOR")

#: Agents the general property does not hold for, each with its reason. An entry
#: here is a strict xfail: when the disagreement is resolved the test starts
#: passing, which fails the run until the entry is removed.
KNOWN_DISAGREEMENTS: Dict[str, str] = {
    "INTENT_CLASSIFIER": (
        "has no entry of its own: reads the 'conversational' entry (120b) on "
        "Groq but passes tier='small' (20b) elsewhere, while its docstring says "
        "'the same model tier as the planner'. Which side is right is an open "
        "decision, not a mechanical fix."
    ),
}


class CallSite(NamedTuple):
    path: str
    line: int
    agent: str
    role: str
    tier: str

    @property
    def table_key(self) -> str:
        return _TABLE_KEY.get(self.agent, self.agent.lower())


def _literal(call: ast.Call, name: str, factory) -> str:
    for kw in call.keywords:
        if kw.arg == name:
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value
            raise AssertionError(f"{name}= is not a string literal; this test cannot read it")
    default = inspect.signature(factory).parameters[name].default
    assert default is not inspect.Parameter.empty, f"{name}= is required and missing"
    return default


def _discover() -> List[CallSite]:
    sites: List[CallSite] = []
    for root in _SCAN_ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if rel in _FACTORY_FILES:
                continue
            source = path.read_text(encoding="utf-8")
            if not any(name in source for name in _FACTORIES):
                continue
            for node in ast.walk(ast.parse(source, filename=rel)):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name not in _FACTORIES:
                    continue
                try:
                    factory = _FACTORIES[name]
                    sites.append(CallSite(
                        rel, node.lineno,
                        _literal(node, "agent", factory),
                        _literal(node, "role", factory),
                        _literal(node, "tier", factory),
                    ))
                except AssertionError as exc:
                    raise AssertionError(f"{rel}:{node.lineno}: {exc}") from None
    return sites


CALL_SITES = _discover()
_BY_AGENT = {site.agent: site for site in CALL_SITES}

#: Sentinel passed as the Groq model. If it ever comes back as the resolved
#: model, the client was built on the Groq path and the test is not measuring
#: the default provider.
_GROQ_ONLY = "groq-path-model-not-used-on-the-default-provider"

_KEY_VARS = (
    "OPENROUTER_API_KEY", "INFERENCE_OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE",
    "GROQ_API_KEY", "GROQ_API_KEY_PLANNING_AGENT", "GROQ_API_KEY_CODING_AGENT",
    "DTA_CODER_MODEL", "DTA_VALIDATOR_MODEL",
)


def _is_routing_var(name: str) -> bool:
    return name.startswith(("INFERENCE_", "AVALOKA_")) or name in _KEY_VARS


@pytest.fixture
def default_provider(monkeypatch):
    """Nothing configured except one OpenRouter key: the documented default."""
    for name in list(os.environ):
        if _is_routing_var(name):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")


def _resolved_on_default_provider(site: CallSite) -> str:
    assert inference.resolve_provider(role=site.role, agent=site.agent) == inference.OPENROUTER
    llm = build_chat_model(
        role=site.role,
        agent=site.agent,
        tier=site.tier,
        temperature=0.0,
        groq_model=_GROQ_ONLY,
        groq_api_key=None,
    )
    assert llm is not None, f"{site.agent}: no client was built on the default provider"
    return llm.model_name


def _site_params():
    params = []
    for site in CALL_SITES:
        reason = KNOWN_DISAGREEMENTS.get(site.agent)
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        params.append(pytest.param(site, id=site.agent, marks=marks))
    return params


def test_the_sweep_finds_the_call_sites():
    """An empty parametrisation passes silently; this is what stops that."""
    agents = [site.agent for site in CALL_SITES]
    assert len(agents) == len(set(agents)), f"an agent id is built at two call sites: {agents}"
    assert {"PLANNER", "CODER", "SUMMARIZER", *GATES} <= set(agents), agents
    stale = sorted(set(KNOWN_DISAGREEMENTS) - set(agents))
    assert not stale, f"KNOWN_DISAGREEMENTS names agents with no call site: {stale}"


@pytest.mark.parametrize("site", _site_params())
def test_every_agent_resolves_to_its_table_entry_on_the_default_provider(site, default_provider):
    where = f"{site.path}:{site.line}"
    assert site.table_key in mc.DEFAULT_MODELS, (
        f"{site.agent} ({where}) builds an LLM but has no entry {site.table_key!r} in "
        "DEFAULT_MODELS, so the table is silent about a model that runs."
    )
    resolved = _resolved_on_default_provider(site)
    assert resolved == mc.DEFAULT_MODELS[site.table_key], (
        f"{site.agent} ({where}) passes tier={site.tier!r} and runs on {resolved!r} on the "
        f"default provider, but DEFAULT_MODELS[{site.table_key!r}] says "
        f"{mc.DEFAULT_MODELS[site.table_key]!r}. The table must say what runs."
    )


@pytest.mark.parametrize("agent", GATES)
def test_a_gate_runs_on_the_large_model_on_the_default_provider(agent, default_provider):
    assert agent in _BY_AGENT, f"gate {agent} has no call site; was it renamed?"
    site = _BY_AGENT[agent]
    tiers = inference._PROVIDER_MODEL_DEFAULTS[inference.OPENROUTER]
    resolved = _resolved_on_default_provider(site)
    assert resolved == tiers["large"], (
        f"{agent} ({site.path}:{site.line}) is a gate and runs on {resolved!r} on the default "
        f"provider, not the large model {tiers['large']!r} (tier={site.tier!r})."
    )
    assert mc.DEFAULT_MODELS[site.table_key] == tiers["large"]


def test_every_validator_is_classified_as_a_gate():
    """A new validator that is not listed in GATES would escape the test above."""
    unclassified = sorted(a for a in _BY_AGENT if "VALIDATOR" in a and a not in GATES)
    assert not unclassified, f"validators not listed in GATES: {unclassified}"


@pytest.mark.parametrize("module,table_key", [
    ("app.agents.validator", "validator"),
    ("app.agents.data_transfer_agent.daft_validator", "dta_validator"),
])
def test_the_gate_object_built_at_import_is_on_its_table_model(module, table_key):
    """The same property on the object the pipeline actually calls.

    ``validator_llm`` is built at import, so it is read in a fresh interpreter
    rather than by reloading a module the rest of the suite has already
    imported. The DTA validator only builds a client when a Groq key is present,
    so one is supplied; with an OpenRouter key also set, OpenRouter is still the
    provider. The sysdocs path is given no scheme so the import opens no
    connection.
    """
    env = {k: v for k, v in os.environ.items() if not _is_routing_var(k)}
    env.update(
        OPENROUTER_API_KEY="test-key-not-real",
        GROQ_API_KEY="test-key-not-real",
        AVALOKA_SYSDOCS_PATH="none://",
    )
    probe = f"import {module} as m; print('MODEL=' + m.validator_llm.model_name)"
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    reported = [ln for ln in proc.stdout.splitlines() if ln.startswith("MODEL=")]
    assert reported == [f"MODEL={mc.DEFAULT_MODELS[table_key]}"], (
        f"{module}.validator_llm is on {reported}, but DEFAULT_MODELS[{table_key!r}] "
        f"says {mc.DEFAULT_MODELS[table_key]!r}"
    )
