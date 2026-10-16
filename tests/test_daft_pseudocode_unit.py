"""The hermetic half of the Daft pseudocode tests: the model is a stub.

Split out of tests/test_daft_pseudocode_integration.py, whose module-level
``integration`` marker (and filename) kept these three out of the gate along
with the twenty prompt cases there that do call a real coding model.
"""
from types import SimpleNamespace
import os
import re

import pytest

from app.agents.data_transfer_agent import daft_coder
from app.agents.data_transfer_agent.daft_coder import parse_pseudo_code as parse_pseudocode


def test_generate_pseudocode_uses_prompt_and_rules(monkeypatch):
    class StubLLM:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="1. Step one")

    stub_llm = StubLLM()
    monkeypatch.setattr(daft_coder, "coder_llm", stub_llm)

    state = {
        "user_prompt": "Create a new column based on Age.",
        "schema": {"Age": "integer"},
    }

    # Pseudocode is its own graph node now; it writes coder_pseudocode.
    result = daft_coder.daft_pseudocode_node(state)

    assert result == {"coder_pseudocode": "1. Step one"}
    assert stub_llm.messages is not None

    system_prompt = stub_llm.messages[0].content
    human_prompt = stub_llm.messages[1].content

    assert "Never rename columns unless the user explicitly requests a rename." in system_prompt
    assert "You MUST include explicit casting steps whenever types are incompatible" in system_prompt
    assert "Create a new column based on Age." in human_prompt
    assert "Schema (column: dtype):" in human_prompt
    assert "Age" in human_prompt


def test_cached_chroma_collection(monkeypatch):
    daft_coder._get_cached_chroma_collection.cache_clear()
    calls = []

    def fake_load(db_path, collection_name, embedding_model):
        calls.append((db_path, collection_name, embedding_model))
        return {"db_path": db_path, "collection_name": collection_name}

    monkeypatch.setattr(daft_coder, "load_chroma_collection", fake_load)

    first = daft_coder._get_cached_chroma_collection("db", "collection", "model")
    second = daft_coder._get_cached_chroma_collection("db", "collection", "model")

    assert first is second
    assert len(calls) == 1


def test_daft_coder_node_builds_prompt_from_inputs(monkeypatch):
    class StubLLM:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="```python\nprint('ok')\n```")

    stub_llm = StubLLM()
    monkeypatch.setattr(daft_coder, "coder_llm", stub_llm)

    # The coder reads what the pseudocode and retrieval nodes left in state
    # (coder_pseudocode, rag_retrieved_docs) and the DTA's source_schema.
    state = {
        "user_prompt": "Filter rows where Age > 30.",
        "source_schema": {"Age": "integer"},
        "coder_pseudocode": "1. Step one",
        "rag_retrieved_docs": "daft filter reference",
        "input_data_type": "csv",
    }

    result = daft_coder.daft_coder_node(state)

    assert "generated_code" in result
    assert "print('ok')" in result["generated_code"]

    human_prompt = stub_llm.messages[1].content
    assert "Filter rows where Age > 30." in human_prompt
    assert "Age" in human_prompt
    assert "Pseudocode blueprint" in human_prompt
    assert "1. Step one" in human_prompt
    assert "daft filter reference" in human_prompt
