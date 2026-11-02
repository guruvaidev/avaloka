"""Memory embeddings stay local unless the operator explicitly opts in to OpenAI.

The defect: EMBEDDING_PROVIDER defaulted to "auto", and "auto" sent the text to
OpenAIEmbeddings whenever OPENAI_API_KEY was set -- a key that is also the chat
key, and that the inference router accepts for a self-hosted OpenAI-spec
server. The text is memory text: raw user queries, learned hints, LLM answers,
and json.dumps(schema_info) with table names, column names and sample values.
An audit saw those synthetic names in a POST to api.openai.com/v1/embeddings,
preceded by tiktoken's GET to openaipublic.blob.core.windows.net.

Every outbound path is intercepted BELOW httpx's event hooks (at the transport)
plus requests and raw sockets, so a recorded request is one that would really
have gone on the wire. Intercepting httpx.Client.send instead sits above
huggingface_hub's offline hook and records Hub requests the hook would have
refused -- that is how the audit came to report HEADs to huggingface.co under
HF_HUB_OFFLINE=1.
"""

from __future__ import annotations

import logging
import socket
import sys
import types

import pytest

SYNTH = ("schema of zz_synth_tbl: columns zz_mrn_col, zz_diag_code_col; "
         "sample 'ZZ-SYNTH-VALUE-42'")
OPENAI_HOSTS = ("api.openai.com", "openaipublic")
LOCAL_DIM = 384


@pytest.fixture
def egress(monkeypatch):
    """Record and refuse every outbound request; yields the list of what was tried."""
    import httpx
    import requests

    seen: list = []

    def _refuse(what: str, exc):
        seen.append(what)
        raise exc

    def transport(self, request):
        _refuse(f"{request.method} {request.url}", httpx.ConnectError("refused by test", request=request))

    async def atransport(self, request):
        _refuse(f"{request.method} {request.url}", httpx.ConnectError("refused by test", request=request))

    def req_send(self, request, **kwargs):
        _refuse(f"{request.method} {request.url}", requests.ConnectionError("refused by test"))

    def connect(self, address):
        _refuse(f"connect {address}", OSError("refused by test"))

    def getaddrinfo(host, *a, **k):
        _refuse(f"dns {host}", OSError("refused by test"))

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", atransport)
    monkeypatch.setattr(requests.Session, "send", req_send)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", lambda addr, *a, **k: getaddrinfo(addr[0]))
    return seen


class _FakeLocalModel:
    """Stands in for the bundled MiniLM: a non-zero 384-d vector per text."""

    def __init__(self):
        self.encoded: list = []

    def encode(self, text):
        self.encoded.append(text)
        return [((i % 7) + 1) / 10.0 for i in range(LOCAL_DIM)]


@pytest.fixture
def eu(monkeypatch):
    """embedding_utils with a clean provider env and the fake local model loaded."""
    import app.services.embedding_utils as mod

    for key in ("EMBEDDING_PROVIDER", "EMBEDDING_OPENAI_API_KEY", "OPENAI_API_KEY",
                "MEMORY_ALLOW_MEMORY_FALLBACKS", "APP_ENV"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(mod, "_warned", set(), raising=False)
    model = _FakeLocalModel()
    monkeypatch.setattr(mod, "_st_instance", model)
    mod._fake_model = model
    return mod


def _openai_attempts(seen):
    return [s for s in seen if any(h in s for h in OPENAI_HOSTS)]


# --- the defect -------------------------------------------------------------


def test_chat_key_present_and_provider_unset_embeds_locally_and_sends_nothing(eu, egress, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    vec = eu.embed_text(SYNTH, target_dim=1536)

    assert _openai_attempts(egress) == [], f"memory text was sent toward OpenAI: {egress}"
    assert egress == [], f"the local path made outbound requests: {egress}"
    assert eu._fake_model.encoded == [SYNTH], "the local model did not embed the text"
    assert len(vec) == 1536 and any(vec[:LOCAL_DIM]) and not any(vec[LOCAL_DIM:])


def test_chat_key_set_provider_unset_never_calls_the_openai_client(eu, egress, monkeypatch):
    """Owner's requirement for the 1.6 hotfix: OPENAI_API_KEY set, EMBEDDING_PROVIDER
    unset, nothing goes to OpenAI. The OpenAI embeddings client is replaced by one
    that records and refuses every use, and the HTTP layer refuses too (egress)."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    used = []

    class _RefusingOpenAIEmbeddings:
        def __init__(self, **kw):
            used.append("constructed")

        def embed_query(self, text):
            used.append(f"embed_query {text[:20]}")
            raise RuntimeError("refused by test: memory text must not go to OpenAI")

        embed_documents = embed_query

    fake = types.ModuleType("langchain_openai")
    fake.OpenAIEmbeddings = _RefusingOpenAIEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_openai", fake)

    for text in (SYNTH, "revenue by zz_region_col", '{"columns": ["zz_mrn_col"]}'):
        eu.embed_text(text, target_dim=1536)

    assert used == [], f"the OpenAI embeddings client was used on the default path: {used}"
    assert _openai_attempts(egress) == [], f"a request went toward OpenAI: {egress}"


def test_local_path_does_not_load_tiktoken_or_the_openai_client(eu, egress, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    calls = []
    fake = types.ModuleType("langchain_openai")
    fake.OpenAIEmbeddings = lambda **kw: calls.append(kw) or (_ for _ in ()).throw(AssertionError)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake)

    eu.embed_text(SYNTH)

    assert calls == [], f"OpenAIEmbeddings was constructed on the default path: {calls}"


def test_auto_is_deprecated_and_means_local(eu, egress, monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "auto")

    with caplog.at_level(logging.WARNING, logger=eu.__name__):
        vec = eu.embed_text(SYNTH, target_dim=LOCAL_DIM)

    assert egress == [], f"EMBEDDING_PROVIDER=auto still reached out: {egress}"
    assert any(vec)
    assert "deprecated" in caplog.text and "auto" in caplog.text


def test_openai_requested_without_dedicated_key_is_local_with_a_warning(eu, egress, monkeypatch, caplog):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")   # the chat key must not count

    with caplog.at_level(logging.WARNING, logger=eu.__name__):
        vec = eu.embed_text(SYNTH, target_dim=LOCAL_DIM)

    assert egress == [], f"no EMBEDDING_OPENAI_API_KEY, yet a request was attempted: {egress}"
    assert eu._fake_model.encoded == [SYNTH], "expected the local model, not a zero-vector"
    assert len(vec) == LOCAL_DIM and any(vec)
    assert "EMBEDDING_OPENAI_API_KEY" in caplog.text


def test_explicit_opt_in_uses_openai_with_the_dedicated_key(eu, egress, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_OPENAI_API_KEY", "sk-embed-optin")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-chat-key")
    built, embedded = [], []

    class _FakeOpenAIEmbeddings:
        def __init__(self, **kw):
            built.append(kw)

        def embed_query(self, text):
            embedded.append(text)
            return [0.25] * 1536

    fake = types.ModuleType("langchain_openai")
    fake.OpenAIEmbeddings = _FakeOpenAIEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_openai", fake)

    vec = eu.embed_text(SYNTH, target_dim=1536)

    assert embedded == [SYNTH], "the opted-in OpenAI path was not attempted"
    assert built and built[0]["api_key"] == "sk-embed-optin", (
        f"OpenAI must be called with EMBEDDING_OPENAI_API_KEY, not the chat key: {built}")
    assert vec == [0.25] * 1536
    assert eu._fake_model.encoded == [], "pinned to OpenAI, the local model must not mix in"
    assert egress == []


def test_opted_in_openai_outage_does_not_fall_back_to_local(eu, egress, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_OPENAI_API_KEY", "sk-embed-optin")

    class _Down:
        def __init__(self, **kw):
            pass

        def embed_query(self, text):
            raise RuntimeError("openai down")

    fake = types.ModuleType("langchain_openai")
    fake.OpenAIEmbeddings = _Down
    monkeypatch.setitem(sys.modules, "langchain_openai", fake)

    assert eu.embed_text(SYNTH, target_dim=64) == [0.0] * 64
    assert eu._fake_model.encoded == []


def test_warmup_loads_the_local_model_by_default_even_with_the_chat_key(eu, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")   # no dedicated key -> local
    assert eu.warmup_local_model() is True
    assert eu._fake_model.encoded == ["warmup"]


def test_production_config_check_does_not_count_the_chat_key(monkeypatch):
    import app.services.memory_runtime as mr

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(mr, "strict_memory_infra_enabled", lambda: True)
    real = mr._module_available
    monkeypatch.setattr(mr, "_module_available",
                        lambda name: False if name == "sentence_transformers" else real(name))

    with pytest.raises(mr.MemoryConfigError) as exc:
        mr.validate_memory_runtime_config()
    assert "embedding provider" in str(exc.value)


# --- stored vectors from the old default ---------------------------------------


def _local_vec(dim=1536):
    return [((i % 5) + 1) / 10.0 for i in range(LOCAL_DIM)] + [0.0] * (dim - LOCAL_DIM)


def _openai_like_vec(dim=1536):
    # Shares the local vector's first 384 components, so cosine is ~0.6 and the
    # old code would accept it as a match; the tail marks it as a native-1536 vector.
    return _local_vec(dim)[:LOCAL_DIM] + [0.1] * (dim - LOCAL_DIM)


def test_redis_semantic_search_skips_vectors_from_another_embedding_space(monkeypatch, caplog):
    from app.services.db.redis_client import RedisClientImpl

    rc = RedisClientImpl()
    rc.use_mock = True
    rc._fallback_store["u1:old_ds"] = {"schema": {"table": "qqq"}, "ttl": 60}
    rc._fallback_vectors["u1:old_ds"] = _openai_like_vec()
    monkeypatch.setattr(rc, "_embed", lambda text: _local_vec())

    with caplog.at_level(logging.WARNING):
        found = rc.search_schema_semantically("u1", "revenue by region")

    assert found is None, "a vector from a different embedding space was scored as a match"
    assert "different embedding provider" in caplog.text


def test_redis_semantic_search_still_matches_same_space_vectors(monkeypatch):
    from app.services.db.redis_client import RedisClientImpl

    rc = RedisClientImpl()
    rc.use_mock = True
    rc._fallback_store["u1:ds"] = {"schema": {"table": "qqq"}, "ttl": 60}
    rc._fallback_vectors["u1:ds"] = _local_vec()
    monkeypatch.setattr(rc, "_embed", lambda text: _local_vec())

    assert rc.search_schema_semantically("u1", "anything") == {"table": "qqq"}


def test_milvus_warns_once_when_stored_rows_come_from_another_space(caplog):
    from app.services.db.milvus_client import MilvusClientImpl

    queries = []

    class _Coll:
        def query(self, **kw):
            queries.append(kw)
            return [{"vector": _openai_like_vec()}, {"vector": _local_vec()}]

        def insert(self, *a, **k):
            pass

        def flush(self):
            pass

    mc = MilvusClientImpl()
    mc._collection = _Coll()
    with caplog.at_level(logging.WARNING):
        mc.insert_insight(session_id="s", content="c", vector=_local_vec())
        mc.insert_insight(session_id="s", content="c", vector=_local_vec())

    assert len(queries) == 1, "the stored-vector check must run once per process"
    assert "1 of 2 sampled rows" in caplog.text and "re-embed" in caplog.text


def test_milvus_space_check_never_breaks_an_insert():
    from app.services.db.milvus_client import MilvusClientImpl

    inserted = []

    class _Coll:
        def query(self, **kw):
            raise RuntimeError("vector output not supported")

        def insert(self, data, **k):
            inserted.append(data)

        def flush(self):
            pass

    mc = MilvusClientImpl()
    mc._collection = _Coll()
    mc.insert_insight(session_id="s", content="c", vector=_local_vec())
    assert len(inserted) == 1


# --- the real bundled model (runs where it is cached, e.g. the API image) -------


def test_real_local_model_returns_nonzero_vectors_and_makes_no_request(egress, monkeypatch):
    pytest.importorskip("sentence_transformers")
    import app.services.embedding_utils as mod

    for key in ("EMBEDDING_PROVIDER", "EMBEDDING_OPENAI_API_KEY", "APP_ENV",
                "SENTENCE_TRANSFORMER_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(mod, "_st_instance", None)
    monkeypatch.setattr(mod, "_warned", set(), raising=False)
    try:
        mod._get_st_model("all-MiniLM-L6-v2")
    except Exception:   # 1.6 has no offline loader: a cache miss surfaces as OSError & co.
        pytest.skip("all-MiniLM-L6-v2 is not in the local HF cache")
    egress.clear()   # this is about embedding, not loading

    vec = mod.embed_text(SYNTH, target_dim=1536)

    assert egress == [], f"outbound requests on the local path: {egress}"
    assert len(vec) == 1536
    assert mod.effective_dim(vec) == LOCAL_DIM, "not a MiniLM vector padded to the target"


def test_embedding_module_never_reads_the_chat_key():
    """Static guard: no env lookup of OPENAI_API_KEY in the embedding modules.

    The C1 egress inventory cannot see this regression (embedding_utils.py is
    declared either way), so it is pinned here.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("app/services/embedding_utils.py", "app/services/memory_runtime.py"):
        tree = ast.parse((root / rel).read_text("utf-8"))
        reads = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, (ast.Call, ast.Subscript))
            and any(isinstance(c, ast.Constant) and c.value == "OPENAI_API_KEY"
                    for c in ast.walk(node.slice if isinstance(node, ast.Subscript) else node)
                    if c is not node and not isinstance(c, ast.JoinedStr))
            and "environ" in ast.unparse(node.value if isinstance(node, ast.Subscript) else node.func)
        ]
        assert reads == [], f"{rel} reads OPENAI_API_KEY from the environment at lines {reads}"
