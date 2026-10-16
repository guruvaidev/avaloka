"""Daft doc retrieval: what comes back, and what happens when nothing can.

Repaired from the quarantined FAISS suite. `app.rag.daft_retriever` and
`retrieve_daft_docs` are gone -- grep finds neither anywhere in app/ or
avaloka/, and no file in the tree mentions FAISS at all. The store is Chroma
(`app/rag/daft_embeddings/chroma.sqlite3`) and the entry point is
`app.rag.daft_retrieval.retrieve_docs`, which takes a *list* of pseudocode
lines rather than one query string.

Four of the nine original tests did not survive and are not replaced here:

  * `test_embeddings_files_exist` asserted index.faiss / index_meta.pkl.
    Those artifacts no longer exist. The equivalent guard -- the store must be
    on disk, because chromadb.PersistentClient would otherwise create an empty
    one and retrieval would silently return nothing -- is already asserted by
    tests/test_dta_bugfixes.py::test_load_chroma_collection_raises_on_missing_db,
    which runs in the gate.

  * `test_filter_semantics`, `test_groupby_and_aggregation_semantics` and
    `test_expression_semantics` asserted that a query about filtering retrieves
    docs containing "filter"/"where"/"predicate", and likewise for groupby and
    expressions. That is a property of the ingested corpus plus the real
    mixedbread-ai/mxbai-embed-large-v1 embedder, which has to be downloaded.
    It cannot be asserted hermetically, so it is not asserted anywhere in the
    gate. That coverage is lost; see the report.

What is kept is the retrieval contract the code can be held to without a model:
which text is returned, that it is deduplicated and non-empty, that blank
pseudocode is never turned into a query, and that a store that cannot answer
degrades to "no docs" instead of taking the coder down with it. None of this
is asserted by anything the gate selects.
"""

from __future__ import annotations

import app.agents.data_transfer_agent.daft_coder as daft_coder
from app.agents.data_transfer_agent.daft_coder import parse_pseudo_code
from app.rag.daft_retrieval import retrieve_docs


class _StubCollection:
    """Stands in for a Chroma collection and records what it was asked.

    `results` is returned verbatim from `query`; `raises` makes the store fail
    the way an unreadable or mismatched-dimension index does.
    """

    def __init__(self, results=None, raises: Exception | None = None):
        self._results = results or {"metadatas": [], "documents": []}
        self._raises = raises
        self.queries: list[list[str]] = []
        self.n_results: list[int] = []

    def query(self, query_texts, n_results):
        self.queries.append(list(query_texts))
        self.n_results.append(n_results)
        if self._raises is not None:
            raise self._raises
        return self._results


# ---------------------------------------------------------------------------
# What retrieval returns
#
# Was: test_basic_retrieval_returns_results --
#      assert isinstance(docs, list); len(docs) > 0; every doc a non-empty str
# ---------------------------------------------------------------------------


def test_retrieval_returns_non_empty_deduplicated_strings_across_all_lines():
    """Every line is queried in one batch and the hits are flattened and deduped.

    The flattening is per-line, so the same chunk is a near-certainty when two
    pseudocode lines are about the same operation. Shipping it twice spends the
    coder's prompt budget on a repeat, and shipping a blank chunk spends it on
    nothing.
    """
    collection = _StubCollection(
        results={
            "metadatas": [
                [{"original_text": "daft.col filters rows"}, {"original_text": "where() predicate"}],
                # Second line re-hits the first chunk, plus an empty one.
                [{"original_text": "daft.col filters rows"}, {"original_text": "   "}],
            ],
            "documents": [["ignored", "ignored"], ["ignored", "ignored"]],
        }
    )

    docs = retrieve_docs(["Filter rows where x > 5", "Keep only rows where y < 3"], collection, 2)

    assert docs == ["daft.col filters rows", "where() predicate"]
    assert all(isinstance(d, str) and d.strip() for d in docs)
    # Both lines went in one batched call at the requested depth.
    assert len(collection.queries) == 1
    assert len(collection.queries[0]) == 2
    assert collection.n_results == [2]


# ---------------------------------------------------------------------------
# Which text comes back
#
# Was: test_metadata_structure -- the metadata paired with the index is a
#      non-empty list. The pickle is gone; what the metadata is *for* is not.
# ---------------------------------------------------------------------------


def test_retrieval_returns_the_stored_text_not_the_embedded_instruction():
    """`metadata['original_text']` wins over the stored document.

    What was embedded carries the mxbai instruction prefix ("Represent this
    sentence for searching relevant passages: ..."). Injecting that into the
    codegen prompt tells the coder model to go searching instead of writing
    Daft. The clean text is kept in metadata for exactly this reason, and the
    document is only a fallback for chunks that have no metadata.
    """
    collection = _StubCollection(
        results={
            "metadatas": [[
                {"original_text": "df.groupby('country').agg(...)"},
                None,  # no metadata: the document is all there is
            ]],
            "documents": [[
                "Represent this sentence for searching relevant passages: df.groupby('country').agg(...)",
                "df.sort(desc=True)",
            ]],
        }
    )

    docs = retrieve_docs(["Group by country and sum revenue"], collection, 2)

    assert docs == ["df.groupby('country').agg(...)", "df.sort(desc=True)"]
    assert not any("Represent this sentence" in d for d in docs), (
        "the embedding instruction prefix leaked into the retrieved docs"
    )


# ---------------------------------------------------------------------------
# Blank input is never retrieved for
#
# Was: test_empty_query_returns_empty_list -- retrieve_daft_docs("") == []
# ---------------------------------------------------------------------------


def test_empty_pseudocode_never_opens_the_store(monkeypatch):
    """With no pseudocode there is nothing to retrieve for, so nothing is loaded.

    Opening the store means constructing the SentenceTransformer embedder --
    seconds of work, and on a cold box a model download -- to answer a question
    nobody asked. The guard is in the node, ahead of the store.
    """
    opened: list[tuple] = []

    # Records rather than raises: the node swallows every Exception, so a stub
    # that raised would make this test pass with the guard deleted. The record
    # is the assertion. It also returns a stub so a regression cannot reach the
    # real store on disk.
    def _record_open(*args, **kwargs):
        opened.append(args)
        return _StubCollection()

    monkeypatch.setattr(daft_coder, "_get_cached_chroma_collection", _record_open)

    out = daft_coder.daft_retrieval_node({"coder_pseudocode": ""})

    assert opened == [], "the store was opened for empty pseudocode"
    assert out == {"rag_retrieved_docs": ""}


# ---------------------------------------------------------------------------
# Was: test_whitespace_query_returns_empty_list --
#      retrieve_daft_docs("   \n  ") == []
# ---------------------------------------------------------------------------


def test_blank_pseudocode_lines_are_dropped_and_never_become_a_query():
    """Whitespace-only pseudocode gets past the emptiness guard; it must still
    not manufacture a query.

    `_create_search_payload` wraps whatever it is handed in the mxbai
    instruction, so a blank line becomes a query that is nothing but the
    instruction -- which matches arbitrary chunks and pads the coder's prompt
    with irrelevant docs. `parse_pseudo_code` is what prevents it.
    """
    assert parse_pseudo_code("   \n  \n\t\n") == []

    collection = _StubCollection()
    docs = retrieve_docs(parse_pseudo_code("   \n  \n\t\n"), collection, 5)

    assert docs == []
    assert collection.queries == [[]], (
        f"a blank line was turned into a query: {collection.queries}"
    )


# ---------------------------------------------------------------------------
# A store that cannot answer does not take the coder down
#
# Was: test_garbage_query_does_not_crash --
#      assert isinstance(docs, list)
# ---------------------------------------------------------------------------


def test_a_store_that_fails_yields_no_docs_rather_than_raising(monkeypatch):
    """Retrieval is an optional prompt enrichment, not a precondition.

    Both layers swallow: `retrieve_docs` returns [] so a partial batch failure
    does not lose the lines that worked, and `daft_retrieval_node` returns ""
    so a transfer still generates code without the docs. If either re-raised,
    an unreadable index would fail the user's transfer outright.
    """
    boom = RuntimeError("Collection expecting embedding with dimension of 1024, got 384")

    collection = _StubCollection(raises=boom)
    assert retrieve_docs(["filter rows where x > 5"], collection, 3) == []

    def _explode(*args, **kwargs):
        raise boom

    monkeypatch.setattr(daft_coder, "_get_cached_chroma_collection", _explode)

    out = daft_coder.daft_retrieval_node({"coder_pseudocode": "Filter rows where x > 5"})

    assert out == {"rag_retrieved_docs": ""}
