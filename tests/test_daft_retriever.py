
from tests._quarantine import requires_api

requires_api("app.rag.daft_retriever", replacement="app.rag.daft_retriever became app.rag.daft_retrieval and the FAISS index was replaced by Chroma (retrieve_docs); this file tests the retired FAISS path.")

import pickle
import os
import pytest

from app.rag.daft_retriever import retrieve_daft_docs

EMBEDDINGS_DIR = "rag-embeddings/daft"
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "index.faiss")
META_PATH = os.path.join(EMBEDDINGS_DIR, "index_meta.pkl")


# Sanity checks for test environment

def test_embeddings_files_exist():
    """
    Ensure ingestion has been run before retrieval tests.
    """
    assert os.path.exists(FAISS_INDEX_PATH), "index.faiss not found. Run daft_ingest.py."
    assert os.path.exists(META_PATH), "index_meta.pkl not found. Run daft_ingest.py."

def test_metadata_structure():
    """
    Validate metadata alignment and structure.
    """
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    assert isinstance(metadata, list)
    assert len(metadata) > 0

# Core retrieval tests

def test_basic_retrieval_returns_results():
    """
    Simple sanity test that retrieval works end-to-end.
    """
    docs = retrieve_daft_docs("filter rows where x > 5", top_k=3)

    assert isinstance(docs, list)
    assert len(docs) > 0

    for doc in docs:
        assert isinstance(doc, str)
        assert len(doc.strip()) > 0


def test_filter_semantics():
    """
    Filtering-related queries should retrieve filter/where docs.
    """
    docs = retrieve_daft_docs(
        "Filter rows where x > 10 and y < 20",
        top_k=5,
    )

    combined_text = " ".join(docs).lower()

    assert any(
        keyword in combined_text
        for keyword in ["filter", "where", "predicate"]
    ), "Filter semantics not retrieved"


def test_groupby_and_aggregation_semantics():
    """
    Groupby + aggregation queries should retrieve groupby and agg docs.
    """
    docs = retrieve_daft_docs(
        """
        Group by country
        Compute sum of revenue
        Sort by revenue descending
        """,
        top_k=6,
    )

    combined_text = " ".join(docs).lower()

    assert "group" in combined_text
    assert any(
        agg in combined_text
        for agg in ["sum", "aggregate", "aggregation"]
    )


def test_expression_semantics():
    """
    Expression-heavy queries should retrieve expression-related docs.
    """
    docs = retrieve_daft_docs(
        "Create a new column using a conditional expression and cast it to int",
        top_k=6,
    )

    combined_text = " ".join(docs).lower()

    assert any(
        keyword in combined_text
        for keyword in ["when", "cast", "expression"]
    )


# Edge case tests

def test_empty_query_returns_empty_list():
    """
    Empty pseudocode should not trigger retrieval.
    """
    docs = retrieve_daft_docs("", top_k=5)
    assert docs == []


def test_whitespace_query_returns_empty_list():
    """
    Whitespace-only queries should not trigger retrieval.
    """
    docs = retrieve_daft_docs("   \n  ", top_k=5)
    assert docs == []


def test_garbage_query_does_not_crash():
    """
    Random input should not crash the retriever.
    """
    docs = retrieve_daft_docs("asdfghjkl qwerty zxcvbn", top_k=5)

    assert isinstance(docs, list)
    
