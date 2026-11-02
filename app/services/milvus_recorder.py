"""
Milvus Recorder — Celery background task for Layer 4 episodic memory writes.

Environment variables (see embedding_utils.py for provider selection):
  EMBEDDING_PROVIDER          – local (default) | openai (needs EMBEDDING_OPENAI_API_KEY)
  SENTENCE_TRANSFORMER_MODEL  – local model (default: all-MiniLM-L6-v2)
  OPENAI_EMBEDDING_MODEL      – OpenAI model name, opt-in only (default: text-embedding-3-small)
  MILVUS_DEFAULT_NOTEBOOK     – default notebook_id (default: default_notebook)
"""
import logging
import os
from typing import Optional

from app.core.celery_app import celery_app
from app.services.db.milvus_client import milvus_client

logger = logging.getLogger(__name__)

_DEFAULT_NOTEBOOK = os.environ.get("MILVUS_DEFAULT_NOTEBOOK", "default_notebook")


def _record_impl(
    session_id:  str,
    content:     str,
    agent_role:  str           = "assistant",
    notebook_id: Optional[str] = None,
) -> None:
    """
    Core recording logic — separated from the Celery decorator so it can be unit-tested directly.

    Delegates embedding resolution to ``app.services.embedding_utils.embed_text``,
    which embeds locally (sentence-transformers) unless the operator opted in
    to OpenAI, with a zero-vector and a loud WARNING as the last resort.
    """
    effective_notebook = notebook_id or _DEFAULT_NOTEBOOK
    vector_dim         = milvus_client._vector_dim

    logger.info(
        f"Recording execution insight: session='{session_id}', "
        f"notebook='{effective_notebook}', role='{agent_role}'"
    )

    # --- Embedding (local by default; OpenAI only on explicit opt-in) ---
    try:
        from app.services.embedding_utils import embed_text
        vector = embed_text(content, target_dim=vector_dim)
    except Exception as exc:
        logger.error(f"embedding_utils.embed_text raised unexpectedly: {exc}. Using zero-vector.")
        vector = [0.0] * vector_dim

    # --- Milvus write ---
    try:
        if not hasattr(milvus_client, "_collection") or milvus_client._collection is None:
            milvus_client.connect()

        milvus_client.insert_insight(
            session_id=session_id,
            content=content,
            agent_role=agent_role,
            vector=vector,
            notebook_id=effective_notebook,
        )
        logger.info(f"Recorded execution insight in Milvus for session '{session_id}'.")
    except Exception as exc:
        logger.error(f"Failed to record execution to Milvus: {exc}")


@celery_app.task(
    name="app.services.milvus_recorder.record_execution_to_milvus",
    ignore_result=True,
)
def record_execution_to_milvus(
    session_id:  str,
    content:     str,
    agent_role:  str           = "assistant",
    notebook_id: Optional[str] = None,
) -> None:
    """Celery background task wrapper — delegates to _record_impl."""
    _record_impl(session_id, content, agent_role, notebook_id)
