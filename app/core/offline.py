"""Offline operation: load models from the local cache, or fail saying so.

The API image bakes every model the stack embeds with (see
``scripts/prefetch_models.py``, run at image build time):

* ``all-MiniLM-L6-v2`` -- the memory plane's sentence-transformer;
* ``mixedbread-ai/mxbai-embed-large-v1`` -- the data-transfer agent's Daft
  documentation retrieval, which used to be downloaded at transfer time;
* chroma's ONNX ``all-MiniLM-L6-v2`` -- chroma's default embedding function,
  which used to be fetched from ``chroma-onnx-models.s3.amazonaws.com`` on the
  first query.

and sets ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1``. Without those,
huggingface_hub asks huggingface.co whether a cached model is current on every
load, which a trial run counted 409 times; with them, a model that is not in
the cache cannot be downloaded at all.

This module turns "not in the cache, and not allowed to download" into one
plain :class:`ModelNotCached` error that names the model and the fix, raised
immediately, instead of an HTTP retry loop against a network that is not
there.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_TRUE = ("1", "true", "yes", "on")
OFFLINE_ENVS = ("AVALOKA_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

# Where the image puts chroma's ONNX model. chroma's own default is under
# ~/.cache, which depends on which user the pod runs as; a fixed path does not.
CHROMA_ONNX_ENV = "CHROMA_ONNX_MODEL_DIR"

FIX = ("Run scripts/prefetch_models.py once with network access (the API image "
       "does this at build time), or unset HF_HUB_OFFLINE/AVALOKA_OFFLINE to allow downloads.")


class ModelNotCached(RuntimeError):
    """A model is not in the local cache and this process may not download it."""


def is_offline() -> bool:
    return any((os.environ.get(k) or "").strip().lower() in _TRUE for k in OFFLINE_ENVS)


def _hf_home() -> str:
    return os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")


def load_sentence_transformer(name: str, **kwargs: Any):
    """``SentenceTransformer(name)``, with a plain error when offline and not cached."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    if is_offline():
        # Offline, huggingface_hub's request hook refuses every Hub call before
        # it reaches a socket, but sentence-transformers still BUILDS a HEAD
        # request per model file first. Asking for cached files only skips that,
        # so nothing (and no model name) is ever handed to an HTTP client.
        kwargs.setdefault("local_files_only", True)
    try:
        return SentenceTransformer(name, **kwargs)
    except Exception as exc:
        if is_offline():
            raise ModelNotCached(
                f"Embedding model {name!r} is not in the local cache (HF_HOME={_hf_home()}) "
                f"and offline mode forbids downloading it. {FIX}"
            ) from exc
        raise


def sentence_transformer_ef(model_name: str, **kwargs: Any):
    """chroma's SentenceTransformerEmbeddingFunction, with the same plain error."""
    from chromadb.utils import embedding_functions  # noqa: PLC0415

    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name, **kwargs)
    except Exception as exc:
        if is_offline():
            raise ModelNotCached(
                f"Embedding model {model_name!r} is not in the local cache (HF_HOME={_hf_home()}) "
                f"and offline mode forbids downloading it. {FIX}"
            ) from exc
        raise


def chroma_onnx_dir() -> Path:
    """The directory chroma's default (ONNX all-MiniLM-L6-v2) model lives in."""
    configured = os.environ.get(CHROMA_ONNX_ENV)
    if configured:
        return Path(configured)
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2  # noqa: PLC0415
    return Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH)


def chroma_default_ef():
    """chroma's default embedding function, pinned to the baked model directory.

    Same model and same function name as chroma's implicit default, so a
    collection created without an explicit function opens with this one. When
    offline and the model is absent this raises before chroma would start its
    download from chroma-onnx-models.s3.amazonaws.com.
    """
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2  # noqa: PLC0415

    ef = ONNXMiniLM_L6_V2()
    ef.DOWNLOAD_PATH = chroma_onnx_dir()
    onnx = Path(ef.DOWNLOAD_PATH) / ef.EXTRACTED_FOLDER_NAME / "model.onnx"
    if is_offline() and not onnx.exists():
        raise ModelNotCached(
            f"chroma's embedding model is not in the local cache ({onnx}) and offline "
            f"mode forbids downloading it from chroma-onnx-models.s3.amazonaws.com. {FIX}"
        )
    return ef
