"""
Shared embedding resolution for the Avaloka memory stack.

Provider priority (first successful wins):
  1. OpenAI            – OPENAI_API_KEY is set → ``text-embedding-3-small`` (or OPENAI_EMBEDDING_MODEL)
  2. Sentence-Transformers – package is installed → local SENTENCE_TRANSFORMER_MODEL  (no API key needed)
  3. Zero-vector       – last resort; logged as WARNING so operators are aware of degraded retrieval

All vectors are padded / trimmed to EMBEDDING_TARGET_DIM so every caller
receives a consistent-length vector regardless of which provider was used.

Environment variables
---------------------
  OPENAI_API_KEY              – activates OpenAI embeddings
  OPENAI_EMBEDDING_MODEL      – model name           (default: text-embedding-3-small)
  SENTENCE_TRANSFORMER_MODEL  – local ST model name  (default: all-MiniLM-L6-v2)
  EMBEDDING_TARGET_DIM        – output vector length  (default: 1536)
"""

import os
import math
import logging
from typing import List
from app.services.memory_runtime import ensure_fallback_allowed

logger = logging.getLogger(__name__)

# Module-level defaults — read once at import so env patching in tests works.
_TARGET_DIM = int(os.environ.get("EMBEDDING_TARGET_DIM", 1536))
_OAI_MODEL  = os.environ.get("OPENAI_EMBEDDING_MODEL",      "text-embedding-3-small")
_ST_MODEL   = os.environ.get("SENTENCE_TRANSFORMER_MODEL",  "all-MiniLM-L6-v2")
# Pin the provider so a shared vector store isn't polluted by mixing
# provider outputs (e.g. a native-1536 OpenAI vector next to a 384-d
# sentence-transformer vector zero-padded to 1536 — their cosine scores are
# meaningless). "auto" keeps the OpenAI→ST→zero cascade; "openai" or
# "sentence_transformers" use only that provider so every stored vector shares
# one embedding space even across a transient provider outage.
_PROVIDER   = os.environ.get("EMBEDDING_PROVIDER", "auto").strip().lower()

# Cache the sentence-transformer model instance to avoid re-loading on every call
_st_instance = None


def _pad_or_trim(vec: List[float], dim: int) -> List[float]:
    """Pad with zeros or trim to *dim* so the caller always gets a consistent length."""
    if len(vec) == dim:
        return vec
    if len(vec) < dim:
        return list(vec) + [0.0] * (dim - len(vec))
    return list(vec[:dim])


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return cosine similarity in [-1, 1]; returns 0.0 for zero vectors."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def embed_text(text: str, target_dim: int = None) -> List[float]:
    """
    Embed *text* and return a float vector of length *target_dim*
    (defaults to EMBEDDING_TARGET_DIM env var, default 1536).

    Walks the provider cascade; the first successful provider wins.
    All exceptions within a provider are caught and logged — the cascade
    continues so the system never hard-fails.
    """
    global _st_instance
    dim = target_dim if target_dim is not None else _TARGET_DIM

    provider = os.environ.get("EMBEDDING_PROVIDER", _PROVIDER).strip().lower()
    allow_openai = provider in ("auto", "openai")
    allow_st     = provider in ("auto", "sentence_transformers")

    api_key = os.environ.get("OPENAI_API_KEY")
    model   = os.environ.get("OPENAI_EMBEDDING_MODEL", _OAI_MODEL)

    if allow_openai and api_key:
        try:
            from langchain_openai import OpenAIEmbeddings
            vec = OpenAIEmbeddings(api_key=api_key, model=model).embed_query(text)
            logger.debug(f"[embed] OpenAI '{model}' → {len(vec)}d (target={dim}d)")
            return _pad_or_trim(vec, dim)
        except Exception as exc:
            # When pinned to OpenAI, do NOT fall through to a different provider:
            # a transient outage must not seed the store with incompatible
            # sentence-transformer vectors. Fail to the zero-vector instead.
            if not allow_st:
                logger.warning(
                    f"[embed] OpenAI embedding failed ({exc}) and provider is pinned "
                    "to 'openai'; returning zero-vector to avoid mixing embedding spaces."
                )
                return [0.0] * dim
            logger.warning(
                f"[embed] OpenAI embedding failed ({exc}). "
                "Falling back to sentence-transformers."
            )

    if not allow_st:
        ensure_fallback_allowed(
            "Memory embeddings",
            f"EMBEDDING_PROVIDER='{provider}' but that provider produced no vector",
        )
        logger.warning(f"[embed] Provider '{provider}' unavailable — returning zero-vector [{dim}d].")
        return [0.0] * dim

    st_model_name = os.environ.get("SENTENCE_TRANSFORMER_MODEL", _ST_MODEL)

    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        if _st_instance is None:
            logger.info(
                f"[embed] Loading sentence-transformer '{st_model_name}' (first call). "
                "Subsequent calls will reuse the cached instance."
            )
            _st_instance = SentenceTransformer(st_model_name)

        raw = _st_instance.encode(text)
        vec = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        logger.info(
            f"[embed] sentence-transformers '{st_model_name}' → {len(vec)}d "
            f"(padded/trimmed to {dim}d). "
            "Set OPENAI_API_KEY for higher-quality OpenAI embeddings."
        )
        return _pad_or_trim(vec, dim)

    except ImportError:
        logger.warning(
            "[embed] sentence-transformers not installed — cannot use local embeddings. "
            "Install with:  pip install sentence-transformers>=2.2.0"
        )
    except Exception as exc:
        logger.warning(f"[embed] sentence-transformers failed ({exc}).")

    ensure_fallback_allowed(
        "Memory embeddings",
        "no OpenAI or sentence-transformers embedding provider is available",
    )

    logger.warning(
        f"[embed] All embedding providers exhausted — returning zero-vector [{dim}d]. "
        "Semantic retrieval will be non-functional until a provider is configured:\n"
        "  Option A: set OPENAI_API_KEY\n"
        "  Option B: pip install sentence-transformers>=2.2.0"
    )
    return [0.0] * dim
