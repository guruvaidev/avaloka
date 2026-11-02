"""
Shared embedding resolution for the Avaloka memory stack.

Embeddings are LOCAL by default. The text embedded here is memory text: user
queries, learned hints and preferences, LLM answers, and cached schema JSON
(table names, column names, sample values). It leaves the cluster only when
the operator opts in explicitly.

Provider selection (``EMBEDDING_PROVIDER``):
  local (default)  – the bundled sentence-transformer (SENTENCE_TRANSFORMER_MODEL);
                     a zero-vector, logged as WARNING, only if it cannot load
  openai           – OpenAI ``text-embedding-3-small`` (or OPENAI_EMBEDDING_MODEL),
                     used ONLY when EMBEDDING_OPENAI_API_KEY is also set; without
                     that key it is treated as local, with a warning
  auto             – deprecated; treated as local, with a warning

OPENAI_API_KEY is never read here. It is the chat key, and the inference router
also accepts it for a self-hosted OpenAI-spec server, so its presence says
nothing about wanting memory text sent to api.openai.com. Until this change the
"auto" default did exactly that whenever the key was set.

All vectors are padded / trimmed to EMBEDDING_TARGET_DIM so every caller
receives a consistent-length vector regardless of which provider was used.

Environment variables
---------------------
  EMBEDDING_PROVIDER          – local (default) | openai
  EMBEDDING_OPENAI_API_KEY    – required for the openai opt-in
  OPENAI_EMBEDDING_MODEL      – model name           (default: text-embedding-3-small)
  SENTENCE_TRANSFORMER_MODEL  – local ST model name  (default: all-MiniLM-L6-v2)
  EMBEDDING_TARGET_DIM        – output vector length  (default: 1536)
"""

import os
import math
import logging
import threading
from typing import List, Sequence
from app.services.memory_runtime import ensure_fallback_allowed

logger = logging.getLogger(__name__)

# Module-level defaults — read once at import so env patching in tests works.
_TARGET_DIM = int(os.environ.get("EMBEDDING_TARGET_DIM", 1536))
_OAI_MODEL  = os.environ.get("OPENAI_EMBEDDING_MODEL",      "text-embedding-3-small")
_ST_MODEL   = os.environ.get("SENTENCE_TRANSFORMER_MODEL",  "all-MiniLM-L6-v2")
# The provider is pinned, never a cascade between providers: a shared vector
# store must not mix outputs (a native-1536 OpenAI vector next to a 384-d
# sentence-transformer vector zero-padded to 1536 -- their cosine scores are
# meaningless). See same_embedding_space() for how mixed stores are detected.
LOCAL, OPENAI = "local", "openai"
_LOCAL_NAMES = ("", LOCAL, "sentence_transformers", "sentence-transformers")
OPENAI_KEY_ENV = "EMBEDDING_OPENAI_API_KEY"

# Each misconfiguration is warned about once per process, not on every embed.
_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg)


def resolve_provider() -> str:
    """Return the embedding provider in effect: ``"local"`` or ``"openai"``.

    OpenAI only on an explicit opt-in: EMBEDDING_PROVIDER=openai AND
    EMBEDDING_OPENAI_API_KEY. Everything else is local.
    """
    raw = (os.environ.get("EMBEDDING_PROVIDER") or LOCAL).strip().lower()
    if raw == OPENAI:
        if (os.environ.get(OPENAI_KEY_ENV) or "").strip():
            return OPENAI
        _warn_once(
            "openai-no-key",
            f"[embed] EMBEDDING_PROVIDER=openai but {OPENAI_KEY_ENV} is not set; embedding "
            "LOCALLY instead. OPENAI_API_KEY is not used for embeddings: set "
            f"{OPENAI_KEY_ENV} to opt in to sending memory text to OpenAI.",
        )
        return LOCAL
    if raw == "auto":
        _warn_once(
            "auto",
            "[embed] EMBEDDING_PROVIDER=auto is deprecated and now means local. It used to "
            "send memory text to OpenAI whenever OPENAI_API_KEY was set. Set "
            "EMBEDDING_PROVIDER=local, or EMBEDDING_PROVIDER=openai with "
            f"{OPENAI_KEY_ENV} to opt in to OpenAI.",
        )
        return LOCAL
    if raw not in _LOCAL_NAMES:
        _warn_once(
            f"unknown:{raw}",
            f"[embed] Unknown EMBEDDING_PROVIDER={raw!r}; embedding locally.",
        )
    return LOCAL


def effective_dim(vec: Sequence[float]) -> int:
    """Length of *vec* up to its last non-zero component.

    A local 384-d vector padded to 1536 has effective_dim 384; a native 1536-d
    OpenAI vector has 1536. A real component is exactly 0.0 with negligible
    probability, so this identifies which embedding space a stored vector came
    from without any metadata.
    """
    for i in range(len(vec) - 1, -1, -1):
        if vec[i]:
            return i + 1
    return 0


def same_embedding_space(a: Sequence[float], b: Sequence[float]) -> bool:
    """False when *a* and *b* visibly come from different embedding spaces.

    Different lengths, or different effective dimensions (see effective_dim),
    mean cosine similarity between them is meaningless. A zero vector carries no
    space, so it never counts as a mismatch. When the target dimension is at or
    below both models' native dimensions, vectors from different providers are
    indistinguishable here; only a re-embed fixes that store.
    """
    if len(a) != len(b):
        return False
    da, db = effective_dim(a), effective_dim(b)
    return da == 0 or db == 0 or da == db


# Cache the sentence-transformer model instance to avoid re-loading on every call
_st_instance = None
# Two first requests at once must not both load the model.
_st_lock = threading.Lock()


def _get_st_model(st_model_name: str):
    """Load the sentence-transformer once (thread-safe) and return it."""
    global _st_instance
    if _st_instance is None:
        with _st_lock:
            if _st_instance is None:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
                logger.info(
                    f"[embed] Loading sentence-transformer '{st_model_name}' (first call). "
                    "Subsequent calls will reuse the cached instance."
                )
                _st_instance = SentenceTransformer(st_model_name)
    return _st_instance


def warmup_local_model() -> bool:
    """Load and run the local model once, on the CALLING thread (server startup).

    Loading torch/sentence-transformers lazily inside a request worker thread
    crashed the process with SIGSEGV (exit 139). The local model is loaded
    whenever the local provider is in effect, which is the default.
    Returns True when the model is loaded and working.
    """
    provider = resolve_provider()
    if provider != LOCAL:
        logger.info("[embed] warm-up skipped: provider pinned to %r", provider)
        return False
    st_model_name = os.environ.get("SENTENCE_TRANSFORMER_MODEL", _ST_MODEL)
    try:
        _get_st_model(st_model_name).encode("warmup")
        return True
    except ImportError:
        logger.info("[embed] warm-up skipped: sentence-transformers not installed")
        return False


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

    Uses the provider resolve_provider() selects -- local unless the operator
    opted in to OpenAI. Provider errors are caught and logged; the last resort is
    a zero-vector, so the system never hard-fails. There is no fall-through from
    one provider to the other (see the note on _LOCAL_NAMES).
    """
    dim = target_dim if target_dim is not None else _TARGET_DIM
    provider = resolve_provider()

    if provider == OPENAI:
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", _OAI_MODEL)
        try:
            from langchain_openai import OpenAIEmbeddings
            vec = OpenAIEmbeddings(
                api_key=os.environ[OPENAI_KEY_ENV].strip(), model=model,
            ).embed_query(text)
            logger.debug(f"[embed] OpenAI '{model}' → {len(vec)}d (target={dim}d)")
            return _pad_or_trim(vec, dim)
        except Exception as exc:
            # Do NOT fall through to the local model: a transient outage must
            # not seed the store with vectors from a second embedding space.
            ensure_fallback_allowed(
                "Memory embeddings",
                f"EMBEDDING_PROVIDER='openai' but OpenAI produced no vector ({exc})",
            )
            logger.warning(
                f"[embed] OpenAI embedding failed ({exc}) and provider is pinned "
                "to 'openai'; returning zero-vector to avoid mixing embedding spaces."
            )
            return [0.0] * dim

    st_model_name = os.environ.get("SENTENCE_TRANSFORMER_MODEL", _ST_MODEL)

    try:
        raw = _get_st_model(st_model_name).encode(text)
        vec = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        logger.debug(
            f"[embed] sentence-transformers '{st_model_name}' → {len(vec)}d "
            f"(padded/trimmed to {dim}d)."
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
        "the local sentence-transformers embedding model is not available",
    )

    logger.warning(
        f"[embed] Local embedding model unavailable — returning zero-vector [{dim}d]. "
        "Semantic retrieval will be non-functional until it is fixed:\n"
        "  pip install sentence-transformers>=2.2.0, and make the model available "
        "(the API image bakes it in; see docs/INSTALL.md)."
    )
    return [0.0] * dim
