"""
Persistent Chroma Client for Usage Context (Layer 2)
Stores user logic signatures and styling preferences locally across sessions.

Environment variables:
  CHROMA_STORAGE_PATH      – Path for persistent ChromaDB storage
                             (default: <repo-root>/artifacts/chromadb_storage)
  CHROMA_COLLECTION        – ChromaDB collection name
                             (default: layer2_usage_context)
  CHROMA_SIGNATURE_CAP     – Max in-memory signatures per user in fallback store
                             (default: 30)
  CHROMA_RETENTION_DAYS    – Discard signatures older than this many days
                             (default: 30)
"""
import os
import logging
from typing import Dict, Optional

try:
    import chromadb
except ImportError:
    chromadb = None

logger = logging.getLogger(__name__)

from app.services.memory_runtime import ensure_fallback_allowed


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


_RETENTION_DAYS = _env_int("CHROMA_RETENTION_DAYS", 30)


class ChromaClientImpl:
    def __init__(self):
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        default_storage = os.path.join(base_dir, "artifacts", "chromadb_storage")
        self.storage_path    = os.environ.get("CHROMA_STORAGE_PATH", default_storage)
        self._collection_name = os.environ.get("CHROMA_COLLECTION",  "layer2_usage_context")
        self._signature_cap   = _env_int("CHROMA_SIGNATURE_CAP", 30)
        self._retention_days  = _env_int("CHROMA_RETENTION_DAYS", _RETENTION_DAYS)

        os.makedirs(self.storage_path, exist_ok=True)

        self.client     = None
        self.collection = None
        self._fallback_namespaces: Dict[str, Dict[str, list]] = {}
        self.use_mock   = False

        logger.info(
            f"Initialized ChromaClientImpl — storage={self.storage_path}, "
            f"collection={self._collection_name}, signature_cap={self._signature_cap}"
        )

    def connect(self) -> bool:
        """Connect to persistent ChromaDB; fall back to in-process dict on failure."""
        if chromadb is None:
            ensure_fallback_allowed("Layer 2 ChromaDB", "chromadb package is not installed")
            logger.warning(
                "chromadb package not installed — Layer 2 falling back to in-process mock. "
                "Install with: pip install chromadb>=0.4.0"
            )
            self.use_mock = True
            return False

        try:
            chroma_host = os.environ.get("CHROMA_HOST")
            chroma_port = os.environ.get("CHROMA_PORT", "8000")
            
            if chroma_host:
                self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
                logger.info(f"Connecting to remote ChromaDB at {chroma_host}:{chroma_port}")
            else:
                self.client = chromadb.PersistentClient(path=self.storage_path)
                
            self.collection = self.client.get_or_create_collection(
                name=self._collection_name,
                metadata={"description": "User-specific ETL style and logic configuration mapping"},
            )
            self.use_mock = False
            logger.info(f"ChromaClientImpl connected — collection='{self._collection_name}'.")
            return True
        except Exception as e:
            ensure_fallback_allowed("Layer 2 ChromaDB", f"bootstrap failed: {e}")
            logger.error(
                f"Failed to bootstrap ChromaDB: {e}. "
                "Activating in-process fallback."
            )
            self.use_mock = True
            return False

    def update_user_signature(self, user_id: str, signature: str) -> None:
        """Append a logic signature for a user (bounded by CHROMA_SIGNATURE_CAP in fallback)."""
        import uuid
        import time

        now        = int(time.time())
        cutoff     = now - (self._retention_days * 86400)
        doc_id     = f"{user_id}_{now}_{uuid.uuid4().hex[:6]}"

        if self.use_mock or not self.collection:
            ns = self._fallback_namespaces.setdefault(user_id, {"signatures": [], "timestamps": []})
            # Ids are kept alongside so a single signature can be deleted;
            # namespaces written without them get placeholder ids.
            stamps = ns.get("timestamps", [])
            ids = ns.get("ids") or [f"{user_id}_{i}" for i in range(len(ns["signatures"]))]
            kept = [
                (sig, ts, sid)
                for sig, ts, sid in zip(ns["signatures"], stamps, ids)
                if ts >= cutoff
            ]
            ns["signatures"]  = [p[0] for p in kept]
            ns["timestamps"]  = [p[1] for p in kept]
            ns["ids"]         = [p[2] for p in kept]
            if len(ns["signatures"]) >= self._signature_cap:
                ns["signatures"].pop(0)
                ns["timestamps"].pop(0)
                ns["ids"].pop(0)
            ns["signatures"].append(signature)
            ns["timestamps"].append(now)
            ns["ids"].append(doc_id)
            logger.debug(
                f"Appended signature for '{user_id}' in fallback "
                f"({len(ns['signatures'])}/{self._signature_cap}, retention={self._retention_days}d)."
            )
            return

        self.collection.upsert(
            documents=[signature],
            metadatas=[{
                "purpose":   "logic_styling",
                "user_id":   user_id,
                "timestamp": now,
            }],
            ids=[doc_id],
        )
        logger.debug(f"Appended logic signature for '{user_id}' in ChromaDB.")

    def get_user_signature(self, user_id: str) -> Optional[str]:
        """Return a merged signature string for a user within the retention window, or None."""
        import time
        cutoff = int(time.time()) - (self._retention_days * 86400)

        if self.use_mock or not self.collection:
            ns = self._fallback_namespaces.get(user_id)
            if ns and ns.get("signatures"):
                valid = [
                    sig for sig, ts in zip(
                        ns["signatures"],
                        ns.get("timestamps", [0] * len(ns["signatures"]))
                    )
                    if ts >= cutoff
                ]
                return " ".join(valid) if valid else None
            return None

        results = self.collection.get(
            where={
                "$and": [
                    {"user_id":   {"$eq": user_id}},
                    {"timestamp": {"$gte": cutoff}},
                ]
            },
            include=["documents", "metadatas"],
        )
        documents = (results or {}).get("documents") or []
        if documents:
            # collection.get() returns rows in unspecified order, so
            # documents[-10:] is not the 10 most recent. Order by the stored
            # timestamp and keep the 10 newest for a deterministic merge.
            metadatas = results.get("metadatas") or []
            paired = [
                (doc, (metadatas[i] or {}).get("timestamp", 0) if i < len(metadatas) else 0)
                for i, doc in enumerate(documents)
            ]
            paired.sort(key=lambda dm: dm[1])
            return " ".join(doc for doc, _ in paired[-10:])

        return None

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------
    # Signatures are filed under the key in the "user_id" metadata field
    # (memory_scope: "<user>#pref", "<user>#thread:<id>", or a bare user or
    # session id for rows written before the scope rule). Each method returns
    # how many signatures it removed; failures of the real store raise, so a
    # caller never takes "nothing deleted" for "deleted".

    def _delete_where_key(self, keep) -> int:
        """Delete every signature whose key satisfies keep(key)."""
        if self.use_mock or not self.collection:
            doomed = [k for k in self._fallback_namespaces if keep(k)]
            count = sum(len(self._fallback_namespaces[k].get("signatures", [])) for k in doomed)
            for k in doomed:
                self._fallback_namespaces.pop(k, None)
            return count
        # Chroma's metadata filters have no prefix match, so the keys are
        # read and matched here. Ids are collected first: deleting while
        # paging by offset would skip rows.
        ids, offset = [], 0
        while True:
            page = self.collection.get(include=["metadatas"], limit=_DELETE_PAGE, offset=offset)
            page_ids = (page or {}).get("ids") or []
            metas = (page or {}).get("metadatas") or []
            for i, doc_id in enumerate(page_ids):
                meta = (metas[i] if i < len(metas) else None) or {}
                if keep(meta.get("user_id")):
                    ids.append(doc_id)
            if len(page_ids) < _DELETE_PAGE:
                break
            offset += len(page_ids)
        for start in range(0, len(ids), _DELETE_PAGE):
            self.collection.delete(ids=ids[start:start + _DELETE_PAGE])
        return len(ids)

    def delete_keys(self, keys) -> int:
        """Delete every signature filed under exactly one of `keys`."""
        wanted = {str(k) for k in keys or [] if k}
        if not wanted:
            return 0
        if self.use_mock or not self.collection:
            return self._delete_where_key(lambda key: key in wanted)
        found = self.collection.get(where={"user_id": {"$in": sorted(wanted)}}, include=[])
        ids = (found or {}).get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def delete_thread(self, scope: str, thread_id: str) -> int:
        """Delete the signature parts one thread of this user (scope) learned."""
        from app.services.memory_scope import thread_key
        if not scope or not thread_id:
            return 0
        return self.delete_keys([thread_key(scope, thread_id)])

    def delete_user(self, scope: str, legacy_keys=None) -> int:
        """Delete every signature of this user: the bare user key (pre-rule
        rows), every key built from it, and `legacy_keys` (session ids rows
        were filed under before user scoping)."""
        from app.services.memory_scope import owns_key
        if not scope:
            return 0
        legacy = {str(k) for k in legacy_keys or [] if k}
        return self._delete_where_key(lambda key: owns_key(key, scope) or str(key or "") in legacy)

    def delete_by_ids(self, ids, scope: str) -> int:
        """Delete single signatures by id -- only those that belong to scope."""
        from app.services.memory_scope import owns_key
        wanted = {str(i) for i in ids or [] if i}
        if not wanted or not scope:
            return 0
        if self.use_mock or not self.collection:
            count = 0
            for key, ns in self._fallback_namespaces.items():
                if not owns_key(key, scope):
                    continue
                sigs = ns.get("signatures", [])
                stamps = ns.get("timestamps") or [0] * len(sigs)
                row_ids = ns.get("ids") or [f"{key}_{i}" for i in range(len(sigs))]
                kept = [r for r in zip(sigs, stamps, row_ids) if r[2] not in wanted]
                count += len(sigs) - len(kept)
                ns["signatures"] = [r[0] for r in kept]
                ns["timestamps"] = [r[1] for r in kept]
                ns["ids"] = [r[2] for r in kept]
            return count
        found = self.collection.get(ids=sorted(wanted), include=["metadatas"])
        page_ids = (found or {}).get("ids") or []
        metas = (found or {}).get("metadatas") or []
        doomed = [doc_id for i, doc_id in enumerate(page_ids)
                  if owns_key(((metas[i] if i < len(metas) else None) or {}).get("user_id"), scope)]
        if doomed:
            self.collection.delete(ids=doomed)
        return len(doomed)


_DELETE_PAGE = 1000


chroma_client = ChromaClientImpl()
