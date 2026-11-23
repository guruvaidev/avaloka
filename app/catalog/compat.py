"""``GET /datasets`` as a view over the catalog (design §C.6, 16g; T16.11).

The datasets tab read the user's Redis session blobs, so a dataset outlived
its session for exactly as long as Redis kept it. The catalog is durable, so
the route now answers with the Redis items it always had **plus** the
catalog's file and derived datasets the sessions no longer mention. Nothing
Redis-only is dropped (that is the bite), and a catalog that is off or
unreachable leaves the route exactly as it was.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Set

logger = logging.getLogger(__name__)

#: Catalog kinds the datasets tab can show as "a dataset": uploads and
#: analysis outputs. Tables and buckets have their own panels.
VIEW_KINDS = ("file", "derived")


def catalog_items(user_id: str, exclude: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """The user's catalog datasets not in ``exclude``, in ``DatasetListItem``
    field shape. Never raises: no catalog, no items."""
    from app.catalog.store import CatalogError, CatalogStore, CatalogUnavailable
    skip: Set[str] = {str(x) for x in exclude}
    try:
        rows = CatalogStore(user_id).list(limit=500)
    except (CatalogUnavailable, CatalogError) as exc:
        logger.debug("[datasets] catalog view unavailable: %s", exc)
        return []
    except Exception:  # noqa: BLE001
        logger.warning("[datasets] catalog view failed", exc_info=True)
        return []
    items: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("kind") not in VIEW_KINDS or r["dataset_id"] in skip:
            continue
        locator = r.get("locator") or {}
        items.append({
            "dataset_id": r["dataset_id"],
            "created_at": r.get("captured_at") or r.get("created_at") or "",
            "size_bytes": int(r.get("bytes") or 0),
            "columns": list(r.get("column_names") or []),
            "session_id": None,
            "group_session_id": None,
            "filename": locator.get("filename") or r.get("display_name") or None,
            "alias": r.get("name"),
            "source_kind": "catalog",
        })
    return items


def merge_dataset_items(redis_items: List[Any], extra: List[Dict[str, Any]], make: Any) -> List[Any]:
    """Redis items first-class, catalog items appended, newest first. ``make``
    builds the response model from a dict (``DatasetListItem``)."""
    out = list(redis_items) + [make(**d) for d in extra]
    out.sort(key=lambda it: getattr(it, "created_at", "") or "", reverse=True)
    return out


__all__ = ["VIEW_KINDS", "catalog_items", "merge_dataset_items"]
