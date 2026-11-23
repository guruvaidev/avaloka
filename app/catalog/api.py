"""``/catalog/datasets``: the personal catalog over HTTP.

Every route resolves the caller first and opens a :class:`CatalogStore` bound
to that user, so there is no route that could list another user's datasets.
``POST .../share`` exists in every edition and answers with the edition
sentence outside Enterprise (403), never a silent no-op; inside Enterprise the
shared catalog is not in this build yet, and it says so (501).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, status

from app.catalog.store import CatalogError, CatalogStore, CatalogUnavailable, KINDS
from app.core.editions import Capabilities, explain, resolve_capabilities

logger = logging.getLogger(__name__)

UserResolver = Callable[[Request], Optional[str]]
CapabilityResolver = Callable[[], Capabilities]

#: The one sentence a non-Enterprise caller reads from ``/share``.
SHARE_SENTENCE = ("Sharing datasets is part of Avaloka Enterprise. Your catalog is personal: "
                  "only you can see the datasets in it.")
SHARE_NOT_BUILT = "The shared catalog is not available in this build yet; your datasets stay personal."


def _default_capabilities() -> Capabilities:
    return resolve_capabilities(os.environ.get("AVALOKA_DEPLOYMENT"), os.environ.get("AVALOKA_EDITION"))


def build_router(resolve_user_id: UserResolver, *, capabilities: Optional[CapabilityResolver] = None,
                 store_factory: Optional[Callable[[str], CatalogStore]] = None) -> APIRouter:
    router = APIRouter(prefix="/catalog", tags=["catalog"])
    caps = capabilities or _default_capabilities
    make_store = store_factory or (lambda user_id: CatalogStore(user_id))

    def _user(request: Request) -> str:
        user_id = resolve_user_id(request)
        if not user_id:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
        return str(user_id)

    def _store(user_id: str) -> CatalogStore:
        granted = caps()
        if not granted.dataset_catalog:
            raise HTTPException(status.HTTP_403_FORBIDDEN, explain("dataset_catalog", granted))
        try:
            return make_store(user_id)
        except CatalogUnavailable as exc:
            logger.warning("[catalog] unavailable: %s", exc)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The dataset catalog is not available.")

    async def _call(fn: Callable[[], Any]) -> Any:
        try:
            return await asyncio.to_thread(fn)
        except CatalogError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
        except CatalogUnavailable as exc:
            logger.warning("[catalog] unavailable: %s", exc)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The dataset catalog is not available.")

    @router.get("/datasets")
    async def list_datasets(request: Request, kind: Optional[str] = None, q: Optional[str] = None,
                            since: Optional[datetime] = None, limit: int = 100) -> Dict[str, Any]:
        store = _store(_user(request))
        if kind is not None and kind not in KINDS:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"kind must be one of {', '.join(KINDS)}")
        items = await _call(lambda: store.list(kind=kind, query=q, since=since, limit=max(1, min(limit, 500))))
        return {"datasets": items, "count": len(items)}

    @router.get("/datasets/{dataset_id}")
    async def get_dataset(request: Request, dataset_id: str) -> Dict[str, Any]:
        store = _store(_user(request))
        detail = await _call(lambda: store.get(dataset_id))
        if detail is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
        return detail

    @router.get("/resolve")
    async def resolve(request: Request, ref: str) -> Dict[str, Any]:
        store = _store(_user(request))
        binding = await _call(lambda: store.resolve(ref))
        return binding.as_dict()

    @router.delete("/datasets/{dataset_id}")
    async def delete_dataset(request: Request, dataset_id: str) -> Dict[str, Any]:
        store = _store(_user(request))
        try:
            return await _call(lambda: store.forget(dataset_id))
        except HTTPException as exc:
            if exc.status_code == status.HTTP_400_BAD_REQUEST and "no such dataset" in str(exc.detail):
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
            raise

    @router.post("/datasets/{dataset_id}/share")
    async def share_dataset(request: Request, dataset_id: str) -> Dict[str, Any]:
        _user(request)
        granted = caps()
        if not granted.shared_catalog:
            raise HTTPException(status.HTTP_403_FORBIDDEN, SHARE_SENTENCE)
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, SHARE_NOT_BUILT)

    return router


__all__ = ["SHARE_NOT_BUILT", "SHARE_SENTENCE", "build_router"]
