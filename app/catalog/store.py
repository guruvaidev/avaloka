"""The application's catalog store: ``avaloka.catalog.store`` plus the lineage graph.

The store itself lives in the ``avaloka`` package so the CLI can ship it
(``avaloka/catalog/store.py``). This module adds what only the application
has: the catalog follows the lineage database when ``AVALOKA_CATALOG_DB_URL``
is unset, ``forget`` removes the dataset from the lineage graph, and glossary
terms on a column are keyed by the graph's salted column digest. Everything
else is re-exported unchanged, so ``from app.catalog.store import X`` keeps
working for every name it ever had.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

from avaloka.catalog import store as _base
from avaloka.catalog.store import (Binding, CatalogError, CatalogUnavailable, ENV_DB_URL, KINDS,
                                   ROW_COUNT_KINDS, Registration, STATES, TABLES, VISIBILITIES,
                                   canonical_locator, catalog_event, check_locator, metadata,
                                   normalise_columns, reset_for_tests, schema_diff, schema_fingerprint,
                                   size_bucket, slug)
from app.core import lineage as _lineage

if _lineage.database_url not in _base.URL_FALLBACKS:
    _base.URL_FALLBACKS.append(_lineage.database_url)


def database_url() -> Optional[str]:
    """``AVALOKA_CATALOG_DB_URL``, else wherever lineage goes. ``None`` is off."""
    return _base.database_url()


class CatalogStore(_base.CatalogStore):
    """One user's catalog, with the lineage graph attached."""

    def __init__(self, owner_user_id: Optional[str], url: Optional[str] = None,
                 *, on_event: Optional[Callable[[Dict[str, str]], None]] = None) -> None:
        try:
            super().__init__(owner_user_id, url, on_event=on_event)
        except CatalogUnavailable:
            raise CatalogUnavailable(
                f"no catalog database: set {ENV_DB_URL}, {_lineage.ENV_DB_URL} or POSTGRES_URL") from None

    def add_terms(self, dataset_id: str, terms: Iterable[str], *, column: Optional[str] = None,
                  column_digest: Optional[str] = None) -> int:
        if column and column_digest is None and _lineage.database_url():
            column_digest = _lineage.LineageStore(self.owner).column_id(dataset_id, column)
        return super().add_terms(dataset_id, terms, column=column, column_digest=column_digest)

    def forget(self, dataset_id: str, *, lineage: Any = None) -> Dict[str, Any]:
        graph = lineage if lineage is not None else (
            _lineage.LineageStore(self.owner) if _lineage.database_url() else None)
        try:
            return super().forget(dataset_id, lineage=graph)
        finally:
            if lineage is None and graph is not None:
                graph.close()


__all__ = [
    "Binding", "CatalogError", "CatalogStore", "CatalogUnavailable", "ENV_DB_URL", "KINDS",
    "ROW_COUNT_KINDS", "Registration", "STATES", "TABLES", "VISIBILITIES", "canonical_locator",
    "catalog_event", "check_locator", "database_url", "metadata", "normalise_columns", "reset_for_tests",
    "schema_diff", "schema_fingerprint", "size_bucket", "slug",
]
