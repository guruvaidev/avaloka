"""The dataset catalog (docs/design/1.7-DATA-PLATFORM.md part C).

``store``  -- the tables and :class:`CatalogStore`: register, list, get,
              resolve, forget. SQLite for one process or the CLI, Postgres
              for a deployment; the same SQLAlchemy Core either way.
``hooks``  -- best-effort registration from the analysis path, next to the
              lineage hooks it mirrors.
``api``    -- the ``/catalog/datasets`` router.
"""

from app.catalog.store import (Binding, CatalogError, CatalogStore, CatalogUnavailable, KINDS,
                               Registration, catalog_event, check_locator, database_url,
                               schema_fingerprint, slug)

__all__ = [
    "Binding", "CatalogError", "CatalogStore", "CatalogUnavailable", "KINDS", "Registration",
    "catalog_event", "check_locator", "database_url", "schema_fingerprint", "slug",
]
