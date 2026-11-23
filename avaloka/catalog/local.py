"""The CLI's catalog: a SQLite file under ``~/.avaloka``, this machine's user.

Every dataset ``avaloka analyze`` (and ``train``, ``coordinate``) reads is
registered here with its columns and a content hash, so ``avaloka catalog
list`` shows what has been analysed and ``avaloka analyze my.<name> --goal ...``
runs on it again by name -- the latest version, or ``my.<name>@3``, or
``analysis:<id>`` for the version a past run read.

Nothing here is required for an analysis: a missing SQLAlchemy (the
``sources`` extra), an unwritable home or a broken file logs once and the run
goes on. ``AVALOKA_CATALOG=off`` turns registration off entirely.
"""

from __future__ import annotations

import getpass
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ENV_HOME = "AVALOKA_HOME"
ENV_USER = "AVALOKA_USER"
ENV_ENABLED = "AVALOKA_CATALOG"
_FALSEY = {"0", "off", "false", "no", "disable", "disabled"}
HASH_LIMIT = 64 * 1024 * 1024
FORMATS = {".csv": "csv", ".tsv": "tsv", ".txt": "csv", ".parquet": "parquet", ".pq": "parquet",
           ".xlsx": "xlsx", ".xlsm": "xlsx", ".xls": "xls", ".json": "json", ".jsonl": "jsonl"}


def home() -> Path:
    return Path(os.environ.get(ENV_HOME) or (Path.home() / ".avaloka")).expanduser()


def catalog_url() -> str:
    """``AVALOKA_CATALOG_DB_URL`` when set, else the SQLite file under :func:`home`."""
    explicit = os.environ.get("AVALOKA_CATALOG_DB_URL", "").strip()
    return explicit or "sqlite:///" + (home() / "catalog.db").as_posix()


def owner() -> str:
    """Whose catalog: ``AVALOKA_USER``, else the OS user. One per machine account."""
    value = os.environ.get(ENV_USER, "").strip()
    if value:
        return value
    try:
        return getpass.getuser() or "local"
    except Exception:  # noqa: BLE001
        return "local"


def enabled() -> bool:
    return os.environ.get(ENV_ENABLED, "").strip().lower() not in _FALSEY


def store():
    """The local :class:`CatalogStore`, or a :class:`CatalogUnavailable` that says
    which extra to install. Imported lazily: ``import avaloka.cli`` must not
    import SQLAlchemy, which is in the ``sources`` extra, not the runtime set."""
    try:
        from avaloka.catalog.store import CatalogStore
    except ImportError as exc:
        raise RuntimeError(f"the catalog needs SQLAlchemy: pip install 'avaloka[sources]' ({exc})") from None
    return CatalogStore(owner(), catalog_url())


def is_reference(text: str) -> bool:
    """``my.<name>``, ``my.<name>@3`` or ``analysis:<id>`` -- a catalog reference, not a path."""
    t = (text or "").strip()
    return t.startswith("my.") or t.startswith("analysis:")


def file_version(path: Path) -> Dict[str, Any]:
    """Size plus content hash (mtime instead past ``HASH_LIMIT``): the same for
    an unchanged file, different when it changed."""
    st = path.stat()
    marker: Dict[str, Any] = {"size": int(st.st_size)}
    if st.st_size <= HASH_LIMIT:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        marker["sha256"] = h.hexdigest()
    else:
        marker["mtime_ns"] = int(st.st_mtime_ns)
    return marker


def register_local_file(path: str | os.PathLike[str], *, columns: Sequence[Any], row_count: Optional[int] = None,
                        row_count_estimated: bool = False, analysis_id: Optional[str] = None):
    """Register a local file this CLI is about to analyse. Best-effort: returns
    the :class:`Registration`, or ``None`` when the catalog is off or fails."""
    if not enabled():
        return None
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            return None
        fmt = FORMATS.get(p.suffix.lower(), p.suffix.lower().lstrip(".") or "file")
        home().mkdir(parents=True, exist_ok=True)
        return store().register(
            "file", {"path": str(p), "filename": p.name, "format": fmt},
            [(str(c), "unknown") if not isinstance(c, (tuple, list)) else (str(c[0]), str(c[1])) for c in columns],
            display_name=p.name, source_version={"ref": p.name, "marker": file_version(p)},
            row_count=int(row_count) if row_count is not None else None,
            row_count_kind=("estimate" if row_count_estimated else "exact") if row_count is not None else "unknown",
            analysis_id=analysis_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[catalog] not registered: %s", exc)
        return None


def resolve_reference(reference: str) -> Tuple[Any, Path]:
    """``(Binding, path)`` for a catalog reference, or a ``RuntimeError`` whose
    message is the sentence the user reads."""
    catalog = store()   # a RuntimeError here already names the extra to install
    from avaloka.catalog.store import CatalogError, CatalogUnavailable
    try:
        binding = catalog.resolve(reference)
    except CatalogUnavailable as exc:
        raise RuntimeError(f"your catalog isn't available: {exc}") from None
    except CatalogError as exc:
        raise RuntimeError(str(exc)) from None
    location = binding.locator.get("path") or binding.locator.get("output_location")
    if not location:
        raise RuntimeError(f"{reference} is in your catalog (version {binding.version_no}, "
                           f"{len(binding.columns)} columns) but its file location was not recorded; "
                           "pass the file path instead")
    path = Path(str(location)).expanduser()
    if not path.exists():
        raise RuntimeError(f"{reference} is in your catalog but its file is no longer at {path}; "
                           "point me at the file and I'll pick up its history")
    return binding, path


__all__ = ["ENV_ENABLED", "ENV_HOME", "ENV_USER", "FORMATS", "catalog_url", "enabled", "file_version", "home",
           "is_reference", "owner", "register_local_file", "resolve_reference", "store"]
