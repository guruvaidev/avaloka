from __future__ import annotations
import logging
from typing import Optional, Tuple
from urllib.parse import urlparse
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

from app.core.settings import Settings
from app.core.storage import AzureBlobStore, S3BlobStore, GCSBlobStore, IBlobStore

logger = logging.getLogger("avaloka.storage")

settings = Settings()

# Will be set by server.py during startup
blob_store: Optional[IBlobStore] = None


class LocalBlobStore:
    """
    Minimal IBlobStore-compatible implementation backed by the local filesystem.

    - Files are stored under a single root directory.
    - put_file() returns just the key (no "://"), so _store_and_key_from_uri
      will route back to this global instance for reads/stats/deletes.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info("Using LOCAL filesystem storage: root=%s", self.root)

    def _path_for(self, key: str) -> Path:
        key = (key or "").lstrip("/")
        return self.root / key

    def put_file(self, local_path: str | Path, key: str) -> str:
        src = Path(local_path)
        dest = self._path_for(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        # We just return the key; callers store this as data_source_location.
        # _store_and_key_from_uri(..., object_name) will route back to this store.
        return key

    def get_file(self, key: str, dest_path: str | Path) -> None:
        src = self._path_for(key)
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    def delete(self, key: str) -> None:
        try:
            self._path_for(key).unlink(missing_ok=True)
        except TypeError:
            # Python < 3.8 has no missing_ok
            p = self._path_for(key)
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def stat(self, key: str) -> Tuple[int, str]:
        p = self._path_for(key)
        st = p.stat()
        size = st.st_size
        created = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        return size, created

    def signed_url(self, key: str, expiry_minutes: int = 15) -> Optional[str]:
        """Local files have no browser-fetchable URL; callers treat None as unavailable."""
        return None

    def list(self, prefix: str = ""):
        base = self._path_for(prefix) if prefix else self.root
        if not base.exists():
            return []
        for p in base.rglob("*"):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                st = p.stat()
                size = st.st_size
                updated = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
                yield rel, size, updated


def _strip_prefix(key: str, prefix: Optional[str]) -> str:
    key = (key or "").lstrip("/")
    p = (prefix or "").strip("/")
    if not p:
        return key
    if key == p:
        return ""
    if key.startswith(p + "/"):
        return key[len(p) + 1 :]
    return key


def _store_and_key_from_uri(storage_uri: Optional[str], object_name: Optional[str]) -> Tuple[IBlobStore, str]:
    """
    Return (store, key_relative_to_that_store)

    Rules:
    - For s3/gs/azure URIs, we build a store with the "prefix" baked in.
    - The returned key is the filename (optionally prefixed by any extra
      path not covered by the store prefix).
    - For anything else (including local keys w/out "://"), we fall back
      to the global blob_store + object_name.
    """
    try:
        if storage_uri and "://" in storage_uri:
            u = urlparse(storage_uri)
            scheme = (u.scheme or "").lower()
            host = u.hostname or ""
            full_path = (u.path or "").lstrip("/")  # e.g. container/prefix/file.csv or container/prefix/

            # ----- Common path split: prefix vs filename -----
            if full_path.endswith("/"):
                prefix_part = full_path.rstrip("/")
                filename_part = ""
            elif "/" in full_path:
                prefix_part, filename_part = full_path.rsplit("/", 1)
            else:
                prefix_part, filename_part = "", full_path

            # What filename/key do we want?
            wanted_key = (object_name or filename_part or "").lstrip("/")

            # ====================== S3 =======================
            if scheme == "s3":
                bucket = u.netloc
                if settings.storage_backend == "s3" and settings.s3_bucket == bucket and blob_store is not None:
                    rel = _strip_prefix(prefix_part, settings.s3_prefix)
                    if rel:
                        wanted_key = f"{rel}/{wanted_key}".lstrip("/") if wanted_key else rel
                    return blob_store, wanted_key

                return (
                    S3BlobStore(
                        bucket,
                        prefix_part,
                        settings.aws_region,
                        endpoint_url=settings.s3_endpoint_url or None,
                    ),
                    wanted_key,
                )

            # ====================== GCS ======================
            if scheme == "gs":
                bucket = u.netloc
                if settings.storage_backend == "gcs" and settings.gcs_bucket == bucket and blob_store is not None:
                    rel = _strip_prefix(prefix_part, settings.gcs_prefix)
                    if rel:
                        wanted_key = f"{rel}/{wanted_key}".lstrip("/") if wanted_key else rel
                    return blob_store, wanted_key

                return GCSBlobStore(bucket, prefix_part), wanted_key

            # ===================== Azure =====================
            if scheme in ("az", "azure") or host.endswith(".blob.core.windows.net"):
                # az://account/container/prefix/file.csv
                if scheme in ("az", "azure"):
                    account = u.netloc or ""
                    raw_path = (u.path or "").lstrip("/")
                    parts = raw_path.split("/", 1)
                    container = parts[0] if parts else ""
                    prefix_part = parts[1] if len(parts) > 1 else ""
                    sas = getattr(settings, "azure_sas", "")
                else:
                    # https://account.blob.core.windows.net/container/prefix/file.csv?SAS
                    account = host.split(".", 1)[0]
                    raw_path = (u.path or "").lstrip("/")
                    parts = raw_path.split("/", 1)
                    container = parts[0] if parts else ""
                    prefix_part = parts[1] if len(parts) > 1 else ""
                    sas = ("?" + u.query) if u.query else settings.azure_sas

                wanted_key = (object_name or filename_part or "").lstrip("/")
                if not wanted_key and prefix_part and not prefix_part.endswith("/"):
                    # Treat last component as filename
                    if "/" in prefix_part:
                        prefix_part, wanted_key = prefix_part.rsplit("/", 1)
                    else:
                        wanted_key, prefix_part = prefix_part, ""

                if (settings.storage_backend == "azure"
                    and settings.azure_account == account
                    and settings.azure_container == container
                    and blob_store is not None):
                    return blob_store, wanted_key

                return AzureBlobStore(account, container, prefix_part, sas), wanted_key

        # Fallback to default (local or whatever init_blob_store created)
        return blob_store, (object_name or "")

    except Exception:
        logger.exception("Failed to parse storage_uri=%s", storage_uri)
        return blob_store, (object_name or "")


def _key_from_uri(uri: Optional[str]) -> Optional[str]:
    if not uri:
        return None
    if "://" not in uri:
        return uri
    try:
        _, rest = uri.split("://", 1)
        _, key = rest.split("/", 1)
        return key
    except Exception:
        return None


def init_blob_store() -> IBlobStore:
    """
    Called by server.lifespan() to create the right blob_store and set
    storage_service.blob_store.

    Behaviour:
    - If STORAGE_BACKEND=s3/gcs/azure and fully configured -> use that backend.
    - If backend is gcs/azure but required config is missing -> fall back to LOCAL.
    - If STORAGE_BACKEND=local or unset/unknown -> LOCAL.
    """
    global blob_store

    backend = (settings.storage_backend or "").lower()

    def _make_local() -> LocalBlobStore:
        # Prefer explicit LOCAL_STORAGE_ROOT, then DATA_DIR, then ./data/filestore
        root = (
            os.getenv("LOCAL_STORAGE_ROOT")
            or os.getenv("DATA_DIR")
            or str(Path.cwd() / "data" / "filestore")
        )
        return LocalBlobStore(root)

    if backend == "s3":
        if not settings.s3_bucket:
            logger.error("S3 bucket not configured")
            raise RuntimeError("S3 not configured")
        # endpoint_url is what points this at an S3-compatible store (in-cluster
        # MinIO) instead of AWS; empty/None keeps the default AWS endpoints.
        blob_store = S3BlobStore(
            settings.s3_bucket,
            settings.s3_prefix,
            settings.aws_region,
            endpoint_url=settings.s3_endpoint_url or None,
        )
        logger.info(
            "Using S3 storage: bucket=%s prefix=%s endpoint=%s",
            settings.s3_bucket,
            settings.s3_prefix,
            settings.s3_endpoint_url or "aws",
        )

    elif backend in ("gcs", "gs", ""):
        # Backend is GCS (or default) – if bucket is missing, fall back to local
        if not settings.gcs_bucket:
            logger.warning(
                "GCS backend selected but GCS bucket is not configured; "
                "falling back to LOCAL filesystem storage.",
            )
            blob_store = _make_local()
        else:
            blob_store = GCSBlobStore(settings.gcs_bucket, settings.gcs_prefix)
            logger.info("Using GCS storage: bucket=%s prefix=%s", settings.gcs_bucket, settings.gcs_prefix)

    elif backend in ("azure", "az"):
        if not (settings.azure_account and settings.azure_container):
            logger.warning(
                "Azure backend selected but Azure account/container not configured; "
                "falling back to LOCAL filesystem storage.",
            )
            blob_store = _make_local()
        else:
            blob_store = AzureBlobStore(
                settings.azure_account,
                settings.azure_container,
                settings.azure_prefix,
                settings.azure_sas,
            )
            logger.info(
                "Using Azure storage: account=%s container=%s prefix=%s",
                settings.azure_account,
                settings.azure_container,
                settings.azure_prefix,
            )

    else:
        # Explicit 'local' or any unknown backend -> local filesystem
        blob_store = _make_local()

    return blob_store
