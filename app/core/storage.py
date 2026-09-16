from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, List
import logging
import json
logger = logging.getLogger(__name__)

# Try to import Azure's ResourceNotFoundError, but don't hard-fail if azure isn't installed.
try:
    from azure.core.exceptions import ResourceNotFoundError  # type: ignore
except Exception:  # azure not installed or not complete
    ResourceNotFoundError = Exception  # type: ignore

class IBlobStore(ABC):
    @abstractmethod
    def put_file(self, local_path: Path, object_name: str) -> str: ...
    @abstractmethod
    def get_file(self, object_name: str, local_path: Path) -> None: ...
    @abstractmethod
    def stat(self, object_name: str) -> Tuple[int, str]: ...
    @abstractmethod
    def delete(self, object_name: str) -> None: ...
    @abstractmethod
    def signed_url(self, object_name: str, expiry_minutes: int = 15) -> Optional[str]: ...

def list_hierarchy(
        self, prefix: str = ""
    ) -> Tuple[List[str], List[Tuple[str, int, str]]]:
        """Single-level listing using a '/' delimiter.

        Returns (folders, files):
          - folders: relative folder prefixes ending in '/', e.g. 'sales/2024/'
          - files:   (rel_key, size_bytes, updated_iso) tuples at this level
        Both are relative to the store's configured prefix, matching list().
        """
        raise NotImplementedError

class GCSBlobStore(IBlobStore):
    def __init__(self, bucket: str, prefix: str = "", service_account_json: Optional[str] = None, project: Optional[str] = None):
        from google.cloud import storage

        if service_account_json:
            from google.oauth2 import service_account
            info = json.loads(service_account_json)
            creds = service_account.Credentials.from_service_account_info(info)
            self._client = storage.Client(credentials=creds, project=project or info.get("project_id"))
        else:
            self._client = storage.Client()

        self._bucket = self._client.bucket(bucket)
        self._prefix = prefix.strip("/")

    def _key(self, name: str) -> str:
        return f"{self._prefix}/{name}" if self._prefix else name

    def put_file(self, local_path: Path, object_name: str) -> str:
        key = self._key(object_name)
        self._bucket.blob(key).upload_from_filename(str(local_path))
        return f"gs://{self._bucket.name}/{key}"

    def get_file(self, object_name: str, local_path: Path) -> None:
        self._bucket.blob(self._key(object_name)).download_to_filename(str(local_path))

    def stat(self, object_name: str) -> Tuple[int, str]:
        b = self._bucket.get_blob(self._key(object_name))
        if not b:
            return 0, ""
        return int(b.size or 0), (b.updated.isoformat() if b.updated else "")

    def delete(self, object_name: str) -> None:
        self._bucket.blob(self._key(object_name)).delete()

    def signed_url(self, object_name: str, expiry_minutes: int = 15) -> Optional[str]:
        from datetime import timedelta

        blob = self._bucket.blob(self._key(object_name))
        kwargs = {
            "version": "v4",
            "expiration": timedelta(minutes=expiry_minutes),
            "method": "GET",
        }
        try:
            return blob.generate_signed_url(**kwargs)
        except Exception:
            # Credentials without a private key (e.g. GCE metadata / ADC) cannot
            # sign locally; sign through the IAM SignBlob API instead.
            import google.auth
            from google.auth.transport import requests as gauth_requests

            credentials, _ = google.auth.default()
            credentials.refresh(gauth_requests.Request())
            return blob.generate_signed_url(
                service_account_email=credentials.service_account_email,
                access_token=credentials.token,
                **kwargs,
            )

    def list(self, prefix: str = ""):
        """
        yield (key, size, updated_iso)
        """
        actual_prefix = self._key(prefix)
        for blob in self._client.list_blobs(self._bucket, prefix=actual_prefix):
            rel_key = (
                blob.name[len(self._prefix) + 1 :]
                if self._prefix and blob.name.startswith(self._prefix + "/")
                else blob.name
            )
            yield (
                rel_key,
                int(blob.size or 0),
                blob.updated.isoformat() if blob.updated else "",
            )
    
    def _strip_store_prefix(self, raw_key: str) -> str:
        if self._prefix and raw_key.startswith(self._prefix + "/"):
            return raw_key[len(self._prefix) + 1:]
        return raw_key

    def list_hierarchy(self, prefix: str = ""):
        actual_prefix = self._key(prefix)
        if actual_prefix and not actual_prefix.endswith("/"):
            actual_prefix += "/"

        # NOTE: iterator.prefixes is only populated AFTER the blobs are
        # consumed, so we must materialize files first, then read prefixes.
        iterator = self._client.list_blobs(
            self._bucket, prefix=actual_prefix, delimiter="/"
        )
        files = []
        for blob in iterator:
            if blob.name == actual_prefix:          # current-dir placeholder
                continue
            rel = self._strip_store_prefix(blob.name)
            if rel.endswith("/"):                   # folder marker object
                continue
            files.append((
                rel,
                int(blob.size or 0),
                blob.updated.isoformat() if blob.updated else "",
            ))
        folders = [self._strip_store_prefix(p) for p in iterator.prefixes]
        return sorted(folders), files


class S3BlobStore(IBlobStore):
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        import boto3
        client_kwargs = {
            "region_name": region or None,
        }
        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key
            if session_token:
                client_kwargs["aws_session_token"] = session_token
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        self._s3 = boto3.client("s3", **client_kwargs)
        self._bucket = bucket
        self._prefix = prefix.strip("/")


    def _key(self, name: str) -> str:
        return f"{self._prefix}/{name}" if self._prefix else name

    def put_file(self, local_path: Path, object_name: str) -> str:
        key = self._key(object_name)
        self._s3.upload_file(str(local_path), self._bucket, key)
        return f"s3://{self._bucket}/{key}"

    def get_file(self, object_name: str, local_path: Path) -> None:
        self._s3.download_file(self._bucket, self._key(object_name), str(local_path))

    def stat(self, object_name: str) -> Tuple[int, str]:
        key = self._key(object_name)
        try:
            head = self._s3.head_object(Bucket=self._bucket, Key=key)
            lm = head.get("LastModified")
            return int(head.get("ContentLength", 0)), (lm.isoformat() if lm else "")
        except Exception:
            return 0, ""

    def delete(self, object_name: str) -> None:
        self._s3.delete_object(Bucket=self._bucket, Key=self._key(object_name))

    def signed_url(self, object_name: str, expiry_minutes: int = 15) -> Optional[str]:
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": self._key(object_name)},
            ExpiresIn=expiry_minutes * 60,
        )

    def list(self, prefix: str = ""):
        """
        yield (key, size, updated_iso)
        """
        actual_prefix = self._key(prefix)
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=actual_prefix):
            for obj in page.get("Contents", []):
                raw_key = obj["Key"]
                # make it relative to self._prefix
                if self._prefix and raw_key.startswith(self._prefix + "/"):
                    rel_key = raw_key[len(self._prefix) + 1 :]
                else:
                    rel_key = raw_key
                lm = obj.get("LastModified")
                yield (
                    rel_key,
                    int(obj.get("Size", 0)),
                    lm.isoformat() if lm else "",
                )
    
    def _strip_store_prefix(self, raw_key: str) -> str:
        if self._prefix and raw_key.startswith(self._prefix + "/"):
            return raw_key[len(self._prefix) + 1:]
        return raw_key

    def list_hierarchy(self, prefix: str = ""):
        actual_prefix = self._key(prefix)
        if actual_prefix and not actual_prefix.endswith("/"):
            actual_prefix += "/"

        paginator = self._s3.get_paginator("list_objects_v2")
        folders, files = [], []
        for page in paginator.paginate(
            Bucket=self._bucket, Prefix=actual_prefix, Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                folders.append(self._strip_store_prefix(cp["Prefix"]))
            for obj in page.get("Contents", []):
                raw_key = obj["Key"]
                if raw_key == actual_prefix:        # current-dir placeholder
                    continue
                rel = self._strip_store_prefix(raw_key)
                if rel.endswith("/"):
                    continue
                lm = obj.get("LastModified")
                files.append((
                    rel,
                    int(obj.get("Size", 0)),
                    lm.isoformat() if lm else "",
                ))
        return sorted(folders), files



class AzureBlobStore(IBlobStore):
    """
    Azure Blob store using a container-level SAS token.

    We NEVER sign requests ourselves; we just stick the SAS query string
    onto the blob URL and let Azure validate it.
    """

    def __init__(self, account: str, container: str, prefix: str = "", sas_token: str = ""):
        from azure.storage.blob import ContainerClient  # type: ignore

        self.account = account
        self.container = container
        self.prefix = (prefix or "").strip("/")

        # Normalize SAS: always start with "?"
        self.sas = ""
        if sas_token:
            s = sas_token.strip()
            self.sas = "?" + s.lstrip("?")

        base = f"https://{self.account}.blob.core.windows.net/{self.container}"
        # ContainerClient will use the SAS embedded in the URL
        self._container = ContainerClient.from_container_url(base + self.sas)

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def put_file(self, local_path: Path, object_name: str) -> str:
        """
        Upload a file using the high-level upload_blob API.
        This avoids manual stage_block/commit_block_list, which were
        causing InvalidBlobOrBlock.
        """
        from azure.storage.blob import BlobClient  # type: ignore

        key = self._key(object_name)
        blob_url = (
            f"https://{self.account}.blob.core.windows.net/"
            f"{self.container}/{key}{self.sas}"
        )
        blob = BlobClient.from_blob_url(blob_url)

        with open(local_path, "rb") as data:
            # overwrite=True is important if the blob may already exist
            blob.upload_blob(data, overwrite=True)

        logger.info(
            "AzureBlobStore.put_file: uploaded '%s' to container '%s'",
            key,
            self.container,
        )
        return f"az://{self.account}/{self.container}/{key}"

    def get_file(self, object_name: str, local_path: Path) -> None:
        from azure.storage.blob import BlobClient  # type: ignore

        key = self._key(object_name)
        blob_url = (
            f"https://{self.account}.blob.core.windows.net/"
            f"{self.container}/{key}{self.sas}"
        )
        blob = BlobClient.from_blob_url(blob_url)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stream = blob.download_blob()
            with open(local_path, "wb") as f:
                for chunk in stream.chunks():
                    f.write(chunk)
            logger.info(
                "AzureBlobStore.get_file: downloaded '%s' from container '%s' to '%s'",
                key,
                self.container,
                local_path,
            )
        except ResourceNotFoundError:
            logger.error(
                "AzureBlobStore.get_file: blob '%s' not found in container '%s'",
                key,
                self.container,
            )
            raise
        except Exception as e:
            logger.error(
                "AzureBlobStore.get_file: failed downloading '%s' from container '%s': %s",
                key,
                self.container,
                e,
                exc_info=True,
            )
            raise

    def stat(self, object_name: str) -> Tuple[int, str]:
        key = self._key(object_name)
        blob = self._container.get_blob_client(key)
        try:
            props = blob.get_blob_properties()
            size = int(props.size or 0)
            updated = props.last_modified.isoformat() if props.last_modified else ""
            return size, updated
        except ResourceNotFoundError:
            return 0, ""
        except Exception as e:
            logger.warning(
                "AzureBlobStore.stat: failed to stat '%s' in container '%s': %s",
                key,
                self.container,
                e,
                exc_info=True,
            )
            return 0, ""

    def signed_url(self, object_name: str, expiry_minutes: int = 15) -> Optional[str]:
        """
        Blob URL authorised by the container SAS token. The SAS itself is the
        credential, so its own expiry (not expiry_minutes) bounds validity.
        """
        if not self.sas:
            return None
        key = self._key(object_name)
        return (
            f"https://{self.account}.blob.core.windows.net/"
            f"{self.container}/{key}{self.sas}"
        )

    def delete(self, object_name: str) -> None:
        key = self._key(object_name)
        blob = self._container.get_blob_client(key)
        try:
            blob.delete_blob()
        except ResourceNotFoundError:
            logger.info(
                "AzureBlobStore.delete: blob '%s' did not exist in container '%s'",
                key,
                self.container,
            )
        except Exception as e:
            logger.warning(
                "AzureBlobStore.delete: failed to delete '%s' in container '%s': %s",
                key,
                self.container,
                e,
                exc_info=True,
            )

    def list(self, prefix: str = ""):
        actual_prefix = self._key(prefix.strip().lstrip("/"))
        for b in self._container.list_blobs(name_starts_with=actual_prefix):
            raw_key = b.name
            if self.prefix and raw_key.startswith(self.prefix + "/"):
                rel_key = raw_key[len(self.prefix) + 1 :]
            else:
                rel_key = raw_key
            yield (
                rel_key,
                int(b.size or 0),
                b.last_modified.isoformat() if b.last_modified else "",
            )
    
    def _strip_store_prefix(self, raw_key: str) -> str:
        if self.prefix and raw_key.startswith(self.prefix + "/"):
            return raw_key[len(self.prefix) + 1:]
        return raw_key

    def list_hierarchy(self, prefix: str = ""):
        from azure.storage.blob import BlobPrefix  # type: ignore

        actual_prefix = self._key(prefix.strip().lstrip("/"))
        if actual_prefix and not actual_prefix.endswith("/"):
            actual_prefix += "/"

        folders, files = [], []
        for item in self._container.walk_blobs(
            name_starts_with=actual_prefix, delimiter="/"
        ):
            if isinstance(item, BlobPrefix):
                folders.append(self._strip_store_prefix(item.name))
                continue
            if item.name == actual_prefix:
                continue
            rel = self._strip_store_prefix(item.name)
            if rel.endswith("/"):
                continue
            files.append((
                rel,
                int(item.size or 0),
                item.last_modified.isoformat() if item.last_modified else "",
            ))
        return sorted(folders), files
