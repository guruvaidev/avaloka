"""Object stores on the connector contract: S3, GCS, Azure, and the local and
in-memory filesystems the tests use (design §B.2 row 2, 15c).

One connector is one bucket (or container) and an optional prefix, reached
through fsspec. Its *credentials* are a :class:`Credentials` whose secret is
a JSON document of storage options -- access key and secret, a service
account, an account key or SAS token, an endpoint -- revealed once, when the
filesystem is built, and never written anywhere else. A signed URL is a
credential too, so a URL with a signature in its query string is refused at
the door: the signature belongs in the sealed options, not in a locator.

What the connector proves, like every other one (``tests/avaloka/
connector_conformance.py``): read, schema and sample are bounded and
receipted; there is no push-down (the local engine filters); ``endpoints()``
names the one host it will open; ``source_version()`` is the object's ETag
(or size and mtime where there is no ETag), and over a prefix it is the set
of them, so a changed or added object moves it.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit

from avaloka.connectors.base import (
    PLANNER_METADATA_MAX_BYTES, READ_MAX_BYTES, READ_MAX_ROWS, SAMPLE_METHODS, Capability, Column,
    ConnectorBase, ConnectorError, Credentials, Endpoint, ReadReceipt, ReadResult, SourceVersion, Table,
    bounded_batches, cap_metadata, requires,
)

#: URL scheme -> provider. ``memory`` and ``file`` are the in-process ones.
PROVIDERS = {"s3": "s3", "s3a": "s3", "gs": "gcs", "gcs": "gcs", "az": "azure", "abfs": "azure",
             "abfss": "azure", "memory": "memory", "file": "file"}
#: Provider -> fsspec protocol, and -> the scheme a store URL is shown with.
PROTOCOLS = {"s3": "s3", "gcs": "gcs", "azure": "az", "memory": "memory", "file": "file"}
SCHEMES = {"s3": "s3", "gcs": "gs", "azure": "az", "memory": "memory", "file": "file"}
#: Object suffixes read as tables.
FORMATS = {".csv": "csv", ".tsv": "csv", ".parquet": "parquet", ".pq": "parquet"}
#: How many objects a catalogue lists; a bucket is not a database.
LIST_LIMIT = 1000
#: Bytes of a CSV read to estimate its row count when it is too large to count.
_ESTIMATE_BYTES = 1 << 20
_SIGNED_QUERY = re.compile(r"(^|[?&])(X-Amz-Signature|X-Amz-Credential|sig|se|sv|Signature|Expires|GoogleAccessId)=",
                           re.IGNORECASE)
_BATCH_ROWS = 65_536


def parse_store_url(url: str) -> tuple[str, str, str]:
    """``(provider, bucket, prefix)`` from a store URL; refuses credentials in it."""
    split = urlsplit(str(url or "").strip())
    scheme = split.scheme.lower()
    if scheme not in PROVIDERS:
        raise ConnectorError(f"unsupported object-store scheme {scheme!r}; one of {sorted(PROVIDERS)}")
    if split.query and _SIGNED_QUERY.search("?" + split.query):
        raise ConnectorError("a signed URL is a credential; put the signature in the sealed options, "
                             "not in the store URL")
    if split.username or split.password:
        raise ConnectorError("credentials do not belong in the store URL")
    provider = PROVIDERS[scheme]
    if provider == "file":
        bucket, prefix = "", (split.netloc + split.path).rstrip("/") or "/"
    else:
        bucket = split.netloc
        if not bucket:
            raise ConnectorError("the store URL names no bucket or container")
        prefix = split.path.strip("/")
    return provider, bucket, prefix


class ObjectStoreConnector(ConnectorBase):
    """One bucket or container, optionally under a prefix, read-only."""

    name = "objectstore"
    capabilities = frozenset({Capability.READ, Capability.SCHEMA, Capability.SAMPLE})
    dialect = None

    def __init__(self, url: str, credentials: Optional[Credentials] = None, *,
                 filesystem: Any = None) -> None:
        self.provider, self.bucket, self.prefix = parse_store_url(url)
        if credentials is not None and not isinstance(credentials, Credentials):
            raise ConnectorError("object-store options must be a Credentials object; the connector "
                                 "does not accept a plain dict or string")
        if credentials is None and self.provider not in ("memory", "file"):
            # Anonymous is a choice, not a fallback: it is made explicit here so a
            # missing key never silently becomes a public read.
            credentials = Credentials(json.dumps({"anonymous": True}))
        self._credentials = credentials or Credentials(json.dumps({"local": True}))
        self._fs = filesystem
        self._options = self._parse_options()
        self._catalogue: Optional[dict[str, Any]] = None
        self._tables: dict[str, Table] = {}

    # -- options --------------------------------------------------------------
    def _parse_options(self) -> dict[str, Any]:
        try:
            options = json.loads(self._credentials.reveal())
        except Exception:  # noqa: BLE001
            raise ConnectorError("object-store options must be a JSON document") from None
        if not isinstance(options, dict):
            raise ConnectorError("object-store options must be a JSON object")
        return options

    def _fs_kwargs(self) -> dict[str, Any]:
        """The options as the fsspec backend wants them. Called once, when the
        filesystem is built; the result is not kept."""
        o = self._options
        if self.provider == "s3":
            kw: dict[str, Any] = {}
            if o.get("anonymous"):
                kw["anon"] = True
            else:
                kw.update(key=o.get("access_key"), secret=o.get("secret_key"), token=o.get("session_token"))
            client_kwargs: dict[str, Any] = {}
            if o.get("endpoint_url"):
                client_kwargs["endpoint_url"] = o["endpoint_url"]
            if o.get("region"):
                client_kwargs["region_name"] = o["region"]
            if client_kwargs:
                kw["client_kwargs"] = client_kwargs
            return {k: v for k, v in kw.items() if v is not None}
        if self.provider == "gcs":
            if o.get("anonymous"):
                return {"token": "anon"}
            sa = o.get("service_account_json")
            if isinstance(sa, str):
                sa = json.loads(sa)
            return {"token": sa} if sa else {}
        if self.provider == "azure":
            kw = {"account_name": o.get("account_name")}
            if o.get("account_key"):
                kw["account_key"] = o["account_key"]
            elif o.get("sas_token"):
                kw["sas_token"] = o["sas_token"]
            elif o.get("connection_string"):
                kw = {"connection_string": o["connection_string"]}
            elif o.get("anonymous"):
                kw["anon"] = True
            return {k: v for k, v in kw.items() if v is not None}
        return {}

    @property
    def fs(self) -> Any:
        if self._fs is None:
            try:
                import fsspec
            except ImportError as exc:  # pragma: no cover
                raise ConnectorError(f"object-store connectors require fsspec: {exc}") from exc
            try:
                self._fs = fsspec.filesystem(PROTOCOLS[self.provider], **self._fs_kwargs())
            except ImportError as exc:
                raise ConnectorError(f"no fsspec driver for {self.provider}: install its backend "
                                     f"({type(exc).__name__})") from None
        return self._fs

    # -- contract plumbing ----------------------------------------------------
    @property
    def credentials(self) -> Credentials:
        return self._credentials

    def describe(self) -> str:
        root = f"{SCHEMES[self.provider]}://{self.bucket}" if self.bucket else "file://"
        return f"{root}/{self.prefix}" if self.prefix and self.bucket else f"{root}{self.prefix}"

    def __repr__(self) -> str:
        return f"ObjectStoreConnector({self.describe()})"

    def close(self) -> None:
        self._fs = None

    def __enter__(self) -> "ObjectStoreConnector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def endpoints(self) -> list[Endpoint]:
        o = self._options
        if self.provider in ("memory", "file"):
            return []
        if self.provider == "s3":
            if o.get("endpoint_url"):
                split = urlsplit(str(o["endpoint_url"]))
                port = split.port or (80 if split.scheme == "http" else 443)
                return [Endpoint(host=split.hostname or "", port=port, scheme=split.scheme or "https")]
            region = o.get("region")
            host = f"s3.{region}.amazonaws.com" if region and region != "us-east-1" else "s3.amazonaws.com"
            return [Endpoint(host=host, port=443, scheme="https")]
        if self.provider == "gcs":
            return [Endpoint(host="storage.googleapis.com", port=443, scheme="https")]
        account = o.get("account_name")
        if not account and o.get("connection_string"):
            m = re.search(r"AccountName=([^;]+)", str(o["connection_string"]))
            account = m.group(1) if m else None
        if not account:
            raise ConnectorError("azure options must name account_name so the endpoint is known")
        return [Endpoint(host=f"{account}.blob.core.windows.net", port=443, scheme="https")]

    # -- paths ----------------------------------------------------------------
    def _root(self) -> str:
        if self.provider == "file":
            return self.prefix
        return f"{self.bucket}/{self.prefix}" if self.prefix else self.bucket

    def _path(self, ref: Optional[str]) -> str:
        """An fsspec path under this connector's root; a ref never escapes it."""
        rel = (ref or "").strip()
        if rel.startswith(("s3://", "gs://", "gcs://", "az://", "abfs://", "abfss://", "memory://", "file://")):
            provider, bucket, prefix = parse_store_url(rel)
            if provider != self.provider or (bucket != self.bucket):
                raise ConnectorError("ref names another store; one connector is one bucket")
            rel = prefix[len(self.prefix):].lstrip("/") if self.prefix and prefix.startswith(self.prefix) else prefix
            if self.provider == "file":
                return prefix
        if ".." in rel.split("/"):
            raise ConnectorError("ref must not climb out of the prefix")
        root = self._root()
        return posixpath.join(root, rel) if rel else root

    def _rel(self, path: str) -> str:
        """A path relative to this connector's root. fsspec backends differ on
        a leading slash (memory keeps it, s3 does not), so both sides lose it."""
        root = self._root().strip("/") + "/"
        clean = path.lstrip("/")
        return clean[len(root):] if clean.startswith(root) else posixpath.basename(path)

    @staticmethod
    def _fmt(path: str) -> Optional[str]:
        return FORMATS.get(posixpath.splitext(path)[1].lower())

    def _objects(self, ref: Optional[str] = None) -> list[dict[str, Any]]:
        """Data objects at ``ref``: one object, or every object under a prefix,
        sorted by name, at most ``LIST_LIMIT``."""
        path = self._path(ref)
        try:
            if self.fs.isfile(path):
                info = self.fs.info(path)
                return [info] if self._fmt(path) else []
            found = self.fs.find(path, maxdepth=None, withdirs=False, detail=True)
        except FileNotFoundError:
            raise ConnectorError(f"no object at {self._rel(path) or '/'}") from None
        infos = [i for i in found.values() if self._fmt(str(i.get("name", "")))]
        infos.sort(key=lambda i: str(i["name"]))
        return infos[:LIST_LIMIT]

    # -- schema -----------------------------------------------------------------
    def _arrow_schema(self, path: str) -> Any:
        fmt = self._fmt(path)
        if fmt == "parquet":
            import pyarrow.parquet as pq
            with self.fs.open(path, "rb") as fh:
                return pq.read_schema(fh)
        import pyarrow.csv as pcsv
        with self.fs.open(path, "rb") as fh:
            reader = pcsv.open_csv(fh, read_options=pcsv.ReadOptions(block_size=1 << 20))
            return reader.schema

    def _table_for(self, info: Mapping[str, Any]) -> Table:
        path = str(info["name"])
        rel = self._rel(path)
        if rel not in self._tables:
            schema = self._arrow_schema(path)
            self._tables[rel] = Table(name=rel, columns=tuple(
                Column(name=f.name, type=str(f.type), nullable=bool(f.nullable)) for f in schema))
        return self._tables[rel]

    @requires(Capability.SCHEMA)
    def catalogue(self) -> dict[str, Any]:
        if self._catalogue is None:
            tables = [self._table_for(i).as_dict() for i in self._objects()]
            self._catalogue = {"backend": self.provider, "bucket": self.bucket, "schema": self.prefix or None,
                               "tables": tables}
        return self._catalogue

    def statistics(self, table: Optional[str] = None) -> dict[str, Any]:
        objects = self._objects(table)
        if not objects:
            raise ConnectorError(f"table {table!r} not found")
        size = sum(int(i.get("size") or 0) for i in objects)
        stats: dict[str, Any] = {"kind": "object", "backend": self.provider, "objects": len(objects),
                                 "size_bytes": size, "n_columns": len(self._table_for(objects[0]).columns)}
        if all(self._fmt(str(i["name"])) == "parquet" for i in objects):
            import pyarrow.parquet as pq
            rows = 0
            for i in objects:
                with self.fs.open(str(i["name"]), "rb") as fh:
                    rows += int(pq.read_metadata(fh).num_rows)
            stats.update(row_count=rows, row_count_estimated=False)
        else:
            stats.update(**self._csv_row_estimate(objects))
        return stats

    def _csv_row_estimate(self, objects: list[Mapping[str, Any]]) -> dict[str, Any]:
        total_size = sum(int(i.get("size") or 0) for i in objects)
        if total_size <= _ESTIMATE_BYTES * 4:
            rows = sum(b.num_rows for b in self._batches(objects))
            return {"row_count": rows, "row_count_estimated": False}
        first = str(objects[0]["name"])
        with self.fs.open(first, "rb") as fh:
            head = fh.read(_ESTIMATE_BYTES)
        lines = max(1, head.count(b"\n") - 1)
        per_row = len(head) / lines
        return {"row_count": int(total_size / per_row), "row_count_estimated": True}

    def planner_metadata(self, max_bytes: int = PLANNER_METADATA_MAX_BYTES) -> dict[str, Any]:
        catalogue = self.catalogue()
        tables = []
        for entry in catalogue["tables"]:
            doc = dict(entry)
            try:
                doc["statistics"] = self.statistics(entry["name"])
            except Exception as exc:  # noqa: BLE001  (an unreadable object is still listed)
                doc["statistics"] = {"statistics_partial": True, "statistics_note": type(exc).__name__}
            tables.append(doc)
        return cap_metadata({"kind": "object_storage", "backend": self.provider, "schema": self.prefix or None,
                             "selected_table": None, "tables": tables}, max_bytes)

    # -- version ------------------------------------------------------------------
    @staticmethod
    def _marker(info: Mapping[str, Any]) -> dict[str, Any]:
        etag = info.get("ETag") or info.get("etag") or info.get("md5Hash") or info.get("content_md5")
        marker: dict[str, Any] = {"size": int(info.get("size") or 0)}
        if etag:
            marker["etag"] = str(etag).strip('"')
        else:
            mtime = info.get("mtime") or info.get("LastModified") or info.get("last_modified") or info.get("created")
            if mtime is not None:
                marker["mtime"] = str(mtime)
        return marker

    def source_version(self, ref: Optional[str] = None) -> SourceVersion:
        objects = self._objects(ref)
        if len(objects) == 1:
            return SourceVersion(ref=self._rel(str(objects[0]["name"])), marker=self._marker(objects[0]))
        markers = {self._rel(str(i["name"])): self._marker(i) for i in objects}
        digest = hashlib.sha256(json.dumps(markers, sort_keys=True).encode()).hexdigest()[:24]
        return SourceVersion(ref=(ref or self.prefix or "/"), marker={"objects": len(markers), "set": digest})

    # -- read -------------------------------------------------------------------
    def _batches(self, objects: list[Mapping[str, Any]]) -> Iterable[Any]:
        import pyarrow.csv as pcsv
        import pyarrow.parquet as pq
        for info in objects:
            path = str(info["name"])
            with self.fs.open(path, "rb") as fh:
                if self._fmt(path) == "parquet":
                    yield from pq.ParquetFile(fh).iter_batches(batch_size=_BATCH_ROWS)
                else:
                    reader = pcsv.open_csv(fh, read_options=pcsv.ReadOptions(block_size=4 << 20))
                    while True:
                        try:
                            yield reader.read_next_batch()
                        except StopIteration:
                            break

    def _result(self, objects: list[Mapping[str, Any]], kept: list[Any], rows: int, used: int,
                stopped: Optional[str], *, ref: Optional[str], method: Optional[str] = None) -> ReadResult:
        import pyarrow as pa
        schema = self._arrow_schema(str(objects[0]["name"]))
        if kept:
            kept = [b if b.schema.equals(schema) else b.cast(schema) for b in kept]
            table = pa.Table.from_batches(kept, schema=schema)
        else:
            table = schema.empty_table()
        receipt = ReadReceipt(connector=self.name, ref=ref or self.prefix or "/", rows=rows, bytes=used,
                              truncated=stopped is not None, stopped_by=stopped, method=method,
                              source_version=self.source_version(ref), columns=tuple(schema.names))
        return ReadResult(table, receipt)

    @requires(Capability.READ)
    def read(self, ref: Any = None, *, max_rows: int = READ_MAX_ROWS, max_bytes: int = READ_MAX_BYTES) -> ReadResult:
        ref = ref if isinstance(ref, str) else None
        objects = self._objects(ref)
        if not objects:
            raise ConnectorError(f"no data object at {ref or self.prefix or '/'}")
        kept, rows, used, stopped = bounded_batches(self._batches(objects), max_rows=max_rows, max_bytes=max_bytes)
        return self._result(objects, kept, rows, used, stopped, ref=ref)

    @requires(Capability.SAMPLE)
    def sample(self, ref: Optional[str] = None, *, n: int = 1000, method: str = "head") -> ReadResult:
        if method not in SAMPLE_METHODS:
            raise ConnectorError(f"sample method {method!r} is not one of {SAMPLE_METHODS}")
        if n <= 0:
            raise ConnectorError("a sample needs n > 0")
        objects = self._objects(ref)
        if not objects:
            raise ConnectorError(f"no data object at {ref or self.prefix or '/'}")
        if method == "head":
            kept, rows, used, _ = bounded_batches(self._batches(objects), max_rows=n, max_bytes=READ_MAX_BYTES)
            return self._result(objects, kept, rows, used, None, ref=ref, method="head")
        if method == "random":
            import numpy as np
            import pyarrow as pa
            full = self.read(ref)
            if full.truncated:
                raise ConnectorError("random sample needs the whole object within the read bound; use head")
            total = full.table.num_rows
            if total <= n:
                return self._result(objects, full.table.to_batches(), total, full.table.nbytes, None, ref=ref, method="random")
            idx = np.sort(np.random.default_rng(0).choice(total, size=n, replace=False))
            picked = full.table.take(pa.array(idx))
            return self._result(objects, picked.to_batches(), picked.num_rows, picked.nbytes, None, ref=ref, method="random")
        raise ConnectorError(f"an object store supports head and random samples, not {method!r}")


__all__ = ["FORMATS", "LIST_LIMIT", "PROTOCOLS", "PROVIDERS", "SCHEMES", "ObjectStoreConnector", "parse_store_url"]
