from __future__ import annotations
import base64
import hashlib
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException, status

from app.core.storage import IBlobStore, GCSBlobStore, S3BlobStore, AzureBlobStore

logger = logging.getLogger("avaloka")

# ---------- Supabase / encryption config ----------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_CLOUD_CONN_TABLE = os.getenv(
    "SUPABASE_CLOUD_CONNECTIONS_TABLE",
    "cloud_datasets",
)

DB_ENCRYPTION_KEY = os.getenv("DB_ENCRYPTION_KEY")
_AESGCM_KEY: Optional[bytes] = None  # cached key

# Fields that hold (or alias) a decrypted credential and must never reach logs
# in plaintext. Kept as the single source of truth for both decryption and
# redaction so a new credential column can't be masked in one place but leaked
# in another.
SENSITIVE_CONNECTION_FIELDS = frozenset({
    "access_key",
    "secret_key",
    "session_token",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "service_account_json",
    "connection_string",
    "account_key",
    "sas",
    "sas_token",
})
# Any field name containing one of these is also treated as a secret, so columns
# added later (e.g. "client_secret", "private_key") are redacted by default.
_SENSITIVE_NAME_HINTS = ("secret", "key", "token", "password", "sas", "credential")


def _is_sensitive_field(name: str) -> bool:
    n = str(name).lower()
    if n in SENSITIVE_CONNECTION_FIELDS:
        return True
    return any(h in n for h in _SENSITIVE_NAME_HINTS)


def _mask_secret_value(v: Any) -> Any:
    if not isinstance(v, str) or not v:
        return v
    # Structured blobs (SA JSON, connection strings, PEM keys) reveal nothing.
    if "\n" in v or len(v) > 40:
        return "***masked***"
    if len(v) > 10:
        return v[:4] + "..." + v[-4:]
    return "***masked***"


def redact_connection(conn: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of a connection row safe to log (secrets masked)."""
    if not isinstance(conn, dict):
        return conn
    return {
        k: (_mask_secret_value(v) if _is_sensitive_field(k) else v)
        for k, v in conn.items()
    }

def _maybe_decrypt_into(data: Dict[str, Any], field: str) -> None:
    """
    Populate data[field] from encrypted columns if present.
    Supports a few common naming patterns:
      - <field>_ciphertext + <field>_iv
      - <field>_ciphertext_b64 + <field>_iv_b64
      - <field>_enc + <field>_iv
    """
    if data.get(field):  # already has plaintext
        if isinstance(data[field], str) and data[field].strip():
            return

    candidates = [
        (f"{field}_ciphertext", f"{field}_iv"),
        (f"{field}_ciphertext_b64", f"{field}_iv_b64"),
        (f"{field}_enc", f"{field}_iv"),
        (f"encrypted_{field}", f"{field}_iv"),
    ]

    for ct_key, iv_key in candidates:
        ct = data.get(ct_key)
        iv = data.get(iv_key)
        if isinstance(ct, str) and isinstance(iv, str) and ct.strip() and iv.strip():
            pt = _decrypt_field(ct, iv)
            if pt:
                data[field] = pt
                return

def _get_aesgcm_key() -> Optional[bytes]:
    """
    Derive AES key from DB_ENCRYPTION_KEY using SHA-256 (matching edge function).

    Currently used for decrypting encrypted fields in cloud connection rows.
    """
    global _AESGCM_KEY
    if _AESGCM_KEY is not None:
        return _AESGCM_KEY

    if not DB_ENCRYPTION_KEY:
        logger.warning("DB_ENCRYPTION_KEY is not set; cannot decrypt cloud credentials")
        return None

    raw = DB_ENCRYPTION_KEY.strip()
    # Hash with SHA-256 to get 32-byte key (matching Supabase edge function)
    key = hashlib.sha256(raw.encode("utf-8")).digest()
    _AESGCM_KEY = key
    return key


def _decrypt_field(ciphertext_b64: Optional[str], iv_b64: Optional[str]) -> Optional[str]:
    """
    Decrypt a single field that was encrypted with AES-GCM using DB_ENCRYPTION_KEY.

    Assumes:
      - iv_b64 is base64(nonce)
      - ciphertext_b64 is base64(ciphertext || tag) from AESGCM.encrypt
    """
    if not ciphertext_b64 or not iv_b64:
        return None

    key = _get_aesgcm_key()
    if not key:
        return None

    try:
        aesgcm = AESGCM(key)
        nonce = base64.b64decode(iv_b64)
        ct = base64.b64decode(ciphertext_b64)
        pt = aesgcm.decrypt(nonce, ct, None)
        return pt.decode("utf-8")
    except Exception:
        logger.exception("[cloud-credentials] failed to decrypt field")
        return None


def encrypt_secret(plaintext: Optional[str]) -> Optional[Dict[str, str]]:
    """
    AES-GCM encrypt a secret for at-rest storage (e.g. in a Redis session).
    Returns {"ciphertext": b64, "iv": b64}, or None if no key is configured.
    """
    if not plaintext:
        return None
    key = _get_aesgcm_key()
    if not key:
        return None
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return {
        "ciphertext": base64.b64encode(ct).decode("ascii"),
        "iv": base64.b64encode(nonce).decode("ascii"),
    }


def decrypt_secret(blob: Optional[Dict[str, str]]) -> Optional[str]:
    """Inverse of encrypt_secret."""
    if not isinstance(blob, dict):
        return None
    return _decrypt_field(blob.get("ciphertext"), blob.get("iv"))


_ICEBERG_METADATA_RE = re.compile(r"(^|/)metadata/[^/]*\.metadata\.json$", re.IGNORECASE)
_HIVE_PARTITION_RE   = re.compile(r"(^|/)[^/]+=[^/]+/")

def detect_folder_table_type(rel_keys: list[str]) -> dict:
    """Classify a folder from object keys relative to the folder root.
    Returns table_type in {iceberg, hive_parquet, parquet_dir, unknown, empty}."""
    keys = [k.lstrip("/") for k in rel_keys if k and not k.endswith("/")]
    if not keys:
        return {"table_type": "empty"}

    # 1) Iceberg — metadata/*.metadata.json present
    meta = [k for k in keys if _ICEBERG_METADATA_RE.search("/" + k)]
    if meta:
        # vN.metadata.json / UUID-suffixed names both sort correctly lexically
        return {"table_type": "iceberg", "read_format": "iceberg",
                "metadata_key": sorted(meta)[-1]}

    parquet = [k for k in keys if k.lower().endswith(".parquet")]

    # 2) Hive-partitioned parquet — any col=value/ segment
    if parquet and any(_HIVE_PARTITION_RE.search("/" + k) for k in parquet):
        return {"table_type": "hive_parquet", "read_format": "parquet",
                "glob": "**/*.parquet", "hive_partitioning": True}

    # 3) Flat multi-file parquet
    if parquet:
        flat = all("/" not in k for k in parquet)
        return {"table_type": "parquet_dir", "read_format": "parquet",
                "glob": "*.parquet" if flat else "**/*.parquet",
                "hive_partitioning": False}

    return {"table_type": "unknown"}

# -------------------------------------------------------------------
# Cloud connection helper (Supabase-backed)
# -------------------------------------------------------------------

async def get_cloud_connection(connection_id: str) -> Dict[str, Any]:
    """
    Look up a cloud storage connection by id from Supabase.

    Expected columns:
      - provider / backend (e.g. "aws", "s3", "azure")
      - region
      - bucket_name / container
      - For AWS:
          * access_key  (PLAINTEXT or encrypted)
          * secret_key  (PLAINTEXT or encrypted)
    """
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or SUPABASE_SERVICE_ROLE_KEY
    if not (SUPABASE_URL and supabase_key):
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Supabase env vars (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY) are not set.",
        )

    try:
        from supabase import create_client  # type: ignore
    except ImportError:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "supabase is not installed. Run: pip install supabase",
        )

    try:
        client = create_client(SUPABASE_URL, supabase_key)
        res = (
            client.table(SUPABASE_CLOUD_CONN_TABLE)
            .select("*")
            .eq("id", connection_id)
            .single()
            .execute()
        )
    except Exception as e:
        logger.exception("[get_cloud_connection] Supabase query failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Supabase query failed for connection_id={connection_id}: {e}",
        )

    data: Dict[str, Any] = getattr(res, "data", None) or {}
    for f in [
        "access_key",
        "secret_key",
        "session_token",
        "service_account_json",
        "connection_string",
        "account_name",
        "account_key",
        "sas",
        "sas_token",
    ]:
        _maybe_decrypt_into(data, f)
    if not data:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Cloud connection not found for id={connection_id}",
        )

    # Normalise provider
    provider = (data.get("provider") or data.get("backend") or "").lower()
    if provider in ("aws_s3", "amazon_s3"):
        provider = "s3"
    data["provider"] = provider

    # helper to strip whitespace
    def _clean(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        v = s.strip()
        return v or None

    # -------- AWS / S3: STRICT + WHITESPACE CLEANUP --------
    if provider in ("aws", "s3"):
        ak = _clean(data.get("access_key") or data.get("aws_access_key_id"))
        sk = _clean(data.get("secret_key") or data.get("aws_secret_access_key"))
        session_token = _clean(data.get("session_token") or data.get("aws_session_token"))

        if not ak or not sk:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "AWS S3 connection is missing access_key/secret_key in Supabase "
                f"for connection_id={connection_id}. Please fix the cloud_datasets row.",
            )

        data["access_key"] = ak
        data["secret_key"] = sk
        if session_token:
            data["session_token"] = session_token

        if data.get("bucket_name"):
            data["bucket_name"] = data["bucket_name"].strip()
        if data.get("region"):
            data["region"] = data["region"].strip()

    # --------- DEBUG LOG (sanitized) ---------
    safe = redact_connection(data)
    logger.info(
        "[get_cloud_connection] id=%s provider=%s bucket/container=%s row=%s",
        connection_id,
        safe.get("provider") or safe.get("backend"),
        safe.get("bucket_name") or safe.get("container"),
        safe,
    )

    return data


# Columns the cloud_datasets row may use to record its owning user. The UI edge
# function scopes by the Supabase auth uid, which our backend resolves as the JWT
# `sub` (see server._resolve_user_id). We try these in order so listing stays
# scoped even if the exact column name differs across schema versions.
_USER_SCOPE_COLUMNS = (
    "user_id", "owner_id", "created_by", "owner", "uid",
    "user_email", "account_id", "profile_id",
)

# Projections to try, widest first. Only the columns the picker below reads --
# select("*") also drags in the `schema` JSONB holding each connection's cached
# column profile: measured at 71.8 MB of a 71.9 MB response for 50 rows (99.8%
# of the payload, none of it used here). That download took ~87s, and since this
# runs twice per transfer (once per endpoint, resolving connection names) it
# added ~3 minutes to a transfer turn -- past the frontend proxy's ~100s
# ceiling, so the UI gave up before the backend replied. The same query with a
# narrow projection returns in 0.6s.
#
# PostgREST rejects the WHOLE query when any named column is absent, and the
# alias spellings below are speculative (this deployment's cloud_datasets has
# none of connection_name/display_name/backend/bucket/container/aws_region), so
# the widest projection is attempted first and we step down on 42703 rather than
# hard-coding one schema. "*" stays last so a genuinely unexpected schema still
# returns rows -- slowly, but correctly.
_LIST_PROJECTIONS = (
    "id,name,connection_name,display_name,provider,backend,"
    "bucket_name,bucket,container,region,aws_region,created_at",
    "id,name,provider,bucket_name,region,created_at",
    "*",
)


async def list_cloud_connections(user_id: Optional[str] = None) -> list[Dict[str, Any]]:
    """List cloud storage connections for one user (metadata only, no secrets).

    Reads the Supabase ``cloud_datasets`` table and returns one lightweight dict
    per connection with just the fields needed to display a picker:
    ``id`` (the connection_id used in transfers), ``name`` (display alias),
    ``provider`` (normalised), ``bucket_name``/``region`` and ``created_at``.

    ``user_id`` scopes the result to the connections that user owns — the same
    per-user filtering the UI's Cloud Dataset panel applies. When ``user_id`` is
    falsy the result is left unscoped (local/dev only); callers behind auth must
    always pass it so a user never sees another user's connections.

    Deliberately does NOT decrypt or return credential fields. Returns ``[]`` on
    any error so the caller can degrade gracefully instead of raising.
    """
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or SUPABASE_SERVICE_ROLE_KEY
    if not (SUPABASE_URL and supabase_key):
        logger.warning("[list_cloud_connections] Supabase env vars not set; returning empty list.")
        return []

    try:
        from supabase import create_client  # type: ignore
    except ImportError:
        logger.warning("[list_cloud_connections] supabase not installed; returning empty list.")
        return []

    try:
        client = create_client(SUPABASE_URL, supabase_key)
        rows = None
        uid = (user_id or "").strip()
        if uid:
            # Prefer an efficient server-side filter on the most likely column.
            # If that column doesn't exist the query errors, so fall back to a
            # full fetch + in-Python scope across candidate columns.
            last_error: Optional[Exception] = None
            for projection in _LIST_PROJECTIONS:
                try:
                    res = (
                        client.table(SUPABASE_CLOUD_CONN_TABLE)
                        .select(projection)
                        .eq("user_id", uid)
                        .order("created_at", desc=True)
                        .execute()
                    )
                    rows = getattr(res, "data", None) or []
                    break
                except Exception as e:
                    # An absent column fails fast and only rules out THIS
                    # projection; keep stepping down before giving up on the
                    # server-side filter entirely.
                    last_error = e
                    logger.debug(
                        "[list_cloud_connections] projection %r rejected: %s",
                        projection, e,
                    )
            if rows is None:
                logger.info(
                    "[list_cloud_connections] server-side user_id filter failed "
                    "for every projection (%s); falling back to in-Python scoping.",
                    last_error,
                )
        if rows is None:
            # Unscoped/fallback path keeps select("*"): which owner column exists
            # is unknown here, and naming a missing one would error the query.
            res = (
                client.table(SUPABASE_CLOUD_CONN_TABLE)
                .select("*")
                .order("created_at", desc=True)
                .execute()
            )
            rows = getattr(res, "data", None) or []
            if uid:
                rows = _scope_rows_to_user(rows, uid)
    except Exception:
        logger.exception("[list_cloud_connections] Supabase query failed")
        return []

    out: list[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        provider = (row.get("provider") or row.get("backend") or "").lower()
        name = row.get("name") or row.get("connection_name") or row.get("display_name")
        out.append({
            "id": row.get("id"),
            "name": name,
            "provider": provider,
            "bucket_name": row.get("bucket_name") or row.get("bucket") or row.get("container"),
            "region": row.get("region") or row.get("aws_region"),
            "created_at": row.get("created_at"),
            "_has_name": bool(name),
            # secrets are simply never copied into this dict
        })
    return out


def _scope_rows_to_user(rows: list, uid: str) -> list:
    """Keep only rows whose owner column matches ``uid``.

    Matches against any of ``_USER_SCOPE_COLUMNS`` that is actually present in the
    data. If none of those columns exist on the rows at all, the schema differs
    from what we expect — we return ``[]`` rather than leak every user's rows.
    """
    present = [c for c in _USER_SCOPE_COLUMNS if any(isinstance(r, dict) and c in r for r in rows)]
    if not present:
        logger.warning(
            "[list_cloud_connections] no known user-scope column found on cloud_datasets "
            "rows (looked for %s); refusing to return unscoped rows.",
            list(_USER_SCOPE_COLUMNS),
        )
        return []
    scoped = []
    for r in rows:
        if isinstance(r, dict) and any(str(r.get(c) or "").strip() == uid for c in present):
            scoped.append(r)
    return scoped


def connection_belongs_to_user(conn: Dict[str, Any], user_id: Optional[str]) -> bool:
    """
    True when the connection row's owner column matches ``user_id``.

    Mirrors ``_scope_rows_to_user``: if the row carries none of the known owner
    columns, refuse rather than treat the connection as shared.
    """
    uid = str(user_id or "").strip()
    if not uid or not isinstance(conn, dict):
        return False
    present = [c for c in _USER_SCOPE_COLUMNS if c in conn]
    if not present:
        return False
    return any(str(conn.get(c) or "").strip() == uid for c in present)


# -------------------------------------------------------------------
# Build IBlobStore from a storage_uri + connection row
# -------------------------------------------------------------------

async def _store_from_connection_uri(
    storage_uri: str,
    conn: Dict[str, Any],
) -> Tuple[IBlobStore, str]:
    """
    Given a storage_uri like:
      - s3://bucket/prefix
      - gs://bucket/prefix
      - az://account/container/prefix
    and a Supabase connection row, return (store, key_prefix).

    For GCS we must be careful because Supabase `bucket_name` may
    include a prefix, e.g. "my-bucket/some-prefix/".
    """
    parsed = urlparse(storage_uri or "")
    scheme = (parsed.scheme or "").lower()
    path_no_slash = (parsed.path or "").lstrip("/").rstrip("/")

    # ---- Normalize provider / backend aliases ----
    raw_provider = (conn.get("provider") or conn.get("backend") or scheme or "").lower()
    if raw_provider in ("google_cloud_platform", "gcp", "google_cloud_storage"):
        provider = "gcs"
    elif raw_provider in ("aws_s3", "amazon_s3"):
        provider = "s3"
    elif raw_provider in ("azure_blob_storage", "azure_blob", "microsoft_azure"):
        provider = "azure"
    else:
        provider = raw_provider or scheme

    # ---------------- S3 / AWS ----------------
    if provider in ("aws", "s3"):
        def _clean(s: Optional[str]) -> Optional[str]:
            if s is None:
                return None
            v = s.strip()
            return v or None

        bucket = parsed.netloc or None
        base_prefix = path_no_slash  # may be ""

        conn_bucket_raw = _clean(conn.get("bucket_name") or conn.get("bucket"))
        if not bucket and conn_bucket_raw:
            parts = conn_bucket_raw.strip("/").split("/", 1)
            bucket = parts[0]
            extra_prefix = parts[1].strip("/") if len(parts) > 1 else ""
            if extra_prefix:
                base_prefix = (
                    f"{extra_prefix}/{base_prefix}".strip("/")
                    if base_prefix
                    else extra_prefix
                )

        if not bucket:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"S3 bucket missing in storage_uri={storage_uri} and connection.",
            )

        region = _clean(conn.get("region") or conn.get("aws_region")) or "us-east-1"
        access_key = _clean(conn.get("access_key") or conn.get("aws_access_key_id"))
        secret_key = _clean(conn.get("secret_key") or conn.get("aws_secret_access_key"))
        session_token = _clean(conn.get("session_token") or conn.get("aws_session_token"))
        endpoint_url = _clean(conn.get("endpoint_url") or conn.get("custom_endpoint_url"))

        store = S3BlobStore(
            bucket=bucket,
            prefix=base_prefix,
            region=region,
            access_key=access_key,
            secret_key=secret_key,
            session_token=session_token,
            endpoint_url=endpoint_url,
        )
        return store, (base_prefix or "").rstrip("/")

    # ---------------- GCS ----------------
    if provider in ("gcs", "gs", "google"):
        bucket = parsed.netloc or None
        base_prefix = path_no_slash

        conn_bucket_raw = (conn.get("bucket_name") or conn.get("bucket") or "").strip()

        if not bucket and conn_bucket_raw:
            parts = conn_bucket_raw.strip("/").split("/", 1)
            bucket = parts[0]
            extra_prefix = parts[1].strip("/") if len(parts) > 1 else ""
            if extra_prefix:
                base_prefix = (
                    f"{extra_prefix}/{base_prefix}".strip("/")
                    if base_prefix
                    else extra_prefix
                )

        if not bucket:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"GCS bucket missing in storage_uri={storage_uri} and connection.",
            )
        

        # sa_json = conn.get("service_account_json") or conn.get("gcp_service_account_json")
        sa_json = (
            conn.get("service_account_json")
            or conn.get("gcp_service_account_json")
            or conn.get("secret_key")   # ← your schema stores SA JSON here
        )
        project = conn.get("project") or conn.get("gcp_project")

        store = GCSBlobStore(bucket=bucket, prefix=base_prefix, service_account_json=sa_json, project=project)

        #store = GCSBlobStore(bucket=bucket, prefix=base_prefix)
        return store, (base_prefix or "").rstrip("/")


    # ---------------- Azure ----------------
    if provider in ("azure", "az"):
        def _clean(s: Optional[str]) -> Optional[str]:
            if s is None:
                return None
            v = str(s).strip()
            return v or None

        # Account, most-trusted first. endpoint_url is authoritative because it
        # is the literal host Azure issued for this storage account.
        conn_account = _clean(
            conn.get("account_name") or conn.get("azure_account") or conn.get("access_key")
        )
        _endpoint = _clean(conn.get("endpoint_url"))
        if _endpoint:
            _host = urlparse(_endpoint).hostname or ""
            if _host.endswith(".blob.core.windows.net"):
                conn_account = _host.split(".", 1)[0] or conn_account

        conn_container = _clean(
            conn.get("container") or conn.get("bucket_name") or conn.get("container_name")
        )

        account_from_uri = _clean(parsed.netloc)

        container_from_uri = None
        extra_prefix = ""
        if path_no_slash:
            parts = path_no_slash.split("/", 1)
            container_from_uri = _clean(parts[0])
            if len(parts) == 2:
                extra_prefix = parts[1]

        # Canonical form is az://<account>/<container>/<prefix>, but callers
        # (the UI's register-existing payload) sometimes send az://<container>/
        # with no account segment. Detect that instead of using the container
        # as a hostname, which DNS-fails after ~90s of SDK retries.
        if account_from_uri and not container_from_uri:
            if (conn_container and account_from_uri == conn_container) or (
                conn_account and account_from_uri != conn_account
            ):
                container_from_uri, account_from_uri = account_from_uri, None

        # The SAS on this row is scoped to the row's own account, so a differing
        # account in the URI could never authenticate — the row always wins.
        if conn_account and account_from_uri and account_from_uri != conn_account:
            logger.warning(
                "[_store_from_connection_uri][azure] storage_uri account %r != connection "
                "account %r; using the connection's account.",
                account_from_uri, conn_account,
            )
        account = conn_account or account_from_uri
        container = container_from_uri or conn_container

        # credentials (prefer conn string, then account key, then sas)
        connection_string = conn.get("connection_string") or conn.get("AZURE_STORAGE_CONNECTION_STRING")
        account_key = conn.get("account_key") or conn.get("AZURE_STORAGE_ACCOUNT_KEY")
        sas_token = conn.get("sas_token") or conn.get("sas") or conn.get("secret_key") or conn.get("AZURE_STORAGE_SAS_TOKEN")
        prefix = extra_prefix.strip("/")

        logger.info(
            "[_store_from_connection_uri][azure] storage_uri=%s provider=%s "
            "account_from_uri=%s container_from_uri=%s account_final=%s "
            "container_final=%s prefix=%s",
            storage_uri,
            provider,
            account_from_uri,
            container_from_uri,
            account,
            container,
            prefix,
        )

        # if not (account and container and sas_token):
        #     raise HTTPException(
        #         status.HTTP_500_INTERNAL_SERVER_ERROR,
        #         "Azure connection is missing account/container/SAS info.",
        #     )
        if not (account and container and sas_token):
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Azure connection is missing account/container/SAS info.",
            )

        # Fail in milliseconds on a malformed account rather than after ~90s of
        # azure-sdk DNS retries against a host that cannot exist.
        if not re.fullmatch(r"[a-z0-9]{3,24}", account or ""):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Resolved Azure storage account {account!r} is not a valid account name "
                f"(from storage_uri={storage_uri!r}). Expected az://<account>/<container>.",
            )

        store = AzureBlobStore(
            account=account,
            container=container,
            prefix=prefix,
            sas_token=sas_token,
        )
        return store, prefix

    # ---------------- Unsupported ----------------
    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        f"Unsupported provider '{provider}' for storage_uri={storage_uri}",
    )

async def normalize_storage_uri(storage_uri: str, conn: Dict[str, Any]) -> str:
    """Rewrite a caller-supplied storage_uri into the canonical
    scheme://<account>/<container>/<prefix> form, using the connection row to
    fill in whatever the caller omitted.

    The UI sends az://<container>/ with no account segment. _store_from_connection_uri
    now tolerates that, but the raw string gets persisted as data_source_location and
    is later re-parsed by storage_service._store_and_key_from_uri, which has no
    connection row to repair it with. Normalise once, on the way in.
    """
    raw = (storage_uri or "").rstrip("/")
    provider = (conn.get("provider") or conn.get("backend") or "").lower()
    if provider not in ("azure", "az", "azure_blob_storage", "azure_blob", "microsoft_azure"):
        return raw
    store, prefix = await _store_from_connection_uri(raw, conn)
    base = f"az://{store.account}/{store.container}"
    return f"{base}/{prefix}" if prefix else base