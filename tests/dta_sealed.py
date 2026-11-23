"""A stand-in for the sealed credential stores, for planner transfer tests. Not a test module.

1.6 port of 2.0's tests/dta_sealed.py (a8e7c1fa). In production a registry
entry is a handle (``conn_id``) plus display fields: a transfer opens a
database handle through the onboarding credentials route
(``_open_db_credentials``) and a bucket's through its cloud_datasets row
(``_cloud_secrets_for``), and a chat registration hands its secret to one of
those stores (``_seal_chat_database`` / ``_seal_chat_cloud``).

Many transfer tests predate that rule and seed the registry with a password or
bucket keys, as if the user had typed them, or register without a signed-in
user. ``install`` keeps those tests about what they test (wizard, routing,
copy, aliases) by playing the sealed store: the first time the planner reads
the registry, every entry that carries a secret (or has no handle) has its
credentials moved into an in-memory store and is left as a handle; the
resolvers answer from that store. The registry the planner works with is still
the scrubbed one -- the production rule runs unchanged.

Adapted from 2.0: 1.6's _cloud_secrets_for takes (entry, alias); 1.6 has no
_handle_key (the handle is ``conn_id`` or the alias).

Tests about the leak itself (test_dta_chat_connection_sealing.py) do not use
this; they go through the real sealed-store calls.
"""
from __future__ import annotations

from typing import Any, Dict

import app.agents.planner as P
from app.graph.etl_state import scrub_registry_entry


_CLOUD_SECRET_FIELDS = ("access_key", "secret_key", "gcp_service_account_json",
                        "az_storage_account", "az_access_key", "az_sas_token",
                        "azure_connection_string")


class SealedStore:
    def __init__(self) -> None:
        self.db: Dict[str, Dict[str, Any]] = {}
        self.cloud: Dict[str, Dict[str, Any]] = {}

    def _seal(self, alias: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        conn_id = entry.get("conn_id") or alias
        if P._is_cloud_entry(entry):
            self.cloud[conn_id] = {k: entry[k] for k in _CLOUD_SECRET_FIELDS if entry.get(k) is not None}
        else:
            self.db[conn_id] = {k: v for k, v in entry.items()
                                if k not in ("db_type", "_source", "conn_id") and v is not None}
        return scrub_registry_entry({**entry, "conn_id": conn_id})

    def _needs_sealing(self, entry: Any) -> bool:
        return isinstance(entry, dict) and (
            not entry.get("conn_id") or scrub_registry_entry(entry) is not entry
        )

    def install(self, monkeypatch) -> "SealedStore":
        real_get = P._get_db_registry
        real_open_db = P._open_db_credentials
        real_cloud = P._cloud_secrets_for

        def get_db_registry(state):
            reg = state.get("dta_database_registry")
            if isinstance(reg, dict):
                for alias, entry in list(reg.items()):
                    if self._needs_sealing(entry):
                        reg[alias] = self._seal(alias, entry)
            return real_get(state)

        def open_db(state, alias, entry):
            if self._needs_sealing(entry):  # injected mid-handler by a mocked resolver
                entry = self._seal(alias, entry)
            stored = self.db.get(str((entry or {}).get("conn_id") or alias))
            if stored is None:
                return real_open_db(state, alias, entry)
            creds = dict(stored)
            creds.setdefault("sslmode", (entry or {}).get("sslmode"))
            return creds

        def cloud_secrets(entry, alias=None):
            if self._needs_sealing(entry):
                entry = self._seal(entry.get("name") or entry.get("bucket_name") or alias or "bucket", entry)
            stored = self.cloud.get((entry or {}).get("conn_id"))
            if stored is None:
                return real_cloud(entry, alias)
            return {**entry, **stored}

        monkeypatch.setattr(P, "_get_db_registry", get_db_registry)
        monkeypatch.setattr(P, "_open_db_credentials", open_db)
        monkeypatch.setattr(P, "_cloud_secrets_for", cloud_secrets)

        def seal_chat_database(state, params, secrets):
            # What the onboarding API keeps for a chat-typed database.
            conn_id = f"chat_{params.alias.lower()}"
            self.db[conn_id] = {k: v for k, v in {
                "host": params.host, "port": params.port, "database": params.database,
                "user": params.user, "password": secrets.get("password"),
                "table": params.table, "schema_name": params.schema_name, "sslmode": params.sslmode,
            }.items() if v is not None}
            return {"conn_id": conn_id}

        def seal_chat_cloud(state, params, secrets):
            # What cloud_datasets keeps for a chat-typed bucket (nothing, without keys).
            keys = {**secrets, **({"az_storage_account": params.az_storage_account}
                                  if params.az_storage_account else {})}
            if not keys:
                return {}
            conn_id = f"chat_{params.alias.lower()}"
            self.cloud[conn_id] = keys
            return {"conn_id": conn_id}

        monkeypatch.setattr(P, "_seal_chat_database", seal_chat_database)
        monkeypatch.setattr(P, "_seal_chat_cloud", seal_chat_cloud)
        return self


def install(monkeypatch) -> SealedStore:
    return SealedStore().install(monkeypatch)
