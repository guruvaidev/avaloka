import logging
import sys
import types

import pytest

from app.api import cloud_connections as cc


GCP_SA_KEY = (
    '{"type":"service_account","private_key":"-----BEGIN PRIVATE KEY-----\\n'
    'MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ==\\n'
    '-----END PRIVATE KEY-----\\n"}'
)
AZURE_CONN_STR = (
    "DefaultEndpointsProtocol=https;AccountName=acct;"
    "AccountKey=abc123def456ghi789base64secretmaterial==;EndpointSuffix=core.windows.net"
)


# -------------------------
# redact_connection unit
# -------------------------

def test_masks_fields_the_old_list_missed():
    row = {
        "service_account_json": GCP_SA_KEY,
        "connection_string": AZURE_CONN_STR,
        "account_key": "Zm9vYmFyYmF6c2VjcmV0a2V5bWF0ZXJpYWw=",
        "sas_token": "?sv=2024-01-01&sig=SUPERSECRETSIGNATURE&se=2026",
    }
    safe = cc.redact_connection(row)
    for k, original in row.items():
        assert safe[k] != original, f"{k} was not masked"
        assert original not in str(safe[k])


def test_large_blobs_fully_masked():
    safe = cc.redact_connection(
        {"service_account_json": GCP_SA_KEY, "connection_string": AZURE_CONN_STR}
    )
    assert safe["service_account_json"] == "***masked***"
    assert safe["connection_string"] == "***masked***"


def test_short_key_partially_masked_for_debugging():
    safe = cc.redact_connection({"access_key": "AKIA1234567890XYZ"})
    masked = safe["access_key"]
    assert masked != "AKIA1234567890XYZ"
    assert masked.startswith("AKIA")
    assert masked.endswith("0XYZ")
    assert "..." in masked


def test_non_secret_fields_untouched():
    row = {
        "provider": "gcs",
        "bucket_name": "my-bucket",
        "region": "us-east-1",
        "id": "conn-1",
        "user_id": "user-123",
    }
    assert cc.redact_connection(row) == row


def test_heuristic_masks_future_secret_columns():
    safe = cc.redact_connection(
        {"client_secret": "shhh-very-secret", "refresh_token": "rt-abcdef123456"}
    )
    assert safe["client_secret"] != "shhh-very-secret"
    assert safe["refresh_token"] != "rt-abcdef123456"


def test_redact_passthrough_non_dict():
    assert cc.redact_connection(None) is None
    assert cc.redact_connection("x") == "x"


def test_every_decrypted_credential_field_is_masked():
    # The finding's root cause was the mask list diverging from the decrypt list.
    for field in (
        "access_key",
        "secret_key",
        "session_token",
        "service_account_json",
        "connection_string",
        "account_key",
        "sas",
        "sas_token",
    ):
        assert cc._is_sensitive_field(field), field
        safe = cc.redact_connection({field: "some-real-credential-value-1234"})
        assert safe[field] != "some-real-credential-value-1234"


# -------------------------
# get_cloud_connection log leak regression
# -------------------------

def _install_fake_supabase(monkeypatch, row):
    class _Res:
        data = row

    class _Query:
        def select(self, *a, **k):
            return self

        def eq(self, *a, **k):
            return self

        def single(self):
            return self

        def execute(self):
            return _Res()

    class _Client:
        def table(self, *a, **k):
            return _Query()

    fake_mod = types.ModuleType("supabase")
    fake_mod.create_client = lambda url, key: _Client()
    monkeypatch.setitem(sys.modules, "supabase", fake_mod)


@pytest.mark.asyncio
async def test_get_cloud_connection_does_not_log_plaintext_secrets(monkeypatch, caplog):
    row = {
        "id": "conn-1",
        "provider": "gcs",
        "bucket_name": "my-bucket",
        "service_account_json": GCP_SA_KEY,
        "connection_string": AZURE_CONN_STR,
        "account_key": "Zm9vYmFyc2VjcmV0a2V5",
    }
    _install_fake_supabase(monkeypatch, dict(row))
    monkeypatch.setattr(cc, "SUPABASE_URL", "https://example.supabase.co", raising=False)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "role-key")

    with caplog.at_level(logging.INFO, logger="avaloka"):
        conn = await cc.get_cloud_connection("conn-1")

    # Caller still gets the real decrypted values.
    assert conn["service_account_json"] == GCP_SA_KEY
    assert conn["connection_string"] == AZURE_CONN_STR

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "BEGIN PRIVATE KEY" not in logged
    assert "MIIEvQIBADANBgkqhkiG9w0" not in logged
    assert "AccountKey=abc123def456" not in logged
    assert "Zm9vYmFyc2VjcmV0a2V5" not in logged
    assert "***masked***" in logged
