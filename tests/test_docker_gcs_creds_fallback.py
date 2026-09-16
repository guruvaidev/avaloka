"""The local Docker runner injects GCS credentials into the (isolated) container.

A GCS source transfer forced to local Docker (e.g. a VPC-private DB destination)
reads GCS inside the container. The container never inherits the host env, so it
gets credentials ONLY from a key the runner writes into the mounted job dir. This
comes from the connection's stored key, or — behind DTA_ALLOW_AMBIENT_GCP_CREDS=1 —
a fallback to the HOST's ambient GOOGLE_APPLICATION_CREDENTIALS so a dev-env export
works for local Docker too. Without either, GCS reads go anonymous →
'storage.objects.get denied'.

The fallback is OPT-IN because on a deployed host that file is the PLATFORM's key
(deploy/.../docker.env sets GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json),
so an unattended fallback would silently run a tenant's transfer as the platform.
"""
import json

from app.data_transfer_docker.docker_run import (
    _write_container_gcs_sa,
    _GCS_SA_FILENAME,
)

_SA = {"type": "service_account", "project_id": "p", "private_key": "k"}


def test_connection_key_is_written_and_takes_precedence(tmp_path, monkeypatch):
    # Even if the host has ambient creds AND the opt-in is on, the connection's key wins.
    monkeypatch.setenv("DTA_ALLOW_AMBIENT_GCP_CREDS", "1")
    host_key = tmp_path / "host.json"
    host_key.write_text(json.dumps({"type": "service_account", "project_id": "host"}))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(host_key))

    path = _write_container_gcs_sa(tmp_path, "job-1", json.dumps(_SA))

    assert path == f"/mnt/gcs/transfers/job-1/{_GCS_SA_FILENAME}"
    written = json.loads((tmp_path / _GCS_SA_FILENAME).read_text())
    assert written["project_id"] == "p"          # connection key, not host


def test_connection_key_accepts_dict(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    path = _write_container_gcs_sa(tmp_path, "job-d", _SA)
    assert path.endswith(_GCS_SA_FILENAME)
    assert json.loads((tmp_path / _GCS_SA_FILENAME).read_text())["project_id"] == "p"


def test_falls_back_to_host_ambient_credentials_when_opted_in(tmp_path, monkeypatch):
    """No connection key + opt-in → copy the host's GOOGLE_APPLICATION_CREDENTIALS in."""
    monkeypatch.setenv("DTA_ALLOW_AMBIENT_GCP_CREDS", "1")
    host_key = tmp_path / "adc.json"
    host_key.write_text(json.dumps(_SA))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(host_key))

    path = _write_container_gcs_sa(tmp_path, "job-2", None)

    assert path == f"/mnt/gcs/transfers/job-2/{_GCS_SA_FILENAME}"
    assert json.loads((tmp_path / _GCS_SA_FILENAME).read_text())["type"] == "service_account"


def test_host_ambient_credentials_are_ignored_by_default(tmp_path, monkeypatch):
    """The security default: a tenant with no key must NOT borrow the platform's.

    On a deployed host GOOGLE_APPLICATION_CREDENTIALS is /secrets/gcp-key.json — the
    platform's own identity. Falling back to it would run the tenant's transfer as us,
    against any bucket we can reach, triggered by nothing louder than a missing key.
    """
    monkeypatch.delenv("DTA_ALLOW_AMBIENT_GCP_CREDS", raising=False)
    host_key = tmp_path / "platform-key.json"
    host_key.write_text(json.dumps(_SA))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(host_key))

    assert _write_container_gcs_sa(tmp_path, "job-esc", None) is None
    assert not (tmp_path / _GCS_SA_FILENAME).exists()


def test_host_env_with_surrounding_quotes_still_works(tmp_path, monkeypatch):
    """Windows `set VAR="C:\\path"` keeps the quotes; the fallback must strip them."""
    monkeypatch.setenv("DTA_ALLOW_AMBIENT_GCP_CREDS", "1")
    host_key = tmp_path / "adc.json"
    host_key.write_text(json.dumps(_SA))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", f'"{host_key}"')  # quoted

    path = _write_container_gcs_sa(tmp_path, "job-q", None)

    assert path == f"/mnt/gcs/transfers/job-q/{_GCS_SA_FILENAME}"
    assert json.loads((tmp_path / _GCS_SA_FILENAME).read_text())["project_id"] == "p"


def test_no_key_and_no_host_creds_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    assert _write_container_gcs_sa(tmp_path, "job-3", None) is None
    assert not (tmp_path / _GCS_SA_FILENAME).exists()


def test_host_env_pointing_at_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("DTA_ALLOW_AMBIENT_GCP_CREDS", "1")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "nope.json"))
    assert _write_container_gcs_sa(tmp_path, "job-4", None) is None


def test_invalid_json_key_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    assert _write_container_gcs_sa(tmp_path, "job-5", "not-json{") is None
    assert not (tmp_path / _GCS_SA_FILENAME).exists()
