"""
Cloud-path (az://) routing.

`az://` was missing from the cloud-path prefix lists in sampling_agent_daft, so
Azure URIs were treated as local and mangled by os.path.abspath (az://c/x ->
/cwd/az:/c/x) even though the io-config and size-estimate az:// handlers exist.

These tests check routing only: a cloud URI must take the cloud branch and must
never be passed to os.path.abspath. daft's reader is stubbed so no credentials
or network are required.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.agents.sampling_agent_daft as sad

CLOUD_URIS = [
    "az://mycontainer/data.csv",
    "gs://mybucket/data.csv",
    "s3://mybucket/data.csv",
    "https://acct.blob.core.windows.net/c/data.csv",
]


def _spy_abspath(monkeypatch):
    seen = []
    real = os.path.abspath
    monkeypatch.setattr(os.path, "abspath", lambda p: (seen.append(p), real(p))[1])
    return seen


@pytest.mark.parametrize("uri", CLOUD_URIS)
def test_load_file_routes_cloud_uri_to_cloud_branch(uri, monkeypatch):
    monkeypatch.setattr(sad, "_build_cloud_io_config", lambda *a, **k: None)
    monkeypatch.setattr(sad.daft, "read_csv", lambda *a, **k: "DF")
    seen = _spy_abspath(monkeypatch)

    df, _ = sad._load_file_with_daft(uri, "csv")

    assert df == "DF"                                  # cloud reader was used
    assert not any(uri in s for s in seen), f"{uri} was abspath-mangled (treated as local)"


def test_estimate_dataset_size_treats_azure_as_cloud(monkeypatch):
    uri = "az://mycontainer/data.csv"
    seen = _spy_abspath(monkeypatch)

    # Unreachable blob -> cloud default, but the URI must not be abspath-mangled.
    rows, _ = sad._estimate_dataset_size(uri, "csv")

    assert rows > 0
    assert not any(uri in s for s in seen), "az:// was abspath-mangled (treated as local)"


def test_local_path_still_uses_abspath(tmp_path, monkeypatch):
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,2\n")
    seen = _spy_abspath(monkeypatch)

    sad._estimate_dataset_size(str(csv), "csv")

    assert any(str(csv) in s for s in seen)            # local paths still go local
