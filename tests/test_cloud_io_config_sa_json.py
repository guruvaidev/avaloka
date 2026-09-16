"""
Service-account JSON normalization for cloud reads.

A service-account credential whose private_key contains RAW newlines is not
strict-valid JSON, so Daft's credential parser rejects it and cloud sampling
degrades. _parse_service_account_json must still parse it, and the GCS branch of
_build_cloud_io_config re-serializes via json.dumps so Daft receives strict JSON.

(The prefix-preservation half lives inline in server.py's register-existing
endpoint — gcs_path joins bucket/prefix/object_name — and is covered there.)
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.agents.sampling_agent_daft as sad

# private_key with RAW newlines -> NOT strict-valid JSON (the failing case).
_PK = "-----BEGIN PRIVATE KEY-----\nABCDEF\nGHIJKL\n-----END PRIVATE KEY-----\n"
SA_RAW_NEWLINES = (
    '{"type":"service_account","project_id":"my-proj","private_key":"'
    + _PK
    + '","client_email":"svc@my-proj.iam.gserviceaccount.com"}'
)


def test_input_is_not_strict_valid_json():
    # Sanity: this is exactly what Daft's strict parser used to choke on.
    with pytest.raises(json.JSONDecodeError):
        json.loads(SA_RAW_NEWLINES)


def test_parse_service_account_json_parses_and_normalizes_to_strict_json():
    parsed = sad._parse_service_account_json(SA_RAW_NEWLINES)

    assert parsed is not None
    assert parsed["project_id"] == "my-proj"
    assert "\n" in parsed["private_key"]                 # real newlines preserved

    # The fix re-dumps to strict JSON; this must parse cleanly (what Daft gets).
    json.loads(json.dumps(parsed))                       # strict by default


def test_parse_service_account_json_accepts_dict_and_rejects_junk():
    d = {"project_id": "p", "private_key": "k"}
    assert sad._parse_service_account_json(d) is d
    assert sad._parse_service_account_json("") is None
    assert sad._parse_service_account_json("not json") is None


def test_build_cloud_io_config_gcs_builds_from_raw_newline_sa():
    cfg = sad._build_cloud_io_config("gs://bucket/prefix/data.csv", {"secret_key": SA_RAW_NEWLINES})
    assert cfg is not None                                # GCS IOConfig built without error

    # No credentials -> no config (ambient auth), not a crash.
    assert sad._build_cloud_io_config("gs://bucket/x.csv", None) is None
