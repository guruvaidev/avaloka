"""The GKE Ray runner ships a GCP service-account key to the cluster so the job
reads the user's buckets as that SA. The key can arrive as raw JSON, a dict, or —
via GOOGLE_APPLICATION_CREDENTIALS — a FILE PATH. Before the fix the runner assumed
raw JSON, so a path failed json.loads() and the PATH STRING itself was written as
gcp_sa.json → the cluster fell back to its own identity → 'Caller does not have
storage.objects.get access'. `_coerce_sa_json` resolves all three shapes to JSON
content, or None (so a bogus value is never shipped).
"""
import json

from app.infra.ray_job_runner import _coerce_sa_json

_SA = {"type": "service_account", "project_id": "p", "private_key": "k"}


def test_dict_is_serialised():
    assert json.loads(_coerce_sa_json(_SA))["project_id"] == "p"


def test_raw_json_string_is_compacted():
    out = _coerce_sa_json(json.dumps(_SA, indent=2))
    assert json.loads(out)["type"] == "service_account"


def test_file_path_is_read(tmp_path):
    key = tmp_path / "sa.json"
    key.write_text(json.dumps(_SA))
    out = _coerce_sa_json(str(key))
    assert json.loads(out)["project_id"] == "p"   # content, NOT the path string
    assert out != str(key)


def test_quoted_file_path_is_read(tmp_path):
    key = tmp_path / "sa.json"
    key.write_text(json.dumps(_SA))
    out = _coerce_sa_json(f'"{key}"')             # Windows `set VAR="..."` artefact
    assert json.loads(out)["project_id"] == "p"


def test_escaped_newlines_in_private_key_are_handled():
    raw = '{"type": "service_account", "private_key": "-----\\nabc\\n-----"}'
    out = _coerce_sa_json(raw)
    assert "\n" in json.loads(out)["private_key"]


def test_invalid_and_empty_return_none(tmp_path):
    assert _coerce_sa_json("") is None
    assert _coerce_sa_json(None) is None
    assert _coerce_sa_json("not-json{") is None
    assert _coerce_sa_json(str(tmp_path / "missing.json")) is None  # nonexistent path
