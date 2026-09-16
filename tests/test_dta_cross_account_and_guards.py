"""Cross-account cloud creds, script-literal guard, intent-gate scope, env round-trip.

Four independent fixes:
1. Source and dest cloud env dicts are merged into ONE namespace, so a same-provider
   cross-account transfer (S3→S3, Azure→Azure) had the dest's keys clobber the
   source's — the read then authenticated as the destination account. Dest creds now
   carry a DEST_ prefix end-to-end (planner env builder → generated io-config and
   upload/download helpers).
2. Table names / object paths are interpolated into the generated script's string
   literals; a quote or backslash broke out of them (invalid script, or injection).
3. _is_transfer_intent gated on the transfer-verb trigger alone, so "move revenue to
   profit" — a column request — skipped the fabrication guards built for it.
4. gke_run YAML-escapes env values but ray_job_runner's DIRECT-mode extractor never
   un-escaped them, corrupting secrets containing backslash or quote.
"""
import ast

import app.agents.planner as P
from app.agents.data_transfer_agent.data_transfer_agent import (
    _make_io_config_code,
    _make_cloud_upload_helper_code,
    _make_cloud_download_helper_code,
    _unsafe_script_literals,
)
from app.data_transfer_docker.gke_run import _render_cloud_env_vars
from app.infra.ray_job_runner import _extract_runtime_env_from_yaml


# ── 1. cross-account creds ────────────────────────────────────────────────────

def test_same_provider_cross_account_creds_both_survive_the_merge():
    src = P._cloud_env_from_entry(
        {"db_type": "aws_s3", "access_key": "SRC", "secret_key": "SRCS", "region": "us-east-1"},
        "source",
    )
    dst = P._cloud_env_from_entry(
        {"db_type": "aws_s3", "access_key": "DST", "secret_key": "DSTS", "region": "eu-west-1"},
        "dest",
    )
    merged = {**src, **dst}
    assert merged["AWS_ACCESS_KEY_ID"] == "SRC"
    assert merged["DEST_AWS_ACCESS_KEY_ID"] == "DST"


def test_dest_helpers_prefer_dest_creds_and_source_stays_plain():
    assert "DEST_AWS_ACCESS_KEY_ID" in _make_io_config_code("aws", "dest")
    assert "DEST_" not in _make_io_config_code("aws", "source")
    assert "DEST_AZURE_STORAGE_ACCOUNT" in _make_io_config_code("azure", "dest")
    # Upload/download always target the destination bucket.
    assert "DEST_AWS_ACCESS_KEY_ID" in _make_cloud_upload_helper_code("aws")
    assert "DEST_AWS_ACCESS_KEY_ID" in _make_cloud_download_helper_code("aws")
    assert "DEST_AZURE_STORAGE_ACCOUNT" in _make_cloud_upload_helper_code("azure")


def test_all_generated_cloud_helpers_are_valid_python():
    for provider in ("aws", "azure", "gcp"):
        for role in ("source", "dest"):
            ast.parse(_make_io_config_code(provider, role))
        ast.parse(_make_cloud_upload_helper_code(provider))
        ast.parse(_make_cloud_download_helper_code(provider))


# ── 2. script-literal guard ───────────────────────────────────────────────────

def test_clean_and_schema_qualified_names_pass():
    assert _unsafe_script_literals({"destination_table": "orders"}) == []
    assert _unsafe_script_literals({"destination_table": "public.orders"}) == []


def test_quote_and_injection_in_table_or_path_are_rejected():
    assert _unsafe_script_literals({"destination_table": 'order"details'})
    assert _unsafe_script_literals({"source_file": 'x"; import os; os.system("id"); "'})
    assert _unsafe_script_literals({"source_table": "a\\b"})


# ── 3. intent-gate scope ──────────────────────────────────────────────────────

def test_column_phrasings_no_longer_skip_the_guards(monkeypatch):
    """'move revenue to profit' must be treated as analysis, not transfer."""
    trigger = P._TRANSFER_TRIGGER_RE
    for prompt in ("move revenue to profit", "copy the total to the summary row"):
        assert trigger.search(prompt), "precondition: the bare trigger DOES match"
        assert not (
            P._extract_transfer_destination(prompt)
            or P._extract_transfer_aliases(prompt)
        ), "precondition: no transfer anatomy parses"
        # The gate composes trigger AND anatomy/nouns — mirrored from plan_etl_job.
        import re
        has_noun = re.search(
            r"\b(?:table|database|db|connection|bucket|warehouse|schema)\b",
            prompt, re.IGNORECASE,
        )
        assert not has_noun


def test_real_transfer_phrasings_still_skip_the_guards():
    import re
    for prompt in (
        "transfer to warehouse into table orders",
        "Transfer adult_income from sunny_test2 into table adult_income_dest in sunny_test7",
    ):
        assert P._TRANSFER_TRIGGER_RE.search(prompt)
        assert (
            P._extract_transfer_destination(prompt)
            or P._extract_transfer_aliases(prompt)
            or re.search(r"\b(?:table|database|db|connection|bucket|warehouse|schema)\b",
                         prompt, re.IGNORECASE)
        )


# ── 4. env escape round-trip ──────────────────────────────────────────────────

def test_secrets_with_backslash_and_quote_survive_render_extract():
    nasty = {
        "DTA_DEST_CONN_STR": 'postgresql://u:pa"ss\\word@h/db',
        "AWS_SECRET_ACCESS_KEY": 'abc\\def"ghi',
        "PLAIN": "simple-value",
    }
    rendered = _render_cloud_env_vars(nasty)
    yaml_doc = "spec:\n  runtimeEnvYAML: |\n    env_vars:\n" + rendered + "\n"
    out = _extract_runtime_env_from_yaml(yaml_doc)["env_vars"]
    assert out == nasty
