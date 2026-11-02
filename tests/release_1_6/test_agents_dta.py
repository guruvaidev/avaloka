"""1.6 release suite: data-transfer agent (plan cases AG-08, AG-09).

AG-08 stands a local Parquet file in for the bucket object and a file-based
SQLite database in for the destination (the SQL sink only fixes the identifier
quote character, so the SQLAlchemy dialect follows the URL; the same stand-in
tests/test_dta_sink_idempotency.py uses). The bucket read itself needs real
credentials and stays manual. See docs/test-reports/1.6-traceability.csv.
"""
from __future__ import annotations

import pandas as pd
import pytest
import sqlalchemy as sa


def test_ag08_cloud_to_database_creates_the_table_and_loads_every_row(tmp_path):
    """AG-08: a file source loaded into a database destination creates the table
    from the source schema with sensible column types, and the row count matches."""
    import daft

    from app.agents.data_transfer_agent.data_transfer_agent import _create_destination_table
    from app.agents.data_transfer_agent.datasink.sql_sink import PostgresDataSink

    source = pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105],
        "price": [9.5, 3.25, 1.0, 12.75, 0.5],
        "sku": ["a-1", "b-2", "c-3", "d-4", "e-5"],
    })
    parquet = tmp_path / "orders.parquet"
    source.to_parquet(parquet)
    url = f"sqlite:///{tmp_path / 'dest.db'}"

    frame = daft.read_parquet(str(parquet))
    schema = {f.name: str(f.dtype) for f in frame.schema()}
    _create_destination_table(url, "orders_copy", schema, "postgresql")
    frame.write_sink(PostgresDataSink(url, "orders_copy", columns=list(schema)))

    engine = sa.create_engine(url)
    try:
        with engine.begin() as conn:
            rows = conn.exec_driver_sql("SELECT order_id, price, sku FROM orders_copy ORDER BY order_id").fetchall()
            # The DECLARED types (SQLAlchemy reflection would normalise them).
            columns = {r[1]: str(r[2]).upper()
                       for r in conn.exec_driver_sql("PRAGMA table_info(orders_copy)").fetchall()}
    finally:
        engine.dispose()

    assert len(rows) == len(source)
    assert [tuple(r) for r in rows] == list(source.itertuples(index=False, name=None))
    assert columns["order_id"] == "BIGINT"
    assert columns["price"] in {"DOUBLE PRECISION", "DOUBLE"}
    assert columns["sku"] == "TEXT"


def _transfer_message(source_type: str) -> str:
    """Run the pipeline for an unsupported source and return what the user sees."""
    from app.agents.data_transfer_agent.data_transfer_agent import data_transfer_pipeline
    from app.agents.data_transfer_agent.dta_state import cloud_storage_credentials

    creds = cloud_storage_credentials(provider="aws", bucket_name="example-bucket",
                                      file_path=f"data/in.{source_type}")
    try:
        final_state, script = data_transfer_pipeline(
            user_prompt="copy it",
            source_type=source_type,
            destination_type="csv",
            source_credentials=creds,
            destination_file="out.csv",
        )
    except RuntimeError as exc:  # today: the LLM pre-flight fails first
        return str(exc)
    assert script is None, "an unsupported source must never produce a transfer script"
    return final_state.get("error_message") or ""


@pytest.fixture()
def _no_model_keys(monkeypatch):
    """Hermetic: with no usable provider the pipeline can never reach a model."""
    import app.agents.data_transfer_agent.daft_coder as daft_coder
    import app.agents.data_transfer_agent.daft_validator as daft_validator

    for key in ("OPENROUTER_API_KEY", "INFERENCE_OPENROUTER_API_KEY", "GROQ_API_KEY",
                "GROQ_API_KEY_CODING_AGENT", "GROQ_API_KEY_PLANNING_AGENT", "OPENAI_API_KEY",
                "AVALOKA_DTA_CODER_PROVIDER", "AVALOKA_DTA_VALIDATOR_PROVIDER",
                "INFERENCE_PROVIDER_CODING", "INFERENCE_PROVIDER_PLANNING"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(daft_coder, "coder_llm", None, raising=False)
    monkeypatch.setattr(daft_validator, "validator_llm", None, raising=False)


@pytest.mark.usefixtures("_no_model_keys")
@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "AG-09: only a cloud JSON source is refused up front "
    "(data_transfer_agent.py:1580-1592); any other unreadable format reaches the "
    "LLM pipeline and fails late, or as an unrelated configuration error"))
@pytest.mark.parametrize("source_type", ["avro", "xlsx"])
def test_ag09_unsupported_source_format_is_refused_with_a_clear_message(source_type):
    """AG-09: a source format the DTA cannot read (anything but CSV, JSON or
    Parquet) is refused before any work, with a message naming the format and
    the formats it can read."""
    message = _transfer_message(source_type).lower()
    assert source_type in message, message
    assert "csv" in message and "parquet" in message, message
