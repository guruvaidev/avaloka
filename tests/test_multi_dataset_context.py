import pytest


@pytest.fixture
def server():
    # Imported lazily so this module still collects when an earlier test in the
    # same session has polluted sys.modules for app.agents.visualization_agent.
    import app.api.server as server

    return server


def _state():
    return [
        {
            "dataset_id": "ds-customers",
            "alias": "customers",
            "filename": "customers.csv",
            "columns": ["customer_id", "name"],
            "preview": [{"customer_id": "C1", "name": "Asha"}],
            "data_source_location": "/tmp/work/customers.csv",
            "full_data_location": "/tmp/work/customers_full.csv",
            "sample_data_location": "/tmp/work/customers_sample.csv",
        },
        {
            "dataset_id": "ds-orders",
            "alias": "orders",
            "filename": "orders.csv",
            "columns": ["order_id", "customer_id", "amount"],
            "preview": [{"order_id": "O1", "customer_id": "C1", "amount": "10"}],
            "full_data_location": "/tmp/work/orders_full.csv",
            "sample_data_location": "/tmp/work/orders_sample.csv",
        },
    ]


def test_context_includes_every_dataset_path(server):
    parts = server._build_multi_dataset_context(_state())
    joined = "\n".join(parts)

    # The regression: the prompt promised "the CSV paths below" but appended none.
    assert "csv_path: /tmp/work/customers.csv" in joined
    # orders has no data_source_location -> falls back to full_data_location
    assert "csv_path: /tmp/work/orders_full.csv" in joined

    for alias in ("customers", "orders"):
        assert f"[Dataset: {alias}]" in joined


def test_context_includes_columns_and_preview(server):
    joined = "\n".join(server._build_multi_dataset_context(_state()))
    assert "columns: customer_id, name" in joined
    assert "columns: order_id, customer_id, amount" in joined
    assert '"name": "Asha"' in joined


def test_path_fallback_to_sample_when_only_sample_present(server):
    state = [
        {
            "dataset_id": "d1",
            "alias": "only_sample",
            "sample_data_location": "/tmp/only_sample.csv",
        }
    ]
    joined = "\n".join(server._build_multi_dataset_context(state))
    assert "csv_path: /tmp/only_sample.csv" in joined


def test_join_suggestions_appended_when_present(server):
    sugs = [
        {
            "left_dataset": "customers",
            "left_key": "customer_id",
            "right_dataset": "orders",
            "right_key": "customer_id",
            "confidence": 0.9,
        }
    ]
    joined = "\n".join(server._build_multi_dataset_context(_state(), sugs))
    assert "[Suggested join keys]" in joined
    assert "customers.customer_id ↔ orders.customer_id" in joined
    assert "confidence=0.9" in joined


def test_no_join_suggestions_section_when_empty(server):
    joined = "\n".join(server._build_multi_dataset_context(_state(), []))
    assert "[Suggested join keys]" not in joined


def test_intro_line_present(server):
    parts = server._build_multi_dataset_context(_state())
    assert "Multiple datasets are available" in parts[0]


def test_format_join_suggestions_handles_missing_fields(server):
    out = server._format_join_suggestions([{"key": "id"}])
    assert "left.id ↔ right.id" in out
    assert server._format_join_suggestions([]) == ""
    assert server._format_join_suggestions(None) == ""
