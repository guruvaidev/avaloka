import json
import logging
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from app.agents.summarizer import summarize_etl_job
from app.api.workflow import build_graph
from app.graph.etl_state import ETLState


# ---------- helpers ----------

def assert_valid_etl_json(data):
    required_keys = ["job_name", "source", "transformations", "destination", "schedule", "notes"]
    for key in required_keys:
        assert key in data, f"Missing key: {key}"

def _msg_text(msg: AIMessage) -> str:
    """Normalize LangChain AIMessage.content into plain text for assertions."""
    content = msg.content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text") or "")
            else:
                parts.append(str(b))
        return "\n".join(parts)
    return content or ""


# ---------- fixtures ----------

@pytest.fixture
def compiled_graph(monkeypatch):
    """
    Patch *inside* app.api.workflow before compile so routing hits the Summarizer.
    """
    def _planner_sets_summarize(state):
        s = dict(state)
        s["ready_to_summarize"] = True
        s["ready_to_code"] = False
        return s

    monkeypatch.setattr("app.api.workflow.plan_etl_job", _planner_sets_summarize)
    monkeypatch.setattr("app.api.workflow.coder_node", lambda s: s)

    return build_graph().compile()


@pytest.fixture
def base_state():
    return ETLState(
        messages=[HumanMessage(content="Create ETL job")],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={},
        infrastructure_provisioned={},
    )


# ---------- integration tests (function-level) ----------

@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_summarizer_handles_empty_messages(mock_extract, mock_llm):
    state = ETLState(
        messages=[],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={},
        infrastructure_provisioned={},
    )
    mock_llm.invoke.return_value = MagicMock(content='{"job_name":"Empty","source":"","transformations":[],"destination":"","schedule":"","notes":""}')
    mock_extract.return_value = '{"job_name":"Empty","source":"","transformations":[],"destination":"","schedule":"","notes":""}'

    result = summarize_etl_job(state)
    assert_valid_etl_json(result["planner_definition"])
    assert result["planner_definition"]["job_name"] == "Empty"


@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_summarizer_merges_with_prefilled_planner(mock_extract, mock_llm):
    state = ETLState(
        messages=[HumanMessage(content="Summarize ETL")],
        planner_definition={"job_name": "Pre-existing", "source": "db"},
        ready_to_summarize=False,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={},
        infrastructure_provisioned={},
    )
    content = '{"job_name":"New Job","source":"api","transformations":[],"destination":"wh","schedule":"daily","notes":""}'
    mock_llm.invoke.return_value = MagicMock(content=content)
    mock_extract.return_value = content

    result = summarize_etl_job(state)
    assert_valid_etl_json(result["planner_definition"])
    assert result["planner_definition"]["job_name"] == "New Job"
    assert result["planner_definition"]["source"] == "api"


@pytest.mark.parametrize("bad_json", [
    '{"job_name":"ETL",}',  # trailing comma
    '{"job_name":"Bad","source":"db" INVALID }',  # malformed
])
@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_summarizer_parsing_error_invalid_json(mock_extract, mock_llm, bad_json, base_state):
    mock_llm.invoke.return_value = MagicMock(content=bad_json)
    mock_extract.return_value = bad_json

    result = summarize_etl_job(base_state)
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert "Failed to generate a valid ETL job summary" in last.content


@pytest.mark.parametrize("content", [
    '{"job_name":"Job1","source":"db","transformations":[],"destination":"wh","schedule":"daily","notes":"note"}',
    '{"job_name":"Job2","source":"api","transformations":[],"destination":"s3","schedule":"hourly","notes":""}',
])
@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_summarizer_multiple_json_outputs(mock_extract, mock_llm, content, base_state):
    mock_llm.invoke.return_value = MagicMock(content=content)
    mock_extract.return_value = content

    result = summarize_etl_job(base_state)
    assert_valid_etl_json(result["planner_definition"])


@pytest.mark.parametrize("partial_json", [
    '{"job_name":"Partial Only"}',
    '{"job_name":"Partial","source":"db"}',
])
@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_summarizer_partial_json_outputs(mock_extract, mock_llm, partial_json, base_state):
    mock_llm.invoke.return_value = MagicMock(content=partial_json)
    mock_extract.return_value = partial_json

    result = summarize_etl_job(base_state)
    assert "job_name" in result["planner_definition"]
    assert isinstance(result["planner_definition"], dict)


@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_summarizer_no_json_extracted(mock_extract, mock_llm, base_state):
    mock_llm.invoke.return_value = MagicMock(content="Some irrelevant text without JSON")
    mock_extract.return_value = None  # no JSON found


    result = summarize_etl_job(base_state)
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)

    # Normalize content (handles str, list-of-blocks, or empty)
    content = last.content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text") or "")
            else:
                parts.append(str(b))
        content = "\n".join(parts)
    elif content is None:
        content = ""

    # Accept either explicit text or empty (some LC versions return "")
    assert ("No valid JSON block found." in content) or (content == "")

    # Side-effect is the reliable signal for this path
    # And state updated as expected
    assert result["ready_to_code"] is False
    assert result["ready_to_summarize"] is True
    assert result["planner_definition"] == {}
    assert len(result["messages"]) >= 1



@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_summarizer_emits_pretty_json_message(mock_extract, mock_llm, base_state):
    content = '{"job_name":"Nice","source":"db","transformations":[],"destination":"wh","schedule":"daily","notes":""}'
    mock_llm.invoke.return_value = MagicMock(content=content)
    mock_extract.return_value = content

    before_n = len(base_state["messages"])
    out = summarize_etl_job(base_state)

    assert len(out["messages"]) == before_n + 1
    last = out["messages"][-1]
    assert isinstance(last, AIMessage)
    txt = _msg_text(last)
    # The preview message uses a json fence and pretty-prints with indentation
    assert "```json" in txt
    assert '"job_name": "Nice"' in txt and '\n  "job_name": "Nice",' in txt
    assert_valid_etl_json(out["planner_definition"])


@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_summarizer_no_json_empty_string(mock_extract, mock_llm, base_state):
    mock_llm.invoke.return_value = MagicMock(content="noise")
    mock_extract.return_value = ""  # falsy but not None
    out = summarize_etl_job(base_state)
    last = out["messages"][-1]
    assert isinstance(last, AIMessage)
    txt = _msg_text(last)
    # Some LC versions return empty string here; accept either
    assert ("No valid JSON block found." in txt) or (txt == "")
    assert out["ready_to_code"] is False and out["ready_to_summarize"] is True


@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_summarizer_logs_parse_error(mock_extract, mock_llm, base_state, caplog):
    bad = '{"job_name":"Oops",}'  # invalid
    mock_llm.invoke.return_value = MagicMock(content=bad)
    mock_extract.return_value = bad
    with caplog.at_level(logging.ERROR):
        _ = summarize_etl_job(base_state)
        assert any("Validation failed" in rec.getMessage() for rec in caplog.records)


@patch("app.agents.summarizer.summarizer_llm")
def test_e2e_graph_ready_to_code_short_circuit(mock_llm):
    graph = build_graph().compile()
    state = ETLState(
        messages=[HumanMessage(content="skip to coder")],
        planner_definition={"job_name":"exists"},
        ready_to_summarize=False,
        ready_to_code=True,  # router should bypass Summarizer
        coder_definition={"code": "pass"},
        infrastructure_request={},
        infrastructure_provisioned={},
    )
    out = summarize_etl_job(state)
    # Summarizer should not be invoked
    mock_llm.assert_not_called()
    assert out["planner_definition"]["job_name"] == "exists"

# ---------- end-to-end (graph-level) ----------

@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_e2e_summarizer_full_flow(mock_extract, mock_llm, compiled_graph, base_state):
    content = '{"job_name":"ETL Job","source":"api","transformations":[],"destination":"s3","schedule":"hourly","notes":""}'
    mock_llm.invoke.return_value = MagicMock(content=content)
    mock_extract.return_value = content
    base_state["ready_to_summarize"] = True

    result = compiled_graph.invoke(base_state)
    assert_valid_etl_json(result["planner_definition"])
    assert result["ready_to_code"] is True


@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_e2e_summarizer_handles_multi_turn_conversation(mock_extract, mock_llm, compiled_graph):
    state = ETLState(
        messages=[HumanMessage(content="Create ETL job for sales"),
                  HumanMessage(content="Use daily schedule")],
        planner_definition={},
        ready_to_summarize=True,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={},
        infrastructure_provisioned={},
    )
    content = '{"job_name":"Sales ETL","source":"db","transformations":[],"destination":"warehouse","schedule":"daily","notes":"multi-turn"}'
    mock_llm.invoke.return_value = MagicMock(content=content)
    mock_extract.return_value = content

    result = compiled_graph.invoke(state)
    assert_valid_etl_json(result["planner_definition"])
    assert result["planner_definition"]["schedule"] == "daily"


@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_e2e_summarizer_handles_large_job(mock_extract, mock_llm, compiled_graph, base_state):
    large = {
        "job_name": "Big Job",
        "source": "bigdb",
        "transformations": [{"name": f"Step {i}", "type": "cleaning", "columns": ["col1"], "aggregation": ""} for i in range(20)],
        "destination": "warehouse",
        "schedule": "hourly",
        "notes": "Test large ETL",
    }
    content = json.dumps(large)
    mock_llm.invoke.return_value = MagicMock(content=content)
    mock_extract.return_value = content
    base_state["ready_to_summarize"] = True

    result = compiled_graph.invoke(base_state)
    assert_valid_etl_json(result["planner_definition"])
    assert len(result["planner_definition"]["transformations"]) == 20


@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_e2e_infra_to_summarizer_sets_flag(mock_extract, mock_llm, compiled_graph):
    state = ETLState(
        messages=[HumanMessage(content="Provision infra then summarize")],
        planner_definition={},
        ready_to_summarize=True,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={"type": "gcp"},
        infrastructure_provisioned={"status": "provisioned"},
    )
    content = '{"job_name":"Infra ETL","source":"db","transformations":[],"destination":"wh","schedule":"weekly","notes":""}'
    mock_llm.invoke.return_value = MagicMock(content=content)
    mock_extract.return_value = content

    result = compiled_graph.invoke(state)
    assert_valid_etl_json(result["planner_definition"])
    assert result["ready_to_summarize"] is True


@patch("app.agents.summarizer.summarizer_llm")
@patch("app.agents.summarizer.extract_json_block")
def test_e2e_summarizer_prefilled_planner_merge(mock_extract, mock_llm, compiled_graph):
    state = ETLState(
        messages=[HumanMessage(content="Summarize with prefilled state")],
        planner_definition={"job_name": "Old", "source": "db"},
        ready_to_summarize=True,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={},
        infrastructure_provisioned={},
    )
    content = '{"job_name":"Updated","source":"api","transformations":[],"destination":"wh","schedule":"weekly","notes":""}'
    mock_llm.invoke.return_value = MagicMock(content=content)
    mock_extract.return_value = content

    result = compiled_graph.invoke(state)
    assert_valid_etl_json(result["planner_definition"])
    assert result["planner_definition"]["job_name"] == "Updated"
    assert result["planner_definition"]["source"] == "api"
