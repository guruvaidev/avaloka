"""When the code ran but its result could not be read back, the reply is plain words.

It used to end with the exception's own line (_brief_error), e.g. "No columns
to parse from file". The exception belongs in the logs and on the trace step.
"""

from langchain_core.messages import HumanMessage

from app.agents import execution_agent as ex

MARKER = "No columns to parse from file ZQ_MARKER"


def test_output_preparation_failure_reply_carries_no_exception_text(tmp_path, monkeypatch):
    data = tmp_path / "in.csv"
    data.write_text("region,units\nnorth,1\nsouth,2\n")

    def boom(*args, **kwargs):
        raise RuntimeError(MARKER)

    monkeypatch.setattr(ex, "render_response_segments", boom)
    state = {
        "messages": [HumanMessage(content="Units by region?")],
        "coder_definition": {"code": "import pandas as pd\ndef main(df):\n    return df\n"},
        "data_source_location": str(data),
        "output_location": str(tmp_path / "out" / "output.csv"),
    }
    out = ex.execution_agent_node_local(state)
    reply = out["messages"][-1].content
    assert out["execution_result"]["status"] == "failed"
    assert "ZQ_MARKER" not in reply and "No columns to parse" not in reply, reply
    assert "RuntimeError" not in reply
    assert "couldn't prepare its result" in reply
