from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from app.graph.etl_state import ETLState

from langgraph.graph import END
from app.agents.coder import coder_node
from app.agents.summarizer import summarize_etl_job
from app.agents.planner import plan_etl_job


def should_summarize(state: ETLState):
    if state["ready_to_code"]:
        return "prepare_code"
    elif state["ready_to_summarize"]:
        return "summarize"
    else:
        return "continue_planning"



def build_graph():
    # Build Graph
    graph_builder = StateGraph(ETLState)
    graph_builder.add_node("plan_etl", plan_etl_job)
    graph_builder.add_node("summarize_etl", summarize_etl_job)
    graph_builder.add_node("code_etl", coder_node)
    graph_builder.set_entry_point("plan_etl")
    graph_builder.add_conditional_edges("plan_etl", should_summarize, {
        "summarize": "summarize_etl",
        "prepare_code": "code_etl",
        "continue_planning": END
    })
    graph_builder.add_edge("summarize_etl", "code_etl")
    graph_builder.add_edge("code_etl", END)
    return graph_builder

