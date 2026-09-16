
from app.api.workflow import build_graph

# LangGraph API handles persistence automatically, so we don't need a checkpointer
# Pass None to build_graph() so subgraphs also don't use a checkpointer
graph = build_graph(checkpointer=None).compile()