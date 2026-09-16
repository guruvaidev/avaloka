import operator
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class _S(TypedDict):
    vals: Annotated[list, operator.add]


def _build(checkpointer):
    def node(state):
        return {"vals": [1]}

    g = StateGraph(_S)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile(checkpointer=checkpointer)


def test_checkpointer_makes_thread_id_persist_state():
    # With a checkpointer, re-invoking the same thread_id resumes prior state.
    compiled = _build(MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}

    r1 = compiled.invoke({"vals": []}, config=cfg)
    r2 = compiled.invoke({"vals": []}, config=cfg)

    assert r1["vals"] == [1]
    assert r2["vals"] == [1, 1], "same thread_id should accumulate across turns"


def test_different_thread_ids_are_isolated():
    compiled = _build(MemorySaver())
    compiled.invoke({"vals": []}, config={"configurable": {"thread_id": "a"}})
    fresh = compiled.invoke({"vals": []}, config={"configurable": {"thread_id": "b"}})
    assert fresh["vals"] == [1], "a new thread_id must start from empty state"


def test_without_checkpointer_thread_id_has_no_effect():
    # This is the MAJ-078 bug shape: no checkpointer -> thread_id is meaningless,
    # every invoke starts fresh regardless of thread_id.
    compiled = _build(None)
    cfg = {"configurable": {"thread_id": "t1"}}
    r1 = compiled.invoke({"vals": []}, config=cfg)
    r2 = compiled.invoke({"vals": []}, config=cfg)
    assert r1["vals"] == [1]
    assert r2["vals"] == [1], "no checkpointer -> no cross-turn persistence"
