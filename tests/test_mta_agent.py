"""
Unit tests for the Model Training Agent graph entry point.

This file used to test a `ModelTrainingAgent` class with `validate_training_task`,
`create_training_plan` and `prepare_training_environment`, plus a node that built
dict training tasks tagged `created_from="state_analysis"`. None of that survives:
`app.agents.model_training_agent` now re-exports the mta_v2 agent, whose entry
point is `execute(state)`, and the task/plan builders it asserted live only in
`app/agents/mta/`, which no production module imports. Those assertions were
deleted rather than rewritten, because the behaviour they described is gone.

What survives is the wiring this file's node tests were actually about: the graph
calls `model_training_agent_node(state)`, and that must drive the agent and hand
back what the agent produced. `tests/test_mta_unit.py` covers it too; when this
was written that file was named `test_mta_integration.py`, which
`tests/conftest.py` auto-marks `integration`, so nothing running in the gate
asserted the wiring.
"""

from unittest.mock import patch

from app.agents.model_training_agent import (
    model_training_agent,
    model_training_agent_node,
)


class TestModelTrainingAgentNode:
    """The node must delegate to the agent, not stand in for it."""

    def test_node_drives_the_agent_with_the_state_it_was_given(self):
        """The node hands the agent the graph's own state object, not a copy.

        `assert_called_once_with(state)` is not enough here: it compares by
        equality, so a node that passed `dict(state)` would satisfy it while the
        agent's in-place updates went to a throwaway. Assert on the identity of
        the argument the mock actually received.
        """
        state = {"messages": [], "ready_to_train": True, "training_task": None}

        with patch("app.agents.model_training_agent.ModelTrainingAgent") as agent_cls:
            agent_cls.return_value.execute.return_value = {"trained": True}
            model_training_agent_node(state)

        execute = agent_cls.return_value.execute
        assert execute.call_count == 1
        (passed_state,), _kwargs = execute.call_args
        assert passed_state is state

    def test_node_returns_the_agent_result_not_the_input_state(self):
        """A node that returns `state` would advance the graph having trained nothing.

        This is the silent form of the bug: the graph keeps running and the state
        looks untouched rather than wrong, so only the identity of the return
        value distinguishes a working node from a no-op.
        """
        state = {"messages": [], "ready_to_train": True, "training_task": None}
        produced = {"messages": [], "training_completed": True}

        with patch("app.agents.model_training_agent.ModelTrainingAgent") as agent_cls:
            agent_cls.return_value.execute.return_value = produced
            result = model_training_agent_node(state)

        assert result is produced
        assert result is not state

    def test_agent_function_and_node_share_one_path(self):
        """`model_training_agent_node` is a thin alias; both must reach execute once."""
        state = {"messages": [], "ready_to_train": False, "training_task": None}

        with patch("app.agents.model_training_agent.ModelTrainingAgent") as agent_cls:
            agent_cls.return_value.execute.return_value = {"ok": True}
            direct = model_training_agent(state)
            via_node = model_training_agent_node(state)

        assert direct == via_node == {"ok": True}
        assert agent_cls.return_value.execute.call_count == 2
