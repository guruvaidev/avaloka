
import unittest
import os

import pytest
from langgraph.graph import StateGraph
from app.api.workflow import build_graph
from app.graph.etl_state import ETLState
from langchain_core.messages import HumanMessage

# This drives the real compiled graph through the real planner and summarizer:
# the job name it asserts on is written by the planning LLM. With no key the
# planner returns {} and the test failed on `'job_name' not found in {}`, which
# reads as a broken graph rather than as an absent model.
#
# `integration` is pytest.ini's marker for "exercises real agents/LLMs", so the
# hermetic CI stage excludes it by marker. The skipif is what keeps a direct
# `pytest tests/test_workflow.py` honest: set the planning key and it runs.
_PLANNER_KEY = os.environ.get("GROQ_API_KEY_PLANNING_AGENT") or os.environ.get("GROQ_API_KEY")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _PLANNER_KEY,
        reason=("no planning LLM configured — set GROQ_API_KEY_PLANNING_AGENT "
                "(or GROQ_API_KEY) to run the end-to-end graph"),
    ),
]


class TestWorkflow(unittest.TestCase):
    def test_e2e_workflow(self):
        # Set up the graph
        graph = build_graph().compile()

        # Initial state
        initial_state = ETLState(
            messages=[HumanMessage(content="I want to group the salaries.csv file by 'job_title' and calculate the sum of 'salary_in_usd'.")],
            planner_definition={},
            ready_to_summarize=False,
            ready_to_code=False,
            coder_definition={},
            infrastructure_request=None,
            infrastructure_provisioned=None
        )

        # Run the planner
        planner_output = graph.invoke(initial_state)

        # Check planner output
        self.assertIn("planner_definition", planner_output)
        self.assertIsNotNone(planner_output["planner_definition"])

        # Manually trigger summarization
        summarizer_input = planner_output.copy()
        summarizer_input["ready_to_summarize"] = True
        summarizer_output = graph.invoke(summarizer_input)

        # Check summarizer output
        self.assertIn("planner_definition", summarizer_output)
        self.assertIsNotNone(summarizer_output["planner_definition"])
        self.assertIn("job_name", summarizer_output["planner_definition"])

        # Manually trigger coding
        coder_input = summarizer_output.copy()
        coder_input["ready_to_code"] = True
        coder_output = graph.invoke(coder_input)

        # Check coder output
        self.assertIn("coder_definition", coder_output)
        self.assertIsNotNone(coder_output["coder_definition"])
        self.assertIn("code", coder_output["coder_definition"])

        # Manually trigger infrastructure provisioning
        infra_input = coder_output.copy()
        infra_input["infrastructure_request"] = {"type": "spark"}
        infra_output = graph.invoke(infra_input)

        # Check infra output
        self.assertIn("infrastructure_provisioned", infra_output)
        self.assertIsNotNone(infra_output["infrastructure_provisioned"])
        self.assertEqual(infra_output["infrastructure_provisioned"]["status"], "provisioned")

if __name__ == "__main__":
    unittest.main()
