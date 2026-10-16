"""The one MTA v2 flow test that reaches a real cloud endpoint.

Needs: network access to Google Cloud Storage. With `training_scheduled=True`
the agent inspects the training dataset's columns before calling the trainer
(`_columns_for_training_data` -> `get_columns`), and the Ray plan fixture
names a `gs://` object. The test stubs the LLM and RayTrainer but not that
read, so it makes an anonymous request to a real bucket and logs
"Anonymous caller does not have storage.objects.get access".

It was briefly in tests/test_mta_v2_flows_unit.py with the rest of that file.
A test that contacts a cloud API does not belong in the hermetic gate, so it is
back behind the `integration` marker until the dataset read is stubbed.
"""
from unittest.mock import MagicMock, patch

from tests.test_mta_v2_flows_unit import _make_ray_training_plan, _make_state


class TestExecuteTrainingRay:
    def test_runs_ray_trainer_when_already_scheduled(self):
        """Second call with training_scheduled=True → RayTrainer.train() is called."""
        from app.agents.mta_v2.agent import ModelTrainingAgent

        plan = _make_ray_training_plan()
        mock_result = {
            "mlflow_run_id": "ray-run-xyz",
            "dashboard_url": "http://localhost:8265",
            "final_accuracy": 0.91,
        }
        agent = ModelTrainingAgent()
        state = _make_state(training_plan=plan, training_scheduled=True)

        mock_ray_trainer = MagicMock()
        mock_ray_trainer.train.return_value = mock_result

        with patch("app.agents.mta_v2.agent.llm") as mock_llm, \
             patch("app.agents.mta_v2.agent.RayTrainer", return_value=mock_ray_trainer):
            mock_llm.invoke.return_value = MagicMock(content="Training done.", tool_calls=[])
            result = agent.execute(state)

        mock_ray_trainer.train.assert_called_once_with(plan)
        assert result["training_result"] == mock_result
