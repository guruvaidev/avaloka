from app.graph.etl_state import ETLState
from app.agents.mta_v2.agent import ModelTrainingAgent

def model_training_agent(state: ETLState) -> ETLState:
    """
    Agent function to execute model training inference.

    Parameters
    ----------
    state : ETLState
        The current state of the ETL process, which should include the training result details.

    Returns
    -------
    ETLState
        The updated state after executing the model training inference.
    """
    agent = ModelTrainingAgent()
    return agent.execute(state)

def model_training_agent_node(state: ETLState) -> ETLState:
    """
    Node function to execute model training inference.

    Parameters
    ----------
    state : ETLState
        The current state of the ETL process, which should include the training result details.

    Returns
    -------
    ETLState
        The updated state after executing the model training inference.
    """
    return model_training_agent(state)