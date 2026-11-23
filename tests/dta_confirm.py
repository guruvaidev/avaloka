"""Answer a transfer's confirmation the way the user does ("yes").

Not a test module. Every chat transfer now stops at one question naming the
source and destination connection and table (J1 follow-up); tests that are
about what happens AFTER that question use these to answer it.
"""
import app.agents.planner as planner


def awaiting_confirmation(state) -> bool:
    pending = state.get("pending_clarification")
    return (isinstance(pending, dict) and pending.get("type") == "dta_transfer"
            and pending.get("awaiting") == "confirm_transfer")


def confirmed(state, msg, reply: str = "yes"):
    """If ``msg`` was the confirmation question, answer it and return the result."""
    if awaiting_confirmation(state):
        return planner._resolve_pending_dta(state, state["pending_clarification"], reply)
    return msg


def run_confirmed(state, params):
    """_handle_initiate_transfer, then "yes" to its confirmation."""
    return confirmed(state, planner._handle_initiate_transfer(state, params))
