"""A data transfer runs once per message (Oct 15, Issue 4).

The planner's DTA fast-path runs the whole transfer itself. A sticky
execution_mode=k8s-ray used to seed infrastructure_request, the router sent the
turn to provision_infra, and provision_infra -> plan_etl ran the transfer again.
"""
import pytest

from app.agents import planner
from app.api import workflow


def _router_state(**extra):
    s = {
        "messages": [],
        "execution_mode": "k8s-ray",
        "data_source_location_cloud": "",
        "infrastructure_request": {"type": "gcp", "app_type": "python-docker", "use_ray": True},
        "infrastructure_provisioned": None,
        "ready_to_code": False,
        "ready_to_summarize": False,
    }
    s.update(extra)
    return s


# ----------------------------------------------------------- route_planner_output

def test_transfer_turn_ends_instead_of_provisioning_infra():
    assert workflow.route_planner_output(_router_state(is_dta_request=True)) == "end"


def test_without_transfer_flag_infra_request_still_provisions():
    # Unchanged behaviour for genuine infrastructure requests.
    assert workflow.route_planner_output(_router_state(is_dta_request=False)) == "provision_infra"


def test_transfer_turn_ends_even_with_code_and_cloud_source():
    state = _router_state(
        is_dta_request=True,
        data_source_location_cloud="gs://bucket/file.csv",
        connection_id="conn-1",
        coder_definition={"code": "print(1)"},
    )
    assert workflow.route_planner_output(state) == "end"


def test_skip_to_training_still_wins():
    assert workflow.route_planner_output(_router_state(is_dta_request=True, skip_to_training=True)) == "train_models"


class _RecordingGraph:
    """Stands in for langgraph's StateGraph and records how build_graph wires it."""

    def __init__(self, *args, **kwargs):
        self.conditional = {}

    def add_conditional_edges(self, source, path, path_map=None, *args, **kwargs):
        self.conditional[source] = (path, dict(path_map or {}))

    def compile(self, *args, **kwargs):
        return self

    def __getattr__(self, name):          # add_node, add_edge, set_entry_point, ...
        return lambda *args, **kwargs: None


def test_end_route_goes_to_verify_claims(monkeypatch):
    # Behaviour of the built graph: the planner's "end" branch leads to
    # verify_claims, so a transfer turn finishes there instead of re-planning.
    monkeypatch.setattr(workflow, "StateGraph", _RecordingGraph)
    graph = workflow.build_graph()
    path, path_map = graph.conditional["plan_etl"]
    assert path is workflow.route_planner_output
    assert path_map["end"] == "verify_claims"
    assert path_map[path(_router_state(is_dta_request=True))] == "verify_claims"


# ------------------------------------------------- is_dta_request reset per pass

@pytest.fixture
def no_gates(monkeypatch):
    monkeypatch.setenv("AVALOKA_DISABLE_PLAN_GATES", "1")


def test_flag_cleared_on_a_non_transfer_pass(monkeypatch, no_gates):
    # Left True by an earlier transfer turn; an analysis turn must not inherit it.
    monkeypatch.setattr(planner, "_plan_etl_job_impl", lambda s: s)
    out = planner.plan_etl_job({"messages": [], "is_dta_request": True})
    assert out["is_dta_request"] is False


def test_flag_kept_when_this_pass_handled_a_transfer(monkeypatch, no_gates):
    def impl(s):
        s["is_dta_request"] = True
        return s

    monkeypatch.setattr(planner, "_plan_etl_job_impl", impl)
    assert planner.plan_etl_job({"messages": []})["is_dta_request"] is True


def test_flag_present_when_impl_returns_a_new_dict(monkeypatch, no_gates):
    # LangGraph keeps the old value for keys a node does not return.
    monkeypatch.setattr(planner, "_plan_etl_job_impl", lambda s: {"messages": []})
    assert planner.plan_etl_job({"messages": [], "is_dta_request": True})["is_dta_request"] is False


def test_planner_then_router_end_to_end(monkeypatch, no_gates):
    def impl(s):
        s["is_dta_request"] = True        # what the DTA fast-path does
        return s

    monkeypatch.setattr(planner, "_plan_etl_job_impl", impl)
    out = planner.plan_etl_job(_router_state())
    assert workflow.route_planner_output(out) == "end"

    monkeypatch.setattr(planner, "_plan_etl_job_impl", lambda s: s)
    out = planner.plan_etl_job(_router_state(is_dta_request=True))
    assert workflow.route_planner_output(out) == "provision_infra"