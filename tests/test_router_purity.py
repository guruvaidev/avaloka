import copy

from app.api.workflow import route_planner_output, route_after_code


VALID_ROUTES = {
    "summarize", "prepare_code", "continue_planning", "task_operation",
    "provision_infra", "execute_locally", "execute_on_k8s", "execute_on_ray",
    "schedule_task", "end", "train_models",
}


def _assert_pure(router, state):
    before = copy.deepcopy(state)
    result = router(state)
    assert state == before, (
        f"{router.__name__} mutated state: "
        f"{ {k: (before.get(k), state.get(k)) for k in set(before) | set(state) if before.get(k) != state.get(k)} }"
    )
    assert result in VALID_ROUTES, result
    return result


# -------------------------
# route_planner_output must not mutate state (MAJ-080)
# -------------------------

def test_quick_sample_does_not_write_execution_mode():
    # The old code set state['execution_mode'] = 'local' here and lost it.
    state = {"analysis_fidelity": "quick_sample", "ready_to_code": True}
    result = _assert_pure(route_planner_output, state)
    assert result == "prepare_code"
    assert "execution_mode" not in state  # router must not have injected it


def test_portfolio_fidelity_pure_and_routes_to_code():
    state = {"analysis_fidelity": "portfolio_samples", "ready_to_code": True}
    result = _assert_pure(route_planner_output, state)
    assert result == "prepare_code"


def test_k8s_ray_fallback_guard_does_not_write_state():
    # k8s-ray requested but missing cloud uri / connection -> the router used to
    # write execution_mode='local' (lost). It must fall through purely.
    state = {
        "execution_mode": "k8s-ray",
        "data_source_location_cloud": "",
        "connection_id": None,
        "ready_to_code": True,
    }
    result = _assert_pure(route_planner_output, state)
    assert result == "prepare_code"
    # execution_mode in the input is left exactly as the caller passed it.
    assert state["execution_mode"] == "k8s-ray"


def test_explicit_task_operation_routes_to_scheduler_without_mutation():
    state = {"task_operation": {"type": "schedule"}, "ready_to_code": True}
    result = _assert_pure(route_planner_output, state)
    assert result == "task_operation"


def test_summarize_takes_precedence_and_is_pure():
    state = {"ready_to_summarize": True, "analysis_fidelity": "quick_sample"}
    result = _assert_pure(route_planner_output, state)
    assert result == "summarize"


def test_k8s_ray_with_code_routes_to_ray_purely():
    state = {
        "execution_mode": "k8s-ray",
        "data_source_location_cloud": "gs://bucket/data.csv",
        "connection_id": "conn-1",
        "coder_definition": {"code": "print(1)"},
    }
    result = _assert_pure(route_planner_output, state)
    assert result == "execute_on_ray"


# -------------------------
# route_after_code is already pure (PR #183) — guard it stays that way
# -------------------------

def test_route_after_code_schedules_without_mutation():
    state = {
        "task_schedule": {"schedule_type": "cron", "task_type": "execute"},
        "coder_definition": {"code": "print(1)"},
    }
    result = _assert_pure(route_after_code, state)
    assert result == "schedule_task"
