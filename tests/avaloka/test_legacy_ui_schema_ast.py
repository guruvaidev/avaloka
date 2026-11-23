"""Import-free guard for the API fields/routes used by the 1.5.2 UI."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _class_fields(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            }
    raise AssertionError(f"class {class_name} not found in {path}")


def _routes(path: Path) -> set[tuple[str, str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    routes = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            function = decorator.func
            if not (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "app"
                and function.attr in {"get", "post", "delete", "patch", "put"}
                and isinstance(decorator.args[0], ast.Constant)
            ):
                continue
            routes.add((function.attr.upper(), decorator.args[0].value))
    return routes


def test_chat_response_keeps_legacy_1_5_2_fields():
    fields = _class_fields(ROOT / "app/api/schemas.py", "ChatResponse")
    expected = {
        "messages",
        "planner_definition",
        "ready_to_summarize",
        "ready_to_code",
        "coder_definition",
        "output_file_data",
        "output_json",
        "reasoning",
        "task_info",
        "analysis_fidelity",
        "selected_sample_name",
        "analysis_task_id",
        "execution_context",
        "datasets",
        "active_dataset_ids",
        "visualization_config",
        "visualization_status",
        "visualization_configs",
        "visualization_statuses",
    }
    assert expected <= fields


def test_server_keeps_legacy_thread_upload_task_and_database_routes():
    routes = _routes(ROOT / "app/api/server.py")
    expected = {
        ("GET", "/health"),
        ("GET", "/version"),
        ("POST", "/threads"),
        ("POST", "/threads/{thread_id}/messages"),
        ("GET", "/threads/{thread_id}/messages"),
        ("DELETE", "/threads/{thread_id}"),
        ("POST", "/api/upload"),
        ("GET", "/tasks/{task_id}/result/{index}"),
        ("POST", "/api/database/connect"),
        ("POST", "/api/v1/database/query"),
    }
    assert expected <= routes


def test_database_models_accept_and_return_active_1_5_2_workspace_fields():
    schemas = ROOT / "app/api/schemas.py"
    connect_fields = _class_fields(schemas, "DatabaseConnectRequest")
    query_fields = _class_fields(schemas, "DatabaseChatRequest")
    response_fields = _class_fields(schemas, "DatabaseChatResponse")

    assert {"connection_id", "customer_id", "api_key"} <= connect_fields
    assert {"connection_id", "session_id", "query", "content"} <= query_fields
    assert {
        "dataset_id",
        "thread_id",
        "session_id",
        "connection_id",
        "results",
        "schema_",
        "row_count",
        "execution_time_ms",
    } <= response_fields
    assert 'Field(None, alias="schema")' in schemas.read_text()


def test_database_shim_resolves_owned_connections_and_keeps_modern_contract():
    server_source = (ROOT / "app/api/server.py").read_text()
    resolver_source = (
        ROOT / "app/api/legacy_database_connections.py"
    ).read_text()

    assert "resolve_legacy_database_credentials" in server_source
    assert "database_query_text(body.content, body.query)" in server_source
    assert "legacy_connect_fields(" in server_source
    assert "legacy_query_fields(" in server_source
    assert '.eq("user_id", user_id)' in resolver_source
    assert '"customer_id": customer_id' in server_source
    assert 'query_result=result' in server_source
