"""The contract between the ingest's vocabularies and the product's own.

The ingest restates every list it enforces -- providers, default models,
editions, agent names, kinds of column, file formats -- because it deploys as
its own image and imports nothing from the product. These tests import both
sides and fail when a list changes on one of them only. Nothing here runs in
the service.

A failure means the product gained or lost a value the ingest does not know
about. The fix is to change ``telemetry_service/schema.py`` to match, and to
ask whether an installation already in the field would now be refused.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from app.analytics import schema as analytics_schema
from app.core import editions, inference, model_config
from telemetry_service import schema

REPO = Path(__file__).resolve().parents[2]


def _source(relpath: str) -> str:
    return (REPO / relpath).read_text(encoding="utf-8")


class TestProviders:
    def test_the_provider_list_is_the_routers_canonical_names(self):
        """Every alias the router accepts resolves to one of these, so the set
        of values is the set of providers a call can be routed to."""
        canonical = set(inference._PROVIDER_ALIASES.values()) | {
            inference.GROQ, inference.LOCAL, inference.OPENAI, inference.OPENROUTER,
            inference.BEDROCK, inference.VERTEX, inference.AZURE,
        }
        assert schema.LLM_PROVIDERS.values == canonical

    def test_every_supported_backend_is_a_provider_the_ingest_accepts(self):
        assert set(model_config.SUPPORTED_BACKENDS) <= schema.LLM_PROVIDERS.values

    def test_there_is_no_catch_all_provider(self):
        assert "other" not in schema.LLM_PROVIDERS.values


class TestModels:
    def test_the_known_models_are_the_products_defaults_and_other(self):
        assert schema.LLM_MODELS.values == set(model_config.DEFAULT_MODELS.values()) | {"other"}

    def test_other_is_the_only_value_that_is_not_a_real_model(self):
        assert "other" not in model_config.DEFAULT_MODELS.values()


class TestEditions:
    def test_the_editions_are_the_products(self):
        assert schema.EDITIONS.values == set(editions.EDITIONS)


class TestAgents:
    def _graph_nodes(self) -> set[str]:
        names: set[str] = set()
        for relpath in ("app/api/workflow.py", "app/infra/workflow.py"):
            names |= set(re.findall(r"\.add_node\(\s*\"([a-z0-9_]+)\"", _source(relpath)))
        return names

    def _fireflies(self) -> set[str]:
        names: set[str] = set()
        for path in sorted((REPO / "avaloka" / "fireflies").glob("*.py")):
            if path.name in {"__init__.py", "base.py"}:
                continue
            names |= set(re.findall(r"^    name = \"([a-z_]+)\"", path.read_text(), re.M))
        return names

    def test_the_sources_were_found(self):
        """Otherwise the equality below could pass against two empty sets."""
        assert {"coder", "execute_code", "plan_etl", "visualize"} <= self._graph_nodes()
        assert {"data_scout", "reporter", "validator"} <= self._fireflies()

    def test_the_agent_names_are_the_graph_nodes_and_the_fireflies(self):
        assert schema.AGENTS.values == self._graph_nodes() | self._fireflies()


class TestColumnKinds:
    def _profiler_roles(self) -> set[str]:
        tree = ast.parse(_source("avaloka/stats.py"))
        fn = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "infer_role")
        return {node.value.value for node in ast.walk(fn)
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)}

    def test_the_kinds_are_the_profilers_roles_and_other(self):
        roles = self._profiler_roles()
        assert {"numeric", "text", "datetime"} <= roles
        assert schema.COLUMN_KINDS == roles | {"other"}

    def test_other_is_an_explicit_bucket_not_a_profiler_role(self):
        assert "other" not in self._profiler_roles()


class TestSharedVocabularies:
    def test_both_producers_use_one_file_format_vocabulary(self):
        process = schema.PROCESS_FIELDS_BY_TYPE["dataset"]["file_format"]
        analytics = schema.ANALYTICS_EVENTS["dataset.connect_completed"]["file_format"]
        assert process is analytics
        assert process.values == analytics_schema.FILE_FORMATS

    def test_both_producers_use_one_outcome_vocabulary(self):
        process = schema.PROCESS_FIELDS_BY_TYPE["agent.completed"]["outcome"]
        analytics = schema.ANALYTICS_EVENTS["question.completed"]["outcome"]
        assert process is analytics
        assert process.values == analytics_schema.OUTCOMES

    def test_the_size_bands_differ_between_producers_and_that_is_known(self):
        """Not unified here on purpose. If this starts failing because they
        were unified, delete it and the comment in schema.py together."""
        process = schema.PROCESS_FIELDS_BY_TYPE["dataset"]
        analytics = schema.ANALYTICS_EVENTS["dataset.connect_completed"]
        assert process["row_count_bucket"].values != analytics["row_count_bucket"].values
        assert process["column_count_bucket"].values != analytics["column_count_bucket"].values
        assert analytics["row_count_bucket"].values == analytics_schema.ROW_BUCKETS

    def test_ids_classes_and_versions_follow_one_rule_for_both_producers(self):
        process = schema.PROCESS_FIELDS_BY_TYPE
        common = schema.ANALYTICS_COMMON
        assert process["install"]["install_id"] is common["install_id"]
        assert process["agent.started"]["session_id"] is common["session_id"]
        assert process["dataset"]["event_id"] is common["event_id"]
        assert (process["agent.completed"]["error_class"]
                is schema.ANALYTICS_EVENTS["error.shown"]["error_class"])

    def test_the_ui_version_is_the_producers_pattern_and_the_others_are_wider(self):
        """`ui_version` is ours and tied to the analytics schema. An
        interpreter's version is not ours, so it has a wider pattern of its
        own; the two are deliberately different rules."""
        install = schema.PROCESS_FIELDS_BY_TYPE["install"]
        assert schema.ANALYTICS_COMMON["ui_version"].kind == "version"
        assert schema._VERSION.pattern == analytics_schema._VERSION.pattern
        assert install["python_version"].kind == install["avaloka_version"].kind == "release_version"
        assert schema._RELEASE_VERSION.pattern != schema._VERSION.pattern
