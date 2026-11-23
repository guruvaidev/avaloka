# Copyright 2026 Avaloka.ai Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for local execution timeout policy, sample runtime validation,
O(N^2) loop complexity guardrails, and post-execution retry routing."""

import textwrap

import pytest

import app.agents.execution_agent as exec_agent
from app.agents.validator import static_semantic_validator_node
from app.api.workflow import route_after_execution


def test_local_exec_timeout_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("AVALOKA_LOCAL_EXEC_TIMEOUT", raising=False)
    assert exec_agent._get_local_exec_timeout() == 180
    assert exec_agent._get_local_exec_timeout("quick_sample") == 180
    assert exec_agent._get_local_exec_timeout("entire_dataset") == 300

    monkeypatch.setenv("AVALOKA_LOCAL_EXEC_TIMEOUT", "240")
    assert exec_agent._get_local_exec_timeout() == 240
    assert exec_agent._get_local_exec_timeout("entire_dataset") == 240


def test_static_semantic_validator_blocks_quadratic_loops_on_entire_dataset():
    schema = {
        "id": "int64",
        "parent_id": "int64",
        "amount": "float64",
    }
    slow_code = textwrap.dedent('''
        import pandas as pd

        def main(df):
            rows = []
            for idx, row in df.iterrows():
                sub = df[df["id"] == row["parent_id"]]
                rows.append(len(sub))
            return pd.DataFrame({"matches": rows})
    ''')
    res = static_semantic_validator_node({
        "generated_code": slow_code,
        "schema": schema,
        "analysis_fidelity": "entire_dataset",
    })
    assert res["static_semantic_error"] is True
    assert "PerformanceError" in res["code_validation_feedback"]


def test_route_after_execution_retries_local_failure_once():
    failed_local_state = {
        "execution_mode": "local",
        "execution_error": "Script execution timed out after 180 seconds",
        "execution_result": {"status": "failed", "message": "Script execution timed out after 180 seconds"},
        "coder_definition": {"code": "def main(df): return df"},
    }
    assert route_after_execution(failed_local_state) == "retry_code"

    exhausted_local_state = {
        **failed_local_state,
        "coder_definition": {"code": "def main(df): return df", "execution_retried": True},
    }
    assert route_after_execution(exhausted_local_state) == "end"

    k8s_failed_state = {
        "execution_error": "Code execution failed",
        "execution_result": {"status": "failed", "message": "Code execution failed"},
        "coder_definition": {"code": "def main(df): return df"},
    }
    assert route_after_execution(k8s_failed_state) == "end"
