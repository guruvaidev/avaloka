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
O(N^2) loop complexity guardrails, fallback script generation, and post-execution retry routing."""

import textwrap

import pandas as pd
import pytest

import app.agents.execution_agent as exec_agent
from app.agents.validator import (
    execute_code_node,
    is_fatal_sample_execution_error,
    static_semantic_validator_node,
)
from app.api.workflow import check_validation_status, route_after_execution


def test_local_exec_timeout_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("AVALOKA_LOCAL_EXEC_TIMEOUT", raising=False)
    assert exec_agent._get_local_exec_timeout() == 180
    assert exec_agent._get_local_exec_timeout("quick_sample") == 180
    assert exec_agent._get_local_exec_timeout("entire_dataset") == 300

    monkeypatch.setenv("AVALOKA_LOCAL_EXEC_TIMEOUT", "240")
    assert exec_agent._get_local_exec_timeout() == 240
    assert exec_agent._get_local_exec_timeout("entire_dataset") == 240


def test_autogenerate_code_honors_derived_column_and_agg_func():
    state = {
        "messages": [
            {
                "role": "user",
                "content": "Return category_code, brand, total_revenue for top 5 by category_code using sum of revenue",
            }
        ],
        "schema": {
            "category_code": "object",
            "brand": "object",
            "revenue": "float64",
        },
    }
    code = exec_agent._autogenerate_code(state)
    assert ".sum()" in code
    assert "'total_revenue'" in code
    assert ".head(5)" in code


def test_execute_code_node_injects_avaloka_result_and_catches_runtime_crash():
    sample_csv = "id,created_at,amount\n1,2026-01-01 10:00:00,15.5\n2,2026-01-01 10:05:00,4.5\n"

    # 1) Calling avaloka_result inside main(df) works during sample validation
    good_code = textwrap.dedent('''
        import pandas as pd

        def main(df):
            total = df["amount"].sum()
            avaloka_result(value=total, kind="sum", columns=["amount"])
            return pd.DataFrame({"total_amount": [total]})
    ''')
    good_res = execute_code_node({"generated_code": good_code, "sample_data": sample_csv})
    assert good_res["execution_error"] is None
    assert good_res["execution_output_data"] == [{"total_amount": pytest.approx(20.0)}]

    # 2) Runtime TypeError in main(df) is flagged for retry
    bad_code = textwrap.dedent('''
        import pandas as pd

        def main(df):
            ts = pd.to_datetime(df["created_at"]).iloc[0]
            if 0 < ts:
                return df
            return df
    ''')
    bad_state = {"generated_code": bad_code, "sample_data": sample_csv, "retry_count": 1}
    bad_res = execute_code_node(bad_state)
    assert is_fatal_sample_execution_error(bad_res["execution_error"]) is True
    assert "RuntimeError during sample execution (TypeError)" in bad_res["code_validation_feedback"]

    merged_state = {**bad_state, **bad_res}
    assert check_validation_status(merged_state) == "refine"

    # Benign sample errors do not trigger refine
    benign_state = {"execution_error": "Execution returned an empty DataFrame.", "retry_count": 1}
    assert check_validation_status(benign_state) == "end"


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
