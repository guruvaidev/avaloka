"""The sample run gives generated code what the full run gives it.

The coder is told ``avaloka_result()`` is pre-injected into every script and
to call it for a scalar answer, and the full-run script's prelude defines it
(app/agents/execution_agent.py). The sandboxed sample run exec'd the code with
``pd`` alone, so every such answer failed validation with
``name 'avaloka_result' is not defined`` and was sent back to the coder.
"""
import textwrap

import pytest

from app.agents.validator import execute_code_node

SAMPLE_CSV = "id,amount\n1,15.5\n2,4.5\n"


def test_a_scalar_answer_written_as_instructed_passes_sample_validation():
    code = textwrap.dedent('''
        import pandas as pd

        def main(df):
            total = df["amount"].sum()
            avaloka_result(value=total, kind="sum", columns=["amount"])
            return pd.DataFrame({"total_amount": [total]})
    ''')
    res = execute_code_node({"generated_code": code, "sample_data": SAMPLE_CSV})
    assert res["execution_error"] is None, res["execution_error"]
    assert res["execution_output_data"] == [{"total_amount": pytest.approx(20.0)}]


def test_numpy_is_in_scope_as_it_is_for_the_full_run():
    code = textwrap.dedent('''
        import pandas as pd

        def main(df):
            return pd.DataFrame({"log_total": [float(np.log1p(df["amount"].sum()))]})
    ''')
    res = execute_code_node({"generated_code": code, "sample_data": SAMPLE_CSV})
    assert res["execution_error"] is None, res["execution_error"]


def test_a_real_crash_is_still_reported():
    code = "import pandas as pd\n\ndef main(df):\n    return df['no_such_column'].sum()\n"
    res = execute_code_node({"generated_code": code, "sample_data": SAMPLE_CSV})
    assert res["execution_error"]
