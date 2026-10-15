import os
import pytest

from app.agents import validator as validator_mod


def _base_state(**overrides):
    """
    Minimal dict-like CodingAgentState stub.
    Your validator code uses both state.get(...) and state["..."] indexing,
    so plain dict is the most compatible.
    """
    state = {
        "generated_code": "",
        "schema": {},
        "user_prompt": "Do something",
        "sample_data": "",
        "uploaded_csv_preview": None,
        "syntax_error": False,
        "static_semantic_error": False,
        "execution_stdout": "",
        "execution_stderr": "",
        "execution_error": None,
        "execution_output_preview": [],
    }
    state.update(overrides)
    return state


# -------------------------
# Syntax validation
# -------------------------

def test_syntactic_validator_flags_invalid_python():
    state = _base_state(
        generated_code="def main(df):\n    return\n    x ="
    )
    out = validator_mod.syntactic_validator_node(state)
    assert out["syntax_error"] is True
    assert "SyntaxError" in out["code_validation_feedback"] or "CompileError" in out["code_validation_feedback"]


# -------------------------
# Static semantic validation
# -------------------------

def test_static_semantic_blocks_unknown_column_read():
    # Reads df['missing'] which is not in schema and not created.
    code = """
import pandas as pd
def main(df):
    x = df['missing']
    return df
"""
    state = _base_state(
        generated_code=code,
        schema={"a": "int64"},
    )
    out = validator_mod.static_semantic_validator_node(state)
    assert out["static_semantic_error"] is True
    assert "Column 'missing' not in schema" in out["code_validation_feedback"]


def test_static_semantic_allows_created_columns():
    # df['new_col'] is created then read; should not be flagged.
    code = """
import pandas as pd
def main(df):
    df['new_col'] = 1
    y = df['new_col']
    return df
"""
    state = _base_state(
        generated_code=code,
        schema={"a": "int64"},
    )
    out = validator_mod.static_semantic_validator_node(state)
    assert out["static_semantic_error"] is False


def test_static_semantic_blocks_str_accessor_on_numeric_schema_column():
    # This is the exact crash class you saw: df['subscribers'] already numeric -> .str fails at runtime.
    code = """
import pandas as pd
def main(df):
    df['subscribers'] = pd.to_numeric(df['subscribers'].str.replace(',', ''))
    return df
"""
    state = _base_state(
        generated_code=code,
        schema={"subscribers": "int64"},
    )
    out = validator_mod.static_semantic_validator_node(state)
    assert out["static_semantic_error"] is True
    assert "do not use .str on it" in out["code_validation_feedback"]


# -------------------------
# Execution validation
# -------------------------

def test_execute_code_node_runs_main_and_returns_preview():
    code = """
import pandas as pd
def main(df):
    print("hello from main")
    return df.head(1).reset_index(drop=True)
"""
    sample_csv = "a,b\n1,2\n3,4\n"
    state = _base_state(
        generated_code=code,
        schema={"a": "int64", "b": "int64"},
        sample_data=sample_csv,
        syntax_error=False,
        static_semantic_error=False,
    )

    out = validator_mod.execute_code_node(state)

    assert out["execution_error"] is None
    assert "hello from main" in (out["execution_stdout"] or "")
    assert isinstance(out["execution_output_preview"], list)
    assert len(out["execution_output_preview"]) == 1
    assert out["execution_output_preview"][0]["a"] in (1, "1")  # depends on sample parsing path


def test_execute_code_node_skips_when_validation_failed():
    state = _base_state(
        syntax_error=True,
        generated_code="def main(df):\n    return df",
        sample_data="a\n1\n",
    )
    out = validator_mod.execute_code_node(state)
    assert out["execution_error"] == "Validation failed."
    assert "Skipped execution" in out["execution_stderr"]


# -------------------------
# Logical semantic validation (strict vs non-strict)
# -------------------------

class _DummyResp:
    def __init__(self, content):
        self.content = content


class _DummyLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        return _DummyResp(self._content)


@pytest.mark.parametrize(
    "strict_mode,expected_blocking",
    [
        ("0", False),  # advisory (explicit opt-out)
        ("1", True),   # strict gating (also the default when unset)
    ],
)
def test_logical_semantic_strict_vs_non_strict(monkeypatch, strict_mode, expected_blocking):
    monkeypatch.setenv(validator_mod.STRICT_LOGIC_ENV, strict_mode)
    monkeypatch.setattr(
        validator_mod,
        "validator_llm",
        _DummyLLM('{"is_logically_correct": false, "rationale": "Logic issue"}'),
    )

    state = _base_state(
        generated_code="def main(df):\n    return df",
        execution_stdout="",
        execution_stderr="",
        execution_error=None,
        execution_output_preview=[{"a": 1}],
        user_prompt="Return a computed output",
    )

    out = validator_mod.logical_semantic_validator_node(state)

    if expected_blocking:
        assert out["logical_semantic_error"] is True
        assert out["code_validation_feedback"] == "Logic issue"
        assert "logical_review_feedback" not in out
    else:
        assert out["logical_semantic_error"] is False
        assert out["code_validation_feedback"] == "APPROVED"
        assert out["logical_review_feedback"] == "Logic issue"


# ===========================================================================
# DeprecatedApiFixer: DataFrame.append() -> pd.concat()
#
# pandas 2.0 removed DataFrame.append. The coder prompt has forbidden it for a
# while and the model wrote it anyway (medication_vs_test_results, and again in
# the YouTube suite), so it is enforced mechanically here rather than asked for.
#
# The hazard is that list.append() is ordinary Python -- the same failing run
# contained a dozen legitimate summary_lines.append(f'...') calls. Two
# unambiguous signals are used: the result is ASSIGNED (list.append returns
# None), or a pandas-only keyword is present. Comparison is by AST, not source
# text, because ast.unparse normalises quote style.
# ===========================================================================

import ast
from app.agents.validator import DeprecatedApiFixer
def _fix(src: str) -> ast.AST:
    tree = DeprecatedApiFixer().visit(ast.parse(src))
    ast.fix_missing_locations(tree)
    return tree


def _unchanged(src: str) -> bool:
    return ast.dump(_fix(src)) == ast.dump(ast.parse(src))


# ---------------------------------------------------------------------------
# Must be rewritten
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    [
        # The exact line from the failing medication_vs_test_results run.
        "combined = combined.append(overall_row)",
        # The idiom pandas removed: a dict row plus ignore_index.
        "df = df.append({'a': 1}, ignore_index=True)",
        # Result discarded, but the keyword gives it away.
        "out.append(row, ignore_index=True)",
        # Chained receiver.
        "res = res.append(df.iloc[[0]], ignore_index=True)",
    ],
)
def test_dataframe_append_becomes_concat(src):
    out = ast.unparse(_fix(src))
    assert "pd.concat" in out, f"not rewritten: {src} -> {out}"
    assert ".append(" not in out, f"append survived the rewrite: {out}"


def test_a_dict_row_is_wrapped_in_a_dataframe():
    """pd.concat takes frames; a bare dict would raise."""
    out = ast.unparse(_fix("df = df.append({'a': 1}, ignore_index=True)"))
    assert "pd.DataFrame([{'a': 1}])" in out, out


def test_an_explicit_ignore_index_false_is_preserved():
    """Rewriting must not quietly change reindexing behaviour."""
    out = ast.unparse(_fix("df = df.append(other, ignore_index=False)"))
    assert "ignore_index=False" in out, out


def test_the_rewritten_code_still_parses():
    out = ast.unparse(_fix("combined = combined.append(overall_row)"))
    compile(out, "<rewritten>", "exec")


# ---------------------------------------------------------------------------
# Must be left alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    [
        # All of these appeared in the same run as the failure above.
        "summary_lines.append(f'Medication: {med} | Abnormal: {pct}%')",
        "encodings.append('utf-16')",
        "results.append(compute(x))",
        "rows.append([1, 2])",
        # A list of frames, built for a later concat -- the receiver is a list
        # even though the argument is a DataFrame.
        "acc.append(df.iloc[0])",
        "frames.append(pd.DataFrame(data))",
    ],
)
def test_list_append_is_untouched(src):
    assert _unchanged(src), (
        f"a list append was rewritten, which turns working code into broken "
        f"code: {src} -> {ast.unparse(_fix(src))}"
    )


def test_the_collect_then_concat_pattern_survives_intact():
    """The pattern the coder is *supposed* to use must pass through unharmed."""
    src = (
        "frames = []\n"
        "for name, group in df.groupby('Medication'):\n"
        "    frames.append(group.head(1))\n"
        "result = pd.concat(frames, ignore_index=True)\n"
    )
    assert _unchanged(src), ast.unparse(_fix(src))
