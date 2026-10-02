"""A Python traceback must never be what a chat user reads.

The reported symptom was raw ``ValueError`` text appearing in conversation.
These tests pin the guarantee at both boundaries where an execution failure
becomes user-facing prose:

  * ``app/agents/result_renderer.py::_clean_error`` -- the execution-failure path
  * ``app/api/workflow.py::_describe_validation_failure`` -- the
    validation-refusal path

Deterministic: no LLM, no network.
"""

from __future__ import annotations

import pytest

from app.agents.result_renderer import _clean_error, render_response


def _exec_result(stderr: str = "", error: str = "") -> dict:
    return {"execution_stderr": stderr, "execution_error": error, "status": "error"}


TRACEBACK = """Traceback (most recent call last):
  File "/tmp/script.py", line 12, in <module>
    main(df)
  File "/tmp/script.py", line 8, in main
    df["rate"] = values
ValueError: Length of values (3) does not match length of index (55500)
"""


def test_the_reported_symptom_is_gone():
    """The exact shape from the bug report: a bare ValueError reaching chat."""
    msg = _clean_error(_exec_result(stderr=TRACEBACK))
    assert not msg.startswith("ValueError")
    assert "Traceback" not in msg
    # It says something a person can act on.
    assert "analysis" in msg.lower()
    # The raw text is preserved, but as a parenthetical, not as the message.
    assert "technical detail" in msg


@pytest.mark.parametrize("exc_line,expect_word", [
    ("KeyError: 'Revenue'", "column"),
    ("ValueError: could not convert string to float: 'abc'", "text"),
    ("TypeError: unsupported operand type(s) for +: 'int' and 'str'", "incompatible"),
    ("ZeroDivisionError: division by zero", "zero"),
    ("MemoryError", "memory"),
    ("AttributeError: 'DataFrame' object has no attribute 'foo'", "operation"),
    ("IndexError: index 5 is out of bounds", "position"),
    ("OverflowError: int too large to convert to float", "large"),
])
def test_known_exception_types_get_a_sentence(exc_line, expect_word):
    msg = _clean_error(_exec_result(stderr=f"Traceback...\n{exc_line}\n"))
    assert expect_word in msg.lower(), msg
    assert not msg.startswith(exc_line.split(":")[0]), msg


def test_unknown_exception_type_still_gets_a_sentence():
    """The open-set problem: the point is that an exception nobody anticipated
    still reads as prose rather than as a traceback line."""
    msg = _clean_error(_exec_result(stderr="Traceback...\nSomeExoticError: kaboom 42\n"))
    assert not msg.startswith("SomeExoticError")
    assert msg[0].isupper()
    assert msg.rstrip().endswith((".", ")"))
    assert "kaboom 42" in msg  # detail kept, just not as the headline


def test_completely_unparseable_stderr_still_gets_a_sentence():
    msg = _clean_error(_exec_result(stderr="???", error=""))
    assert "did not complete successfully" in msg


def test_empty_payload_does_not_crash():
    assert _clean_error({})
    assert _clean_error(None)


def test_timeout_keeps_its_advice():
    msg = _clean_error(_exec_result(error="Script execution timed out after 60 seconds"))
    assert "timed out" in msg
    assert "sample" in msg or "narrower" in msg


def test_render_response_wraps_the_cleaned_error():
    body = render_response(_exec_result(stderr=TRACEBACK))
    assert "Execution failed" in body
    assert "Traceback" not in body
    # The explanation leads; the raw exception is retained only as a trailing
    # parenthetical for whoever wants it. What must not happen is the raw line
    # BEING the message.
    headline, _, detail = body.partition("(technical detail:")
    assert "ValueError" not in headline
    assert "analysis" in headline.lower()
    assert "ValueError: Length of values" in detail


# ---------------------------------------------------------------------------
# the validation-refusal boundary
# ---------------------------------------------------------------------------

def test_validation_refusal_does_not_paste_coder_feedback():
    from app.api.workflow import _describe_validation_failure

    msg = _describe_validation_failure({
        "syntax_error": True,
        "code_validation_feedback": "SyntaxError or CompileError: unexpected EOF "
                                    "while parsing (<string>, line 14). Please fix "
                                    "the Python syntax/structure.",
    })
    assert "SyntaxError" not in msg
    assert "<string>" not in msg
    assert "nothing was executed" in msg


def test_contract_refusal_lists_the_unmet_criteria_in_plain_words():
    from app.api.workflow import _describe_validation_failure

    msg = _describe_validation_failure({
        "contract_error": True,
        "contract_violations": [
            {"rule": "not_aggregated",
             "detail": "the plan promised one row per ['hospital'] but the result "
                       "has 55500 rows against 55500 input rows"},
        ],
    })
    assert "one row per" in msg
    assert "{" not in msg  # no JSON blob in chat
    assert "rephras" in msg.lower()


def test_every_refusal_branch_produces_prose():
    from app.api.workflow import _describe_validation_failure

    for flag in ("syntax_error", "static_semantic_error", "logical_semantic_error"):
        msg = _describe_validation_failure({flag: True, "code_validation_feedback": "raw"})
        assert len(msg.split()) > 8
        assert "raw" not in msg
    # and the catch-all
    assert _describe_validation_failure({})
