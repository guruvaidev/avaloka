"""The chat voice's execution prompt after a turn the replan loop handled.

Before, every failed run got "Shall I retry, or show you what I tried?" with
a Retry button -- after the loop had already retried three times, or had
stopped to ask the user a question the button would not answer. Each test
fails on that tree on the message assertion.
"""

from app.api import chat_voice


def _execution_prompt(final):
    prompts = chat_voice.prompts_after_turn(final, question="total revenue")
    return next(p for p in prompts if p["step"] == "execution")


def test_a_stop_for_the_users_decision_offers_the_two_answers():
    final = {"execution_result": {"status": "failed", "message": "Script failed with exit status 1"},
             "replan_final": "needs_user", "replan_attempts": []}
    ex = _execution_prompt(final)
    assert ex["message"] == "I stopped before changing your data: the repair needs your decision."
    assert ex["question"] == "Should I leave those rows out, or fill them with the median?"
    sends = [r["send"] for r in ex["quick_replies"]]
    assert sends[:2] == ["Leave them out", "Fill with the median"]
    assert "Retry" not in sends
    # Both answers are what the planner's resume step reads (replan_choice).
    from app.agents import replan_choice

    assert replan_choice.parse_choice(sends[0]) == replan_choice.LEAVE_OUT
    assert replan_choice.parse_choice(sends[1]) == replan_choice.MEDIAN


def test_exhausted_attempts_say_how_many_and_do_not_offer_a_retry():
    attempt = {"n": 1, "action": "code_fix", "outcome": "failed_again"}
    final = {"execution_result": {"status": "failed", "message": "Script failed with exit status 1"},
             "replan_final": "exhausted", "replan_attempts": [attempt, {**attempt, "n": 2}, {**attempt, "n": 3}]}
    ex = _execution_prompt(final)
    assert ex["message"] == "I couldn't finish the analysis after 3 attempts to repair it."
    sends = [r["send"] for r in ex["quick_replies"]]
    assert sends[0] == "Show what I tried" and "Retry" not in sends
    assert "exit status" not in ex["message"]


def test_a_failed_run_the_loop_did_not_touch_still_offers_a_retry():
    final = {"execution_result": {"status": "failed", "message": "KeyError: 'x'"}}
    ex = _execution_prompt(final)
    assert ex["quick_replies"][0]["send"] == "Retry"
