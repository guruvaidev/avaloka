"""The chat voice's plain sentences are true of THIS turn (2.0, c-strict).

From the rc17 web regrade (rweb17). The related cell asked "How does the
average order amount differ by customer segment?" and then "What tables do I
have and how do they relate?". The second turn wrote and ran no code
(``coder_definition`` empty, no output), yet its plan step said, unmarked,
"I wrote the code for it and had my validators read it before it ran.": the
card read ``generated_code``, which the checkpointed graph state carries over
from the turn before, byte for byte. That turn reached chit-chat at all
because the keyword 'hey' matched inside "t*hey*".

And under c-strict every word a model wrote carries the commentary mark: the
plan step's "My plan had 3 steps: Filter by Popularity, Group by Cuisine,
Sort by Count." was plain, though the names are the planner model's and the
code that ran grouped by no cuisine column.

The turns are the stored states' minimum fields
(tests/fixtures/chat_voice_live/rweb17_turns.json); the training and drivers
cases put the session-reloaded fields on that live shape and say so.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.agents.avaloka_agent.intent_classifier import _heuristic_intent
from app.api import chat_voice

LIVE = json.loads((Path(__file__).parent / "fixtures" / "chat_voice_live" / "rweb17_turns.json").read_text())


def _turn(name: str, **final_overrides):
    case = copy.deepcopy(LIVE[name])
    case["final"].update(final_overrides)
    return chat_voice.prompts_after_turn(case["final"], question=case["question"],
                                         execution_context=case["execution_context"])


def _plain(prompts) -> list[str]:
    return [s["text"] for p in prompts for s in p["message_segments"] if s["kind"] == "plain"]


def _commentary(prompts) -> list[str]:
    return [s["text"] for p in prompts for s in p["message_segments"] if s["kind"] == "commentary"]


def _step(prompts, step):
    return next((p for p in prompts if p["step"] == step), None)


# ------------------------------------------------------------ this turn's code
def test_a_turn_that_wrote_no_code_does_not_say_it_wrote_code():
    tables = LIVE["related_ask_tables"]["final"]
    # The live shape: this turn's coder_definition is empty and nothing ran;
    # generated_code is the previous turn's, byte for byte.
    assert tables["coder_definition"] == {} and tables["execution_result"] == {} and tables["output_json"] is None
    assert tables["generated_code"] == LIVE["related_ask_seg"]["final"]["generated_code"]
    assert "I wrote the code for it and had my validators read it before it ran." in LIVE["related_ask_tables"]["shown"]

    prompts = _turn("related_ask_tables")
    said = " ".join(p["message"] for p in prompts)
    assert "I wrote the code" not in said, said
    assert _step(prompts, "plan") is None

    # The turn that did write and run code still says so, plain: no true
    # sentence is lost (ask_seg without its planner steps, so the code line shows).
    seg = _turn("related_ask_seg", planner_definition={})
    assert "I wrote the code for it and had my validators read it before it ran." in _plain(seg)
    # ...and code written this turn that has not run yet is not said to have run.
    deferred = _turn("related_ask_seg", planner_definition={}, execution_result={}, output_json=None)
    assert "I wrote the code for it and had my validators read it." in _plain(deferred)


def test_a_check_failure_from_an_earlier_turn_is_not_this_turns():
    """contract_violations is not reset between turns either: only a turn whose
    coding step ran may say its code failed the checks."""
    stale = ["'amount' was declared float64 but holds object"]
    prompts = _turn("related_ask_tables", contract_violations=stale)
    assert not any("didn't pass my own checks" in t for t in _plain(prompts)), _plain(prompts)
    # The same violations on a turn whose coding step ran are still said.
    ran = LIVE["related_ask_tables"]["final"]["agent_trace"] + [
        {"turn_id": LIVE["related_ask_tables"]["final"]["trace_turn_id"], "node": "code_etl"}]
    prompts = _turn("related_ask_tables", contract_violations=stale, agent_trace=ran)
    assert any("didn't pass my own checks" in t for t in _plain(prompts))


# --------------------------------------------- state the server reloads per turn
_SESSION_TRAINING = {  # what the server reloads from the session after a training turn
    "training_plan": {"model_name": "Stroke Prediction Model", "model_type": "classification",
                      "data_config": {"target_column": "stroke"}},
    "training_result": {"status": "success", "mlflow_run_id": "r1"},
    "training_completed": True,
    "evaluation_beats_baseline": True,
}


def test_a_turn_that_trained_nothing_does_not_say_it_trained():
    """server.py hydrates training_plan/training_result/training_completed from
    the session on every turn; on the live chit-chat shape they are not this
    turn's work."""
    prompts = _turn("related_ask_tables", **_SESSION_TRAINING)
    said = " ".join(p["message"] for p in prompts)
    assert "I trained" not in said and "Stroke Prediction Model" not in said, said
    assert "That's my answer" not in said

    # On the turn the training node ran, she says it, and the model's name --
    # the training planner model's words -- is commentary.
    turn = LIVE["related_ask_tables"]["final"]["trace_turn_id"]
    trained = _turn("related_ask_tables", **_SESSION_TRAINING,
                    agent_trace=[{"turn_id": turn, "node": "train_models"}])
    ex = _step(trained, "execution")
    assert ex["message_segments"][0] == {
        "text": "I trained a classification model to predict `stroke`. It beats the trivial baseline by more "
                "than fold noise. ", "kind": "plain"}
    assert ex["message_segments"][1:] == [{"text": "My planner named it Stroke Prediction Model.",
                                           "kind": "commentary"}]


def test_an_intent_from_an_earlier_turn_is_not_how_she_read_this_one():
    """A drivers answer ends the turn at drivers_route: avaloka_agent never ran,
    and avaloka_intent is the previous turn's."""
    turn = LIVE["related_ask_tables"]["final"]["trace_turn_id"]
    prompts = _turn("related_ask_tables", avaloka_intent="statistical_analysis",
                    agent_trace=[{"turn_id": turn, "node": "drivers_route"}])
    assert not any("statistical analysis" in t for t in _plain(prompts)), _plain(prompts)


# ------------------------------------------------------------------ c-strict
def test_the_planners_step_names_carry_the_commentary_mark():
    case = LIVE["recipes_chat"]
    names = [t["name"] for t in case["final"]["planner_definition"]["transformations"]]
    assert "Group by Cuisine" in names
    assert "My plan had 3 steps: Filter by Popularity, Group by Cuisine, Sort by Count." in case["shown"]

    prompts = _turn("recipes_chat")
    plan = _step(prompts, "plan")
    # The count is the plan's, plain; the names are the model's, marked.
    assert plan["message_segments"][0] == {"text": "My plan had 3 steps. ", "kind": "plain"}
    assert plan["message"].endswith(
        "My planner named them: Filter by Popularity, Group by Cuisine, Sort by Count. (commentary — not checked)")
    for name in names:
        assert not any(name in t for t in _plain(prompts)), (name, _plain(prompts))
        assert any(name in t for t in _commentary(prompts)), name
    # The understanding step names the job without the model's step name.
    assert "I read “what kinds of recipes dominate” as a job for my planner." in _plain(prompts)


def test_a_domain_the_model_picked_is_commentary_on_upload():
    domain = LIVE["recipes_upload_domain"]["domain"]
    assert domain["source"] == "model"
    prompts = chat_voice.prompts_after_upload(sample_statistics=None, profiling_result={"domain": domain},
                                              schema=["Name", "Ingredients"], filename="recipes.jsonl")
    assert not any("reads like" in t for t in _plain(prompts)), _plain(prompts)
    assert "It reads like generic data." in _commentary(prompts)
    # A domain the vocabulary decided from the column names stays plain.
    decided = chat_voice.prompts_after_upload(sample_statistics=None,
                                              profiling_result={"domain": {**domain, "source": "vocabulary"}},
                                              schema=["Name", "Ingredients"], filename="recipes.jsonl")
    assert decided[0]["message_segments"] == [
        {"text": "I've had a first look at recipes.jsonl: 2 columns — this reads like generic data. "
                 "I haven't measured how complete the columns are yet, so I can't say whether any are empty.",
         "kind": "plain"}]


# ------------------------------------------------------------------- routing
def test_the_related_tables_question_is_not_chit_chat():
    question = LIVE["related_ask_tables"]["question"]
    assert LIVE["related_ask_tables"]["final"]["avaloka_intent_reasoning"] == "keyword match: 'hey'"
    result = _heuristic_intent(question, False)
    assert result.intent != "chit_chat", result
    assert result.should_delegate()
    # Whole words still match: a greeting is still a greeting.
    for message, expected in [
        ("hey", "chit_chat"), ("Hey!", "chit_chat"), ("hi", "chit_chat"), ("hi there", "chit_chat"),
        ("hello", "chit_chat"), ("who are you?", "chit_chat"), ("what can you do?", "chit_chat"),
        ("run the ETL nightly", "data_transfer"), ("add a cron job", "schedule"),
        ("set up a crontab entry", "schedule"), ("run an ANOVA on price by region", "statistical_analysis"),
    ]:
        assert _heuristic_intent(message, False).intent == expected, message


@pytest.mark.parametrize("message", [
    "How do they relate?",                         # 'hey' in 'they'
    "Total sales in Delhi by month",               # 'hi ' in 'delhi '
    "Who are your best customers by revenue?",     # 'who are you' in 'who are your'
    "Count the Othello games per year",            # 'hello' in 'othello'
    "What can you download from the orders table",  # 'what can you do' in '... download'
])
def test_a_greeting_inside_another_word_is_not_a_greeting(message):
    assert _heuristic_intent(message, False).intent != "chit_chat"


@pytest.mark.parametrize("message,unwanted", [
    ("Compute Cronbach's alpha for the survey items", "schedule"),
    ("Average premium for MetLife policies", "data_transfer"),
    ("Mean score for Villanova applicants", "statistical_analysis"),
])
def test_other_short_keywords_match_whole_words_only(message, unwanted):
    assert _heuristic_intent(message, False).intent != unwanted


def test_a_greeting_in_front_of_a_request_is_not_chit_chat():
    """rc18: chit-chat only when nothing but the greeting is there."""
    for message in ("hey, what's the average age?", "hi — show revenue by country", "Hi! Which country sells most",
                    "hello, can you chart sales", "Hey there, how many rows have a missing age"):
        result = _heuristic_intent(message, False)
        assert result.intent != "chit_chat", (message, result)
        assert result.should_delegate(), (message, result)  # the planner takes it: analysis
    assert _heuristic_intent("hello, can you chart sales", False).intent == "visualization"
    # A greeting alone, with its filler or its own question mark, is still one.
    for message in ("hi", "hey there", "hello Avaloka", "Hello, Avaloka!", "who are you", "who are you?",
                    "what can you do?", "good morning, hey", "hey?"):
        result = _heuristic_intent(message, False)
        assert result.intent == "chit_chat", (message, result)
        assert not result.should_delegate()


def test_schedule_and_provision_are_whole_words_with_their_inflections():
    """"unscheduled" and "provisional" describe the data; "rescheduled" is a
    schedule request (its own keyword), and the plain inflections still match."""
    for message, unwanted in [("count unscheduled maintenance events per plant", "schedule"),
                              ("Average of the provisional figures for Q3", "infrastructure"),
                              ("revenue provisionally booked by region", "infrastructure")]:
        assert _heuristic_intent(message, False).intent != unwanted, message
    for message, expected in [("schedule this every week", "schedule"), ("is the job scheduled", "schedule"),
                              ("rescheduled the export for monday", "schedule"),
                              ("reschedule the report", "schedule"), ("use the scheduler", "schedule"),
                              ("provision a gke cluster", "infrastructure"),
                              ("provisioning ray workers on gke", "infrastructure")]:
        assert _heuristic_intent(message, False).intent == expected, message


def test_a_three_way_tie_is_not_called_a_leader():
    """rweb17 nullxlsx upload: `Outcome` is 20/20/20 over Recovered, Readmitted
    and RECOVERED, and the card said it "leans on “Recovered” (33% of rows)"."""
    stats = LIVE["nullxlsx_upload_statistics"]["sample_statistics"]
    tops = stats["column_statistics"]["Outcome"]["top_values"]
    assert [t["count"] for t in tops] == [20, 20, 20]
    prompts = chat_voice.prompts_after_upload(sample_statistics=stats, filename="null_words_three_sheets [2022].csv",
                                              total_rows=60)
    said = " ".join(_plain(prompts))
    assert "leans on" not in said, said
    assert ("`Outcome` has no single leader: “Recovered”, “Readmitted” and “RECOVERED” hold 33% of rows each"
            in said), said
    # A real leader is still named (stroke's ever_married: Yes 66%).
    lead = copy.deepcopy(stats)
    lead["column_statistics"]["Outcome"]["top_values"] = [
        {"value": "Recovered", "count": 40, "frequency": 0.6667}, {"value": "Readmitted", "count": 20, "frequency": 0.3333}]
    said = " ".join(_plain(chat_voice.prompts_after_upload(sample_statistics=lead, total_rows=60)))
    assert "`Outcome` leans on “Recovered” (67% of rows)" in said, said


# ------------------------------------------------- the message as the graph has it
def _live(words: str, block: str = "single", **renames) -> str:
    """*words* as the web chat hands them to the graph: the stored run context
    appended (augmented_human_message), its column names optionally renamed."""
    context = LIVE["context_blocks"][block]
    for old, new in renames.items():
        context = context.replace(old, new)
    return words + context


def _classified(text: str):
    from langchain_core.messages import HumanMessage

    from app.agents.avaloka_agent.intent_classifier import classify_intent
    return classify_intent([HumanMessage(content=text)])


def test_the_classifier_reads_the_users_words_not_the_run_context(monkeypatch):
    """rc18 live check: "hi" in a session with a dataset routed to
    clarification, "a greeting followed by a request", because the run
    context appended to the message was read as the user's words."""
    monkeypatch.delenv("AVALOKA_LLM_INTENT_CLASSIFY", raising=False)
    for block in ("single", "multi"):
        hi = _classified(_live("hi", block))
        assert hi.intent == "chit_chat", (block, hi)
        assert _classified(_live("hey there", block)).intent == "chit_chat"
        assert _classified(_live("who are you?", block)).intent == "chit_chat"
        # ...and the other direction still holds on the live shape.
        for words in ("hey, what's the average age?", "who are your best customers"):
            result = _classified(_live(words, block))
            assert result.intent != "chit_chat" and result.should_delegate(), (words, block, result)
        assert _classified(_live("hello, can you chart sales", block)).intent == "visualization"


def test_a_column_name_in_the_run_context_is_not_a_keyword(monkeypatch):
    """Every keyword rule read the context block too: a column named
    ``schedule_date`` or ``predicted_value`` routed the question."""
    monkeypatch.delenv("AVALOKA_LLM_INTENT_CLASSIFY", raising=False)
    renamed = {"order_date": "schedule_date", "amount": "predicted_value", "returned": "chart_type",
               "region": "etl_batch"}
    text = _live("What is the average order value by segment?", "multi", **renamed)
    assert "schedule_date" in text and "predicted_value" in text
    result = _classified(text)
    assert result.intent == "clarification" and result.should_delegate(), result
    assert _classified(_live("hi", "multi", **renamed)).intent == "chit_chat"
    # The user's own keyword still decides.
    assert _classified(_live("chart revenue by region", "multi", **renamed)).intent == "visualization"


def test_a_share_that_is_not_all_rows_is_never_printed_as_100_percent():
    """rweb17 taxi upload: `store_and_fwd_flag` is "N" on 199,391 of 200,145
    counted rows (0.9962); the card said "(100% of rows)", hiding the "Y" rows."""
    case = LIVE["taxi_upload_flag"]
    assert "leans on “N” (100% of rows)" in case["shown"]
    prompts = chat_voice.prompts_after_upload(sample_statistics=case["sample_statistics"],
                                              filename="taxi_rowgroups.parquet", total_rows=200_000)
    said = " ".join(_plain(prompts))
    assert "100%" not in said, said
    assert "`store_and_fwd_flag` leans on “N” (99.6% of rows)" in said, said
    # Every share the voice prints goes through one formatter: exact ends stay
    # whole, near ends keep a decimal or say which side they are on.
    assert [chat_voice._pct(r) for r in (1.0, 0.0, 0.9962, 0.0038, 0.9996, 0.0004, 0.33, 0.12)] == [
        "100%", "0%", "99.6%", "0.4%", "over 99.9%", "under 0.1%", "33%", "12%"]


def test_the_verification_card_states_the_count_and_no_verdict():
    """Owner's ruling (2.0): "I checked N claims in my answer against the
    numbers; all of them hold." becomes the count alone. A failed claim is
    still said and flagged."""
    assert "I checked 5 claims in my answer against the numbers; all of them hold." in LIVE["recipes_chat"]["shown"]
    v = _step(_turn("recipes_chat"), "verification")
    assert v["message"] == "I checked 5 claims in my answer against the numbers."
    # The scoped variant (related ask_seg): the count, then the scope line.
    v = _step(_turn("related_ask_seg"), "verification")
    assert v["message"] == ("I checked 5 claims in my answer against the numbers. They are figures of a filtered "
                            "subset, not of every row; the answer now says what was left out.")
    # One claim, and a withheld one with nothing failed: still no verdict.
    v = _step(_turn("recipes_chat", verification_report={"safe_to_present": False, "claims_checked": 1,
                                                         "n_problems": 0, "analysis": "answered"}), "verification")
    assert v["message"] == "I checked 1 claim in my answer against the numbers."
    # The failed variant is unchanged.
    v = _step(_turn("related_ask_tables"), "verification")
    assert v["message"] == ("I checked 1 claim in my answer against the numbers; 1 did not hold and I've flagged it. "
                            "I've held back the statements the numbers don't support.")
