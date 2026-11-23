"""DataAgentBench phase 2: four ways the mission lost answers it had.

Each case is from the rc7 records (benchmarks/results/2.0-rc7/dab on
bench/2.0-rc7-dab, docs/benchmarks/2.0-rc7-dab.md):

(a) the answer review demanded a number from label questions -- a stage
    name, BANT factors, a month, a state code (crmarenapro:1, :3, :4, :9);
(b) gpt-oss-120b sent its answer to a tool that does not exist ("commentary",
    "json") and Groq refused the call, 10 times out of 11 rejections;
(c) context_length_exceeded with every oversized result already capped at
    100 KB: three such results fill gpt-oss's 131k-token window;
(d) the third repeat of a call ended the mission ("three-strikes", 8 of
    gpt-oss's give-ups) instead of asking for a different approach.

Hermetic: scripted providers, fake tools, the real mission. Every test here
uses names that already existed before the fix, so a missing fix shows up as
a failed assertion, not an import or attribute error.
"""

from __future__ import annotations

import copy
import json

import pytest

from avaloka.data_agent import (DataMissionRequest, DataQueryMission, MissionPolicy,
                                ModelResponse, ModelToolCall, ModelUsage,
                                ToolCallRejected, ToolObservation,
                                build_question_plan, review_answer)


class ScriptedProvider:
    """Each script item is a ModelResponse, or a str: the failed_generation of
    a ToolCallRejected raised the way the DAB provider raises Groq's 400."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def complete(self, messages, tools, *, timeout_s):
        self.calls.append(copy.deepcopy(messages))
        item = self.script.pop(0)
        if isinstance(item, str):
            raise ToolCallRejected(
                "provider rejected the model's tool call: BadRequestError: "
                "Error code: 400 - {'error': {'code': 'tool_use_failed'}}",
                failed_generation=item,
            )
        return item


class RegistryTools:
    """Like the DAB registry: unknown names fail, scripted observations by query."""

    def __init__(self, names=("query_db", "execute_python"), observations=None):
        self.names = tuple(names)
        self.observations = dict(observations or {})
        self.calls = []

    def specs(self):
        return [{"type": "function",
                 "function": {"name": name, "parameters": {"type": "object"}}}
                for name in self.names]

    def execute(self, name, arguments, *, call_id):
        self.calls.append((name, arguments, call_id))
        if name not in self.names:
            return ToolObservation(False, f"unknown tool: {name}")
        observation = self.observations.get(json.dumps(arguments, sort_keys=True))
        if observation is not None:
            return observation
        return ToolObservation(True, f'result_key=var_{call_id}; result=[{{"n":7}}]',
                               result_key=f"var_{call_id}", rows_read=1, bytes_read=8)

    def close(self):
        pass


def _response(call_id, name, arguments):
    return ModelResponse(response_id=f"response-{call_id}", content=None,
                         tool_calls=[ModelToolCall(call_id, name, arguments)],
                         usage=ModelUsage(prompt_tokens=8, completion_tokens=2,
                                          total_tokens=10))


def _request(question="What is n?", **kwargs):
    return DataMissionRequest(task_id="dab:fixture:1", question=question,
                              database_description="Read-only fixture",
                              model="fake-model", **kwargs)


def _policy(**overrides):
    # Fields this phase added are dropped on a tree without them, so a missing
    # fix fails on the behaviour assertion rather than a TypeError here.
    known = MissionPolicy.__dataclass_fields__
    return MissionPolicy(**{k: v for k, v in overrides.items() if k in known})


def _kinds(result, kind):
    return [event for event in result.trajectory if event.get("kind") == kind]


# --- (a) a label answer is valid when the question asks for a label ---------

LABEL_QUESTIONS = {
    # The four rc7 questions the numeric finding fired on (Haiku 5.5).
    "crmarenapro:1": (
        "Can this lead be qualified based on the latest discussions? If the answer "
        "is no, which factors—'Budget', 'Authority', 'Need', or 'Timeline'—are "
        "responsible? Return only one or several of the four BANT factors that the "
        "lead qualification fails to meet (i.e. 'Budget', 'Authority', 'Need', "
        "'Timeline')."
    ),
    "crmarenapro:3": (
        "Is the stage name accurately representing the tasks for this opportunity? "
        "If it is not, what should the appropriate stage name be? Return only the "
        "correct stage label among ('Qualification', 'Discovery', 'Quote', "
        "'Negotiation', 'Closed')."
    ),
    "crmarenapro:4": (
        "Is there a particular month in the past 10 months where the number of "
        "SecureAnalytics Pro cases significantly exceeds those of other months? "
        "Return only the month name."
    ),
    "crmarenapro:9": (
        "Which states have the quickest case closure time in the past 6 quarters? "
        "Return only the two-letter abbreviation of the most matching state (eg. CA)."
    ),
    "which-with-numeric-qualifier": (
        "Which store earned the most revenue in USD from the song across all countries?"
    ),
    "what-category": (
        "What category has the highest average rating among businesses in 2016?"
    ),
    "who": "Who had the quickest average turnaround among agents in April 2023?",
}


@pytest.mark.parametrize("name", sorted(LABEL_QUESTIONS))
def test_a_label_question_does_not_require_a_numeric_answer(name):
    plan = build_question_plan(LABEL_QUESTIONS[name])

    assert plan.numeric_result_required is False
    verdict = review_answer(plan, "Negotiation", evidence_refs=["q1"],
                            successful_evidence=1)
    assert verdict.status == "pass", verdict.findings


@pytest.mark.parametrize("question", [
    "How many commit messages are found in repositories that use Shell?",
    "What is the average rating of all businesses located in Indianapolis, Indiana?",
    "What fraction of all articles authored by Amy Jones belong to Science?",
    # A label question that also asks for a number keeps the requirement.
    "Which U.S. state has the highest number of businesses that offer WiFi, and "
    "what is the average rating for those businesses?",
    "Which massage businesses have an average rating of at least 4.0, and what "
    "are their respective average ratings?",
    "Calculate the chi-square statistic for histological types and CDH1 mutations.",
    "What is the overall return of the index since 2000?",
])
def test_an_explicit_numeric_question_still_requires_a_number(question):
    plan = build_question_plan(question)

    assert plan.numeric_result_required is True
    verdict = review_answer(plan, "Pennsylvania", evidence_refs=["q1"],
                            successful_evidence=1)
    assert verdict.status == "repair"


def test_the_imperative_return_is_not_the_financial_noun():
    plan = build_question_plan(
        "Identify the agent with the fewest transfer counts. Return only the Id."
    )
    assert plan.numeric_result_required is False


def test_a_label_answer_to_a_label_question_is_returned_not_discarded():
    # rc7 Haiku crmarenapro:3: two review repairs, then review_failed and no answer.
    provider = ScriptedProvider([
        _response("q1", "query_db", {"query": "select stage"}),
        _response("a1", "return_answer", {"answer": "Negotiation"}),
        _response("a2", "return_answer", {"answer": "Negotiation"}),
    ])

    result = DataQueryMission(provider, RegistryTools()).run(
        _request(LABEL_QUESTIONS["crmarenapro:3"]))

    assert result.answer == "Negotiation"
    assert result.termination_reason == "return_answer"
    assert result.review_repairs == 0


# --- (b) an answer sent to a tool that does not exist -----------------------

@pytest.mark.parametrize("alias", ["final_answer", "submit", "submit_answer",
                                   "answer", "commentary", "json"])
def test_an_answer_sent_to_a_nonexistent_tool_is_the_answer(alias):
    # A provider that passes unknown tool names through (OpenRouter, Ollama).
    provider = ScriptedProvider([
        _response("q1", "query_db", {"query": "select count(*)"}),
        _response("a1", alias, {"answer": "1077", "evidence_refs": ["q1"]}),
    ])
    tools = RegistryTools()

    result = DataQueryMission(provider, tools).run(_request())

    assert result.answer == "1077"
    assert result.termination_reason == "return_answer"
    assert [name for name, _, _ in tools.calls] == ["query_db"]
    assert [(e["tool"], e["path"]) for e in _kinds(result, "answer_tool_alias")] \
        == [(alias, "tool_call")]


@pytest.mark.parametrize("alias", ["commentary", "json"])
def test_an_answer_in_a_provider_rejected_phantom_call_is_the_answer(alias):
    # rc7 gpt-oss music_brainz_20k:2 on Groq: 400 tool_use_failed, ground truth lost.
    generation = json.dumps({"name": alias, "arguments": {
        "answer": "Amazon Music", "evidence_refs": ["q1"]}})
    provider = ScriptedProvider([
        _response("q1", "query_db", {"query": "select store"}),
        generation,
    ])

    result = DataQueryMission(provider, RegistryTools()).run(_request())

    assert result.answer == "Amazon Music"
    assert result.termination_reason == "return_answer"
    assert result.error is None
    assert [(e["tool"], e["path"]) for e in _kinds(result, "answer_tool_alias")] \
        == [(alias, "provider_rejection")]
    # The answer is replayed to the model as a return_answer call, so a review
    # repair (if any) would have a call id to answer.
    assert not _kinds(result, "tool_call_rejected")


def test_a_recovered_answer_still_needs_evidence_and_review():
    generation = json.dumps({"name": "final_answer", "arguments": {"answer": "7"}})
    provider = ScriptedProvider([
        generation,                                    # no evidence yet: refused
        _response("q1", "query_db", {"query": "select n"}),
        _response("a1", "return_answer", {"answer": "7"}),
    ])

    result = DataQueryMission(provider, RegistryTools()).run(_request())

    assert result.answer == "7"
    assert "No successful data-tool evidence" in provider.calls[1][-1]["content"]


@pytest.mark.parametrize("generation", [
    # rc7 PATENTS:2: not an answer at all.
    '{"name": "repo_browser.search", "arguments": {"path": "", "query": "DE-"}}',
    # An alias name carrying a query is a tool call, not an answer.
    '{"name": "json", "arguments": {"query": "select count(*) from t"}}',
    # Answer plus anything else is not answer-shaped.
    '{"name": "commentary", "arguments": {"answer": "7", "query": "select 7"}}',
    # Answer-shaped arguments under a name outside the accepted set.
    '{"name": "python", "arguments": {"answer": "7"}}',
    '{"name": "final_answer", "arguments": {"answer": "   "}}',
    "not json at all",
])
def test_junk_in_a_rejected_call_is_not_accepted_as_an_answer(generation):
    provider = ScriptedProvider([
        _response("q1", "query_db", {"query": "select n"}),
        generation,
        _response("a1", "return_answer", {"answer": "7"}),
    ])

    result = DataQueryMission(provider, RegistryTools()).run(_request())

    assert result.answer == "7"                     # from the real return_answer
    assert not _kinds(result, "answer_tool_alias")
    assert [e["count"] for e in _kinds(result, "tool_call_rejected")] == [1]


def test_a_real_tool_that_shares_an_alias_name_is_still_executed():
    provider = ScriptedProvider([
        _response("q1", "answer", {"answer": "lookup"}),
        _response("a1", "return_answer", {"answer": "7"}),
    ])
    tools = RegistryTools(names=("answer",))

    result = DataQueryMission(provider, tools).run(_request())

    assert [name for name, _, _ in tools.calls] == ["answer"]
    assert result.answer == "7"
    assert not _kinds(result, "answer_tool_alias")


# --- (c) tool output trimmed so the prompt cannot overflow ------------------

def _tool_bytes(messages):
    return sum(len(m["content"].encode("utf-8"))
               for m in messages if m.get("role") == "tool")


def _oversized_run(n_results, size, policy):
    observations = {
        json.dumps({"query": f"q{i}"}): ToolObservation(
            True, f"HEAD{i}:" + ("x" * size) + f":TAIL{i}",
            result_key=f"var_q{i}", rows_read=1, bytes_read=size)
        for i in range(n_results)
    }
    provider = ScriptedProvider(
        [_response(f"q{i}", "query_db", {"query": f"q{i}"}) for i in range(n_results)]
        + [_response("a1", "return_answer", {"answer": "done"})]
    )
    result = DataQueryMission(provider, RegistryTools(observations=observations)).run(
        _request(policy=policy))
    return provider, result


def test_oversized_results_never_overflow_the_default_prompt_bound():
    # rc7 DEPS_DEV_V1:2: 42 MB, 10 MB, 5 MB and 71 MB results, each already cut
    # to 100 KB, took the prompt to 113k tokens and the next call overflowed.
    provider, result = _oversized_run(5, 400_000, MissionPolicy())

    assert result.answer == "done"
    assert max(_tool_bytes(call) for call in provider.calls) <= 240_000
    newest = [m for m in provider.calls[-1] if m.get("role") == "tool"][-1]["content"]
    assert "HEAD4:" in newest and ":TAIL4" in newest      # head and tail kept
    assert "truncated" in newest                           # and how much was cut
    compacted = _kinds(result, "tool_history_compacted")
    assert compacted and all(e["bytes_after"] <= 240_000 for e in compacted)


def test_the_total_bound_is_configurable_and_says_what_was_cut():
    provider, result = _oversized_run(4, 30_000, _policy(
        max_prompt_tool_result_bytes=20_000,
        max_prompt_tool_results_total_bytes=30_000,
        compacted_tool_result_bytes=1_000,
    ))

    assert result.answer == "done"
    assert max(_tool_bytes(call) for call in provider.calls) <= 30_000
    oldest = [m for m in provider.calls[-1] if m.get("role") == "tool"][0]["content"]
    assert len(oldest.encode("utf-8")) <= 1_000
    assert "compacted" in oldest and "var_q0" in oldest   # where the full result is
    assert "truncated" in oldest and "HEAD0:" in oldest


def test_a_single_oversized_result_is_cut_to_head_and_tail_with_a_note():
    provider, _ = _oversized_run(1, 10_000, MissionPolicy(max_prompt_tool_result_bytes=500))

    message = [m for m in provider.calls[1] if m.get("role") == "tool"][0]["content"]
    assert "HEAD0:" in message and ":TAIL0" in message
    assert "truncated" in message
    assert len(message.encode("utf-8")) <= 500 + len("SUCCESS: ")


def test_the_dab_harness_reads_the_bounds_from_env_and_request(monkeypatch):
    from scripts.avaloka_dab_agent import _mission_policy

    monkeypatch.setenv("AVALOKA_DAB_MAX_TOOL_RESULTS_TOTAL_BYTES", "50000")
    monkeypatch.delenv("AVALOKA_DAB_MAX_TOOL_RESULT_BYTES", raising=False)
    monkeypatch.delenv("AVALOKA_DAB_MAX_STRATEGY_SWITCHES", raising=False)
    policy = _mission_policy({"max_prompt_tool_result_bytes": 12_000,
                              "max_strategy_switches": 2})

    assert policy.max_prompt_tool_results_total_bytes == 50_000
    assert policy.max_prompt_tool_result_bytes == 12_000
    assert policy.max_strategy_switches == 2


# --- (d) after three repeats, a second strategy instead of giving up --------

FAILING = {json.dumps({"query": "select bad_col"}): ToolObservation(
    False, 'column "bad_col" does not exist')}


def test_a_third_repeat_switches_strategy_instead_of_ending():
    bad = {"query": "select bad_col"}
    provider = ScriptedProvider([
        _response("r1", "query_db", bad),
        _response("r2", "query_db", bad),
        _response("r3", "query_db", bad),                   # third: switch
        _response("alt", "execute_python", {"code": "print(7)"}),
        _response("a1", "return_answer", {"answer": "7"}),
    ])
    tools = RegistryTools(observations=FAILING)

    result = DataQueryMission(provider, tools).run(_request(
        policy=MissionPolicy(max_iterations=10, repeated_call_limit=3)))

    assert result.answer == "7"
    assert result.termination_reason == "return_answer"
    # The repeated call was blocked, not run a third time.
    assert [c[2] for c in tools.calls] == ["r1", "r2", "alt"]
    assert [e["tool_call_id"] for e in _kinds(result, "strategy_switch")] == ["r3"]
    # What the model saw: the blocked call answered, then the re-plan prompt
    # with the error summary and the other route.
    after_switch = provider.calls[3]
    assert after_switch[-2]["role"] == "tool"
    assert after_switch[-2]["tool_call_id"] == "r3"
    assert after_switch[-2]["content"].startswith("FAILED: not run")
    switch = after_switch[-1]
    assert switch["role"] == "user" and switch["content"].startswith("STRATEGY SWITCH")
    assert 'column "bad_col" does not exist' in switch["content"]
    assert "execute_python" in switch["content"]


def test_repeating_after_the_switch_still_ends_the_mission():
    bad = {"query": "select bad_col"}
    provider = ScriptedProvider([_response(f"r{i}", "query_db", bad) for i in range(1, 5)])

    result = DataQueryMission(provider, RegistryTools(observations=FAILING)).run(
        _request(policy=MissionPolicy(max_iterations=10, repeated_call_limit=3)))

    assert len(_kinds(result, "strategy_switch")) == 1
    assert result.termination_reason == "repeated_tool_call"
    assert result.error_kind == "tool_error"
    assert len(provider.calls) == 4


def test_the_switch_waits_for_every_call_in_the_reply():
    # The re-plan prompt may only follow once each tool_call id has a reply.
    bad = {"query": "select bad_col"}
    double = ModelResponse(
        response_id="r3", content=None,
        tool_calls=[ModelToolCall("r3", "query_db", bad),
                    ModelToolCall("r3b", "query_db", {"query": "select n"})],
        usage=ModelUsage(prompt_tokens=8, completion_tokens=2, total_tokens=10))
    provider = ScriptedProvider([
        _response("r1", "query_db", bad), _response("r2", "query_db", bad), double,
        _response("a1", "return_answer", {"answer": "7"}),
    ])

    result = DataQueryMission(provider, RegistryTools(observations=FAILING)).run(
        _request(policy=MissionPolicy(max_iterations=10, repeated_call_limit=3)))

    assert result.answer == "7"
    tail = provider.calls[3][-3:]
    assert [m["role"] for m in tail] == ["tool", "tool", "user"]
    assert [m.get("tool_call_id") for m in tail[:2]] == ["r3", "r3b"]
    assert tail[2]["content"].startswith("STRATEGY SWITCH")


def test_a_recovered_answer_may_cite_result_keys_as_evidence():
    # Every rc7 phantom-tool answer cited "var_fc_..." result keys, not call
    # ids; refusing them as unknown would lose the recovered answer again.
    generation = json.dumps({"name": "commentary", "arguments": {
        "answer": "Amazon Music", "evidence_refs": ["var_q1"]}})
    provider = ScriptedProvider([
        _response("q1", "query_db", {"query": "select store"}),
        generation,
    ])

    result = DataQueryMission(provider, RegistryTools()).run(_request())

    assert result.answer == "Amazon Music"
    assert _kinds(result, "termination")[-1]["evidence_refs"] == ["q1"]
