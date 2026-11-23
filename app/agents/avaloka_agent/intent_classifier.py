"""Intent classification for the Avaloka conversational agent.

Optional provider-agnostic LLM round-trip using the same model tier as the
planner. Returns a
dataclass with ``intent`` and optional structured params extracted from the
user message.

If the selected provider is unavailable, falls back to a deterministic keyword classifier so
the rest of the graph still works.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.agents.avaloka_agent.prompts import (
    AVALOKA_SYSTEM_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    INTENTS,
)
from app.agents.avaloka_agent.correction import correction_facts
from app.agents.user_text import own_words
from app.core.inference import build_chat_model
from app.core.model_config import resolve as resolve_model
# The CLI's predicts/drives rule, reused rather than copied: the app may
# import avaloka (never the reverse).
from avaloka.goal_analysis import driver_target

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    intent: str
    confidence: float = 0.5
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def is_data_touching(self) -> bool:
        return self.intent in {
            "statistical_analysis",
            "data_transfer",
            "visualization",
            "ml_training",
            "ml_inference",
            "infrastructure",
            "sampling_mode",
            # "exploration" deliberately stays OUT. It already gets its own
            # discovery pass in the agent and is meant to be answered straight
            # from that -- instantly, with no pipeline. Delegating it instead
            # sends "what is in this dataset?" through the planner, which
            # starts a 70s training run to answer a question about column
            # names. The cheap path is the right one; what was broken was the
            # renderer at the end of it, not the routing (see
            # _render_direct_reply).
            "schedule",
            # A correction to the method always re-plans (owner ruling): it is
            # never answered from the results in hand.
            "correction",
        }

    def should_delegate(self) -> bool:
        """Whether the planner sub-agent should handle this turn.

        Data-touching intents always delegate. ``clarification`` also
        delegates because the existing planner has richer tool calls for
        asking clarifying questions; the avaloka_agent direct-reply path
        is reserved for clearly conversational intents.
        """
        return self.is_data_touching() or self.intent == "clarification"


#: Assigned by the heuristic only, never offered to the LLM classifier: it
#: depends on what the turn carries (verified results), not on the words alone.
#: Answered directly, like ``exploration``, but with no discovery pass.
DISCUSSION = "discussion"


_GROQ_MODEL = resolve_model("conversational")
_GROQ_TEMPERATURE = 0.0
_KEY_ENV = "GROQ_API_KEY_PLANNING_AGENT"


_llm: Optional[Any] = None


def _get_llm() -> Optional[Any]:
    global _llm
    if _llm is not None:
        return _llm
    api_key = os.environ.get(_KEY_ENV) or os.environ.get("GROQ_API_KEY")
    try:
        _llm = build_chat_model(
            role="planning",
            agent="INTENT_CLASSIFIER",
            tier="small",
            temperature=_GROQ_TEMPERATURE,
            groq_model=_GROQ_MODEL,
            groq_api_key=api_key,
        )
    except Exception as exc:
        logger.warning("avaloka_agent: ChatGroq init failed: %s", exc)
        _llm = None
    return _llm


def _last_human_message(messages: List[BaseMessage]) -> str:
    """The latest human message in the user's own words.

    The web chat appends "[Analysis context] ..." (fidelity, and per dataset
    its columns and preview rows) to the message it hands the graph. Read as
    the user's words it made "hi" a greeting followed by a request (rc18 live
    check), and a column such as ``schedule_date`` or ``predicted_value``
    could route any question to schedule or inference.
    """
    return own_words(_raw_last_human_message(messages))


def _raw_last_human_message(messages: List[BaseMessage]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            content = m.content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
                return ""
            return content or ""
    return ""


_KEYWORD_RULES = (
    ("ml_inference", (
        "predict", "score this", "run inference", "run the model on",
        "use the model", "deploy inference", "stop inference",
    )),
    ("ml_training", (
        "train a model", "training plan", "fit a model", "train model",
        "hyperparameter", "fine-tune", "fine tune", "build a classifier",
        "build a regressor", "create a training plan", "design training",
        "training strategy", "plan training",
        # The plainest way anyone asks for a model, and it was missing: the
        # table had "build a classifier"/"build a regressor" but not the far
        # commoner "build a model". Without these, "build me a model that
        # predicts revenue" matched only "predict" and went to ml_inference.
        "build a model", "build me a model", "create a model",
        "make a model", "make me a model", "build a model that",
    )),
    ("data_transfer", (
        "convert to parquet", "convert to csv", "convert to avro",
        "convert to delta", "convert to iceberg", "move to s3",
        "move to gcs", "copy to bucket", "etl", " to parquet",
        " to delta", " to iceberg", " to avro",
    )),
    ("visualization", (
        "plot ", "chart", "visualize", "histogram", "scatter",
        "bar chart", "line chart", "show me a", "draw a",
    )),
    ("statistical_analysis", (
        "correlation", "distribution", "hypothesis", "p-value",
        "outlier", "summary statistics", "describe the data",
        "feature importance", "anova", "t-test",
    )),
    ("schedule", (
        "every day", "every hour", "schedule", "reschedule", "cron", "crontab", "at 2am",
        "tomorrow at",
    )),
    ("infrastructure", (
        "provision", "use my gke", "use gke", "use eks", "switch to aws",
        "switch to gcp", "use ray cluster",
    )),
    ("status", (
        "did the training finish", "task status", "is it done",
        "what tasks are running", "list tasks", "show models",
        "list models", "what's running",
    )),
    # Sampling-mode switches are INSTRUCTIONS TO THE PIPELINE, not questions
    # about the data, and the planner already owns the machinery that honours
    # them (parse_fidelity_from_control_text / is_explicit_mode_switch_message /
    # build_mode_switch_confirmation). They used to sit in the "exploration"
    # list, which does not delegate -- so the planner never saw them and the
    # mode silently never changed. The agent's own sampling caveat tells users
    # to say exactly these words to get exact numbers, so this was a documented
    # phrase that did nothing.
    ("sampling_mode", (
        "use entire dataset", "use full dataset", "use the entire dataset",
        "use the full dataset", "switch to entire dataset",
        "switch to full dataset", "run this on the entire dataset",
        "run on the entire dataset", "use quick sample", "use quick samples",
        "switch to quick sample", "switch to quick samples",
        "use portfolio", "use portfolio sample", "use portfolio samples",
        "switch to portfolio", "switch to portfolio sample",
        "switch to portfolio samples",
    )),
    ("exploration", (
        "what's in this", "what is in this", "show me the schema",
        "list columns", "describe the columns", "sample rows",
    )),
    ("chit_chat", (
        "hello", "hi", "hey", "what is avaloka", "who are you",
        "what can you do",
    )),
)

#: Keywords that match only as whole words. A substring match routed "how do
#: *they* relate?" to chit-chat on 'hey' (rc17 web regrade), and would route
#: "Othello", "Delhi sales", "who are your best customers" and "what can you
#: download" the same way; "etl" sits inside "metlife", "cron" inside
#: "acronym" and "Cronbach", "anova" inside "Villanova". The other keywords
#: stay substrings on purpose: "predict" must still catch "predictions",
#: "chart" "charts".
_WHOLE_WORD = frozenset(
    {kw for intent, keywords in _KEYWORD_RULES if intent == "chit_chat" for kw in keywords}
    | {"etl", "cron", "crontab", "anova"}
)

#: Whole words with the inflections that keep their meaning. "unscheduled
#: maintenance" is a column to count, not a job to schedule, and "provisional
#: figures" are not infrastructure; "scheduled", "scheduler", "provisioning"
#: still are. "reschedule" is its own keyword: rescheduling a job is a
#: schedule request.
_INFLECTED = {
    "schedule": r"schedul(?:e|es|ed|er|ers|ing)",
    "reschedule": r"reschedul(?:e|es|ed|ing)",
    "provision": r"provision(?:s|ed|ing)?",
}


def _keyword_in(keyword: str, text: str) -> bool:
    if keyword in _INFLECTED:
        return re.search(rf"(?<![a-z0-9]){_INFLECTED[keyword]}(?![a-z0-9])", text) is not None
    if keyword not in _WHOLE_WORD:
        return keyword in text
    return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None


#: Words a greeting may carry and still be only a greeting: "hey there",
#: "hello Avaloka", "good morning".
_GREETING_FILLER = frozenset({
    "there", "avaloka", "ava", "again", "all", "everyone", "folks", "friend",
    "good", "morning", "afternoon", "evening", "oh", "well",
})


def _only_chit_chat(text: str) -> bool:
    """Whether *text* is nothing but chit-chat once its greeting words go.

    "hey, what's the average age?" opens with a greeting and asks about the
    data: with "hey" removed, "what's the average age" is left, so it is not
    chit-chat. What may remain is filler ("there", "Avaloka") and
    punctuation; a "?" with no words left is the chit-chat's own ("who are
    you?", "what can you do?").
    """
    rest = text
    for intent, keywords in _KEYWORD_RULES:
        if intent != "chit_chat":
            continue
        for kw in sorted(keywords, key=len, reverse=True):
            rest = re.sub(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", " ", rest)
    words = [w for w in re.findall(r"[a-z0-9']+", rest) if w not in _GREETING_FILLER]
    return not words


def asks_what_drives(text: str) -> bool:
    """"What predicts a stroke?", "what drives churn", "drivers of default":
    a question about the data's outcome, answered by analysing the data.

    The rule is the CLI's (avaloka.goal_analysis.driver_target): an explicit
    driver verb or noun followed by the outcome phrase. "What does the model
    predict for row 5" names no outcome and asks a model, so it is not one;
    "predict churn for these customers" has no question word and stays
    inference.
    """
    target = driver_target(text or "", {})
    if not target or not target.get("explicit_verb") or not target.get("phrase"):
        return False
    return "model" not in str(target.get("subject") or "").lower()


def _driver_goal_is_analysis(result: IntentResult, text: str) -> IntentResult:
    """A predicts/drives question is an analysis of the outcome. It never
    routes to inference -- there is no model to run, and "I don't have a
    trained model" answered a question the data can answer (web e2e,
    2026-10-18) -- and it is not left unclassified either."""
    if result.intent in ("ml_inference", "clarification") and asks_what_drives(text):
        return IntentResult(intent="statistical_analysis", confidence=0.7, params=dict(result.params),
                            reasoning="a what-predicts/drives question: analyse the outcome, not inference")
    return result


#: Words that ask for new work on the data -- a computation, a change to the
#: data, a re-run. A message carrying one goes to the planner however it is
#: phrased: "include refunds and recompute" needs the pipeline even when results
#: are already in hand. Prefix-matched, so "recompute", "filtering", "running"
#: and "re-run" count; whole-word at the front, so "account" is not "count".
_NEW_WORK = re.compile(
    r"(?<![a-z])(?:re-?)?(?:comput|calculat|run|count|impute|includ|exclud|filter|train|"
    r"fit|load|join|merg|aggregat|group|sort|convert|export|sav|drop|clean|appl|build|"
    r"creat|generat|updat|fix|redo|tabulat|sum up|break .{0,12}down)")


def asks_about_results(text: str) -> bool:
    """Whether *text* asks about what is already known rather than for new work.

    "Churn fell -- can you confirm?", "That's wrong, it's 42.5", "How did the
    model do?", "What's churn for the enterprise segment?" are answered from
    results in hand: confirmed, corrected, interpreted, or declined because the
    figure is not there. "Compute distinct aisles", "include refunds and
    recompute", "impute the missing values" are new work.
    """
    t = own_words(text or "").lower().strip()
    return bool(t) and _NEW_WORK.search(t) is None


def _results_question(heuristic: IntentResult, text: str, results_in_hand: bool) -> IntentResult:
    """A question about results the reply already holds is answered, not delegated.

    ``clarification`` -- no keyword matched -- delegates to the planner, which
    plans an analysis. When the turn already carries verified results (the
    reply prompt's ``<recommendation>``), a question about them has its answer
    in hand: delegating it handed "can you confirm?" to a planner that never
    sees those results, and the turn ended in a planning failure or an
    execution error. Only ``clarification`` is redirected: a keyword match (a
    chart, a model, a schedule) still goes where it went.
    """
    if (results_in_hand and heuristic.intent == "clarification"
            and asks_about_results(text)):
        return IntentResult(intent="discussion", confidence=0.6,
                            reasoning="a question about verified results already in hand")
    return heuristic


def _heuristic_intent(text: str, has_connection_event: bool) -> IntentResult:
    """Deterministic fallback used when Groq is unavailable."""
    # Only the user's own words: never the run context appended to them.
    t = own_words(text).lower()
    if not t:
        if has_connection_event:
            return IntentResult(intent="onboarding", confidence=0.9,
                                reasoning="empty message + fresh connection_event")
        return IntentResult(intent="chit_chat", confidence=0.5,
                            reasoning="empty message")

    if has_connection_event and len(t) < 12:
        return IntentResult(intent="onboarding", confidence=0.7,
                            reasoning="short message right after connection_event")

    # Longest match wins, rather than the first rule in declaration order.
    # "train a model to predict churn" contains both "predict" (ml_inference)
    # and "train a model" (ml_training); first-rule-wins sent it to INFERENCE
    # -- asking to run a model that does not exist yet -- because ml_inference
    # happens to be declared first. That is the single most common way anyone
    # phrases a training request. Preferring the longer, more specific keyword
    # decides it on evidence rather than on the order someone happened to write
    # the rules in, and keeps the next such collision from reintroducing it.
    best_intent, best_kw = "", ""
    for intent, keywords in _KEYWORD_RULES:
        for kw in keywords:
            if _keyword_in(kw, t) and len(kw) > len(best_kw):
                best_intent, best_kw = intent, kw
    if best_intent == "chit_chat" and not _only_chit_chat(t):
        # A greeting in front of a request is not the message: route on what
        # follows it, or leave it to the planner when no other keyword names it.
        best_intent, best_kw = "", ""
        for intent, keywords in _KEYWORD_RULES:
            for kw in keywords:
                if intent != "chit_chat" and _keyword_in(kw, t) and len(kw) > len(best_kw):
                    best_intent, best_kw = intent, kw
        if not best_kw:
            return _driver_goal_is_analysis(
                IntentResult(intent="clarification", confidence=0.4,
                             reasoning="a greeting followed by a request; no keyword matched the request"), t)
    if best_kw:
        return _driver_goal_is_analysis(
            IntentResult(intent=best_intent, confidence=0.6, reasoning=f"keyword match: '{best_kw}'"), t)

    return _driver_goal_is_analysis(
        IntentResult(intent="clarification", confidence=0.4, reasoning="no keyword matched"), t)


_JSON_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_RE.search(raw)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def classify_intent(
    messages: List[BaseMessage],
    *,
    has_connection_event: bool = False,
    session_summary: str = "",
    results_in_hand: bool = False,
) -> IntentResult:
    """Classify the latest human message into one of ``INTENTS``.

    ``session_summary`` is a short text snippet describing prior thread state
    so the classifier can prefer ``onboarding`` / ``status`` when relevant.
    ``results_in_hand`` says the turn carries verified results the reply can
    answer from; a question about them is then ``discussion``, which is
    answered directly (see :func:`_results_question`).
    """
    user_text = _last_human_message(messages)

    # Heuristic always wins. The keyword set covers all data-touching
    # intents and the obvious conversational ones; ``clarification`` is
    # routed to the planner sub-agent which has the richer toolset for
    # asking follow-ups. Skipping the LLM here keeps classification fast
    # and deterministic, and avoids surprising Groq calls on every chat
    # turn (the LLM still drives reply rendering in agent.py for tone).
    heuristic = _heuristic_intent(user_text, has_connection_event)
    facts = correction_facts(user_text)
    if facts is not None and heuristic.intent in ("clarification", "statistical_analysis"):
        return IntentResult(intent="correction", confidence=0.7, params={"correction": facts.as_dict()},
                            reasoning="a correction to the method: re-plan with it")
    heuristic = _results_question(heuristic, user_text, results_in_hand)

    # Optionally consult the LLM only when explicitly opted in.
    if os.getenv("AVALOKA_LLM_INTENT_CLASSIFY", "0").strip() not in ("1", "true", "True", "yes"):
        return heuristic
    if heuristic.intent != "clarification":
        return heuristic
    llm = _get_llm()
    if llm is None:
        return heuristic

    intents_csv = ", ".join(INTENTS)
    instructions = (
        f"{INTENT_CLASSIFIER_PROMPT}\n\n"
        f"Allowed intents: {intents_csv}\n\n"
        "Return ONLY a JSON object of the form:\n"
        '{"intent": "<one of the allowed intents>", '
        '"confidence": <0..1>, '
        '"params": {"...": "..."}, '
        '"reasoning": "<one short sentence>"}'
    )
    payload = (
        f"<session_context>{session_summary or 'none'}</session_context>\n"
        f"<has_connection_event>{str(bool(has_connection_event)).lower()}</has_connection_event>\n"
        f"<user_message>{user_text}</user_message>"
    )

    try:
        result = llm.invoke([
            SystemMessage(content=AVALOKA_SYSTEM_PROMPT + "\n\n" + instructions),
            HumanMessage(content=payload),
        ])
    except Exception as exc:
        logger.warning("avaloka_agent: intent LLM call failed: %s", exc)
        return _heuristic_intent(user_text, has_connection_event)

    raw = getattr(result, "content", "") or ""
    parsed = _parse_llm_json(raw if isinstance(raw, str) else str(raw))
    if not parsed or "intent" not in parsed or parsed["intent"] not in INTENTS:
        logger.info("avaloka_agent: classifier returned unparseable output, falling back")
        return _heuristic_intent(user_text, has_connection_event)

    intent_str = str(parsed["intent"])
    # Hard guard: "onboarding" only makes sense with a real connection_event,
    # otherwise the user is mid-conversation and the planner should handle it.
    if intent_str == "onboarding" and not has_connection_event:
        intent_str = "clarification"

    return _driver_goal_is_analysis(IntentResult(
        intent=intent_str,
        confidence=float(parsed.get("confidence", 0.5) or 0.5),
        params=dict(parsed.get("params") or {}),
        reasoning=str(parsed.get("reasoning") or ""),
    ), user_text)
