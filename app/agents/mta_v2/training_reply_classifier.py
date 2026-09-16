import json
import logging
import os
import re
from typing import Any, Dict, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.agent_llm import build_agent_llm

load_dotenv()

logger = logging.getLogger(__name__)

TrainingPlanReplyAction = Literal["confirm", "update", "modify_dataset", "cancel", "unclear", "none"]

_api_key = os.environ.get("GROQ_API_KEY_PLANNING_AGENT")
# Cheap JSON classifier: reasoning stays at "low" (mechanical judgment);
# AVALOKA_MTA_CLASSIFIER_MODEL=llama-3.3-70b-versatile restores the old model.
_classifier_llm = build_agent_llm(
    agent="MTA_CLASSIFIER",
    api_key=_api_key,
    default_model="openai/gpt-oss-120b",
    temperature=0,
    default_effort="low",
)


def clean_training_reply_text(text: str) -> str:
    return str(text or "").split("\n\n[Analysis context]", 1)[0].strip()


def _extract_json_object(content: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in classifier response")
    return json.loads(match.group(0))


def classify_training_plan_reply(
    text: str,
    has_pending_plan: bool,
    training_completed: bool,
) -> TrainingPlanReplyAction:
    if not has_pending_plan or training_completed:
        return "none"

    cleaned = clean_training_reply_text(text)
    if not cleaned:
        return "unclear"

    if _classifier_llm is None:
        logger.warning("Training reply classifier LLM unavailable; defaulting to unclear.")
        return "unclear"

    system_prompt = (
        "Classify the user's reply to a pending machine-learning training plan.\n"
        "Return only JSON with this exact schema:\n"
        "{\n"
        '  "action": "confirm" | "update" | "modify_dataset" | "cancel" | "unclear" | "none",\n'
        '  "confidence": number,\n'
        '  "reason": string\n'
        "}\n\n"
        "Definitions:\n"
        "- confirm: the user clearly wants to start training with the current plan as-is.\n"
        "- update: the user gives enough concrete detail to revise at least one part of the plan.\n"
        "- modify_dataset: the user wants to leave the training-plan flow to transform, clean, filter, join, edit, or otherwise modify the dataset before training.\n"
        "- cancel: the user wants to stop, cancel, discard, or exit the pending training flow with no further training-plan work.\n"
        "- unclear: the reply does not clearly choose between starting training and changing the plan.\n\n"
        "- none: the user is not replying to the pending plan; they are making a fresh model-training request or asking for a new/initial training plan.\n\n"
        "Safety rules:\n"
        "- If the reply describes a complete new training task or asks for a new/initial training plan before training, classify it as none.\n"
        "- If the reply contains both approval and a requested change, classify it as update.\n"
        "- Classify as update only when the reply identifies what should change in the plan.\n"
        "- If the reply asks to alter the actual data before training, classify it as modify_dataset.\n"
        "- If the reply asks to stop or cancel training, classify it as cancel.\n"
        "- If the reply says the user wants a change but does not specify what should change, classify it as unclear.\n"
        "- If the reply is ambiguous, classify it as unclear.\n"
        "- Never classify as confirm unless the user clearly wants training to start now."
    )
    try:
        response = _classifier_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User reply:\n{cleaned}"),
            ],
            tools=[],
            tool_choice="none",
        )
        parsed = _extract_json_object(response.content)
        action = str(parsed.get("action", "")).strip().lower()
        if action in {"confirm", "update", "modify_dataset", "cancel", "unclear", "none"}:
            logger.info(
                "Training reply classified as %s (confidence=%s, reason=%s)",
                action,
                parsed.get("confidence"),
                parsed.get("reason"),
            )
            return action  # type: ignore[return-value]
        logger.warning("Unexpected training reply action from classifier: %s", action)
    except Exception as exc:
        logger.warning("Training reply classification failed; defaulting to unclear: %s", exc)

    return "unclear"
