"""
Summarizer Agent for ETL Job Planning

This agent analyzes conversation history and creates structured ETL job definitions
using Pydantic models for validation and JSON schema generation.
"""

import json
import logging
import os
import re
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from app.core.inference import build_chat_model
from app.graph.etl_state import ETLState
from app.utils import extract_json_block

logger = logging.getLogger(__name__)

from app.core.model_config import resolve as resolve_model
from app.core.model_fallback import attach_fallback
from app.core.log_utils import describe_response, preview

_summarizer_api_key = os.environ.get("GROQ_API_KEY_CODING_AGENT")
summarizer_llm = build_chat_model(
    role="coding",
    agent="SUMMARIZER",
    tier="small",
    temperature=0,
    groq_model=resolve_model("summarizer"),
    groq_api_key=_summarizer_api_key,
)
if summarizer_llm is None:
    logger.warning("Summarizer LLM disabled; set GROQ_API_KEY_CODING_AGENT to re-enable remote generation.")


# Pydantic models for schema validation
class Transformation(BaseModel):
    name: str = Field(..., description="e.g., 'Group by Location'")
    type: str = Field(..., description="e.g., 'aggregation', 'filter', 'cleaning'")
    columns: List[str] = Field(..., description="List of columns to apply the transformation on")
    aggregation: Optional[str] = Field(None,
                                       description="The aggregation function, e.g., 'sum', 'count'. Required if type is 'aggregation'")


class ETLJob(BaseModel):
    """A structured representation of an ETL job."""
    job_name: Optional[str] = Field(None, description="A descriptive name for the ETL job.")
    source: Optional[str] = Field(None, description="The data source, e.g., a file path, database table, or API endpoint.")
    transformations: Optional[List[Transformation]] = Field(None,
                                                  description="A list of transformations to be applied to the data.")
    destination: Optional[str] = Field(None, description="The destination for the transformed data.")
    schedule: Optional[str] = Field(None, description="The schedule for running the job, e.g., 'daily at 2am', 'every hour'.")
    notes: Optional[str] = Field(None, description="Any additional notes or comments about the ETL job.")


def _create_summary_prompt(messages: list, error_message: Optional[str] = None):
    """
    Creates a consolidated prompt from the conversation history.
    """
    # Build a single string from the conversation to act as the core input.
    conversation_text = ""
    for msg in messages:
        if isinstance(msg, AIMessage):
            conversation_text += f"\nAI: {msg.content}\n"
        elif isinstance(msg, HumanMessage):
            conversation_text += f"\nHuman: {msg.content}\n"

    # Corrected line for Pydantic v2: get schema dict, then dump to string
    schema_dict = ETLJob.model_json_schema()
    schema_json_str = json.dumps(schema_dict, indent=2)

    system_prompt = (
        "Based on the following conversation, output a valid JSON string that "
        "summarizes the ETL job. The JSON **must** adhere to this exact schema:\n\n"
        f"```json\n{schema_json_str}\n```\n\n"
        "The output **must** be given in the format:\n\n"
        "```json\n{name: value, name: value, ...}\n```\n\n"
        "**Important Rules:**\n"
        "- Output ONLY a valid JSON block. No explanation, markdown, or intro.\n"
        "- The output must not be enclosed in a markdown block.\n"
        "- For optional fields like 'aggregation' or 'notes', use `null` if the value is not present.\n"
        "- All keys and string values must be enclosed in double quotes.\n"
        "- Summarize the transformation with few, but precise words.\n"
        f"{'You failed to generate valid JSON last time. The error was: ' + error_message if error_message else ''}\n"
    )

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=conversation_text)
    ])


def summarize_etl_job(state: ETLState) -> ETLState:
    """
    Summarize ETL job from conversation history with retry mechanism
    
    Args:
        state: Current ETL state
        
    Returns:
        Updated ETL state with job definition
    """
    if state.get("ready_to_code", False):
        # Already processed, return as-is
        state.update(
            messages=state["messages"],
            planner_definition=state["planner_definition"],
            ready_to_summarize=True,
            ready_to_code=True,
            coder_definition={}
        )
        return state

    messages = state.get("messages", [])

    max_retries = 2
    error_message = None

    if summarizer_llm is None:
        logger.info("Skipping summarization because GROQ_API_KEY_CODING_AGENT is not configured.")
        ai_response = AIMessage(
            content="Summary not generated because LLM access is unavailable in this environment."
        )
        state.update(
            messages=state["messages"] + [ai_response],
            planner_definition=state.get("planner_definition"),
            ready_to_summarize=True,
            ready_to_code=False,
            coder_definition={}
        )
        return state

    for attempt in range(max_retries):
        try:
            # Create a fresh, consolidated prompt for each attempt
            prompt = _create_summary_prompt(messages, error_message)
            chain = prompt | summarizer_llm

            response = chain.invoke({})
            raw_response = response.content.strip()
            logger.info("Summarizer LLM answered (attempt %s): %s",
                        attempt + 1, describe_response(response))
            logger.debug("Summarizer raw response (attempt %s): %s",
                         attempt + 1, preview(raw_response, 2000))

            json_str = extract_json_block(raw_response)

            if not json_str:
                raise ValueError("No valid JSON block found.")

            parsed = json.loads(json_str)
            etl_job = ETLJob.model_validate(parsed)
            validated_data = etl_job.model_dump()
            pretty = json.dumps(validated_data, indent=2)

            ai_response = AIMessage(content=f"Here is the JSON summary of your ETL job:\n```json\n{pretty}\n```")

            state.update(
                messages=state["messages"] + [ai_response],
                planner_definition=validated_data,
                ready_to_summarize=True,
                ready_to_code=True,
                coder_definition={}
            )

            return state

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.error(f"Validation failed on attempt {attempt + 1}: {e}")
            error_message = str(e)
            if attempt == max_retries - 1:
                ai_response = AIMessage(
                    content=f"⚠️ Failed to generate a valid ETL job summary after {max_retries} attempts.\n"
                            f"**Error:** `{error_message}`\n\nRaw LLM output:\n```\n{raw_response}\n```"
                )

                state.update(
                    messages=state["messages"] + [ai_response],
                    planner_definition=state.get(
                        "planner_definition",
                        ETLJob(
                            job_name=None,
                            source=None,
                            transformations=None,
                            destination=None,
                            schedule=None,
                            notes=None
                        ).model_dump()
                    ),
                    ready_to_summarize=True,
                    ready_to_code=False,
                    coder_definition={}
                )
                return state

    return state
