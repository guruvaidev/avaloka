import datetime
import re
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def extract_json_block(text):
    """
    Extract the first complete JSON object block from the LLM output.
    """
    matches = re.findall(r'(?s)\{.*\}', text)

    return matches[0] if matches else ""


def generate_filename_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")


def flatten_params(params: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """
    Flatten nested parameter dictionary for MLflow or other logging systems.
    
    Args:
        params: Nested dictionary to flatten
        prefix: Optional prefix for keys
        
    Returns:
        Flattened dictionary with string values
        
    Example:
        >>> flatten_params({"model": {"lr": 0.01, "epochs": 100}})
        {"model.lr": "0.01", "model.epochs": "100"}
    """
    flat_params = {}
    
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            flat_params.update(flatten_params(value, full_key))
        else:
            # Convert to string for MLflow
            flat_params[full_key] = str(value)
    
    return flat_params


def convert_message_dicts_to_objects(messages):
    message_objects = []
    for msg in messages:
        role = msg.get("role") or msg.get("type")
        if role == "human":
            message_objects.append(HumanMessage(content=msg["content"]))
        elif role == "ai":
            message_objects.append(AIMessage(content=msg["content"]))
        elif role == "system":
            message_objects.append(SystemMessage(content=msg["content"]))
    return message_objects

def convert_message_objects_to_dicts(messages):
    message_dicts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            message_dicts.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            message_dicts.append({"role": "ai", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            message_dicts.append({"role": "system", "content": msg.content})
    return message_dicts