"""The user's own words in a turn.

The web chat appends a run-context block to every message it hands the graph
(app/api/server.py, "AUGMENT user message"): "[Analysis context] Current
fidelity: ...", then "[Context] Multiple datasets ..." with each dataset's
columns and preview rows. That block is for the planner and the coder. Read as
the user's question it asks for everything: on rc15 the filter disclosure took
the preview's ``"returned": false`` for "excluding returns" and never showed a
scope line (#588). Anything that asks what the user meant reads it here.

Dependency-free on purpose: the planner, the verifier, the narrator and the
trace all import it.
"""

from __future__ import annotations

import re
from typing import Any, List, Mapping

#: Where the appended run context starts. "[Context]" alone is the multi-dataset
#: block, which follows "[Analysis context]" but is cut at either.
CONTEXT_MARK = re.compile(r"\n\s*\[(?:analysis context|context)\]", re.IGNORECASE)


def own_words(text: Any) -> str:
    """*text* without the run context the web chat appended to it."""
    return CONTEXT_MARK.split(str(text or ""), maxsplit=1)[0].strip()


def _content(message: Any) -> str:
    """A message's text; the text parts of a multi-part content list."""
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, Mapping) else str(p) for p in content)
    return content if isinstance(content, str) else ""


def is_user_message(message: Any) -> bool:
    """A human turn, as a LangChain message or an API/checkpoint dict."""
    if isinstance(message, Mapping):
        return message.get("role") in ("user", "human") or message.get("type") == "human"
    return getattr(message, "type", None) == "human"


def user_texts(state: Mapping[str, Any]) -> List[str]:
    """Every message the user wrote in the thread, oldest first, in their own
    words; ``user_prompt`` when the thread holds none."""
    out = [own_words(_content(m)) for m in state.get("messages") or [] if is_user_message(m)]
    if not out and state.get("user_prompt"):
        out.append(own_words(state["user_prompt"]))
    return out


def last_user_text(state: Mapping[str, Any]) -> str:
    """The question this turn answers: the latest user message in the user's
    own words, else ``user_prompt``."""
    for message in reversed(list(state.get("messages") or [])):
        if is_user_message(message):
            return own_words(_content(message))
    return own_words(state.get("user_prompt"))
