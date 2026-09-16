"""Offering numbered options, and understanding the number that comes back.

Two halves of one promise. The conversational prompt tells the model to end
with a numbered list and to invite a pick — *"Say the number and I'll start"* —
so a reply of "5" has to resolve to what 5 was. Recording the offer and
resolving the pick therefore belong together, in one module both the
conversational agent and the planner can import.

They did not use to. The planner recorded the options it offered through its
``suggest_analysis`` tool, but the conversational agent — which is what a user
actually talks to, and which the prompt instructs to list options — recorded
nothing. So the common path produced a numbered list, promised it could run
one, and then received a bare "5" with no record of what had been offered.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: A message that is *entirely* a reference to a numbered option.
SUGGESTION_PICK_RE = re.compile(
    r"^\s*(?:please\s+)?(?:run|do|go\s+with|start\s+with|pick|option|number|#)?"
    r"\s*#?\s*(\d{1,2})\s*(?:please|thanks)?\s*[.!]?\s*$",
    re.IGNORECASE,
)

#: A numbered list item: "1. text", "2) text", "**3.** text", tolerating
#: bullets and bold markers the model sprinkles in.
_NUMBERED_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\*{0,2}(\d{1,2})\*{0,2}\s*[.)]\s+(.{3,})$"
)


def parse_offered_options(text: str, *, minimum: int = 2) -> List[str]:
    """Extract the numbered options a reply just offered.

    Reads the *last* contiguous run of numbered lines, because a reply often
    explains something with its own numbered steps before ending on the choices
    — and it is the closing list the user is answering.

    Requires the run to start at 1 and increase by one. A list that skips or
    restarts is more likely prose that happens to contain numbers than a menu,
    and mis-recording it would resolve "3" to something never offered, which is
    worse than not resolving it at all.
    """
    if not text:
        return []

    runs: List[List[str]] = []
    current: List[str] = []
    for line in text.splitlines():
        match = _NUMBERED_LINE_RE.match(line)
        if match and int(match.group(1)) == len(current) + 1:
            current.append(match.group(2).strip())
            continue
        if match and int(match.group(1)) == 1:
            if len(current) >= minimum:
                runs.append(current)
            current = [match.group(2).strip()]
            continue
        if current:
            if len(current) >= minimum:
                runs.append(current)
            current = []
    if len(current) >= minimum:
        runs.append(current)

    if not runs:
        return []
    return [re.sub(r"\s*\*\*\s*", "", item).strip(" *_`") for item in runs[-1]]


def resolve_suggestion_reference(user_input: str, state) -> str:
    """Expand "run 3" into the suggestion it refers to.

    The planner offers a numbered list and invites the user to pick one, so the
    number has to mean something on the next turn. Without this the reply goes
    to the LLM as the bare string "run 3", with no record of what 3 was -- the
    same failure as telling a user to type a phrase that does nothing.

    Only a message that is ENTIRELY a reference is expanded. "run 3 but only for
    the north region" is a new instruction that happens to start with a number,
    and rewriting it would discard what the user actually asked for.
    """
    chosen = pick_suggestion(user_input, state)
    return chosen if chosen is not None else user_input


def pick_suggestion(user_input: str, state) -> Optional[str]:
    """The suggestion a message refers to, or ``None`` if it refers to none.

    Separate from :func:`resolve_suggestion_reference` so a caller can tell
    "the user picked option 3" from "the user happened to type something that
    is unchanged by resolution" — the difference matters for deciding whether
    the pick has been consumed.
    """
    pending = (state or {}).get("pending_suggestions") or []
    if not pending:
        return None
    match = SUGGESTION_PICK_RE.match(user_input or "")
    if not match:
        return None
    index = int(match.group(1))
    if not 1 <= index <= len(pending):
        return None
    chosen = str(pending[index - 1]).strip()
    logger.info("Resolved suggestion reference %r -> %.80s", user_input, chosen)
    return chosen


def remember_offered_options(
    updates: Dict[str, Any], reply_text: str
) -> Dict[str, Any]:
    """Record the options a reply offered, or clear a list nothing replaced.

    Clearing matters as much as recording. ``pending_suggestions`` was written
    once and never cleared, so a list offered early in a session stayed live
    indefinitely: a "2" typed twenty turns later, meaning something else
    entirely, resolved against options the user had long forgotten. An offer is
    only good for the turn that follows it.
    """
    options = parse_offered_options(reply_text)
    updates["pending_suggestions"] = options or None
    return updates
