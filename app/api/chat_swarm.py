"""The swarm from the chat: Avaloka clones herself, then tells you what the team did.

The live /swarm page is Professional and Enterprise (app/api/swarm_gate.py).
The swarm itself is everyone's, so the chat is where an open-source or Free
user runs one: a message that asks for it ("swarm this", "send your team",
"clone yourself") starts a coordinated mission on the thread's dataset through
the same service the console uses -- :mod:`app.trace.swarm_service`, capped by
the caller's edition -- waits for the clones, and answers with the team's
story told back from the mission's own trace records by
:mod:`avaloka.swarm_narration` at the team-story level. Nothing in the message
is canned: a line that is not in the records is not in the story.

In editions that have the console the message also links to the live page; in
oss/free it does not mention a page the caller cannot open.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.api.swarm_gate import CONSOLE_PAGE
from app.api.ui_contract import assistant_prompt
from app.core.editions import GateReason, swarm_console_gate

logger = logging.getLogger(__name__)

#: A message that asks for the team. Matched on the user's words only; the
#: UI's quick reply ("Swarm this: <goal>") matches too.
TRIGGER_RE = re.compile(
    r"\b(swarm|clone (yourself|herself|you)|send (in |out )?(the|your|my|a) team|"
    r"fan[ -]?out|split (it|this|the data) (up )?(across|among|between)|"
    r"parallel copies|sub[- ]?avalokas?)\b",
    re.I,
)
_LEAD_IN_RE = re.compile(
    r"^\s*(?:please\s+)?(?:swarm (?:this|it|the data)?|send (?:in |out )?(?:the|your|my|a) team(?: at it)?|"
    r"clone (?:yourself|herself|you)|fan[ -]?out)\s*[:,\-—]?\s*", re.I)

#: Asking after the team, not for a new one: reports on a pending mission and
#: otherwise says so, rather than spending another run.
STATUS_RE = re.compile(
    r"^\s*(?:check (?:on|in on) the (?:swarm|team)|how(?:'s| is| are) (?:the )?(?:swarm|team|clones)(?: doing| going)?|"
    r"(?:swarm|team) (?:status|progress|update))\s*[?.!]*\s*$", re.I)

DEFAULT_TIMEOUT_S = float(os.getenv("AVALOKA_CHAT_SWARM_TIMEOUT_S", "60"))


def wants_swarm(content: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> bool:
    """True when the user asked for the team, in words or via ``metadata.swarm``."""
    if isinstance(metadata, dict) and metadata.get("swarm") is True:
        return True
    return bool(TRIGGER_RE.search(content or "")) or is_status_ask(content)


def is_status_ask(content: Optional[str]) -> bool:
    """"Check on the swarm": tell what is pending, never start a mission for it."""
    return bool(STATUS_RE.match(content or ""))


NOTHING_PENDING = ("Nothing is out right now — the last team report is above. Say “swarm this: "
                   "<your question>” and I'll send them again.")


def mission_goal(content: Optional[str], fallback: str = "understand this dataset") -> str:
    """The goal the team is given: the user's words minus the request itself."""
    text = (content or "").strip()
    stripped = _LEAD_IN_RE.sub("", text, count=1).strip().rstrip("?.! ")
    if stripped and not TRIGGER_RE.fullmatch(stripped):
        return stripped[:200]
    return fallback


async def run_mission(
    user_id: str,
    *,
    dataset_path: str,
    goal: str,
    edition: Optional[str],
    kind: str = "analyze",
    target: Optional[str] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Dict[str, Any]:
    """Start the mission on ``dataset_path`` and wait for the clones.

    Returns the start document plus ``records`` (whatever the trace holds when
    the wait ends) and ``finished``. The mission keeps running past a timeout;
    the story is told from the records so far and says the team is still out.
    """
    from app.trace import swarm_service, swarm_store

    async def _resolve(_thread_id: str) -> Optional[str]:
        return dataset_path

    started = await swarm_service.start_mission(
        user_id, source={"thread_id": "chat"}, goal=goal, kind=kind, target=target,
        edition=edition, thread_resolver=_resolve,
    )
    entry = swarm_store.registry.get(started["mission_id"])
    if entry is not None and entry.thread is not None:
        await asyncio.to_thread(entry.thread.join, max(1.0, float(timeout_s)))
    records: List[Dict[str, Any]] = list(entry.recorder.records) if entry is not None else []
    started["records"] = records
    started["finished"] = bool(entry is not None and entry.finished)
    started["status"] = entry.status if entry is not None else started.get("status")
    return started


async def records_for(user_id: str, mission_id: str) -> Optional[Dict[str, Any]]:
    """A mission started earlier from this chat: its records and whether it is
    done, or ``None`` when nothing serves it any more (restart without Redis)."""
    from app.trace import swarm_service

    page = await swarm_service.records_after(user_id, mission_id, after=0, limit=swarm_service.MAX_RECORDS_PAGE)
    if page is None:
        return None
    return {"mission_id": mission_id, "records": page["records"], "finished": bool(page["done"]),
            "status": page.get("status")}


SESSION_KEY = "chat_swarm"
_TOLD_KEPT = 20


def _state(session: Dict[str, Any]) -> Dict[str, List[str]]:
    state = session.get(SESSION_KEY)
    if not isinstance(state, dict):
        state = {}
    return {"pending": [str(m) for m in (state.get("pending") or [])],
            "told": [str(m) for m in (state.get("told") or [])]}


def pending_missions(session: Dict[str, Any]) -> List[str]:
    """Missions that ran for this thread's dataset -- started from the chat,
    the /swarm page, the planner or any other path that called
    :func:`remember` -- whose end the chat has not yet told, oldest first."""
    return _state(session)["pending"]


def pending_mission(session: Dict[str, Any]) -> Optional[str]:
    """The latest mission still to be told, for an ask that names the team."""
    pending = pending_missions(session)
    return pending[-1] if pending else None


def remember(session: Dict[str, Any], mission_id: str, *, told: bool) -> None:
    """Record a mission on the thread's session. Every path that starts a
    mission for a thread's dataset calls this (the chat does, and
    ``POST /api/swarm/missions`` with ``source.thread_id`` does); the next chat
    turn then tells the team's story. ``told`` once the finished story went out."""
    state = _state(session)
    mid = str(mission_id)
    state["pending"] = [m for m in state["pending"] if m != mid]
    state["told"] = [m for m in state["told"] if m != mid]
    if told:
        state["told"] = (state["told"] + [mid])[-_TOLD_KEPT:]
    else:
        state["pending"].append(mid)
    session[SESSION_KEY] = state


def summary_lines(records: List[Dict[str, Any]]) -> List[str]:
    """The team story at the ``-v`` level, straight from the narrator."""
    from avaloka.swarm_narration import VERBOSE, narrate

    return narrate(records, VERBOSE)


def _merged(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return next((r for r in records if r.get("type") == "swarm.merged"), {})


def _mission_failed(records: List[Dict[str, Any]]) -> bool:
    """The mission ended without its merged result (avaloka.swarm_trace)."""
    finished = next((r for r in records if r.get("type") == "swarm.finished"), None)
    if finished is not None and finished.get("status") != "done":
        return True
    return not _merged(records)


def _with_dataset_label(records: List[Dict[str, Any]], label: Optional[str]) -> List[Dict[str, Any]]:
    """The brief names the file the user knows, not the server's working copy.

    The recorder keeps the path it read (``swarm.requested.source``); the chat
    shows the upload's own name instead. Only that one field is relabelled, so
    every other line stays exactly the narrator's.
    """
    if not label:
        return records
    return [{**r, "source": label} if r.get("type") == "swarm.requested" else r for r in records]


def summary_prompt(
    records: List[Dict[str, Any]],
    *,
    edition: Optional[str],
    mission_id: Optional[str],
    finished: bool = True,
    goal: Optional[str] = None,
    dataset_label: Optional[str] = None,
) -> Dict[str, Any]:
    """The ``swarm`` step: one friendly message built from the records, one
    question about where to go next, and the console link only where the
    console exists."""
    lines = summary_lines(_with_dataset_label(records, dataset_label))
    if not lines:
        head = "I sent the team out, but the trace has nothing to tell yet."
    elif finished and _mission_failed(records):
        # A failed mission is not "back": say it failed, then how far it got.
        head = "My team's mission failed. Here's how far they got:"
    elif finished:
        head = "My team is back. Here's what they did:"
    else:
        head = "My team is still out; here's what they've done so far:"
    body = "\n".join(f"- {line}" for line in lines)
    message = f"{head}\n{body}" if body else head

    merged = _merged(records)
    # "What predicts X": the team's merged ranking as the table the chat and
    # the planner graph show for the same question (avaloka.drivers_answer).
    table = (merged.get("answer") or {}).get("table")
    if table:
        from avaloka.drivers_answer import table_markdown

        message += "\n\n" + "\n".join(table_markdown(table))
    verdict = str(merged.get("verdict") or "").upper()
    clones = [r for r in records if r.get("type") == "clone.returned"]
    replies: List[Any] = []
    if verdict:
        replies.append(f"Explain the {verdict} verdict")
    if len(clones) >= 2:
        replies.append("Compare what the clones found")
    selected = merged.get("selected_model") or {}
    if isinstance(selected, dict) and selected.get("selected"):
        replies.append(f"Tell me about {selected['selected']}")
    if not finished:
        replies.insert(0, "Check on the swarm")
    replies.append("Run it on the full data in one pass")
    question = ("Want me to explain the verdict, or compare what each clone found?" if verdict
                else "Shall I check on the team again in a moment?" if not finished
                else "Want me to go deeper on any part of that?")

    link = None
    if mission_id and swarm_console_gate(edition) is GateReason.OPEN:
        link = {"label": "Watch it on the swarm page", "href": f"{CONSOLE_PAGE}?mission={mission_id}"}
        message += "\nYou can replay every clone live on the swarm page."
    return assistant_prompt("swarm", message, question=question, quick_replies=replies[:4], link=link)


def summary_text(prompt: Dict[str, Any]) -> str:
    """The chat bubble: the message plus the question, as one assistant turn."""
    text = prompt["message"]
    if prompt.get("question"):
        text += f"\n\n{prompt['question']}"
    return text
