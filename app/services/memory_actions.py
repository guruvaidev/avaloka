# app/services/memory_actions.py
#
# Memory never replays a data-moving action into a plan.
#
# rc10: after a transfer was cancelled, the memory plane surfaced the hint
# "Transfer patients table to e2ewhgroq as d1_e" and the coder planned a
# to_sql write. rc11/rc12: the session logic signature ("Use SQL to copy and
# rename the table ...") went into the planner's csv_info, and about 1 plan in
# 4 for "average age of patients with stroke" after a transfer session
# described a transfer (TRANSFER ... TO db://..., CREATE TABLE ..._copy). The
# write guard refused them, so the user saw a false refusal. Without the memory
# context, 0 of 48 plans tripped the guard.
#
# The rule: an action that writes or moves data (transfer, copy, export,
# insert, create table, ...) is never a hint, a signature or a preference --
# whether it completed, was cancelled or was refused, and on any later turn,
# a transfer turn included ("do it again" is never inferred from memory). A
# completed transfer stays in its own records (dta_last_transfer, lineage);
# the memory plane does not record write actions at all.
#
# Two deterministic gates, no LLM:
#   * Learning: a turn that is part of a transfer -- the thread's structured
#     state has a transfer waiting on this reply (pending_clarification of
#     type "dta_transfer"), or the request itself asks to write or move data --
#     is not learned from and is not kept as a past query for the learner.
#   * Serving: every learned hint, preference and signature part that
#     describes a write is dropped before it reaches the planner or coder.
#     The signature is LLM-written free text, merged across sessions in Chroma
#     with nothing linking a sentence to the turn it came from, and rows written
#     before this fix are still stored; so it is filtered by what it says, not
#     by a record of what happened. Dataset facts (schema hints) describe the
#     data, not an action, and are kept whole.
#
# Kept free of heavy imports: the planner and coder import it.

import re
from typing import Any, Iterable, List, Mapping, Optional, Tuple

# Words, not identifiers: "\b" after each verb form keeps column names such as
# transfer_amount or export_value (a "_" continues the word) out of it, and the
# lookbehind keeps pandas' df.copy() out.
_WRITE_ACTION_RE = re.compile(
    r"""
      \btransfer(?:s|red|ring)?\b
    | (?<![.\w])cop(?:y|ies|ied|ying)\b(?!\s*\()
    | \bexport(?:s|ed|ing)?\b
    | \binsert(?:s|ed|ing|ion)?\b
    | \bmigrat(?:e|es|ed|ing|ion)\b
    | \breplicat(?:e|es|ed|ing)\b
    | \bto_sql\b
    | \bif_exists\b
    | \bcreate\s+(?:or\s+replace\s+)?(?:temp(?:orary)?\s+)?(?:table|view|database|schema)\b
    | \bcreat(?:e|es|ed|ing)\s+(?:a\s+|an\s+|the\s+)?new\s+(?:\w+\s+)?table\b
    | \b(?:drop|truncate|alter)\s+table\b
    | \bdelete\s+from\b
    | \bmerge\s+into\b
    | \bupdate\s+\w+\s+set\b
    | \brenam(?:e|es|ed|ing)\s+(?:\w+\s+){0,2}?tables?\b
    | \bwrit(?:e|es|ing|ten)\s+(?:\w+\s+){0,3}?(?:to|into)\s+(?:\w+\s+){0,2}?(?:database|db|table|warehouse|bucket)s?\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Hints this module's caller builds from the dataset's schema, not from what
# the user did.
_DATASET_FACT_PREFIXES = ("Domain insight",)

# Where a merged signature splits into the parts it was made of: sentence ends,
# line breaks, and -- because Chroma joins the stored signatures with a space
# and the LLM rarely ends one with a full stop -- a capitalised word straight
# after a lower-case one ("... exact data fidelity Use precise table ...").
_SIGNATURE_SPLIT_RE = re.compile(
    r"(?<=[.!?;])\s+|\n+|(?<=[a-z0-9)\]'\"])\s+(?=[A-Z][a-z])")


def describes_write_action(text: Any) -> bool:
    """True when the text describes writing or moving data."""
    return bool(text) and bool(_WRITE_ACTION_RE.search(str(text)))


def is_dataset_fact(hint: Any) -> bool:
    return isinstance(hint, str) and hint.startswith(_DATASET_FACT_PREFIXES)


def drop_write_actions(hints: Optional[Iterable[Any]]) -> List[Any]:
    """The hints that do not describe a write. Dataset facts are kept whole."""
    return [h for h in (hints or []) if is_dataset_fact(h) or not describes_write_action(h)]


def strip_write_actions(signature: Any) -> Optional[str]:
    """The signature without the parts that describe a write; None if nothing is left."""
    if not signature:
        return None
    text = str(signature)
    if not describes_write_action(text):
        return text
    kept, start = [], 0
    for sep in list(_SIGNATURE_SPLIT_RE.finditer(text)) + [None]:
        end = sep.start() if sep else len(text)
        part = text[start:end].strip()
        if part and not describes_write_action(part):
            kept.append(part)
        if sep:
            start = sep.end()
    return " ".join(kept) or None


def is_write_action_turn(query: Any, state: Optional[Mapping] = None) -> bool:
    """Is this turn part of a transfer or other write? From the thread's
    structured state first: a reply to a transfer that is waiting on it
    (confirm, cancel, a table name) is such a turn whatever its words."""
    pending = (state or {}).get("pending_clarification")
    if isinstance(pending, Mapping) and pending.get("type") == "dta_transfer":
        return True
    return describes_write_action(query)


def analysis_memory_context(state: Mapping) -> Tuple[List[str], Optional[str]]:
    """(memory_hints, session_logic_signature) as they may enter a plan or a
    code prompt. The memory plane already filters what it returns; this holds
    for anything else that put memory on the state."""
    hints = [h for h in drop_write_actions(state.get("memory_hints") or []) if h]
    return [str(h) for h in hints], strip_write_actions(state.get("session_logic_signature"))
