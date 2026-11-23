"""The model step of intent parsing: one prompt, one recognizer, bounded retries.

The model is asked for a JSON object over the closed vocabularies
(:mod:`avaloka.recipes.base`) and its reply is checked by
:func:`~avaloka.recipes.intent.recognize`. A rejected reply is sent back once
with the rejection reason, because a small model usually fixes a named
mistake and rarely fixes an unnamed one. After ``max_attempts`` the caller
gets ``None`` and decides (design §3.2 rule 1): escalate a tier, or ask the
user for the parameter the heuristics could not bind.

The client is duck-typed on the one method the CLI's ``LLMClient`` exposes
for a system-plus-prompt call::

    client.complete(system: str, prompt: str, *, max_tokens: int) -> str | None

``None`` means the provider failed and has already reported why; this module
does not retry it. No provider code lives here, and nothing here reaches the
network: tests inject an object with the same method returning recorded text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Protocol, Sequence

from avaloka.outcomes.semantics import ColumnSemantics

from .base import GRAINS, INTENT_FLAGS, Intent, RecipeSpec
from .intent import PAYLOAD_KEYS, IntentRejected, recognize
from .specs import WINDOWS

MAX_TOKENS = 300          # design §3.3: identical ceiling for every candidate model


class IntentClient(Protocol):
    def complete(self, system: str, prompt: str, *, max_tokens: int = ...) -> str | None: ...


def _param_line(spec: RecipeSpec) -> str:
    parts = []
    for p in spec.params:
        shape = {"grain": "one of GRAINS", "window": "one of WINDOWS"}.get(
            p.role, f"{'a list of' if p.many else 'one'} {p.role} column name{'s' if p.many else ''}")
        parts.append(f'"{p.name}": {shape}{"" if p.required else " (optional)"}')
    return f"- {spec.id}: {spec.title}. params: {{{', '.join(parts)}}}" if parts else f"- {spec.id}: {spec.title}. params: {{}}"


def intent_prompt(question: str, columns: Mapping[str, str],
                  candidates: Sequence[RecipeSpec]) -> tuple[str, str]:
    """(system, user) for one intent call. Every value the model may return
    is listed; anything else is rejected by :func:`recognize`."""
    system = "\n".join([
        "You map one data question to one analysis recipe and its parameters.",
        "Reply with a single JSON object and nothing else. Keys, all required: "
        + ", ".join(f'"{k}"' for k in PAYLOAD_KEYS) + ".",
        '"recipe" is one of the ids below. "params" is an object whose keys are that recipe\'s parameter names',
        "and whose values are column names from the list given, exactly as spelled; a parameter the question",
        'does not settle is left out, never guessed. "flags" is a list drawn from '
        + ", ".join(f'"{f}"' for f in INTENT_FLAGS) + ":",
        '"causal_question" when the question asks what causes, drives or affects something;',
        '"ambiguous" when two recipes or two columns fit equally; "multi_question" when it asks two different things.',
        '"confidence" is a number between zero and one.',
        "GRAINS: " + ", ".join(GRAINS) + ". WINDOWS: " + ", ".join(WINDOWS) + ".",
        "Recipes:",
        *(_param_line(s) for s in candidates),
    ])
    user = "\n".join([
        "Columns (name: role):",
        *(f"- {n}: {r}" for n, r in columns.items()),
        "",
        f"Question: {question}",
        "",
        "JSON:",
    ])
    return system, user


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_reply(text: str) -> Any:
    """The JSON object in a reply, fenced or bare; :class:`IntentRejected` otherwise."""
    body = text.strip()
    m = _FENCE.search(body)
    if m:
        body = m.group(1)
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise IntentRejected("not_json", f"{exc.msg} at character {exc.pos}") from None


def _retry_prompt(user: str, rejected: IntentRejected) -> str:
    return (f"{user[: -len('JSON:')]}Your previous reply was rejected: {rejected.reason}"
            f"{' — ' + rejected.detail if rejected.detail else ''}.\n"
            "Reply again with only the corrected JSON object.\nJSON:")


def model_intent(client: IntentClient, question: str, sem: Mapping[str, ColumnSemantics],
                 candidates: Sequence[RecipeSpec], *, max_attempts: int = 2) -> Intent | None:
    """Ask ``client`` for the intent, at most ``max_attempts`` times.

    Returns the recognized Intent (``source="model"``), or None when the
    provider returned nothing or every attempt was rejected. A recipe outside
    ``candidates`` is a rejection too, even though it is a valid id: the
    caller narrowed the choice on purpose (§3.1, a tie-break between at most
    three).
    """
    columns = {n: c.role for n, c in sem.items()}
    allowed = {s.id for s in candidates}
    system, user = intent_prompt(question, columns, candidates)
    prompt = user
    for _ in range(max_attempts):
        text = client.complete(system, prompt, max_tokens=MAX_TOKENS)
        if text is None:
            return None
        try:
            payload = parse_reply(text)
            intent = recognize(payload, question=question, columns=columns, source="model")
            if intent.recipe not in allowed:
                raise IntentRejected("not_a_candidate", f"{intent.recipe!r}; choose one of {sorted(allowed)}")
            return intent
        except IntentRejected as rejected:
            prompt = _retry_prompt(user, rejected)
    return None
