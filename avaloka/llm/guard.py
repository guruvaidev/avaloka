"""The backstop between a model's rephrasing and the user's screen.

The prompt is the fix (:data:`avaloka.persona.VOICE_SYSTEM` asks for the
rewritten line and nothing else). This is what catches a small model that
narrates the task anyway. Recorded from ``gemma4:e4b``:

    "I'm going to rephrase this to cleanly communicate that no single variable
     shows dominance. Essentially, the distribution is quite even, ..."
    "To structure this request, I'm going to confirm the necessary inputs ...
     I'll keep it concise and confident: \"I'm ready to dive in. ...\""

Sentences that talk about the rewriting rather than the data are dropped; a
quoted answer after such a preamble is kept. If nothing trustworthy is left,
or the rewrite carries a number or a unit (a currency, a scale) the original
did not, the original line is used: a figure the statistics did not produce,
or a unit the file did not state, must never reach the user.
"""

from __future__ import annotations

import re

from avaloka.units import introduces_units

#: A sentence that is about the text, not about the data.
_META = re.compile(
    r"\b(?:rephras|refram|restructur|reword|rewrit|streamlin|paraphras)\w*"
    r"|\bthis (?:statement|line|sentence|concept|observation|update|request|message|text|idea)\b"
    r"|\bto sound (?:like|more)\b|\bmakes? it sound\b|\bsound (?:more )?like me\b"
    r"|\b(?:tell|telling) the user\b|\bin (?:my|your|her) voice\b"
    r"|\bframing (?:it|this)\b|\bframe it as\b"
    r"|\bthe (?:given|original|provided) (?:line|text|sentence|statement)\b"
    r"|\bkeep it (?:concise|short|brief)\b|\bconcise and confident\b"
    r"|^(?:sure|certainly|of course|okay|ok|here(?:'s| is) (?:a|the|my) (?:version|rewrite|take))\b[,!:.]?",
    re.I,
)
_ECHO = re.compile(r"^(?:reply|line|output|answer)\s*:\s*", re.I)
_LEADIN = re.compile(r"^(?:essentially|basically|in short|in other words|so)\s*,\s*", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[\"'“A-Z0-9*\[])")
_QUOTED = re.compile(r"[\"“]([^\"”]{12,})[\"”]\s*$", re.S)
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def is_meta(text: str) -> bool:
    return bool(_META.search(text))


def clean_rephrasing(original: str, output: str | None) -> str:
    """``output`` with any meta-narration removed, or ``original`` when unsafe."""
    if not output or not output.strip() or output.strip() == original.strip():
        return original
    text = _ECHO.sub("", output.strip().replace("**", ""))
    # Preamble, then the actual line in quotes: keep the quoted line.
    q = _QUOTED.search(text)
    if q and is_meta(text[: q.start()]):
        text = q.group(1).strip()
    text = " ".join(text.split())
    kept = [s for s in _SENTENCE.split(text) if s and not is_meta(s)]
    if not kept:
        return original
    kept[0] = _LEADIN.sub("", kept[0])
    kept[0] = kept[0][:1].upper() + kept[0][1:]
    result = " ".join(kept).strip().strip('"“”').strip()
    if not result or is_meta(result):
        return original
    if not _numbers(result) <= _numbers(original):
        return original
    # Nor a unit: a currency or a scale the original line did not state is a
    # claim about the data no statistic made (avaloka/units.py).
    if introduces_units(original, result):
        return original
    return result
