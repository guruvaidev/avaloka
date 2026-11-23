"""A correction to the method: its facts, written by code, and the reasoning.

Owner ruling: a correction to HOW something was computed ("you excluded
refunds", "include them and recompute", "use median instead of mean") always
re-plans. It is never answered from the results already in hand, and the reply
gives the reasoning: the original approach, the corrected one, why the change
matters, and what the alternative is.

The facts are code's. :func:`correction_facts` reads them from the user's own
words, and :func:`reasoning_text` says them without a model. A model may phrase
the reasoning, but :func:`phrasing_holds` checks that phrasing against these
facts -- both approaches named as written, a reason, the alternative, and no
figure the turn did not already hold -- and the code-written text is used
whenever it does not.

A disputed VALUE with no change of method ("that's wrong, it's 42.5") is not a
correction; that is held, not re-planned.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

from app.agents.user_text import own_words

#: The subject of a correction: a few words, ended by punctuation, the end of
#: the message, or a word that starts a new clause.
_SUBJECT = r"(?:the\s+)?(?P<x>[a-z][a-z0-9_\- ]{0,40}?)(?=\s*(?:[.,;:!?]|$|\band\b|\bthen\b|\bfrom\b|\bwhen\b|\bin\b|\bso\b))"

_YOU_LEFT_OUT = re.compile(
    r"\byou\s+(?:have\s+|had\s+)?(?:excluded|left\s+out|dropped|filtered\s+out|removed|ignored|"
    r"skipped|missed|forgot\s+(?:to\s+include\s+)?)\s*" + _SUBJECT, re.I)
_YOU_KEPT = re.compile(
    r"\byou\s+(?:have\s+|had\s+)?(?:included|kept|counted)\s+" + _SUBJECT, re.I)
_REDO = r"(?:re-?compute|re-?run|recalculate|redo|run\s+it\s+again)"
_INCLUDE_AND_REDO = re.compile(
    r"\b(?:include|add\s+back|count)\s+" + _SUBJECT + r"\s*(?:and|then)\s+" + _REDO, re.I)
_EXCLUDE_AND_REDO = re.compile(
    r"\b(?:exclude|remove|drop|leave\s+out|filter\s+out)\s+" + _SUBJECT + r"\s*(?:and|then)\s+" + _REDO,
    re.I)
_USE_INSTEAD = re.compile(
    r"\buse\s+(?:the\s+)?(?P<new>[a-z][a-z0-9_\- ]{0,30}?)\s+instead\s+of\s+(?:the\s+)?"
    r"(?P<old>[a-z][a-z0-9_\- ]{0,30}?)(?=\s*(?:[.,;:!?]|$))", re.I)

_PRONOUNS = frozenset({"them", "it", "those", "these", "that", "this"})


@dataclass(frozen=True)
class CorrectionFacts:
    """What the correction changes, in the words the reply must use."""

    subject: str
    original: str        # the approach the earlier answer took
    corrected: str       # the approach the correction asks for
    kept_out: bool       # the original left the subject out

    @property
    def alternative(self) -> str:
        """Keeping the original approach -- the one other reading on offer."""
        return self.original

    def as_dict(self) -> dict:
        return {**asdict(self), "alternative": self.alternative}


def _subject(match: "re.Match[str]") -> Optional[str]:
    x = " ".join(match.group("x").split()).strip(" -")
    return x if x and x.lower() not in _PRONOUNS else None


def correction_facts(text: str) -> Optional[CorrectionFacts]:
    """The correction in *text*, or None when it corrects no method."""
    t = own_words(text or "")
    for pattern, kept_out in ((_YOU_LEFT_OUT, True), (_YOU_KEPT, False)):
        m = pattern.search(t)
        if m and (x := _subject(m)):
            return (CorrectionFacts(x, f"excluding {x}", f"including {x}", True) if kept_out
                    else CorrectionFacts(x, f"including {x}", f"excluding {x}", False))
    m = _INCLUDE_AND_REDO.search(t)
    if m and (x := _subject(m)):
        return CorrectionFacts(x, f"excluding {x}", f"including {x}", True)
    m = _EXCLUDE_AND_REDO.search(t)
    if m and (x := _subject(m)):
        return CorrectionFacts(x, f"including {x}", f"excluding {x}", False)
    m = _USE_INSTEAD.search(t)
    if m:
        new, old = " ".join(m.group("new").split()), " ".join(m.group("old").split())
        if new and old and new.lower() != old.lower():
            return CorrectionFacts(new, f"using {old}", f"using {new}", False)
    return None


def reasoning_text(facts: CorrectionFacts) -> str:
    """The reasoning, said by code from the facts alone."""
    if facts.original.startswith("using "):
        why = "that changes how the result is computed"
        intended = f"{facts.original[len('using '):]} was what you meant"
    else:
        why = "that changes which rows the result is computed over"
        intended = f"leaving {facts.subject} {'out' if facts.kept_out else 'in'} was intended"
    return (
        f"Good catch: my approach was {facts.original}. I'm re-planning it {facts.corrected} "
        f"instead, because {why}, so the earlier answer was to a narrower question than "
        f"yours. The alternative is to keep {facts.original}, which is right only if "
        f"{intended}. I'll re-run it and show what changes."
    )


_REPLAN = re.compile(r"\b(re-?plan\w*|re-?run\w*|re-?comput\w*|recalculat\w*|redo\w*|"
                     r"run (?:it|this|that) again)\b", re.I)
_REFUSE = re.compile(r"\b(won'?t|will not|not going to|no need to|don'?t need to)\s+"
                     r"(?:re-?plan|re-?run|re-?comput|recalculat|redo|run (?:it|this|that) again)", re.I)
_REASON = re.compile(r"\b(because|since|so that|which means|that changes|otherwise)\b", re.I)
_ALTERNATIVE = re.compile(r"\b(alternative|the other option)\b", re.I)
_FIGURE = re.compile(r"-?\d[\d,]*\.?\d*")


def _figures(text: str) -> set[float]:
    out = set()
    for raw in _FIGURE.findall(text or ""):
        try:
            out.add(round(float(raw.replace(",", "")), 6))
        except ValueError:
            continue
    return out


def phrasing_holds(candidate: str, facts: CorrectionFacts, known_text: Iterable[str]) -> bool:
    """Whether a model's phrasing of the reasoning keeps to the facts.

    Both approaches as code wrote them, that it is being re-planned (and not
    refused), a reason, the alternative, and no figure beyond small counts that
    is not already in ``known_text`` (the verified results and the user's own
    messages). A phrasing that explains the correction but never says it will
    be re-run is not a re-plan.
    """
    # Typographic apostrophes and hyphens are the ASCII ones: gpt-oss writes
    # "re\u2011run" with a non-breaking hyphen.
    text = (candidate or "").replace("\u2019", "'").replace("\u2010", "-").replace("\u2011", "-")
    lowered = text.lower()
    if facts.original.lower() not in lowered or facts.corrected.lower() not in lowered:
        return False
    if not _REPLAN.search(text) or _REFUSE.search(text):
        return False
    if not _REASON.search(text) or not _ALTERNATIVE.search(text):
        return False
    known = set()
    for chunk in known_text:
        known |= _figures(chunk)
    for n in _figures(text):
        if abs(n) < 10 and float(n).is_integer():
            continue
        if n not in known:
            return False
    return True
