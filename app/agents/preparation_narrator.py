"""Explain, in plain language, what is wrong with a dataset and what to do about it.

The preparation and PII agents produce structured reports: strategies, fill
values, parse rates, confidence scores. Those are correct and unreadable. A user
who is told ``{"action": "impute", "strategy": "median", "missing_rate": 0.32}``
has been given data, not an explanation, and cannot make the decision the report
exists to inform.

This module turns those reports into the sentence Avaloka would say out loud:
what she found, why it matters, what she proposes, and what she will not do
without being asked.

Three rules shape every line it writes.

**Every number is measured.** The narrator has no access to a model and invents
nothing; it reads figures out of the reports and puts words around them. This is
the same reason the Claim Verifier is deterministic — a component whose job is to
explain the data must not be able to make things up about it.

**Consequences, not categories.** "32% missing" is a measurement. "One row in
three has no value here, so any average over this column is really an average
over the two-thirds that do" is an explanation. The second is what lets someone
decide whether they care.

**Refusals are explained, not hidden.** When the agent declines to impute a
column that is 78% empty, saying nothing looks like an oversight. Saying *why*
turns it into a judgement the user can overrule.

Output is plain text with no markup, so it reads correctly in a terminal, in the
chat panel, and in an API response.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Below this, a problem is worth listing but not worth leading with.
MINOR_MISSING_RATE = 0.05


def _plural(n: int, singular: str, plural: Optional[str] = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _pct(value: float) -> str:
    """Percentages a person would say out loud."""
    if value >= 0.1:
        return f"{value:.0%}"
    if value >= 0.01:
        return f"{value:.1%}"
    return "under 1%"


def _fraction_in_words(rate: float) -> Optional[str]:
    """A ratio is easier to picture than a percentage."""
    for threshold, phrase in ((0.45, "almost every other row"),
                              (0.30, "one row in three"),
                              (0.22, "one row in four"),
                              (0.17, "one row in five"),
                              (0.09, "one row in ten")):
        if rate >= threshold:
            return phrase
    return None


# --------------------------------------------------------------------------- #
# The pieces
# --------------------------------------------------------------------------- #

def _describe_coercions(decisions: Sequence[Dict[str, Any]]) -> List[str]:
    lines = []
    for d in decisions:
        if d["action"] != "coerce_numeric":
            continue
        ev = d.get("evidence") or {}
        col = d["column"]
        example = ev.get("example")
        failed = int(ev.get("failed") or 0)
        if ev.get("had_currency_marks"):
            line = (f"- {col} looks like text because the values carry currency symbols "
                    f"(for example {example!r}). Nothing can be averaged or summed while "
                    f"it stays that way, so I'll read it as a number.")
        else:
            line = (f"- {col} is stored as text but its values are numeric "
                    f"(for example {example!r}). I'll convert it so it can be used in "
                    f"calculations.")
        if failed:
            line += (f" {_plural(failed, 'value')} won't parse and will become missing; "
                     f"I'd rather leave a gap than guess at them.")
        lines.append(line)
    return lines


def _describe_missing(decisions: Sequence[Dict[str, Any]],
                      unimputable: Dict[str, float]) -> List[str]:
    lines = []
    for d in decisions:
        if d["action"] != "impute":
            continue
        ev = d.get("evidence") or {}
        rate = float(ev.get("missing_rate") or 0)
        if rate <= 0:
            continue
        col, strategy = d["column"], ev.get("strategy")
        picture = _fraction_in_words(rate)
        opener = (f"- {col} is missing {_pct(rate)} of its values"
                  + (f" — {picture}" if picture else ""))
        if strategy == "median":
            lines.append(f"{opener}. The column is skewed, so I'll fill the gaps with the "
                         f"median rather than the mean; a few large values would drag an "
                         f"average somewhere no real row sits.")
        elif strategy == "mean":
            lines.append(f"{opener}. The values are evenly spread, so the mean is a fair "
                         f"stand-in and that's what I'll use.")
        elif strategy == "mode":
            lines.append(f"{opener}. One value dominates the column, so I'll use it for "
                         f"the gaps.")
        elif strategy == "constant":
            lines.append(f"{opener}. No single value dominates, so filling with the "
                         f"commonest one would invent a majority that isn't there. I'll "
                         f"mark them as missing instead, which keeps the gap visible.")
        elif strategy == "none":
            lines.append(f"{opener}, and I'm going to leave them alone — "
                         f"{d.get('detail', 'filling them would imply something untrue')}.")
        else:
            lines.append(f"{opener}.")

    for col, rate in unimputable.items():
        lines.append(
            f"- {col} is {_pct(rate)} empty. That's past the point where filling it "
            f"repairs anything — I'd be inventing most of the column and then you'd "
            f"analyse my invention. I've left it untouched. Worth deciding whether to "
            f"drop the column, or drop the rows that are missing it.")
    return lines


def _describe_features(decisions: Sequence[Dict[str, Any]]) -> List[str]:
    lines = []
    for d in decisions:
        action, col = d["action"], d["column"]
        produced = d.get("produced") or []
        if action == "add_indicator":
            lines.append(f"- Whether {col} was missing can itself be informative, so I'll "
                         f"keep a flag for it before filling the gaps.")
        elif action == "add_date_parts":
            lines.append(f"- {col} is a timestamp. On its own that's hard to group by, so "
                         f"I'll add the year, month, day, weekday and hour as separate "
                         f"columns.")
        elif action == "add_cyclical":
            lines.append(f"- I'll also encode {col}'s month so that December and January "
                         f"sit next to each other. As a plain number they look eleven "
                         f"apart, which misleads a model.")
        elif action == "group_rare":
            n = len((d.get("evidence") or {}).get("levels") or [])
            lines.append(f"- {col} has {_plural(n, 'category')} appearing in almost no "
                         f"rows. I'll group them together so they don't become noise.")
        elif action == "add_ratio":
            denom = (d.get("evidence") or {}).get("denominator")
            name = produced[0] if produced else f"{col}_per_{denom}"
            lines.append(f"- I'll add {name}, since the ratio usually says more than "
                         f"either column alone.")
        elif action == "parse_datetime":
            lines.append(f"- {col} is stored as text but reads as dates; I'll parse it so "
                         f"it can be sorted and grouped by time.")
        elif action == "coerce_boolean":
            lines.append(f"- {col} holds two values that mean yes and no; I'll store it as "
                         f"a true/false column.")
    return lines


def _describe_pii(pii: Dict[str, Any]) -> List[str]:
    findings = pii.get("findings") or []
    direct = [f for f in findings if f["sensitivity"] == "direct"]
    quasi = [f for f in findings if f["sensitivity"] == "quasi"]
    lines: List[str] = []
    if direct:
        names = ", ".join(f["column"] for f in direct)
        lines.append(f"- {names} identif{'ies' if len(direct) == 1 else 'y'} people "
                     f"directly. I can replace the values with stable stand-ins — the same "
                     f"person always gets the same one, so counts and joins still work — "
                     f"but I won't change anything until you ask.")
    if quasi:
        names = ", ".join(f["column"] for f in quasi)
        lines.append(f"- {names} {'is' if len(quasi) == 1 else 'are'} not identifying "
                     f"alone, but narrow{'s' if len(quasi) == 1 else ''} things down in "
                     f"combination.")
    for risk in pii.get("residual_risk") or []:
        lines.append(f"- {risk}")
    return lines


# --------------------------------------------------------------------------- #
# The whole explanation
# --------------------------------------------------------------------------- #

def narrate_preparation(plan: Dict[str, Any],
                        pii: Optional[Dict[str, Any]] = None,
                        *, dataset_name: Optional[str] = None) -> str:
    """Turn the structured reports into what Avaloka would say.

    Returns plain text. Empty findings produce a short, honest all-clear rather
    than a manufactured list of concerns.
    """
    decisions = plan.get("decisions") or []
    unimputable = plan.get("unimputable") or {}
    name = dataset_name or "this dataset"

    coercions = _describe_coercions(decisions)
    missing = _describe_missing(decisions, unimputable)
    features = _describe_features(decisions)
    pii_lines = _describe_pii(pii) if pii else []

    problems = coercions + missing
    parts: List[str] = []

    if not problems and not features and not pii_lines:
        return (f"I looked over {name} and found nothing that needs cleaning up — no "
                f"missing values, no columns stored as the wrong type. It's ready to "
                f"analyse as it is.")

    # 1. Headline: what shape is it in, and is that a problem?
    if problems:
        parts.append(
            f"Before I analyse {name}, there {'is' if len(problems) == 1 else 'are'} "
            f"{_plural(len(problems), 'thing')} worth fixing. Left alone, "
            f"{'it' if len(problems) == 1 else 'they'} would quietly change the answers "
            f"rather than cause an error, which is the harder kind of problem to notice.")
        parts.append("\n".join(problems))
    else:
        parts.append(f"{name} is in good shape — nothing is missing and every column is "
                     f"the type it should be.")

    # 2. What gets built on top
    if features:
        parts.append("I'd also add a few columns that make the data easier to work with:")
        parts.append("\n".join(features))

    # 3. Privacy, always reported, never applied unasked
    if pii_lines:
        parts.append("On personal data:")
        parts.append("\n".join(pii_lines))

    # 4. The decision, stated as a decision
    fitted = plan.get("fitted_rows")
    closing = "Say the word and I'll apply this."
    if unimputable:
        closing += (f" The {_plural(len(unimputable), 'column')} I've left alone "
                    f"{'is' if len(unimputable) == 1 else 'are'} the one"
                    f"{'' if len(unimputable) == 1 else 's'} I'd like your call on first.")
    elif any(f["sensitivity"] == "direct" for f in (pii or {}).get("findings", [])):
        closing += " Tell me if you want the personal data masked before I go further."
    else:
        closing += " I can also show you the code, or skip any step you disagree with."
    if fitted:
        closing += (f"\n\nEvery figure above comes from the {fitted:,} rows I measured, "
                    f"not an estimate.")
    parts.append(closing)

    return "\n\n".join(parts)


def narrate_headline(plan: Dict[str, Any], pii: Optional[Dict[str, Any]] = None) -> str:
    """One line, for a progress update or a notification."""
    decisions = plan.get("decisions") or []
    repairs = sum(1 for d in decisions if d["action"] in
                  ("coerce_numeric", "impute", "parse_datetime", "coerce_boolean"))
    added = len(plan.get("engineered_columns") or [])
    blocked = len(plan.get("unimputable") or {})
    direct = len([f for f in (pii or {}).get("findings", [])
                  if f["sensitivity"] == "direct"])

    if not (repairs or added or blocked or direct):
        return "Nothing to clean up — the data is ready to analyse."
    bits = []
    if repairs:
        bits.append(f"{_plural(repairs, 'column')} to repair")
    if added:
        bits.append(f"{_plural(added, 'feature')} to add")
    if blocked:
        bits.append(f"{_plural(blocked, 'column')} too sparse to fix")
    if direct:
        bits.append(f"{_plural(direct, 'column')} holding personal data")
    return "Found " + ", ".join(bits[:-1]) + (f" and {bits[-1]}." if len(bits) > 1
                                              else bits[0] + ".")
