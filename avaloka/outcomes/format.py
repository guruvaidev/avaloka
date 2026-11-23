"""Deterministic number formatting for Outcome Report figures.

Every renderer substitutes ``{fN}`` placeholders through this one formatter,
and ``assets/outcome-runtime.js`` implements the same rules line for line, so
the HTML export, the UI, the Markdown and the chat summary print the same
string for the same figure. ``golden/format.json`` holds the two equal: the
pytest and the vitest suites both run it.

Rounding is half-even on the decimal the number prints as, not on its binary
value: 2.675 at two decimals is a tie and goes to 2.68, where Python's
``round`` gives 2.67. Both languages print a float as its shortest round-trip
decimal, so working on that string (and scaling by moving the decimal point)
gives both sides exactly the same digits with no floating-point arithmetic.

Rules, per unit (``decimals`` is the figure's, 0-4):

==========  =========================================================
count       ``1,234`` ; from 10,000 a scale suffix: ``12.3K``, ``4.5M``,
unitless    ``1.2B``, ``3T`` (one decimal, a trailing ``.0`` dropped)
currency    as count, after the symbol (USD ``$``, EUR ``€``, GBP ``£``,
            JPY ``¥``, INR ``₹``) or the ISO code and a space (``CHF 12.5``);
            no code → the plain number, never a guessed symbol
percent     ``8.4%``
pp          ``0.9 pp`` (percentage points)
ratio       ``3.25``
days        ``28 days``, ``1 day``
==========  =========================================================

A negative number starts with ``-`` (before any symbol: ``-$1.9M``); with
``signed=True`` a positive one starts with ``+``. Zero is never signed.

Figures store the real magnitude (1900000, not 1.9 "in millions"); the suffix
is the formatter's job. A figure's optional ``compact`` picks it for the
scaled units (count, currency, unitless; other units ignore it):

- absent: a suffix from 10,000 (``9,999``, ``12.3K``, ``$1.9M``);
- ``true``: a suffix from 1,000 (``1.2K``);
- ``false``: never a suffix (``$1,900,000``).

The mantissa is rounded half-even to one decimal from the exact value, and a
mantissa that reaches 1,000 moves up a tier: 999,950 is ``1M``, never
``1,000.0K``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

UNITS = ("count", "currency", "percent", "pp", "ratio", "days", "unitless")
MAX_DECIMALS = 4
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "INR": "₹"}
_SCALED_UNITS = ("count", "currency", "unitless")
_SUFFIXES = ("", "K", "M", "B", "T")
_SUFFIX_FROM_DIGITS = 5  # an integer part of 10,000 or more takes a suffix
_COMPACT_FROM_DIGITS = 4  # compact=True: from 1,000
_CURRENCY_CODE = re.compile(r"[A-Z]{3}\Z")
_NUMBER_TEXT = re.compile(r"(-?)(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?\Z")
_PLACEHOLDER = re.compile(r"\{(f[0-9]{1,3})\}")


class FormatError(ValueError):
    """A figure the formatter refuses; ``code`` is shared with the JS runtime."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


# ── exact decimal digits ─────────────────────────────────────────────────────
# A number is (negative, digits, exponent): value = ±int(digits) · 10**exponent.

def _decimal(value: Any) -> tuple[bool, str, int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormatError("value", f"{value!r} is not a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise FormatError("value", f"{value!r} is not finite")
    match = _NUMBER_TEXT.match(repr(value) if isinstance(value, float) else str(value))
    if match is None:  # pragma: no cover - repr of a finite float always matches
        raise FormatError("value", f"cannot read {value!r}")
    sign, whole, frac, exp = match.groups()
    frac = frac or ""
    digits = (whole + frac).lstrip("0") or "0"
    return sign == "-", digits, int(exp or 0) - len(frac)


def _increment(digits: str) -> str:
    out = list(digits)
    i = len(out) - 1
    while i >= 0:
        if out[i] == "9":
            out[i] = "0"
            i -= 1
        else:
            out[i] = str(int(out[i]) + 1)
            return "".join(out)
    return "1" + "".join(out)


def _round_digits(digits: str, exponent: int, decimals: int) -> str:
    """Half-even: the integer that is the value times 10**decimals, as digits."""
    if exponent >= -decimals:
        return (digits + "0" * (exponent + decimals)).lstrip("0") or "0"
    cut = -decimals - exponent
    padded = digits.rjust(cut + 1, "0")
    keep, drop = padded[:-cut], padded[-cut:]
    half = "5" + "0" * (cut - 1)
    if drop > half or (drop == half and int(keep[-1]) % 2 == 1):
        keep = _increment(keep)
    return keep.lstrip("0") or "0"


def _place_point(scaled: str, decimals: int) -> tuple[str, str]:
    padded = scaled.rjust(decimals + 1, "0")
    if decimals == 0:
        return padded, ""
    return padded[:-decimals].lstrip("0") or "0", padded[-decimals:]


def round_half_even(value: Any, decimals: int, *, shift: int = 0) -> str:
    """``value · 10**shift`` rounded half-even to ``decimals``, as a canonical string.

    ``shift=2`` reads a fraction as a percent exactly (no float multiply).
    Two figures agree after rounding iff these strings are equal.
    """
    negative, digits, exponent = _decimal(value)
    scaled = _round_digits(digits, exponent + shift, decimals)
    whole, frac = _place_point(scaled, decimals)
    text = f"{whole}.{frac}" if decimals else whole
    return "-" + text if negative and scaled != "0" else text


# ── the formatter ────────────────────────────────────────────────────────────

def _group(whole: str) -> str:
    head = len(whole) % 3 or 3
    return ",".join([whole[:head]] + [whole[i:i + 3] for i in range(head, len(whole), 3)])


def _magnitude(digits: str, exponent: int, decimals: int,
               compact: bool | None = None) -> tuple[str, str]:
    """The unsigned body of a scaled unit: grouped digits or a K/M/B/T mantissa."""
    whole, frac = _place_point(_round_digits(digits, exponent, decimals), decimals)
    tier = 0
    from_digits = _SUFFIX_FROM_DIGITS if compact is None else _COMPACT_FROM_DIGITS
    if compact is not False and len(whole) >= from_digits:
        tier = min((len(whole) - 1) // 3, len(_SUFFIXES) - 1)
    if tier == 0:
        return (_group(whole) + ("." + frac if frac else "")), ""
    while True:
        m_whole, m_frac = _place_point(_round_digits(digits, exponent - 3 * tier, 1), 1)
        if len(m_whole) > 3 and tier < len(_SUFFIXES) - 1:
            tier += 1  # 999,960 is 1,000.0K, which is 1M
            continue
        body = _group(m_whole) + ("" if m_frac == "0" else "." + m_frac)
        return body, _SUFFIXES[tier]


def format_number(value: Any, unit: str, decimals: int, currency: str | None = None,
                  *, signed: bool = False, compact: bool | None = None) -> str:
    """The display string for one number; see the module docstring for the rules."""
    if unit not in UNITS:
        raise FormatError("unit", f"unknown unit {unit!r}")
    # JSON has one number type: 2.0 is the integer 2, as Number.isInteger says.
    if isinstance(decimals, float) and decimals.is_integer():
        decimals = int(decimals)
    if isinstance(decimals, bool) or not isinstance(decimals, int) \
            or not 0 <= decimals <= MAX_DECIMALS:
        raise FormatError("decimals", f"decimals must be an integer 0-{MAX_DECIMALS}")
    if currency is not None and (not isinstance(currency, str)
                                 or not _CURRENCY_CODE.match(currency)):
        raise FormatError("currency", f"{currency!r} is not an ISO 4217 code")
    if compact is not None and not isinstance(compact, bool):
        raise FormatError("compact", f"compact must be true, false or absent, not {compact!r}")
    negative, digits, exponent = _decimal(value)
    is_zero = _round_digits(digits, exponent, decimals) == "0"

    if unit in _SCALED_UNITS:
        body, suffix = _magnitude(digits, exponent, decimals, compact)
        body += suffix
    else:
        whole, frac = _place_point(_round_digits(digits, exponent, decimals), decimals)
        body = _group(whole) + ("." + frac if frac else "")

    if unit == "currency" and currency is not None:
        symbol = CURRENCY_SYMBOLS.get(currency)
        body = symbol + body if symbol else f"{currency} {body}"
    elif unit == "percent":
        body += "%"
    elif unit == "pp":
        body += " pp"
    elif unit == "days":
        body += " day" if body == "1" else " days"

    if is_zero:
        return body
    if negative:
        return "-" + body
    return "+" + body if signed else body


def format_figure(figure: Mapping[str, Any], *, signed: bool = False) -> str:
    """Format a ``figures[]`` entry (``value``, ``unit``, ``decimals``, ``currency``,
    ``compact``)."""
    return format_number(figure["value"], figure["unit"], figure["decimals"],
                         figure.get("currency"), signed=signed, compact=figure.get("compact"))


def render_text(text: str, figures: Any) -> str:
    """Substitute every ``{fN}`` in a text slot with its formatted figure.

    ``figures`` is the report's ``figures`` list or a mapping of id to figure.
    An unknown id raises ``FormatError("unresolved_figure")``.
    """
    by_id = figures if isinstance(figures, Mapping) else {f["id"]: f for f in figures}

    def substitute(match: re.Match[str]) -> str:
        figure = by_id.get(match.group(1))
        if figure is None:
            raise FormatError("unresolved_figure", f"{match.group(1)} is not in figures")
        return format_figure(figure)

    return _PLACEHOLDER.sub(substitute, text)


# ── matching a figure to its evidence (§3.2 rule 3) ─────────────────────────

def evidence_shift(figure_unit: str, evidence_unit: str | None) -> int | None:
    """Decimal places to move the evidence by before comparing, or None on mismatch.

    A ``percent`` figure over evidence stored as a ``fraction`` is scaled by 100;
    otherwise the units must be equal (an evidence leaf that declares no unit is
    taken to be in the figure's unit).
    """
    if evidence_unit is None or evidence_unit == figure_unit:
        return 0
    if figure_unit == "percent" and evidence_unit == "fraction":
        return 2
    return None


def figure_matches(value: Any, decimals: int, evidence: Any, *, shift: int = 0,
                   figure_unit: str | None = None) -> bool:
    """``round_half_even(e·s, d) == round_half_even(v, d)``, exactly.

    ``evidence`` is a bare number (compared at ``shift``) or an evidence leaf
    ``{"value", "unit"}``, compared in its declared unit: with ``figure_unit``
    given, the shift comes from :func:`evidence_shift` and a unit mismatch is
    no match. 8.4 % matches ``{"value": 0.084, "unit": "fraction"}``; it does
    not match a bare 0.084.
    """
    if isinstance(evidence, Mapping):
        if figure_unit is None:
            raise FormatError("unit", "figure_unit is needed to compare with an evidence leaf")
        leaf_shift = evidence_shift(figure_unit, evidence.get("unit"))
        if leaf_shift is None:
            return False
        evidence, shift = evidence.get("value"), leaf_shift
    return round_half_even(evidence, decimals, shift=shift) == round_half_even(value, decimals)


def figure_matches_evidence(figure: Mapping[str, Any], evidence: Any) -> bool:
    """§3.2 rule 3 for one ``figures[]`` entry against the report's ``evidence``.

    False when the pointer does not resolve to a leaf, the units disagree, a
    currency leaf names another ISO code, or the values differ after rounding.
    """
    from .spec import evidence_leaf

    try:
        value, unit, currency = evidence_leaf(evidence, figure["evidence_ref"])
    except (KeyError, ValueError):
        return False
    if figure["unit"] == "currency" and currency is not None \
            and currency != figure.get("currency"):
        return False
    leaf = value if unit is None else {"value": value, "unit": unit}
    return figure_matches(figure["value"], figure["decimals"], leaf, figure_unit=figure["unit"])
