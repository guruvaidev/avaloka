"""Units and scale a file states about its own numbers -- found, never invented.

A finance export says "Revenue by region (£ thousands)" once, in a title row,
and then writes 482.5 for £482,500. Nothing in the header, the column names
or the cell formats repeats it, so a reader that discards the title -- which
is the right thing to do structurally -- discards the units with it, and a
total comes out three orders of magnitude small.

The rule here is the same as for every figure Avaloka reports: code writes it
from what the file says, and it carries where it came from. A unit is only
ever read out of text in the file (a title row, a header cell) through the
fixed tables below; a column's name is never taken to mean a currency, and no
model is asked. A file that states no unit gets none, and nothing downstream
may add one (:func:`introduces_units` is how the narration paths check).

The frame is never multiplied. The figures stay as written, so the profile,
the prepared data and the file agree, and the scale is reported beside them:
goal_analysis states a total both as written and at full scale.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

#: Scale words, as exports write them, and the factor each states.
_SCALES: tuple[tuple[re.Pattern[str], int, str], ...] = (
    (re.compile(r"\bthousands?\b|'000s?\b|\b000s\b|\(000\)|£000\b|\$000\b|€000\b", re.I),
     1_000, "thousands"),
    (re.compile(r"\bmillions?\b|\bmn\b", re.I), 1_000_000, "millions"),
    (re.compile(r"\bbillions?\b|\bbn\b", re.I), 1_000_000_000, "billions"),
)

#: ISO codes read as written.
_CODES = ("USD", "GBP", "EUR", "JPY", "CNY", "INR", "CAD", "AUD", "CHF", "SEK", "NOK",
          "DKK", "NZD", "HKD", "SGD", "ZAR", "BRL", "MXN", "KRW")
_CODE = re.compile(r"\b(%s)\b" % "|".join(_CODES))

#: Symbols mapped to a code only where the symbol is unambiguous in practice.
#: "$" is half a dozen currencies and stays "$" unless the text also states a
#: code (EDGAR writes "USD ($)"); "¥" is yen or yuan and stays a symbol.
_SYMBOLS = {"£": "GBP", "€": "EUR", "₹": "INR"}
_AMBIGUOUS = ("$", "¥")

#: A column whose header says it counts things, or is a rate, is not in the
#: title's currency or scale: "Headcount" on a "£ thousands" sheet is people.
_NOT_SCALED = re.compile(
    r"\b(count|headcount|number|no\.|units?|shares?|per share|rate|ratio|percent|"
    r"pct|margin|employees|staff|fte|days|hours|years?|age|score|index|quantity|qty)\b|#|%",
    re.I)


def _currency(text: str) -> tuple[Optional[str], Optional[str]]:
    """``(code or symbol, symbol)`` stated in ``text``, or ``(None, None)``."""
    code = _CODE.search(text)
    symbol = next((s for s in [*_SYMBOLS, *_AMBIGUOUS] if s in text), None)
    if code:
        return code.group(1), symbol
    if symbol in _SYMBOLS:
        return _SYMBOLS[symbol], symbol
    return (symbol, symbol) if symbol else (None, None)


def read_unit(text: str) -> Optional[dict[str, Any]]:
    """The currency and scale one piece of text states, or None.

    ``{"scale": 1000, "scale_word": "thousands", "currency": "GBP",
    "symbol": "£"}``, with only the keys the text supports.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    unit: dict[str, Any] = {}
    for pattern, factor, word in _SCALES:
        if pattern.search(text):
            unit.update(scale=factor, scale_word=word)
            break
    currency, symbol = _currency(text)
    if currency:
        unit["currency"] = currency
        if symbol:
            unit["symbol"] = symbol
    return unit or None


def unit_words(unit: dict[str, Any]) -> str:
    """"GBP thousands", "$ millions", "USD" -- the unit as the report says it."""
    return " ".join(str(part) for part in (unit.get("currency"), unit.get("scale_word")) if part)


def stated_units(title_rows: Iterable[tuple[int, str]], columns: Iterable[str],
                 numeric: Iterable[str]) -> dict[str, Any]:
    """What a sheet states about its numbers, column by column.

    ``title_rows`` are ``(sheet row, text)`` for the rows above the header.
    The first one that states a currency or a scale is the sheet's unit, and
    it applies to every numeric column except one whose own header states a
    unit (that one wins) or says it counts things or is a rate. Returns
    ``{}`` when nothing is stated, else ``{"stated": {..., "source": ...},
    "columns": {column: unit}, "exempt": [columns]}``.
    """
    stated: Optional[dict[str, Any]] = None
    for sheet_row, text in title_rows:
        unit = read_unit(text)
        if unit:
            stated = {**unit, "source": {"where": "title", "sheet_row": int(sheet_row),
                                         "text": str(text)}}
            break
    numeric = set(numeric)
    per_column: dict[str, Any] = {}
    exempt: list[str] = []
    for column in columns:
        own = read_unit(str(column))
        if own and column in numeric:
            per_column[str(column)] = {**own, "source": {"where": "header", "text": str(column)}}
            continue
        if stated is None or column not in numeric:
            continue
        if _NOT_SCALED.search(str(column)):
            exempt.append(str(column))
            continue
        per_column[str(column)] = {k: v for k, v in stated.items()}
    if stated is None and not per_column:
        return {}
    out: dict[str, Any] = {"columns": per_column, "exempt": exempt}
    if stated is not None:
        out["stated"] = stated
    return out


#: Any unit token a narration could add: codes, symbols, scale words.
_ANY_UNIT = re.compile(
    r"\b(%s)\b|[£€₹$¥]|\b(thousands?|millions?|billions?)\b" % "|".join(_CODES), re.I)


def introduces_units(original: str, rewritten: str) -> bool:
    """Whether ``rewritten`` states a unit ``original`` does not. A model may
    reword a line; it may not give a figure a currency or a scale."""
    def found(text: str) -> set[str]:
        return {m.group(0).lower() for m in _ANY_UNIT.finditer(text or "")}
    return not found(rewritten) <= found(original)


__all__ = ["introduces_units", "read_unit", "stated_units", "unit_words"]
