"""Find personal data, and de-identify it without a vault.

Avaloka deliberately does not store datasets — only context — and that single
constraint decides the whole design of this module.

A conventional tokenisation service keeps a lookup table: token → original,
held in a vault, consulted to reverse. With nowhere durable to put such a table,
that approach is unavailable, and pretending otherwise would mean writing the
very data we promised not to keep. So every transform here is **deterministic
and keyed**: the same input under the same key always yields the same output,
reversal requires the key rather than a stored map, and the key lives outside
Avaloka in ``AVALOKA_PII_KEY``.

That property is not a compromise; it is what makes de-identified data still
useful. A deterministic pseudonym preserves joins and group-bys — two rows for
the same customer still land together — so an analysis over pseudonymised data
answers the same questions as one over the original.

**Unkeyed hashing is refused, not offered as a fallback.** SHA-256 of an email
address is not anonymisation: the space of real email addresses is small enough
to enumerate, so an unkeyed digest is reversible by anyone with a wordlist. When
no key is configured this module raises rather than quietly producing something
that looks protected and is not. Getting this wrong is the difference between a
de-identified dataset and a breach with extra steps.

**What remains identifiable is reported.** Removing direct identifiers does not
make a dataset anonymous — a birth date, a postcode and a gender identify most
individuals uniquely. The report names surviving quasi-identifier combinations
rather than implying the job is finished.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: The key never appears in a frame, a report, or a log line.
PII_KEY_ENV = "AVALOKA_PII_KEY"

#: Below this share of matching values, a pattern hit is noise rather than a
#: column of that kind.
VALUE_MATCH_THRESHOLD = 0.80

#: Sampled rows per column. Classification does not need the whole frame, and
#: reading less of it is the point.
CLASSIFY_SAMPLE_ROWS = 1000


class PIIKind(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    NATIONAL_ID = "national_id"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    PERSON_NAME = "person_name"
    DATE_OF_BIRTH = "date_of_birth"
    POSTCODE = "postcode"
    ADDRESS = "address"
    GENDER = "gender"
    ACCOUNT_ID = "account_id"
    NONE = "none"


class Sensitivity(str, Enum):
    DIRECT = "direct"        # identifies a person on its own
    QUASI = "quasi"          # identifies in combination with others
    NOT_PII = "not_pii"


#: Which kinds identify a person by themselves.
_DIRECT = frozenset({PIIKind.EMAIL, PIIKind.PHONE, PIIKind.NATIONAL_ID,
                     PIIKind.CREDIT_CARD, PIIKind.ACCOUNT_ID})
_QUASI = frozenset({PIIKind.PERSON_NAME, PIIKind.DATE_OF_BIRTH, PIIKind.POSTCODE,
                    PIIKind.ADDRESS, PIIKind.GENDER, PIIKind.IP_ADDRESS})


class Strategy(str, Enum):
    PSEUDONYMISE = "pseudonymise"   # keyed, deterministic, join-preserving
    TOKENISE = "tokenise"           # keyed, format-preserving (keeps last 4)
    GENERALISE = "generalise"       # DOB → age band, postcode → district
    REDACT = "redact"               # constant; irreversible, no utility
    KEEP = "keep"


@dataclass
class ColumnFinding:
    column: str
    kind: PIIKind
    sensitivity: Sensitivity
    confidence: float
    evidence: str
    recommended: Strategy

    def as_dict(self) -> Dict[str, Any]:
        return {"column": self.column, "kind": self.kind.value,
                "sensitivity": self.sensitivity.value,
                "confidence": round(self.confidence, 3), "evidence": self.evidence,
                "recommended": self.recommended.value}


@dataclass
class PIIReport:
    findings: List[ColumnFinding] = field(default_factory=list)
    applied: Dict[str, str] = field(default_factory=dict)
    residual_risk: List[str] = field(default_factory=list)
    scanned_columns: int = 0

    @property
    def direct_identifiers(self) -> List[ColumnFinding]:
        return [f for f in self.findings if f.sensitivity is Sensitivity.DIRECT]

    @property
    def contains_pii(self) -> bool:
        return any(f.sensitivity is not Sensitivity.NOT_PII for f in self.findings)

    def as_dict(self) -> Dict[str, Any]:
        return {"scanned_columns": self.scanned_columns,
                "contains_pii": self.contains_pii,
                "n_direct_identifiers": len(self.direct_identifiers),
                "findings": [f.as_dict() for f in self.findings],
                "applied": dict(self.applied),
                "residual_risk": list(self.residual_risk)}


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_PHONE = re.compile(r"^\+?[\d][\d\s().-]{6,18}\d$")
#: A plain decimal renders as digits and a dot — the same alphabet as the phone
#: pattern — so 20.085536923187668 matches it. Numeric columns must be excluded
#: or a revenue column is classified as a direct identifier and pseudonymised
#: into noise.
_PLAIN_NUMBER = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$")
_IPV4 = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_DIGITS = re.compile(r"\D")
_POSTCODE_UK = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$", re.I)
_POSTCODE_US = re.compile(r"^\d{5}(-\d{4})?$")
_DATE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{1,2}[-/]\d{1,2}[-/]\d{4}$")

#: Column-name hints. A name is corroboration, never proof — a column called
#: "contact" may hold a team name — so a name hit still requires the values to
#: agree unless the values are unreadable.
_NAME_HINTS: Tuple[Tuple[re.Pattern, PIIKind], ...] = (
    (re.compile(r"e[-_]?mail", re.I), PIIKind.EMAIL),
    (re.compile(r"phone|mobile|msisdn|telephone", re.I), PIIKind.PHONE),
    # `\bssn\b` does NOT match inside `customer_ssn`: underscore is a word
    # character, so there is no boundary between "r_" and "ssn".
    (re.compile(r"(?:^|[^a-z])ssn(?:$|[^a-z])|social_?security|national_?id|aadhaar|nino",
                re.I), PIIKind.NATIONAL_ID),
    (re.compile(r"credit_?card|card_?number|\bpan\b|ccnum", re.I), PIIKind.CREDIT_CARD),
    (re.compile(r"ip_?addr|client_?ip", re.I), PIIKind.IP_ADDRESS),
    (re.compile(r"first_?name|last_?name|surname|full_?name|customer_?name|\bname\b", re.I),
     PIIKind.PERSON_NAME),
    (re.compile(r"birth|\bdob\b", re.I), PIIKind.DATE_OF_BIRTH),
    (re.compile(r"post_?code|zip_?code|postal", re.I), PIIKind.POSTCODE),
    (re.compile(r"address|street|addr_?line", re.I), PIIKind.ADDRESS),
    (re.compile(r"gender|\bsex\b", re.I), PIIKind.GENDER),
    (re.compile(r"account_?(id|number)|customer_?id|user_?id|member_?id", re.I),
     PIIKind.ACCOUNT_ID),
)


def _luhn_valid(number: str) -> bool:
    """Credit-card checksum. Without it, any 16-digit id reads as a card number."""
    digits = [int(c) for c in _DIGITS.sub("", number)]
    if not 12 <= len(digits) <= 19:
        return False
    total, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _match_rate(values: Sequence[str], predicate) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if predicate(v)) / len(values)


def classify_column(name: str, values: Sequence[Any]) -> ColumnFinding:
    """Classify one column from its name and a sample of its values."""
    text = [str(v).strip() for v in values if v is not None and str(v).strip()]
    sample = text[:CLASSIFY_SAMPLE_ROWS]

    # Value evidence first: it is the stronger signal, and a matching value
    # pattern is a fact where a column name is only a hint.
    checks = (
        (PIIKind.EMAIL, lambda v: bool(_EMAIL.match(v)), "values are email addresses"),
        (PIIKind.IP_ADDRESS, lambda v: bool(_IPV4.match(v)), "values are IPv4 addresses"),
        (PIIKind.CREDIT_CARD, lambda v: _luhn_valid(v), "values pass the Luhn checksum"),
        # A date is digits and separators of roughly phone length: "1985-04-12"
        # matches a naive phone pattern. Exclude dates before testing for phones,
        # or every date_of_birth column is classified as a direct identifier and
        # pseudonymised instead of generalised into an age band.
        (PIIKind.PHONE,
         lambda v: bool(_PHONE.match(v)) and not _DATE.match(v)
                   and not _PLAIN_NUMBER.match(v),
         "values look like phone numbers"),
    )
    for kind, predicate, why in checks:
        rate = _match_rate(sample, predicate)
        if rate >= VALUE_MATCH_THRESHOLD:
            return _finding(name, kind, rate, f"{rate:.0%} of sampled {why}")

    for pattern, kind in _NAME_HINTS:
        if not pattern.search(name):
            continue
        # A name hint plus agreeing values is strong; a name hint alone is
        # reported at lower confidence rather than ignored, because a column
        # called `ssn` deserves attention even when its sample is empty.
        corroboration = ""
        confidence = 0.6
        if kind is PIIKind.DATE_OF_BIRTH and _match_rate(sample, lambda v: bool(_DATE.match(v))) >= VALUE_MATCH_THRESHOLD:
            confidence, corroboration = 0.95, " and values parse as dates"
        elif kind is PIIKind.POSTCODE and _match_rate(
                sample, lambda v: bool(_POSTCODE_UK.match(v) or _POSTCODE_US.match(v))) >= VALUE_MATCH_THRESHOLD:
            confidence, corroboration = 0.95, " and values match a postcode format"
        elif kind is PIIKind.GENDER and len(set(v.lower() for v in sample)) <= 4:
            confidence, corroboration = 0.85, " and the column is low-cardinality"
        return _finding(name, kind, confidence,
                        f"column name matches {kind.value}{corroboration}")

    return ColumnFinding(name, PIIKind.NONE, Sensitivity.NOT_PII, 1.0,
                         "no identifier pattern in name or values", Strategy.KEEP)


def _finding(name: str, kind: PIIKind, confidence: float, evidence: str) -> ColumnFinding:
    if kind in _DIRECT:
        sensitivity = Sensitivity.DIRECT
        strategy = Strategy.TOKENISE if kind is PIIKind.CREDIT_CARD else Strategy.PSEUDONYMISE
    elif kind in _QUASI:
        sensitivity = Sensitivity.QUASI
        strategy = (Strategy.GENERALISE
                    if kind in (PIIKind.DATE_OF_BIRTH, PIIKind.POSTCODE)
                    else Strategy.PSEUDONYMISE if kind is PIIKind.PERSON_NAME
                    else Strategy.KEEP)
    else:
        sensitivity, strategy = Sensitivity.NOT_PII, Strategy.KEEP
    return ColumnFinding(name, kind, sensitivity, confidence, evidence, strategy)


def scan_dataframe(df, *, columns: Optional[Sequence[str]] = None) -> PIIReport:
    """Classify every column, then report what remains identifiable."""
    report = PIIReport()
    cols = list(columns or df.columns)
    report.scanned_columns = len(cols)
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        # A numeric column is not an email, an IP or a phone number. Skipping the
        # value patterns here also stops a float from matching the phone alphabet.
        numeric = False
        try:
            import pandas as _pd
            numeric = bool(_pd.api.types.is_numeric_dtype(series))
        except Exception:  # noqa: BLE001
            numeric = False
        values = [] if numeric else series.dropna().head(CLASSIFY_SAMPLE_ROWS).tolist()
        finding = classify_column(str(col), values)
        if finding.kind is not PIIKind.NONE:
            report.findings.append(finding)

    quasi = [f.column for f in report.findings if f.sensitivity is Sensitivity.QUASI]
    if len(quasi) >= 2:
        report.residual_risk.append(
            f"{len(quasi)} quasi-identifiers remain ({', '.join(quasi[:6])}). In "
            f"combination these identify most individuals uniquely even after direct "
            f"identifiers are removed; consider generalising or dropping some.")
    return report


# --------------------------------------------------------------------------- #
# De-identification — keyed, deterministic, vault-free
# --------------------------------------------------------------------------- #

class MissingPIIKey(RuntimeError):
    """No key configured. Refused rather than falling back to an unkeyed digest."""


_KEYED_STRATEGIES = frozenset({Strategy.PSEUDONYMISE, Strategy.TOKENISE})


def pii_key_required_columns(report: PIIReport) -> List[str]:
    """Return sensitive columns whose recommended transform needs the PII key."""
    return [
        finding.column
        for finding in report.findings
        if finding.recommended in _KEYED_STRATEGIES
    ]


def require_pii_key(report: PIIReport, *, key: Optional[bytes] = None) -> List[str]:
    """Require keyed protection when selected columns contain supported PII.

    Generalisation-only findings such as dates of birth and postcodes do not
    need a secret. This guard applies only to transformations that use HMAC
    pseudonymisation or format-preserving tokenisation.
    """
    columns = pii_key_required_columns(report)
    if not columns:
        return []
    if key or (os.getenv(PII_KEY_ENV) or "").strip():
        return columns
    labels = ", ".join(repr(column) for column in columns)
    raise MissingPIIKey(
        f"{PII_KEY_ENV} is not set. Sensitive training column(s) {labels} require "
        "keyed pseudonymisation or tokenisation before model training can start. "
        "Configure the PII key or remove those columns from the training plan."
    )


def deidentification_contract(report: PIIReport) -> Dict[str, Dict[str, str]]:
    """Return the safe, serializable transform contract for a PII report.

    The contract deliberately contains only column names, PII kinds, and
    transformation strategies.  It never contains the key or any source value,
    so it is safe to persist alongside the model and reuse during inference.
    """
    return {
        finding.column: {
            "kind": finding.kind.value,
            "strategy": finding.recommended.value,
        }
        for finding in report.findings
        if finding.recommended is not Strategy.KEEP
    }


def _key(explicit: Optional[bytes] = None) -> bytes:
    if explicit:
        return explicit
    raw = (os.getenv(PII_KEY_ENV) or "").strip()
    if not raw:
        raise MissingPIIKey(
            f"{PII_KEY_ENV} is not set. Avaloka refuses to pseudonymise without a key: an "
            f"unkeyed hash of an email address or phone number is reversible by anyone "
            f"with a wordlist, because the space of real values is small enough to "
            f"enumerate. That is not de-identification, and shipping it as though it were "
            f"is worse than leaving the column alone.")
    return raw.encode("utf-8")


def pseudonym(value: Any, *, salt: str = "", key: Optional[bytes] = None,
              length: int = 16) -> str:
    """A stable, keyed pseudonym. Same input and key always give the same output.

    ``salt`` should be the column name, so the same email under two columns does
    not produce the same token and leak that they are equal across contexts.
    """
    digest = hmac.new(_key(key), f"{salt}\x00{value}".encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:length]


def tokenise_preserving_tail(value: Any, *, keep: int = 4, salt: str = "",
                             key: Optional[bytes] = None) -> str:
    """Format-preserving token that keeps the last *keep* characters.

    A card ending 4242 stays recognisable to the person it belongs to and to
    reconciliation, while the rest is unrecoverable without the key.
    """
    text = str(value)
    if len(text) <= keep:
        return pseudonym(text, salt=salt, key=key, length=len(text) or 4)
    head = pseudonym(text[:-keep], salt=salt, key=key, length=max(len(text) - keep, 4))
    return f"{head[:len(text) - keep]}{text[-keep:]}"


def generalise_date_to_age_band(value: Any, *, reference_year: int, width: int = 10
                                ) -> Optional[str]:
    """A birth date becomes an age band. Bands, not ages: an exact age plus a
    postcode is close to an identifier again."""
    import pandas as pd
    stamp = pd.to_datetime(value, errors="coerce")
    if stamp is None or pd.isna(stamp):
        return None
    age = max(reference_year - int(stamp.year), 0)
    low = (age // width) * width
    return f"{low}-{low + width - 1}"


def generalise_postcode(value: Any, *, keep: int = 3) -> Optional[str]:
    text = str(value).strip()
    return text[:keep].upper() if text else None


def transform_pii_value(
    value: Any,
    *,
    column: str,
    kind: PIIKind | str,
    strategy: Strategy | str,
    key: Optional[bytes] = None,
    reference_year: int = 2026,
) -> Any:
    """Apply one persisted PII transform to a raw training/inference value."""
    resolved_strategy_early = (
        strategy if isinstance(strategy, Strategy) else Strategy(str(strategy))
    )
    # REDACT is checked BEFORE the null guard on purpose. Every other strategy
    # must leave a missing value missing -- pseudonymising a null would mint a
    # stable token for "no value", inventing data and letting absence be joined
    # on. Redaction is the opposite case: the column has to be uniformly opaque,
    # because which rows HELD a value is itself informative, and letting nulls
    # through hands back a bit of exactly what redaction removes.
    if resolved_strategy_early is Strategy.REDACT:
        return "__redacted__"
    try:
        import pandas as pd
        if value is None or bool(pd.isna(value)):
            return None
    except (ImportError, TypeError, ValueError):
        if value is None:
            return None

    resolved_kind = kind if isinstance(kind, PIIKind) else PIIKind(str(kind))
    resolved_strategy = (
        strategy if isinstance(strategy, Strategy) else Strategy(str(strategy))
    )
    if resolved_strategy is Strategy.KEEP:
        return value
    if resolved_strategy is Strategy.PSEUDONYMISE:
        return pseudonym(value, salt=column, key=key)
    if resolved_strategy is Strategy.TOKENISE:
        return tokenise_preserving_tail(value, salt=column, key=key)
    if resolved_strategy is Strategy.GENERALISE:
        if resolved_kind is PIIKind.DATE_OF_BIRTH:
            return generalise_date_to_age_band(value, reference_year=reference_year)
        if resolved_kind is PIIKind.POSTCODE:
            return generalise_postcode(value)
        return pseudonym(value, salt=column, key=key)
    if resolved_strategy is Strategy.REDACT:
        return "__redacted__"
    return value


def apply_deidentification(df, report: PIIReport, *,
                           overrides: Optional[Dict[str, Strategy]] = None,
                           key: Optional[bytes] = None,
                           reference_year: int = 2026):
    """Apply each finding's strategy. Returns a new frame; the original is untouched."""
    out = df.copy()
    overrides = overrides or {}
    for finding in report.findings:
        col = finding.column
        if col not in out.columns:
            continue
        strategy = overrides.get(col, finding.recommended)
        if strategy is Strategy.KEEP:
            continue
        out[col] = out[col].map(
            lambda value: transform_pii_value(
                value,
                column=col,
                kind=finding.kind,
                strategy=strategy,
                key=key,
                reference_year=reference_year,
            )
        )
        report.applied[col] = strategy.value
    return out
