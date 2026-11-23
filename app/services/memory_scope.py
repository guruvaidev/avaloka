# app/services/memory_scope.py
#
# Which thread may see a piece of learned memory.
#
# rc12, clean stack: a fresh thread asking "average age of patients with
# stroke" was handed, through the user-scoped Chroma signature, what earlier
# threads had asked for -- "Return the work_type with the highest stroke
# rate ...", "They prefer concise numeric answers about missing data". The
# planner computed top_work_type alongside the age (age3), a contract failed
# because the plan promised average_age and the code returned work_type columns
# (age4), and a verified sentence quoted the total missing-BMI count as one
# work_type's (work4).
#
# The rule: only general, user-level preferences and facts about the dataset
# cross threads. Everything else learned in a thread stays in it.
#
#   kind         scope      examples
#   preference   user       answer style ("concise"), units, chart type, code style
#   domain       dataset    "Domain insight: ...", "Column 'stroke' is 1 when ..."
#   task         thread     "Return the work_type with ...", "compute ...", anything else
#
# The kind is decided when the item is WRITTEN, deterministically, and stored
# with it (as the key it is filed under), so serving is a key lookup and no LLM
# judges at read time what may cross. The learner's output is free text, so it
# is classified here by what it says:
#
#   * A preference is recognised positively: it states how answers should look
#     (style, units, number format, chart type, code style) and names nothing
#     about a particular question -- no imperative task verb at the start, no
#     column or identifier, no topic ("answers about missing data").
#   * A domain fact describes the data: the schema hint or a column's meaning.
#   * Anything else is a task instruction for the thread it came from.
#
# Why a whitelist and not a blacklist: the two mistakes cost different things.
# A preference wrongly kept in its thread costs one style hint in a new thread
# (the default style still applies). A task instruction wrongly let across
# changes what another question computes, which is what rc12 showed three times
# in fifteen turns. So when in doubt an item is a task, and stays where it was
# learned.
#
# Items written before this rule (plain user- or session-keyed rows in Chroma,
# Milvus and Redis) carry no kind. They are treated as task items of an unknown
# thread: never served to any thread.
#
# Deletion (1.6 hotfix) goes by the same keys: a thread's rows are the ones
# under thread_key; a user's are the bare user key and every key built from it
# (owns_key), so deleting a user removes the pre-rule rows too.
#
# Kept free of heavy imports: the memory plane and tests import it.

import hashlib
import re
from typing import Any, Iterable, List, Optional, Tuple

PREFERENCE = "preference"
DOMAIN = "domain"
TASK = "task"

# How a row's key names its scope. Rows keyed by a bare user or session id are
# the pre-rule ones: no key below can equal one, so they are never read.
_SEP = "#"


# Milvus files rows under a VARCHAR(255) session_id; a longer key is cut to a
# stable digest of its tail rather than failing the insert.
_MAX_KEY = 255


def _fit(key: str) -> str:
    if len(key) <= _MAX_KEY:
        return key
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return key[:_MAX_KEY - len(digest) - 1] + "~" + digest


def thread_key(scope: str, thread: str) -> str:
    return _fit(f"{scope}{_SEP}thread:{thread}")


def preference_key(scope: str) -> str:
    return _fit(f"{scope}{_SEP}pref")


def dataset_key(scope: str, dataset: str) -> str:
    return _fit(f"{scope}{_SEP}dataset:{dataset}")


def storage_key(kind: str, scope: str, thread: str, dataset: Optional[str]) -> str:
    """The key an item of this kind is filed under. A domain fact with no known
    dataset cannot be matched to one later, so it stays in its thread."""
    if kind == PREFERENCE:
        return preference_key(scope)
    if kind == DOMAIN and dataset:
        return dataset_key(scope, dataset)
    return thread_key(scope, thread)


def owns_key(key: Any, scope: Any) -> bool:
    """Is a stored row's key one of this scope's: the bare scope (a row written
    before the scope rule) or a key built from it above? Deletion checks every
    row with it before removing it."""
    key, scope = str(key or ""), str(scope or "")
    return bool(scope) and (key == scope or key.startswith(f"{scope}{_SEP}"))


def readable_keys(scope: str, thread: str, dataset: Optional[str]) -> List[str]:
    """Every key a turn of this thread may read: its own thread, the user's
    preferences, and facts about the dataset it is analysing."""
    keys = [thread_key(scope, thread), preference_key(scope)]
    if dataset:
        keys.append(dataset_key(scope, dataset))
    return keys


_DATASET_FACT_PREFIXES = ("Domain insight",)

# A column's meaning: "Column 'stroke' is 1 when ...", "bmi is measured in kg/m2",
# "The work_type field holds ...".
_COLUMN_FACT_RE = re.compile(
    r"""^\s*(?:the\s+)?
        (?:(?:column|field|variable)\s+['"`]?[\w .-]+?['"`]?
         |['"`][\w .-]+['"`]\s+(?:column|field|variable)
         |[\w-]*_[\w-]*\s+(?:column|field)?)
        \s*(?:is|are|means|holds|contains|represents|stores|records|encodes|counts|=)\b""",
    re.IGNORECASE | re.VERBOSE,
)

# How the answer should look: the vocabulary of style, units, number format,
# charts and code style. A preference must use at least one of these.
_STYLE_RE = re.compile(
    r"""\b(?:concise(?:ly)?|brief(?:ly)?|short|terse|succinct(?:ly)?|direct(?:ly)?|to\s+the\s+point
        |detailed|verbose|step[- ]by[- ]step|plain\s+(?:language|english)|explain\w*
        |bullet(?:s|ed)?|tables?\s+format|markdown|narrative|summary\s+first
        |decimal(?:s|\s+places?)?|round(?:ed|ing)?|significant\s+(?:figures|digits)
        |percent(?:age)?s?|units?|metric|imperial|currency|usd|eur|gbp|inr|dollars?|euros?
        |celsius|fahrenheit|kilo(?:gram|metre|meter)s?|miles?|thousands|millions|billions
        |iso\s*8601|date\s+format|time\s*zones?|utc
        |charts?|graphs?|plots?|visuali[sz]\w*|bar|line|pie|scatter|histograms?|heatmaps?
        |colou?rs?|palette|legend|axis|axes
        |pandas|polars|numpy|duckdb|pyspark|sql|code\s+style|clean\s+code|comments?|docstrings?
        |type\s+hints?|vectori[sz]ed)\b""",
    re.IGNORECASE | re.VERBOSE,
)

# An instruction for one question: an imperative task verb up front
# ("Return the work_type ...", "Always compute ...", "Answer with ...").
_TASK_VERB_RE = re.compile(
    r"""^\s*(?:(?:always|please|then|and|also|just)\s+)*
        (?:return|compute|calculate|find|count|list|show|give|provide|answer|report
        |identify|determine|filter|group|sort|order|rank|select|get|fetch|output|include
        |exclude|summari[sz]e|aggregate|average|sum|compare|predict|train|perform|run
        |query|join|merge|drop|remove|add|create|build|generate|produce|display|break\s+down
        |look\s+up|check|verify|use\s+(?:the\s+)?\w+\s+column)\b""",
    re.IGNORECASE | re.VERBOSE,
)

# The subject of a particular question: answers/queries/counts ABOUT something.
_TOPIC_RE = re.compile(
    r"""\b(?:answers?|queries|query|questions?|analys[ie]s|counts?|values?|results?
        |numbers?|figures?|metrics?|rates?|records?|rows?|statistics|stats|insights?)
        \s+(?:about|on|of|for|regarding|concerning|in|from|by|per|across|among)\s+""",
    re.IGNORECASE | re.VERBOSE,
)

# Names something in the data: an identifier (work_type, avgGlucose) or a
# quoted name.
_IDENTIFIER_RE = re.compile(
    r"""\b[A-Za-z]+_[A-Za-z0-9_]+\b|\b[a-z]+[A-Z][A-Za-z0-9]*\b|['"`][^'"`]{1,64}['"`]""",
    re.VERBOSE,
)


_FACT_VERBS = r"(?:is|are|means|holds|contains|represents|stores|records|encodes|counts|=)"


def _states_a_column_meaning(text: str, columns: Iterable[str]) -> bool:
    """'bmi is measured in kg/m2': a known column, then what it is."""
    for col in columns or ():
        col = str(col or "").strip()
        if len(col) >= 2 and re.match(
                rf"\s*(?:the\s+)?['\"`]?{re.escape(col)}['\"`]?\s+(?:column\s+|field\s+)?{_FACT_VERBS}\b",
                text, re.IGNORECASE):
            return True
    return False


def _names_a_column(text: str, columns: Iterable[str]) -> bool:
    low = text.lower()
    for col in columns or ():
        col = str(col or "").strip().lower()
        if len(col) >= 2 and re.search(rf"(?<![\w]){re.escape(col)}(?![\w])", low):
            return True
    return False


def classify(text: Any, columns: Iterable[str] = ()) -> str:
    """preference, domain or task -- see the module comment. `columns` are the
    names of the dataset's columns where known; naming one makes an item
    specific to a question unless it states what the column means."""
    s = str(text or "").strip()
    if not s:
        return TASK
    if (s.startswith(_DATASET_FACT_PREFIXES) or _COLUMN_FACT_RE.search(s)
            or _states_a_column_meaning(s, columns)):
        return DOMAIN
    if (_STYLE_RE.search(s)
            and not _TASK_VERB_RE.search(s)
            and not _TOPIC_RE.search(s)
            and not _IDENTIFIER_RE.search(s)
            and not _names_a_column(s, columns)):
        return PREFERENCE
    return TASK


# Where a merged signature splits into the sentences it was made of (as in
# memory_actions: Chroma joins stored signatures with a space, and the LLM
# rarely ends one with a full stop).
_SIGNATURE_SPLIT_RE = re.compile(
    r"(?<=[.!?;])\s+|\n+|(?<=[a-z0-9)\]'\"])\s+(?=[A-Z][a-z])")


def split_sentences(text: Any) -> List[str]:
    return [p.strip() for p in _SIGNATURE_SPLIT_RE.split(str(text or "")) if p and p.strip()]


def split_by_kind(text: Any, columns: Iterable[str] = ()) -> Tuple[str, str]:
    """(preference part, the rest) of a learned signature, each joined back
    into one string ('' when empty). A domain sentence in a signature is kept
    with the rest: the signature is about style, and the thread that wrote it
    still sees it."""
    pref, rest = [], []
    for part in split_sentences(text):
        (pref if classify(part, columns) == PREFERENCE else rest).append(part)
    return " ".join(pref), " ".join(rest)
