"""The membership test: remove known column, table and dataset names from a question.

This is the control on the path that leaves the cluster. Pattern redaction
(``redact.py``) has to guess what a sensitive string looks like, and a column
name looks like any other word -- ``patient_hiv_status`` matches no pattern.
This does not guess. The deployment knows the names of the analyst's registered
datasets, so every token of the question is tested for membership in that set.

It fails closed, in three ways:

* **No vocabulary, no text.** If the names are unavailable, incomplete or empty,
  ``strip_names`` returns ``None`` and the caller exports no question text.
* **A name that is also an ordinary word is still a name.** A column called
  ``status`` removes the word "status" from every exported question.
* **A token that contains a name goes whole.** ``region_id``, ``regions`` and
  ``byRegion`` are all removed when ``region`` is a column.

Those last two pull against each other -- stripping "status" costs analytic
value, not stripping ``byRegion`` leaks a column -- and this module resolves it
one way: **over-strip, and count precisely.** Every removal is recorded under
one of three counters (``HIT_KINDS``), so a reader of the data can tell a real
schema reference from the English word "status", and the privacy claim and the
analytic cost can both be measured after the fact.

**STATUS: not exercised in production.** No question text is exported
(``export.EXPORT_QUESTION_TEXT`` is ``False``), so nothing calls this module
outside its tests. It is kept, tested, as the mechanism that would have to
guard that path if it were ever opened -- see "If question-text export is ever
enabled" in docs/analytics/DESIGN.md for what must be proven first.

What it cannot do: remove a name the deployment does not know. A column of a
dataset that was never registered here, typed from memory, passes through.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

MASK = "[name]"

#: Below this length a name is matched as a whole word only. ``id`` as a column
#: must not remove "valid", "video" and "paid"; ``region`` may remove "regions".
MIN_SUBSTRING_LEN = 4

_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+|[^\W\d_]+")
_CHUNK = re.compile(r"\S+")
_RUN = re.compile(r"\[name\](?:\s+\[name\])+")


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def words(text: str) -> Tuple[str, ...]:
    """Split on punctuation, underscores and camelCase; casefold. ``orderDate`` -> (order, date)."""
    return tuple(w.casefold() for w in _WORD.findall(_fold(text)))


class Vocabulary:
    """The set of names to remove, in the forms they are matched in."""

    def __init__(self, names: Iterable[Any]) -> None:
        self.tuples: Set[Tuple[str, ...]] = set()
        for name in names:
            if not isinstance(name, str):
                continue
            # ``public.orders`` and ``sales_2024.csv``: each dotted part is a
            # name too, because analysts type the unqualified form.
            for part in {name, *name.split(".")}:
                key = words(part)
                if key:
                    self.tuples.add(key)
        self.squashed: FrozenSet[str] = frozenset("".join(t) for t in self.tuples)
        self.long: Tuple[str, ...] = tuple(s for s in self.squashed if len(s) >= MIN_SUBSTRING_LEN)
        self.max_len = max((len(t) for t in self.tuples), default=0)

    def __bool__(self) -> bool:
        return bool(self.tuples)

    def hits_word(self, word: str) -> bool:
        if word in self.squashed:
            return True
        # Plurals of short names: "ids" when the column is "id".
        for suffix in ("es", "s"):
            if word.endswith(suffix) and word[: -len(suffix)] in self.squashed:
                return True
        return any(name in word for name in self.long)


#: How each removal is classified, so the data can be audited afterwards.
#:
#: ``exact``        the token is, or contains on a word boundary, a known name
#:                  and looks like a schema reference: ``patient_hiv_status``,
#:                  ``region_id``, ``byRegion``, "order date" for ``order_date``.
#: ``common_word``  the token is a plain word that is both a known name and an
#:                  ordinary English word: "status", "region". Removed all the
#:                  same -- but this is the counter that measures over-stripping.
#: ``substring``    a known name sits inside a longer word with no boundary:
#:                  "regions", "subregional", "statuses".
HIT_KINDS = ("exact", "common_word", "substring")

#: Ordinary words that are also plausible column names. Membership here changes
#: only which counter a removal is recorded under, never whether it is removed.
COMMON_WORDS = frozenset("""
account active address age amount area average balance bank base batch brand budget buyer call
campaign capacity case cash category change channel city class client code color comment company
cost count country county created credit currency customer date day days deal delivery department
description device discount distance district division duration email employee end error event
expense failed feature fee field file first flag gender goal grade group growth height hour id
income index industry item job key label language last level limit line link list location loss
manager margin market max mean median member message method min minute model month name net note
number office order owner page paid parent part partner payment percent period person phone plan
point population position price priority product profit project quantity quarter rank rate rating
ratio reason record region rep result return revenue role row sale sales score season section
segment seller session sex shift size source stage start state status stock store street subject
success supplier tag target tax team term test text time title total type unit units updated user
value variant vendor version visit volume week weight year zip zone
""".split())

_PLAIN_WORD = re.compile(r"[^\W\d_]+")


def _aligned(chunk_words: Tuple[str, ...], vocab: "Vocabulary") -> bool:
    """Whether a known name occupies whole words of this token (boundary-aligned)."""
    n = len(chunk_words)
    if "".join(chunk_words) in vocab.squashed:
        return True
    for size in range(1, min(n, vocab.max_len) + 1):
        for i in range(n - size + 1):
            window = chunk_words[i:i + size]
            if window in vocab.tuples or "".join(window) in vocab.squashed:
                return True
    return False


def strip_names(text: Optional[str], names: Optional[Iterable[Any]]) -> Optional[Tuple[str, Dict[str, int]]]:
    """Replace every token that is, or contains, a known name with ``[name]``.

    Over-strips by design, and counts precisely: returns ``(stripped_text,
    {"exact": n, "common_word": n, "substring": n})``. Returns ``None`` when
    the text must not be exported at all: no text, no vocabulary, or an empty
    vocabulary.
    """
    if not isinstance(text, str) or not text.strip() or names is None:
        return None
    try:
        vocab = names if isinstance(names, Vocabulary) else Vocabulary(names)
        if not vocab:
            return None

        chunks = [m.group(0) for m in _CHUNK.finditer(text)]
        flat: List[Tuple[str, int]] = []           # (word, index of the chunk it came from)
        kinds: Dict[int, str] = {}
        for index, raw in enumerate(chunks):
            chunk_words = words(raw)
            flat.extend((w, index) for w in chunk_words)
            if not chunk_words:
                continue
            if _aligned(chunk_words, vocab):
                core = raw.strip(".,;:!?\"'()[]{}<>")
                plain = len(chunk_words) == 1 and _PLAIN_WORD.fullmatch(core) is not None
                kinds[index] = "common_word" if plain and chunk_words[0] in COMMON_WORDS else "exact"
            elif vocab.hits_word("".join(chunk_words)) or any(vocab.hits_word(w) for w in chunk_words):
                kinds[index] = "substring"

        # Names typed as several words: "order date" for ``order_date``. A
        # multi-word match is a schema reference whatever its parts looked like.
        for n in range(min(vocab.max_len, len(flat)), 1, -1):
            for i in range(len(flat) - n + 1):
                window = flat[i:i + n]
                gram = tuple(w for w, _ in window)
                if len({index for _, index in window}) > 1 and (
                        gram in vocab.tuples or "".join(gram) in vocab.squashed):
                    for _, index in window:
                        kinds[index] = "exact"

        out: List[str] = []
        for index, raw in enumerate(chunks):
            if index not in kinds:
                out.append(raw)
                continue
            lead = re.match(r"^[^\w\[\]]*", raw).group(0)       # keep "(" and the like
            trail = re.search(r"[^\w\[\]]*$", raw).group(0)
            out.append(f"{lead}{MASK}{trail}")
        counts = {kind: 0 for kind in HIT_KINDS}
        for kind in kinds.values():
            counts[kind] += 1
        return _RUN.sub(MASK, " ".join(out)), counts
    except Exception:
        return None


def names_from_session_blobs(blobs: Iterable[Any], user_id: str,
                             expected: Optional[int] = None) -> Optional[Set[str]]:
    """Collect column, table and dataset names from a user's dataset sessions.

    Returns ``None`` -- vocabulary unavailable -- if any session cannot be read
    or parsed. A vocabulary built from *some* of the datasets would let the
    names of the others through, so a partial answer is treated as no answer.
    """
    names: Set[str] = set()
    count = 0
    for blob in blobs:
        count += 1
        if not blob:
            return None
        try:
            sess = json.loads(blob) if isinstance(blob, (str, bytes, bytearray)) else blob
        except Exception:
            return None
        if not isinstance(sess, dict):
            return None
        if sess.get("user_id") != user_id:
            continue
        for key in ("alias", "filename", "table_name", "source_table"):
            _collect(sess.get(key), names)
        for key in ("uploaded_csv_columns", "schema", "tables"):
            _collect(sess.get(key), names)
    if expected is not None and count != expected:
        return None
    return names


def _collect(value: Any, into: Set[str], depth: int = 0) -> None:
    """Pull name-like strings out of the shapes sessions store schemas in."""
    if depth > 4 or value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[{":
            try:
                _collect(json.loads(text), into, depth + 1)
                return
            except Exception:
                pass
        if text and len(text) <= 256:
            into.add(text.rsplit("/", 1)[-1])
        return
    if isinstance(value, dict):
        # {"col": "int64"} maps names to types; {"name": "col", "type": ...}
        # describes one column. Take keys in the first shape, "name" in the second.
        if "name" in value and isinstance(value["name"], str):
            into.add(value["name"])
            for key in ("columns", "fields", "children"):
                _collect(value.get(key), into, depth + 1)
            return
        for key, inner in value.items():
            if isinstance(key, str) and key not in ("columns", "fields", "tables"):
                into.add(key)
            if isinstance(inner, (dict, list)):
                _collect(inner, into, depth + 1)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect(item, into, depth + 1)


#: Most dataset sessions read to build one analyst's vocabulary. Past this the
#: vocabulary is reported unavailable rather than built from a subset.
MAX_SESSIONS = 500


def session_store_provider(get_cache, user_sessions_key, session_key, mget):
    """Build a names provider over the API server's session store.

    NOT WIRED ANYWHERE. Tested against fakes only; it has never read a real
    session store, and nobody has confirmed that real session blobs carry names
    under the keys ``names_from_session_blobs`` reads. Wiring it into
    ``app/api/server.py`` is a step of enabling question-text export, and that
    confirmation comes first.

    Arguments are the server's own helpers, passed in so this module does not
    import the server: ``get_cache()`` -> the session cache or None,
    ``user_sessions_key(user_id)``, ``session_key(sid)``, ``mget(keys)``.
    """
    async def provider(request: Any, user_id: str) -> Optional[Set[str]]:
        cache = get_cache()
        if not cache or not user_id:
            return None
        sids = list(await cache.smembers(user_sessions_key(user_id)))
        if len(sids) > MAX_SESSIONS:
            return None
        if not sids:
            return set()
        blobs = await mget([session_key(s) for s in sids])
        return names_from_session_blobs(blobs, user_id, expected=len(sids))

    return provider
