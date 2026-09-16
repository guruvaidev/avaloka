from __future__ import annotations

import base64
import csv
import io
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
from app.agents.sampling_agent import DEFAULT_SAMPLE_MAX_ROWS
from app.services import session_service

logger = logging.getLogger("avaloka")

# ---------- Small in-proc cache ONLY for metadata ----------
# Chat history ("lc_msgs") is write-through persisted to Redis via
# persist_thread_history/hydrate_thread_history below, so it survives
# process restarts; this dict is the hot in-process working copy.
THREAD_META: Dict[str, Dict[str, Any]] = {}


# =============== Generic helpers ===============

def _filename_ext(name: Optional[str]) -> str:
    """Return lowercase extension without the leading dot."""
    if not name:
        return ""
    return Path(name).suffix.lower().lstrip(".")


def _posix(p: Union[str, Path]) -> str:
    """Normalize a path to POSIX style (for JSON/session storage)."""
    return Path(p).resolve().as_posix()


def _safe_rmtree(p: Union[str, Path]):
    """Best-effort recursive delete; logs a warning on failure."""
    try:
        shutil.rmtree(p, ignore_errors=True)
    except Exception:
        logger.warning("rmtree failed for %s", p, exc_info=True)


def _new_id() -> str:
    """UUID4 string helper."""
    return str(uuid.uuid4())


def _now_iso() -> str:
    """Current UTC timestamp in ISO8601."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _normalize_schema_for_state(
    schema_any: Union[List[str], Dict[str, str], None]
) -> Dict[str, str]:
    """
    Normalize a user/schema representation to a dict[str, str] for graph state.

    - dict is returned as-is (copy)
    - list of column names => each mapped to "" (dtype unknown; a bare column
      list carries no type information, so callers must not infer "string" from it)
    - None => empty dict
    """
    if isinstance(schema_any, dict):
        return dict(schema_any)
    if isinstance(schema_any, list):
        return {c: "" for c in schema_any}
    return {}


# NEW: tiny-sample -> CSV writer
def _write_rows_to_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write a list[dict] to CSV with header based on keys of first row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


# ---------- CSV/Files sampling helpers ----------

def _ddl_from_schema(table_name: str, schema: Dict[str, str]) -> str:
    """Generate a simple CREATE TABLE statement from a schema dict."""

    def map_type(t: str) -> str:
        t = (t or "string").lower()
        if t in ("int", "integer", "bigint", "smallint"):
            return "INTEGER"
        if t in ("float", "double", "number", "numeric", "real", "decimal"):
            return "DOUBLE"
        if t in ("bool", "boolean"):
            return "BOOLEAN"
        if t in ("date", "datetime", "timestamp"):
            return "TIMESTAMP"
        return "TEXT"

    cols_sql = ",\n  ".join(f'"{k}" {map_type(v)}' for k, v in schema.items())
    return f'CREATE TABLE "{table_name}" (\n  {cols_sql}\n);'


def _infer_type(v: Optional[str]) -> str:
    """Very light heuristic to infer column type from a probe value."""
    if v is None or str(v).strip() == "":
        return "string"
    s = str(v).strip()
    try:
        int(s)
        return "integer"
    except Exception:
        pass
    try:
        float(s)
        return "number"
    except Exception:
        pass
    if any(ch.isdigit() for ch in s) and any(sep in s for sep in ("-", "/", ".")):
        return "date"
    return "string"


def _fallback_sample_csv(path: str, max_rows: int = DEFAULT_SAMPLE_MAX_ROWS):
    import csv
    from pathlib import Path

    prefix = b""
    try:
        prefix = Path(path).read_bytes()[:4]
    except Exception:
        prefix = b""

    has_utf16_bom = prefix.startswith(b"\xff\xfe") or prefix.startswith(b"\xfe\xff")

    encodings_to_try = (["utf-16"] if has_utf16_bom else []) + [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ]



    def _open_text_guess(p: str):
        last_exc = None
        for enc in encodings_to_try:
            try:
                f = open(p, "r", encoding=enc, newline="")
                # force decode early
                f.read(4096)
                f.seek(0)
                return f, enc
            except UnicodeDecodeError as e:
                last_exc = e
                try:
                    f.close()
                except Exception:
                    pass
            except Exception as e:
                last_exc = e
        # last resort: never crash, replace bad bytes
        f = open(p, "r", encoding="utf-8", errors="replace", newline="")
        return f, "utf-8(replace)"

    f, used_encoding = _open_text_guess(path)
    try:
        # Attempt delimiter sniff (safe fallback to comma)
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
            dialect.delimiter = ","

        reader = csv.DictReader(f, dialect=dialect)
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)

        fieldnames = reader.fieldnames or []
        schema = {c: "string" for c in fieldnames}

        # Best-effort dataset id from filename "input_<dsid>.csv"
        stem = Path(path).stem
        dsid = stem[len("input_"):] if stem.startswith("input_") else stem

        ddl = _ddl_from_schema(dsid or "dataset", schema)
        return rows, schema, ddl

    finally:
        try:
            f.close()
        except Exception:
            pass


def _datauri_csv_to_records(data_uri: str) -> Optional[List[Dict[str, Any]]]:
    """
    Decode a data:text/csv;base64,... URI into list[dict] rows.

    Returns None on parse/decode failure.
    """
    if not data_uri or not data_uri.startswith("data:text/csv;base64,"):
        return None
    try:
        b64 = data_uri.split("base64,", 1)[1]
        csv_text = base64.b64decode(b64).decode()
        sio = io.StringIO(csv_text)
        reader = csv.DictReader(sio)
        return [row for row in reader]
    except Exception:
        return None


# ---------- Message history helpers ----------

def _compact_messages(msgs: List[Any]) -> List[Any]:
    """
    Drop consecutive duplicates (same role + same content) from a list of LC messages.
    """
    out: List[Any] = []
    prev_role, prev_content = None, None
    for m in msgs:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            role = None
        content = getattr(m, "content", None)
        if role == prev_role and content == prev_content:
            continue
        out.append(m)
        prev_role, prev_content = role, content
    return out


def _is_pinned_message(m: Any) -> bool:
    """
    Messages that must survive context truncation: SystemMessage instances and
    anything stamped with additional_kwargs["avaloka_pinned"]. Used so durable
    context (e.g. remembered user preferences) cannot age out of the window.
    """
    if isinstance(m, SystemMessage):
        return True
    ak = getattr(m, "additional_kwargs", None) or {}
    return bool(ak.get("avaloka_pinned"))


def _limit_messages(msgs: List[Any], n: int) -> List[Any]:
    """
    Keep pinned messages plus the last n non-pinned messages.

    Pinned messages sit in front of the window (like a system preamble) so the
    sliding cap only ever evicts ordinary conversational turns. With no pinned
    messages this is exactly the old "last n" behavior.
    """
    pinned = [m for m in msgs if _is_pinned_message(m)]
    rest = [m for m in msgs if not _is_pinned_message(m)]
    if len(rest) > n:
        rest = rest[-n:]
    return pinned + rest if pinned else rest


# ---------- Thread history helpers exposed to endpoints ----------

async def append_thread_msg(thread_id: str, role: str, content: str) -> None:
    """
    Stub for future per-thread persistence.

    Currently no-op because THREAD_META stores only LangChain messages,
    and endpoints manage that directly.
    """
    return


async def read_thread_msgs(thread_id: str) -> List[Dict[str, str]]:
    """
    Convert LangChain messages in THREAD_META[thread_id]['lc_msgs'] into
    a simple {role, content} list for the /threads/{id}/messages endpoint.
    """
    m = THREAD_META.get(thread_id, {}).get("lc_msgs", [])
    out: List[Dict[str, str]] = []
    for x in m:
        role = "user" if isinstance(x, HumanMessage) else "assistant"
        out.append({"role": role, "content": x.content})
    return out


def _ensure_thread_local(thread_id: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Ensure THREAD_META has an entry for a given thread_id.
    """
    t = THREAD_META.get(thread_id)
    if not t:
        THREAD_META[thread_id] = {"lc_msgs": [], "metadata": metadata or {}}
    elif metadata:
        THREAD_META[thread_id]["metadata"].update(metadata or {})


# ---------- Durable thread history (Redis write-through) ----------

def _k_thread_history(thread_id: str) -> str:
    """Redis key for a thread's serialized chat history."""
    return f"thread:{thread_id}:history"


def _lc_msgs_to_jsonable(msgs: List[Any]) -> List[Dict[str, Any]]:
    """
    Serialize LC messages for Redis. Superset of app.utils' role format
    (human/ai/system) plus a "pinned" flag so avaloka_pinned survives the
    round trip; unknown message types are skipped.
    """
    out: List[Dict[str, Any]] = []
    for m in msgs:
        if isinstance(m, HumanMessage):
            role = "human"
        elif isinstance(m, AIMessage):
            role = "ai"
        elif isinstance(m, SystemMessage):
            role = "system"
        else:
            continue
        item: Dict[str, Any] = {"role": role, "content": m.content}
        ak = getattr(m, "additional_kwargs", None) or {}
        if ak.get("avaloka_pinned"):
            item["pinned"] = True
        out.append(item)
    return out


def _jsonable_to_lc_msgs(items: Any) -> List[Any]:
    """Inverse of _lc_msgs_to_jsonable; tolerant of unknown/bad entries."""
    msgs: List[Any] = []
    if not isinstance(items, list):
        return msgs
    for it in items:
        if not isinstance(it, dict):
            continue
        role = it.get("role") or it.get("type")
        content = it.get("content", "")
        kwargs = {"avaloka_pinned": True} if it.get("pinned") else {}
        if role in ("human", "user"):
            msgs.append(HumanMessage(content=content, additional_kwargs=kwargs))
        elif role in ("ai", "assistant"):
            msgs.append(AIMessage(content=content, additional_kwargs=kwargs))
        elif role == "system":
            msgs.append(SystemMessage(content=content, additional_kwargs=kwargs))
    return msgs


async def hydrate_thread_history(thread_id: str) -> None:
    """
    Load a thread's chat history from Redis into THREAD_META.

    Only fills the in-process cache when it has no messages for this thread
    (fresh process after a restart/redeploy); while the process lives,
    THREAD_META stays authoritative. Never raises — history durability must
    not break request handling when Redis is degraded.

    A FAILED read is not the same as an empty one: on failure the entry is
    marked _history_unsynced so persist_thread_history won't blindly
    overwrite a durable copy it never saw (a transient Redis blip at restart
    must not erase the thread's history).
    """
    t = THREAD_META.get(thread_id)
    if t and t.get("lc_msgs"):
        return
    cache = session_service.cache
    if not cache:
        return
    try:
        raw = await cache.get(_k_thread_history(thread_id))
    except Exception as e:
        session_service._note_cache_failure(f"hydrate_thread_history({thread_id})", e)
        if t is not None:
            t["_history_unsynced"] = True
        return
    # Re-validate after the await: a concurrent request (client retry /
    # double-send) may have appended a turn while we were reading Redis —
    # never replace fresher in-process messages with the snapshot we fetched.
    t = THREAD_META.get(thread_id)
    if t and t.get("lc_msgs"):
        return
    if t is not None:
        t.pop("_history_unsynced", None)
    if not raw:
        return
    try:
        msgs = _jsonable_to_lc_msgs(json.loads(raw))
    except Exception:
        logger.warning("Discarding undecodable thread history for %s", thread_id, exc_info=True)
        return
    if not msgs:
        return
    if t is None:
        THREAD_META[thread_id] = {"lc_msgs": msgs, "metadata": {}}
    else:
        t["lc_msgs"] = msgs


async def persist_thread_history(thread_id: str) -> None:
    """
    Write THREAD_META[thread_id]['lc_msgs'] through to Redis. Never raises.

    When the entry is marked _history_unsynced (hydration failed earlier),
    first re-read the durable copy and merge it in front of the local
    messages; if Redis is still unreachable, skip the write entirely so the
    durable history is never replaced by a partial post-restart view.
    (Single-writer per thread is assumed — the deployment runs one replica.)
    """
    cache = session_service.cache
    if not cache:
        return
    tmeta = THREAD_META.get(thread_id, {})
    msgs = tmeta.get("lc_msgs") or []
    if tmeta.get("_history_unsynced"):
        try:
            raw = await cache.get(_k_thread_history(thread_id))
        except Exception as e:
            session_service._note_cache_failure(f"persist_thread_history({thread_id})", e)
            return  # still can't see the durable copy — don't clobber it
        try:
            durable = _jsonable_to_lc_msgs(json.loads(raw)) if raw else []
        except Exception:
            durable = []
        if durable:
            local_contents = {(type(m).__name__, m.content) for m in msgs}
            merged = [m for m in durable if (type(m).__name__, m.content) not in local_contents]
            msgs = _limit_messages(_compact_messages(merged + msgs), len(msgs) + len(merged))
            tmeta["lc_msgs"] = msgs
        tmeta.pop("_history_unsynced", None)
    try:
        await cache.set(
            _k_thread_history(thread_id),
            json.dumps(_lc_msgs_to_jsonable(msgs)),
            ex=session_service.settings.session_ttl_seconds,
        )
    except Exception as e:
        session_service._note_cache_failure(f"persist_thread_history({thread_id})", e)


async def delete_thread_history(thread_id: str) -> None:
    """Remove a thread's persisted history from Redis. Never raises."""
    cache = session_service.cache
    if not cache:
        return
    try:
        await cache.delete(_k_thread_history(thread_id))
    except Exception as e:
        session_service._note_cache_failure(f"delete_thread_history({thread_id})", e)


# ---------- Dataset helpers ----------

def _iso_from_ts(ts: float) -> str:
    """Convert a Unix timestamp into ISO8601 UTC string."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _schema_to_columns(schema_any: Union[List[str], Dict[str, str], None]) -> List[str]:
    """Extract just the column names from a schema representation."""
    if isinstance(schema_any, dict):
        return list(schema_any.keys())
    if isinstance(schema_any, list):
        return schema_any
    return []

def _maybe_json_load(v: Any) -> Any:
    """If v is a JSON string, decode it; otherwise return as-is."""
    if v is None:
        return None
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v
# ---------- Misc helpers ----------
# ---------- Dataset auto-routing helpers ----------

_WORD_RE = re.compile(r"[^a-z0-9]+")


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.lower()
    return _WORD_RE.sub(" ", text)


def _tokenize(text: Optional[str]) -> List[str]:
    return [t for t in _normalize_text(text).split() if len(t) > 1]


def _dataset_descriptor(sess: Dict[str, Any]) -> str:
    """
    Build a textual description of a dataset from metadata we have:
    alias, filename, columns, schema column names, etc.
    """
    parts: List[str] = []

    alias = sess.get("alias") or ""
    filename = sess.get("filename") or ""
    if alias:
        parts.append(alias)
    if filename:
        parts.append(filename)

    # Columns might be JSON string or list
    cols = _maybe_json_load(sess.get("uploaded_csv_columns")) or sess.get(
        "uploaded_csv_columns"
    )
    if isinstance(cols, list):
        parts.extend(str(c) for c in cols)

    # Also include schema column names if present
    schema_any = _maybe_json_load(sess.get("schema")) or sess.get("schema")
    if isinstance(schema_any, dict):
        parts.extend(schema_any.keys())
    elif isinstance(schema_any, list):
        parts.extend(str(c) for c in schema_any)

    return " ".join(parts)


def _score_dataset_for_query(query: str, dsid: str, sess: Dict[str, Any]) -> float:
    """
    Very lightweight scoring:

    - token overlap between query and dataset descriptor
    - strong bonus if alias/filename is mentioned directly
    - small bonus if dataset_id appears in the prompt
    """
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return 0.0

    desc = _dataset_descriptor(sess)
    d_tokens = set(_tokenize(desc))
    if not d_tokens:
        return 0.0

    overlap = q_tokens & d_tokens
    score: float = float(len(overlap))

    q_lower = query.lower()
    alias = (sess.get("alias") or "").lower()
    filename = (sess.get("filename") or "").lower()

    if alias and alias in q_lower:
        score += 5.0
    if filename and filename in q_lower:
        score += 3.0

    if dsid.lower() in q_lower:
        score += 1.0

    return score


def _pick_best_dataset_id(
    query: str, candidates: List[Tuple[str, Dict[str, Any]]]
) -> Optional[str]:
    """
    Decide which dataset to treat as PRIMARY for this query.

    candidates: [(dataset_id, session_dict), ...]
    Returns dataset_id or None.
    """
    if not candidates:
        return None

    best_id: Optional[str] = None
    best_score: float = -1.0

    for dsid, sess in candidates:
        s = _score_dataset_for_query(query, dsid, sess)
        if s > best_score:
            best_id, best_score = dsid, s

    # fallback: if nothing matched, just use the first candidate
    if best_id is None and candidates:
        best_id = candidates[0][0]

    return best_id

def is_database_related_query(content: str) -> bool:
    """
    Simple heuristic used to decide if a query looks database-related.
    """
    content_lower = content.lower()
    database_keywords = [
        "select",
        "query",
        "table",
        "database",
        "sql",
        "show tables",
        "describe table",
        "list tables",
        "count",
        "sum",
        "group by",
        "where",
        "join",
        "order by",
        "limit",
    ]
    return any(k in content_lower for k in database_keywords)

# --- helper: stable UI-friendly dataset alias from filename ---
def _alias_from_filename(filename: str) -> str:
    from pathlib import Path

    stem = Path(filename or "dataset").stem
    alias = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    return alias or "dataset"



import difflib
from datetime import datetime
from typing import Iterable

# -----------------------------
# Join-key suggestion utilities
# -----------------------------

_WORD_RE_JK = re.compile(r"[^a-z0-9]+")

def _jk_norm(s: str) -> str:
    """Normalize a column name for matching."""
    return _WORD_RE_JK.sub("", (s or "").lower().strip())

def _jk_tokens(s: str) -> List[str]:
    s = (s or "").lower()
    s = _WORD_RE_JK.sub(" ", s)
    return [t for t in s.split() if t]

def _jk_is_missing(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}

def _jk_rows_from_preview(ds: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Your datasets_context entries may contain:
      - preview (recommended) : list[dict] truncated
      - uploaded_csv_preview : list[dict] or JSON string
    """
    preview = ds.get("preview")
    if preview is None:
        preview = _maybe_json_load(ds.get("uploaded_csv_preview")) or ds.get("uploaded_csv_preview")
    if isinstance(preview, list) and preview and isinstance(preview[0], dict):
        return preview
    return []

def _jk_cols(ds: Dict[str, Any]) -> List[str]:
    cols = ds.get("columns")
    if cols is None:
        cols = _maybe_json_load(ds.get("uploaded_csv_columns")) or ds.get("uploaded_csv_columns")
    if isinstance(cols, list):
        return [str(c) for c in cols]
    # fallback from schema
    schema_any = _maybe_json_load(ds.get("schema")) or ds.get("schema")
    if isinstance(schema_any, dict):
        return [str(c) for c in schema_any.keys()]
    if isinstance(schema_any, list):
        return [str(c) for c in schema_any]
    return []

def _jk_sample_values(rows: List[Dict[str, Any]], col: str, limit: int = 250) -> List[str]:
    out: List[str] = []
    for r in rows[:limit]:
        if col not in r:
            continue
        v = r.get(col)
        if _jk_is_missing(v):
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out

def _jk_set(vals: Iterable[str]) -> set:
    return set(v.strip().lower() for v in vals if v and str(v).strip())

def _jk_match_rate(vals_a: List[str], vals_b: List[str]) -> float:
    """
    Overlap ratio on sampled unique values.
    Uses min(|A|,|B|) as denominator to avoid punishing different sizes.
    """
    A = _jk_set(vals_a)
    B = _jk_set(vals_b)
    if not A or not B:
        return 0.0
    inter = A & B
    return len(inter) / float(min(len(A), len(B)))

def _jk_uniqueness(vals: List[str]) -> float:
    """Approx uniqueness within sample (0..1)."""
    if not vals:
        return 0.0
    A = _jk_set(vals)
    return min(1.0, len(A) / float(len(vals)))

def _jk_is_id_like(col: str, vals: List[str]) -> bool:
    name = (col or "").lower()
    if name.endswith("id") or name.endswith("_id") or " id" in name:
        return True
    # heuristic: many uniques, mostly alnum, few spaces
    if not vals:
        return False
    uniq = _jk_uniqueness(vals)
    if uniq < 0.75:
        return False
    clean = sum(1 for v in vals[:80] if re.fullmatch(r"[A-Za-z0-9\-_]+", v))
    return clean >= max(5, int(0.6 * min(len(vals[:80]), 80)))

def _jk_parse_date(v: str) -> Optional[datetime]:
    s = (v or "").strip()
    if not s:
        return None
    # Try ISO-ish
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        pass
    # Try common formats
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None

def _jk_date_fraction(vals: List[str]) -> float:
    if not vals:
        return 0.0
    ok = 0
    for v in vals[:60]:
        if _jk_parse_date(v) is not None:
            ok += 1
    return ok / float(min(len(vals), 60))

def _jk_is_date_like(col: str, vals: List[str]) -> bool:
    name = (col or "").lower()
    if any(k in name for k in ["date", "time", "timestamp", "created", "updated", "dt"]):
        return True
    return _jk_date_fraction(vals) >= 0.4

def _jk_is_year_like(col: str, vals: List[str]) -> bool:
    name = (col or "").lower()
    if "year" in name:
        return True
    y = 0
    for v in vals[:60]:
        if re.fullmatch(r"\d{4}", v):
            y += 1
    return y >= 8

def _jk_is_month_like(col: str, vals: List[str]) -> bool:
    name = (col or "").lower()
    if "month" in name:
        return True
    m = 0
    for v in vals[:60]:
        if re.fullmatch(r"(0?[1-9]|1[0-2])", v):
            m += 1
    return m >= 10

def _jk_is_day_like(col: str, vals: List[str]) -> bool:
    name = (col or "").lower()
    if name in {"day", "date_day"} or "day" in name:
        return True
    d = 0
    for v in vals[:60]:
        if re.fullmatch(r"(0?[1-9]|[12][0-9]|3[01])", v):
            d += 1
    return d >= 10

def _jk_is_short_code(vals: List[str]) -> bool:
    """
    Generic 'code' detector (not country-specific): mostly 2-4 uppercase letters/digits.
    """
    if not vals:
        return False
    code = 0
    for v in vals[:80]:
        s = v.strip()
        if re.fullmatch(r"[A-Z0-9]{2,4}", s):
            code += 1
    return code >= 10

def _jk_is_long_label(vals: List[str]) -> bool:
    if not vals:
        return False
    longish = sum(1 for v in vals[:80] if len(v.strip()) >= 6 and " " in v.strip())
    return longish >= 10

def _jk_col_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _jk_norm(a), _jk_norm(b)).ratio()

def _jk_reason(*parts: str) -> str:
    return " ".join([p for p in parts if p])

def _jk_confidence(base: float, match_rate: float, uniq_a: float, uniq_b: float, id_like: bool) -> float:
    # base from name match quality
    score = base
    # overlap is strong evidence
    score += 0.35 * min(1.0, match_rate * 1.2)
    # uniqueness matters for key suitability
    score += 0.15 * min(1.0, (uniq_a + uniq_b) / 2.0)
    if id_like:
        score += 0.10
    return max(0.0, min(1.0, score))

def suggest_join_keys(
    datasets_context: List[Dict[str, Any]],
    max_suggestions: int = 8,
    fuzzy_threshold: float = 0.86,
) -> List[Dict[str, Any]]:
    """
    Suggest join keys across multiple datasets using:
      - exact/canonical column name matches
      - fuzzy column name matches
      - sample value overlap
      - uniqueness/id-likeness
      - generic transform suggestions (date->year/month/day, code->label)

    Returns list of suggestions sorted by confidence desc.

    Suggestion schema:
      {
        left_dataset, right_dataset,
        left_key, right_key,
        confidence, match_rate,
        suggested_transform, reason
      }
    """
    if not datasets_context or len(datasets_context) < 2:
        return []

    prepared = []
    for ds in datasets_context:
        alias = ds.get("alias") or ds.get("filename") or ds.get("dataset_id")
        cols = _jk_cols(ds)
        rows = _jk_rows_from_preview(ds)
        prepared.append((alias, cols, rows))

    suggestions: List[Dict[str, Any]] = []

    # Pairwise dataset comparisons
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            a_alias, a_cols, a_rows = prepared[i]
            b_alias, b_cols, b_rows = prepared[j]

            # normalized lookup
            a_map = {_jk_norm(c): c for c in a_cols}
            b_map = {_jk_norm(c): c for c in b_cols}

            # 1) exact/canonical matches
            common_norm = sorted(set(a_map.keys()) & set(b_map.keys()))
            for k in common_norm:
                a_col = a_map[k]
                b_col = b_map[k]
                a_vals = _jk_sample_values(a_rows, a_col)
                b_vals = _jk_sample_values(b_rows, b_col)
                mr = _jk_match_rate(a_vals, b_vals)
                ua = _jk_uniqueness(a_vals)
                ub = _jk_uniqueness(b_vals)
                id_like = _jk_is_id_like(a_col, a_vals) or _jk_is_id_like(b_col, b_vals)

                conf = _jk_confidence(base=0.65, match_rate=mr, uniq_a=ua, uniq_b=ub, id_like=id_like)

                suggestions.append({
                    "left_dataset": a_alias,
                    "right_dataset": b_alias,
                    "left_key": a_col,
                    "right_key": b_col,
                    "confidence": round(conf, 2),
                    "match_rate": round(mr, 2),
                    "suggested_transform": None,
                    "reason": _jk_reason("Same (canonicalized) column name.", "Looks like an ID key." if id_like else ""),
                })

            # 2) fuzzy matches
            for a_col in a_cols:
                for b_col in b_cols:
                    if _jk_norm(a_col) == _jk_norm(b_col):
                        continue
                    sim = _jk_col_similarity(a_col, b_col)
                    if sim < fuzzy_threshold:
                        continue

                    a_vals = _jk_sample_values(a_rows, a_col)
                    b_vals = _jk_sample_values(b_rows, b_col)
                    mr = _jk_match_rate(a_vals, b_vals)
                    ua = _jk_uniqueness(a_vals)
                    ub = _jk_uniqueness(b_vals)
                    id_like = _jk_is_id_like(a_col, a_vals) or _jk_is_id_like(b_col, b_vals)

                    conf = _jk_confidence(base=0.45 + sim * 0.15, match_rate=mr, uniq_a=ua, uniq_b=ub, id_like=id_like)

                    suggestions.append({
                        "left_dataset": a_alias,
                        "right_dataset": b_alias,
                        "left_key": a_col,
                        "right_key": b_col,
                        "confidence": round(conf, 2),
                        "match_rate": round(mr, 2),
                        "suggested_transform": None,
                        "reason": _jk_reason(f"Similar column names (fuzzy match {sim:.2f}).", "Looks like an ID key." if id_like else ""),
                    })

            # 3) generic date granularity transforms (date/timestamp ↔ year/month/day)
            #    We don't "auto join" on this; we suggest transform.
            for a_col in a_cols:
                a_vals = _jk_sample_values(a_rows, a_col)
                for b_col in b_cols:
                    b_vals = _jk_sample_values(b_rows, b_col)

                    a_date = _jk_is_date_like(a_col, a_vals)
                    b_date = _jk_is_date_like(b_col, b_vals)
                    a_year = _jk_is_year_like(a_col, a_vals)
                    b_year = _jk_is_year_like(b_col, b_vals)
                    a_month = _jk_is_month_like(a_col, a_vals)
                    b_month = _jk_is_month_like(b_col, b_vals)
                    a_day = _jk_is_day_like(a_col, a_vals)
                    b_day = _jk_is_day_like(b_col, b_vals)

                    # date -> year
                    if a_year and b_date:
                        suggestions.append({
                            "left_dataset": a_alias,
                            "right_dataset": b_alias,
                            "left_key": a_col,
                            "right_key": b_col,
                            "confidence": 0.70,
                            "match_rate": None,
                            "suggested_transform": f"Create `year` from `{b_col}` using pd.to_datetime(...).dt.year, then join `{a_col}` ↔ `year`.",
                            "reason": "One side looks like YEAR, the other looks like DATE/TIMESTAMP.",
                        })
                    if a_date and b_year:
                        suggestions.append({
                            "left_dataset": a_alias,
                            "right_dataset": b_alias,
                            "left_key": a_col,
                            "right_key": b_col,
                            "confidence": 0.70,
                            "match_rate": None,
                            "suggested_transform": f"Create `year` from `{a_col}` using pd.to_datetime(...).dt.year, then join `year` ↔ `{b_col}`.",
                            "reason": "One side looks like DATE/TIMESTAMP, the other looks like YEAR.",
                        })

                    # date -> month
                    if a_month and b_date:
                        suggestions.append({
                            "left_dataset": a_alias,
                            "right_dataset": b_alias,
                            "left_key": a_col,
                            "right_key": b_col,
                            "confidence": 0.62,
                            "match_rate": None,
                            "suggested_transform": f"Create `month` from `{b_col}` using pd.to_datetime(...).dt.month, then join `{a_col}` ↔ `month`.",
                            "reason": "One side looks like MONTH, the other looks like DATE/TIMESTAMP.",
                        })
                    if a_date and b_month:
                        suggestions.append({
                            "left_dataset": a_alias,
                            "right_dataset": b_alias,
                            "left_key": a_col,
                            "right_key": b_col,
                            "confidence": 0.62,
                            "match_rate": None,
                            "suggested_transform": f"Create `month` from `{a_col}` using pd.to_datetime(...).dt.month, then join `month` ↔ `{b_col}`.",
                            "reason": "One side looks like DATE/TIMESTAMP, the other looks like MONTH.",
                        })

                    # date -> day (rare but possible)
                    if a_day and b_date:
                        suggestions.append({
                            "left_dataset": a_alias,
                            "right_dataset": b_alias,
                            "left_key": a_col,
                            "right_key": b_col,
                            "confidence": 0.55,
                            "match_rate": None,
                            "suggested_transform": f"Create `day` from `{b_col}` using pd.to_datetime(...).dt.day, then join `{a_col}` ↔ `day`.",
                            "reason": "One side looks like DAY, the other looks like DATE/TIMESTAMP.",
                        })
                    if a_date and b_day:
                        suggestions.append({
                            "left_dataset": a_alias,
                            "right_dataset": b_alias,
                            "left_key": a_col,
                            "right_key": b_col,
                            "confidence": 0.55,
                            "match_rate": None,
                            "suggested_transform": f"Create `day` from `{a_col}` using pd.to_datetime(...).dt.day, then join `day` ↔ `{b_col}`.",
                            "reason": "One side looks like DATE/TIMESTAMP, the other looks like DAY.",
                        })

            # 4) generic short-code ↔ long-label suggestion (not country-specific)
            for a_col in a_cols:
                a_vals = _jk_sample_values(a_rows, a_col)
                for b_col in b_cols:
                    b_vals = _jk_sample_values(b_rows, b_col)
                    if _jk_is_short_code(a_vals) and _jk_is_long_label(b_vals):
                        suggestions.append({
                            "left_dataset": a_alias,
                            "right_dataset": b_alias,
                            "left_key": a_col,
                            "right_key": b_col,
                            "confidence": 0.60,
                            "match_rate": None,
                            "suggested_transform": f"Consider mapping codes in `{a_col}` to labels matching `{b_col}` using a lookup table before joining.",
                            "reason": "One column looks like SHORT CODES, the other like HUMAN-READABLE LABELS.",
                        })
                    if _jk_is_long_label(a_vals) and _jk_is_short_code(b_vals):
                        suggestions.append({
                            "left_dataset": a_alias,
                            "right_dataset": b_alias,
                            "left_key": a_col,
                            "right_key": b_col,
                            "confidence": 0.60,
                            "match_rate": None,
                            "suggested_transform": f"Consider mapping codes in `{b_col}` to labels matching `{a_col}` using a lookup table before joining.",
                            "reason": "One column looks like HUMAN-READABLE LABELS, the other like SHORT CODES.",
                        })

    # Deduplicate suggestions (same ds pair + key pair + transform)
    seen = set()
    deduped = []
    for s in suggestions:
        sig = (
            s.get("left_dataset"), s.get("right_dataset"),
            s.get("left_key"), s.get("right_key"),
            s.get("suggested_transform")
        )
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(s)

    # Sort by confidence then match_rate (if present)
    def _sort_key(x: Dict[str, Any]):
        mr = x.get("match_rate")
        mr_val = mr if isinstance(mr, (int, float)) else -1.0
        return (x.get("confidence", 0.0), mr_val)

    deduped.sort(key=_sort_key, reverse=True)
    return deduped[:max_suggestions]
