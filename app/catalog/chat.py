"""The catalog in chat: ``find_datasets`` and ``use_dataset`` (design §C.6, 16f).

Two things a user says that the catalog answers:

- "which datasets have I used?" / "find the orders data" -> :func:`find_datasets`
  searches the caller's own catalog (names, tags, terms, analysis ids) and
  returns ids with code-written summaries. Nothing here is model text.
- "use my.orders_2026" / "use the one from analysis an:churn" ->
  :func:`use_dataset` resolves the reference to one ``(dataset, version)``
  through ``CatalogStore.resolve`` and binds it to the turn the way an upload
  is bound: ``active_dataset_id``, ``datasets_context`` and
  ``uploaded_csv_columns`` are set from the catalog's schema, and the file is
  located through the session store. The planner then treats it exactly like
  a fresh upload. "The latest" is the latest version; "the one from analysis
  X" pins the version that analysis read.

What this does not do: read the data. A dataset whose copy Avaloka no longer
holds (an upload whose session expired) is reported as such, by name, and is
not bound half-way.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from app.catalog.store import Binding, CatalogError, CatalogStore, CatalogUnavailable, KINDS

logger = logging.getLogger(__name__)

#: "use my.orders_2026", "use my.orders@2", "use analysis:an:churn", "use the
#: dataset from analysis an:churn" -- the reference the planner binds without a
#: model in the loop.
USE_REFERENCE = re.compile(
    r"\b(?:use|load|open|switch\s+to|analy[sz]e)\s+(?:the\s+)?(?:dataset\s+)?"
    r"(?P<ref>my\.[A-Za-z0-9_]+(?:@\d+)?|analysis:\S+|(?:the\s+one\s+)?from\s+analysis\s+\S+)",
    re.IGNORECASE)

#: "my datasets", "which datasets have I used", "find datasets about orders".
FIND_TRIGGERS = ("my datasets", "which datasets", "what datasets", "datasets have i", "datasets did i",
                 "find dataset", "search dataset", "list my datasets", "prior datasets", "previous datasets",
                 "datasets in my catalog", "show my catalog", "my catalog")

Locate = Callable[[str, str], Optional[Mapping[str, Any]]]


def _owner(state: Mapping[str, Any]) -> Optional[str]:
    user_id = state.get("user_id")
    return str(user_id).strip() or None if user_id else None


def _store(state: Mapping[str, Any]) -> Tuple[Optional[CatalogStore], Optional[str]]:
    owner = _owner(state)
    if not owner:
        return None, ("I can't look in your catalog without knowing who you are. "
                      "Please sign in and try again.")
    try:
        return CatalogStore(owner), None
    except CatalogUnavailable:
        return None, ("Your dataset catalog isn't available on this deployment "
                      "(no catalog database is configured), so I can't search it.")


def find_reference(text: str) -> Optional[str]:
    """The ``my.<name>``/``analysis:<id>`` reference in a "use ..." sentence, or None."""
    m = USE_REFERENCE.search(text or "")
    if not m:
        return None
    ref = m.group("ref").strip().rstrip(".,;:!?")
    ref = re.sub(r"^(?:the\s+one\s+)?from\s+analysis\s+", "analysis:", ref, flags=re.IGNORECASE)
    return ref


def is_find_request(text: str) -> bool:
    lower = (text or "").lower()
    return any(t in lower for t in FIND_TRIGGERS)


# --------------------------------------------------------------------------- #
# find_datasets
# --------------------------------------------------------------------------- #

def find_datasets(state: Mapping[str, Any], query: str = "", *, kind: Optional[str] = None,
                  since_days: Optional[int] = None, limit: int = 20) -> str:
    """A Markdown table of the caller's datasets matching ``query``, newest
    first, each with its ``my.<name>`` handle and the code-written summary."""
    store, refusal = _store(state)
    if store is None:
        return refusal or ""
    if kind and kind not in KINDS:
        return f"I don't know the dataset kind `{kind}`; one of {', '.join(KINDS)}."
    since = datetime.now(timezone.utc) - timedelta(days=int(since_days)) if since_days else None
    try:
        hits = store.find(query=query or None, kind=kind, since=since, limit=limit)
    except CatalogUnavailable:
        return "Your dataset catalog isn't reachable right now; the analysis itself is unaffected."
    except CatalogError as exc:
        return f"I couldn't search your catalog: {exc}"
    if not hits:
        what = f"matching `{query}`" if query else "yet"
        return (f"### Your datasets\n\n_No datasets {what}. The catalog records every dataset your analyses "
                "touch, so this fills in as you work._")
    lines = ["### Your datasets", "", "| Use as | Kind | What it is | Id |", "| --- | --- | --- | --- |"]
    for h in hits:
        handle = f"`{h['name']}`" if h.get("name") else "—"
        lines.append(f"| {handle} | {h['kind']} | {h['summary']} | `{h['dataset_id']}` |")
    lines.append("")
    lines.append("_Say `use my.<name>` to analyse one (its latest version), `use my.<name>@2` for a "
                 "specific version, or `use analysis:<id>` for the version a past analysis read._")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# use_dataset: resolve and bind
# --------------------------------------------------------------------------- #

def describe_binding(binding: Binding) -> str:
    rows = ""
    return (f"`my.{_slug(binding.display_name) if binding.display_name else binding.dataset_id}` "
            f"(version {binding.version_no}, {len(binding.columns)} columns{rows}"
            f"{', pinned by analysis ' + binding.pinned_by if binding.pinned_by else ''})")


def _slug(name: str) -> str:
    from app.catalog.store import slug
    return slug(name)


def bind_dataset(state: Dict[str, Any], binding: Binding, located: Optional[Mapping[str, Any]]) -> bool:
    """Bind one catalog version to the turn the way an upload is bound.

    ``located`` is the session blob for the dataset (``data_source_location``,
    ``uploaded_csv_columns``, ``filename``...), or ``None`` when Avaloka no
    longer holds a copy. Returns whether the binding is complete: with no
    location nothing is bound, because a planner that believes a file is there
    produces code that fails later and worse.
    """
    if not located or not located.get("data_source_location"):
        return False
    columns = [name for name, _ in binding.columns] or list(located.get("uploaded_csv_columns") or [])
    entry = {
        "dataset_id": binding.dataset_id,
        "session_id": located.get("session_id"),
        "alias": _slug(binding.display_name) if binding.display_name else binding.dataset_id,
        "filename": located.get("filename") or binding.locator.get("filename") or binding.display_name,
        "columns": columns,
        "schema": located.get("schema"),
        "row_count": located.get("row_count"),
        "data_source_location": located["data_source_location"],
        "active_data_source_location": located.get("active_data_source_location") or located["data_source_location"],
        "full_data_location": located.get("full_data_location"),
        "sample_data_location": located.get("sample_data_location"),
        "catalog_version_id": binding.version_id,
        "catalog_version_no": binding.version_no,
    }
    entry = {k: v for k, v in entry.items() if v is not None}
    state["active_dataset_id"] = binding.dataset_id
    state["active_dataset_ids"] = [binding.dataset_id]
    state["datasets_context"] = [entry]
    state["multi_dataset_state"] = [entry]
    state["uploaded_csv_columns"] = columns
    state["data_source_location"] = located["data_source_location"]
    state["active_data_source_location"] = entry["active_data_source_location"]
    state["catalog_binding"] = {"dataset_id": binding.dataset_id, "version_id": binding.version_id,
                                "version_no": binding.version_no, "kind": binding.kind,
                                "pinned_by": binding.pinned_by}
    return True


def use_dataset(state: Dict[str, Any], reference: str, *, locate: Optional[Locate] = None) -> str:
    """Resolve ``reference`` in the caller's catalog and bind it to this turn.
    Returns the sentence the user reads; the state carries the binding."""
    store, refusal = _store(state)
    if store is None:
        return refusal or ""
    try:
        binding = store.resolve(reference)
    except CatalogUnavailable:
        return "Your dataset catalog isn't reachable right now, so I can't resolve that dataset."
    except CatalogError as exc:
        return f"I couldn't find that dataset: {exc}"
    owner = _owner(state) or ""
    located = None
    if locate is not None:
        try:
            located = locate(binding.dataset_id, owner)
        except Exception:  # noqa: BLE001
            logger.warning("[catalog] locating a dataset's copy failed", exc_info=True)
            located = None
    if not located and binding.kind == "derived" and binding.locator.get("output_location"):
        # An analysis output has no upload session; its locator says where the
        # file landed when the analysis ran (app/catalog/hooks.py).
        located = {"data_source_location": binding.locator["output_location"],
                   "filename": binding.locator.get("filename") or binding.display_name}
    if not bind_dataset(state, binding, located):
        return (f"I know {describe_binding(binding)} from your catalog -- its schema and history -- but "
                "Avaloka no longer holds a copy of the data (the upload's session has expired). "
                "Upload it again and I'll pick up where its history left off.")
    return (f"Using {describe_binding(binding)}. Columns: {', '.join(n for n, _ in binding.columns) or 'unknown'}. "
            "Ask your question.")


def locate_in_session_store(dataset_id: str, user_id: str) -> Optional[Mapping[str, Any]]:
    """The default locator: the upload's Redis session for this user. Returns
    ``None`` without Redis or when the session is gone."""
    try:
        import asyncio
        from app.services import session_service

        async def _find():
            sid = await session_service.find_session_by_dataset_for_user(dataset_id, user_id)
            if not sid:
                return None
            sess = await session_service.get_session(sid)
            if not sess:
                return None
            return {**sess, "session_id": sid}

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(_find())).result()
        return asyncio.run(_find())
    except Exception:  # noqa: BLE001
        logger.debug("[catalog] session store lookup unavailable", exc_info=True)
        return None


__all__ = ["FIND_TRIGGERS", "USE_REFERENCE", "bind_dataset", "describe_binding", "find_datasets",
           "find_reference", "is_find_request", "locate_in_session_store", "use_dataset"]
