"""Stable, additive API contract shared by legacy and swarm-aware UIs.

This module is the only execution-context builder. The synchronous chat
response, the scheduled-task result (``/tasks/{id}/result/{index}``) and the
task SSE stream all call it, so the three cannot drift apart again: before
this, scheduled and SSE results dropped ``execution_context`` entirely.

The base shape is the one develop-1.6 emits and the shipped UI consumes:
``mode``, ``mode_label``, ``sample``, ``rows_analyzed``, ``total_rows`` and
``is_sampled`` are always present. ``swarm`` and ``swarm_message`` are additive
and appear only when the graph produced a clone plan.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

FIDELITY_QUICK = "quick_sample"

#: The steps Avaloka talks through, in the order a turn takes them. The upload
#: produces ``auto_insights``; a chat turn produces the rest as they happened;
#: ``swarm`` is the team story after a mission that fanned out.
PROMPT_STEPS: tuple[str, ...] = (
    "auto_insights", "understanding", "sampling", "plan", "execution",
    "verification", "result", "swarm",
)

_FIDELITY_LABELS = {
    FIDELITY_QUICK: "Quick Sample",
    "portfolio_samples": "Portfolio Samples",
    "entire_dataset": "Entire Dataset",
}


def build_execution_context(
    fidelity: str | None,
    sample_name: str | None,
    *,
    rows_analyzed: int | None = None,
    total_rows: int | None = None,
    swarm: list[dict[str, Any]] | None = None,
    swarm_message: str | None = None,
) -> dict[str, Any]:
    """Return the 1.6 context plus optional swarm metadata.

    Keyword-only past ``sample_name``: develop-1.6 passed row counts third and
    fourth, develop-1.7 passed swarm third and fourth, and a positional call
    from either side would silently land in the other's slot.
    """
    mode = fidelity or FIDELITY_QUICK
    context: dict[str, Any] = {
        "mode": mode,
        "mode_label": _FIDELITY_LABELS.get(mode, mode),
        "sample": sample_name,
        # Frontend badge: "Analyzed 91,713 of 91,713 rows" / "Sample: 7,013 of 91,713"
        "rows_analyzed": rows_analyzed,
        "total_rows": total_rows,
        "is_sampled": (
            rows_analyzed is not None and total_rows is not None
            and rows_analyzed < total_rows
        ),
    }
    # Avaloka's clone plan for this turn (narration only, never routing).
    # Absent on plain turns, early guard/error responses, and in builds that
    # do not ship the swarm module.
    if swarm:
        context["swarm"] = swarm
        if swarm_message:
            context["swarm_message"] = swarm_message
    return context


def assistant_prompt(
    step: str,
    message: str | Iterable[Mapping[str, Any]],
    *,
    question: str | None = None,
    quick_replies: Iterable[Mapping[str, str] | str] = (),
    link: Mapping[str, str] | None = None,
    message_kind: str = "plain",
) -> dict[str, Any]:
    """One entry of ``assistant_prompts``: what Avaloka said at ``step``.

    ``message`` is what she found or did; ``question`` is the one thing she
    asks next; each quick reply is ``{"label": shown on the button, "send":
    the text the UI posts as the user's next message}`` -- a bare string is
    both. ``link`` is an optional ``{"label", "href"}`` the UI renders after
    the text (the /swarm page, in editions that have it). The shape is
    additive to every response that carries it, and a UI that does not know
    it still gets ``messages`` as before.

    ``message_segments`` says who wrote ``message`` (app/api/reply_segments.py):
    ``message_kind`` is ``"plain"`` for a line built from the data (chat_voice,
    the swarm story) or ``"commentary"`` for a model's words, whose sentences
    ``message`` then carries marked. A ``message`` that mixes the two -- a
    template sentence followed by names a model chose -- is passed as its
    segments instead, and ``message_kind`` is not used.
    """
    if step not in PROMPT_STEPS:
        raise ValueError(f"unknown prompt step {step!r}; one of {PROMPT_STEPS}")
    replies = []
    for reply in quick_replies:
        if isinstance(reply, str):
            replies.append({"label": reply, "send": reply})
        else:
            replies.append({"label": str(reply["label"]), "send": str(reply.get("send") or reply["label"])})
    from app.api import reply_segments

    if isinstance(message, str):
        segs = reply_segments.normalize([reply_segments.segment(message.strip(), message_kind)])
    else:
        segs = reply_segments.normalize(message)
    out: dict[str, Any] = {"step": step, "message": reply_segments.render(segs), "message_segments": segs}
    if question:
        out["question"] = question.strip()
    out["quick_replies"] = replies
    if link:
        out["link"] = {"label": str(link["label"]), "href": str(link["href"])}
    return out


def build_execution_context_from_state(
    state: Mapping[str, Any],
    session: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the context for a result that arrives as a graph state.

    Used by the scheduled-task result and SSE paths. Row counts are not
    recorded in a scheduled run's state, so they are reported as unknown
    (``None``) rather than guessed.
    """
    fallback = session or {}
    return build_execution_context(
        state.get("analysis_fidelity") or fallback.get("analysis_fidelity"),
        state.get("selected_sample_name") or fallback.get("selected_sample_name"),
        swarm=state.get("avaloka_swarm"),
        swarm_message=state.get("avaloka_swarm_message"),
    )


def api_capabilities(
    *,
    upload_formats: Iterable[str] = (),
    swarm_limits: Mapping[str, Any] | None = None,
    swarm_console: Mapping[str, Any] | None = None,
    upload_format_status: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Feature discovery for new clients; legacy clients never need this.

    Swarm availability comes from the import probe in
    ``app.agents.avaloka_agent.capabilities`` and from nothing else. An edition
    that does not ship the swarm module must say so here, and no flag can make
    it claim otherwise. The import is deferred so importing this module stays
    cheap for the contract tests.

    ``swarm_console`` is the edition gate on the live /swarm page and the
    swarm/trace routes (``app.api.swarm_gate.console_capability``): the UI
    hides the page's navigation when ``enabled`` is false and shows the
    commercial-gate copy if the route is hit directly. Omitted, the block says
    the console is closed, which is the safe default for a client that asked
    without an edition.

    ``upload_formats`` is what POST /api/upload accepts, read from the server's
    own list; the complex-data block below describes the optional readers for
    documents, images, video and SDK directories, and reports what this build
    can actually open under ``runtime``.
    """
    from app.agents.avaloka_agent import capabilities as _caps

    swarm = _caps.describe()["swarm"]
    try:
        from avaloka.io import complex_data_capabilities
        complex_runtime = complex_data_capabilities()
    except Exception:
        complex_runtime = {}
    available = bool(swarm["available"])
    limits = dict(swarm_limits) if swarm_limits else None
    message = None
    if available and limits:
        cap = limits.get("max_clones_per_mission")
        message = (
            f"Swarm limited to {cap} clones per mission on {limits.get('label')}, run locally."
            if cap is not None else f"Swarm on {limits.get('label')}: no limit on clones per mission."
        )
    return {
        "api_contract": "1.5.2-compatible",
        "compatibility": "additive",
        "features": {
            "swarm_execution_context": {
                "enabled": bool(swarm["available"]),
                "reason": swarm["reason"],
                "path": "execution_context.swarm",
                "optional": True,
                "routing_semantics": False,
            },
            # The swarm as a capability: whether this build has it, and how
            # far one mission may fan out for the caller's edition. The limit
            # is enforced where clones are dispatched (avaloka.coordinator).
            "swarm": {
                "enabled": available,
                "reason": swarm["reason"],
                "limits": limits,
                # Only a shipped swarm can be limited; limits never imply availability.
                "limited": bool(available and limits and limits.get("max_clones_per_mission") is not None),
                "message": message,
            },
            # The live swarm console (the /swarm page and its routes), gated by
            # edition. The swarm itself is "swarm" above and is never gated.
            "swarm_console": dict(swarm_console) if swarm_console else {
                "enabled": False, "reason": "commercial", "page": "/swarm", "message": None,
            },
            # Avaloka talking the user through each step (ChatResponse and
            # UploadResponse ``assistant_prompts``; see assistant_prompt()).
            "assistant_prompts": {
                "enabled": True,
                "optional": True,
                "path": "assistant_prompts",
                "steps": list(PROMPT_STEPS),
            },
            "async_execution_context": True,
            # What POST /api/upload accepts, read from the server's own list.
            # A source the route does not take (MongoDB is a connection, not a
            # file) is not advertised here; JSON Lines joined when the route
            # grew a reader and a route test (2026-10-17).
            "upload_formats": sorted(upload_formats),
            "nested_field_notation": "dotted_paths_v1",
            "complex_data": {
                "optional": True,
                "upload_formats": [
                    "pdf", "jpeg", "png", "tiff", "bmp", "webp",
                    "mp4", "mov", "mkv", "avi", "webm", "m4v",
                ],
                "sdk_directory_formats": ["lerobot", "rlds"],
                "recursive_schema": True,
                "format_validators": ["lerobot_v3", "rlds"],
                "runtime": complex_runtime,
            },
            # Per-format status beside the flat list: accepted formats are
            # "enabled" (with or without the statistics contract); a layout the
            # product knows but does not take is "deferred" with its one-line
            # reason, so a UI can show it greyed out and say why. Additive;
            # absent in older builds.
            "upload_format_status": {
                ext: dict(entry) for ext, entry in sorted((upload_format_status or {}).items())
            },
        },
    }
