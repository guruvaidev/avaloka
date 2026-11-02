"""Credentials a user types into chat, sealed the moment they arrive.

"Register warehouse, postgres on db.example:5432, user etl, password hunter2"
used to travel as typed: into the chat history, the planner's log lines, every
LLM request that carried the transcript, and the session registry. Now the API
server seals each credential-shaped value in the message before anything else
sees it (:func:`seal_typed_secrets`), so the transcript, the logs and the model
see a token -- AES-GCM ciphertext under DB_ENCRYPTION_KEY, the scheme every
other sealed credential here uses -- and never the value.

Only the data transfer agent opens a token (:func:`open_sealed`), when it hands
the connection to the onboarding API's sealed store; the session then keeps
the connection id, nothing else. The planner seals again on entry, for a turn
that reached the graph without passing the API server.

A message is sealed only when it reads as connection details (a host, port,
bucket, DSN, "connect"/"register"...), so "rows where password is null" in an
ordinary analysis question is left alone.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

TOKEN_RE = re.compile(r"⟦sealed:([a-z_]+):([A-Za-z0-9+/=]+)\.([A-Za-z0-9+/=]+)⟧")

_CONNECTION_CUES = re.compile(
    r"\b(host(?:name)?|server|port|bucket|container|database|dsn|connect\w*|regist\w*"
    r"|endpoint|service[ _]account|storage[ _]account|access[ _]key|credential\w*|sas)\b|://",
    re.IGNORECASE,
)

# A quoted or backticked value is taken exactly, quotes excluded, so a secret
# that really ends in punctuation can be typed as 'pa.ss.'. An unquoted value
# ends before sentence punctuation (.,;:!?) that is followed by whitespace or
# the end of the message: "password X." seals X, "a.b" stays a.b.
_VALUE = (r"""(?:(?P<q>["'`])(?P<value>(?:(?!⟦)(?!(?P=q))[^\n])+)(?P=q)"""
          r"""|(?P<bare>(?:(?!⟦)[^\s"'`])+?)(?=[.,;:!?)]*(?:\s|$)))""")
_SEP = r"(?:\s*(?:[:=]|=>|\bis\b)\s*|\s+)"

# (label, pattern). The label names the RegisterDatabaseParams field the value
# belongs to, so the planner can put it back where the user meant it.
_RULES: List[Tuple[str, re.Pattern]] = [
    # scheme://user:PASSWORD@host
    ("password", re.compile(r"(?P<head>[a-z][a-z0-9+.-]*://[^\s/:@]+:)(?P<value>(?!⟦)[^\s@]+)(?=@)", re.IGNORECASE)),
    ("private_key", re.compile(r'(?P<head>"private_key"\s*:\s*")(?P<value>(?!⟦)(?:[^"\\]|\\.)+)(?=")')),
    ("secret_key", re.compile(r"(?P<head>\b(?:aws[_ ]secret[_ ]access[_ ]key|secret(?:[_ ]access)?[_ ]key|secret)\b" + _SEP + r")" + _VALUE, re.IGNORECASE)),
    ("access_key", re.compile(r"(?P<head>\b(?:aws[_ ]access[_ ]key[_ ]id|access[_ ]key(?:[_ ]id)?)\b" + _SEP + r")" + _VALUE, re.IGNORECASE)),
    ("az_sas_token", re.compile(r"(?P<head>\b(?:az[_ ]sas[_ ]token|sas[_ ]token|sas)\b" + _SEP + r")" + _VALUE, re.IGNORECASE)),
    ("az_access_key", re.compile(r"(?P<head>\b(?:az[_ ]access[_ ]key|account[_ ]key)\b" + _SEP + r")" + _VALUE, re.IGNORECASE)),
    ("password", re.compile(r"(?P<head>\b(?:password|passwd|pwd|pass)\b" + _SEP + r")" + _VALUE, re.IGNORECASE)),
]

# Words that follow "password is" in a sentence about a column, not a value.
_NOT_A_VALUE = {"null", "none", "empty", "missing", "blank", "set", "required", "the", "a", "an",
                "and", "or", "for", "to", "of", "in", "column", "field", "is", "with", "auth",
                "authentication", "protected", "reset", "changed"}


def _seal(label: str, value: str) -> Optional[str]:
    from app.api.cloud_connections import encrypt_secret
    blob = encrypt_secret(value)
    if not blob:
        return None
    return f"⟦sealed:{label}:{blob['ciphertext']}.{blob['iv']}⟧"


def looks_like_connection_details(text: str) -> bool:
    return bool(_CONNECTION_CUES.search(text or ""))


def seal_typed_secrets(text: str) -> Tuple[str, int]:
    """``text`` with every credential-shaped value replaced by a sealed token.

    Returns ``(text, how_many)``. Already-sealed tokens are left as they are, so
    sealing twice is harmless. Without DB_ENCRYPTION_KEY a value cannot be
    sealed and is replaced by ``<redacted>`` instead: dropped, never kept.
    """
    if not isinstance(text, str) or not text or not looks_like_connection_details(text):
        return text, 0
    count = 0

    def _sub(label: str):
        def repl(m: re.Match) -> str:
            nonlocal count
            group = "value" if m.group("value") is not None else "bare"
            value = m.group(group)
            if not value or value.lower() in _NOT_A_VALUE:
                return m.group(0)
            count += 1
            token = _seal(label, value) or "<redacted>"
            whole, start = m.group(0), m.start()
            return whole[:m.start(group) - start] + token + whole[m.end(group) - start:]
        return repl

    for label, rx in _RULES:
        # Rules run on the text between tokens only: a token's own label
        # ("sealed:password:...") must never read as a credential to seal.
        out = []
        for m_iter, piece in _interleave(text):
            out.append(piece if m_iter else rx.sub(_sub(label), piece))
        text = "".join(out)
    remember(text)
    return text, count


def _interleave(text: str):
    """Yield ``(is_token, piece)`` over ``text``, tokens and the text between them."""
    pos = 0
    for m in TOKEN_RE.finditer(text):
        if m.start() > pos:
            yield False, text[pos:m.start()]
        yield True, m.group(0)
        pos = m.end()
    if pos < len(text):
        yield False, text[pos:]


def has_sealed(text: Optional[str]) -> bool:
    return bool(text) and bool(TOKEN_RE.search(text))


def sealed_labels(text: Optional[str]) -> Dict[str, List[str]]:
    """``{label: [token, ...]}`` for the tokens in ``text`` (tokens, not values)."""
    out: Dict[str, List[str]] = {}
    for m in TOKEN_RE.finditer(text or ""):
        out.setdefault(m.group(1), []).append(m.group(0))
    return out


def open_sealed(text: Optional[str]) -> Optional[str]:
    """``text`` with every token replaced by the value it seals. In process only.

    A token this deployment's key cannot open becomes ``""``: the caller then
    fails on a wrong credential rather than on a token passed as one.
    """
    if text is None:
        return None
    from app.api.cloud_connections import decrypt_secret

    def repl(m: re.Match) -> str:
        return decrypt_secret({"ciphertext": m.group(2), "iv": m.group(3)}) or ""

    return TOKEN_RE.sub(repl, text)


def redact_sealed(text: Optional[str]) -> Optional[str]:
    """``text`` with each token shown as ``<redacted>``, for anything a person reads."""
    if text is None:
        return None
    return TOKEN_RE.sub("<redacted>", text)


# ---------------------------------------------------------------------------
# What a model sees, and what a person is shown
# ---------------------------------------------------------------------------
#
# A token is ciphertext, but it is still not for a model: a model can copy it,
# quote pieces of it, or carry it into a reply. Every message bound for a chat
# model has its tokens replaced by PLACEHOLDER (install_model_scrub, installed
# once per process by app.core.inference); the stored transcript keeps the
# token, which is what register_database opens. Anything returned to a person
# goes through redact_reply, which removes whole tokens and any run of 8 or
# more characters of a token this process has seen -- a deterministic guard
# that quotes "unknown fields" from the user's text would otherwise echo them.

PLACEHOLDER = "⟦credential⟧"
_PARTIAL_RE = re.compile(r"⟦sealed:[a-z_]*:?[A-Za-z0-9+/=.]*⟧?")
FRAGMENT_MIN = 8
_SEEN_MAX = 2048
_seen: "dict[str, None]" = {}


def remember(text: Optional[str]) -> None:
    """Note the ciphertext and nonce of every token in ``text`` (not secrets)."""
    for m in TOKEN_RE.finditer(text or ""):
        for part in (m.group(2), m.group(3)):
            if len(part) >= FRAGMENT_MIN:
                _seen.pop(part, None)
                _seen[part] = None
    while len(_seen) > _SEEN_MAX:
        _seen.pop(next(iter(_seen)))


def for_model(text: Optional[str]) -> Optional[str]:
    """``text`` as a model may see it: every token is PLACEHOLDER."""
    if not text or "⟦sealed:" not in text:
        return text
    remember(text)
    return TOKEN_RE.sub(PLACEHOLDER, text)


def redact_reply(text: Optional[str], extra: Optional[str] = None) -> Optional[str]:
    """``text`` with tokens and token fragments removed, for a person to read.

    ``extra`` is more text whose tokens count as seen (e.g. the request).
    """
    if not text:
        return text
    remember(text)
    remember(extra)
    text = TOKEN_RE.sub("<redacted>", text)
    # A token cut short (a title or preview truncated mid-token) is still one.
    text = _PARTIAL_RE.sub("<redacted>", text)
    if not _seen:
        return text
    windows = {p[i:i + FRAGMENT_MIN] for p in _seen for i in range(len(p) - FRAGMENT_MIN + 1)}
    marked = bytearray(len(text))
    for i in range(len(text) - FRAGMENT_MIN + 1):
        if text[i:i + FRAGMENT_MIN] in windows:
            marked[i:i + FRAGMENT_MIN] = b"\x01" * FRAGMENT_MIN
    if not any(marked):
        return text
    out, i = [], 0
    while i < len(text):
        if marked[i]:
            while i < len(text) and marked[i]:
                i += 1
            out.append("<redacted>")
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _scrub_value(value):
    if isinstance(value, str):
        return for_model(value)
    if isinstance(value, list):
        return [_scrub_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrub_value(v) for k, v in value.items()}
    return value


def _scrub_message(message):
    """A copy of a LangChain message with every token replaced, or the message itself."""
    try:
        update = {}
        content = getattr(message, "content", None)
        scrubbed = _scrub_value(content)
        if scrubbed != content:
            update["content"] = scrubbed
        for attr in ("tool_calls", "additional_kwargs"):
            value = getattr(message, attr, None)
            if value:
                scrubbed = _scrub_value(value)
                if scrubbed != value:
                    update[attr] = scrubbed
        return message.model_copy(update=update) if update else message
    except Exception:  # never block a model call on the scrub itself
        return message


_installed = False


def install_model_scrub() -> None:
    """Replace tokens in everything any LangChain chat model is sent. Idempotent.

    Patched on BaseChatModel itself, so every agent's model -- whichever
    provider built it, including the fallback models -- is covered: invoke,
    stream and batch pass through ``_convert_input``; generate() through
    ``_generate_with_cache`` / ``_agenerate_with_cache``.
    """
    global _installed
    if _installed:
        return
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
    except Exception:  # noqa: BLE001 - no LangChain, nothing to patch
        return

    original_convert = BaseChatModel._convert_input

    def _convert_input(self, model_input):
        value = original_convert(self, model_input)
        if isinstance(value, StringPromptValue):
            return StringPromptValue(text=for_model(value.text))
        try:
            messages = value.to_messages()
        except Exception:  # noqa: BLE001
            return value
        return ChatPromptValue(messages=[_scrub_message(m) for m in messages])

    original_gen = BaseChatModel._generate_with_cache
    original_agen = BaseChatModel._agenerate_with_cache

    def _generate_with_cache(self, messages, *args, **kwargs):
        return original_gen(self, [_scrub_message(m) for m in messages], *args, **kwargs)

    async def _agenerate_with_cache(self, messages, *args, **kwargs):
        return await original_agen(self, [_scrub_message(m) for m in messages], *args, **kwargs)

    BaseChatModel._convert_input = _convert_input
    BaseChatModel._generate_with_cache = _generate_with_cache
    BaseChatModel._agenerate_with_cache = _agenerate_with_cache
    _installed = True


class RedactRepliesMiddleware:
    """ASGI: every JSON or event-stream response under ``prefixes`` goes through redact_reply.

    The final pass on what a person is shown: thread messages (the reply and
    the history), the trace, and the streamed reply. A JSON body is rewritten
    whole (Content-Length fixed); an event stream chunk by chunk.
    """

    def __init__(self, app, prefixes=("/threads",)):
        self.app = app
        self.prefixes = tuple(prefixes)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not str(scope.get("path", "")).startswith(self.prefixes):
            return await self.app(scope, receive, send)
        state = {"start": None, "mode": None, "chunks": []}

        async def _send(message):
            if message["type"] == "http.response.start":
                ctype = ""
                for k, v in message.get("headers") or []:
                    if k.lower() == b"content-type":
                        ctype = v.decode("latin-1").lower()
                state["mode"] = ("json" if "json" in ctype else
                                 "stream" if "event-stream" in ctype else None)
                if state["mode"] == "json":
                    state["start"] = message
                    return
                return await send(message)
            if message["type"] == "http.response.body" and state["mode"] == "json":
                state["chunks"].append(message.get("body", b""))
                if message.get("more_body"):
                    return
                body = b"".join(state["chunks"])
                text = body.decode("utf-8", "replace")
                redacted = redact_reply(text)
                if redacted != text:
                    body = redacted.encode("utf-8")
                start = state["start"]
                headers = [(k, v) for k, v in start.get("headers") or [] if k.lower() != b"content-length"]
                headers.append((b"content-length", str(len(body)).encode()))
                await send({**start, "headers": headers})
                return await send({"type": "http.response.body", "body": body, "more_body": False})
            if message["type"] == "http.response.body" and state["mode"] == "stream":
                chunk = message.get("body", b"")
                if chunk:
                    text = chunk.decode("utf-8", "replace")
                    redacted = redact_reply(text)
                    if redacted != text:
                        message = {**message, "body": redacted.encode("utf-8")}
            return await send(message)

        return await self.app(scope, receive, _send)
