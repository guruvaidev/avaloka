"""An OpenAI-compatible chat-completions client on the standard library.

One transport for Ollama, Groq, OpenRouter and OpenAI. ``urllib`` only: the
CLI's eight runtime dependencies are pinned (S1.09), and an SDK per provider
would be a ninth, tenth and eleventh.

Every failure becomes :class:`LLMCallError`, which names the provider and the
fix in one line. Nothing here falls back to anything; the caller decides, and
the caller says so when it does.
"""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Optional

from avaloka.data_agent.contracts import ToolCallRejected, tool_use_failed_text
from avaloka.llm.providers import (GROQ, LOCAL, OPENROUTER, Selection, capabilities,
                                   tool_choice_for)

#: Re-prompts after a rejected tool call before giving up. The same bound as
#: ``MissionPolicy.max_tool_call_rejections`` on the data-agent path.
MAX_TOOL_CALL_REJECTIONS = 2

#: What the model is told after a rejected tool call. The request was well
#: formed, so resending it unchanged repeats a deterministic parser failure;
#: the model is told what a valid reply is instead (consistent with the data
#: agent's re-prompt, 0f711844).
REPROMPT = (
    "Your previous reply could not be executed: {why}. Call exactly one of the "
    "provided tools ({names}) by its exact name with JSON arguments that match its "
    "schema. Do not invent tools and do not reply in prose."
)


# A sealed credential token (app/core/typed_secrets.py) can reach the CLI in a
# thread's transcript. It is ciphertext, but no model is sent it: each one goes
# out as the same opaque placeholder the server's models get.
_SEALED_TOKEN = re.compile(r"\u27e6sealed:[a-z_]+:[A-Za-z0-9+/=]+\.[A-Za-z0-9+/=]+\u27e7")
SEALED_PLACEHOLDER = "\u27e6credential\u27e7"


def scrub_sealed(value: Any) -> Any:
    """``value`` (str, list or dict, recursively) with every sealed token replaced."""
    if isinstance(value, str):
        return _SEALED_TOKEN.sub(SEALED_PLACEHOLDER, value) if "\u27e6sealed:" in value else value
    if isinstance(value, list):
        return [scrub_sealed(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub_sealed(v) for k, v in value.items()}
    return value


class LLMCallError(RuntimeError):
    """A provider call failed. ``str()`` is the one line to print."""

    def __init__(self, provider: str, problem: str, fix: str, *, status: int = 0,
                 code: str = "", body: Any = None) -> None:
        super().__init__(f"{problem} Fix: {fix}")
        self.provider, self.problem, self.fix = provider, problem, fix
        self.status, self.code, self.body = status, code, body


class LLMToolCallRejected(LLMCallError, ToolCallRejected):
    """The provider (or the model) produced no usable tool call, after re-prompts."""

    def __init__(self, provider: str, problem: str, fix: str, *,
                 failed_generation: str = "", **kw: Any) -> None:
        LLMCallError.__init__(self, provider, problem, fix, **kw)
        self.failed_generation = failed_generation


def _error_detail(body: Any) -> tuple[str, str]:
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("code") or err.get("type") or ""), str(err.get("message") or "")
    if isinstance(err, str):
        return "", err
    return "", ""


def _is_tool_parse_failure(body: Any, message: str) -> bool:
    if tool_use_failed_text(type("E", (), {"body": body})()) is not None:
        return True
    # Groq's gpt-oss parser, when the model addresses a channel as a tool:
    # "attempted to call tool 'commentary' which was not in request.tools".
    return "not in request.tools" in message


def _fix_for(sel: Selection, status: int, message: str) -> tuple[str, str]:
    label = sel.spec.label
    if status in (401, 403):
        return (f"{label} rejected the API key (HTTP {status}).",
                f"check {sel.spec.key_env}, or choose another provider with --llm-provider.")
    if status == 404:
        if sel.provider == LOCAL:
            return (f"the local server at {sel.base_url} has no model {sel.model!r} (HTTP 404).",
                    f"run `ollama pull {sel.model}`, or set --llm-model / AVALOKA_LLM_MODEL.")
        return (f"{label} has no model {sel.model!r} at {sel.base_url} (HTTP 404).",
                "check --llm-model / AVALOKA_LLM_MODEL and AVALOKA_LLM_BASE_URL.")
    if status == 429:
        return (f"{label} is rate-limiting or out of credit (HTTP 429).",
                "wait and retry, check the account's quota, or choose another provider.")
    detail = f": {message[:160]}" if message else ""
    if status >= 500:
        return (f"{label} returned a server error (HTTP {status}){detail}.",
                "retry later, or choose another provider with --llm-provider.")
    return (f"{label} refused the request (HTTP {status}){detail}.",
            "check --llm-model; if it persists, choose another provider.")


class OpenAICompatibleChat:
    """``POST {base_url}/chat/completions`` for one :class:`Selection`."""

    def __init__(self, selection: Selection, *, timeout_s: float = 60.0) -> None:
        self.sel = selection
        self.timeout_s = timeout_s
        self.caps = capabilities(selection.provider, selection.model)

    # ------------------------------------------------------------------ wire
    def payload(self, messages: list[dict[str, Any]], *, max_tokens: int,
                temperature: Optional[float] = None,
                tools: Optional[list[dict[str, Any]]] = None,
                require_tool: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.sel.model, "messages": messages,
                                "max_tokens": max_tokens, "stream": False}
        if temperature is not None and self.caps.temperature:
            body["temperature"] = temperature
        if self.caps.reasoning_effort:
            if self.sel.provider == GROQ:
                body["reasoning_effort"] = self.caps.reasoning_effort
            elif self.sel.provider == OPENROUTER:
                body["reasoning"] = {"effort": self.caps.reasoning_effort}
        if self.sel.provider == OPENROUTER:
            body["usage"] = {"include": True}   # OpenRouter reports the call's cost
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice_for(self.sel.provider, self.sel.model,
                                                  require_tool=require_tool)
        return body

    def post(self, body: dict[str, Any]) -> dict[str, Any]:
        sel = self.sel
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "avaloka-cli"}
        if sel.api_key:
            headers["Authorization"] = f"Bearer {sel.api_key}"
        url = f"{sel.base_url}/chat/completions"
        body = scrub_sealed(body)
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace") if exc.fp else ""
            try:
                parsed: Any = json.loads(text)
            except ValueError:
                parsed = {"error": {"message": text.strip()[:200]}}
            code, message = _error_detail(parsed)
            if exc.code == 400 and _is_tool_parse_failure(parsed, message):
                fg = tool_use_failed_text(type("E", (), {"body": parsed})()) or ""
                raise LLMToolCallRejected(
                    sel.provider, f"{sel.spec.label} could not parse the model's tool call.",
                    "the model is re-prompted; if it keeps failing, try another model.",
                    status=400, code=code or "tool_use_failed", body=parsed,
                    failed_generation=fg) from None
            problem, fix = _fix_for(sel, exc.code, message)
            raise LLMCallError(sel.provider, problem, fix, status=exc.code, code=code,
                               body=parsed) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
            reason = getattr(exc, "reason", exc)
            if sel.provider == LOCAL:
                fix = ("start the server (`ollama serve`) or set OLLAMA_HOST / "
                       "AVALOKA_LLM_BASE_URL to where it runs.")
            else:
                fix = "check the network, or choose another provider with --llm-provider."
            raise LLMCallError(sel.provider,
                               f"could not reach {sel.spec.label} at {sel.base_url} ({reason}).",
                               fix) from None
        try:
            data = json.loads(raw)
        except ValueError:
            raise LLMCallError(sel.provider, f"{sel.spec.label} returned a non-JSON reply.",
                               "check AVALOKA_LLM_BASE_URL points at an OpenAI-compatible /v1 API.") from None
        if not isinstance(data, dict) or not data.get("choices"):
            code, message = _error_detail(data)
            raise LLMCallError(sel.provider,
                               f"{sel.spec.label} returned no choices{': ' + message[:160] if message else ''}.",
                               "retry, or choose another provider with --llm-provider.")
        return data

    # ------------------------------------------------------------------ calls
    def complete(self, system: str, prompt: str, *, max_tokens: int = 600,
                 temperature: Optional[float] = None) -> tuple[str, dict[str, Any]]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        data = self.post(self.payload(msgs, max_tokens=max_tokens, temperature=temperature))
        choice = data["choices"][0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if not text.strip():
            why = choice.get("finish_reason") or "empty"
            raise LLMCallError(self.sel.provider,
                               f"{self.sel.spec.label} returned an empty reply (finish_reason={why}).",
                               "raise the token budget or choose another model with --llm-model.")
        return text, data.get("usage") or {}

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
             max_tokens: int = 1024, require_tool: bool = True,
             temperature: Optional[float] = None) -> tuple[dict[str, Any], dict[str, Any], int]:
        """One tool-calling turn. Returns ``(message, usage, reprompts)``.

        A provider parse failure (Groq 400 ``tool_use_failed``), a call to a tool
        that was not offered (the phantom ``commentary`` / ``json`` tools), or no
        tool call when one is required, is answered with a re-prompt, at most
        :data:`MAX_TOOL_CALL_REJECTIONS` times; then :class:`LLMToolCallRejected`.
        """
        if not self.caps.tools:
            raise LLMCallError(self.sel.provider,
                               f"model {self.sel.model!r} does not support tool calls.",
                               "use a tool-capable model, e.g. --llm-model gemma4:e4b.")
        names = [t.get("function", {}).get("name") for t in tools]
        msgs = list(messages)
        usage_total: dict[str, Any] = {}
        rejections = 0
        last_generation = ""
        while True:
            why = ""
            try:
                data = self.post(self.payload(msgs, max_tokens=max_tokens, tools=tools,
                                              temperature=temperature, require_tool=require_tool))
            except LLMToolCallRejected as exc:
                why = "the provider could not parse it as a tool call"
                last_generation = exc.failed_generation
            else:
                for k, v in (data.get("usage") or {}).items():
                    if isinstance(v, (int, float)):
                        usage_total[k] = usage_total.get(k, 0) + v
                message = data["choices"][0].get("message") or {}
                calls = message.get("tool_calls") or []
                phantom = [c.get("function", {}).get("name") for c in calls
                           if c.get("function", {}).get("name") not in names]
                if phantom:
                    why = f"it called {phantom[0]!r}, which is not one of the provided tools"
                elif require_tool and not calls:
                    why = "it contained no tool call"
                else:
                    return message, usage_total, rejections
            rejections += 1
            if rejections > MAX_TOOL_CALL_REJECTIONS:
                raise LLMToolCallRejected(
                    self.sel.provider,
                    f"{self.sel.spec.label} produced no valid tool call after "
                    f"{MAX_TOOL_CALL_REJECTIONS} re-prompts ({why}).",
                    "try another model with --llm-model, or another provider.",
                    failed_generation=last_generation)
            msgs.append({"role": "user",
                         "content": REPROMPT.format(why=why, names=", ".join(map(str, names)))})
