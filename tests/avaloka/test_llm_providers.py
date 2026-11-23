"""The CLI's model providers: selection, the wire, and never failing silently.

``avaloka --llm`` used to read ANTHROPIC_API_KEY only, through the Anthropic
SDK, while the README promised GROQ_API_KEY, OPENROUTER_API_KEY and
OPENAI_API_KEY. A Groq key was ignored without a word, and ANTHROPIC_BASE_URL
aimed at Groq returned 404 and fell back to canned text, also without a word.

No test here reaches the network: every HTTP call goes to a mock server on
127.0.0.1 started by the test, and no key is real.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from avaloka.data_agent.contracts import ToolCallRejected
from avaloka.llm import LLMClient
from avaloka.llm.guard import clean_rephrasing, is_meta
from avaloka.llm.openai_compat import (MAX_TOOL_CALL_REJECTIONS, LLMCallError,
                                       LLMToolCallRejected, OpenAICompatibleChat)
from avaloka.llm.providers import (LLMConfigError, capabilities, resolve,
                                   tool_choice_for)

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- #
# A scripted OpenAI-compatible server
# --------------------------------------------------------------------------- #
class MockServer:
    """Answers each POST with the next scripted ``(status, body)``; records requests."""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                outer.requests.append({"path": self.path, "headers": dict(self.headers),
                                       "body": json.loads(self.rfile.read(length) or b"{}")})
                status, body = outer.script.pop(0) if len(outer.script) > 1 else outer.script[0]
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def reply(text="Hello from the model.", usage=None, tool_calls=None):
    msg = {"role": "assistant", "content": text}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return 200, {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                 "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20}}


def call(name, args="{}"):
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": args}}


@pytest.fixture
def server():
    made = []

    def make(*script):
        s = MockServer(script)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


def dead_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


TOOLS = [{"type": "function", "function": {"name": "run_sql", "description": "run",
                                           "parameters": {"type": "object", "properties": {}}}},
         {"type": "function", "function": {"name": "return_answer", "description": "done",
                                           "parameters": {"type": "object", "properties": {}}}}]


# --------------------------------------------------------------------------- #
# (a) selection and order
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("env,expected", [
    ({"GROQ_API_KEY": "k"}, "groq"),
    ({"GROQ_API_KEY": "k", "OLLAMA_HOST": "127.0.0.1:11434"}, "local"),
    ({"GROQ_API_KEY": "k", "AVALOKA_LOCAL_MODEL": "gemma4:e4b"}, "local"),
    ({"GROQ_API_KEY": "k", "INFERENCE_LOCAL_BASE_URL": "http://h:1/v1"}, "local"),
    ({"OPENROUTER_API_KEY": "k", "GROQ_API_KEY": "k"}, "groq"),
    ({"OPENROUTER_API_KEY": "k", "OPENAI_API_KEY": "k"}, "openrouter"),
    ({"OPENAI_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}, "openai"),
    ({"ANTHROPIC_API_KEY": "k"}, "anthropic"),
])
def test_a_auto_order_is_local_then_groq_then_the_rest(env, expected):
    assert resolve(env=env).provider == expected


def test_a_flag_beats_env_beats_auto_and_aliases_resolve():
    env = {"GROQ_API_KEY": "k", "OPENROUTER_API_KEY": "k", "AVALOKA_LLM_PROVIDER": "openrouter"}
    assert resolve(env=env).provider == "openrouter"
    assert resolve(env=env).how == "env"
    assert resolve("groq", env=env).provider == "groq"
    assert resolve("groq", env=env).how == "flag"
    assert resolve("ollama", env={}).provider == "local"


def test_a_defaults_and_overrides_for_model_and_base_url():
    local = resolve("local", env={"OLLAMA_HOST": "10.0.0.5:11434"})
    assert (local.model, local.base_url) == ("gemma4:e4b", "http://10.0.0.5:11434/v1")
    assert resolve("local", env={}).base_url == "http://127.0.0.1:11434/v1"
    groq = resolve(env={"GROQ_API_KEY": "k"})
    assert (groq.model, groq.base_url) == ("openai/gpt-oss-120b", "https://api.groq.com/openai/v1")
    assert resolve(env={"GROQ_API_KEY": "k", "AVALOKA_LLM_MODEL": "m1"}).model == "m1"
    assert resolve(model="m2", env={"GROQ_API_KEY": "k", "AVALOKA_LLM_MODEL": "m1"}).model == "m2"


def test_a_unknown_provider_and_nothing_configured_are_errors_with_a_fix():
    with pytest.raises(LLMConfigError, match="unknown LLM provider 'mistrall'.*Fix: use --llm-provider"):
        resolve("mistrall", env={})
    with pytest.raises(LLMConfigError, match="no model provider is configured.*GROQ_API_KEY"):
        resolve(env={})


def test_a_the_key_never_appears_in_a_repr():
    assert "s3cr3t" not in repr(resolve(env={"GROQ_API_KEY": "s3cr3t"}))


# --------------------------------------------------------------------------- #
# (b) the OpenAI-compatible request shape, against a local mock
# --------------------------------------------------------------------------- #
def test_b_local_request_shape(server, monkeypatch):
    s = server(reply("Ooh, 1,200 rows."))
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", s.url)
    client = LLMClient(provider="local", notify=lambda m: None)
    assert client.complete("SYS", "Line: x\nReply:", max_tokens=77) == "Ooh, 1,200 rows."
    (req,) = s.requests
    assert req["path"] == "/v1/chat/completions"
    assert "Authorization" not in req["headers"]          # local: no key to send
    body = req["body"]
    assert body["model"] == "gemma4:e4b" and body["max_tokens"] == 77 and body["stream"] is False
    assert body["messages"] == [{"role": "system", "content": "SYS"},
                                {"role": "user", "content": "Line: x\nReply:"}]
    assert "tools" not in body and "tool_choice" not in body and "reasoning_effort" not in body
    assert (client.calls, client.last_tokens_in, client.last_tokens_out) == (1, 100, 20)
    assert client.last_cost_usd == 0.0 and client.last_cost_known


def test_b_groq_request_carries_bearer_key_and_gpt_oss_reasoning(server, monkeypatch):
    s = server(reply())
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", s.url)
    LLMClient(notify=lambda m: None).complete("s", "p")
    (req,) = s.requests
    assert req["headers"]["Authorization"] == "Bearer test-key-not-real"
    assert req["body"]["model"] == "openai/gpt-oss-120b"
    assert req["body"]["reasoning_effort"] == "low"


@pytest.mark.parametrize("provider,key_env,expected", [
    ("groq", "GROQ_API_KEY", "required"),
    ("local", None, "required"),           # Ollama answers "auto" with prose
    ("openai", "OPENAI_API_KEY", "required"),
    ("openrouter", "OPENROUTER_API_KEY", "auto"),   # "required" -> finish_reason=error
])
def test_b_tool_choice_is_mapped_per_provider_on_the_wire(server, monkeypatch, provider, key_env, expected):
    s = server(reply("", tool_calls=[call("run_sql")]))
    if key_env:
        monkeypatch.setenv(key_env, "test-key-not-real")
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", s.url)
    LLMClient(provider=provider, notify=lambda m: None).chat([{"role": "user", "content": "q"}], TOOLS)
    body = s.requests[0]["body"]
    assert body["tool_choice"] == expected and body["tools"] == TOOLS
    if provider == "openrouter":
        assert body["usage"] == {"include": True} and body["reasoning"] == {"effort": "low"}


def test_b_capabilities_per_model():
    assert capabilities("local", "gemma3:4b").tools is False       # 400 on any tools
    assert capabilities("local", "gemma4:e4b").tools is True
    assert capabilities("anthropic", "claude-haiku-5-5").temperature is False
    assert tool_choice_for("anthropic", "claude-haiku-5-5", require_tool=True) == {"type": "any"}
    assert tool_choice_for("anthropic", "claude-sonnet-5-5", require_tool=True) == {"type": "auto"}
    assert tool_choice_for("groq", "openai/gpt-oss-120b", require_tool=False) == "auto"


def test_b_a_tool_request_to_a_model_without_tools_is_refused_before_sending(server, monkeypatch):
    s = server(reply())
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", s.url)
    client = LLMClient(provider="local", model="gemma3:4b", notify=lambda m: None)
    with pytest.raises(LLMCallError, match="does not support tool calls.*gemma4:e4b"):
        client.chat([{"role": "user", "content": "q"}], TOOLS)
    assert s.requests == []


def test_b_cost_follows_the_pricing_hooks(server, monkeypatch):
    s = server(reply(usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}))
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", s.url)
    client = LLMClient(notify=lambda m: None)
    client.complete("s", "p")
    assert client.last_cost_usd == 0.0 and client.last_cost_known is False   # unknown, not invented
    monkeypatch.setenv("AVALOKA_MODEL_PRICES", json.dumps({"openai/gpt-oss-120b": [0.15, 0.60]}))
    client.complete("s", "p")
    assert client.last_cost_usd == pytest.approx(0.75) and client.last_cost_known
    # OpenRouter reports the call's cost; that figure is used as given.
    s2 = server(reply(usage={"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0123}))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", s2.url)
    client = LLMClient(provider="openrouter", notify=lambda m: None)
    client.complete("s", "p")
    assert client.last_cost_usd == pytest.approx(0.0123)


# --------------------------------------------------------------------------- #
# (c) a key that is set but unusable is a loud, one-line error
# --------------------------------------------------------------------------- #
def test_c_a_rejected_key_names_the_provider_and_the_fix_once(server, monkeypatch):
    s = server((401, {"error": {"message": "Invalid API Key", "code": "invalid_api_key"}}))
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", s.url)
    lines: list[str] = []
    client = LLMClient(required=True, notify=lines.append)
    assert client.complete("s", "p") is None
    assert client.complete("s", "p") is None          # no second call, no second line
    assert len(s.requests) == 1
    errors = [l for l in lines if l.startswith("LLM error")]
    assert len(errors) == 1 and "\n" not in errors[0]
    assert "(groq)" in errors[0] and "HTTP 401" in errors[0] and "GROQ_API_KEY" in errors[0]
    assert "deterministic" in errors[0]
    assert "test-key-not-real" not in " ".join(lines)
    assert not client.available


def test_c_a_selected_provider_without_its_key_points_at_the_key_that_is_set(monkeypatch):
    with pytest.raises(LLMConfigError) as exc:
        resolve("openai", env={"GROQ_API_KEY": "test-key-not-real"})
    assert "OPENAI_API_KEY is not set" in str(exc.value) and "--llm-provider groq" in str(exc.value)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    lines: list[str] = []
    client = LLMClient(provider="openai", required=True, notify=lines.append)
    assert not client.available and lines and lines[0].startswith("LLM error (openai)")


def test_c_an_anthropic_key_without_the_sdk_says_how_to_install_it(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setitem(sys.modules, "anthropic", None)   # import raises ImportError
    lines: list[str] = []
    client = LLMClient(required=True, notify=lines.append)
    assert not client.available
    assert "LLM error (anthropic)" in lines[0] and "pip install 'avaloka[llm]'" in lines[0]


def test_c_nothing_configured_is_quiet_unless_the_model_was_asked_for():
    lines: list[str] = []
    LLMClient(notify=lines.append)
    assert lines == []
    LLMClient(required=True, notify=lines.append)
    assert len(lines) == 1 and "no model provider is configured" in lines[0]


# --------------------------------------------------------------------------- #
# (d) a provider error is never a silent canned fallback
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status,needle", [
    (404, "ollama pull gemma4:e4b"), (429, "HTTP 429"), (503, "HTTP 503"),
])
def test_d_persona_falls_back_out_loud(server, monkeypatch, status, needle):
    from avaloka.persona import Avaloka
    s = server((status, {"error": {"message": "nope"}}))
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", s.url)
    lines: list[str] = []
    ava = Avaloka(llm=True, provider="local", notify=lines.append)
    text = ava.greet()
    assert "Hi, I'm Avaloka" in text                      # the deterministic line
    errors = [l for l in lines if l.startswith("LLM error")]
    assert len(errors) == 1 and needle in errors[0] and "deterministic" in errors[0]


def test_d_an_unreachable_local_server_is_reported_with_how_to_start_it(monkeypatch):
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", f"http://127.0.0.1:{dead_port()}/v1")
    lines: list[str] = []
    client = LLMClient(provider="local", required=True, notify=lines.append)
    assert client.complete("s", "p") is None
    assert "could not reach local model server" in lines[-1] and "ollama serve" in lines[-1]


def test_d_the_cli_says_so_when_llm_falls_back(churn_csv, tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from avaloka.cli import app
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", f"http://127.0.0.1:{dead_port()}/v1")
    result = CliRunner().invoke(app, ["analyze", str(churn_csv), "--goal", "x", "-o",
                                      str(tmp_path / "out"), "--llm-provider", "local"])
    assert result.exit_code == 0, result.output
    assert "LLM error (local)" in result.output and "deterministic" in result.output
    assert result.output.count("LLM error") == 1


def test_d_an_empty_reply_is_an_error_not_an_empty_line(server, monkeypatch):
    s = server(reply(""))
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", s.url)
    lines: list[str] = []
    client = LLMClient(provider="local", notify=lines.append)
    assert client.complete("s", "p") is None and "empty reply" in lines[-1]


# --------------------------------------------------------------------------- #
# (e) tool_use_failed and phantom tools get a re-prompt
# --------------------------------------------------------------------------- #
TOOL_USE_FAILED = (400, {"error": {"message": "Failed to call a function.", "type": "invalid_request_error",
                                   "code": "tool_use_failed", "failed_generation": "{\"name\": \"json\"}"}})
PHANTOM_400 = (400, {"error": {"message": "tool call validation failed: attempted to call tool "
                                          "'commentary' which was not in request.tools",
                               "type": "invalid_request_error"}})


def _groq(monkeypatch, s):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", s.url)
    return LLMClient(notify=lambda m: None)


def test_e_rejections_are_reprompted_then_succeed(server, monkeypatch):
    s = server(TOOL_USE_FAILED, reply("", tool_calls=[call("commentary")]),
               reply("", tool_calls=[call("run_sql", '{"q": 1}')]))
    msg = _groq(monkeypatch, s).chat([{"role": "user", "content": "how many rows?"}], TOOLS)
    assert msg["tool_calls"][0]["function"]["name"] == "run_sql"
    assert len(s.requests) == 3
    second, third = s.requests[1]["body"]["messages"], s.requests[2]["body"]["messages"]
    assert "could not parse it as a tool call" in second[-1]["content"]
    assert "'commentary', which is not one of the provided tools" in third[-1]["content"]
    assert "run_sql, return_answer" in third[-1]["content"]
    assert second[0] == {"role": "user", "content": "how many rows?"}


def test_e_groqs_phantom_commentary_400_is_a_rejection_too(server, monkeypatch):
    s = server(PHANTOM_400, reply("", tool_calls=[call("return_answer")]))
    msg = _groq(monkeypatch, s).chat([{"role": "user", "content": "q"}], TOOLS)
    assert msg["tool_calls"][0]["function"]["name"] == "return_answer" and len(s.requests) == 2


def test_e_rejections_are_bounded_and_end_as_tool_call_rejected(server, monkeypatch):
    s = server(TOOL_USE_FAILED)
    with pytest.raises(LLMToolCallRejected) as exc:
        _groq(monkeypatch, s).chat([{"role": "user", "content": "q"}], TOOLS)
    assert len(s.requests) == MAX_TOOL_CALL_REJECTIONS + 1
    assert isinstance(exc.value, ToolCallRejected)        # the data agent's class
    assert exc.value.provider == "groq" and "re-prompts" in str(exc.value)


def test_e_no_tool_call_when_one_is_required_is_reprompted(server, monkeypatch):
    s = server(reply("Here is my answer in prose."), reply("", tool_calls=[call("return_answer")]))
    _groq(monkeypatch, s).chat([{"role": "user", "content": "q"}], TOOLS)
    assert "contained no tool call" in s.requests[1]["body"]["messages"][-1]["content"]


def test_e_other_400s_are_not_retried(server, monkeypatch):
    s = server((400, {"error": {"message": "context_length_exceeded", "code": "context_length_exceeded"}}))
    with pytest.raises(LLMCallError) as exc:
        _groq(monkeypatch, s).chat([{"role": "user", "content": "q"}], TOOLS)
    assert not isinstance(exc.value, LLMToolCallRejected) and len(s.requests) == 1


# --------------------------------------------------------------------------- #
# (f) the voice-leak backstop, on what gemma4:e4b actually wrote
# --------------------------------------------------------------------------- #
RECORDED = json.loads((FIXTURES / "gemma4_e4b_voice_recorded.json").read_text())["cases"]


@pytest.mark.parametrize("case", [c for c in RECORDED if c["kind"] == "leak"],
                         ids=lambda c: c["recorded"][:40])
def test_f_recorded_leaks_never_reach_the_user(case):
    out = clean_rephrasing(case["original"], case["recorded"])
    assert is_meta(case["recorded"])
    assert not is_meta(out), out
    assert out != case["recorded"]
    for word in ("rephras", "refram", "restructur", "streamlin", "this statement", "the user"):
        assert word not in out.lower()


def test_f_the_backstop_keeps_the_models_wording_when_only_the_preamble_leaks():
    """Falling back to the original for every leak would also pass the test
    above; the point is to keep the rewording and drop only the narration."""
    leaks = [c for c in RECORDED if c["kind"] == "leak"]
    kept = [c for c in leaks if clean_rephrasing(c["original"], c["recorded"]) != c["original"]]
    assert len(kept) >= len(leaks) - 1, [c["recorded"][:60] for c in leaks if c not in kept]


@pytest.mark.parametrize("case", [c for c in RECORDED if c["kind"] == "clean"],
                         ids=lambda c: c["recorded"][:40])
def test_f_recorded_clean_rephrasings_pass_through(case):
    assert clean_rephrasing(case["original"], case["recorded"]) == case["recorded"]


def test_f_a_number_the_original_did_not_have_falls_back_to_the_original():
    original = "All 9 copies are back."
    assert clean_rephrasing(original, "All 10 copies are back, 95% sure.") == original
    assert clean_rephrasing(original, "All nine copies are back.") == "All nine copies are back."


def test_f_end_to_end_the_persona_never_prints_the_leak(server, monkeypatch):
    from avaloka.persona import VOICE_SYSTEM, Avaloka
    leak = next(c for c in RECORDED if c["recorded"].startswith("I'm going to rephrase this to cleanly"))
    s = server(reply(leak["recorded"]))
    monkeypatch.setenv("AVALOKA_LLM_BASE_URL", s.url)
    ava = Avaloka(llm=True, provider="local", notify=lambda m: None)
    out = ava._voice(leak["original"])
    assert not is_meta(out) and "Gender" in out
    sent = s.requests[0]["body"]["messages"]
    assert sent[0]["content"] == VOICE_SYSTEM and sent[1]["content"].startswith("Line: ")


def test_f_the_prompt_states_the_output_contract():
    from avaloka.persona import VOICE_SYSTEM
    assert "NOTHING else" in VOICE_SYSTEM and "No preamble" in VOICE_SYSTEM
    # The instruction the model was being meta-cognitive about is gone.
    assert "meta-cognitive" not in VOICE_SYSTEM


# --------------------------------------------------------------------------- #
# (g) the transport stays inside the CLI's footprint
# --------------------------------------------------------------------------- #
def test_g_the_llm_package_imports_no_sdk_and_no_app():
    import subprocess
    probe = ("import sys, avaloka.llm, avaloka.cli; "
             "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
             "('app', 'anthropic', 'openai', 'requests', 'httpx', 'langchain_core')); print(bad)")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[2]))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
