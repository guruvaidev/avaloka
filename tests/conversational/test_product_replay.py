"""The conversational probes, replayed through Avaloka's compiled agent graph.

Two halves. The first pins the replay model in-process: a recorded reply comes
back for the same request, and a request with no recording is a recorded
failure, never a fallback. The second runs the real thing -- every probe
through ``build_graph()`` in a fresh interpreter, the model router replaced by
the cassette -- and asserts each probe was sent, walked the graph, was answered
from the recording and was graded.

This file is the product family's CI home. The Benchmark Scorecard step
installs only the benchmark extra, so ``app`` is not importable there and the
product tasks SKIP; the Python test step has the full requirements, and this is
where the replay must pass. A SKIP here would be a green line that measured
nothing, so an unavailable application FAILS.
"""

from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from avaloka.benchmark import conversational as conv
from avaloka.benchmark import conversational_product as product
from avaloka.benchmark.known_defects import KNOWN_DEFECTS, gate

PROBES_FILE = Path(__file__).with_name("probes.py")
CASSETTE = product.default_cassette(PROBES_FILE)


def _models():
    return product._model_classes()


def _request(text="How many rows are in this dataset?"):
    return [SystemMessage(content="You are Avaloka."), HumanMessage(content=text)]


# --------------------------------------------------------------------------- #
# The replay model, in-process
# --------------------------------------------------------------------------- #

def test_a_request_with_no_recording_fails_and_is_ledgered(tmp_path):
    replay_cls, _ = _models()
    cassette = product.Cassette(tmp_path / "absent.jsonl")
    cassette.begin("int_unambiguous_rowcount")
    model = replay_cls(agent="CONVERSATIONAL", cassette=cassette)
    with pytest.raises(product.CassetteMiss, match="CONVERSATIONAL"):
        model.invoke(_request())
    (miss,) = cassette.misses["int_unambiguous_rowcount"]
    assert miss["agent"] == "CONVERSATIONAL" and len(miss["key"]) == 64
    assert "How many rows" in miss["excerpt"]


def test_a_recorded_reply_replays_exactly_and_a_changed_prompt_misses(tmp_path):
    replay_cls, tap_cls = _models()
    path = tmp_path / "product.jsonl"
    recorded = AIMessage(content="There are 20,640 rows.",
                         tool_calls=[{"name": "ask_user", "args": {"q": "gross?"}, "id": "c1"}])
    recorder = product.Cassette(path, recording=True)
    recorder.begin("int_unambiguous_rowcount")
    tap = tap_cls(agent="CONVERSATIONAL", cassette=recorder,
                  inner=GenericFakeChatModel(messages=iter([recorded])))
    assert tap.invoke(_request()).content == "There are 20,640 rows."
    recorder.save({"cassette": "test", "models": ["fake"]})

    cassette = product.Cassette(path)
    assert cassette.loaded and cassette.header["models"] == ["fake"]
    cassette.begin("int_unambiguous_rowcount")
    replay = replay_cls(agent="CONVERSATIONAL", cassette=cassette)
    reply = replay.invoke(_request())
    assert reply.content == "There are 20,640 rows."
    assert reply.tool_calls[0]["name"] == "ask_user" and reply.tool_calls[0]["args"] == {"q": "gross?"}
    assert cassette.calls["int_unambiguous_rowcount"] == 1

    # Asked a second time, the recording has no second answer: a miss.
    with pytest.raises(product.CassetteMiss):
        replay.invoke(_request())
    # A different prompt, or the same prompt from another agent, is a miss too.
    with pytest.raises(product.CassetteMiss):
        replay.invoke(_request("How many columns?"))
    with pytest.raises(product.CassetteMiss):
        replay_cls(agent="PLANNER", cassette=cassette).invoke(_request())
    assert len(cassette.misses["int_unambiguous_rowcount"]) == 3


def test_the_key_ignores_incidental_ids_and_sees_everything_else():
    base, _ = product.request_key("PLANNER", _request("run 3f2b8c1e-1d2a-4b3c-9d8e-7f6a5b4c3d2e"))
    same, _ = product.request_key("PLANNER", _request("run 0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"))
    assert base == same
    assert product.request_key("CODER", _request("run 3f2b8c1e-1d2a-4b3c-9d8e-7f6a5b4c3d2e"))[0] != base
    plain, _ = product.request_key("PLANNER", _request())
    with_choice, _ = product.request_key("PLANNER", _request(), kwargs={"tool_choice": "required"})
    with_stop, _ = product.request_key("PLANNER", _request(), stop=["\n"])
    assert len({plain, with_choice, with_stop}) == 3


def test_a_failed_live_call_is_a_hole_in_the_recording(tmp_path):
    _, tap_cls = _models()

    class Broken(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise TimeoutError("model did not answer")

    cassette = product.Cassette(tmp_path / "c.jsonl", recording=True)
    cassette.begin("resp_quick_answer")
    tap = tap_cls(agent="NARRATOR", cassette=cassette, inner=Broken(messages=iter([])))
    with pytest.raises(TimeoutError):
        tap.invoke(_request())
    assert cassette.misses["resp_quick_answer"][0]["error"].startswith("TimeoutError")
    assert cassette.recorded == []


def test_a_hosted_recording_stops_at_its_spend_cap(tmp_path):
    """Past the cap the next call is refused before it is made, and ledgered."""
    _, tap_cls = _models()
    usage = {"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000}
    replies = iter([AIMessage(content="one", usage_metadata=usage),
                    AIMessage(content="two", usage_metadata=usage)])
    cassette = product.Cassette(tmp_path / "c.jsonl", recording=True)
    cassette.rates, cassette.cap_usd = (0.15, 0.60), 0.10
    cassette.begin("comp_baseline_honesty")
    tap = tap_cls(agent="PLANNER", cassette=cassette, inner=GenericFakeChatModel(messages=replies))
    assert tap.invoke(_request()).content == "one"          # $0.15 spent: over the cap
    with pytest.raises(RuntimeError, match="spend cap"):
        tap.invoke(_request("again"))
    assert "spend cap" in cassette.misses["comp_baseline_honesty"][0]["error"]
    assert len(cassette.recorded) == 1


def test_a_recording_keeps_the_providers_generation_id(tmp_path):
    _, tap_cls = _models()
    reply = AIMessage(content="ok", response_metadata={"id": "gen-1762000000-abc"})
    cassette = product.Cassette(tmp_path / "c.jsonl", recording=True)
    cassette.begin("comp_baseline_honesty")
    tap_cls(agent="PLANNER", cassette=cassette,
            inner=GenericFakeChatModel(messages=iter([reply]))).invoke(_request())
    assert cassette.recorded[0]["generation_id"] == "gen-1762000000-abc"


def test_replay_runs_under_the_recorded_provider(tmp_path, monkeypatch):
    """Provider-dependent request arguments must match the recording, so the
    replay selects the provider the header names -- with a placeholder key that
    satisfies the agents' "is a key set" check and is never sent."""
    cassette = tmp_path / "c.jsonl"
    cassette.write_text('{"cassette": "test", "provider": "openrouter"}\n')
    seen = {}

    def fake_run(command, cwd, env, **kw):
        seen.update(env)
        raise product.subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(product.subprocess, "run", fake_run)
    product.run_product(PROBES_FILE, ["comp_baseline_honesty"], cassette=cassette)
    assert seen["INFERENCE_PROVIDER"] == "openrouter"
    assert seen["OPENROUTER_API_KEY"] == product.REPLAY_KEY_PLACEHOLDER
    assert "GROQ_API_KEY" not in seen


def test_a_hosted_provider_must_be_priced(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.delenv(product.INPUT_RATE_ENV, raising=False)
    with pytest.raises(RuntimeError, match="priced"):
        product.live_settings(None, "openrouter")
    with pytest.raises(RuntimeError, match="Anthropic"):
        product.live_settings("anthropic/claude-haiku-5.5", "openrouter")


def test_the_replay_names_no_anthropic_model():
    with pytest.raises(RuntimeError, match="Anthropic"):
        product._refuse_anthropic("anthropic/claude-haiku-5.5")


# --------------------------------------------------------------------------- #
# The real thing: every probe through the compiled graph, from the cassette
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def replay_results():
    return conv.conversational_results(live=False)


def test_the_cassette_is_committed_and_names_its_models():
    cassette = product.Cassette(CASSETTE)
    assert cassette.loaded, f"no cassette at {CASSETTE}"
    models = " ".join(cassette.header.get("models", []))
    assert "gpt-oss" in models and cassette.header.get("provider") == "openrouter", cassette.header
    assert "claude" not in models.lower() and "anthropic" not in models.lower()
    probe_ids = {p.id for p in conv.load_probe_suite().module.PROBES}
    assert set(cassette.header.get("probes", [])) == probe_ids


def test_recorded_code_imports_no_network_module():
    """``execute_locally`` runs recorded scripts in a child the socket guard does
    not reach, so a re-recording that brings a network client must not land."""
    import json
    import re
    pattern = re.compile(r"(?:^|\\n|;)\s*(?:import|from)\s+"
                         r"(requests|urllib\d?|socket|http|httpx|aiohttp|boto3|ftplib|smtplib|"
                         r"subprocess|paramiko)\b")
    entries = [json.loads(line) for line in CASSETTE.read_text(encoding="utf-8").splitlines()]
    hits = [(e["probe"], e["agent"], m.group(1)) for e in entries if "response" in e
            for m in pattern.finditer(json.dumps(e["response"]))]
    assert hits == []


def test_every_probe_was_sent_walked_the_graph_and_was_answered_from_the_recording(replay_results):
    product_tasks = [r for r in replay_results if r.family == conv.PRODUCT_FAMILY]
    assert len(product_tasks) == 4
    for task in product_tasks:
        assert not task.error, f"{task.task} did not run: {task.error}"
        assert task.source_scheme == "product:replay"
    checks = {c.name: c for t in product_tasks for c in t.checks}
    for probe in conv.load_probe_suite().module.PROBES:
        assert probe.id in checks, f"{probe.id} was not graded"
        for suffix in ("question_sent", "through_graph", "recorded"):
            check = checks[f"{probe.id}::{suffix}"]
            assert check.passed, f"{probe.id}::{suffix}: {check.detail}"


def test_the_measurement_check_passes_on_the_replay(replay_results):
    (gap,) = [r for r in replay_results if r.task == conv.GAP_TASK]
    assert gap.passed, gap.checks[0].detail
    assert gap.source_scheme == "product:replay"


def test_the_gate_agrees_with_the_ledger(replay_results):
    """Strict both ways: every failing product task is declared, and none that
    is declared passes. The product's measured failures are the ledger's."""
    product_runs = [r for r in replay_results
                    if r.family == conv.PRODUCT_FAMILY or r.task == conv.GAP_TASK]
    verdict = gate(product_runs)
    assert verdict.ok, verdict.as_dict()
    failing = {r.task for r in product_runs if not r.passed}
    assert failing == {k for k in KNOWN_DEFECTS if k.startswith("conv_product_")}


def test_a_missing_recording_fails_the_run(tmp_path):
    """No cassette: every model request misses, so the probe and the measurement
    check fail -- the canned text the agents fall back to is never graded as an
    answer."""
    run = product.run_product(PROBES_FILE, ["int_unambiguous_rowcount"], mode=product.REPLAY,
                              cassette=tmp_path / "missing.jsonl")
    assert not run.unavailable, run.unavailable
    (record,) = run.records.values()
    assert record.misses, "a request with no recording must be ledgered"
    assert run.calls == 0
    suite = conv.load_probe_suite()
    gaps = conv._measurement_gaps(suite, run, "")
    assert any("had no recording" in g for g in gaps)


def test_the_replay_opens_no_network_socket(replay_results):
    run = product.run_product(PROBES_FILE, ["comp_fabricate_segment"], mode=product.REPLAY)
    assert not run.unavailable and not run.error
    assert run.external_egress == [], run.egress
