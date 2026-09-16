"""
Integration tests for the persistence naming + versioning changes.

These exercise _persist_assets_background end-to-end with the cloud store,
GitHub write, and session service all mocked — so they run offline and assert
on the KEYS/PATHS produced, which is exactly what Leela's review was about.

IMPORTANT — adjust these two things to your repo before running:
  1. The import below (`from app.services import persistence_service as P`)
     must point at your real module.
  2. The monkeypatch targets patch functions *as seen inside* persistence_service.
     If your module imports them differently, adjust the attribute names.

Run:  pytest tests/test_persistence_integration.py -v
"""
import json
import pytest

# --- point this at your real module -----------------------------------------
from app.services import persistence_service as P   # noqa


# ── Shared fakes ─────────────────────────────────────────────────────────────

class FakeSessionStore:
    """Mimics session_service get/update with an in-memory dict."""
    def __init__(self, initial=None):
        self.data = dict(initial or {})

    async def get_session(self, sid):
        return dict(self.data)

    async def save_session(self, sid, sess):
        self.data = dict(sess)

    async def update_session(self, sid, mutator):
        # Mirror the real atomic read-modify-write: call mutator on live dict.
        mutator(self.data)


@pytest.fixture
def captured(monkeypatch):
    """
    Patch every outbound side effect of _persist_assets_background and capture
    the object keys / git paths it produces. Returns a dict of captures.
    """
    caps = {"code": [], "output": [], "viz": [], "git": []}

    async def fake_code(user_id, session_id, dataset_id, generated_code, prompt_ts,
                        connection_id=None, storage_uri=None, **kw):
        key = P._object_key("code-registry", user_id, session_id, dataset_id,
                            prompt_ts, "transform.py", **kw)
        caps["code"].append(key)
        return {"status": "success", "object_key": key, "prompt_ts": prompt_ts}

    async def fake_output(user_id, session_id, dataset_id, ofd, oj, ol, prompt_ts,
                          connection_id=None, storage_uri=None, **kw):
        key = P._object_key("execution-outputs", user_id, session_id, dataset_id,
                            prompt_ts, "output.csv", **kw)
        caps["output"].append(key)
        return {"status": "success", "object_key": key, "prompt_ts": prompt_ts}

    async def fake_git(user_id, session_id, dataset_id, planner_definition,
                       coder_definition, execution_result, code_object_key,
                       prompt_ts, *, dataset_label="", analysis_label="", version=None):
        # Reproduce the real gate + base-path so the test asserts real behavior.
        status = str((execution_result or {}).get("status", "")).lower()
        if status not in ("success", "succeeded", "completed", "done"):
            return {"status": "skipped", "reason": "execution not successful"}
        base = P._git_base_for_test(dataset_id, session_id, prompt_ts,
                                    dataset_label=dataset_label,
                                    analysis_label=analysis_label,
                                    version=version) \
            if hasattr(P, "_git_base_for_test") else \
            f"jobs/{(P._slug(dataset_label) or 'dataset')}-{P._short(dataset_id)}/" \
            f"{(P._slug(analysis_label) or 'analysis')}-{P._short(session_id)}/" \
            f"v{version or 1}__{prompt_ts}"
        caps["git"].append(f"{base}_transform.py")
        return {"status": "success", "branch": "main", "repo": "acme/repo",
                "source": "env", "file_path": f"{base}_transform.py",
                "written": [f"{base}_transform.py"]}

    monkeypatch.setattr(P, "persist_generated_code_to_store", fake_code)
    monkeypatch.setattr(P, "persist_execution_output_to_store", fake_output)
    monkeypatch.setattr(P, "persist_job_definition_to_git", fake_git)

    # viz is optional in these tests; stub it to a no-op success
    async def fake_viz(*a, **k):
        return {"status": "skipped", "reason": "no viz in test"}
    monkeypatch.setattr(P, "persist_visualization_to_store", fake_viz)

    return caps


def _patch_session(monkeypatch, store):
    import app.services.session_service as S
    monkeypatch.setattr(S, "get_session", store.get_session, raising=False)
    monkeypatch.setattr(S, "save_session", store.save_session, raising=False)
    monkeypatch.setattr(S, "update_session", store.update_session, raising=False)


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_run_produces_meaningful_paths(captured, monkeypatch):
    """send_message-style call WITH labels → readable dataset/analysis folders."""
    store = FakeSessionStore({"code_assets": []})
    _patch_session(monkeypatch, store)

    await P._persist_assets_background(
        user_id="user-abc", session_id="5e0efc71-68b1", dataset_id="fec43785-e3f3",
        generated_code="import pandas as pd\n", planner_definition={"plan": "x"},
        coder_definition={"code": "import pandas as pd\n"},
        execution_result={"status": "success"}, exec_succeeded=True,
        output_file_data=None, output_json=[{"a": 1}], output_location=None,
        visualization_config=None, connection_id=None, storage_uri=None,
        dataset_label="Global YouTube Statistics",
        analysis_label="For each Country and category calculate averages",
    )

    assert captured["code"], "code asset should have been persisted"
    assert "global-youtube-statistics-fec43785" in captured["code"][0]
    assert "dataset-" not in captured["code"][0]      # no GUID fallback
    assert captured["git"], "git write should have happened for a success run"
    assert "global-youtube-statistics-fec43785" in captured["git"][0]
    assert "/v1__" in captured["git"][0]


@pytest.mark.asyncio
async def test_edit_run_without_labels_uses_guid_fallback(captured, monkeypatch):
    """The v2/v5 bug: no labels → dataset-<guid>/analysis-<guid>."""
    store = FakeSessionStore({"code_assets": []})
    _patch_session(monkeypatch, store)

    await P._persist_assets_background(
        user_id="user-abc", session_id="5e0efc71-68b1", dataset_id="fec43785-e3f3",
        generated_code="import pandas as pd\n", planner_definition={"plan": ""},
        coder_definition={"code": "import pandas as pd\n"},
        execution_result={"status": "success"}, exec_succeeded=True,
        output_file_data=None, output_json=None, output_location=None,
        visualization_config=None, connection_id=None, storage_uri=None,
        # no dataset_label / analysis_label -> defaults ""
    )
    assert "/dataset-fec43785/" in captured["code"][0]
    assert "/analysis-5e0efc71/" in captured["code"][0]


@pytest.mark.asyncio
async def test_version_increments_with_history(captured, monkeypatch):
    """Version = len(existing code_assets) + 1."""
    store = FakeSessionStore({"code_assets": [
        {"object_key": "k1", "prompt_ts": "t1"},
        {"object_key": "k2", "prompt_ts": "t2"},
    ]})
    _patch_session(monkeypatch, store)

    await P._persist_assets_background(
        user_id="u", session_id="s-guid", dataset_id="d-guid",
        generated_code="code", planner_definition={"plan": "x"},
        coder_definition={"code": "code"},
        execution_result={"status": "success"}, exec_succeeded=True,
        output_file_data=None, output_json=None, output_location=None,
        visualization_config=None, connection_id=None, storage_uri=None,
        dataset_label="ds", analysis_label="an",
    )
    # 2 existing -> this run is v3
    assert "/v3__" in captured["git"][0]


@pytest.mark.asyncio
async def test_version_prompts_populated(captured, monkeypatch):
    """version_prompts[prompt_ts] should record the analysis label."""
    store = FakeSessionStore({"code_assets": []})
    _patch_session(monkeypatch, store)

    await P._persist_assets_background(
        user_id="u", session_id="s-guid", dataset_id="d-guid",
        generated_code="code", planner_definition={"plan": "x"},
        coder_definition={"code": "code"},
        execution_result={"status": "success"}, exec_succeeded=True,
        output_file_data=None, output_json=None, output_location=None,
        visualization_config=None, connection_id=None, storage_uri=None,
        dataset_label="ds", analysis_label="my question",
    )
    vp = store.data.get("version_prompts")
    assert isinstance(vp, dict)
    assert "my question" in vp.values()


@pytest.mark.asyncio
async def test_dryrun_edit_does_not_reach_git(captured, monkeypatch):
    """Current production behavior: dryrun edit runs skip the Git write."""
    store = FakeSessionStore({"code_assets": []})
    _patch_session(monkeypatch, store)

    await P._persist_assets_background(
        user_id="u", session_id="s-guid", dataset_id="d-guid",
        generated_code="code", planner_definition={"plan": "x"},
        coder_definition={"code": "code"},
        execution_result={"status": "dryrun"}, exec_succeeded=True,
        output_file_data=None, output_json=None, output_location=None,
        visualization_config=None, connection_id=None, storage_uri=None,
        dataset_label="ds", analysis_label="an",
    )
    # Code still persisted to store, but NOTHING committed to Git.
    assert captured["code"], "code should still go to cloud store"
    assert not captured["git"], "dryrun must not reach Git under current gate"