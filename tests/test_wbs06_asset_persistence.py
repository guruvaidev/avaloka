"""
Integration Tests — WBS 0.6 Asset Persistence / Coder Output Registry
======================================================================
Tests all four persistence tasks end-to-end against real GCS + GitHub.

Run with:
    pytest tests/test_wbs06_asset_persistence.py -v

Required env vars (same as .env):
    AVALOKA_GCS_BUCKET          — GCS bucket name
    GITHUB_SYSTEM_TOKEN         — GitHub PAT with repo scope
    GITHUB_JOB_REGISTRY_REPO    — e.g. guruvaidev/avaloka-jobs-intenal

Optional:
    AVALOKA_TEST_SKIP_GIT=1     — skip Task B (GitHub) tests
    AVALOKA_TEST_SKIP_GCS=1     — skip GCS upload tests
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# ── helpers ──────────────────────────────────────────────────────────────────
# Skip when the credentials are absent, not only when someone remembers to opt
# out. As an opt-out these flags defaulted to "run", so a checkout with no GCS
# credentials -- which is every CI run and every new contributor -- attempted
# real uploads and failed 22 tests that had nothing to say about the code. A
# test that cannot reach its dependency should report that it was skipped and
# why, not report the code as broken.
_GCS_CREDENTIALS = bool(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    or os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("AVALOKA_GCS_BUCKET")
)
SKIP_GCS = os.getenv("AVALOKA_TEST_SKIP_GCS") == "1" or not _GCS_CREDENTIALS
SKIP_GIT = os.getenv("AVALOKA_TEST_SKIP_GIT") == "1"

GCS_BUCKET    = os.getenv("AVALOKA_GCS_BUCKET", "avaloka-test-user-filestore")
GITHUB_TOKEN  = os.getenv("GITHUB_SYSTEM_TOKEN", "")
GITHUB_REPO   = os.getenv("GITHUB_JOB_REGISTRY_REPO", "guruvaidev/avaloka-jobs-intenal")


def _rand_id() -> str:
    return str(uuid.uuid4())


def _now_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def user_id():
    return "test-user-wbs06"


@pytest.fixture
def session_id():
    return _rand_id()


@pytest.fixture
def dataset_id():
    return _rand_id()


@pytest.fixture
def prompt_ts():
    return _now_ts()


@pytest.fixture
def sample_code():
    return """
import pandas as pd

def main(df):
    result = df.groupby('category')['subscribers'].sum().reset_index()
    result = result.sort_values('subscribers', ascending=False).head(10)
    return result
""".strip()


@pytest.fixture
def sample_output_json():
    return [
        {"category": "Music",         "subscribers": 500000000},
        {"category": "Entertainment", "subscribers": 400000000},
        {"category": "Education",     "subscribers": 300000000},
    ]


@pytest.fixture
def sample_viz_config():
    return {
        "version": "1.0",
        "charts": [
            {
                "type": "bar",
                "title": "Top Categories by Subscribers",
                "x_col": "category",
                "y_col": "subscribers",
            }
        ],
        "visualization_status": "ready",
    }


@pytest.fixture
def sample_execution_result():
    return {
        "status": "success",
        "message": "ETL job completed successfully",
        "output_file": "/tmp/output.csv",
    }


@pytest.fixture
def sample_planner_definition():
    return {
        "plan": "1. Load data\n2. Group by category\n3. Sum subscribers\n4. Sort descending",
        "steps": ["load", "group", "aggregate", "sort"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Mock GCS BlobStore
# ═══════════════════════════════════════════════════════════════════════════════

class MockGCSBlobStore:
    """In-memory mock of GCSBlobStore for unit tests."""

    def __init__(self):
        self._storage: Dict[str, bytes] = {}

    def put_file(self, local_path: str, key: str) -> str:
        with open(local_path, "rb") as f:
            self._storage[key] = f.read()
        return f"gs://{GCS_BUCKET}/{key}"

    def get_file(self, key: str, dest_path: str) -> None:
        with open(dest_path, "wb") as f:
            f.write(self._storage[key])

    def stat(self, key: str):
        data = self._storage.get(key, b"")
        return len(data), datetime.utcnow().isoformat()

    def delete(self, key: str) -> None:
        self._storage.pop(key, None)

    def list(self, prefix: str = ""):
        for key, data in self._storage.items():
            if key.startswith(prefix):
                yield key, len(data), datetime.utcnow().isoformat()

    def has_key(self, key: str) -> bool:
        return key in self._storage

    def get_content(self, key: str) -> bytes:
        return self._storage.get(key, b"")


@pytest.fixture
def mock_store():
    return MockGCSBlobStore()


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Task A: Generated Code → Cloud Storage
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskA_CodePersistence:

    @pytest.mark.asyncio
    async def test_persist_code_success(
        self, mock_store, user_id, session_id, dataset_id, prompt_ts, sample_code
    ):
        """Task A: generated .py file is uploaded to the correct GCS key."""
        from app.services.persistence_service import persist_generated_code_to_store

        with patch(
            "app.services.persistence_service._get_store_for_connection",
            return_value=(mock_store, ""),
        ):
            result = await persist_generated_code_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                generated_code=sample_code,
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "success"
        assert "object_key" in result

        expected_key = (
            f"code-registry/{user_id}/{session_id}/{dataset_id}_{prompt_ts}_transform.py"
        )
        assert result["object_key"] == expected_key
        assert mock_store.has_key(expected_key)

        stored_content = mock_store.get_content(expected_key).decode("utf-8")
        assert "def main(df):" in stored_content
        assert "groupby" in stored_content

    @pytest.mark.asyncio
    async def test_persist_code_skips_empty_code(
        self, mock_store, user_id, session_id, dataset_id, prompt_ts
    ):
        """Task A: empty code is skipped without error."""
        from app.services.persistence_service import persist_generated_code_to_store

        with patch(
            "app.services.persistence_service._get_store_for_connection",
            return_value=(mock_store, ""),
        ):
            result = await persist_generated_code_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                generated_code="",
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "skipped"
        assert result["reason"] == "no code to persist"

    @pytest.mark.asyncio
    async def test_persist_code_skips_whitespace_only(
        self, mock_store, user_id, session_id, dataset_id, prompt_ts
    ):
        """Task A: whitespace-only code is treated as empty."""
        from app.services.persistence_service import persist_generated_code_to_store

        with patch(
            "app.services.persistence_service._get_store_for_connection",
            return_value=(mock_store, ""),
        ):
            result = await persist_generated_code_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                generated_code="   \n\n\t  ",
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_persist_code_file_naming(
        self, mock_store, user_id, session_id, dataset_id, prompt_ts, sample_code
    ):
        """Task A: key follows the canonical naming pattern."""
        from app.services.persistence_service import persist_generated_code_to_store

        with patch(
            "app.services.persistence_service._get_store_for_connection",
            return_value=(mock_store, ""),
        ):
            result = await persist_generated_code_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                generated_code=sample_code,
                prompt_ts=prompt_ts,
            )

        key = result["object_key"]
        # Validate structure: prefix/user_id/session_id/dataset_id_ts_suffix
        parts = key.split("/")
        assert parts[0] == "code-registry"
        assert parts[1] == user_id
        assert parts[2] == session_id
        assert parts[3].endswith("_transform.py")
        assert dataset_id in parts[3]
        assert prompt_ts in parts[3]

    @pytest.mark.asyncio
    async def test_persist_code_multiple_prompts_create_separate_files(
        self, mock_store, user_id, session_id, dataset_id, sample_code
    ):
        """Task A: each prompt gets its own timestamped file (versioning)."""
        from app.services.persistence_service import persist_generated_code_to_store

        ts1 = "20260424_120000"
        ts2 = "20260424_120500"

        with patch(
            "app.services.persistence_service._get_store_for_connection",
            return_value=(mock_store, ""),
        ):
            r1 = await persist_generated_code_to_store(
                user_id=user_id, session_id=session_id, dataset_id=dataset_id,
                generated_code=sample_code + "\n# version 1",
                prompt_ts=ts1,
            )
            r2 = await persist_generated_code_to_store(
                user_id=user_id, session_id=session_id, dataset_id=dataset_id,
                generated_code=sample_code + "\n# version 2",
                prompt_ts=ts2,
            )

        assert r1["object_key"] != r2["object_key"]
        assert mock_store.has_key(r1["object_key"])
        assert mock_store.has_key(r2["object_key"])

        content1 = mock_store.get_content(r1["object_key"]).decode()
        content2 = mock_store.get_content(r2["object_key"]).decode()
        assert "version 1" in content1
        assert "version 2" in content2


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Task C: Execution Output → Cloud Storage
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskC_OutputPersistence:

    @pytest.mark.asyncio
    async def test_persist_output_from_json(
        self, mock_store, user_id, session_id, dataset_id, prompt_ts, sample_output_json
    ):
        """Task C: output_json rows are serialized to CSV and uploaded."""
        from app.services.persistence_service import persist_execution_output_to_store

        with patch(
            "app.services.persistence_service._get_store_for_connection",
            return_value=(mock_store, ""),
        ):
            result = await persist_execution_output_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                output_file_data=None,
                output_json=sample_output_json,
                output_location=None,
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "success"
        expected_key = (
            f"execution-outputs/{user_id}/{session_id}/{dataset_id}_{prompt_ts}_output.csv"
        )
        assert result["object_key"] == expected_key
        assert mock_store.has_key(expected_key)

        csv_content = mock_store.get_content(expected_key).decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["category"] == "Music"
        assert rows[1]["category"] == "Entertainment"

    @pytest.mark.asyncio
    async def test_persist_output_from_data_uri(
        self, mock_store, user_id, session_id, dataset_id, prompt_ts
    ):
        """Task C: base64 CSV data-uri is decoded and stored correctly."""
        import base64
        from app.services.persistence_service import persist_execution_output_to_store

        csv_text = "category,subscribers\nMusic,500000000\nEntertainment,400000000\n"
        b64 = base64.b64encode(csv_text.encode()).decode()
        output_file_data = {"content": f"data:text/csv;base64,{b64}", "filename": "output.csv", "size": len(csv_text)}

        with patch(
            "app.services.persistence_service._get_store_for_connection",
            return_value=(mock_store, ""),
        ):
            result = await persist_execution_output_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                output_file_data=output_file_data,
                output_json=None,
                output_location=None,
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "success"
        stored = mock_store.get_content(result["object_key"]).decode()
        assert "Music" in stored
        assert "500000000" in stored

    @pytest.mark.asyncio
    async def test_persist_output_skips_when_no_data(
        self, mock_store, user_id, session_id, dataset_id, prompt_ts
    ):
        """Task C: skips gracefully when no output data provided."""
        from app.services.persistence_service import persist_execution_output_to_store

        with patch(
            "app.services.persistence_service._get_store_for_connection",
            return_value=(mock_store, ""),
        ):
            result = await persist_execution_output_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                output_file_data=None,
                output_json=None,
                output_location=None,
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_persist_output_key_naming(
        self, mock_store, user_id, session_id, dataset_id, prompt_ts, sample_output_json
    ):
        """Task C: output key follows canonical naming pattern."""
        from app.services.persistence_service import persist_execution_output_to_store

        with patch(
            "app.services.persistence_service._get_store_for_connection",
            return_value=(mock_store, ""),
        ):
            result = await persist_execution_output_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                output_file_data=None,
                output_json=sample_output_json,
                output_location=None,
                prompt_ts=prompt_ts,
            )

        key = result["object_key"]
        parts = key.split("/")
        assert parts[0] == "execution-outputs"
        assert parts[1] == user_id
        assert parts[2] == session_id
        assert parts[3].endswith("_output.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Task D: Visualization Config → Cloud Storage
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskD_VizPersistence:

    @pytest.mark.asyncio
    async def test_persist_viz_config_success(
        self, mock_store, user_id, session_id, dataset_id, prompt_ts, sample_viz_config
    ):
        """Task D: viz config JSON is serialized and uploaded."""
        from app.services.persistence_service import persist_visualization_to_store

        with patch(
            "app.services.persistence_service._get_store_for_connection",
            return_value=(mock_store, ""),
        ):
            result = await persist_visualization_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                visualization_config=sample_viz_config,
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "success"
        expected_key = (
            f"visualization-configs/{user_id}/{session_id}/{dataset_id}_{prompt_ts}_viz_config.json"
        )
        assert result["object_key"] == expected_key
        assert mock_store.has_key(expected_key)

        stored = json.loads(mock_store.get_content(expected_key).decode())
        assert stored["version"] == "1.0"
        assert len(stored["charts"]) == 1
        assert stored["charts"][0]["type"] == "bar"

    @pytest.mark.asyncio
    async def test_persist_viz_config_skips_empty(
        self, mock_store, user_id, session_id, dataset_id, prompt_ts
    ):
        """Task D: empty viz config is skipped."""
        from app.services.persistence_service import persist_visualization_to_store

        with patch(
            "app.services.persistence_service._get_store_for_connection",
            return_value=(mock_store, ""),
        ):
            result = await persist_visualization_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                visualization_config={},
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_persist_viz_config_key_naming(
        self, mock_store, user_id, session_id, dataset_id, prompt_ts, sample_viz_config
    ):
        """Task D: viz config key uses correct prefix and suffix."""
        from app.services.persistence_service import persist_visualization_to_store

        with patch(
            "app.services.persistence_service._get_store_for_connection",
            return_value=(mock_store, ""),
        ):
            result = await persist_visualization_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                visualization_config=sample_viz_config,
                prompt_ts=prompt_ts,
            )

        key = result["object_key"]
        parts = key.split("/")
        assert parts[0] == "visualization-configs"
        assert parts[3].endswith("_viz_config.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Task B: Job Definition → Git
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskB_GitPersistence:

    @pytest.mark.asyncio
    async def test_persist_job_skips_without_token(
        self, user_id, session_id, dataset_id, prompt_ts,
        sample_planner_definition, sample_execution_result
    ):
        """Task B: skips gracefully when no GitHub token configured."""
        from app.services.persistence_service import persist_job_definition_to_git

        with patch("app.services.persistence_service.GITHUB_TOKEN", ""):
            result = await persist_job_definition_to_git(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                planner_definition=sample_planner_definition,
                coder_definition={"code": "import pandas as pd"},
                execution_result=sample_execution_result,
                code_object_key="code-registry/test/key.py",
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "skipped"
        assert result["reason"] == "GITHUB_SYSTEM_TOKEN not set"

    @pytest.mark.asyncio
    async def test_persist_job_skips_on_failed_execution(
        self, user_id, session_id, dataset_id, prompt_ts, sample_planner_definition
    ):
        """Task B: does not write to Git when execution failed."""
        from app.services.persistence_service import persist_job_definition_to_git

        with patch("app.services.persistence_service.GITHUB_TOKEN", "fake-token"):
            result = await persist_job_definition_to_git(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                planner_definition=sample_planner_definition,
                coder_definition={},
                execution_result={"status": "error", "message": "Script failed"},
                code_object_key="code-registry/test/key.py",
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "skipped"
        assert "not successful" in result["reason"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exec_status", ["success", "succeeded", "completed", "done"])
    async def test_persist_job_accepts_all_success_statuses(
        self, exec_status, user_id, session_id, dataset_id, prompt_ts, sample_planner_definition
    ):
        """Task B: all valid success status strings trigger the Git write."""
        from app.services.persistence_service import persist_job_definition_to_git

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "default_branch": "main",
            "object": {"sha": "abc123"},
        }

        mock_put_response = MagicMock()
        mock_put_response.status_code = 201

        mock_get_ref_response = MagicMock()
        mock_get_ref_response.status_code = 200
        mock_get_ref_response.json.return_value = {"object": {"sha": "abc123"}}

        mock_get_contents_response = MagicMock()
        mock_get_contents_response.status_code = 404

        with patch("app.services.persistence_service.GITHUB_TOKEN", "fake-token"), \
             patch("app.services.persistence_service.GITHUB_JOB_REPO", "test/repo"), \
             patch("httpx.AsyncClient") as mock_client_cls:

            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            # GET repo → 200
            # GET ref/heads/main → 200 with sha
            # GET file contents → 404 (new file)
            # PUT file → 201
            mock_client.get.side_effect = [
                mock_response,
                mock_get_ref_response,
                mock_get_contents_response,
            ]
            mock_client.put.return_value = mock_put_response

            result = await persist_job_definition_to_git(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                planner_definition=sample_planner_definition,
                coder_definition={},
                execution_result={"status": exec_status},
                code_object_key="code-registry/test/key.py",
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "success"
        assert result["branch"] == "main"

    @pytest.mark.asyncio
    async def test_persist_job_pushes_to_main_not_branch(
        self, user_id, session_id, dataset_id, prompt_ts, sample_planner_definition
    ):
        """Task B: pushes directly to main per Leela's guidance, no branch creation."""
        from app.services.persistence_service import persist_job_definition_to_git

        mock_repo_response = MagicMock()
        mock_repo_response.status_code = 200
        mock_repo_response.json.return_value = {"default_branch": "main"}

        mock_ref_response = MagicMock()
        mock_ref_response.status_code = 200
        mock_ref_response.json.return_value = {"object": {"sha": "abc123"}}

        mock_get_contents = MagicMock()
        mock_get_contents.status_code = 404

        mock_put_response = MagicMock()
        mock_put_response.status_code = 201

        with patch("app.services.persistence_service.GITHUB_TOKEN", "fake-token"), \
             patch("app.services.persistence_service.GITHUB_JOB_REPO", "test/repo"), \
             patch("httpx.AsyncClient") as mock_client_cls:

            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.get.side_effect = [mock_repo_response, mock_ref_response, mock_get_contents]
            mock_client.put.return_value = mock_put_response

            result = await persist_job_definition_to_git(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                planner_definition=sample_planner_definition,
                coder_definition={},
                execution_result={"status": "success"},
                code_object_key="code-registry/test/key.py",
                prompt_ts=prompt_ts,
            )

        # Confirm branch is always "main"
        assert result["branch"] == "main"

        # Confirm the PUT was called (file write)
        mock_client.put.assert_called_once()
        put_call_kwargs = mock_client.put.call_args
        put_json = put_call_kwargs.kwargs.get("json") or put_call_kwargs.args[1] if put_call_kwargs.args else {}
        assert put_json.get("branch") == "main"

    @pytest.mark.asyncio
    async def test_persist_job_file_path_structure(
        self, user_id, session_id, dataset_id, prompt_ts, sample_planner_definition
    ):
        """Task B: file_path follows jobs/{user_id}/{session_id}/{...}_job_definition.json."""
        from app.services.persistence_service import persist_job_definition_to_git

        mock_repo_response = MagicMock()
        mock_repo_response.status_code = 200
        mock_repo_response.json.return_value = {"default_branch": "main"}

        mock_ref_response = MagicMock()
        mock_ref_response.status_code = 200
        mock_ref_response.json.return_value = {"object": {"sha": "abc123"}}

        mock_get_contents = MagicMock()
        mock_get_contents.status_code = 404

        mock_put_response = MagicMock()
        mock_put_response.status_code = 201

        with patch("app.services.persistence_service.GITHUB_TOKEN", "fake-token"), \
             patch("app.services.persistence_service.GITHUB_JOB_REPO", "test/repo"), \
             patch("httpx.AsyncClient") as mock_client_cls:

            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.get.side_effect = [mock_repo_response, mock_ref_response, mock_get_contents]
            mock_client.put.return_value = mock_put_response

            result = await persist_job_definition_to_git(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                planner_definition=sample_planner_definition,
                coder_definition={},
                execution_result={"status": "success"},
                code_object_key="code-registry/test/key.py",
                prompt_ts=prompt_ts,
            )

        file_path = result["file_path"]
        assert file_path.startswith(f"jobs/{user_id}/{session_id}/")
        assert file_path.endswith("_job_definition.json")
        assert dataset_id in file_path
        assert prompt_ts in file_path


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Background Orchestrator (_persist_assets_background)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackgroundOrchestrator:

    @pytest.mark.asyncio
    async def test_all_tasks_fire_on_successful_execution(
        self, mock_store, user_id, session_id, dataset_id,
        sample_code, sample_output_json, sample_viz_config,
        sample_execution_result, sample_planner_definition
    ):
        """Orchestrator: all 4 tasks fire when execution succeeded."""
        from app.services.persistence_service import _persist_assets_background

        mock_sess: Dict[str, Any] = {}

        async def mock_get_session(sid):
            return dict(mock_sess)

        async def mock_save_session(sid, data):
            mock_sess.update(data)
            return True

        # with patch("app.services.persistence_service._get_store_for_connection",
        #            return_value=(mock_store, "")), \
        #      patch("app.services.persistence_service.GITHUB_TOKEN", ""), \
        #      patch("app.services.session_service.get_session", side_effect=mock_get_session), \
        #      patch("app.services.session_service.save_session", side_effect=mock_save_session):
        ts_values = ["20260424_120000", "20260424_120500"]
        ts_iter = iter(ts_values)

        with patch("app.services.persistence_service._get_store_for_connection",
                   return_value=(mock_store, "")), \
             patch("app.services.persistence_service.GITHUB_TOKEN", ""), \
             patch("app.services.persistence_service.datetime") as mock_dt, \
             patch("app.services.session_service.get_session", side_effect=mock_get_session), \
             patch("app.services.session_service.save_session", side_effect=mock_save_session):

            mock_dt.utcnow.return_value.strftime.side_effect = lambda fmt: next(ts_iter)

            await _persist_assets_background(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                generated_code=sample_code,
                planner_definition=sample_planner_definition,
                coder_definition={"code": sample_code},
                execution_result=sample_execution_result,
                exec_succeeded=True,
                output_file_data=None,
                output_json=sample_output_json,
                output_location=None,
                visualization_config=sample_viz_config,
                connection_id=None,
                storage_uri=None,
            )

        # Task A — code key written
        assert "gcs_code_object_key" in mock_sess
        assert mock_sess["gcs_code_object_key"].startswith("code-registry/")

        # Task C — output key written
        assert "gcs_output_object_key" in mock_sess
        assert mock_sess["gcs_output_object_key"].startswith("execution-outputs/")

        # Task D — viz key written
        assert "gcs_viz_object_key" in mock_sess
        assert mock_sess["gcs_viz_object_key"].startswith("visualization-configs/")

        # errors must be absent
        assert "persist_errors" not in mock_sess

    @pytest.mark.asyncio
    async def test_does_not_fire_without_code_or_output(
        self, mock_store, user_id, session_id, dataset_id
    ):
        """Orchestrator: returns early when nothing to persist."""
        from app.services.persistence_service import _persist_assets_background

        mock_sess: Dict[str, Any] = {}

        async def mock_get_session(sid):
            return dict(mock_sess)

        async def mock_save_session(sid, data):
            mock_sess.update(data)
            return True

        with patch("app.services.persistence_service._get_store_for_connection",
                   return_value=(mock_store, "")), \
             patch("app.services.persistence_service.GITHUB_TOKEN", ""), \
             patch("app.services.session_service.get_session", side_effect=mock_get_session), \
             patch("app.services.session_service.save_session", side_effect=mock_save_session):

            await _persist_assets_background(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                generated_code=None,
                planner_definition=None,
                coder_definition=None,
                execution_result={},
                exec_succeeded=False,
                output_file_data=None,
                output_json=None,
                output_location=None,
                visualization_config=None,
                connection_id=None,
                storage_uri=None,
            )

        # session was never updated — no keys written
        assert "gcs_code_object_key" not in mock_sess
        assert "gcs_output_object_key" not in mock_sess

    # @pytest.mark.asyncio
    # async def test_session_history_appended_per_prompt(
    #     self, mock_store, user_id, session_id, dataset_id,
    #     sample_code, sample_output_json
    # ):
    #     """Orchestrator: history lists grow with each prompt execution."""
    #     from app.services.persistence_service import _persist_assets_background

    #     mock_sess: Dict[str, Any] = {}

    #     async def mock_get_session(sid):
    #         return dict(mock_sess)

    #     async def mock_save_session(sid, data):
    #         mock_sess.update(data)
    #         return True

    #     with patch("app.services.persistence_service._get_store_for_connection",
    #                return_value=(mock_store, "")), \
    #          patch("app.services.persistence_service.GITHUB_TOKEN", ""), \
    #          patch("app.services.session_service.get_session", side_effect=mock_get_session), \
    #          patch("app.services.session_service.save_session", side_effect=mock_save_session):

    #         # First prompt run
    #         await _persist_assets_background(
    #             user_id=user_id, session_id=session_id, dataset_id=dataset_id,
    #             generated_code=sample_code, planner_definition={"plan": "v1"},
    #             coder_definition={}, execution_result={"status": "success"},
    #             exec_succeeded=True, output_file_data=None,
    #             output_json=sample_output_json, output_location=None,
    #             visualization_config={"charts": []},
    #             connection_id=None, storage_uri=None,
    #         )

    #         # Second prompt run
    #         await _persist_assets_background(
    #             user_id=user_id, session_id=session_id, dataset_id=dataset_id,
    #             generated_code=sample_code + "\n# v2", planner_definition={"plan": "v2"},
    #             coder_definition={}, execution_result={"status": "success"},
    #             exec_succeeded=True, output_file_data=None,
    #             output_json=sample_output_json, output_location=None,
    #             visualization_config={"charts": []},
    #             connection_id=None, storage_uri=None,
    #         )

    #     code_history = mock_sess.get("code_assets", [])
    #     output_history = mock_sess.get("output_assets", [])

    #     assert len(code_history) == 2, "Should have 2 code entries after 2 prompts"
    #     assert len(output_history) == 2, "Should have 2 output entries after 2 prompts"
    #     #assert code_history[0]["prompt_ts"] != code_history[1]["prompt_ts"]
    #     assert code_history[0]["prompt_ts"] == "20260424_120000"
    #     assert code_history[1]["prompt_ts"] == "20260424_120500"
    #     assert code_history[0]["prompt_ts"] != code_history[1]["prompt_ts"]

    @pytest.mark.asyncio
    async def test_session_history_appended_per_prompt(
        self, mock_store, user_id, session_id, dataset_id,
        sample_code, sample_output_json
    ):
        """Orchestrator: history lists grow with each prompt execution."""
        from app.services.persistence_service import _persist_assets_background

        mock_sess: Dict[str, Any] = {}

        async def mock_get_session(sid):
            return dict(mock_sess)

        async def mock_save_session(sid, data):
            mock_sess.update(data)
            return True

        # Return different timestamps for each call so keys don't collide
        ts_values = iter(["20260424_120000", "20260424_120500"])

        with patch("app.services.persistence_service._get_store_for_connection",
                   return_value=(mock_store, "")), \
             patch("app.services.persistence_service.GITHUB_TOKEN", ""), \
             patch("app.services.persistence_service.datetime") as mock_dt, \
             patch("app.services.session_service.get_session", side_effect=mock_get_session), \
             patch("app.services.session_service.save_session", side_effect=mock_save_session):

            mock_dt.utcnow.return_value.strftime.side_effect = lambda fmt: next(ts_values)

            # First prompt run
            await _persist_assets_background(
                user_id=user_id, session_id=session_id, dataset_id=dataset_id,
                generated_code=sample_code, planner_definition={"plan": "v1"},
                coder_definition={}, execution_result={"status": "success"},
                exec_succeeded=True, output_file_data=None,
                output_json=sample_output_json, output_location=None,
                visualization_config={"charts": []},
                connection_id=None, storage_uri=None,
            )

            # Second prompt run
            await _persist_assets_background(
                user_id=user_id, session_id=session_id, dataset_id=dataset_id,
                generated_code=sample_code + "\n# v2", planner_definition={"plan": "v2"},
                coder_definition={}, execution_result={"status": "success"},
                exec_succeeded=True, output_file_data=None,
                output_json=sample_output_json, output_location=None,
                visualization_config={"charts": []},
                connection_id=None, storage_uri=None,
            )

        code_history = mock_sess.get("code_assets", [])
        output_history = mock_sess.get("output_assets", [])

        assert len(code_history) == 2, "Should have 2 code entries after 2 prompts"
        assert len(output_history) == 2, "Should have 2 output entries after 2 prompts"
        assert code_history[0]["prompt_ts"] != code_history[1]["prompt_ts"]
        assert code_history[0]["object_key"] != code_history[1]["object_key"]

    @pytest.mark.asyncio
    async def test_errors_written_to_session_on_store_failure(
        self, user_id, session_id, dataset_id, sample_code, sample_output_json
    ):
        """Orchestrator: persist_errors key set in session on upload failure."""
        from app.services.persistence_service import _persist_assets_background

        broken_store = MockGCSBlobStore()

        def broken_put_file(local_path, key):
            raise ConnectionError("GCS upload failed: network timeout")

        broken_store.put_file = broken_put_file

        mock_sess: Dict[str, Any] = {}

        async def mock_get_session(sid):
            return dict(mock_sess)

        async def mock_save_session(sid, data):
            mock_sess.update(data)
            return True

        with patch("app.services.persistence_service._get_store_for_connection",
                   return_value=(broken_store, "")), \
             patch("app.services.persistence_service.GITHUB_TOKEN", ""), \
             patch("app.services.session_service.get_session", side_effect=mock_get_session), \
             patch("app.services.session_service.save_session", side_effect=mock_save_session):

            await _persist_assets_background(
                user_id=user_id, session_id=session_id, dataset_id=dataset_id,
                generated_code=sample_code, planner_definition=None,
                coder_definition={}, execution_result={"status": "success"},
                exec_succeeded=True, output_file_data=None,
                output_json=sample_output_json, output_location=None,
                visualization_config={"charts": []},
                connection_id=None, storage_uri=None,
            )

        assert "persist_errors" in mock_sess
        assert len(mock_sess["persist_errors"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Retry Logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetryLogic:

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        """Retry: succeeds after one transient failure."""
        from app.services.persistence_service import _retry_async

        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("transient error")
            return "success"

        with patch("app.services.persistence_service.asyncio.sleep", new_callable=AsyncMock):
            result = await _retry_async(flaky_fn, "test-label")

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_raises_after_max_attempts(self):
        """Retry: raises after exhausting all 3 attempts."""
        from app.services.persistence_service import _retry_async

        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError(f"fail attempt {call_count}")

        with patch("app.services.persistence_service.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ConnectionError, match="fail attempt 3"):
                await _retry_async(always_fails, "test-label")

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_uses_correct_delays(self):
        """Retry: delays are 0s, 2s, 4s between attempts."""
        from app.services.persistence_service import _retry_async

        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        async def always_fails():
            raise ConnectionError("fail")

        with patch("app.services.persistence_service.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(ConnectionError):
                await _retry_async(always_fails, "test-label")

        # First attempt: no sleep (delay=0 is skipped)
        # Second attempt: sleep(2)
        # Third attempt: sleep(4)
        assert sleep_calls == [2, 4]


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Object Key Generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestObjectKeyGeneration:

    def test_code_key_format(self):
        from app.services.persistence_service import _object_key
        key = _object_key("code-registry", "user1", "sess1", "ds1", "20260424_120000", "transform.py")
        assert key == "code-registry/user1/sess1/ds1_20260424_120000_transform.py"

    def test_output_key_format(self):
        from app.services.persistence_service import _object_key
        key = _object_key("execution-outputs", "user1", "sess1", "ds1", "20260424_120000", "output.csv")
        assert key == "execution-outputs/user1/sess1/ds1_20260424_120000_output.csv"

    def test_viz_key_format(self):
        from app.services.persistence_service import _object_key
        key = _object_key("visualization-configs", "user1", "sess1", "ds1", "20260424_120000", "viz_config.json")
        assert key == "visualization-configs/user1/sess1/ds1_20260424_120000_viz_config.json"

    def test_key_uses_dataset_id_not_session(self):
        from app.services.persistence_service import _object_key
        key = _object_key("code-registry", "u", "sess-abc", "ds-xyz", "20260424_120000", "transform.py")
        filename = key.split("/")[-1]
        assert filename.startswith("ds-xyz_")


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests — Real GCS (skipped if AVALOKA_TEST_SKIP_GCS=1)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(SKIP_GCS, reason="AVALOKA_TEST_SKIP_GCS=1")
class TestRealGCSIntegration:
    """
    These tests hit the real GCS bucket.
    Run only in CI or when explicitly testing end-to-end.
    """

    @pytest.mark.asyncio
    async def test_real_code_upload_and_verify(
        self, user_id, session_id, dataset_id, prompt_ts, sample_code
    ):
        """E2E Task A: upload code to real GCS and verify it lands."""
        from app.services.persistence_service import persist_generated_code_to_store
        from app.core.storage import GCSBlobStore

        store = GCSBlobStore(GCS_BUCKET)

        with patch("app.services.persistence_service._get_store_for_connection",
                   return_value=(store, "")):
            result = await persist_generated_code_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                generated_code=sample_code,
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "success"
        key = result["object_key"]

        # Verify it exists in GCS
        size, updated = store.stat(key)
        assert size > 0

        # Cleanup
        store.delete(key)

    @pytest.mark.asyncio
    async def test_real_output_upload_and_verify(
        self, user_id, session_id, dataset_id, prompt_ts, sample_output_json
    ):
        """E2E Task C: upload output CSV to real GCS and verify it lands."""
        from app.services.persistence_service import persist_execution_output_to_store
        from app.core.storage import GCSBlobStore

        store = GCSBlobStore(GCS_BUCKET)

        with patch("app.services.persistence_service._get_store_for_connection",
                   return_value=(store, "")):
            result = await persist_execution_output_to_store(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_id,
                output_file_data=None,
                output_json=sample_output_json,
                output_location=None,
                prompt_ts=prompt_ts,
            )

        assert result["status"] == "success"
        key = result["object_key"]

        size, _ = store.stat(key)
        assert size > 0

        # Cleanup
        store.delete(key)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests — Real GitHub (skipped if AVALOKA_TEST_SKIP_GIT=1)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(SKIP_GIT or not GITHUB_TOKEN, reason="No GitHub token or AVALOKA_TEST_SKIP_GIT=1")
class TestRealGitHubIntegration:
    """
    These tests hit the real GitHub repo.
    Requires GITHUB_SYSTEM_TOKEN with repo write scope.
    """

    @pytest.mark.asyncio
    async def test_real_job_definition_write_to_main(
        self, user_id, session_id, dataset_id, prompt_ts,
        sample_planner_definition, sample_execution_result
    ):
        """E2E Task B: writes job_definition.json to main branch on real GitHub repo."""
        import httpx
        from app.services.persistence_service import persist_job_definition_to_git

        result = await persist_job_definition_to_git(
            user_id=user_id,
            session_id=session_id,
            dataset_id=dataset_id,
            planner_definition=sample_planner_definition,
            coder_definition={"code": "import pandas as pd"},
            execution_result=sample_execution_result,
            code_object_key=f"code-registry/{user_id}/{session_id}/test.py",
            prompt_ts=prompt_ts,
        )

        assert result["status"] == "success"
        assert result["branch"] == "main"
        assert result["repo"] == GITHUB_REPO
        assert "_job_definition.json" in result["file_path"]

        # Verify the file actually exists on GitHub
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"https://api.github.com/repos/{GITHUB_REPO}/contents/{result['file_path']}",
                headers=headers,
                params={"ref": "main"},
            )
        assert r.status_code == 200

        content = json.loads(__import__("base64").b64decode(r.json()["content"]))
        assert content["dataset_id"] == dataset_id
        assert content["prompt_ts"] == prompt_ts
        assert "planner_definition" in content