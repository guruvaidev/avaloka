"""
Contract suite C1 -- plan suites E11.05 (repo hygiene) and E11.09 (egress inventory anchor).

Everything here is a pure filesystem/git assertion: no app import, no network, no
fixtures from tests/e2e/conftest.py. Every case runs in every mode; the only skip
path is "git is not on PATH / this is not a work tree", which is reported as a skip
rather than a failure.

The plan's premise for E11.05 was "verify nothing sensitive is committed". That
premise is already false on this branch, so five of these cases are strict xfails
that state the intended end-state and name the remediation. They flip to failures
the day the leak is fixed, which is the signal to delete the xfail.

E11.09 asks for a documented data-egress inventory. There is no document to assert
against, so the inventory is frozen here as a source-level fact: the set of app
modules that talk to a third party is pinned, and a new egress path fails the test
until both the frozen set and the DPA inventory are updated.

Secret hygiene of the suite itself: no assertion message, parametrize id, or repr
in this file may contain a secret value. Findings are reported as
"<relative path>: <pattern name>" only.
"""

from __future__ import annotations

import base64
import fnmatch
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# git plumbing
# ---------------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git is unusable in this environment: {type(exc).__name__}")


def _require_work_tree() -> None:
    proc = _git("rev-parse", "--is-inside-work-tree")
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        pytest.skip(f"{REPO_ROOT} is not a git work tree; repo-hygiene gates need git")


def _tracked_paths() -> List[str]:
    _require_work_tree()
    proc = _git("ls-files", "-z")
    if proc.returncode != 0:
        pytest.skip("git ls-files failed; cannot enumerate tracked files")
    return [p for p in proc.stdout.split("\0") if p]


def _is_tracked(relpath: str) -> bool:
    _require_work_tree()
    proc = _git("ls-files", "--error-unmatch", "--", relpath)
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _is_gitignored(relpath: str) -> bool:
    _require_work_tree()
    return _git("check-ignore", "-q", "--", relpath).returncode == 0


# ---------------------------------------------------------------------------
# .dockerignore evaluation -- git check-ignore only understands .gitignore, so the
# build-context rules have to be interpreted here.
# ---------------------------------------------------------------------------


def _dockerignore_patterns() -> List[str]:
    path = REPO_ROOT / ".dockerignore"
    if not path.is_file():
        return []
    lines = path.read_text("utf-8", errors="ignore").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _dockerignore_pattern_matches(pattern: str, relpath: str) -> bool:
    pattern = pattern.rstrip("/")
    if not pattern:
        return False
    candidate = relpath.strip("/")
    segments = candidate.split("/")
    prefixes = ["/".join(segments[: i + 1]) for i in range(len(segments))]
    if pattern.startswith("**/"):
        tail = pattern[3:]
        return any(fnmatch.fnmatch(seg, tail) for seg in segments) or any(
            fnmatch.fnmatch(p, tail) for p in prefixes
        )
    return any(fnmatch.fnmatch(p, pattern) for p in prefixes)


def _is_dockerignored(relpath: str) -> bool:
    ignored = False
    for pattern in _dockerignore_patterns():
        negated = pattern.startswith("!")
        body = pattern[1:] if negated else pattern
        if _dockerignore_pattern_matches(body, relpath):
            ignored = not negated
    return ignored


# ---------------------------------------------------------------------------
# Secret scanner
# ---------------------------------------------------------------------------

SECRET_PATTERNS: Dict[str, re.Pattern] = {
    "aws_access_key_id": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key_pem": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "groq_api_key": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
}

DB_URI_WITH_CREDENTIALS = re.compile(
    r"(?:postgres(?:ql)?|mysql|mariadb|mongodb)(?:\+[a-z0-9_]+)?://"
    r"(?P<user>[^\s:/@\"']+):(?P<password>[^\s@/\"']+)@"
)

JWT_SHAPED = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")

PLACEHOLDER_PASSWORDS = {
    "p", "pw", "pass", "passwd", "password", "secret", "mypassword", "yourpassword",
    "your-password", "your_password", "changeme", "change-me", "xxx", "xxxx", "***",
    "redacted", "dummy", "example", "placeholder", "todo", "none", "null",
}

PLACEHOLDER_MARKERS = ("{", "}", "$", "<", ">", "%s", "%(")

BINARY_OR_BULK_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf", ".zip", ".gz", ".tgz",
    ".whl", ".so", ".dll", ".dylib", ".pyc", ".class", ".jar", ".sqlite3", ".db",
    ".parquet", ".onnx", ".pth", ".bin", ".woff", ".woff2", ".ttf", ".eot", ".mp4",
    ".mov", ".avi", ".xlsx", ".lock",
}

SCAN_SKIPPED_DIRECTORY_PREFIXES = (
    "venv/", "venvt/", ".venv/", "node_modules/", "ui/node_modules/",
    "tests/fixtures/", "tests/data/", "tests/e2e_kaggle_tests/",
)

MAX_SCANNED_FILE_BYTES = 1_000_000

FIXTURES_WHOSE_MATCHES_ARE_SYNTHETIC_AND_CARRY_NO_LIVE_CREDENTIAL: Dict[str, str] = {
    "tests/test_cloud_connection_redaction.py": (
        "asserts redact_connection() masks a service-account private key; the PEM "
        "header is a literal fragment inside the assertion, there is no key body"
    ),
    "tests/test_cloud_io_config_sa_json.py": (
        "builds a fake service-account JSON whose private_key is the constant "
        "_PK = BEGIN/END markers around the letters ABCDEF/GHIJKL"
    ),
    "tests/test_ray_unit.py": (
        "uses AKIAIOSFODNN7EXAMPLE, the access key id AWS publishes in its own "
        "documentation as the canonical non-functional example value"
    ),
    "tests/test_dta_bugfixes.py": (
        "asserts the DTA redacts source and destination DSN passwords; the "
        "S0urceSecret/D3stSecret literals exist only so the redaction can be seen"
    ),
}


def _password_is_placeholder(password: str) -> bool:
    if password.lower() in PLACEHOLDER_PASSWORDS:
        return True
    return any(marker in password for marker in PLACEHOLDER_MARKERS)


def _jwt_payload_text(token: str) -> str:
    try:
        segment = token.split(".")[1]
        segment += "=" * (-len(segment) % 4)
        return base64.urlsafe_b64decode(segment).decode("utf-8", "replace")
    except (ValueError, IndexError, UnicodeDecodeError):
        return ""


def _readable_tracked_files() -> Iterable[Tuple[str, str]]:
    for relpath in _tracked_paths():
        if relpath.startswith(SCAN_SKIPPED_DIRECTORY_PREFIXES):
            continue
        if relpath in FIXTURES_WHOSE_MATCHES_ARE_SYNTHETIC_AND_CARRY_NO_LIVE_CREDENTIAL:
            continue
        path = REPO_ROOT / relpath
        if not path.is_file() or path.suffix.lower() in BINARY_OR_BULK_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_SCANNED_FILE_BYTES:
                continue
            text = path.read_text("utf-8", errors="ignore")
        except OSError:
            continue
        if "\0" in text[:4096]:
            continue
        yield relpath, text


def _scan_for_secrets() -> List[str]:
    findings: List[str] = []
    for relpath, text in _readable_tracked_files():
        matched: Set[str] = set()
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                matched.add(name)
        for match in DB_URI_WITH_CREDENTIALS.finditer(text):
            if not _password_is_placeholder(match.group("password")):
                matched.add("db_uri_with_password")
        for token in JWT_SHAPED.findall(text):
            if "service_role" in _jwt_payload_text(token):
                matched.add("supabase_service_role_jwt")
        for name in sorted(matched):
            findings.append(f"{relpath}: {name}")
    return sorted(findings)


# ---------------------------------------------------------------------------
# E11.05 -- repo hygiene
# ---------------------------------------------------------------------------


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E11.05 (P0): .env is git-tracked and carries a live Supabase "
        "service_role key plus SUPABASE_JWT_SECRET (.gitignore:56 lists .env but "
        "ignore rules never apply to already-tracked paths). Remediation: rotate the "
        "Supabase service_role key and JWT secret, git rm --cached .env, purge it "
        "from history (git filter-repo / BFG), force-push and re-clone. "
        "Remove this xfail when fixed."
    ),
)
def test_e11_05_env_file_is_not_tracked() -> None:
    """.env must never be under version control; git ls-files .env returns nothing."""
    assert not _is_tracked(".env"), (
        ".env is tracked by git; anyone with repo read access holds the Supabase "
        "service-role key and the JWT signing secret"
    )


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E11.05 (P0): tracked files contain live credentials -- .env holds a "
        "Supabase service_role JWT and a gsk_ Groq key, app/mcp_server/customers.json "
        "holds per-customer api_keys and a Postgres DSN with a password and is "
        "auto-loaded into the MCP registry at startup. Remediation: rotate every "
        "matched credential, untrack the files, purge from history. "
        "Remove this xfail when fixed."
    ),
)
def test_e11_05_secret_material_absent_from_tracked_files() -> None:
    """No tracked file matches a live-credential pattern (paths and pattern names only)."""
    findings = _scan_for_secrets()
    assert not findings, (
        "tracked files matched credential patterns (values withheld by design):\n  "
        + "\n  ".join(findings)
    )


CREDENTIAL_FILE_CANDIDATES = [
    "avalokagcpbucketserviceaccount.json",
    "gcp-serviceaccount.json",
]


@pytest.mark.defect
@pytest.mark.parametrize("candidate", CREDENTIAL_FILE_CANDIDATES)
@pytest.mark.parametrize("mechanism", ["gitignore", "dockerignore"])
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E11.05 (P0): SA key neither committed nor ignored -- "
        "avalokagcpbucketserviceaccount.json sits untracked in the repo root with no "
        "matching rule in .gitignore and none in .dockerignore (which only excludes "
        ".env/*.env/env_bak), so one `git add .` commits a GCP private key and one "
        "`docker build .` bakes it into the image layer. Remediation: add "
        "*serviceaccount*.json (and *-sa.json / *credentials*.json) to both files. "
        "Remove this xfail when fixed."
    ),
)
def test_e11_05_credential_files_are_ignored(mechanism: str, candidate: str) -> None:
    """Service-account key filenames are excluded from both git and the Docker build context."""
    ignored = _is_gitignored(candidate) if mechanism == "gitignore" else _is_dockerignored(candidate)
    assert ignored, (
        f"{candidate} is not excluded by .{mechanism}; a service-account private key "
        f"can be committed or baked into an image layer"
    )


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E11.05 (P0): app/mcp_server/customers.json is git-tracked and holds "
        "16 customer records with ak_* api_keys and a Postgres DSN carrying a "
        "password; it is loaded into the MCP registry at server startup. Remediation: "
        "rotate every embedded ak_* api_key and the database password, move the "
        "registry to a mounted secret/DB, git rm --cached the file (.gitignore "
        "already lists customers.json) and purge it from history. "
        "Remove this xfail when fixed."
    ),
)
def test_e11_05_mcp_customers_registry_not_tracked() -> None:
    """The MCP customer registry, which embeds api_keys and a DB password, is not committed."""
    assert not _is_tracked("app/mcp_server/customers.json"), (
        "app/mcp_server/customers.json is tracked; it embeds per-customer API keys "
        "and a database password"
    )


KEY_SHAPED_LITERALS = {
    "mcp_api_key": re.compile(r"\bak_[0-9a-fA-F]{24,}\b"),
    "groq_api_key": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key_id": re.compile(r"AKIA[0-9A-Z]{16}"),
}


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT E11.05 (P2): the tracked repo-root debug script test.py hardcodes an "
        "MCP api_key literal next to the customer id it belongs to, mirroring an entry "
        "in app/mcp_server/customers.json. Remediation: delete test.py (or move it "
        "under scripts/ reading the key from the environment) and rotate that key. "
        "Remove this xfail when fixed."
    ),
)
def test_e11_05_no_debug_scripts_with_keys_in_repo_root() -> None:
    """The repo root holds no debug script carrying a key-shaped literal."""
    script = REPO_ROOT / "test.py"
    if not script.is_file():
        return
    text = script.read_text("utf-8", errors="ignore")
    matched = sorted(name for name, pattern in KEY_SHAPED_LITERALS.items() if pattern.search(text))
    assert not matched, (
        f"test.py contains key-shaped literals (values withheld): {', '.join(matched)}"
    )


STRAY_ARTIFACT_SAMPLES = [
    "server.log",
    "run.out",
    "run.err",
    "scratch.tmp",
    "notes.bak",
    pytest.param(
        "console_output.txt",
        marks=[
            pytest.mark.defect,
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "DEFECT E11.05 (P2): no .gitignore rule covers a console capture "
                    "-- the file covers *.log/*.out/*.err/*.tmp/*.bak but nothing "
                    "matches console_output.txt, which has already appeared untracked "
                    "in the repo root; a console dump routinely contains tokens and "
                    "DSNs and is one `git add .` from being committed. Remediation: "
                    "add console_output.txt / *_output.txt to .gitignore. "
                    "Remove this xfail when fixed."
                ),
            ),
        ],
    ),
]


@pytest.mark.parametrize("artifact", STRAY_ARTIFACT_SAMPLES)
def test_e11_05_stray_artifacts_are_ignored(artifact: str) -> None:
    """Console and log capture artifacts are covered by .gitignore and are not tracked."""
    assert not _is_tracked(artifact), f"{artifact} is tracked; build/console output must not be"
    assert _is_gitignored(artifact), (
        f"no .gitignore rule matches {artifact}; a stray capture can be staged by `git add .`"
    )


# ---------------------------------------------------------------------------
# E11.09 -- data-egress inventory anchor
# ---------------------------------------------------------------------------

EGRESS_INVENTORY_UPDATE_INSTRUCTION = (
    "A data-egress path was added, moved or removed. Update BOTH the frozen set in "
    "tests/contract/test_c1_repo_hygiene.py AND the customer-facing egress inventory / "
    "DPA sub-processor list before merging. Every entry below is user data or a user "
    "prompt leaving the cluster to a third party."
)

EGRESS_CHANNELS: Dict[str, Tuple[re.Pattern, str, Set[str]]] = {
    "groq_llm_prompts": (
        re.compile(r"^[ \t]*(?:from[ \t]+langchain_groq|import[ \t]+langchain_groq)", re.M),
        "agent prompts, which embed sampled rows, schemas and DDL, go to Groq's hosted API",
        {
            "app/agents/coder.py",
            "app/agents/data_transfer_agent/daft_coder.py",
            "app/agents/data_transfer_agent/daft_validator.py",
            "app/agents/mta/task_builder.py",
            "app/agents/mta_v2/agent.py",
            "app/agents/planner.py",
            "app/agents/profiling_agent.py",
            "app/agents/summarizer.py",
            "app/agents/validator.py",
            "app/agents/visualization_agent.py",
            "app/core/agent_llm.py",
            "app/services/memory_plane.py",
        },
    ),
    "embedding_provider": (
        re.compile(r"langchain_openai|OpenAIEmbeddings|OPENAI_API_KEY"),
        "memory/RAG text is embedded by OpenAI when OPENAI_API_KEY is set, otherwise by "
        "local sentence-transformers, otherwise a zero-vector -- only the first leaves the cluster",
        {
            "app/services/embedding_utils.py",
            "app/services/memory_runtime.py",
            "app/services/milvus_recorder.py",
        },
    ),
    "supabase_dataset_rows": (
        re.compile(r"dataset_profiles|dataset_samples"),
        "sampled ROWS and column profiles are written to cloud Supabase tables with the "
        "service-role key; the most sensitive channel and the one the plan omitted",
        {
            "app/agents/sampling_persistence.py",
        },
    ),
    "cloud_object_storage": (
        re.compile(
            r"from[ \t]+google\.cloud[ \t]+import[ \t]+storage|^[ \t]*import[ \t]+boto3|azure\.storage\.blob",
            re.M,
        ),
        "uploaded datasets, artifacts and models move through GCS/S3/Azure SDK clients "
        "(by design), and the AWS SDK is also used for EKS control-plane calls",
        {
            "app/agents/data_transfer_agent/data_transfer_agent.py",
            "app/agents/mta/inference_docker/inference_job.py",
            "app/agents/mta/local_inference.py",
            "app/agents/mta/mlflow_integration.py",
            "app/agents/mta/temporary_inference.py",
            "app/agents/mta_v2/inference.py",
            "app/agents/mta_v2/inference_docker_image/ray_job.py",
            "app/agents/mta_v2/inference_service_image/inference.py",
            "app/agents/sampling_agent_daft.py",
            "app/core/storage.py",
            "app/infra/k8s_invoker.py",
            "app/infra/providers/aws_eks.py",
        },
    ),
}


def _tracked_app_modules() -> List[str]:
    return [p for p in _tracked_paths() if p.startswith("app/") and p.endswith(".py")]


def _modules_matching(pattern: re.Pattern, modules: Sequence[str]) -> Set[str]:
    found: Set[str] = set()
    for relpath in modules:
        path = REPO_ROOT / relpath
        if not path.is_file():
            continue
        try:
            text = path.read_text("utf-8", errors="ignore")
        except OSError:
            continue
        if pattern.search(text):
            found.add(relpath)
    return found


@pytest.mark.parametrize("channel", sorted(EGRESS_CHANNELS))
def test_e11_09_egress_inventory_is_complete(channel: str) -> None:
    """The app modules that send user data to each third party are exactly the inventoried set."""
    pattern, description, expected = EGRESS_CHANNELS[channel]
    actual = _modules_matching(pattern, _tracked_app_modules())
    added = sorted(actual - expected)
    removed = sorted(expected - actual)
    assert actual == expected, (
        f"egress channel '{channel}' drifted -- {description}\n"
        f"  new egress modules: {added or 'none'}\n"
        f"  inventoried modules that no longer match: {removed or 'none'}\n"
        f"{EGRESS_INVENTORY_UPDATE_INSTRUCTION}"
    )


def test_e11_09_pins_supabase_row_egress_is_undocumented_in_the_plan() -> None:
    """Pins the decision that sampled rows leaving to cloud Supabase is a real egress channel the plan omitted."""
    module = REPO_ROOT / "app" / "agents" / "sampling_persistence.py"
    if not module.is_file():
        pytest.skip("app/agents/sampling_persistence.py is absent; nothing to pin")
    text = module.read_text("utf-8", errors="ignore")
    assert "dataset_samples" in text and "dataset_profiles" in text, (
        "sampling_persistence no longer writes dataset_samples/dataset_profiles; "
        + EGRESS_INVENTORY_UPDATE_INSTRUCTION
    )
    assert re.search(r"SUPABASE_(?:SERVICE_ROLE_KEY|URL)", text), (
        "sampling_persistence no longer reaches Supabase directly; "
        + EGRESS_INVENTORY_UPDATE_INSTRUCTION
    )
