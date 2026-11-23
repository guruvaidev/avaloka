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
until both the frozen set and the DPA inventory are updated. The scan covers both
Python packages that ship -- ``app/`` (the API and agents) and ``avaloka/`` (the
CLI) -- because the CLI reads customer databases, document stores and object
storage directly (``avaloka/io/sources.py``) and used to sit outside the inventory.

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


#: The scanner's own source. It necessarily contains the patterns it looks for
#: and, in the comment explaining the AWS placeholder, a literal
#: AKIAIOSFODNN7EXAMPLE -- so it matches itself. Excluding one file by exact
#: path is narrow enough to stay honest; excluding a glob would not be.
_SCANNER_SOURCE = "tests/contract/test_c1_repo_hygiene.py"


def _scan_for_secrets() -> List[str]:
    findings: List[str] = []
    for relpath, text in _readable_tracked_files():
        if relpath == _SCANNER_SOURCE:
            continue
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


def test_e11_05_env_file_is_not_tracked() -> None:
    """.env must never be under version control; git ls-files .env returns nothing.

    Was DEFECT E11.05 (P0), xfail(strict): .env was tracked and carried a live
    Supabase service_role key and JWT signing secret, which .gitignore could not
    help with -- ignore rules never apply to already-tracked paths. The 1.0
    repository was rebuilt without it. The assertion stays, because re-adding
    the file is exactly the mistake worth catching.
    """
    assert not _is_tracked(".env"), (
        ".env is tracked by git; anyone with repo read access holds the Supabase "
        "service-role key and the JWT signing secret"
    )


def test_e11_05_secret_material_absent_from_tracked_files() -> None:
    """No tracked file matches a live-credential pattern (paths and pattern names only).

    Was DEFECT E11.05 (P0), xfail(strict). The tracked material is gone: .env
    when the 1.0 tree was rebuilt, app/mcp_server/customers.json and its
    sixteen ak_* keys here, and the literal local Postgres password in
    test_mcp_server_integration.py, now read from the environment.

    What remains and still matches are documented placeholders -- AWS's own
    AKIAIOSFODNN7EXAMPLE, "-----BEGIN PRIVATE KEY-----\nABCDEF", and
    user:pass@host in docstrings. The credentials are still in git history on
    both branches; this asserts only that the working tree is clean.
    """
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


def test_e11_05_mcp_customers_registry_not_tracked() -> None:
    """The MCP customer registry, which embeds api_keys and a DB password, is not committed.

    Was DEFECT E11.05 (P0), xfail(strict). .gitignore already listed the file,
    which did nothing, because ignore rules never apply to an already-tracked
    path -- the same trap that kept .env in the tree. Untracked now, with
    customers.example.json committed in its place so the format stays
    documented. The embedded ak_* keys and database password still want
    rotating, and the file still wants purging from history.
    """
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


def test_e11_05_no_debug_scripts_with_keys_in_repo_root() -> None:
    """The repo root holds no debug script carrying a key-shaped literal.

    DEFECT E11.05 is fixed. test.py was a tracked repo-root debug script
    hardcoding an MCP api_key next to the customer id it belonged to, and it
    was deleted while preparing the public 1.0 -- which is the remediation the
    xfail here prescribed. The marker is gone with it.

    Widened while removing the marker: this now checks every script at the repo
    root, not just the one that happened to be the offender. The original was
    a pin on one known file; the property worth holding is that no root-level
    script carries a key-shaped literal.

    The key itself still needs rotating. Deleting the file removes it from the
    working tree, not from the history of a repository about to be published.
    """
    offenders: list[str] = []
    for script in sorted(REPO_ROOT.glob("*.py")):
        text = script.read_text("utf-8", errors="ignore")
        matched = sorted(name for name, pattern in KEY_SHAPED_LITERALS.items()
                         if pattern.search(text))
        if matched:
            offenders.append(f"{script.name}: {', '.join(matched)}")
    assert not offenders, (
        "repo-root scripts contain key-shaped literals (values withheld): "
        + "; ".join(offenders)
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

#: Modules that import daft, each of which must disable daft's usage beacon
#: before it does (see the ``daft_usage_beacon`` channel).
DAFT_IMPORTERS: Tuple[str, ...] = (
    "app/agents/avaloka_agent/discovery_tools.py",
    "app/agents/data_transfer_agent/daft_execution.py",
    "app/agents/data_transfer_agent/data_transfer_agent.py",
    "app/agents/data_transfer_agent/datasink/sql_sink.py",
    "app/agents/sampling_agent_daft.py",
    "app/agents/sampling_async.py",
    "app/data_transfer_docker/container_file.py",
)

#: Modules that import onnxruntime, or chromadb (whose default embedding
#: function runs onnxruntime), each of which would start onnxruntime's 1DS
#: telemetry uploader unless ORT_DISABLE_TELEMETRY is set first (see the
#: ``onnxruntime_1ds_telemetry`` channel).
ORT_IMPORTERS: Tuple[str, ...] = (
    "app/agents/mta/inference_docker/inference_job.py",
    "app/agents/mta/local_inference.py",
    "app/agents/mta/onnx_exporter.py",
    "app/agents/mta_v2/inference_service_manager.py",
    "app/core/offline.py",
    "app/rag/daft_ingest.py",
    "app/rag/daft_retrieval.py",
    "app/serve/inference.py",
    "app/services/db/chroma_client.py",
)
#: Of those, the ones run as a standalone script inside their own image, where
#: app/__init__.py never runs: each must set the switch itself before the import.
ORT_STANDALONE_SCRIPTS: Tuple[str, ...] = (
    "app/agents/mta/inference_docker/inference_job.py",
)

#: Every channel is matched against these trees unless it names its own
#: roots (the fourth tuple element). Egress is about what leaves the cluster,
#: not which package it left from, so every tree that can hold an outbound
#: call is here.
#:
#: This was ``app/`` alone, then ``app/`` and ``avaloka/``, and always ``.py``
#: only. ``ui/`` failed both conditions, so no channel could fail on anything
#: the web app or the Supabase edge functions sent, however much they sent.
#: Proved rather than argued: a module under ``ui/`` posting ``dataset_samples``
#: to a third party was committed to a clone and every case here still passed.
#: The edge functions were in fact POSTing chart data to quickchart.io the
#: whole time (DEF-013).
EGRESS_SCAN_ROOTS: Tuple[str, ...] = ("app/", "avaloka/", "ui/src/", "ui/supabase/")
EGRESS_SCAN_SUFFIXES: Tuple[str, ...] = (".py", ".ts", ".tsx")

EGRESS_CHANNELS: Dict[str, tuple] = {
    "product_analytics_export": (
        # Anchored on the endpoint variable and its one accessor, NOT on an import
        # like the entries around it. The sender uses stdlib urllib, so there is
        # no third-party package to anchor on, and matching urllib itself would
        # claim unrelated modules. Any module that starts reading the export
        # endpoint shows up here as a new egress module.
        # Deliberately does not mention PR #329's host or its endpoint variable,
        # so this entry and that PR's `telemetry_usage_stats` entry cannot claim
        # each other's modules when both are on one branch.
        re.compile(r"AVALOKA_ANALYTICS_EXPORT_ENDPOINT|export_endpoint\("),
        "product analytics events are forwarded to an Avaloka-operated ingest (not a "
        "third party, but outside the customer's cluster) when an operator sets BOTH "
        "AVALOKA_ANALYTICS_EXPORT=on and an https endpoint. There is no default host and "
        "nothing schedules the export on develop-1.6, so by default this channel carries "
        "nothing. When enabled it carries metadata only (owner decision D16.5) -- event "
        "types, timings, counts, outcomes, error class names, route templates, chart "
        "types, versions, held to an explicit allowlist at the one egress point -- plus a "
        "per-analyst pseudonym re-keyed every 28 days with a secret that stays in the "
        "deployment. It carries NO question text in any form: the analyst's question, "
        "which contains column and table names, is stored only in the deployment's own "
        "database and is never read by the export; the one prompt-derived value sent is "
        "which of six length bands the question fell in. "
        "LIMIT OF THIS ENTRY: an env-var anchor detects that code can be configured to "
        "send, an import anchor detects that a package is present, and neither proves a "
        "route is live or dead. This entry would not notice a module that posted to a "
        "hard-coded host without calling export_endpoint()",
        {
            "app/analytics/config.py",
            "app/analytics/export.py",
        },
    ),
    "analysis_email_delivery": (
        # Anchored on the SMTP credentials, not on a mail library: these are Deno
        # edge functions and the transport is hand-rolled in _shared/raw-smtp.ts,
        # so there is no package name to match on.
        re.compile(r"SMTP_HOST|SMTP_USERNAME|SMTP_PASSWORD"),
        "analysis and report results are emailed to recipient addresses the sharing user "
        "types in. The result rows themselves are encrypted -- buildEncryptedPdf sets a "
        "userPassword and the handler rejects one outside 6-128 characters -- but the "
        "SUBJECT LINE carries the analysis NAME verbatim (`Analysis results: ${title}`) "
        "and the attachment filename is that same name with punctuation substituted "
        "(DEF-010). A subject line is never end-to-end encrypted, is indexed by the "
        "recipient's mail provider, and is retained in transit logs at every hop. The "
        "sending user's own e-mail address appears in the body. "
        "LIMIT OF THIS ENTRY: it anchors on the credentials these functions read, so it "
        "would not notice a sender that hard-coded a relay host.",
        {
            "ui/supabase/functions/_shared/raw-smtp.ts",
            "ui/supabase/functions/send-analysis-email/index.ts",
            "ui/supabase/functions/send-report-email/index.ts",
            "ui/supabase/functions/send-smtp-email/index.ts",
        },
    ),
    # The TypeScript twin of http_client_to_literal_host below: a fetch() aimed
    # at a host written into the source is a destination no operator can
    # configure away. Until the commit that emptied this set, the two mail
    # functions POSTed every chart's data values and labels to quickchart.io,
    # and send-report-email pulled its watermark from a Lovable-hosted asset on
    # every report (DEF-013); neither host was in the egress inventory or on
    # the sub-processor list. Charts are now drawn locally with the pdf-lib the
    # functions already embed and the watermark is text, so this entry is the
    # tripwire: the next hard-coded host in ui/ shows up here.
    "ui_fetch_to_literal_host": (
        re.compile(r"\bfetch\(\s*[\"'`]https?://(?!localhost|127\.0\.0\.1)"),
        "a fetch() in the web app or an edge function whose destination is written "
        "into the source",
        set(),
        ("ui/src/", "ui/supabase/"),
    ),
    "groq_llm_prompts": (
        re.compile(r"^[ \t]*(?:from[ \t]+langchain_groq|import[ \t]+langchain_groq)", re.M),
        "agent prompts, which embed sampled rows, schemas and DDL, go to Groq's hosted API",
        # This set shrank, which is the good direction: nine agents used to
        # construct a Groq client each, and now build through app/core. The
        # prompts still carry sampled rows, schemas and DDL -- the egress did
        # not go away, it moved to one place where it can be seen.
        {
            "app/agents/data_transfer_agent/daft_coder.py",
            "app/agents/data_transfer_agent/daft_validator.py",
            "app/agents/mta_v2/agent.py",
            "app/core/inference.py",
        },
    ),
    # The OpenAI-compatible CHAT path, which is not the embedding path below.
    # langchain_openai carries both: ChatOpenAI for chat completions and
    # OpenAIEmbeddings for vectors. Matching one regex on the package name
    # conflated them, so a chart-generation LLM call landed in a channel
    # described as embedding memory/RAG text. Where the prompt actually goes
    # depends on configuration: AVALOKA_VISUALIZATION_BACKEND=openrouter sends
    # it to a third party, =local sends it to a server the operator runs, and
    # =groq does not use this client at all (see groq_llm_prompts above).
    "openai_compatible_llm_prompts": (
        re.compile(r"^[ \t]*(?:from[ \t]+langchain_openai|import[ \t]+langchain_openai)", re.M),
        "chart and insight prompts, which embed column names and sampled values from the "
        "result table (profile_columns, visualization_agent.py:255-283), go to OpenRouter "
        "when the backend is openrouter -- a third party -- or to a local server when it is "
        "local, which does not leave the operator's control",
        {
            "app/agents/visualization_agent.py",
            "app/core/inference.py",
            "app/core/model_fallback.py",
            "app/services/embedding_utils.py",
        },
    ),
    # Embeddings are local by default since 2.0. Before, the "auto" default sent
    # memory text (raw queries, hints, LLM answers, schema JSON with table and
    # column names) to OpenAI whenever the CHAT key OPENAI_API_KEY was set. The
    # pattern now names the dedicated opt-in key separately: \b keeps
    # OPENAI_API_KEY from matching inside EMBEDDING_OPENAI_API_KEY, so a module
    # that reads only the opt-in key, or only the chat key, is still seen.
    "embedding_provider": (
        re.compile(r"OpenAIEmbeddings|EMBEDDING_OPENAI_API_KEY|\bOPENAI_API_KEY\b"),
        "memory/RAG text (queries, hints, answers, schema JSON with table and column names) "
        "is embedded by the bundled local sentence-transformer by default, a zero-vector as "
        "the last resort; it goes to OpenAI only on the explicit opt-in "
        "EMBEDDING_PROVIDER=openai plus EMBEDDING_OPENAI_API_KEY -- OPENAI_API_KEY never selects it",
        {
            # The conversational agent reads OPENAI_API_KEY to decide whether a
            # hosted model is available before it routes a turn.
            "app/agents/avaloka_agent/agent.py",
            "app/core/inference.py",
            # The CLI's provider table names OPENAI_API_KEY to select OpenAI as a
            # chat provider for --llm; it embeds nothing. Its chat egress is
            # declared under cli_llm_prompts.
            "avaloka/llm/providers.py",
            # model_fallback.py is NOT listed here any more: its only match was
            # the langchain_openai package name, which is the chat client, not
            # OpenAIEmbeddings. It is declared under openai_compatible_llm_prompts.
            # The one module that can send: it reads EMBEDDING_OPENAI_API_KEY and
            # calls OpenAIEmbeddings().embed_query, only after the opt-in.
            "app/services/embedding_utils.py",
            # Reads no key itself now; its production check names the opt-in
            # key in the message it raises.
            "app/services/memory_runtime.py",
            # milvus_recorder.py is NOT listed: its only key mentions are in
            # docstrings documenting provider selection. It delegates to
            # embedding_utils.embed_text (above), which is where the opt-in key is
            # read and OpenAIEmbeddings().embed_query is called. The
            # third party and the module that reaches it are both still declared;
            # only the module that merely describes the behaviour is not.
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
    # ---- the CLI's source connectors (avaloka/io/sources.py and what it
    # ---- is built on). Scoped to avaloka/: the app's SQLAlchemy use is its
    # ---- own lineage/telemetry store and the MCP server's registered
    # ---- customer databases, which the MCP inventory above describes.
    "cli_relational_database_sources": (
        re.compile(r"^[ \t]*(?:import[ \t]+sqlalchemy|from[ \t]+sqlalchemy(?:\.[\w.]+)?[ \t]+import|import[ \t]+psycopg)", re.M),
        "the CLI reads whole tables, SELECT-only queries and catalogue/statistics from "
        "any SQLAlchemy URL the user names (Postgres via psycopg); the destination is the "
        "user's own database, and the connection string is held in process and redacted "
        "in provenance. avaloka/catalog/store.py is the one WRITER in this set: the dataset "
        "catalog records what Avaloka touched -- dataset kind, a locator (connection id, "
        "schema, table; check_locator refuses secret-bearing keys), typed column names, row "
        "counts, which analysis used which version -- to AVALOKA_CATALOG_DB_URL, which is "
        "the operator's own lineage database in the application and a SQLite file under "
        "~/.avaloka in the CLI; never a DSN, a key or a data row",
        {
            "avaloka/benchmark/runner.py",
            "avaloka/catalog/store.py",
            "avaloka/connectors/postgres.py",
            "avaloka/io/sources.py",
            "avaloka/source_statistics.py",
        },
        ("avaloka/",),
    ),
    "cli_document_database_sources": (
        re.compile(r"^[ \t]*(?:import[ \t]+pymongo|from[ \t]+pymongo(?:\.[\w.]+)?[ \t]+import)", re.M),
        "the CLI reads a bounded MongoDB collection selection the user names "
        "(AVALOKA_MONGO_MAX_DOCUMENTS); the destination is the user's own cluster",
        {
            "avaloka/io/sources.py",
        },
        ("avaloka/",),
    ),
    "cli_object_storage_sources": (
        re.compile(r"^[ \t]*(?:import[ \t]+fsspec|from[ \t]+fsspec(?:\.[\w.]+)?[ \t]+import)", re.M),
        "the CLI streams one object the user names from s3/gs/az/http through fsspec "
        "into a bounded local file; the destination is the user's own bucket or URL. "
        "The connector contract's object store (avaloka/connectors/objectstore.py) is "
        "read-only: it lists, stats and reads bounded byte ranges of the objects in the "
        "one bucket or container its URL names, at the host endpoints() reports -- the "
        "configured S3 endpoint_url, else AWS S3; storage.googleapis.com; or "
        "<account>.blob.core.windows.net. Each request carries the bucket and object key "
        "and is authenticated with the user's own storage options (access key, secret and "
        "session token; a service account, whose token exchange also reaches Google's "
        "OAuth endpoint; an account key, SAS token or connection string). With no options "
        "it reads anonymously; options that name no key leave the fsspec backend's default "
        "credential lookup in charge, which for S3 and GCS can query the cloud "
        "instance-metadata service. It writes nothing, and sends no row, column name or "
        "object name anywhere but that store",
        {
            "avaloka/benchmark/runner.py",
            "avaloka/connectors/objectstore.py",
            "avaloka/io/sources.py",
        },
        ("avaloka/",),
    ),
    # ---- the CLI's optional narrator (avaloka --llm). Off unless the user
    # ---- passes --llm or --llm-provider; then each line Avaloka says (which
    # ---- can quote column names and measured figures) goes to the provider.
    "cli_llm_prompts": (
        re.compile(r"/chat/completions|^[ \t]*import[ \t]+anthropic\b", re.M),
        "with --llm, the CLI sends each persona line (which can name columns and quote "
        "measured statistics) to the selected model provider: a local Ollama / "
        "OpenAI-compatible server the user runs, or Groq, OpenRouter, OpenAI or Anthropic. "
        "No provider is contacted without --llm or --llm-provider. The opt-in live run of "
        "the conversational benchmark sends its synthetic probe evidence to Groq's chat "
        "endpoint when no local model answers",
        {
            "avaloka/benchmark/conversational.py",
            "avaloka/llm/client.py",
            "avaloka/llm/openai_compat.py",
        },
        ("avaloka/",),
    ),
    # A third-party destination that is DISABLED BY DEFAULT. daft's
    # __init__ calls scarf_telemetry.track_import_on_scarf(), and every query
    # calls track_runner_on_scarf(): each sends daft's version, the OS, the
    # Python version and the CPU arch (no customer data) to
    # daft.gateway.scarf.sh and osstelemetry.io -- unless DO_NOT_TRACK or
    # SCARF_NO_ANALYTICS is set before the import. app/core/thirdparty_optout.py
    # sets both (as defaults), app/__init__.py calls it before any app module
    # loads, every module below calls it again before its first daft import
    # (test_daft_importers_disable_the_beacon_first), and the images and the
    # chart set the same variables. tests/test_thirdparty_optout.py imports
    # daft in a subprocess and fails if a connection to either host is tried.
    "daft_usage_beacon": (
        re.compile(r"^[ \t]*(?:import[ \t]+daft\b|from[ \t]+daft(?:\.[\w.]+)?[ \t]+import\b)", re.M),
        "DISABLED BY DEFAULT: importing daft would send daft's version, platform, Python "
        "version and CPU arch -- never data, column or file names -- to daft.gateway.scarf.sh "
        "and osstelemetry.io; every module that imports daft first sets DO_NOT_TRACK and "
        "SCARF_NO_ANALYTICS (app/core/thirdparty_optout.py), and a deployment must unset "
        "both to opt in",
        set(DAFT_IMPORTERS),
    ),
    # A third-party destination that is DISABLED BY DEFAULT. onnxruntime's
    # official builds (1.31.0 is pinned) start Microsoft's 1DS uploader when the
    # ORT environment is created -- chroma's default embedding function does that
    # on its first query -- and POST trace events (ORT version, session and
    # execution-provider metadata, a device id; never data, column or file
    # names) to mobile.events.data.microsoft.com/OneCollector/1.0 ~15 s later.
    # ORT_DISABLE_TELEMETRY=1 before initialisation stops the uploader;
    # onnxruntime.disable_telemetry_events() does not (measured). Defaulted by
    # app/core/thirdparty_optout.py (via app/__init__.py), every image, the
    # quickstart compose and the chart; tests/test_thirdparty_optout.py loads an
    # ONNX model behind a recording proxy and fails on a CONNECT to the collector.
    "onnxruntime_1ds_telemetry": (
        re.compile(
            r"^[ \t]*(?:import[ \t]+(?:onnxruntime|chromadb)\b"
            r"|from[ \t]+(?:onnxruntime|chromadb)(?:\.[\w.]+)?[ \t]+import\b)",
            re.M,
        ),
        "DISABLED BY DEFAULT: loading an ONNX model (directly, or through chroma's default "
        "embedding function) would send onnxruntime trace events -- never data, column or "
        "file names -- to Microsoft's mobile.events.data.microsoft.com; "
        "app/core/thirdparty_optout.py sets ORT_DISABLE_TELEMETRY=1 before any of these "
        "modules load, and a deployment must unset it to opt in",
        set(ORT_IMPORTERS),
    ),
    # Cloud instance-metadata services. A destination inside the provider, not a
    # third party -- but answering it hands the process the VM's credentials and
    # identity. Only describe_environment(allow_network=True) (an explicit MCP
    # tool argument) probes it; detection is env-only otherwise. botocore's own
    # credential-chain probe of 169.254.169.254 is off by default through
    # AWS_EC2_METADATA_DISABLED=true (app/core/thirdparty_optout.py, images,
    # compose, chart), and the DTA sysdocs load that triggered it is off unless
    # AVALOKA_SYSDOCS_PATH is set.
    "cloud_instance_metadata": (
        re.compile(r"169\.254\.169\.254|metadata\.google\.internal"),
        "OPT-IN ONLY: the MCP describe_environment tool asks the GCP/Azure/AWS instance-"
        "metadata service which cloud it runs on when called with allow_network=True; "
        "nothing else contacts it, and botocore's credential probe is disabled by "
        "AWS_EC2_METADATA_DISABLED=true unless an operator opts in to instance credentials",
        {"app/execution/environment.py"},
    ),
    # An HTTP client aimed at a literal, non-local host is a destination the
    # operator cannot configure away. Nothing in either package does this
    # today (the GitHub call in persistence_service goes through a client
    # variable, and the benchmark fetches go to a URL argument), so this
    # entry is the tripwire: the first module that posts to a hard-coded
    # host shows up here.
    "http_client_to_literal_host": (
        re.compile(
            r"(?:requests|httpx|urllib\.request)\.(?:get|post|put|patch|delete|urlopen|Request)\("
            r"\s*f?[\"']https?://(?!localhost|127\.0\.0\.1)",
        ),
        "an HTTP client call whose destination is written into the source",
        set(),
    ),
}


def _tracked_modules(roots: Sequence[str] = EGRESS_SCAN_ROOTS) -> List[str]:
    return [p for p in _tracked_paths()
            if p.startswith(tuple(roots)) and p.endswith(EGRESS_SCAN_SUFFIXES)]


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
        if pattern.search(_code_only(text, relpath)):
            found.add(relpath)
    return found


C_STYLE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx")

#: What may precede a `/` for it to open a regex literal rather than divide.
#: The usual tokenizer heuristic: after an operator, an opening bracket, a
#: separator or one of these keywords the parser expects an operand; after an
#: identifier, a number or a closing bracket it expects an operator.
_REGEX_PRECEDERS = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_KEYWORDS = {"return", "typeof", "case", "in", "of", "delete", "void", "throw", "new", "do", "else"}


def _opens_regex(text: str, i: int) -> bool:
    j = i - 1
    while j >= 0 and text[j] in " \t":
        j -= 1
    if j < 0 or text[j] in "\n\r" or text[j] in _REGEX_PRECEDERS:
        return True
    k = j
    while k >= 0 and (text[k].isalnum() or text[k] == "_" or text[k] == "$"):
        k -= 1
    return text[k + 1:j + 1] in _REGEX_KEYWORDS


def _code_only_cstyle(text: str) -> str:
    """TypeScript/JavaScript with comments blanked in place, offsets preserved.

    Without this the TS files fall through to the Python path, `tokenize` raises
    on them, and it returns the raw text -- so a `.ts` file whose only mention of
    OPENAI_API_KEY is inside a `//` comment is reported as an egress module. That
    is the same false positive docstrings caused for Python, which listed
    milvus_recorder.py under embedding_provider on comment hits alone.

    String-aware, because `"https://host"` is not the start of a comment. Inside a
    template literal everything up to the closing backtick is treated as string,
    including a `${...}` hole: that can leave a comment unblanked, which costs a
    false positive, never a missed call.

    Regex-literal-aware too, because `/[&<>"']/g` is not the start of a string.
    The first port of this stripper was not: send-analysis-email escapes HTML
    with exactly that literal on its line 30, the unmatched `"` put the scanner
    into string state, and from then on every `"` toggled it the wrong way --
    so `fetch("https://quickchart.io/chart"` was blanked as a `//` comment and
    the one call this widening exists to catch was invisible to it. Whether a
    `/` opens a regex or divides is decided from the token before it
    (``_opens_regex``); a mistake in the divide direction can only leave a
    comment unblanked, never hide code.
    """
    out = list(text)
    i, n = 0, len(text)
    quote = None
    while i < n:
        ch = text[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"`":
            quote = ch
            i += 1
            continue
        if ch == "/" and i + 1 < n:
            following = text[i + 1]
            if following == "/":
                while i < n and text[i] != "\n":
                    out[i] = " "
                    i += 1
                continue
            if following == "*":
                end = text.find("*/", i + 2)
                end = n if end == -1 else end + 2
                for j in range(i, end):
                    if out[j] != "\n":
                        out[j] = " "
                i = end
                continue
            if _opens_regex(text, i):
                # Skip the literal as code: escapes pass, a `/` inside a
                # character class does not close it, and the line ends it
                # (an unterminated regex is a syntax error, not a string).
                i += 1
                in_class = False
                while i < n and text[i] != "\n":
                    if text[i] == "\\":
                        i += 2
                        continue
                    if text[i] == "[":
                        in_class = True
                    elif text[i] == "]":
                        in_class = False
                    elif text[i] == "/" and not in_class:
                        i += 1
                        break
                    i += 1
                continue
        i += 1
    return "".join(out)


def _code_only(text: str, relpath: str = "") -> str:
    """Source with comments and docstrings blanked, everything else byte-identical.

    TypeScript and JavaScript (by ``relpath`` suffix) go through
    ``_code_only_cstyle``; everything else is treated as Python.

    Blanks the comment/docstring spans IN PLACE rather than reassembling the file
    from tokens. The first version joined tokens with newlines, which split
    `boto3.client("s3")` into seven lines and broke every multi-token channel
    pattern -- eleven modules including app/core/storage.py stopped matching
    cloud_object_storage, so the inventory reported them as removed when the code
    had not changed at all.

    The reason to blank at all: the inventory asks which modules *send* data to a
    third party. app/api/server.py mentions dataset_samples in a comment
    explaining what is capped, and was reported as a new egress module on that
    basis.
    """
    if relpath.endswith(C_STYLE_SUFFIXES):
        return _code_only_cstyle(text)

    import io
    import tokenize

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text  # unparseable: fall back to the stricter raw match

    lines = text.splitlines(keepends=True)
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    def at(row: int, col: int) -> int:
        return offsets[row - 1] + col

    spans, prev = [], None
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            spans.append((at(*tok.start), at(*tok.end)))
            continue
        if tok.type == tokenize.STRING and prev in (
            None, tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL,
        ):
            spans.append((at(*tok.start), at(*tok.end)))
        if tok.type not in (tokenize.NL, tokenize.COMMENT):
            prev = tok.type

    out = list(text)
    for a, b in spans:
        for i in range(a, min(b, len(out))):
            if out[i] != "\n":
                out[i] = " "      # keep offsets and line structure intact
    return "".join(out)


@pytest.mark.parametrize("channel", sorted(EGRESS_CHANNELS))
def test_e11_09_egress_inventory_is_complete(channel: str) -> None:
    """The app and CLI modules that send user data to each third party are exactly the inventoried set."""
    pattern, description, expected, *scope = EGRESS_CHANNELS[channel]
    roots = scope[0] if scope else EGRESS_SCAN_ROOTS
    actual = _modules_matching(pattern, _tracked_modules(roots))
    added = sorted(actual - expected)
    removed = sorted(expected - actual)
    assert actual == expected, (
        f"egress channel '{channel}' drifted -- {description}\n"
        f"  new egress modules: {added or 'none'}\n"
        f"  inventoried modules that no longer match: {removed or 'none'}\n"
        f"{EGRESS_INVENTORY_UPDATE_INSTRUCTION}"
    )


def test_e11_09_inventory_scans_the_cli_package() -> None:
    """The scan reaches avaloka/: a CLI module with a known anchor is seen, so a new one would be."""
    assert "avaloka/io/sources.py" in _tracked_modules(), (
        "avaloka/io/sources.py is not in the scanned set; the CLI package has dropped out "
        "of the egress inventory"
    )
    pattern = EGRESS_CHANNELS["cli_object_storage_sources"][0]
    assert "avaloka/io/sources.py" in _modules_matching(pattern, _tracked_modules()), (
        "the fsspec anchor no longer matches avaloka/io/sources.py; the inventory would "
        "not notice a new object-storage reader"
    )


def test_e11_09_inventory_scans_the_web_app_and_edge_functions() -> None:
    """The scan's scope, asserted -- a scan that cannot see a tree cannot fail on it.

    This is the regression guard for the defect that let the quickchart.io
    sender ship unnoticed (DEF-009, DEF-013): ``_tracked_modules`` filtered on
    ``app/`` and ``.py``, so every EGRESS_CHANNELS case was vacuous outside one
    tree. Narrowing the scope again fails here rather than silently going green.
    """
    scanned = _tracked_modules()
    for root in EGRESS_SCAN_ROOTS:
        assert any(p.startswith(root) for p in scanned), (
            f"no tracked file under {root} is scanned, so no egress channel can "
            f"fail on anything in it; see EGRESS_SCAN_ROOTS")
    # The sender that proved the point. Named so a move does not quietly
    # un-scan it.
    assert "ui/supabase/functions/send-analysis-email/index.ts" in scanned
    pattern = EGRESS_CHANNELS["analysis_email_delivery"][0]
    assert "ui/supabase/functions/send-analysis-email/index.ts" in _modules_matching(
        pattern, scanned), (
        "the SMTP anchor no longer matches send-analysis-email; the inventory would "
        "not notice a new mail sender")


@pytest.mark.parametrize("source,fragment,visible", [
    ("// uses OPENAI_API_KEY historically\nconst a = 1;\n", "OPENAI_API_KEY", False),
    ("/* dataset_samples used to go here */\nconst b = 2;\n", "dataset_samples", False),
    ("const k = process.env.OPENAI_API_KEY;\n", "OPENAI_API_KEY", True),
    # The trap: a URL is not a comment.
    ('const u = "https://api.openai.com/v1"; const k = OPENAI_API_KEY;\n', "OPENAI_API_KEY", True),
    ('const u = "https://x"; // OPENAI_API_KEY note\n', "OPENAI_API_KEY", False),
    ("const u = `https://h/${OPENAI_API_KEY}`;\n", "OPENAI_API_KEY", True),
    ('const s = "a\\" // not a comment OPENAI_API_KEY";\n', "OPENAI_API_KEY", True),
    # The trap that hid quickchart.io: a regex literal holding a quote is not a
    # string, and the fetch on the next line is code.
    ('s.replace(/[&<>"\']/g, esc);\nfetch("https://quickchart.io/OPENAI_API_KEY");\n',
     "OPENAI_API_KEY", True),
    ("const r = /x/g; // OPENAI_API_KEY in a comment after a regex\n", "OPENAI_API_KEY", False),
    ("const q = a / b; // OPENAI_API_KEY after a division\n", "OPENAI_API_KEY", False),
    ("const d = a / b / OPENAI_API_KEY;\n", "OPENAI_API_KEY", True),
    ("return /\\/\\//.test(OPENAI_API_KEY);\n", "OPENAI_API_KEY", True),
], ids=["line-comment", "block-comment", "real-code", "url-in-string",
        "url-then-comment", "template-literal", "escaped-quote",
        "regex-with-quote", "comment-after-regex", "comment-after-division",
        "two-divisions", "slashes-inside-regex"])
def test_the_comment_stripper_understands_typescript(source: str, fragment: str, visible: bool) -> None:
    """A mention in a TS comment is not egress; a URL in a string is not a comment."""
    out = _code_only(source, "example.ts")
    assert (fragment in out) is visible
    assert len(out) == len(source), "offsets must survive: the matcher reports line numbers"


def test_the_python_comment_stripper_still_blanks_docstrings_only() -> None:
    """The TS path must not have changed what Python files do."""
    out = _code_only('"""mentions OPENAI_API_KEY"""\nimport os\nk = os.environ["OPENAI_API_KEY"]\n',
                     "example.py")
    assert out.count("OPENAI_API_KEY") == 1


# ---------------------------------------------------------------------------
# Edge functions authenticate before they act
# ---------------------------------------------------------------------------
#
# The egress inventory above only ever catches outbound DATA. It was never
# going to notice an inbound endpoint that answers without asking who is
# calling, however wide its path filter gets -- and widening that filter
# should not be mistaken for covering this class too.
#
# The class is real here because the edge runtime sets VERIFY_JWT=false
# (deploy/helm/avaloka/templates/supabase.yaml, ui/supabase/config.toml), so
# the platform enforces nothing and each function must check for itself. One
# omission is an unauthenticated endpoint, and the pod it runs in holds
# SUPABASE_SERVICE_ROLE_KEY and the Postgres password as plaintext env values
# (DEF-011). Kong fronts it as a NodePort whose functions-v1 route carries only
# the cors plugin, so "unauthenticated" means reachable, not merely internal.
#
# Cheap while the surface is three functions. That is the moment to write it.

#: The functions this suite knows about. Frozen deliberately: a new function is
#: a visible edit here with a reason, not something that quietly joins an
#: audited set -- and a test that enumerated the directory itself would pass
#: happily if the directory ever moved.
EDGE_FUNCTIONS = (
    "ui/supabase/functions/send-analysis-email/index.ts",
    "ui/supabase/functions/send-report-email/index.ts",
    "ui/supabase/functions/send-smtp-email/index.ts",
)

#: Where each of these functions puts its gate: a 401. Matching the refusal
#: rather than the check means this does not care whether the function reads a
#: bearer header, calls auth.getUser or compares a shared token.
_AUTH_GATE = re.compile(r"\b401\b")

#: What a handler must not do before that gate. `Array.from` and `Object.from`
#: are excluded because they are not database reads, and mislabelling one would
#: send the next reader hunting for a query that is not there. Reading the
#: request body counts too: a malformed-JSON 400 in front of the gate is an
#: oracle anyone can query.
_HANDLER_EFFECTS = {
    "an outbound SMTP login": re.compile(r"\blogin\s*\("),
    "an outbound fetch": re.compile(r"\bfetch\s*\("),
    "a database query": re.compile(r"(?<!Array)(?<!Object)\.from\s*\("),
    "a raw socket": re.compile(r"Deno\.connect\s*\("),
    "a read of the request body": re.compile(r"\breq\.(?:json|text|formData|arrayBuffer)\s*\("),
}


def _handler_region(text: str) -> str:
    """Everything from `Deno.serve(` on -- the request handler and nothing above it.

    Scoped on purpose. These files define their helpers above the handler, so a
    whole-file offset comparison would find `Deno.connect` inside an SMTP client
    class declared at the top and conclude the socket is opened before the auth
    check in every function, including the two that are correct.
    """
    start = text.find("Deno.serve(")
    return "" if start < 0 else text[start:]


def test_the_edge_function_set_is_what_this_suite_thinks_it_is() -> None:
    """The audited set matches the tree, so the check below cannot audit nothing."""
    # Enumerated from the filesystem, not `git ls-files`: `_tracked_paths()`
    # skips when git is unusable, and this test is what stops the strict xfail
    # below from going quiet if a function disappears. A guard that skips is not
    # a guard. The stricter direction is also the right one here -- an untracked
    # function directory is still a function somebody deployed.
    root = REPO_ROOT / "ui" / "supabase" / "functions"
    found = tuple(sorted(
        str(entry.relative_to(REPO_ROOT)).replace("\\", "/")
        for entry in root.glob("*/index.ts"))) if root.is_dir() else ()
    assert found == EDGE_FUNCTIONS, (
        f"the edge-function set changed: {found}\n"
        "(An empty tuple means the directory itself is gone, which would leave "
        "the authentication check below auditing nothing.)\n"
        "Every function must authenticate before it acts (VERIFY_JWT=false means "
        "nothing else will). Add it to EDGE_FUNCTIONS and to the parametrisation "
        "below, deliberately.")


@pytest.mark.parametrize("relpath", EDGE_FUNCTIONS)
def test_an_edge_function_authenticates_before_it_acts(relpath: str) -> None:
    """No outbound call, socket, query or body read in the handler before the 401.

    send-smtp-email carried a strict xfail here (DEF-012: the diagnose branch
    ran ahead of the bearer check). The fix moved the check to the top of the
    handler and the marker went with it; the case now holds like the other two.
    """
    path = REPO_ROOT / relpath
    # Deliberately NOT pytest.skip. A check that skips when the file is missing
    # goes quiet rather than red. An absent function is a loud failure here; the
    # frozen-set test above says which files must exist.
    assert path.is_file(), (
        f"{relpath} is absent. If a function was renamed or removed, update "
        "EDGE_FUNCTIONS and this parametrisation deliberately -- do not let this "
        "check go quiet.")
    handler = _handler_region(_code_only(path.read_text("utf-8", errors="ignore"), relpath))
    assert handler, f"{relpath} has no Deno.serve handler; this check cannot see it"

    gate = _AUTH_GATE.search(handler)
    assert gate, (
        f"{relpath} never returns 401, so it refuses nobody. With VERIFY_JWT=false "
        "the platform will not do it either.")

    earliest = None
    for label, pattern in _HANDLER_EFFECTS.items():
        found = pattern.search(handler)
        if found and (earliest is None or found.start() < earliest[1]):
            earliest = (label, found.start())
    if earliest is None:
        return

    label, where = earliest
    assert gate.start() < where, (
        f"{relpath} performs {label} before it authenticates anyone. An early "
        "return above the auth check is an unauthenticated endpoint, and the "
        "functions pod holds the service-role key and the Postgres password.")


_DAFT_IMPORT = re.compile(r"^[ \t]*(?:import[ \t]+daft\b|from[ \t]+daft(?:\.[\w.]+)?[ \t]+import\b)", re.M)
# The helper call, or -- for a script that may run without the app package --
# setting the variables directly.
_BEACON_OFF = re.compile(r"disable_thirdparty_telemetry\(\)|DO_NOT_TRACK|SCARF_NO_ANALYTICS")


@pytest.mark.parametrize("relpath", DAFT_IMPORTERS)
def test_daft_importers_disable_the_beacon_first(relpath: str) -> None:
    """Every daft import is preceded, in its own module, by the opt-out."""
    text = _code_only((REPO_ROOT / relpath).read_text("utf-8", errors="ignore"))
    first = _DAFT_IMPORT.search(text)
    assert first, f"{relpath}: no daft import; update DAFT_IMPORTERS"
    assert _BEACON_OFF.search(text[: first.start()]), (
        f"{relpath}: imports daft before disabling its usage beacon "
        "(call app.core.thirdparty_optout.disable_thirdparty_telemetry() first)"
    )


_ORT_IMPORT = re.compile(
    r"^[ \t]*(?:import[ \t]+(?:onnxruntime|chromadb)\b"
    r"|from[ \t]+(?:onnxruntime|chromadb)(?:\.[\w.]+)?[ \t]+import\b)",
    re.M,
)
_ORT_OFF = re.compile(r"disable_thirdparty_telemetry\(\)|ORT_DISABLE_TELEMETRY")


@pytest.mark.parametrize("relpath", ORT_STANDALONE_SCRIPTS)
def test_standalone_onnxruntime_scripts_disable_its_telemetry_first(relpath: str) -> None:
    """A script that runs without app/__init__.py sets ORT_DISABLE_TELEMETRY itself, first."""
    text = _code_only((REPO_ROOT / relpath).read_text("utf-8", errors="ignore"))
    first = _ORT_IMPORT.search(text)
    assert first, f"{relpath}: no onnxruntime/chromadb import; update ORT_STANDALONE_SCRIPTS"
    assert _ORT_OFF.search(text[: first.start()]), (
        f"{relpath}: imports onnxruntime before setting ORT_DISABLE_TELEMETRY"
    )


def test_app_package_disables_third_party_beacons_on_import() -> None:
    """``import app.<anything>`` runs the opt-out before the module that asked for it."""
    text = _code_only((REPO_ROOT / "app" / "__init__.py").read_text("utf-8"))
    assert re.search(r"disable_thirdparty_telemetry\(\)", text), (
        "app/__init__.py no longer calls disable_thirdparty_telemetry()"
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


def test_no_test_module_exits_the_interpreter_on_import() -> None:
    """A `sys.exit()` that runs at import time aborts the whole pytest session.

    Not the module -- the session. pytest reports INTERNALERROR and no test in
    the repository runs, whatever was actually wrong. One file did this, and it
    made CI unpassable: `tests/planner_graph_agent/test_planner_graph_agent.py`
    began as a standalone script and kept the script's `except ImportError:
    sys.exit(1)` when it became a test.

    `pytest.skip(..., allow_module_level=True)` is the idiom that expresses the
    same intent without taking everything else down with it.

    Calls inside a function body, or under `if __name__ == "__main__"`, are
    fine: neither runs on import.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent

    def _is_main_guard(node: ast.stmt) -> bool:
        return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__")

    def _import_time(body: list[ast.stmt]):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if _is_main_guard(node):
                continue
            yield node
            for field in ("body", "orelse", "finalbody"):
                yield from _import_time(getattr(node, field, []) or [])
            for handler in getattr(node, "handlers", []) or []:
                yield from _import_time(handler.body)

    offenders: list[str] = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text("utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in _import_time(tree.body):
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
                continue
            func = node.value.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            module = getattr(getattr(func, "value", None), "id", None)
            if name == "exit" and module in (None, "sys", "os"):
                offenders.append(f"{path.relative_to(root)}:{node.value.lineno}")

    assert not offenders, (
        "these call exit() at import time, which aborts the entire pytest run "
        "rather than skipping the module — use pytest.skip(..., "
        f"allow_module_level=True) instead: {', '.join(offenders)}"
    )
