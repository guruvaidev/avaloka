"""Suite C9 -- the tree the generator would publish from this commit carries no credential.

C1 scans the tracked files of this checkout. C8 proves the generator's branch
model on a toy repository. Neither looks at the tree ``scripts/generate-oss.sh``
would actually publish from HEAD: a new top-level path the manifest does not
exclude, an overlay file that carries a key, a splice that pulls an internal
section through, a sample dataset with a connection string in a cell. These
cases build that tree -- the real script, the real manifest, the content of
HEAD -- and read every file in it twice:

* for credential *shapes* (C1's patterns, a few more providers, a PEM block
  that has a body, a JWT that is not the public Supabase demo key, a database
  URL whose password is not a placeholder);
* for the SHA-256 of *values* that have at one time sat in a tracked file.
  Only the digests are here. A hit is reported by label, never by value, so
  the assertion message cannot itself become the leak.

The tree is built the way ``--publish`` builds it, in a throwaway clone whose
``github`` remote is a bare repository under ``tmp_path``: the real remote is
never contacted and nothing is pushed (``PUBLISH_CONFIRM`` is stripped from the
environment and the stand-in's refs are compared before and after). The
checkout the tests run in is not touched: no ref is created, no object written.

A scan that cannot fail is the failure mode this repository keeps producing, so
the suite proves its own teeth: ``test_c9_02`` plants one example of every
class into a temporary tree and requires each to be found, and ``test_c9_01``
refuses to pass on a tree that is implausibly small or tokenised nothing.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import pytest

from tests.contract import test_c1_repo_hygiene as c1

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate-oss.sh"

#: The ref the throwaway clone exposes HEAD under, so the script's default
#: REMOTE=origin resolves to this commit's content and nothing else.
_SOURCE_REF = "c9-public-source"


# ---------------------------------------------------------------------------
# Values known to have been in a tracked file at some point: SHA-256 only.
#
# Labels are the variable the value was assigned to. The digests are of the
# bare value, as it would be written after ``NAME=`` with no quotes. A URL
# that carries a password is listed twice: the whole URL and the password
# alone. A hit on any of these in the generated tree is a publish of a
# credential, whatever the surrounding file looks like.
# ---------------------------------------------------------------------------

KNOWN_VALUE_HASHES: Dict[str, str] = {
    "BITBUCKET_ACCESS_TOKEN #1": "7e9f83c5d4bde3d9401ece76a2e11019cc0db5c9728b11c4a8d14d0d22dca453",
    "BITBUCKET_PIPELINE_TOKEN #1": "5a0a51b50a8c7fd2651a7cb00a5cf1948b06f371d31ddeee15ee5233eb84d733",
    "GEMINI_API_KEY #1": "580b0dd64f29c55fdff2348f87247f2d404f08b09eeec32aa85a4feeae7f373b",
    "GITHUB_SYSTEM_TOKEN #1": "de3f26d63fd176f7bda4a5092ce85fc31745a224ca6dc6be317721cd48487ed0",
    "GROQ_API_KEY #1": "59c21f0de23a8409aceebb4ef6ece75a1293ba0ca54ee57cafb79288f8540908",
    "GROQ_API_KEY #2": "6dde2f4f3f2892a7d021f290403cf58f87338bc29c4e54cdc1c5191343a7784e",
    "KAGGLE_API_TOKEN #1": "8725e89fe123d2b72aaf62c165dde9105df1676b19d8f8504ea8ff0e48a1e0c0",
    "MLFLOW_BACKEND_STORE_URI #1": "b14ac1d06432e4bc22f4066bfc732c7f79413aecf2557912cbe36254eff61cf6",
    "OPENROUTER_API_KEY #1": "b7b4544e6659eafae2aef8e53b35431bf1f1f009b935e48e4085c40d08e5ebe6",
    "OPENROUTER_API_KEY #2": "c34fe5f8f1df82901d72bbaf06cdc749f8789651955aa13fa9025bb3f20303d8",
    "SUPABASE_JWT_SECRET #1": "6cce3a31568dfcde2e617592d7d3a751acadc2b2d301d0cf62ce5bd3b31d1697",
    "SUPABASE_JWT_SECRET #2": "74bc96cf9795445e5b8212d3ac6a0ad001b5bdf30a6af7c9ca47ef682d4c4889",
    "SUPABASE_SERVICE_ROLE_KEY #1": "54f34081cb327d1bab45f3515b956db16ceb5f1bf59a6b971cfa1bc64959ec1a",
    "SUPABASE_SERVICE_ROLE_KEY #2": "5cfcca43c2b4608cfa9db5c5aa200895f4301926e47e9e5d2180be2dc846d9af",
}

#: Proves the tokeniser -> digest -> lookup path end to end in test_c9_02. Not a
#: secret; its digest sits in the same dict as the real ones so the lookup the
#: self-test exercises is the lookup the real scan uses.
_CANARY_LABEL = "canary (self-test only)"
# Assembled at runtime: this file is itself part of the public tree, and a
# literal here would be found by the scan it exists to prove.
_CANARY_VALUE = "c9-canary-" + "6d1f0a9b" + "2e7c4d58"
KNOWN_VALUE_HASHES[_CANARY_LABEL] = hashlib.sha256(_CANARY_VALUE.encode()).hexdigest()

_HASH_TO_LABEL: Dict[str, str] = {digest: label for label, digest in KNOWN_VALUE_HASHES.items()}


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

SHAPE_PATTERNS: Dict[str, re.Pattern] = {
    "aws_access_key_id": c1.SECRET_PATTERNS["aws_access_key_id"],
    "aws_temporary_key_id": re.compile(r"\bASIA[0-9A-Z]{16}\b"),
    "groq_api_key": c1.SECRET_PATTERNS["groq_api_key"],
    "mcp_api_key": c1.KEY_SHAPED_LITERALS["mcp_api_key"],
    # Also matches the OpenRouter (sk-or-v1-) and Anthropic (sk-ant-) prefixes.
    "sk_api_key": c1.KEY_SHAPED_LITERALS["openai_api_key"],
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    "gitlab_token": re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "atlassian_token": re.compile(r"\bAT(?:ATT|CTT|BB)[A-Za-z0-9_=-]{20,}"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "azure_storage_key": re.compile(r"AccountKey=[A-Za-z0-9+/=]{40,}"),
}

#: AWS publishes this id in its own documentation as the canonical
#: non-functional example; test_ray_unit.py and C1 use it for that reason.
_AWS_DOCUMENTED_EXAMPLE_ID = "AKIA" + "IOSFODNN7EXAMPLE"

#: A PEM block with a body. C1 flags the header alone, which is why two test
#: fixtures that build "BEGIN ... ABCDEF ... END" sit on its allow-list. Real
#: private keys are over a thousand base64 characters; sixty-four is far below
#: any real key and far above every fixture.
_PEM_BLOCK = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?P<body>.*?)-----END [A-Z ]*PRIVATE KEY-----", re.S
)
_PEM_MIN_BODY = 64

#: C1's pattern covers postgres/mysql/mariadb/mongodb. The others here carry
#: passwords the same way and are all in use somewhere in the tree.
_URL_WITH_PASSWORD = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb|redis|rediss|amqps?|mssql|oracle|clickhouse|snowflake)"
    r"(?:\+[a-z0-9_]+)?://(?P<user>[^\s:/@\"'`]+):(?P<password>[^\s@/\"'`]+)@(?P<host>[^\s/\"'`?#]+)"
)

_JWT = c1.JWT_SHAPED

#: The signing secret of the Supabase local-development stack. The anon and
#: service_role tokens it signs are published in Supabase's own documentation
#: and ship in every `supabase start`; they unlock nothing but a developer's
#: laptop. Any other JWT in the public tree is somebody's real token.
_SUPABASE_DEMO_SECRET = b"super-secret-jwt-token-with-at-least-32-characters-long"

#: Files whose *shape* matches are synthetic by construction, with the reason.
#: They are exempt from SHAPE_PATTERNS and the URL password check only; the
#: digest scan, the PEM-with-body check and the JWT check still apply to them,
#: so a real key pasted into one of these files is still a failure.
FIXTURES_WHOSE_SHAPE_MATCHES_ARE_SYNTHETIC: Dict[str, str] = {
    **c1.FIXTURES_WHOSE_MATCHES_ARE_SYNTHETIC_AND_CARRY_NO_LIVE_CREDENTIAL,
    "tests/analytics/test_redaction.py": (
        "asserts the analytics redactor masks an sk- shaped token; the literal is "
        "an invented 27-character value that exists so the masking can be seen"
    ),
}

#: Media that cannot carry a credential anyone would type. Everything else --
#: including the sample CSVs and the embeddings sqlite file -- is read.
_MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mov"}

#: A run of characters that could be a value: everything but whitespace and the
#: delimiters a value is written between in .env, YAML, JSON, TOML, shell and
#: Markdown. Twelve is below the length of every value in the dict.
_TOKEN = re.compile(r"[^\s\"'`,;()\[\]{}<>|\\]{12,}")


def _candidate_values(text: str) -> Iterator[str]:
    """Every substring that would be the value if this text were the leak."""
    for match in _TOKEN.finditer(text):
        token = match.group(0)
        yield token
        stripped = token.rstrip(".:,;")
        if stripped != token:
            yield stripped
        if "=" in token:                      # NAME=value, with the value possibly containing '='
            yield token.split("=", 1)[1]
        if ":" in token and "://" not in token:   # name:value with no spaces
            yield token.split(":", 1)[1]
    for match in _URL_WITH_PASSWORD.finditer(text):
        yield match.group("password")


def _jwt_signed_by_demo_secret(token: str) -> bool:
    try:
        header, payload, signature = token.split(".")
    except ValueError:
        return False
    digest = hmac.new(_SUPABASE_DEMO_SECRET, f"{header}.{payload}".encode(), hashlib.sha256).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return hmac.compare_digest(expected, signature)


def _pem_body_length(body: str) -> int:
    return len(re.sub(r"\\n|\\r|\s|\"|'", "", body))


@dataclass(frozen=True)
class ScanStats:
    files_read: int
    tokens_hashed: int


def scan_tree(root: Path) -> Tuple[List[str], ScanStats]:
    """Return findings as ``"<relative path>: <kind>[ (<label>)]"`` and how much was read."""
    findings: Set[str] = set()
    files_read = tokens_hashed = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        if path.suffix.lower() in _MEDIA_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")      # lossless for the ASCII a credential is made of
        files_read += 1

        for value in _candidate_values(text):
            tokens_hashed += 1
            label = _HASH_TO_LABEL.get(hashlib.sha256(value.encode()).hexdigest())
            if label is not None:
                findings.add(f"{rel}: known_value ({label})")

        for match in _PEM_BLOCK.finditer(text):
            if _pem_body_length(match.group("body")) >= _PEM_MIN_BODY:
                findings.add(f"{rel}: private_key_with_body")

        for token in _JWT.findall(text):
            if "service_role" in c1._jwt_payload_text(token):
                findings.add(f"{rel}: supabase_service_role_jwt")
            elif not _jwt_signed_by_demo_secret(token):
                findings.add(f"{rel}: jwt_not_the_supabase_demo_key")

        if rel in FIXTURES_WHOSE_SHAPE_MATCHES_ARE_SYNTHETIC:
            continue
        for name, pattern in SHAPE_PATTERNS.items():
            for match in pattern.finditer(text):
                if name == "aws_access_key_id" and match.group(0) == _AWS_DOCUMENTED_EXAMPLE_ID:
                    continue
                findings.add(f"{rel}: {name}")
                break
        for match in _URL_WITH_PASSWORD.finditer(text):
            if not c1._password_is_placeholder(match.group("password")):
                findings.add(f"{rel}: db_url_with_password")
                break
    return sorted(findings), ScanStats(files_read=files_read, tokens_hashed=tokens_hashed)


# ---------------------------------------------------------------------------
# Generating the tree
# ---------------------------------------------------------------------------


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        message = f"{tool} is not installed; the public tree was not generated or scanned"
        if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
            pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
        pytest.skip(message)


def _clean_env(home: Path, extra: Dict[str, str]) -> Dict[str, str]:
    drop = ("SRC_BRANCH", "REMOTE", "PUBLISH_REMOTE", "PUBLISH_BRANCH", "PUBLISH_BASE", "PUBLISH_CONFIRM")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k not in drop}
    env.update({
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "c9", "GIT_AUTHOR_EMAIL": "c9@example.invalid",
        "GIT_COMMITTER_NAME": "c9", "GIT_COMMITTER_EMAIL": "c9@example.invalid",
    })
    env.update(extra)
    return env


def _git(cwd: Path, env: Dict[str, str], *args: str) -> str:
    run = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, f"git {' '.join(args)} failed in {cwd}: {run.stderr}"
    return run.stdout


@dataclass(frozen=True)
class GeneratedTree:
    commit: str
    root: Path
    log: str


def generate_public_tree(work: Path) -> GeneratedTree:
    """Build the tree ``--publish`` would publish from HEAD, in a clone, against a stand-in remote."""
    for tool in ("git", "bash", "python3"):
        _require(tool)
    env = _clean_env(work, {})
    head = _git(REPO_ROOT, env, "rev-parse", "HEAD").strip()

    clone = work / "clone"
    _git(REPO_ROOT, env, "clone", "-q", "--shared", "--no-checkout", str(REPO_ROOT), str(clone))
    _git(clone, env, "checkout", "-q", "--detach", head)
    _git(clone, env, "update-ref", f"refs/remotes/origin/{_SOURCE_REF}", head)

    # The "published tip": one commit on an empty tree, in a bare repository
    # that is the clone's `github` remote. --publish fetches it, parents the
    # squash commit on it, and -- without PUBLISH_CONFIRM -- stops there.
    stand_in = work / "published.git"
    _git(work, env, "init", "-q", "--bare", str(stand_in))
    empty_tree = _git(clone, env, "hash-object", "-t", "tree", os.devnull).strip()
    seed = _git(clone, env, "commit-tree", empty_tree, "-m", "stand-in for the published tip").strip()
    _git(clone, env, "push", "-q", str(stand_in), f"{seed}:refs/heads/main")
    _git(clone, env, "remote", "add", "github", str(stand_in))
    stand_in_refs_before = _git(stand_in, env, "for-each-ref")

    run_env = _clean_env(work, {
        "SRC_BRANCH": _SOURCE_REF, "REMOTE": "origin",
        "PUBLISH_REMOTE": "github", "PUBLISH_BRANCH": "main",
    })
    run = subprocess.run(["bash", "scripts/generate-oss.sh", "--publish"], cwd=clone, env=run_env,
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, f"generator exited {run.returncode}\n{run.stdout}\n{run.stderr}"
    assert "dry run. Set PUBLISH_CONFIRM=yes" in run.stdout, run.stdout
    assert _git(stand_in, env, "for-each-ref") == stand_in_refs_before, "the stand-in remote was pushed to"

    match = re.search(r"^   commit: ([0-9a-f]{7,40})", run.stdout, re.M)
    assert match, f"no commit id in the generator's output:\n{run.stdout}"
    commit = _git(clone, env, "rev-parse", match.group(1)).strip()
    assert _git(clone, env, "rev-parse", f"{commit}^").strip() == seed, "the squash commit is not parented on the published tip"

    root = work / "tree"
    root.mkdir()
    archive = subprocess.run(["git", "archive", "--format=tar", commit], cwd=clone, env=env,
                             capture_output=True, check=True, timeout=600).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(root, filter="data")
        else:  # pragma: no cover - interpreters older than the filter
            tar.extractall(root)
    return GeneratedTree(commit=commit, root=root, log=run.stdout)


@pytest.fixture(scope="module")
def public_tree(tmp_path_factory: pytest.TempPathFactory) -> GeneratedTree:
    return generate_public_tree(tmp_path_factory.mktemp("c9"))


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def test_c9_00_denylist_is_well_formed() -> None:
    """Every entry is a full SHA-256 hex digest, labels are unique, and the real entries are there."""
    assert len(KNOWN_VALUE_HASHES) >= 10, "the known-value list has lost entries"
    for label, digest in KNOWN_VALUE_HASHES.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), f"{label}: not a sha256 hex digest"
    assert len(set(KNOWN_VALUE_HASHES.values())) == len(KNOWN_VALUE_HASHES), "duplicate digests"
    for name in ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_JWT_SECRET", "MLFLOW_BACKEND_STORE_URI"):
        assert any(label.startswith(name) for label in KNOWN_VALUE_HASHES), f"no entry for {name}"


def test_c9_01_generated_public_tree_carries_no_credential(public_tree: GeneratedTree) -> None:
    """The tree --publish would push from HEAD has no credential shape and none of the known values."""
    root = public_tree.root
    for must in ("README.md", "oss/manifest.yaml", "scripts/generate-oss.sh", "LICENSE"):
        assert (root / must).is_file(), f"{must} missing: this is not the public tree"
    for must_not in (".env", ".env.bak.pre-docker-switch", ".claude"):
        assert not (root / must_not).exists(), f"{must_not} is in the generated tree"
    assert "ok   .env unreachable from the published ref" in public_tree.log, public_tree.log

    findings, stats = scan_tree(root)
    # A scan of a near-empty tree would pass vacuously. The public tree is well
    # over a thousand files and tokenises hundreds of thousands of candidates.
    assert stats.files_read >= 1000, f"only {stats.files_read} files read; the tree is not the public tree"
    assert stats.tokens_hashed >= 100_000, f"only {stats.tokens_hashed} candidates hashed; the scan did not read the tree"
    assert not findings, (
        f"the public tree generated from {public_tree.commit[:12]} carries credential material "
        "(values withheld by design):\n  " + "\n  ".join(findings)
    )


def test_c9_02_scanner_bites(tmp_path: Path) -> None:
    """Plant one of every class into a tree and the scan names each; a clean tree yields nothing."""
    planted = tmp_path / "planted"
    planted.mkdir()

    def write(rel: str, text: str) -> None:
        path = planted / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    # The canary's digest is in KNOWN_VALUE_HASHES; six syntaxes a value gets written in.
    write("conf/.env", f"SOME_SETTING={_CANARY_VALUE}\n")
    write("conf/values.yaml", f"supabase:\n  key: \"{_CANARY_VALUE}\"\n")
    write("conf/config.json", json.dumps({"token": _CANARY_VALUE}))
    write("conf/notes.md", f"Use `{_CANARY_VALUE}` for the demo.\n")
    write("conf/run.sh", f"export SETTING='{_CANARY_VALUE}'\n")
    write("conf/trailing.txt", f"the value is {_CANARY_VALUE}.\n")
    # Shapes. None of these literals is a real credential; the suffixes are
    # arbitrary characters of the right class and length.
    write("shapes/openrouter.py", "KEY = 'sk-or-v1-" + "0123456789abcdef" * 4 + "'\n")
    write("shapes/aws.txt", "aws_access_key_id = AKIA" + "ABCDEFGHIJKLMNOP" + "\n")
    write("shapes/github.txt", "token: ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8" + "\n")
    write("shapes/groq.txt", "GROQ=gsk_" + "Zy9Xw8Vu7Ts6Rq5Po4Nm3Lk2Ji1Hg0Fe" + "\n")
    write("shapes/db.py", 'DSN = "postgresql://app:' + "Tr0ub4dor-3x-9q" + '@db.internal:5432/app"\n')
    pem_body = base64.b64encode(bytes(range(96))).decode()
    write("shapes/sa.json", json.dumps({
        "type": "service_account",
        "private_key": "-----BEGIN " + f"PRIVATE KEY-----\\n{pem_body}\\n-----END PRIVATE KEY-----\\n",
    }))
    header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(b'{"iss":"supabase","ref":"abcdefghijklmnopqrst","role":"anon"}').rstrip(b"=").decode()
    signature = base64.urlsafe_b64encode(
        hmac.new(b"not-the-demo-secret-at-all-0123456789", f"{header}.{payload}".encode(), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    write("shapes/supabase.ts", f'export const KEY = "{header}.{payload}.{signature}";\n')
    sr_payload = base64.urlsafe_b64encode(b'{"iss":"supabase","role":"service_role"}').rstrip(b"=").decode()
    write("shapes/service.ts", f'export const KEY = "{header}.{sr_payload}.{signature}";\n')

    findings, stats = scan_tree(planted)
    assert stats.files_read == 14
    kinds = {f for f in findings}
    for expected in (
        f"conf/.env: known_value ({_CANARY_LABEL})",
        f"conf/values.yaml: known_value ({_CANARY_LABEL})",
        f"conf/config.json: known_value ({_CANARY_LABEL})",
        f"conf/notes.md: known_value ({_CANARY_LABEL})",
        f"conf/run.sh: known_value ({_CANARY_LABEL})",
        f"conf/trailing.txt: known_value ({_CANARY_LABEL})",
        "shapes/openrouter.py: sk_api_key",
        "shapes/aws.txt: aws_access_key_id",
        "shapes/github.txt: github_token",
        "shapes/groq.txt: groq_api_key",
        "shapes/db.py: db_url_with_password",
        "shapes/sa.json: private_key_with_body",
        "shapes/supabase.ts: jwt_not_the_supabase_demo_key",
        "shapes/service.ts: supabase_service_role_jwt",
    ):
        assert expected in kinds, f"not found: {expected}\nfound: {findings}"

    # And the things the real tree legitimately contains are not findings.
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "doc.md").write_text(
        "Connect with `postgresql://user:password@host/db#table`.\n"
        f"Example access key id: {_AWS_DOCUMENTED_EXAMPLE_ID}\n"
        'private_key = "-----BEGIN ' + 'PRIVATE KEY-----\\nABCDEF\\n-----END PRIVATE KEY-----\\n"\n'
        'url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"\n',
        encoding="utf-8",
    )
    demo_payload = base64.urlsafe_b64encode(b'{"iss":"supabase-demo","role":"anon","exp":1799535600}').rstrip(b"=").decode()
    demo_sig = base64.urlsafe_b64encode(
        hmac.new(_SUPABASE_DEMO_SECRET, f"{header}.{demo_payload}".encode(), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    (clean / "config.ts").write_text(f'export const ANON = "{header}.{demo_payload}.{demo_sig}";\n', encoding="utf-8")
    findings, stats = scan_tree(clean)
    assert stats.files_read == 2
    assert findings == [], findings


def test_c9_03_generation_touches_nothing_outside_its_sandbox(public_tree: GeneratedTree) -> None:
    """Building the tree created no ref here and left no `github` fetch mark on this checkout."""
    env = _clean_env(public_tree.root.parent, {})
    refs = _git(REPO_ROOT, env, "for-each-ref", "--format=%(refname)")
    assert f"refs/remotes/origin/{_SOURCE_REF}" not in refs, "the source ref leaked into this checkout"
    assert not (REPO_ROOT / "FETCH_HEAD").exists() or _SOURCE_REF not in (REPO_ROOT / "FETCH_HEAD").read_text(errors="ignore")
