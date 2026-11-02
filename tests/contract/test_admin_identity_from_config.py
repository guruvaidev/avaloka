"""Administrators come from configuration, never from a hard-coded identity.

Two things shipped in the 1.6 line that this guards against:

* a migration that inserted an ``admin`` row into ``public.user_roles`` for one
  literal auth-user UUID (and an older one keyed to a literal email). The chart's
  migrate Job re-ran every migration on every deploy, so those grants came back on
  every upgrade of every installation;
* UI server code that decided "is this the admin?" by comparing the caller's
  email with a string literal.

Now the admin list is ``admin.emails`` in the chart, which reaches the web UI and
the migrate hook as ``ADMIN_EMAILS``. Empty means nobody.

The scan covers both migration directories (and any seed SQL beside them),
``ui/src``, ``app/`` and the CLI package ``avaloka/``. Failure messages name the
file, line and kind of match, never the matched address, so a failing run does
not copy personal data into CI logs.

The helm checks render the chart and look at what reaches the pods.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHART_ROOT = REPO_ROOT / "deploy" / "helm" / "avaloka"
UI_MIGRATIONS = REPO_ROOT / "ui" / "supabase" / "migrations"
CHART_MIGRATIONS = CHART_ROOT / "files" / "supabase-migrations"

# RFC 2606 / 6761 names are placeholders by definition, so a literal on one of
# them is not an identity.
_PLACEHOLDER_DOMAIN = re.compile(r"@(?:[\w-]+\.)*(?:example\.(?:com|org|net)|[\w-]+\.(?:test|invalid|example))$", re.I)
_EMAIL = r"[A-Za-z0-9._%+'-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_EMAIL_RE = re.compile(_EMAIL)
_UUID_RE = re.compile(_UUID)
_SQL_EMAIL_LITERAL = re.compile(r"'(" + _EMAIL + r")'")
_SQL_UUID_LITERAL = re.compile(r"'(" + _UUID + r")'")

# Source rules. Each names a way a literal identity decides admin status.
_Q = r"[\"'`]"
_SOURCE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email literal assigned to an admin-named identifier",
        re.compile(r"\b\w*admin\w*\b\s*(?::\s*[\w\[\]<>|, ]+)?\s*[=:]\s*[\[\(\{]?\s*" + _Q + r"(" + _EMAIL + r")" + _Q, re.I),
    ),
    (
        "email literal in an equality comparison",
        re.compile(r"(?:===?|!==?)\s*" + _Q + r"(" + _EMAIL + r")" + _Q + r"|" + _Q + r"(" + _EMAIL + r")" + _Q + r"\s*(?:===?|!==?)"),
    ),
    (
        "email literal in a membership test",
        re.compile(r"\bin\s*[\[\(\{]\s*" + _Q + r"(" + _EMAIL + r")" + _Q),
    ),
    (
        "UUID literal compared or assigned on an admin/role line",
        re.compile(r"(?=.*\b(?:admin|role)\w*)(?:===?|!==?|=|:)\s*" + _Q + r"(" + _UUID + r")" + _Q, re.I),
    ),
)

_SOURCE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py"}
_SKIP_DIRS = {"node_modules", "__pycache__", ".venv", "dist", "build", ".output"}


def _is_real_identity(literal: str) -> bool:
    return bool(_EMAIL_RE.fullmatch(literal)) and not _PLACEHOLDER_DOMAIN.search(literal)


def _sql_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for root in (UI_MIGRATIONS, CHART_MIGRATIONS, REPO_ROOT / "ui" / "supabase"):
        if root.is_dir():
            files += sorted(root.glob("*.sql"))
    seeds = REPO_ROOT / "ui" / "supabase" / "seed.sql"
    if seeds.is_file():
        files.append(seeds)
    return sorted(set(files))


def _sql_statements(text: str) -> list[tuple[int, str]]:
    """(1-based line of the statement start, statement) with -- comments dropped."""
    stripped = "\n".join(line.split("--", 1)[0] for line in text.splitlines())
    out: list[tuple[int, str]] = []
    line = 1
    for chunk in stripped.split(";"):
        lead = len(chunk) - len(chunk.lstrip("\n"))
        out.append((line + lead, chunk))
        line += chunk.count("\n")
    return out


def _grant_hits(text: str) -> list[tuple[int, str]]:
    """(line, kind) for each statement that grants a role to a literal identity."""
    found: list[tuple[int, str]] = []
    for line, stmt in _sql_statements(text):
        low = stmt.lower()
        writes = re.search(r"\b(insert|update)\b", low)
        if not writes or not ("user_roles" in low or re.search(r"'admin'", low)):
            continue
        if _SQL_UUID_LITERAL.search(stmt):
            found.append((line, "role grant keyed to a literal auth-user UUID"))
        if any(_is_real_identity(m) for m in _SQL_EMAIL_LITERAL.findall(stmt)):
            found.append((line, "role grant keyed to a literal email"))
    return found


def _grant_violations_in_sql(path: pathlib.Path) -> list[str]:
    rel = path.relative_to(REPO_ROOT).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    return [f"{rel}:{line}: {kind}" for line, kind in _grant_hits(text)]


# ---- function bodies and policy expressions --------------------------------
#
# A role grant is not the only way SQL can hard-code an admin: a SECURITY DEFINER
# helper such as is_support_admin() compared auth.jwt() ->> 'email' with a literal,
# and RLS policies call it. What counts is the definition in force once every
# migration has run, so each directory is replayed in the hook's order (file-name
# sort) and only the LAST CREATE FUNCTION per name, and the last CREATE POLICY per
# (policy, table), is judged. An old literal superseded by a later definition is
# history, not a live grant.

_FUNC_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+(?P<name>[\w.\"]+)\s*\((?P<rest>.*?)"
    r"\bAS\s+(?P<q>\$\w*\$)(?P<body>.*?)(?P=q)",
    re.I | re.S,
)
_POLICY_RE = re.compile(
    r"CREATE\s+POLICY\s+(?P<name>\"[^\"]+\"|\w+)\s+ON\s+(?P<table>[\w.\"]+)(?P<expr>[^;]*);",
    re.I | re.S,
)
_IDENTITY_COMPARISON = re.compile(
    r"(?:auth\.uid\(\)|\buser_id\b|\bid\b|auth\.jwt\(\)\s*->>?\s*'(?:sub|email)')\s*(?:=|<>|!=|IN\s*\()\s*'(" + _UUID + r")'"
    r"|'(" + _UUID + r")'\s*(?:=|<>|!=)\s*(?:auth\.uid\(\)|\buser_id\b|\bid\b)",
    re.I,
)


def _norm(name: str) -> str:
    return name.replace('"', "").lower().removeprefix("public.")


def _identity_kinds(text: str) -> list[str]:
    kinds: list[str] = []
    if any(_is_real_identity(m) for m in _SQL_EMAIL_LITERAL.findall(text)):
        kinds.append("literal email")
    if _IDENTITY_COMPARISON.search(text):
        kinds.append("literal UUID compared with the caller's identity")
    return kinds


def _effective_definitions(files: list[tuple[str, str]]) -> dict[tuple[str, str], tuple[str, int, str]]:
    """{(kind, key): (file, line, text)} for the last definition of each function/policy."""
    out: dict[tuple[str, str], tuple[str, int, str]] = {}
    for rel, text in files:  # already in replay order
        for m in _FUNC_RE.finditer(text):
            out[("function", _norm(m.group("name")))] = (rel, text.count("\n", 0, m.start()) + 1, m.group("body"))
        for m in _POLICY_RE.finditer(text):
            key = f"{m.group('name').strip(chr(34))} ON {_norm(m.group('table'))}"
            out[("policy", key)] = (rel, text.count("\n", 0, m.start()) + 1, m.group("expr"))
    return out


def _definition_hits(files: list[tuple[str, str]]) -> list[str]:
    found: list[str] = []
    for (kind, key), (rel, line, body) in sorted(_effective_definitions(files).items()):
        for what in _identity_kinds(body):
            found.append(f"{rel}:{line}: {kind} {key} decides access by a {what}")
    return found


def _migration_dirs() -> list[pathlib.Path]:
    return [d for d in (UI_MIGRATIONS, CHART_MIGRATIONS) if d.is_dir()]


def _source_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for root in (REPO_ROOT / "ui" / "src", REPO_ROOT / "app", REPO_ROOT / "avaloka"):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix in _SOURCE_SUFFIXES and path.is_file() and not (_SKIP_DIRS & set(path.parts)):
                out.append(path)
    return sorted(out)


def _line_hits(line: str) -> list[str]:
    """Kinds of literal-identity admin decision on one source line."""
    hits: list[str] = []
    for kind, rule in _SOURCE_RULES:
        for match in rule.finditer(line):
            literal = next((g for g in match.groups() if g), "")
            if _UUID_RE.fullmatch(literal) or _is_real_identity(literal):
                hits.append(kind)
                break
    return hits


def _source_violations(path: pathlib.Path) -> list[str]:
    rel = path.relative_to(REPO_ROOT).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        f"{rel}:{lineno}: {kind}"
        for lineno, line in enumerate(text.splitlines(), 1)
        for kind in _line_hits(line)
    ]


def test_scan_covers_the_migration_directories_and_sources() -> None:
    # A scan that silently finds no files passes for the wrong reason.
    sql = _sql_files()
    assert any(p.parent == UI_MIGRATIONS for p in sql), "no UI migrations scanned"
    assert any(p.parent == CHART_MIGRATIONS for p in sql), "no chart migrations scanned"
    sources = _source_files()
    for part in ("ui/src/", "app/", "avaloka/"):
        assert any(p.relative_to(REPO_ROOT).as_posix().startswith(part) for p in sources), f"nothing scanned under {part}"


def test_no_migration_or_seed_grants_a_role_to_a_hardcoded_identity() -> None:
    violations = [v for path in _sql_files() for v in _grant_violations_in_sql(path)]
    assert not violations, (
        f"{len(violations)} SQL statement(s) grant a role to a hard-coded identity. "
        "Admins come from admin.emails / ADMIN_EMAILS at deploy time:\n  " + "\n  ".join(violations)
    )


def test_no_function_or_policy_in_force_decides_access_by_a_hardcoded_identity() -> None:
    violations: list[str] = []
    for d in _migration_dirs():
        files = [(p.relative_to(REPO_ROOT).as_posix(), p.read_text(encoding="utf-8", errors="replace"))
                 for p in sorted(d.glob("*.sql"))]
        violations += _definition_hits(files)
    assert not violations, (
        f"{len(violations)} SQL function/policy definition(s) in force decide access by a "
        "hard-coded identity. Check public.user_roles (granted from ADMIN_EMAILS) instead:\n  "
        + "\n  ".join(violations)
    )


def test_definition_rule_judges_only_the_definition_in_force() -> None:
    literal_fn = (
        "CREATE OR REPLACE FUNCTION public.is_support_admin()\nRETURNS boolean LANGUAGE sql\n"
        "SECURITY DEFINER SET search_path = public\nAS $$\n"
        "  SELECT COALESCE(lower(auth.jwt() ->> 'email') = 'desk@company.io', false)\n$$;\n"
    )
    roles_fn = (
        "CREATE OR REPLACE FUNCTION public.is_support_admin()\nRETURNS boolean LANGUAGE sql\n"
        "SET search_path = ''\nAS $$ SELECT EXISTS (SELECT 1 FROM public.user_roles r "
        "WHERE r.user_id = auth.uid() AND r.role = 'admin') $$;\n"
    )
    uuid_policy = (
        'CREATE POLICY "owner sees all" ON public.support_tickets FOR SELECT TO authenticated\n'
        "  USING (auth.uid() = '3f0c171b-5b72-4762-9d02-e1822ec2fccf');\n"
    )
    clean_policy = (
        'CREATE POLICY "owner sees all" ON public.support_tickets FOR SELECT TO authenticated\n'
        "  USING (public.is_support_admin());\n"
    )
    assert _definition_hits([("a.sql", literal_fn)]) == [
        "a.sql:1: function is_support_admin decides access by a literal email"
    ]
    assert _definition_hits([("a.sql", literal_fn), ("b.sql", roles_fn)]) == []
    assert _definition_hits([("a.sql", roles_fn), ("b.sql", literal_fn)]) != []
    assert _definition_hits([("p.sql", uuid_policy)]) == [
        "p.sql:1: policy owner sees all ON support_tickets decides access by a "
        "literal UUID compared with the caller's identity"
    ]
    assert _definition_hits([("p.sql", uuid_policy), ("q.sql", clean_policy)]) == []
    # A plan id or a placeholder address is not an identity.
    assert _definition_hits([("s.sql", "CREATE FUNCTION f() RETURNS uuid AS $$ SELECT "
                                       "'3f0c171b-5b72-4762-9d02-e1822ec2fccf'::uuid $$;")]) == []
    assert _definition_hits([("s.sql", "CREATE FUNCTION f() RETURNS bool AS $$ SELECT "
                                       "auth.jwt() ->> 'email' = 'x@example.com' $$;")]) == []


def test_no_source_decides_admin_by_a_hardcoded_identity() -> None:
    violations = [v for path in _source_files() for v in _source_violations(path)]
    assert not violations, (
        f"{len(violations)} source line(s) decide admin status from a literal identity. "
        "Read ADMIN_EMAILS instead (ui/src/lib/admin-emails.ts):\n  " + "\n  ".join(violations)
    )


@pytest.mark.parametrize(
    "line",
    [
        'const ADMIN_EMAIL = "someone@company.io";',
        "export const SUPPORT_ADMIN_EMAIL = 'desk@company.io';",
        'if (claimEmail === "desk@company.io") {',
        'userEmail?.toLowerCase() === "desk@company.io"',
        'if email in {"desk@company.io", "x@company.io"}:',
        "ADMIN_EMAILS = ['desk@company.io']",
        "if (user.id === '3f0c171b-5b72-4762-9d02-e1822ec2fccf' && role) {",
    ],
)
def test_source_rules_catch_each_known_shape(line: str) -> None:
    assert _line_hits(line), f"no rule matched: {line}"


@pytest.mark.parametrize(
    "line",
    [
        'placeholder="you@company.com"',
        'head_email: "olivia@avaloka.ai", role: "Administrator"',
        "if (isAdminEmail(claimEmail)) {",
        'expect(isAdminEmail("ops@example.com", "ops@example.com")).toBe(true);',
        'if (claimEmail === "desk@example.com") {',
    ],
)
def test_source_rules_leave_non_identity_lines_alone(line: str) -> None:
    assert not _line_hits(line), f"false positive: {line}"


def test_sql_rule_catches_uuid_and_email_grants() -> None:
    text = (
        "-- seed\n"
        "INSERT INTO public.user_roles (user_id, role) VALUES ('3f0c171b-5b72-4762-9d02-e1822ec2fccf', 'admin');\n"
        "INSERT INTO public.user_roles (user_id, role)\n"
        "SELECT id, 'admin'::public.app_role FROM auth.users WHERE email = 'desk@company.io';\n"
        "INSERT INTO public.user_roles (user_id, role) SELECT id, 'admin' FROM auth.users WHERE email = 'ops@example.com';\n"
        "UPDATE public.profiles SET role = 'admin' WHERE id = '3f0c171b-5b72-4762-9d02-e1822ec2fccf';\n"
        "SELECT public.has_role('3f0c171b-5b72-4762-9d02-e1822ec2fccf', 'admin');\n"
    )
    assert [line for line, _ in _grant_hits(text)] == [2, 3, 6]


# --------------------------------------------------------------------------
# Rendered chart
# --------------------------------------------------------------------------

SENTINELS = {
    "supabase.db.password": "SENTINEL-DB-PASSWORD-7f3a",
    "supabase.jwtSecret": "SENTINEL-JWT-SECRET-0c41-aaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "supabase.anonKey": "SENTINEL-ANON-KEY-91be",
    "supabase.serviceKey": "SENTINEL-SERVICE-KEY-55d2",
}


def _helm_run(*extra: str) -> subprocess.CompletedProcess:
    """Raw `helm template` with the sentinel credentials; the caller checks the result."""
    helm = shutil.which("helm")
    if helm is None:
        message = "helm is not installed, so the admin.emails / migrate-hook render cannot be checked"
        if os.environ.get("CI") or os.environ.get("AVALOKA_REQUIRE_HELM"):
            pytest.fail(message + "; install helm in this pipeline")
        pytest.skip(message)
    args = ["--set", "supabase.enabled=true", "--set", "supabase.requireOwnCredentials=false"]
    for key, value in SENTINELS.items():
        args += ["--set-string", f"{key}={value}"]
    return subprocess.run(
        [helm, "template", "avaloka", str(CHART_ROOT), *args, *extra],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


def _helm_template(*extra: str) -> list[dict]:
    helm = shutil.which("helm")
    if helm is None:
        message = "helm is not installed, so the admin.emails / migrate-hook render cannot be checked"
        if os.environ.get("CI") or os.environ.get("AVALOKA_REQUIRE_HELM"):
            pytest.fail(message + "; install helm in this pipeline")
        pytest.skip(message)
    args = ["--set", "supabase.enabled=true", "--set", "supabase.requireOwnCredentials=false"]
    for key, value in SENTINELS.items():
        args += ["--set-string", f"{key}={value}"]
    result = subprocess.run(
        [helm, "template", "avaloka", str(CHART_ROOT), *args, *extra],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"helm template failed:\n{result.stderr}"
    return [doc for doc in yaml.safe_load_all(result.stdout) if isinstance(doc, dict)]


def _find(docs: list[dict], kind: str, suffix: str) -> dict:
    hits = [d for d in docs if d.get("kind") == kind and d["metadata"]["name"].endswith(suffix)]
    assert len(hits) == 1, f"expected one {kind} *{suffix}, found {[d['metadata']['name'] for d in hits]}"
    return hits[0]


def _env(container: dict) -> dict[str, str]:
    return {e["name"]: e.get("value") for e in container.get("env", [])}


def _migrate_job(docs: list[dict]) -> dict:
    return _find(docs, "Job", "-supabase-migrate")


def _container(workload: dict, name: str) -> dict:
    containers = workload["spec"]["template"]["spec"]["containers"]
    return next(c for c in containers if c["name"] == name)


@pytest.fixture(scope="module")
def default_render() -> list[dict]:
    return _helm_template()


def test_migrate_hook_keeps_image_timing_and_runs_the_tracking_script(default_render: list[dict]) -> None:
    job = _migrate_job(default_render)
    ann = job["metadata"]["annotations"]
    assert ann["helm.sh/hook"] == "post-install,post-upgrade"
    assert ann["helm.sh/hook-weight"] == "5"
    assert ann["helm.sh/hook-delete-policy"] == "before-hook-creation"
    migrate = _container(job, "migrate")
    assert migrate["image"] == "supabase/postgres:15.8.1.060"
    assert migrate["command"] == ["/bin/sh", "/migrations/migrate.sh"]

    cm = _find(default_render, "ConfigMap", "-supabase-migrations")
    assert "migrate.sh" in cm["data"] and "baseline.txt" in cm["data"], "ConfigMap lacks the runner or baseline"
    script = cm["data"]["migrate.sh"]
    for needle in (
        "avaloka_meta.schema_migrations",      # the version table
        "'backfilled'",                        # existing installs are backfilled
        "to_regclass('public.ray_jobs')",      # ... detected from the newest baseline objects
        "--single-transaction",                # new migrations run in one transaction
        "ON_ERROR_STOP=1",
        "ON CONFLICT (user_id, role) DO NOTHING",  # the admin bootstrap only adds
    ):
        assert needle in script, f"migrate.sh lost {needle!r}"
    assert "DELETE FROM public.user_roles" not in script.upper().replace("DELETE  FROM", "DELETE FROM")
    assert "REVOKE" not in "\n".join(l for l in script.splitlines() if "user_roles" in l).upper()

    baseline = [l for l in cm["data"]["baseline.txt"].splitlines() if l and not l.startswith("#")]
    sql_keys = {k[:-4] for k in cm["data"] if k.endswith(".sql")}
    assert baseline and set(baseline) <= sql_keys, "baseline lists a migration the chart no longer ships"


def test_baseline_is_frozen_and_markers_come_from_it() -> None:
    """The backfill trusts the baseline; it must name only real, pre-tracking files,
    and the objects the detection looks for must be created by baseline files."""
    path = CHART_ROOT / "files" / "db-migrations.baseline"
    assert path.is_file(), "no frozen baseline: the migrate hook cannot backfill an existing install"
    lines = path.read_text().splitlines()
    baseline = [l for l in lines if l and not l.startswith("#")]
    assert len(baseline) == len(set(baseline))
    shipped = {p.stem for p in CHART_MIGRATIONS.glob("*.sql")}
    assert set(baseline) <= shipped
    text = "\n".join((CHART_MIGRATIONS / f"{v}.sql").read_text().lower() for v in baseline)
    for table in ("public.ray_jobs", "public.notifications", "public.user_roles"):
        assert re.search(r"create table (if not exists )?" + re.escape(table) + r"\b", text), table


def test_empty_admin_emails_reaches_nobody(default_render: list[dict]) -> None:
    for workload, container in ((_migrate_job(default_render), "migrate"),
                                (_find(default_render, "Deployment", "-webui"), "webui")):
        env = _env(_container(workload, container))
        assert "ADMIN_EMAILS" in env, f"{workload['metadata']['name']} has no ADMIN_EMAILS"
        assert env["ADMIN_EMAILS"] == "", f"{workload['metadata']['name']} has a default admin"


def test_configured_admin_emails_reach_the_ui_and_the_hook() -> None:
    docs = _helm_template(
        "--set", "admin.emails={ Ops@Example.com , lead@example.com}",
        "--set", "supabase.auth.autoconfirm=false",
    )
    for workload, container in ((_migrate_job(docs), "migrate"), (_find(docs, "Deployment", "-webui"), "webui")):
        env = _env(_container(workload, container))
        assert env.get("ADMIN_EMAILS") == "ops@example.com,lead@example.com", workload["metadata"]["name"]


def test_no_plaintext_secret_in_the_migrate_hook_beyond_pgpassword(default_render: list[dict]) -> None:
    """The DB password reaches the hook only as PGPASSWORD (the existing handling);
    it is never in the script, the command, the ConfigMap, or another env var, and
    the script never prints the libpq variables."""
    job = _migrate_job(default_render)
    migrate = _container(job, "migrate")
    env = _env(migrate)
    assert env["PGPASSWORD"] == SENTINELS["supabase.db.password"]
    elsewhere = yaml.safe_dump({k: v for k, v in env.items() if k != "PGPASSWORD"}) + yaml.safe_dump(
        {"command": migrate.get("command"), "args": migrate.get("args")}
    )
    cm = yaml.safe_dump(_find(default_render, "ConfigMap", "-supabase-migrations"))
    for name, secret in SENTINELS.items():
        assert secret not in elsewhere, f"{name} appears in the migrate container outside PGPASSWORD"
        assert secret not in cm, f"{name} appears in the migrations ConfigMap"
    script_path = CHART_ROOT / "files" / "db-migrate.sh"
    assert script_path.is_file(), "the migrate hook has no tracking script"
    script = script_path.read_text()
    assert not re.search(r"^\s*set\s+-[a-z]*x", script, re.M), "set -x would echo the connection settings"
    assert not re.search(r"(echo|printf)[^\n]*\$\{?PG(PASSWORD|HOST|USER)", script), "script prints a PG* variable"
    assert not re.search(r"(echo|printf|cat)[^\n]*\$\{?ADMIN_EMAILS", script), "script prints the admin addresses"
    # psql prints query results: SELECT set_config(..., :'admin_emails') would echo the list.
    for line in script.splitlines():
        if "-v admin_emails=" in line:
            assert ">/dev/null" in line, "the psql call holding the admin list does not discard its output"


# --------------------------------------------------------------------------
# admin.emails with Supabase email autoconfirm: refused
# --------------------------------------------------------------------------

_AUTOCONFIRM_FIX = "set supabase.auth.autoconfirm=false"


@pytest.mark.parametrize("autoconfirm", ["true", "True"])
def test_admin_emails_with_autoconfirm_refuses_to_render(autoconfirm: str) -> None:
    """With autoconfirm on, a listed address with no account could be claimed by anyone."""
    result = _helm_run("--set", "admin.emails={ops@example.com}", "--set-string", f"supabase.auth.autoconfirm={autoconfirm}")
    assert result.returncode != 0, "chart rendered admin.emails together with autoconfirm"
    assert "admin.emails is set while supabase.auth.autoconfirm is true" in result.stderr, result.stderr
    assert _AUTOCONFIRM_FIX in result.stderr and "leave admin.emails empty" in result.stderr
    assert "ops@example.com" not in result.stderr, "the refusal echoed an address"


def test_admin_emails_with_the_chart_default_autoconfirm_refuses_to_render() -> None:
    result = _helm_run("--set", "admin.emails={ops@example.com}")
    assert result.returncode != 0 and "supabase.auth.autoconfirm is true" in result.stderr


def test_admin_emails_alone_renders_with_autoconfirm_off() -> None:
    docs = _helm_template("--set", "admin.emails={ops@example.com}", "--set", "supabase.auth.autoconfirm=false")
    env = _env(_container(_migrate_job(docs), "migrate"))
    assert env.get("ADMIN_EMAILS") == "ops@example.com"
    assert env.get("GOTRUE_MAILER_AUTOCONFIRM") == "false", "the hook cannot see the autoconfirm setting"


def test_autoconfirm_alone_renders_with_no_admins() -> None:
    docs = _helm_template("--set", "supabase.auth.autoconfirm=true")
    env = _env(_container(_migrate_job(docs), "migrate"))
    assert env.get("ADMIN_EMAILS") == ""
    assert env.get("GOTRUE_MAILER_AUTOCONFIRM") == "true"


def test_hosted_supabase_is_not_judged_by_the_in_cluster_autoconfirm() -> None:
    """supabase.enabled=false: no gotrue is rendered, its autoconfirm value means nothing."""
    result = _helm_run("--set", "supabase.enabled=false", "--set", "admin.emails={ops@example.com}")
    assert result.returncode == 0, result.stderr
