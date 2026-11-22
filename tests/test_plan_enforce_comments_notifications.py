"""Comments, and notifying other people, need a Professional plan or higher.

Owner ruling 2026-11-01: share_analysis, comment_on_analysis and notifications
are Professional and Enterprise (app/core/editions.py). Migration
20261103100000_plan_enforce_comments_notifications.sql enforces the comment and
notification half in the database; this suite runs it against real Postgres
under the chart's own tracked migrate hook, with the Supabase stub and the
plan-state world from tests/test_plan_enforced_org_features.py.

Pinned, for every plan state (Free, Professional, Enterprise, an Enterprise
trial, missing / canceled / past-due / unknown plans, and app admins):
  * analysis_comments and report_comments: INSERT and UPDATE are refused below
    Professional; SELECT and DELETE of existing rows are unchanged.
  * notifications: a notification to someone else is refused below
    Professional; a notification to yourself is not. A Free user still reads,
    marks read and deletes what is sent to them.
  * the app-admin bypass writes exactly one audit row, and a self-notification
    writes none.

Every refusal is shown to come from the new policy alone: the same statement,
by the same caller, succeeds when only that policy is dropped (in a rolled-back
transaction).

AVALOKA_PLANENFORCE_MIGRATION, when set, replaces the migration in the copied
chart set with that file. The bite driver uses it to run this suite against
mutants; it is never set in CI.

Needs AVALOKA_TEST_PG_ADMIN_URL, psql and sh. Skips on a laptop without them and
FAILS when CI is set.
"""

from __future__ import annotations

import json
import os
import pathlib
import secrets

import pytest

from tests.test_plan_enforced_org_features import (
    ALLOWED_COLLAB,
    LOG,
    MIGRATIONS,
    REPO_ROOT,
    RLS_DENIED,
    TIERS,
    Db,
    Org,
    World,
    _baseline_versions,
    _chart_set,
    _drop,
    _new_db,
    _ok,
    _seed,
    _uuid,
)

pytestmark = pytest.mark.integration

VERSION = "20261103100000_plan_enforce_comments_notifications"
UI_COPY = REPO_ROOT / "ui" / "supabase" / "migrations"
OVERRIDE = os.environ.get("AVALOKA_PLANENFORCE_MIGRATION", "").strip()

P_ADD_ANALYSIS = "pro plan to add analysis comments"
P_CHANGE_ANALYSIS = "pro plan to change analysis comments"
P_ADD_REPORT = "pro plan to add report comments"
P_CHANGE_REPORT = "pro plan to change report comments"
P_NOTIFY = "pro plan to notify others"
POLICIES = {
    P_ADD_ANALYSIS: ("analysis_comments", "INSERT"),
    P_CHANGE_ANALYSIS: ("analysis_comments", "UPDATE"),
    P_ADD_REPORT: ("report_comments", "INSERT"),
    P_CHANGE_REPORT: ("report_comments", "UPDATE"),
    P_NOTIFY: ("notifications", "INSERT"),
}


# ---- the world --------------------------------------------------------------------


class Extra:
    """Per organisation: an analysis and a report owned by the org owner, one
    existing comment on each by the owner, and another person to notify."""

    def __init__(self) -> None:
        self.analysis = _uuid()
        self.report = _uuid()
        self.analysis_comment = _uuid()
        self.report_comment = _uuid()
        self.other_auth = _uuid()
        self.other_profile = _uuid()


def _seed_extra(d: Db, w: World) -> dict[str, Extra]:
    extras = {k: Extra() for k in w.orgs}
    sql = []
    for k, o in w.orgs.items():
        e = extras[k]
        sql += [
            f"INSERT INTO public.analyses (id, owner_id, name) VALUES ('{e.analysis}', '{o.profile}', 'Analysis {k}');",
            f"INSERT INTO public.reports (id, organization_id, analysis_id, title, created_by) "
            f"VALUES ('{e.report}', '{o.org}', '{e.analysis}', 'Report {k}', '{o.auth}');",
            f"INSERT INTO public.analysis_comments (id, analysis_id, content, author_id) "
            f"VALUES ('{e.analysis_comment}', '{e.analysis}', 'existing', '{o.profile}');",
            f"INSERT INTO public.report_comments (id, report_id, author_id, body) "
            f"VALUES ('{e.report_comment}', '{e.report}', '{o.profile}', 'existing');",
            f"INSERT INTO auth.users (id, email, email_confirmed_at) VALUES ('{e.other_auth}', 'other-{k}@example.com', now());",
            f"INSERT INTO public.profiles (id, user_id, full_name, organization_id) VALUES ('{e.other_profile}', '{e.other_auth}', 'Other', NULL);",
        ]
    d.admin("\n".join(sql))
    return extras


@pytest.fixture(scope="module")
def applied(tmp_path_factory):
    d = _new_db()
    try:
        mig = _chart_set(tmp_path_factory.mktemp("planenforce"))
        if OVERRIDE:
            (mig / f"{VERSION}.sql").write_text(pathlib.Path(OVERRIDE).read_text())
        first = _ok(d.run_hook(mig))
        second = _ok(d.run_hook(mig))
        given = mig / f"{VERSION}.sql"
        # What this database was given; empty when the file is absent, so a tree
        # without the migration fails on the assertions below, not on a read.
        d.applied_sql = given.read_text() if given.is_file() else ""
        yield d, first, second
    finally:
        _drop(d)


@pytest.fixture(scope="module")
def world(applied):
    w = _seed(applied[0])
    return w, _seed_extra(applied[0], w)


def _without(d: Db, policy: str, auth: str, sql: str):
    """The same statement by the same caller, with only `policy` dropped, rolled back."""
    table = POLICIES[policy][0]
    script = (
        f"BEGIN;\nDROP POLICY \"{policy}\" ON public.{table};\nSET LOCAL ROLE authenticated;\n"
        f"SET LOCAL \"request.jwt.claim.sub\" = '{auth}';\n"
        "SET LOCAL \"request.jwt.claim.role\" = 'authenticated';\n"
        f"SET LOCAL \"request.jwt.claims\" = '{{\"sub\": \"{auth}\", \"role\": \"authenticated\"}}';\n"
        f"{sql}\nROLLBACK;\n"
    )
    return d.psql(script, check=False)


def _refused_only_by(d: Db, policy: str, o: Org, sql: str, expect_out: str) -> None:
    r = _without(d, policy, o.auth, sql)
    assert r.returncode == 0 and r.stdout.strip() == expect_out, (
        f"{o.key}: without {policy!r} the write should pass the pre-existing policies: {r.stdout}{r.stderr}")


# ---- the migration itself ------------------------------------------------------------


def test_migration_is_new_not_baseline_and_both_copies_are_identical():
    assert VERSION not in _baseline_versions()
    prefix = VERSION.split("_", 1)[0]
    assert prefix > "20261102130000", "must sort after the reconciled #646's last migration"
    for copy in (MIGRATIONS, UI_COPY):
        assert sorted(p.name for p in copy.glob(f"{prefix}_*.sql")) == [f"{VERSION}.sql"]
    assert (UI_COPY / f"{VERSION}.sql").read_bytes() == (MIGRATIONS / f"{VERSION}.sql").read_bytes()


def test_applied_once_by_the_tracked_hook_and_policies_are_restrictive(applied):
    d, first, second = applied
    assert f"applying {VERSION}\n" in first
    assert f"applying {VERSION}" not in second
    got = d.scalar("SELECT string_agg(policyname || ':' || tablename || ':' || permissive || ':' || cmd, ',' "
                   "ORDER BY policyname) FROM pg_policies WHERE policyname LIKE 'pro plan to %' "
                   "AND tablename IN ('analysis_comments', 'report_comments', 'notifications')")
    assert got == ",".join(f"{p}:{t}:RESTRICTIVE:{c}" for p, (t, c) in sorted(POLICIES.items()))


def test_applying_the_sql_again_is_a_no_op(applied):
    d = applied[0]
    q = ("SELECT string_agg(tablename || '.' || policyname || ':' || permissive, ',' ORDER BY tablename, policyname) "
         "FROM pg_policies WHERE schemaname = 'public'")
    t = ("SELECT string_agg(tgrelid::regclass::text || '.' || tgname, ',' ORDER BY tgrelid::regclass::text) "
         "FROM pg_trigger WHERE tgname = 'audit_org_feature_bypass'")
    before = (d.scalar(q), d.scalar(t))
    d.psql(d.applied_sql)
    assert (d.scalar(q), d.scalar(t)) == before


# ---- comments -----------------------------------------------------------------------


@pytest.mark.parametrize("key", TIERS)
def test_analysis_comment_insert(applied, world, key):
    d, (w, ex) = applied[0], world
    o, e = w.orgs[key], ex[key]
    sql = (f"INSERT INTO public.analysis_comments (analysis_id, content, author_id) "
           f"VALUES ('{e.analysis}', 'new by {key}', '{o.profile}') RETURNING 'ok';")
    _refused_only_by(d, P_ADD_ANALYSIS, o, sql, "ok")
    r = d.as_user(o.auth, sql)
    if key in ALLOWED_COLLAB:
        assert r.returncode == 0 and r.stdout.strip() == "ok", f"{key} refused: {r.stderr}"
    else:
        assert r.returncode != 0 and RLS_DENIED in r.stderr, f"{key} commented: {r.stdout}{r.stderr}"


@pytest.mark.parametrize("key", TIERS)
def test_report_comment_insert(applied, world, key):
    d, (w, ex) = applied[0], world
    o, e = w.orgs[key], ex[key]
    sql = (f"INSERT INTO public.report_comments (report_id, author_id, body) "
           f"VALUES ('{e.report}', '{o.profile}', 'new by {key}') RETURNING 'ok';")
    _refused_only_by(d, P_ADD_REPORT, o, sql, "ok")
    r = d.as_user(o.auth, sql)
    if key in ALLOWED_COLLAB:
        assert r.returncode == 0 and r.stdout.strip() == "ok", f"{key} refused: {r.stderr}"
    else:
        assert r.returncode != 0 and RLS_DENIED in r.stderr, f"{key} commented: {r.stdout}{r.stderr}"


@pytest.mark.parametrize("key", TIERS)
@pytest.mark.parametrize("table", ["analysis_comments", "report_comments"])
def test_comment_update(applied, world, key, table):
    d, (w, ex) = applied[0], world
    o, e = w.orgs[key], ex[key]
    rid, col, policy = ((e.analysis_comment, "content", P_CHANGE_ANALYSIS) if table == "analysis_comments"
                        else (e.report_comment, "body", P_CHANGE_REPORT))
    sql = f"WITH u AS (UPDATE public.{table} SET {col} = 'edited by {key}' WHERE id = '{rid}' RETURNING 1) SELECT count(*) FROM u;"
    _refused_only_by(d, policy, o, sql, "1")
    r = d.as_user(o.auth, sql)
    assert r.returncode == 0, r.stderr
    now = d.scalar(f"SELECT {col} FROM public.{table} WHERE id = '{rid}'")
    if key in ALLOWED_COLLAB:
        assert (r.stdout.strip(), now) == ("1", f"edited by {key}")
    else:
        assert (r.stdout.strip(), now) == ("0", "existing"), f"{key} edited a comment"


@pytest.mark.parametrize("key", ["free", "canceled", "missing"])
def test_existing_comments_stay_readable_and_deletable_below_pro(applied, world, key):
    d, (w, ex) = applied[0], world
    o, e = w.orgs[key], ex[key]
    for table, rid in (("analysis_comments", e.analysis_comment), ("report_comments", e.report_comment)):
        r = d.as_user(o.auth, f"SELECT count(*) FROM public.{table} WHERE id = '{rid}';")
        assert r.stdout.strip() == "1", f"{key}: {table} not readable: {r.stderr}"
    r = d.as_user(o.auth, f"WITH x AS (DELETE FROM public.analysis_comments WHERE id = '{e.analysis_comment}' "
                          "RETURNING 1) SELECT count(*) FROM x;")
    assert r.stdout.strip() == "1", f"{key}: own analysis comment not deletable: {r.stderr}"


# ---- notifications ---------------------------------------------------------------


def _notify(o: Org, recipient: str, tag: str) -> str:
    return (f"INSERT INTO public.notifications (recipient_id, actor_id, type, title) "
            f"VALUES ('{recipient}', '{o.profile}', 'generic', '{tag}') RETURNING 'ok';")


@pytest.mark.parametrize("key", TIERS)
def test_notifying_someone_else(applied, world, key):
    d, (w, ex) = applied[0], world
    o, e = w.orgs[key], ex[key]
    sql = _notify(o, e.other_profile, f"to other by {key}")
    _refused_only_by(d, P_NOTIFY, o, sql, "ok")
    r = d.as_user(o.auth, sql)
    if key in ALLOWED_COLLAB:
        assert r.returncode == 0 and r.stdout.strip() == "ok", f"{key} refused: {r.stderr}"
    else:
        assert r.returncode != 0 and RLS_DENIED in r.stderr, f"{key} notified someone else: {r.stdout}{r.stderr}"


@pytest.mark.parametrize("key", TIERS)
def test_notifying_yourself_is_open_on_every_plan(applied, world, key):
    d, (w, _) = applied[0], world
    o = w.orgs[key]
    r = d.as_user(o.auth, _notify(o, o.profile, f"to self by {key}"))
    assert r.returncode == 0 and r.stdout.strip() == "ok", f"{key}: self-notification refused: {r.stderr}"


def test_a_free_user_still_receives_reads_and_clears_notifications(applied, world):
    """What a Free account depends on: an invitation from a Professional user, and
    the backend's completed / failed notices (service role)."""
    d, (w, ex) = applied[0], world
    pro, free = w.orgs["professional"], w.orgs["free"]
    r = d.as_user(pro.auth, _notify(pro, free.profile, "collaborator.invited"))
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    nid = _uuid()
    d.psql(f"BEGIN; SET LOCAL ROLE service_role; INSERT INTO public.notifications (id, recipient_id, type, title) "
           f"VALUES ('{nid}', '{free.profile}', 'analysis.completed', 'done'); COMMIT;")
    r = d.as_user(free.auth, "SELECT string_agg(title, ',' ORDER BY title) FROM public.notifications "
                             f"WHERE recipient_id = '{free.profile}' AND title IN ('collaborator.invited', 'done');")
    assert r.stdout.strip() == "collaborator.invited,done", r.stderr
    r = d.as_user(free.auth, f"WITH u AS (UPDATE public.notifications SET read_at = now() WHERE id = '{nid}' "
                             "RETURNING 1) SELECT count(*) FROM u;")
    assert r.stdout.strip() == "1", f"Free cannot mark its notification read: {r.stderr}"
    r = d.as_user(free.auth, f"WITH x AS (DELETE FROM public.notifications WHERE id = '{nid}' RETURNING 1) SELECT count(*) FROM x;")
    assert r.stdout.strip() == "1", f"Free cannot delete its notification: {r.stderr}"


# ---- audit of the app-admin bypass -------------------------------------------------


def _log(d: Db, actor: str) -> list[dict]:
    out = d.scalar(f"SELECT coalesce(string_agg(row_to_json(l)::text, E'\\n' ORDER BY id), '') FROM {LOG} l "
                   f"WHERE actor_user_id = '{actor}'")
    return [json.loads(ln) for ln in out.splitlines() if ln]


@pytest.mark.parametrize("action", ["analysis_comments", "report_comments", "notifications"])
def test_admin_bypass_is_audited_once_without_customer_data(applied, world, action):
    d, (w, ex) = applied[0], world
    o, e = w.orgs["admin_on_free"], ex["admin_on_free"]
    marker = f"Secret-{secrets.token_hex(4)}"
    stmt = {
        "analysis_comments": f"INSERT INTO public.analysis_comments (analysis_id, content, author_id) "
                             f"VALUES ('{e.analysis}', '{marker}', '{o.profile}') RETURNING 'ok';",
        "report_comments": f"INSERT INTO public.report_comments (report_id, author_id, body) "
                           f"VALUES ('{e.report}', '{o.profile}', '{marker}') RETURNING 'ok';",
        "notifications": _notify(o, e.other_profile, marker),
    }[action]
    before = len(_log(d, o.auth))
    r = d.as_user(o.auth, stmt)
    assert r.returncode == 0 and r.stdout.strip() == "ok", f"bypass write refused: {r.stderr}"
    rows = _log(d, o.auth)
    assert len(rows) == before + 1, f"expected one new audit row, got {len(rows) - before}"
    row = rows[-1]
    assert (row["source"], row["action"], row["operation"]) == ("rls", action, "INSERT")
    assert marker not in json.dumps(row) and "@example.com" not in json.dumps(row)


def test_a_self_notification_by_an_admin_is_not_logged_as_a_bypass(applied, world):
    d, (w, _) = applied[0], world
    o = w.orgs["admin_on_free"]
    before = len(_log(d, o.auth))
    r = d.as_user(o.auth, _notify(o, o.profile, "self"))
    assert r.returncode == 0, r.stderr
    assert len(_log(d, o.auth)) == before


@pytest.mark.parametrize("key", ["professional", "enterprise"])
def test_a_paying_caller_is_never_logged(applied, world, key):
    d, (w, ex) = applied[0], world
    o, e = w.orgs[key], ex[key]
    before = len(_log(d, o.auth))
    r = d.as_user(o.auth, _notify(o, e.other_profile, "paid"))
    assert r.returncode == 0, r.stderr
    assert len(_log(d, o.auth)) == before
