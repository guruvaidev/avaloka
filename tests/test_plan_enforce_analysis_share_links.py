"""Analysis share links need a Professional plan or higher; report links do not.

Owner ruling 2026-11-02. Migration 20261103110000_plan_enforce_analysis_share_links.sql
adds RESTRICTIVE INSERT and UPDATE policies on resource_share_links that apply
only to resource_type = 'analysis'. Run against real Postgres under the chart's
tracked migrate hook, with the harness and plan-state world of
tests/test_plan_enforced_org_features.py.

Pinned, for every plan state:
  * creating or changing an analysis link is refused below Professional, and
    each refusal comes from the new policy alone (the same statement passes
    with only that policy dropped, rolled back);
  * report and project links are unaffected on every plan;
  * an analysis link a Free user created BEFORE the migration still resolves
    afterwards (the resolver's service-role read by token), stays readable to
    its owner, and can still be revoked; no row is removed;
  * the app-admin bypass on an analysis link writes one audit row; a report
    link by the same admin writes none.

AVALOKA_PLANENFORCE_SHARE_MIGRATION, when set, replaces the migration in the
copied chart set with that file (the bite driver uses it; never set in CI).

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

VERSION = "20261103110000_plan_enforce_analysis_share_links"
PREVIOUS = "20261103100000_plan_enforce_comments_notifications"
UI_COPY = REPO_ROOT / "ui" / "supabase" / "migrations"
OVERRIDE = os.environ.get("AVALOKA_PLANENFORCE_SHARE_MIGRATION", "").strip()

P_ADD = "pro plan to add analysis share links"
P_CHANGE = "pro plan to change analysis share links"


class Extra:
    def __init__(self) -> None:
        self.analysis = _uuid()
        self.report = _uuid()
        self.analysis_link = _uuid()   # an existing analysis link, created by the seed
        self.report_link = _uuid()     # an existing report link


def _seed_extra(d: Db, w: World, *, links: bool = True) -> dict[str, Extra]:
    extras = {k: Extra() for k in w.orgs}
    sql = []
    for k, o in w.orgs.items():
        e = extras[k]
        sql += [
            f"INSERT INTO public.analyses (id, owner_id, name) VALUES ('{e.analysis}', '{o.profile}', 'Analysis {k}');",
            f"INSERT INTO public.reports (id, organization_id, analysis_id, title, created_by) "
            f"VALUES ('{e.report}', '{o.org}', '{e.analysis}', 'Report {k}', '{o.auth}');",
        ]
        if links:
            sql += [
                f"INSERT INTO public.resource_share_links (id, resource_type, resource_id) VALUES ('{e.analysis_link}', 'analysis', '{e.analysis}');",
                f"INSERT INTO public.resource_share_links (id, resource_type, resource_id) VALUES ('{e.report_link}', 'report', '{e.report}');",
            ]
    d.admin("\n".join(sql))
    return extras


def _migrate(d: Db, mig: pathlib.Path) -> str:
    if OVERRIDE:
        (mig / f"{VERSION}.sql").write_text(pathlib.Path(OVERRIDE).read_text())
    return _ok(d.run_hook(mig))


@pytest.fixture(scope="module")
def applied(tmp_path_factory):
    d = _new_db()
    try:
        mig = _chart_set(tmp_path_factory.mktemp("sharelinks"))
        first = _migrate(d, mig)
        second = _ok(d.run_hook(mig))
        given = mig / f"{VERSION}.sql"
        d.applied_sql = given.read_text() if given.is_file() else ""
        yield d, first, second
    finally:
        _drop(d)


@pytest.fixture(scope="module")
def world(applied):
    w = _seed(applied[0])
    return w, _seed_extra(applied[0], w)


def _without(d: Db, policy: str, auth: str, sql: str):
    script = (
        f"BEGIN;\nDROP POLICY \"{policy}\" ON public.resource_share_links;\nSET LOCAL ROLE authenticated;\n"
        f"SET LOCAL \"request.jwt.claim.sub\" = '{auth}';\n"
        "SET LOCAL \"request.jwt.claim.role\" = 'authenticated';\n"
        f"SET LOCAL \"request.jwt.claims\" = '{{\"sub\": \"{auth}\", \"role\": \"authenticated\"}}';\n"
        f"{sql}\nROLLBACK;\n"
    )
    return d.psql(script, check=False)


def _passes_without(d: Db, policy: str, o: Org, sql: str, out: str) -> None:
    r = _without(d, policy, o.auth, sql)
    assert r.returncode == 0 and r.stdout.strip() == out, (
        f"{o.key}: without {policy!r} the write should pass the pre-existing policies: {r.stdout}{r.stderr}")


# ---- the migration ------------------------------------------------------------------


def test_migration_is_new_not_baseline_after_the_previous_one_and_copies_identical():
    assert VERSION not in _baseline_versions()
    prefix = VERSION.split("_", 1)[0]
    assert prefix > PREVIOUS.split("_", 1)[0] > "20261102130000"
    for copy in (MIGRATIONS, UI_COPY):
        assert sorted(p.name for p in copy.glob(f"{prefix}_*.sql")) == [f"{VERSION}.sql"]
    assert (UI_COPY / f"{VERSION}.sql").read_bytes() == (MIGRATIONS / f"{VERSION}.sql").read_bytes()


def test_applied_once_and_policies_are_restrictive(applied):
    d, first, second = applied
    assert f"applying {VERSION}\n" in first
    assert f"applying {VERSION}" not in second
    got = d.scalar("SELECT string_agg(policyname || ':' || permissive || ':' || cmd, ',' ORDER BY policyname) "
                   "FROM pg_policies WHERE tablename = 'resource_share_links' AND policyname LIKE 'pro plan to %'")
    assert got == f"{P_ADD}:RESTRICTIVE:INSERT,{P_CHANGE}:RESTRICTIVE:UPDATE"


def test_applying_the_sql_again_is_a_no_op(applied):
    d = applied[0]
    q = ("SELECT string_agg(policyname || ':' || permissive || ':' || coalesce(qual, '') || ':' || coalesce(with_check, ''), ',' "
         "ORDER BY policyname) FROM pg_policies WHERE tablename = 'resource_share_links'")
    t = "SELECT count(*) FROM pg_trigger WHERE tgrelid = 'public.resource_share_links'::regclass AND NOT tgisinternal"
    before = (d.scalar(q), d.scalar(t))
    d.psql(d.applied_sql)
    assert (d.scalar(q), d.scalar(t)) == before


# ---- analysis links by plan ----------------------------------------------------------


@pytest.mark.parametrize("key", TIERS)
def test_creating_an_analysis_link(applied, world, key):
    d, (w, ex) = applied[0], world
    o, e = w.orgs[key], ex[key]
    new_analysis = _uuid()
    d.admin(f"INSERT INTO public.analyses (id, owner_id, name) VALUES ('{new_analysis}', '{o.profile}', 'Fresh {key}');")
    sql = (f"INSERT INTO public.resource_share_links (resource_type, resource_id) "
           f"VALUES ('analysis', '{new_analysis}') RETURNING 'ok';")
    _passes_without(d, P_ADD, o, sql, "ok")
    r = d.as_user(o.auth, sql)
    if key in ALLOWED_COLLAB:
        assert r.returncode == 0 and r.stdout.strip() == "ok", f"{key} refused: {r.stderr}"
    else:
        assert r.returncode != 0 and RLS_DENIED in r.stderr, f"{key} created an analysis link: {r.stdout}{r.stderr}"


@pytest.mark.parametrize("key", TIERS)
def test_changing_an_analysis_link(applied, world, key):
    d, (w, ex) = applied[0], world
    o, e = w.orgs[key], ex[key]
    d.admin(f"UPDATE public.resource_share_links SET is_public = false WHERE id = '{e.analysis_link}';")
    sql = (f"WITH u AS (UPDATE public.resource_share_links SET is_public = true WHERE id = '{e.analysis_link}' "
           "RETURNING 1) SELECT count(*) FROM u;")
    _passes_without(d, P_CHANGE, o, sql, "1")
    r = d.as_user(o.auth, sql)
    assert r.returncode == 0, r.stderr
    now = d.scalar(f"SELECT is_public FROM public.resource_share_links WHERE id = '{e.analysis_link}'")
    if key in ALLOWED_COLLAB:
        assert (r.stdout.strip(), now) == ("1", "t")
    else:
        assert (r.stdout.strip(), now) == ("0", "f"), f"{key} changed an analysis link"


@pytest.mark.parametrize("key", ["free", "canceled", "missing"])
def test_a_report_link_cannot_be_turned_into_an_analysis_link_below_pro(applied, world, key):
    d, (w, ex) = applied[0], world
    o, e = w.orgs[key], ex[key]
    r = d.as_user(o.auth, f"UPDATE public.resource_share_links SET resource_type = 'analysis', resource_id = '{e.analysis}' "
                          f"WHERE id = '{e.report_link}';")
    assert d.scalar(f"SELECT resource_type FROM public.resource_share_links WHERE id = '{e.report_link}'") == "report", \
        f"{key}: {r.stdout}{r.stderr}"


# ---- report and project links stay open ---------------------------------------------


@pytest.mark.parametrize("key", TIERS)
def test_report_links_are_open_on_every_plan(applied, world, key):
    d, (w, ex) = applied[0], world
    o, e = w.orgs[key], ex[key]
    new_report = _uuid()
    d.admin(f"INSERT INTO public.reports (id, organization_id, analysis_id, title, created_by) "
            f"VALUES ('{new_report}', '{o.org}', '{e.analysis}', 'Fresh report', '{o.auth}');")
    r = d.as_user(o.auth, f"INSERT INTO public.resource_share_links (resource_type, resource_id) "
                          f"VALUES ('report', '{new_report}') RETURNING 'ok';")
    assert r.returncode == 0 and r.stdout.strip() == "ok", f"{key}: report link refused: {r.stderr}"
    r = d.as_user(o.auth, f"WITH u AS (UPDATE public.resource_share_links SET is_public = NOT is_public "
                          f"WHERE id = '{e.report_link}' RETURNING 1) SELECT count(*) FROM u;")
    assert r.returncode == 0 and r.stdout.strip() == "1", f"{key}: report link change refused: {r.stderr}"


@pytest.mark.parametrize("key", ["free", "professional"])
def test_project_links_are_unaffected(applied, world, key):
    d, (w, _) = applied[0], world
    o = w.orgs[key]
    pid = _uuid()
    d.admin(f"INSERT INTO public.projects (id, owner_id, name, organization_id) VALUES ('{pid}', '{o.profile}', 'P', '{o.org}');")
    r = d.as_user(o.auth, f"INSERT INTO public.resource_share_links (resource_type, resource_id) "
                          f"VALUES ('project', '{pid}') RETURNING 'ok';")
    assert r.returncode == 0 and r.stdout.strip() == "ok", f"{key}: project link refused: {r.stderr}"


# ---- links that already exist ---------------------------------------------------------


def test_an_analysis_link_a_free_user_made_before_the_migration_still_resolves(tmp_path):
    """Migrate without the new file, let a Free owner create an analysis link as
    themselves, then apply the new migration and resolve that link."""
    d = _new_db()
    try:
        mig = _chart_set(tmp_path)
        held = mig / f"{VERSION}.sql"
        if OVERRIDE:
            held.write_text(pathlib.Path(OVERRIDE).read_text())
        text = held.read_text() if held.is_file() else ""
        if held.is_file():
            held.unlink()
        out = _ok(d.run_hook(mig))
        assert f"applying {VERSION}" not in out
        w = _seed(d)
        ex = _seed_extra(d, w, links=False)
        free, e = w.orgs["free"], ex["free"]
        r = d.as_user(free.auth, f"INSERT INTO public.resource_share_links (resource_type, resource_id, is_public) "
                                 f"VALUES ('analysis', '{e.analysis}', true) RETURNING share_token;")
        assert r.returncode == 0, f"before the migration a Free owner could create an analysis link: {r.stderr}"
        token = r.stdout.strip()
        rows_before = d.scalar("SELECT count(*) FROM public.resource_share_links")

        held.write_text(text)
        out = _ok(d.run_hook(mig))
        assert f"applying {VERSION}\n" in out, "the new migration did not run on the existing database"
        assert d.scalar("SELECT count(*) FROM public.resource_share_links") == rows_before, "rows were removed"

        # What /api/shared/$token does: read the link by token with the service role.
        r = d.psql("BEGIN; SET LOCAL ROLE service_role; SELECT resource_type || '|' || resource_id || '|' || is_public "
                   f"FROM public.resource_share_links WHERE share_token = '{token}'; COMMIT;", check=False)
        assert r.returncode == 0 and r.stdout.strip() == f"analysis|{e.analysis}|true", r.stderr
        # Its owner still sees it, and can still revoke it.
        r = d.as_user(free.auth, f"SELECT count(*) FROM public.resource_share_links WHERE share_token = '{token}';")
        assert r.stdout.strip() == "1", r.stderr
        r = d.as_user(free.auth, f"WITH x AS (DELETE FROM public.resource_share_links WHERE share_token = '{token}' "
                                 "RETURNING 1) SELECT count(*) FROM x;")
        assert r.stdout.strip() == "1", f"Free owner cannot revoke the old link: {r.stderr}"
    finally:
        _drop(d)


# ---- audit of the app-admin bypass ------------------------------------------------------


def _log(d: Db, actor: str) -> list[dict]:
    out = d.scalar(f"SELECT coalesce(string_agg(row_to_json(l)::text, E'\\n' ORDER BY id), '') FROM {LOG} l "
                   f"WHERE actor_user_id = '{actor}'")
    return [json.loads(ln) for ln in out.splitlines() if ln]


def test_admin_bypass_on_an_analysis_link_is_audited_once(applied, world):
    d, (w, _) = applied[0], world
    o = w.orgs["admin_on_free"]
    aid = _uuid()
    marker = f"Secret-{secrets.token_hex(4)}"
    d.admin(f"INSERT INTO public.analyses (id, owner_id, name) VALUES ('{aid}', '{o.profile}', '{marker}');")
    before = len(_log(d, o.auth))
    r = d.as_user(o.auth, f"INSERT INTO public.resource_share_links (resource_type, resource_id) "
                          f"VALUES ('analysis', '{aid}') RETURNING 'ok';")
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    rows = _log(d, o.auth)
    assert len(rows) == before + 1, f"expected one new audit row, got {len(rows) - before}"
    assert (rows[-1]["source"], rows[-1]["action"], rows[-1]["operation"]) == ("rls", "resource_share_links", "INSERT")
    assert marker not in json.dumps(rows[-1])


def test_a_report_link_by_an_admin_is_not_a_bypass(applied, world):
    d, (w, ex) = applied[0], world
    o, e = w.orgs["admin_on_free"], ex["admin_on_free"]
    rid = _uuid()
    d.admin(f"INSERT INTO public.reports (id, organization_id, analysis_id, title, created_by) "
            f"VALUES ('{rid}', '{o.org}', '{e.analysis}', 'R', '{o.auth}');")
    before = len(_log(d, o.auth))
    r = d.as_user(o.auth, f"INSERT INTO public.resource_share_links (resource_type, resource_id) "
                          f"VALUES ('report', '{rid}') RETURNING 'ok';")
    assert r.returncode == 0, r.stderr
    assert len(_log(d, o.auth)) == before
