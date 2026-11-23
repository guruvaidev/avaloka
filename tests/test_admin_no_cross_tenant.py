"""D-2: the app 'admin' role reaches no other tenant's customer data, and no user
reads another tenant's profile rows (migration 20261103120000_admin_no_cross_tenant).

Against real Postgres, with the chart's own migration set applied by the chart's
own migrate hook (deploy/helm/avaloka/files/db-migrate.sh). Two databases are
built from the same seed:

  * base   every chart migration dated before this one, i.e. develop-1.6 before
           the fix;
  * fixed  every chart migration.

Each leak test asserts the leak on base first and its absence on fixed, so the
test proves both that the hole is real and that this migration is what closes it
(dropping or weakening any part of the migration turns the fixed half red).

The world:
  org A  Enterprise. owner_a, member_a, and admin_a (user_roles 'admin', the
         ADMIN_EMAILS / support-desk role, an ordinary member of A).
  org B  Enterprise. owner_b, with a report, its analysis and project, a comment,
         a support ticket, and org structure (department, team, roster rows).
  org C  Free, owned by collab_c, who is invited onto B's report as a viewer (its
         one roster row is in B, as the invite flow writes it).
  loner  no organisation.

Needs AVALOKA_TEST_PG_ADMIN_URL, psql and sh; skips without them on a laptop and
FAILS when CI is set (the shared harness in test_plan_enforced_org_features).
"""

from __future__ import annotations

import pathlib
import shutil
import uuid
from dataclasses import dataclass, field

import pytest

from tests.test_plan_enforced_org_features import (
    BASELINE,
    MIGRATIONS,
    REPO_ROOT,
    Db,
    _baseline_versions,
    _drop,
    _new_db,
    _ok,
)

pytestmark = pytest.mark.integration

VERSION = "20261103120000_admin_no_cross_tenant"
RLS_DENIED = "row-level security"


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class Person:
    key: str
    org: str | None
    auth: str = field(default_factory=_uuid)
    profile: str = field(default_factory=_uuid)
    app_user: str = field(default_factory=_uuid)

    @property
    def email(self) -> str:
        return f"{self.key}@example.com"


@dataclass
class World:
    org_a: str = field(default_factory=_uuid)
    org_b: str = field(default_factory=_uuid)
    org_c: str = field(default_factory=_uuid)
    plan_ent: str = field(default_factory=_uuid)
    plan_free: str = field(default_factory=_uuid)
    report_b: str = field(default_factory=_uuid)
    analysis_b: str = field(default_factory=_uuid)
    project_b: str = field(default_factory=_uuid)
    comment_b: str = field(default_factory=_uuid)
    ticket_b: str = field(default_factory=_uuid)
    dept_b: str = field(default_factory=_uuid)
    team_b: str = field(default_factory=_uuid)
    collab_row: str = field(default_factory=_uuid)
    pending_b: str = field(default_factory=_uuid)   # an invite row in B, no account linked yet
    people: dict[str, Person] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, org in (("owner_a", self.org_a), ("member_a", self.org_a), ("admin_a", self.org_a),
                         ("owner_b", self.org_b), ("collab_c", self.org_c), ("loner", None)):
            self.people[key] = Person(key, org)

    def __getitem__(self, key: str) -> Person:
        return self.people[key]


WORLD = World()


def _seed(d: Db, w: World) -> None:
    p = w.people
    sql = [
        f"INSERT INTO public.plans (id, plan_type, name) VALUES ('{w.plan_ent}', 'enterprise', 'Enterprise'), "
        f"('{w.plan_free}', 'free', 'Free');",
    ]
    for person in p.values():
        org = f"'{person.org}'" if person.org else "NULL"
        sql += [
            f"INSERT INTO auth.users (id, email, email_confirmed_at) VALUES ('{person.auth}', '{person.email}', now());",
            f"INSERT INTO public.profiles (id, user_id, full_name, phone, company_email, organization_id) VALUES "
            f"('{person.profile}', '{person.auth}', 'Name {person.key}', '+1555{person.key}', "
            f"'company-{person.email}', {org});",
        ]
    owners = {w.org_a: p["owner_a"], w.org_b: p["owner_b"], w.org_c: p["collab_c"]}
    for org, owner in owners.items():
        sql.append(f"INSERT INTO public.organizations (id, name, owner_profile_id) VALUES "
                   f"('{org}', 'Org {owner.key}', '{owner.profile}');")
    # app_users.auth_user_id is unique: collab_c's one roster row is the invite in B
    # below; it owns org C through organizations.owner_profile_id.
    for key in ("owner_a", "member_a", "admin_a", "owner_b"):
        person = p[key]
        role = "owner" if key.startswith("owner") else "member"
        sql.append(f"INSERT INTO public.app_users (id, organization_id, email, auth_user_id, profile_id, role, status) VALUES "
                   f"('{person.app_user}', '{person.org}', '{person.email}', '{person.auth}', '{person.profile}', '{role}', 'Active');")
    collab_in_b = _uuid()
    sql += [
        f"INSERT INTO public.subscriptions (owner_profile_id, organization_id, plan_id, payment_provider, status) VALUES "
        f"('{p['owner_a'].profile}', '{w.org_a}', '{w.plan_ent}', 'stripe', 'active'), "
        f"('{p['owner_b'].profile}', '{w.org_b}', '{w.plan_ent}', 'stripe', 'active'), "
        f"('{p['collab_c'].profile}', '{w.org_c}', '{w.plan_free}', 'stripe', 'active');",
        f"INSERT INTO public.user_roles (user_id, role) VALUES ('{p['admin_a'].auth}', 'admin');",
        # B's resources.
        f"INSERT INTO public.projects (id, owner_id, name, organization_id) VALUES "
        f"('{w.project_b}', '{p['owner_b'].profile}', 'Project B', '{w.org_b}');",
        f"INSERT INTO public.analyses (id, owner_id, name, organization_id, project_id) VALUES "
        f"('{w.analysis_b}', '{p['owner_b'].profile}', 'Analysis B', '{w.org_b}', '{w.project_b}');",
        f"INSERT INTO public.reports (id, organization_id, project_id, analysis_id, title, created_by) VALUES "
        f"('{w.report_b}', '{w.org_b}', '{w.project_b}', '{w.analysis_b}', 'Report B', '{p['owner_b'].auth}');",
        f"INSERT INTO public.report_comments (id, report_id, author_id, body) VALUES "
        f"('{w.comment_b}', '{w.report_b}', '{p['owner_b'].profile}', 'comment by owner b');",
        f"INSERT INTO public.support_tickets (id, user_id, subject, message) VALUES "
        f"('{w.ticket_b}', '{p['owner_b'].auth}', 'help', 'ticket from owner b');",
        f"INSERT INTO public.departments (id, organization_id, department, code) VALUES "
        f"('{w.dept_b}', '{w.org_b}', 'Finance B', 'FINB');",
        f"INSERT INTO public.organization_teams (id, organization_id, name, branch) VALUES "
        f"('{w.team_b}', '{w.org_b}', 'Team B', 'HQ');",
        # collab_c is invited onto B's report: a roster row in B, as the invite flow writes it.
        f"INSERT INTO public.app_users (id, organization_id, email, auth_user_id, role, status) VALUES "
        f"('{collab_in_b}', '{w.org_b}', '{p['collab_c'].email}', '{p['collab_c'].auth}', 'member', 'Active');",
        f"INSERT INTO public.resource_collaborators (id, resource_type, resource_id, app_user_id, access_level, invited_by) VALUES "
        f"('{w.collab_row}', 'report', '{w.report_b}', '{collab_in_b}', 'view', '{p['owner_b'].auth}');",
        # A pending invite in B whose address the loner will name as its company_email.
        f"INSERT INTO public.app_users (id, organization_id, email, role, status) VALUES "
        f"('{w.pending_b}', '{w.org_b}', 'pending-b@example.com', 'member', 'Invited');",
    ]
    r = d.psql("SET session_replication_role = replica;\n" + "\n".join(sql), check=False)
    assert r.returncode == 0, f"seed failed: {r.stderr[-2000:]}"


def _build(tmp: pathlib.Path, *, with_fix: bool) -> Db:
    mig = tmp / "migrations"
    shutil.copytree(MIGRATIONS, mig)
    shutil.copyfile(BASELINE, mig / "baseline.txt")   # as the chart mounts it
    if not with_fix:
        # develop-1.6 before this fix: drop it and every migration dated after it,
        # since later fixes (e.g. #646's BG-01) close some of the same holes.
        for f in mig.glob("2*.sql"):
            if f.stem >= VERSION:
                f.unlink()
    d = _new_db()
    out = _ok(d.run_hook(mig))
    assert (f"applying {VERSION}\n" in out) == with_fix
    _seed(d, WORLD)
    return d


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = _build(tmp_path_factory.mktemp("noxtenant_base"), with_fix=False)
    try:
        yield d
    finally:
        _drop(d)


@pytest.fixture(scope="module")
def fixed(tmp_path_factory):
    d = _build(tmp_path_factory.mktemp("noxtenant_fixed"), with_fix=True)
    try:
        yield d
    finally:
        _drop(d)


def _rows(d: Db, who: str, sql: str) -> list[str]:
    r = d.as_user(WORLD[who].auth, sql)
    assert r.returncode == 0, f"{who}: {r.stderr}"
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def _one(d: Db, who: str, sql: str) -> str:
    rows = _rows(d, who, sql)
    assert len(rows) == 1, rows
    return rows[0]


# ---- the migration under tracking ------------------------------------------------


def test_migration_is_new_not_baseline_and_ships_in_both_copies():
    assert VERSION not in _baseline_versions(), "a new migration must not be in the frozen baseline"
    ui = (REPO_ROOT / "ui" / "supabase" / "migrations" / f"{VERSION}.sql").read_text()
    assert ui == (MIGRATIONS / f"{VERSION}.sql").read_text(), "the UI and chart copies differ"


def test_it_runs_after_the_migrations_it_rewrites():
    """It redefines objects #649, #650 and #661 also touch, and must sort after the
    newest migration on develop-1.6 (20261103090000), which `supabase db push`
    requires on CLI-managed projects."""
    must_follow = [f.stem for f in MIGRATIONS.glob("2026110*.sql")
                   if f.stem.startswith(("20261101", "20261103090000"))]
    assert must_follow and all(VERSION > v for v in must_follow), must_follow


def test_applying_the_sql_again_is_a_no_op(fixed):
    snapshot = ("SELECT string_agg(tablename || '.' || policyname || ':' || cmd || ':' || coalesce(qual, '') "
                "|| ':' || coalesce(with_check, ''), E'\\n' ORDER BY tablename, policyname) FROM pg_policies "
                "WHERE schemaname = 'public'")
    before = fixed.scalar(snapshot)
    fixed.psql((MIGRATIONS / f"{VERSION}.sql").read_text())
    assert fixed.scalar(snapshot) == before


# ---- C1-C4: another tenant's report --------------------------------------------------

REPORT = f"SELECT id FROM public.reports WHERE id = '{WORLD.report_b}';"
COMMENTS = f"SELECT id FROM public.report_comments WHERE report_id = '{WORLD.report_b}';"
COLLABS = f"SELECT id FROM public.resource_collaborators WHERE id = '{WORLD.collab_row}';"
ACCESS = f"SELECT public.get_my_report_access('{WORLD.report_b}');"
TEAMS = f"SELECT member_id FROM public.get_report_teams(ARRAY['{WORLD.report_b}']::uuid[]);"


@pytest.mark.parametrize("what,sql", [("report", REPORT), ("comments", COMMENTS), ("collaborators", COLLABS)],
                         ids=["report", "comments", "collaborators"])
def test_admin_reads_another_tenants_report_data_on_base_and_not_after(base, fixed, what, sql):
    assert _rows(base, "admin_a", sql), f"base: the admin could not read B's {what}; the test proves nothing"
    assert _rows(fixed, "admin_a", sql) == [], f"the admin still reads org B's {what}"


TEAMS_SRC = "SELECT pg_get_functiondef('public.get_report_teams(uuid[])'::regprocedure);"


def _base_report_teams_is_broken_or_leaks(base: Db, who: str, sql: str) -> bool:
    """On develop-1.6 a chart install's get_report_teams() fails on every call
    ("column reference "email" is ambiguous": its RETURNS TABLE column and an
    unqualified app_users.email collide). Where it runs (a hosted project may
    differ), it leaks; either way its source carries the admin term."""
    assert "has_role(auth.uid(), 'admin')" in base.scalar(TEAMS_SRC)
    r = base.as_user(WORLD[who].auth, sql)
    return r.returncode != 0 and 'column reference "email" is ambiguous' in r.stderr or bool(r.stdout.strip())


def test_admin_reads_another_tenants_report_teams_on_base_and_not_after(base, fixed):
    assert _base_report_teams_is_broken_or_leaks(base, "admin_a", TEAMS)
    assert "has_role" not in fixed.scalar(TEAMS_SRC)
    assert _rows(fixed, "admin_a", TEAMS) == [], "the admin still reads org B's report team"


def test_report_teams_work_again_for_owners_and_collaborators(fixed):
    """The redefinition also qualifies every column, so the function runs."""
    assert len(_rows(fixed, "owner_b", TEAMS)) == 2       # owner_b (creator) and collab_c
    assert len(_rows(fixed, "collab_c", TEAMS)) == 2


def test_admin_gets_edit_on_another_tenants_report_on_base_and_not_after(base, fixed):
    assert _one(base, "admin_a", ACCESS) == "edit"
    assert _one(fixed, "admin_a", ACCESS) == "none"


def test_a_plain_member_of_another_org_never_could(base, fixed):
    for d in (base, fixed):
        assert _rows(d, "member_a", REPORT) == []


# ---- owners and collaborators are unaffected -----------------------------------------


def test_owner_keeps_its_report_comments_collaborators_and_edit(fixed):
    assert _rows(fixed, "owner_b", REPORT) == [WORLD.report_b]
    assert _rows(fixed, "owner_b", COMMENTS) == [WORLD.comment_b]
    assert _rows(fixed, "owner_b", COLLABS) == [WORLD.collab_row]
    assert _one(fixed, "owner_b", ACCESS) == "edit"


def test_collaborator_keeps_view_access_and_the_comments(fixed):
    assert _rows(fixed, "collab_c", REPORT) == [WORLD.report_b]
    assert _one(fixed, "collab_c", ACCESS) == "view"
    assert _rows(fixed, "collab_c", COMMENTS) == [WORLD.comment_b]
    assert _rows(fixed, "collab_c", COLLABS) == [WORLD.collab_row]   # its own collaboration row


# ---- C4: report teams carry no email across tenants -------------------------------------

TEAM_EMAILS = (f"SELECT coalesce(email, '<null>') || '|' || name FROM public.get_report_teams("
               f"ARRAY['{WORLD.report_b}']::uuid[]) ORDER BY 1;")


def test_report_teams_show_a_collaborator_no_email_from_another_tenant(base, fixed):
    assert _base_report_teams_is_broken_or_leaks(base, "collab_c", TEAM_EMAILS)
    rows = _rows(fixed, "collab_c", TEAM_EMAILS)
    assert rows, "the collaborator lost the report's team list"
    assert all(r.split("|")[0] == "<null>" for r in rows), rows
    assert not any("@" in r.split("|")[1] for r in rows), f"an email came back as a name: {rows}"


def test_report_teams_still_show_the_owner_its_own_roster_emails(fixed):
    rows = _rows(fixed, "owner_b", TEAM_EMAILS)
    assert any(r.startswith(WORLD["collab_c"].email + "|") for r in rows), rows


# ---- C5: an admin moving its own profile into another org ------------------------------

MOVE = f"UPDATE public.profiles SET organization_id = '{WORLD.org_b}' WHERE user_id = '{WORLD['admin_a'].auth}';"
AFTER_MOVE = (f"{MOVE}\nSELECT public.current_org_id()::text || '|' || public.is_org_admin()::text || '|' || "
              f"(SELECT count(*) FROM public.departments WHERE organization_id = '{WORLD.org_b}')::text;")


def test_admin_org_move_works_on_base_and_is_refused_after(base, fixed):
    r = base.as_user(WORLD["admin_a"].auth, AFTER_MOVE)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == f"{WORLD.org_b}|true|1", \
        f"base: the move did not hand org B to the admin: {r.stdout}"
    base.admin(f"UPDATE public.profiles SET organization_id = '{WORLD.org_a}' WHERE user_id = '{WORLD['admin_a'].auth}';")
    r = fixed.as_user(WORLD["admin_a"].auth, AFTER_MOVE)
    assert r.returncode != 0 and "organization_id cannot be modified" in r.stderr, r.stdout + r.stderr
    assert fixed.scalar(f"SELECT organization_id FROM public.profiles WHERE user_id = '{WORLD['admin_a'].auth}'") == WORLD.org_a


def test_no_signed_in_user_moves_its_profile(fixed):
    r = fixed.as_user(WORLD["member_a"].auth,
                      f"UPDATE public.profiles SET organization_id = '{WORLD.org_b}' WHERE user_id = '{WORLD['member_a'].auth}';")
    assert r.returncode != 0 and "organization_id cannot be modified" in r.stderr


def test_server_side_code_still_moves_profiles(fixed):
    """The invite-accept and upgrade flows link a profile with the service role."""
    loner = WORLD["loner"]
    r = fixed.psql(
        "BEGIN; SET LOCAL ROLE service_role;\n"
        "SET LOCAL \"request.jwt.claims\" = '{\"role\": \"service_role\"}';\n"
        f"UPDATE public.profiles SET organization_id = '{WORLD.org_c}' WHERE id = '{loner.profile}';\n"
        f"UPDATE public.profiles SET organization_id = NULL WHERE id = '{loner.profile}';\nCOMMIT;", check=False)
    assert r.returncode == 0, r.stderr


# ---- C6: an admin granting roles --------------------------------------------------------

GRANT = f"INSERT INTO public.user_roles (user_id, role) VALUES ('{WORLD['member_a'].auth}', 'admin');"


def test_admin_grants_admin_on_base_and_cannot_after(base, fixed):
    r = base.as_user(WORLD["admin_a"].auth, GRANT)
    assert r.returncode == 0, f"base: the admin could not grant; the test proves nothing: {r.stderr}"
    base.admin(f"DELETE FROM public.user_roles WHERE user_id = '{WORLD['member_a'].auth}';")
    r = fixed.as_user(WORLD["admin_a"].auth, GRANT)
    assert r.returncode != 0 and RLS_DENIED in r.stderr, r.stdout + r.stderr
    assert fixed.scalar(f"SELECT count(*) FROM public.user_roles WHERE user_id = '{WORLD['member_a'].auth}'") == "0"


def test_admin_cannot_revoke_another_admin(fixed):
    other = _uuid()
    fixed.admin(f"INSERT INTO public.user_roles (user_id, role) VALUES ('{other}', 'admin');")
    _rows(fixed, "admin_a", f"DELETE FROM public.user_roles WHERE user_id = '{other}';")
    assert fixed.scalar(f"SELECT count(*) FROM public.user_roles WHERE user_id = '{other}'") == "1"


def test_users_still_read_their_own_roles(fixed):
    assert _rows(fixed, "admin_a", "SELECT role FROM public.user_roles;") == ["admin"]


# ---- C13: claiming an org through company_email ------------------------------------------


def test_company_email_claims_an_org_on_base_and_not_after(base, fixed):
    loner = WORLD["loner"]
    for d in (base, fixed):
        d.admin(f"UPDATE public.profiles SET company_email = 'pending-b@example.com' WHERE id = '{loner.profile}';")
    assert _one(base, "loner", "SELECT coalesce(public.current_org_id()::text, '<none>');") == WORLD.org_b
    assert _one(fixed, "loner", "SELECT coalesce(public.current_org_id()::text, '<none>');") == "<none>"


def test_members_keep_their_org(fixed):
    for key, org in (("owner_a", WORLD.org_a), ("member_a", WORLD.org_a), ("owner_b", WORLD.org_b)):
        assert _one(fixed, key, "SELECT public.current_org_id();") == org


def test_a_roster_row_without_a_profile_org_still_resolves(fixed):
    """Fallback by auth link (not by an address the user wrote)."""
    p = Person("rostered", None)
    fixed.admin(
        f"INSERT INTO auth.users (id, email) VALUES ('{p.auth}', '{p.email}');"
        f"INSERT INTO public.profiles (id, user_id, full_name) VALUES ('{p.profile}', '{p.auth}', 'R');"
        f"INSERT INTO public.app_users (id, organization_id, email, auth_user_id, role, status) VALUES "
        f"('{p.app_user}', '{WORLD.org_a}', '{p.email}', '{p.auth}', 'member', 'Active');")
    r = fixed.as_user(p.auth, "SELECT public.current_org_id();")
    assert r.stdout.strip() == WORLD.org_a, r.stderr


# ---- profiles -------------------------------------------------------------------------------

PROFILE_OF = "SELECT id FROM public.profiles WHERE user_id = '{auth}';"


def test_a_user_reads_another_tenants_profile_on_base_and_not_after(base, fixed):
    sql = PROFILE_OF.format(auth=WORLD["owner_b"].auth)
    assert _rows(base, "member_a", sql), "base: profiles were not world-readable; the test proves nothing"
    for who in ("member_a", "owner_a", "admin_a", "collab_c", "loner"):
        assert _rows(fixed, who, sql) == [], f"{who} reads org B's owner's profile row"


def test_nobody_reads_phone_or_company_email_of_another_tenant(fixed):
    for who in ("member_a", "admin_a", "collab_c", "loner"):
        assert _rows(fixed, who, "SELECT phone FROM public.profiles WHERE phone LIKE '%owner_b%';") == []


def test_everyone_reads_their_own_profile(fixed):
    for key, person in WORLD.people.items():
        assert _rows(fixed, key, "SELECT id FROM public.profiles;").count(person.profile) == 1, key


def test_an_org_admin_still_reads_its_own_orgs_profiles(fixed):
    """Kept on purpose ("Admins read org profiles", own org only)."""
    rows = set(_rows(fixed, "admin_a", "SELECT id FROM public.profiles;"))
    assert {WORLD["owner_a"].profile, WORLD["member_a"].profile} <= rows
    assert WORLD["owner_b"].profile not in rows


DISPLAY = "SELECT id || '|' || coalesce(full_name, '') FROM public.get_display_profiles(ARRAY['{ids}']::uuid[]);"


def test_display_fields_for_own_org(fixed):
    rows = _rows(fixed, "member_a", DISPLAY.format(ids=WORLD["owner_a"].auth))
    assert rows == [f"{WORLD['owner_a'].profile}|Name owner_a"]


def test_display_fields_for_a_report_collaborator_across_tenants(fixed):
    # The collaborator sees the report's creator; the creator sees the collaborator.
    assert _rows(fixed, "collab_c", DISPLAY.format(ids=WORLD["owner_b"].auth)) == \
        [f"{WORLD['owner_b'].profile}|Name owner_b"]
    assert _rows(fixed, "owner_b", DISPLAY.format(ids=WORLD["collab_c"].auth)) == \
        [f"{WORLD['collab_c'].profile}|Name collab_c"]


def test_display_fields_refuse_unrelated_people(fixed):
    for who in ("member_a", "admin_a", "loner"):
        assert _rows(fixed, who, DISPLAY.format(ids=WORLD["owner_b"].auth)) == [], who


def test_display_profiles_return_no_contact_or_billing_columns(fixed):
    cols = fixed.scalar("SELECT string_agg(a, ',' ORDER BY n) FROM unnest(("
                        "SELECT proargnames FROM pg_proc WHERE proname = 'get_display_profiles')) WITH ORDINALITY AS t(a, n)")
    assert cols == "p_ids,id,user_id,full_name,first_name,last_name,avatar_url", cols


def test_display_profiles_are_not_callable_anonymously(fixed):
    r = fixed.psql("BEGIN; SET LOCAL ROLE anon; SELECT * FROM public.get_display_profiles(ARRAY[]::uuid[]); COMMIT;",
                   check=False)
    assert r.returncode != 0 and "permission denied" in r.stderr


# ---- support tickets stay with support ---------------------------------------------------------

TICKETS = f"SELECT id FROM public.support_tickets WHERE id = '{WORLD.ticket_b}';"


def test_support_admin_still_reads_tickets(base, fixed):
    for d in (base, fixed):
        assert _rows(d, "admin_a", TICKETS) == [WORLD.ticket_b]
        assert _rows(d, "member_a", TICKETS) == []


# ---- the #650 plan rules still hold -------------------------------------------------------------


def test_plan_rules_still_hold(fixed):
    free = fixed.as_user(WORLD["collab_c"].auth,
                         f"INSERT INTO public.organization_teams (organization_id, name, branch) "
                         f"VALUES ('{WORLD.org_c}', 'free team', 'HQ');")
    assert free.returncode != 0 and RLS_DENIED in free.stderr, free.stdout + free.stderr
    ent = fixed.as_user(WORLD["owner_b"].auth,
                        f"INSERT INTO public.organization_teams (organization_id, name, branch) "
                        f"VALUES ('{WORLD.org_b}', 'enterprise team', 'HQ') RETURNING name;")
    assert ent.returncode == 0 and ent.stdout.strip() == "enterprise team", ent.stderr
    # Org structure stays Enterprise-only (#650, revised by #661).
    free = fixed.as_user(WORLD["collab_c"].auth,
                         f"INSERT INTO public.departments (organization_id, department, code) "
                         f"VALUES ('{WORLD.org_c}', 'free dept', 'FD');")
    assert free.returncode != 0 and RLS_DENIED in free.stderr, free.stdout + free.stderr
    ent = fixed.as_user(WORLD["owner_b"].auth,
                        f"INSERT INTO public.departments (organization_id, department, code) "
                        f"VALUES ('{WORLD.org_b}', 'enterprise dept', 'ED') RETURNING department;")
    assert ent.returncode == 0 and ent.stdout.strip() == "enterprise dept", ent.stderr


def test_admin_cannot_write_another_tenants_org_structure(fixed):
    _rows(fixed, "admin_a", f"UPDATE public.departments SET department = 'hijacked' WHERE id = '{WORLD.dept_b}';")
    _rows(fixed, "admin_a", f"UPDATE public.organization_teams SET name = 'hijacked' WHERE id = '{WORLD.team_b}';")
    assert fixed.scalar(f"SELECT department FROM public.departments WHERE id = '{WORLD.dept_b}'") == "Finance B"
    assert fixed.scalar(f"SELECT name FROM public.organization_teams WHERE id = '{WORLD.team_b}'") == "Team B"
