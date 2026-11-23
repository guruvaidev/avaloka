"""#646's database fixes (BG-01, BG-03, BG-05, BG-06), as reconciled
with migration tracking, #650 and D-2.

Real Postgres, the chart's own migration set and migrate hook. Two databases from
one seed: "base" is every chart migration except these three (develop-1.6 with
D-2), "fixed" is every chart migration. Each behaviour test asserts the old
behaviour on base first, then the new one on fixed. An existing-install test runs
the pre-tracking hook over the baseline, seeds data the migrations must cope with
(duplicate department names, a Free organisation with department text), then the
tracked hook.

Needs AVALOKA_TEST_PG_ADMIN_URL, psql and sh (shared harness in
test_plan_enforced_org_features): skips on a laptop without them, FAILS under CI.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import uuid

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

D2 = "20261103120000_admin_no_cross_tenant"
BG03 = "20261103130000_fix_bg03_analysis_visibility"
BG01 = "20261103140000_fix_bg01_org_membership"
BG0506 = "20261103150000_bg05_bg06_roles_departments"
OURS = (BG03, BG01, BG0506)


def _uuid() -> str:
    return str(uuid.uuid4())


class W:
    """The seed: Enterprise org E (owner, Active member, admin), Free org F."""
    plan_ent, plan_free = _uuid(), _uuid()
    org_e, org_f = _uuid(), _uuid()
    owner_e, member_e, invitee_e, admin_e, owner_f, drifter = (
        {"auth": _uuid(), "profile": _uuid(), "app_user": _uuid(), "key": k}
        for k in ("owner_e", "member_e", "invitee_e", "admin_e", "owner_f", "drifter"))
    project_e, analysis_proj, analysis_loose = _uuid(), _uuid(), _uuid()


def _seed(d: Db) -> None:
    w = W
    people = [(w.owner_e, w.org_e), (w.member_e, w.org_e), (w.invitee_e, None), (w.admin_e, w.org_e),
              (w.owner_f, w.org_f), (w.drifter, None)]
    sql = [f"INSERT INTO public.plans (id, plan_type, name) VALUES ('{w.plan_ent}', 'enterprise', 'Enterprise'), "
           f"('{w.plan_free}', 'free', 'Free');"]
    for p, org in people:
        o = f"'{org}'" if org else "NULL"
        sql += [f"INSERT INTO auth.users (id, email, email_confirmed_at) VALUES ('{p['auth']}', '{p['key']}@example.com', now());",
                f"INSERT INTO public.profiles (id, user_id, full_name, organization_id) VALUES "
                f"('{p['profile']}', '{p['auth']}', '{p['key']}', {o});"]
    sql += [
        f"INSERT INTO public.organizations (id, name, owner_profile_id) VALUES ('{w.org_e}', 'E', '{w.owner_e['profile']}'),"
        f" ('{w.org_f}', 'F', '{w.owner_f['profile']}');",
        f"INSERT INTO public.subscriptions (owner_profile_id, organization_id, plan_id, payment_provider, status) VALUES "
        f"('{w.owner_e['profile']}', '{w.org_e}', '{w.plan_ent}', 'stripe', 'active'),"
        f" ('{w.owner_f['profile']}', '{w.org_f}', '{w.plan_free}', 'stripe', 'active');",
        f"INSERT INTO public.app_users (id, organization_id, email, auth_user_id, profile_id, role, status) VALUES "
        f"('{w.owner_e['app_user']}', '{w.org_e}', 'owner_e@example.com', '{w.owner_e['auth']}', '{w.owner_e['profile']}', 'owner', 'Active'),"
        f" ('{w.member_e['app_user']}', '{w.org_e}', 'member_e@example.com', '{w.member_e['auth']}', '{w.member_e['profile']}', 'member', 'Active'),"
        f" ('{w.admin_e['app_user']}', '{w.org_e}', 'admin_e@example.com', '{w.admin_e['auth']}', '{w.admin_e['profile']}', 'member', 'Active'),"
        # An invite that was never accepted, already linked to the account (get_my_report_access backfills that link).
        f" ('{w.invitee_e['app_user']}', '{w.org_e}', 'invitee_e@example.com', '{w.invitee_e['auth']}', NULL, 'member', 'Invited');",
        f"INSERT INTO public.user_roles (user_id, role) VALUES ('{w.admin_e['auth']}', 'admin');",
        f"INSERT INTO public.projects (id, owner_id, name, organization_id) VALUES "
        f"('{w.project_e}', '{w.owner_e['profile']}', 'P', '{w.org_e}');",
        f"INSERT INTO public.analyses (id, owner_id, name, organization_id, project_id) VALUES "
        f"('{w.analysis_proj}', '{w.owner_e['profile']}', 'in a project', '{w.org_e}', '{w.project_e}'),"
        f" ('{w.analysis_loose}', '{w.owner_e['profile']}', 'personal', '{w.org_e}', NULL);",
    ]
    r = d.psql("SET session_replication_role = replica;\n" + "\n".join(sql), check=False)
    assert r.returncode == 0, f"seed failed: {r.stderr[-2000:]}"


def _chart(tmp: pathlib.Path, *, ours: bool) -> pathlib.Path:
    mig = tmp / "migrations"
    shutil.copytree(MIGRATIONS, mig)
    shutil.copyfile(BASELINE, mig / "baseline.txt")
    if not ours:
        # Before these fixes: drop them and every migration dated after them.
        for f in mig.glob("2*.sql"):
            if f.stem >= OURS[0]:
                f.unlink()
    return mig


def _build(tmp: pathlib.Path, *, ours: bool) -> Db:
    d = _new_db()
    out = _ok(d.run_hook(_chart(tmp, ours=ours)))
    for v in OURS:
        assert (f"applying {v}\n" in out) == ours, v
    _seed(d)
    return d


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = _build(tmp_path_factory.mktemp("migrec_base"), ours=False)
    try:
        yield d
    finally:
        _drop(d)


@pytest.fixture(scope="module")
def fixed(tmp_path_factory):
    d = _build(tmp_path_factory.mktemp("migrec_fixed"), ours=True)
    try:
        yield d
    finally:
        _drop(d)


def _q(d: Db, who: dict, sql: str) -> list[str]:
    r = d.as_user(who["auth"], sql)
    assert r.returncode == 0, f"{who['key']}: {r.stderr}"
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


ORG = "SELECT coalesce(public.current_org_id()::text, '<none>');"


# ---- the migrations under tracking ------------------------------------------------


@pytest.mark.parametrize("version", OURS)
def test_new_not_baseline_both_copies_no_inner_transaction(version):
    assert version not in _baseline_versions()
    ui = (REPO_ROOT / "ui" / "supabase" / "migrations" / f"{version}.sql").read_text()
    assert ui == (MIGRATIONS / f"{version}.sql").read_text(), "the UI and chart copies differ"
    assert not re.search(r"(?im)^\s*(BEGIN|COMMIT|ROLLBACK)\s*;", ui), \
        "the hook wraps each file in one transaction with its record; an inner COMMIT ends it early"


#: The 2026110* migrations on develop-1.6 when these were renumbered to follow them.
#: Comparing with every 2026110* file made each later migration a failure here.
FOLLOWS = (
    "20261101120000_plan_enforced_org_features", "20261101120000_support_admin_from_roles",
    "20261101140000_plan_rules_pro_collab", "20261102100000_owner_only_project_visibility",
    "20261103090000_restore_support_ticket_policies", "20261103100000_plan_enforce_comments_notifications",
    "20261103110000_plan_enforce_analysis_share_links", D2, "20261103120000_bg01_no_org_for_self_serve",
)
# Merged later, to run between BG-01 and BG-05/06: the unique indexes need it first.
DEDUPE = "20261103145000_dedupe_seed_departments_roles"
# The unique indexes for every organisation without duplicates (owner's ruling).
PER_ORG = "20261117110000_org_name_keys_per_org"


def test_they_run_after_the_migrations_they_follow():
    shipped = {f.stem for f in MIGRATIONS.glob("*.sql")}
    assert set(FOLLOWS) <= shipped, sorted(set(FOLLOWS) - shipped)
    assert all(v > o for v in OURS for o in FOLLOWS)
    assert list(OURS) == sorted(OURS)
    assert DEDUPE in shipped and BG01 < DEDUPE < BG0506


def test_the_old_versions_are_gone():
    names = {f.name for f in MIGRATIONS.parent.glob("*/*.sql")} | {f.name for f in MIGRATIONS.glob("*.sql")}
    for old in ("20261101100000_fix_bg03_analysis_visibility.sql", "20261101110000_fix_bg01_org_membership.sql",
                "20261101120000_bg05_bg06_roles_departments.sql",
                "20261102110000_fix_bg03_analysis_visibility.sql", "20261102120000_fix_bg01_org_membership.sql",
                "20261102130000_bg05_bg06_roles_departments.sql"):
        assert old not in names
        assert not (REPO_ROOT / "ui" / "supabase" / "migrations" / old).exists()


# ---- BG-01: current_org_id() needs membership --------------------------------------


def test_a_profile_org_without_membership_is_honoured_on_base_and_not_after(base, fixed):
    """A profile pointing at E without owning it or an Active row there (moved by
    any path other than the user, which D-2 already refuses)."""
    for d in (base, fixed):
        d.psql(f"SET session_replication_role = replica; UPDATE public.profiles SET organization_id = '{W.org_e}' "
               f"WHERE id = '{W.drifter['profile']}';")
    assert _q(base, W.drifter, ORG) == [W.org_e]
    assert _q(fixed, W.drifter, ORG) == ["<none>"]


def test_an_unaccepted_invite_gives_org_context_on_base_and_not_after(base, fixed):
    assert _q(base, W.invitee_e, ORG) == [W.org_e]
    assert _q(fixed, W.invitee_e, ORG) == ["<none>"]


def test_owners_and_active_members_keep_their_org(fixed):
    for who, org in ((W.owner_e, W.org_e), (W.member_e, W.org_e), (W.admin_e, W.org_e), (W.owner_f, W.org_f)):
        assert _q(fixed, who, ORG) == [org], who["key"]


def test_the_admin_org_move_stays_refused(fixed):
    r = fixed.as_user(W.admin_e["auth"], f"UPDATE public.profiles SET organization_id = '{W.org_f}' "
                                         f"WHERE user_id = '{W.admin_e['auth']}';")
    assert r.returncode != 0 and "organization_id cannot be modified" in r.stderr


# ---- BG-03: org-wide analysis visibility is project-scoped ----------------------------

ANALYSES = "SELECT name FROM public.analyses ORDER BY name;"


def test_a_member_sees_personal_analyses_of_others_on_base_and_not_after(base, fixed):
    assert _q(base, W.member_e, ANALYSES) == ["in a project", "personal"]
    assert _q(fixed, W.member_e, ANALYSES) == ["in a project"]


def test_the_owner_keeps_its_own_analyses(fixed):
    assert _q(fixed, W.owner_e, ANALYSES) == ["in a project", "personal"]


def test_another_org_sees_none(fixed):
    assert _q(fixed, W.owner_f, ANALYSES) == []


def test_api_roles_cannot_truncate(base, fixed):
    probe = "SELECT has_table_privilege('authenticated', 'public.analyses', 'TRUNCATE');"
    assert base.scalar(probe) == "t", "base: authenticated could not truncate; the test proves nothing"
    assert fixed.scalar(probe) == "f"
    granted = ("SELECT coalesce(string_agg(DISTINCT c.relname, ',' ORDER BY c.relname), '') FROM pg_class c "
               "JOIN pg_namespace n ON n.oid = c.relnamespace, aclexplode(c.relacl) a WHERE n.nspname = 'public' "
               "AND c.relkind = 'r' AND a.privilege_type = 'TRUNCATE' "
               "AND a.grantee IN ('anon'::regrole, 'authenticated'::regrole)")
    assert "analyses" in base.scalar(granted).split(",")
    # integration_connections.sql (undated) sorts after every dated migration, so on
    # a fresh install its table is created, with grants, after the REVOKE ran. A
    # known gap, listed in the PR; every other table is covered.
    assert set(filter(None, fixed.scalar(granted).split(","))) <= {"integration_connections"}


# ---- BG-05/06 on a fresh install ----------------------------------------------------------


def test_links_and_indexes_exist(fixed):
    assert fixed.scalar("SELECT count(*) FROM information_schema.columns WHERE table_schema = 'public' AND "
                        "(table_name, column_name) IN (('app_users', 'department_id'), ('app_users', 'role_id'), "
                        "('config_roles', 'department_id'))") == "3"
    assert fixed.scalar("SELECT count(*) FROM pg_indexes WHERE indexname IN "
                        "('departments_org_name_key', 'config_roles_org_role_dept_key')") == "2"


def test_the_sync_trigger_fires_before_any_enforce_trigger(fixed):
    names = fixed.scalar("SELECT string_agg(tgname, ',' ORDER BY tgname) FROM pg_trigger "
                         "WHERE tgrelid = 'public.app_users'::regclass AND NOT tgisinternal").split(",")
    assert "app_users_sync_role_department" in names and "trg_app_users_sync_role_department" not in names
    assert all("app_users_sync_role_department" < n for n in names if n.startswith("enforce_"))
    assert "app_users_sync_role_department" < "enforce_app_users_plan"   # the #650 revision's name


def test_every_function_has_an_empty_search_path(fixed):
    rows = fixed.scalar(
        "SELECT string_agg(proname || '=' || coalesce(array_to_string(proconfig, ';'), '<none>'), ',' ORDER BY proname) "
        "FROM pg_proc WHERE pronamespace = 'public'::regnamespace AND proname IN ('role_permissions_jsonb', "
        "'config_roles_resolve_department', 'app_users_sync_role_department', 'config_roles_propagate_to_users', "
        "'departments_propagate_rename', 'current_org_id')")
    for item in rows.split(","):
        assert item.endswith('=search_path=""'), item


def _role(d: Db, org: str, role: str, dept: str, perms: str) -> str:
    rid = _uuid()
    d.psql(f"INSERT INTO public.departments (organization_id, department, code) VALUES ('{org}', '{dept}', 'X') "
           f"ON CONFLICT DO NOTHING;"
           f"INSERT INTO public.config_roles (id, organization_id, role, department, permissions) VALUES "
           f"('{rid}', '{org}', '{role}', '{dept}', '{perms}');")
    return rid


def test_a_team_only_update_is_not_rewritten(fixed):
    rid = _role(fixed, W.org_e, "Analyst", "Sales", "{reports:view}")
    fixed.psql(f"UPDATE public.app_users SET role_id = '{rid}' WHERE id = '{W.member_e['app_user']}';")
    assert fixed.scalar(f"SELECT permissions::text FROM public.app_users WHERE id = '{W.member_e['app_user']}'") \
        == '{"reports": {"View": true}}'
    # Drift the stored permissions with triggers off, then change only team_id.
    fixed.psql("SET session_replication_role = replica; UPDATE public.app_users SET permissions = '{\"x\": 1}' "
               f"WHERE id = '{W.member_e['app_user']}';")
    fixed.psql(f"UPDATE public.app_users SET team_id = NULL, updated_at = now() WHERE id = '{W.member_e['app_user']}';")
    assert fixed.scalar(f"SELECT permissions::text FROM public.app_users WHERE id = '{W.member_e['app_user']}'") \
        == '{"x": 1}', "a team-only update rewrote the user's permissions"


def test_a_role_change_normalises_and_a_foreign_role_is_refused(fixed):
    rid = _role(fixed, W.org_e, "Lead", "Ops", "{reports:edit}")
    fixed.psql(f"UPDATE public.app_users SET role_id = '{rid}' WHERE id = '{W.owner_e['app_user']}';")
    assert fixed.scalar(f"SELECT role || '|' || department FROM public.app_users WHERE id = '{W.owner_e['app_user']}'") \
        == "Lead|Ops"
    foreign = _role(fixed, W.org_f, "Lead", "Ops", "{}")
    r = fixed.psql(f"UPDATE public.app_users SET role_id = '{foreign}' WHERE id = '{W.owner_e['app_user']}';", check=False)
    assert r.returncode != 0 and "does not belong to organization" in r.stderr


# ---- BG-05/06 on an existing install ---------------------------------------------------------


def test_existing_install_with_duplicates_and_a_free_org(tmp_path):
    d = _new_db()
    try:
        mig = _chart(tmp_path, ours=True)
        d.run_old_hook_on_baseline(mig)
        _seed(d)
        dup = _uuid()
        d.psql(
            "SET session_replication_role = replica;"
            f"UPDATE public.app_users SET department = 'Sales', role = 'Analyst', permissions = '{{\"reports\": {{\"View\": true}}}}' "
            f"WHERE organization_id = '{W.org_e}';"
            f"INSERT INTO public.app_users (organization_id, email, role, department, status) VALUES "
            f"('{W.org_f}', 'free-member@example.com', 'Clerk', 'Ops', 'Active');"
            # Two departments that differ only in case, in another organisation.
            f"INSERT INTO public.organizations (id, name, owner_profile_id) VALUES ('{dup}', 'D', '{W.drifter['profile']}');"
            f"INSERT INTO public.departments (organization_id, department, code) VALUES "
            f"('{dup}', 'Finance', 'F1'), ('{dup}', 'finance', 'F2');")
        r = d.run_hook(mig)
        out = _ok(r)
        for v in OURS:
            assert f"applying {v}\n" in out, v
        # The dedupe keeps duplicates that differ and goes on; BG-05/06 still skips the
        # index while they exist.
        assert "departments still have duplicate names that differ (code) in 1 organisation(s)" in out
        assert "departments_org_name_key not created" in out
        # Re-pinned for the owner's ruling (warn and continue, skip the index only for
        # the organisations with duplicates): the index used to be absent for every
        # organisation here; 20261117110000 now creates it for all but D.
        assert "departments_org_name_key created without 1 organisation(s)" in out
        assert d.scalar("SELECT count(*) FROM pg_indexes WHERE indexname = 'departments_org_name_key'") == "1"
        assert d.scalar("SELECT count(*) FROM pg_indexes WHERE indexname = 'config_roles_org_role_dept_key'") == "1"
        # Per organisation, not global: E (no duplicates) is held to unique names ...
        r = d.psql(f"INSERT INTO public.departments (organization_id, department, code) VALUES ('{W.org_e}', 'SALES', 'S2');",
                   check=False)
        assert r.returncode != 0 and 'unique constraint "departments_org_name_key"' in r.stderr, r.stdout + r.stderr
        # ... while D keeps its two departments until an admin merges them.
        assert d.scalar(f"SELECT count(*) FROM public.departments WHERE organization_id = '{dup}'") == "2"
        r = d.psql(f"INSERT INTO public.departments (organization_id, department, code) VALUES ('{dup}', 'FINANCE', 'F3');",
                   check=False)
        assert r.returncode == 0, r.stderr
        # Enterprise E is backfilled and linked; Free F gets no org structure.
        assert d.scalar(f"SELECT count(*) FROM public.departments WHERE organization_id = '{W.org_e}' AND department = 'Sales'") == "1"
        assert d.scalar(f"SELECT count(*) FROM public.config_roles WHERE organization_id = '{W.org_e}' AND role = 'Analyst'") == "1"
        assert d.scalar(f"SELECT count(*) FROM public.app_users WHERE organization_id = '{W.org_e}' AND role_id IS NULL "
                        "AND department = 'Sales'") == "0"
        assert d.scalar(f"SELECT count(*) FROM public.departments WHERE organization_id = '{W.org_f}'") == "0"
        assert d.scalar(f"SELECT count(*) FROM public.config_roles WHERE organization_id = '{W.org_f}'") == "0"
        # Applied once: a second run applies nothing.
        out = _ok(d.run_hook(mig))
        assert "migrations complete: 0 applied," in out
        # The runbook (docs/deployment.md): once an admin has merged D's duplicates,
        # re-applying the file gives every organisation the index.
        full = ("SELECT count(*) FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                "WHERE c.relname = 'departments_org_name_key' AND i.indpred IS NULL")
        assert d.scalar(full) == "0"
        d.psql(f"DELETE FROM public.departments WHERE organization_id = '{dup}' AND department <> 'Finance';")
        d.psql((mig / f"{PER_ORG}.sql").read_text())
        assert d.scalar(full) == "1"
    finally:
        _drop(d)


def test_existing_install_with_the_doubled_seed_upgrades(tmp_path):
    """The organisation seed that ran twice: the same department and roles, twice.

    The dedupe (DEDUPE) removes the copies before BG-05/06 creates its unique
    indexes. While the pre-renumbering BG-05/06 also shipped, the hook ran it
    first, in its old position, and the upgrade stopped on the first index.
    """
    d = _new_db()
    try:
        mig = _chart(tmp_path, ours=True)
        d.run_old_hook_on_baseline(mig)
        _seed(d)
        rows = []
        for ms in (0, 7):
            at = f"now() - interval '1 day' + interval '{ms} milliseconds'"
            rows.append(f"INSERT INTO public.departments (organization_id, department, code, created_at) VALUES "
                        f"('{W.org_e}', 'Management', 'MGMT', {at});")
            rows.append(f"INSERT INTO public.config_roles (organization_id, role, department, permissions, created_at) VALUES "
                        f"('{W.org_e}', 'Admin', 'Management', '{{reports:edit}}', {at}), "
                        f"('{W.org_e}', 'Analyst', 'Management', '{{reports:view}}', {at});")
        d.psql("SET session_replication_role = replica;" + "".join(rows))
        out = _ok(d.run_hook(mig))
        assert f"applying {DEDUPE}\n" in out and f"applying {BG0506}\n" in out
        assert out.index(f"applying {DEDUPE}\n") < out.index(f"applying {BG0506}\n")
        assert "not created" not in out
        assert d.scalar("SELECT count(*) FROM pg_indexes WHERE indexname IN "
                        "('departments_org_name_key', 'config_roles_org_role_dept_key')") == "2"
        assert d.scalar(f"SELECT count(*) FROM public.departments WHERE organization_id = '{W.org_e}' "
                        "AND department = 'Management'") == "1"
        assert d.scalar(f"SELECT string_agg(role, ',' ORDER BY role) FROM public.config_roles "
                        f"WHERE organization_id = '{W.org_e}' AND department = 'Management'") == "Admin,Analyst"
        out = _ok(d.run_hook(mig))
        assert "migrations complete: 0 applied," in out
    finally:
        _drop(d)
