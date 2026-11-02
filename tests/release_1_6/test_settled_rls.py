"""1.6 release suite: enterprise rows settled after the owner's answers, checked
against the Supabase migrations a self-hosted install applies.

Reuses the harness in tests/release_1_6/test_supabase_rls.py (replay of
deploy/helm/avaloka/files/supabase-migrations onto a throwaway Postgres with a
minimal Supabase stub; queries run as a signed-in user under RLS).

Marker ``integration``: needs ``AVALOKA_TEST_POSTGRES_URL`` (an admin URL).
"""
from __future__ import annotations

import pytest

from tests.release_1_6.test_supabase_rls import _uuid, rls_db, world  # noqa: F401  (fixtures)

pytestmark = pytest.mark.integration


def test_newbg02_project_creator_sees_own_project(rls_db, world):
    """NEW-BG-02: an organization member who is not the org owner creates a project
    the way the UI does (insert with owner_id = own profile, then read it back) and
    can see it, including in the project list query (deleted_at IS NULL); and so can
    a signed-in user with no organization, whose only path is the owner policies.

    Settled from the bug's expected behaviour ("whoever created a project should see
    it") and ui/src/lib/projects.ts (createProjectWithAnalysis / fetchProjects); the
    policy under test is "owners read own projects" (owner_id = current_profile_id()).
    """
    assert rls_db.has_policy("projects", "owners read own projects")
    # A member of the organization also sees the project through the org-wide
    # policies (and a trigger stamps the creator's org onto the row), so the creator's
    # own access is isolated with a signed-in user who has no organization: only the
    # owner policies can show them their project.
    solo_auth, solo_profile = _uuid(), _uuid()
    with rls_db.admin() as cur:
        cur.execute("SET session_replication_role = replica")
        cur.execute("INSERT INTO auth.users (id, email) VALUES (%s, %s)", (solo_auth, "solo@example.com"))
        cur.execute("INSERT INTO public.profiles (id, user_id, full_name, organization_id) VALUES (%s, %s, %s, NULL)",
                    (solo_profile, solo_auth, "Jane Doe Solo"))
        cur.execute("SET session_replication_role = origin")

    for auth, email, profile, org in (
        (world.auth("B"), world.email("B"), world.profile("B"), world.org),
        (solo_auth, "solo@example.com", solo_profile, None),
    ):
        pid = _uuid()
        with rls_db.as_user(auth, email) as cur:
            cur.execute(
                "INSERT INTO public.projects (id, name, owner_id, organization_id) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (pid, "Member project", profile, org),
            )
            assert str(cur.fetchone()[0]) == pid
        with rls_db.as_user(auth, email) as cur:
            cur.execute("SELECT id, name FROM public.projects WHERE deleted_at IS NULL AND id = %s", (pid,))
            assert [(str(r[0]), r[1]) for r in cur.fetchall()] == [(pid, "Member project")], f"org={org}"


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "NEW-BG-05: app_users.role and app_users.department are free-text columns "
    "(migration 20260617080245) with no foreign key to config_roles / departments, and the "
    "UI copies role permissions onto each user, so a role change never reaches its users"))
def test_newbg05_app_users_reference_roles_and_departments_by_key(rls_db):
    """NEW-BG-05: a user's role and department reference the config_roles and
    departments rows by key, so a change to a role's permissions applies to every
    user holding it (the bug's stated expectation)."""
    with rls_db.admin() as cur:
        cur.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='public' "
            "AND table_name IN ('app_users','config_roles','departments')"
        )
        assert cur.fetchone()[0] == 3, "precondition: the three tables exist after the replay"
        cur.execute(
            """
            SELECT DISTINCT ccu.table_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name AND tc.constraint_schema = ccu.constraint_schema
            WHERE tc.table_schema = 'public' AND tc.table_name = 'app_users'
              AND tc.constraint_type = 'FOREIGN KEY'
            """
        )
        referenced = {r[0] for r in cur.fetchall()}
    assert {"config_roles", "departments"} <= referenced
