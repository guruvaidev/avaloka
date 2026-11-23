"""Who may delete a project (migration 20261013130000_fix_project_delete_and_report_access).

Before that migration, the policy "owners and collaborators delete projects" let ANY
collaborator on a project delete it, a viewer included. 20261013130000 drops it and
creates "owners and editors delete projects": the owner, or a collaborator whose access
is 'edit' (public.is_resource_editor), and nobody else.

Against real Postgres, with the chart's own migration set applied by the chart's own
migrate hook, queried as a signed-in user (the shared harness in
test_plan_enforced_org_features). Each case deletes its own project, so the cases do
not depend on each other's order.

  org X  owner_x (owns every project), editor_x ('edit' collaborator), viewer_x ('view'
         collaborator), each with a roster row in X.
  org Y  outsider_y, unrelated to X's projects.

The viewer's refusal is only meaningful if the viewer can see the project (otherwise RLS
hides the row and no DELETE policy is consulted), so that is asserted first. The
outsider cannot see it at all, so its refusal is held by the SELECT policies as well.

Needs AVALOKA_TEST_PG_ADMIN_URL, psql and sh; skips without them on a laptop and FAILS
when CI is set.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import uuid
from dataclasses import dataclass, field

import pytest

from tests.test_plan_enforced_org_features import BASELINE, MIGRATIONS, Db, _drop, _new_db, _ok

pytestmark = pytest.mark.integration

VERSION = "20261013130000_fix_project_delete_and_report_access"
POLICY = "owners and editors delete projects"


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class Person:
    key: str
    org: str
    auth: str = field(default_factory=_uuid)
    profile: str = field(default_factory=_uuid)
    app_user: str = field(default_factory=_uuid)


@dataclass
class World:
    org_x: str = field(default_factory=_uuid)
    org_y: str = field(default_factory=_uuid)
    people: dict[str, Person] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, org in (("owner_x", self.org_x), ("editor_x", self.org_x),
                         ("viewer_x", self.org_x), ("outsider_y", self.org_y)):
            self.people[key] = Person(key, org)

    def __getitem__(self, key: str) -> Person:
        return self.people[key]


WORLD = World()


def _seed(d: Db, w: World) -> None:
    p = w.people
    sql = []
    for person in p.values():
        sql += [
            f"INSERT INTO auth.users (id, email, email_confirmed_at) VALUES "
            f"('{person.auth}', '{person.key}@example.com', now());",
            f"INSERT INTO public.profiles (id, user_id, full_name, organization_id) VALUES "
            f"('{person.profile}', '{person.auth}', 'Name {person.key}', '{person.org}');",
        ]
    sql += [
        f"INSERT INTO public.organizations (id, name, owner_profile_id) VALUES "
        f"('{w.org_x}', 'Org X', '{p['owner_x'].profile}'), ('{w.org_y}', 'Org Y', '{p['outsider_y'].profile}');",
    ]
    for person in p.values():
        role = "owner" if person.key.startswith(("owner", "outsider")) else "member"
        sql.append(
            f"INSERT INTO public.app_users (id, organization_id, email, auth_user_id, profile_id, role, status) VALUES "
            f"('{person.app_user}', '{person.org}', '{person.key}@example.com', '{person.auth}', "
            f"'{person.profile}', '{role}', 'Active');")
    r = d.psql("SET session_replication_role = replica;\n" + "\n".join(sql), check=False)
    assert r.returncode == 0, f"seed failed: {r.stderr[-2000:]}"


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("project_delete")
    mig = tmp / "migrations"
    shutil.copytree(MIGRATIONS, mig)
    shutil.copyfile(BASELINE, mig / "baseline.txt")   # as the chart mounts it
    d = _new_db()
    try:
        out = _ok(d.run_hook(mig))
        # Since the develop-1.6 merge this file is a baseline migration (1.6 shipped it
        # before tracking), so on a fresh database the hook logs it "(baseline)".
        assert re.search(rf"applying {re.escape(VERSION)}(?: \(baseline\))?\n", out), \
            "the hook did not apply the migration under test"
        _seed(d, WORLD)
        yield d
    finally:
        _drop(d)


def _project(d: Db) -> str:
    """A fresh project of owner_x in org X, shared with editor_x ('edit') and viewer_x ('view')."""
    w, pid = WORLD, _uuid()
    d.admin(
        "SET session_replication_role = replica;\n"
        f"INSERT INTO public.projects (id, owner_id, name, organization_id) VALUES "
        f"('{pid}', '{w['owner_x'].profile}', 'P {pid[:8]}', '{w.org_x}');\n"
        "INSERT INTO public.resource_collaborators (resource_type, resource_id, app_user_id, access_level, invited_by) VALUES "
        f"('project', '{pid}', '{w['editor_x'].app_user}', 'edit', '{w['owner_x'].auth}'), "
        f"('project', '{pid}', '{w['viewer_x'].app_user}', 'view', '{w['owner_x'].auth}');"
    )
    return pid


def _delete(d: Db, who: str, pid: str) -> str:
    r = d.as_user(WORLD[who].auth, f"WITH x AS (DELETE FROM public.projects WHERE id = '{pid}' RETURNING 1) "
                                   "SELECT count(*) FROM x;")
    assert r.returncode == 0, f"{who}: {r.stderr}"
    return r.stdout.strip()


def _exists(d: Db, pid: str) -> bool:
    return d.scalar(f"SELECT count(*) FROM public.projects WHERE id = '{pid}'") == "1"


def test_the_policy_in_force_is_owners_and_editors(db):
    names = db.scalar("SELECT string_agg(policyname, '|') FROM pg_policies "
                      "WHERE schemaname = 'public' AND tablename = 'projects' AND cmd IN ('DELETE', 'ALL')")
    assert POLICY in (names or "").split("|"), names
    assert "owners and collaborators delete projects" not in (names or ""), names


def test_owner_deletes_its_project(db):
    pid = _project(db)
    assert _delete(db, "owner_x", pid) == "1"
    assert not _exists(db, pid)


def test_editor_deletes_a_shared_project(db):
    pid = _project(db)
    assert _delete(db, "editor_x", pid) == "1"
    assert not _exists(db, pid)


def test_viewer_is_refused(db):
    pid = _project(db)
    seen = db.as_user(WORLD["viewer_x"].auth, f"SELECT count(*) FROM public.projects WHERE id = '{pid}';")
    assert seen.stdout.strip() == "1", f"precondition: the viewer must see the project ({seen.stderr})"
    assert _delete(db, "viewer_x", pid) == "0", "a 'view' collaborator deleted the project"
    assert _exists(db, pid)


def test_unrelated_user_is_refused(db):
    """An unrelated user cannot even see the project, and SELECT policies also filter the
    rows a DELETE can reach, so this refusal holds even if the DELETE policy were loosened.
    The viewer case above is the one that pins the DELETE policy itself."""
    pid = _project(db)
    seen = db.as_user(WORLD["outsider_y"].auth, f"SELECT count(*) FROM public.projects WHERE id = '{pid}';")
    assert seen.stdout.strip() == "0", f"the outsider can see org X's project ({seen.stderr})"
    assert _delete(db, "outsider_y", pid) == "0", "a user unrelated to the project deleted it"
    assert _exists(db, pid)
