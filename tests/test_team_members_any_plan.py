"""Team members may be on any plan; creating a team stays Professional or higher.

Owner ruling: a user on any plan (Free, Professional or Enterprise) may be invited
to and added as a member of a Professional team. Creating a team stays
Professional and above, the same rule as the UI's canCreateTeams
(ui/src/lib/use-plan-features.ts). Migration
20261116140000_team_members_any_plan.sql.

Run against real Postgres under the chart's tracked migrate hook, with the
harness and plan-state world of tests/test_plan_enforced_org_features.py. Every
write is made as a signed-in browser (role authenticated, JWT claims set), so the
app's own RLS policies and triggers decide.

Pinned:
  * an invitation to a Free, Professional or Enterprise user is created by the
    team owner, and the invitee accepts it and becomes a member;
  * a public.teams INSERT is refused below Professional, and the refusal comes
    from the new policy alone (the same statement passes with only that policy
    dropped, rolled back); Professional and Enterprise may create one; the
    app-admin bypass is allowed and audited once;
  * the migration ships identically in both copies, is not in the baseline, and
    re-applies as a no-op;
  * a later ruling: any member of the sharer's team can receive a shared
    analysis, whatever their plan, directly or through the team; someone on none
    of the sharer's teams cannot. The database already allowed this (no
    migration changes it); these pin it.

Needs AVALOKA_TEST_PG_ADMIN_URL, psql and sh. Skips on a laptop without them and
FAILS when CI is set.
"""

from __future__ import annotations

import json

import pytest

from tests.test_plan_enforced_org_features import (
    MIGRATIONS,
    REPO_ROOT,
    RLS_DENIED,
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

VERSION = "20261116140000_team_members_any_plan"
UI_COPY = REPO_ROOT / "ui" / "supabase" / "migrations"
POLICY = "pro plan to create teams"
LOG = "avaloka_meta.org_feature_bypass_log"

#: Invitee plan -> the world org whose owner is on that plan.
INVITEES = {"free": "free", "professional": "professional", "enterprise": "enterprise"}


class Inviter:
    """A Professional user who owns a standalone (profile-owned) team."""

    def __init__(self) -> None:
        self.auth = _uuid()
        self.profile = _uuid()
        self.team = _uuid()


def _seed_inviter(d: Db, w: World) -> Inviter:
    i = Inviter()
    d.admin("\n".join([
        f"INSERT INTO auth.users (id, email, email_confirmed_at) VALUES ('{i.auth}', 'inviter@example.com', now());",
        f"INSERT INTO public.profiles (id, user_id, full_name) VALUES ('{i.profile}', '{i.auth}', 'Inviter');",
        f"INSERT INTO public.subscriptions (owner_profile_id, organization_id, plan_id, payment_provider, status) "
        f"VALUES ('{i.profile}', NULL, '{w.plans['professional']}', 'stripe', 'active');",
        f"INSERT INTO public.teams (id, owner_profile_id, lead_profile_id, name) "
        f"VALUES ('{i.team}', '{i.profile}', '{i.profile}', 'Pro team');",
    ]))
    return i


@pytest.fixture(scope="module")
def applied(tmp_path_factory):
    d = _new_db()
    try:
        mig = _chart_set(tmp_path_factory.mktemp("teamany"))
        first = _ok(d.run_hook(mig))
        second = _ok(d.run_hook(mig))
        yield d, first, second
    finally:
        _drop(d)


@pytest.fixture(scope="module")
def world(applied):
    d = applied[0]
    w = _seed(d)
    return w, _seed_inviter(d, w)


def _jwt(auth: str) -> str:
    return (
        "SET LOCAL ROLE authenticated;\n"
        f"SET LOCAL \"request.jwt.claim.sub\" = '{auth}';\n"
        "SET LOCAL \"request.jwt.claim.role\" = 'authenticated';\n"
        f"SET LOCAL \"request.jwt.claims\" = '{{\"sub\": \"{auth}\", \"role\": \"authenticated\"}}';\n"
    )


def _invite(d: Db, i: Inviter, invitee: Org):
    """The team owner invites, from their own browser session."""
    inv = _uuid()
    r = d.as_user(i.auth, (
        "INSERT INTO public.team_invitations (id, team_id, invited_profile_id, invited_email, invited_by, status) "
        f"VALUES ('{inv}', '{i.team}', '{invitee.profile}', 'owner-{invitee.key}@example.com', '{i.profile}', 'pending') "
        "RETURNING 'ok';"))
    return inv, r


# ---- the migration ------------------------------------------------------------------


def test_migration_is_new_not_baseline_and_both_copies_are_identical():
    assert VERSION not in _baseline_versions()
    chart = MIGRATIONS / f"{VERSION}.sql"
    ui = UI_COPY / f"{VERSION}.sql"
    assert chart.is_file() and ui.is_file()
    assert chart.read_bytes() == ui.read_bytes(), "the chart and UI copies differ"


def test_fresh_install_applies_it_once(applied):
    d, first, second = applied
    assert f"applying {VERSION}" in first
    assert d.scalar(f"SELECT how FROM avaloka_meta.schema_migrations WHERE version = '{VERSION}'") == "applied"
    assert f"applying {VERSION}" not in second
    assert "migrations complete: 0 applied," in second


def test_applying_the_sql_again_is_a_no_op(applied):
    d = applied[0]
    sql = (MIGRATIONS / f"{VERSION}.sql").read_text()
    before = d.scalar("SELECT count(*) FROM pg_policies WHERE tablename = 'teams'")
    r = d.psql(sql, check=False)
    assert r.returncode == 0, r.stderr
    assert d.scalar("SELECT count(*) FROM pg_policies WHERE tablename = 'teams'") == before
    assert d.scalar(f"SELECT count(*) FROM pg_policies WHERE tablename = 'teams' AND policyname = '{POLICY}'") == "1"


# ---- invitations: any plan ------------------------------------------------------------


@pytest.mark.parametrize("plan", list(INVITEES))
def test_owner_invites_and_the_invitee_accepts_on_any_plan(applied, world, plan):
    d, (w, i) = applied[0], world
    invitee = w.orgs[INVITEES[plan]]
    assert d.scalar(f"SELECT avaloka_meta.plan_tier_of('{invitee.auth}')") == plan
    inv, r = _invite(d, i, invitee)
    assert (r.returncode, r.stdout.strip()) == (0, "ok"), f"{plan} invitee refused at invite: {r.stderr}"
    # The invitee accepts from their own session (the invitee's UPDATE policy).
    r = d.as_user(invitee.auth, (
        f"WITH u AS (UPDATE public.team_invitations SET status = 'accepted' WHERE id = '{inv}' RETURNING 1) "
        "SELECT count(*) FROM u;"))
    assert (r.returncode, r.stdout.strip()) == (0, "1"), f"{plan} invitee refused at accept: {r.stderr}"
    assert d.scalar(f"SELECT count(*) FROM public.team_memberships WHERE team_id = '{i.team}' "
                    f"AND user_id = '{invitee.profile}'") == "1"


# ---- creating a team: Professional or higher --------------------------------------------


def _team_insert(o: Org, tag: str) -> str:
    return (f"INSERT INTO public.teams (owner_profile_id, name) "
            f"VALUES ('{o.profile}', 'New team {tag}') RETURNING 'ok';")


@pytest.mark.parametrize("key", ["free", "missing", "canceled", "past_due", "no_plan_row"])
def test_team_insert_is_refused_below_professional(applied, world, key):
    d, (w, _) = applied[0], world
    o = w.orgs[key]
    r = d.as_user(o.auth, _team_insert(o, key))
    assert r.returncode != 0 and RLS_DENIED in r.stderr, f"{key} created a team: {r.stdout}{r.stderr}"
    # The refusal comes from the new policy alone: without it the same write passes.
    r = d.psql(f"BEGIN;\nDROP POLICY \"{POLICY}\" ON public.teams;\n{_jwt(o.auth)}{_team_insert(o, key)}\nROLLBACK;\n",
               check=False)
    assert (r.returncode, r.stdout.strip()) == (0, "ok"), f"{key}: without {POLICY!r} the write should pass: {r.stderr}"


@pytest.mark.parametrize("key", ["professional", "enterprise", "enterprise_trial"])
def test_team_insert_is_allowed_from_professional(applied, world, key):
    d, (w, _) = applied[0], world
    o = w.orgs[key]
    r = d.as_user(o.auth, _team_insert(o, key))
    assert (r.returncode, r.stdout.strip()) == (0, "ok"), f"{key} refused: {r.stderr}"


def test_admin_bypass_team_insert_is_allowed_and_audited_once(applied, world):
    d, (w, _) = applied[0], world
    o = w.orgs["admin_on_free"]
    count = f"SELECT count(*) FROM {LOG} WHERE actor_user_id = '{o.auth}' AND action = 'teams'"
    before = int(d.scalar(count))
    r = d.as_user(o.auth, _team_insert(o, "bypass"))
    assert (r.returncode, r.stdout.strip()) == (0, "ok"), r.stderr
    assert int(d.scalar(count)) == before + 1
    row = json.loads(d.scalar(f"SELECT row_to_json(l) FROM {LOG} l WHERE actor_user_id = '{o.auth}' "
                              "AND action = 'teams' ORDER BY id DESC LIMIT 1"))
    assert (row["source"], row["operation"]) == ("rls", "INSERT")


def test_professional_team_insert_is_not_a_bypass_row(applied, world):
    d, (w, _) = applied[0], world
    o = w.orgs["admin_on_pro"]
    count = f"SELECT count(*) FROM {LOG} WHERE actor_user_id = '{o.auth}' AND action = 'teams'"
    before = int(d.scalar(count))
    r = d.as_user(o.auth, _team_insert(o, "pro-admin"))
    assert (r.returncode, r.stdout.strip()) == (0, "ok"), r.stderr
    assert int(d.scalar(count)) == before


# ---- sharing an analysis with a teammate: any plan -------------------------------------
#
# Owner ruling: any member of the sharer's team can receive its shared analyses,
# whatever their own plan. The server writes shares as service_role (the
# analysis_shares table grants nothing to authenticated), so a share is written
# here the same way, with its triggers on; the recipient then reads the analysis
# from their own browser session, through the analyses RLS policies.


class Sharing:
    def __init__(self) -> None:
        self.team = _uuid()
        self.analysis = _uuid()


#: Teammate plan -> the world org whose owner is on that plan; the outsider is on
#: none of the sharer's teams.
TEAMMATES = {"free": "free", "professional": "professional", "enterprise": "enterprise"}
OUTSIDER = "enterprise_trial"


@pytest.fixture(scope="module")
def sharing(applied, world):
    d, (w, i) = applied[0], world
    s = Sharing()
    sql = [
        f"INSERT INTO public.teams (id, owner_profile_id, lead_profile_id, name) "
        f"VALUES ('{s.team}', '{i.profile}', '{i.profile}', 'Sharing team');",
        f"INSERT INTO public.analyses (id, owner_id, name) VALUES ('{s.analysis}', '{i.profile}', 'Standalone analysis');",
    ]
    sql += [f"INSERT INTO public.team_memberships (team_id, user_id) VALUES ('{s.team}', '{w.orgs[k].profile}');"
            for k in TEAMMATES.values()]
    d.admin("\n".join(sql))
    return s


def _share_as_server(d: Db, s: Sharing, sharer: Inviter, recipient: Org, level: str = "edit"):
    return d.psql(
        "BEGIN;\nSET LOCAL ROLE service_role;\n"
        "INSERT INTO public.analysis_shares (analysis_id, recipient_profile_id, access_level, shared_by) "
        f"VALUES ('{s.analysis}', '{recipient.profile}', '{level}', '{sharer.profile}') RETURNING 'ok';\nCOMMIT;\n",
        check=False)


def _reads(d: Db, o: Org, s: Sharing) -> str:
    r = d.as_user(o.auth, f"SELECT count(*) FROM public.analyses WHERE id = '{s.analysis}';")
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.mark.parametrize("plan", list(TEAMMATES))
def test_a_teammate_on_any_plan_receives_a_direct_share(applied, world, sharing, plan):
    d, (w, i) = applied[0], world
    o = w.orgs[TEAMMATES[plan]]
    assert d.scalar(f"SELECT avaloka_meta.plan_tier_of('{o.auth}')") == plan
    r = _share_as_server(d, sharing, i, o)
    assert (r.returncode, r.stdout.strip()) == (0, "ok"), f"{plan} teammate refused a share: {r.stderr}"
    assert _reads(d, o, sharing) == "1", f"{plan} teammate cannot read the analysis shared with them"


def test_an_outsider_cannot_receive_a_direct_share_or_read_it(applied, world, sharing):
    d, (w, i) = applied[0], world
    o = w.orgs[OUTSIDER]
    r = _share_as_server(d, sharing, i, o)
    assert r.returncode != 0 and "Direct sharing requires a current shared team." in r.stderr, r.stdout + r.stderr
    assert _reads(d, o, sharing) == "0"


@pytest.mark.parametrize("plan", list(TEAMMATES))
def test_a_team_share_reaches_a_member_on_any_plan_and_not_an_outsider(applied, world, sharing, plan):
    d, (w, i) = applied[0], world
    team_analysis = _uuid()
    d.admin(f"INSERT INTO public.analyses (id, owner_id, name) VALUES ('{team_analysis}', '{i.profile}', 'Team analysis {plan}');")
    r = d.psql(
        "BEGIN;\nSET LOCAL ROLE service_role;\n"
        "INSERT INTO public.analysis_shares (analysis_id, team_id, access_level, shared_by) "
        f"VALUES ('{team_analysis}', '{sharing.team}', 'view', '{i.profile}') RETURNING 'ok';\nCOMMIT;\n", check=False)
    assert (r.returncode, r.stdout.strip()) == (0, "ok"), r.stderr
    for who, expected in ((w.orgs[TEAMMATES[plan]], "1"), (w.orgs[OUTSIDER], "0")):
        r = d.as_user(who.auth, f"SELECT count(*) FROM public.analyses WHERE id = '{team_analysis}';")
        assert (r.returncode, r.stdout.strip()) == (0, expected), f"{who.key}: {r.stdout}{r.stderr}"
