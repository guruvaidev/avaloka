"""Stored chat messages keep no typed credentials (20261121100000_chat_credentials_redacted).

The migration adds public.redact_chat_credentials(text), redacts every user
message in public.analysis_messages as it is written or edited, and redacts the
messages already stored. Against real Postgres, with the chart's own migration
set applied by the chart's own migrate hook (the harness of
tests/test_plan_enforced_org_features.py):

* the upgrade: messages stored before the migration lose only their credential
  values; ids, the other columns and every other character stay as they were;
* the hook logs how many messages changed, and nothing of their content;
* running it again changes nothing;
* a message written afterwards, by the browser's role through row-level
  security, is stored redacted; ordinary text and assistant replies are not
  touched;
* the SQL function gives the Python redaction's result on every corpus message.

Needs AVALOKA_TEST_PG_ADMIN_URL (a role that may CREATE DATABASE), psql and sh.
Skips on a laptop without them and FAILS when CI is set. The file checks at the
top need nothing.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest

from app.core import typed_secrets
from tests.test_chat_credentials_redacted import CORPUS, ORDINARY, TYPED, _pw
from tests.test_plan_enforced_org_features import (BASELINE, MIGRATIONS, REPO_ROOT,
                                                   _chart_set, _drop, _new_db, _ok)

VERSION = "20261121100000_chat_credentials_redacted"
UI_COPY = REPO_ROOT / "ui" / "supabase" / "migrations"
NEWEST_BEFORE = "20261117130000"
COUNT_LINE = re.compile(r"credentials redacted in (\d+) stored user message")

PROBES = [p for _, _, probes in TYPED for p in probes]
ASSISTANT_TEXT = "Use password=" + _pw("assistant") + " only in the example config shown above."


def _python_redact(text: str) -> str:
    fn = getattr(typed_secrets, "redact_typed_secrets", None)
    return fn(text)[0] if fn else text


# ---- the files (no database) ----------------------------------------------------

def test_both_copies_exist_and_are_byte_identical():
    chart = MIGRATIONS / f"{VERSION}.sql"
    ui = UI_COPY / f"{VERSION}.sql"
    assert chart.is_file() and ui.is_file(), f"migration {VERSION} is missing from a migrations directory"
    assert chart.read_bytes() == ui.read_bytes()


def test_version_sorts_after_the_migrations_it_was_written_after_and_is_not_in_the_baseline():
    # Later migrations may follow it; their order is the migration-order guard's to check.
    assert VERSION[:14] > NEWEST_BEFORE
    baseline = {ln.strip() for ln in BASELINE.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    assert VERSION not in baseline


# ---- real Postgres ----------------------------------------------------------------

def _q(text: str) -> str:
    assert "$c7$" not in text
    return f"$c7${text}$c7$"


def _rows(d) -> dict:
    out = d.scalar(
        "SELECT coalesce(json_agg(json_build_object('id', id, 'analysis_id', analysis_id, 'role', role, "
        "'content', content, 'output', output, 'created_at', created_at, 'author_id', author_id) ORDER BY id), '[]') "
        "FROM public.analysis_messages;")
    return {r["id"]: r for r in json.loads(out)}


class _World:
    def __init__(self):
        self.auth = str(uuid.uuid4())
        self.profile = str(uuid.uuid4())
        self.analysis = str(uuid.uuid4())


def _seed_owner(d) -> _World:
    w = _World()
    d.admin(
        f"INSERT INTO auth.users (id, email, email_confirmed_at) VALUES ('{w.auth}', 'chat-{w.auth[:8]}@example.com', now());\n"
        f"INSERT INTO public.profiles (id, user_id, full_name) VALUES ('{w.profile}', '{w.auth}', 'Chat owner');\n"
        f"INSERT INTO public.analyses (id, owner_id, name) VALUES ('{w.analysis}', '{w.profile}', 'Chat');\n")
    return w


def _stored_before(d, w: _World) -> dict:
    """Messages as an older client stored them: written with triggers off, so exactly as typed."""
    rows = [("user", msg) for msg in CORPUS if msg] + [("assistant", ASSISTANT_TEXT)]
    values = ",\n".join(
        f"('{uuid.uuid4()}', '{w.analysis}', '{role}', {_q(text)}, "
        f"{'NULL' if role == 'user' else _q(json.dumps({'n': i}))}::jsonb, now() - interval '{i} minutes', "
        f"{'NULL' if role == 'assistant' else repr(w.profile)})"
        for i, (role, text) in enumerate(rows))
    d.admin("INSERT INTO public.analysis_messages (id, analysis_id, role, content, output, created_at, author_id) "
            f"VALUES\n{values};")
    return _rows(d)


@pytest.fixture(scope="module")
def upgraded(tmp_path_factory):
    """An install migrated without this migration and holding typed credentials,
    then the hook run again with the migration present, then a third time."""
    d = _new_db()
    try:
        mig = _chart_set(tmp_path_factory.mktemp("chatcred"))
        new = mig / f"{VERSION}.sql"
        assert new.is_file(), f"migration {VERSION} is missing from the chart's migration set"
        held = new.read_bytes()
        new.unlink()
        _ok(d.run_hook(mig))
        w = _seed_owner(d)
        before = _stored_before(d, w)
        new.write_bytes(held)
        upgrade_log = _ok(d.run_hook(mig))
        after = _rows(d)
        rerun_log = _ok(d.run_hook(mig))
        yield d, w, before, after, upgrade_log, rerun_log
    finally:
        _drop(d)


@pytest.mark.integration
def test_the_seed_holds_the_typed_passwords_before_the_upgrade(upgraded):
    _, _, before, _, _, _ = upgraded
    held = [p for p in PROBES if any(p in r["content"] for r in before.values())]
    assert held == PROBES, "the upgrade case did not start from stored credentials"


@pytest.mark.integration
def test_the_upgrade_leaves_no_typed_password_in_a_stored_message(upgraded):
    _, _, _, after, _, _ = upgraded
    leaked = [r["id"] for r in after.values() if r["role"] == "user" and any(p in r["content"] for p in PROBES)]
    assert not leaked, f"{len(leaked)} stored user message(s) still hold a typed password"
    for _, msg, probes in TYPED:
        expected = msg
        for p in probes:
            expected = expected.replace(p, "[redacted]")
        assert any(r["content"] == expected for r in after.values()), "a typed message lost more than its password"


@pytest.mark.integration
def test_only_the_credential_values_change(upgraded):
    _, _, before, after, _, _ = upgraded
    assert set(after) == set(before), "row ids changed"
    for rid, old in before.items():
        new = after[rid]
        assert {k: v for k, v in new.items() if k != "content"} == {k: v for k, v in old.items() if k != "content"}, \
            "a column other than content changed"
        if old["role"] == "user":
            assert new["content"] == _python_redact(old["content"]), "a stored message differs from the Python redaction"
        else:
            assert new["content"] == old["content"], "an assistant reply was changed"
    for text in ORDINARY:
        assert sum(r["content"] == text for r in after.values()) == 1, "ordinary text about passwords was changed"


@pytest.mark.integration
def test_the_hook_applies_it_once_and_logs_only_a_count(upgraded):
    _, _, before, after, upgrade_log, rerun_log = upgraded
    assert VERSION in upgrade_log
    assert VERSION not in rerun_log
    counts = [int(n) for n in COUNT_LINE.findall(upgrade_log)]
    changed = sum(before[i]["content"] != after[i]["content"] for i in before)
    assert counts == [changed] and changed >= len(TYPED), f"logged {counts}, changed {changed}"
    assert not [p for p in PROBES if p in upgrade_log or p in rerun_log], "the migration log carries a password"


@pytest.mark.integration
def test_running_the_sql_again_changes_nothing(upgraded):
    d = upgraded[0]
    sql = (MIGRATIONS / f"{VERSION}.sql").read_text()
    stamp = ("SELECT md5(string_agg(id || '|' || analysis_id || '|' || role || '|' || content || '|' || "
             "coalesce(output::text, '') || '|' || created_at || '|' || coalesce(author_id::text, ''), ',' ORDER BY id)) "
             "FROM public.analysis_messages;")
    before = d.scalar(stamp)
    result = d.psql(sql, check=True)
    assert d.scalar(stamp) == before, "a second run changed stored messages"
    assert [int(n) for n in COUNT_LINE.findall(result.stdout + result.stderr)] == [0]


@pytest.mark.integration
def test_a_message_written_afterwards_is_stored_redacted(upgraded):
    """The browser's own insert, as role authenticated through row-level security."""
    d, w, _, _, _, _ = upgraded
    ids = {}
    for name, msg, _ in TYPED:
        ids[name] = str(uuid.uuid4())
        res = d.as_user(w.auth, "INSERT INTO public.analysis_messages (id, analysis_id, role, content, author_id) "
                                f"VALUES ('{ids[name]}', '{w.analysis}', 'user', {_q(msg)}, '{w.profile}');")
        assert res.returncode == 0, f"{name}: the browser role could not write its message: {res.stderr[-300:]}"
    plain_id, reply_id = str(uuid.uuid4()), str(uuid.uuid4())
    d.as_user(w.auth, "INSERT INTO public.analysis_messages (id, analysis_id, role, content) VALUES "
                      f"('{plain_id}', '{w.analysis}', 'user', {_q(ORDINARY[0])}), "
                      f"('{reply_id}', '{w.analysis}', 'assistant', {_q(ASSISTANT_TEXT)});")
    rows = _rows(d)
    for name, msg, probes in TYPED:
        stored = rows[ids[name]]["content"]
        assert not [p for p in probes if p in stored], f"{name}: a new message was stored with its password"
        assert stored == _python_redact(msg)
    assert rows[plain_id]["content"] == ORDINARY[0]
    assert rows[reply_id]["content"] == ASSISTANT_TEXT
    # An edit is redacted as well.
    d.psql(f"UPDATE public.analysis_messages SET content = {_q(TYPED[1][1])} WHERE id = '{plain_id}';")
    assert TYPED[1][2][0] not in _rows(d)[plain_id]["content"], "an edited message was stored with its password"


@pytest.mark.integration
def test_the_sql_redaction_matches_the_python_one(upgraded):
    d = upgraded[0]
    values = ",\n".join(f"({i}, {_q(text)})" for i, text in enumerate(CORPUS))
    out = d.scalar(f"SELECT json_agg(public.redact_chat_credentials(t) ORDER BY n) FROM (VALUES {values}) AS v(n, t);")
    sql = json.loads(out)
    mismatched = [i for i, text in enumerate(CORPUS) if sql[i] != _python_redact(text)]
    assert not mismatched, f"SQL and Python redact corpus message(s) {mismatched} differently"


@pytest.mark.integration
def test_on_a_fresh_install_it_applies_and_redacts_nothing(tmp_path_factory):
    d = _new_db()
    try:
        log = _ok(d.run_hook(_chart_set(tmp_path_factory.mktemp("chatcred_fresh"))))
        assert VERSION in log
        assert [int(n) for n in COUNT_LINE.findall(log)] == [0]
        assert d.scalar("SELECT count(*) FROM pg_trigger WHERE tgname = 'analysis_messages_redact_credentials';") == "1"
    finally:
        _drop(d)


# ---- a Professional team member with edit access to a shared analysis -----------
#
# The chart's whole migration set, the team-sharing migrations included, with every
# write made the way the app makes it: sign-up through auth.users (its trigger makes
# the profile), the billing row and the team, invitation and share writes as
# service_role (the server functions use the service key; their triggers still run:
# the invitee plan gate, the membership-on-accept trigger, the standalone and
# teammate checks on analysis_shares), the analysis and the message as the browser
# (role authenticated, under row-level security). The member is not the owner, so
# only the "editors append shared analysis messages" policy can admit the message;
# the restrictive Professional-project policy applies too. It runs on whatever
# migration set the tree has, so on a tree without this migration it shows the
# member's message stored as typed.

PROFESSIONAL_PLAN = "2abfb8c8-e811-4f7c-aa1a-8f70b773d7db"


def _as_service(d, sql: str) -> str:
    res = d.psql(f"BEGIN;\nSET LOCAL ROLE service_role;\n{sql}\nCOMMIT;\n", check=False)
    assert res.returncode == 0, f"service write failed: {res.stderr[-400:]}"
    return res.stdout.strip()


def _sign_up(d, label: str) -> tuple[str, str]:
    """A user signing up: the auth row, and the profile its trigger makes."""
    auth = str(uuid.uuid4())
    d.psql(f"INSERT INTO auth.users (id, email, email_confirmed_at, raw_user_meta_data) VALUES "
           f"('{auth}', '{label}-{auth[:8]}@example.com', now(), '{{\"full_name\": \"{label}\"}}');")
    profile = d.scalar(f"SELECT id FROM public.profiles WHERE user_id = '{auth}';")
    assert profile, f"signing up {label} made no profile"
    return auth, profile


@pytest.fixture(scope="module")
def team_share(tmp_path_factory):
    d = _new_db()
    try:
        _ok(d.run_hook(_chart_set(tmp_path_factory.mktemp("chatcred_team"))))
        owner_auth, owner = _sign_up(d, "owner")
        member_auth, member = _sign_up(d, "member")
        # Both on Professional: a Free recipient of a share is held to view
        # (analysis_share_access_level, 20261111210000), and only an editor appends.
        for who in (owner, member):
            _as_service(d, "INSERT INTO public.subscriptions (owner_profile_id, organization_id, plan_id, payment_provider, "
                           f"status) VALUES ('{who}', NULL, '{PROFESSIONAL_PLAN}', 'stripe', 'active');")
        org = d.scalar(f"SELECT coalesce(organization_id::text, 'NULL') FROM public.profiles WHERE id = '{owner}';")
        org_sql = "NULL" if org == "NULL" else f"'{org}'"
        team = _as_service(d, "INSERT INTO public.teams (owner_profile_id, lead_profile_id, organization_id, name) "
                              f"VALUES ('{owner}', '{owner}', {org_sql}, 'Data team') RETURNING id;").splitlines()[0]
        invite = _as_service(d, "INSERT INTO public.team_invitations (team_id, invited_profile_id, invited_email, "
                                f"invited_by, status) VALUES ('{team}', '{member}', 'member@example.com', '{owner}', "
                                "'pending') RETURNING id;").splitlines()[0]
        _as_service(d, f"UPDATE public.team_invitations SET status = 'accepted' WHERE id = '{invite}' "
                       f"AND invited_profile_id = '{member}' AND status = 'pending';")
        analysis = str(uuid.uuid4())
        res = d.as_user(owner_auth, f"INSERT INTO public.analyses (id, owner_id, name) VALUES ('{analysis}', '{owner}', 'Shared');")
        assert res.returncode == 0, f"the owner could not create the analysis: {res.stderr[-300:]}"
        before_share = d.as_user(member_auth, "INSERT INTO public.analysis_messages (analysis_id, role, content, author_id) "
                                              f"VALUES ('{analysis}', 'user', 'hello', '{member}');")
        _as_service(d, "INSERT INTO public.analysis_shares (analysis_id, recipient_profile_id, access_level, can_share, "
                       f"shared_by) VALUES ('{analysis}', '{member}', 'edit', false, '{owner}');")
        yield d, member_auth, member, owner, team, analysis, before_share
    finally:
        _drop(d)


def _member_level(d, member_auth: str, analysis: str) -> str:
    res = d.as_user(member_auth, f"SELECT public.analysis_share_access_level('{analysis}');")
    assert res.returncode == 0, res.stderr[-300:]
    return [ln for ln in res.stdout.splitlines() if ln.strip()][-1].strip()


@pytest.mark.integration
def test_the_team_share_is_set_up_by_the_apps_own_rules(team_share):
    d, member_auth, member, owner, team, analysis, before_share = team_share
    assert d.scalar(f"SELECT count(*) FROM public.team_memberships WHERE team_id = '{team}' AND user_id = '{member}';") == "1", \
        "accepting the invitation did not make the member a team member"
    assert member != owner
    assert _member_level(d, member_auth, analysis) == "3", "the member does not have edit access to the shared analysis"
    assert before_share.returncode != 0 and "row-level security" in before_share.stderr, \
        "the member could write to the analysis before it was shared, so the share policy is not what is tested"


@pytest.mark.integration
@pytest.mark.parametrize("name,message,probes", TYPED, ids=[t[0] for t in TYPED])
def test_a_team_editor_appending_to_a_shared_analysis_stores_it_redacted(team_share, name, message, probes):
    d, member_auth, member, _owner, _team, analysis, _ = team_share
    msg_id = str(uuid.uuid4())
    res = d.as_user(member_auth, "INSERT INTO public.analysis_messages (id, analysis_id, role, content, author_id) "
                                 f"VALUES ('{msg_id}', '{analysis}', 'user', {_q(message)}, '{member}');")
    assert res.returncode == 0, f"{name}: the shared-analysis policies refused the editor's message: {res.stderr[-300:]}"
    stored = d.scalar(f"SELECT content FROM public.analysis_messages WHERE id = '{msg_id}';")
    assert not [p for p in probes if p in stored], f"{name}: the editor's message was stored with its password"
    assert stored == _python_redact(message)
