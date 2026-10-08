"""Plan resolution and limits (app/core/plan_limits.py).

The fake Supabase client below serves the same table shapes as production
(profiles, subscriptions, organizations, organization_teams, team_memberships,
plans), so the tests exercise the real query chain without a network.
"""
from datetime import datetime, timedelta, timezone

import pytest

import app.core.plan_limits as pl

MB = 1024 * 1024
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)

PLANS = [
    {"id": "plan-free", "plan_type": "free"},
    {"id": "plan-pro-m", "plan_type": "professional"},
    {"id": "plan-pro-y", "plan_type": "professional"},
    {"id": "plan-ent-m", "plan_type": "enterprise"},
]


class _Res:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, rows):
        self._rows = rows
        self._filters = []
        self._limit = None

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self._filters.append(lambda r: r.get(col) == val)
        return self

    def in_(self, col, vals):
        vals = set(vals)
        self._filters.append(lambda r: r.get(col) in vals)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def execute(self):
        rows = [r for r in self._rows if all(f(r) for f in self._filters)]
        return _Res(rows[: self._limit] if self._limit else rows)


class FakeSupabase:
    def __init__(self, **tables):
        self.tables = {"plans": PLANS, **tables}
        self.calls = 0

    def table(self, name):
        self.calls += 1
        return _Query(self.tables.get(name, []))


def _iso(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f+00")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    pl.clear_plan_cache()
    monkeypatch.delenv("AVALOKA_FREE_MAX_TOTAL_BYTES", raising=False)
    monkeypatch.delenv("AVALOKA_PLAN_GRACE_DAYS", raising=False)
    # A CI or developer shell may run as an OSS edition; these tests are about
    # the hosted (billed) behaviour unless a test sets them explicitly.
    monkeypatch.delenv("AVALOKA_EDITION", raising=False)
    monkeypatch.delenv("AVALOKA_PLAN_LIMITS", raising=False)
    monkeypatch.delenv("AVALOKA_DEFAULT_PLAN", raising=False)
    yield
    pl.clear_plan_cache()


def _use(monkeypatch, client):
    monkeypatch.setattr(pl, "_get_client", lambda: client)


def _sub(**kw):
    base = {"status": "active", "plan_id": "plan-pro-m", "current_period_end": None,
            "trial_ends_at": None, "organization_id": None, "owner_profile_id": None}
    base.update(kw)
    return base


PROFILE = {"id": "prof-1", "user_id": "auth-1"}


# ---- limits ---------------------------------------------------------------


def test_free_allows_100_mb_in_total():
    lim = pl.limits_for("free")
    assert lim.max_total_bytes == 100 * MB
    assert lim.batch_jobs is False


def test_unknown_or_missing_plan_is_free():
    assert pl.limits_for(None).plan == "free"
    assert pl.limits_for("gold").plan == "free"


def test_paid_plans_allow_batch_jobs():
    assert pl.limits_for("professional").batch_jobs
    assert pl.limits_for("enterprise").batch_jobs


def test_free_limit_is_configurable(monkeypatch):
    monkeypatch.setenv("AVALOKA_FREE_MAX_TOTAL_BYTES", str(50 * MB))
    assert pl.limits_for("free").max_total_bytes == 50 * MB


def test_several_small_free_files_fit_under_the_total():
    assert pl.check_sizes(pl.limits_for("free"), [30 * MB, 30 * MB, 30 * MB]) is None


def test_free_files_over_the_total_are_refused_with_an_upgrade_offer():
    err = pl.check_sizes(pl.limits_for("free"), [60 * MB, 60 * MB])
    assert err["code"] == "plan_limit_exceeded"
    assert err["reason"] == "total_size"
    assert err["actual"] == 120 * MB
    assert err["upgrade_available"] is True
    assert "upgrade to Professional" in err["message"]


def test_one_byte_over_is_refused():
    assert pl.check_sizes(pl.limits_for("free"), [100 * MB]) is None
    assert pl.check_sizes(pl.limits_for("free"), [100 * MB + 1]) is not None


def test_paid_plan_is_not_offered_an_upgrade():
    err = pl.check_sizes(pl.limits_for("professional"), [150 * MB])
    assert err["reason"] == "file_size"
    assert err["upgrade_available"] is False
    assert err["upgrade_url"] is None


def test_too_many_files_is_its_own_reason():
    err = pl.check_sizes(pl.limits_for("free"), [1] * 11)
    assert err["reason"] == "file_count"
    assert err["upgrade_available"] is False


# ---- subscription validity ------------------------------------------------


def test_active_within_period_is_current():
    assert pl.subscription_is_current(_sub(current_period_end=_iso(NOW + timedelta(days=10))), NOW)


def test_active_without_an_end_date_is_current():
    assert pl.subscription_is_current(_sub(current_period_end=None), NOW)


def test_active_long_past_its_period_is_not_current():
    """The Stripe row in production: status 'active', period ended."""
    assert not pl.subscription_is_current(_sub(current_period_end=_iso(NOW - timedelta(days=30))), NOW)


def test_a_late_renewal_webhook_is_covered_by_the_grace_period():
    assert pl.subscription_is_current(_sub(current_period_end=_iso(NOW - timedelta(days=2))), NOW)


def test_a_stale_trial_is_not_current():
    """The PayPal row in production: 'trialing' five weeks after trial_ends_at."""
    sub = _sub(status="trialing", trial_ends_at=_iso(NOW - timedelta(days=37)))
    assert not pl.subscription_is_current(sub, NOW)


@pytest.mark.parametrize("status", ["canceled", "past_due", "unpaid", "incomplete", "incomplete_expired"])
def test_other_statuses_are_not_current(status):
    assert not pl.subscription_is_current(_sub(status=status), NOW)


def test_postgres_short_offset_timestamps_parse():
    assert pl._parse_ts("2027-08-21 00:00:00+00") == datetime(2027, 8, 21, tzinfo=timezone.utc)


# ---- plan resolution ------------------------------------------------------


def test_no_profile_is_free(monkeypatch):
    _use(monkeypatch, FakeSupabase(profiles=[]))
    assert pl.resolve_plan("auth-1") == "free"


def test_no_subscription_is_free(monkeypatch):
    _use(monkeypatch, FakeSupabase(profiles=[PROFILE], subscriptions=[]))
    assert pl.resolve_plan("auth-1") == "free"


def test_personal_professional_subscription(monkeypatch):
    _use(monkeypatch, FakeSupabase(profiles=[PROFILE],
                                   subscriptions=[_sub(owner_profile_id="prof-1")]))
    assert pl.resolve_plan("auth-1") == "professional"


def test_expired_subscription_falls_back_to_free(monkeypatch):
    old = _sub(owner_profile_id="prof-1", current_period_end="2025-01-01 00:00:00+00")
    _use(monkeypatch, FakeSupabase(profiles=[PROFILE], subscriptions=[old]))
    assert pl.resolve_plan("auth-1") == "free"


def test_the_highest_current_plan_wins(monkeypatch):
    subs = [_sub(owner_profile_id="prof-1", plan_id="plan-pro-m"),
            _sub(owner_profile_id="prof-1", plan_id="plan-ent-m")]
    _use(monkeypatch, FakeSupabase(profiles=[PROFILE], subscriptions=subs))
    assert pl.resolve_plan("auth-1") == "enterprise"


def test_an_organisation_owner_gets_the_org_plan(monkeypatch):
    org_sub = _sub(owner_profile_id="someone-else", organization_id="org-1", plan_id="plan-ent-m")
    _use(monkeypatch, FakeSupabase(
        profiles=[PROFILE], subscriptions=[org_sub],
        organizations=[{"id": "org-1", "owner_profile_id": "prof-1"}]))
    assert pl.resolve_plan("auth-1") == "enterprise"


@pytest.mark.parametrize("member_id", ["auth-1", "prof-1"])
def test_a_team_member_gets_the_org_plan_whichever_id_the_membership_stores(monkeypatch, member_id):
    org_sub = _sub(owner_profile_id="owner", organization_id="org-1", plan_id="plan-ent-m")
    _use(monkeypatch, FakeSupabase(
        profiles=[PROFILE], subscriptions=[org_sub], organizations=[],
        team_memberships=[{"team_id": "team-1", "user_id": member_id}],
        organization_teams=[{"id": "team-1", "organization_id": "org-1"}]))
    assert pl.resolve_plan("auth-1") == "enterprise"


def test_the_plan_is_cached(monkeypatch):
    client = FakeSupabase(profiles=[PROFILE], subscriptions=[_sub(owner_profile_id="prof-1")])
    _use(monkeypatch, client)
    assert pl.resolve_plan("auth-1") == "professional"
    calls = client.calls
    assert pl.resolve_plan("auth-1") == "professional"
    assert client.calls == calls


def test_a_failed_lookup_is_free_and_not_cached_long(monkeypatch):
    def boom():
        raise RuntimeError("supabase down")
    monkeypatch.setattr(pl, "_get_client", boom)
    assert pl.resolve_plan("auth-1") == "free"
    assert pl._CACHE["auth-1"][0] - __import__("time").monotonic() <= pl._FAILURE_TTL_S


def test_no_user_is_free():
    assert pl.resolve_plan(None) == "free"


# ---- installs without billing (self-hosted / OSS) -------------------------


def _no_supabase(monkeypatch):
    def must_not_be_called():
        pytest.fail("an install without billing must not query Supabase")
    monkeypatch.setattr(pl, "_get_client", must_not_be_called)


@pytest.mark.parametrize("var,value", [
    ("AVALOKA_EDITION", "oss"),
    ("AVALOKA_EDITION", "community"),
    ("AVALOKA_EDITION", "OSS"),
    ("AVALOKA_PLAN_LIMITS", "off"),
    ("AVALOKA_PLAN_LIMITS", "0"),
    ("AVALOKA_PLAN_LIMITS", "false"),
    ("AVALOKA_PLAN_LIMITS", "disabled"),
])
def test_an_install_without_billing_is_not_limited_and_never_asks_supabase(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    _no_supabase(monkeypatch)
    assert pl.resolve_plan("auth-1") == "enterprise"


def test_an_install_without_billing_is_not_offered_an_upgrade(monkeypatch):
    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    _no_supabase(monkeypatch)
    err = pl.check_sizes(pl.limits_for(pl.resolve_plan("auth-1")), [150 * MB])
    assert err is None or err["upgrade_available"] is False


def test_an_install_without_billing_applies_even_with_no_user(monkeypatch):
    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    _no_supabase(monkeypatch)
    assert pl.resolve_plan(None) == "enterprise"


def test_the_plan_of_an_unbilled_install_is_configurable(monkeypatch):
    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    monkeypatch.setenv("AVALOKA_DEFAULT_PLAN", "professional")
    _no_supabase(monkeypatch)
    assert pl.resolve_plan("auth-1") == "professional"


def test_an_unknown_default_plan_falls_back_to_enterprise(monkeypatch):
    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    monkeypatch.setenv("AVALOKA_DEFAULT_PLAN", "gold")
    _no_supabase(monkeypatch)
    assert pl.resolve_plan("auth-1") == "enterprise"


@pytest.mark.parametrize("value", ["", "on", "1", "hosted"])
def test_plan_limits_stay_on_unless_explicitly_turned_off(monkeypatch, value):
    monkeypatch.setenv("AVALOKA_PLAN_LIMITS", value)
    _use(monkeypatch, FakeSupabase(profiles=[PROFILE], subscriptions=[]))
    assert pl.resolve_plan("auth-1") == "free"


def test_a_hosted_edition_still_reads_the_subscription(monkeypatch):
    monkeypatch.setenv("AVALOKA_EDITION", "cloud")
    _use(monkeypatch, FakeSupabase(profiles=[PROFILE],
                                   subscriptions=[_sub(owner_profile_id="prof-1")]))
    assert pl.resolve_plan("auth-1") == "professional"


def test_a_failed_lookup_is_logged_as_an_error(monkeypatch, caplog):
    def boom():
        raise RuntimeError("supabase down")
    monkeypatch.setattr(pl, "_get_client", boom)
    with caplog.at_level("ERROR", logger=pl.logger.name):
        assert pl.resolve_plan("auth-1") == "free"
    assert any("plan lookup failed" in r.getMessage() and r.levelname == "ERROR"
               for r in caplog.records)