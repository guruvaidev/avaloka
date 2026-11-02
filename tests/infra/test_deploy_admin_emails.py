"""ADMIN_EMAILS reaches the chart's admin.emails through deploy_stack, and only that.

No cluster: run_command is mocked and the argv it would run is inspected.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.infra import deploy_stack


def _argv(monkeypatch, admin_emails: str | None) -> list[str]:
    if admin_emails is None:
        monkeypatch.delenv("ADMIN_EMAILS", raising=False)
    else:
        monkeypatch.setenv("ADMIN_EMAILS", admin_emails)
    with patch.object(deploy_stack, "run_command") as rc:
        rc.return_value = {"step_name": "x", "status": "SUCCESS", "message": "", "details": ""}
        deploy_stack.deploy_avaloka(namespace="avaloka-test")
        return rc.call_args.args[0]


@pytest.mark.parametrize("value", [None, "", "  ", " , "])
def test_unset_admin_emails_passes_nothing(monkeypatch, value):
    argv = _argv(monkeypatch, value)
    assert not any("admin.emails" in a for a in argv), "an empty ADMIN_EMAILS must not set an admin"


def test_admin_emails_are_normalised_into_the_helm_list(monkeypatch):
    argv = _argv(monkeypatch, " Ops@Example.com , lead@example.com,")
    want = "admin.emails={ops@example.com,lead@example.com}"
    assert want in argv, "ADMIN_EMAILS did not reach admin.emails"
    assert argv[argv.index(want) - 1] == "--set"


@pytest.mark.parametrize("bad", ["not-an-email", "a@example.com,b c@example.com", "x@example.com}", "a=b@example.com"])
def test_malformed_admin_emails_fail_the_deploy(monkeypatch, bad):
    with pytest.raises(ValueError) as err:
        _argv(monkeypatch, bad)
    assert "@" not in str(err.value), "the error must not echo the addresses"
