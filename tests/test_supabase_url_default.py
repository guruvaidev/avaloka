"""
SUPABASE_URL must not default to a hardcoded project.

SUPABASE_URL defaulted to a hardcoded third-party project. A deploy that set only
SUPABASE_SERVICE_ROLE_KEY would send the key + profiling data to that fixed project.
The default is removed; a missing URL now skips persistence instead of leaking.
"""

import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import app.agents.sampling_persistence as sp


#: Any Supabase project URL. Matching the *shape* rather than one project ref
#: makes the guard stronger and keeps a real project identifier out of a public
#: repository -- naming the project we were leaking to is its own small leak.
_SUPABASE_PROJECT_URL = re.compile(r"https://[a-z0-9]{15,}\.supabase\.co")


def test_no_hardcoded_supabase_url_in_source():
    for rel in ("app/agents/sampling_persistence.py", "app/api/cloud_connections.py"):
        text = (ROOT / rel).read_text()
        found = _SUPABASE_PROJECT_URL.search(text)
        assert not found, f"{rel} hardcodes a Supabase project URL: {found.group(0)}"


def test_missing_url_raises_instead_of_using_foreign_project(monkeypatch):
    monkeypatch.setattr(sp, "SUPABASE_URL", None)
    monkeypatch.setattr(sp, "SUPABASE_SERVICE_ROLE_KEY", "svc-key")

    with pytest.raises(RuntimeError):
        sp.get_supabase_client()   # must not build a client to any hardcoded project


def test_configured_url_is_honored(monkeypatch):
    captured = {}
    fake = types.ModuleType("supabase")
    fake.create_client = lambda url, key: (captured.update(url=url, key=key), "client")[1]
    monkeypatch.setitem(sys.modules, "supabase", fake)
    monkeypatch.setattr(sp, "SUPABASE_URL", "https://myproject.supabase.co")
    monkeypatch.setattr(sp, "SUPABASE_SERVICE_ROLE_KEY", "svc-key")

    client = sp.get_supabase_client()

    assert client == "client"
    assert captured["url"] == "https://myproject.supabase.co"
