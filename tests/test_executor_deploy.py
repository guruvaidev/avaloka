"""The executor's isolation lives in the deployment files; pin it there.

The live check (generated code gets connection refused to postgres, sees no
secret and no db_connections.json) needs a running stack. These are the
static facts it depends on, so an edit that quietly removes one fails here.
"""

import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.quickstart.yml"
EXECUTOR_TPL = ROOT / "deploy" / "helm" / "avaloka" / "templates" / "executor.yaml"
SECRET_NAMES = ("DB_ENCRYPTION_KEY", "MCP_SERVICE_TOKEN", "AVALOKA_LOCAL_AUTH_SECRET", "AVALOKA_LOCAL_AUTH_PASSWORD",
                "GROQ_API_KEY", "OPENROUTER_API_KEY", "REDIS_URL", "CELERY_REDIS_URL")


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(COMPOSE.read_text("utf-8"))


def test_compose_executor_is_alone_on_an_internal_network(compose):
    ex = compose["services"]["executor"]
    assert ex["networks"] == ["sandbox"]
    assert compose["networks"]["sandbox"]["internal"] is True
    members = sorted(n for n, s in compose["services"].items() if "sandbox" in (s.get("networks") or []))
    assert members == ["api", "executor"], members
    # langgraph dev serves its runs API without auth: it must not be reachable.
    for db in ("postgres", "redis", "mcp", "langgraph"):
        assert "sandbox" not in compose["services"][db]["networks"]


def test_compose_executor_holds_no_secret_and_no_state(compose):
    ex = compose["services"]["executor"]
    env = ex.get("environment") or {}
    assert not [n for n in SECRET_NAMES if n in env]
    assert "AVALOKA_EXECUTOR_URL" not in env
    assert not ex.get("volumes"), "the executor must not mount a volume (no /state)"
    assert ex["user"] != "0" and not str(ex["user"]).startswith("0:")
    assert ex["read_only"] is True and ex["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in ex["security_opt"]


def test_compose_api_points_at_the_executor(compose):
    env = compose["services"]["api"]["environment"]
    assert env["AVALOKA_EXECUTOR_URL"] == "http://executor:8020"


def test_chart_executor_has_no_egress_no_token_no_secrets():
    text = EXECUTOR_TPL.read_text("utf-8")
    assert re.search(r"^\s*egress:\s*\[\]\s*$", text, re.M), "executor egress must be deny-all"
    assert re.search(r"policyTypes:\s*\n\s*- Ingress\s*\n\s*- Egress", text)
    assert "automountServiceAccountToken: false" in text
    assert "readOnlyRootFilesystem: true" in text and "allowPrivilegeEscalation: false" in text
    assert 'drop: ["ALL"]' in text and "runAsNonRoot: true" in text
    assert not re.search(r"^\s*envFrom:", text, re.M) and "secretKeyRef" not in text
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert not [n for n in SECRET_NAMES if n in code]
    assert "persistentVolumeClaim" not in code
    # The datastore-side policy excludes the executor's label.
    assert re.search(r"operator: NotIn\s*\n\s*values: \[code-executor\]", text)
