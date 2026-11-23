"""Generated secrets: created on first install, persisted, reused on every start.

Avaloka has five secrets nobody needs to choose, only to keep:

* ``DB_ENCRYPTION_KEY`` -- the AES-GCM key every sealed credential uses
  (app/core/encryption_key.py);
* ``MCP_SERVICE_TOKEN`` -- the shared API <-> MCP/onboarding service secret;
* ``EXECUTE_SERVICE_TOKEN`` -- the shared graph pods <-> ``/execute`` service
  secret (app/executor/execute_service.py);
* ``LANGGRAPH_SERVICE_TOKEN`` -- the shared API <-> LangGraph server secret
  (app/api/langgraph_service_auth.py);
* ``AVALOKA_PII_KEY``   -- the HMAC key for PII pseudonymisation.

When one is unset or empty, :func:`provision` generates it
(``secrets.token_hex(32)``, cryptographically random) and saves it to the
.env file, so the next start reuses it instead of regenerating. A value that
has been published in this repository (a chart default, a documented
placeholder) protects nothing, so it is replaced the same way, with a warning.
A generated value is never printed or logged; only "generated and saved to
<file>".

Where it runs:

* ``scripts/install.sh`` and ``python -m app.core.provisioning`` -- local
  installs and any compose stack that reads the same .env (``env_file``);
* ``scripts/quickstart.sh`` -- the same rules for ``.env.quickstart``, in bash
  (it runs before any Python dependency is installed);
* each service's startup, for the same .env, so a plain
  ``uvicorn app.api.server:app`` on a host needs no manual step either;
* Helm does the same in the chart (``avaloka.generatedSecret``: ``lookup`` of
  the release Secret, else ``randAlphaNum``).

Inside a container a service never generates by default: every container of
one deployment must share one value, and a key generated inside one container
lives only in that container's filesystem. A container adopts what its env
file already holds (:func:`adopt_from_file`); one that finds no key is a broken
deploy and refuses to start (encryption_key.py). ``AVALOKA_AUTO_PROVISION=1``
opts a (non-Kubernetes) container in; Kubernetes pods never generate.

Standalone on purpose (stdlib only): the installer runs it before the app's
dependencies are importable.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

#: Every secret Avaloka generates for itself: name -> what it is for.
GENERATED_SECRETS: Dict[str, str] = {
    "DB_ENCRYPTION_KEY": "AES-GCM key for every sealed credential",
    "MCP_SERVICE_TOKEN": "shared API <-> MCP/onboarding service secret",
    "EXECUTE_SERVICE_TOKEN": "shared graph pods <-> /execute service secret",
    "LANGGRAPH_SERVICE_TOKEN": "shared API <-> LangGraph server secret",
    "AVALOKA_PII_KEY": "HMAC key for PII pseudonymisation",
}

#: Values that have shipped in this repository's chart, docs or examples. A
#: secret equal to one of them (case-insensitive, whitespace stripped) is
#: treated as unset and replaced.
PUBLISHED_VALUES = frozenset({
    # deploy/helm/avaloka/values.yaml supabase.functions.dbEncryptionKey, through 1.6.
    "avaloka-local-dev-encryption-key",
    # .env.example placeholders, through 1.6.
    "<generate-with-openssl-rand-hex-32>",
    "<generate-and-store-a-strong-random-secret>",
    "changeme", "change-me", "secret", "test", "dev", "development",
    # MLflow's basic_auth.ini default admin password before 3.17 (GHSA-gq3w-7jj3-x7gr).
    "password", "password1234",
})

#: What replacing a published value costs, per secret: said with every
#: replacement warning, so an operator upgrading an install learns it there.
REPLACEMENT_CONSEQUENCE: Dict[str, str] = {
    "DB_ENCRYPTION_KEY": (
        "Anything sealed under the old key can no longer be opened: saved database "
        "connections, cloud connection keys, MCP and GitHub tokens, and passwords typed "
        "into chat. Register those connections again (and re-enter those tokens)."),
    "MCP_SERVICE_TOKEN": (
        "Nothing stored depends on it; restart every service (API, LangGraph, MCP) so "
        "they all use the new value."),
    "EXECUTE_SERVICE_TOKEN": (
        "Nothing stored depends on it; restart the graph pods and the execute service so "
        "they all use the new value."),
    "LANGGRAPH_SERVICE_TOKEN": (
        "Nothing stored depends on it; restart the API and the LangGraph server so "
        "they both use the new value."),
    "AVALOKA_PII_KEY": (
        "Columns pseudonymised under the old key no longer match new pseudonyms; "
        "retrain models trained on pseudonymised data."),
}

_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def is_published(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in PUBLISHED_VALUES


def is_usable(value: Optional[str]) -> bool:
    v = (value or "").strip()
    return bool(v) and not is_published(v)


def default_env_file() -> Path:
    """``AVALOKA_ENV_FILE`` if set, else ``.env`` in the working directory."""
    return Path(os.getenv("AVALOKA_ENV_FILE") or ".env").expanduser()


def in_kubernetes() -> bool:
    return bool(os.getenv("KUBERNETES_SERVICE_HOST"))


def in_container() -> bool:
    return in_kubernetes() or os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def startup_may_generate() -> bool:
    """Whether a service may generate a missing secret at startup.

    ``AVALOKA_AUTO_PROVISION``: ``0`` never; ``1`` yes, except inside Kubernetes
    (pods must share the release Secret); unset, only outside a container.
    """
    mode = os.getenv("AVALOKA_AUTO_PROVISION", "").strip()
    if mode == "0" or in_kubernetes():
        return False
    if mode == "1":
        return True
    return not in_container()


def adopt_from_file(env_file: Optional[Path] = None, names: Iterable[str] = GENERATED_SECRETS,
                    environ=None) -> Dict[str, str]:
    """Take each unusable secret in ``names`` from the env file, if it holds a usable one. Never generates.

    Returns ``{name: "environment" | "file" | "missing"}`` -- never a value.
    """
    environ = os.environ if environ is None else environ
    path = Path(env_file) if env_file else default_env_file()
    try:
        on_disk = read_env_file(path)
    except OSError:
        on_disk = {}
    done: Dict[str, str] = {}
    for name in names:
        if is_usable(environ.get(name)):
            done[name] = "environment"
        elif is_usable(on_disk.get(name)):
            environ[name] = on_disk[name].strip()
            done[name] = "file"
        else:
            done[name] = "missing"
    return done


def _unquote(raw: str) -> str:
    v = raw.strip()
    if " #" in v and not v.startswith(("'", '"')):
        v = v.split(" #", 1)[0].strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    return v


def read_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        text = path.read_text()
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if m and not line.lstrip().startswith("#"):
            out[m.group(1)] = _unquote(m.group(2))
    return out


def _write_values(path: Path, values: Dict[str, str]) -> None:
    """Set ``values`` in the env file: replace existing lines, append the rest. Mode 0600."""
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        lines = []
    pending = dict(values)
    for i, line in enumerate(lines):
        m = _LINE_RE.match(line)
        if m and not line.lstrip().startswith("#") and m.group(1) in pending:
            lines[i] = f"{m.group(1)}={pending.pop(m.group(1))}"
    if pending:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# Generated by Avaloka (app/core/provisioning.py). Keep these: changing")
        lines.append("# DB_ENCRYPTION_KEY makes every sealed credential unreadable.")
        lines.extend(f"{k}={v}" for k, v in pending.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    os.chmod(path, 0o600)


@contextmanager
def _locked(path: Path):
    """An exclusive lock beside the env file, so two services starting at once agree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows: best effort
            pass
        yield
    finally:
        os.close(fd)


def provision(env_file: Optional[Path] = None, names: Iterable[str] = GENERATED_SECRETS,
              environ=None, persist: bool = True) -> Dict[str, str]:
    """Make every generated secret in ``names`` usable; return what was done per name.

    For each name: a usable value in ``environ`` (default os.environ) wins; else
    a usable value in the env file is adopted; else (unset, empty or published)
    a new one is generated and, with ``persist``, saved to the env file. The
    adopted or generated value is put into ``environ``. Results are
    ``"environment"``, ``"file"``, ``"generated"`` or ``"replaced-published"``
    -- never the value.
    """
    environ = os.environ if environ is None else environ
    path = Path(env_file) if env_file else default_env_file()
    done: Dict[str, str] = {}
    with _locked(path):
        on_disk = read_env_file(path)
        to_write: Dict[str, str] = {}
        for name in names:
            current = environ.get(name)
            if is_usable(current):
                done[name] = "environment"
                continue
            if is_usable(on_disk.get(name)):
                environ[name] = on_disk[name].strip()
                done[name] = "file"
                continue
            published = is_published(current) or is_published(on_disk.get(name))
            value = secrets.token_hex(32)
            environ[name] = value
            to_write[name] = value
            done[name] = "replaced-published" if published else "generated"
        if to_write and persist:
            _write_values(path, to_write)
    for name, how in done.items():
        if how == "generated":
            logger.warning("%s was not set: generated and saved to %s", name, path)
        elif how == "replaced-published":
            logger.warning("%s was a published value (it ships in this repository) and protects "
                           "nothing: replaced with a generated one, saved to %s. %s If it was "
                           "exported in your shell, remove the export.", name, path,
                           REPLACEMENT_CONSEQUENCE.get(name, ""))
    return done


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Generate and persist Avaloka's generated secrets.")
    p.add_argument("--env-file", default=None, help="env file to read and update (default: $AVALOKA_ENV_FILE or ./.env)")
    args = p.parse_args(argv)
    path = Path(args.env_file) if args.env_file else default_env_file()
    done = provision(path, environ=dict(os.environ))
    for name, how in done.items():
        if how == "generated":
            print(f"[provision] {name}: generated and saved to {path}")
        elif how == "replaced-published":
            print(f"[provision] {name}: WARNING the value was a published one; replaced with a generated "
                  f"one, saved to {path}. {REPLACEMENT_CONSEQUENCE.get(name, '')}")
        elif how == "file":
            print(f"[provision] {name}: kept the value already in {path}")
        else:
            print(f"[provision] {name}: set in the environment; {path} left as it is")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
