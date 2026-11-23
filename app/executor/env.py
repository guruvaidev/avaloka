"""The environment a model-written script runs with.

An allowlist, not a denylist. A denylist has to name every secret the
deployment will ever add (DB_ENCRYPTION_KEY, AVALOKA_LOCAL_AUTH_SECRET,
MCP_SERVICE_TOKEN, AVALOKA_PII_KEY, each provider's *_API_KEY, REDIS_URL,
CELERY_REDIS_URL, S3 keys, the Supabase JWT secret, every database password,
...) and fails open the day someone adds one more. The allowlist names what a
pandas/daft script needs to start and fails closed: anything new is withheld
until someone decides it belongs here.

What is allowed, and why:

* ``PATH``, ``HOME``, ``LANG``/``LC_*``, ``TZ``: the interpreter finds its
  binaries, caches, encodings and timestamps through them. Callers may point
  ``HOME`` at a private temp dir.
* ``PYTHON*`` switches that change only how the interpreter behaves (hash
  seed, unbuffered output, no .pyc writes on a read-only root, UTF-8 I/O).
* ``TMPDIR``, ``MPLCONFIGDIR``: somewhere writable for temp files and the
  matplotlib font cache when the root filesystem is read-only. Callers may
  point ``TMPDIR`` at a private temp dir.
* Thread-count caps and ``MALLOC_ARENA_MAX``, so a script cannot be told by
  the host to use more cores than the deployment gave it, and cannot be
  starved either.
* ``HF_HOME`` and the HF/transformers offline switches: a script that imports
  sentence-transformers finds the baked model and never tries the Hub.
* Every third-party telemetry/IMDS opt-out (``THIRDPARTY_OPTOUT_ENV``), so a
  script that imports daft, mlflow, chromadb or boto does not phone home.
  They carry no host value unless the host set one.

Generated scripts read no data-path variable: the callers bake the input and
output paths into the script as literals (``execution_agent``,
``daft_execution``, ``executor.sample``), so none is allowed through.

Nothing here names a database, a bucket, a queue or a key. The only outbound
connection is here, and ``tests/test_executor_env.py`` and
``tests/test_generated_code_env_scrub.py`` keep it that way.
"""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional

from app.core.thirdparty_optout import THIRDPARTY_OPTOUT_ENV

#: Variables passed through to a model-written script when the host has them.
ALLOWED_ENV_NAMES = frozenset({
    "PATH",
    "HOME",
    "TMPDIR",
    "MPLCONFIGDIR",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LC_NUMERIC",
    "LC_TIME",
    "LC_COLLATE",
    "LC_MONETARY",
    "LC_MESSAGES",
    "TZ",
    "PYTHONHASHSEED",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONFAULTHANDLER",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MALLOC_ARENA_MAX",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    *THIRDPARTY_OPTOUT_ENV,
})

#: Substrings that never belong in a script's environment, whatever the
#: allowlist says. A second check, so a careless addition above is caught
#: by the test suite rather than shipped.
FORBIDDEN_NAME_PARTS = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH",
    "REDIS", "CELERY", "DATABASE", "_DSN", "CONN", "SUPABASE", "MCP_",
    "LANGGRAPH_API", "_URL", "_URI",
)


def _forbidden(name: str) -> bool:
    upper = name.upper()
    return any(part in upper for part in FORBIDDEN_NAME_PARTS)


def scrubbed_env(source: Optional[Mapping[str, str]] = None,
                 extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment for a model-written script.

    ``source`` defaults to this process's environment. ``extra`` sets values
    the caller decides (``PYTHONIOENCODING``, a private ``HOME``/``TMPDIR``);
    it goes through the same filter, so it cannot be used to smuggle a secret
    in. Every opt-out is defaulted on, as ``disable_thirdparty_telemetry``
    does for this process; a host that set one keeps its value.
    """
    src = os.environ if source is None else source
    env = {k: v for k, v in src.items() if k in ALLOWED_ENV_NAMES and not _forbidden(k)}
    for name, value in THIRDPARTY_OPTOUT_ENV.items():
        env.setdefault(name, value)
    for name, value in (extra or {}).items():
        if name in ALLOWED_ENV_NAMES and not _forbidden(name):
            env[name] = value
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return env
