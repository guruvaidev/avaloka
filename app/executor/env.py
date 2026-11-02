"""The environment a model-written script is allowed to see.

Ported from 2.0's ``app/executor/env.py`` (#575, used by the bench scripts in
#613). An allowlist, not a denylist. A denylist has to name every secret the
deployment will ever add (DB_ENCRYPTION_KEY, MCP_SERVICE_TOKEN,
AVALOKA_PII_KEY, each provider's *_API_KEY, the Supabase keys, every database
password, ...) and fails open the day someone adds one more. The allowlist
names what a pandas/daft script needs to start and fails closed: anything new
is withheld until someone decides it belongs here.

What is here and why:

* ``PATH``, ``HOME``, locale and ``TZ``: the interpreter and libraries resolve
  binaries, caches, encodings and timestamps through them. Callers point
  ``HOME`` at a private temp dir.
* ``PYTHON*`` switches that change only how the interpreter behaves (hash
  seed, unbuffered output, no .pyc writes, UTF-8 I/O).
* ``TMPDIR`` and ``MPLCONFIGDIR``: somewhere writable for tempfile and the
  matplotlib font cache. Callers point ``TMPDIR`` at a private temp dir.
* Thread-count caps and ``MALLOC_ARENA_MAX``, so a script uses the cores and
  arenas the deployment gave it.
* The third-party usage-beacon opt-outs below, set to "off", so a script that
  imports daft, mlflow, chromadb or boto does not phone home. They carry no
  host value unless the host set one.

1.6's generated scripts read no data-path variable: the executor bakes the
input and output paths into the script as literals
(``execution_agent._build_local_main_harness``, ``daft_execution``), so none
is allowed through.

Nothing that names a credential, a URL of an internal service, or a
connection is here, and ``tests/test_generated_code_env_scrub.py`` keeps it
that way.
"""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional

#: Third-party usage beacons and the cloud-metadata credential probe, off.
#: The subset of 2.0's ``app/core/thirdparty_optout.py`` a script can trigger.
THIRDPARTY_OPTOUT_ENV: Dict[str, str] = {
    "DO_NOT_TRACK": "true",                # daft; also mlflow, HF hub
    "SCARF_NO_ANALYTICS": "true",          # daft's Scarf gateway
    "ORT_DISABLE_TELEMETRY": "1",          # onnxruntime 1DS
    "ANONYMIZED_TELEMETRY": "False",       # chromadb
    "MLFLOW_DISABLE_TELEMETRY": "true",    # mlflow 3.x
    "HF_HUB_DISABLE_TELEMETRY": "1",       # huggingface_hub
    "RAY_USAGE_STATS_ENABLED": "0",        # ray
    "AWS_EC2_METADATA_DISABLED": "true",   # botocore: no 169.254.169.254 probe
}

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
    in. Every opt-out is defaulted on; a host that set one keeps its value.
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
