"""Ray token authentication on the app's side (RAY_AUTH_MODE=token, Ray >= 2.52).

Ray's own clients need nothing from here: ``ray.init`` with a ``ray://``
address and
``JobSubmissionClient`` read ``RAY_AUTH_MODE`` and ``RAY_AUTH_TOKEN`` (or
``RAY_AUTH_TOKEN_PATH``) from the environment and send the token themselves.
This module covers what Ray does not:

* the app's plain HTTP calls to a Ray dashboard (``/api/version``,
  ``/api/jobs/...``), which must carry ``Authorization: Bearer <token>``;
* KubeRay-mode RayJobs, whose pods and whose KubeRay ``authOptions`` read the
  token from the job's own Secret under ``auth_token``.

The token is never logged, and never returned in anything but a header or the
job Secret's data.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Mapping, Optional

#: Key KubeRay's ``authOptions.secretName`` requires, and the key the Ray pods
#: read by secretKeyRef (deploy/helm/ray/*.yaml, app/infra/rayjob*.yaml).
RAYJOB_AUTH_KEY = "auth_token"


class RayAuthTokenMissing(RuntimeError):
    """RAY_AUTH_MODE=token is set but no token is available."""


def auth_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return (env.get("RAY_AUTH_MODE") or "").strip().lower() == "token"


def auth_token(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The token Ray itself would use: RAY_AUTH_TOKEN, else RAY_AUTH_TOKEN_PATH's file."""
    env = os.environ if environ is None else environ
    token = (env.get("RAY_AUTH_TOKEN") or "").strip()
    if token:
        return token
    path = (env.get("RAY_AUTH_TOKEN_PATH") or "").strip()
    if path:
        try:
            token = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
    return token or None


def require_token(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The token when auth is on (raising if there is none); None when auth is off."""
    if not auth_enabled(environ):
        return None
    token = auth_token(environ)
    if not token:
        raise RayAuthTokenMissing(
            "RAY_AUTH_MODE=token but neither RAY_AUTH_TOKEN nor RAY_AUTH_TOKEN_PATH "
            "provides a token. In Kubernetes it comes from the release Secret "
            "(the chart's ray.auth); elsewhere set RAY_AUTH_TOKEN."
        )
    return token


def auth_headers(environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """``{"Authorization": "Bearer <token>"}`` when token auth is on, else ``{}``.

    Raises RayAuthTokenMissing when auth is on and there is no token, so a
    request is not sent unauthenticated only to be refused with a vaguer error.
    """
    token = require_token(environ)
    return {"Authorization": f"Bearer {token}"} if token else {}


def with_job_auth(secret_env: Optional[Mapping[str, str]],
                  environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """A KubeRay-mode job's Secret data, plus the Ray token when auth is on.

    The RayJob's pods read it by secretKeyRef (key ``auth_token``) and its
    ``authOptions.secretName`` names the same Secret, so the job's Ray cluster
    enforces the token and the KubeRay operator can still read its status.
    """
    data = {str(k): str(v) for k, v in (secret_env or {}).items() if v}
    token = require_token(environ)
    if token:
        data[RAYJOB_AUTH_KEY] = token
    return data


_AUTH_BLOCK_RE = re.compile(r"(?ms)^[ \t]*# ray-auth:begin\b.*?^[ \t]*# ray-auth:end[ \t]*\n")


def strip_job_auth(yaml_text: str) -> str:
    """The RayJob YAML without its ``# ray-auth:begin`` ... ``# ray-auth:end`` blocks.

    The shipped templates (app/infra/rayjob.yaml, rayjob_data_transfer.yaml)
    turn token auth on. When the app itself runs without RAY_AUTH_MODE=token
    (the chart's ray.auth.enabled=false) the job's Secret has no token to give,
    and a job that demands one would never start; the runner removes the blocks
    instead.
    """
    return _AUTH_BLOCK_RE.sub("", yaml_text)
