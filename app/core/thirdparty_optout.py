"""Third-party usage beacons and cloud-metadata probes, off before the library loads.

Several libraries the stack imports phone home unless told not to. None of
them sends customer data, but each is an outbound call nobody configured, from
every pod that loads it. Each variable below is that library's documented
switch:

* ``import daft`` calls ``scarf_telemetry.track_import_on_scarf()``, which
  sends daft's version, the platform, the Python version and the CPU
  architecture to ``daft.gateway.scarf.sh`` and ``osstelemetry.io``; every
  query sends a ``daft-runner`` event the same way. daft documents
  ``DO_NOT_TRACK=true``; ``daft.scarf_telemetry.opted_out()`` also honours
  ``SCARF_NO_ANALYTICS``.
* onnxruntime's official builds (1.31.0 here) start Microsoft's 1DS uploader
  when the ORT environment is created -- chroma's default embedding function
  does that on its first query -- and POST trace events to
  ``mobile.events.data.microsoft.com/OneCollector/1.0`` about 15 s later.
  onnxruntime's Privacy.md: "Set ``ORT_DISABLE_TELEMETRY=1`` before ONNX
  Runtime initializes". ``onnxruntime.disable_telemetry_events()`` is NOT
  enough: it is called after initialisation and the initialisation event has
  already been queued (measured: the connection still happens).
* chromadb's product telemetry: ``ANONYMIZED_TELEMETRY=False``.
* MLflow 3.x usage telemetry: ``MLFLOW_DISABLE_TELEMETRY=true`` (it also
  honours ``DO_NOT_TRACK``).
* huggingface_hub's usage pings: ``HF_HUB_DISABLE_TELEMETRY=1`` (also honours
  ``DO_NOT_TRACK``).
* DVC's own analytics daemon: ``DVC_NO_ANALYTICS=1`` (``dvc/analytics.py``
  reads only this; its iterative-telemetry dependency reads ``DO_NOT_TRACK``).
* Ray's cluster usage report to ``usage-stats.ray.io``:
  ``RAY_USAGE_STATS_ENABLED=0``.
* Streamlit's browser-side usage stats: ``STREAMLIT_BROWSER_GATHER_USAGE_STATS=false``.
* ``langgraph dev`` (langgraph-cli's ``@log_command``) POSTs the OS, Python and
  CLI version and the command's flags to a LangChain-owned
  ``kzrlppojinpcyyaipxnb.supabase.co/rest/v1/logs`` on every start:
  ``LANGGRAPH_CLI_NO_ANALYTICS=1``. The CLI starts before any app module, so
  this one only works from the image/compose/chart environment.
* langgraph-api's server asks ``pypi.org/pypi/langgraph-api/json`` for the
  latest version on every start (seen in the quickstart capture as a SYN from
  the langgraph container to a Fastly address (151.101.x.x) on :443): ``LANGGRAPH_NO_VERSION_CHECK=true``.

And one that is not telemetry but is the same kind of surprise:

* botocore, when it has no explicit credentials, walks its provider chain and
  asks the instance-metadata service at ``169.254.169.254`` for the VM's role.
  On a cloud VM that silently hands any S3 call the instance's credentials.
  ``AWS_EC2_METADATA_DISABLED=true`` removes that step; a deployment that
  wants instance credentials sets it to ``false`` (see ``CLOUD_METADATA_OPTOUT_ENV``).

``app/__init__.py`` calls ``disable_thirdparty_telemetry()``, which puts this
ahead of every import reached through the ``app`` package; each module that
imports daft calls it again, as does every image (Dockerfile ``ENV``), the
quickstart compose, the chart's ConfigMap and the Ray runtime_env a job is
submitted with.

Defaults, not overrides: a deployment that sets any of these itself keeps its
value, so opting back in is a deliberate act.
"""

from __future__ import annotations

import os
from typing import Dict

#: Every variable here, set to this value, disables a third-party usage beacon.
THIRDPARTY_TELEMETRY_OPTOUT_ENV: Dict[str, str] = {
    "DO_NOT_TRACK": "true",                          # daft; also mlflow, HF hub, iterative-telemetry
    "SCARF_NO_ANALYTICS": "true",                    # the Scarf gateway's own switch, also read by daft
    "ORT_DISABLE_TELEMETRY": "1",                    # onnxruntime 1DS -> events.data.microsoft.com
    "ANONYMIZED_TELEMETRY": "False",                 # chromadb product telemetry
    "MLFLOW_DISABLE_TELEMETRY": "true",              # mlflow 3.x usage telemetry
    "HF_HUB_DISABLE_TELEMETRY": "1",                 # huggingface_hub usage pings
    "DVC_NO_ANALYTICS": "1",                         # dvc analytics daemon
    "RAY_USAGE_STATS_ENABLED": "0",                  # Ray -> usage-stats.ray.io
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false", # streamlit browser usage stats
    "LANGGRAPH_CLI_NO_ANALYTICS": "1",               # `langgraph dev` -> a LangChain-owned supabase.co
    "LANGGRAPH_NO_VERSION_CHECK": "true",            # langgraph-api -> pypi.org on every server start
}

#: Cloud instance-metadata lookups nobody asked for. Not a beacon, but the
#: same default: off until the operator opts in.
CLOUD_METADATA_OPTOUT_ENV: Dict[str, str] = {
    "AWS_EC2_METADATA_DISABLED": "true",  # botocore: no 169.254.169.254 credential probe
}

#: Everything this module defaults, in one map (what the images, compose,
#: chart and Ray runtime_env carry).
THIRDPARTY_OPTOUT_ENV: Dict[str, str] = {
    **THIRDPARTY_TELEMETRY_OPTOUT_ENV,
    **CLOUD_METADATA_OPTOUT_ENV,
}


def disable_thirdparty_telemetry() -> Dict[str, str]:
    """Default the opt-out variables in this process; returns them as now set.

    Idempotent and cheap, so every module that loads one of these libraries
    can call it first. A child process (a generated script, a Ray worker on
    this node) inherits them from ``os.environ``. The returned map carries the
    deployment's own value wherever it set one, so passing it on (a Ray
    runtime_env) propagates an opt-in rather than overriding it.
    """
    for name, value in THIRDPARTY_OPTOUT_ENV.items():
        os.environ.setdefault(name, value)
    return {name: os.environ[name] for name in THIRDPARTY_OPTOUT_ENV}
