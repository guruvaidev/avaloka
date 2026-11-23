"""Third-party beacons and cloud-metadata probes are never attempted from our code.

daft: ``import daft`` sends daft's version, platform, Python and arch to
``daft.gateway.scarf.sh`` (and ``osstelemetry.io``) from a daemon thread,
unless ``DO_NOT_TRACK`` / ``SCARF_NO_ANALYTICS`` is set before the import.
onnxruntime: its 1DS uploader posts to ``mobile.events.data.microsoft.com``
unless ``ORT_DISABLE_TELEMETRY=1`` is set before ORT initialises.
botocore: with no credentials it asks ``169.254.169.254`` for the instance's.
``app/core/thirdparty_optout.py`` defaults every switch, and ``app/__init__.py``
runs it.

Each case runs in a fresh interpreter with the opt-out variables stripped from
its environment and a recorder that refuses every outbound attempt (nothing
reaches the network) and reports what was tried. Each group has a control case
without the opt-out that must show the call, or the recorder is not seeing what
it is meant to and the other cases would pass vacuously.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BEACON_HOSTS = ("daft.gateway.scarf.sh", "osstelemetry.io")
OPTOUT_VARS = ("DO_NOT_TRACK", "SCARF_NO_ANALYTICS", "DAFT_ANALYTICS_ENABLED")  # daft cases

needs_daft = pytest.mark.skipif(
    importlib.util.find_spec("daft") is None, reason="daft is not installed"
)

_RECORDER = textwrap.dedent(
    """
    import json, socket, sys, threading

    tried = []

    def _refuse(host):
        tried.append(str(host))
        raise OSError("network disabled by test")

    _real_getaddrinfo = socket.getaddrinfo
    socket.getaddrinfo = lambda host, *a, **k: _refuse(host)
    socket.create_connection = lambda address, *a, **k: _refuse(address[0])
    _real_connect = socket.socket.connect
    def _connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _refuse(address[0])
        return _real_connect(self, address)
    socket.socket.connect = _connect

    exec(compile(sys.argv[1], "<case>", "exec"))

    for t in threading.enumerate():
        if t is not threading.current_thread():
            t.join(timeout=10)
    print("TRIED=" + json.dumps(tried))
    """
)


def _hosts_tried(case: str) -> list:
    env = {k: v for k, v in os.environ.items() if k not in OPTOUT_VARS}
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _RECORDER, case],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("TRIED="))
    return json.loads(line[len("TRIED="):])


def _beacons(hosts: list) -> list:
    return [h for h in hosts if any(b in h for b in BEACON_HOSTS)]


@needs_daft
def test_control_bare_import_daft_attempts_the_beacon():
    """Without the opt-out, the recorder sees daft phone home -- so the cases below can fail."""
    tried = _hosts_tried("import daft")
    if not _beacons(tried) and "dev" in _hosts_tried(
        "import os\nos.environ['DO_NOT_TRACK'] = 'true'\nimport daft\ntried.append(daft.get_build_type())"
    ):
        pytest.skip("a dev build of daft sends no beacon; nothing to control against")
    assert _beacons(tried), f"recorder saw no beacon from a bare import: {tried}"


@needs_daft
@pytest.mark.parametrize("case", [
    # Any module of ours: the app package disables the beacon first.
    "import app\nimport daft",
    # The helper, as each daft-importing module calls it.
    "from app.core.thirdparty_optout import disable_thirdparty_telemetry\n"
    "disable_thirdparty_telemetry()\nimport daft",
    # The DTA's sink, which imports daft at module level.
    "import app.agents.data_transfer_agent.datasink.sql_sink",
    # The sampling agent's lazy daft, on first use.
    "import app.agents.sampling_agent_daft as s\ns.daft.from_pydict({'a': [1]})",
])
def test_no_beacon_from_our_imports(case):
    tried = _hosts_tried(case)
    assert not _beacons(tried), f"daft usage beacon attempted: {_beacons(tried)}"


def test_opt_out_is_a_default_not_an_override(monkeypatch):
    from app.core.thirdparty_optout import THIRDPARTY_OPTOUT_ENV, disable_thirdparty_telemetry

    for name in THIRDPARTY_OPTOUT_ENV:
        monkeypatch.delenv(name, raising=False)
    assert disable_thirdparty_telemetry() == THIRDPARTY_OPTOUT_ENV
    monkeypatch.setenv("DO_NOT_TRACK", "false")
    assert disable_thirdparty_telemetry()["DO_NOT_TRACK"] == "false"
    # An operator who wants the instance's own credentials says so, and keeps it.
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "false")
    assert disable_thirdparty_telemetry()["AWS_EC2_METADATA_DISABLED"] == "false"


def test_every_documented_switch_is_in_the_helper():
    """The switches this suite and the deploy files rely on are all in one map."""
    from app.core.thirdparty_optout import THIRDPARTY_OPTOUT_ENV

    assert THIRDPARTY_OPTOUT_ENV == {
        "DO_NOT_TRACK": "true",
        "SCARF_NO_ANALYTICS": "true",
        "ORT_DISABLE_TELEMETRY": "1",
        "ANONYMIZED_TELEMETRY": "False",
        "MLFLOW_DISABLE_TELEMETRY": "true",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DVC_NO_ANALYTICS": "1",
        "RAY_USAGE_STATS_ENABLED": "0",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "LANGGRAPH_CLI_NO_ANALYTICS": "1",
        "LANGGRAPH_NO_VERSION_CHECK": "true",
        "AWS_EC2_METADATA_DISABLED": "true",
    }


# ---------------------------------------------------------------------------
# onnxruntime's 1DS telemetry -> mobile.events.data.microsoft.com
#
# onnxruntime's official builds start Microsoft's 1DS uploader when the ORT
# environment is created and POST to mobile.events.data.microsoft.com about
# 15 s later. The uploader is C++ (libcurl), so the Python socket recorder
# above never sees it. libcurl does honour HTTPS_PROXY, so the recorder here is
# a local proxy: every child gets HTTPS_PROXY pointing at it, it records the
# CONNECT line and answers 403. Nothing reaches the network, and a DNS lookup
# for the collector would only happen after the proxy said yes, which it never
# does. The control case (no opt-out) must show the CONNECT, or the recorder is
# blind and the opt-out cases would pass vacuously.
# ---------------------------------------------------------------------------

ORT_COLLECTOR = "events.data.microsoft.com"
ORT_WAIT_S = float(os.environ.get("AVALOKA_ORT_TELEMETRY_WAIT_S", "35"))

# A one-node ONNX model built in memory: any InferenceSession initialises the
# ORT environment, which is what starts the uploader. chroma's ONNX MiniLM is
# loaded too when its model is on disk (the API image bakes it).
_ORT_LOAD = textwrap.dedent(
    """
    import os, time
    import numpy as np
    import onnxruntime as ort
    from onnx import TensorProto, helper
    v = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1])
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [v("x")], [v("y")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    s.run(None, {"x": np.ones(1, dtype=np.float32)})
    if os.environ.get("CHROMA_ONNX_MODEL_DIR") and os.path.isdir(os.environ["CHROMA_ONNX_MODEL_DIR"]):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        ef = ONNXMiniLM_L6_V2()
        ef.DOWNLOAD_PATH = os.environ["CHROMA_ONNX_MODEL_DIR"]
        ef(["the model chroma embeds with"])
    time.sleep(float(os.environ["ORT_CASE_WAIT_S"]))
    """
)

ORT_CASES = {
    # Control: no opt-out anywhere. Must reach the recorder.
    "bare": _ORT_LOAD,
    # The non-fix the evidence ruled out, kept so nobody "simplifies" to it.
    "api_call_only": _ORT_LOAD.replace(
        "import onnxruntime as ort\n",
        "import onnxruntime as ort\nort.disable_telemetry_events()\n",
    ),
    # Any module of ours: app/__init__ runs the helper before onnxruntime loads.
    "via_app_package": "import app\n" + _ORT_LOAD,
    # The helper, as a module that loads onnxruntime or chroma would call it.
    "via_helper": (
        "from app.core.thirdparty_optout import disable_thirdparty_telemetry\n"
        "disable_thirdparty_telemetry()\n" + _ORT_LOAD
    ),
}


def _ort_run_all() -> dict:
    """Run every case concurrently behind one recording proxy; CONNECT lines per case."""
    import socket
    import threading
    import time

    servers, seen = {}, {name: [] for name in ORT_CASES}

    def _serve(srv, name):
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            try:
                conn.settimeout(5)
                first = conn.recv(4096).decode("latin-1", "replace").splitlines()
                seen[name].append(first[0] if first else "<empty>")
                conn.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            except OSError:
                pass
            finally:
                conn.close()

    procs = {}
    for name, case in ORT_CASES.items():
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(16)
        servers[name] = srv
        threading.Thread(target=_serve, args=(srv, name), daemon=True).start()
        proxy = f"http://127.0.0.1:{srv.getsockname()[1]}"
        env = {k: v for k, v in os.environ.items() if k not in ("ORT_DISABLE_TELEMETRY", "NO_PROXY", "no_proxy")}
        env.update(
            HTTPS_PROXY=proxy, https_proxy=proxy, HTTP_PROXY=proxy, http_proxy=proxy, ALL_PROXY=proxy,
            ORT_CASE_WAIT_S=str(ORT_WAIT_S),
            PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        )
        procs[name] = subprocess.Popen(
            [sys.executable, "-c", case], cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    for name, proc in procs.items():
        _, err = proc.communicate(timeout=ORT_WAIT_S + 240)
        assert proc.returncode == 0, f"{name}: {err[-2000:]}"
    time.sleep(0.5)
    for srv in servers.values():
        srv.close()
    return {name: list(seen[name]) for name in ORT_CASES}


@pytest.fixture(scope="module")
def ort_connects():
    if importlib.util.find_spec("onnxruntime") is None or importlib.util.find_spec("onnx") is None:
        pytest.skip("onnxruntime/onnx is not installed")
    return _ort_run_all()


def _to_collector(lines: list) -> list:
    return [ln for ln in lines if ORT_COLLECTOR in ln]


def test_control_bare_onnxruntime_reaches_microsofts_collector(ort_connects):
    """Without the opt-out, ORT's 1DS uploader asks for the collector -- the recorder can see it."""
    import onnxruntime

    if not _to_collector(ort_connects["bare"]):
        pytest.skip(
            f"onnxruntime {onnxruntime.__version__} sent no 1DS event in {ORT_WAIT_S:.0f}s "
            "(a build without telemetry); nothing to control against"
        )
    assert _to_collector(ort_connects["bare"])


def test_disable_telemetry_events_alone_does_not_stop_it(ort_connects):
    """Documents why the fix is an env var: the Python API is called too late."""
    if not _to_collector(ort_connects["bare"]):
        pytest.skip("no control signal")
    assert _to_collector(ort_connects["api_call_only"]), (
        "ort.disable_telemetry_events() now stops the 1DS upload on its own; "
        "the env-var opt-out is still right, but update the docstring in "
        "app/core/thirdparty_optout.py"
    )


@pytest.mark.parametrize("case", ["via_app_package", "via_helper"])
def test_no_onnxruntime_telemetry_from_our_imports(ort_connects, case):
    if not _to_collector(ort_connects["bare"]):
        pytest.skip("no control signal: this onnxruntime build sends nothing to suppress")
    assert not _to_collector(ort_connects[case]), (
        f"onnxruntime telemetry attempted ({case}): {ort_connects[case]}"
    )


# ---------------------------------------------------------------------------
# The DTA's sysdocs: no object store, and no instance-metadata probe, unless
# the operator configured one.
# ---------------------------------------------------------------------------

IMDS = "169.254.169.254"
_SYSDOCS_ENV = (
    "AVALOKA_SYSDOCS_PATH", "AVALOKA_SYSDOCS_URI", "AWS_EC2_METADATA_DISABLED",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
    "AWS_ENDPOINT_URL", "S3_ENDPOINT_URL", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD",
    "AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE",
)
# What a transfer does with the validator: it imports it (the module-level load
# runs there) and the loader is called again on reload.
_TRANSFER_LOAD = (
    "import app.agents.data_transfer_agent.daft_validator as v\n"
    "tried.append('DOCS=%d' % len(v.load_sysdocs_from_gcs()))\n"
)


def _hosts_tried_env(case: str, **env_overrides: str) -> list:
    env = {k: v for k, v in os.environ.items() if k not in _SYSDOCS_ENV}
    env.update(env_overrides)
    # No ~/.aws on the test box may supply credentials and mask the probe.
    env["AWS_SHARED_CREDENTIALS_FILE"] = env["AWS_CONFIG_FILE"] = os.devnull
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _RECORDER, case],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("TRIED="))
    return json.loads(line[len("TRIED="):])


needs_s3fs = pytest.mark.skipif(
    importlib.util.find_spec("s3fs") is None or importlib.util.find_spec("gcsfs") is None,
    reason="s3fs/gcsfs is not installed",
)


@needs_s3fs
def test_control_configured_sysdocs_without_credentials_probes_imds():
    """An s3:// sysdocs path with no credentials and no opt-out asks the metadata service."""
    tried = _hosts_tried_env(
        _TRANSFER_LOAD,
        AVALOKA_SYSDOCS_PATH="s3://avaloka/dta_md_files/sysdocs_markdown",
        AWS_EC2_METADATA_DISABLED="false",
    )
    assert IMDS in tried, f"recorder saw no IMDS probe: {tried}"


@needs_s3fs
def test_transfer_without_sysdocs_contacts_no_object_store_and_no_imds():
    """Sysdocs not configured: the validator loads nothing and opens no connection at all."""
    # The IMDS switch is turned OFF on purpose: the sysdocs gate alone must keep
    # the transfer off the network, whatever AWS_EC2_METADATA_DISABLED says.
    tried = _hosts_tried_env(_TRANSFER_LOAD, AWS_EC2_METADATA_DISABLED="false")
    hosts = [h for h in tried if not h.startswith("DOCS=")]
    assert "DOCS=0" in tried
    assert IMDS not in hosts and not any("amazonaws" in h or "avaloka" in h for h in hosts), hosts


@needs_s3fs
def test_configured_sysdocs_never_probe_imds_by_default():
    """With sysdocs configured, the app's default still keeps botocore off 169.254.169.254."""
    tried = _hosts_tried_env(
        "import app\n" + _TRANSFER_LOAD,
        AVALOKA_SYSDOCS_PATH="s3://avaloka/dta_md_files/sysdocs_markdown",
    )
    assert IMDS not in tried, f"IMDS probed despite AWS_EC2_METADATA_DISABLED default: {tried}"


# ---------------------------------------------------------------------------
# The deploy surfaces carry the same switches. A beacon that fires before any
# app module loads (langgraph-cli, the Ray head, the chroma/mlflow servers)
# is only silenced from the environment, so the helper alone is not enough.
# ---------------------------------------------------------------------------

def _python_dockerfiles() -> list:
    skip = {".git", "node_modules", ".venv", "venv"}
    out = [
        p.relative_to(REPO_ROOT).as_posix()
        for p in REPO_ROOT.rglob("*Dockerfile*")
        if p.is_file() and not skip.intersection(p.relative_to(REPO_ROOT).parts)
    ]
    return [p for p in out if "DO_NOT_TRACK" in (REPO_ROOT / p).read_text("utf-8", "ignore")]


def test_every_python_image_sets_every_switch():
    from app.core.thirdparty_optout import THIRDPARTY_OPTOUT_ENV

    files = _python_dockerfiles()
    assert "deploy/docker/Dockerfile.api" in files and "deploy/docker/Dockerfile.ray" in files
    missing = {}
    for rel in files:
        text = (REPO_ROOT / rel).read_text("utf-8")
        gaps = [k for k, v in THIRDPARTY_OPTOUT_ENV.items() if f"{k}={v}" not in text]
        if gaps:
            missing[rel] = gaps
    assert not missing, f"images missing opt-out ENV: {missing}"


@pytest.mark.parametrize("rel", [
    "docker-compose.quickstart.yml",
    "deploy/helm/avaloka/templates/configmap.yaml",
    "deploy/helm/avaloka/templates/mcp.yaml",
])
def test_compose_and_chart_set_every_switch(rel):
    from app.core.thirdparty_optout import THIRDPARTY_OPTOUT_ENV

    text = (REPO_ROOT / rel).read_text("utf-8")
    # The chart keeps the one list in _helpers.tpl (avaloka.thirdPartyOptout) and
    # each template includes it; the rendered result is held to the same values
    # by tests/contract/test_thirdparty_telemetry_optout.py. Here: the template
    # includes the list, and the list is complete.
    if rel.startswith("deploy/helm/avaloka/templates/"):
        assert 'include "avaloka.thirdPartyOptout' in text, f"{rel} does not include avaloka.thirdPartyOptout"
        text += (REPO_ROOT / "deploy/helm/avaloka/templates/_helpers.tpl").read_text("utf-8")
    gaps = []
    for k, v in THIRDPARTY_OPTOUT_ENV.items():
        as_map = f'{k}: "{v}"' in text
        as_list = re.search(rf'- name: {k}\s*\n\s*value: "{re.escape(v)}"', text)
        # The chart's IMDS switch is a template on config.instanceMetadataCredentials.
        templated = k == "AWS_EC2_METADATA_DISABLED" and re.search(
            rf'{k}(?::|\s*\n\s*value:) {{{{ ternary "false" "true" \(\.Values\.config\.instanceMetadataCredentials', text)
        if not (as_map or as_list or templated):
            gaps.append(k)
    assert not gaps, f"{rel} is missing: {gaps}"
