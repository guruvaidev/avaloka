"""Offline mode: the embedding paths make no network call, and fail plainly.

A trial of the web stack counted 409 requests to huggingface.co from a server
whose model was already baked in, a 670 MB model download in the middle of a
data transfer, and chroma fetching its ONNX model from
chroma-onnx-models.s3.amazonaws.com on the first query. The image now bakes
all three and sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
(deploy/docker/Dockerfile.api, app/core/offline.py).

Each case runs in a fresh interpreter (huggingface_hub reads its offline flag
at import) with every name lookup and connection recorded and refused, and
with EMPTY model caches: the worst case, where a download is the only way to
get the model. Offline, the code must not try; it must raise ModelNotCached
saying which model and what to do.

The control cases run the same code with the offline flags removed and
assert the recorder DOES see huggingface.co and the chroma bucket. Without
them, a recorder that never fired would pass the offline cases too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("chromadb")

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("huggingface.co", "hf.co", "chroma-onnx-models.s3.amazonaws.com")

_PROBE = textwrap.dedent(r'''
    import json, socket, sys
    seen = []
    def _deny(host):
        seen.append(str(host))
        raise OSError("network disabled by test_offline_egress")
    _real_gai = socket.getaddrinfo
    def getaddrinfo(host, *a, **k):
        if str(host) in ("localhost", "127.0.0.1", "::1"):
            return _real_gai(host, *a, **k)
        _deny(host)
    def create_connection(addr, *a, **k):
        _deny(addr[0])
    socket.getaddrinfo = getaddrinfo
    socket.create_connection = create_connection

    out = {"errors": {}}
    case = sys.argv[1]
    try:
        if case == "memory":
            from app.services import embedding_utils
            embedding_utils._st_instance = None
            embedding_utils._get_st_model("all-MiniLM-L6-v2")
        elif case == "dta":
            from app.core.offline import sentence_transformer_ef
            sentence_transformer_ef("mixedbread-ai/mxbai-embed-large-v1", normalize_embeddings=True)
        elif case == "chroma":
            from app.core.offline import chroma_default_ef
            chroma_default_ef()(["hello"])
        out["ok"] = True
    except Exception as exc:
        out["errors"][case] = f"{type(exc).__name__}: {exc}"
    out["seen"] = seen
    print("RESULT " + json.dumps(out))
''')


def _run(case: str, tmp_path: Path, offline: bool) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "AVALOKA_OFFLINE",
                        "HF_HOME", "CHROMA_ONNX_MODEL_DIR", "SENTENCE_TRANSFORMERS_HOME")}
    env.update({
        "HOME": str(tmp_path / "home"),
        "HF_HOME": str(tmp_path / "hf"),            # empty: nothing cached
        "CHROMA_ONNX_MODEL_DIR": str(tmp_path / "onnx"),
        "ANONYMIZED_TELEMETRY": "False",
        "PYTHONPATH": str(ROOT),
        "HF_HUB_ETAG_TIMEOUT": "2", "HF_HUB_DOWNLOAD_TIMEOUT": "2",
    })
    if offline:
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE)
    proc = subprocess.run([sys.executable, str(probe), case], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=240)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, f"probe printed no result (rc={proc.returncode}):\n{proc.stderr[-3000:]}"
    return json.loads(line[len("RESULT "):])


def _forbidden(seen):
    return [h for h in seen if any(h == f or h.endswith("." + f) for f in FORBIDDEN)]


@pytest.mark.parametrize("case", ["memory", "dta", "chroma"])
def test_offline_makes_no_download_attempt_and_fails_plainly(case, tmp_path):
    out = _run(case, tmp_path, offline=True)
    assert _forbidden(out["seen"]) == [], f"offline {case} path tried the network: {out['seen']}"
    assert out["seen"] == [], f"offline {case} path looked up hosts: {out['seen']}"
    err = out["errors"].get(case, "")
    assert err.startswith("ModelNotCached"), f"expected a plain ModelNotCached, got {err!r}"
    assert "prefetch_models.py" in err


@pytest.mark.parametrize("case,host", [
    ("memory", "huggingface.co"),
    ("chroma", "chroma-onnx-models.s3.amazonaws.com"),
])
def test_control_without_offline_flags_the_recorder_sees_the_download(case, host, tmp_path):
    out = _run(case, tmp_path, offline=False)
    assert any(h == host or h.endswith("." + host) for h in out["seen"]), (
        f"the recorder saw {out['seen']} for {case} with downloads allowed; if it cannot see "
        f"{host} here, the offline assertions above prove nothing"
    )


def test_api_image_is_offline_by_default():
    """The image, not just the compose file, carries the offline flags and the models."""
    text = (ROOT / "deploy" / "docker" / "Dockerfile.api").read_text()
    runtime = text.split("AS runtime", 1)[1]
    for flag in ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "CHROMA_ONNX_MODEL_DIR="):
        assert flag in runtime, f"runtime stage lacks {flag}"
    assert "prefetch_models.py" in text.split("AS runtime", 1)[0], "builder stage does not prefetch"
    assert "COPY --from=builder /opt/chroma /opt/chroma" in runtime
