"""Every local model Avaloka ships as a default must carry a permissive licence.

The default local model was ``qwen2.5:3b-instruct`` until 2026-11-01. Its weights
are under the Qwen RESEARCH licence (non-commercial), which a commercial edition
must not default to. Nothing caught it, because nothing recorded what licence a
model tag carries. This file does: every tag that ships as a default -- in code,
chart values, the inference manifests, the example env and (where present) the
quickstart -- must appear in ``LOCAL_MODEL_LICENCES`` with its licence beside it,
and that licence must be one of ``PERMITTED_LICENCES``.

Adding a model: verify the licence on the model card (Hugging Face ``license``
field and LICENSE file) AND on the tag you actually pull (for Ollama, the
``license`` layer of ``registry.ollama.ai/v2/library/<name>/manifests/<tag>``),
then add the tag here. A tag is a pointer to weights; record the tag that names
the weights you checked, not a floating alias.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

#: Licences a shipped default may carry.
PERMITTED_LICENCES = frozenset({"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause"})

#: Tag (as served) -> licence of the weights it names. Evidence, 2026-11-01:
#:  qwen3:4b-instruct-2507-q4_K_M  HF Qwen/Qwen3-4B-Instruct-2507 license: apache-2.0;
#:      Ollama licence layer sha256:d18a5cc7... is the Apache License 2.0 text.
#:  Qwen/Qwen2.5-14B-Instruct      HF license: apache-2.0 (vLLM GPU default).
#:  gemma4:e4b                     HF google/gemma-4-E4B-it license: apache-2.0;
#:      Ollama licence layer sha256:7339fa41... is the Apache License 2.0 text.
#:  qwen2.5:3b-instruct            HF license: other (Qwen RESEARCH LICENSE
#:      AGREEMENT); Ollama licence layer sha256:b5c0e5cf... is the same text.
#:      Kept here so the reason it may not ship is on record, not forgotten.
#:  google/gemma-4-31b-it          HF google/gemma-4-31B-it license: apache-2.0
#:      (model_fallback cluster default).
#:  qwen/qwen3.8-27b               HF Qwen/Qwen3.8-27B license: apache-2.0
#:      (model_fallback cluster alternate).
#:  google/gemma-3-4b-it           HF license: gemma (Gemma Terms of Use). The
#:      laptop fallback default until 2026-11-01; recorded so it cannot return.
LOCAL_MODEL_LICENCES = {
    "qwen3:4b-instruct-2507-q4_K_M": "Apache-2.0",
    "Qwen/Qwen2.5-14B-Instruct": "Apache-2.0",
    "gemma4:e4b": "Apache-2.0",
    "qwen2.5:3b-instruct": "Qwen-Research",
    "google/gemma-4-31b-it": "Apache-2.0",
    "qwen/qwen3.8-27b": "Apache-2.0",
    "google/gemma-3-4b-it": "Gemma-Terms-of-Use",
}

#: The vLLM manifest's client-facing alias, not a model: the weights behind it
#: are the ``--model`` repo, which is checked instead.
_SERVED_ALIASES = frozenset({"qwen-local"})


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _env_assignments(text: str, name_re: str) -> list[str]:
    """Values of NAME=value lines, commented out or not (examples count)."""
    pat = re.compile(rf"^[#\s]*(?:{name_re})=([^\s#]+)", re.MULTILINE)
    return pat.findall(text)


def _fallback_defaults() -> dict[str, str]:
    """The built-in defaults of model_fallback's LOCAL_MODEL_* (the env-free value).

    Read from the source, not the imported constants, so an AVALOKA_LOCAL_MODEL_*
    set in the environment cannot hide what ships.
    """
    from app.core import model_fallback

    found: dict[str, str] = {}
    for node in ast.parse(_read("app/core/model_fallback.py")).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith("LOCAL_MODEL_")):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and len(call.args) == 2):
            continue
        default = call.args[1]
        if isinstance(default, ast.Constant):
            found[node.targets[0].id] = default.value
        elif isinstance(default, ast.Name):
            found[node.targets[0].id] = getattr(model_fallback, default.id)
    return found


def _shipped_local_models() -> dict[str, str]:
    """Every default local model tag we ship, keyed by where it ships."""
    from app.core import inference

    found: dict[str, str] = {}
    for tier, tag in inference._PROVIDER_MODEL_DEFAULTS[inference.LOCAL].items():
        found[f"app/core/inference.py LOCAL[{tier}]"] = tag

    values = yaml.safe_load(_read("deploy/helm/avaloka/values.yaml"))["localLLM"]
    found["values.yaml localLLM.model"] = values["model"]
    found["values.yaml localLLM.servedModelName"] = values["servedModelName"]

    for doc in yaml.safe_load_all(_read("deploy/inference/ollama-cpu.yaml")):
        if (doc or {}).get("kind") != "Deployment":
            continue
        for c in doc["spec"]["template"]["spec"]["containers"]:
            for e in c.get("env", []):
                if e["name"] == "MODEL":
                    found["ollama-cpu.yaml env MODEL"] = e["value"]
    for i, tag in enumerate(_env_assignments(
            _read("deploy/inference/ollama-cpu.yaml"), r"INFERENCE_LOCAL_MODEL_\w+")):
        found[f"ollama-cpu.yaml example env #{i}"] = tag

    for doc in yaml.safe_load_all(_read("deploy/inference/vllm-gpu.yaml")):
        if (doc or {}).get("kind") != "Deployment":
            continue
        for c in doc["spec"]["template"]["spec"]["containers"]:
            args = c.get("args", [])
            if "--model" in args:
                found["vllm-gpu.yaml --model"] = args[args.index("--model") + 1]

    for i, tag in enumerate(_env_assignments(
            _read(".env.example"), r"INFERENCE_LOCAL_MODEL(?:_\w+)?")):
        found[f".env.example INFERENCE_LOCAL_MODEL* #{i}"] = tag

    for name, tag in _fallback_defaults().items():
        found[f"model_fallback.py {name}"] = tag

    from avaloka.benchmark.conversational import LOCAL_MODEL_PREFERENCE
    found["conversational.py LOCAL_MODEL_PREFERENCE[0]"] = LOCAL_MODEL_PREFERENCE[0]

    # Quickstart (2.0 and later): the compose default and the generated env.
    if (ROOT / "docker-compose.quickstart.yml").exists():
        m = re.search(r"INFERENCE_LOCAL_MODEL:\s*\$\{INFERENCE_LOCAL_MODEL:-([^}]+)\}",
                      _read("docker-compose.quickstart.yml"))
        assert m, "docker-compose.quickstart.yml no longer defaults INFERENCE_LOCAL_MODEL"
        found["docker-compose.quickstart.yml INFERENCE_LOCAL_MODEL"] = m.group(1)
    if (ROOT / "scripts/quickstart.sh").exists():
        qs = _read("scripts/quickstart.sh")
        for i, tag in enumerate(re.findall(r"INFERENCE_LOCAL_MODEL=([\w.:/-]+)", qs)):
            found[f"quickstart.sh INFERENCE_LOCAL_MODEL #{i}"] = tag
        for i, tag in enumerate(re.findall(r"model:-([\w.:/-]+)\}", qs)):
            found[f"quickstart.sh model default #{i}"] = tag

    return {k: v for k, v in found.items() if v not in _SERVED_ALIASES}


def test_the_scan_finds_the_defaults_it_is_meant_to_guard():
    """A scan that silently finds nothing would pass every licence check."""
    shipped = _shipped_local_models()
    for where in ("app/core/inference.py LOCAL[large]", "app/core/inference.py LOCAL[small]",
                  "values.yaml localLLM.model", "ollama-cpu.yaml env MODEL",
                  "vllm-gpu.yaml --model", "conversational.py LOCAL_MODEL_PREFERENCE[0]",
                  "model_fallback.py LOCAL_MODEL_LAPTOP", "model_fallback.py LOCAL_MODEL_CLUSTER",
                  "model_fallback.py LOCAL_MODEL_CLUSTER_ALT"):
        assert where in shipped, f"the licence scan no longer reads {where}"


def test_every_shipped_local_model_has_a_permitted_licence():
    problems = []
    for where, tag in sorted(_shipped_local_models().items()):
        licence = LOCAL_MODEL_LICENCES.get(tag)
        if licence is None:
            problems.append(f"{where} ships {tag!r}, which has no recorded licence: verify "
                            f"the model card and the pulled tag, then add it to "
                            f"LOCAL_MODEL_LICENCES")
        elif licence not in PERMITTED_LICENCES:
            problems.append(f"{where} ships {tag!r} under {licence}, not one of "
                            f"{sorted(PERMITTED_LICENCES)}")
    assert not problems, "\n".join(problems)


def test_the_default_local_model_is_one_name_everywhere():
    """Code, chart, manifest and benchmark name the same small default."""
    shipped = _shipped_local_models()
    small = {k: v for k, v in shipped.items()
             if not k.startswith(("vllm-gpu.yaml", "docker-compose.quickstart", "quickstart.sh",
                                  "model_fallback.py LOCAL_MODEL_CLUSTER"))}
    assert len(set(small.values())) == 1, f"the default local model has drifted: {small}"
