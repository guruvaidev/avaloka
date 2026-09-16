import re
from pathlib import Path
from typing import Dict


def _sanitize_k8s_name(name: str, max_len: int = 63) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s:
        s = "rayjob"
    return (s[:max_len].strip("-")) or "rayjob"


def render_rayjob_yaml_from_file(
    template_path: str,
    *,
    rayjob_name: str,          # ex: "rayjob-sample-3w"
    ray_namespace: str,        # ex: "ray-jobs"
    data_source_uri: str,      # ex: "s3://bucket/key.csv"
    cloud_secret_name: str,    # ex: "avaloka-raycreds-..."
) -> str:
    p = Path(template_path)
    text = p.read_text(encoding="utf-8")

    name = _sanitize_k8s_name(rayjob_name)
    ns = (ray_namespace or "").strip() or "default"

    data_source_uri = (data_source_uri or "").strip()
    if data_source_uri and "://" not in data_source_uri:
        raise ValueError(f"Invalid data_source_uri: {data_source_uri!r}")

    cloud_secret_name = (cloud_secret_name or "").strip()
    if not cloud_secret_name:
        raise ValueError("cloud_secret_name is required")


    replacements: Dict[str, str] = {
        "__RAYJOB_NAME__": name,
        "__RAY_NAMESPACE__": ns,
        "__DATA_SOURCE_URI__": data_source_uri,
        "__CLOUD_SECRET_NAME__": cloud_secret_name,
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    leftovers = re.findall(r"__([A-Z0-9_]+)__", text)
    if leftovers:
        raise ValueError(f"Template render incomplete; leftover placeholders: {sorted(set(leftovers))}")

    return text
