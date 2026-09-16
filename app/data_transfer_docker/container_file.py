import time
import json
import sys
import traceback
import os
import importlib.util

import ray
import daft


def log_event(status, message, extra=None):
    log_entry = {"timestamp": time.time(), "status": status, "message": message}
    if extra:
        log_entry.update(extra)
    print(json.dumps(log_entry))
    sys.stdout.flush()

def load_config():
    config = {
        "job_id":           os.environ.get("JOB_ID"),
        # Kuberay automatically sets the Ray head address if run as a job, 
        # but "auto" is needed for workers to find the head.
        "ray_head_address": os.environ.get("RAY_HEAD_ADDRESS", "auto"),
        "scripts_mount_path": os.environ.get("SCRIPTS_MOUNT_PATH", "/mnt/gcs/transfers"),
    }

    missing = [k for k, v in config.items() if v is None]
    if missing:
        log_event("ERROR", f"Missing required environment variables: {missing}")
        sys.exit(1)

    return config

def ray_enabled():
    """Should this job run on Ray, or on Daft's native (single-process) runner?

    Ray earns its ~45s startup only across a real cluster. In a single local
    container it buys nothing -- Daft's native runner is already multi-threaded
    across the same CPUs -- and measured on a 995-row transfer the cluster spent
    35s on ray.init() plus 10s on set_runner_ray() to do 17s of actual work.
    That overhead pushed the whole request past the frontend proxy's ceiling, so
    a transfer that succeeded was reported to the user as an unknown outcome.

    Under KubeRay this is NOT optional: the job was scheduled onto a cluster and
    must attach to it, so that case ignores the flag entirely. Defaults to ON so
    any caller that does not set DTA_USE_RAY keeps the previous behaviour;
    docker_run.py opts the local runner out.
    """
    if "KUBERAY_INIT_CLUSTER" in os.environ or "RAY_JOB_ID" in os.environ:
        return True
    return os.environ.get("DTA_USE_RAY", "1").strip().lower() not in ("0", "false", "no")


def init_ray(ray_head_address):
    # If running inside Kubernetes (KubeRay), force 'auto' so it connects to the cluster
    # instead of starting a standalone local instance.
    if "KUBERAY_INIT_CLUSTER" in os.environ or "RAY_JOB_ID" in os.environ:
        ray_head_address = "auto"
        
    log_event("INFO", f"Connecting to Ray cluster with address: {ray_head_address}")
    try:
        # If running inside a KubeRay cluster as a worker/job, ray.init() 
        # often connects automatically without an explicit address.
        if ray_head_address == "local":
            ray.init(address="local")
        elif ray_head_address == "auto":
            ray.init(address="auto")
        else:
            ray.init(address=ray_head_address)
            
        log_event("INFO", "Ray initialized", {
            "nodes": len(ray.nodes()),
            "available_resources": ray.cluster_resources()
        })
    except Exception as e:
        log_event("ERROR", "Ray initialization failed", {
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        sys.exit(1)

def init_daft():
    log_event("INFO", "Initializing Daft with Ray runner")
    try:
        daft.set_runner_ray()
        log_event("INFO", "Daft Ray runner set")
    except Exception as e:
        log_event("ERROR", "Daft initialization failed", {
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        sys.exit(1)

def load_injection_script(scripts_mount_path, job_id):
    script_path = os.path.join(scripts_mount_path, job_id, "injection_script.py")
    log_event("INFO", f"Loading injection script", {"path": script_path})

    if not os.path.exists(script_path):
        log_event("ERROR", "Injection script not found", {"path": script_path})
        sys.exit(1)

    try:
        spec = importlib.util.spec_from_file_location("injection_script", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        log_event("INFO", "Injection script loaded successfully")
        return module
    except Exception as e:
        log_event("ERROR", "Failed to load injection script", {
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        sys.exit(1)


def run_transfer(module, job_id):
    if not hasattr(module, "run_pipeline"):
        log_event("ERROR", "Injection script does not define run_pipeline()")
        sys.exit(1)

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            log_event("STARTING", f"Attempt {attempt + 1}", {"job_id": job_id})

            result = module.run_pipeline()

            log_event("SUCCESS", "Transfer complete", {
                "job_id": job_id,
                "result": result
            })
            return

        except Exception as e:
            log_event(
                "RETRYING" if attempt < max_retries else "FAILURE",
                str(e),
                {
                    "job_id": job_id,
                    "attempt": attempt + 1,
                    "traceback": traceback.format_exc()
                }
            )
            if attempt < max_retries:
                time.sleep(5)
            else:
                sys.exit(1)

if __name__ == "__main__":
    config = load_config()
    log_event("INFO", "Configuration loaded", {"job_id": config["job_id"]})

    if ray_enabled():
        init_ray(config["ray_head_address"])
        init_daft()
    else:
        # Leave Daft on its default native runner -- the generated injection
        # scripts only use the daft DataFrame API, which runs on either.
        log_event("INFO", "Ray disabled (DTA_USE_RAY=0) — using Daft native runner")

    module = load_injection_script(config["scripts_mount_path"], config["job_id"])
    run_transfer(module, config["job_id"])