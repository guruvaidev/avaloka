import logging
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from langchain_core.messages import AIMessage, HumanMessage
from celery import Celery
from celery.utils.log import get_logger
from redbeat.schedulers import RedBeatScheduler, RedBeatSchedulerEntry, RedBeatJSONEncoder, RedBeatJSONDecoder, get_redis, ensure_conf
from dotenv import load_dotenv

from app.agents.execution_agent import execution_agent_node, execution_agent_node_ssh
from app.agents.mta_v2.failure_diagnostics import (
    TRAINING_FAILURE_MESSAGE,
    build_training_failure,
    format_training_failure,
)
from app.graph.etl_state import ETLState
from app.core.scheduled_run_context import scheduled_run_context
from app.utils import convert_message_objects_to_dicts, convert_message_dicts_to_objects

logger = logging.getLogger(__name__)

load_dotenv()
REDIS_URL = os.getenv("CELERY_REDIS_URL", "redis://avaloka-redis:6379/0")
SERVERS = json.loads(os.getenv("CELERY_SSH_SERVERS", "[]"))
SERVERS_IN_USE = []


celery_app = Celery('celery_app', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    result_extended=True,
    result_expires=None,
    task_time_limit=300,
    task_soft_time_limit=240,
    worker_prefetch_multiplier=1,
)
ensure_conf(celery_app)

class AvalokaEntry(RedBeatSchedulerEntry):
    def __init__(self, name=None, task=None, schedule=None, args=None, kwargs=None, enabled=True, options=None, max_runs: int = -1, **clsargs):
        super().__init__(name, task, schedule, args, kwargs, enabled, options, **clsargs)
        self.max_runs = max_runs

    def save(self):
        definition = {
            'name': self.name,
            'task': self.task,
            'args': self.args,
            'kwargs': self.kwargs,
            'options': self.options,
            'schedule': self.schedule,
            'enabled': self.enabled,
            'max_runs': self.max_runs
        }
        meta = {
            'last_run_at': self.last_run_at,
        }
        with get_redis(self.app).pipeline() as pipe:
            pipe.hset(self.key, 'definition', json.dumps(definition, cls=RedBeatJSONEncoder))
            pipe.hsetnx(self.key, 'meta', json.dumps(meta, cls=RedBeatJSONEncoder))
            pipe.zadd(self.app.redbeat_conf.schedule_key, {self.key: self.score})
            pipe.execute()

        return self

class AvalokaScheduler(RedBeatScheduler):
    Entry = AvalokaEntry

    def __init__(self, *args, **kwargs):
        del kwargs["max_interval"]
        super().__init__(max_interval=60, *args, **kwargs)

    def maybe_due(self, entry: AvalokaEntry, **kwargs):
        logger = get_logger('celery.beat')

        is_due, next_time_to_run = entry.is_due()

        if is_due:
            logger.info('Scheduler: Sending due task %s (%s)', entry.name, entry.task)
            # entry.name == the task_id the API queries runs by. Hand it to the
            # worker (transient, per-dispatch) so it can key scheduled_runs rows.
            try:
                if entry.args and isinstance(entry.args[0], dict):
                    entry.args[0]["schedule_id"] = entry.name
            except Exception:
                logger.debug('Scheduler: could not tag schedule_id on %s', entry.name)
            try:
                result = self.apply_async(entry, **kwargs)
            except Exception as exc:
                logger.exception('Scheduler: Message Error: %s', exc)
            else:
                if result and hasattr(result, 'id'):
                    with get_redis(self.app).pipeline() as pipe:
                        if entry.max_runs > -1 and entry.total_run_count + 1 >= entry.max_runs:
                            entry.enabled = False
                            definition = {
                                'name': entry.name,
                                'task': entry.task,
                                'args': entry.args,
                                'kwargs': entry.kwargs,
                                'options': entry.options,
                                'schedule': entry.schedule,
                                'enabled': entry.enabled,
                                'max_runs': entry.max_runs
                            }
                            pipe.hset(entry.key, 'definition', json.dumps(definition, cls=RedBeatJSONEncoder))

                        pipe.hget(entry.key, 'task_data')

                        task_data = json.loads(pipe.execute()[-1] or '{}', cls=RedBeatJSONDecoder)
                        task_ids = task_data.get("task_ids", [])
                        task_run_at = task_data.get("task_run_at", [])

                        task_data["task_ids"] = [*task_ids, result.id]
                        task_data["task_run_at"] = [*task_run_at, datetime.now(timezone.utc)]

                        pipe.hset(
                            entry.key, 'task_data', json.dumps(task_data, cls=RedBeatJSONEncoder)
                        )
                        pipe.execute()
                    logger.debug('Scheduler: %s sent. id->%s', entry.task, result.id)
                else:
                    logger.debug('Scheduler: %s sent.', entry.task)
        return next_time_to_run
    
    @staticmethod
    def get_last_run_task_id(task_id: str, generate_key: bool = True) -> Optional[str]:
        key = (
            RedBeatSchedulerEntry.generate_key(celery_app, task_id)
            if generate_key
            else task_id
        )
        with get_redis(celery_app).pipeline() as pipe:
            pipe.hget(key, 'task_data')
            task_ids = json.loads(pipe.execute()[0] or '{}', cls=RedBeatJSONDecoder).get("task_ids", [])
            return (task_ids[-1] if task_ids else None)
        
    @staticmethod
    def get_last_run_at(task_id: str, generate_key: bool = True) -> Optional[str]:
        key = (
            RedBeatSchedulerEntry.generate_key(celery_app, task_id)
            if generate_key
            else task_id
        )
        with get_redis(celery_app).pipeline() as pipe:
            pipe.hget(key, 'task_data')
            task_run_at = json.loads(pipe.execute()[0] or '{}', cls=RedBeatJSONDecoder).get("task_run_at", [])
            return (task_run_at[-1] if task_run_at else None)
        
    @staticmethod
    def get_task_ids(task_id: str, generate_key: bool = True) -> list[str]:
        key = (
            RedBeatSchedulerEntry.generate_key(celery_app, task_id)
            if generate_key
            else task_id
        )
        with get_redis(celery_app).pipeline() as pipe:
            pipe.hget(key, 'task_data')
            task_ids = json.loads(pipe.execute()[0] or '{}', cls=RedBeatJSONDecoder).get("task_ids", [])
            return task_ids or []
    
    @staticmethod
    def get_all_timestamps(task_id: str, generate_key: bool = True) -> list[str]:
        key = (
            RedBeatSchedulerEntry.generate_key(celery_app, task_id)
            if generate_key
            else task_id
        )
        with get_redis(celery_app).pipeline() as pipe:
            pipe.hget(key, 'task_data')
            task_run_at = json.loads(pipe.execute()[0] or '{}', cls=RedBeatJSONDecoder).get("task_run_at", [])
            return task_run_at or []
    
    @staticmethod
    def get_task_ids_with_timestamp(task_id: str, generate_key: bool = True) -> list[tuple[str, datetime]]:
        key = (
            RedBeatSchedulerEntry.generate_key(celery_app, task_id)
            if generate_key
            else task_id
        )
        with get_redis(celery_app).pipeline() as pipe:
            pipe.hget(key, 'task_data')
            task_data = json.loads(pipe.execute()[0] or '{}', cls=RedBeatJSONDecoder)
            return list(zip(task_data.get("task_ids", []), task_data.get("task_run_at", []))) or []


from celery import current_task
_SCHEDULED_RUNS_TABLE = os.getenv("SUPABASE_SCHEDULED_RUNS_TABLE", "scheduled_runs")
_SCHEDULED_RUN_ROW_CAP = int(os.getenv("SCHEDULED_RUN_OUTPUT_ROW_CAP", "5000"))


def _supabase():
    # Lazy import to avoid an import-time cycle with the persistence layer.
    from app.agents.sampling_persistence import get_supabase_client
    return get_supabase_client()


def _record_run_start(schedule_id: str, execution_id: str, state: dict) -> None:
    def _q():
        client = _supabase()
        existing = (client.table(_SCHEDULED_RUNS_TABLE)
                    .select("run_id", count="exact")
                    .eq("schedule_id", schedule_id).execute())
        run_number = (getattr(existing, "count", None) or 0) + 1
        client.table(_SCHEDULED_RUNS_TABLE).upsert({
            "run_id":      execution_id,
            "schedule_id": schedule_id,
            "session_id":  state.get("session_id"),
            "user_id":     state.get("user_id") or "",
            "run_number":  run_number,
            "status":      "running",
            "started_at":  datetime.now(timezone.utc).isoformat(),
        }).execute()
    try:
        _q()
    except Exception:
        logger.warning("[runs] start-write failed for %s", execution_id, exc_info=True)


def _record_ray_job_submission(
    schedule_id: str,
    execution_id: str,
    metadata: dict,
) -> None:
    """Persist the nested Ray identity while its Celery run is still active."""
    ray_job = {
        key: str(metadata[key])
        for key in ("job_id", "status", "dashboard_url", "namespace")
        if metadata.get(key)
    }
    if not ray_job.get("job_id"):
        return

    try:
        (_supabase().table(_SCHEDULED_RUNS_TABLE)
         .update({"status": "running", "result": {"ray_job": ray_job}})
         .eq("schedule_id", schedule_id)
         .eq("run_id", execution_id)
         .execute())
    except Exception:
        logger.warning(
            "[runs] Ray submission metadata write failed for %s",
            execution_id,
            exc_info=True,
        )


def _output_json_from_state(final: dict):
    """Decode the inlined output CSV to rows; None if too big or absent."""
    ofd = (final or {}).get("output_file_data")
    if not (isinstance(ofd, dict) and str(ofd.get("content", "")).startswith("data:text/csv;base64,")):
        return None
    import base64, csv, io
    try:
        b64 = ofd["content"].split("base64,", 1)[1]
        rows = list(csv.DictReader(io.StringIO(base64.b64decode(b64).decode())))
        return rows if len(rows) <= _SCHEDULED_RUN_ROW_CAP else None
    except Exception:
        return None


def _training_failure_from_final(final: dict) -> dict | None:
    """Return the structured MTA failure carried by a scheduled final state."""
    if not isinstance(final, dict):
        return None
    training_result = final.get("training_result") or {}
    if not isinstance(training_result, dict):
        return None
    failure = training_result.get("failure")
    if isinstance(failure, dict):
        return failure
    error = (
        training_result.get("error")
        or training_result.get("execution_error")
        or final.get("training_error")
    )
    if error or training_result.get("status") == "error":
        return build_training_failure(error or "Model training failed without an error message.")
    return None


def _ray_job_from_final(final: dict) -> dict | None:
    """Return user-visible Ray metadata from a completed MTA state."""
    if not isinstance(final, dict):
        return None
    training_result = final.get("training_result") or {}
    if not isinstance(training_result, dict) or not training_result.get("ray_job_id"):
        return None
    return {
        key: str(value)
        for key, value in {
            "job_id": training_result.get("ray_job_id"),
            "status": training_result.get("ray_job_status") or training_result.get("status"),
            "dashboard_url": training_result.get("dashboard_url"),
            "namespace": training_result.get("ray_namespace"),
        }.items()
        if value
    }


def _run_status_from_final(final: dict, task_type: str | None = None):
    if not isinstance(final, dict):
        return "success", None
    if task_type == "training":
        failure = _training_failure_from_final(final)
        if failure:
            return "failure", format_training_failure(failure)
        training_result = final.get("training_result") or {}
        mlflow_run_id = (
            training_result.get("mlflow_run_id")
            if isinstance(training_result, dict)
            else None
        ) or final.get("mlflow_run_id")
        if not final.get("training_completed") or not mlflow_run_id:
            return "failure", TRAINING_FAILURE_MESSAGE
    er = final.get("execution_result") or {}
    st = str(er.get("status", "")).lower()
    if st and st not in ("success", "succeeded", "completed", "done"):
        return "failure", str(er.get("error") or "execution failed")
    for _k in ("configure_inference_error", "stop_inference_error", "inference_error", "training_error"):
        if final.get(_k):
            return "failure", str(final.get(_k))
    if final.get("error"):
        return "failure", str(final.get("error"))
    return "success", None


def _record_run_finish(
    schedule_id: str,
    execution_id: str,
    final: dict,
    t0: float,
    scheduled_state: dict | None = None,
) -> None:
    schedule_data = (scheduled_state or {}).get("task_schedule") or {}
    task_type = str(schedule_data.get("task_type") or "execute")
    status, error = _run_status_from_final(final, task_type=task_type)
    completion_message = None
    if task_type == "training" and status == "failure":
        completion_message = error or TRAINING_FAILURE_MESSAGE
    def _q():
        client = _supabase()
        client.table(_SCHEDULED_RUNS_TABLE).update({
            "status":      status,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": int((time.perf_counter() - t0) * 1000),
            #"result":      {"output_json": _output_json_from_state(final)},
            "result": {
                "output_json": _output_json_from_state(final),
                "logs": None if task_type == "training" and status == "failure" else _logs_from_final(final),
                "message": completion_message,
                "ray_job": _ray_job_from_final(final),
            },
            "error":       error,
        }).eq("run_id", execution_id).execute()
    try:
        _q()
    except Exception:
        logger.warning("[runs] finish-write failed for %s", execution_id, exc_info=True)


import traceback as _tb
def _record_run_failure(
    schedule_id: str,
    execution_id: str,
    exc: Exception,
    t0: float,
    scheduled_state: dict | None = None,
) -> None:
    logs = [f"[error] {exc}", _tb.format_exc()]
    is_training = ((scheduled_state or {}).get("task_schedule") or {}).get("task_type") == "training"
    result = {"logs": logs}
    error = str(exc)
    if is_training:
        failure = build_training_failure(exc, logs="\n".join(logs))
        safe_message = format_training_failure(failure)
        result = {"logs": None, "message": safe_message}
        error = safe_message
    def _q():
        client = _supabase()
        client.table(_SCHEDULED_RUNS_TABLE).update({
            "status":      "failure",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": int((time.perf_counter() - t0) * 1000),
            "result":      result,
            "error":       error,
        }).eq("run_id", execution_id).execute()
    try:
        _q()
    except Exception:
        logger.warning("[runs] failure-write failed for %s", execution_id, exc_info=True)

@celery_app.task
def agent_task(json_state: dict) -> None:
    schedule_id  = (json_state or {}).get("schedule_id")
    execution_id = getattr(getattr(current_task, "request", None), "id", None)
    _t0 = time.perf_counter()

    if schedule_id and execution_id:
        _record_run_start(schedule_id, execution_id, json_state)

    try:
        if schedule_id and execution_id:
            with scheduled_run_context(
                schedule_id,
                execution_id,
                ray_job_callback=_record_ray_job_submission,
            ):
                final_state = _run_agent_task(json_state)
        else:
            final_state = _run_agent_task(json_state)
    except Exception as exc:
        if schedule_id and execution_id:
            _record_run_failure(
                schedule_id,
                execution_id,
                exc,
                _t0,
                scheduled_state=json_state,
            )
        raise

    if schedule_id and execution_id:
        _record_run_finish(
            schedule_id,
            execution_id,
            final_state,
            _t0,
            scheduled_state=json_state,
        )
    return final_state



def _run_agent_task(json_state: dict) -> dict:
    if not isinstance(json_state, dict):
        raise TypeError("json_state must be a dictionary")

    print(
        f"[agent_task] ENTERED task_type={json_state.get('task_schedule')}",
        flush=True,
    )
    logger.info("[agent_task] ENTERED")

    statistics = {
        "start": datetime.now(
            timezone.utc
        ).timestamp()
    }
    start_time = time.perf_counter()

    state: ETLState = json_state.copy()

    # ---------------------------------------------------------
    # Fallback execution routing
    # ---------------------------------------------------------
    file_size = int(
        state.get("file_size_bytes") or 0
    )
    cloud_uri = str(
        state.get("data_source_location_cloud") or ""
    ).strip()

    ray_threshold = 1 * 1024 * 1024 * 1024
    fidelity = str(
        state.get("analysis_fidelity") or ""
    ).strip()

    has_explicit_mode = bool(
        str(state.get("execution_mode") or "").strip()
    )

    if not has_explicit_mode:
        if cloud_uri and file_size >= ray_threshold:
            state["execution_mode"] = "k8s-ray"

            logger.warning(
                "[agent_task] fallback routing "
                "file_size=%.2fGB >= 1GB -> k8s-ray",
                file_size / (1024 ** 3),
            )

        elif cloud_uri and file_size > 0:
            state["execution_mode"] = "local"

            logger.info(
                "[agent_task] fallback routing "
                "file_size=%.2fGB < 1GB -> local",
                file_size / (1024 ** 3),
            )
    else:
        logger.info(
            "[agent_task] honoring execution_mode=%s "
            "analysis_fidelity=%s",
            state.get("execution_mode"),
            fidelity or "unset",
        )

    schedule_data = (
        state.pop("task_schedule", None)
        or {"task_type": "execute"}
    )
    task_type = schedule_data.get("task_type")

    print(
        f"[agent_task] task_type={task_type} "
        f"execution_mode={state.get('execution_mode')}",
        flush=True,
    )

    # ---------------------------------------------------------
    # Background sampling
    # ---------------------------------------------------------
    if task_type == "sample_profile":
        from app.agents.sampling_async import (
            run_background_sampling,
        )

        logger.info(
            "[agent_task] Running background sampling "
            "for dataset=%s",
            state.get("dataset_id"),
        )

        result = run_background_sampling(state)

        final_state = (
            result
            if isinstance(result, dict)
            else {"result": result}
        )

    # ---------------------------------------------------------
    # Scheduled analysis execution
    # ---------------------------------------------------------
    elif task_type == "execute":
        state["messages"] = (
            convert_message_dicts_to_objects(
                state.get("messages") or []
            )
        )

        execution_mode = (
            state.get("execution_mode") or "local"
        )

        if execution_mode == "k8s-ray":
            from app.agents.execution_agent import (
                execution_agent_node_ray,
            )

            state = execution_agent_node_ray(state)

        elif state.get("deploy_on_k8s"):
            state = execution_agent_node(state)

        elif SERVERS:
            state = handle_ssh_execute(state)

        else:
            from app.agents.execution_agent import (
                execution_agent_node_local,
            )

            state = execution_agent_node_local(state)

        state["messages"] = (
            convert_message_objects_to_dicts(
                state.get("messages") or []
            )
        )

        final_state = state

        # Inline a successful output so the API can access the
        # worker's result even when it has a different filesystem.
        execution_result = (
            final_state.get("execution_result") or {}
        )
        execution_status = str(
            execution_result.get("status", "")
        ).lower()

        if (
            execution_status in {"success", "succeeded"}
            and not final_state.get("output_file_data")
        ):
            output_path = final_state.get(
                "output_location"
            )

            if (
                output_path
                and os.path.isfile(output_path)
                and os.path.getsize(output_path) > 0
            ):
                import base64

                with open(output_path, "rb") as output_file:
                    raw_output = output_file.read()

                final_state["output_file_data"] = {
                    "filename": os.path.basename(
                        output_path
                    ),
                    "content": (
                        "data:text/csv;base64,"
                        + base64.b64encode(
                            raw_output
                        ).decode("utf-8")
                    ),
                    "size": len(raw_output),
                }

    # ---------------------------------------------------------
    # Scheduled training
    # ---------------------------------------------------------
    elif task_type == "training":
        state["messages"] = (
            convert_message_dicts_to_objects(
                state.get("messages") or []
            )
        )
        state["training_scheduled"] = True

        from app.agents.model_training_agent import (
            model_training_agent_node,
        )

        logger.info(
            "[agent_task] Running scheduled training "
            "directly via MTA"
        )

        final_state = model_training_agent_node(
            state
        )

        final_state["messages"] = (
            convert_message_objects_to_dicts(
                final_state.get("messages") or []
            )
        )

    # ---------------------------------------------------------
    # Scheduled inference service action
    # ---------------------------------------------------------
    elif task_type in {
        "start_inference",
        "stop_inference",
    }:
        state["messages"] = (
            convert_message_dicts_to_objects(
                state.get("messages") or []
            )
        )
        state["task_schedule"] = schedule_data

        from app.agents.model_training_agent import (
            model_training_agent_node,
        )

        logger.info(
            "[agent_task] Running scheduled inference "
            "service operation"
        )

        final_state = model_training_agent_node(
            state
        )

        final_state["messages"] = (
            convert_message_objects_to_dicts(
                final_state.get("messages") or []
            )
        )

    else:
        raise ValueError(
            f"Unsupported scheduled task type: {task_type}"
        )

    statistics["time_spent"] = (
        time.perf_counter() - start_time
    )
    statistics["end"] = datetime.now(
        timezone.utc
    ).timestamp()

    final_state["task_run_statistics"] = statistics

    execution_result = (
        final_state.get("execution_result") or {}
    )

    logger.info(
        "[agent_task] FINISHED task_type=%s status=%s",
        task_type,
        execution_result.get("status", "completed"),
    )

    return final_state

@celery_app.task(name="app.core.celery_app.execution_agent_task")
def execution_agent_task(json_state: dict) -> None:
    """Alias for agent_task — kept for backward compat with old Redis schedule entries."""
    return agent_task(json_state)
    
def handle_ssh_execute(state: ETLState) -> ETLState:
    waited = 0
    while not SERVERS:
        if waited >= 30:
            logger.error("handle_ssh_execute: no SSH servers available after 30s")
            new_state = state.copy()
            new_state["messages"] = new_state["messages"] + [
                AIMessage("Failed to run task - no execution server available")
            ]
            return new_state
        time.sleep(1)
        waited += 1

    server = SERVERS.pop(-1)
    SERVERS_IN_USE.append(server)
    new_state = state.copy()

    for dataset in (state.get("multi_dataset_state") or []):
        logger.info("Creating input folder")
        data_source_location = os.path.dirname(dataset["data_source_location"])
        p = subprocess.run(["ssh", server, f"mkdir -p {data_source_location}"], capture_output=True)
        if p.returncode:
            logger.info(f"Stderr:\n" + p.stderr.decode())
            logger.info(f"Stdout:\n" + p.stdout.decode())
            new_state.update({
                "messages": new_state["messages"] + [AIMessage("Failed to run task - failed to create input folder to server")]
            })
            SERVERS_IN_USE.remove(server)
            SERVERS.append(server)
            return new_state
        
        logger.info("Copying input file")
        p = subprocess.run(["scp", "-r", data_source_location, f"{server}:{os.path.dirname(data_source_location)}"], capture_output=True)
        if p.returncode:
            logger.info(f"Stderr:\n" + p.stderr.decode())
            logger.info(f"Stdout:\n" + p.stdout.decode())
            new_state.update({
                "messages": new_state["messages"] + [AIMessage("Failed to run task - failed to copy input data to server")]
            })
            SERVERS_IN_USE.remove(server)
            SERVERS.append(server)
            return new_state
        

    logger.info("Creating output folder")
    output_location = os.path.dirname(state["output_location"])
    p = subprocess.run(["ssh", server, f"mkdir -p {output_location}"], capture_output=True)
    if p.returncode:
        logger.info(f"Stderr:\n" + p.stderr.decode())
        logger.info(f"Stdout:\n" + p.stdout.decode())
        new_state.update({
            "messages": new_state["messages"] + [AIMessage("Failed to run task - failed to create output folder on server")]
        })
        SERVERS_IN_USE.remove(server)
        SERVERS.append(server)
        return new_state

        
    logger.info("Copying execute script")
    script = os.path.join(output_location, "execute.sh")
    p = subprocess.run(["scp", os.path.join("scripts", "execute.sh"), f"{server}:{script}"], capture_output=True)
    if p.returncode:
        logger.info(f"Stderr:\n" + p.stderr.decode())
        logger.info(f"Stdout:\n" + p.stdout.decode())
        new_state.update({
            "messages": new_state["messages"] + [AIMessage("Failed to run task - failed to copy execute script to server")]
        })
        SERVERS_IN_USE.remove(server)
        SERVERS.append(server)
        return new_state
    
    new_state = execution_agent_node_ssh(new_state, server)

    SERVERS_IN_USE.remove(server)
    SERVERS.append(server)
    return new_state



def _logs_from_final(final: dict) -> list:
    """Assemble a run log (list of lines) from a finished agent state so the
    UI can show execution logs instead of the output table."""
    if not isinstance(final, dict):
        return []
    lines: list = []

    stats = final.get("task_run_statistics") or {}
    if stats.get("start"):
        lines.append(f"[start] {datetime.fromtimestamp(stats['start'], timezone.utc).isoformat()}")

    er = final.get("execution_result") or {}
    if er.get("status"):
        lines.append(f"[execution] status={er.get('status')}")
    # Adjust these keys to whatever execution_agent_node_local/_ray actually
    # captures (stdout/stderr from the generated-code subprocess, etc.).
    for k in ("command", "stdout", "logs", "stderr", "message", "error", "traceback"):
        v = er.get(k)
        if v:
            lines.append(f"[{k}] {v}")

    try:
        for m in convert_message_dicts_to_objects(final.get("messages") or []):
            if isinstance(m, AIMessage) and str(m.content).strip():
                lines.append(f"[assistant] {m.content}")
    except Exception:
        pass

    for _k in ("configure_inference_error", "stop_inference_error", "inference_error", "training_error"):
        if final.get(_k):
            lines.append(f"[{_k}] {final[_k]}")
    for _k in ("configure_inference_traceback", "stop_inference_traceback"):
        if final.get(_k):
            lines.append(f"[traceback] {final[_k]}")

    if stats.get("time_spent") is not None:
        lines.append(f"[duration] {stats['time_spent']:.2f}s")
    if stats.get("end"):
        lines.append(f"[end] {datetime.fromtimestamp(stats['end'], timezone.utc).isoformat()}")
    return lines
