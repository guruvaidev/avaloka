from uuid import uuid4
import calendar
import logging
import datetime
import os

from dateutil.relativedelta import relativedelta

from celery.schedules import crontab
from celery.result import AsyncResult
from langchain_core.messages import AIMessage
from tabulate import tabulate

from app.graph.etl_state import ETLState
from app.core.celery_app import celery_app, AvalokaScheduler, AvalokaEntry
from app.core import celery_results
from app.core.task_metadata import scheduled_task_metadata
from app.utils import convert_message_objects_to_dicts, convert_message_dicts_to_objects
from redis import Redis
from redis.connection import parse_url


logger = logging.getLogger(__name__)
CELERY_URL = parse_url(os.getenv("CELERY_REDIS_URL", "redis://avaloka-redis:6379/0"))

# RayTrainer can wait for a submitted job for up to 12 hours. Give scheduled
# training a little extra time to persist the final result after that wait,
# without weakening the default 240/300-second limits for other Celery tasks.
SCHEDULED_TRAINING_SOFT_TIME_LIMIT = (12 * 60 * 60) + (5 * 60)
SCHEDULED_TRAINING_TIME_LIMIT = (12 * 60 * 60) + (10 * 60)

# Full-dataset analyses on Ray outlive the global 240/300 s Celery limits.
SCHEDULED_RAY_EXECUTE_SOFT_TIME_LIMIT = int(os.getenv("AVALOKA_RAY_EXECUTE_SOFT_LIMIT_S", str(2 * 60 * 60)))
SCHEDULED_RAY_EXECUTE_TIME_LIMIT = SCHEDULED_RAY_EXECUTE_SOFT_TIME_LIMIT + 5 * 60

# Absolute-schedule units, coarsest first. The rollover step for a past time is
# the unit one order ABOVE the coarsest field the user actually specified
# (e.g. only 'hour' given -> roll by 'day' for "next 9am").
_SCHEDULE_UNIT_ORDER = ["year", "month", "day", "hour", "minute", "second"]
_RELATIVEDELTA_KW = {
    "year": "years", "month": "months", "day": "days",
    "hour": "hours", "minute": "minutes", "second": "seconds",
}


def _celery_options_for_schedule(schedule_data: dict, execution_mode: str | None = None) -> dict:
    """Return execution limits appropriate for this scheduled task type."""
    task_type = (schedule_data or {}).get("task_type")
    if task_type == "training":
        return {
            "soft_time_limit": SCHEDULED_TRAINING_SOFT_TIME_LIMIT,
            "time_limit": SCHEDULED_TRAINING_TIME_LIMIT,
        }
    if task_type == "execute" and execution_mode == "k8s-ray":
        return {
            "soft_time_limit": SCHEDULED_RAY_EXECUTE_SOFT_TIME_LIMIT,
            "time_limit": SCHEDULED_RAY_EXECUTE_TIME_LIMIT,
        }
    return {}


# Local dev has no Celery worker. When set to "1", one-shot "run now" jobs
# execute in-process instead of being refused. Keep it off in production:
# there, a missing worker is an outage, not a reason to run a long Ray job
# inside an HTTP request.
_RUN_INLINE_WITHOUT_WORKER = os.getenv("AVALOKA_RUN_INLINE_WITHOUT_WORKER", "0") == "1"


def is_run_now(schedule_data: dict | None) -> bool:
    """A single zero-delay run. This is how full-dataset (k8s-ray) analyses
    are dispatched, not a schedule the user asked for."""
    if not schedule_data:
        return False
    return (
        schedule_data.get("schedule_type") == "relative"
        and int(schedule_data.get("second") or 0) <= 0
        and int(schedule_data.get("max_runs") or 1) == 1
    )


def _prepare_celery_state(state: ETLState) -> ETLState:
    """State handed to agent_task, whether queued or run inline."""
    celery_state: ETLState = state.copy()
    celery_state["messages"] = convert_message_objects_to_dicts(state["messages"])
    celery_state["execution_output_data"] = None
    celery_state["execution_output_preview"] = None
    celery_state["enable_training"] = False
    celery_state["ready_to_code"] = False
    celery_state["ready_to_summarize"] = False
    return celery_state


def _run_now_inline(state: ETLState) -> ETLState:
    """Run a one-shot job in this process when no Celery worker exists."""
    # Calling a Celery task directly runs it in-process.
    from app.core.celery_app import _run_status_from_final, agent_task

    result: ETLState = agent_task(_prepare_celery_state(state))
    msgs = convert_message_dicts_to_objects(result.get("messages") or [])
    reply = msgs[-1].content if msgs else "The analysis finished."
    # Present output only for a run that succeeded. A failed run, or one the
    # output contract refused, has no table of its own: output_file_data is None
    # and output_location is whatever the session carried in (an earlier turn's
    # file), which the chat would otherwise show as this question's answer. The
    # reply text already says what happened.
    status, _error = _run_status_from_final(result, task_type="execute")
    ok = status == "success"
    state.update(
        messages=state["messages"] + [AIMessage(content=reply)],
        output_location=result.get("output_location") if ok else None,
        output_file_data=result.get("output_file_data") if ok else None,
    )
    return state


def _run_training_in_background(state: ETLState) -> "threading.Thread":
    """Start a one-shot training run on a daemon thread (dev mode only)."""
    import threading

    from app.core.celery_app import agent_task

    celery_state = _prepare_celery_state(state)

    def _run() -> None:
        try:
            agent_task(celery_state)
        except Exception:  # noqa: BLE001 - the thread must not take the API down
            logger.exception("Background training (dev mode) failed")

    thread = threading.Thread(target=_run, name="avaloka-dev-training", daemon=True)
    thread.start()
    return thread


def _cleared(state: ETLState) -> ETLState:
    new_state: ETLState = state.copy()
    new_state.update(task_operation={}, task_schedule={}, ready_to_code=False,
                     ready_to_summarize=False, skip_to_training=False)
    return new_state

def _safe_replace(base: datetime.datetime, fields: dict) -> datetime.datetime:
    """datetime.replace() that clamps an out-of-range day to the month's last
    valid day, so e.g. day=31 in a 30-day month doesn't raise ValueError."""
    target = dict(fields)
    if "day" in target:
        year = target.get("year", base.year)
        month = target.get("month", base.month)
        target["day"] = min(target["day"], calendar.monthrange(year, month)[1])
    return base.replace(**target)


def _resolve_absolute_run_at(schedule_data: dict, now: datetime.datetime) -> datetime.datetime:
    """
    Compute the next fire time for an absolute schedule.

    Fields are interpreted in `now`'s timezone (UTC), matching how schedules are
    rendered elsewhere. Unspecified fields inherit from `now`; if the resulting
    time is already in the past, it rolls forward by the unit above the coarsest
    specified field so a requested time-of-day fires on its next occurrence
    instead of immediately.
    """
    field_map = {
        "year": schedule_data.get("year"),
        "month": schedule_data.get("month"),
        "day": schedule_data.get("day_of_month"),
        "hour": schedule_data.get("hour"),
        "minute": schedule_data.get("minute"),
        "second": schedule_data.get("second"),
    }
    provided: dict = {}
    for key, value in field_map.items():
        if value is None:
            continue
        if isinstance(value, str):
            if not value.isdigit():
                continue
            value = int(value)
        provided[key] = value

    run_at = _safe_replace(now, provided)

    coarsest = next((u for u in _SCHEDULE_UNIT_ORDER if u in provided), None)
    idx = _SCHEDULE_UNIT_ORDER.index(coarsest) if coarsest else 0
    # idx == 0 means a fully-anchored (year-level) time; there's no larger unit
    # to roll into, so a past time simply fires as soon as possible.
    if idx > 0 and run_at <= now:
        step = relativedelta(**{_RELATIVEDELTA_KW[_SCHEDULE_UNIT_ORDER[idx - 1]]: 1})
        while run_at <= now:
            run_at = run_at + step
    return run_at


def is_celery_worker_running():
    """Checks if at least one Celery worker is available and running."""
    try:
        r = Redis(host=CELERY_URL["host"], port=CELERY_URL["port"], db=CELERY_URL["db"], socket_connect_timeout=1)
        r.ping()

        inspector = celery_app.control.inspect(timeout=1.0)
        available = inspector.ping()
        return bool(available)
    except Exception:
        return False
    
def handle_task_info(state: ETLState):
    logger.info("Retrieve info on task")
    task_id = state["task_operation"]["task_id"]
    key = AvalokaEntry.generate_key(celery_app, task_id)
    try:
        entry = AvalokaEntry.from_key(key, celery_app)
    except Exception as e:
        logger.error(f"Task {task_id} couldnt be found", exc_info=e)
        state.update(messages=state["messages"] + [AIMessage(content=f"Task {task_id} couldnt be found")])
        return state

    last_run_at = AvalokaScheduler.get_last_run_at(task_id)
    message_append = []
    message_append.append(
        f" and was last run on {last_run_at.astimezone(datetime.timezone.utc).strftime('%B %d, %Y %I:%M %p')} UTC"
        if last_run_at
        else ""
    )

    if isinstance(entry.schedule, crontab):
        schedule = {
            "month_of_year": entry.schedule._orig_month_of_year,
            "day_of_month": entry.schedule._orig_day_of_month,
            "day_of_week": entry.schedule._orig_day_of_week,
            "hour": entry.schedule._orig_hour,
            "minute": entry.schedule._orig_minute,
        }
        message_append.append(f". The schedule is {schedule}")
    else:
        message_append.append(f". The schedule is {entry.schedule}")


    message = AIMessage(
        content=f"The task {task_id} has run a {entry.total_run_count} times" + "".join(message_append)
    )
    
    state.update(messages=state["messages"] + [message])
    return state

def _finished_run(celery_task_id: str, result) -> dict:
    """What a finished run shows: its reply, output and output location.

    A result stored as a summary (app/core/celery_results.py) has no messages
    and no inline output; those come from the run's durable scheduled_runs row.
    A result stored before that (the full state) is read as it always was.
    """
    result = result if isinstance(result, dict) else {}
    if celery_results.is_summary(result):
        run = celery_results.load_scheduled_run(celery_task_id) or {}
        run_result = run.get("result") or {}
        msgs = celery_results.assistant_messages_from_logs(run_result.get("logs"))
        return {
            "reply": msgs[-1]["content"] if msgs else "",
            "output_location": result.get("output_location"),
            "output_file_data": celery_results.output_file_data_from_rows(
                run_result.get("output_json"), result.get("output_location")),
            "statistics": result.get("task_run_statistics") or {},
        }
    msgs = convert_message_dicts_to_objects(result.get("messages") or [])
    return {
        "reply": msgs[-1].content if msgs else "",
        "output_location": result.get("output_location"),
        "output_file_data": result.get("output_file_data"),
        "statistics": result.get("task_run_statistics") or {},
    }


def _not_finished_message(celery_task_id: str, task_state: str, running: str) -> str:
    """``running`` unless the durable run record says the run is over.

    A result that is no longer stored reads as PENDING, which is not "running".
    """
    finished = celery_results.finished_run_state(celery_results.load_scheduled_run(celery_task_id))
    if finished == "SUCCESS":
        return "The task finished successfully, but its stored result is no longer available."
    if finished == "FAILURE":
        return "The task failed. Its stored result is no longer available."
    return running


def _utc_text(ts) -> str:
    return datetime.datetime.fromtimestamp(ts).astimezone(datetime.timezone.utc).strftime('%B %d, %Y %I:%M %p')


def handle_task_result(state: ETLState):
    task_id = state["task_operation"]["task_id"]
    result_index = state["task_operation"]["result_index"]
    logger.info(f"Retrieving result of task {task_id}")
    key = AvalokaEntry.generate_key(celery_app, task_id)
    try:
        entry = AvalokaEntry.from_key(key, celery_app)
    except Exception as e:
        logger.error(f"Task {task_id} couldnt be found", exc_info=e)
        state.update(messages=state["messages"] + [AIMessage(content=f"Task {task_id} couldn't be found")])
        return state

    celery_task_ids = AvalokaScheduler.get_task_ids(task_id)
    if result_index < 0 or result_index >= len(celery_task_ids):
        state.update(messages=state["messages"] + [AIMessage(f"Couldn't find the result for the {task_id}")])
    else:
        celery_task_id = celery_task_ids[result_index]
        logger.info(f"Found celery task {celery_task_id} for {task_id}")
        task = AsyncResult(celery_task_id, app=celery_app)
        base_message = f"Results for task {task_id} run #{result_index + 1}:\n"
        if task.state == "SUCCESS":
            run = _finished_run(celery_task_id, task.result)
            message = f"\n\n{run['reply']}" if run["reply"] else ""
            state.update(
                messages=state["messages"] + [AIMessage(f"{base_message}The task ran successfully{message}")],
                output_location=run["output_location"],
                output_file_data=run["output_file_data"],
            )
            return state
        if task.state == "FAILURE":
            message = AIMessage(content=f"{base_message}The latest task failed to execute. Traceback:\n{task.traceback}")
        elif task.state == "REVOKED":
            message = AIMessage(content=f"{base_message}The latest task failed to execute was cancelled")
        else:
            logger.info(f"Task with id {task_id} is currently {task.state}")
            message = AIMessage(content=base_message + _not_finished_message(
                celery_task_id, task.state, f"Task with id {task_id} is currently running"))
        state.update(messages=state["messages"] + [message])
    return state

def handle_task_status(state: ETLState):
    task_id = state["task_operation"]["task_id"]
    logger.info(f"Retrieving status of task {task_id}")
    key = AvalokaEntry.generate_key(celery_app, task_id)
    try:
        entry = AvalokaEntry.from_key(key, celery_app)
    except Exception as e:
        logger.error(f"Task {task_id} couldnt be found", exc_info=e)
        state.update(messages=state["messages"] + [AIMessage(content=f"Task {task_id} couldn't be found")])
        return state

    celery_task_id = AvalokaScheduler.get_last_run_task_id(task_id)
    due_at = entry.due_at.astimezone(datetime.timezone.utc).strftime('%B %d, %Y %I:%M %p')
    is_unlimited = entry.max_runs == -1
    remaining_str = "unlimited" if is_unlimited else str(entry.max_runs - entry.total_run_count)
    total_str = "unlimited" if is_unlimited else str(entry.max_runs)
    has_more_runs = is_unlimited or (entry.max_runs - entry.total_run_count) > 0
    if celery_task_id is None:
        state.update(
            messages=state["messages"] + [AIMessage(content=f"The task is scheduled to run at {due_at} UTC. There are {remaining_str} runs remaining for a total of {total_str} runs")],
        )
    else:
        logger.info(f"Found celery task {celery_task_id} for {task_id}")
        task = AsyncResult(celery_task_id, app=celery_app)
        if task.state == "SUCCESS":
            run = _finished_run(celery_task_id, task.result)
            message = f"The task is scheduled to run again at {due_at} UTC" if has_more_runs else ""
            statistics = run["statistics"]
            timing = ""
            if all(isinstance(statistics.get(k), (int, float)) for k in ("start", "end", "time_spent")):
                timing = (
                    f"The task started at {_utc_text(statistics['start'])} UTC and finished at {_utc_text(statistics['end'])} UTC. "
                    f"The task took {datetime.timedelta(seconds=statistics['time_spent'])} to complete. "
                )
            saved = f"The output is saved to {run['output_location']}. " if run["output_location"] else ""
            state.update(
                messages=state["messages"] + [AIMessage(
                    "The task ran successfully. "
                    + timing
                    + saved
                    + f"There are {remaining_str} runs remaining. "
                    + message
                )],
                output_location=run["output_location"],
                output_file_data=run["output_file_data"],
            )
            return state

        if task.state == "FAILURE":
            message = AIMessage(content=f"The latest task failed to execute. Traceback:\n{task.traceback}")
        elif task.state == "REVOKED":
            message = AIMessage(content=f"The latest task failed to execute was cancelled")
        else:
            logger.info(f"Task with id {task_id} is currently {task.state}")
            finished = _not_finished_message(celery_task_id, task.state, "")
            message = AIMessage(content=finished) if finished else AIMessage(content=f"Task with id {task_id} is currently running. The task started at {entry.last_run_at.astimezone(datetime.timezone.utc).strftime('%B %d, %Y %I:%M %p')} UTC. There are {remaining_str} runs remaining for a total of {total_str} runs")
        state.update(messages=state["messages"] + [message])
    return state

def handle_task_list(state: ETLState):
    task_ids = state.get("task_list", [])
    if not task_ids:
        state.update(
            messages=state["messages"] + [AIMessage(content=f"There are no current tasks")]
        )
        return state

    entries: list[AvalokaEntry] = []
    for task_id in task_ids:
        try:
            entry = AvalokaEntry.from_key(AvalokaEntry.generate_key(celery_app, task_id), celery_app)
        except Exception as e:
            logger.error(f"Task {task_id} couldnt be found", exc_info=e)
            continue
        entries.append(entry)

    if not entries:
        state.update(
            messages=state["messages"] + [AIMessage(content=f"There are no current tasks")]
        )
        return state

    data = []
    for entry in entries:
        task_id = entry.name
        runs_completed = entry.total_run_count
        if entry.max_runs != -1:
            remaining_runs = entry.max_runs - entry.total_run_count
            remaining_str = str(remaining_runs)
            has_next = remaining_runs > 0
        else:
            remaining_str = "Unlimited"
            has_next = True

        if has_next:
            next_due = entry.due_at.astimezone(datetime.timezone.utc).strftime('%B %d, %Y %I:%M %p UTC')
        else:
            next_due = "N/A"
        
        if entry.total_run_count > 0:
            last_id = AvalokaScheduler.get_last_run_task_id(task_id)
            task = AsyncResult(last_id, app=celery_app)
            task_state = task.state
            if task_state not in ("SUCCESS", "FAILURE", "REVOKED"):
                task_state = celery_results.finished_run_state(
                    celery_results.load_scheduled_run(last_id)) or task_state
            if task_state == "SUCCESS":
                last_status = f"Success ({entry.last_run_at.astimezone(datetime.timezone.utc).strftime('%B %d, %Y %I:%M %p UTC')})"
            elif task_state == "FAILURE":
                last_status = f"Failed ({entry.last_run_at.astimezone(datetime.timezone.utc).strftime('%B %d, %Y %I:%M %p UTC')})"
            elif task_state == "REVOKED":
                last_status = "Cancelled"
            else:
                last_status = "Running"
        else:
            last_status = "Not run yet"
        
        if isinstance(entry.schedule, crontab):
            schedule = str({
                "month_of_year": entry.schedule._orig_month_of_year,
                "day_of_month": entry.schedule._orig_day_of_month,
                "day_of_week": entry.schedule._orig_day_of_week,
                "hour": entry.schedule._orig_hour,
                "minute": entry.schedule._orig_minute,
            })
        else:
            schedule = str(entry.schedule.run_every)
        
        data.append([task_id, runs_completed, remaining_str, next_due, last_status, str(schedule)])
    
    headers = [
        "Task ID",
        "Runs Completed",
        "Remaining Runs",
        "Next Due",
        "Last Status",
        "Schedule"
    ]
    state.update(
        messages=state["messages"] + [AIMessage(content=str(tabulate(data, headers=headers, tablefmt="pipe")))]
    )
    return state

def _cancel_training_job_if_any(task_id: str, state: ETLState) -> bool:
    """A training job id (ml_jobs) given to cancel_task: delete its RayJob, mark it cancelled."""
    try:
        from app.infra import training_tasks
        job = training_tasks.cancel_training_job(task_id, str(state.get("user_id") or ""))
    except Exception:  # noqa: BLE001
        return False
    if job is None:
        return False
    state.update(messages=state["messages"] + [AIMessage(
        content=f"Training job {job.id[:8]} is cancelled." if job.status == "cancelled"
        else f"Training job {job.id[:8]} had already finished ({job.status}).")])
    return True


def handle_task_cancel(state: ETLState):
    task_id = state["task_operation"]["task_id"]
    logger.info(f"Cancelling task {task_id}")
    if _cancel_training_job_if_any(task_id, state):
        return state
    key = AvalokaEntry.generate_key(celery_app, task_id)
    try:
        entry = AvalokaEntry.from_key(key, celery_app)
    except Exception as e:
        logger.exception(f"Failed to cancel task", exc_info=e)
        message = AIMessage(content=f"Task {task_id} doesn't exist")
    else:
        entry.delete()
        message = AIMessage(content=f"Task with id {task_id} has been cancelled")
    
    task_list = state.get("task_list") or []
    if task_id in task_list:
        task_list = [*task_list]
        task_list.remove(task_id)

    state.update(
        messages=state["messages"] + [message],
        task_list=task_list
    )
    return state

def handle_task_scheduling(state: ETLState) -> None:
    logger.info("Scheduling periodic task")
    celery_state = _prepare_celery_state(state)
    schedule_data = state["task_schedule"]
    task_metadata = scheduled_task_metadata(schedule_data)
    max_runs = schedule_data.get("max_runs", 1)
    schedule_type = schedule_data["schedule_type"]
   

    if schedule_type == "absolute":
        now = datetime.datetime.now(datetime.timezone.utc)
        run_at = _resolve_absolute_run_at(schedule_data, now)
        schedule = (run_at - now).total_seconds()
    elif schedule_type == "relative":
        schedule = schedule_data.get("second", -1)
    elif schedule_type == "repetitive":
        schedule = crontab(
            day_of_month=schedule_data["day_of_month"],
            day_of_week=schedule_data["day_of_week"],
            hour=schedule_data["hour"],
            minute=schedule_data["minute"],
        )
        # Force unlimited for repetitive crons — the planner sometimes emits
        # max_runs=1, which fires once then disables an every-minute schedule.
        max_runs = -1
        
    entry = AvalokaEntry(
        str(uuid4()),
        "app.core.celery_app.agent_task",
        schedule,
        args=[celery_state],
        options=_celery_options_for_schedule(schedule_data, state.get("execution_mode")),
        app=celery_app,
        max_runs=max_runs,
    ).save()

    due_at = entry.due_at.astimezone(datetime.timezone.utc)
    logger.info(f"Task due at: {due_at}")
    logger.info(f"Schedule: {entry.schedule}")

    state.update(
        messages=state["messages"] + [AIMessage(content=f"{task_metadata['scheduled_message']} The task id is {entry.name}.")],
        task_list=[*(state.get("task_list") or []), entry.name],
        task_info={
            "task_id": entry.name,
            "next_due_at": (due_at - datetime.datetime.now(datetime.timezone.utc)).total_seconds() + 2,
            **task_metadata,
        }
    )
    return state

def task_scheduler_node(state: ETLState) -> None:
    """LangGraph node that schedules a periodic Celery task"""
    logger.info("Entered scheduler node")
    if not is_celery_worker_running():
        schedule_data = state.get("task_schedule") or {}
        run_now = not state.get("task_operation") and is_run_now(schedule_data)
        if run_now and _RUN_INLINE_WITHOUT_WORKER and schedule_data.get("task_type") == "training":
            # Dev box without a worker: training still never runs inside the
            # chat turn. It runs in a background thread of this process and the
            # turn returns at once; the job row (ml_jobs) and the pending-turn
            # poll report the outcome, as they would with a worker.
            logger.warning("No Celery worker; running training in a background thread "
                           "(AVALOKA_RUN_INLINE_WITHOUT_WORKER=1).")
            _run_training_in_background(state)
            state.update(messages=state["messages"] + [AIMessage(
                content="Model training has started in the background (dev mode, no worker). "
                        "The result will appear in the Inference panel when it finishes.")])
            return _cleared(state)
        if run_now and _RUN_INLINE_WITHOUT_WORKER and schedule_data.get("task_type") != "training":
            logger.warning("No Celery worker; running one-shot job inline (AVALOKA_RUN_INLINE_WITHOUT_WORKER=1).")
            try:
                state = _run_now_inline(state)
            except Exception as exc:
                logger.exception("Inline run failed")
                state.update(messages=state["messages"] + [AIMessage(
                    content=f"The analysis couldn't run: {exc}")])
            return _cleared(state)

        if run_now:
            logger.warning("Run-now job needs a Celery worker and none is running.")
            content = (
                "This analysis runs on the full dataset as a background job, but no "
                "background worker (Celery) is running, so it couldn't start. Start "
                "the worker, or switch 'Analysis on' to a sample to run it right away."
            )
        else:
            logger.warning("Scheduling requested but no Celery worker is running; replying instead of failing.")
            content = (
                "Scheduling isn't available in this deployment — no task worker "
                "(Celery) is running, so I can't create or manage scheduled jobs "
                "right now."
            )
        state.update(messages=state["messages"] + [AIMessage(content=content)])
        return _cleared(state)

    
    
    if state.get("task_operation"):
        if state["task_operation"]["operation"] == "INFO":
            state = handle_task_info(state)

        elif state["task_operation"]["operation"] == "RETRIEVE_RESULT":
            state = handle_task_result(state)

        elif state["task_operation"]["operation"] == "STATUS":
            state = handle_task_status(state)

        elif state["task_operation"]["operation"] == "LIST":
            state = handle_task_list(state)
        
        elif state["task_operation"]["operation"] == "CANCEL":
            state = handle_task_cancel(state)

        else:
            state.update(messages=state["messages"] + [AIMessage(content="Failed to handle the task.")])

        return _cleared(state)

    if state.get("task_schedule"):
        state = handle_task_scheduling(state)

    new_state: ETLState = state.copy()
    new_state.update(task_operation={}, task_schedule={}, ready_to_code=False, ready_to_summarize=False, skip_to_training=False)
    return new_state
