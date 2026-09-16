from app.agents.scheduler import (
    SCHEDULED_TRAINING_SOFT_TIME_LIMIT,
    SCHEDULED_TRAINING_TIME_LIMIT,
    _celery_options_for_schedule,
)


def test_scheduled_training_outlives_ray_wait_window():
    options = _celery_options_for_schedule({"task_type": "training"})

    assert options == {
        "soft_time_limit": SCHEDULED_TRAINING_SOFT_TIME_LIMIT,
        "time_limit": SCHEDULED_TRAINING_TIME_LIMIT,
    }
    assert options["soft_time_limit"] > 12 * 60 * 60
    assert options["time_limit"] > options["soft_time_limit"]


def test_non_training_schedule_keeps_celery_defaults():
    assert _celery_options_for_schedule({"task_type": "execute"}) == {}
    assert _celery_options_for_schedule({}) == {}
