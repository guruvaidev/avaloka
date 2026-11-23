-- Durable Celery task results (app/core/celery_results.py).
--
-- Redis was the only copy of every task result, and kept each one forever. A
-- finished result is now written here first and cached in Redis for
-- CELERY_RESULT_EXPIRES seconds; a reader that misses Redis loads it from this
-- table. Rows are summaries: no conversation messages, no task arguments, no
-- inline output, no traceback text (the run's error, logs and output table are
-- on its scheduled_runs row).
--
-- Service role only. RLS is on with no policy, so anon and authenticated read
-- and write nothing; the worker and the API use the service role.
CREATE TABLE IF NOT EXISTS public.celery_task_results (
  task_id     text PRIMARY KEY,
  status      text NOT NULL
    CHECK (status IN ('PENDING', 'RECEIVED', 'STARTED', 'RETRY', 'SUCCESS', 'FAILURE', 'REVOKED')),
  result      jsonb,
  traceback   text,
  date_done   timestamptz,
  name        text,
  user_id     text,
  schedule_id text,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS celery_task_results_date_done_idx
  ON public.celery_task_results (date_done);
CREATE INDEX IF NOT EXISTS celery_task_results_schedule_id_idx
  ON public.celery_task_results (schedule_id);

ALTER TABLE public.celery_task_results ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.celery_task_results FROM PUBLIC, anon, authenticated;
GRANT ALL ON public.celery_task_results TO service_role;

-- Retention (CELERY_RESULT_RETENTION_DAYS): delete at most batch_size finished
-- rows older than cutoff, oldest first on the date_done index. SKIP LOCKED and
-- the bounded batch keep each call short; the daily purge task calls it until a
-- batch comes back short. An unfinished task (PENDING, RECEIVED, STARTED,
-- RETRY) is never deleted.
CREATE OR REPLACE FUNCTION public.purge_celery_task_results(cutoff timestamptz, batch_size integer)
RETURNS integer
LANGUAGE plpgsql
SET search_path = public
AS $$
DECLARE
  deleted integer;
BEGIN
  IF cutoff IS NULL THEN
    RAISE EXCEPTION 'cutoff is required';
  END IF;
  IF batch_size IS NULL OR batch_size < 1 OR batch_size > 10000 THEN
    RAISE EXCEPTION 'batch_size must be between 1 and 10000';
  END IF;
  WITH doomed AS (
    SELECT task_id
      FROM public.celery_task_results
     WHERE date_done < cutoff
       AND status IN ('SUCCESS', 'FAILURE', 'REVOKED')
     ORDER BY date_done
     LIMIT batch_size
     FOR UPDATE SKIP LOCKED
  )
  DELETE FROM public.celery_task_results r
   USING doomed
   WHERE r.task_id = doomed.task_id;
  GET DIAGNOSTICS deleted = ROW_COUNT;
  RETURN deleted;
END;
$$;

REVOKE ALL ON FUNCTION public.purge_celery_task_results(timestamptz, integer) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.purge_celery_task_results(timestamptz, integer) TO service_role;
