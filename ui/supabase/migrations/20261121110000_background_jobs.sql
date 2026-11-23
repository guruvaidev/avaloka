-- =====================================================================
-- Background jobs: one ledger for every long-running operation, plus the
-- chat-thread outbox that tells the user's conversation when one finishes.
--
-- Builds on ml_jobs (20261117130000_ml_jobs_endpoints.sql). This file is
-- safe in EITHER order with that migration: the CREATE TABLE below repeats
-- its exact column set and is a no-op when it already exists; every column
-- here is ADD COLUMN IF NOT EXISTS; the kind check is dropped and
-- re-created with the union of both vocabularies. Running this file twice
-- changes nothing.
--
-- Rows are owned by user_id (auth.uid()). The backend writes with the
-- service-role key and enforces ownership in code; RLS is the safety net
-- for any user-JWT query. No customer data lives here: no column names, no
-- file names, no URIs, no raw error text in anything a notification reads.
-- =====================================================================

create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table if not exists public.ml_jobs (
    id              uuid primary key default gen_random_uuid(),
    user_id         uuid not null references auth.users(id) on delete cascade,
    session_id      text,
    kind            text not null,
    status          text not null check (status in ('queued','running','succeeded','failed','cancelled')),
    mlflow_run_id   text,
    ray_job_name    text,
    celery_task_id  text,
    error           text,
    attempts        integer not null default 0,
    created_at      timestamptz not null default now(),
    started_at      timestamptz,
    finished_at     timestamptz,
    updated_at      timestamptz not null default now()
);
create index if not exists ml_jobs_user_created on public.ml_jobs (user_id, created_at desc);
create index if not exists ml_jobs_run on public.ml_jobs (mlflow_run_id);

-- The ledger columns.
alter table public.ml_jobs add column if not exists thread_id       text;
alter table public.ml_jobs add column if not exists analysis_id     uuid;
alter table public.ml_jobs add column if not exists resource_type   text;
alter table public.ml_jobs add column if not exists resource_id     text;
alter table public.ml_jobs add column if not exists label           text;
alter table public.ml_jobs add column if not exists fail_reason     text;
alter table public.ml_jobs add column if not exists notified_at     timestamptz;
alter table public.ml_jobs add column if not exists notification_id uuid;
alter table public.ml_jobs add column if not exists origin          text not null default 'api';

-- Closed vocabularies. The kind check is the union of the training/serving
-- kinds and the ledger kinds; re-created so either apply order ends here.
alter table public.ml_jobs drop constraint if exists ml_jobs_kind_check;
alter table public.ml_jobs add constraint ml_jobs_kind_check check (kind in (
    'training','register','deploy_endpoint','stop_endpoint','cross_validate','reap',
    'analysis','scheduled_run','transfer','ray_job','sampling','inference_batch','swarm'));
alter table public.ml_jobs drop constraint if exists ml_jobs_resource_type_check;
alter table public.ml_jobs add constraint ml_jobs_resource_type_check check (resource_type is null or resource_type in (
    'analysis','dataset','model','endpoint','transfer','schedule','mission'));
alter table public.ml_jobs drop constraint if exists ml_jobs_fail_reason_check;
alter table public.ml_jobs add constraint ml_jobs_fail_reason_check check (fail_reason is null or fail_reason in (
    'failed','timed_out','cancelled','no_worker','cluster_error','refused','not_answered'));
alter table public.ml_jobs drop constraint if exists ml_jobs_origin_check;
alter table public.ml_jobs add constraint ml_jobs_origin_check check (origin in ('api','celery','thread'));

create index if not exists ml_jobs_thread_open on public.ml_jobs (thread_id)
    where status in ('queued','running');

drop trigger if exists ml_jobs_set_updated_at on public.ml_jobs;
create trigger ml_jobs_set_updated_at before update on public.ml_jobs
for each row execute function public.set_updated_at();

alter table public.ml_jobs enable row level security;
drop policy if exists "ml_jobs_select_own" on public.ml_jobs;
create policy "ml_jobs_select_own" on public.ml_jobs for select using (user_id = auth.uid());

-- Chat-thread outbox. Written by whoever finishes a job (API, Celery worker,
-- a mission thread); read and marked delivered ONLY by the API, at a turn
-- boundary of the thread, so a note never lands inside a turn in progress.
-- `text` is the rendered note, already redacted; `unique (job_id)` is the
-- "one note per job, ever" rule, so a retried finish inserts nothing.
create table if not exists public.job_chat_notes (
    id            uuid primary key default gen_random_uuid(),
    job_id        uuid not null references public.ml_jobs(id) on delete cascade,
    user_id       uuid not null references auth.users(id) on delete cascade,
    thread_id     text not null,
    analysis_id   uuid,
    text          text not null,
    created_at    timestamptz not null default now(),
    delivered_at  timestamptz,
    batch_id      uuid,
    unique (job_id)
);
create index if not exists job_chat_notes_pending on public.job_chat_notes (thread_id, created_at)
    where delivered_at is null;

alter table public.job_chat_notes enable row level security;
drop policy if exists "job_chat_notes_select_own" on public.job_chat_notes;
create policy "job_chat_notes_select_own" on public.job_chat_notes for select using (user_id = auth.uid());
-- No insert/update/delete policy: the service role alone writes here.

-- RLS does not apply to TRUNCATE, and Supabase's default privileges grant it
-- to the API roles on every new table (see 20261121090000_api_roles_no_truncate).
-- This file runs after that one, so it revokes for its own tables.
revoke truncate on table public.ml_jobs, public.job_chat_notes from anon, authenticated;
