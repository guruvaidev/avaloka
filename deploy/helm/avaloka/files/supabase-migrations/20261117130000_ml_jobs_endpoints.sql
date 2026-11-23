-- =====================================================================
-- ml_jobs / ml_models / ml_endpoints / ml_cross_validation
-- Durable state for model training, registration, serving endpoints and
-- cross-validation. The chat and the inference panel read THIS, never the
-- Celery result backend or an MLflow artifact (the Redis-outage lesson:
-- state that lives only in Redis vanishes with it).
--
-- Rows are owned by user_id (auth.uid()). The backend writes with the
-- service-role key and enforces ownership in code, as scheduled_runs does;
-- the RLS policies below are the safety net for any user-JWT query.
-- No customer data lives here beyond what the user already owns in MLflow:
-- no column names, no file names, no dataset URIs.
-- =====================================================================

create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

-- One row per background operation. `error` is a one-line, user-safe reason.
create table if not exists public.ml_jobs (
    id              uuid primary key default gen_random_uuid(),
    user_id         uuid not null references auth.users(id) on delete cascade,
    session_id      text,
    kind            text not null check (kind in ('training','register','deploy_endpoint','stop_endpoint','cross_validate','reap')),
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

-- The MLflow model-registry record for a run: name, version, alias.
create table if not exists public.ml_models (
    mlflow_run_id    text primary key,
    user_id          uuid not null references auth.users(id) on delete cascade,
    registered_name  text not null,
    version          integer not null,
    alias            text,
    task_type        text not null check (task_type in ('classification','regression')),
    requires_gpu     boolean not null default false,
    signature_logged boolean not null default false,
    metrics          jsonb not null default '{}'::jsonb,
    created_at       timestamptz not null default now(),
    unique (registered_name, version)
);

-- The serving endpoint for a run and where it is in its life.
--   pending -> deploying -> ready | failed ; ready -> stopped ; failed|stopped -> pending
-- Idle scale-to-zero never changes `status`: a ready endpoint whose pod is
-- asleep wakes on the next request.
create table if not exists public.ml_endpoints (
    mlflow_run_id   text primary key,
    user_id         uuid not null references auth.users(id) on delete cascade,
    status          text not null check (status in ('pending','deploying','ready','failed','stopped')),
    server          text not null default 'cpu' check (server in ('cpu','gpu')),
    model_uri       text,
    error           text,
    last_request_at timestamptz,
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

-- k-fold result for a run (written on 1.7+; the table exists on 1.6 so the
-- port is additive). `aggregate`/`folds` hold matrices and fold metrics.
create table if not exists public.ml_cross_validation (
    mlflow_run_id   text primary key,
    user_id         uuid not null references auth.users(id) on delete cascade,
    status          text not null check (status in ('queued','running','succeeded','skipped','failed')),
    k               integer,
    splitter        text,
    reason          text,
    aggregate       jsonb,
    folds           jsonb,
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

drop trigger if exists ml_jobs_set_updated_at on public.ml_jobs;
create trigger ml_jobs_set_updated_at before update on public.ml_jobs
for each row execute function public.set_updated_at();
drop trigger if exists ml_endpoints_set_updated_at on public.ml_endpoints;
create trigger ml_endpoints_set_updated_at before update on public.ml_endpoints
for each row execute function public.set_updated_at();
drop trigger if exists ml_cross_validation_set_updated_at on public.ml_cross_validation;
create trigger ml_cross_validation_set_updated_at before update on public.ml_cross_validation
for each row execute function public.set_updated_at();

alter table public.ml_jobs enable row level security;
alter table public.ml_models enable row level security;
alter table public.ml_endpoints enable row level security;
alter table public.ml_cross_validation enable row level security;

drop policy if exists "ml_jobs_select_own" on public.ml_jobs;
create policy "ml_jobs_select_own" on public.ml_jobs for select using (user_id = auth.uid());
drop policy if exists "ml_models_select_own" on public.ml_models;
create policy "ml_models_select_own" on public.ml_models for select using (user_id = auth.uid());
drop policy if exists "ml_endpoints_select_own" on public.ml_endpoints;
create policy "ml_endpoints_select_own" on public.ml_endpoints for select using (user_id = auth.uid());
drop policy if exists "ml_cross_validation_select_own" on public.ml_cross_validation;
create policy "ml_cross_validation_select_own" on public.ml_cross_validation for select using (user_id = auth.uid());
