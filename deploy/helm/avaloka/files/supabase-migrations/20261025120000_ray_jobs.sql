-- Ray job records for the Ray jobs panel (app/infra/ray_jobs_store.py).
-- One row per Ray submission; written by app/infra/ray_job_runner.py.
-- Run once in the Supabase SQL editor (or your migration tool).

create table if not exists public.ray_jobs (
    submission_id  text primary key,                 -- Ray submission id, e.g. dta-dta-e8bc6d2358664de9
    user_id        text not null,                    -- auth.uid() of the user who started it (JWT sub)
    thread_id      text,                             -- chat thread that started it
    analysis_id    text,                             -- analyses.id, when known
    kind           text not null default 'data_transfer',  -- data_transfer | mta_training | inference
    dashboard_url  text not null,                    -- Ray cluster that ran it; never returned to the UI
    source         text,                             -- credentials are stripped before writing
    destination    text,
    status         text not null default 'PENDING',  -- PENDING | RUNNING | SUCCEEDED | FAILED | STOPPED
    message        text,
    rows_written   integer,
    runtime_s      double precision,
    created_at     timestamptz not null default now(),
    ended_at       timestamptz,
    updated_at     timestamptz not null default now()
);

create index if not exists ray_jobs_user_created_idx
    on public.ray_jobs (user_id, created_at desc);

-- Only the backend (service-role key) reads and writes this table; it filters by
-- user_id itself. RLS on with no policies keeps browsers using the anon key out.
alter table public.ray_jobs enable row level security;
