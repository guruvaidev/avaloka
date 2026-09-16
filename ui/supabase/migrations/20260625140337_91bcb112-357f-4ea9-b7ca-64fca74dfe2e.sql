ALTER TABLE public.analyses
  ADD COLUMN IF NOT EXISTS dataset_id text,
  ADD COLUMN IF NOT EXISTS thread_id  text,
  ADD COLUMN IF NOT EXISTS filename   text,
  ADD COLUMN IF NOT EXISTS schema     jsonb,
  ADD COLUMN IF NOT EXISTS samples    jsonb,
  ADD COLUMN IF NOT EXISTS viz_config jsonb;