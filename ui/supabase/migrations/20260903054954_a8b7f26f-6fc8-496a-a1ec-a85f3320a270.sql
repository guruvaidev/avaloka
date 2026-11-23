-- session_snapshot: JSON snapshot of session state captured per analysis.
-- Read by server.py _q(); adds column missing from migrations (present in hosted DB via manual edit).
ALTER TABLE public.analyses
  ADD COLUMN IF NOT EXISTS session_snapshot jsonb;
