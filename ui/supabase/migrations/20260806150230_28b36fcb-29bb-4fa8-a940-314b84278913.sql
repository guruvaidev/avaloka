ALTER TABLE public.analyses ADD COLUMN IF NOT EXISTS parent_analysis_id uuid REFERENCES public.analyses(id) ON DELETE CASCADE;
CREATE INDEX IF NOT EXISTS analyses_parent_analysis_id_idx ON public.analyses(parent_analysis_id);
CREATE INDEX IF NOT EXISTS analyses_session_id_idx ON public.analyses(session_id);