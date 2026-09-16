CREATE TABLE public.analysis_comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id UUID NOT NULL REFERENCES public.analyses(id) ON DELETE CASCADE,
  author_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  parent_id UUID REFERENCES public.analysis_comments(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  attachments JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_analysis_comments_analysis ON public.analysis_comments(analysis_id, created_at);
CREATE INDEX idx_analysis_comments_parent ON public.analysis_comments(parent_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.analysis_comments TO authenticated;
GRANT ALL ON public.analysis_comments TO service_role;

ALTER TABLE public.analysis_comments ENABLE ROW LEVEL SECURITY;

-- Read: owner of analysis, or collaborator on it
CREATE POLICY "Read comments on accessible analyses"
ON public.analysis_comments FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM public.analyses a
    WHERE a.id = analysis_comments.analysis_id
      AND (
        a.owner_id = public.current_profile_id()
        OR public.is_resource_collaborator('analysis', a.id)
      )
  )
);

-- Insert: must be author, and must have access to the analysis
CREATE POLICY "Insert comments on accessible analyses"
ON public.analysis_comments FOR INSERT TO authenticated
WITH CHECK (
  author_id = public.current_profile_id()
  AND EXISTS (
    SELECT 1 FROM public.analyses a
    WHERE a.id = analysis_comments.analysis_id
      AND (
        a.owner_id = public.current_profile_id()
        OR public.is_resource_collaborator('analysis', a.id)
      )
  )
);

-- Update own
CREATE POLICY "Update own comments"
ON public.analysis_comments FOR UPDATE TO authenticated
USING (author_id = public.current_profile_id())
WITH CHECK (author_id = public.current_profile_id());

-- Delete own
CREATE POLICY "Delete own comments"
ON public.analysis_comments FOR DELETE TO authenticated
USING (author_id = public.current_profile_id());