CREATE TABLE public.report_comments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  report_id uuid NOT NULL REFERENCES public.reports(id) ON DELETE CASCADE,
  author_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  parent_id uuid REFERENCES public.report_comments(id) ON DELETE CASCADE,
  content text NOT NULL,
  attachments jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamp with time zone NOT NULL DEFAULT now()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.report_comments TO authenticated;
GRANT ALL ON public.report_comments TO service_role;

ALTER TABLE public.report_comments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read report comments they can access"
ON public.report_comments
FOR SELECT
TO authenticated
USING (
  EXISTS (
    SELECT 1
    FROM public.reports r
    WHERE r.id = report_id
      AND (
        r.created_by = auth.uid()
        OR public.is_resource_collaborator('report', r.id)
      )
  )
);

CREATE POLICY "Users can comment on reports they can access"
ON public.report_comments
FOR INSERT
TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1
    FROM public.reports r
    WHERE r.id = report_id
      AND (
        r.created_by = auth.uid()
        OR public.is_resource_collaborator('report', r.id)
      )
  )
  AND author_id = public.current_profile_id()
);

CREATE POLICY "Authors can update their own report comments"
ON public.report_comments
FOR UPDATE
TO authenticated
USING (author_id = public.current_profile_id())
WITH CHECK (author_id = public.current_profile_id());

CREATE POLICY "Authors can delete their own report comments"
ON public.report_comments
FOR DELETE
TO authenticated
USING (author_id = public.current_profile_id());