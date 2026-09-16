GRANT SELECT, INSERT, UPDATE, DELETE ON public.report_comments TO authenticated;
GRANT ALL ON public.report_comments TO service_role;

DROP POLICY IF EXISTS "Authors can delete their own report comments" ON public.report_comments;
DROP POLICY IF EXISTS "Authors can update their own report comments" ON public.report_comments;
DROP POLICY IF EXISTS "Users can comment on reports they can access" ON public.report_comments;
DROP POLICY IF EXISTS "Users can read report comments they can access" ON public.report_comments;
DROP POLICY IF EXISTS "report_collaborators_can_comment" ON public.report_comments;
DROP POLICY IF EXISTS "report_collaborators_can_read_comments" ON public.report_comments;
DROP POLICY IF EXISTS "report_owner_admin_can_manage_comments" ON public.report_comments;

CREATE POLICY "Report viewers can read comments"
ON public.report_comments
FOR SELECT
TO authenticated
USING (
  EXISTS (
    SELECT 1
    FROM public.reports r
    WHERE r.id = report_comments.report_id
  )
);

CREATE POLICY "Report viewers can add comments"
ON public.report_comments
FOR INSERT
TO authenticated
WITH CHECK (
  author_id = public.current_profile_id()
  AND EXISTS (
    SELECT 1
    FROM public.reports r
    WHERE r.id = report_comments.report_id
  )
);

CREATE POLICY "Authors can update their report comments"
ON public.report_comments
FOR UPDATE
TO authenticated
USING (author_id = public.current_profile_id())
WITH CHECK (
  author_id = public.current_profile_id()
  AND EXISTS (
    SELECT 1
    FROM public.reports r
    WHERE r.id = report_comments.report_id
  )
);

CREATE POLICY "Authors can delete their report comments"
ON public.report_comments
FOR DELETE
TO authenticated
USING (author_id = public.current_profile_id());