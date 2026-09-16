CREATE POLICY "read projects via accessible report"
ON public.projects
FOR SELECT
TO authenticated
USING (
  EXISTS (
    SELECT 1
    FROM public.reports r
    WHERE r.project_id = projects.id
      AND (
        r.created_by = auth.uid()
        OR public.is_resource_collaborator('report', r.id)
      )
  )
);