DROP POLICY IF EXISTS "collaborators read shared analyses" ON public.analyses;
CREATE POLICY "collaborators read shared analyses"
ON public.analyses
FOR SELECT
TO authenticated
USING (
  public.is_resource_collaborator('analysis', id)
  OR (project_id IS NOT NULL AND public.is_resource_collaborator('project', project_id))
);