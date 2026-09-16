DROP POLICY IF EXISTS "Read creator profiles of accessible reports" ON public.profiles;

CREATE POLICY "Read creator profiles of accessible reports"
  ON public.profiles
  FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1
      FROM public.reports r
      WHERE (r.created_by = profiles.id OR r.created_by = profiles.user_id)
        AND r.organization_id = public.current_org_id()
        AND (
          r.created_by = auth.uid()
          OR public.is_resource_collaborator('analysis'::text, r.analysis_id)
          OR (r.project_id IS NOT NULL AND public.is_resource_collaborator('project'::text, r.project_id))
          OR public.is_resource_collaborator('report'::text, r.id)
        )
    )
  );