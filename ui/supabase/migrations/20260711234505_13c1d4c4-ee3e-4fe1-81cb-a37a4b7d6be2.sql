-- Allow users invited directly to a report (resource_type='report') to view it,
-- and let their name/avatar resolve via the creator profile policy.

DROP POLICY IF EXISTS "Users view own or shared reports" ON public.reports;
CREATE POLICY "Users view own or shared reports"
  ON public.reports
  FOR SELECT
  USING (
    (organization_id = current_org_id())
    AND (
      (created_by = auth.uid())
      OR is_resource_collaborator('analysis', analysis_id)
      OR ((project_id IS NOT NULL) AND is_resource_collaborator('project', project_id))
      OR is_resource_collaborator('report', id)
    )
  );

DROP POLICY IF EXISTS "Read creator profiles of accessible reports" ON public.profiles;
CREATE POLICY "Read creator profiles of accessible reports"
  ON public.profiles
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.reports r
      WHERE r.created_by = profiles.id
        AND (
          r.created_by = auth.uid()
          OR is_resource_collaborator('analysis', r.analysis_id)
          OR (r.project_id IS NOT NULL AND is_resource_collaborator('project', r.project_id))
          OR is_resource_collaborator('report', r.id)
        )
    )
  );