
-- Allow reading projects when the user collaborates on any analysis inside the project
CREATE POLICY "collaborators read projects via analyses"
ON public.projects FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM public.analyses a
    WHERE a.project_id = projects.id
      AND public.is_resource_collaborator('analysis', a.id)
  )
);

-- Allow reading a profile when that profile authored a report the current user can access
CREATE POLICY "Read creator profiles of accessible reports"
ON public.profiles FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM public.reports r
    WHERE r.created_by = profiles.id
      AND (
        r.created_by = auth.uid()
        OR public.is_resource_collaborator('analysis', r.analysis_id)
        OR (r.project_id IS NOT NULL AND public.is_resource_collaborator('project', r.project_id))
      )
  )
);
