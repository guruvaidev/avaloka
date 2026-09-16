
-- Allow collaborators (shared-with users) to read the project row so the sidebar can list it
CREATE POLICY "collaborators read shared projects"
ON public.projects
FOR SELECT
TO authenticated
USING (public.is_resource_collaborator('project', id));

-- Allow same-org members to read minimal profile info (needed to show report creator name)
CREATE POLICY "Org members read profiles"
ON public.profiles
FOR SELECT
TO authenticated
USING (organization_id IS NOT NULL AND organization_id = public.current_org_id());
