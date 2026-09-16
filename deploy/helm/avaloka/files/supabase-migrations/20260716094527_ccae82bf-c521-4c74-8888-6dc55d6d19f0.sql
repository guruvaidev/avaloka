CREATE POLICY "Collaborators can view shared projects"
ON public.projects
FOR SELECT
TO authenticated
USING (public.is_resource_collaborator('project', id));