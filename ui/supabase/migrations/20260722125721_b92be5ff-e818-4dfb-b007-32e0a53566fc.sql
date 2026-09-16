DROP POLICY IF EXISTS "Report creators insert collaborators" ON public.resource_collaborators;

CREATE POLICY "Report creators insert collaborators"
ON public.resource_collaborators
FOR INSERT
TO authenticated
WITH CHECK (
  resource_type = 'report'
  AND EXISTS (
    SELECT 1
    FROM public.reports r
    JOIN public.app_users invited_user ON invited_user.id = resource_collaborators.app_user_id
    WHERE r.id = resource_collaborators.resource_id
      AND r.created_by = public.current_profile_id()
      AND invited_user.organization_id = r.organization_id
  )
);

-- Also allow project/analysis creators to insert collaborators consistently.
DROP POLICY IF EXISTS "Project creators insert collaborators" ON public.resource_collaborators;
CREATE POLICY "Project creators insert collaborators"
ON public.resource_collaborators
FOR INSERT
TO authenticated
WITH CHECK (
  resource_type = 'project'
  AND EXISTS (
    SELECT 1
    FROM public.projects p
    JOIN public.app_users invited_user ON invited_user.id = resource_collaborators.app_user_id
    WHERE p.id = resource_collaborators.resource_id
      AND p.owner_id = public.current_profile_id()
      AND invited_user.organization_id = (
        SELECT organization_id FROM public.profiles WHERE id = public.current_profile_id()
      )
  )
);

DROP POLICY IF EXISTS "Analysis creators insert collaborators" ON public.resource_collaborators;
CREATE POLICY "Analysis creators insert collaborators"
ON public.resource_collaborators
FOR INSERT
TO authenticated
WITH CHECK (
  resource_type = 'analysis'
  AND EXISTS (
    SELECT 1
    FROM public.analyses a
    JOIN public.app_users invited_user ON invited_user.id = resource_collaborators.app_user_id
    WHERE a.id = resource_collaborators.resource_id
      AND a.owner_id = public.current_profile_id()
      AND invited_user.organization_id = (
        SELECT organization_id FROM public.profiles WHERE id = public.current_profile_id()
      )
  )
);