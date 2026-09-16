ALTER TABLE public.resource_collaborators
DROP CONSTRAINT IF EXISTS resource_collaborators_resource_type_check;

ALTER TABLE public.resource_collaborators
ADD CONSTRAINT resource_collaborators_resource_type_check
CHECK (resource_type IN ('project', 'analysis', 'report'));

CREATE POLICY "Report creators insert collaborators"
ON public.resource_collaborators
FOR INSERT
TO authenticated
WITH CHECK (
  resource_type = 'report'
  AND invited_by = auth.uid()
  AND EXISTS (
    SELECT 1
    FROM public.reports r
    JOIN public.app_users invited_user
      ON invited_user.id = resource_collaborators.app_user_id
    WHERE r.id = resource_collaborators.resource_id
      AND r.created_by = auth.uid()
      AND invited_user.organization_id = r.organization_id
  )
);