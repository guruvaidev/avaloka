DROP POLICY IF EXISTS "Owner can update their organization" ON public.organizations;

CREATE POLICY "Owner or admin can update their organization"
ON public.organizations
FOR UPDATE TO authenticated
USING (
  owner_profile_id = auth.uid()
  OR (id = public.current_org_id() AND public.has_role(auth.uid(), 'admin'))
)
WITH CHECK (
  owner_profile_id = auth.uid()
  OR (id = public.current_org_id() AND public.has_role(auth.uid(), 'admin'))
);