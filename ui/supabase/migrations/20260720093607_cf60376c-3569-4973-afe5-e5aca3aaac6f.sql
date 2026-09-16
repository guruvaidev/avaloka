CREATE POLICY "Read profiles in same organization"
ON public.profiles
FOR SELECT
TO authenticated
USING (
  organization_id IS NOT NULL
  AND organization_id = public.current_org_id()
);