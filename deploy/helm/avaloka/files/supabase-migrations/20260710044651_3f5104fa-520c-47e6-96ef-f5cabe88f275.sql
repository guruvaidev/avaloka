DROP POLICY IF EXISTS "Creators delete their reports" ON public.reports;
CREATE POLICY "Creators or admins delete reports"
ON public.reports FOR DELETE TO authenticated
USING (
  organization_id = current_org_id()
  AND (created_by = auth.uid() OR public.has_role(auth.uid(), 'admin'))
);