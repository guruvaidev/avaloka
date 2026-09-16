DROP POLICY IF EXISTS "Users view own or shared reports" ON public.reports;

CREATE POLICY "Users view own or shared reports"
  ON public.reports
  FOR SELECT
  TO authenticated
  USING (
    created_by = auth.uid()
    OR public.is_resource_collaborator('report', id)
  );