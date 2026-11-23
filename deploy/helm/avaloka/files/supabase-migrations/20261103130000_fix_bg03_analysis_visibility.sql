-- BG-03 (#646): organisation members see the organisation's
-- analyses only when they belong to a project; the org-admin update path is
-- narrowed the same way. TRUNCATE is revoked from the API roles on every public
-- table.
--
-- Reconciled for develop-1.6 (bugfix/1.6-migration-fix-reconciled):
--   * renumbered from 20261101100000 to run after #649, #650, #661, develop-1.6's
--     newest (20261103090000) and D-2 (20261103120000); in both the UI and chart copies;
--   * its own BEGIN/COMMIT removed: the chart's migrate hook runs each file in one
--     transaction together with its schema_migrations record, and an inner COMMIT
--     ended that transaction early;
--   * otherwise unchanged. Both policies stay inside the caller's own organisation
--     (current_org_id()), so no admin cross-tenant path returns (D-2).

DROP POLICY IF EXISTS "org members read org analyses" ON public.analyses;
DROP POLICY IF EXISTS "org members read org project analyses" ON public.analyses;
CREATE POLICY "org members read org project analyses"
  ON public.analyses FOR SELECT TO authenticated
  USING (project_id IS NOT NULL AND organization_id IS NOT NULL
         AND organization_id = public.current_org_id());
DROP POLICY IF EXISTS "org admins update org analyses" ON public.analyses;
DROP POLICY IF EXISTS "org admins update org project analyses" ON public.analyses;
CREATE POLICY "org admins update org project analyses"
  ON public.analyses FOR UPDATE TO authenticated
  USING (project_id IS NOT NULL AND organization_id IS NOT NULL
         AND organization_id = public.current_org_id() AND public.is_org_admin())
  WITH CHECK (organization_id = public.current_org_id());
DO $$
DECLARE r record;
BEGIN
  FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP
    EXECUTE format('REVOKE TRUNCATE ON TABLE public.%I FROM anon, authenticated', r.tablename);
  END LOOP;
END $$;
