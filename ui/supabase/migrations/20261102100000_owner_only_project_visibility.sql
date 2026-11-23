CREATE OR REPLACE FUNCTION public.is_owner_of_org(_org uuid)
  RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER SET search_path TO 'public' AS $$
  SELECT _org IS NOT NULL AND EXISTS (
    SELECT 1 FROM public.organizations o
    WHERE o.id = _org AND o.owner_profile_id = public.current_profile_id())
$$;
REVOKE ALL ON FUNCTION public.is_owner_of_org(uuid) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.is_owner_of_org(uuid) TO authenticated, service_role;
DROP TRIGGER IF EXISTS trg_projects_set_org ON public.projects;
CREATE TRIGGER trg_projects_set_org BEFORE INSERT ON public.projects
  FOR EACH ROW EXECUTE FUNCTION public.set_row_organization();
UPDATE public.projects pr SET organization_id = p.organization_id FROM public.profiles p
 WHERE p.id = pr.owner_id AND pr.organization_id IS NULL AND p.organization_id IS NOT NULL;
DROP POLICY IF EXISTS "org members read org projects"   ON public.projects;
DROP POLICY IF EXISTS "org admins update org projects"  ON public.projects;
DROP POLICY IF EXISTS "org owner reads org projects"    ON public.projects;
DROP POLICY IF EXISTS "org owner updates org projects"  ON public.projects;
CREATE POLICY "org owner reads org projects" ON public.projects FOR SELECT TO authenticated
  USING (public.is_owner_of_org(organization_id));
CREATE POLICY "org owner updates org projects" ON public.projects FOR UPDATE TO authenticated
  USING (public.is_owner_of_org(organization_id)) WITH CHECK (public.is_owner_of_org(organization_id));
DROP POLICY IF EXISTS "org members read org project analyses"  ON public.analyses;
DROP POLICY IF EXISTS "org admins update org project analyses" ON public.analyses;
DROP POLICY IF EXISTS "org owners read member analyses"        ON public.analyses;
DROP POLICY IF EXISTS "org owner reads org project analyses"   ON public.analyses;
DROP POLICY IF EXISTS "org owner updates org project analyses" ON public.analyses;
CREATE POLICY "org owner reads org project analyses" ON public.analyses FOR SELECT TO authenticated
  USING (project_id IS NOT NULL AND public.is_owner_of_org(organization_id));
CREATE POLICY "org owner updates org project analyses" ON public.analyses FOR UPDATE TO authenticated
  USING (project_id IS NOT NULL AND public.is_owner_of_org(organization_id))
  WITH CHECK (public.is_owner_of_org(organization_id));
