ALTER TABLE public.projects ADD COLUMN IF NOT EXISTS organization_id uuid REFERENCES public.organizations(id) ON DELETE SET NULL;
ALTER TABLE public.analyses ADD COLUMN IF NOT EXISTS organization_id uuid REFERENCES public.organizations(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_projects_org ON public.projects(organization_id);
CREATE INDEX IF NOT EXISTS idx_analyses_org ON public.analyses(organization_id);

UPDATE public.projects p
SET organization_id = pr.organization_id
FROM public.profiles pr
WHERE p.organization_id IS NULL
  AND pr.organization_id IS NOT NULL
  AND (pr.id = p.owner_id OR pr.user_id = p.owner_id)
  AND EXISTS (SELECT 1 FROM public.organizations o WHERE o.id = pr.organization_id);

UPDATE public.analyses a
SET organization_id = pr.organization_id
FROM public.profiles pr
WHERE a.organization_id IS NULL
  AND pr.organization_id IS NOT NULL
  AND (pr.id = a.owner_id OR pr.user_id = a.owner_id)
  AND EXISTS (SELECT 1 FROM public.organizations o WHERE o.id = pr.organization_id);

UPDATE public.analyses a
SET organization_id = p.organization_id
FROM public.projects p
WHERE a.organization_id IS NULL
  AND a.project_id = p.id
  AND p.organization_id IS NOT NULL;

CREATE OR REPLACE FUNCTION public.set_row_organization()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
  _org uuid;
BEGIN
  IF NEW.organization_id IS NULL THEN
    _org := public.current_org_id();
    IF _org IS NOT NULL AND EXISTS (SELECT 1 FROM public.organizations o WHERE o.id = _org) THEN
      NEW.organization_id := _org;
    END IF;
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_projects_set_org ON public.projects;
CREATE TRIGGER trg_projects_set_org BEFORE INSERT ON public.projects
FOR EACH ROW EXECUTE FUNCTION public.set_row_organization();

DROP TRIGGER IF EXISTS trg_analyses_set_org ON public.analyses;
CREATE TRIGGER trg_analyses_set_org BEFORE INSERT ON public.analyses
FOR EACH ROW EXECUTE FUNCTION public.set_row_organization();

DROP POLICY IF EXISTS "org members read org projects" ON public.projects;
CREATE POLICY "org members read org projects" ON public.projects
FOR SELECT TO authenticated
USING (organization_id IS NOT NULL AND organization_id = public.current_org_id());

DROP POLICY IF EXISTS "org admins update org projects" ON public.projects;
CREATE POLICY "org admins update org projects" ON public.projects
FOR UPDATE TO authenticated
USING (organization_id IS NOT NULL AND organization_id = public.current_org_id() AND public.is_org_admin())
WITH CHECK (organization_id = public.current_org_id());

DROP POLICY IF EXISTS "org members read org analyses" ON public.analyses;
CREATE POLICY "org members read org analyses" ON public.analyses
FOR SELECT TO authenticated
USING (organization_id IS NOT NULL AND organization_id = public.current_org_id());

DROP POLICY IF EXISTS "org admins update org analyses" ON public.analyses;
CREATE POLICY "org admins update org analyses" ON public.analyses
FOR UPDATE TO authenticated
USING (organization_id IS NOT NULL AND organization_id = public.current_org_id() AND public.is_org_admin())
WITH CHECK (organization_id = public.current_org_id());