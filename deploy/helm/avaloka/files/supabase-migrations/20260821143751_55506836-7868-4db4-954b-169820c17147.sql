ALTER TABLE public.projects DROP CONSTRAINT IF EXISTS projects_organization_id_fkey;
ALTER TABLE public.analyses DROP CONSTRAINT IF EXISTS analyses_organization_id_fkey;

CREATE OR REPLACE FUNCTION public.set_row_organization()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
  IF NEW.organization_id IS NULL THEN
    NEW.organization_id := public.current_org_id();
  END IF;
  RETURN NEW;
END;
$$;
REVOKE ALL ON FUNCTION public.set_row_organization() FROM PUBLIC, anon, authenticated;

UPDATE public.projects p
SET organization_id = pr.organization_id
FROM public.profiles pr
WHERE p.organization_id IS NULL
  AND pr.organization_id IS NOT NULL
  AND (pr.id = p.owner_id OR pr.user_id = p.owner_id);

UPDATE public.analyses a
SET organization_id = pr.organization_id
FROM public.profiles pr
WHERE a.organization_id IS NULL
  AND pr.organization_id IS NOT NULL
  AND (pr.id = a.owner_id OR pr.user_id = a.owner_id);

UPDATE public.analyses a
SET organization_id = p.organization_id
FROM public.projects p
WHERE a.organization_id IS NULL
  AND a.project_id = p.id
  AND p.organization_id IS NOT NULL;