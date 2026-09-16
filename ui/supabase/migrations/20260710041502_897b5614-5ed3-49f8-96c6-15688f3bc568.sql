
-- resource_collaborators
CREATE TABLE IF NOT EXISTS public.resource_collaborators (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  resource_type TEXT NOT NULL CHECK (resource_type IN ('project','analysis')),
  resource_id UUID NOT NULL,
  app_user_id UUID NOT NULL REFERENCES public.app_users(id) ON DELETE CASCADE,
  access_level TEXT NOT NULL DEFAULT 'view' CHECK (access_level IN ('view','edit','comment')),
  department TEXT,
  invited_by UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (resource_type, resource_id, app_user_id)
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.resource_collaborators TO authenticated;
GRANT ALL ON public.resource_collaborators TO service_role;
ALTER TABLE public.resource_collaborators ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members read collaborators"
  ON public.resource_collaborators FOR SELECT TO authenticated
  USING (EXISTS (
    SELECT 1 FROM public.app_users au
    WHERE au.id = resource_collaborators.app_user_id
      AND au.organization_id = public.current_org_id()
  ));

CREATE POLICY "Org members write collaborators"
  ON public.resource_collaborators FOR INSERT TO authenticated
  WITH CHECK (EXISTS (
    SELECT 1 FROM public.app_users au
    WHERE au.id = resource_collaborators.app_user_id
      AND au.organization_id = public.current_org_id()
  ));

CREATE POLICY "Org members update collaborators"
  ON public.resource_collaborators FOR UPDATE TO authenticated
  USING (EXISTS (
    SELECT 1 FROM public.app_users au
    WHERE au.id = resource_collaborators.app_user_id
      AND au.organization_id = public.current_org_id()
  ));

CREATE POLICY "Org members delete collaborators"
  ON public.resource_collaborators FOR DELETE TO authenticated
  USING (EXISTS (
    SELECT 1 FROM public.app_users au
    WHERE au.id = resource_collaborators.app_user_id
      AND au.organization_id = public.current_org_id()
  ));

CREATE TRIGGER trg_resource_collab_updated_at
  BEFORE UPDATE ON public.resource_collaborators
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- resource_share_links
CREATE TABLE IF NOT EXISTS public.resource_share_links (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  resource_type TEXT NOT NULL CHECK (resource_type IN ('project','analysis')),
  resource_id UUID NOT NULL,
  share_token TEXT NOT NULL DEFAULT replace(gen_random_uuid()::text,'-',''),
  link_access TEXT NOT NULL DEFAULT 'view' CHECK (link_access IN ('view','edit','comment','restricted')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (resource_type, resource_id),
  UNIQUE (share_token)
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.resource_share_links TO authenticated;
GRANT ALL ON public.resource_share_links TO service_role;
ALTER TABLE public.resource_share_links ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Authenticated read share links"
  ON public.resource_share_links FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated write share links"
  ON public.resource_share_links FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "Authenticated update share links"
  ON public.resource_share_links FOR UPDATE TO authenticated USING (true);

CREATE TRIGGER trg_resource_share_links_updated_at
  BEFORE UPDATE ON public.resource_share_links
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- Helper: check if the current auth user is a collaborator on a resource
CREATE OR REPLACE FUNCTION public.is_resource_collaborator(_resource_type TEXT, _resource_id UUID)
RETURNS BOOLEAN
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.resource_collaborators rc
    JOIN public.app_users au ON au.id = rc.app_user_id
    WHERE rc.resource_type = _resource_type
      AND rc.resource_id = _resource_id
      AND au.auth_user_id = auth.uid()
  )
$$;

-- Tighten reports SELECT: creator OR shared collaborator on the analysis/project
DROP POLICY IF EXISTS "Org members view reports" ON public.reports;

CREATE POLICY "Users view own or shared reports"
  ON public.reports FOR SELECT TO authenticated
  USING (
    organization_id = public.current_org_id()
    AND (
      created_by = auth.uid()
      OR public.is_resource_collaborator('analysis', analysis_id)
      OR (project_id IS NOT NULL AND public.is_resource_collaborator('project', project_id))
    )
  );
