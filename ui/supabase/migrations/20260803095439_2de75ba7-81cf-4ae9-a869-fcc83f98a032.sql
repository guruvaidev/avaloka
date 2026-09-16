-- 1) app_users: split broad ALL policy into read-for-org-members + admin-only writes
CREATE OR REPLACE FUNCTION public.is_org_admin()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    public.has_role(auth.uid(), 'admin'::public.app_role)
    OR EXISTS (
      SELECT 1 FROM public.organizations o
      WHERE o.owner_profile_id = public.current_profile_id()
        AND o.id = public.current_org_id()
    )
    OR EXISTS (
      SELECT 1 FROM public.app_users au
      WHERE au.organization_id = public.current_org_id()
        AND (au.auth_user_id = auth.uid() OR au.profile_id = public.current_profile_id())
        AND lower(au.role) IN ('owner', 'admin')
    )
$$;

REVOKE ALL ON FUNCTION public.is_org_admin() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.is_org_admin() TO authenticated, service_role;

DROP POLICY IF EXISTS "org members manage app_users" ON public.app_users;

CREATE POLICY "org members read app_users"
ON public.app_users FOR SELECT TO authenticated
USING (organization_id = public.current_org_id());

CREATE POLICY "org admins insert app_users"
ON public.app_users FOR INSERT TO authenticated
WITH CHECK (organization_id = public.current_org_id() AND public.is_org_admin());

CREATE POLICY "org admins update app_users"
ON public.app_users FOR UPDATE TO authenticated
USING (organization_id = public.current_org_id() AND public.is_org_admin())
WITH CHECK (organization_id = public.current_org_id() AND public.is_org_admin());

CREATE POLICY "org admins delete app_users"
ON public.app_users FOR DELETE TO authenticated
USING (organization_id = public.current_org_id() AND public.is_org_admin());

-- 2) user_roles: block self-assignment of roles entirely (bootstrap via service role)
CREATE POLICY "no self role assignment insert"
ON public.user_roles AS RESTRICTIVE FOR INSERT TO authenticated
WITH CHECK (user_id <> auth.uid());

CREATE POLICY "no self role assignment update"
ON public.user_roles AS RESTRICTIVE FOR UPDATE TO authenticated
USING (user_id <> auth.uid())
WITH CHECK (user_id <> auth.uid());

-- 3) projects: scope collaborator policy to authenticated
DROP POLICY IF EXISTS "collaborators read projects via analyses" ON public.projects;
CREATE POLICY "collaborators read projects via analyses"
ON public.projects FOR SELECT TO authenticated
USING (EXISTS (
  SELECT 1 FROM public.analyses a
  WHERE a.project_id = projects.id
    AND public.is_resource_collaborator('analysis', a.id)
));

-- 4) resource_share_links: scope all owner policies to authenticated
DROP POLICY IF EXISTS "Owners read share links for their resources" ON public.resource_share_links;
DROP POLICY IF EXISTS "Owners insert share links for their resources" ON public.resource_share_links;
DROP POLICY IF EXISTS "Owners update share links for their resources" ON public.resource_share_links;
DROP POLICY IF EXISTS "Owners delete share links for their resources" ON public.resource_share_links;

CREATE POLICY "Owners read share links for their resources"
ON public.resource_share_links FOR SELECT TO authenticated
USING (
  (resource_type = 'analysis' AND EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = resource_id AND a.owner_id = public.current_profile_id()))
  OR (resource_type = 'project' AND EXISTS (SELECT 1 FROM public.projects p WHERE p.id = resource_id AND p.owner_id = public.current_profile_id()))
  OR (resource_type = 'report' AND EXISTS (SELECT 1 FROM public.reports r WHERE r.id = resource_id AND r.created_by = auth.uid()))
);

CREATE POLICY "Owners insert share links for their resources"
ON public.resource_share_links FOR INSERT TO authenticated
WITH CHECK (
  (resource_type = 'analysis' AND EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = resource_id AND a.owner_id = public.current_profile_id()))
  OR (resource_type = 'project' AND EXISTS (SELECT 1 FROM public.projects p WHERE p.id = resource_id AND p.owner_id = public.current_profile_id()))
  OR (resource_type = 'report' AND EXISTS (SELECT 1 FROM public.reports r WHERE r.id = resource_id AND r.created_by = auth.uid()))
);

CREATE POLICY "Owners update share links for their resources"
ON public.resource_share_links FOR UPDATE TO authenticated
USING (
  (resource_type = 'analysis' AND EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = resource_id AND a.owner_id = public.current_profile_id()))
  OR (resource_type = 'project' AND EXISTS (SELECT 1 FROM public.projects p WHERE p.id = resource_id AND p.owner_id = public.current_profile_id()))
  OR (resource_type = 'report' AND EXISTS (SELECT 1 FROM public.reports r WHERE r.id = resource_id AND r.created_by = auth.uid()))
)
WITH CHECK (
  (resource_type = 'analysis' AND EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = resource_id AND a.owner_id = public.current_profile_id()))
  OR (resource_type = 'project' AND EXISTS (SELECT 1 FROM public.projects p WHERE p.id = resource_id AND p.owner_id = public.current_profile_id()))
  OR (resource_type = 'report' AND EXISTS (SELECT 1 FROM public.reports r WHERE r.id = resource_id AND r.created_by = auth.uid()))
);

CREATE POLICY "Owners delete share links for their resources"
ON public.resource_share_links FOR DELETE TO authenticated
USING (
  (resource_type = 'analysis' AND EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = resource_id AND a.owner_id = public.current_profile_id()))
  OR (resource_type = 'project' AND EXISTS (SELECT 1 FROM public.projects p WHERE p.id = resource_id AND p.owner_id = public.current_profile_id()))
  OR (resource_type = 'report' AND EXISTS (SELECT 1 FROM public.reports r WHERE r.id = resource_id AND r.created_by = auth.uid()))
);